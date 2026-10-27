//! Cascaded second-order sections (biquads) on complex and real signals.

use core::marker::PhantomData;

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::timing::{Delay, Timing};
use crate::types::{C32, CQ15, ComplexSample, Q15, RealSample};

/// Most sections a cascade may have.
pub const MAX_SECTIONS: usize = 32;

/// The period of the explicit flush of the `f32` states (`C32` and `f32`): section `k` flushes
/// its two states with `flush_denormal` after every sample `n` (counted from the creation or
/// the last `reset` of the state) with `(n + k) % FLUSH_PERIOD == FLUSH_PERIOD - 1`. The
/// schedule depends only on the absolute sample index, so the result does not depend on how
/// the input is split into calls (decision D17). The skew by the section index lets the
/// vector paths, where section `k + 1` runs one sample behind section `k`, flush all
/// sections after the same step.
pub const FLUSH_PERIOD: usize = 16;

/// The coefficient type of a sample type: `f32` for `C32` and `f32`, `i16` for `CQ15` and `Q15`
/// (Q2.14: value = raw / 2^14, range -2 ... 2 - 2^-14).
///
/// ```
/// use sdrk_core::kernels::biquad::BiquadSample;
/// fn coef_of<T: BiquadSample>(c: T::Coef) -> T::Coef { c }
/// assert_eq!(coef_of::<sdrk_core::Q15>(16384), 16384i16);
/// assert_eq!(coef_of::<f32>(0.5), 0.5f32);
/// ```
pub trait BiquadSample {
    /// One coefficient: `f32` or Q2.14 `i16`.
    type Coef: Copy + Send + Sync + core::fmt::Debug + PartialEq;
}

impl BiquadSample for C32 {
    type Coef = f32;
}
impl BiquadSample for f32 {
    type Coef = f32;
}
impl BiquadSample for CQ15 {
    type Coef = i16;
}
impl BiquadSample for Q15 {
    type Coef = i16;
}

/// The checks shared by both state types: `1 ..= 32` sections and four history entries each.
fn check_shape(sections: usize, history: usize) -> Result<()> {
    if sections == 0 || sections > MAX_SECTIONS {
        return Err(Error::InvalidParameter(
            "a biquad cascade needs 1 to 32 sections",
        ));
    }
    let expected = 4 * sections;
    if history != expected {
        return Err(Error::LengthMismatch {
            expected,
            got: history,
        });
    }
    Ok(())
}

macro_rules! biquad_state {
    (
        $(#[$meta:meta])*
        $name:ident, $bound:ident
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone)]
        pub struct $name<T, Coefs, Hist>
        where
            T: $bound + BiquadSample,
            Coefs: AsRef<[[T::Coef; 5]]>,
            Hist: AsRef<[T]> + AsMut<[T]>,
        {
            coefs: Coefs,
            hist: Hist,
            /// The number of samples processed since `new` or `reset`, modulo
            /// [`FLUSH_PERIOD`]: the phase of the flush schedule of the `f32` states.
            phase: u8,
            _t: PhantomData<T>,
        }

        impl<T, Coefs, Hist> $name<T, Coefs, Hist>
        where
            T: $bound + BiquadSample,
            Coefs: AsRef<[[T::Coef; 5]]>,
            Hist: AsRef<[T]> + AsMut<[T]>,
        {
            /// Creates a state: `1 ..= 32` sections (else `InvalidParameter`) and
            /// `history.len() == 4 * sections` (else `LengthMismatch`). The history is zeroed.
            pub fn new(coefs: Coefs, mut history: Hist) -> Result<Self> {
                check_shape(coefs.as_ref().len(), history.as_ref().len())?;
                history.as_mut().fill(T::ZERO);
                Ok(Self {
                    coefs,
                    hist: history,
                    phase: 0,
                    _t: PhantomData,
                })
            }

            /// Replaces the coefficients (runtime retuning). The number of sections must stay
            /// the same (else `LengthMismatch`, and nothing changes); the history and the
            /// phase of the flush schedule are kept.
            pub fn set_coefs(&mut self, coefs: Coefs) -> Result<()> {
                let expected = self.coefs.as_ref().len();
                let got = coefs.as_ref().len();
                if got != expected {
                    return Err(Error::LengthMismatch { expected, got });
                }
                self.coefs = coefs;
                Ok(())
            }

            /// Clears the history and the phase of the flush schedule (the coefficients are
            /// kept).
            pub fn reset(&mut self) {
                self.hist.as_mut().fill(T::ZERO);
                self.phase = 0;
            }

            /// The number of sections.
            pub fn sections(&self) -> usize {
                self.coefs.as_ref().len()
            }

            /// The coefficients `[b0, b1, b2, a1, a2]` per section.
            pub fn coefs(&self) -> &[[T::Coef; 5]] {
                self.coefs.as_ref()
            }

            /// The history, four entries per section (see the kernel for the layout).
            pub fn history(&self) -> &[T] {
                self.hist.as_ref()
            }

            /// Coefficients and mutable history at once, for kernel implementations in backend
            /// crates.
            pub fn parts_mut(&mut self) -> (&[[T::Coef; 5]], &mut [T]) {
                (self.coefs.as_ref(), self.hist.as_mut())
            }

            /// The phase of the flush schedule of the `f32` states: the number of samples
            /// processed since `new` or `reset`, modulo [`FLUSH_PERIOD`] (always below it).
            /// The `Q15` kernels have no flush and leave it at zero.
            pub fn phase(&self) -> usize {
                usize::from(self.phase)
            }

            /// Coefficients, mutable history and the phase of the flush schedule at once, for
            /// kernel implementations in backend crates: an `f32` kernel reads the phase and
            /// advances it by the number of samples it processed with
            /// [`advance_phase`](Self::advance_phase).
            pub fn parts_phase_mut(&mut self) -> (&[[T::Coef; 5]], &mut [T], usize) {
                (self.coefs.as_ref(), self.hist.as_mut(), usize::from(self.phase))
            }

            /// Advances the phase of the flush schedule by `samples` (modulo
            /// [`FLUSH_PERIOD`]), for kernel implementations after a call of `samples`
            /// samples.
            pub fn advance_phase(&mut self, samples: usize) {
                let next = (usize::from(self.phase) + samples % FLUSH_PERIOD) % FLUSH_PERIOD;
                self.phase = u8::try_from(next).unwrap_or(0);
            }
        }

        impl<T, Coefs, Hist> $name<T, Coefs, Hist>
        where
            T: $bound + BiquadSample,
            Coefs: AsRef<[[T::Coef; 5]]> + AsMut<[[T::Coef; 5]]>,
            Hist: AsRef<[T]> + AsMut<[T]>,
        {
            /// The coefficients for in-place retuning with caller-provided storage that
            /// cannot be replaced by value; the number of sections cannot change and the
            /// history is kept.
            pub fn coefs_mut(&mut self) -> &mut [[T::Coef; 5]] {
                self.coefs.as_mut()
            }
        }

        impl<T, Coefs, Hist> Timing for $name<T, Coefs, Hist>
        where
            T: $bound + BiquadSample,
            Coefs: AsRef<[[T::Coef; 5]]>,
            Hist: AsRef<[T]> + AsMut<[T]>,
        {
            fn ratio(&self) -> Ratio {
                Ratio::ONE
            }

            /// Zero; the delay of an IIR filter depends on the frequency and is not part of
            /// the exact event timing (as for `deemphasis`).
            fn group_delay(&self) -> Delay {
                Delay::ZERO
            }
        }
    };
}

biquad_state! {
    /// State of the complex `biquad` kernel: the coefficients `[b0, b1, b2, a1, a2]` of
    /// `1 ..= 32` sections (`a0 = 1`) and four history entries per section.
    ///
    /// The caller provides the storage (`[T; N]`, `&mut [T]` or, in upper layers, `Vec<T>`).
    ///
    /// ```
    /// use sdrk_core::{kernels::biquad::BiquadState, C32};
    /// let unit = [[1.0f32, 0.0, 0.0, 0.0, 0.0]];
    /// let state = BiquadState::<C32, _, _>::new(unit, [C32::default(); 4])?;
    /// assert_eq!(state.sections(), 1);
    /// // The history needs four entries per section.
    /// assert!(BiquadState::<C32, _, _>::new(unit, [C32::default(); 2]).is_err());
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    BiquadState, ComplexSample
}

biquad_state! {
    /// State of the real `biquad_real` kernel: the coefficients `[b0, b1, b2, a1, a2]` of
    /// `1 ..= 32` sections (`a0 = 1`) and four history entries per section.
    ///
    /// The caller provides the storage (`[T; N]`, `&mut [T]` or, in upper layers, `Vec<T>`).
    ///
    /// ```
    /// use sdrk_core::{kernels::biquad::BiquadRealState, Q15};
    /// let mut state =
    ///     BiquadRealState::<Q15, _, _>::new(vec![[16384i16, 0, 0, 0, 0]], vec![Q15(0); 4])?;
    /// // Retuning keeps the number of sections.
    /// assert!(state.set_coefs(vec![[8192i16, 0, 0, 0, 0]]).is_ok());
    /// assert!(state.set_coefs(vec![[8192i16, 0, 0, 0, 0]; 2]).is_err());
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    BiquadRealState, RealSample
}

/// Cascaded second-order sections on a complex signal.
///
/// ```
/// use sdrk_core::{kernels::biquad::{Biquad, BiquadState}, Fallback, C32};
/// // b = [0.5, 0.5, 0], a = [0, 0]: a two-point average.
/// let coefs = [[0.5f32, 0.5, 0.0, 0.0, 0.0]];
/// let mut state = BiquadState::<C32, _, _>::new(coefs, [C32::default(); 4])?;
/// let input = [C32::new(2.0, 0.0), C32::new(4.0, 2.0)];
/// let mut output = [C32::default(); 2];
/// Fallback.biquad(&mut state, &input, &mut output)?;
/// assert_eq!(output, [C32::new(1.0, 0.0), C32::new(3.0, 1.0)]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Biquad<T: ComplexSample + BiquadSample> {
    /// Filters `re` and `im` independently with the same coefficients. Per section
    /// `H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)`; the output of section `k`
    /// is the input of section `k + 1`. The lengths must be equal.
    ///
    /// `C32` (transposed direct form II, `f32` without FMA, per section and component):
    /// `y = b0 x + s1`; `s1 = (b1 x + s2) - a1 y`; `s2 = b2 x - a2 y`. Both states of section
    /// `k` are flushed with `flush_denormal` after every sample `n` with
    /// `(n + k) % FLUSH_PERIOD == FLUSH_PERIOD - 1` ([`FLUSH_PERIOD`]; `n` counts the samples
    /// since `new` or `reset`). The history holds `s1, s2` of section `k` in the entries `4k`
    /// and `4k + 1`; the rest is unused. `CQ15` (direct form I): `acc = b0 x + b1 x1 +
    /// b2 x2 - a1 y1 - a2 y2` in `i64`, `y = sat16(round_shift(acc, 14))` per section; the
    /// history holds `x1, x2, y1, y2` of section `k` in the entries `4k ..= 4k + 3`. Bit-exact
    /// between `Fallback` and `Reference`. A `C32` state is only valid for the implementation
    /// that filled it (the reference keeps direct form I states).
    fn biquad<C, H>(
        &self,
        s: &mut BiquadState<T, C, H>,
        input: &[T],
        output: &mut [T],
    ) -> Result<()>
    where
        C: AsRef<[[T::Coef; 5]]>,
        H: AsRef<[T]> + AsMut<[T]>;
}

/// Cascaded second-order sections on a real signal.
///
/// ```
/// use sdrk_core::{kernels::biquad::{BiquadReal, BiquadRealState}, Fallback};
/// let coefs = [[0.5f32, 0.5, 0.0, 0.0, 0.0]];
/// let mut state = BiquadRealState::<f32, _, _>::new(coefs, [0.0f32; 4])?;
/// let mut output = [0.0f32; 3];
/// Fallback.biquad_real(&mut state, &[2.0, 4.0, 6.0], &mut output)?;
/// assert_eq!(output, [1.0, 3.0, 5.0]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait BiquadReal<T: RealSample + BiquadSample> {
    /// The real variant of [`Biquad::biquad`]: the same recursions on a real signal (`f32`:
    /// transposed direct form II, `Q15`: direct form I, bit-exact between `Fallback` and
    /// `Reference`), the same history layout. The lengths must be equal.
    fn biquad_real<C, H>(
        &self,
        s: &mut BiquadRealState<T, C, H>,
        input: &[T],
        output: &mut [T],
    ) -> Result<()>
    where
        C: AsRef<[[T::Coef; 5]]>,
        H: AsRef<[T]> + AsMut<[T]>;
}
