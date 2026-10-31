//! Halfband decimators and the real-to-complex converter for devices with real sampling.
//!
//! A halfband filter has `N = 4 k + 3` taps, is symmetric and has the centre tap `h[m]`
//! (`m = (N - 1) / 2`, odd); every tap at an even offset from the centre, `h[m ± 2 i]` with
//! `i >= 1`, is exactly zero. The non-zero taps are therefore the even indices `0, 2, ..., N - 1`
//! and the centre `m`: about half of the taps. The decimators evaluate only those.

use core::marker::PhantomData;

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::timing::{Delay, Progress, Timing};
use crate::types::{ComplexSample, Q15, RealSample, Sample};

/// Negation of a real sample as the shift of [`RealToComplex`] needs it: `f32` flips the sign,
/// `Q15` saturates (`-(-32768) = 32767`).
pub(crate) trait ShiftNeg: Sample {
    fn shift_neg(self) -> Self;
}

impl ShiftNeg for f32 {
    #[inline]
    fn shift_neg(self) -> f32 {
        -self
    }
}

impl ShiftNeg for Q15 {
    #[inline]
    fn shift_neg(self) -> Q15 {
        Q15(self.0.saturating_neg())
    }
}

/// `u[n] = x[n] (-j)^n` for the phase `n mod 4`: `(x, 0)`, `(0, -x)`, `(-x, 0)`, `(0, x)`.
#[inline]
pub(crate) fn shifted<T: ComplexSample>(x: T::Real, n4: usize) -> T
where
    T::Real: ShiftNeg,
{
    let zero = <T::Real as Sample>::ZERO;
    match n4 & 3 {
        0 => T::new(x, zero),
        1 => T::new(zero, x.shift_neg()),
        2 => T::new(x.shift_neg(), zero),
        _ => T::new(zero, x),
    }
}

/// Checks the halfband rules of the taps and the history length; zeroes the history.
fn validate<P: Sample, X: Sample>(taps: &[P], history: &mut [X]) -> Result<()> {
    let n = taps.len();
    if n < 3 || n % 4 != 3 {
        return Err(Error::InvalidParameter(
            "a halfband filter needs N = 4 k + 3 taps",
        ));
    }
    let m = (n - 1) / 2;
    let clean = taps
        .iter()
        .enumerate()
        .all(|(k, t)| k == m || k.abs_diff(m) % 2 == 1 || *t == P::ZERO);
    if !clean {
        return Err(Error::InvalidParameter(
            "the taps at even offsets from the centre of a halfband filter must be exactly zero",
        ));
    }
    let expected = n - 1;
    let got = history.len();
    if got != expected {
        return Err(Error::LengthMismatch { expected, got });
    }
    history.fill(X::ZERO);
    Ok(())
}

/// Storage and phase shared by the three states.
#[derive(Debug, Clone)]
struct Core<P: Sample, X: Sample, Taps: AsRef<[P]>, Hist: AsRef<[X]> + AsMut<[X]>> {
    taps: Taps,
    hist: Hist,
    /// `HalfbandState`, `HalfbandRealState`: input samples to skip before the next output
    /// (0 or 1). `RealToComplexState`: `n mod 4` of the next input sample.
    phase: usize,
    _t: PhantomData<(P, X)>,
}

impl<P: Sample, X: Sample, Taps: AsRef<[P]>, Hist: AsRef<[X]> + AsMut<[X]>> Core<P, X, Taps, Hist> {
    fn new(taps: Taps, mut history: Hist) -> Result<Self> {
        validate(taps.as_ref(), history.as_mut())?;
        Ok(Self {
            taps,
            hist: history,
            phase: 0,
            _t: PhantomData,
        })
    }

    fn reset(&mut self) {
        self.hist.as_mut().fill(X::ZERO);
        self.phase = 0;
    }

    fn group_delay(&self) -> Delay {
        let n = self.taps.as_ref().len() as u64;
        Delay::new(n.saturating_sub(1), 2).unwrap_or(Delay::ZERO)
    }
}

macro_rules! halfband_state_api {
    ($tap:ty, $hist:ty) => {
        /// Clears the history and restarts the phase (the taps are kept).
        pub fn reset(&mut self) {
            self.core.reset();
        }

        /// The filter taps (`N = 4 k + 3`, zero at the even offsets from the centre).
        pub fn taps(&self) -> &[$tap] {
            self.core.taps.as_ref()
        }

        /// The history, oldest sample first (`taps().len() - 1` samples).
        pub fn history(&self) -> &[$hist] {
            self.core.hist.as_ref()
        }

        /// The number of taps `N`.
        pub fn len(&self) -> usize {
            self.core.taps.as_ref().len()
        }

        /// Always `false`: a state has at least three taps.
        pub fn is_empty(&self) -> bool {
            false
        }
    };
}

/// Borrowed parts of a [`HalfbandState`] or [`HalfbandRealState`] for kernel implementations in
/// backend crates.
pub struct HalfbandParts<'a, P, T> {
    /// The filter taps.
    pub taps: &'a [P],
    /// The history, oldest sample first.
    pub hist: &'a mut [T],
    /// Input samples still to skip before the next output (0 or 1).
    pub countdown: &'a mut usize,
}

/// Borrowed parts of a [`RealToComplexState`] for kernel implementations in backend crates.
pub struct RealToComplexParts<'a, P, T> {
    /// The filter taps.
    pub taps: &'a [P],
    /// The history of the shifted samples `u`, oldest first.
    pub hist: &'a mut [T],
    /// `n mod 4` of the next input sample (0 to 3).
    pub n4: &'a mut usize,
}

/// State of the complex halfband decimator: halfband taps, a history of `N - 1` samples and the
/// decimation phase.
///
/// The definition is that of `fir_decimate` with factor 2 and the same taps: with
/// `y_full[n] = sum h[k] x[n-k]` over the whole stream (zero history at the start and after
/// `reset`), the outputs are `y[m] = y_full[2 m]`; the first input sample after a reset produces
/// the first output. The group delay is `(N - 1) / 2` input samples.
///
/// ```
/// use sdrk_core::{kernels::halfband::{HalfbandDecimate, HalfbandState}, Fallback, C32};
/// // N = 3: h = [0.25, 0.5, 0.25] (the taps at the even offsets from the centre vanish only for i >= 1).
/// let mut state = HalfbandState::<C32, _, _>::new([0.25f32, 0.5, 0.25], [C32::default(); 2])?;
/// let input: Vec<C32> = (1..=6).map(|v| C32::new(v as f32, 0.0)).collect();
/// let mut output = [C32::default(); 3];
/// let p = Fallback.halfband_decimate(&mut state, &input, &mut output)?;
/// assert_eq!((p.consumed, p.produced), (6, 3));
/// assert_eq!(output.map(|s| s.re), [0.25, 2.0, 4.0]);
/// // Taps with a non-zero tap at an even offset from the centre are rejected.
/// assert!(HalfbandState::<C32, _, _>::new([0.1f32, 0.2, 0.4, 0.2, 0.1, 0.0, 0.0], [C32::default(); 6]).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct HalfbandState<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>> {
    core: Core<T::Real, T, Taps, Hist>,
}

impl<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>>
    HalfbandState<T, Taps, Hist>
{
    /// Creates a state: `taps.len() >= 3` with `taps.len() % 4 == 3`, the taps at the even
    /// offsets `m ± 2 i` (`i >= 1`) from the centre `m = (N - 1) / 2` exactly zero, and
    /// `history.len() == taps.len() - 1`. The history is zeroed.
    ///
    /// # Errors
    /// `InvalidParameter` for a wrong tap count or a non-zero tap at an even offset;
    /// `LengthMismatch` for a wrong history length.
    pub fn new(taps: Taps, history: Hist) -> Result<Self> {
        Ok(Self {
            core: Core::new(taps, history)?,
        })
    }

    halfband_state_api!(T::Real, T);

    /// Number of input samples that are skipped before the next output is computed (0 or 1).
    pub fn countdown(&self) -> usize {
        self.core.phase
    }

    /// The largest number of input samples that [`HalfbandDecimate::halfband_decimate`] consumes
    /// while producing at most `output_len` outputs: `countdown + 2 * output_len` (saturating).
    pub fn max_input_for_output(&self, output_len: usize) -> usize {
        self.core.phase.saturating_add(output_len.saturating_mul(2))
    }

    /// The borrowed parts of the state, for kernel implementations in backend crates.
    pub fn parts_mut(&mut self) -> HalfbandParts<'_, T::Real, T> {
        HalfbandParts {
            taps: self.core.taps.as_ref(),
            hist: self.core.hist.as_mut(),
            countdown: &mut self.core.phase,
        }
    }
}

impl<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>> Timing
    for HalfbandState<T, Taps, Hist>
{
    /// `1 / 2`.
    fn ratio(&self) -> Ratio {
        Ratio::new(1, 2).unwrap_or(Ratio::ONE)
    }

    /// `(N - 1) / 2` input samples, built as `Delay::new(N - 1, 2)`.
    fn group_delay(&self) -> Delay {
        self.core.group_delay()
    }
}

/// Complex FIR with halfband taps followed by decimation by 2.
///
/// The kernel consumes input only as far as the output space allows: an input sample that would
/// produce an output is not consumed when `output` is full. The caller resubmits the unconsumed
/// input.
pub trait HalfbandDecimate<T: ComplexSample> {
    /// Computes `y[m] = y_full[2 m]` (see [`HalfbandState`]) over the non-zero taps only and
    /// returns how many input samples were consumed and how many outputs were produced.
    /// Accumulation: `C32` without FMA, the even taps in ascending tap index and the centre
    /// tap last (the zero taps are omitted; the bits are therefore not those of `fir_decimate`
    /// with factor 2, whose centre tap comes in its own position), `CQ15` in `i64` with one rounding and saturation per
    /// component, bit-exact with `fir_decimate` with factor 2.
    fn halfband_decimate<Taps, Hist>(
        &self,
        state: &mut HalfbandState<T, Taps, Hist>,
        input: &[T],
        output: &mut [T],
    ) -> Result<Progress>
    where
        Taps: AsRef<[T::Real]>,
        Hist: AsRef<[T]> + AsMut<[T]>;
}

/// State of the real halfband decimator (`f32` or `Q15` samples and taps). The definition is the
/// one of [`HalfbandState`].
///
/// ```
/// use sdrk_core::{kernels::halfband::{HalfbandDecimateReal, HalfbandRealState}, Fallback};
/// let mut state = HalfbandRealState::<f32, _, _>::new([0.25f32, 0.5, 0.25], [0.0f32; 2])?;
/// let mut output = [0.0f32; 2];
/// let p = Fallback.halfband_decimate_real(&mut state, &[4.0, 4.0, 4.0, 4.0], &mut output)?;
/// assert_eq!((p.consumed, p.produced), (4, 2));
/// assert_eq!(output, [1.0, 4.0]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct HalfbandRealState<T: RealSample, Taps: AsRef<[T]>, Hist: AsRef<[T]> + AsMut<[T]>> {
    core: Core<T, T, Taps, Hist>,
}

impl<T: RealSample, Taps: AsRef<[T]>, Hist: AsRef<[T]> + AsMut<[T]>>
    HalfbandRealState<T, Taps, Hist>
{
    /// Creates a state with the rules of [`HalfbandState::new`].
    ///
    /// # Errors
    /// `InvalidParameter` for a wrong tap count or a non-zero tap at an even offset;
    /// `LengthMismatch` for a wrong history length.
    pub fn new(taps: Taps, history: Hist) -> Result<Self> {
        Ok(Self {
            core: Core::new(taps, history)?,
        })
    }

    halfband_state_api!(T, T);

    /// Number of input samples that are skipped before the next output is computed (0 or 1).
    pub fn countdown(&self) -> usize {
        self.core.phase
    }

    /// The largest number of input samples that [`HalfbandDecimateReal::halfband_decimate_real`]
    /// consumes while producing at most `output_len` outputs: `countdown + 2 * output_len`
    /// (saturating).
    pub fn max_input_for_output(&self, output_len: usize) -> usize {
        self.core.phase.saturating_add(output_len.saturating_mul(2))
    }

    /// The borrowed parts of the state, for kernel implementations in backend crates.
    pub fn parts_mut(&mut self) -> HalfbandParts<'_, T, T> {
        HalfbandParts {
            taps: self.core.taps.as_ref(),
            hist: self.core.hist.as_mut(),
            countdown: &mut self.core.phase,
        }
    }
}

impl<T: RealSample, Taps: AsRef<[T]>, Hist: AsRef<[T]> + AsMut<[T]>> Timing
    for HalfbandRealState<T, Taps, Hist>
{
    /// `1 / 2`.
    fn ratio(&self) -> Ratio {
        Ratio::new(1, 2).unwrap_or(Ratio::ONE)
    }

    /// `(N - 1) / 2` input samples, built as `Delay::new(N - 1, 2)`.
    fn group_delay(&self) -> Delay {
        self.core.group_delay()
    }
}

/// Real FIR with halfband taps followed by decimation by 2.
pub trait HalfbandDecimateReal<T: RealSample> {
    /// Computes `y[m] = y_full[2 m]` (see [`HalfbandState`]) over the non-zero taps only.
    /// Accumulation: `f32` without FMA, the even taps in ascending tap index and the centre
    /// tap last, `Q15` in `i64` with one rounding
    /// and saturation, bit-exact with the integer model of `fir_decimate` with factor 2.
    fn halfband_decimate_real<Taps, Hist>(
        &self,
        state: &mut HalfbandRealState<T, Taps, Hist>,
        input: &[T],
        output: &mut [T],
    ) -> Result<Progress>
    where
        Taps: AsRef<[T]>,
        Hist: AsRef<[T]> + AsMut<[T]>;
}

/// State of the real-to-complex converter: halfband taps, the history of the shifted samples
/// `u` (`N - 1` of them) and `n mod 4`.
///
/// With `n` counted from construction or `reset`, `u[n] = x[n] (-j)^n`, that is `(x, 0)`,
/// `(0, -x)`, `(-x, 0)`, `(0, x)` for `n mod 4 = 0, 1, 2, 3` (for `Q15` the negation saturates:
/// `-(-32768) = 32767`). The output is the halfband decimation of `u` with the same taps
/// ([`HalfbandState`]): `y[m] = sum h[k] u[2 m - k]`. A real input component at the frequency
/// `f` (0 to fs/2) appears at `f - fs/4` in the complex output (rate fs/2; the output frequency
/// 0 is the input frequency fs/4); the mirror component at `-f` lands outside the pass band and
/// is removed. The group delay is `(N - 1) / 2` input samples.
///
/// ```
/// use sdrk_core::{kernels::halfband::{RealToComplex, RealToComplexState}, Fallback, C32};
/// let mut state = RealToComplexState::<C32, _, _>::new([0.25f32, 0.5, 0.25], [C32::default(); 2])?;
/// let mut output = [C32::default(); 2];
/// let p = Fallback.real_to_complex(&mut state, &[1.0, 2.0, 3.0, 4.0], &mut output)?;
/// assert_eq!((p.consumed, p.produced), (4, 2));
/// // y[0] = 0.25 u[0] = 0.25; y[1] = 0.25 u[2] + 0.5 u[1] + 0.25 u[0] = -0.75 - 1.0j + 0.25.
/// assert_eq!(output, [C32::new(0.25, 0.0), C32::new(-0.5, -1.0)]);
/// assert_eq!(state.n_mod_4(), 0);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct RealToComplexState<
    T: ComplexSample,
    Taps: AsRef<[T::Real]>,
    Hist: AsRef<[T]> + AsMut<[T]>,
> {
    core: Core<T::Real, T, Taps, Hist>,
}

impl<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>>
    RealToComplexState<T, Taps, Hist>
{
    /// Creates a state with the rules of [`HalfbandState::new`]; `n` starts at 0.
    ///
    /// # Errors
    /// `InvalidParameter` for a wrong tap count or a non-zero tap at an even offset;
    /// `LengthMismatch` for a wrong history length.
    pub fn new(taps: Taps, history: Hist) -> Result<Self> {
        Ok(Self {
            core: Core::new(taps, history)?,
        })
    }

    halfband_state_api!(T::Real, T);

    /// `n mod 4` of the next input sample (0 to 3); `reset` sets it to 0.
    pub fn n_mod_4(&self) -> usize {
        self.core.phase
    }

    /// Number of input samples that are skipped before the next output is computed (0 or 1):
    /// the output samples are the inputs with an even `n`.
    pub fn countdown(&self) -> usize {
        self.core.phase & 1
    }

    /// The largest number of input samples that [`RealToComplex::real_to_complex`] consumes while
    /// producing at most `output_len` outputs: `countdown + 2 * output_len` (saturating).
    pub fn max_input_for_output(&self, output_len: usize) -> usize {
        self.countdown()
            .saturating_add(output_len.saturating_mul(2))
    }

    /// The borrowed parts of the state, for kernel implementations in backend crates.
    pub fn parts_mut(&mut self) -> RealToComplexParts<'_, T::Real, T> {
        RealToComplexParts {
            taps: self.core.taps.as_ref(),
            hist: self.core.hist.as_mut(),
            n4: &mut self.core.phase,
        }
    }
}

impl<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>> Timing
    for RealToComplexState<T, Taps, Hist>
{
    /// `1 / 2`.
    fn ratio(&self) -> Ratio {
        Ratio::new(1, 2).unwrap_or(Ratio::ONE)
    }

    /// `(N - 1) / 2` input samples, built as `Delay::new(N - 1, 2)`.
    fn group_delay(&self) -> Delay {
        self.core.group_delay()
    }
}

/// Shift by fs/4 (`x[n] (-j)^n`) and halfband decimation in one kernel.
///
/// Like [`HalfbandDecimate`] it consumes input only as far as the output space allows.
pub trait RealToComplex<T: ComplexSample> {
    /// Computes the outputs defined at [`RealToComplexState`]. Every `u[n]` has one zero
    /// component, so the kernel evaluates the real part over the taps at even indices and the
    /// imaginary part over the centre tap only. Accumulation: `C32` without FMA, the even taps
    /// in ascending tap index and the centre tap last, `CQ15` in `i64` with one rounding and saturation per component (bit-exact
    /// with the definition).
    fn real_to_complex<Taps, Hist>(
        &self,
        state: &mut RealToComplexState<T, Taps, Hist>,
        input: &[T::Real],
        output: &mut [T],
    ) -> Result<Progress>
    where
        Taps: AsRef<[T::Real]>,
        Hist: AsRef<[T]> + AsMut<[T]>;
}
