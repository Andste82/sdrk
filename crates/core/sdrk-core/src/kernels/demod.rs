//! Demodulation and conditioning kernels: FM discriminator, de-emphasis, DC blocker.

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::timing::{Delay, Rate, Timing};
use crate::types::{ComplexSample, RealSample};

/// State of the quadrature (FM) discriminator: the previous input sample.
///
/// `new()` and `reset()` set it to zero, so the first output is `arg(0) = 0`.
///
/// ```
/// use sdrk_core::{kernels::demod::{QuadDemod, QuadDemodState}, Fallback, C32};
/// let mut state = QuadDemodState::<C32>::new();
/// // A quarter turn per sample gives pi / 2 after the first sample.
/// let input = [C32::new(1.0, 0.0), C32::new(0.0, 1.0), C32::new(-1.0, 0.0)];
/// let mut output = [0.0f32; 3];
/// Fallback.quad_demod(&mut state, &input, &mut output, 1.0)?;
/// assert_eq!(output[0], 0.0);
/// assert!((output[1] - core::f32::consts::FRAC_PI_2).abs() < 1e-6);
/// assert!((output[2] - core::f32::consts::FRAC_PI_2).abs() < 1e-6);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadDemodState<T: ComplexSample> {
    prev: T,
}

impl<T: ComplexSample> Default for QuadDemodState<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ComplexSample> QuadDemodState<T> {
    /// A state with `prev = 0`.
    pub fn new() -> Self {
        Self { prev: T::ZERO }
    }

    /// Sets `prev` back to zero.
    pub fn reset(&mut self) {
        self.prev = T::ZERO;
    }

    /// The previous input sample.
    pub fn prev(&self) -> T {
        self.prev
    }

    /// Stores the last input sample; used by kernel implementations in backend crates.
    pub fn set_prev(&mut self, prev: T) {
        self.prev = prev;
    }
}

impl<T: ComplexSample> Timing for QuadDemodState<T> {
    fn ratio(&self) -> Ratio {
        Ratio::ONE
    }

    /// One input sample (the discriminator looks at `x[n]` and `x[n-1]`), `Delay::new(1, 2)`.
    fn group_delay(&self) -> Delay {
        Delay::new(1, 2).unwrap_or(Delay::ZERO)
    }
}

/// Quadrature (FM) discriminator.
///
/// ```
/// use sdrk_core::{kernels::demod::{QuadDemod, QuadDemodState}, Fallback, CQ15, Q15};
/// let mut state = QuadDemodState::<CQ15>::new();
/// // A quarter turn per sample is 0.5 in units of pi.
/// let q = |re, im| CQ15 { re: Q15(re), im: Q15(im) };
/// let input = [q(16384, 0), q(0, 16384), q(-16384, 0)];
/// let mut output = [Q15(0); 3];
/// Fallback.quad_demod(&mut state, &input, &mut output, Q15(32767))?;
/// assert_eq!(output[0], Q15(0));
/// assert!((i32::from(output[1].0) - 16384).abs() <= 1);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait QuadDemod<T: ComplexSample> {
    /// `output[n] = gain * arg(x[n] * conj(x[n-1]))`, with `x[-1]` taken from the state
    /// (zero after a reset) and `arg(0) = 0`. The `f32` output is in radians times the
    /// gain; the `Q15` output has the unit phase / pi, scaled by the Q15 gain.
    /// A typical FM gain is `fs / (2 pi deviation)` so that full deviation maps to +-1.
    /// The lengths must be equal. The state keeps the last input sample.
    fn quad_demod(
        &self,
        state: &mut QuadDemodState<T>,
        input: &[T],
        output: &mut [T::Real],
        gain: T::Real,
    ) -> Result<()>;
}

/// State of the single-pole de-emphasis filter: coefficient `alpha` and the last output.
///
/// The delay of this IIR filter depends on the frequency, so `group_delay` reports 0
/// and is not part of exact event timing.
///
/// ```
/// use sdrk_core::{kernels::demod::DeemphasisState, Rate};
/// let state = DeemphasisState::<f32>::new(50e-6, Rate::hz(48_000)?)?;
/// assert!((state.alpha() - 0.34).abs() < 0.01);
/// assert!(DeemphasisState::<f32>::new(0.0, Rate::hz(48_000)?).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeemphasisState<T: RealSample> {
    alpha: T,
    y: T,
}

impl<T: RealSample> DeemphasisState<T> {
    /// Creates a state with time constant `tau_s` (seconds) for samples at `rate`:
    /// `alpha = 1 - exp(-1 / (fs * tau))`, computed in f64 and then rounded to the
    /// sample type (cast for `f32`, the Q15 rule for `Q15`). `tau_s` must be finite
    /// and positive, otherwise `InvalidParameter`.
    pub fn new(tau_s: f64, rate: Rate) -> Result<Self> {
        if !(tau_s.is_finite() && tau_s > 0.0) {
            return Err(Error::InvalidParameter(
                "time constant must be finite and positive",
            ));
        }
        let alpha = 1.0 - libm::exp(-1.0 / (rate.as_f64() * tau_s));
        Ok(Self {
            alpha: T::from_f64_round(alpha),
            y: T::ZERO,
        })
    }

    /// Sets the last output back to zero (the coefficient is kept).
    pub fn reset(&mut self) {
        self.y = T::ZERO;
    }

    /// The coefficient `alpha`.
    pub fn alpha(&self) -> T {
        self.alpha
    }

    /// The last output sample.
    pub fn last(&self) -> T {
        self.y
    }

    pub(crate) fn parts_mut(&mut self) -> (T, &mut T) {
        (self.alpha, &mut self.y)
    }
}

impl<T: RealSample> Timing for DeemphasisState<T> {
    fn ratio(&self) -> Ratio {
        Ratio::ONE
    }

    /// Zero; the real delay of the IIR filter depends on the frequency.
    fn group_delay(&self) -> Delay {
        Delay::ZERO
    }
}

/// Single-pole low-pass for audio de-emphasis.
///
/// ```
/// use sdrk_core::{kernels::demod::{Deemphasis, DeemphasisState}, Fallback, Rate};
/// let mut state = DeemphasisState::<f32>::new(1.0, Rate::hz(1)?)?;
/// let mut output = [0.0f32; 2];
/// Fallback.deemphasis(&mut state, &[1.0, 1.0], &mut output)?;
/// assert!(output[0] > 0.6 && output[1] > output[0]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Deemphasis<T: RealSample> {
    /// `y[n] = y[n-1] + alpha * (x[n] - y[n-1])`. The lengths must be equal.
    fn deemphasis(
        &self,
        state: &mut DeemphasisState<T>,
        input: &[T],
        output: &mut [T],
    ) -> Result<()>;
}

/// State of the DC blocker: the pole `r` and the previous input and output samples.
///
/// ```
/// use sdrk_core::{kernels::demod::DcBlockState, C32};
/// let state = DcBlockState::<C32>::new(0.999)?;
/// assert_eq!(state.r(), 0.999);
/// assert!(DcBlockState::<C32>::new(1.0).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DcBlockState<T: ComplexSample> {
    r: T::Real,
    x_prev: T,
    y_prev: T,
}

impl<T: ComplexSample> DcBlockState<T> {
    /// Creates a state with pole `r`, `0 < r < 1` (for Q15 `1 <= raw <= 32767`);
    /// anything else, including NaN, gives `InvalidParameter`.
    pub fn new(r: T::Real) -> Result<Self> {
        let v = r.to_f64();
        if !(v > 0.0 && v < 1.0) {
            return Err(Error::InvalidParameter(
                "the pole must be greater than 0 and less than 1",
            ));
        }
        Ok(Self {
            r,
            x_prev: T::ZERO,
            y_prev: T::ZERO,
        })
    }

    /// Clears the previous input and output (the pole is kept).
    pub fn reset(&mut self) {
        self.x_prev = T::ZERO;
        self.y_prev = T::ZERO;
    }

    /// The pole `r`.
    pub fn r(&self) -> T::Real {
        self.r
    }

    /// The previous input sample.
    pub fn x_prev(&self) -> T {
        self.x_prev
    }

    /// The previous output sample (flushed for `f32`).
    pub fn y_prev(&self) -> T {
        self.y_prev
    }

    pub(crate) fn parts_mut(&mut self) -> (T::Real, &mut T, &mut T) {
        (self.r, &mut self.x_prev, &mut self.y_prev)
    }
}

impl<T: ComplexSample> Timing for DcBlockState<T> {
    fn ratio(&self) -> Ratio {
        Ratio::ONE
    }

    /// Zero (a one-pole high-pass; its delay depends on the frequency).
    fn group_delay(&self) -> Delay {
        Delay::ZERO
    }
}

/// DC blocker for complex signals.
///
/// ```
/// use sdrk_core::{kernels::demod::{DcBlock, DcBlockState}, Fallback, C32};
/// let mut state = DcBlockState::<C32>::new(0.5)?;
/// let input = [C32::new(1.0, 0.0); 3];
/// let mut output = [C32::default(); 3];
/// Fallback.dc_block(&mut state, &input, &mut output)?;
/// assert_eq!(output.map(|s| s.re), [1.0, 0.5, 0.25]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait DcBlock<T: ComplexSample> {
    /// `y[n] = x[n] - x[n-1] + r * y[n-1]` per component. The lengths must be equal.
    fn dc_block(&self, state: &mut DcBlockState<T>, input: &[T], output: &mut [T]) -> Result<()>;
}

/// State of the DC blocker for real signals: the pole `r` and the previous input and
/// output samples. The real variant of [`DcBlockState`] with the same rules.
///
/// ```
/// use sdrk_core::kernels::demod::DcBlockRealState;
/// let state = DcBlockRealState::<f32>::new(0.999)?;
/// assert_eq!(state.r(), 0.999);
/// assert!(DcBlockRealState::<f32>::new(1.0).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DcBlockRealState<T: RealSample> {
    r: T,
    x_prev: T,
    y_prev: T,
}

impl<T: RealSample> DcBlockRealState<T> {
    /// Creates a state with pole `r`, `0 < r < 1` (for Q15 `1 <= raw <= 32767`);
    /// anything else, including NaN, gives `InvalidParameter`.
    pub fn new(r: T) -> Result<Self> {
        let v = r.to_f64();
        if !(v > 0.0 && v < 1.0) {
            return Err(Error::InvalidParameter(
                "the pole must be greater than 0 and less than 1",
            ));
        }
        Ok(Self {
            r,
            x_prev: T::ZERO,
            y_prev: T::ZERO,
        })
    }

    /// Clears the previous input and output (the pole is kept).
    pub fn reset(&mut self) {
        self.x_prev = T::ZERO;
        self.y_prev = T::ZERO;
    }

    /// The pole `r`.
    pub fn r(&self) -> T {
        self.r
    }

    /// The previous input sample.
    pub fn x_prev(&self) -> T {
        self.x_prev
    }

    /// The previous output sample (flushed for `f32`).
    pub fn y_prev(&self) -> T {
        self.y_prev
    }

    pub(crate) fn parts_mut(&mut self) -> (T, &mut T, &mut T) {
        (self.r, &mut self.x_prev, &mut self.y_prev)
    }
}

impl<T: RealSample> Timing for DcBlockRealState<T> {
    fn ratio(&self) -> Ratio {
        Ratio::ONE
    }

    /// Zero (a one-pole high-pass; its delay depends on the frequency).
    fn group_delay(&self) -> Delay {
        Delay::ZERO
    }
}

/// DC blocker for real signals.
///
/// For `f32`, a NaN input makes the stored `x_prev` and `y_prev` NaN (the denormal flush
/// leaves NaN unchanged), so every following output is NaN until `reset()`; this is the
/// behaviour of `dc_block`.
///
/// ```
/// use sdrk_core::{kernels::demod::{DcBlockReal, DcBlockRealState}, Fallback};
/// let mut state = DcBlockRealState::<f32>::new(0.5)?;
/// let mut output = [0.0f32; 3];
/// Fallback.dc_block_real(&mut state, &[1.0; 3], &mut output)?;
/// assert_eq!(output, [1.0, 0.5, 0.25]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait DcBlockReal<T: RealSample> {
    /// `y[n] = x[n] - x[n-1] + r * y[n-1]`. `f32` keeps `flush_denormal(y)` as the
    /// feedback; `Q15` computes `sat16((x - x_prev) + round_shift(r * y_prev, 15))` in
    /// `i64`, so a signal that stays within a dead band of a few LSB around a constant
    /// does not decay to zero (see `dc_block`). The lengths must be equal.
    fn dc_block_real(
        &self,
        state: &mut DcBlockRealState<T>,
        input: &[T],
        output: &mut [T],
    ) -> Result<()>;
}
