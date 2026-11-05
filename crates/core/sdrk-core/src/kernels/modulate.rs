//! Modulators (transmit side).

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::timing::{Delay, Rate, Timing};
use crate::types::ComplexSample;

const TWO_POW_32: f64 = 4_294_967_296.0;

/// State of the FM modulator: the phase accumulator in units of 2π/2^32 and the scale `k`
/// (phase units per unit of input amplitude).
///
/// `k = deviation_hz / sample_rate * 2^32`, so an input of amplitude 1.0 advances the phase
/// by `deviation_hz / sample_rate` turns per sample.
///
/// ```
/// use sdrk_core::{kernels::modulate::{FmMod, FmModState}, Fallback, Rate, C32};
/// // Deviation of a quarter of the sample rate: an input of 1.0 turns 90 degrees per sample.
/// let mut state = FmModState::new(250.0, Rate::hz(1000)?)?;
/// let mut output = [C32::default(); 2];
/// Fallback.fm_mod(&mut state, &[1.0f32, 1.0], &mut output)?;
/// assert!((output[0].im - 1.0).abs() < 1e-6);
/// assert!((output[1].re + 1.0).abs() < 1e-6);
/// assert_eq!(state.phase(), 1 << 31);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FmModState {
    phase: u32,
    k: f64,
}

impl FmModState {
    /// Creates a modulator state with phase 0. `deviation_hz` must be finite and not
    /// negative, otherwise `InvalidParameter`.
    pub fn new(deviation_hz: f64, rate: Rate) -> Result<Self> {
        if !(deviation_hz.is_finite() && deviation_hz >= 0.0) {
            return Err(Error::InvalidParameter(
                "deviation must be finite and not negative",
            ));
        }
        Ok(Self {
            phase: 0,
            k: deviation_hz / rate.as_f64() * TWO_POW_32,
        })
    }

    /// Sets the phase back to 0 (the deviation is kept).
    pub fn reset(&mut self) {
        self.phase = 0;
    }

    /// The phase accumulator in units of 2π/2^32.
    pub fn phase(&self) -> u32 {
        self.phase
    }

    /// Sets the phase accumulator.
    pub fn set_phase(&mut self, phase: u32) {
        self.phase = phase;
    }

    /// The scale `k` in phase units per unit of amplitude.
    pub fn scale(&self) -> f64 {
        self.k
    }

    /// The scale and the phase accumulator at once, for kernel implementations in backend crates.
    pub fn parts_mut(&mut self) -> (f64, &mut u32) {
        (self.k, &mut self.phase)
    }
}

impl Timing for FmModState {
    fn ratio(&self) -> Ratio {
        Ratio::ONE
    }

    fn group_delay(&self) -> Delay {
        Delay::ZERO
    }
}

/// FM modulator: real baseband in, complex constant-envelope signal out.
pub trait FmMod<T: ComplexSample> {
    /// `phase += floor(x[n] * k + 0.5)` (computed in f64, as `i64`, wrapping into `u32`),
    /// then `output[n] = e^{j 2 pi phase / 2^32}`. `Q15` input is `x = q / 32768` in f64
    /// before the formula. The output is the cosine/sine of the phase from the table and
    /// interpolation of [`Rotate`](crate::kernels::mix::Rotate), which is `Rotate` applied to
    /// a unit vector without the final multiplication; a `CQ15` output therefore has the
    /// table amplitude `1 - 2^-15`. The lengths must be equal.
    fn fm_mod(&self, s: &mut FmModState, input: &[T::Real], output: &mut [T]) -> Result<()>;
}

/// AM modulator: real audio in, complex baseband with carrier out.
///
/// `C32` (input `f32`): `x' = clamp(x, -1, 1)` with NaN giving 0, then `t = depth·x'`,
/// `u = 1 + t`, `re = carrier·u` in `f32` in that order (no FMA), `im = 0`.
///
/// `CQ15` (input `Q15`): `c_q = round(carrier·32768)` and `m_q = round(depth·32768)`, each
/// limited to 32767 (the `f32` to `Q15` conversion of the numeric conventions), then
/// `t = round_shift(m_q·x, 15)`, `u = 32768 + t`, `re = sat16(round_shift(c_q·u, 15))` in
/// `i64`, `im = 0`.
///
/// `carrier` and `depth` must lie in `0 ..= 1` (NaN and the infinities are outside), else
/// `InvalidParameter`; unequal lengths give `LengthMismatch`. The parameters are checked
/// first, and nothing is written on an error. Stateless, ratio 1, delay 0, bit-exact class.
///
/// ```
/// use sdrk_core::{kernels::modulate::AmMod, Fallback, C32, CQ15, Q15};
/// let mut out = [C32::default(); 3];
/// Fallback.am_mod(0.5, 0.8, &[1.0, -1.0, f32::NAN], &mut out)?;
/// assert_eq!(out[0], C32::new(0.5 * (1.0 + 0.8), 0.0));
/// assert_eq!(out[1], C32::new(0.5 * (1.0 - 0.8), 0.0));
/// assert_eq!(out[2], C32::new(0.5, 0.0)); // NaN counts as 0: the bare carrier
///
/// let mut out = [CQ15::default(); 2];
/// Fallback.am_mod(1.0, 1.0, &[Q15::MIN, Q15::MAX], &mut out)?;
/// assert_eq!(out[0].re, Q15(1)); // the envelope touches 0 up to one LSB
/// assert_eq!(out[1].re, Q15::MAX); // saturated
/// assert!(Fallback.am_mod(1.5, 0.8, &[Q15(0)], &mut out[..1]).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait AmMod<T: ComplexSample> {
    /// `out[n] = (c·(1 + m·x[n]), 0)` with `x` clamped to `[−1, 1]`.
    fn am_mod(&self, carrier: f32, depth: f32, input: &[T::Real], output: &mut [T]) -> Result<()>;
}

/// `Ok` when `carrier` and `depth` both lie in `0 ..= 1`, else `InvalidParameter`.
pub(crate) fn check_am_params(carrier: f32, depth: f32) -> Result<()> {
    if !(0.0..=1.0).contains(&carrier) {
        return Err(Error::InvalidParameter("carrier must be 0 to 1"));
    }
    if !(0.0..=1.0).contains(&depth) {
        return Err(Error::InvalidParameter("depth must be 0 to 1"));
    }
    Ok(())
}
