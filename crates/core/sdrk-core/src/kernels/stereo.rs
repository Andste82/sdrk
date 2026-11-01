//! FM stereo: the pilot phase-locked loop with the 38 kHz demodulation (`stereo_pll`) and the
//! sum/difference matrix (`stereo_matrix`).
//!
//! The multiplex signal (ITU-R BS.450) is
//! `m(t) = 0.45 (L + R) + 0.45 (L - R) sin(2 w_p t) + a_p sin(w_p t)` with `w_p = 2 pi 19 kHz`
//! and the pilot amplitude `a_p` of about 0.1. The loop locks to the pilot (the multiplex
//! signal through a band-pass around 19 kHz), regenerates `sin(2 theta)` from its phase and
//! multiplies the multiplex signal with it. After the audio low-pass, the sum `S` and the
//! difference `D` give `L = S + 2 D` and `R = S - 2 D` ([`StereoMatrix`]).

#![allow(clippy::similar_names)]

use core::marker::PhantomData;

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::timing::{Delay, Rate, Timing};
use crate::types::{ComplexSample, RealSample, SampleType};

/// Limit of the integrator in hertz (BS.450 allows +-2 Hz of pilot deviation).
const IMAX_HZ: f64 = 50.0;
/// Time constant of the lock detector in seconds.
const LOCK_TAU_S: f64 = 0.05;
/// `2^64` as f64: scale of the integrator and limit of the `Q15` variant.
const Q64: f64 = 18_446_744_073_709_551_616.0;
/// `2^34` as f64: scale of the `Q15` loop gains.
const Q34: f64 = 17_179_869_184.0;
/// `2^31` as f64: scale of the `Q15` lock coefficient.
const Q31: f64 = 2_147_483_648.0;
/// `2^61` as f64: scale of the `Q15` lock level.
const Q61: f64 = 2_305_843_009_213_693_952.0;
/// `2^32` as f64: one turn in phase units.
const TURN: f64 = 4_294_967_296.0;

/// The loop coefficients exactly as computed by [`StereoPllState::new`] in f64, before the
/// conversion to `f32` or to the `Q15` integers.
///
/// ```
/// use sdrk_core::{kernels::stereo::StereoPllState, Rate};
/// let st = StereoPllState::<f32>::new(19_000.0, 50.0, 0.1, Rate::hz(192_000)?)?;
/// let c = st.coefficients();
/// assert_eq!(c.step0, (19_000.0_f64 / 192_000.0 * 4294967296.0 + 0.5).floor() as u32);
/// assert!(c.kp > 0.0 && c.ki > 0.0);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoPllCoefficients {
    /// Proportional gain in turns per sample per unit of detector output.
    pub kp: f64,
    /// Integral gain in turns per sample per unit of detector output.
    pub ki: f64,
    /// Integrator limit in turns per sample.
    pub imax: f64,
    /// Smoothing coefficient of the lock detector.
    pub beta: f64,
    /// Level at or above which the stereo decision switches on.
    pub on: f64,
    /// Level below which the stereo decision switches off.
    pub off: f64,
    /// Nominal phase increment in units of `2^-32` turn.
    pub step0: u32,
}

/// Mutable view of the numeric fields, used by the kernel implementations of this crate.
pub(crate) struct StereoParts<'a> {
    pub(crate) kp: f32,
    pub(crate) ki: f32,
    pub(crate) imax: f32,
    pub(crate) beta: f32,
    pub(crate) on: f32,
    pub(crate) off: f32,
    pub(crate) kp_q: i64,
    pub(crate) ki_q: i64,
    pub(crate) imax_q: i64,
    pub(crate) beta_q31: i64,
    pub(crate) on_q61: i64,
    pub(crate) off_q61: i64,
    pub(crate) step0: u32,
    pub(crate) enabled: bool,
    pub(crate) phase: &'a mut u32,
    pub(crate) integ: &'a mut f32,
    pub(crate) level: &'a mut f32,
    pub(crate) integ_q: &'a mut i64,
    pub(crate) level_q61: &'a mut i64,
    pub(crate) stereo: &'a mut bool,
}

/// State of the stereo pilot loop: the 32-bit phase, the integrator, the lock level, the
/// stereo decision, the enable flag and the coefficients.
///
/// Both numeric representations (`f32` and the `i64` integers of `Q15`) are derived from the
/// same parameters, and only the one of the sample type `T` is used.
///
/// ```
/// use sdrk_core::{kernels::stereo::StereoPllState, Q15, Rate};
/// let rate = Rate::hz(480_000)?;
/// let st = StereoPllState::<f32>::new(19_000.0, 20.0, 0.1, rate)?;
/// assert!(!st.is_stereo() && st.phase() == 0);
/// assert!(StereoPllState::<f32>::new(120_000.0, 20.0, 0.1, rate).is_err());
/// assert!(StereoPllState::<Q15>::new(19_000.0, 20.0, 0.0625, rate).is_ok());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoPllState<T: RealSample> {
    coef: StereoPllCoefficients,
    kp: f32,
    ki: f32,
    imax: f32,
    beta: f32,
    on: f32,
    off: f32,
    kp_q: i64,
    ki_q: i64,
    imax_q: i64,
    beta_q31: i64,
    on_q61: i64,
    off_q61: i64,
    phase: u32,
    integ: f32,
    level: f32,
    integ_q: i64,
    level_q61: i64,
    stereo: bool,
    enabled: bool,
    marker: PhantomData<T>,
}

fn invalid(msg: &'static str) -> Error {
    Error::InvalidParameter(msg)
}

fn q(value: f64, scale: f64) -> Option<i64> {
    let v = libm::floor(value * scale + 0.5);
    // `i64::MAX as f64` is 2^63, which does not fit.
    if v.abs() < 9_223_372_036_854_775_808.0 {
        Some(v as i64)
    } else {
        None
    }
}

impl<T: RealSample> StereoPllState<T> {
    /// Creates a state with phase 0, integrator 0, level 0, not stereo and enabled.
    ///
    /// `pilot_hz` is the nominal pilot frequency (19 000), `natural_hz` the natural frequency
    /// of the loop, `pilot_level` the nominal pilot amplitude in units of the input signal
    /// (`f32`: 0.1; `Q15`: 0.1 times the discriminator scale). Requirements:
    /// `0 < pilot_hz < fs / 4`, `0 < natural_hz <= pilot_hz / 100`, `0 < pilot_level <= 1`;
    /// NaN and infinities are rejected.
    ///
    /// The coefficients are computed in f64 (`kd = pilot_level / 2`, damping `1 / sqrt 2`,
    /// `wn = 2 pi natural_hz / fs`, `kp = 2 zeta wn / (kd 2 pi)`, `ki = wn^2 / (kd 2 pi)`,
    /// `step0 = floor(pilot_hz / fs 2^32 + 0.5)`, `imax = 50 / fs`,
    /// `beta = 1 - exp(-1 / (fs 0.05))`, `on = 0.4 kd`, `off = 0.3 kd`). The `f32` variant
    /// also needs finite non-zero `f32` values of `kp`, `ki`, `beta`, `on` and `off`. The
    /// `Q15` variant returns `InvalidParameter` when `kp_q 2^30 + imax_q` or
    /// `ki_q 2^30 + imax_q` does not fit in `i64`, or when `ki_q` or the Q31 lock coefficient
    /// is zero.
    ///
    /// ```
    /// use sdrk_core::{kernels::stereo::StereoPllState, Rate};
    /// let rate = Rate::hz(192_000)?;
    /// assert!(StereoPllState::<f32>::new(19_000.0, 50.0, 0.1, rate).is_ok());
    /// // natural_hz above pilot_hz / 100, a zero level and NaN are rejected
    /// assert!(StereoPllState::<f32>::new(19_000.0, 200.0, 0.1, rate).is_err());
    /// assert!(StereoPllState::<f32>::new(19_000.0, 50.0, 0.0, rate).is_err());
    /// assert!(StereoPllState::<f32>::new(f64::NAN, 50.0, 0.1, rate).is_err());
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn new(pilot_hz: f64, natural_hz: f64, pilot_level: f64, rate: Rate) -> Result<Self> {
        let fs = rate.as_f64();
        if !(pilot_hz.is_finite() && pilot_hz > 0.0 && pilot_hz < fs / 4.0) {
            return Err(invalid(
                "stereo pilot frequency must be positive and below a quarter of the sample rate",
            ));
        }
        if !(natural_hz.is_finite() && natural_hz > 0.0 && natural_hz <= pilot_hz / 100.0) {
            return Err(invalid(
                "stereo loop natural frequency must be positive and at most a hundredth of the pilot frequency",
            ));
        }
        if !(pilot_level.is_finite() && pilot_level > 0.0 && pilot_level <= 1.0) {
            return Err(invalid("stereo pilot level must be in (0, 1]"));
        }
        let two_pi = 2.0 * core::f64::consts::PI;
        let kd = pilot_level / 2.0;
        let zeta = 1.0 / libm::sqrt(2.0);
        let wn = two_pi * natural_hz / fs;
        let kp = 2.0 * zeta * wn / (kd * two_pi);
        let ki = wn * wn / (kd * two_pi);
        let imax = IMAX_HZ / fs;
        let beta = 1.0 - libm::exp(-1.0 / (fs * LOCK_TAU_S));
        let (on, off) = (0.4 * kd, 0.3 * kd);
        let step0 = libm::floor(pilot_hz / fs * TURN + 0.5) as u32;
        let coef = StereoPllCoefficients {
            kp,
            ki,
            imax,
            beta,
            on,
            off,
            step0,
        };
        let finite_f32 = |v: f64| (v as f32).is_finite() && v as f32 != 0.0;
        if !(finite_f32(kp)
            && finite_f32(ki)
            && finite_f32(imax)
            && finite_f32(beta)
            && finite_f32(on)
            && finite_f32(off))
        {
            return Err(invalid(
                "stereo loop coefficients do not fit in f32 for this combination of rate and level",
            ));
        }
        let (kp_q, ki_q, imax_q, beta_q31, on_q61, off_q61) = if T::TYPE == SampleType::Q15 {
            let conv = (|| {
                let (kp_q, ki_q, imax_q) = (q(kp, Q34)?, q(ki, Q34)?, q(imax, Q64)?);
                let (beta_q31, on_q61, off_q61) = (q(beta, Q31)?, q(on, Q61)?, q(off, Q61)?);
                // |e| <= 2^30 in the Q15 loop; neither `integ + ki_q e` nor `kp_q e + integ` may
                // overflow.
                kp_q.checked_mul(1 << 30)?.checked_add(imax_q)?;
                ki_q.checked_mul(1 << 30)?.checked_add(imax_q)?;
                Some((kp_q, ki_q, imax_q, beta_q31, on_q61, off_q61))
            })();
            let Some(v) = conv else {
                return Err(invalid("stereo loop gain does not fit in the Q15 integers"));
            };
            if v.1 == 0 || v.3 == 0 {
                return Err(invalid(
                    "stereo loop gain or lock coefficient rounds to zero in Q15",
                ));
            }
            v
        } else {
            (0, 0, 0, 0, 0, 0)
        };
        Ok(Self {
            coef,
            kp: kp as f32,
            ki: ki as f32,
            imax: imax as f32,
            beta: beta as f32,
            on: on as f32,
            off: off as f32,
            kp_q,
            ki_q,
            imax_q,
            beta_q31,
            on_q61,
            off_q61,
            phase: 0,
            integ: 0.0,
            level: 0.0,
            integ_q: 0,
            level_q61: 0,
            stereo: false,
            enabled: true,
            marker: PhantomData,
        })
    }

    /// Enables or disables the difference output. While disabled the imaginary output is
    /// zero; the loop and the lock decision keep running, and enabling restores the output
    /// from the next sample.
    ///
    /// ```
    /// use sdrk_core::{kernels::stereo::StereoPllState, Rate};
    /// let mut st = StereoPllState::<f32>::new(19_000.0, 20.0, 0.1, Rate::hz(480_000)?)?;
    /// st.set_enabled(false);
    /// assert!(!st.is_enabled());
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// True while the difference output is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// The lock decision of the last sample (independent of [`StereoPllState::set_enabled`]).
    pub fn is_stereo(&self) -> bool {
        self.stereo
    }

    /// The phase of the loop in units of `2^-32` turn.
    pub fn phase(&self) -> u32 {
        self.phase
    }

    /// The f64 coefficients of the loop (see [`StereoPllState::new`]).
    pub fn coefficients(&self) -> StereoPllCoefficients {
        self.coef
    }

    /// Phase 0, integrator 0, level 0, not stereo; `enabled` is kept.
    ///
    /// ```
    /// use sdrk_core::{kernels::stereo::StereoPllState, Rate};
    /// let mut st = StereoPllState::<f32>::new(19_000.0, 20.0, 0.1, Rate::hz(480_000)?)?;
    /// st.set_enabled(false);
    /// st.reset();
    /// assert!(!st.is_enabled() && st.phase() == 0 && !st.is_stereo());
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn reset(&mut self) {
        self.phase = 0;
        self.integ = 0.0;
        self.level = 0.0;
        self.integ_q = 0;
        self.level_q61 = 0;
        self.stereo = false;
    }

    /// The integrator in turns per sample (for tests and diagnostics): the `f32` value
    /// widened, or the `i64` integer divided by `2^64`.
    pub fn integrator(&self) -> f64 {
        if T::TYPE == SampleType::Q15 {
            self.integ_q as f64 / Q64
        } else {
            f64::from(self.integ)
        }
    }

    /// The integrator limit in turns per sample that the active sample type clamps to: the
    /// `f32` limit widened, or `imax_q / 2^64` for `Q15`.
    pub fn integrator_limit(&self) -> f64 {
        if T::TYPE == SampleType::Q15 {
            self.imax_q as f64 / Q64
        } else {
            f64::from(self.imax)
        }
    }

    /// The smoothed detector level (`kd` when locked at the nominal level): the `f32` value
    /// widened, or the Q61 integer divided by `2^61`.
    pub fn lock_level(&self) -> f64 {
        if T::TYPE == SampleType::Q15 {
            self.level_q61 as f64 / Q61
        } else {
            f64::from(self.level)
        }
    }

    pub(crate) fn parts_mut(&mut self) -> StereoParts<'_> {
        StereoParts {
            kp: self.kp,
            ki: self.ki,
            imax: self.imax,
            beta: self.beta,
            on: self.on,
            off: self.off,
            kp_q: self.kp_q,
            ki_q: self.ki_q,
            imax_q: self.imax_q,
            beta_q31: self.beta_q31,
            on_q61: self.on_q61,
            off_q61: self.off_q61,
            step0: self.coef.step0,
            enabled: self.enabled,
            phase: &mut self.phase,
            integ: &mut self.integ,
            level: &mut self.level,
            integ_q: &mut self.integ_q,
            level_q61: &mut self.level_q61,
            stereo: &mut self.stereo,
        }
    }
}

impl<T: RealSample> Timing for StereoPllState<T> {
    fn ratio(&self) -> Ratio {
        Ratio::ONE
    }

    /// Zero; the output belongs to the input sample at the same index.
    fn group_delay(&self) -> Delay {
        Delay::ZERO
    }
}

/// Pilot phase-locked loop and 38 kHz demodulation.
///
/// ```
/// use sdrk_core::{kernels::stereo::{StereoPll, StereoPllState}, Fallback, Rate, C32};
/// let mut st = StereoPllState::<f32>::new(19_000.0, 20.0, 0.1, Rate::hz(480_000)?)?;
/// let mpx = [0.5_f32; 64];
/// let pilot = [0.0_f32; 64];
/// let mut out = [C32::new(9.0, 9.0); 64];
/// Fallback.stereo_pll(&mut st, &mpx, &pilot, &mut out)?;
/// // no pilot: not stereo, the difference channel is zero
/// assert_eq!(out[63], C32::new(0.5, 0.0));
/// assert!(!st.is_stereo());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait StereoPll<T: RealSample> {
    /// `mpx` is the multiplex signal, `pilot` the same signal through the pilot band-pass;
    /// `output[n] = (mpx[n], mpx[n] sin(2 theta))` while the loop is in the stereo state and
    /// enabled, else `(mpx[n], 0)`. The three slices must have equal lengths, otherwise
    /// `LengthMismatch` is returned and nothing changes.
    ///
    /// Sine and cosine come from the cosine tables of `rotate` with its interpolation, and
    /// `sin(2 theta)` from the phase `2 theta` (`phase.wrapping_mul(2)`). Per sample (`f32`,
    /// no FMA; a non-finite `pilot[n]` gives `e = q = 0` and leaves the level unchanged):
    /// `e = p c`, `q = p s`; `level = flush(level + beta (q - level))`; stereo on at
    /// `level >= on`, off at `level < off`; `integ = clamp(flush(integ + ki e), +-imax)`;
    /// `f = kp e + integ`; `phase += step0 + floor(f 2^32 + 0.5)` (wrapping). `Q15` does the
    /// same in `i64` (Q30 detector, Q61 level, integrator in units of `2^-64` turn); see
    /// `docs/kernels.md`. Both variants are bit-exact between `Fallback` and `Reference`.
    fn stereo_pll(
        &self,
        state: &mut StereoPllState<T>,
        mpx: &[T],
        pilot: &[T],
        output: &mut [T::Complex],
    ) -> Result<()>;
}

/// Sum and difference to left and right.
///
/// ```
/// use sdrk_core::{kernels::stereo::StereoMatrix, Fallback, C32};
/// let input = [C32::new(0.5, 0.125)];
/// let mut output = [C32::new(0.0, 0.0)];
/// Fallback.stereo_matrix(&input, &mut output)?;
/// assert_eq!(output[0], C32::new(0.75, 0.25));
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait StereoMatrix<T: ComplexSample> {
    /// `input`: `re` = S (sum), `im` = D (difference after demodulation); `output`: `re` =
    /// S + 2D (left), `im` = S - 2D (right). Stateless; equal lengths, otherwise
    /// `LengthMismatch` and nothing changes.
    ///
    /// `C32`: `d2 = d + d`, `l = s + d2`, `r = s - d2` in f32. `CQ15`: `d2 = 2 d` in `i64`,
    /// `l = sat16(s + d2)`, `r = sat16(s - d2)`.
    fn stereo_matrix(&self, input: &[T], output: &mut [T]) -> Result<()>;
}
