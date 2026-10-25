//! Power squelch: a gate that passes the signal only while its average power is high enough.

use core::marker::PhantomData;

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::num::q15_from_f64;
use crate::timing::{Delay, Rate, Timing};
use crate::types::{ComplexSample, SampleType};

/// Scale of the Q30 power values of the `CQ15` variant (`2^30`, full scale is 1.0).
const Q30: f64 = 1_073_741_824.0;

/// Mutable view of the numeric fields, used by the kernel implementations of this crate.
pub(crate) struct SquelchParts<'a> {
    pub(crate) alpha_f: f32,
    pub(crate) open_f: f32,
    pub(crate) close_f: f32,
    pub(crate) p_f: &'a mut f32,
    pub(crate) alpha_q: i64,
    pub(crate) open_q: i64,
    pub(crate) close_q: i64,
    pub(crate) p_q: &'a mut i64,
    pub(crate) open: &'a mut bool,
}

/// State of the power squelch: smoothing coefficient, average power `p`, the open and
/// close levels and the gate.
///
/// The state starts closed with `p = 0`. Both numeric representations (`f32` for
/// `C32`, Q30 `i64` for `CQ15`) are derived from the same parameters, and only the one
/// of the sample type `T` is used.
///
/// ```
/// use sdrk_core::{kernels::squelch::SquelchState, Rate, C32};
/// let state = SquelchState::<C32>::new(-40.0, 3.0, 0.01, Rate::hz(48_000)?)?;
/// assert!(!state.is_open());
/// assert!(SquelchState::<C32>::new(-40.0, -1.0, 0.01, Rate::hz(48_000)?).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SquelchState<T: ComplexSample> {
    threshold_db: f64,
    hysteresis_db: f64,
    alpha_f: f32,
    alpha_q: i64,
    open_f: f32,
    close_f: f32,
    open_q: i64,
    close_q: i64,
    p_f: f32,
    p_q: i64,
    open: bool,
    marker: PhantomData<T>,
}

/// `(P_open, P_close)` as `f32` (computed in f64 and cast) and as Q30 `i64`
/// (`floor(10^(db/10) * 2^30 + 0.5)`).
fn levels(threshold_db: f64, hysteresis_db: f64) -> (f32, f32, i64, i64) {
    let lin = |db: f64| libm::pow(10.0, db / 10.0);
    let q30 = |p: f64| libm::floor(p * Q30 + 0.5) as i64;
    let (open, close) = (lin(threshold_db), lin(threshold_db - hysteresis_db));
    (open as f32, close as f32, q30(open), q30(close))
}

fn check_levels(threshold_db: f64, hysteresis_db: f64) -> Result<()> {
    if !(threshold_db.is_finite() && (-150.0..=0.0).contains(&threshold_db)) {
        return Err(Error::InvalidParameter(
            "squelch threshold must be between -150 and 0 dB",
        ));
    }
    if !(hysteresis_db.is_finite() && hysteresis_db >= 0.0) {
        return Err(Error::InvalidParameter(
            "squelch hysteresis must be finite and not negative",
        ));
    }
    Ok(())
}

impl<T: ComplexSample> SquelchState<T> {
    /// Creates a closed state with `p = 0`.
    ///
    /// `threshold_db` is the open level in dBFS (-150 to 0, a full-scale complex sample
    /// `|x|^2 = 1` is 0 dBFS), `hysteresis_db >= 0` lowers the close level
    /// (`threshold_db - hysteresis_db`), `tau_s > 0` is the time constant of the power
    /// average at `rate`. The smoothing coefficient is
    /// `alpha = 1 - exp(-1 / (fs * tau))` in f64 and must satisfy `0 < alpha <= 0.5`.
    /// Anything else, including NaN, gives `InvalidParameter`.
    ///
    /// For `CQ15` the coefficient is `q15_from_f64(alpha)`; a coefficient below
    /// `2^-16` rounds to zero and then the average never moves, so use
    /// `fs * tau_s` below about 20 000 samples there.
    pub fn new(threshold_db: f64, hysteresis_db: f64, tau_s: f64, rate: Rate) -> Result<Self> {
        check_levels(threshold_db, hysteresis_db)?;
        if !(tau_s.is_finite() && tau_s > 0.0) {
            return Err(Error::InvalidParameter(
                "squelch time constant must be finite and positive",
            ));
        }
        let alpha = 1.0 - libm::exp(-1.0 / (rate.as_f64() * tau_s));
        if !(alpha > 0.0 && alpha <= 0.5) {
            return Err(Error::InvalidParameter(
                "squelch smoothing coefficient must be in (0, 0.5]",
            ));
        }
        let (open_f, close_f, open_q, close_q) = levels(threshold_db, hysteresis_db);
        Ok(Self {
            threshold_db,
            hysteresis_db,
            alpha_f: alpha as f32,
            alpha_q: i64::from(q15_from_f64(alpha).0),
            open_f,
            close_f,
            open_q,
            close_q,
            p_f: 0.0,
            p_q: 0,
            open: false,
            marker: PhantomData,
        })
    }

    /// Changes the open level and the hysteresis; the average power and the gate are
    /// kept and the new levels apply from the next sample. The arguments follow the
    /// rules of [`SquelchState::new`]; on an error nothing changes.
    ///
    /// ```
    /// use sdrk_core::{kernels::squelch::SquelchState, Rate, CQ15};
    /// let mut state = SquelchState::<CQ15>::new(-40.0, 3.0, 0.01, Rate::hz(48_000)?)?;
    /// state.set_thresholds(-30.0, 6.0)?;
    /// assert_eq!(state.threshold_db(), -30.0);
    /// assert!(state.set_thresholds(1.0, 6.0).is_err());
    /// assert_eq!(state.threshold_db(), -30.0);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn set_thresholds(&mut self, threshold_db: f64, hysteresis_db: f64) -> Result<()> {
        check_levels(threshold_db, hysteresis_db)?;
        let (open_f, close_f, open_q, close_q) = levels(threshold_db, hysteresis_db);
        self.threshold_db = threshold_db;
        self.hysteresis_db = hysteresis_db;
        self.open_f = open_f;
        self.close_f = close_f;
        self.open_q = open_q;
        self.close_q = close_q;
        Ok(())
    }

    /// True while the gate is open.
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Sets the average power back to zero and closes the gate (levels and
    /// coefficient are kept).
    pub fn reset(&mut self) {
        self.p_f = 0.0;
        self.p_q = 0;
        self.open = false;
    }

    /// The open level in dBFS.
    pub fn threshold_db(&self) -> f64 {
        self.threshold_db
    }

    /// The hysteresis in dB.
    pub fn hysteresis_db(&self) -> f64 {
        self.hysteresis_db
    }

    /// The average power as a linear value (full scale `|x|^2 = 1`), exact for both
    /// sample types (`f32` widened, or the Q30 integer divided by `2^30`).
    pub fn average_power(&self) -> f64 {
        if T::TYPE == SampleType::CQ15 {
            self.p_q as f64 / Q30
        } else {
            f64::from(self.p_f)
        }
    }

    pub(crate) fn parts_mut(&mut self) -> SquelchParts<'_> {
        SquelchParts {
            alpha_f: self.alpha_f,
            open_f: self.open_f,
            close_f: self.close_f,
            p_f: &mut self.p_f,
            alpha_q: self.alpha_q,
            open_q: self.open_q,
            close_q: self.close_q,
            p_q: &mut self.p_q,
            open: &mut self.open,
        }
    }
}

impl<T: ComplexSample> Timing for SquelchState<T> {
    fn ratio(&self) -> Ratio {
        Ratio::ONE
    }

    /// Zero; the gate acts on the sample it looks at.
    fn group_delay(&self) -> Delay {
        Delay::ZERO
    }
}

/// Power squelch.
///
/// ```
/// use sdrk_core::{kernels::squelch::{Squelch, SquelchState}, Fallback, Rate, C32};
/// let mut state = SquelchState::<C32>::new(-20.0, 3.0, 0.001, Rate::hz(48_000)?)?;
/// let mut input = [C32::new(0.0, 0.0); 1000];
/// input[100..300].fill(C32::new(0.5, 0.5));
/// let mut output = [C32::new(9.0, 9.0); 1000];
/// Fallback.squelch(&mut state, &input, &mut output)?;
/// assert_eq!(output[0], C32::new(0.0, 0.0));
/// assert_eq!(output[299], C32::new(0.5, 0.5));
/// assert!(!state.is_open());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Squelch<T: ComplexSample> {
    /// Per sample: update the average power, then the gate (open when `p >= P_open`,
    /// closed when `p < P_close`, otherwise unchanged), then write `input[n]` when the
    /// gate is open and zero when it is closed. The lengths must be equal.
    ///
    /// `C32`: `q = re*re + im*im`, `p = flush_denormal(p + alpha * (q - p))` in f32
    /// without FMA; a non-finite `q` leaves `p` and the gate unchanged. `CQ15`: `q` is
    /// the Q30 value `re^2 + im^2`, `p = p + round_shift(alpha_raw * (q - p), 15)` in
    /// `i64`. Both are bit-exact between `Fallback` and `Reference`.
    fn squelch(&self, state: &mut SquelchState<T>, input: &[T], output: &mut [T]) -> Result<()>;
}
