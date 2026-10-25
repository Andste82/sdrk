//! Automatic gain control: a multiplicative feedback loop that holds the output amplitude
//! at a set point.

use core::marker::PhantomData;

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::timing::{Delay, Rate, Timing};
use crate::types::{ComplexSample, SampleType};

/// Scale of the Q.32 gain of the `CQ15` variant (`2^32`).
const Q32: f64 = 4_294_967_296.0;
/// Scale of the Q.48 loop coefficient of the `CQ15` variant (`2^48`).
const Q48: f64 = 281_474_976_710_656.0;

/// Mutable view of the numeric fields, used by the kernel implementations of this crate.
pub(crate) struct AgcParts<'a> {
    pub(crate) ref_f: f32,
    pub(crate) ka_f: f32,
    pub(crate) kr_f: f32,
    pub(crate) gmin_f: f32,
    pub(crate) gmax_f: f32,
    pub(crate) g_f: &'a mut f32,
    pub(crate) g_d: &'a mut f64,
    pub(crate) ref_q: i64,
    pub(crate) ka_q: i64,
    pub(crate) kr_q: i64,
    pub(crate) gmin_q: i64,
    pub(crate) gmax_q: i64,
    pub(crate) g_q: &'a mut i64,
}

/// The constants derived from the parameters, in both numeric representations.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Consts {
    ref_f: f32,
    ka_f: f32,
    kr_f: f32,
    gmin_f: f32,
    gmax_f: f32,
    ref_q: i64,
    ka_q: i64,
    kr_q: i64,
    gmin_q: i64,
    gmax_q: i64,
}

fn lin(db: f64) -> f64 {
    libm::pow(10.0, db / 20.0)
}

fn q32(gain: f64) -> i64 {
    libm::floor(gain * Q32 + 0.5) as i64
}

/// Per-sample loop rate `1 - exp(-1 / (fs * tau))`, which must lie in `(0, 0.5]`.
fn loop_rate(rate: Rate, tau_s: f64, what: &'static str) -> Result<f64> {
    if !(tau_s.is_finite() && tau_s > 0.0) {
        return Err(Error::InvalidParameter(what));
    }
    let r = 1.0 - libm::exp(-1.0 / (rate.as_f64() * tau_s));
    if !(r > 0.0 && r <= 0.5) {
        return Err(Error::InvalidParameter(
            "agc time constant must give a per-sample rate in (0, 0.5]",
        ));
    }
    Ok(r)
}

fn check_gains(min_db: f64, max_db: f64) -> Result<()> {
    if !(min_db.is_finite() && max_db.is_finite() && min_db >= -40.0 && max_db <= 90.0) {
        return Err(Error::InvalidParameter(
            "agc gain limits must be finite and between -40 and 90 dB",
        ));
    }
    if min_db > max_db {
        return Err(Error::InvalidParameter(
            "agc minimum gain must not exceed the maximum gain",
        ));
    }
    Ok(())
}

fn consts(
    reference: f64,
    attack_s: f64,
    release_s: f64,
    min_db: f64,
    max_db: f64,
    rate: Rate,
) -> Result<Consts> {
    if !(reference.is_finite() && (0.03125..=1.0).contains(&reference)) {
        return Err(Error::InvalidParameter(
            "agc reference must be between 2^-5 and 1",
        ));
    }
    check_gains(min_db, max_db)?;
    let r_a = loop_rate(
        rate,
        attack_s,
        "agc attack time must be finite and positive",
    )?;
    let r_r = loop_rate(
        rate,
        release_s,
        "agc release time must be finite and positive",
    )?;
    let ref_q = libm::floor(reference * 32768.0 + 0.5) as i64;
    let k_q = |r: f64| libm::floor(r * Q48 / ref_q as f64 + 0.5) as i64;
    // A loop coefficient that rounds to zero would freeze the `CQ15` gain silently.
    if k_q(r_a) == 0 || k_q(r_r) == 0 {
        return Err(Error::InvalidParameter(
            "agc time constant is too long for the Q.48 loop coefficient",
        ));
    }
    Ok(Consts {
        ref_f: reference as f32,
        ka_f: (r_a / reference) as f32,
        kr_f: (r_r / reference) as f32,
        gmin_f: lin(min_db) as f32,
        gmax_f: lin(max_db) as f32,
        ref_q,
        ka_q: k_q(r_a),
        kr_q: k_q(r_r),
        gmin_q: q32(lin(min_db)),
        gmax_q: q32(lin(max_db)),
    })
}

/// State of the automatic gain control: the gain, its limits, the reference amplitude and
/// the attack and release constants.
///
/// Both numeric representations (`f32` for `C32`, a Q.32 `i64` for `CQ15`) are derived
/// from the same parameters, and only the one of the sample type `T` is used.
///
/// ```
/// use sdrk_core::{kernels::agc::AgcState, Rate, C32};
/// let state = AgcState::<C32>::new(0.5, 0.005, 0.05, -20.0, 60.0, 0.0, Rate::hz(48_000)?)?;
/// assert_eq!(state.gain_db(), 0.0);
/// // The reference must be between 2^-5 and 1.
/// assert!(AgcState::<C32>::new(2.0, 0.005, 0.05, -20.0, 60.0, 0.0, Rate::hz(48_000)?).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgcState<T: ComplexSample> {
    c: Consts,
    initial_db: f64,
    g_f: f32,
    /// The gain of the reference implementation (f64); the fallback leaves it alone.
    g_d: f64,
    g_q: i64,
    marker: PhantomData<T>,
}

impl<T: ComplexSample> AgcState<T> {
    /// Creates a state with the gain `initial_gain_db`.
    ///
    /// `reference` is the output amplitude `|y|` the loop settles at (linear, `2^-5` to
    /// `1`), `attack_s` and `release_s` are the time constants of the loop at `rate`
    /// (attack when the output is too loud, release when it is too quiet), the gains are
    /// in dB with `-40 <= min_gain_db <= initial_gain_db <= max_gain_db <= 90`. The
    /// per-sample rates `1 - exp(-1 / (fs * tau))` (f64) must lie in `(0, 0.5]`, i.e. a
    /// time constant of at least `1 / (fs * ln 2)`, and the `CQ15` loop coefficient
    /// `floor(r * 2^48 / reference_raw + 0.5)` must not be zero. Anything else, including
    /// NaN, gives `InvalidParameter` (kind `InvalidArgument`; `sdrk_core::Error` has no
    /// `OutOfRange`).
    pub fn new(
        reference: f64,
        attack_s: f64,
        release_s: f64,
        min_gain_db: f64,
        max_gain_db: f64,
        initial_gain_db: f64,
        rate: Rate,
    ) -> Result<Self> {
        let c = consts(
            reference,
            attack_s,
            release_s,
            min_gain_db,
            max_gain_db,
            rate,
        )?;
        if !(initial_gain_db >= min_gain_db && initial_gain_db <= max_gain_db) {
            return Err(Error::InvalidParameter(
                "agc initial gain must lie between the minimum and maximum gain",
            ));
        }
        let mut state = Self {
            c,
            initial_db: initial_gain_db,
            g_f: 0.0,
            g_d: 0.0,
            g_q: 0,
            marker: PhantomData,
        };
        state.reset();
        Ok(state)
    }

    /// Changes every parameter except the initial gain. The current gain is kept and
    /// clamped to the new range. The arguments follow the rules of [`AgcState::new`]; on
    /// an error nothing changes.
    ///
    /// ```
    /// use sdrk_core::{kernels::agc::AgcState, Rate, CQ15};
    /// let rate = Rate::hz(48_000)?;
    /// let mut state = AgcState::<CQ15>::new(0.5, 0.005, 0.05, -20.0, 60.0, 30.0, rate)?;
    /// state.set_params(0.5, 0.01, 0.1, -20.0, 20.0, rate)?;
    /// assert!((state.gain_db() - 20.0).abs() < 1e-6);
    /// assert!(state.set_params(0.5, 0.01, 0.1, 30.0, 20.0, rate).is_err());
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn set_params(
        &mut self,
        reference: f64,
        attack_s: f64,
        release_s: f64,
        min_gain_db: f64,
        max_gain_db: f64,
        rate: Rate,
    ) -> Result<()> {
        self.c = consts(
            reference,
            attack_s,
            release_s,
            min_gain_db,
            max_gain_db,
            rate,
        )?;
        self.g_f = self.g_f.max(self.c.gmin_f).min(self.c.gmax_f);
        self.g_d = self
            .g_d
            .max(f64::from(self.c.gmin_f))
            .min(f64::from(self.c.gmax_f));
        self.g_q = self.g_q.max(self.c.gmin_q).min(self.c.gmax_q);
        Ok(())
    }

    /// The current gain in dB (`20 * log10(g)`).
    pub fn gain_db(&self) -> f64 {
        let g = if T::TYPE == SampleType::CQ15 {
            self.g_q as f64 / Q32
        } else {
            f64::from(self.g_f)
        };
        20.0 * libm::log10(g)
    }

    /// Sets the gain back to the initial gain of [`AgcState::new`], clamped to the
    /// current range (the parameters are kept).
    pub fn reset(&mut self) {
        let g = lin(self.initial_db);
        self.g_f = (g as f32).max(self.c.gmin_f).min(self.c.gmax_f);
        self.g_d = f64::from(self.g_f);
        self.g_q = q32(g).max(self.c.gmin_q).min(self.c.gmax_q);
    }

    pub(crate) fn parts_mut(&mut self) -> AgcParts<'_> {
        AgcParts {
            ref_f: self.c.ref_f,
            ka_f: self.c.ka_f,
            kr_f: self.c.kr_f,
            gmin_f: self.c.gmin_f,
            gmax_f: self.c.gmax_f,
            g_f: &mut self.g_f,
            g_d: &mut self.g_d,
            ref_q: self.c.ref_q,
            ka_q: self.c.ka_q,
            kr_q: self.c.kr_q,
            gmin_q: self.c.gmin_q,
            gmax_q: self.c.gmax_q,
            g_q: &mut self.g_q,
        }
    }
}

impl<T: ComplexSample> Timing for AgcState<T> {
    fn ratio(&self) -> Ratio {
        Ratio::ONE
    }

    /// Zero; the gain acts on the sample it is applied to.
    fn group_delay(&self) -> Delay {
        Delay::ZERO
    }
}

/// Automatic gain control.
///
/// ```
/// use sdrk_core::{kernels::agc::{Agc, AgcState}, Fallback, Rate, C32};
/// let mut state = AgcState::<C32>::new(0.5, 0.001, 0.01, -20.0, 60.0, 0.0, Rate::hz(48_000)?)?;
/// let input = [C32::new(0.01, 0.0); 4800];
/// let mut output = [C32::default(); 4800];
/// Fallback.agc(&mut state, &input, &mut output)?;
/// // A weak signal is amplified towards the reference amplitude.
/// assert!(output[4799].re > 0.4);
/// assert!(state.gain_db() > 30.0);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Agc<T: ComplexSample> {
    /// Per sample: write `y = g * x`, measure `m = |y|`, then update the gain
    /// multiplicatively, `g = g * (1 + (r / reference) * e)` with the error
    /// `e = reference - m` and `r` the attack rate when `e < 0` and the release rate
    /// otherwise, and clamp `g` to the gain limits. Near the set point this is a
    /// first-order loop with the time constant `1 / r` samples, independent of the input
    /// level. The lengths must be equal.
    ///
    /// `C32`: `f32` arithmetic without FMA in this order; a NaN `m` leaves the gain
    /// unchanged, an infinite one drives it to the minimum. `CQ15`: the gain is a Q.32
    /// `i64`, `y = sat16(round_shift(x * g, 32))`, `m` is the rounded integer square root
    /// of `re^2 + im^2` (not saturated) and the step `(g * e * k + 2^47) >> 48` is
    /// computed in `i128` with `k` a Q.48 coefficient; bit-exact between `Fallback` and
    /// `Reference`.
    fn agc(&self, s: &mut AgcState<T>, input: &[T], output: &mut [T]) -> Result<()>;
}
