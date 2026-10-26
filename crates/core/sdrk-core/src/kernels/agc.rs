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
    pub(crate) lo_f: f32,
    pub(crate) hi_f: f32,
    pub(crate) ra_f: f32,
    pub(crate) rr_f: f32,
    pub(crate) gmin_f: f32,
    pub(crate) gmax_f: f32,
    /// The constants of the update forms of the fallback (see [`Derived`]).
    pub(crate) d: Derived,
    pub(crate) g_f: &'a mut f32,
    pub(crate) g_d: &'a mut f64,
    pub(crate) ref_q: i64,
    pub(crate) lo_q: i64,
    pub(crate) hi_q: i64,
    pub(crate) ra_q: i64,
    pub(crate) rr_q: i64,
    pub(crate) gmin_q: i64,
    pub(crate) gmax_q: i64,
    pub(crate) g_q: &'a mut i64,
}

/// The constants of the `C32` update in the forms that the fallback evaluates (the same
/// function as the specification, see `fallback/agc.rs`), each the f64 value computed from the
/// f64 values of the f32 constants, rounded once to f32.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Derived {
    /// `r_r ref`: release from `lo`, `g + (r_r ref / |x| - r_r g)`.
    pub(crate) cr: f32,
    /// `r_r / lo`: release below `lo`, `g + g ((r_r / lo) (ref - m))`.
    pub(crate) br: f32,
}

impl Derived {
    fn new(reference: f32, lo: f32, r_r: f32) -> Self {
        let (reference, lo, r_r) = (f64::from(reference), f64::from(lo), f64::from(r_r));
        Self {
            cr: (r_r * reference) as f32,
            br: (r_r / lo) as f32,
        }
    }
}

/// The constants derived from the parameters, in both numeric representations.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Consts {
    ref_f: f32,
    lo_f: f32,
    hi_f: f32,
    ra_f: f32,
    rr_f: f32,
    gmin_f: f32,
    gmax_f: f32,
    d: Derived,
    ref_q: i64,
    lo_q: i64,
    hi_q: i64,
    ra_q: i64,
    rr_q: i64,
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
    let k_q = |r: f64| libm::floor(r * Q48 + 0.5) as i64;
    // A loop coefficient that rounds to zero would freeze the `CQ15` gain silently.
    if k_q(r_a) == 0 || k_q(r_r) == 0 {
        return Err(Error::InvalidParameter(
            "agc time constant is too long for the Q.48 loop coefficient",
        ));
    }
    let ref_f = reference as f32;
    // The products are rounded in f32, as every constant.
    let lo_f = ref_f * 0.25;
    let rate_r = r_r as f32;
    Ok(Consts {
        ref_f,
        lo_f,
        hi_f: ref_f * 1.5,
        ra_f: r_a as f32,
        rr_f: rate_r,
        gmin_f: lin(min_db) as f32,
        gmax_f: lin(max_db) as f32,
        d: Derived::new(ref_f, lo_f, rate_r),
        ref_q,
        lo_q: ref_q >> 2,
        hi_q: ref_q + (ref_q >> 1),
        ra_q: k_q(r_a),
        rr_q: k_q(r_r),
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
    /// `floor(r * 2^48 + 0.5)` must not be zero. Anything else, including
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
            lo_f: self.c.lo_f,
            hi_f: self.c.hi_f,
            ra_f: self.c.ra_f,
            rr_f: self.c.rr_f,
            gmin_f: self.c.gmin_f,
            gmax_f: self.c.gmax_f,
            d: self.c.d,
            g_f: &mut self.g_f,
            g_d: &mut self.g_d,
            ref_q: self.c.ref_q,
            lo_q: self.c.lo_q,
            hi_q: self.c.hi_q,
            ra_q: self.c.ra_q,
            rr_q: self.c.rr_q,
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
    /// Per sample: write `y = g * x`, measure `m = |y|`, then update the gain with the
    /// error `e = reference - m` (`u = m / reference`) and clamp it to the gain limits:
    ///
    /// - too loud (`e < 0`, attack rate `r_a`): `g = g * d / (d + r_a * (m - reference))`
    ///   with `d = min(m, 1.5 * reference)`;
    /// - too quiet (`e >= 0`, release rate `r_r`): `g = g * (d + r_r * e) / d` with
    ///   `d = max(m, reference / 4)`.
    ///
    /// For `0.25 <= u <= 1.5` this is `g * (1 + r * e / m)` to first order, i.e. the error is
    /// normalised by the OUTPUT level (the update of GNU Radio's `agc3_cc`): a release is
    /// exactly exponential towards the set point, `u` approaches 1 by `r_r` per sample at
    /// any starting level, and a step of 30 dB settles within 1 dB after 5 time
    /// constants in both directions. Beyond that range the divisor is limited, which
    /// keeps the loop fast for big steps and bounded for silence. Near the set point the
    /// loop is first order with the time constant `1 / r` samples, independent of the
    /// input level. The attack form is divisive, so the factor stays in `(0, 1)` for any
    /// overshoot and the loop cannot collapse for a large `r_a * u`. An input sample of
    /// exactly zero (silence, a closed squelch) carries no information: the gain is
    /// held. The lengths must be equal.
    ///
    /// `C32`: `f32` arithmetic without FMA, with `m` computed as `g * |x|` and the release
    /// in two rearranged forms of the same function (one per side of the divisor limit, see
    /// `docs/kernels.md`); a NaN or zero `m` leaves the gain unchanged, an infinite one
    /// drives it to the minimum. `CQ15`: the gain is a
    /// Q.32 `i64`, `y = sat16(round_shift(x * g, 32))`, `m` is the rounded integer square
    /// root of `re^2 + im^2` (not saturated) and the new gain is computed in `i128` with
    /// Q.48 rates and rounded half up; bit-exact between `Fallback` and `Reference`.
    fn agc(&self, s: &mut AgcState<T>, input: &[T], output: &mut [T]) -> Result<()>;
}
