//! Pre-emphasis and peak limiter in one pass (`preemph_limit`), the audio stage of an FM
//! transmitter.
//!
//! A first-order pre-emphasis `H(z) = (b0 + b1 z^-1) / (1 + a1 z^-1)` (the coefficients come from
//! `sdrk_blocks::design::preemphasis_coefs`, or `[1, 0, 0]` for none) is followed by a limiter
//! without look-ahead: the envelope jumps to every new peak (instant attack), stays for a hold
//! time and then falls with a fixed factor per sample (release). The sample is scaled by
//! `limit / envelope` whenever the envelope is above the limit, so no output sample is above
//! `limit`.
//!
//! The pre-emphasis can boost the treble by up to 56 dB (1000 us at 240 kHz), so the `Q15`
//! variant keeps the pre-emphasized value `v` in Q30 in an `i64` until the limiter has seen it.
//!
//! Both variants are bit-exact between `Fallback` and `Reference`. The limiter compares and
//! switches, so the expected values of the golden cases come from the exact float32 model and the
//! integer model (an exception to the f64 reference rule, as for `squelch`).
//!
//! # Numeric contract
//!
//! `H = floor(hold_s * fs + 0.5)` samples, `r = exp(-1 / (fs * release_s))` (f64) and `T` is the
//! limit.
//!
//! `f32` (every operation rounded once, no FMA): per sample `x` is the input clamped to
//! `[-1024, 1024]`, or `0.0` when it is NaN or infinite; `v = (b0 x + b1 x1) - a1 v1`; `x1 = x`; `v1 = flush_denormal(v)`; `m = |v|`;
//! if `m >= env` then `env = m` and `hold = H`, else if `hold > 0` then `hold -= 1`, else
//! `env = flush_denormal(env r)`; `y = v (T / env)` when `env > T`, else `y = v`; the output is
//! `y` clamped to `[-T, T]` (a NaN stays a NaN).
//!
//! `Q15` (`i64`, with `i128` for the products): the coefficients are Q24
//! (`floor(c 2^24 + 0.5)`), `v` and `env` are Q30. `v = round_shift((b0 x + b1 x1) 2^15 - a1 v1, 24)`
//! (in `i128`, saturated to `+-2^62`, which a stable filter never reaches); `m = |v|`; hold as in
//! `f32`; the release is `env = round_shift(env r_q31, 31)` with `r_q31 = floor(r 2^31 + 0.5)`
//! (the product in `i128`); with `T_q30 = floor(T 2^30 + 0.5)`: `y = div_round(v T_q30, env)` when
//! `env > T_q30` (the product in `i128`, `div_round(n, d) = sign(n) floor((|n| + floor(d / 2)) / d)`),
//! else `y = v`; the output is `sat16(round_shift(y, 15))` clamped to `+-T_q15` with
//! `T_q15 = min(floor(T 32768 + 0.5), 32767)`. The clamp keeps `-32768` from ever being an
//! output, also with the limiter idle at `limit` 1.0.
//!
//! The gain of the envelope after a release can be larger than `T / |v|` for the sample that
//! follows (the envelope falls below the current peak), which the final clamp catches.
//!
//! The clamp of the input (`2^10`, 60 dB above full scale) only matters for a corrupted input: with
//! it, `v` stays below about `1e6` for every coefficient set of `preemphasis_coefs` (`tau` up to
//! 1 ms, rates up to 384 kHz), so `v`, the envelope and `T / env` stay finite and the state cannot
//! become NaN. Without a clamp, a single finite input of `1e35` to `1e38` (depending on the
//! coefficients) made `v` infinite and the state NaN until `reset`. Inputs within the clamp give
//! the same bits as before. `Q15` needs no clamp: its input is a 16-bit integer and `v` saturates.

use core::marker::PhantomData;

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::timing::{Delay, Rate, Timing};
use crate::types::{RealSample, SampleType};

/// `2^24`: scale of the `Q15` coefficients.
const Q24: f64 = 16_777_216.0;
/// `2^30`: scale of `T_q30`.
const Q30: f64 = 1_073_741_824.0;
/// `2^31`: scale of the `Q15` release factor.
const Q31: f64 = 2_147_483_648.0;
/// Largest absolute coefficient (exclusive): `2^14`.
const COEF_LIMIT: f64 = 16_384.0;
/// Largest hold time in samples.
const MAX_HOLD: f64 = 2_147_483_648.0;

/// Saturation of the `Q15` pre-emphasized value in Q30 (`2^62`).
pub(crate) const V_LIMIT: i64 = 1 << 62;

/// Largest absolute `f32` input (`2^10`, 60 dB above full scale): a finite input beyond it is
/// clamped to `+-INPUT_LIMIT`.
pub(crate) const INPUT_LIMIT: f32 = 1024.0;

/// Mutable view of the numeric fields, used by the kernel implementations of this crate.
pub(crate) struct PreemphLimitParts<'a> {
    pub(crate) b0_f: f32,
    pub(crate) b1_f: f32,
    pub(crate) a1_f: f32,
    pub(crate) r_f: f32,
    pub(crate) t_f: f32,
    pub(crate) x1_f: &'a mut f32,
    pub(crate) v1_f: &'a mut f32,
    pub(crate) env_f: &'a mut f32,
    pub(crate) b0_q: i64,
    pub(crate) b1_q: i64,
    pub(crate) a1_q: i64,
    pub(crate) r_q: i64,
    pub(crate) t_q30: i64,
    pub(crate) t_q15: i64,
    pub(crate) x1_q: &'a mut i64,
    pub(crate) v1_q: &'a mut i64,
    pub(crate) env_q: &'a mut i64,
    pub(crate) hold_len: u32,
    pub(crate) hold: &'a mut u32,
}

/// State of the pre-emphasis and limiter: coefficients, the previous input and pre-emphasized
/// value, the envelope, the hold counter and the limit.
///
/// Both numeric representations (`f32`, and Q24/Q30 `i64` for `Q15`) are derived from the same
/// parameters, and only the one of the sample type `T` is used. The state starts at rest
/// (`x1 = v1 = 0`, envelope 0, hold 0).
///
/// ```
/// use sdrk_core::{kernels::preemph_limit::PreemphLimitState, Rate};
/// let state = PreemphLimitState::<f32>::new([1.0, 0.0, 0.0], 0.5, 0.02, 0.2, Rate::hz(48_000)?)?;
/// assert_eq!(state.gain(), 1.0);
/// assert_eq!(state.hold_samples(), 960);
/// // The limit must be in (0, 1].
/// assert!(PreemphLimitState::<f32>::new([1.0, 0.0, 0.0], 1.5, 0.02, 0.2, Rate::hz(48_000)?).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreemphLimitState<T: RealSample> {
    limit: f64,
    hold_len: u32,
    b0_f: f32,
    b1_f: f32,
    a1_f: f32,
    r_f: f32,
    t_f: f32,
    x1_f: f32,
    v1_f: f32,
    env_f: f32,
    b0_q: i64,
    b1_q: i64,
    a1_q: i64,
    r_q: i64,
    t_q30: i64,
    t_q15: i64,
    x1_q: i64,
    v1_q: i64,
    env_q: i64,
    hold: u32,
    marker: PhantomData<T>,
}

/// `floor(x + 0.5)` for a coefficient scaled by `scale`.
fn round_scaled(x: f64, scale: f64) -> i64 {
    libm::floor(x * scale + 0.5) as i64
}

/// `(T_f32, T_q30, T_q15)` of a validated limit.
fn limits(limit: f64) -> (f32, i64, i64) {
    let q15 = round_scaled(limit, 32_768.0).min(32_767);
    (limit as f32, round_scaled(limit, Q30), q15)
}

fn check_limit(limit: f64) -> Result<()> {
    if limit > 0.0 && limit <= 1.0 {
        Ok(())
    } else {
        Err(Error::InvalidParameter(
            "limit must be greater than 0 and at most 1",
        ))
    }
}

impl<T: RealSample> PreemphLimitState<T> {
    /// Creates a state at rest.
    ///
    /// `coefs` is `[b0, b1, a1]` of the pre-emphasis (from `preemphasis_coefs`, or `[1, 0, 0]`
    /// for none), `limit` the output peak in `(0, 1]`, `hold_s >= 0` the hold time and
    /// `release_s > 0` the release time constant at `rate`. Rejected with `InvalidParameter`
    /// (NaN included): a `limit` outside `(0, 1]`, a `release_s` that is not finite and positive,
    /// a `hold_s` that is negative or not finite, a hold time above `2^31` samples, a coefficient
    /// that is not finite or has `|c| >= 2^14`, and a release that never completes (the factor
    /// per sample rounds to 1 in the representation of `T`, `fs * release_s` at least `3.4e7`
    /// samples for `f32` and `4.3e9` for `Q15`).
    pub fn new(
        coefs: [f64; 3],
        limit: f64,
        hold_s: f64,
        release_s: f64,
        rate: Rate,
    ) -> Result<Self> {
        check_limit(limit)?;
        if !(release_s.is_finite() && release_s > 0.0) {
            return Err(Error::InvalidParameter(
                "release time must be finite and positive",
            ));
        }
        if !(hold_s.is_finite() && hold_s >= 0.0) {
            return Err(Error::InvalidParameter(
                "hold time must be finite and not negative",
            ));
        }
        let fs = rate.as_f64();
        let hold = libm::floor(hold_s * fs + 0.5);
        if hold.is_nan() || hold > MAX_HOLD {
            return Err(Error::InvalidParameter("hold time is too long"));
        }
        if !coefs.iter().all(|c| c.is_finite() && c.abs() < COEF_LIMIT) {
            return Err(Error::InvalidParameter(
                "pre-emphasis coefficients must be finite with an absolute value below 2^14",
            ));
        }
        let r = libm::exp(-1.0 / (fs * release_s));
        let r_f = r as f32;
        let r_q = round_scaled(r, Q31);
        let never = if T::TYPE == SampleType::Q15 {
            r_q >= 1 << 31
        } else {
            r_f >= 1.0
        };
        if never {
            return Err(Error::InvalidParameter(
                "release time is too long for the rate",
            ));
        }
        let (t_f, t_q30, t_q15) = limits(limit);
        Ok(Self {
            limit,
            hold_len: hold as u32,
            b0_f: coefs[0] as f32,
            b1_f: coefs[1] as f32,
            a1_f: coefs[2] as f32,
            r_f,
            t_f,
            x1_f: 0.0,
            v1_f: 0.0,
            env_f: 0.0,
            b0_q: round_scaled(coefs[0], Q24),
            b1_q: round_scaled(coefs[1], Q24),
            a1_q: round_scaled(coefs[2], Q24),
            r_q,
            t_q30,
            t_q15,
            x1_q: 0,
            v1_q: 0,
            env_q: 0,
            hold: 0,
            marker: PhantomData,
        })
    }

    /// Changes the limit; it applies from the next sample, the envelope and the hold counter are
    /// kept. The rules of [`PreemphLimitState::new`] apply; on an error nothing changes.
    ///
    /// ```
    /// use sdrk_core::{kernels::preemph_limit::PreemphLimitState, Rate};
    /// let mut state = PreemphLimitState::<f32>::new([1.0, 0.0, 0.0], 1.0, 0.0, 0.2, Rate::hz(48_000)?)?;
    /// state.set_limit(0.5)?;
    /// assert_eq!(state.limit(), 0.5);
    /// assert!(state.set_limit(0.0).is_err());
    /// assert_eq!(state.limit(), 0.5);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn set_limit(&mut self, limit: f64) -> Result<()> {
        check_limit(limit)?;
        let (t_f, t_q30, t_q15) = limits(limit);
        self.limit = limit;
        self.t_f = t_f;
        self.t_q30 = t_q30;
        self.t_q15 = t_q15;
        Ok(())
    }

    /// The limit.
    pub fn limit(&self) -> f64 {
        self.limit
    }

    /// The hold time in samples, `floor(hold_s * fs + 0.5)`.
    pub fn hold_samples(&self) -> u32 {
        self.hold_len
    }

    /// The current gain reduction for meters: `1.0` while the envelope is at or below the limit,
    /// otherwise `limit / envelope` computed from the stored values (the `f32` values widened, or
    /// `T_q30 / envelope` for `Q15`). The samples after a release can be scaled by a larger gain
    /// than a peak needs; the output clamp keeps them within the limit.
    ///
    /// ```
    /// use sdrk_core::{kernels::preemph_limit::{PreemphLimit, PreemphLimitState}, Fallback, Rate};
    /// let mut state = PreemphLimitState::<f32>::new([1.0, 0.0, 0.0], 0.5, 0.0, 0.2, Rate::hz(48_000)?)?;
    /// let mut out = [0.0f32; 1];
    /// Fallback.preemph_limit(&mut state, &[1.0], &mut out)?;
    /// assert_eq!(out[0], 0.5);
    /// assert_eq!(state.gain(), 0.5);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn gain(&self) -> f64 {
        if T::TYPE == SampleType::Q15 {
            if self.env_q > self.t_q30 {
                self.t_q30 as f64 / self.env_q as f64
            } else {
                1.0
            }
        } else if self.env_f > self.t_f {
            f64::from(self.t_f) / f64::from(self.env_f)
        } else {
            1.0
        }
    }

    /// Sets the previous input and pre-emphasized value, the envelope and the hold counter back
    /// to zero (coefficients, limit and times are kept).
    pub fn reset(&mut self) {
        self.x1_f = 0.0;
        self.v1_f = 0.0;
        self.env_f = 0.0;
        self.x1_q = 0;
        self.v1_q = 0;
        self.env_q = 0;
        self.hold = 0;
    }

    pub(crate) fn parts_mut(&mut self) -> PreemphLimitParts<'_> {
        PreemphLimitParts {
            b0_f: self.b0_f,
            b1_f: self.b1_f,
            a1_f: self.a1_f,
            r_f: self.r_f,
            t_f: self.t_f,
            x1_f: &mut self.x1_f,
            v1_f: &mut self.v1_f,
            env_f: &mut self.env_f,
            b0_q: self.b0_q,
            b1_q: self.b1_q,
            a1_q: self.a1_q,
            r_q: self.r_q,
            t_q30: self.t_q30,
            t_q15: self.t_q15,
            x1_q: &mut self.x1_q,
            v1_q: &mut self.v1_q,
            env_q: &mut self.env_q,
            hold_len: self.hold_len,
            hold: &mut self.hold,
        }
    }
}

impl<T: RealSample> Timing for PreemphLimitState<T> {
    fn ratio(&self) -> Ratio {
        Ratio::ONE
    }

    /// Zero; the real delay of the IIR filter depends on the frequency and the gain acts on the
    /// sample it looks at.
    fn group_delay(&self) -> Delay {
        Delay::ZERO
    }
}

/// Pre-emphasis and peak limiter.
///
/// ```
/// use sdrk_core::{kernels::preemph_limit::{PreemphLimit, PreemphLimitState}, Fallback, Rate};
/// let mut state = PreemphLimitState::<f32>::new([1.0, 0.0, 0.0], 0.5, 0.02, 0.2, Rate::hz(48_000)?)?;
/// let input = [0.25, -2.0, 0.25, 0.0];
/// let mut output = [0.0f32; 4];
/// Fallback.preemph_limit(&mut state, &input, &mut output)?;
/// assert_eq!(output[0], 0.25);
/// assert_eq!(output[1], -0.5);
/// // The gain stays down for the hold time.
/// assert_eq!(output[2], 0.0625);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait PreemphLimit<T: RealSample> {
    /// Pre-emphasizes `input` and limits the peaks, see the module documentation for the
    /// arithmetic. The lengths must be equal.
    fn preemph_limit(
        &self,
        state: &mut PreemphLimitState<T>,
        input: &[T],
        output: &mut [T],
    ) -> Result<()>;
}
