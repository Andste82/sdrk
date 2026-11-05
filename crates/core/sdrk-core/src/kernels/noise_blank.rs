//! Noise blanker: removes short impulses from a complex signal by setting them to zero.
//!
//! The blanker compares the power of every sample with a running average of the power. A
//! sample whose power exceeds the average by the threshold is an impulse: the output is set
//! to zero from the oldest sample of a delay line of `D` samples (the lookahead, which covers
//! the rising edge of the impulse) until `H` samples (the hold) after the last impulse sample.
//! Everything else passes unchanged, delayed by `D`.

use core::marker::PhantomData;

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::timing::{Delay, Rate, Timing};
use crate::types::{ComplexSample, SampleType};

/// Scale of the Q31 smoothing coefficient of the `CQ15` variant (`2^31`).
const Q31: f64 = 2_147_483_648.0;
/// Scale of the Q16 threshold factor of the `CQ15` variant (`2^16`).
const Q16: f64 = 65_536.0;
/// Largest lookahead in seconds.
const MAX_LOOKAHEAD_S: f64 = 0.001;

/// Mutable view of the numeric fields, used by the kernel implementations of this crate.
pub(crate) struct NoiseBlankParts<'a, T> {
    pub(crate) hist: &'a mut [T],
    pub(crate) alpha_f: f32,
    pub(crate) k_f: f32,
    pub(crate) alpha_q: i64,
    pub(crate) k_q: i64,
    /// `D + H + 1`: the counter value an impulse sets.
    pub(crate) span: u64,
    pub(crate) p_f: &'a mut f32,
    pub(crate) p_q: &'a mut i64,
    pub(crate) count: &'a mut u64,
    pub(crate) blanked: &'a mut u64,
}

/// State of the noise blanker: the delay line of `D` samples, the average power `p`, the
/// blanking counter `c` and the coefficients.
///
/// `D` is the length of the caller's history storage, `D = round(lookahead_s * fs)` for a
/// lookahead of 0 to 1 ms. The state starts with a zero delay line, `p = 0` and `c = 0`.
/// Both numeric representations (`f32` for `C32`, Q61 `i64` for `CQ15`) are derived from the
/// same parameters, and only the one of the sample type `T` is used.
///
/// ```
/// use sdrk_core::{kernels::noise_blank::NoiseBlankState, Rate, C32};
/// let rate = Rate::hz(192_000)?;
/// // Lookahead 50 us: D = round(0.00005 * 192000) = 10 samples.
/// let state = NoiseBlankState::<C32, _>::new(15.0, 0.01, 0.0002, [C32::default(); 10], rate)?;
/// assert_eq!(state.lookahead(), 10);
/// assert_eq!(state.hold(), 38);
/// assert_eq!(state.blanked(), 0);
/// assert!(NoiseBlankState::<C32, _>::new(5.0, 0.01, 0.0002, [C32::default(); 10], rate).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct NoiseBlankState<T: ComplexSample, Hist: AsRef<[T]> + AsMut<[T]>> {
    hist: Hist,
    threshold_db: f64,
    tau_s: f64,
    hold_s: f64,
    hold: u64,
    alpha_f: f32,
    alpha_q: i64,
    k_f: f32,
    k_q: i64,
    p_f: f32,
    p_q: i64,
    count: u64,
    blanked: u64,
    marker: PhantomData<T>,
}

/// `k = 10^(threshold_db / 10)` as `f32` (computed in f64 and cast) and as the Q16 integer
/// `floor(k * 2^16 + 0.5)`.
fn factors(threshold_db: f64) -> Result<(f32, i64)> {
    if !(threshold_db.is_finite() && (6.0..=40.0).contains(&threshold_db)) {
        return Err(Error::InvalidParameter(
            "noise blank threshold must be between 6 and 40 dB",
        ));
    }
    let k = libm::pow(10.0, threshold_db / 10.0);
    Ok((k as f32, libm::floor(k * Q16 + 0.5) as i64))
}

impl<T: ComplexSample, Hist: AsRef<[T]> + AsMut<[T]>> NoiseBlankState<T, Hist> {
    /// Creates a state with a zero delay line, `p = 0` and `c = 0`.
    ///
    /// `threshold_db` (6 to 40): an impulse is a sample whose power exceeds the average power
    /// by this much. `tau_s` (0.001 to 1): the time constant of the power average,
    /// `alpha = 1 - exp(-1 / (fs * tau_s))` in f64, which must not exceed 0.5 (at least 1.45
    /// samples per time constant). `hold_s` (0 to 0.01): the blanking time after the last
    /// impulse sample, `H = floor(hold_s * fs + 0.5)` samples. `history` holds exactly
    /// `D = round(lookahead_s * fs)` elements for a lookahead of 0 to 1 ms, so `D` must not
    /// exceed `floor(0.001 * fs + 0.5)`; it is zeroed. For `CQ15` the Q31 coefficient
    /// `floor(alpha * 2^31 + 0.5)` must not be zero. Anything else, including NaN, gives
    /// `InvalidParameter`.
    pub fn new(
        threshold_db: f64,
        tau_s: f64,
        hold_s: f64,
        mut history: Hist,
        rate: Rate,
    ) -> Result<Self> {
        let (k_f, k_q) = factors(threshold_db)?;
        if !(tau_s.is_finite() && (0.001..=1.0).contains(&tau_s)) {
            return Err(Error::InvalidParameter(
                "noise blank time constant must be between 0.001 and 1 s",
            ));
        }
        if !(hold_s.is_finite() && (0.0..=0.01).contains(&hold_s)) {
            return Err(Error::InvalidParameter(
                "noise blank hold must be between 0 and 0.01 s",
            ));
        }
        let fs = rate.as_f64();
        let max_d = libm::floor(MAX_LOOKAHEAD_S * fs + 0.5);
        if history.as_ref().len() as f64 > max_d {
            return Err(Error::InvalidParameter(
                "noise blank lookahead must not exceed 0.001 s",
            ));
        }
        let alpha = 1.0 - libm::exp(-1.0 / (fs * tau_s));
        if !(alpha > 0.0 && alpha <= 0.5) {
            return Err(Error::InvalidParameter(
                "noise blank smoothing coefficient must be in (0, 0.5]",
            ));
        }
        let alpha_q = libm::floor(alpha * Q31 + 0.5) as i64;
        if T::TYPE == SampleType::CQ15 && alpha_q == 0 {
            return Err(Error::InvalidParameter(
                "noise blank time constant is too long for CQ15",
            ));
        }
        history.as_mut().fill(T::ZERO);
        Ok(Self {
            hist: history,
            threshold_db,
            tau_s,
            hold_s,
            hold: libm::floor(hold_s * fs + 0.5) as u64,
            alpha_f: alpha as f32,
            alpha_q,
            k_f,
            k_q,
            p_f: 0.0,
            p_q: 0,
            count: 0,
            blanked: 0,
            marker: PhantomData,
        })
    }

    /// Changes the threshold (6 to 40 dB, as in [`NoiseBlankState::new`]); the delay line,
    /// the average power and the counter are kept and the new threshold applies from the
    /// next sample. On an error nothing changes.
    ///
    /// ```
    /// use sdrk_core::{kernels::noise_blank::NoiseBlankState, Rate, CQ15};
    /// let rate = Rate::hz(48_000)?;
    /// let mut state = NoiseBlankState::<CQ15, _>::new(15.0, 0.01, 0.0, [CQ15::default(); 0], rate)?;
    /// state.set_threshold_db(20.0)?;
    /// assert_eq!(state.threshold_db(), 20.0);
    /// assert!(state.set_threshold_db(41.0).is_err());
    /// assert_eq!(state.threshold_db(), 20.0);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn set_threshold_db(&mut self, threshold_db: f64) -> Result<()> {
        let (k_f, k_q) = factors(threshold_db)?;
        self.threshold_db = threshold_db;
        self.k_f = k_f;
        self.k_q = k_q;
        Ok(())
    }

    /// Clears the delay line, the average power, the counter and the count of blanked
    /// samples (the coefficients are kept).
    pub fn reset(&mut self) {
        self.hist.as_mut().fill(T::ZERO);
        self.p_f = 0.0;
        self.p_q = 0;
        self.count = 0;
        self.blanked = 0;
    }

    /// The number of output samples set to zero since construction or the last
    /// [`reset`](NoiseBlankState::reset).
    pub fn blanked(&self) -> u64 {
        self.blanked
    }

    /// The threshold in dB.
    pub fn threshold_db(&self) -> f64 {
        self.threshold_db
    }

    /// The time constant of the power average in seconds.
    pub fn tau_s(&self) -> f64 {
        self.tau_s
    }

    /// The hold time in seconds.
    pub fn hold_s(&self) -> f64 {
        self.hold_s
    }

    /// The lookahead `D` in samples (the length of the delay line and the group delay).
    pub fn lookahead(&self) -> usize {
        self.hist.as_ref().len()
    }

    /// The hold `H` in samples.
    pub fn hold(&self) -> u64 {
        self.hold
    }

    /// The average power as a linear value (full scale `|x|^2 = 1`): the `f32` value
    /// widened, or the Q61 integer converted to f64 and divided by `2^61`.
    pub fn average_power(&self) -> f64 {
        if T::TYPE == SampleType::CQ15 {
            self.p_q as f64 / (Q31 * Q31 / 2.0)
        } else {
            f64::from(self.p_f)
        }
    }

    pub(crate) fn parts_mut(&mut self) -> NoiseBlankParts<'_, T> {
        let d = self.hist.as_ref().len() as u64;
        NoiseBlankParts {
            span: d.saturating_add(self.hold).saturating_add(1),
            hist: self.hist.as_mut(),
            alpha_f: self.alpha_f,
            k_f: self.k_f,
            alpha_q: self.alpha_q,
            k_q: self.k_q,
            p_f: &mut self.p_f,
            p_q: &mut self.p_q,
            count: &mut self.count,
            blanked: &mut self.blanked,
        }
    }
}

impl<T: ComplexSample, Hist: AsRef<[T]> + AsMut<[T]>> Timing for NoiseBlankState<T, Hist> {
    fn ratio(&self) -> Ratio {
        Ratio::ONE
    }

    /// `D` samples, the lookahead.
    fn group_delay(&self) -> Delay {
        Delay::new(self.hist.as_ref().len() as u64, 1).unwrap_or(Delay::ZERO)
    }
}

/// Noise blanker.
///
/// ```
/// use sdrk_core::{kernels::noise_blank::{NoiseBlank, NoiseBlankState}, Fallback, Rate, C32};
/// let rate = Rate::hz(48_000)?;
/// let mut state = NoiseBlankState::<C32, _>::new(15.0, 0.01, 0.0, [C32::default(); 2], rate)?;
/// let mut input = [C32::new(0.01, 0.0); 1000];
/// input[500] = C32::new(1.0, 0.0);
/// let mut output = [C32::default(); 1000];
/// Fallback.noise_blank(&mut state, &input, &mut output)?;
/// // The output is the input delayed by D = 2; the impulse and the D samples before it
/// // are zero.
/// assert_eq!(output[1], C32::default());
/// assert_eq!(output[2], C32::new(0.01, 0.0));
/// assert!(output[500..503].iter().all(|y| *y == C32::default()));
/// assert_eq!(output[503], C32::new(0.01, 0.0));
/// assert_eq!(state.blanked(), 3);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait NoiseBlank<T: ComplexSample> {
    /// Per sample `x`, in this order: (1) the power `q`; (2) when `q` is not finite, or when
    /// the average `p` is positive and `q > k p`, the sample is an impulse: the counter
    /// becomes `c = D + H + 1` and the average moves towards `k p` instead of `q` (the power
    /// of an impulse never enters the average); otherwise, when `p` is positive, the average
    /// moves towards `q`, and when `p` is zero (after construction, `reset` or silence) it
    /// starts at `q`; (3) `y` is the oldest sample of the delay line (`x` itself for `D = 0`),
    /// which then takes `x`; (4) when `c > 0` the output is zero and `c = c - 1`, otherwise
    /// `y`. So an impulse at input `n` sets the outputs of the inputs `n - D` to `n + H` to
    /// zero. The lengths must be equal.
    ///
    /// `C32`: `q = re*re + im*im`; the impulse test is `!(q <= f32::MAX)` or
    /// `p > 0 && q > k * p`; then `p = flush_denormal(p + alpha * (t - p))` with the target
    /// `t = k * p` for an impulse and `t = q` otherwise, or `p = flush_denormal(q)` when
    /// `p == 0` and the sample is no impulse; all in f32 without FMA. `CQ15`: `q` is the Q30
    /// value `re^2 + im^2`, `p` is Q61 and `p30 = p >> 31`; the impulse test is `p30 > 0` and
    /// `q 2^16 > k_q p30` with `k_q = floor(k 2^16 + 0.5)`; then
    /// `p = p + alpha_q31 * (t - round_shift(p, 31))` in `i64` with `t = (k_q p30) >> 16` for
    /// an impulse and `t = q` otherwise, or `p = q 2^31` when `p30 == 0`. Both are bit-exact
    /// between `Fallback` and `Reference` (see `docs/kernels.md`).
    fn noise_blank<H>(
        &self,
        s: &mut NoiseBlankState<T, H>,
        input: &[T],
        output: &mut [T],
    ) -> Result<()>
    where
        H: AsRef<[T]> + AsMut<[T]>;
}
