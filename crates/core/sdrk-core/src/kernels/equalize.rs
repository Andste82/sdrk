//! Adaptive equalizer: an FIR filter on the symbol stream (one sample per symbol) whose complex
//! weights adapt blind with the constant-modulus algorithm (CMA) and then decision-directed,
//! both normalized by the power of the filter window (NLMS), optionally with decision feedback
//! (DFE) and with a decision-directed phase tracker on its output.
//!
//! Per symbol the joint vector `x` (the forward history `u`, newest first, then the fed-back
//! decisions) is filtered with the weights `w`, `y = Σ w[i] x[i]` (the [`DotComplex`] kernel
//! and its order, one call for both parts). The tracker phase `theta` turns the output back,
//! `z = y e^(-j theta)`, the decision `d` is made on `z`, and the weights move against the
//! error, `w[i] = w[i] - (mu / (P + eps)) e conj(x[i])` with `P = Σ |x[i]|²`: in CMA
//! `e = clip(y (|y|² - R2))` (Godard 2-2, each component clipped to `[-1, 1]`), in decision mode
//! `e = (z - d) e^(j theta)`, the decision error turned back into the equalizer domain. The
//! fed-back decisions are `d e^(j theta)` (decision mode only). The arithmetic is fixed (float32
//! model, see [`Equalize`] and `docs/kernels.md`), so `Fallback`, `Reference` and the SIMD
//! backend are bit-identical.

#![allow(clippy::many_single_char_names)]

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::fallback::costas::{integrate, not_finite, phase_step_f32};
use crate::fallback::dot::dot_complex_tree;
use crate::fallback::mix::interp_f32_pair;
use crate::kernels::check_len;
use crate::kernels::constellation::Qam;
use crate::kernels::costas::{derotate, loop_gains, octant_point};
use crate::kernels::dot::{DOT_COMPLEX_PARTIALS, DotComplex};
use crate::num::flush_denormal;
use crate::timing::{Delay, Timing};
use crate::types::C32;

/// Smallest and largest number of forward taps (odd only).
const TAPS: core::ops::RangeInclusive<usize> = 3..=63;
/// Largest number of feedback taps.
pub(crate) const MAX_FEEDBACK: usize = 63;
/// Smallest and largest (normalized) step size.
const MU: core::ops::RangeInclusive<f32> = 0.000_01..=1.0;
/// Smallest and largest bandwidth of the phase tracker (0 turns it off).
const PHASE_BW: core::ops::RangeInclusive<f64> = 0.0001..=0.2;
/// The damping of the phase tracker (that of `costas`, the value of the blocks).
#[allow(
    clippy::approx_constant,
    reason = "the damping 0.7071 of the costas block, not 1 / sqrt(2)"
)]
pub(crate) const PHASE_DAMPING: f64 = 0.7071;
/// The limit of the tracker's frequency integrator in cycles per symbol.
pub(crate) const PHASE_MAX_FREQ: f32 = 0.125;
/// The constant added to the window power of the NLMS update, `mu / (P + NLMS_EPS)`.
pub(crate) const NLMS_EPS: f32 = 1e-6;
/// The limit of each component of the CMA error.
pub(crate) const CMA_CLIP: f32 = 1.0;
/// The smoothing factor of the decision error power, `mse = mse + MSE_ALPHA (|e|² - mse)`: a
/// time constant of 100 symbols.
pub(crate) const MSE_ALPHA: f32 = 0.01;
/// The value of [`EqualizeState::mse`] after construction and `reset`: the power of a unit
/// constellation, as if the equalizer knew nothing yet.
pub(crate) const MSE_START: f32 = 1.0;
/// `1 / sqrt(2)` rounded once to `f32`: the components of the QPSK points.
pub(crate) const QPSK_A: f32 = core::f32::consts::FRAC_1_SQRT_2;
/// The CMA radius of 16-QAM, [`Qam::cma_r2`] rounded once to `f32`.
const QAM16_R2: f32 = Qam::Qam16.cma_r2() as f32;
/// The CMA radius of 64-QAM, [`Qam::cma_r2`] rounded once to `f32`.
const QAM64_R2: f32 = Qam::Qam64.cma_r2() as f32;

/// The adaptation mode of the equalizer.
///
/// ```
/// use sdrk_core::kernels::equalize::{EqMode, EqParams, EqualizeState};
/// let p = EqParams { cma_symbols: 0, ..EqParams::default() };
/// let st = EqualizeState::new([Default::default(); 5], [Default::default(); 5], p)?;
/// assert_eq!(st.mode(), EqMode::Decision);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EqMode {
    /// Blind adaptation with the constant-modulus algorithm.
    Cma,
    /// Decision-directed adaptation against the nearest constellation point.
    Decision,
}

/// The constellation the equalizer decides on; it also gives the CMA radius `R2`.
///
/// ```
/// use sdrk_core::{kernels::equalize::EqConstellation, C32};
/// let d = EqConstellation::Qpsk.decide(C32::new(0.2, -3.0));
/// assert_eq!(d, C32::new(core::f32::consts::FRAC_1_SQRT_2, -core::f32::consts::FRAC_1_SQRT_2));
/// assert_eq!(EqConstellation::Psk8.cma_r2(), 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EqConstellation {
    /// BPSK, the points `+1` and `-1`.
    Bpsk,
    /// QPSK, the points `(±1 ± j) / sqrt(2)`.
    Qpsk,
    /// 8PSK, the points `exp(j (2k + 1) pi / 8)` of M40.1.
    Psk8,
    /// 16-QAM, the unit-power Gray points of [`Qam::Qam16`] (M44).
    Qam16,
    /// 64-QAM, the unit-power Gray points of [`Qam::Qam64`] (M44).
    Qam64,
}

impl EqConstellation {
    /// The nearest constellation point, with the decision rule of `slice` and `costas`:
    /// BPSK `+1` for `re >= 0` (including `-0.0`), else `-1`; QPSK the signs of both components
    /// in the same way, scaled by `1 / sqrt(2)` (as `f32`); 8PSK the point of the half-open
    /// octant of `costas` `8psk` (zero takes point 0). A NaN component counts as not negative.
    /// 16-QAM and 64-QAM the grid decision of [`Qam::decide`] (a NaN component takes level 0).
    ///
    /// ```
    /// use sdrk_core::{kernels::{constellation::Qam, equalize::EqConstellation}, C32};
    /// assert_eq!(EqConstellation::Bpsk.decide(C32::new(-0.1, 5.0)), C32::new(-1.0, 0.0));
    /// assert_eq!(EqConstellation::Bpsk.decide(C32::new(0.0, -1.0)), C32::new(1.0, 0.0));
    /// let d = EqConstellation::Psk8.decide(C32::new(0.0, 1.0));
    /// assert!(d.re < 0.0 && d.im > 0.9); // the octant [pi/2, 3pi/4)
    /// let y = C32::new(0.2, -2.0);
    /// assert_eq!(EqConstellation::Qam16.decide(y), Qam::Qam16.decide(y));
    /// ```
    #[inline]
    pub fn decide(self, y: C32) -> C32 {
        let side = |v: f32, a: f32| if v < 0.0 { -a } else { a };
        match self {
            Self::Bpsk => C32::new(side(y.re, 1.0), 0.0),
            Self::Qpsk => C32::new(side(y.re, QPSK_A), side(y.im, QPSK_A)),
            Self::Psk8 => {
                let (re, im) = octant_point(y);
                C32::new(re, im)
            }
            Self::Qam16 => Qam::Qam16.decide(y),
            Self::Qam64 => Qam::Qam64.decide(y),
        }
    }

    /// The CMA radius `R2 = E|s|⁴ / E|s|²` of the constellation: 1 for every PSK constellation,
    /// for 16-QAM and 64-QAM the exact value of [`Qam::cma_r2`] (`33 / 25` and `29 / 21`)
    /// rounded once to `f32`.
    ///
    /// ```
    /// use sdrk_core::kernels::equalize::EqConstellation;
    /// assert_eq!(EqConstellation::Bpsk.cma_r2(), 1.0);
    /// assert_eq!(EqConstellation::Qam16.cma_r2(), 1.32);
    /// assert_eq!(EqConstellation::Qam64.cma_r2(), (29.0f64 / 21.0) as f32);
    /// ```
    #[inline]
    pub fn cma_r2(self) -> f32 {
        match self {
            Self::Bpsk | Self::Qpsk | Self::Psk8 => 1.0,
            Self::Qam16 => QAM16_R2,
            Self::Qam64 => QAM64_R2,
        }
    }
}

/// The parameters of [`EqualizeState::new`]; [`Default`] gives the defaults of the `equalize`
/// block.
///
/// ```
/// use sdrk_core::kernels::equalize::{EqConstellation, EqParams};
/// let p = EqParams::default();
/// assert_eq!((p.constellation, p.feedback_taps, p.cma_symbols), (EqConstellation::Qpsk, 0, 1000));
/// assert_eq!((p.mu_cma, p.mu_dd, p.phase_bw), (0.05, 0.02, 0.02));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EqParams {
    /// The constellation of the decisions and of the CMA radius.
    pub constellation: EqConstellation,
    /// Feedback taps `M` of the decision feedback (0 to 63; 0 is the linear equalizer).
    pub feedback_taps: usize,
    /// Normalized CMA step size (1e-5 to 1).
    pub mu_cma: f32,
    /// Normalized decision-mode step size (1e-5 to 1).
    pub mu_dd: f32,
    /// Symbols in CMA mode before the switch to decision mode (0 starts in decision mode,
    /// `u32::MAX` never switches).
    pub cma_symbols: u32,
    /// Loop bandwidth of the phase tracker normalized to the symbol rate (0 = off, else 1e-4 to
    /// 0.2; damping 0.7071).
    pub phase_bw: f64,
}

impl Default for EqParams {
    /// QPSK, no feedback, `mu_cma` 0.05, `mu_dd` 0.02, 1000 CMA symbols, `phase_bw` 0.02.
    fn default() -> Self {
        Self {
            constellation: EqConstellation::Qpsk,
            feedback_taps: 0,
            mu_cma: 0.05,
            mu_dd: 0.02,
            cma_symbols: 1000,
            phase_bw: 0.02,
        }
    }
}

/// State of the equalizer: `N + M` complex weights (`N` forward taps, then `M` feedback taps),
/// the joint history of the same length (the last `N` symbols, newest first, then the last `M`
/// fed-back decisions, newest first), the parameters, the mode with its symbol counter, the
/// smoothed decision error power and the phase and frequency of the phase tracker.
///
/// The caller provides the storage (`[C32; N + M]`, `&mut [C32]` or, in upper layers,
/// `Vec<C32>`).
///
/// ```
/// use sdrk_core::{kernels::equalize::{EqMode, EqParams, EqualizeState}, C32};
/// let st = EqualizeState::new([C32::default(); 11], [C32::default(); 11], EqParams::default())?;
/// assert_eq!((st.taps(), st.feedback_taps(), st.mode(), st.mse()), (11, 0, EqMode::Cma, 1.0));
/// // The centre tap starts at 1, the others at 0.
/// assert_eq!(st.weights()[5], C32::new(1.0, 0.0));
/// // 15 forward and 5 feedback taps: 20 weights and 20 history values.
/// let dfe = EqParams { feedback_taps: 5, ..EqParams::default() };
/// let st = EqualizeState::new([C32::default(); 20], [C32::default(); 20], dfe)?;
/// assert_eq!((st.taps(), st.feedback_taps(), st.weights()[7]), (15, 5, C32::new(1.0, 0.0)));
/// // An even number of forward taps, a history of another length and a step size above 1 fail.
/// assert!(EqualizeState::new([C32::default(); 10], [C32::default(); 10],
///     EqParams::default()).is_err());
/// assert!(EqualizeState::new([C32::default(); 11], [C32::default(); 9],
///     EqParams::default()).is_err());
/// assert!(EqualizeState::new([C32::default(); 11], [C32::default(); 11],
///     EqParams { mu_cma: 2.0, ..EqParams::default() }).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct EqualizeState<W: AsRef<[C32]> + AsMut<[C32]>, H: AsRef<[C32]> + AsMut<[C32]>> {
    weights: W,
    hist: H,
    params: EqParams,
    kp: f32,
    ki: f32,
    mode: EqMode,
    /// Symbols processed in CMA mode since construction or `reset`.
    count: u32,
    mse: f32,
    /// The tracker phase in units of `2^-32` turn.
    phase: u32,
    /// The tracker's frequency integrator in cycles per symbol.
    freq: f32,
}

/// Mutable view of the state, used by the kernel implementations of this crate.
pub(crate) struct EqualizeParts<'a> {
    pub(crate) weights: &'a mut [C32],
    pub(crate) hist: &'a mut [C32],
    /// The forward taps `N`.
    pub(crate) taps: usize,
    pub(crate) constellation: EqConstellation,
    pub(crate) mu_cma: f32,
    pub(crate) mu_dd: f32,
    pub(crate) cma_symbols: u32,
    pub(crate) kp: f32,
    pub(crate) ki: f32,
    pub(crate) mode: &'a mut EqMode,
    pub(crate) count: &'a mut u32,
    pub(crate) mse: &'a mut f32,
    pub(crate) phase: &'a mut u32,
    pub(crate) freq: &'a mut f32,
}

fn check_mu(mu: f32, msg: &'static str) -> Result<()> {
    if mu.is_finite() && MU.contains(&mu) {
        Ok(())
    } else {
        Err(Error::InvalidParameter(msg))
    }
}

const MU_CMA_MSG: &str = "equalizer cma step size must be between 1e-5 and 1";
const MU_DD_MSG: &str = "equalizer decision step size must be between 1e-5 and 1";

/// The start weights: the centre forward tap `(N - 1) / 2` at 1, the other weights (the
/// feedback weights too) at 0.
fn start_weights(w: &mut [C32], taps: usize) {
    let centre = taps / 2;
    for (i, v) in w.iter_mut().enumerate() {
        *v = if i == centre {
            C32::new(1.0, 0.0)
        } else {
            C32::new(0.0, 0.0)
        };
    }
}

/// The mode at the start: decision mode when `cma_symbols` is 0, CMA otherwise.
fn start_mode(cma_symbols: u32) -> EqMode {
    if cma_symbols == 0 {
        EqMode::Decision
    } else {
        EqMode::Cma
    }
}

impl<W: AsRef<[C32]> + AsMut<[C32]>, H: AsRef<[C32]> + AsMut<[C32]>> EqualizeState<W, H> {
    /// Creates a state with `M = params.feedback_taps` feedback taps (0 to 63) and
    /// `N = weights.len() - M` forward taps (odd, 3 to 63), the centre forward tap
    /// `w[(N - 1) / 2] = 1` and the other weights 0, a zero history (`history.len()` must be
    /// `weights.len()`, `LengthMismatch` otherwise), the normalized step sizes `mu_cma` and
    /// `mu_dd` (1e-5 to 1), `cma_symbols` symbols in CMA mode before the switch to decision
    /// mode (0 starts in decision mode, `u32::MAX` never switches), the phase tracker of
    /// bandwidth `phase_bw` (0 or 1e-4 to 0.2, damping 0.7071, phase and frequency 0) and `mse`
    /// at 1. The tracker gains are those of `costas` (`CostasState::new`) for `phase_bw` and the
    /// damping 0.7071; `phase_bw` 0 gives the gains 0, a phase that stays 0. The other
    /// violations, including NaN, give `InvalidParameter`.
    pub fn new(mut weights: W, mut history: H, params: EqParams) -> Result<Self> {
        let len = weights.as_ref().len();
        if params.feedback_taps > MAX_FEEDBACK {
            return Err(Error::InvalidParameter(
                "equalizer needs 0 to 63 feedback taps",
            ));
        }
        let taps = len.saturating_sub(params.feedback_taps);
        if !TAPS.contains(&taps) || taps % 2 == 0 || len < params.feedback_taps {
            return Err(Error::InvalidParameter(
                "equalizer needs an odd number of forward taps from 3 to 63",
            ));
        }
        let got = history.as_ref().len();
        if got != len {
            return Err(Error::LengthMismatch { expected: len, got });
        }
        check_mu(params.mu_cma, MU_CMA_MSG)?;
        check_mu(params.mu_dd, MU_DD_MSG)?;
        let bw = params.phase_bw;
        if !(bw == 0.0 || (bw.is_finite() && PHASE_BW.contains(&bw))) {
            return Err(Error::InvalidParameter(
                "equalizer phase tracker bandwidth must be 0 or between 0.0001 and 0.2",
            ));
        }
        let (kp, ki) = loop_gains(bw, PHASE_DAMPING);
        start_weights(weights.as_mut(), taps);
        history.as_mut().fill(C32::new(0.0, 0.0));
        Ok(Self {
            weights,
            hist: history,
            params,
            kp,
            ki,
            mode: start_mode(params.cma_symbols),
            count: 0,
            mse: MSE_START,
            phase: 0,
            freq: 0.0,
        })
    }

    /// The adaptation mode.
    pub fn mode(&self) -> EqMode {
        self.mode
    }

    /// The decision error power `|z - d|²` of the phase-corrected output `z` smoothed over
    /// about 100 symbols (linear, not dB): after every symbol with a finite output, in both
    /// modes, `mse = flush_denormal(mse + 0.01 (p - mse))` with `p = e.re e.re + e.im e.im`,
    /// `e = z - d`, if `p` is finite. It starts at 1.
    pub fn mse(&self) -> f32 {
        self.mse
    }

    /// Returns to the state after construction: the start weights, a zero history, the start
    /// mode with the symbol counter at 0, `mse` at 1 and the tracker at phase and frequency 0
    /// (the parameters and step sizes are kept).
    ///
    /// ```
    /// use sdrk_core::{kernels::equalize::{EqConstellation, EqMode, EqParams, Equalize,
    ///     EqualizeState}, Fallback, C32};
    /// let p = EqParams { constellation: EqConstellation::Bpsk, cma_symbols: 4, ..EqParams::default() };
    /// let mut st = EqualizeState::new([C32::default(); 3], [C32::default(); 3], p)?;
    /// let mut out = [C32::default(); 8];
    /// Fallback.equalize(&mut st, &[C32::new(0.5, 0.0); 8], &mut out)?;
    /// assert_eq!(st.mode(), EqMode::Decision);
    /// st.reset();
    /// assert_eq!((st.mode(), st.mse(), st.weights()[1]), (EqMode::Cma, 1.0, C32::new(1.0, 0.0)));
    /// assert_eq!((st.phase(), st.frequency()), (0, 0.0));
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn reset(&mut self) {
        let taps = self.taps();
        start_weights(self.weights.as_mut(), taps);
        self.hist.as_mut().fill(C32::new(0.0, 0.0));
        self.mode = start_mode(self.params.cma_symbols);
        self.count = 0;
        self.mse = MSE_START;
        self.phase = 0;
        self.freq = 0.0;
    }

    /// Changes the CMA step size (1e-5 to 1, `InvalidParameter` otherwise and nothing
    /// changes); the new value applies from the next symbol.
    ///
    /// ```
    /// use sdrk_core::{kernels::equalize::{EqParams, EqualizeState}, C32};
    /// let mut st = EqualizeState::new([C32::default(); 3], [C32::default(); 3], EqParams::default())?;
    /// st.set_mu_cma(0.1)?;
    /// assert_eq!(st.mu_cma(), 0.1);
    /// assert!(st.set_mu_cma(f32::NAN).is_err());
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn set_mu_cma(&mut self, mu: f32) -> Result<()> {
        check_mu(mu, MU_CMA_MSG)?;
        self.params.mu_cma = mu;
        Ok(())
    }

    /// Changes the decision-mode step size (1e-5 to 1, `InvalidParameter` otherwise and
    /// nothing changes); the new value applies from the next symbol.
    ///
    /// ```
    /// use sdrk_core::{kernels::equalize::{EqParams, EqualizeState}, C32};
    /// let mut st = EqualizeState::new([C32::default(); 3], [C32::default(); 3], EqParams::default())?;
    /// st.set_mu_dd(0.01)?;
    /// assert_eq!(st.mu_dd(), 0.01);
    /// assert!(st.set_mu_dd(0.0).is_err());
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn set_mu_dd(&mut self, mu: f32) -> Result<()> {
        check_mu(mu, MU_DD_MSG)?;
        self.params.mu_dd = mu;
        Ok(())
    }

    /// The number of forward taps `N`.
    pub fn taps(&self) -> usize {
        self.weights
            .as_ref()
            .len()
            .saturating_sub(self.params.feedback_taps)
    }

    /// The number of feedback taps `M`.
    pub fn feedback_taps(&self) -> usize {
        self.params.feedback_taps
    }

    /// The weights `w[0..N + M]`: `w[i]` for `i < N` multiplies the symbol `i` symbols back,
    /// `w[N + k]` the decision `k + 1` symbols back.
    pub fn weights(&self) -> &[C32] {
        self.weights.as_ref()
    }

    /// The joint history `x[0..N + M]`: the last `N` symbols, newest first (a non-finite input
    /// is stored as zero), then the last `M` fed-back decisions `d e^(j theta)`, newest first
    /// (zero in CMA mode and for a non-finite symbol).
    pub fn history(&self) -> &[C32] {
        self.hist.as_ref()
    }

    /// The parameters (with the current step sizes).
    pub fn params(&self) -> EqParams {
        self.params
    }

    /// The constellation of the decisions.
    pub fn constellation(&self) -> EqConstellation {
        self.params.constellation
    }

    /// The CMA step size.
    pub fn mu_cma(&self) -> f32 {
        self.params.mu_cma
    }

    /// The decision-mode step size.
    pub fn mu_dd(&self) -> f32 {
        self.params.mu_dd
    }

    /// The symbols in CMA mode before the switch (0: decision mode from the start, `u32::MAX`:
    /// never).
    pub fn cma_symbols(&self) -> u32 {
        self.params.cma_symbols
    }

    /// The phase of the tracker in units of `2^-32` turn; the output is the filter output
    /// turned back by it.
    pub fn phase(&self) -> u32 {
        self.phase
    }

    /// The tracker's frequency integrator in cycles per symbol.
    pub fn frequency(&self) -> f32 {
        self.freq
    }

    /// Sets the tracker's frequency integrator in cycles per symbol (a frequency estimate from
    /// outside, as the acquisition of `qam_demod`); the value is clipped to `[-0.125, 0.125]`
    /// (the limit of the integrator), flushed when subnormal, and NaN gives 0. The phase, the
    /// weights and the mode stay.
    ///
    /// ```
    /// use sdrk_core::{kernels::equalize::{EqParams, EqualizeState}, C32};
    /// let mut st = EqualizeState::new([C32::default(); 3], [C32::default(); 3], EqParams::default())?;
    /// st.set_frequency(0.04);
    /// assert_eq!(st.frequency(), 0.04);
    /// st.set_frequency(-0.3);
    /// assert_eq!(st.frequency(), -0.125);
    /// st.set_frequency(f32::NAN);
    /// assert_eq!(st.frequency(), 0.0);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn set_frequency(&mut self, freq: f32) {
        self.freq = if freq.is_nan() {
            0.0
        } else {
            clip(flush_denormal(freq), PHASE_MAX_FREQ)
        };
    }

    /// The tracker gains `(kp, ki)` in cycles per symbol per unit of the detector output.
    pub fn phase_gains(&self) -> (f32, f32) {
        (self.kp, self.ki)
    }

    pub(crate) fn parts_mut(&mut self) -> EqualizeParts<'_> {
        let taps = self.taps();
        EqualizeParts {
            weights: self.weights.as_mut(),
            hist: self.hist.as_mut(),
            taps,
            constellation: self.params.constellation,
            mu_cma: self.params.mu_cma,
            mu_dd: self.params.mu_dd,
            cma_symbols: self.params.cma_symbols,
            kp: self.kp,
            ki: self.ki,
            mode: &mut self.mode,
            count: &mut self.count,
            mse: &mut self.mse,
            phase: &mut self.phase,
            freq: &mut self.freq,
        }
    }
}

impl<W: AsRef<[C32]> + AsMut<[C32]>, H: AsRef<[C32]> + AsMut<[C32]>> Timing
    for EqualizeState<W, H>
{
    fn ratio(&self) -> Ratio {
        Ratio::ONE
    }

    /// `(N - 1) / 2` symbols: the centre forward tap carries the signal at the start.
    fn group_delay(&self) -> Delay {
        let n = self.taps() as u64;
        Delay::new(n.saturating_sub(1) / 2, 1).unwrap_or(Delay::ZERO)
    }
}

/// `v` limited to `[-limit, limit]` as two selects (NaN stays NaN).
#[inline]
pub(crate) fn clip(v: f32, limit: f32) -> f32 {
    let v = if v > limit { limit } else { v };
    if v < -limit { -limit } else { v }
}

/// `v e^(j theta)` from `c = cos theta`, `s = sin theta`: `(v.re c - v.im s, v.re s + v.im c)`.
#[inline]
pub(crate) fn rotate(v: C32, c: f32, s: f32) -> C32 {
    C32::new(v.re * c - v.im * s, v.re * s + v.im * c)
}

/// The window power `P = Σ (x.re x.re + x.im x.im)` in ascending `i`, from `+0.0`, and the
/// partial sum over the first `n` elements (the forward part).
#[inline]
pub(crate) fn window_power(x: &[C32], n: usize) -> (f32, f32) {
    let mut p = 0.0f32;
    let mut forward = 0.0f32;
    for (i, v) in x.iter().enumerate() {
        p += v.re * v.re + v.im * v.im;
        if i + 1 == n {
            forward = p;
        }
    }
    (forward, p)
}

/// One weight of the update: `flush_denormal(w - g conj(x))` per component with
/// `g conj(x) = (g.re x.re + g.im x.im, g.im x.re - g.re x.im)`.
#[inline]
pub(crate) fn update_weight(w: C32, g: C32, x: C32) -> C32 {
    let cr = g.re * x.re + g.im * x.im;
    let ci = g.im * x.re - g.re * x.im;
    C32::new(flush_denormal(w.re - cr), flush_denormal(w.im - ci))
}

/// The mode switch after a symbol: in CMA mode the counter advances, and decision mode starts
/// once it reaches `cma_symbols` (never for `u32::MAX`).
#[inline]
pub(crate) fn advance(mode: &mut EqMode, count: &mut u32, cma_symbols: u32) {
    if *mode == EqMode::Cma && cma_symbols != u32::MAX {
        *count = count.saturating_add(1);
        if *count >= cma_symbols {
            *mode = EqMode::Decision;
        }
    }
}

/// The tracker after a symbol with the output `z` and the decision `d`: the detector
/// `pd = z.im d.re - z.re d.im` (0 if not finite), `freq = clip(flush(freq + ki pd), 0.125)`,
/// `phase += phase_step(freq + kp pd)`.
///
/// Computed with the forms of the carrier loop (D1100, D1417), the same bits: the integrator
/// inside its limit after one well-predicted branch ([`integrate`]), the phase step from the
/// bits of the `f32` ([`phase_step_f32`]) and the 0 of a non-finite detector value out of line.
#[inline]
pub(crate) fn track(phase: &mut u32, freq: &mut f32, kp: f32, ki: f32, z: C32, d: C32) {
    let pd = z.im * d.re - z.re * d.im;
    let pd = if pd.is_finite() { pd } else { not_finite() };
    *freq = integrate(*freq + ki * pd, PHASE_MAX_FREQ);
    *phase = phase.wrapping_add(phase_step_f32(*freq + kp * pd));
}

/// One symbol after the filter output `y` (finite), with `Pf` (`forward`) and the step
/// `m = mu / (P + eps)` of the mode: the tracker rotation, the decision, `mse`, the error and
/// the factor `g = m e`. Returns the output `z`, the value to feed
/// back (`d e^(j theta)` in decision mode, else 0) and `g` (`None` when it is not finite).
struct Step {
    z: C32,
    feedback: C32,
    g: Option<C32>,
}

#[allow(
    clippy::inline_always,
    reason = "a part of the symbol loop of every path: two instances call it, and LLVM keeps a helper with two callers out of line"
)]
#[inline(always)]
fn step<K: Decide>(
    p: &mut EqualizeParts<'_>,
    k: K,
    mode: EqMode,
    y: C32,
    forward: f32,
    m: f32,
) -> Result<Step> {
    // Phase 0 is the identity (the table's sine at 0 is not exactly 0).
    let (c, s) = if *p.phase == 0 {
        (1.0, 0.0)
    } else {
        interp_f32_pair(*p.phase).ok_or(Error::Internal("cosine table index"))?
    };
    let z = derotate(y, c, s);
    let d = k.decide(z);
    let ed = C32::new(z.re - d.re, z.im - d.im);
    let pe = ed.re * ed.re + ed.im * ed.im;
    if pe.is_finite() {
        *p.mse = flush_denormal(*p.mse + MSE_ALPHA * (pe - *p.mse));
    }
    let (e, feedback) = match mode {
        EqMode::Cma => {
            let f = (y.re * y.re + y.im * y.im) - k.cma_r2();
            let e = C32::new(clip(y.re * f, CMA_CLIP), clip(y.im * f, CMA_CLIP));
            (e, C32::new(0.0, 0.0))
        }
        EqMode::Decision => (rotate(ed, c, s), rotate(d, c, s)),
    };
    let g = C32::new(m * e.re, m * e.im);
    track(p.phase, p.freq, p.kp, p.ki, z, d);
    // A forward window of silence adapts nothing and feeds nothing back (D839).
    let live = forward != 0.0;
    Ok(Step {
        z,
        feedback: if live { feedback } else { C32::new(0.0, 0.0) },
        g: (live && g.re.is_finite() && g.im.is_finite()).then_some(g),
    })
}

/// The equalizer of [`Equalize::equalize`] (see there for the arithmetic). Without feedback taps
/// it runs the linear path of this crate (runs of symbols on a split copy of the state, its own
/// filter part in the order of [`DotComplex`], D840) and `dot` is not used; with feedback taps
/// the filter part `y = Σ w[i] x[i]` over the joint vector of forward history and fed-back
/// decisions is computed by `dot` (any implementation of [`DotComplex`] `C32`, all of them
/// bit-identical). `Fallback`, the SIMD backend and the facade run it with `Fallback` (D841).
///
/// ```
/// use sdrk_core::{kernels::equalize::{equalize_with, EqParams, EqualizeState}, Fallback, C32};
/// let p = EqParams { cma_symbols: 0, mu_dd: 1e-5, phase_bw: 0.0, ..EqParams::default() };
/// let mut st = EqualizeState::new([C32::default(); 3], [C32::default(); 3], p)?;
/// let mut out = [C32::default(); 3];
/// let x = C32::new(0.5, 0.5);
/// equalize_with(&Fallback, &mut st, &[x; 3], &mut out)?;
/// // The centre tap delays by one symbol.
/// assert_eq!(out[0], C32::new(0.0, 0.0));
/// assert!((out[1] - x).norm_sqr() < 1e-6);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub fn equalize_with<D, W, H>(
    dot: &D,
    s: &mut EqualizeState<W, H>,
    input: &[C32],
    output: &mut [C32],
) -> Result<()>
where
    D: DotComplex<C32>,
    W: AsRef<[C32]> + AsMut<[C32]>,
    H: AsRef<[C32]> + AsMut<[C32]>,
{
    check_len(input.len(), output.len())?;
    let mut p = s.parts_mut();
    match p.constellation {
        EqConstellation::Qam16 => equalize_qam(dot, &mut p, QamDecide(Qam::Qam16), input, output),
        EqConstellation::Qam64 => equalize_qam(dot, &mut p, QamDecide(Qam::Qam64), input, output),
        EqConstellation::Bpsk => equalize_paths(dot, &mut p, PskDecide::Bpsk, input, output),
        EqConstellation::Psk8 => equalize_paths(dot, &mut p, PskDecide::Psk8, input, output),
        _ => equalize_paths(dot, &mut p, PskDecide::Qpsk, input, output),
    }
}

/// The decision and the CMA radius of the loops of [`equalize_with`], each loop compiled for one
/// implementation: [`PskDecide`] (the three PSK constellations, chosen per symbol as in M43) and
/// [`QamDecide`] (M44), so that the PSK loops keep the code of M43 and the QAM loops are a
/// function of their own.
trait Decide: Copy {
    /// [`EqConstellation::decide`].
    fn decide(self, y: C32) -> C32;
    /// [`EqConstellation::cma_r2`].
    fn cma_r2(self) -> f32;
}

/// The PSK constellations, as the three-variant enum of M43 (so that the decision compiles to
/// the code of M43).
#[derive(Clone, Copy)]
enum PskDecide {
    Bpsk,
    Qpsk,
    Psk8,
}

impl Decide for PskDecide {
    #[inline]
    fn decide(self, y: C32) -> C32 {
        let side = |v: f32, a: f32| if v < 0.0 { -a } else { a };
        match self {
            Self::Bpsk => C32::new(side(y.re, 1.0), 0.0),
            Self::Qpsk => C32::new(side(y.re, QPSK_A), side(y.im, QPSK_A)),
            Self::Psk8 => {
                let (re, im) = octant_point(y);
                C32::new(re, im)
            }
        }
    }

    #[inline]
    fn cma_r2(self) -> f32 {
        1.0
    }
}

/// A square QAM constellation.
#[derive(Clone, Copy)]
struct QamDecide(Qam);

impl Decide for QamDecide {
    #[inline]
    fn decide(self, y: C32) -> C32 {
        self.0.decide(y)
    }

    #[inline]
    fn cma_r2(self) -> f32 {
        match self.0 {
            Qam::Qam16 => QAM16_R2,
            _ => QAM64_R2,
        }
    }
}

/// The paths of [`equalize_with`] for the QAM constellations, a function of their own.
#[inline(never)]
fn equalize_qam<D: DotComplex<C32>>(
    dot: &D,
    p: &mut EqualizeParts<'_>,
    k: QamDecide,
    input: &[C32],
    output: &mut [C32],
) -> Result<()> {
    equalize_paths(dot, p, k, input, output)
}

/// The linear path without feedback taps, the joint path with them.
#[inline]
fn equalize_paths<D: DotComplex<C32>, K: Decide>(
    dot: &D,
    p: &mut EqualizeParts<'_>,
    k: K,
    input: &[C32],
    output: &mut [C32],
) -> Result<()> {
    if p.hist.len() == p.taps {
        equalize_linear(p, k, input, output)
    } else {
        equalize_joint(dot, p, k, input, output)
    }
}

/// `v`, or zero when a component is not finite (the value the history stores).
#[inline]
fn sanitize(v: C32) -> C32 {
    if v.re.is_finite() && v.im.is_finite() {
        v
    } else {
        C32::new(0.0, 0.0)
    }
}

/// Step 5 for every weight of the joint path: `w[i] = flush_denormal(w[i] - g conj(x[i]))`, and
/// the start weights (`taps` forward taps) when a new weight is not finite. The finite test is
/// an `or` of integer bits over all weights (no branch per weight), so the loop stays one
/// straight pass: an exponent of all ones plus one carries into the sign bit.
#[inline]
fn update_weights(w: &mut [C32], x: &[C32], g: C32, taps: usize) {
    const EXP: u32 = 0x7f80_0000;
    const CARRY: u32 = 0x0080_0000;
    let mut bad = 0u32;
    for (w, &x) in w.iter_mut().zip(x) {
        let v = update_weight(*w, g, x);
        bad |=
            (v.re.to_bits() & EXP).wrapping_add(CARRY) | (v.im.to_bits() & EXP).wrapping_add(CARRY);
        *w = v;
    }
    if bad & 0x8000_0000 != 0 {
        start_weights(w, taps);
    }
}

/// Symbols per run of the linear path ([`equalize_linear`]).
const RUN: usize = 64;
/// The most forward taps rounded up to a multiple of [`QUAD`].
const PADDED_MAX: usize = 64;
/// The linear path pads the forward taps to a multiple of four lanes: one vector of SSE2,
/// NEON and SIMD128, half a block of the filter part (whose last half block goes to the partial
/// sums 0 to 3).
const QUAD: usize = DOT_COMPLEX_PARTIALS / 2;
/// A block of [`DOT_COMPLEX_PARTIALS`] values of one component.
type Lanes = [f32; DOT_COMPLEX_PARTIALS];

/// The history part of the working copy of the linear path, split into its real and imaginary
/// parts. The two parts live in the frames of two functions ([`equalize_linear`] and
/// [`linear_state`]), so that no frame holds both (the stack limit of the library, 4096 bytes
/// per frame as Clippy counts them).
struct History {
    /// The history, newest first: symbol `k` of a run at `RUN - 1 - k`, the history before the
    /// run from `RUN` on.
    xr: [f32; RUN + PADDED_MAX],
    xi: [f32; RUN + PADDED_MAX],
    /// `-xi`, for the imaginary part of the filter part (see [`dot_split`]).
    xn: [f32; RUN + PADDED_MAX],
}

/// The weight part of the working copy of the linear path, with the scratch of a run.
struct Weights {
    /// The weights, zero after the `N` forward taps.
    wr: [f32; PADDED_MAX],
    wi: [f32; PADDED_MAX],
    /// All ones for the `N` weights, zero for the padding.
    mask: [u32; PADDED_MAX],
    /// `|x|²` of the history values.
    mag: [f32; RUN + PADDED_MAX],
    /// The window power and the step `mu / (P + eps)` of the symbol at each position of a run.
    power: [f32; RUN],
    steps: [f32; RUN],
}

impl History {
    /// All zero (the padding stays so).
    const ZERO: Self = Self {
        xr: [0.0; RUN + PADDED_MAX],
        xi: [0.0; RUN + PADDED_MAX],
        xn: [0.0; RUN + PADDED_MAX],
    };
}

impl Weights {
    /// All zero (the padding stays so).
    const ZERO: Self = Self {
        wr: [0.0; PADDED_MAX],
        wi: [0.0; PADDED_MAX],
        mask: [0; PADDED_MAX],
        mag: [0.0; RUN + PADDED_MAX],
        power: [0.0; RUN],
        steps: [0.0; RUN],
    };
}

/// The working copy of the linear path.
struct Linear<'a> {
    h: &'a mut History,
    w: &'a mut Weights,
}

/// `dot_complex` of the window and the weights in the order of its contract (element `i` to
/// partial sum `i mod 8`, the products `x.re w.re - x.im w.im` and `x.re w.im + x.im w.re`,
/// then the tree), over blocks of split parts (a multiple of [`QUAD`] samples): each block is
/// one vector step per component on every level with vectors of four or eight lanes. The
/// imaginary product is
/// formed as `x.re w.im - (-x.im) w.re` from the negated imaginary parts `xn`: `(-a) b` is
/// `-(a b)` and `p - (-q)` is `p + q` exactly (IEEE 754 defines the difference as the sum with
/// the negated operand), so the bits are the same. The point is the form: with a sum of two
/// products LLVM pairs the products into one vector and adds them horizontally (many
/// shuffles per block on SSE2, AVX2 and NEON); with a difference both components compile to
/// plain lane-wise multiplies and subtractions.
#[allow(
    clippy::inline_always,
    reason = "a part of the symbol loop of every path: two instances call it, and LLVM keeps a helper with two callers out of line"
)]
#[inline(always)]
fn dot_split(xr: &[f32], xi: &[f32], xn: &[f32], wr: &[f32], wi: &[f32]) -> C32 {
    /// `acc[k] = acc[k] + (a c - b d)` per lane, by whole blocks; a last half block (the length
    /// is a multiple of [`QUAD`]) goes to the partial sums 0 to 3, as its elements do.
    #[inline(always)]
    fn pass(a: &[f32], b: &[f32], c: &[f32], d: &[f32]) -> Lanes {
        #[inline]
        fn add(acc: &mut [f32], a: &[f32], b: &[f32], c: &[f32], d: &[f32]) {
            let lanes = acc.iter_mut().zip(a).zip(b).zip(c.iter().zip(d));
            for (((s, &a), &b), (&c, &d)) in lanes {
                *s += a * c - b * d;
            }
        }
        let mut acc: Lanes = [0.0; DOT_COMPLEX_PARTIALS];
        let (ab, ar) = a.as_chunks::<DOT_COMPLEX_PARTIALS>();
        let (bb, br) = b.as_chunks::<DOT_COMPLEX_PARTIALS>();
        let (cb, cr) = c.as_chunks::<DOT_COMPLEX_PARTIALS>();
        let (db, dr) = d.as_chunks::<DOT_COMPLEX_PARTIALS>();
        for (((a, b), c), d) in ab.iter().zip(bb).zip(cb).zip(db) {
            add(&mut acc, a, b, c, d);
        }
        if let (Some(a), Some(b), Some(c), Some(d)) = (
            ar.first_chunk::<QUAD>(),
            br.first_chunk::<QUAD>(),
            cr.first_chunk::<QUAD>(),
            dr.first_chunk::<QUAD>(),
        ) && let Some(lo) = acc.first_chunk_mut::<QUAD>()
        {
            add(lo, a, b, c, d);
        }
        acc
    }
    let re = pass(xr, xi, wr, wi);
    let im = pass(xr, xn, wi, wr);
    C32::new(dot_complex_tree(re), dot_complex_tree(im))
}

/// The bits of `f32::MAX`: a value is not finite when its magnitude bits are above.
const MAX_BITS: u32 = 0x7f7f_ffff;

/// The accumulators of the finite test of [`update_split`] after the weight bits `vr`, `vi`, one
/// per component and lane: where an unsigned maximum of 32-bit lanes is one instruction (AArch64,
/// SSE4.1 and up) the largest magnitude bits (`v << 1`, the sign shifted out); at SSE2 and on
/// Cortex-M4 the or of the all-ones masks of a signed compare of the magnitude bits with those
/// of `f32::MAX`; on `wasm32` (counted in instructions) the or of the compare as 0 or 1, the
/// form of M43 with one accumulator, which takes the fewest instructions there (D1417).
#[cfg(any(target_arch = "aarch64", target_feature = "sse4.1"))]
#[inline]
fn not_finite_acc(acc_r: u32, acc_i: u32, vr: u32, vi: u32) -> (u32, u32) {
    (acc_r.max(vr << 1), acc_i.max(vi << 1))
}

/// See the first variant.
#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "wasm32",
    target_feature = "sse4.1"
)))]
#[inline]
fn not_finite_acc(acc_r: u32, acc_i: u32, vr: u32, vi: u32) -> (u32, u32) {
    let mask = |v: u32| (-i32::from((v & 0x7fff_ffff) as i32 > MAX_BITS as i32)) as u32;
    (acc_r | mask(vr), acc_i | mask(vi))
}

/// See the first variant.
#[cfg(target_arch = "wasm32")]
#[inline]
fn not_finite_acc(acc_r: u32, acc_i: u32, vr: u32, vi: u32) -> (u32, u32) {
    let big = |v: u32| u32::from((v & 0x7fff_ffff) as i32 > MAX_BITS as i32);
    (acc_r | big(vr) | big(vi), acc_i)
}

/// Whether the accumulators of [`not_finite_acc`] saw a value that is not finite.
#[cfg(any(target_arch = "aarch64", target_feature = "sse4.1"))]
#[inline]
fn not_finite_any(a: u32, b: u32) -> bool {
    a.max(b) > MAX_BITS << 1
}

/// See the first variant (both other forms).
#[cfg(not(any(target_arch = "aarch64", target_feature = "sse4.1")))]
#[inline]
fn not_finite_any(a: u32, b: u32) -> bool {
    a | b != 0
}

/// Step 5 on the split weights over whole blocks: `w[i] = flush_denormal(w[i] - g conj(x[i]))`
/// for the `N` weights, over `lanes` (`N` rounded up to a multiple of [`QUAD`]); the padding is
/// computed too and masked back to `+0`, so only the `N` weights take part in the finite test
/// ([`not_finite_acc`] per weight and component into two accumulators, one test after the
/// loop, no branch). One flat loop, which LLVM vectorizes.
/// Returns whether a new weight is not finite.
#[allow(
    clippy::inline_always,
    reason = "a part of the symbol loop of every path: two instances call it, and LLVM keeps a helper with two callers out of line"
)]
#[inline(always)]
fn update_split(l: &mut Linear<'_>, at: usize, lanes: usize, g: C32) -> Option<bool> {
    let xr = l.h.xr.get(at..at + lanes)?;
    let xi = l.h.xi.get(at..at + lanes)?;
    let wr = l.w.wr.get_mut(..lanes)?;
    let wi = l.w.wi.get_mut(..lanes)?;
    let (mut big_r, mut big_i) = (0u32, 0u32);
    let each = wr
        .iter_mut()
        .zip(wi.iter_mut())
        .zip(xr)
        .zip(xi)
        .zip(&l.w.mask);
    for ((((wr, wi), &a), &b), &m) in each {
        let cr = g.re * a + g.im * b;
        let ci = g.im * a - g.re * b;
        let vr = flush_denormal(*wr - cr).to_bits() & m;
        let vi = flush_denormal(*wi - ci).to_bits() & m;
        (big_r, big_i) = not_finite_acc(big_r, big_i, vr, vi);
        *wr = f32::from_bits(vr);
        *wi = f32::from_bits(vi);
    }
    Some(not_finite_any(big_r, big_i))
}

/// The equalizer without feedback taps (`M = 0`), the arithmetic of [`Equalize::equalize`] in
/// runs of [`RUN`] symbols on a working copy of the state ([`Linear`]):
///
/// - The history is kept newest first with the symbol `k` of a run at `RUN - 1 - k`, so the
///   forward window of symbol `k` is the slice from `RUN - 1 - k`: nothing moves per symbol, the
///   window moves once per run.
/// - The filter part runs over the window and the weights padded with zero weights to a
///   multiple of [`QUAD`] samples ([`dot_split`]): the window values there are finite (stored
///   symbols or zero), so their products are `+0` or `-0`, and adding either to a partial sum
///   does not change it (a partial sum that starts at `+0` is never `-0` under
///   round-to-nearest). The sum has the bits of the call over `N` samples.
/// - The window power depends on the input only (`M = 0`), so it is computed for the whole run
///   first: `|x|²` once per symbol, then the sums of all windows side by side, each in
///   ascending order from `+0.0` (its own chain in the order of the contract), which the
///   compiler vectorizes across the windows; so are the steps `mu / (P + eps)` of the run, the
///   mode of every symbol known from the symbol counter (the rule of `advance`). The divisions
///   leave the chain from symbol to symbol.
/// - The weight update runs over the padding too, which is masked back to zero
///   ([`update_split`]).
// A function of its own per decision (as the one linear path of M43), so that the symbol loop
// keeps its helpers inlined.
#[inline(never)]
fn equalize_linear<K: Decide>(
    p: &mut EqualizeParts<'_>,
    k: K,
    input: &[C32],
    output: &mut [C32],
) -> Result<()> {
    let mut h = History::ZERO;
    linear_state(p, k, &mut h, input, output)
}

/// The rest of [`equalize_linear`]: the weight part of the working copy, the copy in and out of
/// the state and the runs.
fn linear_state<K: Decide>(
    p: &mut EqualizeParts<'_>,
    k: K,
    h: &mut History,
    input: &[C32],
    output: &mut [C32],
) -> Result<()> {
    let mut w = Weights::ZERO;
    let mut l = Linear { h, w: &mut w };
    load_linear(p, &mut l);
    let result = linear_runs(p, k, &mut l, input, output);
    // The state is written back also after an (internal) error.
    store_linear(p, &l);
    result
}

/// The state into the working copy.
fn load_linear(p: &EqualizeParts<'_>, l: &mut Linear<'_>) {
    let hist =
        l.h.xr
            .iter_mut()
            .zip(l.h.xi.iter_mut())
            .zip(l.h.xn.iter_mut())
            .skip(RUN);
    for (((r, i), m), h) in hist.zip(p.hist.iter()) {
        (*r, *i, *m) = (h.re, h.im, -h.im);
    }
    let weights =
        l.w.wr
            .iter_mut()
            .zip(l.w.wi.iter_mut())
            .zip(l.w.mask.iter_mut());
    for (((r, i), m), w) in weights.zip(p.weights.iter()) {
        (*r, *i, *m) = (w.re, w.im, u32::MAX);
    }
}

/// The working copy back into the state.
fn store_linear(p: &mut EqualizeParts<'_>, l: &Linear<'_>) {
    for ((w, &r), &i) in p.weights.iter_mut().zip(&l.w.wr).zip(&l.w.wi) {
        *w = C32::new(r, i);
    }
    let hist = l.h.xr.iter().zip(&l.h.xi).skip(RUN);
    for (h, (&r, &i)) in p.hist.iter_mut().zip(hist) {
        *h = C32::new(r, i);
    }
}

/// The runs of [`equalize_linear`]; on return the history is at `RUN..RUN + N`.
#[inline]
fn linear_runs<K: Decide>(
    p: &mut EqualizeParts<'_>,
    dec: K,
    l: &mut Linear<'_>,
    input: &[C32],
    output: &mut [C32],
) -> Result<()> {
    let index = || Error::Internal("equalizer index");
    let n = p.taps;
    let lanes = n.next_multiple_of(QUAD);
    let padded = lanes;
    for (xs, ys) in input.chunks(RUN).zip(output.chunks_mut(RUN)) {
        let first = RUN - xs.len();
        prepare_run(p, l, xs, first).ok_or_else(index)?;
        for (k, (o, &raw)) in ys.iter_mut().zip(xs).enumerate() {
            let at = RUN - 1 - k;
            let y = dot_window(l, at, padded).ok_or_else(index)?;
            let finite = raw.re.is_finite() && raw.im.is_finite();
            *o = if finite && y.re.is_finite() && y.im.is_finite() {
                let (&pq, &m) = l.w.power.get(at).zip(l.w.steps.get(at)).ok_or_else(index)?;
                let mode = *p.mode;
                let st = step(p, dec, mode, y, pq, m)?;
                if let Some(g) = st.g
                    && update_split(l, at, lanes, g).ok_or_else(index)?
                {
                    start_split(l, n);
                }
                st.z
            } else {
                C32::new(0.0, 0.0)
            };
            advance(p.mode, p.count, p.cma_symbols);
        }
        // The window of the last symbol becomes the history before the next run.
        l.h.xr.copy_within(first..first + n, RUN);
        l.h.xi.copy_within(first..first + n, RUN);
        l.h.xn.copy_within(first..first + n, RUN);
    }
    Ok(())
}

/// The start of a run of [`linear_runs`]: the symbols of the run `xs` into the history (from
/// `first` on), `|x|²` of every value of its windows, the window powers side by side and the
/// steps `mu / (P + eps)` of the run.
#[allow(
    clippy::inline_always,
    reason = "a part of the symbol loop of every path: two instances call it, and LLVM keeps a helper with two callers out of line"
)]
#[inline(always)]
fn prepare_run(p: &EqualizeParts<'_>, l: &mut Linear<'_>, xs: &[C32], first: usize) -> Option<()> {
    let n = p.taps;
    let fresh =
        l.h.xr
            .get_mut(first..RUN)?
            .iter_mut()
            .zip(l.h.xi.get_mut(first..RUN)?.iter_mut())
            .zip(l.h.xn.get_mut(first..RUN)?.iter_mut())
            .rev();
    for (((r, i), m), &x) in fresh.zip(xs) {
        let v = sanitize(x);
        (*r, *i, *m) = (v.re, v.im, -v.im);
    }
    // |x|² of every value of the windows of the run, then the window powers side by side.
    let used = RUN + n - 1;
    let vals =
        l.h.xr
            .get(first..used)?
            .iter()
            .zip(l.h.xi.get(first..used)?);
    for (m, (&r, &i)) in l.w.mag.get_mut(first..used)?.iter_mut().zip(vals) {
        *m = r * r + i * i;
    }
    let pw = l.w.power.get_mut(first..)?;
    pw.fill(0.0);
    for i in 0..n {
        let ms = l.w.mag.get(first + i..RUN + i)?;
        for (q, &m) in pw.iter_mut().zip(ms) {
            *q += m;
        }
    }
    // The steps `mu / (P + eps)` of the run: the symbols before `cma_left` in CMA mode (the
    // rule of `advance`), the others in decision mode.
    let cma_left = match *p.mode {
        EqMode::Decision => 0,
        EqMode::Cma if p.cma_symbols == u32::MAX => RUN,
        EqMode::Cma => p.cma_symbols.saturating_sub(*p.count) as usize,
    };
    let sw = l.w.steps.get_mut(first..)?;
    for (j, (st, &q)) in sw.iter_mut().zip(pw.iter()).enumerate() {
        // Position `first + j` holds symbol `RUN - 1 - first - j` of the run.
        let mu = if RUN - 1 - first - j < cma_left {
            p.mu_cma
        } else {
            p.mu_dd
        };
        *st = mu / (q + NLMS_EPS);
    }
    Some(())
}

/// [`dot_split`] of the window from `at` and the weights, over `padded` samples.
#[allow(
    clippy::inline_always,
    reason = "a part of the symbol loop of every path: two instances call it, and LLVM keeps a helper with two callers out of line"
)]
#[inline(always)]
fn dot_window(l: &Linear<'_>, at: usize, padded: usize) -> Option<C32> {
    let x = at..at + padded;
    Some(dot_split(
        l.h.xr.get(x.clone())?,
        l.h.xi.get(x.clone())?,
        l.h.xn.get(x)?,
        l.w.wr.get(..padded)?,
        l.w.wi.get(..padded)?,
    ))
}

/// The start weights on the split weights (the padding stays zero).
fn start_split(l: &mut Linear<'_>, taps: usize) {
    let centre = taps / 2;
    for (i, (r, im)) in
        l.w.wr
            .iter_mut()
            .zip(l.w.wi.iter_mut())
            .take(taps)
            .enumerate()
    {
        *r = if i == centre { 1.0 } else { 0.0 };
        *im = 0.0;
    }
}

/// The equalizer with feedback taps (`M > 0`): the joint history moves by one per symbol.
fn equalize_joint<D: DotComplex<C32>, K: Decide>(
    dot: &D,
    p: &mut EqualizeParts<'_>,
    k: K,
    input: &[C32],
    output: &mut [C32],
) -> Result<()> {
    let n = p.taps;
    let len = p.hist.len();
    for (o, &raw) in output.iter_mut().zip(input) {
        let finite = raw.re.is_finite() && raw.im.is_finite();
        // The forward history, newest first: shift by one and store the input (zero when it
        // is not finite).
        p.hist.copy_within(..n.saturating_sub(1), 1);
        if let Some(first) = p.hist.first_mut() {
            *first = sanitize(raw);
        }
        let y = dot.dot_complex(p.hist, p.weights)?;
        let mode = *p.mode;
        let mut feedback = C32::new(0.0, 0.0);
        *o = if finite && y.re.is_finite() && y.im.is_finite() {
            let (forward, power) = window_power(p.hist, n);
            let mu = match mode {
                EqMode::Cma => p.mu_cma,
                EqMode::Decision => p.mu_dd,
            };
            let st = step(p, k, mode, y, forward, mu / (power + NLMS_EPS))?;
            if let Some(g) = st.g {
                update_weights(p.weights, p.hist, g, n);
            }
            feedback = st.feedback;
            st.z
        } else {
            C32::new(0.0, 0.0)
        };
        // The fed-back decisions, newest first.
        if len > n {
            p.hist.copy_within(n..len - 1, n + 1);
            if let Some(slot) = p.hist.get_mut(n) {
                *slot = feedback;
            }
        }
        advance(p.mode, p.count, p.cma_symbols);
    }
    Ok(())
}

/// Adaptive equalizer on one sample per symbol (`C32`).
///
/// ```
/// use sdrk_core::{kernels::equalize::{EqMode, EqParams, Equalize, EqualizeState}, Fallback, C32};
/// // QPSK through the channel 1 + 0.4 z^-1, turned by 0.5 rad: CMA, then decisions with the
/// // phase tracker, remove the echo and the phase.
/// let h = core::f32::consts::FRAC_1_SQRT_2;
/// let sym: Vec<C32> = (0u32..6000)
///     .map(|k| {
///         let b = k.wrapping_mul(2_654_435_761) >> 30;
///         C32::new(if b & 1 == 0 { h } else { -h }, if b & 2 == 0 { h } else { -h })
///     })
///     .collect();
/// let turn = C32::new(0.5f32.cos(), 0.5f32.sin());
/// let input: Vec<C32> = (0..sym.len())
///     .map(|k| (if k == 0 { sym[0] } else { sym[k] + sym[k - 1] * 0.4 }) * turn)
///     .collect();
/// let mut st = EqualizeState::new([C32::default(); 11], [C32::default(); 11], EqParams::default())?;
/// let mut out = vec![C32::default(); input.len()];
/// Fallback.equalize(&mut st, &input, &mut out)?;
/// assert_eq!(st.mode(), EqMode::Decision);
/// assert!(st.mse() < 1e-3);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Equalize {
    /// Per symbol `x`, in f32 without FMA, with `N` forward and `M` feedback taps and the joint
    /// vector `h` (the forward history `h[0..N]`, then the fed-back decisions `h[N..N + M]`):
    ///
    /// 1. The forward history moves by one and `h[0] = x` (`0` when `x` is not finite);
    ///    `y = Σ w[i] h[i]` over all `N + M` elements with the order of [`DotComplex`] `C32`
    ///    (not conjugated, one call).
    /// 2. If `x` and `y` are finite: `(c, s)` = cosine and sine of the tracker phase from the
    ///    table of `rotate` (the interpolation of `costas`; exactly `(1, 0)` at phase 0),
    ///    `z = (y.re c + y.im s, y.im c - y.re s)`, the decision `d` on `z` (the rule of
    ///    [`EqConstellation::decide`]), `ed = z - d`, `p = ed.re ed.re + ed.im ed.im`, and if
    ///    `p` is finite `mse = flush_denormal(mse + 0.01 (p - mse))`.
    /// 3. The error: CMA `f = (y.re y.re + y.im y.im) - R2` (`R2` = 1 for PSK, the `f32` of
    ///    `33 / 25` for 16-QAM and of `29 / 21` for 64-QAM, [`EqConstellation::cma_r2`]) and
    ///    `e = (clip(y.re f), clip(y.im f))` with `clip` to `[-1, 1]` (two selects; NaN stays),
    ///    step `mu_cma`, feedback value 0; decision mode `e = (ed.re c - ed.im s,
    ///    ed.re s + ed.im c)` (`ed e^(j theta)`), step `mu_dd`, feedback value
    ///    `(d.re c - d.im s, d.re s + d.im c)`.
    /// 4. The window power `P = Σ (h.re h.re + h.im h.im)` over the `N + M` elements in
    ///    ascending order from `+0.0` (`Pf` its value after the `N` forward elements),
    ///    `m = mu / (P + 1e-6)`, `g = (m e.re, m e.im)`.
    /// 5. If `Pf` is not 0 and `g` is finite, for every `i`: `w[i] = flush_denormal(w[i] - g conj(h[i]))` per
    ///    component with `g conj(h) = (g.re h.re + g.im h.im, g.im h.re - g.re h.im)`; if a new
    ///    weight is not finite (an overflow), the weights return to the start weights.
    /// 6. The tracker: `pd = z.im d.re - z.re d.im` (0 if not finite),
    ///    `freq = clip(flush_denormal(freq + ki pd), 0.125)`, `phase = phase +
    ///    floor((freq + kp pd) 2^32 + 0.5)` (wrapping, the step of `costas`).
    /// 7. The fed-back decisions move by one and `h[N]` = the feedback value of step 3 (0 when
    ///    `x` or `y` is not finite or `Pf` is 0).
    /// 8. The output is `z`; it is `0` and the weights, `mse` and the tracker stay when `x` or
    ///    `y` is not finite.
    /// 9. In CMA mode the symbol counter advances (for every symbol, also a non-finite one);
    ///    decision mode starts after `cma_symbols` symbols (never for `u32::MAX`) and does not
    ///    switch back; `reset` returns to the start.
    ///
    /// The lengths must be equal, otherwise `LengthMismatch` and nothing changes.
    fn equalize<W, H>(
        &self,
        s: &mut EqualizeState<W, H>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()>
    where
        W: AsRef<[C32]> + AsMut<[C32]>,
        H: AsRef<[C32]> + AsMut<[C32]>;
}
