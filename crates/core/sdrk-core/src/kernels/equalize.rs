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
use crate::fallback::mix::interp_f32_pair;
use crate::fallback::modulate::phase_step;
use crate::kernels::check_len;
use crate::kernels::costas::{derotate, loop_gains, octant_point};
use crate::kernels::dot::DotComplex;
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
/// A full turn of the tracker phase, `2^32`.
pub(crate) const TURN: f64 = 4_294_967_296.0;

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
}

impl EqConstellation {
    /// The nearest constellation point, with the decision rule of `slice` and `costas`:
    /// BPSK `+1` for `re >= 0` (including `-0.0`), else `-1`; QPSK the signs of both components
    /// in the same way, scaled by `1 / sqrt(2)` (as `f32`); 8PSK the point of the half-open
    /// octant of `costas` order 8 (zero takes point 0). A NaN component counts as not negative.
    ///
    /// ```
    /// use sdrk_core::{kernels::equalize::EqConstellation, C32};
    /// assert_eq!(EqConstellation::Bpsk.decide(C32::new(-0.1, 5.0)), C32::new(-1.0, 0.0));
    /// assert_eq!(EqConstellation::Bpsk.decide(C32::new(0.0, -1.0)), C32::new(1.0, 0.0));
    /// let d = EqConstellation::Psk8.decide(C32::new(0.0, 1.0));
    /// assert!(d.re < 0.0 && d.im > 0.9); // the octant [pi/2, 3pi/4)
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
        }
    }

    /// The CMA radius `R2 = E|s|⁴ / E|s|²` of the constellation: 1 for every PSK constellation
    /// (QAM in M44 has its own value).
    ///
    /// ```
    /// use sdrk_core::kernels::equalize::EqConstellation;
    /// assert_eq!(EqConstellation::Bpsk.cma_r2(), 1.0);
    /// ```
    #[inline]
    pub fn cma_r2(self) -> f32 {
        match self {
            Self::Bpsk | Self::Qpsk | Self::Psk8 => 1.0,
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
#[inline]
pub(crate) fn track(phase: &mut u32, freq: &mut f32, kp: f32, ki: f32, z: C32, d: C32) {
    let pd = z.im * d.re - z.re * d.im;
    let pd = if pd.is_finite() { pd } else { 0.0 };
    *freq = clip(flush_denormal(*freq + ki * pd), PHASE_MAX_FREQ);
    *phase = phase.wrapping_add(phase_step(f64::from(*freq + kp * pd), TURN));
}

/// One symbol after the filter output `y` (finite): the tracker rotation, the decision, `mse`,
/// the error and the factor `g = (mu / (P + eps)) e`. Returns the output `z`, the value to feed
/// back (`d e^(j theta)` in decision mode, else 0) and `g` (`None` when it is not finite).
struct Step {
    z: C32,
    feedback: C32,
    g: Option<C32>,
}

#[inline]
fn step(p: &mut EqualizeParts<'_>, mode: EqMode, y: C32) -> Result<Step> {
    // Phase 0 is the identity (the table's sine at 0 is not exactly 0).
    let (c, s) = if *p.phase == 0 {
        (1.0, 0.0)
    } else {
        interp_f32_pair(*p.phase).ok_or(Error::Internal("cosine table index"))?
    };
    let z = derotate(y, c, s);
    let d = p.constellation.decide(z);
    let ed = C32::new(z.re - d.re, z.im - d.im);
    let pe = ed.re * ed.re + ed.im * ed.im;
    if pe.is_finite() {
        *p.mse = flush_denormal(*p.mse + MSE_ALPHA * (pe - *p.mse));
    }
    let (e, mu, feedback) = match mode {
        EqMode::Cma => {
            let f = (y.re * y.re + y.im * y.im) - p.constellation.cma_r2();
            let e = C32::new(clip(y.re * f, CMA_CLIP), clip(y.im * f, CMA_CLIP));
            (e, p.mu_cma, C32::new(0.0, 0.0))
        }
        EqMode::Decision => (rotate(ed, c, s), p.mu_dd, rotate(d, c, s)),
    };
    let (forward, power) = window_power(p.hist, p.taps);
    let m = mu / (power + NLMS_EPS);
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

/// The equalizer with the filter part `y = Σ w[i] x[i]` computed by `dot` (any implementation
/// of [`DotComplex`] `C32`, all of them bit-identical) over the joint vector of forward
/// history and fed-back decisions: `Fallback` runs it with `Fallback`, the SIMD backend with
/// its own `dot_complex`. The rest of the loop is the scalar code of this crate. See
/// [`Equalize::equalize`] for the arithmetic.
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
    let n = p.taps;
    let len = p.hist.len();
    for (o, &raw) in output.iter_mut().zip(input) {
        let finite = raw.re.is_finite() && raw.im.is_finite();
        // The forward history, newest first: shift by one and store the input (zero when it
        // is not finite).
        p.hist.copy_within(..n.saturating_sub(1), 1);
        if let Some(first) = p.hist.first_mut() {
            *first = if finite { raw } else { C32::new(0.0, 0.0) };
        }
        let y = dot.dot_complex(p.hist, p.weights)?;
        let mode = *p.mode;
        let mut feedback = C32::new(0.0, 0.0);
        *o = if finite && y.re.is_finite() && y.im.is_finite() {
            let st = step(&mut p, mode, y)?;
            if let Some(g) = st.g {
                let mut bad = false;
                for (w, &x) in p.weights.iter_mut().zip(p.hist.iter()) {
                    let v = update_weight(*w, g, x);
                    bad |= !(v.re.is_finite() && v.im.is_finite());
                    *w = v;
                }
                if bad {
                    start_weights(p.weights, n);
                }
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
    /// 3. The error: CMA `f = (y.re y.re + y.im y.im) - R2` (`R2` = 1 for PSK) and
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
