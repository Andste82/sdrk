//! Symbol timing recovery: a feedback loop that finds the symbol instants of a signal with
//! `sps` samples per symbol and outputs one interpolated sample per symbol.
//!
//! A timing error detector (Gardner or Mueller–Müller, [`Ted`]) measures how far the strobe
//! lies from the symbol centre, a proportional-integral loop filter corrects the strobe
//! position `t` and the symbol period `omega`, and a cubic Lagrange interpolator computes the
//! sample at the fractional position `t`. The arithmetic is fixed (float32 model, see
//! [`SymbolSync`] and `docs/kernels.md`), so `Fallback` and `Reference` are bit-identical.

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::timing::{Delay, Progress, RatioKind, Timing};
use crate::types::{C32, Sample, SampleType};

/// The smallest samples per symbol.
pub const MIN_SPS: f64 = 2.0;
/// The largest samples per symbol (D531).
pub const MAX_SPS: f64 = 1_000_000.0;

/// The strobe position at construction and after `reset`, counted from the first sample of the
/// history: the three zero samples of the history come first, so 4 is the input sample 1, the
/// first position with a full interpolation window `x[0] .. x[3]`.
pub(crate) const START: usize = 4;
/// The samples the history keeps between calls.
pub(crate) const HIST: usize = 3;

/// `1/3`, `1/2` and `1/6` as `f32`, each rounded once.
pub(crate) const THIRD: f32 = 1.0 / 3.0;
pub(crate) const HALF: f32 = 0.5;
pub(crate) const SIXTH: f32 = 1.0 / 6.0;

/// The timing error detector of [`SymbolSyncState`].
///
/// ```
/// use sdrk_core::kernels::symbol_sync::Ted;
/// assert_ne!(Ted::Gardner, Ted::MuellerMuller);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Ted {
    /// Gardner: `(y_prev - y) m` with the mid sample `m` half a symbol before `y`; works
    /// without decisions and without carrier lock (two interpolations per symbol).
    Gardner,
    /// Mueller–Müller: `d_prev y - d y_prev` with the decisions `d = sign(y)`; one
    /// interpolation per symbol, needs symbols that are about `+-1` per component.
    MuellerMuller,
}

/// The loop constants of [`SymbolSyncState`], computed in f64 by
/// [`SymbolSyncState::new`].
///
/// ```
/// use sdrk_core::{kernels::symbol_sync::{SymbolSyncState, Ted}, C32};
/// let st = SymbolSyncState::<C32>::new(4.0, Ted::Gardner, 0.01, 1.0, 0.005, 1.0)?;
/// let g = st.gains();
/// assert!(g.kp > 0.0 && g.ki > 0.0 && g.ki < g.kp);
/// assert_eq!(g.max_integ, 0.005 * 4.0);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymbolSyncGains {
    /// Proportional gain `kp = (4 damping theta / d) / ted_gain` in samples per unit of the
    /// detector output.
    pub kp: f64,
    /// Integral gain `ki = (4 theta^2 / d) / ted_gain`.
    pub ki: f64,
    /// The limit of the integrator, `max_deviation * sps` samples.
    pub max_integ: f64,
}

/// State of the symbol synchronizer.
///
/// It holds the last 3 input samples, the next strobe position `t` (an integer index into the
/// history followed by the input, and the fraction `mu` in f64), the symbol period `omega` and
/// its nominal value `sps`, the integrator of the loop filter, the last output and its
/// decision, and the mid sample of the next strobe (Gardner) once it has been computed.
/// `T` is `C32` or `f32` (the digital receivers have no `Q15` variants, D529).
///
/// ```
/// use sdrk_core::{kernels::symbol_sync::{SymbolSyncState, Ted}, C32, Ratio, RatioKind, Timing};
/// let st = SymbolSyncState::<C32>::new(4.1, Ted::Gardner, 0.01, core::f64::consts::FRAC_1_SQRT_2, 0.005, 1.0)?;
/// assert_eq!(st.sps_estimate(), 4.1);
/// assert_eq!(st.ratio(), Ratio::new(10, 41)?);
/// assert_eq!(st.ratio_kind(), RatioKind::Variable);
/// // loop_bw above 0.1 and sps below 2 are rejected
/// assert!(SymbolSyncState::<C32>::new(4.1, Ted::Gardner, 0.2, 0.7, 0.005, 1.0).is_err());
/// assert!(SymbolSyncState::<f32>::new(1.5, Ted::Gardner, 0.01, 0.7, 0.005, 1.0).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymbolSyncState<T: Sample> {
    pub(crate) sps: f64,
    pub(crate) ted: Ted,
    pub(crate) loop_bw: f64,
    pub(crate) damping: f64,
    pub(crate) max_deviation: f64,
    pub(crate) ted_gain: f64,
    pub(crate) kp: f64,
    pub(crate) ki: f64,
    pub(crate) lim: f64,
    /// The last `HIST` samples, oldest first; index 0 of the strobe positions.
    pub(crate) hist: [T; HIST],
    /// Integer part of the next strobe position.
    pub(crate) i: usize,
    /// Fraction of the next strobe position, `0 <= mu < 1`.
    pub(crate) mu: f64,
    pub(crate) omega: f64,
    pub(crate) integ: f64,
    pub(crate) y_prev: T,
    pub(crate) d_prev: T,
    /// The mid sample of the next strobe, valid when `mid_ready`.
    pub(crate) mid: T,
    pub(crate) mid_ready: bool,
    /// False until the first strobe after construction or `reset` (which uses `e = 0`).
    pub(crate) started: bool,
}

fn invalid(msg: &'static str) -> Error {
    Error::InvalidParameter(msg)
}

impl<T: Sample> SymbolSyncState<T> {
    /// Creates a state with a zero history, the first strobe at input sample 1, `omega = sps`
    /// and a zero integrator.
    ///
    /// `sps = fs / symbol_rate` (2 to 1 000 000, not necessarily an integer); `loop_bw` (0.0001
    /// to 0.1) is the loop bandwidth normalized to the symbol rate; `damping` 0.1 to 4 (1/√2
    /// is the usual choice); `max_deviation` (0 to 0.05) limits `omega` to
    /// `sps (1 +- max_deviation)`; `ted_gain > 0` is the slope of the detector, by which the
    /// loop gains are divided. The loop constants, in f64:
    /// `theta = loop_bw / (damping + 1 / (4 damping))`,
    /// `d = (1 + (2 damping) theta) + theta theta`, `kp = (((4 damping) theta) / d) / ted_gain`,
    /// `ki = ((4 (theta theta)) / d) / ted_gain` (both must be finite). Anything else,
    /// including NaN, gives `InvalidParameter`; a sample type other than `C32` and `f32` gives
    /// `Unsupported`.
    ///
    /// ```
    /// use sdrk_core::{kernels::symbol_sync::{SymbolSyncState, Ted}, CQ15};
    /// assert!(SymbolSyncState::<f32>::new(8.0, Ted::Gardner, 0.01, 0.7071, 0.005, 1.0).is_ok());
    /// assert!(SymbolSyncState::<f32>::new(8.0, Ted::Gardner, 0.01, 0.7071, 0.06, 1.0).is_err());
    /// assert!(SymbolSyncState::<f32>::new(8.0, Ted::Gardner, 0.01, 0.7071, 0.005, 0.0).is_err());
    /// assert!(SymbolSyncState::<CQ15>::new(8.0, Ted::Gardner, 0.01, 0.7071, 0.005, 1.0).is_err());
    /// ```
    pub fn new(
        sps: f64,
        ted: Ted,
        loop_bw: f64,
        damping: f64,
        max_deviation: f64,
        ted_gain: f64,
    ) -> Result<Self> {
        if !matches!(T::TYPE, SampleType::C32 | SampleType::F32) {
            return Err(Error::Unsupported(
                "symbol sync supports only the sample types c32 and f32",
            ));
        }
        if !(sps.is_finite() && (MIN_SPS..=MAX_SPS).contains(&sps)) {
            return Err(invalid(
                "symbol sync samples per symbol must be between 2 and 1000000",
            ));
        }
        if !(loop_bw.is_finite() && (0.0001..=0.1).contains(&loop_bw)) {
            return Err(invalid(
                "symbol sync loop bandwidth must be between 0.0001 and 0.1",
            ));
        }
        if !(damping.is_finite() && (0.1..=4.0).contains(&damping)) {
            return Err(invalid("symbol sync damping must be between 0.1 and 4"));
        }
        if !(max_deviation.is_finite() && (0.0..=0.05).contains(&max_deviation)) {
            return Err(invalid(
                "symbol sync maximum deviation must be between 0 and 0.05",
            ));
        }
        if !(ted_gain.is_finite() && ted_gain > 0.0) {
            return Err(invalid(
                "symbol sync detector gain must be finite and positive",
            ));
        }
        let theta = loop_bw / (damping + 1.0 / (4.0 * damping));
        let d = (1.0 + (2.0 * damping) * theta) + theta * theta;
        // The detector error is a timing error in symbols (`ted_gain` is the detector slope per
        // symbol); the factor `sps` turns the corrections into samples, so `loop_bw` and
        // `ted_gain` mean the same at every `sps` (D565).
        let kp = ((((4.0 * damping) * theta) / d) / ted_gain) * sps;
        let ki = (((4.0 * (theta * theta)) / d) / ted_gain) * sps;
        if !(kp.is_finite() && ki.is_finite()) {
            return Err(invalid(
                "symbol sync loop gains must be finite (detector gain too small)",
            ));
        }
        Ok(Self {
            sps,
            ted,
            loop_bw,
            damping,
            max_deviation,
            ted_gain,
            kp,
            ki,
            lim: max_deviation * sps,
            hist: [T::ZERO; HIST],
            i: START,
            mu: 0.0,
            omega: sps,
            integ: 0.0,
            y_prev: T::ZERO,
            d_prev: T::ZERO,
            mid: T::ZERO,
            mid_ready: false,
            started: false,
        })
    }

    /// Returns to the state after construction: zero history, the first strobe at the next
    /// input sample 1, `omega = sps`, zero integrator, and `e = 0` for the next strobe (the
    /// parameters are kept).
    ///
    /// ```
    /// use sdrk_core::{kernels::symbol_sync::{SymbolSync, SymbolSyncState, Ted}, Fallback};
    /// let mut st = SymbolSyncState::<f32>::new(4.0, Ted::Gardner, 0.05, 0.7071, 0.05, 1.0)?;
    /// let input: Vec<f32> = (0..400).map(|n| if (n / 4) % 2 == 0 { 1.0 } else { -1.0 }).collect();
    /// let mut out = [0.0f32; 128];
    /// Fallback.symbol_sync(&mut st, &input, &mut out)?;
    /// st.reset();
    /// assert_eq!(st, SymbolSyncState::<f32>::new(4.0, Ted::Gardner, 0.05, 0.7071, 0.05, 1.0)?);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn reset(&mut self) {
        self.hist = [T::ZERO; HIST];
        self.i = START;
        self.mu = 0.0;
        self.omega = self.sps;
        self.integ = 0.0;
        self.y_prev = T::ZERO;
        self.d_prev = T::ZERO;
        self.mid = T::ZERO;
        self.mid_ready = false;
        self.started = false;
    }

    /// The current estimate of the samples per symbol (`omega = sps + integrator`).
    pub fn sps_estimate(&self) -> f64 {
        self.omega
    }

    /// The position of the next strobe in input samples, counted from the first sample of the
    /// next call's input (`i - 3 + mu`; it can be negative while the strobe lies in the history).
    /// A test that adds the input consumed so far gets the absolute strobe position (D556).
    ///
    /// ```
    /// use sdrk_core::{kernels::symbol_sync::{SymbolSync, SymbolSyncState, Ted}, Fallback};
    /// let mut st = SymbolSyncState::<f32>::new(4.0, Ted::Gardner, 0.01, 0.7071, 0.005, 1.0)?;
    /// assert_eq!(st.next_strobe(), 1.0); // input sample 1, the first full window
    /// let mut out = [0.0f32; 4];
    /// let p = Fallback.symbol_sync(&mut st, &[0.0; 4], &mut out)?;
    /// // The strobe at sample 1 is out (e = 0 on the first strobe), the next one is at 1 + sps.
    /// assert_eq!((p.consumed, p.produced), (4, 1));
    /// assert_eq!(p.consumed as f64 + st.next_strobe(), 5.0);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn next_strobe(&self) -> f64 {
        (self.i as f64 - HIST as f64) + self.mu
    }

    /// The nominal samples per symbol.
    pub fn sps(&self) -> f64 {
        self.sps
    }

    /// The timing error detector.
    pub fn ted(&self) -> Ted {
        self.ted
    }

    /// The loop bandwidth, normalized to the symbol rate.
    pub fn loop_bw(&self) -> f64 {
        self.loop_bw
    }

    /// The damping of the loop.
    pub fn damping(&self) -> f64 {
        self.damping
    }

    /// The largest deviation of `omega` from `sps`, as a fraction of `sps`.
    pub fn max_deviation(&self) -> f64 {
        self.max_deviation
    }

    /// The detector gain the loop gains are divided by.
    pub fn ted_gain(&self) -> f64 {
        self.ted_gain
    }

    /// The loop constants.
    pub fn gains(&self) -> SymbolSyncGains {
        SymbolSyncGains {
            kp: self.kp,
            ki: self.ki,
            max_integ: self.lim,
        }
    }
}

impl<T: Sample> Timing for SymbolSyncState<T> {
    /// The nominal ratio `1 / sps` as the reduced fraction `10^6 / round(sps 10^6)` (exact for
    /// every `sps` with at most six decimals; the block builds the exact ratio from the symbol
    /// rate, D532).
    fn ratio(&self) -> Ratio {
        let den = libm::floor(self.sps * 1_000_000.0 + 0.5) as u64;
        Ratio::new(1_000_000, den).unwrap_or(Ratio::ONE)
    }

    /// Variable: the loop moves the strobes.
    fn ratio_kind(&self) -> RatioKind {
        RatioKind::Variable
    }

    /// One input sample: the first strobe is input sample 1.
    fn group_delay(&self) -> Delay {
        Delay::new(1, 1).unwrap_or(Delay::ZERO)
    }
}

/// The per-sample arithmetic of the float32 model, implemented for `f32` and `C32` (used by
/// `Fallback`; `Reference` writes the same formulas out on its own).
pub(crate) trait SyncSample: Sample {
    /// Cubic Lagrange interpolation at `i + mu` from `x[i-1], x[i], x[i+1], x[i+2]`.
    fn interp(w: &[Self; 4], mu: f32) -> Self;
    /// The Gardner error `re((y_prev - y) conj(m))`.
    fn gardner(y_prev: Self, y: Self, m: Self) -> f32;
    /// The decision `sign(y)` per component (0 for zero and NaN).
    fn decide(y: Self) -> Self;
    /// The Mueller–Müller error `re(conj(d_prev) y - conj(d) y_prev)`.
    fn mm(d_prev: Self, y: Self, d: Self, y_prev: Self) -> f32;
}

#[inline]
fn interp1(xm1: f32, x0: f32, x1: f32, x2: f32, mu: f32) -> f32 {
    let c1 = ((-(xm1 * THIRD) - x0 * HALF) + x1) - x2 * SIXTH;
    let c2 = (xm1 * HALF - x0) + x1 * HALF;
    let c3 = ((-(xm1 * SIXTH) + x0 * HALF) - x1 * HALF) + x2 * SIXTH;
    ((c3 * mu + c2) * mu + c1) * mu + x0
}

#[inline]
fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

impl SyncSample for f32 {
    #[inline]
    fn interp(w: &[f32; 4], mu: f32) -> f32 {
        let [xm1, x0, x1, x2] = *w;
        interp1(xm1, x0, x1, x2, mu)
    }

    #[inline]
    fn gardner(y_prev: f32, y: f32, m: f32) -> f32 {
        (y_prev - y) * m
    }

    #[inline]
    fn decide(y: f32) -> f32 {
        sign(y)
    }

    #[inline]
    fn mm(d_prev: f32, y: f32, d: f32, y_prev: f32) -> f32 {
        d_prev * y - d * y_prev
    }
}

impl SyncSample for C32 {
    #[inline]
    fn interp(w: &[C32; 4], mu: f32) -> C32 {
        let [xm1, x0, x1, x2] = *w;
        C32::new(
            interp1(xm1.re, x0.re, x1.re, x2.re, mu),
            interp1(xm1.im, x0.im, x1.im, x2.im, mu),
        )
    }

    #[inline]
    fn gardner(y_prev: C32, y: C32, m: C32) -> f32 {
        (y_prev.re - y.re) * m.re + (y_prev.im - y.im) * m.im
    }

    #[inline]
    fn decide(y: C32) -> C32 {
        C32::new(sign(y.re), sign(y.im))
    }

    #[inline]
    fn mm(d_prev: C32, y: C32, d: C32, y_prev: C32) -> f32 {
        (d_prev.re * y.re + d_prev.im * y.im) - (d.re * y_prev.re + d.im * y_prev.im)
    }
}

/// Symbol timing recovery.
///
/// ```
/// use sdrk_core::{kernels::symbol_sync::{SymbolSync, SymbolSyncState, Ted}, Fallback};
/// // Alternating symbols +1, -1 with 4 samples each (a square wave of period 8 samples).
/// let input: Vec<f32> = (0..4000).map(|n| if (n / 4) % 2 == 0 { 1.0 } else { -1.0 }).collect();
/// let mut st = SymbolSyncState::<f32>::new(4.0, Ted::Gardner, 0.01, 0.7071, 0.005, 1.0)?;
/// let mut out = vec![0.0f32; 1100];
/// let p = Fallback.symbol_sync(&mut st, &input, &mut out)?;
/// // All input is taken (the last 3 samples stay in the state), about one output per 4.
/// assert_eq!(p.consumed, 4000);
/// assert!((995..=1005).contains(&p.produced));
/// // The strobes settle in the middle of the symbols.
/// assert!(out[p.produced - 100..p.produced].iter().all(|y| y.abs() == 1.0));
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait SymbolSync<T: Sample> {
    /// Interpolates one output per symbol and adjusts the strobes; returns the input consumed
    /// and the outputs produced.
    ///
    /// The strobe positions count the samples `b` = the 3 history samples followed by
    /// `input`. Per strobe at `t = i + mu`, while `i + 2` is an index of `b` and `output` has
    /// room (all f32 operations one at a time, no FMA):
    ///
    /// 1. `y = interp(t)`; Gardner also `m = interp(t - omega/2)` (computed as soon as its
    ///    window is available, possibly in an earlier call). `interp` is the cubic Lagrange
    ///    interpolation from `b[i-1] .. b[i+2]` with `mu as f32`, per component.
    /// 2. The error `e` (f32): Gardner `re((y_prev - y) conj(m))`; Mueller–Müller with the
    ///    decisions `d = sign(y)` per component, `re(conj(d_prev) y - conj(d) y_prev)`. The
    ///    first strobe after construction or `reset` uses `e = 0`, and so does a non-finite
    ///    `e` (D533).
    /// 3. Output `y`; with `e` as f64: `integ = clamp(flush(integ + ki e), -lim, lim)`,
    ///    `omega = sps + integ`, `v = (mu + omega) + clamp(kp e, -omega/4, omega/4)`,
    ///    `i = i + floor(v)`, `mu = v - floor(v)` (D530, D534).
    ///
    /// At the end, `k = min(i_p - 1, input.len())` input samples are consumed, where `i_p`
    /// is the integer part of the next position the loop needs (the mid sample of the next
    /// strobe when it is still missing, else the strobe); the history becomes `b[k .. k+3]`
    /// and `k` is subtracted from the positions. When `output` has room this is the whole
    /// input; when it is full the caller resubmits the rest.
    fn symbol_sync(
        &self,
        s: &mut SymbolSyncState<T>,
        input: &[T],
        output: &mut [T],
    ) -> Result<Progress>;
}
