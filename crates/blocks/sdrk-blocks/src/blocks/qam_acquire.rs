//! `qam_acquire`: the frequency acquisition of square QAM by the fourth power (the estimate of
//! the carrier offset of `qam_demod`, M44.1 §2.2), with an optional correction of the offset.

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, Scaling};
use sdrk_core::num::flush_denormal;

use super::{head, head_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};

/// Symbols skipped at the start of a search before its first window (the pull-in of the timing
/// loop in front of the acquisition).
pub const QAM_ACQUISITION_SKIP: usize = 256;
/// Symbols of one window of the fourth-power frequency estimate.
pub const QAM_ACQUISITION_SYMBOLS: usize = 512;
/// The symbol of a search or a verify window (counted from its start) at which its window is
/// complete.
pub(crate) const WINDOW_END: u16 = (QAM_ACQUISITION_SKIP + QAM_ACQUISITION_SYMBOLS) as u16;
/// The FFT size of the estimate: the symbols zero-padded to twice their number.
const ACQUISITION_FFT: usize = 2 * QAM_ACQUISITION_SYMBOLS;
/// The smallest peak of the periodogram of the fourth powers, relative to their energy
/// (`|X_k|^2 >= ACQUISITION_MIN_PEAK sum |q_n|^2`), for which the estimate is used. Gaussian
/// noise reaches it in 2 of 10000 windows; QAM gives 13 to 46 in the tests (D1403).
const ACQUISITION_MIN_PEAK: f32 = 12.0;
/// The smoothing factor of the symbol power `p` by which the fourth powers are normalized
/// (`p = p + ACQUISITION_POWER_ALPHA (|x|^2 - p)`, a time constant of 64 symbols), so that a
/// level that still changes (an AGC that settles) does not weight the end of the window more.
const ACQUISITION_POWER_ALPHA: f32 = 1.0 / 64.0;

/// Description of the `qam_acquire` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "qam_acquire",
    summary: "Estimates the carrier offset of square QAM symbols (4-QAM to 1024-QAM) by the \
        fourth power and optionally turns it out.",
    details: "The input is one complex sample per symbol (after symbol_sync). From symbol 256 \
        on, windows of 512 symbols of x^4 / p^2 (p the symbol power smoothed over 64 symbols) \
        are zero-padded to 1024 points and transformed by fft; the largest bin, refined by a \
        parabola, gives the offset in the range +-0.125 cycles per symbol, which is used when \
        the peak is at least 12 times the energy of the window; otherwise the next window \
        follows at once. After a used estimate the block stops searching. With correct false \
        the output is the input, bit for bit (the block only estimates; the mode of qam_demod, \
        whose controller reads the estimate); with correct true the output is the input turned \
        back by the estimate, x[n] exp(-j 2 pi f n), with the phase continuous across \
        estimates (for example in front of a costas). The fourth power cannot resolve a \
        quarter cycle per symbol, and the estimate has the 90 degree ambiguity of the square \
        constellation.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32],
        description: "Symbols, one sample per symbol, unit average power.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32],
        description: "The symbols, unchanged or turned back by the estimate.",
    }],
    params: &[ParamDesc {
        name: "correct",
        unit: Unit::None,
        kind: ParamKind::Bool { default: false },
        typical: "false behind a receiver that takes the estimate; true in front of a costas",
        mutability: Mutability::Build,
        description: "Whether the output is turned back by the estimate (true) or is the input \
            (false).",
    }],
    required: &[],
};

/// Configuration of [`QamAcquireBlock`].
///
/// ```
/// use sdrk_blocks::{ParamValue, QamAcquireConfig};
/// let mut cfg = QamAcquireConfig::default();
/// assert!(!cfg.correct);
/// cfg.set("correct", ParamValue::Bool(true))?;
/// assert_eq!(cfg.get("correct")?, ParamValue::Bool(true));
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QamAcquireConfig {
    /// Whether the output is turned back by the estimate (default `false`: the output is the
    /// input).
    pub correct: bool,
}

impl QamAcquireConfig {
    /// Sets a parameter by its description name; checks the kind.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "correct" => self.correct = value.as_bool()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    ///
    /// # Errors
    /// `UnknownParameter`.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "correct" => Ok(ParamValue::Bool(self.correct)),
            _ => unknown(),
        }
    }
}

/// A window of [`QamAcquireBlock`] that completed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcquireWindow {
    /// The estimate of the window in cycles per symbol when its peak was clear.
    pub estimate: Option<f64>,
    /// Whether it was a verify window (which does not replace the held estimate).
    pub verify: bool,
}

/// What the acquisition does with the symbols.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Searching: windows of the fourth powers until one has a clear peak.
    Search,
    /// One verify window, whose result does not replace the estimate.
    Verify,
    /// No window (an estimate held): the symbols only pass.
    Idle,
}

/// The frequency acquisition of square QAM (4-QAM to 1024-QAM, one window length for every order, D1519) by the fourth power: `C32` symbols
/// in, `C32` symbols out, one sample per symbol (M44.1 §2.2; the acquisition of `qam_demod`,
/// D1403 to D1405, D1421).
///
/// Per symbol `x` the smoothed power `p = flush(p + (|x|^2 - p) / 64)` (finite values only, from
/// 0 at the start; only while a window runs). From symbol [`QAM_ACQUISITION_SKIP`] of a search or
/// a verify window on, [`QAM_ACQUISITION_SYMBOLS`] symbols of `q = x^4 / (p p)` (0 when not finite
/// or `p` is not above 0). For square QAM the mean of `x^4` is `E[s^4] e^(j 2 pi 4 f n)` with the
/// offset `f` in cycles per symbol (`E[s^4]` = -0.68 and -0.62 at unit power, not 0). The
/// periodogram of `q` zero-padded to 1024 points (the kernel `fft`) has its largest bin `k` near
/// `4096 f`; the parabola through the magnitudes of the bins `k - 1`, `k`, `k + 1` refines it,
/// and `f = ((k + delta) / 1024 wrapped into [-0.5, 0.5)) / 4`, the range `[-0.125, 0.125)` cycles
/// per symbol. A window is *clear* when its peak stands clear of the noise (`|X_k|^2 >= 12 sum
/// |q_n|^2`).
///
/// A search (from the start, a [`reset`](Block::reset) or [`search`](QamAcquireBlock::search))
/// takes windows until one is clear, each following the last at once (without the skip), and
/// then holds its estimate ([`estimate`](QamAcquireBlock::estimate)) and takes no further window.
/// [`verify`](QamAcquireBlock::verify) takes one more window (with the skip; the power `p` runs
/// from its first symbol) and reports it without replacing the estimate. The window that
/// completes in a call is returned by [`process_in_place`](QamAcquireBlock::process_in_place). A window completes in the call of its last symbol;
/// [`remaining`](QamAcquireBlock::remaining) tells how many symbols that is away, so that a
/// caller can end a run there.
///
/// With `correct` false the output is the input, bit for bit, and
/// [`process_in_place`](QamAcquireBlock::process_in_place) only estimates; with `correct` true
/// every symbol is turned back by the estimate, `x[n] exp(-j 2 pi phi[n])` with the phase `phi`
/// in units of `2^-32` turn advancing by `round(f 2^32)` per symbol (continuous across
/// estimates; a new estimate applies from the symbol after the one that completes its window).
/// Timing: constant ratio 1, delay 0. Storage: the 1024 `C32` of the FFT and its 512 twiddles.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, QamAcquireBlock, QamAcquireConfig};
/// let rate = Rate::hz(1000)?;
/// let cfg = QamAcquireConfig::default();
/// let mut mem = vec![0u8; QamAcquireBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate)? + 16];
/// let mut b = QamAcquireBlock::new(&cfg, rate, &mut ArenaStorage::new(&mut mem))?;
/// // 16-QAM corner points turned by 0.01 cycles per symbol
/// let x: Vec<C32> = (0..1000u32)
///     .map(|n| {
///         let a = 0.3162 * if (n * 7 + n / 3) % 2 == 0 { 3.0 } else { 1.0 };
///         let t = std::f32::consts::TAU * (0.01 * n as f32 + 0.125);
///         C32::new(a * t.cos(), a * t.sin())
///     })
///     .collect();
/// let mut y = vec![C32::default(); x.len()];
/// let p = b.process(&x, &mut y)?;
/// assert_eq!((p.consumed, p.produced), (1000, 1000));
/// assert_eq!(y, x); // correct false: the input
/// assert!((b.estimate().unwrap() - 0.01).abs() < 1e-4);
/// assert_eq!(b.window_clear(), Some(true));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct QamAcquireBlock<S: Storage> {
    /// The twiddles of the FFT (the plan is made from them for each window, so that the block
    /// stays small).
    twiddles: S::Buf<C32>,
    /// The fourth powers, then the spectrum.
    buf: S::Buf<C32>,
    /// The held estimate in cycles per symbol, NaN while there is none.
    estimate: f64,
    /// The smoothed symbol power.
    power: f32,
    /// The phase of the correction in units of `2^-32` turn.
    phase: u32,
    /// Symbols of the current search or verify window seen (from [`QAM_ACQUISITION_SKIP`] on
    /// after a window without a clear peak), up to [`WINDOW_END`].
    seen: u16,
    mode: Mode,
    /// Whether a window was completed: 0 none since the start or `reset`, 1 not clear, 2 clear.
    clear: u8,
    correct: bool,
}

impl<S: Storage> core::fmt::Debug for QamAcquireBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QamAcquireBlock")
            .field("mode", &self.mode)
            .field("seen", &self.seen)
            .field("estimate", &self.estimate())
            .field("window_clear", &self.window_clear())
            .field("correct", &self.correct)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> QamAcquireBlock<S> {
    /// Creates the block, searching; the twiddles and the window buffer come from `storage`.
    ///
    /// # Errors
    /// `Storage` when the storage is too small.
    pub fn new(config: &QamAcquireConfig, _rate_in: Rate, storage: &mut S) -> Result<Self> {
        let mut twiddles = storage.alloc::<C32>(ACQUISITION_FFT / 2)?;
        FftPlan::new(ACQUISITION_FFT, twiddles.as_mut())?;
        Ok(Self {
            twiddles,
            buf: storage.alloc::<C32>(ACQUISITION_FFT)?,
            estimate: f64::NAN,
            power: 0.0,
            phase: 0,
            seen: 0,
            mode: Mode::Search,
            clear: 0,
            correct: config.correct,
        })
    }

    /// Adds the storage of the block to `plan`: the twiddles, then the window buffer.
    pub(crate) fn plan(plan: &mut Plan) -> Result<()> {
        plan.add::<C32>(ACQUISITION_FFT / 2)?;
        plan.add::<C32>(ACQUISITION_FFT)?;
        Ok(())
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more): 512 and 1024 `C32`.
    ///
    /// # Errors
    /// None for this configuration; the signature matches the other blocks.
    pub fn storage_bytes(_config: &QamAcquireConfig, _rate_in: Rate) -> Result<usize> {
        let mut plan = Plan::default();
        Self::plan(&mut plan)?;
        Ok(plan.bytes())
    }

    /// Output rate: the input rate.
    ///
    /// # Errors
    /// None; the signature matches the other blocks.
    pub fn rate_out(_config: &QamAcquireConfig, rate_in: Rate) -> Result<Rate> {
        Ok(rate_in)
    }

    /// The configuration of the block.
    pub fn config(&self) -> QamAcquireConfig {
        QamAcquireConfig {
            correct: self.correct,
        }
    }

    /// The held estimate in cycles per symbol: that of the clear window that ended the last
    /// search; `None` while searching.
    pub fn estimate(&self) -> Option<f64> {
        (!self.estimate.is_nan()).then_some(self.estimate)
    }

    /// Whether the last completed window had a clear peak; `None` before the first window since
    /// the start or `reset`.
    pub fn window_clear(&self) -> Option<bool> {
        match self.clear {
            0 => None,
            c => Some(c == 2),
        }
    }

    /// Whether the block takes a window (a search or a verify window runs).
    pub fn active(&self) -> bool {
        self.mode != Mode::Idle
    }

    /// Symbols until the current window completes (the window completes with the last of
    /// them); `usize::MAX` while no window runs.
    pub fn remaining(&self) -> usize {
        match self.mode {
            Mode::Idle => usize::MAX,
            _ => usize::from(WINDOW_END - self.seen),
        }
    }

    /// Starts a new search: drops the estimate and begins a window at the next symbol (after the
    /// skip of [`QAM_ACQUISITION_SKIP`] symbols; the power `p` is kept).
    pub fn search(&mut self) {
        self.estimate = f64::NAN;
        self.mode = Mode::Search;
        self.seen = 0;
    }

    /// Takes one window from the next symbol on (after the skip) as a verify window: its result
    /// is reported by [`process_in_place`](QamAcquireBlock::process_in_place) and
    /// [`window_clear`](QamAcquireBlock::window_clear) and does not replace the estimate.
    pub fn verify(&mut self) {
        self.mode = Mode::Verify;
        self.seen = 0;
    }

    /// Takes the symbols of `data` in place: the same as [`Block::process`] with `data` as input
    /// and output (with `correct` false the samples do not change; the call of `qam_demod`).
    /// Returns the last window that completed in the call, if one did.
    ///
    /// # Errors
    /// `Core` when the FFT fails (it does not for the storage of the block).
    pub fn process_in_place(&mut self, data: &mut [C32]) -> Result<Option<AcquireWindow>> {
        let mut done = None;
        let mut rest = data;
        while !rest.is_empty() {
            let n = rest.len().min(self.remaining());
            let (now, later) = rest.split_at_mut(n);
            if self.correct {
                self.turn(now);
            }
            if self.mode != Mode::Idle {
                self.collect(now);
                if self.seen == WINDOW_END {
                    done = Some(self.window_end()?);
                }
            }
            rest = later;
        }
        Ok(done)
    }

    /// Turns `x` back by the estimate (nothing while there is none and the phase is 0).
    fn turn(&mut self, x: &mut [C32]) {
        let step = match self.estimate() {
            Some(f) => libm::round(f * 4_294_967_296.0) as i64 as u32,
            None if self.phase == 0 => return,
            None => 0,
        };
        for v in x {
            let t = (self.phase as i32) as f32 * (core::f32::consts::TAU / 4_294_967_296.0);
            let (s, c) = (libm::sinf(t), libm::cosf(t));
            *v = C32::new(v.re * c + v.im * s, v.im * c - v.re * s);
            self.phase = self.phase.wrapping_add(step);
        }
    }

    /// Takes the next symbols of a window (at most [`QamAcquireBlock::remaining`]).
    fn collect(&mut self, x: &[C32]) {
        let buf = self.buf.as_mut();
        for &v in x {
            let p = v.re * v.re + v.im * v.im;
            if p.is_finite() {
                self.power =
                    flush_denormal(self.power + ACQUISITION_POWER_ALPHA * (p - self.power));
            }
            let n = usize::from(self.seen);
            if let Some(i) = n
                .checked_sub(QAM_ACQUISITION_SKIP)
                .filter(|&i| i < QAM_ACQUISITION_SYMBOLS)
            {
                // The fourth power, and the zero of the padding that the last FFT overwrote (a
                // store per symbol instead of clearing the padding at once).
                if let Some(slot) = buf.get_mut(i) {
                    *slot = normalized_fourth_power(v, self.power);
                }
                if let Some(pad) = buf.get_mut(i + QAM_ACQUISITION_SYMBOLS) {
                    *pad = C32::new(0.0, 0.0);
                }
            }
            self.seen += 1;
        }
    }

    /// The end of a window (out of line: once per window).
    #[inline(never)]
    #[cold]
    fn window_end(&mut self) -> Result<AcquireWindow> {
        let f = self.estimate_now()?;
        self.clear = if f.is_some() { 2 } else { 1 };
        let window = AcquireWindow {
            estimate: f,
            verify: self.mode == Mode::Verify,
        };
        match (self.mode, f) {
            (Mode::Search, Some(f)) => {
                self.estimate = f;
                self.mode = Mode::Idle;
            }
            // The next window at once, without the skip.
            (Mode::Search, None) => self.seen = QAM_ACQUISITION_SKIP as u16,
            _ => self.mode = Mode::Idle,
        }
        Ok(window)
    }

    fn estimate_now(&mut self) -> Result<Option<f64>> {
        let buf = self.buf.as_mut();
        // The padding `buf[512..]` was zeroed by `collect`, symbol by symbol.
        let energy: f32 = buf
            .get(..QAM_ACQUISITION_SYMBOLS)
            .unwrap_or_default()
            .iter()
            .map(|v| v.re * v.re + v.im * v.im)
            .sum();
        let plan = FftPlan::from_twiddles(ACQUISITION_FFT, self.twiddles.as_ref())?;
        <Sdrk as Fft<C32>>::fft(&Sdrk, &plan, buf, Direction::Forward, Scaling::Default)?;
        let power = |v: &C32| v.re * v.re + v.im * v.im;
        let mut bin = 0usize;
        let mut peak = -1.0f32;
        for (i, v) in buf.iter().enumerate() {
            let p = power(v);
            if p > peak {
                peak = p;
                bin = i;
            }
        }
        if !(energy.is_finite() && peak.is_finite() && energy > 0.0)
            || peak < ACQUISITION_MIN_PEAK * energy
        {
            return Ok(None);
        }
        let size = ACQUISITION_FFT;
        let mag = |i: usize| {
            buf.get(i % size)
                .map_or(0.0, |v| libm::sqrt(f64::from(power(v))))
        };
        let (below, top, above) = (mag(bin + size - 1), mag(bin), mag(bin + 1));
        let den = below - 2.0 * top + above;
        let delta = if den < 0.0 {
            0.5 * (below - above) / den
        } else {
            0.0
        };
        let mut cycles = (bin as f64 + delta) / size as f64;
        if cycles >= 0.5 {
            cycles -= 1.0;
        }
        Ok(Some(cycles / 4.0))
    }
}

/// `x^4 / (p p)` in `f32`: `a = x^2 = (x.re x.re - x.im x.im, 2 x.re x.im)`,
/// `q = a^2 = (a.re a.re - a.im a.im, 2 a.re a.im)`, each part divided by `p p`; 0 when `p` is
/// not above 0 or the result is not finite.
#[inline]
fn normalized_fourth_power(x: C32, p: f32) -> C32 {
    let a = C32::new(x.re * x.re - x.im * x.im, 2.0 * x.re * x.im);
    let pp = p * p;
    let q = C32::new((a.re * a.re - a.im * a.im) / pp, (2.0 * a.re * a.im) / pp);
    if p > 0.0 && q.re.is_finite() && q.im.is_finite() {
        q
    } else {
        C32::new(0.0, 0.0)
    }
}

impl<S: Storage> Block for QamAcquireBlock<S> {
    type In = C32;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[C32], output: &mut [C32]) -> Result<Progress> {
        let n = input.len().min(output.len());
        let out = head_mut(output, n)?;
        out.copy_from_slice(head(input, n)?);
        self.process_in_place(out)?;
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: Ratio::ONE,
            kind: RatioKind::Constant,
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {
        self.estimate = f64::NAN;
        self.power = 0.0;
        self.phase = 0;
        self.seen = 0;
        self.mode = Mode::Search;
        self.clear = 0;
    }

    fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
        match DESCRIPTION.param(name) {
            Some(_) => Err(Error::ReadOnlyParameter),
            None => unknown(),
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config().get(name)
    }

    fn min_input(&self) -> usize {
        1
    }
}
