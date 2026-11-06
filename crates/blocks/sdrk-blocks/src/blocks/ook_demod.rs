//! `ook_demod`: on-off keying receiver (magnitude, boxcar matched filter, swing
//! normalization, Gardner symbol timing, decision-directed level trackers after `rtl_433`), built
//! from the kernels `mag`, `resample_real` and `symbol_sync`.

use sdrk::{Delay, Progress, Rate, RatioKind, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::decim::{ResampleReal, ResampleRealState};
use sdrk_core::kernels::spectrum::Mag;
use sdrk_core::kernels::symbol_sync::SymbolSync;

use super::symbol_sync::SymbolSyncBlock;
use super::{HZ_LIMIT, check_count, head, head_mut, hz, tail, tail_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};
use crate::{SymbolSyncConfig, Ted};

/// Input samples per piece the block processes at a time (the size of its two work buffers).
/// Pieces are aligned to multiples of this count since construction or reset; the swing
/// normalization of a block of this many samples is fixed at its first sample (D720).
pub const OOK_DEMOD_CHUNK: usize = 256;

/// Ratio of the time constants of the off and the on level trackers: `rtl_433`
/// (`src/pulse_detect.c`) moves its low (noise) estimate by 1/1024 and its high estimate by 1/64
/// of the distance per sample (`OOK_EST_LOW_RATIO`, `OOK_EST_HIGH_RATIO`), 16 times slower
/// (D720).
pub const OOK_LOW_RATIO: f64 = 16.0;

/// Position of the threshold between the tracked off and on levels after the lead-in,
/// `t = lo + OOK_THRESHOLD (hi - lo)` (D727). The envelope of an on symbol (Rician) spreads
/// more than that of an off symbol (Rayleigh): measured behind the channel filter of
/// `ook_receiver` at Eb/N0 15 dB, 0.142 and 0.096 of the swing, so the midpoint of `rtl_433`
/// turns most errors into lost ones; the point of equal distance in units of the spread is
/// 0.40, but the off symbols between two on symbols have a longer tail (the timing jitter lets
/// some energy of the neighbours in). Measured on test 4 of M39.1, the fewest wrong payload
/// bits are at 0.44 to 0.46.
pub const OOK_THRESHOLD: f32 = 0.45;

/// Smallest swing `hi - lo` the boxcar output is divided by (D720).
pub const OOK_MIN_SWING: f32 = 1e-6;

/// Fewest samples per symbol (`rate_unsupported` below; D639).
const MIN_SPS: f64 = 4.0;

/// Most samples per symbol (bounds the matched filter to 1024 taps; D639).
const MAX_SPS: f64 = 1024.0;

/// Largest `track_symbols` (D639).
const MAX_TRACK: usize = 100_000;

/// Bandwidth of the timing loop normalized to the symbol rate (the `symbol_sync` default).
const LOOP_BW: f64 = 0.01;

/// Damping of the timing loop (the `symbol_sync` default).
#[allow(clippy::approx_constant)]
const DAMPING: f64 = 0.7071;

/// Largest deviation of the symbol period (the `symbol_sync` default).
const MAX_DEVIATION: f64 = 0.005;

/// Gardner slope per symbol of the boxcar-filtered magnitude of on-off keying with the levels
/// 0 and 1 (rectangular pulses), measured (D636): a quarter of the slope of 2-FSK with the
/// levels ±1, since the swing is half as large and the detector multiplies two differences.
/// The block divides the boxcar output by the tracked swing `hi - lo` before the timing loop,
/// so the slope holds at any input level; a common level of both symbols adds no mean error
/// (D720).
pub const TED_GAIN_OOK: f64 = 0.37;

/// Description of the `ook_demod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "ook_demod",
    summary: "Demodulates on-off keying to soft bits with an adaptive threshold.",
    details: "A chain of five parts: mag (the envelope, so a carrier offset does not matter), \
        a matched filter of round(sps) equal taps 1 / round(sps) (a boxcar over one symbol), \
        a gain 1 / max(hi - lo, 1e-6) from the level trackers, fixed per block of 256 input \
        samples (so the swing between the levels is about 1 and the timing loop runs at its \
        nominal gain at any input level), symbol_sync with the Gardner detector at \
        symbol_rate_hz (loop bandwidth 0.01 of the symbol rate; the detector slope of a \
        boxcar filtered envelope with a swing of 1 is built in) and decision-directed level \
        trackers after rtl_433: the threshold is t = lo + 0.45 (hi - lo), below the middle as \
        the on level spreads more in noise than the off level; a symbol v above t is a \
        1 and moves the on level hi by d = 1 - exp(-1 / (track_symbols 0.5)) of its distance \
        towards v, any other symbol is a 0 and moves the off level lo 16 times slower; the \
        soft bit is (v - t) times the gain, so a value above 0 is a 1 and the output feeds \
        frame_decode directly. The first track_symbols symbols are a lead-in with peak \
        trackers (hi = max(v, hi - d (hi - lo)), lo = min(v, lo + d (hi - lo))) that give the \
        mean trackers their start; after more than track_symbols equal decisions in a row the \
        other level also moves towards v with the slow step, so the trackers recover from a \
        change of the noise floor. The sample rate must be at least 4 times the symbol rate. \
        The output rate follows the clock of the signal, so the ratio kind is variable; the \
        nominal output rate is the symbol rate. Pitfall: put a channel filter (ddc) in front \
        that is not much wider than the signal: the envelope is detected before the boxcar, so \
        the noise of a wide channel raises the off level and narrows the eye. Pitfall: the \
        threshold needs a few symbols of both levels after the lead-in; a run of equal bits \
        longer than track_symbols pulls the other level towards it.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32],
        description: "Complex baseband, channel-filtered, at least 4 samples per symbol.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32],
        description: "Soft bits, one per symbol; above 0 is a 1.",
    }],
    params: &[
        ParamDesc {
            name: "symbol_rate_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 0.001,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "2000 for a 433 MHz sensor",
            mutability: Mutability::Build,
            description: "Nominal symbol rate in Hz, rounded to 1/1000 Hz; the sample rate \
                divided by it must lie between 4 and 1024. Required.",
        },
        ParamDesc {
            name: "track_symbols",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: MAX_TRACK as i64,
                default: Some(32),
            },
            typical: "32; 8 for short bursts",
            mutability: Mutability::Build,
            description: "Time constant of the level trackers in symbols: the on level moves \
                by d = 1 - exp(-2 / track_symbols) of its distance per symbol decided 1, the off \
                level by 1 - exp(-1 / (8 track_symbols)) per symbol decided 0; also the length \
                of the lead-in and the run of equal decisions after which the other level \
                follows.",
        },
    ],
    required: &["symbol_rate_hz"],
};

/// Configuration of [`OokDemodBlock`].
///
/// ```
/// use sdrk_blocks::OokDemodConfig;
/// let cfg = OokDemodConfig { symbol_rate_hz: 2_000.0, ..OokDemodConfig::default() };
/// assert_eq!(cfg.track_symbols, 32);
/// assert!(cfg.validate(sdrk::Rate::hz(250_000)?).is_ok()); // 125 samples per symbol
/// assert!(cfg.validate(sdrk::Rate::hz(6_000)?).is_err()); // 3 are too few
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OokDemodConfig {
    /// Nominal symbol rate in Hz (required by the pipeline; rounded to 1/1000 Hz).
    pub symbol_rate_hz: f64,
    /// Time constant of the level trackers in symbols (1 to 100000).
    pub track_symbols: usize,
}

impl Default for OokDemodConfig {
    /// `symbol_rate_hz` 2000 (a placeholder: the pipeline requires it), `track_symbols` 32.
    fn default() -> Self {
        Self {
            symbol_rate_hz: 2_000.0,
            track_symbols: 32,
        }
    }
}

impl OokDemodConfig {
    /// The configuration of the inner `symbol_sync`: Gardner, loop bandwidth 0.01, damping
    /// 0.7071, `max_deviation` 0.005 and the measured detector slope [`TED_GAIN_OOK`] as
    /// `ted_gain`.
    ///
    /// ```
    /// use sdrk_blocks::{OokDemodConfig, TED_GAIN_OOK};
    /// assert_eq!(OokDemodConfig::default().symbol_sync().ted_gain, TED_GAIN_OOK);
    /// ```
    pub fn symbol_sync(&self) -> SymbolSyncConfig {
        SymbolSyncConfig {
            symbol_rate_hz: self.symbol_rate_hz,
            ted: Ted::Gardner,
            loop_bw: LOOP_BW,
            damping: DAMPING,
            max_deviation: MAX_DEVIATION,
            ted_gain: TED_GAIN_OOK,
        }
    }

    /// Samples per symbol `fs / symbol_rate_hz` (the symbol rate on the 1/1000 Hz grid) after
    /// the checks of [`OokDemodConfig::validate`].
    ///
    /// # Errors
    /// As [`OokDemodConfig::validate`].
    pub fn sps(&self, rate_in: Rate) -> Result<f64> {
        check_count("track_symbols", self.track_symbols, 1, MAX_TRACK)?;
        let sps = self.symbol_sync().sps(rate_in)?;
        if (MIN_SPS..=MAX_SPS).contains(&sps) {
            Ok(sps)
        } else {
            let fs = hz(rate_in);
            Err(Error::OutOfRange {
                name: "symbol_rate_hz",
                min: fs / MAX_SPS,
                max: fs / MIN_SPS,
                got: self.symbol_rate_hz,
            })
        }
    }

    /// Number of taps of the matched filter, `round(sps)`.
    ///
    /// # Errors
    /// As [`OokDemodConfig::validate`].
    pub fn num_taps(&self, rate_in: Rate) -> Result<usize> {
        Ok(libm::round(self.sps(rate_in)?) as usize)
    }

    /// Checks the configuration against the input sample rate.
    ///
    /// # Errors
    /// `OutOfRange` for a `track_symbols` outside 1 to 100000 or a `symbol_rate_hz` that
    /// gives fewer than 4 or more than 1024 samples per symbol.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        self.sps(rate_in).map(|_| ())
    }

    /// The nominal output rate: the symbol rate (on the 1/1000 Hz grid).
    ///
    /// # Errors
    /// As [`OokDemodConfig::validate`].
    pub fn rate_out(&self, rate_in: Rate) -> Result<Rate> {
        self.validate(rate_in)?;
        SymbolSyncBlock::<f32, crate::ArenaStorage<'_>>::rate_out(&self.symbol_sync(), rate_in)
    }

    /// The step `d = 1 - exp(-1 / (track_symbols 0.5))` of the on level tracker (and of the
    /// peak trackers of the lead-in), in `f64`.
    ///
    /// ```
    /// use sdrk_blocks::OokDemodConfig;
    /// let d = OokDemodConfig::default().track_step();
    /// assert!((d - (1.0 - (-1.0f64 / 16.0).exp())).abs() < 1e-15);
    /// ```
    pub fn track_step(&self) -> f64 {
        1.0 - libm::exp(-1.0 / (self.track_symbols as f64 * 0.5))
    }

    /// The step `1 - exp(-1 / (track_symbols 0.5 OOK_LOW_RATIO))` of the off level tracker
    /// (16 times the time constant of the on level, as in `rtl_433`), in `f64`.
    ///
    /// ```
    /// use sdrk_blocks::OokDemodConfig;
    /// let d = OokDemodConfig::default().low_step();
    /// assert!((d - (1.0 - (-1.0f64 / 256.0).exp())).abs() < 1e-15);
    /// ```
    pub fn low_step(&self) -> f64 {
        1.0 - libm::exp(-1.0 / (self.track_symbols as f64 * 0.5 * OOK_LOW_RATIO))
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "symbol_rate_hz" => self.symbol_rate_hz = value.as_f64()?,
            "track_symbols" => self.track_symbols = value.as_usize()?,
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
            "symbol_rate_hz" => Ok(ParamValue::F64(self.symbol_rate_hz)),
            "track_symbols" => Ok(ParamValue::I64(self.track_symbols as i64)),
            _ => unknown(),
        }
    }
}

/// OOK demodulator: `C32` in, `f32` soft bits out.
///
/// Per piece of at most [`OOK_DEMOD_CHUNK`] input samples (pieces never cross a multiple of
/// [`OOK_DEMOD_CHUNK`] samples since construction or reset) the block calls `mag` and
/// `resample_real` (the boxcar with `L = M = 1`) once each, multiplies the boxcar output by the
/// gain of the block, `1 / max(hi - lo, OOK_MIN_SWING)` taken at its first sample (1 before the
/// first symbol), calls `symbol_sync` once and turns the symbols into soft bits with the level
/// trackers (in Rust, in `f32`, on the symbols divided by the gain again). The first
/// `track_symbols` symbols are a lead-in with the peak trackers of M39.1 §4 (the threshold of a
/// symbol uses `hi` and `lo` after it); after it the trackers are decision-directed means after
/// `rtl_433` (the threshold of a symbol, `lo + OOK_THRESHOLD (hi - lo)`, uses the levels
/// before it; D727). The trackers start at 0. A
/// piece can give more symbols than the output has room for; the rest waits in the block and
/// comes out first in the next call. Timing: `RatioKind::Variable` with the nominal ratio
/// `symbol_rate / fs`, delay `(round(sps) + 1) / 2` input samples (filter and interpolator).
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, OokDemodBlock, OokDemodConfig};
/// let cfg = OokDemodConfig { symbol_rate_hz: 1_000.0, ..OokDemodConfig::default() };
/// let rate = Rate::hz(8_000)?;
/// let bytes = OokDemodBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut block = OokDemodBlock::new(&cfg, rate, &mut arena)?;
/// // Alternating on and off symbols of 8 samples each, a carrier at 300 Hz.
/// let input: Vec<C32> = (0..8000)
///     .map(|n| {
///         let a = if (n / 8) % 2 == 0 { 0.8f64 } else { 0.0 };
///         let ph = 2.0 * core::f64::consts::PI * 300.0 * n as f64 / 8000.0;
///         C32::new((a * ph.cos()) as f32, (a * ph.sin()) as f32)
///     })
///     .collect();
/// let mut bits = vec![0.0f32; 1100];
/// let p = block.process(&input, &mut bits)?;
/// assert_eq!(p.consumed, 8000);
/// // After the loop has locked the soft bits alternate in sign.
/// let tail = &bits[p.produced - 100..p.produced];
/// assert!(tail.windows(2).all(|w| w[0] * w[1] < 0.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct OokDemodBlock<S: Storage> {
    config: OokDemodConfig,
    filter: ResampleRealState<f32, S::Buf<f32>, S::Buf<f32>>,
    sync: SymbolSyncBlock<f32, S>,
    /// Nominal samples per symbol.
    sps: f64,
    /// Symbols of a piece at most.
    max_symbols: usize,
    /// The level trackers.
    levels: Levels,
    /// The swing `max(hi - lo, OOK_MIN_SWING)` of the current block (1 before the first
    /// symbol).
    swing: f32,
    /// The gain `1 / swing` of the current block.
    gain: f32,
    /// Input samples of the current block of [`OOK_DEMOD_CHUNK`] samples already processed.
    phase: usize,
    /// Magnitudes; after the filter the symbols of the piece.
    env: S::Buf<f32>,
    /// Filter output.
    filt: S::Buf<f32>,
    /// Soft bits `env[pending.0..pending.1]` that did not fit into the output yet.
    pending: (usize, usize),
}

impl<S: Storage> core::fmt::Debug for OokDemodBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OokDemodBlock")
            .field("config", &self.config)
            .field("levels", &self.levels)
            .field("swing", &self.swing)
            .field("pending", &self.pending)
            .finish_non_exhaustive()
    }
}

/// Most strobes of `symbol_sync` in `m` new input samples (as in `fsk_demod`): at most
/// `floor(m / step) + 1` with the smallest step `0.75 sps (1 - max_deviation)`.
fn max_strobes(m: usize, sps: f64) -> usize {
    let step = 0.75 * sps * (1.0 - MAX_DEVIATION) * (1.0 - 1e-9);
    (libm::floor(m as f64 / step) as usize).saturating_add(1)
}

impl<S: Storage> OokDemodBlock<S> {
    /// Creates the block; the filter taps and history and the two work buffers come from
    /// `storage`.
    ///
    /// # Errors
    /// As [`OokDemodConfig::validate`]; `Storage` if the storage is too small.
    pub fn new(config: &OokDemodConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        let sps = config.sps(rate_in)?;
        let n = libm::round(sps) as usize;
        let mut taps = storage.alloc::<f32>(n)?;
        taps.as_mut().fill((1.0 / n as f64) as f32);
        let hist = storage.alloc::<f32>(n)?;
        let env = storage.alloc::<f32>(OOK_DEMOD_CHUNK)?;
        let filt = storage.alloc::<f32>(OOK_DEMOD_CHUNK)?;
        let sync = SymbolSyncBlock::new(&config.symbol_sync(), rate_in, storage)?;
        Ok(Self {
            config: *config,
            filter: ResampleRealState::new(taps, hist, 1, 1)?,
            sync,
            sps,
            max_symbols: max_strobes(OOK_DEMOD_CHUNK, sps),
            levels: Levels::new(config),
            swing: 1.0,
            gain: 1.0,
            phase: 0,
            env,
            filt,
            pending: (0, 0),
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// # Errors
    /// As [`OokDemodConfig::validate`].
    pub fn storage_bytes(config: &OokDemodConfig, rate_in: Rate) -> Result<usize> {
        let n = config.num_taps(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<f32>(n)?;
        plan.add::<f32>(n)?;
        plan.add::<f32>(OOK_DEMOD_CHUNK)?;
        plan.add::<f32>(OOK_DEMOD_CHUNK)?;
        Ok(plan.bytes())
    }

    /// The nominal output rate (see [`OokDemodConfig::rate_out`]).
    ///
    /// # Errors
    /// As [`OokDemodConfig::validate`].
    pub fn rate_out(config: &OokDemodConfig, rate_in: Rate) -> Result<Rate> {
        config.rate_out(rate_in)
    }

    /// The current estimate of the samples per symbol of the timing loop.
    pub fn sps_estimate(&self) -> f64 {
        self.sync.sps_estimate()
    }

    /// Number of taps of the matched filter.
    pub fn num_taps(&self) -> usize {
        self.filter.taps().len()
    }

    /// The current on and off levels `(hi, lo)` of the trackers, in units of the boxcar
    /// output (the magnitude of the input).
    pub fn levels(&self) -> (f32, f32) {
        (self.levels.hi, self.levels.lo)
    }

    /// The swing the boxcar output of the current block is divided by before `symbol_sync`:
    /// `max(hi - lo, OOK_MIN_SWING)` at the first sample of the block, 1 before the first
    /// symbol.
    pub fn swing(&self) -> f32 {
        self.swing
    }

    /// Writes the pending soft bits that fit into `out`; returns the number written.
    fn flush(&mut self, out: &mut [f32]) -> Result<usize> {
        let (a, b) = self.pending;
        let n = (b - a).min(out.len());
        let src = head(tail(self.env.as_ref(), a)?, n)?;
        head_mut(out, n)?.copy_from_slice(src);
        self.pending = if a + n == b { (0, 0) } else { (a + n, b) };
        Ok(n)
    }

    /// Runs the three kernels and the gain over one piece `x` (at most the rest of the
    /// current block of [`OOK_DEMOD_CHUNK`] samples) and writes the soft bits that fit into
    /// `out`; the rest becomes pending. Returns the soft bits written.
    fn piece(&mut self, x: &[C32], out: &mut [f32]) -> Result<usize>
    where
        Sdrk: Mag<C32> + ResampleReal<f32> + SymbolSync<f32>,
    {
        let m = x.len();
        if self.phase == 0
            && let Some(swing) = self.levels.swing()
        {
            self.swing = swing;
            self.gain = 1.0 / swing;
        }
        self.phase = (self.phase + m) % OOK_DEMOD_CHUNK;
        <Sdrk as Mag<C32>>::mag(&Sdrk, x, head_mut(self.env.as_mut(), m)?)?;
        let p = <Sdrk as ResampleReal<f32>>::resample_real(
            &Sdrk,
            &mut self.filter,
            head(self.env.as_ref(), m)?,
            head_mut(self.filt.as_mut(), m)?,
        )?;
        if p.consumed != m || p.produced != m {
            return Err(piece_error());
        }
        let gain = self.gain;
        for y in head_mut(self.filt.as_mut(), m)? {
            *y *= gain;
        }
        let filt = head(self.filt.as_ref(), m)?;
        let (swing, gain) = (self.swing, self.gain);
        if out.len() > max_strobes(m, self.sps) {
            // Straight into the output: it cannot fill up, so the piece is consumed.
            let p = self.sync.process(filt, out)?;
            if p.consumed != m {
                return Err(piece_error());
            }
            self.levels.track(head_mut(out, p.produced)?, swing, gain);
            return Ok(p.produced);
        }
        let p = self.sync.process(filt, self.env.as_mut())?;
        if p.consumed != m || p.produced > self.max_symbols {
            return Err(piece_error());
        }
        self.levels
            .track(head_mut(self.env.as_mut(), p.produced)?, swing, gain);
        self.pending = (0, p.produced);
        self.flush(out)
    }
}

/// The level trackers of `ook_demod` (D720).
#[derive(Debug, Clone, Copy)]
struct Levels {
    /// The on level.
    hi: f32,
    /// The off level.
    lo: f32,
    /// Step of the on level and of the lead-in peak trackers (rounded once to `f32`).
    d_hi: f32,
    /// Step of the off level (rounded once to `f32`).
    d_lo: f32,
    /// Symbols of the lead-in still to come.
    lead: usize,
    /// `track_symbols`: the lead-in and the longest run of equal decisions that leaves the
    /// other level alone.
    track: usize,
    /// Equal decisions in a row (up to `track + 1`).
    run: usize,
    /// The last decision.
    last: bool,
}

impl Levels {
    fn new(config: &OokDemodConfig) -> Self {
        Self {
            hi: 0.0,
            lo: 0.0,
            d_hi: config.track_step() as f32,
            d_lo: config.low_step() as f32,
            lead: config.track_symbols,
            track: config.track_symbols,
            run: 0,
            last: false,
        }
    }

    /// The state after construction (the steps are kept).
    fn reset(&mut self) {
        self.hi = 0.0;
        self.lo = 0.0;
        self.lead = self.track;
        self.run = 0;
        self.last = false;
    }

    /// The swing `max(hi - lo, OOK_MIN_SWING)` once a symbol has been tracked (`None`
    /// before).
    fn swing(&self) -> Option<f32> {
        (self.lead < self.track).then(|| {
            let s = self.hi - self.lo;
            if s > OOK_MIN_SWING { s } else { OOK_MIN_SWING }
        })
    }

    /// Turns the symbols `v` (normalized: the boxcar output times `gain`) into soft bits in
    /// place: the trackers see `v swing`, the soft bit is `(v swing - t) gain`.
    fn track(&mut self, v: &mut [f32], swing: f32, gain: f32) {
        let flush = sdrk_core::num::flush_denormal;
        for x in v {
            let sym = *x * swing;
            if self.lead > 0 {
                // Lead-in: the peak trackers; the threshold uses the values after the symbol.
                self.lead -= 1;
                let span = self.d_hi * (self.hi - self.lo);
                let up = flush(self.hi - span);
                let down = flush(self.lo + span);
                // `max` and `min` written out: a NaN symbol leaves the trackers unchanged.
                self.hi = if sym > up { sym } else { up };
                self.lo = if sym < down { sym } else { down };
                *x = (sym - self.hi.midpoint(self.lo)) * gain;
                continue;
            }
            let t = self.lo + OOK_THRESHOLD * (self.hi - self.lo);
            *x = (sym - t) * gain;
            // Decision-directed means (`rtl_433` `pulse_detect_package`); a NaN symbol is neither
            // above nor at or below the threshold and leaves the trackers unchanged.
            if sym.is_nan() {
                continue;
            }
            let one = sym > t;
            self.run = if one == self.last {
                (self.run + 1).min(self.track + 1)
            } else {
                1
            };
            self.last = one;
            let long = self.run > self.track;
            if one {
                self.hi = flush(self.hi + self.d_hi * (sym - self.hi));
                if long {
                    self.lo = flush(self.lo + self.d_lo * (sym - self.lo));
                }
            } else {
                self.lo = flush(self.lo + self.d_lo * (sym - self.lo));
                if long {
                    self.hi = flush(self.hi + self.d_lo * (sym - self.hi));
                }
            }
        }
    }
}

fn piece_error() -> Error {
    Error::Core(sdrk_core::Error::Internal("ook_demod piece"))
}

impl<S: Storage> Block for OokDemodBlock<S>
where
    Sdrk: Mag<C32> + ResampleReal<f32> + SymbolSync<f32>,
{
    type In = C32;
    type Out = f32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[C32], output: &mut [f32]) -> Result<Progress> {
        let mut produced = self.flush(output)?;
        let mut consumed = 0usize;
        while self.pending.0 == self.pending.1 && consumed < input.len() && produced < output.len()
        {
            let m = (input.len() - consumed).min(OOK_DEMOD_CHUNK - self.phase);
            let x = head(tail(input, consumed)?, m)?;
            produced += self.piece(x, tail_mut(output, produced)?)?;
            consumed += m;
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        let sync = self.sync.timing();
        let taps = self.filter.taps().len() as u64;
        BlockTiming {
            ratio: sync.ratio,
            kind: RatioKind::Variable,
            // (taps - 1) / 2 (boxcar) + 1 (interpolator).
            delay: Delay::new(taps + 1, 2).unwrap_or(Delay::ZERO),
        }
    }

    fn reset(&mut self) {
        self.filter.reset();
        self.sync.reset();
        self.levels.reset();
        self.swing = 1.0;
        self.gain = 1.0;
        self.phase = 0;
        self.pending = (0, 0);
    }

    fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
        match DESCRIPTION.param(name) {
            Some(_) => Err(Error::ReadOnlyParameter),
            None => unknown(),
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        self.sync.min_input()
    }
}
