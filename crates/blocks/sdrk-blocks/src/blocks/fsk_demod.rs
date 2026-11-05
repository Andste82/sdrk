//! `fsk_demod`: non-coherent 2-FSK and 4-FSK receiver (discriminator, boxcar matched filter,
//! Gardner symbol timing, slicer), built from the kernels `quad_demod`, `resample_real` and
//! `symbol_sync`.

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::decim::{ResampleReal, ResampleRealState};
use sdrk_core::kernels::demod::{QuadDemod, QuadDemodState};
use sdrk_core::kernels::symbol_sync::SymbolSync;

use super::fsk_mod::bits_per_symbol;
use super::slice::fsk4_bits;
use super::symbol_sync::SymbolSyncBlock;
use super::{HZ_LIMIT, head, head_mut, hz, multiply_rate, tail, tail_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};
use crate::{BlockSample, SymbolSyncConfig, Ted};

/// Input samples per piece the block processes at a time (the size of its two work buffers).
pub const FSK_DEMOD_CHUNK: usize = 256;

/// Fewest samples per symbol (`rate_unsupported` below).
const MIN_SPS: f64 = 4.0;

/// Most samples per symbol (bounds the matched filter to 1024 taps, D573).
const MAX_SPS: f64 = 1024.0;

/// Damping of the timing loop (the `symbol_sync` default).
#[allow(clippy::approx_constant)]
const DAMPING: f64 = 0.7071;

/// Largest deviation of the symbol period (the `symbol_sync` default).
const MAX_DEVIATION: f64 = 0.005;

/// Gardner slope per symbol of the boxcar-filtered discriminator output of 2-FSK (levels ±1,
/// rectangular frequency pulses), measured (D571).
pub const TED_GAIN_2FSK: f64 = 1.48;

/// Gardner slope per symbol of the boxcar-filtered discriminator output of 4-FSK (levels ±1,
/// ±1/3), measured (D571).
pub const TED_GAIN_4FSK: f64 = 0.84;

/// Description of the `fsk_demod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "fsk_demod",
    summary: "Demodulates 2-FSK or 4-FSK (also GFSK and GMSK) to soft bits without a carrier \
        lock.",
    details: "A chain of four parts: quad_demod with deviation_hz (the outer frequencies read \
        +1 and -1), a matched filter of round(sps) equal taps 1 / round(sps) (a boxcar over one \
        symbol; for GFSK the same boxcar, the Gaussian is not inverted), symbol_sync with the \
        Gardner detector at symbol_rate_hz with loop_bw (the detector slope of boxcar \
        filtered rectangular FSK is built in, so for plain FSK loop_bw keeps its meaning) \
        and a slicer: 2 levels \
        give one soft bit per symbol, the level itself (slice real); 4 levels give two, -v \
        and |v| - 2/3 (slice fsk4). A soft bit above 0 means bit 1, so the output feeds \
        frame_decode directly. Place it after a ddc that filters the channel (and an afc \
        when the carrier may be off); the input level does not matter. The sample rate must \
        be at least 4 times the symbol rate. The output rate follows the clock of the \
        signal, so the ratio kind is variable; the nominal output rate is the symbol rate \
        times the bits per symbol. Pitfall: a frequency offset of the carrier shifts every \
        level; keep it well below the deviation (afc). Pitfall: the built-in detector slope is \
        that of rectangular FSK; with a Gaussian pulse the slope is lower (about 0.78 at bt \
        0.5, 0.21 at bt 0.3), so the loop is that much narrower than loop_bw: raise loop_bw \
        by about 1.9 for bt 0.5.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32],
        description: "Complex baseband, channel-filtered, at least 4 samples per symbol.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32],
        description: "Soft bits, bits per symbol samples per symbol; above 0 is a 1.",
    }],
    params: &[
        ParamDesc {
            name: "levels",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 2,
                max: 4,
                default: Some(2),
            },
            typical: "2; 4 for M17 and DMR",
            mutability: Mutability::Build,
            description: "Number of frequencies: 2 (one bit per symbol) or 4 (two bits).",
        },
        ParamDesc {
            name: "symbol_rate_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 0.001,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "50000 for a GFSK packet link, 4800 for M17",
            mutability: Mutability::Build,
            description: "Nominal symbol rate in Hz, rounded to 1/1000 Hz; the sample rate \
                divided by it must lie between 4 and 1024. Required.",
        },
        ParamDesc {
            name: "deviation_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "the deviation of the transmitter (its outer frequencies)",
            mutability: Mutability::Build,
            description: "Frequency of the outer levels in Hz; at most half the sample rate. \
                Required.",
        },
        ParamDesc {
            name: "loop_bw",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0001,
                max: 0.1,
                default: Some(0.01),
            },
            typical: "0.01; 0.005 for weak signals, 0.02 for fast acquisition",
            mutability: Mutability::Build,
            description: "Bandwidth of the timing loop normalized to the symbol rate.",
        },
    ],
    required: &["symbol_rate_hz", "deviation_hz"],
};

/// Configuration of [`FskDemodBlock`].
///
/// ```
/// use sdrk_blocks::FskDemodConfig;
/// let cfg = FskDemodConfig { symbol_rate_hz: 50_000.0, deviation_hz: 25_000.0, ..FskDemodConfig::default() };
/// assert!(cfg.validate(sdrk::Rate::hz(200_000)?).is_ok()); // 4 samples per symbol
/// assert!(cfg.validate(sdrk::Rate::hz(150_000)?).is_err()); // 3 are too few
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FskDemodConfig {
    /// Number of frequencies, 2 or 4.
    pub levels: usize,
    /// Nominal symbol rate in Hz (required by the pipeline; rounded to 1/1000 Hz).
    pub symbol_rate_hz: f64,
    /// Frequency of the outer levels in Hz (required by the pipeline).
    pub deviation_hz: f64,
    /// Bandwidth of the timing loop normalized to the symbol rate (0.0001 to 0.1).
    pub loop_bw: f64,
}

impl Default for FskDemodConfig {
    /// `levels` 2, `symbol_rate_hz` 1200 and `deviation_hz` 600 (placeholders: the pipeline
    /// requires both), `loop_bw` 0.01.
    fn default() -> Self {
        Self {
            levels: 2,
            symbol_rate_hz: 1200.0,
            deviation_hz: 600.0,
            loop_bw: 0.01,
        }
    }
}

impl FskDemodConfig {
    /// The configuration of the inner `symbol_sync`: Gardner, `loop_bw`, damping 0.7071,
    /// `max_deviation` 0.005 and the measured detector slope ([`TED_GAIN_2FSK`] or
    /// [`TED_GAIN_4FSK`]) as `ted_gain`.
    ///
    /// ```
    /// use sdrk_blocks::{FskDemodConfig, TED_GAIN_4FSK};
    /// let cfg = FskDemodConfig { levels: 4, ..FskDemodConfig::default() };
    /// assert_eq!(cfg.symbol_sync()?.ted_gain, TED_GAIN_4FSK);
    /// # Ok::<(), sdrk_blocks::Error>(())
    /// ```
    ///
    /// # Errors
    /// `InvalidParameter` for `levels` other than 2 and 4.
    pub fn symbol_sync(&self) -> Result<SymbolSyncConfig> {
        let ted_gain = if bits_per_symbol(self.levels)? == 1 {
            TED_GAIN_2FSK
        } else {
            TED_GAIN_4FSK
        };
        Ok(SymbolSyncConfig {
            symbol_rate_hz: self.symbol_rate_hz,
            ted: Ted::Gardner,
            loop_bw: self.loop_bw,
            damping: DAMPING,
            max_deviation: MAX_DEVIATION,
            ted_gain,
        })
    }

    /// Samples per symbol `fs / symbol_rate_hz` (the symbol rate on the 1/1000 Hz grid) after
    /// the checks of [`FskDemodConfig::validate`].
    ///
    /// # Errors
    /// As [`FskDemodConfig::validate`].
    pub fn sps(&self, rate_in: Rate) -> Result<f64> {
        let sync = self.symbol_sync()?;
        let half = hz(rate_in) / 2.0;
        if !(self.deviation_hz > 0.0 && self.deviation_hz <= half) {
            return Err(Error::OutOfRange {
                name: "deviation_hz",
                min: 0.0,
                max: half,
                got: self.deviation_hz,
            });
        }
        let sps = sync.sps(rate_in)?;
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
    /// ```
    /// use sdrk_blocks::FskDemodConfig;
    /// let cfg = FskDemodConfig { symbol_rate_hz: 4_800.0, deviation_hz: 2_400.0, ..FskDemodConfig::default() };
    /// assert_eq!(cfg.num_taps(sdrk::Rate::hz(48_000)?)?, 10);
    /// # Ok::<(), sdrk_blocks::Error>(())
    /// ```
    ///
    /// # Errors
    /// As [`FskDemodConfig::validate`].
    pub fn num_taps(&self, rate_in: Rate) -> Result<usize> {
        Ok(libm::round(self.sps(rate_in)?) as usize)
    }

    /// Checks the configuration against the input sample rate.
    ///
    /// # Errors
    /// `InvalidParameter` for `levels` other than 2 and 4; `OutOfRange` for a `deviation_hz`
    /// above half the sample rate, a `loop_bw` outside 0.0001 to 0.1 or a `symbol_rate_hz`
    /// that gives fewer than 4 or more than 1024 samples per symbol.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        self.sps(rate_in).map(|_| ())
    }

    /// The nominal output rate: the symbol rate (on the 1/1000 Hz grid) times the bits per
    /// symbol.
    ///
    /// # Errors
    /// As [`FskDemodConfig::validate`].
    pub fn rate_out(&self, rate_in: Rate) -> Result<Rate> {
        self.validate(rate_in)?;
        let sym = SymbolSyncBlock::<f32, crate::ArenaStorage<'_>>::rate_out(
            &self.symbol_sync()?,
            rate_in,
        )?;
        multiply_rate(sym, bits_per_symbol(self.levels)?)
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "levels" => self.levels = value.as_usize()?,
            "symbol_rate_hz" => self.symbol_rate_hz = value.as_f64()?,
            "deviation_hz" => self.deviation_hz = value.as_f64()?,
            "loop_bw" => self.loop_bw = value.as_f64()?,
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
            "levels" => Ok(ParamValue::I64(self.levels as i64)),
            "symbol_rate_hz" => Ok(ParamValue::F64(self.symbol_rate_hz)),
            "deviation_hz" => Ok(ParamValue::F64(self.deviation_hz)),
            "loop_bw" => Ok(ParamValue::F64(self.loop_bw)),
            _ => unknown(),
        }
    }
}

/// FSK demodulator: `C32` in, `f32` soft bits out (no Q15 variant, D529).
///
/// Per piece of at most [`FSK_DEMOD_CHUNK`] input samples the block calls `quad_demod`,
/// `resample_real` (the boxcar with `L = M = 1`) and `symbol_sync` once each, then maps the
/// symbols to soft bits (2 levels: the level; 4 levels: the `fsk4` map of `slice`). A piece
/// can give more symbols than the output has room for; the rest waits in the block and comes
/// out first in the next call (at most one piece of symbols). Timing: `RatioKind::Variable`
/// with the nominal ratio `bits_per_symbol symbol_rate / fs`, delay `round(sps) / 2 + 1`
/// input samples (discriminator, filter, interpolator).
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, FskDemodBlock, FskDemodConfig};
/// let cfg = FskDemodConfig { symbol_rate_hz: 1000.0, deviation_hz: 500.0, ..FskDemodConfig::default() };
/// let rate = Rate::hz(8000)?;
/// let bytes = FskDemodBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut block = FskDemodBlock::new(&cfg, rate, &mut arena)?;
/// // Alternating symbols +500 Hz, -500 Hz with 8 samples each (continuous phase).
/// let mut phase = 0.0f64;
/// let input: Vec<C32> = (0..8000)
///     .map(|n| {
///         let f = if (n / 8) % 2 == 0 { 500.0 } else { -500.0 };
///         phase += 2.0 * core::f64::consts::PI * f / 8000.0;
///         C32::new(phase.cos() as f32, phase.sin() as f32)
///     })
///     .collect();
/// let mut bits = vec![0.0f32; 1100];
/// let p = block.process(&input, &mut bits)?;
/// assert_eq!(p.consumed, 8000);
/// // After the loop has locked the soft bits alternate between about +1 and -1.
/// let tail = &bits[p.produced - 100..p.produced];
/// assert!(tail.windows(2).all(|w| w[0] * w[1] < -0.8));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct FskDemodBlock<S: Storage> {
    config: FskDemodConfig,
    bps: usize,
    gain: f32,
    quad: QuadDemodState<C32>,
    filter: ResampleRealState<f32, S::Buf<f32>, S::Buf<f32>>,
    sync: SymbolSyncBlock<f32, S>,
    /// Nominal samples per symbol.
    sps: f64,
    /// Symbols of a piece at most: `floor(FSK_DEMOD_CHUNK / min_step) + 1`.
    max_symbols: usize,
    /// Discriminator output; after the filter the symbols of the piece.
    disc: S::Buf<f32>,
    /// Filter output.
    filt: S::Buf<f32>,
    /// Symbols `disc[pending.0..pending.1]` that did not fit into the output yet.
    pending: (usize, usize),
}

impl<S: Storage> core::fmt::Debug for FskDemodBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FskDemodBlock")
            .field("config", &self.config)
            .field("pending", &self.pending)
            .finish_non_exhaustive()
    }
}

/// Most strobes of `symbol_sync` in `m` new input samples: the strobes lie less than `m`
/// samples apart in total and at least `0.75 sps (1 - max_deviation)` apart each (the step of
/// the loop is clamped to a quarter of the period, D534), so at most `floor(m / step) + 1`.
fn max_strobes(m: usize, sps: f64) -> usize {
    // A margin of 1e-9 covers the rounding of the f64 positions.
    let step = 0.75 * sps * (1.0 - MAX_DEVIATION) * (1.0 - 1e-9);
    (libm::floor(m as f64 / step) as usize).saturating_add(1)
}

impl<S: Storage> FskDemodBlock<S> {
    /// Creates the block; the filter taps and history and the two work buffers come from
    /// `storage`.
    ///
    /// # Errors
    /// As [`FskDemodConfig::validate`]; `Storage` if the storage is too small.
    pub fn new(config: &FskDemodConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        let sps = config.sps(rate_in)?;
        let bps = bits_per_symbol(config.levels)?;
        let n = libm::round(sps) as usize;
        let mut taps = storage.alloc::<f32>(n)?;
        let tap = 1.0 / n as f64;
        for t in taps.as_mut() {
            *t = tap as f32;
        }
        let hist = storage.alloc::<f32>(n)?;
        let disc = storage.alloc::<f32>(FSK_DEMOD_CHUNK)?;
        let filt = storage.alloc::<f32>(FSK_DEMOD_CHUNK)?;
        let sync = SymbolSyncBlock::new(&config.symbol_sync()?, rate_in, storage)?;
        Ok(Self {
            config: *config,
            bps,
            gain: C32::demod_gain(config.deviation_hz, hz(rate_in)),
            quad: QuadDemodState::new(),
            filter: ResampleRealState::new(taps, hist, 1, 1)?,
            sync,
            sps,
            max_symbols: max_strobes(FSK_DEMOD_CHUNK, sps),
            disc,
            filt,
            pending: (0, 0),
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// # Errors
    /// As [`FskDemodConfig::validate`].
    pub fn storage_bytes(config: &FskDemodConfig, rate_in: Rate) -> Result<usize> {
        let n = config.num_taps(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<f32>(n)?;
        plan.add::<f32>(n)?;
        plan.add::<f32>(FSK_DEMOD_CHUNK)?;
        plan.add::<f32>(FSK_DEMOD_CHUNK)?;
        Ok(plan.bytes())
    }

    /// The nominal output rate (see [`FskDemodConfig::rate_out`]).
    ///
    /// # Errors
    /// As [`FskDemodConfig::validate`].
    pub fn rate_out(config: &FskDemodConfig, rate_in: Rate) -> Result<Rate> {
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

    /// Writes the pending symbols that fit into `out` as soft bits; returns the soft bits
    /// written.
    fn flush(&mut self, out: &mut [f32]) -> Result<usize> {
        let (a, b) = self.pending;
        let n = (b - a).min(out.len() / self.bps);
        let symbols = head(tail(self.disc.as_ref(), a)?, n)?;
        write_bits(self.bps, symbols, out);
        self.pending = (a + n, b);
        if a + n == b {
            self.pending = (0, 0);
        }
        Ok(n * self.bps)
    }

    /// Runs the three kernels over one piece `x` (at most [`FSK_DEMOD_CHUNK`] samples) and
    /// writes the soft bits that fit into `out`; the rest becomes pending. Returns the soft
    /// bits written.
    fn piece(&mut self, x: &[C32], out: &mut [f32]) -> Result<usize>
    where
        Sdrk: QuadDemod<C32> + ResampleReal<f32> + SymbolSync<f32>,
    {
        let m = x.len();
        let disc = head_mut(self.disc.as_mut(), m)?;
        <Sdrk as QuadDemod<C32>>::quad_demod(&Sdrk, &mut self.quad, x, disc, self.gain)?;
        let filt = head_mut(self.filt.as_mut(), m)?;
        let p = <Sdrk as ResampleReal<f32>>::resample_real(
            &Sdrk,
            &mut self.filter,
            head(self.disc.as_ref(), m)?,
            filt,
        )?;
        if p.consumed != m || p.produced != m {
            return Err(piece_error());
        }
        let filt = head(self.filt.as_ref(), m)?;
        if self.bps == 1 && out.len() > max_strobes(m, self.sps) {
            // Straight into the output: it cannot fill up, so the piece is consumed.
            let p = self.sync.process(filt, out)?;
            if p.consumed != m {
                return Err(piece_error());
            }
            return Ok(p.produced);
        }
        let p = self.sync.process(filt, self.disc.as_mut())?;
        if p.consumed != m || p.produced > self.max_symbols {
            return Err(piece_error());
        }
        self.pending = (0, p.produced);
        self.flush(out)
    }
}

/// The soft bits of `symbols` into `out` (which has room for them): the level for 2-FSK, the
/// `fsk4` pair for 4-FSK.
fn write_bits(bps: usize, symbols: &[f32], out: &mut [f32]) {
    if bps == 1 {
        for (o, &v) in out.iter_mut().zip(symbols) {
            *o = v;
        }
    } else {
        for (o, &v) in out.chunks_exact_mut(2).zip(symbols) {
            if let [b0, b1] = o {
                (*b0, *b1) = fsk4_bits(1.0, v);
            }
        }
    }
}

fn piece_error() -> Error {
    Error::Core(sdrk_core::Error::Internal("fsk_demod piece"))
}

impl<S: Storage> Block for FskDemodBlock<S>
where
    Sdrk: QuadDemod<C32> + ResampleReal<f32> + SymbolSync<f32>,
{
    type In = C32;
    type Out = f32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[C32], output: &mut [f32]) -> Result<Progress> {
        let bps = self.bps;
        let mut produced = self.flush(output)?;
        let mut consumed = 0usize;
        while self.pending.0 == self.pending.1
            && consumed < input.len()
            && output.len() - produced >= bps
        {
            let m = (input.len() - consumed).min(FSK_DEMOD_CHUNK);
            let x = head(tail(input, consumed)?, m)?;
            produced += self.piece(x, tail_mut(output, produced)?)?;
            consumed += m;
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        let sync = self.sync.timing();
        let bps = self.bps as u64;
        let taps = self.filter.taps().len() as u64;
        BlockTiming {
            ratio: sync
                .ratio
                .mul(Ratio::new(bps, 1).unwrap_or(Ratio::ONE))
                .unwrap_or(sync.ratio),
            kind: RatioKind::Variable,
            // 1/2 (discriminator) + (taps - 1) / 2 (boxcar) + 1 (interpolator).
            delay: Delay::new(taps + 2, 2).unwrap_or(Delay::ZERO),
        }
    }

    fn reset(&mut self) {
        self.quad.reset();
        self.filter.reset();
        self.sync.reset();
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
