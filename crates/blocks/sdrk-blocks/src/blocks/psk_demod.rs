//! `psk_demod`: BPSK and QPSK demodulator (matched filter, symbol timing, carrier recovery or
//! differential detection, soft bits).

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::costas::{Costas, CostasState};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::symbol_sync::{SymbolSync, SymbolSyncState};

use super::costas::{
    DEFAULT_DAMPING, DEFAULT_LOOP_BW, DEFAULT_MAX_OFFSET, MAX_LOOP_BW, MAX_OFFSET_LIMIT,
    MIN_LOOP_BW, check_f64, check_order,
};
use super::slice::soft_bits;
use super::symbol_sync::SymbolSyncBlock;
use super::{HZ_LIMIT, check_count, head, head_mut, tail, tail_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::{RRC_MAX_SPAN, RRC_MAX_SPS, RRC_MIN_ROLLOFF, RRC_MIN_SPAN, Rrc};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};
use crate::{SymbolSyncConfig, Ted};

/// Input samples per piece the block processes at a time (the size of its work buffers).
pub const PSK_DEMOD_CHUNK: usize = 256;

/// Smallest `symbol_rate_hz`.
const MIN_SYMBOL_RATE_HZ: f64 = 0.001;
/// The largest relative distance of `fs / symbol_rate_hz` from an integer.
const SPS_TOLERANCE: f64 = 0.01;
/// The largest deviation of the symbol clock the timing loop follows (the default of
/// `symbol_sync`).
const SYNC_MAX_DEVIATION: f64 = 0.005;

/// The reason of the rate error (a hint to resample).
const RATE_REASON: &str = "the sample rate must be within 1 % of an integer multiple (2 to \
    1024) of the symbol rate and at least twice it; put a resample before psk_demod";

/// Description of the `psk_demod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "psk_demod",
    summary: "Demodulates BPSK or QPSK to soft bits: matched filter, symbol timing and carrier \
        recovery.",
    details: "The input is the complex baseband at roughly the nominal level (an agc before \
        it) with fs / symbol_rate_hz within 1 % of an integer and at least 2 (otherwise the \
        rate is unsupported: resample first). A matched root-raised-cosine filter (rolloff, \
        span symbols, normalized so that the pulse of psk_mod gives amplitude 1 at the symbol \
        centres) is followed by symbol_sync (Gardner, loop bandwidth sync_bw, damping 0.7071, \
        max_deviation 0.005, ted_gain 1) and, for coherent detection, the Costas loop costas \
        (carrier_bw, damping 0.7071, max_offset cycles per symbol); with differential set the \
        loop is replaced by z_k = y_k conj(y_k-1) (y_-1 = 1). The symbols become soft bits as \
        slice does (BPSK re; QPSK re then im): above 0 is a 1. The phase ambiguity of \
        coherent detection (180 degrees for BPSK, 90 degrees for QPSK) is not resolved: use \
        frame_decode with polarity both or differential coding. The output rate is the \
        symbol rate times the bits per symbol and follows the clock of the signal (variable \
        ratio).",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32],
        description: "Complex baseband, an integer number (2 or more) of samples per symbol, \
            amplitude about 1.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32],
        description: "Soft bits, bits per symbol samples per symbol; above 0 is a 1.",
    }],
    params: &[
        ParamDesc {
            name: "order",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 2,
                max: 4,
                default: Some(2),
            },
            typical: "2 for BPSK, 4 for QPSK",
            mutability: Mutability::Build,
            description: "Order of the phase-shift keying: 2 (BPSK) or 4 (QPSK).",
        },
        ParamDesc {
            name: "symbol_rate_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: MIN_SYMBOL_RATE_HZ,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "25000 for 50 kbit/s QPSK",
            mutability: Mutability::Build,
            description: "Symbol rate in Hz; the sample rate must be within 1 % of an integer \
                multiple (2 to 1024) of it and at least twice it. Required.",
        },
        ParamDesc {
            name: "rolloff",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: RRC_MIN_ROLLOFF,
                max: 1.0,
                default: Some(0.35),
            },
            typical: "0.35 (the roll-off of the transmitter)",
            mutability: Mutability::Build,
            description: "Roll-off of the matched root-raised-cosine filter.",
        },
        ParamDesc {
            name: "span",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: RRC_MIN_SPAN as i64,
                max: RRC_MAX_SPAN as i64,
                default: Some(10),
            },
            typical: "10",
            mutability: Mutability::Build,
            description: "Length of the matched filter in symbols.",
        },
        ParamDesc {
            name: "differential",
            unit: Unit::None,
            kind: ParamKind::Bool { default: false },
            typical: "false; true for differentially coded signals (no carrier loop)",
            mutability: Mutability::Build,
            description: "Differential detection instead of the carrier loop.",
        },
        ParamDesc {
            name: "sync_bw",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0001,
                max: 0.1,
                default: Some(0.01),
            },
            typical: "0.01",
            mutability: Mutability::Build,
            description: "Loop bandwidth of the symbol timing, normalized to the symbol rate.",
        },
        ParamDesc {
            name: "carrier_bw",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: MIN_LOOP_BW,
                max: MAX_LOOP_BW,
                default: Some(DEFAULT_LOOP_BW),
            },
            typical: "0.02; wider for large frequency offsets",
            mutability: Mutability::Build,
            description: "Loop bandwidth of the carrier loop, normalized to the symbol rate.",
        },
        ParamDesc {
            name: "max_offset",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: MAX_OFFSET_LIMIT,
                default: Some(DEFAULT_MAX_OFFSET),
            },
            typical: "0.05; above the largest carrier offset in cycles per symbol",
            mutability: Mutability::Build,
            description: "Limit of the carrier loop's frequency integrator in cycles per \
                symbol.",
        },
    ],
    required: &["symbol_rate_hz"],
};

/// Configuration of [`PskDemodBlock`].
///
/// ```
/// use sdrk_blocks::PskDemodConfig;
/// let cfg = PskDemodConfig { order: 4, symbol_rate_hz: 25_000.0, ..PskDemodConfig::default() };
/// assert_eq!(cfg.samples_per_symbol(sdrk::Rate::hz(200_000)?)?, 8);
/// // 7.5 samples per symbol is not within 1 % of an integer
/// assert!(cfg.validate(sdrk::Rate::hz(187_500)?).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PskDemodConfig {
    /// Order: 2 (BPSK) or 4 (QPSK).
    pub order: u32,
    /// Symbol rate in Hz (required).
    pub symbol_rate_hz: f64,
    /// Roll-off of the matched filter (0.05 to 1).
    pub rolloff: f64,
    /// Length of the matched filter in symbols (4 to 32).
    pub span: usize,
    /// Differential detection instead of the carrier loop.
    pub differential: bool,
    /// Loop bandwidth of the symbol timing (0.0001 to 0.1).
    pub sync_bw: f64,
    /// Loop bandwidth of the carrier loop (0.0001 to 0.2).
    pub carrier_bw: f64,
    /// Limit of the frequency integrator in cycles per symbol (0 to 0.25).
    pub max_offset: f64,
}

impl Default for PskDemodConfig {
    /// Order 2, `symbol_rate_hz` 1200 (a placeholder: the pipeline requires the parameter),
    /// `rolloff` 0.35, `span` 10, coherent, `sync_bw` 0.01, `carrier_bw` 0.02, `max_offset`
    /// 0.05.
    fn default() -> Self {
        Self {
            order: 2,
            symbol_rate_hz: 1200.0,
            rolloff: 0.35,
            span: 10,
            differential: false,
            sync_bw: 0.01,
            carrier_bw: DEFAULT_LOOP_BW,
            max_offset: DEFAULT_MAX_OFFSET,
        }
    }
}

impl PskDemodConfig {
    /// Bits per symbol: 1 for order 2, 2 for order 4.
    pub fn bits_per_symbol(&self) -> usize {
        if self.order == 4 { 2 } else { 1 }
    }

    /// The configuration of the timing loop.
    fn sync(&self) -> SymbolSyncConfig {
        SymbolSyncConfig {
            symbol_rate_hz: self.symbol_rate_hz,
            ted: Ted::Gardner,
            loop_bw: self.sync_bw,
            damping: DEFAULT_DAMPING,
            max_deviation: SYNC_MAX_DEVIATION,
            ted_gain: 1.0,
        }
    }

    /// The ranges, and the integer samples per symbol of the matched filter.
    fn derive(&self, rate_in: Rate) -> Result<usize> {
        check_order(self.order)?;
        check_f64(
            "symbol_rate_hz",
            self.symbol_rate_hz,
            MIN_SYMBOL_RATE_HZ,
            HZ_LIMIT,
        )?;
        check_f64("rolloff", self.rolloff, RRC_MIN_ROLLOFF, 1.0)?;
        check_count("span", self.span, RRC_MIN_SPAN, RRC_MAX_SPAN)?;
        check_f64("sync_bw", self.sync_bw, 0.0001, 0.1)?;
        check_f64("carrier_bw", self.carrier_bw, MIN_LOOP_BW, MAX_LOOP_BW)?;
        check_f64("max_offset", self.max_offset, 0.0, MAX_OFFSET_LIMIT)?;
        let sps = rate_in.as_f64() / self.symbol_rate_hz;
        let n = libm::floor(sps + 0.5);
        let near = libm::fabs(sps - n) <= SPS_TOLERANCE * n;
        if !(near && n >= 2.0 && n <= RRC_MAX_SPS as f64) {
            return Err(Error::InvalidParameter {
                name: "symbol_rate_hz",
                reason: RATE_REASON,
            });
        }
        // The exact checks of the timing loop (the 1/1000 Hz grid of the symbol rate). Its
        // range error of the symbol rate (exact samples per symbol below 2, as 1.99 with n = 2)
        // is the same rate rule: reported with the hint to resample.
        match self.sync().validate(rate_in) {
            Err(Error::OutOfRange {
                name: "symbol_rate_hz",
                ..
            }) => Err(Error::InvalidParameter {
                name: "symbol_rate_hz",
                reason: RATE_REASON,
            }),
            other => other,
        }?;
        Ok(n as usize)
    }

    /// The integer samples per symbol of the matched filter: `fs / symbol_rate_hz` rounded.
    ///
    /// # Errors
    /// As [`PskDemodConfig::validate`].
    pub fn samples_per_symbol(&self, rate_in: Rate) -> Result<usize> {
        self.derive(rate_in)
    }

    /// Checks the configuration against the input sample rate.
    ///
    /// # Errors
    /// `OutOfRange` for a value outside its range; `InvalidParameter` for the order 3 and for
    /// a symbol rate whose samples per symbol are not within 1 % of an integer from 2 to 1024
    /// or are below 2 (the pipeline reports it as `rate_unsupported` with the hint to
    /// resample).
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        self.derive(rate_in).map(|_| ())
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`; `InvalidParameter` for the order 3.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "order" => {
                let o = u32::try_from(value.as_i64()?).unwrap_or(0);
                check_order(o)?;
                self.order = o;
            }
            "symbol_rate_hz" => self.symbol_rate_hz = value.as_f64()?,
            "rolloff" => self.rolloff = value.as_f64()?,
            "span" => self.span = value.as_usize()?,
            "differential" => self.differential = value.as_bool()?,
            "sync_bw" => self.sync_bw = value.as_f64()?,
            "carrier_bw" => self.carrier_bw = value.as_f64()?,
            "max_offset" => self.max_offset = value.as_f64()?,
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
            "order" => Ok(ParamValue::I64(i64::from(self.order))),
            "symbol_rate_hz" => Ok(ParamValue::F64(self.symbol_rate_hz)),
            "rolloff" => Ok(ParamValue::F64(self.rolloff)),
            "span" => Ok(ParamValue::I64(self.span as i64)),
            "differential" => Ok(ParamValue::Bool(self.differential)),
            "sync_bw" => Ok(ParamValue::F64(self.sync_bw)),
            "carrier_bw" => Ok(ParamValue::F64(self.carrier_bw)),
            "max_offset" => Ok(ParamValue::F64(self.max_offset)),
            _ => unknown(),
        }
    }
}

/// BPSK and QPSK demodulator: `C32` in, `f32` soft bits out (`bits_per_symbol` per symbol,
/// above 0 is a 1).
///
/// Per piece of at most [`PSK_DEMOD_CHUNK`] input samples: the kernel `fir` (matched filter,
/// taps `rrc(rolloff, n, span) / sqrt(n)` as `f32` with `n` the rounded samples per symbol),
/// the kernel `symbol_sync` (Gardner at the exact `fs / symbol_rate_hz`), then the kernel
/// `costas` (coherent) or the differential product `z_k = y_k conj(y_(k-1))` with
/// `y_(-1) = 1` (`f32`: `(y.re p.re + y.im p.im, y.im p.re - y.re p.im)`), and the soft bits
/// of `slice` (scale 1). Soft bits that do not fit into the output wait in a buffer and are
/// written first by the next call (which then consumes no input). Timing: variable ratio
/// `symbol_rate bits / fs`, delay `span n / 2 + 1` input samples. Storage: the taps, the
/// filter history and three work buffers (filter output, symbols, waiting soft bits).
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{
///     ArenaStorage, Block, PskDemodBlock, PskDemodConfig, PskModBlock, PskModConfig,
/// };
/// let rate = Rate::hz(4000)?; // 1000 symbols per second, 4 samples per symbol
/// let tx = PskModConfig { order: 4, samples_per_symbol: 4, ..PskModConfig::default() };
/// let rx = PskDemodConfig { order: 4, symbol_rate_hz: 1000.0, differential: false, ..PskDemodConfig::default() };
/// let mut mem = vec![0u8; 1 << 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut m = PskModBlock::new(&tx, Rate::hz(2000)?, &mut arena)?;
/// let mut d = PskDemodBlock::new(&rx, rate, &mut arena)?;
/// // Alternating bit pairs; the loop settles on the diagonals.
/// let bits: Vec<f32> = (0..8000).map(|k| if (k / 2) % 3 == 0 { 1.0 } else { -1.0 }).collect();
/// let mut sig = vec![C32::default(); 16_000];
/// let p = m.process(&bits, &mut sig)?;
/// assert_eq!(p.produced, 16_000);
/// let mut soft = vec![0.0f32; 8100];
/// let q = d.process(&sig, &mut soft)?;
/// assert_eq!(q.consumed, 16_000);
/// assert!((7980..=8000).contains(&q.produced));
/// assert!(soft[q.produced - 100..q.produced].iter().all(|s| (s.abs() - 0.7071).abs() < 0.05));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct PskDemodBlock<S: Storage> {
    config: PskDemodConfig,
    bps: usize,
    sps: usize,
    ratio: Ratio,
    min_input: usize,
    fir: FirState<C32, S::Buf<f32>, S::Buf<C32>>,
    sync: SymbolSyncState<C32>,
    costas: CostasState,
    /// The previous symbol of the differential detector.
    prev: C32,
    /// Matched-filter output, then the symbols after the carrier loop.
    a: S::Buf<C32>,
    /// The symbols of `symbol_sync`.
    b: S::Buf<C32>,
    /// Soft bits that did not fit into the output, `pending[at..len]`.
    pending: S::Buf<f32>,
    at: usize,
    len: usize,
}

impl<S: Storage> core::fmt::Debug for PskDemodBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PskDemodBlock")
            .field("config", &self.config)
            .field("costas", &self.costas)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> PskDemodBlock<S> {
    /// Designs the matched filter and creates the block; taps, history and work buffers come
    /// from `storage`.
    ///
    /// # Errors
    /// As [`PskDemodConfig::validate`]; `Storage` when the storage is too small.
    pub fn new(config: &PskDemodConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        let n = config.derive(rate_in)?;
        let rrc = Rrc::new(config.rolloff, n, config.span)?;
        let taps_len = rrc.len();
        let gain = 1.0 / libm::sqrt(n as f64);
        let mut taps = storage.alloc::<f32>(taps_len)?;
        for (k, t) in taps.as_mut().iter_mut().enumerate() {
            *t = (rrc.tap(k) * gain) as f32;
        }
        let hist = storage.alloc::<C32>(taps_len - 1)?;
        let a = storage.alloc::<C32>(PSK_DEMOD_CHUNK)?;
        let b = storage.alloc::<C32>(PSK_DEMOD_CHUNK)?;
        let bps = config.bits_per_symbol();
        let pending = storage.alloc::<f32>(PSK_DEMOD_CHUNK * bps)?;
        let sync_cfg = config.sync();
        let block = SymbolSyncBlock::<C32, S>::new(&sync_cfg, rate_in, storage)?;
        let ratio = block.timing().ratio.mul(Ratio::new(bps as u64, 1)?)?;
        let min_input = block.min_input();
        let sync = *block.state();
        let costas = CostasState::new(
            config.order,
            config.carrier_bw,
            DEFAULT_DAMPING,
            config.max_offset,
        )?;
        Ok(Self {
            config: *config,
            bps,
            sps: n,
            ratio,
            min_input,
            fir: FirState::new(taps, hist)?,
            sync,
            costas,
            prev: C32::new(1.0, 0.0),
            a,
            b,
            pending,
            at: 0,
            len: 0,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// # Errors
    /// As [`PskDemodConfig::validate`].
    pub fn storage_bytes(config: &PskDemodConfig, rate_in: Rate) -> Result<usize> {
        let n = config.derive(rate_in)?;
        let taps = config.span * n + 1;
        let mut plan = Plan::default();
        plan.add::<f32>(taps)?;
        plan.add::<C32>(taps - 1)?;
        plan.add::<C32>(PSK_DEMOD_CHUNK)?;
        plan.add::<C32>(PSK_DEMOD_CHUNK)?;
        plan.add::<f32>(PSK_DEMOD_CHUNK * config.bits_per_symbol())?;
        Ok(plan.bytes())
    }

    /// Output rate: the symbol rate (on the 1/1000 Hz grid) times the bits per symbol.
    ///
    /// # Errors
    /// As [`PskDemodConfig::validate`].
    pub fn rate_out(config: &PskDemodConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        let sym = SymbolSyncBlock::<C32, S>::rate_out(&config.sync(), rate_in)?;
        Ok(sym.scale(Ratio::new(config.bits_per_symbol() as u64, 1)?)?)
    }

    /// Number of taps of the matched filter (`span n + 1`).
    pub fn num_taps(&self) -> usize {
        self.fir.taps().len()
    }

    /// The integer samples per symbol of the matched filter.
    pub fn samples_per_symbol(&self) -> usize {
        self.sps
    }

    /// Whether the carrier loop reports lock (always false with `differential`).
    pub fn locked(&self) -> bool {
        !self.config.differential && self.costas.locked()
    }

    /// The carrier loop's frequency estimate in cycles per symbol (0 with `differential`).
    pub fn frequency(&self) -> f64 {
        if self.config.differential {
            0.0
        } else {
            self.costas.frequency()
        }
    }

    /// The current estimate of the samples per symbol of the timing loop.
    pub fn sps_estimate(&self) -> f64 {
        self.sync.sps_estimate()
    }

    /// Writes waiting soft bits to `output`; returns how many.
    fn flush(&mut self, output: &mut [f32]) -> Result<usize> {
        let waiting = head(tail(self.pending.as_ref(), self.at)?, self.len - self.at)?;
        let k = waiting.len().min(output.len());
        head_mut(output, k)?.copy_from_slice(head(waiting, k)?);
        self.at += k;
        if self.at == self.len {
            self.at = 0;
            self.len = 0;
        }
        Ok(k)
    }

    /// One piece of `m` input samples: the symbols in `a[..ns]`; returns `ns`.
    fn symbols(&mut self, piece: &[C32]) -> Result<usize> {
        let m = piece.len();
        let mf = head_mut(self.a.as_mut(), m)?;
        <Sdrk as Fir<C32>>::fir(&Sdrk, &mut self.fir, piece, mf)?;
        let p = <Sdrk as SymbolSync<C32>>::symbol_sync(
            &Sdrk,
            &mut self.sync,
            head(self.a.as_ref(), m)?,
            self.b.as_mut(),
        )?;
        if p.consumed != m {
            return Err(Error::Core(sdrk_core::Error::Internal(
                "psk_demod timing loop did not take the whole piece",
            )));
        }
        let ns = p.produced;
        let syms = head(self.b.as_ref(), ns)?;
        let out = head_mut(self.a.as_mut(), ns)?;
        if self.config.differential {
            let mut prev = self.prev;
            for (z, &y) in out.iter_mut().zip(syms) {
                *z = C32::new(
                    y.re * prev.re + y.im * prev.im,
                    y.im * prev.re - y.re * prev.im,
                );
                prev = y;
            }
            self.prev = prev;
        } else {
            <Sdrk as Costas>::costas(&Sdrk, &mut self.costas, syms, out)?;
        }
        Ok(ns)
    }
}

impl<S: Storage> Block for PskDemodBlock<S> {
    type In = C32;
    type Out = f32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[C32], output: &mut [f32]) -> Result<Progress> {
        let mut produced = self.flush(output)?;
        let mut consumed = 0usize;
        let qpsk = self.bps == 2;
        while self.len == 0 && produced < output.len() {
            let m = (input.len() - consumed).min(PSK_DEMOD_CHUNK);
            if m == 0 {
                break;
            }
            let ns = self.symbols(head(tail(input, consumed)?, m)?)?;
            consumed += m;
            let syms = head(self.a.as_ref(), ns)?;
            let room = tail_mut(output, produced)?;
            let direct = soft_bits(qpsk, 1.0, syms, room);
            produced += direct * self.bps;
            let rest = tail(syms, direct)?;
            if !rest.is_empty() {
                let n = soft_bits(qpsk, 1.0, rest, self.pending.as_mut());
                self.at = 0;
                self.len = n * self.bps;
                produced += self.flush(tail_mut(output, produced)?)?;
            }
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: self.ratio,
            kind: RatioKind::Variable,
            delay: Delay::new((self.config.span * self.sps + 2) as u64, 2).unwrap_or(Delay::ZERO),
        }
    }

    fn reset(&mut self) {
        self.fir.reset();
        self.sync.reset();
        self.costas.reset();
        self.prev = C32::new(1.0, 0.0);
        self.at = 0;
        self.len = 0;
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
        self.min_input
    }
}
