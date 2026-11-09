//! `psk_demod`: BPSK, QPSK and 8PSK demodulator (matched filter, symbol timing, carrier
//! recovery or differential detection, soft bits), with the QPSK variants pi/4-DQPSK and
//! offset QPSK.

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::costas::{Costas, CostasGains, CostasState};
use sdrk_core::kernels::equalize::{Equalize, EqualizeState};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::kernels::symbol_sync::{SymbolSync, SymbolSyncState};
use sdrk_core::num::flush_denormal;

use super::costas::{
    DEFAULT_DAMPING, DEFAULT_LOOP_BW, DEFAULT_MAX_OFFSET, MAX_LOOP_BW, MAX_OFFSET_LIMIT,
    MIN_LOOP_BW, check_f64, check_order, costas_constellation_of_order,
};
use super::equalize::{
    DEFAULT_CMA_SYMBOLS, DEFAULT_MU_CMA, DEFAULT_MU_DD, EQ_FEEDBACK_MAX, EQ_TAPS, EqualizeConfig,
    constellation_of_order,
};
use super::psk_mod::{PskVariant, VARIANT_PARAM, bits_of_order, check_variant};
use super::slice::{psk8_bits, soft_bits};
use super::symbol_sync::{SymbolSyncBlock, min_input as sync_min_input};
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

/// A full turn of the phase accumulators, `2^32`.
const TURN: f64 = 4_294_967_296.0;
/// The smoothing factor of the `offset` lock detector (that of `costas`).
const LOCK_ALPHA: f32 = 0.01;
/// The `offset` lock detector turns on above this level (that of `costas`).
const LOCK_ON: f32 = 0.5;
/// The `offset` lock detector turns off below this level (that of `costas`).
const LOCK_OFF: f32 = 0.3;

/// The reason of the rate error (a hint to resample).
const RATE_REASON: &str = "the sample rate must be within 1 % of an integer multiple (2 to \
    1024) of the symbol rate and at least twice it; put a resample before psk_demod";

/// Description of the `psk_demod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "psk_demod",
    summary: "Demodulates BPSK, QPSK or 8PSK to soft bits: matched filter, symbol timing and \
        carrier recovery.",
    details: "The input is the complex baseband at roughly the nominal level (an agc before \
        it) with fs / symbol_rate_hz within 1 % of an integer and at least 2 (otherwise the \
        rate is unsupported: resample first). A matched root-raised-cosine filter (rolloff, \
        span symbols, normalized so that the pulse of psk_mod gives amplitude 1 at the symbol \
        centres) is followed by symbol_sync (Gardner, loop bandwidth sync_bw, damping 0.7071, \
        max_deviation 0.005, ted_gain 1) and, for coherent detection, the Costas loop costas \
        (carrier_bw, damping 0.7071, max_offset cycles per symbol); with differential set the \
        loop is replaced by z_k = y_k conj(y_k-1) (y_-1 = 1). The symbols become soft bits as \
        slice does (BPSK re; QPSK re then im; 8PSK the max-log soft bits of slice 8psk): \
        above 0 is a 1. The variant pi4 (order 4, differential) maps z_k back to the bits of \
        pi/4-DQPSK: -im for bit 0, then -re for bit 1. The variant offset (order 4, not \
        differential, even samples per symbol) has its own carrier loop at the sample rate: \
        the matched filter output is derotated sample by sample with the phase and frequency \
        of the loop, then the in-phase part is delayed by half a symbol (which undoes the \
        half-symbol offset of OQPSK) before symbol_sync; the loop is fed back per symbol \
        from the OQPSK phase error of the in-phase and quadrature parts half a symbol apart \
        (carrier_bw, damping 0.7071, max_offset as for costas). The phase ambiguity of \
        coherent detection (180 degrees for BPSK, 90 degrees for QPSK, 45 degrees for 8PSK) \
        is not resolved: use frame_decode with polarity both or differential coding. The \
        output rate is the symbol rate times the bits per symbol and follows the clock of the \
        signal (variable ratio). With equalizer_taps above 0 an equalize stage (taps \
        equalizer_taps, feedback_taps equalizer_feedback_taps, the constellation of order, \
        mu_cma 0.05, mu_dd 0.02, 1000 CMA symbols, phase_bw carrier_bw) runs between \
        symbol_sync and costas and removes the inter-symbol interference of multipath and \
        filters; with differential it runs before the differential product in CMA only and \
        without its phase tracker, with offset after the carrier loop of offset; it adds \
        (equalizer_taps - 1) / 2 symbols of delay.",
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
                max: 8,
                default: Some(2),
            },
            typical: "2 for BPSK, 4 for QPSK, 8 for 8PSK",
            mutability: Mutability::Build,
            description: "Order of the phase-shift keying: 2 (BPSK), 4 (QPSK) or 8 (8PSK).",
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
        VARIANT_PARAM,
        ParamDesc {
            name: "equalizer_taps",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 0,
                max: EQ_TAPS.1 as i64,
                default: Some(0),
            },
            typical: "0 (off); 11 to 15 against multipath",
            mutability: Mutability::Build,
            description: "Forward taps of the adaptive equalizer between the timing loop and \
                the carrier loop: 0 (off) or odd from 3 to 63.",
        },
        ParamDesc {
            name: "equalizer_feedback_taps",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 0,
                max: EQ_FEEDBACK_MAX as i64,
                default: Some(0),
            },
            typical: "0 (linear); 5 for channels with deep spectral nulls",
            mutability: Mutability::Build,
            description: "Feedback taps of the equalizer (a decision feedback equalizer, 0 to \
                63); only with equalizer_taps above 0.",
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
    /// Order: 2 (BPSK), 4 (QPSK) or 8 (8PSK).
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
    /// The variant: plain, pi/4-DQPSK or offset QPSK (M40.1).
    pub variant: PskVariant,
    /// Forward taps of the equalizer between the timing loop and the carrier loop (M43): 0
    /// (off) or odd from 3 to 63.
    pub equalizer_taps: usize,
    /// Feedback taps of the equalizer (0 to 63; 0 = linear); only with `equalizer_taps` above
    /// 0.
    pub equalizer_feedback_taps: usize,
}

impl Default for PskDemodConfig {
    /// Order 2, `symbol_rate_hz` 1200 (a placeholder: the pipeline requires the parameter),
    /// `rolloff` 0.35, `span` 10, coherent, `sync_bw` 0.01, `carrier_bw` 0.02, `max_offset`
    /// 0.05, variant plain, no equalizer.
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
            variant: PskVariant::Plain,
            equalizer_taps: 0,
            equalizer_feedback_taps: 0,
        }
    }
}

/// `equalizer_taps`: 0 or odd from 3 to 63; `OutOfRange` above 63, `InvalidParameter` for 1
/// and the even counts (inside the range of the description but not allowed).
/// `equalizer_feedback_taps`: 0 to 63 (`OutOfRange` above), and 0 without the equalizer
/// (`InvalidParameter`).
fn check_feedback_taps(feedback: usize, taps: usize) -> Result<()> {
    check_count("equalizer_feedback_taps", feedback, 0, EQ_FEEDBACK_MAX)?;
    if feedback != 0 && taps == 0 {
        return Err(Error::InvalidParameter {
            name: "equalizer_feedback_taps",
            reason: "the equalizer feedback taps need equalizer_taps above 0",
        });
    }
    Ok(())
}

fn check_equalizer_taps(taps: usize) -> Result<()> {
    check_count("equalizer_taps", taps, 0, EQ_TAPS.1)?;
    if taps == 0 || (taps >= EQ_TAPS.0 && !taps.is_multiple_of(2)) {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name: "equalizer_taps",
            reason: "the equalizer taps must be 0 (off) or odd from 3 to 63",
        })
    }
}

impl PskDemodConfig {
    /// Bits per symbol: 1 for order 2, 2 for order 4, 3 for order 8.
    pub fn bits_per_symbol(&self) -> usize {
        bits_of_order(self.order)
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

    /// The configuration of the equalizer (`None` for `equalizer_taps` 0): the forward and
    /// feedback taps, the constellation of the order, the step sizes and CMA symbols of the
    /// `equalize` defaults and the phase tracker at `carrier_bw`; with `differential` CMA only
    /// (`cma_symbols` `u32::MAX`, D815) and without the tracker (`phase_bw` 0, D827).
    pub fn equalizer(&self) -> Option<EqualizeConfig> {
        (self.equalizer_taps != 0).then(|| EqualizeConfig {
            taps: self.equalizer_taps,
            feedback_taps: self.equalizer_feedback_taps,
            constellation: constellation_of_order(self.order),
            mu_cma: DEFAULT_MU_CMA,
            mu_dd: DEFAULT_MU_DD,
            cma_symbols: if self.differential {
                u32::MAX
            } else {
                DEFAULT_CMA_SYMBOLS
            },
            phase_bw: if self.differential {
                0.0
            } else {
                self.carrier_bw
            },
        })
    }

    /// The ranges, and the integer samples per symbol of the matched filter.
    fn derive(&self, rate_in: Rate) -> Result<usize> {
        check_order(self.order)?;
        check_equalizer_taps(self.equalizer_taps)?;
        check_feedback_taps(self.equalizer_feedback_taps, self.equalizer_taps)?;
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
        let n = n as usize;
        check_variant(
            self.variant,
            self.order,
            self.differential,
            n.is_multiple_of(2),
        )?;
        Ok(n)
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
    /// `OutOfRange` for a value outside its range; `InvalidParameter` for the orders 3, 5, 6
    /// and 7, for a symbol rate whose samples per symbol are not within 1 % of an integer from
    /// 2 to 1024 or are below 2 (the pipeline reports it as `rate_unsupported` with the hint to
    /// resample), for an `equalizer_taps` of 1 or an even count, and for a `variant` that does
    /// not fit (`pi4` without order 4 and
    /// `differential`, `offset` without order 4, with `differential` or with an odd rounded
    /// number of samples per symbol; the pipeline reports it as `invalid_value`).
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        self.derive(rate_in).map(|_| ())
    }

    /// Sets a parameter by its description name; checks kind and range (the combination of
    /// the parameters is checked by [`PskDemodConfig::validate`]).
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`; `InvalidParameter` for the orders
    /// 3, 5, 6 and 7 and for an `equalizer_taps` of 1 or an even count.
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
            "variant" => self.variant = PskVariant::from_index(value.as_choice()?),
            "equalizer_taps" => {
                let t = value.as_usize()?;
                check_equalizer_taps(t)?;
                self.equalizer_taps = t;
            }
            "equalizer_feedback_taps" => self.equalizer_feedback_taps = value.as_usize()?,
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
            "variant" => Ok(ParamValue::Choice(self.variant.index())),
            "equalizer_taps" => Ok(ParamValue::I64(self.equalizer_taps as i64)),
            "equalizer_feedback_taps" => Ok(ParamValue::I64(self.equalizer_feedback_taps as i64)),
            _ => unknown(),
        }
    }
}

/// BPSK, QPSK and 8PSK demodulator: `C32` in, `f32` soft bits out (`bits_per_symbol` per
/// symbol, above 0 is a 1).
///
/// Per piece of at most [`PSK_DEMOD_CHUNK`] input samples: the kernel `fir` (matched filter,
/// taps `rrc(rolloff, n, span) / sqrt(n)` as `f32` with `n` the rounded samples per symbol),
/// the kernel `symbol_sync` (Gardner at the exact `fs / symbol_rate_hz`), then the kernel
/// `costas` of the order (coherent) or the differential product `z_k = y_k conj(y_(k-1))`
/// with `y_(-1) = 1` (`f32`: `(y.re p.re + y.im p.im, y.im p.re - y.re p.im)`), and the
/// soft bits of `slice` (scale 1: BPSK `re`, QPSK `re` then `im`, 8PSK the max-log bits of
/// `8psk`; the variant `pi4` gives `-im` then `-re`). The variant `offset` replaces `costas` by a carrier loop at the sample
/// rate before the timing loop (D686 to D688): between two strobes the matched-filter
/// output is derotated with the kernel `rotate`, the in-phase part is delayed by `n / 2`
/// samples, `symbol_sync` takes the samples up to the next strobe, and the OQPSK phase error
/// of that symbol updates the loop before the next sample (see [`PskDemodBlock::locked`]).
/// Soft bits that do not fit into the output wait in a buffer and are written first by the
/// next call (which then consumes no input). Timing: variable ratio
/// `symbol_rate bits / fs`, delay `span n / 2 + 1` input samples (also for `offset`: the
/// in-phase delay realigns the early in-phase part with the quadrature part, D658).
/// Storage: the taps, the filter history, one work buffer of `512 + 128 bits` `C32` (filter
/// output, symbols, waiting soft bits as `f32`), for `offset` one buffer of `4 + n / 2 + PSK_DEMOD_CHUNK` `C32`
/// (the interpolation window, the ring of the in-phase delay and the derotated run) and for
/// `equalizer_taps` `t > 0` the `t + f` weights and the `t + f` history values of the equalizer
/// (`f` = `equalizer_feedback_taps`).
///
/// With `equalizer_taps` above 0 (M43 §3 as amended, D825) the kernel `equalize` (the
/// constellation of the order, `equalizer_feedback_taps` feedback taps, `mu_cma` 0.05, `mu_dd`
/// 0.02, 1000 CMA symbols, its phase tracker at `carrier_bw`) runs on the symbols of
/// `symbol_sync` before `costas`; for `offset` after its loop; with `differential` before the
/// differential product with `cma_symbols` `u32::MAX` (CMA only) and `phase_bw` 0. It adds
/// `(t - 1) / 2` symbols to the delay; with `equalizer_taps` 0 the block is unchanged.
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
    /// Bits per symbol and the rounded samples per symbol (`u32`, and `min_input` computed
    /// from the timing loop, keep the block within 512 bytes with the equalizer, D817).
    bps: u32,
    sps: u32,
    ratio: Ratio,
    fir: FirState<C32, S::Buf<f32>, S::Buf<C32>>,
    sync: SymbolSyncState<C32>,
    costas: CostasState,
    /// The previous symbol of the differential detector.
    prev: C32,
    /// One work buffer `[a | b | pending]` (see [`split_work`]): `a` the matched-filter
    /// output, then the symbols after the carrier loop; `b` the symbols of `symbol_sync`;
    /// `pending` (as `f32`) the soft bits that did not fit into the output, `pending[at..len]`.
    work: S::Buf<C32>,
    at: u32,
    len: u32,
    /// The carrier loop of `offset`; `None` for the other variants.
    offset: Option<OffsetLoop<S>>,
    /// The equalizer before the carrier loop; `None` for `equalizer_taps` 0.
    eq: Option<EqState<S>>,
}

/// The state of the equalizer of `psk_demod` on the storage `S`.
type EqState<S> = EqualizeState<<S as Storage>::Buf<C32>, <S as Storage>::Buf<C32>>;

/// The timing loop of `psk_demod` (through `SymbolSyncBlock`, which checks the rate) and the
/// ratio of soft bits per input sample.
fn timing_loop<S: Storage>(
    config: &PskDemodConfig,
    rate_in: Rate,
    bps: usize,
    storage: &mut S,
) -> Result<(Ratio, SymbolSyncState<C32>)> {
    let block = SymbolSyncBlock::<C32, S>::new(&config.sync(), rate_in, storage)?;
    let ratio = block.timing().ratio.mul(Ratio::new(bps as u64, 1)?)?;
    Ok((ratio, *block.state()))
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
        let bps = config.bits_per_symbol();
        let work = storage.alloc::<C32>(work_len(bps))?;
        let offset = if config.variant == PskVariant::Offset {
            Some(OffsetLoop::new(n, storage)?)
        } else {
            None
        };
        let eq = match config.equalizer() {
            Some(e) => Some(e.state(storage)?),
            None => None,
        };
        let (ratio, sync) = timing_loop(config, rate_in, bps, storage)?;
        let costas = CostasState::new(
            costas_constellation_of_order(config.order),
            config.carrier_bw,
            DEFAULT_DAMPING,
            config.max_offset,
        )?;
        Ok(Self {
            config: *config,
            bps: to_u32(bps)?,
            sps: to_u32(n)?,
            ratio,
            fir: FirState::new(taps, hist)?,
            sync,
            costas,
            prev: C32::new(1.0, 0.0),
            work,
            at: 0,
            len: 0,
            offset,
            eq,
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
        plan.add::<C32>(work_len(config.bits_per_symbol()))?;
        if config.variant == PskVariant::Offset {
            plan.add::<C32>(WINDOW + n / 2 + PSK_DEMOD_CHUNK)?;
        }
        if let Some(e) = config.equalizer() {
            e.plan(&mut plan)?;
        }
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
        self.sps as usize
    }

    /// Whether the carrier loop reports lock (always false with `differential`).
    ///
    /// For `offset` the lock detector of the sample-rate loop smooths the order-4 metric of
    /// `costas`, `-cos(4 arg y)` of the realigned symbol `y`, with the factor 0.01 and the
    /// thresholds 0.5 (on) and 0.3 (off) of `costas` (D687).
    pub fn locked(&self) -> bool {
        if self.config.differential {
            false
        } else if let Some(o) = &self.offset {
            o.locked
        } else {
            self.costas.locked()
        }
    }

    /// The carrier loop's frequency estimate in cycles per symbol (0 with `differential`).
    pub fn frequency(&self) -> f64 {
        if self.config.differential {
            0.0
        } else if let Some(o) = &self.offset {
            f64::from(o.freq)
        } else {
            self.costas.frequency()
        }
    }

    /// The state of the equalizer (`None` for `equalizer_taps` 0).
    pub fn equalizer(&self) -> Option<&EqState<S>> {
        self.eq.as_ref()
    }

    /// The current estimate of the samples per symbol of the timing loop.
    pub fn sps_estimate(&self) -> f64 {
        self.sync.sps_estimate()
    }

    /// The soft-bit map of the order and the variant.
    fn map(&self) -> SoftMap {
        match (self.bps, self.config.variant) {
            (3, _) => SoftMap::Psk8,
            (2, PskVariant::Pi4) => SoftMap::Pi4,
            (2, _) => SoftMap::Qpsk,
            _ => SoftMap::Bpsk,
        }
    }

    /// Writes waiting soft bits to `output`; returns how many.
    fn flush(&mut self, output: &mut [f32]) -> Result<usize> {
        let (at, len) = (self.at as usize, self.len as usize);
        let (_, _, pending) = split_work(self.work.as_mut())?;
        let waiting = head(tail(pending, at)?, len - at)?;
        let k = waiting.len().min(output.len());
        head_mut(output, k)?.copy_from_slice(head(waiting, k)?);
        // `k <= len <= PSK_DEMOD_CHUNK bits`.
        self.at += k as u32;
        if self.at == self.len {
            self.at = 0;
            self.len = 0;
        }
        Ok(k)
    }

    /// One piece of `m` input samples: the symbols in `a[..ns]`, or in `b[..ns]` when the
    /// second value is true (coherent detection with the equalizer); returns `ns`.
    #[allow(clippy::many_single_char_names)]
    fn symbols(&mut self, piece: &[C32]) -> Result<(usize, bool)> {
        let m = piece.len();
        let (a, b, _) = split_work(self.work.as_mut())?;
        <Sdrk as Fir<C32>>::fir(&Sdrk, &mut self.fir, piece, head_mut(a, m)?)?;
        if let Some(o) = self.offset.as_mut() {
            let gains = self.costas.gains();
            let ns = o.symbols(&gains, &mut self.sync, head(a, m)?, b)?;
            let syms = head(b, ns)?;
            let out = head_mut(a, ns)?;
            match self.eq.as_mut() {
                Some(eq) => <Sdrk as Equalize>::equalize(&Sdrk, eq, syms, out)?,
                None => out.copy_from_slice(syms),
            }
            return Ok((ns, false));
        }
        let p = <Sdrk as SymbolSync<C32>>::symbol_sync(&Sdrk, &mut self.sync, head(a, m)?, b)?;
        if p.consumed != m {
            return Err(Error::Core(sdrk_core::Error::Internal(
                "psk_demod timing loop did not take the whole piece",
            )));
        }
        let carrier = Carrier::<S> {
            differential: self.config.differential,
            prev: &mut self.prev,
            costas: &mut self.costas,
            eq: self.eq.as_mut(),
        };
        carrier.run(a, b, p.produced)
    }
}

/// The stage after the timing loop of `psk_demod` (plain and `pi4`): the equalizer, then the
/// differential product or `costas`.
struct Carrier<'a, S: Storage> {
    differential: bool,
    prev: &'a mut C32,
    costas: &'a mut CostasState,
    eq: Option<&'a mut EqState<S>>,
}

impl<S: Storage> Carrier<'_, S> {
    /// The `ns` symbols of `symbol_sync` in `b`; the result in `a[..ns]`, or in `b[..ns]` when
    /// the returned flag is true (coherent detection with the equalizer).
    fn run(self, a: &mut [C32], b: &mut [C32], ns: usize) -> Result<(usize, bool)> {
        let syms = head(b, ns)?;
        let out = head_mut(a, ns)?;
        if self.differential {
            let mut prev = *self.prev;
            if let Some(eq) = self.eq {
                // CMA only before the differential product, which then runs in place.
                <Sdrk as Equalize>::equalize(&Sdrk, eq, syms, out)?;
                for z in out.iter_mut() {
                    let y = *z;
                    *z = differential(y, prev);
                    prev = y;
                }
            } else {
                for (z, &y) in out.iter_mut().zip(syms) {
                    *z = differential(y, prev);
                    prev = y;
                }
            }
            *self.prev = prev;
            Ok((ns, false))
        } else if let Some(eq) = self.eq {
            // The equalizer before the carrier loop (D825): `b` -> `a` -> `b`.
            <Sdrk as Equalize>::equalize(&Sdrk, eq, syms, out)?;
            <Sdrk as Costas>::costas(&Sdrk, self.costas, head(a, ns)?, head_mut(b, ns)?)?;
            Ok((ns, true))
        } else {
            <Sdrk as Costas>::costas(&Sdrk, self.costas, syms, out)?;
            Ok((ns, false))
        }
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
        let map = self.map();
        while self.len == 0 && produced < output.len() {
            let m = (input.len() - consumed).min(PSK_DEMOD_CHUNK);
            if m == 0 {
                break;
            }
            let (ns, in_b) = self.symbols(head(tail(input, consumed)?, m)?)?;
            consumed += m;
            let bps = self.bps as usize;
            let (a, b, pending) = split_work(self.work.as_mut())?;
            let syms = head(if in_b { b } else { a }, ns)?;
            let room = tail_mut(output, produced)?;
            let direct = demod_bits(map, syms, room);
            produced += direct * bps;
            let rest = tail(syms, direct)?;
            if !rest.is_empty() {
                let n = demod_bits(map, rest, pending);
                self.at = 0;
                self.len = to_u32(n * bps)?;
                produced += self.flush(tail_mut(output, produced)?)?;
            }
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: self.ratio,
            kind: RatioKind::Variable,
            // `offset` adds nothing: the in-phase delay only realigns the in-phase part, which
            // is half a symbol early, with the quadrature part, the reference of the delay that
            // `psk_mod` reports (D658).
            // The equalizer adds `(taps - 1) / 2` symbols of `sps` input samples.
            delay: Delay::new(
                (self.config.span * self.sps as usize
                    + 2
                    + self.config.equalizer_taps.saturating_sub(1) * self.sps as usize)
                    as u64,
                2,
            )
            .unwrap_or(Delay::ZERO),
        }
    }

    fn reset(&mut self) {
        self.fir.reset();
        self.sync.reset();
        self.costas.reset();
        self.prev = C32::new(1.0, 0.0);
        self.at = 0;
        self.len = 0;
        if let Some(o) = self.offset.as_mut() {
            o.reset();
        }
        if let Some(eq) = self.eq.as_mut() {
            eq.reset();
        }
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
        // That of the timing loop (`SymbolSyncBlock::min_input`) at its exact samples per
        // symbol.
        sync_min_input(self.sync.sps(), SYNC_MAX_DEVIATION)
    }
}

/// The soft-bit maps of `psk_demod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SoftMap {
    /// `re` (BPSK).
    Bpsk,
    /// `re`, then `im` (QPSK, plain and offset).
    Qpsk,
    /// `-im`, then `-re`: the quadrants of `z_k` back to the bit pairs of pi/4-DQPSK
    /// (`+pi/4` is `00`, `+3 pi/4` `01`, `-3 pi/4` `11`, `-pi/4` `10`).
    Pi4,
    /// The max-log soft bits of `slice` `8psk`.
    Psk8,
}

/// The soft bits of the symbols in `input` (scale 1) with the map `map`; takes
/// `min(input.len(), output.len() / bits)` symbols and returns that count.
fn demod_bits(map: SoftMap, input: &[C32], output: &mut [f32]) -> usize {
    match map {
        SoftMap::Bpsk => soft_bits(false, 1.0, input, output),
        SoftMap::Qpsk => soft_bits(true, 1.0, input, output),
        SoftMap::Pi4 => {
            for (o, z) in output.chunks_exact_mut(2).zip(input) {
                if let [b0, b1] = o {
                    *b0 = -z.im;
                    *b1 = -z.re;
                }
            }
            input.len().min(output.len() / 2)
        }
        SoftMap::Psk8 => {
            for (o, z) in output.chunks_exact_mut(3).zip(input) {
                o.copy_from_slice(&psk8_bits(1.0, z.re, z.im));
            }
            input.len().min(output.len() / 3)
        }
    }
}

/// The length in `C32` of the work buffer `[a | b | pending]`: two pieces of symbols and
/// `PSK_DEMOD_CHUNK bps` soft bits as `f32` pairs.
fn work_len(bps: usize) -> usize {
    2 * PSK_DEMOD_CHUNK + (PSK_DEMOD_CHUNK * bps).div_ceil(2)
}

/// Splits the work buffer into `a` and `b` (`PSK_DEMOD_CHUNK` symbols each) and the soft-bit
/// buffer `pending` (`f32`).
fn split_work(work: &mut [C32]) -> Result<(&mut [C32], &mut [C32], &mut [f32])> {
    let bad = || Error::Core(sdrk_core::Error::Internal("psk_demod work buffer"));
    let (a, rest) = work.split_at_mut_checked(PSK_DEMOD_CHUNK).ok_or_else(bad)?;
    let (b, pending) = rest.split_at_mut_checked(PSK_DEMOD_CHUNK).ok_or_else(bad)?;
    Ok((a, b, bytemuck::cast_slice_mut(pending)))
}

/// `v` as `u32`, or an internal error.
fn to_u32(v: usize) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::Core(sdrk_core::Error::Internal("psk_demod count")))
}

/// The differential product `y conj(prev)` in `f32`:
/// `(y.re prev.re + y.im prev.im, y.im prev.re - y.re prev.im)`.
#[inline]
fn differential(y: C32, prev: C32) -> C32 {
    C32::new(
        y.re * prev.re + y.im * prev.im,
        y.im * prev.re - y.re * prev.im,
    )
}

/// `floor(v 2^32 + 0.5)` as a wrapping phase step (through `i64`, as `costas` rounds).
fn turn_step(v: f64) -> u32 {
    libm::floor(v * TURN + 0.5) as i64 as u32
}

/// `sign(x)`: 1 above zero, -1 below, 0 for zero and NaN.
fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// The cubic Lagrange interpolation of `symbol_sync` at `mu` from `x[-1], x[0], x[1], x[2]`
/// (the same f32 operations in the same order, per component).
fn interp1(xm1: f32, x0: f32, x1: f32, x2: f32, mu: f32) -> f32 {
    const THIRD: f32 = 1.0 / 3.0;
    const HALF: f32 = 0.5;
    const SIXTH: f32 = 1.0 / 6.0;
    let c1 = ((-(xm1 * THIRD) - x0 * HALF) + x1) - x2 * SIXTH;
    let c2 = (xm1 * HALF - x0) + x1 * HALF;
    let c3 = ((-(xm1 * SIXTH) + x0 * HALF) - x1 * HALF) + x2 * SIXTH;
    ((c3 * mu + c2) * mu + c1) * mu + x0
}

/// The carrier loop of the variant `offset` (D686 to D688): an oscillator at the sample rate
/// that derotates the matched-filter output before the in-phase delay, a proportional-integral
/// loop filter per symbol with the gains of `costas` order 4 and the lock detector of `costas`
/// order 4 on the realigned symbols.
///
/// Per sample `n` the derotated `r_n = x_n exp(j psi_n)` (kernel `rotate`, phase `psi` and
/// increment `inc` as `u32` turns) goes into a ring of `h = n / 2` samples; `symbol_sync`
/// gets `(re r_(n-h), im r_n)` (the in-phase part delayed), and a window of the last four
/// `(re r_n, im r_(n-h))` (the other two parts) is kept. The samples are given to
/// `symbol_sync` in runs that end with the last sample of the next strobe's window: right
/// after a strobe (and after construction or `reset`) `t = next_strobe()` gives the run length
/// `max(1, floor(t) + 3)` and `mu = (t - floor(t)) as f32`, which are kept until the strobe
/// (a run cut by the end of a piece continues in the next one), so a symbol `y` always comes
/// out at the end of a run; then `z` is the interpolation of the window at `mu` (the cubic
/// Lagrange interpolation of `symbol_sync`), the phase error is
/// `e = sign(y.re) z.im - sign(y.im) z.re` (`im r` at the in-phase centre and `re r` at the
/// quadrature centre half a symbol later; a non-finite `e` counts as 0), and in f32 without
/// FMA `freq = clamp(flush_denormal(freq + ki e), -max, max)` and `psi -= turn_step(f64(kp
/// e))`; the increment of every run is `inc = turn_step(-f64(freq) / sps)` with `sps` the
/// exact samples per symbol of `symbol_sync` (`fs / symbol_rate_hz`); the lock metric
/// `((4 a) b - d d) / (p p)` of `y` (`a = y.re y.re`, `b = y.im y.im`, `p = a + b`,
/// `d = a - b`), when finite, updates `l = flush_denormal(l + 0.01 (c - l))` with lock on
/// above 0.5 and off below 0.3. The update acts from the sample after the strobe's window on,
/// wherever the input is cut.
struct OffsetLoop<S: Storage> {
    /// One buffer `[window | ring | work]`: `window` (4) the `(re r_n, im r_(n-h))` of the
    /// last four samples, the newest last; `ring` (`h`) the last `h` derotated samples,
    /// `ring[pos]` the oldest; `work` ([`PSK_DEMOD_CHUNK`]) the run being realigned.
    mem: S::Buf<C32>,
    /// `h = n / 2`.
    half: u32,
    pos: u32,
    /// The oscillator phase `psi` before the next sample, in `2^-32` turns.
    phase: u32,
    /// The frequency integrator in cycles per symbol.
    freq: f32,
    level: f32,
    /// The `mu` of the next strobe and the samples still missing up to the end of its
    /// window, taken from `symbol_sync` once after each strobe (0: not taken yet).
    mu: f32,
    remaining: u32,
    locked: bool,
}

/// The four samples of the interpolation window.
const WINDOW: usize = 4;

impl<S: Storage> OffsetLoop<S> {
    /// The loop gains are those of `costas` order 4 (`kp`, `ki`, `max_offset` of
    /// [`CostasState::gains`]), passed in by the block, which keeps that state.
    fn new(n: usize, storage: &mut S) -> Result<Self> {
        let half = n / 2;
        Ok(Self {
            mem: storage.alloc::<C32>(WINDOW + half + PSK_DEMOD_CHUNK)?,
            half: u32::try_from(half).map_err(|_| {
                Error::Core(sdrk_core::Error::Internal("psk_demod offset ring length"))
            })?,
            pos: 0,
            phase: 0,
            freq: 0.0,
            level: 0.0,
            mu: 0.0,
            remaining: 0,
            locked: false,
        })
    }

    fn reset(&mut self) {
        let keep = WINDOW + self.half as usize;
        if let Some(state) = self.mem.as_mut().get_mut(..keep) {
            state.fill(C32::default());
        }
        self.pos = 0;
        self.phase = 0;
        self.freq = 0.0;
        self.level = 0.0;
        self.mu = 0.0;
        self.remaining = 0;
        self.locked = false;
    }

    /// Derotates `x`, realigns it and runs `symbol_sync`; writes the symbols to `out` and
    /// returns their count.
    #[allow(clippy::many_single_char_names)]
    fn symbols(
        &mut self,
        gains: &CostasGains,
        sync: &mut SymbolSyncState<C32>,
        x: &[C32],
        out: &mut [C32],
    ) -> Result<usize> {
        let mut done = 0usize;
        let mut ns = 0usize;
        let sps = sync.sps();
        while done < x.len() {
            if self.remaining == 0 {
                // Right after a strobe (or reset) the state of `symbol_sync` is the same
                // wherever the input was cut, so `mu` and the run length are too.
                let next = sync.next_strobe();
                let whole = libm::floor(next);
                self.mu = (next - whole) as f32;
                self.remaining = (whole as i64 + 3).clamp(1, i64::from(u32::MAX)) as u32;
            }
            let need = self.remaining as usize;
            let l = need.min(x.len() - done);
            let (state, work) = self.mem.as_mut().split_at_mut(WINDOW + self.half as usize);
            let (window, ring) = state.split_at_mut(WINDOW);
            let run = head_mut(work, l)?;
            let mut nco = Nco::from_increment(turn_step(-f64::from(self.freq) / sps));
            nco.set_phase(self.phase);
            <Sdrk as Rotate<C32>>::rotate(&Sdrk, &mut nco, head(tail(x, done)?, l)?, run)?;
            self.phase = nco.phase();
            for v in run.iter_mut() {
                let r = *v;
                let old = match ring.get_mut(self.pos as usize) {
                    Some(slot) => core::mem::replace(slot, r),
                    None => r,
                };
                self.pos += 1;
                if self.pos >= self.half {
                    self.pos = 0;
                }
                *v = C32::new(old.re, r.im);
                window.copy_within(1.., 0);
                if let Some(w) = window.last_mut() {
                    *w = C32::new(r.re, old.im);
                }
            }
            let p = <Sdrk as SymbolSync<C32>>::symbol_sync(
                &Sdrk,
                sync,
                head(run, l)?,
                tail_mut(out, ns)?,
            )?;
            // `l <= need <= u32::MAX`.
            self.remaining -= l as u32;
            if p.consumed != l || p.produced > 1 || (p.produced == 1) != (self.remaining == 0) {
                return Err(Error::Core(sdrk_core::Error::Internal(
                    "psk_demod offset run did not end at a strobe",
                )));
            }
            if p.produced == 1 {
                let y = *tail(out, ns)?
                    .first()
                    .ok_or(Error::Core(sdrk_core::Error::Internal(
                        "psk_demod offset symbol",
                    )))?;
                let w = match *window {
                    [w0, w1, w2, w3] => [w0, w1, w2, w3],
                    _ => [C32::default(); WINDOW],
                };
                self.update(gains, &w, y);
                ns += 1;
            }
            done += l;
        }
        Ok(ns)
    }

    /// The loop filter and the lock detector after the symbol `y` of the strobe at `self.mu`,
    /// with the interpolation window `w`.
    #[allow(clippy::many_single_char_names)]
    fn update(&mut self, g: &CostasGains, w: &[C32; WINDOW], y: C32) {
        let mu = self.mu;
        let [w0, w1, w2, w3] = *w;
        let z = C32::new(
            interp1(w0.re, w1.re, w2.re, w3.re, mu),
            interp1(w0.im, w1.im, w2.im, w3.im, mu),
        );
        let e = sign(y.re) * z.im - sign(y.im) * z.re;
        let e = if e.is_finite() { e } else { 0.0 };
        let f = flush_denormal(self.freq + g.ki * e);
        self.freq = if f > g.max_offset {
            g.max_offset
        } else if f < -g.max_offset {
            -g.max_offset
        } else {
            f
        };
        self.phase = self.phase.wrapping_sub(turn_step(f64::from(g.kp * e)));
        let a = y.re * y.re;
        let b = y.im * y.im;
        let p = a + b;
        let d = a - b;
        let c = ((4.0 * a) * b - d * d) / (p * p);
        if c.is_finite() {
            self.level = flush_denormal(self.level + LOCK_ALPHA * (c - self.level));
            if self.level > LOCK_ON {
                self.locked = true;
            } else if self.level < LOCK_OFF {
                self.locked = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::psk_mod::{phase_index, point16};
    use super::{SoftMap, demod_bits};
    use sdrk_core::C32;

    /// M40.1 §2: every bit pair of pi/4-DQPSK steps the phase by its increment in the
    /// transmitter, and the receiver maps the quadrant of `z_k` (the increment itself) back
    /// to the same pair, for any previous phase.
    #[test]
    fn pi4_quadrants_map_back_to_their_bit_pairs() {
        // (b0, b1), the increment in units of pi / 8
        for (b0, b1, step) in [
            (false, false, 2u8), // +pi/4
            (false, true, 6),    // +3 pi/4
            (true, true, 10),    // -3 pi/4
            (true, false, 14),   // -pi/4
        ] {
            let soft = [if b0 { 1.0f32 } else { -1.0 }, if b1 { 1.0 } else { -1.0 }];
            assert_eq!(phase_index(&soft, true), step, "{b0} {b1}");
            for prev in (0u8..16).step_by(2) {
                let (p, y) = (point16(prev), point16(prev.wrapping_add(step)));
                let z = C32::new(y.re * p.re + y.im * p.im, y.im * p.re - y.re * p.im);
                let mut out = [0.0f32; 2];
                assert_eq!(demod_bits(SoftMap::Pi4, &[z], &mut out), 1);
                assert_eq!(
                    (out.first().map(|v| *v > 0.0), out.get(1).map(|v| *v > 0.0)),
                    (Some(b0), Some(b1)),
                    "{b0} {b1} from phase {prev}: {out:?}"
                );
            }
        }
        // the plain QPSK map is a different table: +pi/4 is 11 there
        let mut out = [0.0f32; 2];
        demod_bits(SoftMap::Qpsk, &[point16(2)], &mut out);
        assert!(out.iter().all(|v| *v > 0.0));
    }
}
