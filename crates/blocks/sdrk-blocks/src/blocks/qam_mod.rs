//! `qam_mod`: 16-QAM and 64-QAM modulator (soft bits to root-raised-cosine shaped symbols).

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_core::C32;
use sdrk_core::kernels::constellation::Qam;

use super::costas::check_f64;
use super::interpolate::InterpolateBlock;
use super::symbol_map::{SymbolMapBlock, SymbolMapConfig, SymbolMapConstellation};
use super::{check_count, head, head_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::chain::Chain;
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::{RRC_MAX_SPAN, RRC_MIN_ROLLOFF, RRC_MIN_SPAN, Rrc};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};

/// The capacity in symbols of the buffer between `symbol_map` and the interpolation.
pub const QAM_MOD_CHUNK: usize = 256;

/// The largest `samples_per_symbol` of `qam_mod`.
const MAX_SPS: usize = 64;

/// The `order` parameter of `qam_mod` and `qam_demod`.
pub(crate) const ORDER_PARAM: ParamDesc = ParamDesc {
    name: "order",
    unit: Unit::None,
    kind: ParamKind::I64 {
        min: 16,
        max: 64,
        default: Some(16),
    },
    typical: "16 for 16-QAM, 64 for 64-QAM",
    mutability: Mutability::Build,
    description: "Order of the quadrature amplitude modulation: 16 (16-QAM, 4 bits per \
        symbol) or 64 (64-QAM, 6 bits per symbol).",
};

/// The constellation of a QAM order: 16 or 64, otherwise `InvalidParameter` (17 to 63 are
/// inside the range of the description but not supported) or `OutOfRange`.
pub(crate) fn qam_of_order(order: u32) -> Result<Qam> {
    match order {
        16 => Ok(Qam::Qam16),
        64 => Ok(Qam::Qam64),
        o if (16..=64).contains(&o) => Err(Error::InvalidParameter {
            name: "order",
            reason: "the order must be 16 (16-QAM) or 64 (64-QAM)",
        }),
        o => Err(Error::OutOfRange {
            name: "order",
            min: 16.0,
            max: 64.0,
            got: f64::from(o),
        }),
    }
}

/// The QAM order of a parameter value (`OutOfRange` for values that do not fit `u32`).
pub(crate) fn order_of_value(value: &ParamValue) -> Result<u32> {
    let o = u32::try_from(value.as_i64()?).unwrap_or(0);
    qam_of_order(o)?;
    Ok(o)
}

/// The `samples_per_symbol` parameter of `qam_mod` and `qam_frame_encode`.
pub(crate) const SAMPLES_PER_SYMBOL_PARAM: ParamDesc = ParamDesc {
    name: "samples_per_symbol",
    unit: Unit::None,
    kind: ParamKind::I64 {
        min: 2,
        max: MAX_SPS as i64,
        default: None,
    },
    typical: "8",
    mutability: Mutability::Build,
    description: "Output samples per symbol, 2 to 64. Required.",
};

/// The `rolloff` parameter of `qam_mod` and `qam_frame_encode`.
pub(crate) const MOD_ROLLOFF_PARAM: ParamDesc = ParamDesc {
    name: "rolloff",
    unit: Unit::None,
    kind: ParamKind::F64 {
        min: RRC_MIN_ROLLOFF,
        max: 1.0,
        default: Some(0.35),
    },
    typical: "0.35; 0.2 for a narrow spectrum",
    mutability: Mutability::Build,
    description: "Roll-off of the root-raised-cosine pulse.",
};

/// The `span` parameter of `qam_mod` and `qam_frame_encode`.
pub(crate) const MOD_SPAN_PARAM: ParamDesc = ParamDesc {
    name: "span",
    unit: Unit::None,
    kind: ParamKind::I64 {
        min: RRC_MIN_SPAN as i64,
        max: RRC_MAX_SPAN as i64,
        default: Some(10),
    },
    typical: "10; 16 or more for less intersymbol interference",
    mutability: Mutability::Build,
    description: "Length of the pulse in symbols (span samples_per_symbol + 1 taps).",
};

/// Description of the `qam_mod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "qam_mod",
    summary: "Turns soft bits into 16-QAM or 64-QAM symbols shaped with a root-raised-cosine \
        pulse.",
    details: "Reads one soft bit per input sample (above 0 is a 1, anything else, NaN \
        included, a 0; frame_encode produces them) and maps 4 (order 16) or 6 (order 64) bits, \
        the first one the most significant, to a point of the square constellation: the first \
        half of the bits selects the in-phase level, the second half the quadrature level, \
        each by the Gray code of the level index i (i XOR (i >> 1)); the level value is \
        2 i - (m - 1) with m = 4 or 8 levels per axis, scaled by 1 / sqrt(E) with E = 10 or 42 \
        (unit average power). These are the constellations qam16 and qam64 of symbol_demap, \
        equalize \
        and costas. The symbols are interpolated by samples_per_symbol with the \
        root-raised-cosine pulse (rolloff, span symbols) scaled by sqrt(samples_per_symbol), \
        so the matched filter of qam_demod gives the points at the symbol centres. The output \
        rate is the bit rate times samples_per_symbol / bits per symbol; the group delay is \
        span / 2 symbols. A call takes whole symbols only. Follow it with duc to reach the \
        radio rate.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32],
        description: "Soft bits, one sample per bit; above 0 is a 1.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32],
        description: "The shaped baseband signal, samples_per_symbol samples per symbol, unit \
            average power at the symbol centres.",
    }],
    params: &[
        ORDER_PARAM,
        SAMPLES_PER_SYMBOL_PARAM,
        MOD_ROLLOFF_PARAM,
        MOD_SPAN_PARAM,
    ],
    required: &["samples_per_symbol"],
};

/// Configuration of [`QamModBlock`].
///
/// ```
/// use sdrk_blocks::QamModConfig;
/// let cfg = QamModConfig { order: 64, samples_per_symbol: 8, ..QamModConfig::default() };
/// assert!(cfg.validate().is_ok());
/// assert_eq!(cfg.bits_per_symbol(), 6);
/// assert!(QamModConfig { order: 32, ..cfg }.validate().is_err());
/// assert!(QamModConfig { samples_per_symbol: 1, ..cfg }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QamModConfig {
    /// Order: 16 (16-QAM) or 64 (64-QAM).
    pub order: u32,
    /// Output samples per symbol, 2 to 64 (required).
    pub samples_per_symbol: usize,
    /// Roll-off of the root-raised-cosine pulse (0.05 to 1).
    pub rolloff: f64,
    /// Length of the pulse in symbols (4 to 32).
    pub span: usize,
}

impl Default for QamModConfig {
    /// Order 16, `samples_per_symbol` 8 (a placeholder: the pipeline requires the parameter),
    /// `rolloff` 0.35, `span` 10.
    fn default() -> Self {
        Self {
            order: 16,
            samples_per_symbol: 8,
            rolloff: 0.35,
            span: 10,
        }
    }
}

impl QamModConfig {
    /// Bits per symbol: 4 for order 16, 6 for order 64 (4 for an invalid order).
    pub fn bits_per_symbol(&self) -> usize {
        qam_of_order(self.order).map_or(4, Qam::bits_per_symbol)
    }

    /// Checks the ranges.
    ///
    /// # Errors
    /// `OutOfRange` for a value outside its range; `InvalidParameter` for the orders 17 to 63.
    pub fn validate(&self) -> Result<()> {
        qam_of_order(self.order)?;
        check_count("samples_per_symbol", self.samples_per_symbol, 2, MAX_SPS)?;
        check_f64("rolloff", self.rolloff, RRC_MIN_ROLLOFF, 1.0)?;
        check_count("span", self.span, RRC_MIN_SPAN, RRC_MAX_SPAN)
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`; `InvalidParameter` for the orders
    /// 17 to 63.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "order" => self.order = order_of_value(&value)?,
            "samples_per_symbol" => self.samples_per_symbol = value.as_usize()?,
            "rolloff" => self.rolloff = value.as_f64()?,
            "span" => self.span = value.as_usize()?,
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
            "samples_per_symbol" => Ok(ParamValue::I64(self.samples_per_symbol as i64)),
            "rolloff" => Ok(ParamValue::F64(self.rolloff)),
            "span" => Ok(ParamValue::I64(self.span as i64)),
            _ => unknown(),
        }
    }

    /// The output/input ratio `samples_per_symbol / bits_per_symbol`.
    fn ratio(&self) -> Result<Ratio> {
        Ok(Ratio::new(
            self.samples_per_symbol as u64,
            self.bits_per_symbol() as u64,
        )?)
    }
}

/// The stages of `qam_mod`: `symbol_map` then the interpolation with the pulse.
type QamModStages<S> = Chain<SymbolMapBlock<S>, InterpolateBlock<C32, S>, S>;

/// The pulse of the modulators `qam_mod` and `psk_mod`: the `span sps + 1` taps
/// `rrc(rolloff, sps, span) sqrt(sps)` in `f64`, rounded once to `f32` by
/// [`InterpolateBlock::with_taps`].
pub(crate) fn pulse_stage<S: Storage>(
    rolloff: f64,
    sps: usize,
    span: usize,
    storage: &mut S,
) -> Result<InterpolateBlock<C32, S>> {
    let rrc = Rrc::new(rolloff, sps, span)?;
    let gain = libm::sqrt(sps as f64);
    InterpolateBlock::with_taps((0..rrc.len()).map(|k| rrc.tap(k) * gain), sps, storage)
}

/// 16-QAM and 64-QAM modulator: `f32` soft bits in (one per sample), `C32` out at
/// `samples_per_symbol` samples per symbol.
///
/// A composite block (M44.1 §4): [`SymbolMapBlock`] (`qam16` or `qam64` from `order`, the `f32`
/// points of `sdrk_core::kernels::constellation::Qam`, M44 §1) then
/// [`InterpolateBlock::with_taps`] with the taps `rrc(rolloff, sps, span) sqrt(sps)` as `f32`
/// (the kernel `fir_interpolate` through the facade, as in `psk_mod`), as a [`Chain`] whose
/// buffer holds [`QAM_MOD_CHUNK`] symbols. A call takes `min(input / bits, output / sps)` whole
/// symbols. Timing: constant ratio `sps / bits`, delay `span bits / 2` input samples. Storage:
/// the taps, the filter history and the buffer of the chain.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, QamModBlock, QamModConfig};
/// let cfg = QamModConfig { order: 16, samples_per_symbol: 4, ..QamModConfig::default() };
/// let rate = Rate::hz(40_000)?;
/// let bytes = QamModBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut m = QamModBlock::new(&cfg, rate, &mut arena)?;
/// let bits = [1.0f32, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0];
/// let mut out = [C32::default(); 64];
/// let p = m.process(&bits, &mut out)?;
/// assert_eq!((p.consumed, p.produced), (8, 8)); // two symbols; the ninth bit waits
/// assert_eq!(QamModBlock::<ArenaStorage<'_>>::rate_out(&cfg, rate)?, Rate::hz(40_000)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct QamModBlock<S: Storage> {
    config: QamModConfig,
    bps: usize,
    sps: usize,
    ratio: Ratio,
    stages: QamModStages<S>,
}

impl<S: Storage> core::fmt::Debug for QamModBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QamModBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> QamModBlock<S> {
    /// Designs the pulse and creates the block; taps, history and the buffer of the chain
    /// come from `storage`.
    ///
    /// # Errors
    /// As [`QamModConfig::validate`]; `Storage` when the storage is too small.
    pub fn new(config: &QamModConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate()?;
        let constellation = match qam_of_order(config.order)? {
            Qam::Qam64 => SymbolMapConstellation::Qam64,
            _ => SymbolMapConstellation::Qam16,
        };
        let map = SymbolMapBlock::new(
            &SymbolMapConfig {
                constellation,
                ..SymbolMapConfig::default()
            },
            rate_in,
            storage,
        )?;
        let sps = config.samples_per_symbol;
        let pulse = pulse_stage(config.rolloff, sps, config.span, storage)?;
        Ok(Self {
            config: *config,
            bps: config.bits_per_symbol(),
            sps,
            ratio: config.ratio()?,
            stages: Chain::new(map, pulse, QAM_MOD_CHUNK, storage)?,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// # Errors
    /// As [`QamModConfig::validate`].
    pub fn storage_bytes(config: &QamModConfig, _rate_in: Rate) -> Result<usize> {
        let mut plan = Plan::default();
        Self::plan(config, &mut plan)?;
        Ok(plan.bytes())
    }

    /// Adds the buffers of the block to `plan`, in the order `new` takes them.
    pub(crate) fn plan(config: &QamModConfig, plan: &mut Plan) -> Result<()> {
        config.validate()?;
        let sps = config.samples_per_symbol;
        let n = config.span * sps + 1;
        plan.add::<f32>(n)?;
        plan.add::<C32>(n.div_ceil(sps))?;
        plan.add::<C32>(QAM_MOD_CHUNK)
    }

    /// The configuration of the block.
    pub fn config(&self) -> QamModConfig {
        self.config
    }

    /// Output rate: the bit rate times `samples_per_symbol / bits_per_symbol`.
    ///
    /// # Errors
    /// As [`QamModConfig::validate`]; `OutOfRange` when the rate overflows.
    pub fn rate_out(config: &QamModConfig, rate_in: Rate) -> Result<Rate> {
        config.validate()?;
        Ok(rate_in.scale(config.ratio()?)?)
    }

    /// Number of taps of the pulse (`span samples_per_symbol + 1`).
    pub fn num_taps(&self) -> usize {
        self.stages.second().num_taps()
    }
}

/// Runs whole symbols through the stages of a modulator: `min(input / bps, output / sps)`
/// symbols, so that no bit is taken whose samples do not fit (the buffer of the chain is empty
/// after every call).
pub(crate) fn modulate<B: Block<In = f32, Out = C32>>(
    stages: &mut B,
    bps: usize,
    sps: usize,
    input: &[f32],
    output: &mut [C32],
    what: &'static str,
) -> Result<Progress> {
    let syms = (input.len() / bps).min(output.len() / sps);
    if syms == 0 {
        return Ok(Progress::default());
    }
    let p = stages.process(head(input, syms * bps)?, head_mut(output, syms * sps)?)?;
    if p.consumed != syms * bps || p.produced != syms * sps {
        return Err(Error::Core(sdrk_core::Error::Internal(what)));
    }
    Ok(p)
}

impl<S: Storage> Block for QamModBlock<S> {
    type In = f32;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[f32], output: &mut [C32]) -> Result<Progress> {
        modulate(
            &mut self.stages,
            self.bps,
            self.sps,
            input,
            output,
            "qam_mod stages did not take the whole piece",
        )
    }

    fn timing(&self) -> BlockTiming {
        // The figures of the block of M44, as it reported them (the delay of the chain is the
        // same value, `span / 2` symbols of the interpolation).
        BlockTiming {
            ratio: self.ratio,
            kind: RatioKind::Constant,
            delay: Delay::new((self.config.span * self.bps) as u64, 2).unwrap_or(Delay::ZERO),
        }
    }

    fn reset(&mut self) {
        self.stages.reset();
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
        self.bps
    }
}
