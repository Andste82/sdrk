//! `qam_mod`: 16-QAM and 64-QAM modulator (soft bits to root-raised-cosine shaped symbols).

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_core::C32;
use sdrk_core::kernels::constellation::Qam;

use super::costas::check_f64;
use super::interpolate::InterpolateBlock;
use super::symbol_map::{
    CONSTELLATION_OPTIONS, SymbolMapBlock, SymbolMapConfig, SymbolMapConstellation,
};
use super::{check_count, head, head_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::chain::Chain;
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::{RRC_MAX_SPAN, RRC_MIN_ROLLOFF, RRC_MIN_SPAN, Rrc};
use crate::error::{Error, Result};
use crate::param_map::{
    ParamMap, Stages, check_build_param, check_composite_param, composite_param,
    composite_set_param, stage_of,
};
use crate::storage::{Plan, Storage};

/// The capacity in symbols of the buffer between `symbol_map` and the interpolation.
pub const QAM_MOD_CHUNK: usize = 256;

/// The largest `samples_per_symbol` of `qam_mod`.
const MAX_SPS: usize = 64;

/// The `constellation` parameter of `qam_mod`, `qam_demod`, `qam_frame_encode` and
/// `qam_frame_decode`: the choice of `symbol_map` (the same names at the same indices in
/// `costas`, `equalize` and `symbol_demap`), of which these blocks take `qam16` and `qam64`
/// (M44.1.1 §2).
pub(crate) const QAM_CONSTELLATION_PARAM: ParamDesc = ParamDesc {
    name: "constellation",
    unit: Unit::None,
    kind: ParamKind::Choice {
        options: CONSTELLATION_OPTIONS,
        default: 3,
    },
    typical: "qam16 or qam64 (bpsk, qpsk and 8psk are refused: psk_mod)",
    mutability: Mutability::Build,
    description: "The constellation: qam16 (16-QAM, 4 bits per symbol) or qam64 (64-QAM, 6 bits \
        per symbol); bpsk, qpsk and 8psk are those of psk_mod and psk_demod.",
};

/// The square QAM of a constellation: `qam16` or `qam64`, otherwise `InvalidParameter` of
/// `constellation` (`bpsk`, `qpsk` and `8psk` are in the choice but not QAM).
pub(crate) fn qam_of(c: SymbolMapConstellation) -> Result<Qam> {
    match c {
        SymbolMapConstellation::Qam16 => Ok(Qam::Qam16),
        SymbolMapConstellation::Qam64 => Ok(Qam::Qam64),
        _ => Err(Error::InvalidParameter {
            name: "constellation",
            reason: "the constellation must be qam16 or qam64",
        }),
    }
}

/// The QAM constellation of a parameter value (its range checked by the description).
pub(crate) fn qam_constellation_of_value(value: &ParamValue) -> Result<SymbolMapConstellation> {
    let c = SymbolMapConstellation::from_index(value.as_choice()?);
    qam_of(c)?;
    Ok(c)
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
        included, a 0; frame_encode produces them) and maps 4 (qam16) or 6 (qam64) bits, \
        the first one the most significant, to a point of the square constellation: the first \
        half of the bits selects the in-phase level, the second half the quadrature level, \
        each by the Gray code of the level index i (i XOR (i >> 1)); the level value is \
        2 i - (m - 1) with m = 4 or 8 levels per axis, scaled by 1 / sqrt(E) with E = 10 or 42 \
        (unit average power). These are the constellations qam16 and qam64 of symbol_map, \
        symbol_demap, equalize and costas. The symbols are interpolated by samples_per_symbol with the \
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
        QAM_CONSTELLATION_PARAM,
        SAMPLES_PER_SYMBOL_PARAM,
        MOD_ROLLOFF_PARAM,
        MOD_SPAN_PARAM,
    ],
    required: &["samples_per_symbol"],
};

/// Configuration of [`QamModBlock`].
///
/// ```
/// use sdrk_blocks::{QamModConfig, SymbolMapConstellation};
/// let cfg = QamModConfig {
///     constellation: SymbolMapConstellation::Qam64,
///     samples_per_symbol: 8,
///     ..QamModConfig::default()
/// };
/// assert!(cfg.validate().is_ok());
/// assert_eq!(cfg.bits_per_symbol(), 6);
/// let qpsk = QamModConfig { constellation: SymbolMapConstellation::Qpsk, ..cfg };
/// assert!(qpsk.validate().is_err());
/// assert!(QamModConfig { samples_per_symbol: 1, ..cfg }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QamModConfig {
    /// The constellation: `Qam16` or `Qam64`.
    pub constellation: SymbolMapConstellation,
    /// Output samples per symbol, 2 to 64 (required).
    pub samples_per_symbol: usize,
    /// Roll-off of the root-raised-cosine pulse (0.05 to 1).
    pub rolloff: f64,
    /// Length of the pulse in symbols (4 to 32).
    pub span: usize,
}

impl Default for QamModConfig {
    /// `qam16`, `samples_per_symbol` 8 (a placeholder: the pipeline requires the parameter),
    /// `rolloff` 0.35, `span` 10.
    fn default() -> Self {
        Self {
            constellation: SymbolMapConstellation::Qam16,
            samples_per_symbol: 8,
            rolloff: 0.35,
            span: 10,
        }
    }
}

impl QamModConfig {
    /// Bits per symbol: 4 for `qam16`, 6 for `qam64` (that of the constellation for another
    /// one, which [`QamModConfig::validate`] refuses).
    pub fn bits_per_symbol(&self) -> usize {
        self.constellation.bits_per_symbol()
    }

    /// Checks the ranges.
    ///
    /// # Errors
    /// `OutOfRange` for a value outside its range; `InvalidParameter` for a constellation other
    /// than `qam16` and `qam64`.
    pub fn validate(&self) -> Result<()> {
        qam_of(self.constellation)?;
        check_count("samples_per_symbol", self.samples_per_symbol, 2, MAX_SPS)?;
        check_f64("rolloff", self.rolloff, RRC_MIN_ROLLOFF, 1.0)?;
        check_count("span", self.span, RRC_MIN_SPAN, RRC_MAX_SPAN)
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`; `InvalidParameter` for a
    /// constellation other than `qam16` and `qam64`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "constellation" => self.constellation = qam_constellation_of_value(&value)?,
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
            "constellation" => Ok(ParamValue::Choice(self.constellation.index())),
            "samples_per_symbol" => Ok(ParamValue::I64(self.samples_per_symbol as i64)),
            "rolloff" => Ok(ParamValue::F64(self.rolloff)),
            "span" => Ok(ParamValue::I64(self.span as i64)),
            _ => unknown(),
        }
    }

    /// The pulse filter of the configuration.
    pub(crate) fn pulse(&self) -> RrcPulse {
        RrcPulse {
            samples_per_symbol: self.samples_per_symbol,
            rolloff: self.rolloff,
            span: self.span,
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

/// The flat parameter names of `qam_mod` and the building blocks they reach (M44.1.1 §4): `0` is
/// `symbol_map`, `1` the root-raised-cosine pulse filter (the interpolation with the taps of
/// [`QamModConfig`]'s `rolloff`, `span` and `samples_per_symbol`; the building block
/// `pulse_shape` of M44.1.1 §3 takes these parameters over).
pub const QAM_MOD_PARAMS: ParamMap = ParamMap::new(&[
    ("constellation", "0.constellation"),
    ("samples_per_symbol", "1.samples_per_symbol"),
    ("rolloff", "1.rolloff"),
    ("span", "1.span"),
]);

/// The root-raised-cosine pulse of the modulators `qam_mod` and `psk_mod`: its parameters and
/// the design of its taps (the stage `1` of their maps). The building block `pulse_shape` of
/// M44.1.1 §3 replaces it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RrcPulse {
    /// Output samples per symbol (2 to 64).
    pub(crate) samples_per_symbol: usize,
    /// Roll-off (0.05 to 1).
    pub(crate) rolloff: f64,
    /// Length in symbols (4 to 32).
    pub(crate) span: usize,
}

impl RrcPulse {
    /// The description of a parameter of the pulse.
    pub(crate) fn desc(name: &str) -> Option<&'static ParamDesc> {
        [
            &SAMPLES_PER_SYMBOL_PARAM,
            &MOD_ROLLOFF_PARAM,
            &MOD_SPAN_PARAM,
        ]
        .into_iter()
        .find(|p| p.name == name)
    }

    /// The current value of a parameter of the pulse.
    pub(crate) fn param(&self, name: &str) -> Result<ParamValue> {
        match name {
            "samples_per_symbol" => Ok(ParamValue::I64(self.samples_per_symbol as i64)),
            "rolloff" => Ok(ParamValue::F64(self.rolloff)),
            "span" => Ok(ParamValue::I64(self.span as i64)),
            _ => unknown(),
        }
    }

    /// Checks a value of a parameter of the pulse: all are build parameters.
    pub(crate) fn check_param(name: &str, value: &ParamValue) -> Result<()> {
        check_build_param(Self::desc(name).ok_or(Error::UnknownParameter)?, value)
    }

    /// The interpolation by `samples_per_symbol` with the `span sps + 1` taps
    /// `rrc(rolloff, sps, span) sqrt(sps)` in `f64`, rounded once to `f32` by
    /// [`InterpolateBlock::with_taps`].
    pub(crate) fn stage<S: Storage>(&self, storage: &mut S) -> Result<InterpolateBlock<C32, S>> {
        let sps = self.samples_per_symbol;
        let rrc = Rrc::new(self.rolloff, sps, self.span)?;
        let gain = libm::sqrt(sps as f64);
        InterpolateBlock::with_taps((0..rrc.len()).map(|k| rrc.tap(k) * gain), sps, storage)
    }
}

/// 16-QAM and 64-QAM modulator: `f32` soft bits in (one per sample), `C32` out at
/// `samples_per_symbol` samples per symbol.
///
/// A composite block (M44.1 §4): [`SymbolMapBlock`] (the `constellation` `qam16` or `qam64`, the `f32`
/// points of `sdrk_core::kernels::constellation::Qam`, M44 §1) then
/// [`InterpolateBlock::with_taps`] with the taps `rrc(rolloff, sps, span) sqrt(sps)` as `f32`
/// (the kernel `fir_interpolate` through the facade, as in `psk_mod`), as a [`Chain`] whose
/// buffer holds [`QAM_MOD_CHUNK`] symbols. A call takes `min(input / bits, output / sps)` whole
/// symbols. Timing: constant ratio `sps / bits`, delay `span bits / 2` input samples. Storage:
/// the taps, the filter history and the buffer of the chain. Its parameters are answered by
/// its building blocks through [`QAM_MOD_PARAMS`] (M44.1.1 §4).
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, ParamValue, QamModBlock, QamModConfig};
/// let cfg = QamModConfig { samples_per_symbol: 4, ..QamModConfig::default() };
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
/// assert_eq!(m.param("constellation")?, ParamValue::Choice(3)); // qam16, from symbol_map
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct QamModBlock<S: Storage> {
    pulse: RrcPulse,
    bps: usize,
    ratio: Ratio,
    stages: QamModStages<S>,
}

impl<S: Storage> core::fmt::Debug for QamModBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QamModBlock")
            .field("config", &self.config())
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
        let map = SymbolMapBlock::new(
            &SymbolMapConfig {
                constellation: config.constellation,
                ..SymbolMapConfig::default()
            },
            rate_in,
            storage,
        )?;
        let pulse = config.pulse();
        let interp = pulse.stage(storage)?;
        Ok(Self {
            pulse,
            bps: config.bits_per_symbol(),
            ratio: config.ratio()?,
            stages: Chain::new(map, interp, QAM_MOD_CHUNK, storage)?,
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

    /// The configuration of the block, from its building blocks.
    pub fn config(&self) -> QamModConfig {
        QamModConfig {
            constellation: self.stages.first().config().constellation,
            samples_per_symbol: self.pulse.samples_per_symbol,
            rolloff: self.pulse.rolloff,
            span: self.pulse.span,
        }
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
            self.pulse.samples_per_symbol,
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
            delay: Delay::new((self.pulse.span * self.bps) as u64, 2).unwrap_or(Delay::ZERO),
        }
    }

    fn reset(&mut self) {
        self.stages.reset();
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        composite_set_param(self, name, value)
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        composite_param(self, name)
    }

    fn check_param(&self, name: &str, value: &ParamValue) -> Result<()> {
        check_composite_param(self, name, value)
    }

    fn min_input(&self) -> usize {
        self.bps
    }
}

impl<S: Storage> Stages for QamModBlock<S> {
    const PARAMS: ParamMap = QAM_MOD_PARAMS;

    fn stage_param(&self, path: &str) -> Result<ParamValue> {
        match stage_of(&Self::PARAMS, path)? {
            ("0", n) => self.stages.first().param(n),
            ("1", n) => self.pulse.param(n),
            _ => unknown(),
        }
    }

    fn check_stage_param(&self, path: &str, value: &ParamValue) -> Result<()> {
        match stage_of(&Self::PARAMS, path)? {
            ("0", n) => self.stages.first().check_param(n, value),
            ("1", n) => RrcPulse::check_param(n, value),
            _ => unknown(),
        }
    }

    fn set_stage_param(&mut self, path: &str, value: ParamValue) -> Result<()> {
        match stage_of(&Self::PARAMS, path)? {
            ("0", n) => self.stages.first_mut().set_param(n, value),
            ("1", _) => Err(Error::ReadOnlyParameter),
            _ => unknown(),
        }
    }
}
