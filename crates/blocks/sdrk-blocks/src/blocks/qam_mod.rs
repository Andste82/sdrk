//! `qam_mod`: square QAM modulator, 4-QAM (`qpsk`) to 1024-QAM (soft bits to root-raised-cosine
//! shaped symbols).

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_core::C32;
use sdrk_core::kernels::constellation::Qam;

use super::pulse_shape::{
    PulseShapeBlock, PulseShapeConfig, ROLLOFF_PARAM, SAMPLES_PER_SYMBOL_PARAM, SPAN_PARAM,
};
use super::symbol_map::{
    CONSTELLATION_OPTIONS, SymbolMapBlock, SymbolMapConfig, SymbolMapConstellation,
};
use super::{head, head_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::chain::Chain;
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::param_map::ParamMap;
use crate::storage::{Plan, Storage};

/// The capacity in symbols of the buffer between `symbol_map` and the interpolation.
pub const QAM_MOD_CHUNK: usize = 256;

/// The `constellation` parameter of `qam_mod`, `qam_demod`, `qam_frame_encode` and
/// `qam_frame_decode`: the choice of `symbol_map` (the same names at the same indices in
/// `costas`, `equalize` and `symbol_demap`), of which these blocks take the square orders `qpsk`
/// (4-QAM), `qam16`, `qam64`, `qam256` and `qam1024` (M44.1.1 §2, M44.2 §4).
pub(crate) const QAM_CONSTELLATION_PARAM: ParamDesc = ParamDesc {
    name: "constellation",
    unit: Unit::None,
    kind: ParamKind::Choice {
        options: CONSTELLATION_OPTIONS,
        default: 3,
    },
    typical: "qam16; qam64 to qam1024 for a good SNR, qpsk for a weak one (bpsk and 8psk are \
        refused: psk_mod)",
    mutability: Mutability::Build,
    description: "The square constellation: qpsk (4-QAM, 2 bits per symbol), qam16 (4 bits), \
        qam64 (6 bits), qam256 (8 bits) or qam1024 (10 bits); bpsk and 8psk are those of \
        psk_mod and psk_demod.",
};

/// The square QAM of a constellation: `qpsk` (4-QAM, M44.2), `qam16`, `qam64`, `qam256` or
/// `qam1024`, otherwise `InvalidParameter` of `constellation` (`bpsk` and `8psk` are in the
/// choice but not square QAM).
pub(crate) fn qam_of(c: SymbolMapConstellation) -> Result<Qam> {
    match c {
        SymbolMapConstellation::Qpsk => Ok(Qam::Qpsk),
        SymbolMapConstellation::Qam16 => Ok(Qam::Qam16),
        SymbolMapConstellation::Qam64 => Ok(Qam::Qam64),
        SymbolMapConstellation::Qam256 => Ok(Qam::Qam256),
        SymbolMapConstellation::Qam1024 => Ok(Qam::Qam1024),
        _ => Err(Error::InvalidParameter {
            name: "constellation",
            reason: "the constellation must be qpsk, qam16, qam64, qam256 or qam1024",
        }),
    }
}

/// The QAM constellation of a parameter value (its range checked by the description).
pub(crate) fn qam_constellation_of_value(value: &ParamValue) -> Result<SymbolMapConstellation> {
    let c = SymbolMapConstellation::from_index(value.as_choice()?);
    qam_of(c)?;
    Ok(c)
}

/// Description of the `qam_mod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "qam_mod",
    summary: "Turns soft bits into square QAM symbols (4-QAM to 1024-QAM) shaped with a \
        root-raised-cosine pulse.",
    details: "Reads one soft bit per input sample (above 0 is a 1, anything else, NaN \
        included, a 0; frame_encode produces them) and maps 2 (qpsk, 4-QAM), 4 (qam16), 6 \
        (qam64), 8 (qam256) or 10 (qam1024) bits, \
        the first one the most significant, to a point of the square constellation: the first \
        half of the bits selects the in-phase level, the second half the quadrature level, \
        each by the Gray code of the level index i (i XOR (i >> 1)); the level value is \
        2 i - (m - 1) with m = 2, 4, 8, 16 or 32 levels per axis, scaled by 1 / sqrt(E) with \
        E = 2, 10, 42, 170 or 682 (unit average power; qpsk is the QPSK of psk_mod point for \
        point). These are the constellations of symbol_map, symbol_demap, equalize and costas. The symbols are interpolated by samples_per_symbol with the \
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
        ROLLOFF_PARAM,
        SPAN_PARAM,
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
/// // 4-QAM is `qpsk`; `bpsk` and `8psk` are not square QAM
/// let qpsk = QamModConfig { constellation: SymbolMapConstellation::Qpsk, ..cfg };
/// assert_eq!(qpsk.bits_per_symbol(), 2);
/// assert!(qpsk.validate().is_ok());
/// let bpsk = QamModConfig { constellation: SymbolMapConstellation::Bpsk, ..cfg };
/// assert!(bpsk.validate().is_err());
/// assert!(QamModConfig { samples_per_symbol: 1, ..cfg }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QamModConfig {
    /// The constellation: `Qpsk` (4-QAM), `Qam16`, `Qam64`, `Qam256` or `Qam1024`.
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
    /// Bits per symbol: 2 for `qpsk`, 4 for `qam16`, 6 for `qam64`, 8 for `qam256`, 10 for
    /// `qam1024` (that of the constellation for another one, which [`QamModConfig::validate`]
    /// refuses).
    pub fn bits_per_symbol(&self) -> usize {
        self.constellation.bits_per_symbol()
    }

    /// Checks the ranges.
    ///
    /// # Errors
    /// `OutOfRange` for a value outside its range; `InvalidParameter` for a constellation that
    /// is not a square QAM (`bpsk`, `8psk`).
    pub fn validate(&self) -> Result<()> {
        qam_of(self.constellation)?;
        self.pulse().validate()
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`; `InvalidParameter` for a
    /// constellation that is not a square QAM (`bpsk`, `8psk`).
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

    /// The configuration of the building block `pulse_shape`.
    pub fn pulse(&self) -> PulseShapeConfig {
        PulseShapeConfig {
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

/// The stages of `qam_mod`: `symbol_map` then `pulse_shape`.
type QamModStages<S> = Chain<SymbolMapBlock<S>, PulseShapeBlock<S>, S>;

/// The flat parameter names of `qam_mod` and the building blocks they reach (M44.1.1 §4): `0` is
/// `symbol_map`, `1` `pulse_shape`.
pub const QAM_MOD_PARAMS: ParamMap = ParamMap::new(&[
    ("constellation", "0.constellation"),
    ("samples_per_symbol", "1.samples_per_symbol"),
    ("rolloff", "1.rolloff"),
    ("span", "1.span"),
]);

/// Square QAM modulator (4-QAM to 1024-QAM): `f32` soft bits in (one per sample), `C32` out at
/// `samples_per_symbol` samples per symbol.
///
/// A composite block (M44.1 §4, M44.1.1 §3.3): [`SymbolMapBlock`] (the `constellation` `qpsk`,
/// `qam16`, `qam64`, `qam256` or `qam1024`, the `f32` points of
/// `sdrk_core::kernels::constellation::Qam`, M44 §1, M44.2 §1) then
/// [`PulseShapeBlock`] (the taps `rrc(rolloff, sps, span) sqrt(sps)` as `f32`, the kernel
/// `fir_interpolate` through the facade, as in `psk_mod`), as a [`Chain`] whose buffer holds
/// [`QAM_MOD_CHUNK`] symbols. A call takes `min(input / bits, output / sps)` whole
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
    bps: usize,
    ratio: Ratio,
    pub(super) stages: QamModStages<S>,
}

impl<S: Storage> core::fmt::Debug for QamModBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QamModBlock")
            .field("config", &self.config())
            .finish_non_exhaustive()
    }
}

impl<S: Storage> QamModBlock<S> {
    /// Creates the block of its building blocks; the taps and history of `pulse_shape` and the
    /// buffer of the chain come from `storage`.
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
        let pulse = PulseShapeBlock::new(&config.pulse(), rate_in, storage)?;
        Ok(Self {
            bps: config.bits_per_symbol(),
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
        PulseShapeBlock::<S>::plan(&config.pulse(), plan)?;
        plan.add::<C32>(QAM_MOD_CHUNK)
    }

    /// The configuration of the block, from its building blocks.
    pub fn config(&self) -> QamModConfig {
        let pulse = self.stages.second().config();
        QamModConfig {
            constellation: self.stages.first().config().constellation,
            samples_per_symbol: pulse.samples_per_symbol,
            rolloff: pulse.rolloff,
            span: pulse.span,
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
        let sps = self.stages.second().samples_per_symbol();
        modulate(
            &mut self.stages,
            self.bps,
            sps,
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
            delay: Delay::new((self.config().span * self.bps) as u64, 2).unwrap_or(Delay::ZERO),
        }
    }

    fn reset(&mut self) {
        self.stages.reset();
    }

    // Through the map alone: the chain routes its paths to `symbol_map` and `pulse_shape`.
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        QAM_MOD_PARAMS.set(&mut self.stages, name, value)
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        QAM_MOD_PARAMS.get(&self.stages, name)
    }

    fn check_param(&self, name: &str, value: &ParamValue) -> Result<()> {
        QAM_MOD_PARAMS.check(&self.stages, name, value)
    }

    fn min_input(&self) -> usize {
        self.bps
    }
}
