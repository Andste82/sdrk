//! `psk_mod`: BPSK, QPSK and 8PSK modulator (soft bits to root-raised-cosine shaped symbols),
//! coherent or differential, with the QPSK variants pi/4-DQPSK and offset QPSK.

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_core::C32;

use super::costas::check_f64;
use super::interpolate::InterpolateBlock;
use super::iq_offset::{IqOffsetBlock, IqOffsetConfig, IqPart};
use super::qam_mod::{RrcPulse, modulate};
use super::symbol_map::{
    CONSTELLATION_OPTIONS, SymbolMapBlock, SymbolMapConfig, SymbolMapConstellation,
    SymbolMapVariant,
};
use super::unknown;
use super::{check_count, head_mut};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::chain::Chain;
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::{RRC_MAX_SPAN, RRC_MIN_ROLLOFF, RRC_MIN_SPAN};
use crate::error::{Error, Result};
use crate::param_map::{
    ParamMap, Stages, check_composite_param, composite_param, composite_set_param, stage_of,
};
use crate::storage::{Plan, Storage};

/// The capacity in symbols of the buffer between `symbol_map` and the interpolation (and with
/// `offset` in samples of the buffer between the interpolation and `iq_offset`).
pub const PSK_MOD_CHUNK: usize = 256;

/// The largest `samples_per_symbol` of `psk_mod`.
const MAX_SPS: usize = 64;

/// The names of the `variant` choice, in index order.
const VARIANT_OPTIONS: &[&str] = &["plain", "pi4", "offset"];

/// The variant of the PSK blocks `psk_mod` and `psk_demod` (M40.1).
///
/// ```
/// use sdrk_blocks::{PskModConfig, PskVariant, SymbolMapConstellation};
/// let pi4 = PskModConfig {
///     constellation: SymbolMapConstellation::Qpsk,
///     differential: true,
///     variant: PskVariant::Pi4,
///     ..PskModConfig::default()
/// };
/// assert!(pi4.validate().is_ok());
/// // pi/4-DQPSK is differential QPSK only
/// assert!(PskModConfig { differential: false, ..pi4 }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum PskVariant {
    /// The plain constellation (BPSK, QPSK or 8PSK, coherent or differential).
    #[default]
    Plain,
    /// pi/4-DQPSK: `qpsk` and differential only; the phase steps by `+pi/4` (bits `00`),
    /// `+3 pi/4` (`01`), `-3 pi/4` (`11`) or `-pi/4` (`10`), bit 0 first.
    Pi4,
    /// Offset QPSK (OQPSK): `qpsk`, not differential, an even `samples_per_symbol`; the
    /// transmitter delays the quadrature part by half a symbol, the receiver the in-phase part,
    /// after derotating the signal sample by sample with its own carrier loop (D686).
    Offset,
}

impl PskVariant {
    pub(crate) const fn index(self) -> u16 {
        match self {
            Self::Plain => 0,
            Self::Pi4 => 1,
            Self::Offset => 2,
        }
    }

    pub(crate) const fn from_index(i: u16) -> Self {
        match i {
            1 => Self::Pi4,
            2 => Self::Offset,
            _ => Self::Plain,
        }
    }
}

/// The description of the `variant` parameter (shared by `psk_mod` and `psk_demod`).
pub(crate) const VARIANT_PARAM: ParamDesc = ParamDesc {
    name: "variant",
    unit: Unit::None,
    kind: ParamKind::Choice {
        options: VARIANT_OPTIONS,
        default: 0,
    },
    typical: "plain; pi4 for pi/4-DQPSK (TETRA), offset for OQPSK (IEEE 802.15.4)",
    mutability: Mutability::Build,
    description: "plain, pi4 (pi/4-DQPSK: needs qpsk and differential) or offset (OQPSK: \
        needs qpsk, not differential and an even number of samples per symbol).",
};

/// The `constellation` parameter of `psk_mod` and `psk_demod`: the choice of `symbol_map` (the
/// same names at the same indices in `costas`, `equalize` and `symbol_demap`), of which these
/// blocks take `bpsk`, `qpsk` and `8psk` (M44.1.1 §2).
pub(crate) const PSK_CONSTELLATION_PARAM: ParamDesc = ParamDesc {
    name: "constellation",
    unit: Unit::None,
    kind: ParamKind::Choice {
        options: CONSTELLATION_OPTIONS,
        default: 0,
    },
    typical: "bpsk, qpsk or 8psk (qam16 and qam64 are refused: qam_mod)",
    mutability: Mutability::Build,
    description: "The constellation: bpsk (1 bit per symbol), qpsk (2 bits per symbol) or 8psk \
        (3 bits per symbol); qam16 and qam64 are those of qam_mod and qam_demod.",
};

/// A phase-shift keying (`bpsk`, `qpsk` or `8psk`), otherwise `InvalidParameter` of
/// `constellation` (`qam16` and `qam64` are in the choice but not PSK).
pub(crate) fn check_psk(c: SymbolMapConstellation) -> Result<()> {
    if c.is_psk() {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name: "constellation",
            reason: "the constellation must be bpsk, qpsk or 8psk",
        })
    }
}

/// The PSK constellation of a parameter value (its range checked by the description).
pub(crate) fn psk_constellation_of_value(value: &ParamValue) -> Result<SymbolMapConstellation> {
    let c = SymbolMapConstellation::from_index(value.as_choice()?);
    check_psk(c)?;
    Ok(c)
}

/// The combination rules of `variant` (M40.1 §2 to §4): `pi4` needs `qpsk` and
/// `differential`, `offset` needs `qpsk`, not `differential` and `even_sps`; any other
/// variant than `plain` needs `qpsk`. Otherwise `InvalidParameter` of `variant`.
pub(crate) fn check_variant(
    variant: PskVariant,
    constellation: SymbolMapConstellation,
    differential: bool,
    even_sps: bool,
) -> Result<()> {
    let qpsk = constellation == SymbolMapConstellation::Qpsk;
    let reason = match variant {
        PskVariant::Plain => return Ok(()),
        PskVariant::Pi4 if qpsk && differential => return Ok(()),
        PskVariant::Pi4 => "pi4 needs qpsk and differential",
        PskVariant::Offset if qpsk && !differential && even_sps => return Ok(()),
        PskVariant::Offset => {
            "offset needs qpsk, not differential and an even number of samples per symbol"
        }
    };
    Err(Error::InvalidParameter {
        name: "variant",
        reason,
    })
}

/// Description of the `psk_mod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "psk_mod",
    summary: "Turns soft bits into BPSK, QPSK or 8PSK symbols shaped with a root-raised-cosine \
        pulse.",
    details: "Reads one soft bit per input sample (above 0 is a 1, anything else, NaN \
        included, a 0; frame_encode produces them) and maps 1 (constellation bpsk), 2 (qpsk) \
        or 3 (8psk) bits to a symbol with the Gray mapping of symbol_map and symbol_demap: BPSK \
        (2 b - 1, 0), QPSK ((2 b0 - 1) / sqrt(2), (2 b1 - 1) / sqrt(2)), so bit 1 is positive; \
        8PSK exp(j (2 pi k / 8 + pi / 8)) with k from the Gray code b0 b1 b2 (000 -> 0, 001 -> 1, \
        011 -> 2, 010 -> 3, 110 -> 4, 111 -> 5, 101 -> 6, 100 -> 7). With differential set, \
        the phase of a symbol is the phase of the previous symbol plus the phase of the mapped \
        symbol (start phase 0), which psk_demod with differential undoes without a carrier \
        loop. The variant pi4 (qpsk, differential) is pi/4-DQPSK: the phase steps by +pi/4 \
        (bits 00), +3 pi/4 (01), -3 pi/4 (11) or -pi/4 (10). The variant offset (qpsk, not \
        differential, even samples_per_symbol) is OQPSK: the quadrature part is delayed by \
        samples_per_symbol / 2 output samples after the pulse shaping, which keeps the \
        envelope away from zero. The symbols are interpolated by samples_per_symbol with the root-raised-cosine \
        pulse (rolloff, span symbols) scaled by sqrt(samples_per_symbol), so the matched \
        filter of psk_demod gives amplitude 1 at the symbol centres. The output rate is the \
        bit rate times samples_per_symbol / bits per symbol; the group delay is span / 2 \
        symbols (offset: plus half a symbol). A call takes whole symbols only. Follow it with duc to reach the radio rate.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32],
        description: "Soft bits, one sample per bit; above 0 is a 1.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32],
        description: "The shaped baseband signal, samples_per_symbol samples per symbol, peak \
            amplitude about 1.",
    }],
    params: &[
        PSK_CONSTELLATION_PARAM,
        ParamDesc {
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
        },
        ParamDesc {
            name: "rolloff",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: RRC_MIN_ROLLOFF,
                max: 1.0,
                default: Some(0.35),
            },
            typical: "0.35; 0.2 for a narrow spectrum, 0.5 to 1 for easy timing recovery",
            mutability: Mutability::Build,
            description: "Roll-off of the root-raised-cosine pulse.",
        },
        ParamDesc {
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
        },
        ParamDesc {
            name: "differential",
            unit: Unit::None,
            kind: ParamKind::Bool { default: false },
            typical: "false; true for a receiver without carrier recovery",
            mutability: Mutability::Build,
            description: "Differential coding: each symbol adds its phase to the previous one.",
        },
        VARIANT_PARAM,
    ],
    required: &["samples_per_symbol"],
};

/// Configuration of [`PskModBlock`].
///
/// ```
/// use sdrk_blocks::{PskModConfig, PskVariant, SymbolMapConstellation};
/// let cfg = PskModConfig {
///     constellation: SymbolMapConstellation::Qpsk,
///     samples_per_symbol: 8,
///     ..PskModConfig::default()
/// };
/// assert!(cfg.validate().is_ok());
/// assert_eq!(cfg.bits_per_symbol(), 2);
/// assert!(PskModConfig { samples_per_symbol: 1, ..cfg }.validate().is_err());
/// let qam = PskModConfig { constellation: SymbolMapConstellation::Qam16, ..cfg };
/// assert!(qam.validate().is_err());
/// let psk8 = PskModConfig { constellation: SymbolMapConstellation::Psk8, ..cfg };
/// assert_eq!(psk8.bits_per_symbol(), 3);
/// // the variants other than plain are QPSK only
/// assert!(PskModConfig { variant: PskVariant::Offset, ..psk8 }.validate().is_err());
/// assert!(PskModConfig { variant: PskVariant::Offset, ..cfg }.validate().is_ok());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PskModConfig {
    /// The constellation: `Bpsk`, `Qpsk` or `Psk8`.
    pub constellation: SymbolMapConstellation,
    /// Output samples per symbol, 2 to 64 (required).
    pub samples_per_symbol: usize,
    /// Roll-off of the root-raised-cosine pulse (0.05 to 1).
    pub rolloff: f64,
    /// Length of the pulse in symbols (4 to 32).
    pub span: usize,
    /// Differential coding.
    pub differential: bool,
    /// The variant: plain, pi/4-DQPSK or offset QPSK (M40.1).
    pub variant: PskVariant,
}

impl Default for PskModConfig {
    /// `bpsk`, `samples_per_symbol` 8 (a placeholder: the pipeline requires the parameter),
    /// `rolloff` 0.35, `span` 10, not differential, variant plain.
    fn default() -> Self {
        Self {
            constellation: SymbolMapConstellation::Bpsk,
            samples_per_symbol: 8,
            rolloff: 0.35,
            span: 10,
            differential: false,
            variant: PskVariant::Plain,
        }
    }
}

impl PskModConfig {
    /// Bits per symbol: 1 for `bpsk`, 2 for `qpsk`, 3 for `8psk`.
    pub fn bits_per_symbol(&self) -> usize {
        self.constellation.bits_per_symbol()
    }

    /// Checks the ranges and the combination of `variant` with `constellation`, `differential`
    /// and `samples_per_symbol`.
    ///
    /// # Errors
    /// `OutOfRange` for a value outside its range; `InvalidParameter` for `qam16` and `qam64`
    /// and for a `variant` that does not fit (`pi4` without `qpsk` and `differential`, `offset`
    /// without `qpsk`, with `differential` or with an odd `samples_per_symbol`).
    pub fn validate(&self) -> Result<()> {
        check_psk(self.constellation)?;
        check_count("samples_per_symbol", self.samples_per_symbol, 2, MAX_SPS)?;
        check_f64("rolloff", self.rolloff, RRC_MIN_ROLLOFF, 1.0)?;
        check_count("span", self.span, RRC_MIN_SPAN, RRC_MAX_SPAN)?;
        check_variant(
            self.variant,
            self.constellation,
            self.differential,
            self.samples_per_symbol.is_multiple_of(2),
        )
    }

    /// Sets a parameter by its description name; checks kind and range (the combination of
    /// the parameters is checked by [`PskModConfig::validate`]).
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`; `InvalidParameter` for `qam16` and
    /// `qam64`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "constellation" => self.constellation = psk_constellation_of_value(&value)?,
            "samples_per_symbol" => self.samples_per_symbol = value.as_usize()?,
            "rolloff" => self.rolloff = value.as_f64()?,
            "span" => self.span = value.as_usize()?,
            "differential" => self.differential = value.as_bool()?,
            "variant" => self.variant = PskVariant::from_index(value.as_choice()?),
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
            "differential" => Ok(ParamValue::Bool(self.differential)),
            "variant" => Ok(ParamValue::Choice(self.variant.index())),
            _ => unknown(),
        }
    }

    /// The configuration of the `symbol_map` stage: `constellation`, `differential` and the
    /// variant `pi4` (`offset` acts after the pulse filter and maps as `plain`).
    fn map_config(&self) -> SymbolMapConfig {
        SymbolMapConfig {
            constellation: self.constellation,
            differential: self.differential,
            variant: if self.variant == PskVariant::Pi4 {
                SymbolMapVariant::Pi4
            } else {
                SymbolMapVariant::Plain
            },
        }
    }

    /// The pulse filter of the configuration.
    fn pulse(&self) -> RrcPulse {
        RrcPulse {
            samples_per_symbol: self.samples_per_symbol,
            rolloff: self.rolloff,
            span: self.span,
        }
    }

    /// The half symbol of the `offset` delay in output samples (0 for the other variants).
    fn half(&self) -> usize {
        if self.variant == PskVariant::Offset {
            self.samples_per_symbol / 2
        } else {
            0
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

/// The stages of `psk_mod`: `symbol_map` then the interpolation.
type PskModStages<S> = Chain<SymbolMapBlock<S>, InterpolateBlock<C32, S>, S>;

/// BPSK, QPSK and 8PSK modulator: `f32` soft bits in (one per sample), `C32` out at
/// `samples_per_symbol` samples per symbol.
///
/// A composite block (M44.1 §4): [`SymbolMapBlock`] (the `constellation` `bpsk`, `qpsk` or `8psk`,
/// with `differential` and the variant `plain` or `pi4`: one of the sixteen `f32` points at
/// multiples of 22.5 degrees, the differential phase in units of `pi / 8`) then
/// [`InterpolateBlock::with_taps`] with the taps `rrc(rolloff, sps, span) sqrt(sps)` as `f32`
/// (the kernel `fir_interpolate` through the facade), as a [`Chain`] whose buffer holds
/// [`PSK_MOD_CHUNK`] symbols; with the variant `offset` the output of the chain then passes
/// [`IqOffsetBlock`] in place with `samples` `sps / 2` (the imaginary part delayed through a
/// ring of `sps / 2` `f32`, zeros at the start). A call takes `min(input / bits, output / sps)`
/// whole symbols. Timing: constant ratio `sps / bits`, delay `span bits / 2` input samples
/// (`offset`: `(span bits + 2) / 2`, the half symbol of the quadrature part is one bit).
/// Storage: the taps, the filter history, the buffer of the chain and for `offset` the ring.
/// Its parameters are answered by its building blocks through [`PSK_MOD_PARAMS`] (M44.1.1 §4;
/// `variant` `offset` is the presence of the `iq_offset` stage, open item I44.1.1-1).
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, PskModBlock, PskModConfig};
/// use sdrk_blocks::SymbolMapConstellation;
/// let cfg = PskModConfig {
///     constellation: SymbolMapConstellation::Qpsk,
///     samples_per_symbol: 4,
///     ..PskModConfig::default()
/// };
/// let rate = Rate::hz(9600)?;
/// let bytes = PskModBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut m = PskModBlock::new(&cfg, rate, &mut arena)?;
/// let bits = [1.0f32, -1.0, -1.0, 1.0, 1.0];
/// let mut out = [C32::default(); 64];
/// let p = m.process(&bits, &mut out)?;
/// assert_eq!((p.consumed, p.produced), (4, 8)); // two symbols; the odd bit waits
/// assert_eq!(PskModBlock::<ArenaStorage<'_>>::rate_out(&cfg, rate)?, Rate::hz(19_200)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct PskModBlock<S: Storage> {
    pulse: RrcPulse,
    bps: usize,
    stages: PskModStages<S>,
    /// The delay of the quadrature part for `offset`; `None` for the other variants.
    offset: Option<IqOffsetBlock<S>>,
}

impl<S: Storage> core::fmt::Debug for PskModBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PskModBlock")
            .field("config", &self.config())
            .finish_non_exhaustive()
    }
}

impl<S: Storage> PskModBlock<S> {
    /// Designs the pulse and creates the block; taps, history, the buffer of the chain and the
    /// ring of `offset` come from `storage`.
    ///
    /// # Errors
    /// As [`PskModConfig::validate`]; `Storage` when the storage is too small.
    pub fn new(config: &PskModConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate()?;
        let map = SymbolMapBlock::new(&config.map_config(), rate_in, storage)?;
        let pulse = config.pulse();
        let interp = pulse.stage(storage)?;
        let stages = Chain::new(map, interp, PSK_MOD_CHUNK, storage)?;
        let half = config.half();
        let offset = if half > 0 {
            Some(IqOffsetBlock::new(
                &IqOffsetConfig {
                    samples: half,
                    part: IqPart::Imag,
                },
                rate_in,
                storage,
            )?)
        } else {
            None
        };
        Ok(Self {
            pulse,
            bps: config.bits_per_symbol(),
            stages,
            offset,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// # Errors
    /// As [`PskModConfig::validate`].
    pub fn storage_bytes(config: &PskModConfig, _rate_in: Rate) -> Result<usize> {
        config.validate()?;
        let sps = config.samples_per_symbol;
        let n = config.span * sps + 1;
        let mut plan = Plan::default();
        plan.add::<f32>(n)?;
        plan.add::<C32>(n.div_ceil(sps))?;
        plan.add::<C32>(PSK_MOD_CHUNK)?;
        let half = config.half();
        if half > 0 {
            plan.add::<f32>(half)?;
        }
        Ok(plan.bytes())
    }

    /// Output rate: the bit rate times `samples_per_symbol / bits_per_symbol`.
    ///
    /// # Errors
    /// As [`PskModConfig::validate`]; `OutOfRange` when the rate overflows.
    pub fn rate_out(config: &PskModConfig, rate_in: Rate) -> Result<Rate> {
        config.validate()?;
        Ok(rate_in.scale(config.ratio()?)?)
    }

    /// Number of taps of the pulse (`span samples_per_symbol + 1`).
    pub fn num_taps(&self) -> usize {
        self.stages.second().num_taps()
    }

    /// The configuration of the block, from its building blocks.
    pub fn config(&self) -> PskModConfig {
        let map = self.stages.first().config();
        PskModConfig {
            constellation: map.constellation,
            samples_per_symbol: self.pulse.samples_per_symbol,
            rolloff: self.pulse.rolloff,
            span: self.pulse.span,
            differential: map.differential,
            variant: self.variant(),
        }
    }

    /// The variant: `offset` where the `iq_offset` stage is there, otherwise that of
    /// `symbol_map`.
    fn variant(&self) -> PskVariant {
        if self.offset.is_some() {
            PskVariant::Offset
        } else if self.stages.first().config().variant == SymbolMapVariant::Pi4 {
            PskVariant::Pi4
        } else {
            PskVariant::Plain
        }
    }
}

/// The flat parameter names of `psk_mod` and the building blocks they reach (M44.1.1 §4): `0`
/// is `symbol_map`, `1` the root-raised-cosine pulse filter (as in [`QAM_MOD_PARAMS`](crate::QAM_MOD_PARAMS)).
/// `variant` reaches the `variant` of `symbol_map` (`plain`, `pi4`); `offset` is not a value of
/// any building block but the `iq_offset` stage after the pulse filter, which the block reports
/// as the variant `offset` (open item I44.1.1-1).
pub const PSK_MOD_PARAMS: ParamMap = ParamMap::new(&[
    ("constellation", "0.constellation"),
    ("samples_per_symbol", "1.samples_per_symbol"),
    ("rolloff", "1.rolloff"),
    ("span", "1.span"),
    ("differential", "0.differential"),
    ("variant", "0.variant"),
]);

impl<S: Storage> Stages for PskModBlock<S> {
    const PARAMS: ParamMap = PSK_MOD_PARAMS;

    fn stage_param(&self, path: &str) -> Result<ParamValue> {
        match stage_of(&Self::PARAMS, path)? {
            ("0", "variant") => Ok(ParamValue::Choice(self.variant().index())),
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

impl<S: Storage> Block for PskModBlock<S> {
    type In = f32;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[f32], output: &mut [C32]) -> Result<Progress> {
        let p = modulate(
            &mut self.stages,
            self.bps,
            self.pulse.samples_per_symbol,
            input,
            output,
            "psk_mod stages did not take the whole piece",
        )?;
        if let Some(offset) = self.offset.as_mut() {
            offset.process_in_place(head_mut(output, p.produced)?);
        }
        Ok(p)
    }

    fn timing(&self) -> BlockTiming {
        // The figures of the block of M40.1, as it reported them: the delay of the chain plus,
        // with `offset`, the half symbol (one bit) of the quadrature part, which `iq_offset`
        // does not report.
        let bits = self.bps;
        let offset = usize::from(self.offset.is_some());
        BlockTiming {
            ratio: Ratio::new(self.pulse.samples_per_symbol as u64, bits as u64)
                .unwrap_or(Ratio::ONE),
            kind: RatioKind::Constant,
            delay: Delay::new((self.pulse.span * bits + 2 * offset) as u64, 2)
                .unwrap_or(Delay::ZERO),
        }
    }

    fn reset(&mut self) {
        self.stages.reset();
        if let Some(offset) = self.offset.as_mut() {
            offset.reset();
        }
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
