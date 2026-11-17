//! `pulse_shape`: root-raised-cosine pulse shaping of a transmitter (symbols in, samples out).

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::interp::{FirInterpState, FirInterpolate};
use sdrk_core::timing::Timing;

use super::costas::check_f64;
use super::{check_count, multiply_rate, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::{RRC_MAX_SPAN, RRC_MIN_ROLLOFF, RRC_MIN_SPAN, Rrc};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};

/// The largest `samples_per_symbol` of `pulse_shape` (and of the modulators `qam_mod` and
/// `psk_mod` that use it).
pub const PULSE_SHAPE_MAX_SPS: usize = 64;

/// The `samples_per_symbol` parameter of `pulse_shape`, `qam_mod`, `psk_mod` and
/// `qam_frame_encode`.
pub(crate) const SAMPLES_PER_SYMBOL_PARAM: ParamDesc = ParamDesc {
    name: "samples_per_symbol",
    unit: Unit::None,
    kind: ParamKind::I64 {
        min: 2,
        max: PULSE_SHAPE_MAX_SPS as i64,
        default: None,
    },
    typical: "8",
    mutability: Mutability::Build,
    description: "Output samples per symbol, 2 to 64. Required.",
};

/// The `rolloff` parameter of `pulse_shape`, `qam_mod`, `psk_mod` and `qam_frame_encode`.
pub(crate) const ROLLOFF_PARAM: ParamDesc = ParamDesc {
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
};

/// The `span` parameter of `pulse_shape`, `qam_mod`, `psk_mod` and `qam_frame_encode`.
pub(crate) const SPAN_PARAM: ParamDesc = ParamDesc {
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

/// Description of the `pulse_shape` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "pulse_shape",
    summary: "Shapes complex symbols with a root-raised-cosine pulse (interpolation by \
        samples_per_symbol).",
    details: "Interpolates the symbols by samples_per_symbol with the span samples_per_symbol \
        + 1 taps of the root-raised-cosine pulse (rolloff, span symbols, unit energy) scaled \
        by sqrt(samples_per_symbol), each rounded once to f32: every symbol gives \
        samples_per_symbol output samples, and the matched filter of pulse_match (or of \
        psk_demod) gives the symbols back at the symbol centres. It is the transmit filter of \
        qam_mod and psk_mod. The rate rises by samples_per_symbol; the group delay is span / 2 \
        symbols.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32],
        description: "The symbols, one sample per symbol.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32],
        description: "The shaped signal, samples_per_symbol samples per symbol.",
    }],
    params: &[SAMPLES_PER_SYMBOL_PARAM, ROLLOFF_PARAM, SPAN_PARAM],
    required: &["samples_per_symbol"],
};

/// Configuration of [`PulseShapeBlock`].
///
/// ```
/// use sdrk_blocks::PulseShapeConfig;
/// let cfg = PulseShapeConfig { samples_per_symbol: 4, ..PulseShapeConfig::default() };
/// assert!(cfg.validate().is_ok());
/// assert_eq!((cfg.rolloff, cfg.span), (0.35, 10));
/// assert!(PulseShapeConfig { samples_per_symbol: 1, ..cfg }.validate().is_err());
/// assert!(PulseShapeConfig { span: 33, ..cfg }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PulseShapeConfig {
    /// Output samples per symbol, 2 to 64 (required).
    pub samples_per_symbol: usize,
    /// Roll-off of the root-raised-cosine pulse (0.05 to 1).
    pub rolloff: f64,
    /// Length of the pulse in symbols (4 to 32).
    pub span: usize,
}

impl Default for PulseShapeConfig {
    /// `samples_per_symbol` 8 (a placeholder: the pipeline requires the parameter), `rolloff`
    /// 0.35, `span` 10.
    fn default() -> Self {
        Self {
            samples_per_symbol: 8,
            rolloff: 0.35,
            span: 10,
        }
    }
}

impl PulseShapeConfig {
    /// Checks the ranges.
    ///
    /// # Errors
    /// `OutOfRange` for a value outside its range.
    pub fn validate(&self) -> Result<()> {
        check_count(
            "samples_per_symbol",
            self.samples_per_symbol,
            2,
            PULSE_SHAPE_MAX_SPS,
        )?;
        check_f64("rolloff", self.rolloff, RRC_MIN_ROLLOFF, 1.0)?;
        check_count("span", self.span, RRC_MIN_SPAN, RRC_MAX_SPAN)
    }

    /// Number of taps: `span samples_per_symbol + 1`.
    pub fn num_taps(&self) -> usize {
        self.span * self.samples_per_symbol + 1
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
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
            "samples_per_symbol" => Ok(ParamValue::I64(self.samples_per_symbol as i64)),
            "rolloff" => Ok(ParamValue::F64(self.rolloff)),
            "span" => Ok(ParamValue::I64(self.span as i64)),
            _ => unknown(),
        }
    }
}

/// Root-raised-cosine pulse shaping: `C32` symbols in, `C32` samples out at
/// `samples_per_symbol` samples per symbol.
///
/// Designs the `span sps + 1` taps `rrc(rolloff, sps, span) sqrt(sps)` in `f64`
/// ([`design::rrc`](crate::design::rrc)) and rounds each once to `f32`; the kernel
/// `fir_interpolate` runs through the facade. A call fills as much of the output as the input
/// allows (the samples of a symbol may be split across calls). Timing: constant ratio `sps`, delay `span / 2` input symbols. Storage: the taps and the
/// filter history of `ceil((span sps + 1) / sps)` symbols. The building block of `qam_mod` and
/// `psk_mod` (M44.1.1 §3.1).
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, PulseShapeBlock, PulseShapeConfig};
/// let cfg = PulseShapeConfig { samples_per_symbol: 4, ..PulseShapeConfig::default() };
/// let rate = Rate::hz(2400)?;
/// let bytes = PulseShapeBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut b = PulseShapeBlock::new(&cfg, rate, &mut arena)?;
/// let mut out = [C32::default(); 16];
/// let p = b.process(&[C32::new(1.0, 0.0); 5], &mut out)?;
/// assert_eq!((p.consumed, p.produced), (4, 16)); // the fifth symbol does not fit
/// assert_eq!(b.num_taps(), 41);
/// assert_eq!(PulseShapeBlock::<ArenaStorage<'_>>::rate_out(&cfg, rate)?, Rate::hz(9600)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct PulseShapeBlock<S: Storage> {
    fir: FirInterpState<C32, S::Buf<f32>, S::Buf<C32>>,
    rolloff: f64,
}

impl<S: Storage> core::fmt::Debug for PulseShapeBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PulseShapeBlock")
            .field("config", &self.config())
            .finish_non_exhaustive()
    }
}

impl<S: Storage> PulseShapeBlock<S> {
    /// Designs the pulse and creates the block; taps and history come from `storage`, in this
    /// order.
    ///
    /// # Errors
    /// As [`PulseShapeConfig::validate`]; `Storage` when the storage is too small.
    pub fn new(config: &PulseShapeConfig, _rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate()?;
        let sps = config.samples_per_symbol;
        let rrc = Rrc::new(config.rolloff, sps, config.span)?;
        let gain = libm::sqrt(sps as f64);
        let n = rrc.len();
        let mut taps = storage.alloc::<f32>(n)?;
        for (k, t) in taps.as_mut().iter_mut().enumerate() {
            *t = (rrc.tap(k) * gain) as f32;
        }
        let hist = storage.alloc::<C32>(n.div_ceil(sps))?;
        Ok(Self {
            fir: FirInterpState::new(taps, hist, sps)?,
            rolloff: config.rolloff,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// # Errors
    /// As [`PulseShapeConfig::validate`].
    pub fn storage_bytes(config: &PulseShapeConfig, _rate_in: Rate) -> Result<usize> {
        let mut plan = Plan::default();
        Self::plan(config, &mut plan)?;
        Ok(plan.bytes())
    }

    /// Adds the buffers of the block to `plan`, in the order `new` takes them.
    pub(crate) fn plan(config: &PulseShapeConfig, plan: &mut Plan) -> Result<()> {
        config.validate()?;
        let n = config.num_taps();
        plan.add::<f32>(n)?;
        plan.add::<C32>(n.div_ceil(config.samples_per_symbol))
    }

    /// Output rate: the symbol rate times `samples_per_symbol`.
    ///
    /// # Errors
    /// As [`PulseShapeConfig::validate`]; `OutOfRange` when the rate overflows.
    pub fn rate_out(config: &PulseShapeConfig, rate_in: Rate) -> Result<Rate> {
        config.validate()?;
        multiply_rate(rate_in, config.samples_per_symbol)
    }

    /// The configuration of the block.
    pub fn config(&self) -> PulseShapeConfig {
        let sps = self.samples_per_symbol();
        PulseShapeConfig {
            samples_per_symbol: sps,
            rolloff: self.rolloff,
            span: (self.num_taps() - 1) / sps.max(1),
        }
    }

    /// Output samples per symbol.
    pub fn samples_per_symbol(&self) -> usize {
        self.fir.factor()
    }

    /// Number of taps of the pulse (`span samples_per_symbol + 1`).
    pub fn num_taps(&self) -> usize {
        self.fir.taps().len()
    }

    /// The taps of the pulse as `f32`, for tests.
    pub fn taps(&self) -> &[f32] {
        self.fir.taps()
    }
}

impl<S: Storage> Block for PulseShapeBlock<S> {
    type In = C32;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[C32], output: &mut [C32]) -> Result<Progress> {
        Ok(<Sdrk as FirInterpolate<C32>>::fir_interpolate(
            &Sdrk,
            &mut self.fir,
            input,
            output,
        )?)
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: self.fir.ratio(),
            kind: self.fir.ratio_kind(),
            delay: self.fir.group_delay(),
        }
    }

    fn reset(&mut self) {
        self.fir.reset();
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
