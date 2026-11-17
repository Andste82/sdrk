//! `pulse_match`: the root-raised-cosine matched filter of a receiver, at the input rate.

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::timing::Timing;

use super::costas::check_f64;
use super::psk_demod::{MIN_SYMBOL_RATE_HZ, SPS_TOLERANCE};
use super::{HZ_LIMIT, check_count, head, head_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::{RRC_MAX_SPAN, RRC_MAX_SPS, RRC_MIN_ROLLOFF, RRC_MIN_SPAN, Rrc};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};

/// The reason of the rate error of `pulse_match` (a hint to resample).
const RATE_REASON: &str = "the sample rate must be within 1 % of an integer multiple (2 to \
    1024) of the symbol rate; put a resample before the matched filter";

/// The `symbol_rate_hz` parameter of `pulse_match`, `qam_demod` and `qam_frame_decode`.
pub(crate) const SYMBOL_RATE_PARAM: ParamDesc = ParamDesc {
    name: "symbol_rate_hz",
    unit: Unit::Hz,
    kind: ParamKind::F64 {
        min: MIN_SYMBOL_RATE_HZ,
        max: HZ_LIMIT,
        default: None,
    },
    typical: "25000",
    mutability: Mutability::Build,
    description: "Symbol rate in Hz; the sample rate must be within 1 % of an integer \
            multiple (2 to 1024) of it. Required.",
};

/// The `rolloff` parameter of `pulse_match`, `qam_demod` and `qam_frame_decode`.
pub(crate) const MATCH_ROLLOFF_PARAM: ParamDesc = ParamDesc {
    name: "rolloff",
    unit: Unit::None,
    kind: ParamKind::F64 {
        min: RRC_MIN_ROLLOFF,
        max: 1.0,
        default: Some(0.35),
    },
    typical: "0.35 (the roll-off of the transmitter)",
    mutability: Mutability::Build,
    description: "Roll-off of the root-raised-cosine pulse.",
};

/// The `span` parameter of `pulse_match`, `qam_demod` and `qam_frame_decode`.
pub(crate) const MATCH_SPAN_PARAM: ParamDesc = ParamDesc {
    name: "span",
    unit: Unit::None,
    kind: ParamKind::I64 {
        min: RRC_MIN_SPAN as i64,
        max: RRC_MAX_SPAN as i64,
        default: Some(10),
    },
    typical: "10 (the span of the transmitter)",
    mutability: Mutability::Build,
    description: "Length of the filter in symbols (span n + 1 taps).",
};

/// Description of the `pulse_match` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "pulse_match",
    summary: "Filters a complex signal with the root-raised-cosine pulse matched to pulse_shape \
        (the receive filter).",
    details: "Filters with the span n + 1 taps of the root-raised-cosine pulse (rolloff, span \
        symbols, unit energy) divided by sqrt(n), each rounded once to f32, where n is the \
        sample rate divided by symbol_rate_hz rounded to an integer: the ratio must lie within \
        1 % of an integer from 2 to 1024 (otherwise the rate is unsupported: resample first). \
        After the pulse of pulse_shape (or qam_mod, psk_mod) the output has the symbols at \
        the symbol centres with their transmitted amplitude. It is the matched filter of \
        qam_demod. The rate does not change; the group delay is span n / 2 samples.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32],
        description: "The received baseband signal.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32],
        description: "The filtered signal at the input rate.",
    }],
    params: &[SYMBOL_RATE_PARAM, MATCH_ROLLOFF_PARAM, MATCH_SPAN_PARAM],
    required: &["symbol_rate_hz"],
};

/// Configuration of [`PulseMatchBlock`].
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::PulseMatchConfig;
/// let cfg = PulseMatchConfig { symbol_rate_hz: 25_000.0, ..PulseMatchConfig::default() };
/// assert_eq!(cfg.samples_per_symbol(Rate::hz(200_000)?)?, 8);
/// // 7.5 samples per symbol is not within 1 % of an integer
/// assert!(cfg.validate(Rate::hz(187_500)?).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PulseMatchConfig {
    /// Symbol rate in Hz (required).
    pub symbol_rate_hz: f64,
    /// Roll-off of the root-raised-cosine pulse (0.05 to 1).
    pub rolloff: f64,
    /// Length of the filter in symbols (4 to 32).
    pub span: usize,
}

impl Default for PulseMatchConfig {
    /// `symbol_rate_hz` 1200 (a placeholder: the pipeline requires the parameter), `rolloff`
    /// 0.35, `span` 10.
    fn default() -> Self {
        Self {
            symbol_rate_hz: 1200.0,
            rolloff: 0.35,
            span: 10,
        }
    }
}

impl PulseMatchConfig {
    /// The integer samples per symbol `n` of the filter (the rate rule): `fs / symbol_rate_hz`
    /// rounded, which must lie within 1 % of `n` with `n` from 2 to 1024.
    ///
    /// # Errors
    /// `OutOfRange` for a value outside its range; `InvalidParameter` of `symbol_rate_hz` (with
    /// the hint to resample; `rate_unsupported` in the pipeline) when the rate rule fails.
    pub fn samples_per_symbol(&self, rate_in: Rate) -> Result<usize> {
        check_f64(
            "symbol_rate_hz",
            self.symbol_rate_hz,
            MIN_SYMBOL_RATE_HZ,
            HZ_LIMIT,
        )?;
        check_f64("rolloff", self.rolloff, RRC_MIN_ROLLOFF, 1.0)?;
        check_count("span", self.span, RRC_MIN_SPAN, RRC_MAX_SPAN)?;
        let sps = rate_in.as_f64() / self.symbol_rate_hz;
        let n = libm::floor(sps + 0.5);
        let near = libm::fabs(sps - n) <= SPS_TOLERANCE * n;
        if !(near && n >= 2.0 && n <= RRC_MAX_SPS as f64) {
            return Err(Error::InvalidParameter {
                name: "symbol_rate_hz",
                reason: RATE_REASON,
            });
        }
        Ok(n as usize)
    }

    /// Checks the configuration against the input sample rate.
    ///
    /// # Errors
    /// As [`PulseMatchConfig::samples_per_symbol`].
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        self.samples_per_symbol(rate_in).map(|_| ())
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
            "symbol_rate_hz" => Ok(ParamValue::F64(self.symbol_rate_hz)),
            "rolloff" => Ok(ParamValue::F64(self.rolloff)),
            "span" => Ok(ParamValue::I64(self.span as i64)),
            _ => unknown(),
        }
    }
}

/// The root-raised-cosine matched filter: `C32` in, `C32` out at the input rate.
///
/// Designs the `span n + 1` taps `rrc(rolloff, n, span) / sqrt(n)` in `f64`
/// ([`design::rrc`](crate::design::rrc)), with `n` from the rate rule of
/// [`PulseMatchConfig::samples_per_symbol`], and rounds each once to `f32`; the kernel `fir`
/// runs through the facade. A call takes `min(input, output)` samples. Timing: constant ratio
/// 1, delay `span n / 2` samples. Storage: the taps and the filter history of `span n`
/// samples. The building block of `qam_demod` (M44.1.1 §3.2).
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, PulseMatchBlock, PulseMatchConfig};
/// let cfg = PulseMatchConfig { symbol_rate_hz: 2400.0, ..PulseMatchConfig::default() };
/// let rate = Rate::hz(9600)?;
/// let bytes = PulseMatchBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut b = PulseMatchBlock::new(&cfg, rate, &mut arena)?;
/// assert_eq!((b.samples_per_symbol(), b.num_taps()), (4, 41));
/// let mut out = [C32::default(); 8];
/// let p = b.process(&[C32::new(1.0, 0.0); 8], &mut out)?;
/// assert_eq!((p.consumed, p.produced), (8, 8));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct PulseMatchBlock<S: Storage> {
    fir: FirState<C32, S::Buf<f32>, S::Buf<C32>>,
    symbol_rate_hz: f64,
    rolloff: f64,
    /// The integer samples per symbol `n` (2 to 1024).
    sps: u16,
}

impl<S: Storage> core::fmt::Debug for PulseMatchBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PulseMatchBlock")
            .field("config", &self.config())
            .field("samples_per_symbol", &self.sps)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> PulseMatchBlock<S> {
    /// Designs the filter and creates the block; taps and history come from `storage`, in this
    /// order.
    ///
    /// # Errors
    /// As [`PulseMatchConfig::validate`]; `Storage` when the storage is too small.
    pub fn new(config: &PulseMatchConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        let n = config.samples_per_symbol(rate_in)?;
        let rrc = Rrc::new(config.rolloff, n, config.span)?;
        let len = rrc.len();
        let gain = 1.0 / libm::sqrt(n as f64);
        let mut taps = storage.alloc::<f32>(len)?;
        for (k, t) in taps.as_mut().iter_mut().enumerate() {
            *t = (rrc.tap(k) * gain) as f32;
        }
        let hist = storage.alloc::<C32>(len - 1)?;
        Ok(Self {
            fir: FirState::new(taps, hist)?,
            symbol_rate_hz: config.symbol_rate_hz,
            rolloff: config.rolloff,
            // At most `RRC_MAX_SPS` (1024).
            sps: u16::try_from(n).map_err(|_| Error::Core(sdrk_core::Error::Internal("sps")))?,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// # Errors
    /// As [`PulseMatchConfig::validate`].
    pub fn storage_bytes(config: &PulseMatchConfig, rate_in: Rate) -> Result<usize> {
        let mut plan = Plan::default();
        Self::plan(config, rate_in, &mut plan)?;
        Ok(plan.bytes())
    }

    /// Adds the buffers of the block to `plan`, in the order `new` takes them.
    pub(crate) fn plan(config: &PulseMatchConfig, rate_in: Rate, plan: &mut Plan) -> Result<()> {
        let n = config.samples_per_symbol(rate_in)?;
        let taps = config.span * n + 1;
        plan.add::<f32>(taps)?;
        plan.add::<C32>(taps - 1)
    }

    /// Output rate: the input rate.
    ///
    /// # Errors
    /// As [`PulseMatchConfig::validate`].
    pub fn rate_out(config: &PulseMatchConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }

    /// The configuration of the block.
    pub fn config(&self) -> PulseMatchConfig {
        PulseMatchConfig {
            symbol_rate_hz: self.symbol_rate_hz,
            rolloff: self.rolloff,
            span: (self.num_taps() - 1) / usize::from(self.sps).max(1),
        }
    }

    /// The integer samples per symbol `n` of the filter.
    pub fn samples_per_symbol(&self) -> usize {
        usize::from(self.sps)
    }

    /// Number of taps (`span n + 1`).
    pub fn num_taps(&self) -> usize {
        self.fir.taps().len()
    }

    /// The taps as `f32`, for tests.
    pub fn taps(&self) -> &[f32] {
        self.fir.taps()
    }
}

impl<S: Storage> Block for PulseMatchBlock<S> {
    type In = C32;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[C32], output: &mut [C32]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as Fir<C32>>::fir(&Sdrk, &mut self.fir, head(input, n)?, head_mut(output, n)?)?;
        Ok(Progress {
            consumed: n,
            produced: n,
        })
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
