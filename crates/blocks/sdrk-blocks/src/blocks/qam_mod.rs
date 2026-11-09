//! `qam_mod`: 16-QAM and 64-QAM modulator (soft bits to root-raised-cosine shaped symbols).

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::constellation::Qam;
use sdrk_core::kernels::interp::{FirInterpState, FirInterpolate};

use super::costas::check_f64;
use super::{check_count, head, head_mut, tail, tail_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::{RRC_MAX_SPAN, RRC_MIN_ROLLOFF, RRC_MIN_SPAN, Rrc};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};

/// Symbols per piece the block processes at a time (the size of its symbol buffer).
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
        (unit average power). These are the constellations qam16 and qam64 of slice, equalize \
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
            typical: "0.35; 0.2 for a narrow spectrum",
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

/// The symbol of `k` soft bits (the first the most significant, above 0 is a 1).
#[inline]
pub(crate) fn symbol_of(bits: &[f32]) -> u32 {
    bits.iter()
        .fold(0u32, |s, &b| (s << 1) | u32::from(b > 0.0))
}

/// 16-QAM and 64-QAM modulator: `f32` soft bits in (one per sample), `C32` out at
/// `samples_per_symbol` samples per symbol.
///
/// Every symbol is the `f32` point of `sdrk_core::kernels::constellation::Qam` of its bits (M44
/// §1); the interpolation is the kernel `fir_interpolate` through the facade with the taps
/// `rrc(rolloff, sps, span) sqrt(sps)` as `f32`, as in `psk_mod`. A call takes
/// `min(input / bits, output / sps)` whole symbols, in pieces of at most [`QAM_MOD_CHUNK`].
/// Timing: constant ratio `sps / bits`, delay `span bits / 2` input samples. Storage: the taps,
/// the filter history and a symbol buffer.
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
    qam: Qam,
    bps: usize,
    sps: usize,
    ratio: Ratio,
    fir: FirInterpState<C32, S::Buf<f32>, S::Buf<C32>>,
    buf: S::Buf<C32>,
}

impl<S: Storage> core::fmt::Debug for QamModBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QamModBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> QamModBlock<S> {
    /// Designs the pulse and creates the block; taps, history and the symbol buffer come
    /// from `storage`.
    ///
    /// # Errors
    /// As [`QamModConfig::validate`]; `Storage` when the storage is too small.
    pub fn new(config: &QamModConfig, _rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate()?;
        let qam = qam_of_order(config.order)?;
        let sps = config.samples_per_symbol;
        let rrc = Rrc::new(config.rolloff, sps, config.span)?;
        let n = rrc.len();
        let gain = libm::sqrt(sps as f64);
        let mut taps = storage.alloc::<f32>(n)?;
        for (k, t) in taps.as_mut().iter_mut().enumerate() {
            *t = (rrc.tap(k) * gain) as f32;
        }
        let hist = storage.alloc::<C32>(n.div_ceil(sps))?;
        let buf = storage.alloc::<C32>(QAM_MOD_CHUNK)?;
        Ok(Self {
            config: *config,
            qam,
            bps: qam.bits_per_symbol(),
            sps,
            ratio: config.ratio()?,
            fir: FirInterpState::new(taps, hist, sps)?,
            buf,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// # Errors
    /// As [`QamModConfig::validate`].
    pub fn storage_bytes(config: &QamModConfig, _rate_in: Rate) -> Result<usize> {
        config.validate()?;
        let sps = config.samples_per_symbol;
        let n = config.span * sps + 1;
        let mut plan = Plan::default();
        plan.add::<f32>(n)?;
        plan.add::<C32>(n.div_ceil(sps))?;
        plan.add::<C32>(QAM_MOD_CHUNK)?;
        Ok(plan.bytes())
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
        self.fir.taps().len()
    }
}

impl<S: Storage> Block for QamModBlock<S> {
    type In = f32;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[f32], output: &mut [C32]) -> Result<Progress> {
        let (bps, sps) = (self.bps, self.sps);
        let points = self.qam.point_table();
        let (mut consumed, mut produced) = (0usize, 0usize);
        loop {
            let syms = ((input.len() - consumed) / bps)
                .min((output.len() - produced) / sps)
                .min(QAM_MOD_CHUNK);
            if syms == 0 {
                break;
            }
            let bits = head(tail(input, consumed)?, syms * bps)?;
            let buf = head_mut(self.buf.as_mut(), syms)?;
            for (s, b) in buf.iter_mut().zip(bits.chunks_exact(bps)) {
                // The symbol has `bps` bits, below the table length.
                *s = points
                    .get(symbol_of(b) as usize)
                    .copied()
                    .unwrap_or_default();
            }
            let out = head_mut(tail_mut(output, produced)?, syms * sps)?;
            let p = <Sdrk as FirInterpolate<C32>>::fir_interpolate(
                &Sdrk,
                &mut self.fir,
                head(self.buf.as_ref(), syms)?,
                out,
            )?;
            if p.consumed != syms {
                return Err(Error::Core(sdrk_core::Error::Internal(
                    "qam_mod filter did not take the whole piece",
                )));
            }
            consumed += syms * bps;
            produced += p.produced;
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: self.ratio,
            kind: RatioKind::Constant,
            delay: Delay::new((self.config.span * self.bps) as u64, 2).unwrap_or(Delay::ZERO),
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
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        self.bps
    }
}
