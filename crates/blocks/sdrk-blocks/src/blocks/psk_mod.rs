//! `psk_mod`: BPSK and QPSK modulator (soft bits to root-raised-cosine shaped symbols),
//! coherent or differential.

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::interp::{FirInterpState, FirInterpolate};

use super::costas::{check_f64, check_order};
use super::{check_count, head, head_mut, tail, tail_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::{RRC_MAX_SPAN, RRC_MIN_ROLLOFF, RRC_MIN_SPAN, Rrc};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};

/// Symbols per piece the block processes at a time (the size of its symbol buffer).
pub const PSK_MOD_CHUNK: usize = 256;

/// The largest `samples_per_symbol` of `psk_mod`.
const MAX_SPS: usize = 64;

/// `1 / sqrt(2)` rounded once to `f32`.
const H: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// Description of the `psk_mod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "psk_mod",
    summary: "Turns soft bits into BPSK or QPSK symbols shaped with a root-raised-cosine pulse.",
    details: "Reads one soft bit per input sample (above 0 is a 1, anything else, NaN \
        included, a 0; frame_encode produces them) and maps 1 (order 2, BPSK) or 2 (order 4, \
        QPSK) bits to a symbol with the Gray mapping of slice: BPSK (2 b - 1, 0), QPSK ((2 b0 - \
        1) / sqrt(2), (2 b1 - 1) / sqrt(2)), so bit 1 is positive. With differential set, the \
        phase of a symbol is the phase of the previous symbol plus the phase of the mapped \
        symbol (start phase 0), which psk_demod with differential undoes without a carrier \
        loop. The symbols are interpolated by samples_per_symbol with the root-raised-cosine \
        pulse (rolloff, span symbols) scaled by sqrt(samples_per_symbol), so the matched \
        filter of psk_demod gives amplitude 1 at the symbol centres. The output rate is the \
        bit rate times samples_per_symbol / bits per symbol; the group delay is span / 2 \
        symbols. A call takes whole symbols only. Follow it with duc to reach the radio rate.",
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
            description: "Order of the phase-shift keying: 2 (BPSK, 1 bit per symbol) or 4 \
                (QPSK, 2 bits per symbol).",
        },
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
    ],
    required: &["samples_per_symbol"],
};

/// The constellation point at the phase `k pi / 4` (`k` mod 8), each part rounded once to
/// `f32`; the multiples of `pi / 4` keep the points exact while the differential phase runs
/// (D613).
pub(crate) const fn point(k: u8) -> C32 {
    let (re, im) = match k & 7 {
        0 => (1.0, 0.0),
        1 => (H, H),
        2 => (0.0, 1.0),
        3 => (-H, H),
        4 => (-1.0, 0.0),
        5 => (-H, -H),
        6 => (0.0, -1.0),
        _ => (H, -H),
    };
    C32 { re, im }
}

/// The phase index (in units of `pi / 4`) of the Gray symbol of `bits` (1 or 2 soft bits,
/// above 0 is a 1): BPSK 1 -> 0, 0 -> 4; QPSK (1, 1) -> 1, (0, 1) -> 3, (0, 0) -> 5,
/// (1, 0) -> 7.
fn phase_index(bits: &[f32]) -> u8 {
    match bits {
        [b] => {
            if *b > 0.0 {
                0
            } else {
                4
            }
        }
        [b0, b1, ..] => match (*b0 > 0.0, *b1 > 0.0) {
            (true, true) => 1,
            (false, true) => 3,
            (false, false) => 5,
            (true, false) => 7,
        },
        [] => 0,
    }
}

/// Configuration of [`PskModBlock`].
///
/// ```
/// use sdrk_blocks::PskModConfig;
/// let cfg = PskModConfig { order: 4, samples_per_symbol: 8, ..PskModConfig::default() };
/// assert!(cfg.validate().is_ok());
/// assert_eq!(cfg.bits_per_symbol(), 2);
/// assert!(PskModConfig { samples_per_symbol: 1, ..cfg }.validate().is_err());
/// assert!(PskModConfig { order: 3, ..cfg }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PskModConfig {
    /// Order: 2 (BPSK) or 4 (QPSK).
    pub order: u32,
    /// Output samples per symbol, 2 to 64 (required).
    pub samples_per_symbol: usize,
    /// Roll-off of the root-raised-cosine pulse (0.05 to 1).
    pub rolloff: f64,
    /// Length of the pulse in symbols (4 to 32).
    pub span: usize,
    /// Differential coding.
    pub differential: bool,
}

impl Default for PskModConfig {
    /// Order 2, `samples_per_symbol` 8 (a placeholder: the pipeline requires the parameter),
    /// `rolloff` 0.35, `span` 10, not differential.
    fn default() -> Self {
        Self {
            order: 2,
            samples_per_symbol: 8,
            rolloff: 0.35,
            span: 10,
            differential: false,
        }
    }
}

impl PskModConfig {
    /// Bits per symbol: 1 for order 2, 2 for order 4.
    pub fn bits_per_symbol(&self) -> usize {
        if self.order == 4 { 2 } else { 1 }
    }

    /// Checks the ranges.
    ///
    /// # Errors
    /// `OutOfRange` for a value outside its range; `InvalidParameter` for the order 3.
    pub fn validate(&self) -> Result<()> {
        check_order(self.order)?;
        check_count("samples_per_symbol", self.samples_per_symbol, 2, MAX_SPS)?;
        check_f64("rolloff", self.rolloff, RRC_MIN_ROLLOFF, 1.0)?;
        check_count("span", self.span, RRC_MIN_SPAN, RRC_MAX_SPAN)
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
            "samples_per_symbol" => self.samples_per_symbol = value.as_usize()?,
            "rolloff" => self.rolloff = value.as_f64()?,
            "span" => self.span = value.as_usize()?,
            "differential" => self.differential = value.as_bool()?,
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
            "differential" => Ok(ParamValue::Bool(self.differential)),
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

/// BPSK and QPSK modulator: `f32` soft bits in (one per sample), `C32` out at
/// `samples_per_symbol` samples per symbol.
///
/// Every symbol is one of the eight `f32` points at multiples of 45 degrees (BPSK and the
/// differential phase need the axes, QPSK the diagonals); the interpolation is the kernel
/// `fir_interpolate` through the facade with the taps `rrc(rolloff, sps, span) sqrt(sps)` as
/// `f32`. A call takes `min(input / bits, output / sps)` whole symbols, in pieces of at most
/// [`PSK_MOD_CHUNK`]. Timing: constant ratio `sps / bits`, delay `span bits / 2` input
/// samples. Storage: the taps, the filter history and a symbol buffer.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, PskModBlock, PskModConfig};
/// let cfg = PskModConfig { order: 4, samples_per_symbol: 4, ..PskModConfig::default() };
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
    config: PskModConfig,
    bps: usize,
    sps: usize,
    ratio: Ratio,
    fir: FirInterpState<C32, S::Buf<f32>, S::Buf<C32>>,
    buf: S::Buf<C32>,
    /// The differential phase in units of `pi / 4`.
    phase: u8,
}

impl<S: Storage> core::fmt::Debug for PskModBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PskModBlock")
            .field("config", &self.config)
            .field("phase", &self.phase)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> PskModBlock<S> {
    /// Designs the pulse and creates the block; taps, history and the symbol buffer come
    /// from `storage`.
    ///
    /// # Errors
    /// As [`PskModConfig::validate`]; `Storage` when the storage is too small.
    pub fn new(config: &PskModConfig, _rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate()?;
        let sps = config.samples_per_symbol;
        let rrc = Rrc::new(config.rolloff, sps, config.span)?;
        let n = rrc.len();
        let gain = libm::sqrt(sps as f64);
        let mut taps = storage.alloc::<f32>(n)?;
        for (k, t) in taps.as_mut().iter_mut().enumerate() {
            *t = (rrc.tap(k) * gain) as f32;
        }
        let hist = storage.alloc::<C32>(n.div_ceil(sps))?;
        let buf = storage.alloc::<C32>(PSK_MOD_CHUNK)?;
        Ok(Self {
            config: *config,
            bps: config.bits_per_symbol(),
            sps,
            ratio: config.ratio()?,
            fir: FirInterpState::new(taps, hist, sps)?,
            buf,
            phase: 0,
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
        self.fir.taps().len()
    }
}

impl<S: Storage> Block for PskModBlock<S> {
    type In = f32;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[f32], output: &mut [C32]) -> Result<Progress> {
        let (bps, sps) = (self.bps, self.sps);
        let (mut consumed, mut produced) = (0usize, 0usize);
        loop {
            let syms = ((input.len() - consumed) / bps)
                .min((output.len() - produced) / sps)
                .min(PSK_MOD_CHUNK);
            if syms == 0 {
                break;
            }
            let bits = head(tail(input, consumed)?, syms * bps)?;
            let buf = head_mut(self.buf.as_mut(), syms)?;
            let mut phase = self.phase;
            for (s, b) in buf.iter_mut().zip(bits.chunks_exact(bps)) {
                let k = phase_index(b);
                *s = if self.config.differential {
                    phase = phase.wrapping_add(k) & 7;
                    point(phase)
                } else {
                    point(k)
                };
            }
            self.phase = phase;
            let out = head_mut(tail_mut(output, produced)?, syms * sps)?;
            let p = <Sdrk as FirInterpolate<C32>>::fir_interpolate(
                &Sdrk,
                &mut self.fir,
                head(self.buf.as_ref(), syms)?,
                out,
            )?;
            if p.consumed != syms {
                return Err(Error::Core(sdrk_core::Error::Internal(
                    "psk_mod filter did not take the whole piece",
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
        self.phase = 0;
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
