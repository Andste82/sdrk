//! `psk_mod`: BPSK, QPSK and 8PSK modulator (soft bits to root-raised-cosine shaped symbols),
//! coherent or differential, with the QPSK variants pi/4-DQPSK and offset QPSK.

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::interp::{FirInterpState, FirInterpolate};

use super::costas::{check_f64, check_order};
use super::slice::{C8, PSK8_GRAY, S8};
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

/// The names of the `variant` choice, in index order.
const VARIANT_OPTIONS: &[&str] = &["plain", "pi4", "offset"];

/// The variant of the PSK blocks `psk_mod` and `psk_demod` (M40.1).
///
/// ```
/// use sdrk_blocks::{PskModConfig, PskVariant};
/// let pi4 = PskModConfig {
///     order: 4,
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
    /// The plain constellation of the order (BPSK, QPSK or 8PSK, coherent or differential).
    #[default]
    Plain,
    /// pi/4-DQPSK: order 4 and differential only; the phase steps by `+pi/4` (bits `00`),
    /// `+3 pi/4` (`01`), `-3 pi/4` (`11`) or `-pi/4` (`10`), bit 0 first.
    Pi4,
    /// Offset QPSK (OQPSK): order 4, not differential, an even `samples_per_symbol`; the
    /// transmitter delays the quadrature part by half a symbol, the receiver the in-phase part.
    /// The receiver corrects the carrier only after that delay, so it needs the carrier phase
    /// on the axes and no carrier offset (open item I40.1-3).
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
    description: "plain, pi4 (pi/4-DQPSK: needs order 4 and differential) or offset (OQPSK: \
        needs order 4, not differential and an even number of samples per symbol).",
};

/// The combination rules of `variant` (M40.1 §2 to §4): `pi4` needs order 4 and
/// `differential`, `offset` needs order 4, not `differential` and `even_sps`; any other
/// variant than `plain` needs order 4. Otherwise `InvalidParameter` of `variant`.
pub(crate) fn check_variant(
    variant: PskVariant,
    order: u32,
    differential: bool,
    even_sps: bool,
) -> Result<()> {
    let reason = match variant {
        PskVariant::Plain => return Ok(()),
        PskVariant::Pi4 if order == 4 && differential => return Ok(()),
        PskVariant::Pi4 => "pi4 needs order 4 and differential",
        PskVariant::Offset if order == 4 && !differential && even_sps => return Ok(()),
        PskVariant::Offset => {
            "offset needs order 4, not differential and an even number of samples per symbol"
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
        included, a 0; frame_encode produces them) and maps 1 (order 2, BPSK), 2 (order 4, \
        QPSK) or 3 (order 8, 8PSK) bits to a symbol with the Gray mapping of slice: BPSK (2 b \
        - 1, 0), QPSK ((2 b0 - 1) / sqrt(2), (2 b1 - 1) / sqrt(2)), so bit 1 is positive; 8PSK \
        exp(j (2 pi k / 8 + pi / 8)) with k from the Gray code b0 b1 b2 (000 -> 0, 001 -> 1, \
        011 -> 2, 010 -> 3, 110 -> 4, 111 -> 5, 101 -> 6, 100 -> 7). With differential set, \
        the phase of a symbol is the phase of the previous symbol plus the phase of the mapped \
        symbol (start phase 0), which psk_demod with differential undoes without a carrier \
        loop. The variant pi4 (order 4, differential) is pi/4-DQPSK: the phase steps by +pi/4 \
        (bits 00), +3 pi/4 (01), -3 pi/4 (11) or -pi/4 (10). The variant offset (order 4, not \
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
            description: "Order of the phase-shift keying: 2 (BPSK, 1 bit per symbol), 4 \
                (QPSK, 2 bits per symbol) or 8 (8PSK, 3 bits per symbol).",
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
        VARIANT_PARAM,
    ],
    required: &["samples_per_symbol"],
};

/// The constellation point at the phase `k pi / 8` (`k` mod 16), each part rounded once to
/// `f32`; the multiples of `pi / 8` keep the points exact while the differential phase runs
/// (D613). The even entries are the eight points of M40 at multiples of `pi / 4` (BPSK, QPSK
/// and the differential phase), the odd ones the 8PSK points (`cos(pi / 8)`, `sin(pi / 8)`).
pub(crate) const fn point16(k: u8) -> C32 {
    let (re, im) = match k & 15 {
        0 => (1.0, 0.0),
        1 => (C8, S8),
        2 => (H, H),
        3 => (S8, C8),
        4 => (0.0, 1.0),
        5 => (-S8, C8),
        6 => (-H, H),
        7 => (-C8, S8),
        8 => (-1.0, 0.0),
        9 => (-C8, -S8),
        10 => (-H, -H),
        11 => (-S8, -C8),
        12 => (0.0, -1.0),
        13 => (S8, -C8),
        14 => (H, -H),
        _ => (C8, -S8),
    };
    C32 { re, im }
}

/// The phase index (in units of `pi / 8`) of the symbol of `bits` (1, 2 or 3 soft bits, above
/// 0 is a 1). BPSK: 1 -> 0, 0 -> 8. QPSK (Gray, plain): (1, 1) -> 2, (0, 1) -> 6, (0, 0) ->
/// 10, (1, 0) -> 14. pi/4-DQPSK (the increments of M40.1 §2): (0, 0) -> 2 (`+pi/4`), (0, 1)
/// -> 6 (`+3 pi/4`), (1, 1) -> 10 (`-3 pi/4`), (1, 0) -> 14 (`-pi/4`). 8PSK: `2 k + 1` with `k`
/// the entry `b0 4 + b1 2 + b2` of [`PSK8_GRAY`].
pub(super) fn phase_index(bits: &[f32], pi4: bool) -> u8 {
    match bits {
        [b] => {
            if *b > 0.0 {
                0
            } else {
                8
            }
        }
        [b0, b1] => match (*b0 > 0.0, *b1 > 0.0, pi4) {
            (true, true, false) | (false, false, true) => 2,
            (false, true, _) => 6,
            (false, false, false) | (true, true, true) => 10,
            (true, false, _) => 14,
        },
        [b0, b1, b2, ..] => {
            let g =
                usize::from(*b0 > 0.0) * 4 + usize::from(*b1 > 0.0) * 2 + usize::from(*b2 > 0.0);
            let k = PSK8_GRAY.get(g).copied().unwrap_or(0);
            2 * k + 1
        }
        [] => 0,
    }
}

/// Configuration of [`PskModBlock`].
///
/// ```
/// use sdrk_blocks::{PskModConfig, PskVariant};
/// let cfg = PskModConfig { order: 4, samples_per_symbol: 8, ..PskModConfig::default() };
/// assert!(cfg.validate().is_ok());
/// assert_eq!(cfg.bits_per_symbol(), 2);
/// assert!(PskModConfig { samples_per_symbol: 1, ..cfg }.validate().is_err());
/// assert!(PskModConfig { order: 3, ..cfg }.validate().is_err());
/// let psk8 = PskModConfig { order: 8, ..cfg };
/// assert_eq!(psk8.bits_per_symbol(), 3);
/// // the variants other than plain are QPSK only
/// assert!(PskModConfig { variant: PskVariant::Offset, ..psk8 }.validate().is_err());
/// assert!(PskModConfig { variant: PskVariant::Offset, ..cfg }.validate().is_ok());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PskModConfig {
    /// Order: 2 (BPSK), 4 (QPSK) or 8 (8PSK).
    pub order: u32,
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
    /// Order 2, `samples_per_symbol` 8 (a placeholder: the pipeline requires the parameter),
    /// `rolloff` 0.35, `span` 10, not differential, variant plain.
    fn default() -> Self {
        Self {
            order: 2,
            samples_per_symbol: 8,
            rolloff: 0.35,
            span: 10,
            differential: false,
            variant: PskVariant::Plain,
        }
    }
}

/// Bits per symbol of an order: 3 for 8, 2 for 4, otherwise 1.
pub(crate) const fn bits_of_order(order: u32) -> usize {
    match order {
        8 => 3,
        4 => 2,
        _ => 1,
    }
}

impl PskModConfig {
    /// Bits per symbol: 1 for order 2, 2 for order 4, 3 for order 8.
    pub fn bits_per_symbol(&self) -> usize {
        bits_of_order(self.order)
    }

    /// Checks the ranges and the combination of `variant` with `order`, `differential` and
    /// `samples_per_symbol`.
    ///
    /// # Errors
    /// `OutOfRange` for a value outside its range; `InvalidParameter` for the orders 3, 5, 6
    /// and 7 and for a `variant` that does not fit (`pi4` without order 4 and `differential`,
    /// `offset` without order 4, with `differential` or with an odd `samples_per_symbol`).
    pub fn validate(&self) -> Result<()> {
        check_order(self.order)?;
        check_count("samples_per_symbol", self.samples_per_symbol, 2, MAX_SPS)?;
        check_f64("rolloff", self.rolloff, RRC_MIN_ROLLOFF, 1.0)?;
        check_count("span", self.span, RRC_MIN_SPAN, RRC_MAX_SPAN)?;
        check_variant(
            self.variant,
            self.order,
            self.differential,
            self.samples_per_symbol.is_multiple_of(2),
        )
    }

    /// Sets a parameter by its description name; checks kind and range (the combination of
    /// the parameters is checked by [`PskModConfig::validate`]).
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`; `InvalidParameter` for the orders
    /// 3, 5, 6 and 7.
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
            "order" => Ok(ParamValue::I64(i64::from(self.order))),
            "samples_per_symbol" => Ok(ParamValue::I64(self.samples_per_symbol as i64)),
            "rolloff" => Ok(ParamValue::F64(self.rolloff)),
            "span" => Ok(ParamValue::I64(self.span as i64)),
            "differential" => Ok(ParamValue::Bool(self.differential)),
            "variant" => Ok(ParamValue::Choice(self.variant.index())),
            _ => unknown(),
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

/// BPSK, QPSK and 8PSK modulator: `f32` soft bits in (one per sample), `C32` out at
/// `samples_per_symbol` samples per symbol.
///
/// Every symbol is one of the sixteen `f32` points at multiples of 22.5 degrees (BPSK, QPSK
/// and their differential phase use the even ones at multiples of 45 degrees, 8PSK the odd
/// ones; the differential phase runs in units of `pi / 8`); the interpolation is the kernel
/// `fir_interpolate` through the facade with the taps `rrc(rolloff, sps, span) sqrt(sps)` as
/// `f32`. With the variant `offset` the imaginary part of the filter output then passes a
/// delay of `sps / 2` samples (a ring of `sps / 2` `f32`, zeros at the start). A call takes
/// `min(input / bits, output / sps)` whole symbols, in pieces of at most [`PSK_MOD_CHUNK`].
/// Timing: constant ratio `sps / bits`, delay `span bits / 2` input samples (`offset`:
/// `(span bits + 2) / 2`, the half symbol of the quadrature part is one bit). Storage: the
/// taps, the filter history, a symbol buffer and for `offset` the ring.
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
    /// The quadrature delay of `offset` (`sps / 2` samples); `None` for the other variants.
    ring: Option<S::Buf<f32>>,
    /// The differential phase in units of `pi / 8`.
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
        let half = config.half();
        let ring = if half > 0 {
            Some(storage.alloc::<f32>(half)?)
        } else {
            None
        };
        Ok(Self {
            config: *config,
            bps: config.bits_per_symbol(),
            sps,
            ratio: config.ratio()?,
            fir: FirInterpState::new(taps, hist, sps)?,
            buf,
            ring,
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
            let pi4 = self.config.variant == PskVariant::Pi4;
            for (s, b) in buf.iter_mut().zip(bits.chunks_exact(bps)) {
                let k = phase_index(b, pi4);
                *s = if self.config.differential {
                    phase = phase.wrapping_add(k) & 15;
                    point16(phase)
                } else {
                    point16(k)
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
            if let Some(ring) = self.ring.as_mut() {
                // The piece is whole symbols, so a multiple of the ring length: every block of
                // `sps / 2` samples swaps its imaginary parts with the ring, which delays them
                // by `sps / 2` samples and needs no position.
                let ring = ring.as_mut();
                for block in out.chunks_exact_mut(ring.len()) {
                    for (y, r) in block.iter_mut().zip(ring.iter_mut()) {
                        core::mem::swap(&mut y.im, r);
                    }
                }
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
            delay: Delay::new(
                (self.config.span * self.bps + 2 * usize::from(self.ring.is_some())) as u64,
                2,
            )
            .unwrap_or(Delay::ZERO),
        }
    }

    fn reset(&mut self) {
        self.fir.reset();
        self.phase = 0;
        if let Some(ring) = self.ring.as_mut() {
            ring.as_mut().fill(0.0);
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
        self.bps
    }
}
