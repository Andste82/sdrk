//! `symbol_map`: soft bits to the points of a constellation (BPSK, QPSK, 8PSK and the square
//! QAM of 16 to 1024 points), coherent or differential, without pulse shaping; the transmit side
//! of `symbol_demap`.

use core::marker::PhantomData;

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_core::C32;
use sdrk_core::kernels::constellation::{Qam, level_of_code};

use sdrk_core::kernels::costas::CostasConstellation;
use sdrk_core::kernels::equalize::EqConstellation;

use super::symbol_demap::{C8, PSK8_GRAY, S8, SymbolDemapConstellation};
use super::unknown;
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};

/// `1 / sqrt(2)` rounded once to `f32`.
const H: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// The names of the `constellation` choice, in index order: the same names at the same indices
/// in `costas`, `equalize`, `symbol_demap` and the `constellation` of the QAM and PSK composites
/// (M44.1.1 §1).
pub(crate) const CONSTELLATION_OPTIONS: &[&str] = &[
    "bpsk", "qpsk", "8psk", "qam16", "qam64", "qam256", "qam1024",
];

/// The names of the `variant` choice, in index order.
const VARIANT_OPTIONS: &[&str] = &["plain", "pi4"];

/// Description of the `symbol_map` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "symbol_map",
    summary: "Turns soft bits into the points of a constellation (BPSK to 1024-QAM), one \
        sample per symbol.",
    details: "Reads one soft bit per input sample (above 0 is a 1, anything else, NaN \
        included, a 0; frame_encode produces them) and maps bits-per-symbol of them, the first \
        one the most significant, to one point of the constellation; it is the transmit side \
        of symbol_demap and uses its names and its Gray codes. bpsk (1 bit): 1 -> (1, 0), \
        0 -> (-1, 0). qpsk (2 bits): ((2 b0 - 1) / sqrt(2), (2 b1 - 1) / sqrt(2)). 8psk (3 \
        bits): exp(j (2 pi k / 8 + pi / 8)) with k from the Gray code b0 b1 b2 (000 -> 0, 001 \
        -> 1, 011 -> 2, 010 -> 3, 110 -> 4, 111 -> 5, 101 -> 6, 100 -> 7). The PSK points are \
        the sixteen f32 points at multiples of 22.5 degrees, each part rounded once to f32. \
        qam16, qam64, qam256 and qam1024 (4, 6, 8 and 10 bits): the unit-power Gray square \
        constellations of qam_mod, \
        equalize, costas and symbol_demap (the first half of the bits selects the in-phase \
        level, the second half the quadrature level). With differential set (bpsk, qpsk and \
        8psk only) the phase of a symbol is the phase of the previous symbol plus the phase of \
        the mapped symbol, kept as a multiple of 22.5 degrees, start phase 0. The variant pi4 \
        (qpsk with differential only) is pi/4-DQPSK: the phase steps by +pi/4 (bits 00), +3 \
        pi/4 (01), -3 pi/4 (11) or -pi/4 (10). Any other combination is invalid. The output \
        rate is the bit rate divided by the bits per symbol; delay 0. A call takes whole \
        symbols only. Follow it with interpolate (or the pulse shaping of a modulator).",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32],
        description: "Soft bits, one sample per bit; above 0 is a 1.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32],
        description: "The points, one sample per symbol.",
    }],
    params: &[
        ParamDesc {
            name: "constellation",
            unit: Unit::None,
            kind: ParamKind::Choice {
                options: CONSTELLATION_OPTIONS,
                default: 0,
            },
            typical: "bpsk; qpsk, 8psk or qam16 to qam1024 for more bits per symbol",
            mutability: Mutability::Build,
            description: "The constellation: bpsk (1 bit per symbol), qpsk (2), 8psk (3), qam16 \
                (4), qam64 (6), qam256 (8) or qam1024 (10), the names of symbol_demap.",
        },
        ParamDesc {
            name: "differential",
            unit: Unit::None,
            kind: ParamKind::Bool { default: false },
            typical: "false; true for a receiver without carrier recovery",
            mutability: Mutability::Build,
            description: "Differential coding (bpsk, qpsk and 8psk only): each symbol adds its \
                phase to the previous one.",
        },
        ParamDesc {
            name: "variant",
            unit: Unit::None,
            kind: ParamKind::Choice {
                options: VARIANT_OPTIONS,
                default: 0,
            },
            typical: "plain; pi4 for pi/4-DQPSK (TETRA)",
            mutability: Mutability::Build,
            description: "plain or pi4 (pi/4-DQPSK: needs qpsk and differential).",
        },
    ],
    required: &[],
};

/// The constellation of [`SymbolMapBlock`] (the transmit names of
/// [`SymbolDemapConstellation`](crate::SymbolDemapConstellation)).
///
/// ```
/// use sdrk_blocks::SymbolMapConstellation;
/// assert_eq!(SymbolMapConstellation::Psk8.bits_per_symbol(), 3);
/// assert_eq!(SymbolMapConstellation::Qam64.bits_per_symbol(), 6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum SymbolMapConstellation {
    /// BPSK, one bit: 1 -> `(1, 0)`, 0 -> `(-1, 0)`.
    #[default]
    Bpsk,
    /// QPSK, two bits (Gray): `((2 b0 - 1) / √2, (2 b1 - 1) / √2)`.
    Qpsk,
    /// 8PSK, three bits `b0 b1 b2` of the Gray code [`PSK8_GRAY`](crate::PSK8_GRAY): the point
    /// `exp(j (2 pi k / 8 + pi / 8))`.
    Psk8,
    /// 16-QAM, four bits: the points of [`Qam::Qam16`].
    Qam16,
    /// 64-QAM, six bits: the points of [`Qam::Qam64`].
    Qam64,
    /// 256-QAM, eight bits: the points of [`Qam::Qam256`] (M44.2).
    Qam256,
    /// 1024-QAM, ten bits: the points of [`Qam::Qam1024`] (M44.2).
    Qam1024,
}

impl SymbolMapConstellation {
    /// Bits per symbol: 1, 2, 3, 4, 6, 8 or 10.
    pub const fn bits_per_symbol(self) -> usize {
        match self {
            Self::Bpsk => 1,
            Self::Qpsk => 2,
            Self::Psk8 => 3,
            Self::Qam16 => 4,
            Self::Qam64 => 6,
            Self::Qam256 => 8,
            Self::Qam1024 => 10,
        }
    }

    /// Whether the constellation is a phase-shift keying (BPSK, QPSK or 8PSK).
    pub const fn is_psk(self) -> bool {
        matches!(self, Self::Bpsk | Self::Qpsk | Self::Psk8)
    }

    pub(crate) const fn index(self) -> u16 {
        match self {
            Self::Bpsk => 0,
            Self::Qpsk => 1,
            Self::Psk8 => 2,
            Self::Qam16 => 3,
            Self::Qam64 => 4,
            Self::Qam256 => 5,
            Self::Qam1024 => 6,
        }
    }

    pub(crate) const fn from_index(i: u16) -> Self {
        match i {
            1 => Self::Qpsk,
            2 => Self::Psk8,
            3 => Self::Qam16,
            4 => Self::Qam64,
            5 => Self::Qam256,
            6 => Self::Qam1024,
            _ => Self::Bpsk,
        }
    }
}

/// The same constellation in the carrier loop (`costas`).
impl From<SymbolMapConstellation> for CostasConstellation {
    fn from(c: SymbolMapConstellation) -> Self {
        match c {
            SymbolMapConstellation::Bpsk => Self::Bpsk,
            SymbolMapConstellation::Qpsk => Self::Qpsk,
            SymbolMapConstellation::Psk8 => Self::Psk8,
            SymbolMapConstellation::Qam16 => Self::Qam16,
            SymbolMapConstellation::Qam64 => Self::Qam64,
            SymbolMapConstellation::Qam256 => Self::Qam256,
            SymbolMapConstellation::Qam1024 => Self::Qam1024,
        }
    }
}

/// The same constellation in the equalizer (`equalize`).
impl From<SymbolMapConstellation> for EqConstellation {
    fn from(c: SymbolMapConstellation) -> Self {
        match c {
            SymbolMapConstellation::Bpsk => Self::Bpsk,
            SymbolMapConstellation::Qpsk => Self::Qpsk,
            SymbolMapConstellation::Psk8 => Self::Psk8,
            SymbolMapConstellation::Qam16 => Self::Qam16,
            SymbolMapConstellation::Qam64 => Self::Qam64,
            SymbolMapConstellation::Qam256 => Self::Qam256,
            SymbolMapConstellation::Qam1024 => Self::Qam1024,
        }
    }
}

/// The same constellation in the soft bits (`symbol_demap`).
impl From<SymbolMapConstellation> for SymbolDemapConstellation {
    fn from(c: SymbolMapConstellation) -> Self {
        match c {
            SymbolMapConstellation::Bpsk => Self::Bpsk,
            SymbolMapConstellation::Qpsk => Self::Qpsk,
            SymbolMapConstellation::Psk8 => Self::Psk8,
            SymbolMapConstellation::Qam16 => Self::Qam16,
            SymbolMapConstellation::Qam64 => Self::Qam64,
            SymbolMapConstellation::Qam256 => Self::Qam256,
            SymbolMapConstellation::Qam1024 => Self::Qam1024,
        }
    }
}

/// The variant of [`SymbolMapBlock`].
///
/// ```
/// use sdrk_blocks::{SymbolMapConfig, SymbolMapConstellation, SymbolMapVariant};
/// let pi4 = SymbolMapConfig {
///     constellation: SymbolMapConstellation::Qpsk,
///     differential: true,
///     variant: SymbolMapVariant::Pi4,
/// };
/// assert!(pi4.validate().is_ok());
/// assert!(SymbolMapConfig { differential: false, ..pi4 }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum SymbolMapVariant {
    /// The plain constellation, coherent or differential.
    #[default]
    Plain,
    /// pi/4-DQPSK (`qpsk` with `differential` only): the phase steps by `+pi/4` (bits `00`),
    /// `+3 pi/4` (`01`), `-3 pi/4` (`11`) or `-pi/4` (`10`), bit 0 first.
    Pi4,
}

/// Configuration of [`SymbolMapBlock`].
///
/// ```
/// use sdrk_blocks::{SymbolMapConfig, SymbolMapConstellation};
/// let cfg = SymbolMapConfig {
///     constellation: SymbolMapConstellation::Qam16,
///     ..SymbolMapConfig::default()
/// };
/// assert!(cfg.validate().is_ok());
/// assert_eq!(cfg.bits_per_symbol(), 4);
/// // differential coding is for the PSK constellations only
/// assert!(SymbolMapConfig { differential: true, ..cfg }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SymbolMapConfig {
    /// The constellation (default `bpsk`).
    pub constellation: SymbolMapConstellation,
    /// Differential coding (`bpsk`, `qpsk` and `8psk` only; default `false`).
    pub differential: bool,
    /// The variant (default `plain`).
    pub variant: SymbolMapVariant,
}

impl SymbolMapConfig {
    /// Bits per symbol of the constellation.
    pub const fn bits_per_symbol(&self) -> usize {
        self.constellation.bits_per_symbol()
    }

    /// Checks the combination: `differential` needs `bpsk`, `qpsk` or `8psk`; `pi4` needs
    /// `qpsk` and `differential`.
    ///
    /// # Errors
    /// `InvalidParameter` of `differential` for a differential QAM, of `variant` for `pi4`
    /// without `qpsk` and `differential`.
    pub fn validate(&self) -> Result<()> {
        if self.differential && !self.constellation.is_psk() {
            return Err(Error::InvalidParameter {
                name: "differential",
                reason: "differential needs bpsk, qpsk or 8psk",
            });
        }
        if self.variant == SymbolMapVariant::Pi4
            && !(self.constellation == SymbolMapConstellation::Qpsk && self.differential)
        {
            return Err(Error::InvalidParameter {
                name: "variant",
                reason: "pi4 needs qpsk and differential",
            });
        }
        Ok(())
    }

    /// Sets a parameter by its description name; checks kind and range (the combination is
    /// checked by [`SymbolMapConfig::validate`]).
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "constellation" => {
                self.constellation = SymbolMapConstellation::from_index(value.as_choice()?);
            }
            "differential" => self.differential = value.as_bool()?,
            "variant" => {
                self.variant = if value.as_choice()? == 1 {
                    SymbolMapVariant::Pi4
                } else {
                    SymbolMapVariant::Plain
                };
            }
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
            "differential" => Ok(ParamValue::Bool(self.differential)),
            "variant" => Ok(ParamValue::Choice(match self.variant {
                SymbolMapVariant::Plain => 0,
                SymbolMapVariant::Pi4 => 1,
            })),
            _ => unknown(),
        }
    }
}

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
pub(crate) fn phase_index(bits: &[f32], pi4: bool) -> u8 {
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

/// The symbol of `K` soft bits (the first the most significant, above 0 is a 1).
#[inline]
fn symbol_of<const K: usize>(bits: &[f32; K]) -> u32 {
    bits.iter()
        .fold(0u32, |s, &b| (s << 1) | u32::from(b > 0.0))
}

/// The points of the symbols of `K` bits each in `bits` (whole symbols) to `out`.
#[inline]
fn map_points<const K: usize>(points: &[C32], bits: &[f32], out: &mut [C32]) {
    for (s, b) in out.iter_mut().zip(bits.chunks_exact(K)) {
        // The symbol has `K` bits, below the table length.
        *s = <&[f32; K]>::try_from(b)
            .ok()
            .and_then(|b| points.get(symbol_of(b) as usize))
            .copied()
            .unwrap_or_default();
    }
}

/// The level values of one axis of `q` by the Gray code of the axis: entry `g` is the level of
/// the level index whose Gray code is `g` (the per-axis form of the point table).
#[allow(
    clippy::indexing_slicing,
    reason = "const evaluation only: an index out of range fails the build, never at run time"
)]
const fn code_levels<const M: usize>(q: Qam) -> [f32; M] {
    let levels = q.levels();
    let mut out = [0.0; M];
    let mut g = 0;
    while g < M {
        out[g] = levels[level_of_code(g as u32) as usize];
        g += 1;
    }
    out
}

/// The 256-QAM levels by Gray code (16 values).
const QAM256_CODE_LEVELS: [f32; 16] = code_levels::<16>(Qam::Qam256);
/// The 1024-QAM levels by Gray code (32 values).
const QAM1024_CODE_LEVELS: [f32; 32] = code_levels::<32>(Qam::Qam1024);

/// The points of the symbols of `2 B` bits each in `bits` (whole symbols) to `out`, per axis:
/// the first `B` bits are the Gray code of the in-phase level, the last `B` that of the
/// quadrature level, each looked up in `levels` (the levels by Gray code). The same bits as the
/// point table of [`Qam`].
#[inline]
fn map_axes<const B: usize>(levels: &[f32], bits: &[f32], out: &mut [C32]) {
    for (s, b) in out.iter_mut().zip(bits.chunks_exact(2 * B)) {
        let (bi, bq) = b.split_at(B);
        let level = |half: &[f32]| {
            <&[f32; B]>::try_from(half)
                .ok()
                .and_then(|h| levels.get(symbol_of(h) as usize))
                .copied()
                .unwrap_or_default()
        };
        *s = C32::new(level(bi), level(bq));
    }
}

/// Soft bits to constellation points: `f32` soft bits in (one per sample), `C32` points out,
/// one per symbol of `bits_per_symbol` bits.
///
/// `bpsk`, `qpsk` and `8psk` take the sixteen constant `f32` points at multiples of 22.5
/// degrees of `psk_mod` (D649) and the Gray code [`PSK8_GRAY`](crate::PSK8_GRAY); `qam16` and
/// `qam64` the table of [`Qam`] (shared with `symbol_demap`, `equalize` and `costas`), `qam256`
/// and `qam1024` the same points from the levels per axis (D1509). With
/// `differential` the phase is a counter in units of 22.5 degrees (modulo 16), start 0
/// (D613); `pi4` takes the steps of D648. A call takes `min(input.len() / bits,
/// output.len())` whole symbols. Timing: constant ratio `1 / bits`, delay 0. The block needs no
/// storage.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, SymbolMapBlock, SymbolMapConfig, SymbolMapConstellation};
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = SymbolMapConfig {
///     constellation: SymbolMapConstellation::Qpsk,
///     ..SymbolMapConfig::default()
/// };
/// let mut map = SymbolMapBlock::new(&cfg, Rate::hz(2400)?, &mut arena)?;
/// let mut points = [C32::default(); 4];
/// let p = map.process(&[1.0, -1.0, -1.0, -1.0, 1.0], &mut points)?;
/// assert_eq!((p.consumed, p.produced), (4, 2)); // two symbols; the fifth bit waits
/// let h = core::f32::consts::FRAC_1_SQRT_2;
/// assert_eq!(points[..2], [C32::new(h, -h), C32::new(-h, -h)]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SymbolMapBlock<S: Storage> {
    config: SymbolMapConfig,
    bits: usize,
    /// The differential phase in units of `pi / 8`.
    phase: u8,
    _marker: PhantomData<fn() -> S>,
}

impl<S: Storage> core::fmt::Debug for SymbolMapBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SymbolMapBlock")
            .field("config", &self.config)
            .field("phase", &self.phase)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> SymbolMapBlock<S> {
    /// Creates the block; it needs no storage.
    ///
    /// # Errors
    /// As [`SymbolMapConfig::validate`].
    pub fn new(config: &SymbolMapConfig, _rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config: *config,
            bits: config.bits_per_symbol(),
            phase: 0,
            _marker: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    ///
    /// # Errors
    /// As [`SymbolMapConfig::validate`].
    pub fn storage_bytes(config: &SymbolMapConfig, _rate_in: Rate) -> Result<usize> {
        config.validate()?;
        Ok(Plan::default().bytes())
    }

    /// Output rate: the bit rate divided by the bits per symbol.
    ///
    /// # Errors
    /// As [`SymbolMapConfig::validate`]; `OutOfRange` when the rate overflows.
    pub fn rate_out(config: &SymbolMapConfig, rate_in: Rate) -> Result<Rate> {
        config.validate()?;
        Ok(rate_in.scale(Ratio::new(1, config.bits_per_symbol() as u64)?)?)
    }

    /// The configuration.
    pub fn config(&self) -> &SymbolMapConfig {
        &self.config
    }
}

impl<S: Storage> Block for SymbolMapBlock<S> {
    type In = f32;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[f32], output: &mut [C32]) -> Result<Progress> {
        let bits = self.bits;
        let syms = (input.len() / bits).min(output.len());
        let (input, output) = (
            super::head(input, syms * bits)?,
            super::head_mut(output, syms)?,
        );
        match self.config.constellation {
            // The bits of a symbol are a constant of the order, so the compiler unrolls them.
            SymbolMapConstellation::Qam16 => {
                map_points::<4>(Qam::Qam16.point_table(), input, output);
            }
            SymbolMapConstellation::Qam64 => {
                map_points::<6>(Qam::Qam64.point_table(), input, output);
            }
            // 256-QAM and 1024-QAM per axis: as fast as the point table or faster, without its
            // 2 KiB and 8 KiB (D1509).
            SymbolMapConstellation::Qam256 => {
                map_axes::<4>(&QAM256_CODE_LEVELS, input, output);
            }
            SymbolMapConstellation::Qam1024 => {
                map_axes::<5>(&QAM1024_CODE_LEVELS, input, output);
            }
            _ => {
                let pi4 = self.config.variant == SymbolMapVariant::Pi4;
                let mut phase = self.phase;
                for (s, b) in output.iter_mut().zip(input.chunks_exact(bits)) {
                    let k = phase_index(b, pi4);
                    *s = if self.config.differential {
                        phase = phase.wrapping_add(k) & 15;
                        point16(phase)
                    } else {
                        point16(k)
                    };
                }
                self.phase = phase;
            }
        }
        Ok(Progress {
            consumed: syms * bits,
            produced: syms,
        })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: Ratio::new(1, self.bits as u64).unwrap_or(Ratio::ONE),
            kind: RatioKind::Constant,
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {
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
        self.bits
    }
}
