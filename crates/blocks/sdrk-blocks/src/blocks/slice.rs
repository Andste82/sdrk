//! `slice`: symbols to soft bits (`f32`, a value above 0 means bit 1) for real levels, BPSK,
//! QPSK, 4-FSK levels, 8PSK (max-log soft bits) and 16-/64-QAM (piecewise-linear max-log soft
//! bits). A stateless per-symbol map without a kernel.

use core::marker::PhantomData;

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_core::C32;
use sdrk_core::kernels::constellation::{QAM16_SQRT_E, QAM64_SQRT_E};

use super::{multiply_rate, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::StreamElem;
use crate::storage::{Plan, Storage};

/// The names of the `constellation` choice, in index order (`8psk` of M40.1 and `qam16`,
/// `qam64` of M44 appended, so the indices of M38 and M39 stay).
const CONSTELLATION_OPTIONS: &[&str] = &["real", "bpsk", "qpsk", "fsk4", "8psk", "qam16", "qam64"];

/// The Gray mapping of 8PSK (M40.1 §1): entry `g = b0 4 + b1 2 + b2` (bit 0 first, the most
/// significant) is the symbol index `k` of the point `exp(j (2 pi k / 8 + pi / 8))`:
/// `000 -> 0, 001 -> 1, 011 -> 2, 010 -> 3, 110 -> 4, 111 -> 5, 101 -> 6, 100 -> 7`, so
/// neighbouring points differ in one bit. Shared by `slice` (`8psk`), `psk_mod` and the tests.
///
/// ```
/// use sdrk_blocks::PSK8_GRAY;
/// assert_eq!(PSK8_GRAY, [0, 1, 3, 2, 7, 6, 4, 5]);
/// // Neighbouring symbols differ in exactly one bit.
/// let mut g_of_k = [0u8; 8];
/// for (g, &k) in PSK8_GRAY.iter().enumerate() {
///     g_of_k[k as usize] = g as u8;
/// }
/// for k in 0..8 {
///     assert_eq!((g_of_k[k] ^ g_of_k[(k + 1) % 8]).count_ones(), 1);
/// }
/// ```
pub const PSK8_GRAY: [u8; 8] = [0, 1, 3, 2, 7, 6, 4, 5];

/// The inverse of [`PSK8_GRAY`]: entry `k` is the Gray code `g` of the symbol index `k`.
pub(crate) const PSK8_CODE: [u8; 8] = [0, 1, 3, 2, 6, 7, 5, 4];

/// `cos(pi / 8)` rounded once to `f32`.
#[allow(
    clippy::excessive_precision,
    reason = "the literal is the shortest decimal that reads back as the f32 of cos(pi / 8)"
)]
pub(crate) const C8: f32 = 0.923_879_5;
/// `sin(pi / 8)` rounded once to `f32`.
#[allow(
    clippy::excessive_precision,
    reason = "the literal is the shortest decimal that reads back as the f32 of sin(pi / 8)"
)]
pub(crate) const S8: f32 = 0.382_683_43;

/// The 8PSK points `exp(j (2k + 1) pi / 8)` in index order `k = 0 .. 7`, each part the `f32`
/// of the exact value.
pub(crate) const PSK8_POINTS: [(f32, f32); 8] = [
    (C8, S8),
    (S8, C8),
    (-S8, C8),
    (-C8, S8),
    (-C8, -S8),
    (-S8, -C8),
    (S8, -C8),
    (C8, -S8),
];

/// The `f32` nearest to 2/3: the threshold between the inner (±1/3) and the outer (±1) levels
/// of 4-FSK.
const TWO_THIRDS: f32 = 2.0 / 3.0;

/// Smallest `scale` (D553).
const MIN_SCALE: f64 = 1e-9;
/// Largest `scale` (D553).
const MAX_SCALE: f64 = 1e9;

/// Description of the `slice` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "slice",
    summary: "Turns symbols into soft bits: f32 values whose sign is the bit and whose size is \
        the confidence.",
    details: "Maps every input symbol to bits-per-symbol soft bits in the order of the \
        constellation: real (f32 input, 1 bit: scale x), bpsk (C32 input, 1 bit: scale re), \
        qpsk (C32 input, 2 bits, Gray: scale re for bit 0, then scale im for bit 1), fsk4 (f32 \
        input, the 4-FSK levels +1, +1/3, -1/3, -1 of fsk_mod, 2 bits: scale (-v) for bit 0, \
        which is 1 for the negative levels, then scale (|v| - 2/3) for bit 1, which is 1 for \
        the outer levels), 8psk (C32 input, the points exp(j (2 pi k / 8 + pi / 8)) of psk_mod \
        with the Gray code b0 b1 b2, 3 bits, max-log: scale (min |y - p|^2 over the points \
        whose bit is 0 - the same minimum over the points whose bit is 1) / 4 for each bit), \
        qam16 and qam64 (C32 input, the unit-power Gray square QAM of qam_mod: the first half \
        of the bits from the in-phase level, the second half from the quadrature level, 4 or 6 \
        bits, the piecewise-linear max-log form per axis with v = component sqrt(E), E = 10 or \
        42: qam16 scale v, scale (2 - |v|); qam64 scale v, scale (4 - |v|), scale (2 - ||v| - \
        4|)). A soft bit above 0 means bit 1, at or below 0 bit 0, and its magnitude is the confidence; \
        this convention holds for the whole project (frame_decode reads it). Place it after \
        symbol_sync (and carrier recovery for PSK). A receiver that wants log-likelihood \
        ratios sets scale to 2 / sigma^2 of the noise per component. NaN and infinite \
        symbols pass through as NaN and infinite soft bits, except for 8psk: there an \
        infinite symbol, or one so large that its squared distances overflow (a part above \
        about 1.8e19), gives NaN soft bits (inf - inf). No state, delay 0; the output \
        rate is the symbol rate times the bits per symbol.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::F32],
        description: "Symbols, one sample per symbol: f32 for real and fsk4, C32 for bpsk, qpsk, \
            8psk, qam16 and qam64.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32],
        description: "Soft bits, bits-per-symbol samples per input symbol; above 0 is a 1.",
    }],
    params: &[
        ParamDesc {
            name: "constellation",
            unit: Unit::None,
            kind: ParamKind::Choice {
                options: CONSTELLATION_OPTIONS,
                default: 1,
            },
            typical: "bpsk; real after an FSK discriminator, qpsk for QPSK, fsk4 for 4-FSK, \
                8psk for 8PSK, qam16 and qam64 for QAM",
            mutability: Mutability::Build,
            description: "The symbol alphabet: real (f32, 1 bit), bpsk (C32, 1 bit), qpsk \
                (C32, 2 bits, re then im), fsk4 (f32, 2 bits, -v then |v| - 2/3), 8psk (C32, \
                3 bits, max-log, Gray), qam16 (C32, 4 bits) or qam64 (C32, 6 bits; both Gray, \
                in-phase bits first, piecewise-linear max-log).",
        },
        ParamDesc {
            name: "scale",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: MIN_SCALE,
                max: MAX_SCALE,
                default: Some(1.0),
            },
            typical: "1.0; 2 / sigma^2 for log-likelihood ratios",
            mutability: Mutability::Runtime,
            description: "Factor of every soft bit.",
        },
    ],
    required: &[],
};

/// The symbol alphabet of [`SliceBlock`].
///
/// ```
/// use sdrk_blocks::SliceConstellation;
/// assert_eq!(SliceConstellation::Qpsk.bits_per_symbol(), 2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SliceConstellation {
    /// Real levels (`f32` input), one bit: `scale x`.
    Real,
    /// BPSK (`C32` input), one bit: `scale re`.
    Bpsk,
    /// QPSK (`C32` input), two bits (Gray): `scale re` (bit 0), `scale im` (bit 1).
    Qpsk,
    /// 4-FSK levels `+1, +1/3, -1/3, -1` (`f32` input), two bits: `scale (-v)` (bit 0, 1 for
    /// the negative levels), `scale (|v| - 2/3)` (bit 1, 1 for the outer levels).
    Fsk4,
    /// 8PSK (`C32` input), three bits `b0 b1 b2` of the Gray code [`PSK8_GRAY`], max-log:
    /// `scale ((m0 - m1) / 4)` with `m0`, `m1` the smallest squared distance to the points
    /// whose bit is 0 and 1 (M40.1).
    Psk8,
    /// 16-QAM (`C32` input), four bits: the Gray code of the in-phase level (`b0 b1`), then of
    /// the quadrature level (`b2 b3`) of [`sdrk_core::kernels::constellation::Qam::Qam16`]; per
    /// axis with `v = component √10`: `scale v`, `scale (2 - |v|)` (M44 §2).
    Qam16,
    /// 64-QAM (`C32` input), six bits: the Gray code of the in-phase level (`b0 b1 b2`), then
    /// of the quadrature level (`b3 b4 b5`) of
    /// [`sdrk_core::kernels::constellation::Qam::Qam64`]; per axis with `v = component √42`:
    /// `scale v`, `scale (4 - |v|)`, `scale (2 - ||v| - 4|)` (M44 §2).
    Qam64,
}

impl SliceConstellation {
    /// Soft bits per input symbol.
    pub const fn bits_per_symbol(self) -> usize {
        match self {
            Self::Real | Self::Bpsk => 1,
            Self::Qpsk | Self::Fsk4 => 2,
            Self::Psk8 => 3,
            Self::Qam16 => 4,
            Self::Qam64 => 6,
        }
    }

    /// The input type the constellation takes.
    ///
    /// ```
    /// use sdrk_blocks::{PortType, SliceConstellation};
    /// assert_eq!(SliceConstellation::Real.input_type(), PortType::F32);
    /// assert_eq!(SliceConstellation::Bpsk.input_type(), PortType::C32);
    /// assert_eq!(SliceConstellation::Fsk4.input_type(), PortType::F32);
    /// ```
    pub const fn input_type(self) -> PortType {
        match self {
            Self::Real | Self::Fsk4 => PortType::F32,
            Self::Bpsk | Self::Qpsk | Self::Psk8 | Self::Qam16 | Self::Qam64 => PortType::C32,
        }
    }

    const fn index(self) -> u16 {
        match self {
            Self::Real => 0,
            Self::Bpsk => 1,
            Self::Qpsk => 2,
            Self::Fsk4 => 3,
            Self::Psk8 => 4,
            Self::Qam16 => 5,
            Self::Qam64 => 6,
        }
    }

    const fn from_index(i: u16) -> Self {
        match i {
            0 => Self::Real,
            2 => Self::Qpsk,
            3 => Self::Fsk4,
            4 => Self::Psk8,
            5 => Self::Qam16,
            6 => Self::Qam64,
            _ => Self::Bpsk,
        }
    }
}

/// Configuration of [`SliceBlock`].
///
/// ```
/// use sdrk_blocks::{SliceConfig, SliceConstellation};
/// let cfg = SliceConfig::default();
/// assert_eq!((cfg.constellation, cfg.scale), (SliceConstellation::Bpsk, 1.0));
/// assert!(cfg.validate().is_ok());
/// assert!(SliceConfig { scale: 0.0, ..cfg }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SliceConfig {
    /// The symbol alphabet.
    pub constellation: SliceConstellation,
    /// Factor of every soft bit (1e-9 to 1e9; changeable at run time).
    pub scale: f64,
}

impl Default for SliceConfig {
    fn default() -> Self {
        Self {
            constellation: SliceConstellation::Bpsk,
            scale: 1.0,
        }
    }
}

fn check_scale(v: f64) -> Result<()> {
    if (MIN_SCALE..=MAX_SCALE).contains(&v) {
        Ok(())
    } else {
        Err(Error::OutOfRange {
            name: "scale",
            min: MIN_SCALE,
            max: MAX_SCALE,
            got: v,
        })
    }
}

impl SliceConfig {
    /// Checks the range of `scale`.
    ///
    /// # Errors
    /// `OutOfRange` for `scale` outside 1e-9 to 1e9 (NaN included).
    pub fn validate(&self) -> Result<()> {
        check_scale(self.scale)
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "constellation" => {
                self.constellation = SliceConstellation::from_index(value.as_choice()?);
            }
            "scale" => self.scale = value.as_f64()?,
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
            "scale" => Ok(ParamValue::F64(self.scale)),
            _ => unknown(),
        }
    }
}

/// An input sample type of [`SliceBlock`]: `f32` (constellations `real` and `fsk4`) or `C32`
/// (`bpsk`, `qpsk`, `8psk`, `qam16`, `qam64`).
///
/// ```
/// use sdrk_blocks::SliceElem;
/// use sdrk::C32;
/// assert_eq!(C32::new(0.5, -2.0).parts(), (0.5, -2.0));
/// assert_eq!(0.25f32.parts(), (0.25, 0.0));
/// ```
pub trait SliceElem: StreamElem {
    /// Real and imaginary part (`f32`: the value and 0).
    fn parts(self) -> (f32, f32);
}

impl SliceElem for f32 {
    #[inline]
    fn parts(self) -> (f32, f32) {
        (self, 0.0)
    }
}

impl SliceElem for C32 {
    #[inline]
    fn parts(self) -> (f32, f32) {
        (self.re, self.im)
    }
}

/// The soft bits of the symbols in `input`, `scale * component` each: one per symbol (`re`) or,
/// with `qpsk`, two (`re`, then `im`). Takes `min(input.len(), output.len() / bits)` symbols
/// and returns that count (shared by `slice` and `psk_demod`).
pub(crate) fn soft_bits<T: SliceElem>(
    qpsk: bool,
    scale: f32,
    input: &[T],
    output: &mut [f32],
) -> usize {
    if qpsk {
        for (o, &x) in output.chunks_exact_mut(2).zip(input) {
            let (re, im) = x.parts();
            if let [b0, b1] = o {
                *b0 = scale * re;
                *b1 = scale * im;
            }
        }
        input.len().min(output.len() / 2)
    } else {
        for (o, &x) in output.iter_mut().zip(input) {
            *o = scale * x.parts().0;
        }
        input.len().min(output.len())
    }
}

/// Symbol slicer: `T` symbols in (`f32` for `real` and `fsk4`, `C32` for `bpsk`, `qpsk`,
/// `8psk`, `qam16` and `qam64`), `f32` soft bits out, `bits_per_symbol` per symbol; a soft bit `> 0` means bit 1 and
/// its magnitude is the confidence.
///
/// Each soft bit is one `f32` product `scale * component` (`scale` rounded once to `f32`); for
/// `fsk4` the components are `-v` and `|v| - 2/3` (the `f32` nearest to 2/3); for `8psk` they
/// are `(m0 - m1) * 0.25` from the squared distances to the points; for `qam16` and `qam64`
/// `v`, `2 - |v|` (and `4 - |v|`, `2 - ||v| - 4|`) per axis with `v = component * √E` (one
/// `f32` product, `√E` rounded once to `f32`; see [`SliceConstellation`]).
/// A call takes `min(input.len(), output.len() / bits_per_symbol)` symbols. Timing: constant
/// ratio `bits_per_symbol / 1`, delay 0. The block needs no storage.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, ParamValue, SliceBlock, SliceConfig, SliceConstellation};
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = SliceConfig { constellation: SliceConstellation::Qpsk, scale: 2.0 };
/// let mut slice = SliceBlock::<C32, _>::new(&cfg, Rate::hz(1200)?, &mut arena)?;
/// let mut bits = [0.0f32; 4];
/// let p = slice.process(&[C32::new(0.5, -1.0), C32::new(-0.25, 0.75)], &mut bits)?;
/// assert_eq!((p.consumed, p.produced), (2, 4));
/// assert_eq!(bits, [1.0, -2.0, -0.5, 1.5]); // re (bit 0), then im (bit 1)
/// slice.set_param("scale", ParamValue::F64(1.0))?;
/// // `real` needs f32 symbols
/// let real = SliceConfig { constellation: SliceConstellation::Real, scale: 1.0 };
/// assert!(SliceBlock::<C32, _>::new(&real, Rate::hz(1200)?, &mut arena).is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct SliceBlock<T: SliceElem, S: Storage> {
    config: SliceConfig,
    scale: f32,
    _marker: PhantomData<fn() -> (T, S)>,
}

impl<T: SliceElem, S: Storage> SliceBlock<T, S> {
    /// Checks the range and that the constellation takes `T`.
    fn check(config: &SliceConfig) -> Result<()> {
        config.validate()?;
        if config.constellation.input_type() == T::PORT_TYPE {
            Ok(())
        } else {
            Err(Error::InvalidParameter {
                name: "constellation",
                reason: "real and fsk4 need f32 input; bpsk, qpsk, 8psk, qam16 and qam64 need c32 \
                    input",
            })
        }
    }

    /// Creates the block; it needs no storage.
    ///
    /// # Errors
    /// `OutOfRange` for `scale`; `InvalidParameter` for a constellation that does not take
    /// `T` (`real` and `fsk4` need `f32`, `bpsk`, `qpsk`, `8psk`, `qam16` and `qam64` need
    /// `C32`).
    pub fn new(config: &SliceConfig, _rate_in: Rate, _storage: &mut S) -> Result<Self> {
        Self::check(config)?;
        Ok(Self {
            config: *config,
            scale: config.scale as f32,
            _marker: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    ///
    /// # Errors
    /// As [`SliceBlock::new`].
    pub fn storage_bytes(config: &SliceConfig, _rate_in: Rate) -> Result<usize> {
        Self::check(config)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate: the input rate times the bits per symbol.
    ///
    /// # Errors
    /// As [`SliceBlock::new`].
    pub fn rate_out(config: &SliceConfig, rate_in: Rate) -> Result<Rate> {
        Self::check(config)?;
        multiply_rate(rate_in, config.constellation.bits_per_symbol())
    }
}

impl<T: SliceElem, S: Storage> Block for SliceBlock<T, S> {
    type In = T;
    type Out = f32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [f32]) -> Result<Progress> {
        let s = self.scale;
        let n = match self.config.constellation {
            SliceConstellation::Qpsk => soft_bits(true, s, input, output),
            SliceConstellation::Fsk4 => {
                for (o, &x) in output.chunks_exact_mut(2).zip(input) {
                    let v = x.parts().0;
                    if let [b0, b1] = o {
                        (*b0, *b1) = fsk4_bits(s, v);
                    }
                }
                input.len().min(output.len() / 2)
            }
            SliceConstellation::Psk8 => {
                for (o, &x) in output.chunks_exact_mut(3).zip(input) {
                    let (re, im) = x.parts();
                    o.copy_from_slice(&psk8_bits(s, re, im));
                }
                input.len().min(output.len() / 3)
            }
            SliceConstellation::Qam16 => qam16_soft_bits(s, input, output),
            SliceConstellation::Qam64 => qam64_soft_bits(s, input, output),
            _ => soft_bits(false, s, input, output),
        };
        Ok(Progress {
            consumed: n,
            produced: n * self.config.constellation.bits_per_symbol(),
        })
    }

    fn timing(&self) -> BlockTiming {
        let bits = self.config.constellation.bits_per_symbol() as u64;
        BlockTiming {
            ratio: Ratio::new(bits, 1).unwrap_or(Ratio::ONE),
            kind: RatioKind::Constant,
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {}

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match runtime_param(&DESCRIPTION, name, &value)? {
            "scale" => {
                let v = value.as_f64()?;
                self.config.scale = v;
                self.scale = v as f32;
                Ok(())
            }
            _ => unknown(),
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        1
    }
}

/// The two soft bits of a 4-FSK level `v` with the factor `s`: `s (-v)` (bit 0, 1 for the
/// negative levels) and `s (|v| - 2/3)` (bit 1, 1 for the outer levels). Shared by `slice`
/// (`fsk4`) and `fsk_demod`.
#[inline]
pub(crate) fn fsk4_bits(s: f32, v: f32) -> (f32, f32) {
    (s * -v, s * (libm::fabsf(v) - TWO_THIRDS))
}

/// The three max-log soft bits `b0, b1, b2` of the 8PSK symbol `(re, im)` with the factor `s`
/// (M40.1 §1): for bit `i`, `s ((m0 - m1) * 0.25)` with `m0` (`m1`) the smallest squared
/// distance `dr dr + di di` (`dr = re - p.re`, `di = im - p.im`) to the points whose Gray bit
/// `i` is 0 (1). Each minimum starts at the first point of its set in index order and takes a
/// later point only when it is strictly nearer (`d < m`), so a tie keeps the lower index and a
/// NaN distance propagates; an infinite input gives `inf - inf`, a NaN soft bit (D642). Shared
/// by `slice` (`8psk`) and `psk_demod`.
#[inline]
pub(crate) fn psk8_bits(s: f32, re: f32, im: f32) -> [f32; 3] {
    let mut dist = [0.0f32; 8];
    for (d, &(pr, pi)) in dist.iter_mut().zip(&PSK8_POINTS) {
        let dr = re - pr;
        let di = im - pi;
        *d = dr * dr + di * di;
    }
    let mut out = [0.0f32; 3];
    for (bit, o) in out.iter_mut().enumerate() {
        let shift = 2 - bit;
        let (mut m0, mut m1) = (None::<f32>, None::<f32>);
        for (&d, &g) in dist.iter().zip(&PSK8_CODE) {
            let m = if (g >> shift) & 1 == 0 {
                &mut m0
            } else {
                &mut m1
            };
            *m = match *m {
                Some(v) if d < v => Some(d),
                Some(v) => Some(v),
                None => Some(d),
            };
        }
        let (m0, m1) = (m0.unwrap_or(0.0), m1.unwrap_or(0.0));
        *o = s * ((m0 - m1) * 0.25);
    }
    out
}

/// The components of the first `n` symbols of `input` as `f32`, in memory order (`re`, `im` for
/// `C32`; the value for `f32`), or none if `input` is shorter.
#[inline]
fn components<T: SliceElem>(input: &[T], n: usize) -> &[f32] {
    let per = core::mem::size_of::<T>() / core::mem::size_of::<f32>();
    bytemuck::try_cast_slice::<T, f32>(input)
        .ok()
        .and_then(|c| c.get(..n * per))
        .unwrap_or(&[])
}

/// The four soft bits of 16-QAM (M44 §2) for every symbol of `input` with the factor `s`: per
/// axis (in-phase first) with `v = component * √10`, `s v` (the first bit of the Gray code, 1
/// on the positive side) and `s (2 - |v|)` (the second bit, 1 for the inner levels). Takes
/// `min(input.len(), output.len() / 4)` symbols and returns that count (shared by `slice` and
/// `qam_demod`). The loop runs over the components in memory order, two soft bits each, which
/// the compiler vectorizes.
#[inline]
pub(crate) fn qam16_soft_bits<T: SliceElem>(s: f32, input: &[T], output: &mut [f32]) -> usize {
    let n = input.len().min(output.len() / 4);
    for (o, &x) in output.chunks_exact_mut(2).zip(components(input, n)) {
        let v = x * QAM16_SQRT_E;
        if let [b0, b1] = o {
            *b0 = s * v;
            *b1 = s * (2.0 - libm::fabsf(v));
        }
    }
    n
}

/// The six soft bits of 64-QAM (M44 §2) for every symbol of `input` with the factor `s`: per
/// axis (in-phase first) with `v = component * √42` and `a = |v|`, `s v` (1 on the positive
/// side), `s (4 - a)` (1 for the levels ±1, ±3) and `s (2 - |a - 4|)` (1 for ±3, ±5). Takes
/// `min(input.len(), output.len() / 6)` symbols and returns that count (shared by `slice` and
/// `qam_demod`). The loop runs over the components in memory order, three soft bits each.
#[inline]
pub(crate) fn qam64_soft_bits<T: SliceElem>(s: f32, input: &[T], output: &mut [f32]) -> usize {
    let n = input.len().min(output.len() / 6);
    for (o, &x) in output.chunks_exact_mut(3).zip(components(input, n)) {
        let v = x * QAM64_SQRT_E;
        let a = libm::fabsf(v);
        if let [b0, b1, b2] = o {
            *b0 = s * v;
            *b1 = s * (4.0 - a);
            *b2 = s * (2.0 - libm::fabsf(a - 4.0));
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::{C8, PSK8_CODE, PSK8_GRAY, PSK8_POINTS, S8, psk8_bits};

    #[test]
    fn the_8psk_tables_agree() {
        let a = core::f64::consts::PI / 8.0;
        assert_eq!(C8.to_bits(), (libm::cos(a) as f32).to_bits());
        assert_eq!(S8.to_bits(), (libm::sin(a) as f32).to_bits());
        for (k, &(re, im)) in PSK8_POINTS.iter().enumerate() {
            let a = f64::from(2 * k as u32 + 1) * core::f64::consts::PI / 8.0;
            assert_eq!(re.to_bits(), (libm::cos(a) as f32).to_bits(), "{k}");
            assert_eq!(im.to_bits(), (libm::sin(a) as f32).to_bits(), "{k}");
        }
        for (g, &k) in PSK8_GRAY.iter().enumerate() {
            assert_eq!(PSK8_CODE.get(usize::from(k)).copied(), Some(g as u8));
        }
    }

    #[test]
    fn every_point_slices_to_its_own_code() {
        for (k, &(re, im)) in PSK8_POINTS.iter().enumerate() {
            let g = PSK8_CODE.get(k).copied().unwrap_or(0);
            let bits = psk8_bits(1.0, re, im);
            for (i, b) in bits.iter().enumerate() {
                assert_eq!(*b > 0.0, (g >> (2 - i)) & 1 == 1, "point {k}, bit {i}: {b}");
            }
        }
        // NaN propagates; an infinite input gives NaN (inf - inf).
        assert!(psk8_bits(1.0, f32::NAN, 0.5).iter().all(|b| b.is_nan()));
        assert!(
            psk8_bits(1.0, f32::INFINITY, 0.5)
                .iter()
                .all(|b| b.is_nan())
        );
    }
}
