//! `slice`: symbols to soft bits (`f32`, a value above 0 means bit 1) for real levels, BPSK,
//! QPSK and 4-FSK levels. A stateless per-symbol map without a kernel.

use core::marker::PhantomData;

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_core::C32;

use super::{multiply_rate, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::StreamElem;
use crate::storage::{Plan, Storage};

/// The names of the `constellation` choice, in index order.
const CONSTELLATION_OPTIONS: &[&str] = &["real", "bpsk", "qpsk", "fsk4"];

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
        the outer levels). A soft \
        bit above 0 means bit 1, at or below 0 bit 0, and its magnitude is the confidence; \
        this convention holds for the whole project (frame_decode reads it). Place it after \
        symbol_sync (and carrier recovery for PSK). A receiver that wants log-likelihood \
        ratios sets scale to 2 / sigma^2 of the noise per component. NaN and infinite \
        symbols pass through as NaN and infinite soft bits. No state, delay 0; the output \
        rate is the symbol rate times the bits per symbol.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::F32],
        description: "Symbols, one sample per symbol: f32 for real, C32 for bpsk and qpsk.",
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
            typical: "bpsk; real after an FSK discriminator, qpsk for QPSK, fsk4 for 4-FSK",
            mutability: Mutability::Build,
            description: "The symbol alphabet: real (f32, 1 bit), bpsk (C32, 1 bit), qpsk \
                (C32, 2 bits, re then im) or fsk4 (f32, 2 bits, -v then |v| - 2/3).",
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
}

impl SliceConstellation {
    /// Soft bits per input symbol.
    pub const fn bits_per_symbol(self) -> usize {
        match self {
            Self::Real | Self::Bpsk => 1,
            Self::Qpsk | Self::Fsk4 => 2,
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
            Self::Bpsk | Self::Qpsk => PortType::C32,
        }
    }

    const fn index(self) -> u16 {
        match self {
            Self::Real => 0,
            Self::Bpsk => 1,
            Self::Qpsk => 2,
            Self::Fsk4 => 3,
        }
    }

    const fn from_index(i: u16) -> Self {
        match i {
            0 => Self::Real,
            2 => Self::Qpsk,
            3 => Self::Fsk4,
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
/// (`bpsk`, `qpsk`).
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

/// Symbol slicer: `T` symbols in (`f32` for `real` and `fsk4`, `C32` for `bpsk` and `qpsk`), `f32` soft
/// bits out, `bits_per_symbol` per symbol; a soft bit `> 0` means bit 1 and its magnitude is
/// the confidence.
///
/// Each soft bit is one `f32` product `scale * component` (`scale` rounded once to `f32`); for
/// `fsk4` the components are `-v` and `|v| - 2/3` (the `f32` nearest to 2/3).
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
                reason: "real and fsk4 need f32 input; bpsk and qpsk need c32 input",
            })
        }
    }

    /// Creates the block; it needs no storage.
    ///
    /// # Errors
    /// `OutOfRange` for `scale`; `InvalidParameter` for a constellation that does not take
    /// `T` (`real` and `fsk4` need `f32`, `bpsk` and `qpsk` need `C32`).
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
            SliceConstellation::Qpsk => {
                for (o, &x) in output.chunks_exact_mut(2).zip(input) {
                    let (re, im) = x.parts();
                    if let [b0, b1] = o {
                        *b0 = s * re;
                        *b1 = s * im;
                    }
                }
                input.len().min(output.len() / 2)
            }
            SliceConstellation::Fsk4 => {
                for (o, &x) in output.chunks_exact_mut(2).zip(input) {
                    let v = x.parts().0;
                    if let [b0, b1] = o {
                        (*b0, *b1) = fsk4_bits(s, v);
                    }
                }
                input.len().min(output.len() / 2)
            }
            _ => {
                for (o, &x) in output.iter_mut().zip(input) {
                    *o = s * x.parts().0;
                }
                input.len().min(output.len())
            }
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
