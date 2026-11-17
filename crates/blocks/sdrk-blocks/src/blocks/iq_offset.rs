//! `iq_offset`: delays one part of a complex signal (the imaginary or the real part) against
//! the other by half a symbol, the offset of OQPSK on both sides of a link.

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_core::C32;

use super::{check_count, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};

/// The largest `samples_per_symbol` of `iq_offset` (a delay of at most 512 samples).
pub const IQ_OFFSET_MAX_SPS: usize = 1024;

/// The names of the `part` choice, in index order.
const PART_OPTIONS: &[&str] = &["imag", "real"];

/// Description of the `iq_offset` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "iq_offset",
    summary: "Delays the imaginary or the real part of a complex signal against the other \
        part by half a symbol (the offset of OQPSK).",
    details: "Every output sample keeps the part that is not delayed of the input sample at \
        the same position and takes the delayed part from the input sample \
        samples_per_symbol / 2 samples earlier (zeros at the start and after a reset); \
        samples_per_symbol must be even. It is the same operation on both sides of an offset \
        QPSK link: the transmitter delays the imaginary part by half a symbol after the pulse \
        shaping (part imag, as psk_mod with the variant offset does), a receiver delays the \
        real part by the same amount (part real) to align the two parts again. The rate does \
        not change; the reported delay is 0, because the part that is not delayed passes \
        straight through.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32],
        description: "The complex signal.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32],
        description: "The signal with the part delayed by half a symbol.",
    }],
    params: &[
        ParamDesc {
            name: "samples_per_symbol",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 2,
                max: IQ_OFFSET_MAX_SPS as i64,
                default: None,
            },
            typical: "8 (the samples per symbol of the modulator)",
            mutability: Mutability::Build,
            description: "Samples per symbol of the signal, even, 2 to 1024; the part is \
                delayed by half a symbol, samples_per_symbol / 2 samples. Required.",
        },
        ParamDesc {
            name: "part",
            unit: Unit::None,
            kind: ParamKind::Choice {
                options: PART_OPTIONS,
                default: 0,
            },
            typical: "imag in a transmitter, real in a receiver",
            mutability: Mutability::Build,
            description: "The part that is delayed: imag (the imaginary part) or real.",
        },
    ],
    required: &["samples_per_symbol"],
};

/// The part of the signal that [`IqOffsetBlock`] delays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum IqPart {
    /// The imaginary part (the quadrature part; the transmitter of OQPSK).
    #[default]
    Imag,
    /// The real part (the in-phase part; the receiver of OQPSK).
    Real,
}

/// Configuration of [`IqOffsetBlock`].
///
/// ```
/// use sdrk_blocks::{IqOffsetConfig, IqPart};
/// let cfg = IqOffsetConfig { samples_per_symbol: 8, part: IqPart::Imag };
/// assert!(cfg.validate().is_ok());
/// assert_eq!(cfg.delay(), 4);
/// assert!(IqOffsetConfig { samples_per_symbol: 0, ..cfg }.validate().is_err());
/// assert!(IqOffsetConfig { samples_per_symbol: 7, ..cfg }.validate().is_err()); // odd
/// assert!(IqOffsetConfig { samples_per_symbol: 1026, ..cfg }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IqOffsetConfig {
    /// Samples per symbol, even, 2 to 1024 (required); the delay is half of it.
    pub samples_per_symbol: usize,
    /// The part that is delayed (default `imag`).
    pub part: IqPart,
}

impl Default for IqOffsetConfig {
    /// `samples_per_symbol` 8 (a placeholder: the pipeline requires the parameter), `part`
    /// `imag`.
    fn default() -> Self {
        Self {
            samples_per_symbol: 8,
            part: IqPart::Imag,
        }
    }
}

impl IqOffsetConfig {
    /// Checks the range of `samples_per_symbol`.
    ///
    /// # Errors
    /// `OutOfRange` for `samples_per_symbol` outside 2 to 1024; `InvalidParameter` for an odd
    /// `samples_per_symbol`.
    pub fn validate(&self) -> Result<()> {
        check_count(
            "samples_per_symbol",
            self.samples_per_symbol,
            2,
            IQ_OFFSET_MAX_SPS,
        )?;
        check_even(self.samples_per_symbol)
    }

    /// The delay of the part in samples: half a symbol, `samples_per_symbol / 2`.
    pub fn delay(&self) -> usize {
        self.samples_per_symbol / 2
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`; `InvalidParameter` for an odd
    /// `samples_per_symbol`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "samples_per_symbol" => {
                let sps = value.as_usize()?;
                check_even(sps)?;
                self.samples_per_symbol = sps;
            }
            "part" => {
                self.part = if value.as_choice()? == 1 {
                    IqPart::Real
                } else {
                    IqPart::Imag
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
            "samples_per_symbol" => Ok(ParamValue::I64(self.samples_per_symbol as i64)),
            "part" => Ok(ParamValue::Choice(match self.part {
                IqPart::Imag => 0,
                IqPart::Real => 1,
            })),
            _ => unknown(),
        }
    }
}

/// An even `samples_per_symbol`, otherwise `InvalidParameter` (the delay is half a symbol).
fn check_even(sps: usize) -> Result<()> {
    if sps.is_multiple_of(2) {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name: "samples_per_symbol",
            reason: "samples_per_symbol must be even (the delay is half a symbol)",
        })
    }
}

/// The delay of one part of a complex signal by half a symbol: `C32` in, `C32` out, one sample
/// per sample.
///
/// The delayed part runs through a ring of `samples_per_symbol / 2` `f32` in the storage (zeros at the start
/// and after [`reset`](Block::reset)): each output sample takes the part from the ring and puts
/// the part of its input sample in its place. A call takes `min(input.len(), output.len())`
/// samples. Timing: constant ratio 1, delay 0 (the other part is not delayed). Storage: the
/// ring.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, IqOffsetBlock, IqOffsetConfig, IqPart};
/// let cfg = IqOffsetConfig { samples_per_symbol: 4, part: IqPart::Imag };
/// let rate = Rate::hz(9600)?;
/// let bytes = IqOffsetBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut b = IqOffsetBlock::new(&cfg, rate, &mut arena)?;
/// let x = [C32::new(1.0, 10.0), C32::new(2.0, 20.0), C32::new(3.0, 30.0)];
/// let mut y = [C32::default(); 3];
/// let p = b.process(&x, &mut y)?;
/// assert_eq!((p.consumed, p.produced), (3, 3));
/// assert_eq!(y, [C32::new(1.0, 0.0), C32::new(2.0, 0.0), C32::new(3.0, 10.0)]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct IqOffsetBlock<S: Storage> {
    /// The ring of `samples_per_symbol / 2` values of the delayed part.
    ring: S::Buf<f32>,
    /// The position in the ring of the next sample (below `samples_per_symbol / 2`, at most
    /// 511).
    pos: u16,
    part: IqPart,
}

impl<S: Storage> core::fmt::Debug for IqOffsetBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IqOffsetBlock")
            .field("samples_per_symbol", &(2 * self.ring.as_ref().len()))
            .field("part", &self.part)
            .field("pos", &self.pos)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> IqOffsetBlock<S> {
    /// Creates the block; the ring comes from `storage`.
    ///
    /// # Errors
    /// As [`IqOffsetConfig::validate`]; `Storage` when the storage is too small.
    pub fn new(config: &IqOffsetConfig, _rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            ring: storage.alloc::<f32>(config.delay())?,
            pos: 0,
            part: config.part,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// # Errors
    /// As [`IqOffsetConfig::validate`].
    pub fn storage_bytes(config: &IqOffsetConfig, _rate_in: Rate) -> Result<usize> {
        config.validate()?;
        let mut plan = Plan::default();
        plan.add::<f32>(config.delay())?;
        Ok(plan.bytes())
    }

    /// Output rate: the input rate.
    ///
    /// # Errors
    /// As [`IqOffsetConfig::validate`].
    pub fn rate_out(config: &IqOffsetConfig, rate_in: Rate) -> Result<Rate> {
        config.validate()?;
        Ok(rate_in)
    }

    /// The configuration of the block.
    pub fn config(&self) -> IqOffsetConfig {
        IqOffsetConfig {
            samples_per_symbol: 2 * self.ring.as_ref().len(),
            part: self.part,
        }
    }

    /// Delays the part of `data` in place: the same as [`Block::process`] with `data` as input
    /// and output (the call of a composite on its own output, `psk_mod` with `offset`).
    ///
    /// ```
    /// use sdrk::{Rate, C32};
    /// use sdrk_blocks::{ArenaStorage, IqOffsetBlock, IqOffsetConfig, IqPart};
    /// let cfg = IqOffsetConfig { samples_per_symbol: 2, part: IqPart::Real };
    /// let mut mem = [0u8; 8];
    /// let mut b = IqOffsetBlock::new(&cfg, Rate::hz(9600)?, &mut ArenaStorage::new(&mut mem))?;
    /// let mut x = [C32::new(1.0, 10.0), C32::new(2.0, 20.0)];
    /// b.process_in_place(&mut x);
    /// assert_eq!(x, [C32::new(0.0, 10.0), C32::new(1.0, 20.0)]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn process_in_place(&mut self, data: &mut [C32]) {
        let ring = self.ring.as_mut();
        let pos = usize::from(self.pos);
        let pos = match self.part {
            IqPart::Imag => swap_parts::<true>(ring, pos, data),
            IqPart::Real => swap_parts::<false>(ring, pos, data),
        };
        // Below the ring length, which is at most `IQ_OFFSET_MAX_SPS / 2`.
        self.pos = u16::try_from(pos).unwrap_or(0);
    }
}

/// Swaps the part (`IMAG`: the imaginary part, otherwise the real part) of `out` with the ring
/// from `pos` on and returns the new position.
#[inline]
fn swap_parts<const IMAG: bool>(ring: &mut [f32], mut pos: usize, out: &mut [C32]) -> usize {
    let mut rest = out;
    while !rest.is_empty() {
        let Some(tail) = ring.get_mut(pos..) else {
            return 0;
        };
        let n = tail.len().min(rest.len());
        let (now, later) = rest.split_at_mut(n);
        for (y, r) in now.iter_mut().zip(tail.iter_mut()) {
            if IMAG {
                core::mem::swap(&mut y.im, r);
            } else {
                core::mem::swap(&mut y.re, r);
            }
        }
        pos += n;
        if pos == ring.len() {
            pos = 0;
        }
        rest = later;
    }
    pos
}

impl<S: Storage> Block for IqOffsetBlock<S> {
    type In = C32;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[C32], output: &mut [C32]) -> Result<Progress> {
        let n = input.len().min(output.len());
        let out = super::head_mut(output, n)?;
        out.copy_from_slice(super::head(input, n)?);
        self.process_in_place(out);
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: Ratio::ONE,
            kind: RatioKind::Constant,
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {
        self.ring.as_mut().fill(0.0);
        self.pos = 0;
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
