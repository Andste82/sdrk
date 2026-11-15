//! `iq_offset`: delays one part of a complex signal (the imaginary or the real part) against
//! the other by a number of samples, the offset of OQPSK on both sides of a link.

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_core::C32;

use super::{check_count, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};

/// The largest `samples` of `iq_offset`.
pub const IQ_OFFSET_MAX_SAMPLES: usize = 512;

/// The names of the `part` choice, in index order.
const PART_OPTIONS: &[&str] = &["imag", "real"];

/// Description of the `iq_offset` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "iq_offset",
    summary: "Delays the imaginary or the real part of a complex signal against the other \
        part (the offset of OQPSK).",
    details: "Every output sample keeps the part that is not delayed of the input sample at \
        the same position and takes the delayed part from the input samples earlier (zeros \
        at the start and after a reset). It is the same operation on both sides of an offset \
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
        description: "The signal with the part delayed by samples samples.",
    }],
    params: &[
        ParamDesc {
            name: "samples",
            unit: Unit::Samples,
            kind: ParamKind::I64 {
                min: 1,
                max: IQ_OFFSET_MAX_SAMPLES as i64,
                default: None,
            },
            typical: "samples_per_symbol / 2 for OQPSK",
            mutability: Mutability::Build,
            description: "The delay of the part in samples, 1 to 512. Required.",
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
    required: &["samples"],
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
/// let cfg = IqOffsetConfig { samples: 4, part: IqPart::Imag };
/// assert!(cfg.validate().is_ok());
/// assert!(IqOffsetConfig { samples: 0, ..cfg }.validate().is_err());
/// assert!(IqOffsetConfig { samples: 513, ..cfg }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IqOffsetConfig {
    /// The delay in samples, 1 to 512 (required).
    pub samples: usize,
    /// The part that is delayed (default `imag`).
    pub part: IqPart,
}

impl Default for IqOffsetConfig {
    /// `samples` 4 (a placeholder: the pipeline requires the parameter), `part` `imag`.
    fn default() -> Self {
        Self {
            samples: 4,
            part: IqPart::Imag,
        }
    }
}

impl IqOffsetConfig {
    /// Checks the range of `samples`.
    ///
    /// # Errors
    /// `OutOfRange` for `samples` outside 1 to 512.
    pub fn validate(&self) -> Result<()> {
        check_count("samples", self.samples, 1, IQ_OFFSET_MAX_SAMPLES)
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "samples" => self.samples = value.as_usize()?,
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
            "samples" => Ok(ParamValue::I64(self.samples as i64)),
            "part" => Ok(ParamValue::Choice(match self.part {
                IqPart::Imag => 0,
                IqPart::Real => 1,
            })),
            _ => unknown(),
        }
    }
}

/// The delay of one part of a complex signal: `C32` in, `C32` out, one sample per sample.
///
/// The delayed part runs through a ring of `samples` `f32` in the storage (zeros at the start
/// and after [`reset`](Block::reset)): each output sample takes the part from the ring and puts
/// the part of its input sample in its place. A call takes `min(input.len(), output.len())`
/// samples. Timing: constant ratio 1, delay 0 (the other part is not delayed). Storage: the
/// ring.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, IqOffsetBlock, IqOffsetConfig, IqPart};
/// let cfg = IqOffsetConfig { samples: 2, part: IqPart::Imag };
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
    /// The ring of `samples` values of the delayed part.
    ring: S::Buf<f32>,
    /// The position in the ring of the next sample (below `samples`, at most 511).
    pos: u16,
    part: IqPart,
}

impl<S: Storage> core::fmt::Debug for IqOffsetBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IqOffsetBlock")
            .field("samples", &self.ring.as_ref().len())
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
            ring: storage.alloc::<f32>(config.samples)?,
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
        plan.add::<f32>(config.samples)?;
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
            samples: self.ring.as_ref().len(),
            part: self.part,
        }
    }

    /// Delays the part of `data` in place: the same as [`Block::process`] with `data` as input
    /// and output (the call of a composite on its own output, `psk_mod` with `offset`).
    ///
    /// ```
    /// use sdrk::{Rate, C32};
    /// use sdrk_blocks::{ArenaStorage, IqOffsetBlock, IqOffsetConfig, IqPart};
    /// let cfg = IqOffsetConfig { samples: 1, part: IqPart::Real };
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
        // Below the ring length, which is at most `IQ_OFFSET_MAX_SAMPLES`.
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
