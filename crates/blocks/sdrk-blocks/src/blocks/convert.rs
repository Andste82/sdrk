//! `convert`: raw device bytes to complex samples.

use core::marker::PhantomData;

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::kernels::convert::{Convert, ConvertReal};
use sdrk_device::{Container, Layout, SampleFormat, formats};

use super::{head, head_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::sample::{BlockReal, BlockSample};
use crate::storage::{Plan, Storage};

/// Description of the `convert` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "convert",
    summary: "Converts raw device bytes in a sample format to complex or real samples.",
    details: "Reads interleaved complex samples in any supported device format (8, 10, 12 or 16 \
        bit integers, 32 bit floats, packed formats, either byte order, optional I/Q swap) and \
        produces C32 or CQ15 samples. A format with a real layout (for example rs16 from a \
        real-sampling device) produces f32 or Q15 samples instead. The input is bytes; the \
        block only consumes whole packing groups (for example 3 bytes for two 12 bit samples) \
        and leaves a partial group for the next call. Planar layouts cannot be streamed in \
        arbitrary pieces and are rejected. The sample rate does not change.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::Raw],
        description: "Bytes in the configured sample format.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
        description: "Complex samples (real samples for a real format) at the same sample rate.",
    }],
    params: &[ParamDesc {
        name: "format",
        unit: Unit::None,
        kind: ParamKind::Format,
        typical: "cu8 for RTL-SDR, cs16 for HackRF-class devices",
        mutability: Mutability::Build,
        description: "Sample format of the input bytes, given by name (for example cu8).",
    }],
    required: &["format"],
};

/// Configuration of [`ConvertBlock`].
///
/// ```
/// use sdrk_blocks::ConvertConfig;
/// let cfg = ConvertConfig::default();
/// assert!(cfg.validate(sdrk::Rate::hz(2_400_000)?).is_ok());
/// # Ok::<(), sdrk::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConvertConfig {
    /// Format of the input bytes (interleaved complex, not 1-bit).
    pub format: SampleFormat,
}

impl Default for ConvertConfig {
    fn default() -> Self {
        Self {
            format: formats::CU8,
        }
    }
}

/// Samples and bytes of the smallest packing group of a streaming format of any supported
/// layout (interleaved complex or real).
pub(crate) fn stream_group(format: &SampleFormat) -> Result<(usize, usize)> {
    if !matches!(format.layout(), Layout::Interleaved | Layout::Real) {
        return Err(Error::InvalidParameter {
            name: "format",
            reason: "only interleaved complex and real formats can be streamed",
        });
    }
    if format.container() == Container::Bits1 {
        return Err(Error::InvalidParameter {
            name: "format",
            reason: "1-bit containers are not supported",
        });
    }
    for samples in 1..=64usize {
        if let Ok(bytes) = format.bytes_for_samples(samples) {
            return Ok((samples, bytes));
        }
    }
    Err(Error::InvalidParameter {
        name: "format",
        reason: "no packing group found",
    })
}

/// Samples and bytes of the smallest packing group of an interleaved complex streaming format
/// (a real format is an error).
pub(crate) fn packing_group(format: &SampleFormat) -> Result<(usize, usize)> {
    if format.layout() != Layout::Interleaved {
        return Err(Error::InvalidParameter {
            name: "format",
            reason: "only interleaved complex formats can be streamed",
        });
    }
    stream_group(format)
}

impl ConvertConfig {
    /// Checks the configuration.
    pub fn validate(&self, _rate_in: Rate) -> Result<()> {
        stream_group(&self.format).map(|_| ())
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "format" => self.format = value.as_format()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "format" => Ok(ParamValue::Format(self.format)),
            _ => unknown(),
        }
    }
}

/// Raw bytes to `C32` or `CQ15`.
///
/// `Progress` counts bytes in and samples out. The block needs no storage.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, ConvertBlock, ConvertConfig};
/// use sdrk::{Rate, CQ15};
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = ConvertConfig::default(); // cu8
/// let mut block = ConvertBlock::<CQ15, _>::new(&cfg, Rate::hz(2_400_000)?, &mut arena)?;
/// let mut out = [CQ15::default(); 2];
/// let p = block.process(&[255, 128, 0], &mut out)?; // the last byte is a partial sample
/// assert_eq!((p.consumed, p.produced), (2, 1));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct ConvertBlock<T: BlockSample, S: Storage> {
    format: SampleFormat,
    group_samples: usize,
    group_bytes: usize,
    _marker: PhantomData<fn() -> (T, S)>,
}

impl<T: BlockSample, S: Storage> ConvertBlock<T, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &ConvertConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        if config.format.layout() == Layout::Real {
            return Err(Error::InvalidParameter {
                name: "format",
                reason: "a real format needs ConvertRealBlock",
            });
        }
        let (group_samples, group_bytes) = stream_group(&config.format)?;
        Ok(Self {
            format: config.format,
            group_samples,
            group_bytes,
            _marker: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &ConvertConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate for the configuration (equal to the input sample rate).
    pub fn rate_out(config: &ConvertConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }
}

impl<T: BlockSample, S: Storage> Block for ConvertBlock<T, S>
where
    Sdrk: Convert<T>,
{
    type In = u8;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[u8], output: &mut [T]) -> Result<Progress> {
        let samples = self.format.samples_in_bytes(input.len()).min(output.len());
        let groups = samples / self.group_samples;
        let n = groups * self.group_samples;
        let bytes = groups * self.group_bytes;
        if n == 0 {
            return Ok(Progress::default());
        }
        let produced = <Sdrk as Convert<T>>::convert(
            &Sdrk,
            &self.format,
            head(input, bytes)?,
            head_mut(output, n)?,
        )?;
        Ok(Progress {
            consumed: bytes,
            produced,
        })
    }

    fn timing(&self) -> BlockTiming {
        let ratio =
            Ratio::new(self.group_samples as u64, self.group_bytes as u64).unwrap_or(Ratio::ONE);
        BlockTiming {
            ratio,
            kind: RatioKind::Constant,
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {}

    fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
        if DESCRIPTION.param(name).is_some() {
            Err(Error::ReadOnlyParameter)
        } else {
            unknown()
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        match name {
            "format" => Ok(ParamValue::Format(self.format)),
            _ => unknown(),
        }
    }

    fn min_input(&self) -> usize {
        self.group_bytes
    }
}

/// Raw bytes of a real format (for example `rs16`) to `f32` or `Q15`.
///
/// Like [`ConvertBlock`], `Progress` counts bytes in and samples out and the block needs no
/// storage. The pipeline kind `convert` chooses this block for a format with a real layout.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, ConvertConfig, ConvertRealBlock};
/// use sdrk::{Q15, Rate};
/// use sdrk_device::formats;
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = ConvertConfig { format: formats::RS16 };
/// let mut block = ConvertRealBlock::<Q15, _>::new(&cfg, Rate::hz(4_800_000)?, &mut arena)?;
/// let mut out = [Q15(0); 4];
/// let p = block.process(&[0x00, 0x40, 0x00, 0xC0, 0x7F], &mut out)?; // the last byte is partial
/// assert_eq!((p.consumed, p.produced), (4, 2));
/// assert_eq!(out[..2], [Q15(16384), Q15(-16384)]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct ConvertRealBlock<R: BlockReal, S: Storage> {
    format: SampleFormat,
    group_samples: usize,
    group_bytes: usize,
    _marker: PhantomData<fn() -> (R, S)>,
}

impl<R: BlockReal, S: Storage> ConvertRealBlock<R, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &ConvertConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        if config.format.layout() != Layout::Real {
            return Err(Error::InvalidParameter {
                name: "format",
                reason: "ConvertRealBlock needs a format with a real layout",
            });
        }
        let (group_samples, group_bytes) = stream_group(&config.format)?;
        Ok(Self {
            format: config.format,
            group_samples,
            group_bytes,
            _marker: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &ConvertConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate for the configuration (equal to the input sample rate).
    pub fn rate_out(config: &ConvertConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }
}

impl<R: BlockReal, S: Storage> Block for ConvertRealBlock<R, S>
where
    Sdrk: ConvertReal<R>,
{
    type In = u8;
    type Out = R;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[u8], output: &mut [R]) -> Result<Progress> {
        let samples = self.format.samples_in_bytes(input.len()).min(output.len());
        let groups = samples / self.group_samples;
        let n = groups * self.group_samples;
        let bytes = groups * self.group_bytes;
        if n == 0 {
            return Ok(Progress::default());
        }
        let produced = <Sdrk as ConvertReal<R>>::convert_real(
            &Sdrk,
            &self.format,
            head(input, bytes)?,
            head_mut(output, n)?,
        )?;
        Ok(Progress {
            consumed: bytes,
            produced,
        })
    }

    fn timing(&self) -> BlockTiming {
        let ratio =
            Ratio::new(self.group_samples as u64, self.group_bytes as u64).unwrap_or(Ratio::ONE);
        BlockTiming {
            ratio,
            kind: RatioKind::Constant,
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {}

    fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
        if DESCRIPTION.param(name).is_some() {
            Err(Error::ReadOnlyParameter)
        } else {
            unknown()
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        match name {
            "format" => Ok(ParamValue::Format(self.format)),
            _ => unknown(),
        }
    }

    fn min_input(&self) -> usize {
        self.group_bytes
    }
}
