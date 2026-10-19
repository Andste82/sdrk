//! `convert_out`: complex samples to raw device bytes.

use core::marker::PhantomData;

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::kernels::convert::ConvertOut;
use sdrk_device::{SampleFormat, formats};

use super::convert::packing_group;
use super::{head, head_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Description of the `convert_out` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "convert_out",
    summary: "Converts complex samples to raw device bytes in a sample format.",
    details: "Writes C32 or CQ15 samples as interleaved bytes in any supported device format, \
        rounding half up and saturating at full scale. The output is bytes; the block only \
        produces whole packing groups (for example two 12 bit samples in 3 bytes) and leaves \
        the remaining input for the next call. Planar and real layouts cannot be streamed in \
        arbitrary pieces and are rejected. The sample rate does not change.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::Raw],
        description: "Bytes in the configured sample format.",
    }],
    params: &[ParamDesc {
        name: "format",
        unit: Unit::None,
        kind: ParamKind::Format,
        typical: "cs8 for HackRF-class transmitters, cs16 for others",
        mutability: Mutability::Build,
        description: "Sample format of the output bytes, given by name (for example cs8).",
    }],
    required: &["format"],
};

/// Configuration of [`ConvertOutBlock`].
///
/// ```
/// use sdrk_blocks::ConvertOutConfig;
/// assert!(ConvertOutConfig::default().validate(sdrk::Rate::hz(1_000_000)?).is_ok());
/// # Ok::<(), sdrk::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConvertOutConfig {
    /// Format of the output bytes (interleaved complex, not 1-bit).
    pub format: SampleFormat,
}

impl Default for ConvertOutConfig {
    fn default() -> Self {
        Self {
            format: formats::CS8,
        }
    }
}

impl ConvertOutConfig {
    /// Checks the configuration.
    pub fn validate(&self, _rate_in: Rate) -> Result<()> {
        packing_group(&self.format).map(|_| ())
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

/// `C32` or `CQ15` to raw bytes.
///
/// `Progress` counts samples in and bytes out. The block needs no storage.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, ConvertOutBlock, ConvertOutConfig};
/// use sdrk::{Rate, C32};
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = ConvertOutConfig::default(); // cs8
/// let mut block = ConvertOutBlock::<C32, _>::new(&cfg, Rate::hz(1_000_000)?, &mut arena)?;
/// let mut bytes = [0u8; 4];
/// let p = block.process(&[C32::new(0.5, -0.5)], &mut bytes)?;
/// assert_eq!((p.consumed, p.produced), (1, 2));
/// assert_eq!(&bytes[..2], &[64, 192]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct ConvertOutBlock<T: BlockSample, S: Storage> {
    format: SampleFormat,
    group_samples: usize,
    group_bytes: usize,
    _marker: PhantomData<fn() -> (T, S)>,
}

impl<T: BlockSample, S: Storage> ConvertOutBlock<T, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &ConvertOutConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let (group_samples, group_bytes) = packing_group(&config.format)?;
        Ok(Self {
            format: config.format,
            group_samples,
            group_bytes,
            _marker: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &ConvertOutConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output sample rate for the configuration (equal to the input rate).
    pub fn rate_out(config: &ConvertOutConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }
}

impl<T: BlockSample, S: Storage> Block for ConvertOutBlock<T, S>
where
    Sdrk: ConvertOut<T>,
{
    type In = T;
    type Out = u8;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [u8]) -> Result<Progress> {
        let room = self.format.samples_in_bytes(output.len());
        let groups = input.len().min(room) / self.group_samples;
        let n = groups * self.group_samples;
        if n == 0 {
            return Ok(Progress::default());
        }
        let bytes = groups * self.group_bytes;
        let written = <Sdrk as ConvertOut<T>>::convert_out(
            &Sdrk,
            &self.format,
            head(input, n)?,
            head_mut(output, bytes)?,
        )?;
        Ok(Progress {
            consumed: n,
            produced: written,
        })
    }

    fn timing(&self) -> BlockTiming {
        let ratio =
            Ratio::new(self.group_bytes as u64, self.group_samples as u64).unwrap_or(Ratio::ONE);
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
        self.group_samples
    }
}
