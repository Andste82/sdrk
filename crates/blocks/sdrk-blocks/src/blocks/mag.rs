//! `mag`: magnitude (envelope) of complex samples.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::spectrum::Mag;

use super::{head, head_mut, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{BlockDescription, PortDesc, PortType};
use crate::error::{Error, Result};
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Description of the `mag` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "mag",
    summary: "Magnitude (envelope) of complex samples.",
    details: "Computes sqrt(re^2 + im^2) of every sample; the result is real. It is the \
        envelope detector of an AM receiver (after the channel filter, before the DC blocker \
        and the audio decimation) and uses the kernel of am_demod. With CQ15 the square root \
        is rounded to the nearest Q15 value and full-scale corners saturate. The block has no \
        parameters and keeps no state. Pitfall: the output keeps the DC offset of the carrier; \
        remove it with a high-pass if the audio is wanted.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32, PortType::Q15],
        description: "Magnitudes (f32 for C32 input, Q15 for CQ15 input), one per input.",
    }],
    params: &[],
    required: &[],
};

/// Configuration of [`MagBlock`] (it has no parameters).
///
/// ```
/// use sdrk_blocks::MagConfig;
/// assert!(MagConfig.validate(sdrk::Rate::hz(48_000)?).is_ok());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MagConfig;

impl MagConfig {
    /// Checks the configuration against the input sample rate (always valid).
    pub fn validate(&self, _rate_in: Rate) -> Result<()> {
        Ok(())
    }

    /// Sets a parameter; the block has none, so this is always `UnknownParameter`.
    pub fn set(&mut self, _name: &str, _value: ParamValue) -> Result<()> {
        unknown()
    }

    /// Current value of a parameter; always `UnknownParameter`.
    pub fn get(&self, _name: &str) -> Result<ParamValue> {
        unknown()
    }
}

/// Magnitude: `C32` to `f32` or `CQ15` to `Q15`.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, MagBlock, MagConfig};
/// use sdrk::{Rate, C32};
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut block = MagBlock::<C32, _>::new(&MagConfig, Rate::hz(48_000)?, &mut arena)?;
/// let mut out = [0.0f32; 1];
/// block.process(&[C32::new(3.0, 4.0)], &mut out)?;
/// assert_eq!(out, [5.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct MagBlock<T: BlockSample, S: Storage> {
    _marker: PhantomData<fn() -> (T, S)>,
}

impl<T: BlockSample, S: Storage> MagBlock<T, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &MagConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        Ok(Self {
            _marker: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &MagConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    pub fn rate_out(config: &MagConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }
}

impl<T: BlockSample, S: Storage> Block for MagBlock<T, S>
where
    Sdrk: Mag<T>,
{
    type In = T;
    type Out = T::Real;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T::Real]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as Mag<T>>::mag(&Sdrk, head(input, n)?, head_mut(output, n)?)?;
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        unit_timing()
    }

    fn reset(&mut self) {}

    fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
        let _ = name;
        Err(Error::UnknownParameter)
    }

    fn param(&self, _name: &str) -> Result<ParamValue> {
        unknown()
    }

    fn min_input(&self) -> usize {
        1
    }
}
