//! `imag_part`: imaginary part of a complex stream.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::math::Interleave;

use super::{head, head_mut, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{BlockDescription, PortDesc, PortType};
use crate::error::Result;
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Description of the `imag_part` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "imag_part",
    summary: "Outputs the imaginary part of a complex stream as a real stream.",
    details: "Copies the imaginary part of every complex sample to the output: C32 gives f32, CQ15 \
        gives Q15. The rate does not change. Use it to take one channel out of a complex \
        stream, for example the left channel of the stereo audio of fm_stereo_demod (re is \
        left, im is right). The other part is dropped.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32, PortType::Q15],
        description: "The imaginary parts (C32 gives f32, CQ15 gives Q15) at the input rate.",
    }],
    params: &[],
    required: &[],
};

/// The imaginary part of a complex stream: `C32` to `f32`, `CQ15` to `Q15`. The block needs no storage.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, ImagPartBlock};
/// use sdrk::{Rate, C32};
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut block = ImagPartBlock::<C32, _>::new(Rate::hz(48_000)?, &mut arena)?;
/// let mut out = [0.0f32; 2];
/// block.process(&[C32::new(1.0, 3.0), C32::new(2.0, 4.0)], &mut out)?;
/// assert_eq!(out, [3.0, 4.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct ImagPartBlock<T: BlockSample, S: Storage> {
    _marker: PhantomData<fn() -> (T, S)>,
}

impl<T: BlockSample, S: Storage> ImagPartBlock<T, S> {
    /// Creates the block; it has no parameters and needs no storage.
    pub fn new(_rate_in: Rate, _storage: &mut S) -> Result<Self> {
        Ok(Self {
            _marker: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(_rate_in: Rate) -> Result<usize> {
        Ok(Plan::default().bytes())
    }

    /// Output rate (equal to the input rate).
    pub fn rate_out(rate_in: Rate) -> Result<Rate> {
        Ok(rate_in)
    }
}

impl<T: BlockSample, S: Storage> Block for ImagPartBlock<T, S>
where
    Sdrk: Interleave<T>,
{
    type In = T;
    type Out = T::Real;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T::Real]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as Interleave<T>>::deinterleave(
            &Sdrk,
            head(input, n)?,
            None,
            Some(head_mut(output, n)?),
        )?;
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        unit_timing()
    }

    fn reset(&mut self) {}

    fn set_param(&mut self, _name: &str, _value: ParamValue) -> Result<()> {
        unknown()
    }

    fn param(&self, _name: &str) -> Result<ParamValue> {
        unknown()
    }

    fn min_input(&self) -> usize {
        1
    }
}
