//! `to_complex`: real stream to a complex stream with zero imaginary part.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::math::Interleave;

use super::{head, head_mut, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{BlockDescription, PortDesc, PortType};
use crate::error::Result;
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Description of the `to_complex` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "to_complex",
    summary: "Turns a real stream into a complex stream with a zero imaginary part.",
    details: "Copies every real sample into the real part of a complex sample and sets the \
        imaginary part to 0: out = in + j 0. f32 gives C32, Q15 gives CQ15. The spectrum of the \
        result is the symmetric spectrum of the real signal; the rate does not change. Use it \
        to feed a real signal (for example audio) to a block that needs complex input, such \
        as a modulator or an oscillator clock. real_part is the inverse.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32, PortType::Q15],
        description: "Real samples.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples (f32 gives C32, Q15 gives CQ15) at the input rate.",
    }],
    params: &[],
    required: &[],
};

/// Real stream to a complex stream: `f32` to `C32`, `Q15` to `CQ15` (`T` is the complex output
/// type). The imaginary part is 0. The block needs no storage.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, ToComplexBlock};
/// use sdrk::{Rate, CQ15, Q15};
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut block = ToComplexBlock::<CQ15, _>::new(Rate::hz(48_000)?, &mut arena)?;
/// let mut out = [CQ15::default(); 2];
/// block.process(&[Q15(100), Q15(-200)], &mut out)?;
/// assert_eq!(out, [CQ15 { re: Q15(100), im: Q15(0) }, CQ15 { re: Q15(-200), im: Q15(0) }]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct ToComplexBlock<T: BlockSample, S: Storage> {
    _marker: PhantomData<fn() -> (T, S)>,
}

impl<T: BlockSample, S: Storage> ToComplexBlock<T, S> {
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

impl<T: BlockSample, S: Storage> Block for ToComplexBlock<T, S>
where
    Sdrk: Interleave<T>,
{
    type In = T::Real;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T::Real], output: &mut [T]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as Interleave<T>>::interleave(&Sdrk, head(input, n)?, None, head_mut(output, n)?)?;
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
