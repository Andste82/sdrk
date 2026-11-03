//! `stereo_matrix`: sum and difference to left and right.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::stereo::StereoMatrix;

use super::{head, head_mut, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{BlockDescription, PortDesc, PortType};
use crate::error::Result;
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Description of the `stereo_matrix` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "stereo_matrix",
    summary: "Turns sum and difference (S, D) into left and right: L = S + 2 D, R = S - 2 D.",
    details: "The input is a complex stream with the sum signal S (L+R) in re and the demodulated \
        difference signal D in im, where D is the product of the difference signal and the \
        regenerated 38 kHz carrier (so its amplitude is half of L-R). The output has the left \
        channel L = S + 2 D in re and the right channel R = S - 2 D in im, the interleaved \
        two-channel layout of sound cards. C32 computes in f32, CQ15 in integers with \
        saturation. The block is stateless and does not change the rate. It is the last step \
        of the stereo decoder of fm_stereo_demod; use it after your own filtering.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples: re is the sum S, im is the demodulated difference D.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "Stereo audio: re is the left channel, im the right channel.",
    }],
    params: &[],
    required: &[],
};

/// Sum and difference to left and right for `C32` and `CQ15`. The block needs no storage.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, StereoMatrixBlock};
/// use sdrk::{Rate, C32};
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut block = StereoMatrixBlock::<C32, _>::new(Rate::hz(48_000)?, &mut arena)?;
/// let mut out = [C32::default(); 1];
/// block.process(&[C32::new(0.5, 0.125)], &mut out)?;
/// assert_eq!(out[0], C32::new(0.75, 0.25));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct StereoMatrixBlock<T: BlockSample, S: Storage> {
    _marker: PhantomData<fn() -> (T, S)>,
}

impl<T: BlockSample, S: Storage> StereoMatrixBlock<T, S> {
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

impl<T: BlockSample, S: Storage> Block for StereoMatrixBlock<T, S>
where
    Sdrk: StereoMatrix<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as StereoMatrix<T>>::stereo_matrix(&Sdrk, head(input, n)?, head_mut(output, n)?)?;
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
