//! `make_complex`: real and imaginary streams to one complex stream.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::math::Interleave;

use super::{head, head_mut, lockstep, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{BlockDescription, PortDesc, PortType};
use crate::error::Result;
use crate::ports::Two;
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Description of the `make_complex` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "make_complex",
    summary: "Combines a real and an imaginary stream into one complex stream.",
    details: "Takes two real streams of the same type and rate and forms the complex stream \
        out = re + j im, sample by sample. f32 inputs give C32, Q15 inputs give CQ15. Use it to \
        build a complex signal from two separate channels, for example the I and Q branches of a \
        modulator. The two inputs advance together; their lengths in a call must be equal. \
        real_part and imag_part are the inverse.",
    inputs: &[
        PortDesc {
            name: "re",
            types: &[PortType::F32, PortType::Q15],
            description: "Real parts.",
        },
        PortDesc {
            name: "im",
            types: &[PortType::F32, PortType::Q15],
            description: "Imaginary parts, the same type and rate as the real parts.",
        },
    ],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples (f32 gives C32, Q15 gives CQ15) at the input rate.",
    }],
    params: &[],
    required: &[],
};

/// Real and imaginary stream to a complex stream: `f32` pairs to `C32`, `Q15` pairs to `CQ15`
/// (`T` is the complex output type).
///
/// The inputs are a [`Two`] port: both slices have the same length, otherwise `process`
/// returns `LengthMismatch` and changes nothing. The block needs no storage.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, MakeComplexBlock};
/// use sdrk::{Rate, C32};
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut block = MakeComplexBlock::<C32, _>::new(Rate::hz(48_000)?, &mut arena)?;
/// let mut out = [C32::default(); 2];
/// let p = block.process((&[1.0, 2.0], &[3.0, 4.0]), &mut out)?;
/// assert_eq!((p.consumed, p.produced), (2, 2));
/// assert_eq!(out, [C32::new(1.0, 3.0), C32::new(2.0, 4.0)]);
/// assert!(block.process((&[1.0], &[]), &mut out).is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct MakeComplexBlock<T: BlockSample, S: Storage> {
    _marker: PhantomData<fn() -> (T, S)>,
}

impl<T: BlockSample, S: Storage> MakeComplexBlock<T, S> {
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

impl<T: BlockSample, S: Storage> Block for MakeComplexBlock<T, S>
where
    Sdrk: Interleave<T>,
{
    type In = Two<T::Real, T::Real>;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: (&[T::Real], &[T::Real]), output: &mut [T]) -> Result<Progress> {
        let (re, im) = input;
        let n = lockstep(re, im)?.min(output.len());
        <Sdrk as Interleave<T>>::interleave(
            &Sdrk,
            head(re, n)?,
            Some(head(im, n)?),
            head_mut(output, n)?,
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
