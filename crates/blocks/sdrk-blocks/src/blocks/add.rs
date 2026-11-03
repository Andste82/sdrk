//! `add`: sum of two streams.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::math::{Add, AddReal};

use super::{head, head_mut, lockstep, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{BlockDescription, PortDesc, PortType};
use crate::error::Result;
use crate::ports::Two;
use crate::sample::{BlockReal, BlockSample};
use crate::storage::{Plan, Storage};

/// Description of the `add` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "add",
    summary: "Adds two streams of the same type, sample by sample.",
    details: "Adds the two input streams component by component: out = a + b. Complex inputs give a complex sum, real inputs a real sum. Both inputs have the same type and sample rate and advance together; their lengths in a call must be equal. Q15 sums saturate at full scale instead of wrapping. Pitfall: the streams must be aligned in time and in rate; the block does not resample or delay.",
    inputs: &[
        PortDesc {
            name: "a",
            types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
            description: "First operand.",
        },
        PortDesc {
            name: "b",
            types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
            description: "Second operand, the same type and rate as the first.",
        },
    ],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
        description: "The sum a + b, same type and rate as the inputs.",
    }],
    params: &[],
    required: &[],
};

/// Sum of two streams for complex samples (`C32` or `CQ15`).
///
/// The inputs are a [`Two`] port: both slices have the same length, otherwise `process`
/// returns `LengthMismatch` and changes nothing. The block needs no storage.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, AddBlock};
/// use sdrk::{Rate, C32};
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut block = AddBlock::<C32, _>::new(Rate::hz(48_000)?, &mut arena)?;
/// let a = [C32::new(1.0, 2.0)];
/// let b = [C32::new(0.5, -4.0)];
/// let mut out = [C32::default(); 1];
/// let p = block.process((&a, &b), &mut out)?;
/// assert_eq!((p.consumed, p.produced), (1, 1));
/// assert_eq!(out[0], C32::new(1.5, -2.0));
/// // Unequal input lengths are rejected.
/// assert!(block.process((&a, &[]), &mut out).is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct AddBlock<T: BlockSample, S: Storage> {
    _marker: PhantomData<fn() -> (T, S)>,
}

impl<T: BlockSample, S: Storage> AddBlock<T, S> {
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

impl<T: BlockSample, S: Storage> Block for AddBlock<T, S>
where
    Sdrk: Add<T>,
{
    type In = Two<T, T>;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: (&[T], &[T]), output: &mut [T]) -> Result<Progress> {
        let (a, b) = input;
        let n = lockstep(a, b)?.min(output.len());
        <Sdrk as Add<T>>::add(&Sdrk, head(a, n)?, head(b, n)?, head_mut(output, n)?)?;
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

/// Sum of two streams for real samples (`f32` or `Q15`).
///
/// Like [`AddBlock`], the inputs are a [`Two`] port in lockstep and the block needs no storage.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, AddRealBlock};
/// use sdrk::{Rate, Q15};
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut block = AddRealBlock::<Q15, _>::new(Rate::hz(48_000)?, &mut arena)?;
/// let mut out = [Q15(0); 2];
/// block.process((&[Q15(20000), Q15(-3)], &[Q15(20000), Q15(5)]), &mut out)?;
/// assert_eq!(out, [Q15(32767), Q15(2)]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct AddRealBlock<R: BlockReal, S: Storage> {
    _marker: PhantomData<fn() -> (R, S)>,
}

impl<R: BlockReal, S: Storage> AddRealBlock<R, S> {
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

impl<R: BlockReal, S: Storage> Block for AddRealBlock<R, S>
where
    Sdrk: AddReal<R>,
{
    type In = Two<R, R>;
    type Out = R;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: (&[R], &[R]), output: &mut [R]) -> Result<Progress> {
        let (a, b) = input;
        let n = lockstep(a, b)?.min(output.len());
        <Sdrk as AddReal<R>>::add_real(&Sdrk, head(a, n)?, head(b, n)?, head_mut(output, n)?)?;
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
