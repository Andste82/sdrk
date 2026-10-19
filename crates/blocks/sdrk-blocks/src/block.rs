//! The uniform block interface.

use sdrk::{Delay, Progress, Ratio, RatioKind};
use sdrk_device::SampleFormat;

use crate::describe::BlockDescription;
use crate::error::{Error, Result};

/// Exact timing of a block: output rate divided by input rate, whether it can change, and the
/// group delay in input samples.
///
/// ```
/// use sdrk_blocks::BlockTiming;
/// use sdrk::{Delay, Ratio, RatioKind};
/// let t = BlockTiming { ratio: Ratio::new(1, 10)?, kind: RatioKind::Constant, delay: Delay::new(126, 2)? };
/// assert_eq!(t.delay.reduced().num(), 63);
/// # Ok::<(), sdrk::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockTiming {
    /// Output rate divided by input rate (`Ratio` of the nominal rate for a variable block).
    pub ratio: Ratio,
    /// Whether the ratio is fixed or can change at run time.
    pub kind: RatioKind,
    /// Delay between an input sample and its effect on the output, in input samples.
    pub delay: Delay,
}

/// The value of a parameter, as given by name through `set` and `set_param`.
///
/// ```
/// use sdrk_blocks::ParamValue;
/// assert_eq!(ParamValue::F64(1.5).as_f64(), Ok(1.5));
/// assert!(ParamValue::Bool(true).as_f64().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ParamValue {
    /// A floating-point number.
    F64(f64),
    /// An integer.
    I64(i64),
    /// A flag.
    Bool(bool),
    /// Index into the options of a `ParamKind::Choice`.
    Choice(u16),
    /// A sample format.
    Format(SampleFormat),
}

impl ParamValue {
    /// The number, or `WrongParamType`.
    pub fn as_f64(&self) -> Result<f64> {
        match self {
            Self::F64(v) => Ok(*v),
            _ => Err(Error::WrongParamType),
        }
    }

    /// The integer, or `WrongParamType`.
    pub fn as_i64(&self) -> Result<i64> {
        match self {
            Self::I64(v) => Ok(*v),
            _ => Err(Error::WrongParamType),
        }
    }

    /// The flag, or `WrongParamType`.
    pub fn as_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            _ => Err(Error::WrongParamType),
        }
    }

    /// The choice index, or `WrongParamType`.
    pub fn as_choice(&self) -> Result<u16> {
        match self {
            Self::Choice(v) => Ok(*v),
            _ => Err(Error::WrongParamType),
        }
    }

    /// The sample format, or `WrongParamType`.
    pub fn as_format(&self) -> Result<SampleFormat> {
        match self {
            Self::Format(v) => Ok(*v),
            _ => Err(Error::WrongParamType),
        }
    }

    /// The integer as `usize` (after the range check of the parameter it came from).
    pub(crate) fn as_usize(&self) -> Result<usize> {
        usize::try_from(self.as_i64()?).map_err(|_| Error::WrongParamType)
    }
}

/// Looks up `name` in the description, checks that a block may change it at run time and that
/// the value has the right type and range. Returns the canonical (static) name.
pub(crate) fn runtime_param(
    desc: &'static BlockDescription,
    name: &str,
    value: &ParamValue,
) -> Result<&'static str> {
    let p = desc.param(name).ok_or(Error::UnknownParameter)?;
    if p.mutability != crate::describe::Mutability::Runtime {
        return Err(Error::ReadOnlyParameter);
    }
    p.check(value)?;
    Ok(p.name)
}

/// A processing block with one streaming input and one streaming output (multi-port routing
/// is the pipeline's job).
///
/// Blocks use caller-provided storage, never allocate and never panic. `process` follows the
/// contract of the rate-changing kernels: it consumes as much input as it can and produces at
/// most `output.len()` samples; the caller resubmits the unconsumed input.
pub trait Block: Send {
    /// Input sample type.
    type In: Copy + Send;
    /// Output sample type.
    type Out: Copy + Send;

    /// The machine-readable description of this block kind.
    fn description(&self) -> &'static BlockDescription;

    /// Consumes as much input as possible and produces at most `output.len()` samples.
    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> Result<Progress>;

    /// Exact ratio out/in and group delay (in input samples).
    fn timing(&self) -> BlockTiming;

    /// Clears all history (used by the pipeline on gaps); the parameters stay.
    fn reset(&mut self);

    /// Changes a run-time parameter. The new value applies to the first sample of the next
    /// `process` call.
    ///
    /// # Errors
    /// `UnknownParameter`, `ReadOnlyParameter` (a build parameter), `WrongParamType`,
    /// `OutOfRange`.
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()>;

    /// Current value of a parameter (for a lossless readout of the configuration).
    ///
    /// # Errors
    /// `UnknownParameter`.
    fn param(&self, name: &str) -> Result<ParamValue>;

    /// Smallest input count that is guaranteed to produce at least one output (used for
    /// buffer sizing).
    fn min_input(&self) -> usize;
}

/// Metadata of a completed frame.
///
/// ```
/// use sdrk_blocks::FrameMeta;
/// let m = FrameMeta { first_sample: 0, averaged: 4, exponent: 0 };
/// assert_eq!(m.averaged, 4);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameMeta {
    /// Input index of the first sample of the frame's first window, counted since construction
    /// or the last reset.
    pub first_sample: u64,
    /// Number of windows that were averaged into the frame.
    pub averaged: u32,
    /// Power of two the values still have to be multiplied by (always 0: already applied).
    pub exponent: i32,
}

/// Blocks that output frames (spectrum, measurements) instead of a stream.
pub trait FrameBlock: Send {
    /// Input sample type.
    type In: Copy + Send;

    /// The machine-readable description of this block kind.
    fn description(&self) -> &'static BlockDescription;

    /// Number of `f32` values in a frame.
    fn frame_len(&self) -> usize;

    /// Consumes input; when a frame completes, writes it into `frame` and returns its metadata.
    /// At most one frame per call: the call stops consuming right after completing a frame.
    /// Returns the number of consumed input samples and the frame metadata, if any.
    ///
    /// # Errors
    /// `LengthMismatch` if `frame.len()` differs from [`FrameBlock::frame_len`]. The content of
    /// `frame` is unspecified unless a frame is returned.
    fn push(&mut self, input: &[Self::In], frame: &mut [f32])
    -> Result<(usize, Option<FrameMeta>)>;

    /// Clears all history and the sample counter; the parameters stay.
    fn reset(&mut self);

    /// Changes a run-time parameter (see [`Block::set_param`]).
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()>;

    /// Current value of a parameter (see [`Block::param`]).
    fn param(&self, name: &str) -> Result<ParamValue>;
}
