//! The uniform block interface.

use sdrk::{Delay, Progress, Ratio, RatioKind};
use sdrk_device::SampleFormat;

use crate::describe::BlockDescription;
use crate::error::{Error, Result};
use crate::ports::Ports;

/// Exact timing of a block: output rate divided by input rate, whether it can change, and the
/// group delay in input samples.
///
/// A block with a packet output reports `kind: RatioKind::Packets { max_per_input }` and the
/// same bound as `ratio` (packets per input sample, for buffer sizing). A pure encoder with a
/// packet input and a stream output reports `kind: RatioKind::Source`; its `ratio` bounds the
/// output samples per packet (the output rate comes from the packet input of the pipeline).
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

/// A processing block. Its input and output are port types ([`Ports`]): a stream of one sample
/// type, a pair of streams ([`Two`](crate::Two)), frames ([`Frame`](crate::Frame)), packets
/// ([`Packets`](crate::Packets)) or a clock stream with packets ([`Clocked`](crate::Clocked)).
/// A stream block writes `type In = C32;` and
/// `fn process(&mut self, input: &[C32], output: &mut [C32])`.
///
/// Lockstep rule for [`Two`](crate::Two) ports: both slices of a pair have the same length
/// (otherwise `LengthMismatch`, nothing changes) and `Progress` counts pairs.
///
/// Frame rule for `Out = Frame`: `process` consumes input and stops right after completing a
/// frame; it then writes the frame into `FrameOut::values`, sets `*FrameOut::meta` to `Some`
/// and returns `produced: 1`. Without a completed frame `produced` is 0 and `*meta` is left
/// unchanged. `values.len() != frame_len()` is `LengthMismatch`; the content of `values` is
/// unspecified unless a frame is returned.
///
/// Blocks use caller-provided storage, never allocate and never panic. `process` follows the
/// contract of the rate-changing kernels: it consumes as much input as it can and produces at
/// most `output.len()` samples; the caller resubmits the unconsumed input.
pub trait Block: Send {
    /// Input port type.
    type In: Ports;
    /// Output port type.
    type Out: Ports;

    /// The machine-readable description of this block kind.
    fn description(&self) -> &'static BlockDescription;

    /// Consumes as much input as possible and produces at most `output.len()` samples.
    fn process(
        &mut self,
        input: <Self::In as Ports>::Ref<'_>,
        output: <Self::Out as Ports>::Mut<'_>,
    ) -> Result<Progress>;

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

    /// Maps a count of input samples since the last origin (construction or `reset`) to the
    /// number of output samples whose computation includes at least one of them: the first
    /// output sample that includes input sample `n` is `map_index(n)` (relative to the origin).
    ///
    /// The default uses the nominal ratio `L/M` of [`Block::timing`]: `ceil(rel_in * L / M)`.
    /// A block whose ratio is not exactly constant (the tuned resampler) overrides it. The
    /// result saturates at `u64::MAX`.
    fn map_index(&self, rel_in: u64) -> u64 {
        let ratio = self.timing().ratio;
        let v = (u128::from(rel_in) * u128::from(ratio.num())).div_ceil(u128::from(ratio.den()));
        u64::try_from(v).unwrap_or(u64::MAX)
    }

    /// Number of `f32` values of one frame for `Out = Frame`; 0 for every other block.
    fn frame_len(&self) -> usize {
        0
    }

    /// Number of packets of port b that the last `process` call consumed, for a block with a
    /// [`Clocked`](crate::Clocked) input (`Progress::consumed` counts its clock samples); 0
    /// for every other block. The caller resubmits the packets after these together with the
    /// unconsumed clock samples.
    fn consumed_packets(&self) -> usize {
        0
    }

    /// The reading `index` of the block, a measured value with its name (for example the
    /// carrier offset `offset_hz` of `afc`), or `None` past the last one. Readings are numbered
    /// from 0 without gaps; their number and names never change. The pipeline copies them
    /// into its statistics after every call. The default has none.
    ///
    /// ```
    /// use sdrk::{Rate, C32};
    /// use sdrk_blocks::{ArenaStorage, Block, ScaleBlock, ScaleConfig};
    /// let mut arena = ArenaStorage::new(&mut []);
    /// let block = ScaleBlock::<C32, _>::new(&ScaleConfig::default(), Rate::hz(1000)?, &mut arena)?;
    /// assert_eq!(block.reading(0), None);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn reading(&self, index: usize) -> Option<(&'static str, f64)> {
        let _ = index;
        None
    }
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
