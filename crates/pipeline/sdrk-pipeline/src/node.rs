//! Type-erased nodes: object-safe wrappers around the generic blocks.

use sdrk::{C32, CQ15, Progress, Q15};
use sdrk_blocks::{
    Block, BlockDescription, BlockTiming, Error as BlockError, FrameBlock, FrameMeta, ParamValue,
};

/// Typed, borrowed input views used by the type-erased runner. The variant always matches the
/// input port type of the node (checked at build).
#[derive(Debug)]
pub enum NodeInput<'a> {
    /// Raw bytes in a sample format.
    Raw(&'a [u8]),
    /// Complex `f32`.
    C32(&'a [C32]),
    /// Complex Q15.
    CQ15(&'a [CQ15]),
    /// Real `f32`.
    F32(&'a [f32]),
    /// Real Q15.
    Q15(&'a [Q15]),
}

/// Typed, borrowed output views used by the type-erased runner.
#[derive(Debug)]
pub enum NodeOutput<'a> {
    /// Raw bytes in a sample format.
    Raw(&'a mut [u8]),
    /// Complex `f32`.
    C32(&'a mut [C32]),
    /// Complex Q15.
    CQ15(&'a mut [CQ15]),
    /// Real `f32`.
    F32(&'a mut [f32]),
    /// Real Q15.
    Q15(&'a mut [Q15]),
    /// One frame of `f32` values.
    Frame(&'a mut [f32]),
}

mod sealed {
    pub trait Sealed {}
}

/// An element type that can travel through a [`NodeInput`] or [`NodeOutput`] (`u8`, `C32`,
/// `CQ15`, `f32`, `Q15`). Sealed.
pub trait NodeElem: Copy + Send + 'static + sealed::Sealed {
    /// The slice of this type, if the view has the matching variant.
    fn input(i: NodeInput<'_>) -> Option<&[Self]>;
    /// The mutable slice of this type, if the view has the matching variant.
    fn output(o: NodeOutput<'_>) -> Option<&mut [Self]>;
}

macro_rules! elem {
    ($t:ty, $v:ident) => {
        impl sealed::Sealed for $t {}
        impl NodeElem for $t {
            fn input(i: NodeInput<'_>) -> Option<&[Self]> {
                match i {
                    NodeInput::$v(s) => Some(s),
                    _ => None,
                }
            }
            fn output(o: NodeOutput<'_>) -> Option<&mut [Self]> {
                match o {
                    NodeOutput::$v(s) => Some(s),
                    _ => None,
                }
            }
        }
    };
}

elem!(u8, Raw);
elem!(C32, C32);
elem!(CQ15, CQ15);
elem!(f32, F32);
elem!(Q15, Q15);

fn mismatch() -> BlockError {
    BlockError::Core(sdrk_core::Error::Internal("node port type mismatch"))
}

/// Object-safe wrapper around a `Block` or `FrameBlock`.
///
/// Implemented by [`StreamNode`] and [`FrameNode`]; a custom block can be wrapped the same way.
pub trait Node: Send {
    /// Stream nodes return produced samples in `Progress`; frame nodes return `produced = 1`
    /// and fill `meta` when a frame completed.
    fn process(
        &mut self,
        input: NodeInput<'_>,
        output: NodeOutput<'_>,
        meta: &mut Option<FrameMeta>,
    ) -> Result<Progress, BlockError>;
    /// Clears all history; the parameters stay.
    fn reset(&mut self);
    /// Changes a run-time parameter.
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<(), BlockError>;
    /// Current value of a parameter.
    fn param(&self, name: &str) -> Result<ParamValue, BlockError>;
    /// Timing of the block; `None` for frame nodes.
    fn timing(&self) -> Option<BlockTiming>;
    /// Smallest input count that is guaranteed to produce an output.
    fn min_input(&self) -> usize;
    /// The description of the block kind.
    fn description(&self) -> &'static BlockDescription;
}

/// A [`Node`] around a [`Block`].
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ScaleBlock, ScaleConfig};
/// use sdrk_pipeline::{Node, NodeInput, NodeOutput, StreamNode, VecStorage};
/// let block = ScaleBlock::<C32, _>::new(&ScaleConfig::default(), Rate::hz(1000)?, &mut VecStorage)?;
/// let mut node = StreamNode::new(block);
/// let input = [C32::new(1.0, 0.0); 4];
/// let mut output = [C32::default(); 4];
/// let mut meta = None;
/// let p = node.process(NodeInput::C32(&input), NodeOutput::C32(&mut output), &mut meta)?;
/// assert_eq!((p.consumed, p.produced), (4, 4));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct StreamNode<B: Block> {
    block: B,
}

impl<B: Block> StreamNode<B> {
    /// Wraps a block.
    pub fn new(block: B) -> Self {
        Self { block }
    }
}

impl<B> Node for StreamNode<B>
where
    B: Block,
    B::In: NodeElem,
    B::Out: NodeElem,
{
    fn process(
        &mut self,
        input: NodeInput<'_>,
        output: NodeOutput<'_>,
        _meta: &mut Option<FrameMeta>,
    ) -> Result<Progress, BlockError> {
        let input = <B::In as NodeElem>::input(input).ok_or_else(mismatch)?;
        let output = <B::Out as NodeElem>::output(output).ok_or_else(mismatch)?;
        self.block.process(input, output)
    }

    fn reset(&mut self) {
        self.block.reset();
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<(), BlockError> {
        self.block.set_param(name, value)
    }

    fn param(&self, name: &str) -> Result<ParamValue, BlockError> {
        self.block.param(name)
    }

    fn timing(&self) -> Option<BlockTiming> {
        Some(self.block.timing())
    }

    fn min_input(&self) -> usize {
        self.block.min_input()
    }

    fn description(&self) -> &'static BlockDescription {
        self.block.description()
    }
}

/// A [`Node`] around a [`FrameBlock`].
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{SpectrumBlock, SpectrumConfig};
/// use sdrk_pipeline::{FrameNode, Node, NodeInput, NodeOutput, VecStorage};
/// let cfg = SpectrumConfig { fft_size: 16, ..SpectrumConfig::default() };
/// let block = SpectrumBlock::<C32, _>::new(&cfg, Rate::hz(1000)?, &mut VecStorage)?;
/// let mut node = FrameNode::new(block);
/// let input = [C32::new(1.0, 0.0); 16];
/// let mut frame = [0.0f32; 16];
/// let mut meta = None;
/// let p = node.process(NodeInput::C32(&input), NodeOutput::Frame(&mut frame), &mut meta)?;
/// assert_eq!((p.consumed, p.produced), (16, 1));
/// assert!(meta.is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct FrameNode<B: FrameBlock> {
    block: B,
}

impl<B: FrameBlock> FrameNode<B> {
    /// Wraps a frame block.
    pub fn new(block: B) -> Self {
        Self { block }
    }
}

impl<B> Node for FrameNode<B>
where
    B: FrameBlock,
    B::In: NodeElem,
{
    fn process(
        &mut self,
        input: NodeInput<'_>,
        output: NodeOutput<'_>,
        meta: &mut Option<FrameMeta>,
    ) -> Result<Progress, BlockError> {
        let input = <B::In as NodeElem>::input(input).ok_or_else(mismatch)?;
        let NodeOutput::Frame(frame) = output else {
            return Err(mismatch());
        };
        let (consumed, done) = self.block.push(input, frame)?;
        *meta = done;
        Ok(Progress {
            consumed,
            produced: usize::from(done.is_some()),
        })
    }

    fn reset(&mut self) {
        self.block.reset();
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<(), BlockError> {
        self.block.set_param(name, value)
    }

    fn param(&self, name: &str) -> Result<ParamValue, BlockError> {
        self.block.param(name)
    }

    fn timing(&self) -> Option<BlockTiming> {
        None
    }

    fn min_input(&self) -> usize {
        1
    }

    fn description(&self) -> &'static BlockDescription {
        self.block.description()
    }
}
