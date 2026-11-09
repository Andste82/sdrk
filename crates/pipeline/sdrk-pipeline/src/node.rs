//! Type-erased nodes: object-safe wrappers around the generic blocks.

use sdrk::{C32, CQ15, Progress, Q15};
use sdrk_blocks::{
    Block, BlockDescription, BlockTiming, Clocked, Error as BlockError, Frame, FrameMeta, FrameOut,
    Lanes, LanesMut, LanesRef, Packet, Packets, ParamValue, Ports, StreamElem, Two,
};

/// One typed, borrowed stream: the data of one input port of a node.
///
/// ```
/// use sdrk_pipeline::NodeElem;
/// let samples = [1.0f32, 2.0];
/// assert!(matches!(NodeElem::F32(&samples), NodeElem::F32(s) if s.len() == 2));
/// ```
#[derive(Debug, Clone, Copy)]
pub enum NodeElem<'a> {
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

/// Typed, borrowed input views used by the type-erased runner. The variant always matches the
/// input port type of the node (checked at build).
#[derive(Debug)]
pub enum NodeInput<'a> {
    /// The two input ports of a two-input node, in lockstep (`a`, then `b`).
    Two {
        /// The stream of port `a`.
        a: NodeElem<'a>,
        /// The stream of port `b`.
        b: NodeElem<'a>,
    },
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
    /// Packets (the input of an encoder).
    Packets(&'a [Packet]),
    /// The input of a clocked encoder: the clock stream of port `a` and the queued packets of
    /// port `b`. Lockstep applies to the clock only.
    Clocked {
        /// The clock stream of port `a`.
        a: NodeElem<'a>,
        /// The packets of port `b` (any number, also none).
        packets: &'a [Packet],
    },
    /// A lane stream of `C32`: whole time steps of `width` values, step-major.
    Lanes {
        /// The values, `width` per step.
        data: &'a [C32],
        /// Number of lanes.
        width: usize,
    },
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
    /// Room for packets (the output of a decoder).
    Packets(&'a mut [Packet]),
    /// Room for whole time steps of a `C32` lane stream (`width` values per step).
    Lanes {
        /// The room, a multiple of `width` values.
        data: &'a mut [C32],
        /// Number of lanes.
        width: usize,
    },
}

/// How a stream element is taken out of a [`NodeInput`] or [`NodeOutput`]. Implemented for
/// every [`StreamElem`] (`u8`, `C32`, `CQ15`, `f32`, `Q15`); the supertrait seals it.
pub trait NodeSlice: StreamElem {
    /// The slice of this type, if the view has the matching variant.
    fn input(i: NodeInput<'_>) -> Option<&[Self]>;
    /// The slice of this type out of one stream of a pair, if it has the matching variant.
    fn elem(e: NodeElem<'_>) -> Option<&[Self]>;
    /// The mutable slice of this type, if the view has the matching variant.
    fn output(o: NodeOutput<'_>) -> Option<&mut [Self]>;
}

/// How the port views of a block are taken out of a [`NodeInput`] or [`NodeOutput`]: the slices
/// of a stream element (every [`NodeSlice`]) or the views of a lane stream (`Lanes<C32>`).
/// The trait of the ports of [`LanesNode`] and of the input of [`PacketOutNode`] (also
/// [`Packets`], the input of a packet to packet block such as `sensor_decode`, M46).
pub trait NodePort: Ports {
    /// The read-only view of this port type, if the input has the matching variant.
    fn input(i: NodeInput<'_>) -> Option<Self::Ref<'_>>;
    /// The writable view of this port type, if the output has the matching variant.
    fn output(o: NodeOutput<'_>) -> Option<Self::Mut<'_>>;
}

impl NodePort for Lanes<C32> {
    fn input(i: NodeInput<'_>) -> Option<LanesRef<'_, C32>> {
        match i {
            NodeInput::Lanes { data, width } => LanesRef::new(data, width).ok(),
            _ => None,
        }
    }
    fn output(o: NodeOutput<'_>) -> Option<LanesMut<'_, C32>> {
        match o {
            NodeOutput::Lanes { data, width } => LanesMut::new(data, width).ok(),
            _ => None,
        }
    }
}

impl NodePort for Packets {
    fn input(i: NodeInput<'_>) -> Option<&[Packet]> {
        match i {
            NodeInput::Packets(s) => Some(s),
            _ => None,
        }
    }
    fn output(o: NodeOutput<'_>) -> Option<&mut [Packet]> {
        match o {
            NodeOutput::Packets(s) => Some(s),
            _ => None,
        }
    }
}

macro_rules! elem {
    ($t:ty, $v:ident) => {
        impl NodePort for $t {
            fn input(i: NodeInput<'_>) -> Option<&[Self]> {
                <Self as NodeSlice>::input(i)
            }
            fn output(o: NodeOutput<'_>) -> Option<&mut [Self]> {
                <Self as NodeSlice>::output(o)
            }
        }
        impl NodeSlice for $t {
            fn input(i: NodeInput<'_>) -> Option<&[Self]> {
                match i {
                    NodeInput::$v(s) => Some(s),
                    _ => None,
                }
            }
            fn elem(e: NodeElem<'_>) -> Option<&[Self]> {
                match e {
                    NodeElem::$v(s) => Some(s),
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

/// Object-safe wrapper around a `Block` with stream or frame ports.
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
    /// Input samples since the last origin to output samples (see `Block::map_index`); frame
    /// nodes map every index to itself.
    fn map_index(&self, rel_in: u64) -> u64;
    /// Packets of port `b` that the last `process` call of a clocked node consumed (see
    /// `Block::consumed_packets`); 0 for every other node.
    fn consumed_packets(&self) -> usize {
        0
    }
    /// The reading `index` of the block (see `Block::reading`); `None` for nodes without
    /// readings.
    fn reading(&self, index: usize) -> Option<(&'static str, f64)> {
        let _ = index;
        None
    }
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

impl<B, I, O> Node for StreamNode<B>
where
    B: Block<In = I, Out = O>,
    I: NodeSlice,
    O: NodeSlice,
{
    fn process(
        &mut self,
        input: NodeInput<'_>,
        output: NodeOutput<'_>,
        _meta: &mut Option<FrameMeta>,
    ) -> Result<Progress, BlockError> {
        let input = <I as NodeSlice>::input(input).ok_or_else(mismatch)?;
        let output = <O as NodeSlice>::output(output).ok_or_else(mismatch)?;
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

    fn map_index(&self, rel_in: u64) -> u64 {
        self.block.map_index(rel_in)
    }

    fn reading(&self, index: usize) -> Option<(&'static str, f64)> {
        self.block.reading(index)
    }
}

/// A [`Node`] around a [`Block`] with two stream inputs (`In = Two<X, Y>`): the node of a
/// two-input kind. The runner passes both ports in [`NodeInput::Two`].
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{AddBlock};
/// use sdrk_pipeline::{Node, NodeElem, NodeInput, NodeOutput, PairNode, VecStorage};
/// let block = AddBlock::<C32, _>::new(Rate::hz(1000)?, &mut VecStorage)?;
/// let mut node = PairNode::new(block);
/// let a = [C32::new(1.0, 0.0); 4];
/// let b = [C32::new(0.0, 2.0); 4];
/// let mut out = [C32::default(); 4];
/// let mut meta = None;
/// let p = node.process(
///     NodeInput::Two { a: NodeElem::C32(&a), b: NodeElem::C32(&b) },
///     NodeOutput::C32(&mut out),
///     &mut meta,
/// )?;
/// assert_eq!((p.consumed, p.produced), (4, 4));
/// assert_eq!(out[0], C32::new(1.0, 2.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct PairNode<B: Block> {
    block: B,
}

impl<B: Block> PairNode<B> {
    /// Wraps a block with a [`Two`] input.
    pub fn new(block: B) -> Self {
        Self { block }
    }
}

impl<B, X, Y, O> Node for PairNode<B>
where
    B: Block<In = Two<X, Y>, Out = O>,
    X: NodeSlice,
    Y: NodeSlice,
    O: NodeSlice,
{
    fn process(
        &mut self,
        input: NodeInput<'_>,
        output: NodeOutput<'_>,
        _meta: &mut Option<FrameMeta>,
    ) -> Result<Progress, BlockError> {
        let NodeInput::Two { a, b } = input else {
            return Err(mismatch());
        };
        let a = <X as NodeSlice>::elem(a).ok_or_else(mismatch)?;
        let b = <Y as NodeSlice>::elem(b).ok_or_else(mismatch)?;
        let output = <O as NodeSlice>::output(output).ok_or_else(mismatch)?;
        self.block.process((a, b), output)
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

    fn map_index(&self, rel_in: u64) -> u64 {
        self.block.map_index(rel_in)
    }
}

/// A [`Node`] around a [`Block`] with `Out = Frame`.
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
pub struct FrameNode<B: Block<Out = Frame>> {
    block: B,
}

impl<B: Block<Out = Frame>> FrameNode<B> {
    /// Wraps a frame block.
    pub fn new(block: B) -> Self {
        Self { block }
    }
}

impl<B, I> Node for FrameNode<B>
where
    B: Block<In = I, Out = Frame>,
    I: NodeSlice,
{
    fn process(
        &mut self,
        input: NodeInput<'_>,
        output: NodeOutput<'_>,
        meta: &mut Option<FrameMeta>,
    ) -> Result<Progress, BlockError> {
        let input = <I as NodeSlice>::input(input).ok_or_else(mismatch)?;
        let NodeOutput::Frame(frame) = output else {
            return Err(mismatch());
        };
        *meta = None;
        self.block.process(
            input,
            FrameOut {
                values: frame,
                meta,
            },
        )
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

    fn map_index(&self, rel_in: u64) -> u64 {
        rel_in
    }
}

/// The methods of [`Node`] that only forward to the block.
macro_rules! forward_node {
    () => {
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

        fn map_index(&self, rel_in: u64) -> u64 {
            self.block.map_index(rel_in)
        }
    };
}

/// A [`Node`] around a [`Block`] with a packet input and a stream output (`In = Packets`):
/// the node of an encoder such as `frame_encode`. An empty packet slice asks the block for
/// its idle signal (pull mode).
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::{FrameEncodeBlock, FrameEncodeConfig, Packet, PacketFlags};
/// use sdrk_pipeline::{Node, NodeInput, NodeOutput, PacketInNode, VecStorage};
/// let block = FrameEncodeBlock::new(&FrameEncodeConfig::default(), Rate::hz(9600)?, &mut VecStorage)?;
/// let mut node = PacketInNode::new(block);
/// let packets = [Packet::new(0, PacketFlags::NONE, b"hi")?];
/// let mut out = [0.0f32; 200];
/// let mut meta = None;
/// let p = node.process(NodeInput::Packets(&packets), NodeOutput::F32(&mut out), &mut meta)?;
/// assert_eq!((p.consumed, p.produced), (1, 104));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct PacketInNode<B: Block> {
    block: B,
}

impl<B: Block> PacketInNode<B> {
    /// Wraps a block with a [`Packets`] input.
    pub fn new(block: B) -> Self {
        Self { block }
    }
}

impl<B, O> Node for PacketInNode<B>
where
    B: Block<In = Packets, Out = O>,
    O: NodeSlice,
{
    fn process(
        &mut self,
        input: NodeInput<'_>,
        output: NodeOutput<'_>,
        _meta: &mut Option<FrameMeta>,
    ) -> Result<Progress, BlockError> {
        let NodeInput::Packets(input) = input else {
            return Err(mismatch());
        };
        let output = <O as NodeSlice>::output(output).ok_or_else(mismatch)?;
        self.block.process(input, output)
    }

    forward_node!();
}

/// A [`Node`] around a [`Block`] with a packet output (`Out = Packets`): the node of a
/// decoder such as `frame_decode` (a stream input) or `band_pulse_detect` (a lane input,
/// [`Lanes<C32>`](Lanes), M45.1); the input is taken out of the [`NodeInput`] by [`NodePort`].
/// The packet indices are those of the block (counted from its construction or reset, in
/// samples of a stream input or time steps of a lane input); the runner moves them into the
/// index domain of the node's input. A block with a [`Packets`] input (`sensor_decode`) copies
/// the index of the packet it reads, which is already in the domain of the upstream node, so
/// the runner does not move it again (`NodeRt::packets_in`).
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::{FrameDecodeBlock, FrameDecodeConfig, Packet};
/// use sdrk_pipeline::{Node, NodeInput, NodeOutput, PacketOutNode, VecStorage};
/// let block = FrameDecodeBlock::new(&FrameDecodeConfig::default(), Rate::hz(9600)?, &mut VecStorage)?;
/// let mut node = PacketOutNode::new(block);
/// let noise = [-1.0f32; 100];
/// let mut out = [Packet::default(); 2];
/// let mut meta = None;
/// let p = node.process(NodeInput::F32(&noise), NodeOutput::Packets(&mut out), &mut meta)?;
/// assert_eq!((p.consumed, p.produced), (100, 0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct PacketOutNode<B: Block> {
    block: B,
}

impl<B: Block> PacketOutNode<B> {
    /// Wraps a block with a [`Packets`] output.
    pub fn new(block: B) -> Self {
        Self { block }
    }
}

impl<B, I> Node for PacketOutNode<B>
where
    B: Block<In = I, Out = Packets>,
    I: NodePort,
{
    fn process(
        &mut self,
        input: NodeInput<'_>,
        output: NodeOutput<'_>,
        _meta: &mut Option<FrameMeta>,
    ) -> Result<Progress, BlockError> {
        let input = <I as NodePort>::input(input).ok_or_else(mismatch)?;
        let NodeOutput::Packets(output) = output else {
            return Err(mismatch());
        };
        self.block.process(input, output)
    }

    forward_node!();
}

/// A [`Node`] around a [`Block`] with a [`Clocked`] input (`In = Clocked<A, Packets>`): an
/// encoder driven by the clock stream of port `a` that takes the packets of port `b`.
/// `Progress::consumed` counts clock samples; [`Node::consumed_packets`] reports the packets
/// the same call consumed.
#[derive(Debug)]
pub struct ClockedNode<B: Block> {
    block: B,
}

impl<B: Block> ClockedNode<B> {
    /// Wraps a block with a [`Clocked`] input.
    pub fn new(block: B) -> Self {
        Self { block }
    }
}

impl<B, A, O> Node for ClockedNode<B>
where
    B: Block<In = Clocked<A, Packets>, Out = O>,
    A: NodeSlice,
    O: NodeSlice,
{
    fn process(
        &mut self,
        input: NodeInput<'_>,
        output: NodeOutput<'_>,
        _meta: &mut Option<FrameMeta>,
    ) -> Result<Progress, BlockError> {
        let NodeInput::Clocked { a, packets } = input else {
            return Err(mismatch());
        };
        let a = <A as NodeSlice>::elem(a).ok_or_else(mismatch)?;
        let output = <O as NodeSlice>::output(output).ok_or_else(mismatch)?;
        self.block.process((a, packets), output)
    }

    fn consumed_packets(&self) -> usize {
        self.block.consumed_packets()
    }

    fn reading(&self, index: usize) -> Option<(&'static str, f64)> {
        self.block.reading(index)
    }

    forward_node!();
}

/// A [`Node`] around a [`Block`] with a lane port ([`Lanes<C32>`](Lanes)) on either side: a
/// channelizer (`In = C32`, `Out = Lanes<C32>`) or a block that reads lanes (`In = Lanes<C32>`).
/// The runner passes lane streams as [`NodeInput::Lanes`] and [`NodeOutput::Lanes`];
/// `Progress` counts time steps on a lane port.
///
/// ```
/// use sdrk::{C32, Delay, Progress, Ratio, RatioKind};
/// use sdrk_blocks::{all_descriptions, Block, BlockDescription, BlockTiming, Lanes, LanesRef, ParamValue};
/// use sdrk_pipeline::{LanesNode, Node, NodeInput, NodeOutput};
///
/// /// Sums the lanes of every step.
/// struct LaneSum;
/// impl Block for LaneSum {
///     type In = Lanes<C32>;
///     type Out = C32;
///     fn description(&self) -> &'static BlockDescription { all_descriptions()[0] }
///     fn process(&mut self, input: LanesRef<'_, C32>, output: &mut [C32]) -> sdrk_blocks::Result<Progress> {
///         let n = input.steps().min(output.len());
///         for (i, o) in output[..n].iter_mut().enumerate() {
///             *o = input.step(i).unwrap().iter().sum();
///         }
///         Ok(Progress { consumed: n, produced: n })
///     }
///     fn timing(&self) -> BlockTiming {
///         BlockTiming { ratio: Ratio::ONE, kind: RatioKind::Constant, delay: Delay::ZERO }
///     }
///     fn reset(&mut self) {}
///     fn set_param(&mut self, _: &str, _: ParamValue) -> sdrk_blocks::Result<()> {
///         Err(sdrk_blocks::Error::UnknownParameter)
///     }
///     fn param(&self, _: &str) -> sdrk_blocks::Result<ParamValue> {
///         Err(sdrk_blocks::Error::UnknownParameter)
///     }
///     fn min_input(&self) -> usize { 1 }
/// }
///
/// let mut node = LanesNode::new(LaneSum);
/// let lanes = [C32::new(1.0, 0.0); 8]; // two steps of four lanes
/// let mut out = [C32::default(); 2];
/// let mut meta = None;
/// let p = node.process(NodeInput::Lanes { data: &lanes, width: 4 }, NodeOutput::C32(&mut out), &mut meta)?;
/// assert_eq!((p.consumed, p.produced), (2, 2));
/// assert_eq!(out[1], C32::new(4.0, 0.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct LanesNode<B: Block> {
    block: B,
}

impl<B: Block> LanesNode<B> {
    /// Wraps a block with a lane port.
    pub fn new(block: B) -> Self {
        Self { block }
    }
}

impl<B, I, O> Node for LanesNode<B>
where
    B: Block<In = I, Out = O>,
    I: NodePort,
    O: NodePort,
{
    fn process(
        &mut self,
        input: NodeInput<'_>,
        output: NodeOutput<'_>,
        _meta: &mut Option<FrameMeta>,
    ) -> Result<Progress, BlockError> {
        let input = <I as NodePort>::input(input).ok_or_else(mismatch)?;
        let output = <O as NodePort>::output(output).ok_or_else(mismatch)?;
        self.block.process(input, output)
    }

    fn reading(&self, index: usize) -> Option<(&'static str, f64)> {
        self.block.reading(index)
    }

    forward_node!();
}
