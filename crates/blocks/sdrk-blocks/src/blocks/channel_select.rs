//! `channel_select`: one lane of a lane stream as an ordinary stream (M45 §5).

use sdrk::Progress;
use sdrk_core::C32;

use super::{unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, LaneElem, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::{Lanes, LanesRef};

/// The largest width of a lane stream (the most channels of `channelize`).
const MAX_WIDTH: i64 = sdrk_core::kernels::channelize::MAX_CHANNELS as i64;

/// Description of the `channel_select` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "channel_select",
    summary: "Copies one channel of a lane stream into an ordinary complex stream.",
    details: "Reads a lane stream of M lanes (the output of channelize) and writes lane channel \
        mod M of every time step: one output sample per step, at the step rate of the input \
        (2 fs / M after channelize). channel is the signed channel number: 0 is the centre \
        channel, -1 the one below, +1 the one above; the valid range is -M/2 to M/2 - 1 for an \
        input of M lanes. Any number of channel_select nodes may read the same channelize node, \
        also with the same channel. No delay, no storage.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::Lanes {
            elem: LaneElem::C32,
            width_param: "",
        }],
        description: "A lane stream (any width).",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32],
        description: "The selected channel, one sample per time step of the input.",
    }],
    params: &[ParamDesc {
        name: "channel",
        unit: Unit::None,
        kind: ParamKind::I64 {
            min: -(MAX_WIDTH / 2),
            max: MAX_WIDTH / 2 - 1,
            default: None,
        },
        typical: "0 (the centre channel), -1 the channel below",
        mutability: Mutability::Build,
        description: "Signed channel number, -M/2 to M/2 - 1 for an input of M lanes (lane \
            channel mod M). Required.",
    }],
    required: &["channel"],
};

/// Configuration of [`ChannelSelectBlock`].
///
/// ```
/// use sdrk_blocks::ChannelSelectConfig;
/// // Channel -1 of 16 lanes is lane 15, channel +7 is lane 7; +8 is out of range.
/// assert_eq!(ChannelSelectConfig { channel: -1 }.lane(16)?, 15);
/// assert_eq!(ChannelSelectConfig { channel: 7 }.lane(16)?, 7);
/// assert!(ChannelSelectConfig { channel: 8 }.lane(16).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChannelSelectConfig {
    /// Signed channel number: 0 the centre, negative below, positive above.
    pub channel: i64,
}

impl ChannelSelectConfig {
    /// The lane of the channel in a lane stream of `width` lanes: `channel mod width`.
    ///
    /// # Errors
    /// `InvalidParameter` (`width`) for width 0; `OutOfRange` (`channel`, with the range
    /// `-width/2` to `width - width/2 - 1`) for a channel outside it.
    pub fn lane(&self, width: usize) -> Result<usize> {
        if width == 0 {
            return Err(Error::InvalidParameter {
                name: "width",
                reason: "a lane stream has at least one lane",
            });
        }
        let w = i64::try_from(width).map_err(|_| Error::InvalidParameter {
            name: "width",
            reason: "the width of the lane stream is too large",
        })?;
        let (min, max) = (-(w / 2), w - w / 2 - 1);
        if !(min..=max).contains(&self.channel) {
            return Err(Error::OutOfRange {
                name: "channel",
                min: min as f64,
                max: max as f64,
                got: self.channel as f64,
            });
        }
        // In range, so the remainder is in 0 .. width.
        Ok(self.channel.rem_euclid(w).unsigned_abs() as usize)
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "channel" => self.channel = value.as_i64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "channel" => Ok(ParamValue::I64(self.channel)),
            _ => unknown(),
        }
    }
}

/// One lane of a lane stream as a `C32` stream: a strided copy of lane `channel mod M` of
/// every time step. The block needs no storage.
///
/// `Lanes<C32>` is no chain item, so a Rust user wires a channelizer and its readers by hand,
/// with one buffer for the lanes and one per reader:
///
/// ```
/// use sdrk_blocks::{
///     AlignedBytes, ArenaStorage, Block, ChannelSelectBlock, ChannelSelectConfig,
///     ChannelizeBlock, ChannelizeConfig, LanesMut, LanesRef,
/// };
/// use sdrk::C32;
/// let cfg = ChannelizeConfig::default(); // 16 channels
/// let mut mem = AlignedBytes([0u8; 4096]);
/// let mut arena = ArenaStorage::new(&mut mem.0);
/// let mut chan = ChannelizeBlock::new(&cfg, &mut arena)?;
/// let mut below = ChannelSelectBlock::new(&ChannelSelectConfig { channel: -1 }, 16)?;
///
/// // A tone one channel spacing below the centre (-fs / 16).
/// let input: Vec<C32> = (0..1024)
///     .map(|n| {
///         let p = -2.0 * core::f32::consts::PI * (n % 16) as f32 / 16.0;
///         C32::new(p.cos(), p.sin())
///     })
///     .collect();
/// let mut lanes = vec![C32::default(); 16 * 128]; // buffer 1: 128 steps of 16 lanes
/// let p = chan.process(&input, LanesMut::new(&mut lanes, 16)?)?;
/// assert_eq!(p.produced, 128);
/// let mut out = vec![C32::default(); 128]; // buffer 2: the channel below the centre
/// below.process(LanesRef::new(&lanes, 16)?, &mut out)?;
/// // After the filter has settled the channel carries the tone at full level.
/// assert!((out[127].norm_sqr() - 1.0).abs() < 1e-3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct ChannelSelectBlock {
    config: ChannelSelectConfig,
    width: usize,
    lane: usize,
}

impl ChannelSelectBlock {
    /// Creates the block for an input of `width` lanes.
    ///
    /// # Errors
    /// The errors of [`ChannelSelectConfig::lane`].
    pub fn new(config: &ChannelSelectConfig, width: usize) -> Result<Self> {
        let lane = config.lane(width)?;
        Ok(Self {
            config: *config,
            width,
            lane,
        })
    }

    /// The width of the input (lanes per step).
    pub fn width(&self) -> usize {
        self.width
    }

    /// The lane that the block copies (`channel mod width`).
    pub fn lane(&self) -> usize {
        self.lane
    }
}

impl Block for ChannelSelectBlock {
    type In = Lanes<C32>;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: LanesRef<'_, C32>, output: &mut [C32]) -> Result<Progress> {
        if input.width() != self.width {
            return Err(Error::Core(sdrk_core::Error::LengthMismatch {
                expected: self.width,
                got: input.width(),
            }));
        }
        let n = input.steps().min(output.len());
        let lane = input.data().iter().skip(self.lane).step_by(self.width);
        for (o, v) in output.iter_mut().take(n).zip(lane) {
            *o = *v;
        }
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
        if DESCRIPTION.param(name).is_some() {
            Err(Error::ReadOnlyParameter)
        } else {
            unknown()
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        1
    }
}
