//! `channelize`: the polyphase channelizer (M45 §4), `C32` to a lane stream of `channels`
//! channels.

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::channelize::{Channelize, ChannelizeState, history_len};
use sdrk_core::timing::Timing;

use super::{check_count, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, LaneElem, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::{
    CHANNELIZER_MAX_ATTENUATION_DB, CHANNELIZER_MAX_STOPBAND, CHANNELIZER_MIN_ATTENUATION_DB,
    channelizer_lowpass, channelizer_taps_per_branch,
};
use crate::error::{Error, Result};
use crate::ports::{Lanes, LanesMut};
use crate::storage::{Plan, Storage};

/// Smallest and largest channel count (the kernel's range; a power of two).
const CHANNELS: (usize, usize) = (
    sdrk_core::kernels::channelize::MIN_CHANNELS,
    sdrk_core::kernels::channelize::MAX_CHANNELS,
);
/// Smallest and largest prototype taps per branch (the kernel's range).
const TAPS_PER_BRANCH: (usize, usize) = (
    sdrk_core::kernels::channelize::MIN_TAPS_PER_BRANCH,
    sdrk_core::kernels::channelize::MAX_TAPS_PER_BRANCH,
);
/// The smallest edge of the description (the edges are fractions of the channel spacing above
/// 0; the design checks `passband < stopband`).
const MIN_EDGE: f64 = 1e-6;

/// Description of the `channelize` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "channelize",
    summary: "Polyphase channelizer: splits a complex band into channels lanes of a lane stream \
        at twice the channel spacing.",
    details: "An analysis filter bank with 2x oversampling (the M/2 hop form, kernel \
        channelize): per M/2 input samples the M branch filters of taps_per_branch taps each \
        give M values, which are shifted by the parity of the output step and transformed by \
        an M-point FFT. Lane k of the output is the channel at k fs / M from the input centre in \
        FFT order (lane 0 is the centre, lanes above M/2 are the negative frequencies; \
        channel_select takes a signed channel number) at the rate 2 fs / M. The prototype is a \
        Kaiser-windowed low-pass of M taps_per_branch taps with the passband edge at passband \
        fs / M and the stopband from stopband fs / M (fractions of the channel spacing fs / M). \
        The defaults (0.5 / 1.0) are a search grid: every narrowband signal lies in the flat \
        passband of at least one channel. A fixed channel plan uses sharper edges such as 0.4 / \
        0.6, which need more taps: if the Kaiser estimate needs more than M taps_per_branch \
        taps, the configuration is rejected with the required taps_per_branch. Group delay \
        (M taps_per_branch - 1) / 2 input samples. Storage: the taps, a history of 2 M \
        taps_per_branch samples and M/2 FFT twiddles.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32],
        description: "Complex samples of the band.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::Lanes {
            elem: LaneElem::C32,
            width_param: "channels",
        }],
        description: "The channels as a lane stream of channels lanes, one step per channels / 2 \
            input samples.",
    }],
    params: &[
        ParamDesc {
            name: "channels",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: CHANNELS.0 as i64,
                max: CHANNELS.1 as i64,
                default: Some(16),
            },
            typical: "16",
            mutability: Mutability::Build,
            description: "Number of channels M, a power of two from 4 to 256; the channel \
                spacing is fs / M.",
        },
        ParamDesc {
            name: "taps_per_branch",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: TAPS_PER_BRANCH.0 as i64,
                max: TAPS_PER_BRANCH.1 as i64,
                default: Some(8),
            },
            typical: "8 for 0.5 / 1.0, 24 for 0.4 / 0.6",
            mutability: Mutability::Build,
            description: "Prototype taps per branch; the prototype has channels taps_per_branch \
                taps and must be at least as long as the Kaiser estimate of the edges.",
        },
        ParamDesc {
            name: "passband",
            unit: Unit::Ratio,
            kind: ParamKind::F64 {
                min: MIN_EDGE,
                max: CHANNELIZER_MAX_STOPBAND,
                default: Some(0.5),
            },
            typical: "0.5 (search grid), 0.4 (channel plan)",
            mutability: Mutability::Build,
            description: "Passband edge as a fraction of the channel spacing, below stopband.",
        },
        ParamDesc {
            name: "stopband",
            unit: Unit::Ratio,
            kind: ParamKind::F64 {
                min: MIN_EDGE,
                max: CHANNELIZER_MAX_STOPBAND,
                default: Some(1.0),
            },
            typical: "1.0 (search grid), 0.6 (channel plan)",
            mutability: Mutability::Build,
            description: "Stopband edge as a fraction of the channel spacing, above passband \
                and at most 1.5 (the largest edge that does not alias into the passband at the \
                output rate 2 fs / M).",
        },
        ParamDesc {
            name: "attenuation_db",
            unit: Unit::Decibel,
            kind: ParamKind::F64 {
                min: CHANNELIZER_MIN_ATTENUATION_DB,
                max: CHANNELIZER_MAX_ATTENUATION_DB,
                default: Some(60.0),
            },
            typical: "60",
            mutability: Mutability::Build,
            description: "Stopband attenuation of the prototype in dB.",
        },
    ],
    required: &[],
};

/// Configuration of [`ChannelizeBlock`].
///
/// ```
/// use sdrk_blocks::ChannelizeConfig;
/// let grid = ChannelizeConfig::default(); // 16 channels, 8 taps per branch, 0.5 / 1.0
/// assert!(grid.validate().is_ok());
/// // A sharp channel plan needs 19 taps per branch at 60 dB.
/// let plan = ChannelizeConfig { passband: 0.4, stopband: 0.6, ..grid };
/// assert_eq!(plan.required_taps_per_branch()?, 19);
/// assert!(plan.validate().is_err());
/// assert!(ChannelizeConfig { taps_per_branch: 24, ..plan }.validate().is_ok());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelizeConfig {
    /// Number of channels `M` (a power of two, 4 to 256).
    pub channels: usize,
    /// Prototype taps per branch `L` (4 to 32).
    pub taps_per_branch: usize,
    /// Passband edge as a fraction of the channel spacing.
    pub passband: f64,
    /// Stopband edge as a fraction of the channel spacing.
    pub stopband: f64,
    /// Stopband attenuation in dB.
    pub attenuation_db: f64,
}

impl Default for ChannelizeConfig {
    fn default() -> Self {
        Self {
            channels: 16,
            taps_per_branch: 8,
            passband: 0.5,
            stopband: 1.0,
            attenuation_db: 60.0,
        }
    }
}

impl ChannelizeConfig {
    /// The taps per branch that the Kaiser estimate of the edges and the attenuation needs
    /// (`design::channelizer_taps_per_branch`).
    ///
    /// # Errors
    /// `InvalidParameter` for a channel count that is not a power of two from 4 to 256 or for
    /// edges outside `0 < passband < stopband <= 1.5`; `OutOfRange` for the attenuation.
    pub fn required_taps_per_branch(&self) -> Result<usize> {
        channelizer_taps_per_branch(
            self.channels,
            self.passband,
            self.stopband,
            self.attenuation_db,
        )
    }

    /// Checks the whole configuration (it does not depend on the sample rate).
    ///
    /// # Errors
    /// The errors of [`required_taps_per_branch`](Self::required_taps_per_branch);
    /// `OutOfRange` (`taps_per_branch`) with the required taps per branch as `min` when the
    /// prototype is shorter than the Kaiser estimate (or `taps_per_branch` is outside 4 to 32).
    pub fn validate(&self) -> Result<()> {
        check_count(
            "taps_per_branch",
            self.taps_per_branch,
            TAPS_PER_BRANCH.0,
            TAPS_PER_BRANCH.1,
        )?;
        let need = self.required_taps_per_branch()?;
        if self.taps_per_branch < need {
            return Err(Error::OutOfRange {
                name: "taps_per_branch",
                min: need as f64,
                max: TAPS_PER_BRANCH.1 as f64,
                got: self.taps_per_branch as f64,
            });
        }
        Ok(())
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "channels" => self.channels = value.as_usize()?,
            "taps_per_branch" => self.taps_per_branch = value.as_usize()?,
            "passband" => self.passband = value.as_f64()?,
            "stopband" => self.stopband = value.as_f64()?,
            "attenuation_db" => self.attenuation_db = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "channels" => Ok(ParamValue::I64(self.channels as i64)),
            "taps_per_branch" => Ok(ParamValue::I64(self.taps_per_branch as i64)),
            "passband" => Ok(ParamValue::F64(self.passband)),
            "stopband" => Ok(ParamValue::F64(self.stopband)),
            "attenuation_db" => Ok(ParamValue::F64(self.attenuation_db)),
            _ => unknown(),
        }
    }
}

/// The state type of the block.
type State<S> = ChannelizeState<
    C32,
    <S as Storage>::Buf<f32>,
    <S as Storage>::Buf<C32>,
    <S as Storage>::Buf<C32>,
>;

/// The polyphase channelizer: `C32` in, a lane stream of `channels` lanes out (one time step
/// per `channels / 2` input samples). A thin block over the kernel `channelize` and
/// `design::channelizer_prototype`.
///
/// ```
/// use sdrk_blocks::{AlignedBytes, ArenaStorage, Block, ChannelizeBlock, ChannelizeConfig, LanesMut};
/// use sdrk::C32;
/// let cfg = ChannelizeConfig::default(); // 16 channels
/// let mut mem = AlignedBytes([0u8; 4096]);
/// let bytes = ChannelizeBlock::<ArenaStorage<'_>>::storage_bytes(&cfg)?;
/// let mut arena = ArenaStorage::new(&mut mem.0[..bytes]);
/// let mut block = ChannelizeBlock::new(&cfg, &mut arena)?;
/// let input = [C32::new(1.0, 0.0); 64];
/// let mut room = [C32::default(); 16 * 8];
/// let p = block.process(&input, LanesMut::new(&mut room, 16)?)?;
/// assert_eq!((p.consumed, p.produced), (64, 8)); // 8 steps of 16 channels
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ChannelizeBlock<S: Storage> {
    config: ChannelizeConfig,
    state: State<S>,
}

impl<S: Storage> core::fmt::Debug for ChannelizeBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ChannelizeBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> ChannelizeBlock<S> {
    /// Designs the prototype and creates the block; the taps, the history and the twiddles
    /// come from `storage`.
    ///
    /// # Errors
    /// The errors of [`ChannelizeConfig::validate`] and of the storage.
    pub fn new(config: &ChannelizeConfig, storage: &mut S) -> Result<Self> {
        config.validate()?;
        let (m, l) = (config.channels, config.taps_per_branch);
        let lp = channelizer_lowpass(
            m,
            l,
            config.passband,
            config.stopband,
            config.attenuation_db,
        )?;
        let mut taps = storage.alloc::<f32>(m * l)?;
        for (i, t) in taps.as_mut().iter_mut().enumerate() {
            *t = lp.tap(i) as f32;
        }
        let hist = storage.alloc::<C32>(history_len(m, l))?;
        let twiddles = storage.alloc::<C32>(m / 2)?;
        Ok(Self {
            config: *config,
            state: ChannelizeState::new(taps, m, hist, twiddles)?,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address: `M L` taps (`f32`), a history of `2 M L` and `M / 2` twiddles (`C32`).
    ///
    /// ```
    /// use sdrk_blocks::{ArenaStorage, ChannelizeBlock, ChannelizeConfig};
    /// let bytes = ChannelizeBlock::<ArenaStorage<'_>>::storage_bytes(&ChannelizeConfig::default())?;
    /// assert_eq!(bytes, 128 * 4 + 256 * 8 + 8 * 8);
    /// # Ok::<(), sdrk_blocks::Error>(())
    /// ```
    ///
    /// # Errors
    /// The errors of [`ChannelizeConfig::validate`].
    pub fn storage_bytes(config: &ChannelizeConfig) -> Result<usize> {
        config.validate()?;
        let (m, l) = (config.channels, config.taps_per_branch);
        let mut plan = Plan::default();
        plan.add::<f32>(m * l)?;
        plan.add::<C32>(history_len(m, l))?;
        plan.add::<C32>(m / 2)?;
        Ok(plan.bytes())
    }

    /// The step rate of the output: `2 rate_in / M` time steps per second.
    ///
    /// # Errors
    /// The errors of [`ChannelizeConfig::validate`].
    pub fn rate_out(config: &ChannelizeConfig, rate_in: Rate) -> Result<Rate> {
        config.validate()?;
        Ok(rate_in.scale(sdrk::Ratio::new(2, config.channels as u64)?)?)
    }

    /// Number of channels (lanes) `M`.
    pub fn channels(&self) -> usize {
        self.config.channels
    }

    /// The prototype taps, time-reversed (as the kernel stores them).
    pub fn taps(&self) -> &[f32] {
        self.state.taps()
    }
}

impl<S: Storage> Block for ChannelizeBlock<S>
where
    Sdrk: Channelize<C32>,
{
    type In = C32;
    type Out = Lanes<C32>;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[C32], mut output: LanesMut<'_, C32>) -> Result<Progress> {
        let m = self.config.channels;
        if output.width() != m {
            return Err(Error::Core(sdrk_core::Error::LengthMismatch {
                expected: m,
                got: output.width(),
            }));
        }
        let p = Sdrk.channelize(&mut self.state, input, output.data_mut())?;
        Ok(Progress {
            consumed: p.consumed,
            produced: p.produced / m,
        })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: self.state.ratio(),
            kind: self.state.ratio_kind(),
            delay: self.state.group_delay(),
        }
    }

    fn reset(&mut self) {
        self.state.reset();
    }

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
        self.config.channels / 2
    }
}
