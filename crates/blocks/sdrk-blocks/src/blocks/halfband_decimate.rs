//! `halfband_decimate`: halfband low-pass filter and decimation by 2 (complex and real).

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::halfband::{
    HalfbandDecimate, HalfbandDecimateReal, HalfbandRealState, HalfbandState,
};
use sdrk_core::timing::Timing;

use super::fir::MAX_TAPS_LIMIT;
use super::{HZ_LIMIT, divide_rate, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::{HalfbandDesign, halfband_len, kaiser_beta};
use crate::error::{Error, Result};
use crate::sample::{BlockReal, BlockSample, TunedElem};
use crate::storage::{Plan, Storage};

/// The parameters shared by the blocks `halfband_decimate` and `real_to_iq`.
pub(crate) const PARAMS: &[ParamDesc] = &[
    ParamDesc {
        name: "transition_hz",
        unit: Unit::Hz,
        kind: ParamKind::F64 {
            min: 1e-6,
            max: HZ_LIMIT,
            default: None,
        },
        typical: "default 0.1 times the input rate",
        mutability: Mutability::Build,
        description: "Width of the transition band in Hz, centred on a quarter of the input \
            rate. The default is 0.1 * fs; it must be below fs / 2.",
    },
    ParamDesc {
        name: "attenuation_db",
        unit: Unit::Decibel,
        kind: ParamKind::F64 {
            min: 8.0,
            max: 200.0,
            default: Some(60.0),
        },
        typical: "60",
        mutability: Mutability::Build,
        description: "Stop-band attenuation in dB (at least 8).",
    },
    ParamDesc {
        name: "max_taps",
        unit: Unit::None,
        kind: ParamKind::I64 {
            min: 3,
            max: MAX_TAPS_LIMIT,
            default: Some(1023),
        },
        typical: "1023",
        mutability: Mutability::Build,
        description: "Upper limit of the number of taps; a design that needs more fails.",
    },
];

/// Description of the `halfband_decimate` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "halfband_decimate",
    summary: "Halfband low-pass filter and decimation by 2.",
    details: "A Kaiser-window halfband filter with a cutoff at a quarter of the input rate, \
        followed by a decimation by 2; every other tap is zero, so the block needs about a \
        quarter of the multiplications of a general decimating FIR. The number of taps (4 k + 3) \
        follows from the transition width and the attenuation. The usable band after the \
        decimation is plus or minus (fs / 4 - transition / 2) of the input rate; a signal above \
        fs / 4 + transition / 2 is attenuated by the stop-band attenuation. The group delay is \
        (taps - 1) / 2 input samples. Several halfband_decimate blocks in a row decimate by powers of \
        two, for example in front of a ddc. It accepts complex and real samples.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
        description: "Samples of any of the four types.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
        description: "Samples of the input type at half the input rate.",
    }],
    params: PARAMS,
    required: &[],
};

/// Configuration of [`HalfbandDecimateBlock`], [`HalfbandDecimateRealBlock`] and
/// [`RealToIqBlock`](crate::RealToIqBlock).
///
/// ```
/// use sdrk_blocks::HalfbandDecimateConfig;
/// let cfg = HalfbandDecimateConfig::default();
/// let rate = sdrk::Rate::hz(48_000)?;
/// assert_eq!(cfg.num_taps(rate)? % 4, 3);
/// assert!(HalfbandDecimateConfig { transition_hz: Some(30_000.0), ..cfg }.validate(rate).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HalfbandDecimateConfig {
    /// Width of the transition band in Hz; `None` means `0.1 * fs`.
    pub transition_hz: Option<f64>,
    /// Stop-band attenuation in dB.
    pub attenuation_db: f64,
    /// Upper limit of the number of taps.
    pub max_taps: usize,
}

impl Default for HalfbandDecimateConfig {
    fn default() -> Self {
        Self {
            transition_hz: None,
            attenuation_db: 60.0,
            max_taps: 1023,
        }
    }
}

impl HalfbandDecimateConfig {
    /// The transition width for the input rate with the default filled in.
    pub fn transition_for(&self, rate_in: Rate) -> f64 {
        self.transition_hz.unwrap_or(0.1 * super::hz(rate_in))
    }

    /// Number of taps (`4 k + 3`) for the input rate; also checks the whole configuration.
    ///
    /// # Errors
    /// `OutOfRange` if the transition width is not below half the input rate or the design
    /// needs more than `max_taps` taps; `InvalidParameter` for an unusable attenuation.
    pub fn num_taps(&self, rate_in: Rate) -> Result<usize> {
        let fs = super::hz(rate_in);
        let transition = self.transition_for(rate_in);
        if !(transition > 0.0 && transition < fs / 2.0) {
            return Err(Error::OutOfRange {
                name: "transition_hz",
                min: 0.0,
                max: fs / 2.0,
                got: transition,
            });
        }
        let n = halfband_len(transition, self.attenuation_db, fs)?;
        if n > self.max_taps {
            return Err(Error::OutOfRange {
                name: "max_taps",
                min: n as f64,
                max: MAX_TAPS_LIMIT as f64,
                got: self.max_taps as f64,
            });
        }
        Ok(n)
    }

    /// Checks the configuration against the input sample rate.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        self.num_taps(rate_in)?;
        divide_rate(rate_in, 2).map(|_| ())
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "transition_hz" => self.transition_hz = Some(value.as_f64()?),
            "attenuation_db" => self.attenuation_db = value.as_f64()?,
            "max_taps" => self.max_taps = value.as_usize()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter. `transition_hz` is an error while it is not set (its
    /// default depends on the sample rate; the blocks report the value in use).
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "transition_hz" => {
                self.transition_hz
                    .map(ParamValue::F64)
                    .ok_or(Error::InvalidParameter {
                        name: "transition_hz",
                        reason: "not set; the default depends on the sample rate",
                    })
            }
            "attenuation_db" => Ok(ParamValue::F64(self.attenuation_db)),
            "max_taps" => Ok(ParamValue::I64(self.max_taps as i64)),
            _ => unknown(),
        }
    }

    /// The configuration with the defaults filled in for the input rate.
    pub(crate) fn resolved(&self, rate_in: Rate) -> Self {
        Self {
            transition_hz: Some(self.transition_for(rate_in)),
            ..*self
        }
    }

    /// Designs the taps into a new buffer of `storage` (zeros stay exactly zero).
    pub(crate) fn alloc_taps<P, S>(
        &self,
        rate_in: Rate,
        storage: &mut S,
        convert: impl Fn(f64) -> Result<P>,
    ) -> Result<S::Buf<P>>
    where
        P: bytemuck::Pod + Default + Send,
        S: Storage,
    {
        let n = self.num_taps(rate_in)?;
        let design = HalfbandDesign::new(kaiser_beta(self.attenuation_db), n)?;
        let mut taps = storage.alloc::<P>(n)?;
        for (k, t) in taps.as_mut().iter_mut().enumerate() {
            *t = convert(design.tap(k))?;
        }
        Ok(taps)
    }

    /// Bytes of storage for `n` taps of `P` and a history of `n - 1` elements of `X`.
    pub(crate) fn plan_bytes<P: bytemuck::Pod, X: bytemuck::Pod>(
        &self,
        rate_in: Rate,
    ) -> Result<usize> {
        let n = self.num_taps(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<P>(n)?;
        plan.add::<X>(n - 1)?;
        Ok(plan.bytes())
    }
}

/// Timing of a halfband decimator with `taps` taps.
pub(crate) fn timing_of<S: Timing>(state: &S) -> BlockTiming {
    BlockTiming {
        ratio: state.ratio(),
        kind: state.ratio_kind(),
        delay: state.group_delay(),
    }
}

/// Halfband filter and decimation by 2 for `C32` and `CQ15`.
///
/// The first input sample after construction or `reset` produces the first output (like a
/// decimating FIR with factor 2).
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, HalfbandDecimateBlock, HalfbandDecimateConfig};
/// use sdrk::{Rate, C32};
/// let cfg = HalfbandDecimateConfig::default();
/// let rate = Rate::hz(48_000)?;
/// let mut mem = vec![0u8; HalfbandDecimateBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)? + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut block = HalfbandDecimateBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// let mut out = vec![C32::default(); 100];
/// let p = block.process(&[C32::new(1.0, 0.0); 200], &mut out)?;
/// assert_eq!((p.consumed, p.produced), (200, 100));
/// assert!((out[99].re - 1.0).abs() < 1e-5); // unity gain at DC
/// assert_eq!(HalfbandDecimateBlock::<C32, ArenaStorage<'_>>::rate_out(&cfg, rate)?, Rate::hz(24_000)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct HalfbandDecimateBlock<T: BlockSample, S: Storage> {
    config: HalfbandDecimateConfig,
    state: HalfbandState<T, S::Buf<T::Real>, S::Buf<T>>,
}

impl<T: BlockSample, S: Storage> core::fmt::Debug for HalfbandDecimateBlock<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HalfbandDecimateBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<T: BlockSample, S: Storage> HalfbandDecimateBlock<T, S> {
    /// Designs the taps and creates the block; taps and history come from `storage`.
    pub fn new(config: &HalfbandDecimateConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let taps = config.alloc_taps::<T::Real, S>(rate_in, storage, <T as TunedElem>::tap)?;
        let hist = storage.alloc::<T>(taps.as_ref().len() - 1)?;
        Ok(Self {
            config: config.resolved(rate_in),
            state: HalfbandState::new(taps, hist)?,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    pub fn storage_bytes(config: &HalfbandDecimateConfig, rate_in: Rate) -> Result<usize> {
        config.plan_bytes::<T::Real, T>(rate_in)
    }

    /// Output rate for the configuration (half the input rate).
    pub fn rate_out(config: &HalfbandDecimateConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        divide_rate(rate_in, 2)
    }

    /// Number of taps of the designed filter.
    pub fn num_taps(&self) -> usize {
        self.state.taps().len()
    }
}

impl<T: BlockSample, S: Storage> Block for HalfbandDecimateBlock<T, S>
where
    Sdrk: HalfbandDecimate<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        Ok(<Sdrk as HalfbandDecimate<T>>::halfband_decimate(
            &Sdrk,
            &mut self.state,
            input,
            output,
        )?)
    }

    fn timing(&self) -> BlockTiming {
        timing_of(&self.state)
    }

    fn reset(&mut self) {
        self.state.reset();
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        runtime_param(&DESCRIPTION, name, &value).map(|_| ())
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        2
    }
}

/// Halfband filter and decimation by 2 for `f32` and `Q15`.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, HalfbandDecimateConfig, HalfbandDecimateRealBlock};
/// use sdrk::Rate;
/// let cfg = HalfbandDecimateConfig::default();
/// let rate = Rate::hz(48_000)?;
/// let mut mem = vec![0u8; HalfbandDecimateRealBlock::<f32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)? + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut block = HalfbandDecimateRealBlock::<f32, _>::new(&cfg, rate, &mut arena)?;
/// let mut out = vec![0.0f32; 100];
/// let p = block.process(&[0.5f32; 200], &mut out)?;
/// assert_eq!((p.consumed, p.produced), (200, 100));
/// assert!((out[99] - 0.5).abs() < 1e-5);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct HalfbandDecimateRealBlock<R: BlockReal, S: Storage> {
    config: HalfbandDecimateConfig,
    state: HalfbandRealState<R, S::Buf<R>, S::Buf<R>>,
}

impl<R: BlockReal, S: Storage> core::fmt::Debug for HalfbandDecimateRealBlock<R, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HalfbandDecimateRealBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<R: BlockReal, S: Storage> HalfbandDecimateRealBlock<R, S> {
    /// Designs the taps and creates the block; taps and history come from `storage`.
    pub fn new(config: &HalfbandDecimateConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let taps = config.alloc_taps::<R, S>(rate_in, storage, <R as TunedElem>::tap)?;
        let hist = storage.alloc::<R>(taps.as_ref().len() - 1)?;
        Ok(Self {
            config: config.resolved(rate_in),
            state: HalfbandRealState::new(taps, hist)?,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    pub fn storage_bytes(config: &HalfbandDecimateConfig, rate_in: Rate) -> Result<usize> {
        config.plan_bytes::<R, R>(rate_in)
    }

    /// Output rate for the configuration (half the input rate).
    pub fn rate_out(config: &HalfbandDecimateConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        divide_rate(rate_in, 2)
    }

    /// Number of taps of the designed filter.
    pub fn num_taps(&self) -> usize {
        self.state.taps().len()
    }
}

impl<R: BlockReal, S: Storage> Block for HalfbandDecimateRealBlock<R, S>
where
    Sdrk: HalfbandDecimateReal<R>,
{
    type In = R;
    type Out = R;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[R], output: &mut [R]) -> Result<Progress> {
        Ok(<Sdrk as HalfbandDecimateReal<R>>::halfband_decimate_real(
            &Sdrk,
            &mut self.state,
            input,
            output,
        )?)
    }

    fn timing(&self) -> BlockTiming {
        timing_of(&self.state)
    }

    fn reset(&mut self) {
        self.state.reset();
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        runtime_param(&DESCRIPTION, name, &value).map(|_| ())
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        2
    }
}
