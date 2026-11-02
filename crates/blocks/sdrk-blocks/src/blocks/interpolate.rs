//! `interpolate`: zero insertion and image-rejection filter by an integer factor (complex and
//! real).

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::interp::{
    FirInterpRealState, FirInterpState, FirInterpolate, FirInterpolateReal,
};
use sdrk_core::timing::Timing;

use super::decimate::unset;
use super::fir::MAX_TAPS_LIMIT;
use super::{HZ_LIMIT, check_count, multiply_rate, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::DesignSpec;
use crate::error::{Error, Result};
use crate::sample::{BlockReal, BlockSample, TunedElem};
use crate::storage::{Plan, Storage};

/// Description of the `interpolate` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "interpolate",
    summary: "Interpolation by an integer factor with an image-rejection low-pass.",
    details: "Inserts factor - 1 zeros after every sample and removes the images with a \
        Kaiser-window low-pass designed at the output rate; the taps are multiplied by factor \
        so the gain stays 1. The output rate is the input rate times factor. Cutoff and \
        transition width are relative to the input rate (default 0.4 and 0.1 times the input \
        rate), so the pass band is the whole input band. It is the first stage of a transmit \
        chain (interpolate, then rotate) and the inverse of decimate. The polyphase \
        structure computes only the non-zero products. The group delay is (taps - 1) / (2 \
        factor) input samples. Pitfall: the stop band, cutoff plus half the transition width, \
        must not exceed half the input rate, otherwise the images are not removed.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
        description: "Samples at the input rate.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
        description: "Samples of the same type at the input rate times factor.",
    }],
    params: &[
        ParamDesc {
            name: "factor",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: 1024,
                default: None,
            },
            typical: "10 to go from 240 kHz to 2.4 MHz",
            mutability: Mutability::Build,
            description: "Interpolation factor, 1 to 1024. Required.",
        },
        ParamDesc {
            name: "cutoff_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "default 0.4 times the input rate",
            mutability: Mutability::Build,
            description: "Cutoff (-6 dB point) in Hz, relative to the input rate. The default is \
                0.4 * fs.",
        },
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
            description: "Width of the transition band in Hz. The default is 0.1 * fs. Cutoff \
                plus half the transition must not exceed half the input rate.",
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
            description: "Image attenuation in dB (at least 8).",
        },
        ParamDesc {
            name: "max_taps",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: MAX_TAPS_LIMIT,
                default: Some(1023),
            },
            typical: "1023",
            mutability: Mutability::Build,
            description: "Upper limit of the number of taps; a design that needs more fails.",
        },
    ],
    required: &["factor"],
};

/// Configuration of [`InterpolateBlock`] and [`InterpolateRealBlock`].
///
/// ```
/// use sdrk_blocks::InterpolateConfig;
/// let cfg = InterpolateConfig { factor: 10, ..InterpolateConfig::default() };
/// assert!(cfg.validate(sdrk::Rate::hz(240_000)?).is_ok());
/// let wide = InterpolateConfig { cutoff_hz: Some(120e3), ..cfg };
/// assert!(wide.validate(sdrk::Rate::hz(240_000)?).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterpolateConfig {
    /// Interpolation factor, 1 to 1024.
    pub factor: usize,
    /// Cutoff in Hz, relative to the input rate; `None` means `0.4 * fs`.
    pub cutoff_hz: Option<f64>,
    /// Transition width in Hz, relative to the input rate; `None` means `0.1 * fs`.
    pub transition_hz: Option<f64>,
    /// Image attenuation in dB.
    pub attenuation_db: f64,
    /// Upper limit of the number of taps.
    pub max_taps: usize,
}

impl Default for InterpolateConfig {
    fn default() -> Self {
        Self {
            factor: 10,
            cutoff_hz: None,
            transition_hz: None,
            attenuation_db: 60.0,
            max_taps: 1023,
        }
    }
}

impl InterpolateConfig {
    /// The filter specification (frequencies in Hz) with the defaults filled in.
    ///
    /// ```
    /// use sdrk_blocks::InterpolateConfig;
    /// let spec = InterpolateConfig::default().design_spec(sdrk::Rate::hz(240_000)?);
    /// assert_eq!((spec.cutoff_hz, spec.transition_hz), (96_000.0, 24_000.0));
    /// # Ok::<(), sdrk::Error>(())
    /// ```
    pub fn design_spec(&self, rate_in: Rate) -> DesignSpec {
        let fin = super::hz(rate_in);
        DesignSpec {
            cutoff_hz: self.cutoff_hz.unwrap_or(0.4 * fin),
            transition_hz: self.transition_hz.unwrap_or(0.1 * fin),
            attenuation_db: self.attenuation_db,
        }
    }

    /// Number of taps (designed at the output rate) for the input rate; also checks the whole
    /// configuration.
    pub fn num_taps(&self, rate_in: Rate) -> Result<usize> {
        check_count("factor", self.factor, 1, 1024)?;
        let fin = super::hz(rate_in);
        let spec = self.design_spec(rate_in);
        // The images start at the input Nyquist frequency; the stop band must not pass it.
        if spec.cutoff_hz + spec.transition_hz / 2.0 > fin / 2.0 {
            return Err(Error::OutOfRange {
                name: "cutoff_hz",
                min: 0.0,
                max: fin / 2.0 - spec.transition_hz / 2.0,
                got: spec.cutoff_hz,
            });
        }
        spec.num_taps(fin * self.factor as f64, self.max_taps)
    }

    /// Checks the configuration against the input sample rate.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        self.num_taps(rate_in).map(|_| ())
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "factor" => self.factor = value.as_usize()?,
            "cutoff_hz" => self.cutoff_hz = Some(value.as_f64()?),
            "transition_hz" => self.transition_hz = Some(value.as_f64()?),
            "attenuation_db" => self.attenuation_db = value.as_f64()?,
            "max_taps" => self.max_taps = value.as_usize()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter. `cutoff_hz` and `transition_hz` are an error while they
    /// are not set (their default depends on the sample rate; the block reports the value in
    /// use).
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "factor" => Ok(ParamValue::I64(self.factor as i64)),
            "cutoff_hz" => self
                .cutoff_hz
                .map(ParamValue::F64)
                .ok_or_else(|| unset(name)),
            "transition_hz" => self
                .transition_hz
                .map(ParamValue::F64)
                .ok_or_else(|| unset(name)),
            "attenuation_db" => Ok(ParamValue::F64(self.attenuation_db)),
            "max_taps" => Ok(ParamValue::I64(self.max_taps as i64)),
            _ => unknown(),
        }
    }

    /// The configuration with the rate-dependent defaults filled in.
    fn resolved(&self, rate_in: Rate) -> Self {
        let spec = self.design_spec(rate_in);
        Self {
            cutoff_hz: Some(spec.cutoff_hz),
            transition_hz: Some(spec.transition_hz),
            ..*self
        }
    }
}

/// Interpolation by an integer factor for `C32` and `CQ15`.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, InterpolateBlock, InterpolateConfig};
/// use sdrk::{Rate, C32};
/// let cfg = InterpolateConfig { factor: 4, ..InterpolateConfig::default() };
/// let rate = Rate::hz(12_000)?;
/// let bytes = InterpolateBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut block = InterpolateBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// let mut out = [C32::default(); 64];
/// let p = block.process(&[C32::new(1.0, 0.0); 16], &mut out)?;
/// assert_eq!((p.consumed, p.produced), (16, 64));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct InterpolateBlock<T: BlockSample, S: Storage> {
    config: InterpolateConfig,
    fir: FirInterpState<T, S::Buf<T::Real>, S::Buf<T>>,
}

impl<T: BlockSample, S: Storage> core::fmt::Debug for InterpolateBlock<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("InterpolateBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<T: BlockSample, S: Storage> InterpolateBlock<T, S> {
    /// Designs the filter and creates the block; taps and history come from `storage`.
    pub fn new(config: &InterpolateConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let l = config.factor;
        let rate_out = multiply_rate(rate_in, l)?;
        let lp = config
            .design_spec(rate_in)
            .lowpass(super::hz(rate_out), config.max_taps)?;
        let n = lp.len();
        let mut taps = storage.alloc::<T::Real>(n)?;
        for (i, t) in taps.as_mut().iter_mut().enumerate() {
            *t = <T as TunedElem>::tap(lp.tap(i) * l as f64)?;
        }
        let hist = storage.alloc::<T>(n.div_ceil(l))?;
        Ok(Self {
            config: config.resolved(rate_in),
            fir: FirInterpState::new(taps, hist, l)?,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    pub fn storage_bytes(config: &InterpolateConfig, rate_in: Rate) -> Result<usize> {
        let n = config.num_taps(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<T::Real>(n)?;
        plan.add::<T>(n.div_ceil(config.factor))?;
        Ok(plan.bytes())
    }

    /// Output rate: the input rate times the factor.
    pub fn rate_out(config: &InterpolateConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        multiply_rate(rate_in, config.factor)
    }

    /// Number of taps of the designed filter.
    pub fn num_taps(&self) -> usize {
        self.fir.taps().len()
    }
}

impl<T: BlockSample, S: Storage> Block for InterpolateBlock<T, S>
where
    Sdrk: FirInterpolate<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        Ok(<Sdrk as FirInterpolate<T>>::fir_interpolate(
            &Sdrk,
            &mut self.fir,
            input,
            output,
        )?)
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: self.fir.ratio(),
            kind: self.fir.ratio_kind(),
            delay: self.fir.group_delay(),
        }
    }

    fn reset(&mut self) {
        self.fir.reset();
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
        1
    }
}

/// Interpolation by an integer factor for `f32` and `Q15`.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, InterpolateConfig, InterpolateRealBlock};
/// use sdrk::Rate;
/// let cfg = InterpolateConfig { factor: 4, ..InterpolateConfig::default() };
/// let rate = Rate::hz(12_000)?;
/// let bytes = InterpolateRealBlock::<f32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut block = InterpolateRealBlock::<f32, _>::new(&cfg, rate, &mut arena)?;
/// let mut out = [0.0f32; 64];
/// let p = block.process(&[1.0; 16], &mut out)?;
/// assert_eq!((p.consumed, p.produced), (16, 64));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct InterpolateRealBlock<R: BlockReal, S: Storage> {
    config: InterpolateConfig,
    fir: FirInterpRealState<R, S::Buf<R>, S::Buf<R>>,
}

impl<R: BlockReal, S: Storage> core::fmt::Debug for InterpolateRealBlock<R, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("InterpolateRealBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<R: BlockReal, S: Storage> InterpolateRealBlock<R, S> {
    /// Designs the filter and creates the block; taps and history come from `storage`.
    pub fn new(config: &InterpolateConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let l = config.factor;
        let rate_out = multiply_rate(rate_in, l)?;
        let lp = config
            .design_spec(rate_in)
            .lowpass(super::hz(rate_out), config.max_taps)?;
        let n = lp.len();
        let mut taps = storage.alloc::<R>(n)?;
        for (i, t) in taps.as_mut().iter_mut().enumerate() {
            *t = <R as TunedElem>::tap(lp.tap(i) * l as f64)?;
        }
        let hist = storage.alloc::<R>(n.div_ceil(l))?;
        Ok(Self {
            config: config.resolved(rate_in),
            fir: FirInterpRealState::new(taps, hist, l)?,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    pub fn storage_bytes(config: &InterpolateConfig, rate_in: Rate) -> Result<usize> {
        let n = config.num_taps(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<R>(n)?;
        plan.add::<R>(n.div_ceil(config.factor))?;
        Ok(plan.bytes())
    }

    /// Output rate: the input rate times the factor.
    pub fn rate_out(config: &InterpolateConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        multiply_rate(rate_in, config.factor)
    }

    /// Number of taps of the designed filter.
    pub fn num_taps(&self) -> usize {
        self.fir.taps().len()
    }
}

impl<R: BlockReal, S: Storage> Block for InterpolateRealBlock<R, S>
where
    Sdrk: FirInterpolateReal<R>,
{
    type In = R;
    type Out = R;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[R], output: &mut [R]) -> Result<Progress> {
        Ok(<Sdrk as FirInterpolateReal<R>>::fir_interpolate_real(
            &Sdrk,
            &mut self.fir,
            input,
            output,
        )?)
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: self.fir.ratio(),
            kind: self.fir.ratio_kind(),
            delay: self.fir.group_delay(),
        }
    }

    fn reset(&mut self) {
        self.fir.reset();
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
        1
    }
}
