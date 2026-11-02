//! `decimate`: low-pass filter and decimation by an integer factor (complex and real).

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::decim::{FirDecimState, FirDecimate, ResampleReal, ResampleRealState};
use sdrk_core::timing::Timing;

use super::fir::MAX_TAPS_LIMIT;
use super::{HZ_LIMIT, check_count, divide_rate, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::DesignSpec;
use crate::error::{Error, Result};
use crate::sample::{BlockReal, BlockSample, TunedElem};
use crate::storage::{Plan, Storage};

/// Description of the `decimate` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "decimate",
    summary: "Low-pass filter and decimation by an integer factor.",
    details: "A Kaiser-window low-pass designed at the input rate followed by keeping every \
        factor-th sample; the output rate is the input rate divided by factor. Unless given, \
        the cutoff is 0.4 times the output rate and the transition width 0.1 times the output \
        rate, so the stop band starts at half the output rate and nothing aliases. It is the \
        channel filter of a receiver after rotate, and the audio decimator after the \
        discriminator. Complex signals use the FIR decimator, real signals the polyphase \
        resampler with an up factor of 1 (the audio stage of fm_demod and am_demod). For a \
        factor of 2 with a cheap filter use halfband_decimate. The group delay is (taps - 1) / \
        2 input samples. Pitfall: the stop band, cutoff plus half the transition width, must \
        not exceed half the output rate.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
        description: "Samples at the input rate.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
        description: "Samples of the same type at the input rate divided by factor.",
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
            typical: "10 to go from 2.4 MHz to 240 kHz",
            mutability: Mutability::Build,
            description: "Decimation factor, 1 to 1024. Required.",
        },
        ParamDesc {
            name: "cutoff_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "100e3 for FM broadcast; default 0.4 times the output rate",
            mutability: Mutability::Build,
            description: "Cutoff (-6 dB point) of the low-pass in Hz. The default is \
                0.4 * fs / factor.",
        },
        ParamDesc {
            name: "transition_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "default 0.1 times the output rate",
            mutability: Mutability::Build,
            description: "Width of the transition band in Hz. The default is 0.1 * fs / factor. \
                Cutoff plus half the transition must not exceed half the output rate.",
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

/// Error of the `get` of a filter parameter that is still unset (the default depends on the
/// sample rate).
pub(crate) fn unset(name: &str) -> Error {
    Error::InvalidParameter {
        name: if name == "cutoff_hz" {
            "cutoff_hz"
        } else {
            "transition_hz"
        },
        reason: "not set; the default depends on the sample rate",
    }
}

/// Configuration of [`DecimateBlock`] and [`DecimateRealBlock`].
///
/// ```
/// use sdrk_blocks::DecimateConfig;
/// let cfg = DecimateConfig { factor: 10, ..DecimateConfig::default() };
/// assert_eq!(cfg.num_taps(sdrk::Rate::hz(2_400_000)?)?, 365);
/// // A stop band above the output Nyquist frequency is rejected.
/// let wide = DecimateConfig { cutoff_hz: Some(130e3), ..cfg };
/// assert!(wide.validate(sdrk::Rate::hz(2_400_000)?).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecimateConfig {
    /// Decimation factor, 1 to 1024.
    pub factor: usize,
    /// Cutoff in Hz; `None` means `0.4 * fs / factor`.
    pub cutoff_hz: Option<f64>,
    /// Transition width in Hz; `None` means `0.1 * fs / factor`.
    pub transition_hz: Option<f64>,
    /// Stop-band attenuation in dB.
    pub attenuation_db: f64,
    /// Upper limit of the number of taps.
    pub max_taps: usize,
}

impl Default for DecimateConfig {
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

impl DecimateConfig {
    /// The filter specification for the input rate with the defaults filled in.
    ///
    /// ```
    /// use sdrk_blocks::DecimateConfig;
    /// let spec = DecimateConfig::default().design_spec(sdrk::Rate::hz(2_400_000)?);
    /// assert_eq!((spec.cutoff_hz, spec.transition_hz), (96_000.0, 24_000.0));
    /// # Ok::<(), sdrk::Error>(())
    /// ```
    pub fn design_spec(&self, rate_in: Rate) -> DesignSpec {
        let out = super::hz(rate_in) / self.factor.max(1) as f64;
        DesignSpec {
            cutoff_hz: self.cutoff_hz.unwrap_or(0.4 * out),
            transition_hz: self.transition_hz.unwrap_or(0.1 * out),
            attenuation_db: self.attenuation_db,
        }
    }

    /// Number of taps for the input rate (also checks the whole configuration).
    pub fn num_taps(&self, rate_in: Rate) -> Result<usize> {
        check_count("factor", self.factor, 1, 1024)?;
        let fs = super::hz(rate_in);
        let spec = self.design_spec(rate_in);
        // The stop band must start below the output Nyquist frequency, otherwise the wanted
        // band aliases.
        let out_half = fs / self.factor as f64 / 2.0;
        if spec.cutoff_hz + spec.transition_hz / 2.0 > out_half {
            return Err(Error::OutOfRange {
                name: "cutoff_hz",
                min: 0.0,
                max: out_half - spec.transition_hz / 2.0,
                got: spec.cutoff_hz,
            });
        }
        spec.num_taps(fs, self.max_taps)
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

/// Low-pass and decimation by an integer factor for `C32` and `CQ15`.
///
/// ```
/// use sdrk_blocks::{AlignedBytes, ArenaStorage, Block, DecimateBlock, DecimateConfig};
/// use sdrk::{Rate, C32};
/// let cfg = DecimateConfig { factor: 4, ..DecimateConfig::default() };
/// let rate = Rate::hz(48_000)?;
/// let bytes = DecimateBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = AlignedBytes([0u8; 4096]);
/// let mut arena = ArenaStorage::new(&mut mem.0[..bytes]);
/// let mut block = DecimateBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// let mut out = [C32::default(); 16];
/// let p = block.process(&[C32::new(1.0, 0.0); 64], &mut out)?;
/// assert_eq!((p.consumed, p.produced), (64, 16));
/// assert_eq!(DecimateBlock::<C32, ArenaStorage<'_>>::rate_out(&cfg, rate)?, Rate::hz(12_000)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct DecimateBlock<T: BlockSample, S: Storage> {
    config: DecimateConfig,
    fir: FirDecimState<T, S::Buf<T::Real>, S::Buf<T>>,
}

impl<T: BlockSample, S: Storage> core::fmt::Debug for DecimateBlock<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DecimateBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<T: BlockSample, S: Storage> DecimateBlock<T, S> {
    /// Designs the filter and creates the block; taps and history come from `storage`.
    pub fn new(config: &DecimateConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let lp = config
            .design_spec(rate_in)
            .lowpass(super::hz(rate_in), config.max_taps)?;
        let n = lp.len();
        let mut taps = storage.alloc::<T::Real>(n)?;
        for (i, t) in taps.as_mut().iter_mut().enumerate() {
            *t = <T as TunedElem>::tap(lp.tap(i))?;
        }
        let hist = storage.alloc::<T>(n - 1)?;
        Ok(Self {
            config: config.resolved(rate_in),
            fir: FirDecimState::new(taps, hist, config.factor)?,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    pub fn storage_bytes(config: &DecimateConfig, rate_in: Rate) -> Result<usize> {
        let n = config.num_taps(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<T::Real>(n)?;
        plan.add::<T>(n - 1)?;
        Ok(plan.bytes())
    }

    /// Output rate: the input rate divided by the factor.
    pub fn rate_out(config: &DecimateConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        divide_rate(rate_in, config.factor)
    }

    /// Number of taps of the designed filter.
    pub fn num_taps(&self) -> usize {
        self.fir.taps().len()
    }
}

impl<T: BlockSample, S: Storage> Block for DecimateBlock<T, S>
where
    Sdrk: FirDecimate<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        Ok(<Sdrk as FirDecimate<T>>::fir_decimate(
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
        self.config.factor
    }
}

/// Low-pass and decimation by an integer factor for `f32` and `Q15`.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, DecimateConfig, DecimateRealBlock};
/// use sdrk::Rate;
/// let cfg = DecimateConfig { factor: 4, ..DecimateConfig::default() };
/// let rate = Rate::hz(48_000)?;
/// let bytes = DecimateRealBlock::<f32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut block = DecimateRealBlock::<f32, _>::new(&cfg, rate, &mut arena)?;
/// let mut out = [0.0f32; 16];
/// let p = block.process(&[1.0; 64], &mut out)?;
/// assert_eq!((p.consumed, p.produced), (64, 16));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct DecimateRealBlock<R: BlockReal, S: Storage> {
    config: DecimateConfig,
    state: ResampleRealState<R, S::Buf<R>, S::Buf<R>>,
}

impl<R: BlockReal, S: Storage> core::fmt::Debug for DecimateRealBlock<R, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DecimateRealBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<R: BlockReal, S: Storage> DecimateRealBlock<R, S> {
    /// Designs the filter and creates the block; taps and history come from `storage`.
    pub fn new(config: &DecimateConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let lp = config
            .design_spec(rate_in)
            .lowpass(super::hz(rate_in), config.max_taps)?;
        let n = lp.len();
        let mut taps = storage.alloc::<R>(n)?;
        for (i, t) in taps.as_mut().iter_mut().enumerate() {
            *t = <R as TunedElem>::tap(lp.tap(i))?;
        }
        let hist = storage.alloc::<R>(n)?;
        Ok(Self {
            config: config.resolved(rate_in),
            state: ResampleRealState::new(taps, hist, 1, config.factor)?,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    pub fn storage_bytes(config: &DecimateConfig, rate_in: Rate) -> Result<usize> {
        let n = config.num_taps(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<R>(n)?;
        plan.add::<R>(n)?;
        Ok(plan.bytes())
    }

    /// Output rate: the input rate divided by the factor.
    pub fn rate_out(config: &DecimateConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        divide_rate(rate_in, config.factor)
    }

    /// Number of taps of the designed filter.
    pub fn num_taps(&self) -> usize {
        self.state.taps().len()
    }
}

impl<R: BlockReal, S: Storage> Block for DecimateRealBlock<R, S>
where
    Sdrk: ResampleReal<R>,
{
    type In = R;
    type Out = R;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[R], output: &mut [R]) -> Result<Progress> {
        Ok(<Sdrk as ResampleReal<R>>::resample_real(
            &Sdrk,
            &mut self.state,
            input,
            output,
        )?)
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
        self.config.factor
    }
}
