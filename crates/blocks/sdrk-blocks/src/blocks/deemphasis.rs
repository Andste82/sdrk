//! `deemphasis`: single-pole audio de-emphasis on its own.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::biquad::{Biquad, BiquadSample, BiquadState};
use sdrk_core::kernels::demod::{Deemphasis, DeemphasisState};
use sdrk_core::timing::Timing;

use super::biquad::BiquadElem;
use super::fm_stereo_demod::deemphasis_section;
use super::{head, head_mut, hz, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::sample::{BlockReal, BlockSample};
use crate::storage::{Plan, Storage};

/// Description of the `deemphasis` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "deemphasis",
    summary: "Single-pole low-pass that undoes the pre-emphasis of FM broadcast audio.",
    details: "A one-pole low-pass y[n] = y[n-1] + alpha (x[n] - y[n-1]) with alpha = 1 - exp(-1 / \
        (fs tau)). It sits after the quadrature discriminator of an FM receiver (before or \
        after the audio decimation; the filter is cheaper at the lower rate, but then tau must \
        stay well below the output Nyquist frequency). Real samples (f32, Q15) get the same \
        filter as the de-emphasis inside fm_demod. Complex samples (C32, CQ15) are stereo \
        audio with the left channel in re and the right channel in im, as stereo_matrix and \
        fm_stereo_demod write it: both channels are filtered alike, with the biquad section \
        that fm_stereo_demod uses for its de-emphasis, so the block gives the same samples as \
        that block bit for bit. Its group delay depends on the frequency and is reported as \
        zero. Pitfall: tau is the time constant in microseconds (50 in Europe, 75 in North \
        America), not a frequency.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32, PortType::Q15, PortType::C32, PortType::CQ15],
        description: "Real samples, for example the output of quad_demod, or stereo audio as \
            complex samples (left in re, right in im).",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32, PortType::Q15, PortType::C32, PortType::CQ15],
        description: "The de-emphasised samples, same type and rate as the input.",
    }],
    params: &[ParamDesc {
        name: "tau_us",
        unit: Unit::Microseconds,
        kind: ParamKind::F64 {
            min: 1.0,
            max: 10_000.0,
            default: Some(50.0),
        },
        typical: "50 in Europe, 75 in North America",
        mutability: Mutability::Build,
        description: "Time constant of the de-emphasis in microseconds.",
    }],
    required: &[],
};

/// Configuration of [`DeemphasisRealBlock`].
///
/// ```
/// use sdrk_blocks::DeemphasisConfig;
/// let cfg = DeemphasisConfig { tau_us: 75.0 };
/// assert!(cfg.validate(sdrk::Rate::hz(48_000)?).is_ok());
/// assert!(DeemphasisConfig { tau_us: 0.0 }.validate(sdrk::Rate::hz(48_000)?).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeemphasisConfig {
    /// Time constant in microseconds.
    pub tau_us: f64,
}

impl Default for DeemphasisConfig {
    fn default() -> Self {
        Self { tau_us: 50.0 }
    }
}

impl DeemphasisConfig {
    /// Checks the configuration against the input sample rate.
    pub fn validate(&self, _rate_in: Rate) -> Result<()> {
        let v = self.tau_us;
        if (1.0..=10_000.0).contains(&v) {
            Ok(())
        } else {
            Err(Error::OutOfRange {
                name: "tau_us",
                min: 1.0,
                max: 10_000.0,
                got: v,
            })
        }
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "tau_us" => self.tau_us = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "tau_us" => Ok(ParamValue::F64(self.tau_us)),
            _ => unknown(),
        }
    }
}

/// De-emphasis for real samples, `f32` and `Q15`. For stereo audio as complex samples see
/// [`DeemphasisBlock`].
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, DeemphasisConfig, DeemphasisRealBlock};
/// use sdrk::Rate;
/// let rate = Rate::hz(48_000)?;
/// let cfg = DeemphasisConfig::default();
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut block = DeemphasisRealBlock::<f32, _>::new(&cfg, rate, &mut arena)?;
/// let mut out = [0.0f32; 400];
/// block.process(&[1.0; 400], &mut out)?;
/// assert!(out[0] < 0.5 && (out[399] - 1.0).abs() < 1e-3); // a step settles
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct DeemphasisRealBlock<R: BlockReal, S: Storage> {
    config: DeemphasisConfig,
    state: DeemphasisState<R>,
    _storage: PhantomData<fn() -> S>,
}

impl<R: BlockReal, S: Storage> DeemphasisRealBlock<R, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &DeemphasisConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        Ok(Self {
            config: *config,
            state: DeemphasisState::new(config.tau_us * 1e-6, rate_in)?,
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &DeemphasisConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    pub fn rate_out(config: &DeemphasisConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }
}

impl<R: BlockReal, S: Storage> Block for DeemphasisRealBlock<R, S>
where
    Sdrk: Deemphasis<R>,
{
    type In = R;
    type Out = R;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[R], output: &mut [R]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as Deemphasis<R>>::deemphasis(
            &Sdrk,
            &mut self.state,
            head(input, n)?,
            head_mut(output, n)?,
        )?;
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            delay: self.state.group_delay(),
            ..unit_timing()
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
        1
    }
}

/// The state of the complex de-emphasis: one biquad section.
type SectionState<T, S> =
    BiquadState<T, <S as Storage>::Buf<[<T as BiquadSample>::Coef; 5]>, <S as Storage>::Buf<T>>;

/// De-emphasis of stereo audio for `C32` and `CQ15` (left in `re`, right in `im`, `T` is the
/// complex type).
///
/// Both channels are filtered with the biquad section that `fm_stereo_demod` uses for its
/// de-emphasis (`[alpha, 0, 0, -(1 - alpha), 0]`), so the output equals the output of that block
/// bit for bit when it follows `stereo_matrix` at the same rate. The block needs storage for
/// the section and its four state values.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, DeemphasisBlock, DeemphasisConfig};
/// use sdrk::{Rate, C32};
/// let rate = Rate::hz(48_000)?;
/// let cfg = DeemphasisConfig::default();
/// let bytes = DeemphasisBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut block = DeemphasisBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// let mut out = [C32::default(); 400];
/// block.process(&[C32::new(1.0, -1.0); 400], &mut out)?;
/// // A step settles in both channels.
/// assert!((out[399].re - 1.0).abs() < 1e-3 && (out[399].im + 1.0).abs() < 1e-3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct DeemphasisBlock<T: BlockSample + BiquadElem, S: Storage> {
    config: DeemphasisConfig,
    state: SectionState<T, S>,
}

impl<T: BlockSample + BiquadElem, S: Storage> core::fmt::Debug for DeemphasisBlock<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeemphasisBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<T: BlockSample + BiquadElem, S: Storage> DeemphasisBlock<T, S> {
    /// Creates the block; the section and the state values come from `storage`.
    pub fn new(config: &DeemphasisConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let section = deemphasis_section(config.tau_us, hz(rate_in));
        let mut coefs = storage.alloc::<[<T as BiquadSample>::Coef; 5]>(1)?;
        <T as BiquadElem>::convert(&[section], coefs.as_mut())?;
        let hist = storage.alloc::<T>(4)?;
        Ok(Self {
            config: *config,
            state: BiquadState::new(coefs, hist)?,
        })
    }

    /// Exact bytes of storage the block needs.
    pub fn storage_bytes(config: &DeemphasisConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<[<T as BiquadSample>::Coef; 5]>(1)?;
        plan.add::<T>(4)?;
        Ok(plan.bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    pub fn rate_out(config: &DeemphasisConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }
}

impl<T: BlockSample + BiquadElem, S: Storage> Block for DeemphasisBlock<T, S>
where
    Sdrk: Biquad<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as Biquad<T>>::biquad(
            &Sdrk,
            &mut self.state,
            head(input, n)?,
            head_mut(output, n)?,
        )?;
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        unit_timing()
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
        1
    }
}
