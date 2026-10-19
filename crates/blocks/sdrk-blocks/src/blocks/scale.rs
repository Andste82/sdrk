//! `scale`: multiplies a complex signal by a gain in decibels.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::RealSample;
use sdrk_core::kernels::basic::Scale;

use super::{head, head_mut, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Description of the `scale` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "scale",
    summary: "Multiplies a complex signal by a gain given in decibels.",
    details: "Applies a real gain to both components of every sample. For C32 the gain may be \
        between -120 and +40 dB. Q15 cannot represent a gain above 1, so for CQ15 the gain is \
        limited to 0 dB and 0 dB itself is the largest Q15 value (about -0.0003 dB). The gain \
        can be changed while the block runs; it applies from the next call. Pitfall: a gain \
        above the headroom of the signal saturates Q15 samples.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "The scaled samples, same type and rate as the input.",
    }],
    params: &[ParamDesc {
        name: "gain_db",
        unit: Unit::Decibel,
        kind: ParamKind::F64 {
            min: -120.0,
            max: 40.0,
            default: Some(0.0),
        },
        typical: "0 for unity gain, -6 to halve the amplitude",
        mutability: Mutability::Runtime,
        description: "Gain in dB; at most +40 for C32 and at most 0 for CQ15.",
    }],
    required: &[],
};

/// Configuration of [`ScaleBlock`].
///
/// ```
/// use sdrk_blocks::ScaleConfig;
/// let mut cfg = ScaleConfig::default();
/// cfg.set("gain_db", sdrk_blocks::ParamValue::F64(-6.0))?;
/// assert_eq!(cfg.gain_db, -6.0);
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleConfig {
    /// Gain in dB (-120 to +40; at most 0 for CQ15).
    pub gain_db: f64,
}

impl Default for ScaleConfig {
    fn default() -> Self {
        Self { gain_db: 0.0 }
    }
}

impl ScaleConfig {
    /// Checks the configuration (the range of `gain_db` that is common to both sample types).
    pub fn validate(&self, _rate_in: Rate) -> Result<()> {
        DESCRIPTION
            .params
            .first()
            .ok_or(Error::UnknownParameter)?
            .check(&ParamValue::F64(self.gain_db))
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "gain_db" => self.gain_db = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "gain_db" => Ok(ParamValue::F64(self.gain_db)),
            _ => unknown(),
        }
    }
}

/// Gain block for `C32` and `CQ15`.
///
/// ```
/// use sdrk_blocks::{AlignedBytes, ArenaStorage, Block, ScaleBlock, ScaleConfig};
/// use sdrk::{Rate, C32};
/// let mut mem = AlignedBytes([0u8; 0]);
/// let mut arena = ArenaStorage::new(&mut mem.0);
/// let cfg = ScaleConfig { gain_db: -6.020599913279624 };
/// let mut block = ScaleBlock::<C32, _>::new(&cfg, Rate::hz(48_000)?, &mut arena)?;
/// let mut out = [C32::default(); 1];
/// block.process(&[C32::new(2.0, -4.0)], &mut out)?;
/// assert!((out[0].re - 1.0).abs() < 1e-6 && (out[0].im + 2.0).abs() < 1e-6);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct ScaleBlock<T: BlockSample, S: Storage> {
    config: ScaleConfig,
    gain: T::Real,
    _storage: PhantomData<fn() -> S>,
}

impl<T: BlockSample, S: Storage> ScaleBlock<T, S> {
    fn gain_for(gain_db: f64) -> Result<T::Real> {
        if gain_db > T::MAX_GAIN_DB {
            return Err(Error::OutOfRange {
                name: "gain_db",
                min: -120.0,
                max: T::MAX_GAIN_DB,
                got: gain_db,
            });
        }
        Ok(T::Real::from_f64_round(libm::pow(10.0, gain_db / 20.0)))
    }

    /// Creates the block; it needs no storage.
    pub fn new(config: &ScaleConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        Ok(Self {
            config: *config,
            gain: Self::gain_for(config.gain_db)?,
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &ScaleConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    pub fn rate_out(config: &ScaleConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }
}

impl<T: BlockSample, S: Storage> Block for ScaleBlock<T, S>
where
    Sdrk: Scale<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as Scale<T>>::scale(&Sdrk, head(input, n)?, head_mut(output, n)?, self.gain)?;
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        unit_timing()
    }

    fn reset(&mut self) {}

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match runtime_param(&DESCRIPTION, name, &value)? {
            "gain_db" => {
                let db = value.as_f64()?;
                self.gain = Self::gain_for(db)?;
                self.config.gain_db = db;
                Ok(())
            }
            _ => unknown(),
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        1
    }
}
