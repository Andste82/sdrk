//! `rotate`: frequency shift by a numerically controlled oscillator.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::mix::{Nco, Rotate};

use super::{HZ_LIMIT, head, head_mut, hz, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Description of the `rotate` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "rotate",
    summary: "Shifts the spectrum of a complex signal by a frequency.",
    details: "Multiplies the signal with a complex oscillator, so a positive frequency_hz moves \
        the spectrum up and a negative one moves it down; to bring a carrier at +f to 0 Hz \
        use -f. It is the mixer of a digital down-converter (rotate, then decimate) or up-converter \
        (interpolate, then rotate). frequency_hz can change while the block runs and the \
        oscillator phase stays continuous. The oscillator resolution is fs / 2^32. Pitfall: \
        the frequency must be within plus or minus half the sample rate, exclusive; the \
        shifted signal still has to fit into the band, nothing is filtered.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "The shifted samples, same type and rate as the input.",
    }],
    params: &[ParamDesc {
        name: "frequency_hz",
        unit: Unit::Hz,
        kind: ParamKind::F64 {
            min: -HZ_LIMIT,
            max: HZ_LIMIT,
            default: None,
        },
        typical: "-600e3 to bring a carrier 600 kHz above the tuner centre to 0 Hz",
        mutability: Mutability::Runtime,
        description: "Shift in Hz; positive values shift the spectrum up. Within plus or minus \
            half the input rate, exclusive. Required.",
    }],
    required: &["frequency_hz"],
};

/// Configuration of [`RotateBlock`].
///
/// ```
/// use sdrk_blocks::RotateConfig;
/// let cfg = RotateConfig { frequency_hz: -600e3 };
/// assert!(cfg.validate(sdrk::Rate::hz(2_400_000)?).is_ok());
/// assert!(RotateConfig { frequency_hz: 1_200e3 }.validate(sdrk::Rate::hz(2_400_000)?).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RotateConfig {
    /// Shift in Hz (positive shifts up), within plus or minus half the input rate, exclusive.
    pub frequency_hz: f64,
}

/// Checks `|f| < fs / 2`.
fn check_frequency(f: f64, rate: Rate) -> Result<()> {
    let half = hz(rate) / 2.0;
    if f > -half && f < half {
        Ok(())
    } else {
        Err(Error::OutOfRange {
            name: "frequency_hz",
            min: -half,
            max: half,
            got: f,
        })
    }
}

impl RotateConfig {
    /// Checks the configuration against the input sample rate.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        check_frequency(self.frequency_hz, rate_in)
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "frequency_hz" => self.frequency_hz = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "frequency_hz" => Ok(ParamValue::F64(self.frequency_hz)),
            _ => unknown(),
        }
    }
}

/// Frequency shifter for `C32` and `CQ15`.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, RotateBlock, RotateConfig};
/// use sdrk::{Rate, C32};
/// let rate = Rate::hz(8)?;
/// let mut arena = ArenaStorage::new(&mut []);
/// // A quarter of the sample rate turns a constant 1 into 1, j, -1, -j.
/// let mut block = RotateBlock::<C32, _>::new(&RotateConfig { frequency_hz: 2.0 }, rate, &mut arena)?;
/// let mut out = [C32::default(); 4];
/// block.process(&[C32::new(1.0, 0.0); 4], &mut out)?;
/// assert!((out[1].im - 1.0).abs() < 1e-6 && (out[2].re + 1.0).abs() < 1e-6);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct RotateBlock<T: BlockSample, S: Storage> {
    config: RotateConfig,
    rate_in: Rate,
    nco: Nco,
    _marker: PhantomData<fn() -> (T, S)>,
}

impl<T: BlockSample, S: Storage> RotateBlock<T, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &RotateConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        Ok(Self {
            config: *config,
            rate_in,
            nco: Nco::new(config.frequency_hz, rate_in)?,
            _marker: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &RotateConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    pub fn rate_out(config: &RotateConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }
}

impl<T: BlockSample, S: Storage> Block for RotateBlock<T, S>
where
    Sdrk: Rotate<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as Rotate<T>>::rotate(&Sdrk, &mut self.nco, head(input, n)?, head_mut(output, n)?)?;
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        unit_timing()
    }

    fn reset(&mut self) {
        self.nco.reset();
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match runtime_param(&DESCRIPTION, name, &value)? {
            "frequency_hz" => {
                let f = value.as_f64()?;
                check_frequency(f, self.rate_in)?;
                self.nco.set_frequency(f, self.rate_in)?;
                self.config.frequency_hz = f;
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
