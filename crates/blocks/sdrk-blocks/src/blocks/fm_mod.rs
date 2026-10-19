//! `fm_mod`: frequency modulator (real audio to complex baseband).

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::modulate::{FmMod, FmModState};

use super::{HZ_LIMIT, head, head_mut, hz, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Description of the `fm_mod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "fm_mod",
    summary: "Frequency-modulates a real signal onto a complex baseband carrier.",
    details: "The input is the real modulating signal with full scale 1.0 (f32) or 32767 (Q15); \
        the output is a unit-magnitude complex signal whose instantaneous frequency is the input \
        times deviation_hz. The block runs at the rate of its input, so the audio must be \
        interpolated to the baseband rate before it (for broadcast FM, 240 kHz). The phase \
        accumulator is 32 bits wide and never drifts. The deviation can change while the \
        block runs and the phase stays continuous.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32, PortType::Q15],
        description: "Real modulating signal at the baseband rate.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex baseband signal at the same rate.",
    }],
    params: &[ParamDesc {
        name: "deviation_hz",
        unit: Unit::Hz,
        kind: ParamKind::F64 {
            min: 0.0,
            max: HZ_LIMIT,
            default: Some(75_000.0),
        },
        typical: "75e3 for FM broadcast, 5e3 for narrow-band voice",
        mutability: Mutability::Runtime,
        description: "Frequency deviation in Hz for a full-scale input; at most half the \
            sample rate.",
    }],
    required: &[],
};

/// Configuration of [`FmModBlock`].
///
/// ```
/// use sdrk_blocks::FmModConfig;
/// assert_eq!(FmModConfig::default().deviation_hz, 75_000.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FmModConfig {
    /// Frequency deviation in Hz for a full-scale input.
    pub deviation_hz: f64,
}

impl Default for FmModConfig {
    fn default() -> Self {
        Self {
            deviation_hz: 75_000.0,
        }
    }
}

impl FmModConfig {
    /// Checks the configuration against the input sample rate.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        let half = hz(rate_in) / 2.0;
        if self.deviation_hz >= 0.0 && self.deviation_hz <= half {
            Ok(())
        } else {
            Err(Error::OutOfRange {
                name: "deviation_hz",
                min: 0.0,
                max: half,
                got: self.deviation_hz,
            })
        }
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "deviation_hz" => self.deviation_hz = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "deviation_hz" => Ok(ParamValue::F64(self.deviation_hz)),
            _ => unknown(),
        }
    }
}

/// FM modulator: `f32` to `C32` or `Q15` to `CQ15` (`T` is the complex output type).
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, FmModBlock, FmModConfig};
/// use sdrk::{Rate, C32};
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = FmModConfig { deviation_hz: 1_000.0 };
/// let mut block = FmModBlock::<C32, _>::new(&cfg, Rate::hz(8_000)?, &mut arena)?;
/// let mut out = [C32::default(); 4];
/// block.process(&[1.0; 4], &mut out)?; // +1 kHz at 8 kHz: 45 degrees per sample
/// assert!((out[1].re - 0.0).abs() < 1e-3 && (out[1].im - 1.0).abs() < 1e-3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct FmModBlock<T: BlockSample, S: Storage> {
    config: FmModConfig,
    rate_in: Rate,
    state: FmModState,
    _marker: PhantomData<fn() -> (T, S)>,
}

impl<T: BlockSample, S: Storage> FmModBlock<T, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &FmModConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        Ok(Self {
            config: *config,
            rate_in,
            state: FmModState::new(config.deviation_hz, rate_in)?,
            _marker: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &FmModConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    pub fn rate_out(config: &FmModConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }
}

impl<T: BlockSample, S: Storage> Block for FmModBlock<T, S>
where
    Sdrk: FmMod<T>,
{
    type In = T::Real;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T::Real], output: &mut [T]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as FmMod<T>>::fm_mod(
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

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match runtime_param(&DESCRIPTION, name, &value)? {
            "deviation_hz" => {
                let d = value.as_f64()?;
                FmModConfig { deviation_hz: d }.validate(self.rate_in)?;
                let phase = self.state.phase();
                self.state = FmModState::new(d, self.rate_in)?;
                self.state.set_phase(phase);
                self.config.deviation_hz = d;
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
