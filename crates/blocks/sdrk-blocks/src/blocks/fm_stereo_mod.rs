//! `fm_stereo_mod`: stereo audio to the FM multiplex signal (sum, pilot, difference on the
//! 38 kHz subcarrier).

use core::marker::PhantomData;

use sdrk::{C32, Progress, Rate, Sdrk};
use sdrk_core::kernels::stereo::{StereoEncode, StereoEncodeState};

use super::{head, head_mut, hz, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};

/// Lowest input rate of `fm_stereo_mod` in Hz: the difference band reaches 53 kHz.
const MIN_RATE_HZ: f64 = 120_000.0;

/// Description of the `fm_stereo_mod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "fm_stereo_mod",
    summary: "Stereo encoder: stereo audio to the FM multiplex signal with 19 kHz pilot and 38 kHz subcarrier.",
    details: "The input is stereo audio (c32, left channel in re, right channel in im, full \
        scale 1.0); the output is the real multiplex signal main (L + R) / 2 + main (L - R) / 2 \
        sin(2 phi) + pilot sin(phi), where phi is the phase of the pilot oscillator at pilot_hz. \
        The subcarrier is the doubled pilot phase, so a receiver that locks to the pilot (the \
        kind fm_stereo_demod) decodes left as left. For audio within full scale the peak is at \
        most main + pilot (0.99 at the defaults), so the following fm_mod with deviation_hz \
        75000 stays within 75 kHz. Place it after preemph_limit (at the audio rate) and the \
        up-sampling resample, at 192 kHz or more, and follow it with fm_mod. pilot 0 turns \
        the pilot off at runtime and receivers fall back to mono. Pitfalls: the input rate \
        must be at least 120 kHz (the difference band reaches 53 kHz); the audio must be \
        band-limited to 15 kHz, otherwise it overlaps the pilot and the subcarrier.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32],
        description: "Stereo audio (re left, im right), pre-emphasised and limited, at 120 kHz \
            or more.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32],
        description: "The real multiplex signal at the same rate, for fm_mod.",
    }],
    params: &[
        ParamDesc {
            name: "pilot_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 18_000.0,
                max: 20_000.0,
                default: Some(19_000.0),
            },
            typical: "19000 (broadcast)",
            mutability: Mutability::Build,
            description: "Frequency of the pilot; the subcarrier is at twice this frequency.",
        },
        ParamDesc {
            name: "main",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 1.0,
                default: Some(0.9),
            },
            typical: "0.9 (with pilot 0.09: peak 0.99)",
            mutability: Mutability::Build,
            description: "Gain of the sum and of the difference signal.",
        },
        ParamDesc {
            name: "pilot",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 0.2,
                default: Some(0.09),
            },
            typical: "0.09 (9 % of the deviation), 0 for mono",
            mutability: Mutability::Runtime,
            description: "Amplitude of the pilot; 0 turns it off.",
        },
    ],
    required: &[],
};

/// Configuration of [`FmStereoModBlock`].
///
/// ```
/// use sdrk_blocks::FmStereoModConfig;
/// let cfg = FmStereoModConfig::default();
/// assert_eq!((cfg.pilot_hz, cfg.main, cfg.pilot), (19_000.0, 0.9, 0.09));
/// assert!(cfg.validate(sdrk::Rate::hz(192_000)?).is_ok());
/// // The difference band reaches 53 kHz: the rate must be at least 120 kHz.
/// assert!(cfg.validate(sdrk::Rate::hz(96_000)?).is_err());
/// # Ok::<(), sdrk::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FmStereoModConfig {
    /// Pilot frequency in Hz (18000 to 20000).
    pub pilot_hz: f64,
    /// Gain of the sum and difference signals (0 to 1).
    pub main: f64,
    /// Pilot amplitude (0 to 0.2).
    pub pilot: f64,
}

impl Default for FmStereoModConfig {
    fn default() -> Self {
        Self {
            pilot_hz: 19_000.0,
            main: 0.9,
            pilot: 0.09,
        }
    }
}

impl FmStereoModConfig {
    /// Checks the configuration against the input sample rate.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        for p in DESCRIPTION.params {
            p.check(&self.get(p.name)?)?;
        }
        if hz(rate_in) < MIN_RATE_HZ {
            return Err(Error::OutOfRange {
                name: "rate_in",
                min: MIN_RATE_HZ,
                max: f64::MAX,
                got: hz(rate_in),
            });
        }
        self.state(rate_in)?;
        Ok(())
    }

    /// The kernel state of the configuration.
    fn state(&self, rate_in: Rate) -> Result<StereoEncodeState> {
        Ok(StereoEncodeState::new(
            self.pilot_hz,
            self.main as f32,
            self.pilot as f32,
            rate_in,
        )?)
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "pilot_hz" => self.pilot_hz = value.as_f64()?,
            "main" => self.main = value.as_f64()?,
            "pilot" => self.pilot = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "pilot_hz" => Ok(ParamValue::F64(self.pilot_hz)),
            "main" => Ok(ParamValue::F64(self.main)),
            "pilot" => Ok(ParamValue::F64(self.pilot)),
            _ => unknown(),
        }
    }
}

/// FM stereo encoder: stereo audio (`C32`, `re` left, `im` right) to the real multiplex signal
/// (`f32`) through the kernel `stereo_encode`.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, FmStereoModBlock, FmStereoModConfig, ParamValue};
/// use sdrk::{Rate, C32};
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = FmStereoModConfig::default();
/// let mut block = FmStereoModBlock::<_>::new(&cfg, Rate::hz(192_000)?, &mut arena)?;
/// block.set_param("pilot", ParamValue::F64(0.0))?;
/// // Mono audio without pilot is the sum signal alone: 0.9 * 0.5.
/// let mut out = [0.0f32; 8];
/// block.process(&[C32::new(0.5, 0.5); 8], &mut out)?;
/// assert!(out.iter().all(|&y| y == 0.45));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct FmStereoModBlock<S: Storage> {
    config: FmStereoModConfig,
    state: StereoEncodeState,
    _storage: PhantomData<fn() -> S>,
}

impl<S: Storage> FmStereoModBlock<S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &FmStereoModConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        Ok(Self {
            config: *config,
            state: config.state(rate_in)?,
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &FmStereoModConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    pub fn rate_out(config: &FmStereoModConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }

    /// The pilot phase before the next sample, in units of `2^-32` turn.
    pub fn phase(&self) -> u32 {
        self.state.phase()
    }
}

impl<S: Storage> Block for FmStereoModBlock<S> {
    type In = C32;
    type Out = f32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[C32], output: &mut [f32]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as StereoEncode>::stereo_encode(
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
            "pilot" => {
                let v = value.as_f64()?;
                self.state.set_pilot(v as f32)?;
                self.config.pilot = v;
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
