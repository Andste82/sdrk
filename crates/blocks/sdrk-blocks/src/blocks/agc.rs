//! `agc`: automatic gain control that settles the output amplitude at a reference level.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::agc::{Agc, AgcState};

use super::{head, head_mut, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Description of the `agc` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "agc",
    summary: "Automatic gain control: scales a complex signal to a constant output amplitude.",
    details: "Multiplies every sample by a gain g and then corrects the gain from the amplitude \
        m of the output sample: g = g (1 + (r / ref) (ref - m)), with ref the reference \
        amplitude and r the attack rate when the output is too loud and the release rate when \
        it is too quiet. Near the set point this is a first-order loop whose time constant is \
        attack_s or release_s, independent of the input level. The gain stays between \
        min_gain_db and max_gain_db. Place it after the channel filter (ddc) and, if there is \
        one, after the squelch. For AM use slow time constants (attack 0.1 s, release 0.5 s), \
        well above the audio period: a fast AGC follows the modulation and distorts it. The \
        defaults (attack 0.005 s, release 0.2 s) suit SSB. Pitfalls: a time constant below \
        1 / (fs ln 2) seconds (about 1.5 samples) is rejected; a signal that needs more than \
        max_gain_db stays below the reference; a runtime change keeps the current gain and \
        only clamps it to the new range; a NaN sample (f32) leaves the gain unchanged and an \
        infinite one drives it to min_gain_db.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples of the channel signal.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "The scaled samples; same type and rate as the input.",
    }],
    params: &[
        ParamDesc {
            name: "reference_db",
            unit: Unit::Decibel,
            kind: ParamKind::F64 {
                min: -30.0,
                max: 0.0,
                default: Some(-6.0),
            },
            typical: "-6 (output amplitude 0.5)",
            mutability: Mutability::Runtime,
            description: "Output amplitude in dBFS the loop settles at (a full-scale sample is \
                0 dB); the linear value must be at least 2^-5.",
        },
        ParamDesc {
            name: "attack_s",
            unit: Unit::Seconds,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: 100.0,
                default: Some(0.005),
            },
            typical: "0.005 for SSB, 0.1 for AM",
            mutability: Mutability::Runtime,
            description: "Time constant in seconds when the output is too loud (gain falls).",
        },
        ParamDesc {
            name: "release_s",
            unit: Unit::Seconds,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: 100.0,
                default: Some(0.2),
            },
            typical: "0.2 for SSB, 0.5 for AM",
            mutability: Mutability::Runtime,
            description: "Time constant in seconds when the output is too quiet (gain rises).",
        },
        ParamDesc {
            name: "min_gain_db",
            unit: Unit::Decibel,
            kind: ParamKind::F64 {
                min: -40.0,
                max: 90.0,
                default: Some(-20.0),
            },
            typical: "-20",
            mutability: Mutability::Runtime,
            description: "Lowest gain in dB.",
        },
        ParamDesc {
            name: "max_gain_db",
            unit: Unit::Decibel,
            kind: ParamKind::F64 {
                min: -40.0,
                max: 90.0,
                default: Some(60.0),
            },
            typical: "60",
            mutability: Mutability::Runtime,
            description: "Highest gain in dB.",
        },
        ParamDesc {
            name: "initial_gain_db",
            unit: Unit::Decibel,
            kind: ParamKind::F64 {
                min: -40.0,
                max: 90.0,
                default: Some(0.0),
            },
            typical: "0",
            mutability: Mutability::Build,
            description: "Gain in dB at the start and after a reset; between min_gain_db and \
                max_gain_db.",
        },
    ],
    required: &[],
};

/// Configuration of [`AgcBlock`].
///
/// ```
/// use sdrk_blocks::AgcConfig;
/// let cfg = AgcConfig::default();
/// assert_eq!(cfg.reference_db, -6.0);
/// assert!(cfg.validate(sdrk::Rate::hz(48_000)?).is_ok());
/// assert!(AgcConfig { min_gain_db: 10.0, ..cfg }.validate(sdrk::Rate::hz(48_000)?).is_err());
/// # Ok::<(), sdrk::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgcConfig {
    /// Output amplitude in dBFS (-30 to 0).
    pub reference_db: f64,
    /// Time constant when the output is too loud, in seconds.
    pub attack_s: f64,
    /// Time constant when the output is too quiet, in seconds.
    pub release_s: f64,
    /// Lowest gain in dB (-40 to 90).
    pub min_gain_db: f64,
    /// Highest gain in dB (-40 to 90).
    pub max_gain_db: f64,
    /// Gain at the start in dB, between the two limits.
    pub initial_gain_db: f64,
}

impl Default for AgcConfig {
    fn default() -> Self {
        Self {
            reference_db: -6.0,
            attack_s: 0.005,
            release_s: 0.2,
            min_gain_db: -20.0,
            max_gain_db: 60.0,
            initial_gain_db: 0.0,
        }
    }
}

fn check_range(name: &'static str, v: f64, min: f64, max: f64) -> Result<()> {
    if v >= min && v <= max {
        Ok(())
    } else {
        Err(Error::OutOfRange {
            name,
            min,
            max,
            got: v,
        })
    }
}

/// Linear amplitude of a level in dBFS.
fn linear(db: f64) -> f64 {
    libm::pow(10.0, db / 20.0)
}

impl AgcConfig {
    /// Checks the configuration against the input sample rate.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        check_range("reference_db", self.reference_db, -30.0, 0.0)?;
        check_range("attack_s", self.attack_s, 1e-6, 100.0)?;
        check_range("release_s", self.release_s, 1e-6, 100.0)?;
        check_range("min_gain_db", self.min_gain_db, -40.0, 90.0)?;
        check_range("max_gain_db", self.max_gain_db, -40.0, 90.0)?;
        check_range("initial_gain_db", self.initial_gain_db, -40.0, 90.0)?;
        // The kernel checks the ordering of the gains, the minimum reference and the rates.
        AgcState::<C32>::new(
            linear(self.reference_db),
            self.attack_s,
            self.release_s,
            self.min_gain_db,
            self.max_gain_db,
            self.initial_gain_db,
            rate_in,
        )?;
        Ok(())
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "reference_db" => self.reference_db = value.as_f64()?,
            "attack_s" => self.attack_s = value.as_f64()?,
            "release_s" => self.release_s = value.as_f64()?,
            "min_gain_db" => self.min_gain_db = value.as_f64()?,
            "max_gain_db" => self.max_gain_db = value.as_f64()?,
            "initial_gain_db" => self.initial_gain_db = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "reference_db" => Ok(ParamValue::F64(self.reference_db)),
            "attack_s" => Ok(ParamValue::F64(self.attack_s)),
            "release_s" => Ok(ParamValue::F64(self.release_s)),
            "min_gain_db" => Ok(ParamValue::F64(self.min_gain_db)),
            "max_gain_db" => Ok(ParamValue::F64(self.max_gain_db)),
            "initial_gain_db" => Ok(ParamValue::F64(self.initial_gain_db)),
            _ => unknown(),
        }
    }
}

/// Automatic gain control for `C32` and `CQ15`.
///
/// ```
/// use sdrk_blocks::{AgcBlock, AgcConfig, ArenaStorage, Block};
/// use sdrk::{Rate, C32};
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = AgcConfig { attack_s: 0.001, release_s: 0.002, ..AgcConfig::default() };
/// let mut block = AgcBlock::<C32, _>::new(&cfg, Rate::hz(48_000)?, &mut arena)?;
/// let weak = [C32::new(0.01, 0.0); 4800];
/// let mut out = [C32::default(); 4800];
/// block.process(&weak, &mut out)?;
/// // The gain rose until the output reaches the reference of -6 dBFS (0.5).
/// assert!((out[4799].re - 0.5).abs() < 0.01);
/// assert!(block.gain_db() > 30.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct AgcBlock<T: BlockSample, S: Storage> {
    config: AgcConfig,
    rate_in: Rate,
    state: AgcState<T>,
    _storage: PhantomData<fn() -> S>,
}

impl<T: BlockSample, S: Storage> AgcBlock<T, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &AgcConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        Ok(Self {
            config: *config,
            rate_in,
            state: AgcState::new(
                linear(config.reference_db),
                config.attack_s,
                config.release_s,
                config.min_gain_db,
                config.max_gain_db,
                config.initial_gain_db,
                rate_in,
            )?,
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &AgcConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    pub fn rate_out(config: &AgcConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }

    /// The current gain in dB.
    pub fn gain_db(&self) -> f64 {
        self.state.gain_db()
    }
}

impl<T: BlockSample, S: Storage> Block for AgcBlock<T, S>
where
    Sdrk: Agc<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as Agc<T>>::agc(
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
        let name = runtime_param(&DESCRIPTION, name, &value)?;
        let mut candidate = self.config;
        candidate.set(name, value)?;
        // The kernel keeps the current gain (clamped) and checks the ordering of the limits
        // and the rates; on an error nothing changes. The initial gain is not checked
        // against the new limits, `reset` clamps it.
        self.state.set_params(
            linear(candidate.reference_db),
            candidate.attack_s,
            candidate.release_s,
            candidate.min_gain_db,
            candidate.max_gain_db,
            self.rate_in,
        )?;
        self.config = candidate;
        Ok(())
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        1
    }
}
