//! `squelch`: power gate that passes the signal only while its average power is high enough.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::squelch::{Squelch, SquelchState};

use super::{head, head_mut, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Description of the `squelch` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "squelch",
    summary: "Passes a complex signal only while its average power is above a threshold.",
    details: "Tracks the average power p of the input with a one-pole filter (time constant \
        tau_s) and opens when p reaches the threshold_db level in dBFS (a full-scale complex \
        sample has power 1, i.e. 0 dBFS). The gate closes again when p falls below \
        threshold_db - hysteresis_db; in between it keeps its state, so a signal that hovers \
        around the threshold does not make the gate chatter. While the gate is closed the \
        output is exactly zero, so a following FM demodulator outputs exact silence. Place \
        it after the channel filter (ddc) and before the demodulator and any gain control: \
        after an AGC every signal, noise included, has the same level and the gate would \
        always open. Pitfalls: the gate reacts after about tau_s to a level change, a \
        smaller tau_s opens faster but is noisier; with CQ15 a tau_s above about 20000 \
        samples (tau_s times the sample rate) rounds the smoothing coefficient to zero and \
        the average no longer moves; with CQ15 the rounding of the \
        update stalls the average near -66 dBFS after the signal ends, so a close level \
        (threshold_db - hysteresis_db) below that never closes an open gate, and a threshold \
        below about -93 dB rounds to zero and opens the gate on the first sample, even for \
        silence.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples of the channel signal.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "The input while the gate is open, zeros while it is closed; same type \
            and rate as the input.",
    }],
    params: &[
        ParamDesc {
            name: "threshold_db",
            unit: Unit::Decibel,
            kind: ParamKind::F64 {
                min: -150.0,
                max: 0.0,
                default: Some(-40.0),
            },
            typical: "-40 for a strong local signal, -55 for weak signals",
            mutability: Mutability::Runtime,
            description: "Level in dBFS at which the gate opens.",
        },
        ParamDesc {
            name: "hysteresis_db",
            unit: Unit::Decibel,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 40.0,
                default: Some(3.0),
            },
            typical: "3",
            mutability: Mutability::Runtime,
            description: "The gate closes this many dB below threshold_db.",
        },
        ParamDesc {
            name: "tau_s",
            unit: Unit::Seconds,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: 100.0,
                default: Some(0.01),
            },
            typical: "0.01 at 48 kHz",
            mutability: Mutability::Build,
            description: "Time constant of the power average in seconds.",
        },
    ],
    required: &[],
};

/// Configuration of [`SquelchBlock`].
///
/// ```
/// use sdrk_blocks::SquelchConfig;
/// let cfg = SquelchConfig::default();
/// assert_eq!(cfg.threshold_db, -40.0);
/// assert!(cfg.validate(sdrk::Rate::hz(48_000)?).is_ok());
/// assert!(SquelchConfig { threshold_db: 3.0, ..cfg }.validate(sdrk::Rate::hz(48_000)?).is_err());
/// # Ok::<(), sdrk::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SquelchConfig {
    /// Open level in dBFS (-150 to 0).
    pub threshold_db: f64,
    /// The gate closes this many dB below the open level (0 to 40).
    pub hysteresis_db: f64,
    /// Time constant of the power average in seconds.
    pub tau_s: f64,
}

impl Default for SquelchConfig {
    fn default() -> Self {
        Self {
            threshold_db: -40.0,
            hysteresis_db: 3.0,
            tau_s: 0.01,
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

impl SquelchConfig {
    /// Checks the configuration against the input sample rate.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        check_range("threshold_db", self.threshold_db, -150.0, 0.0)?;
        check_range("hysteresis_db", self.hysteresis_db, 0.0, 40.0)?;
        check_range("tau_s", self.tau_s, 1e-6, 100.0)?;
        // The smoothing coefficient must lie in (0, 0.5]; the kernel checks that.
        SquelchState::<C32>::new(self.threshold_db, self.hysteresis_db, self.tau_s, rate_in)?;
        Ok(())
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "threshold_db" => self.threshold_db = value.as_f64()?,
            "hysteresis_db" => self.hysteresis_db = value.as_f64()?,
            "tau_s" => self.tau_s = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "threshold_db" => Ok(ParamValue::F64(self.threshold_db)),
            "hysteresis_db" => Ok(ParamValue::F64(self.hysteresis_db)),
            "tau_s" => Ok(ParamValue::F64(self.tau_s)),
            _ => unknown(),
        }
    }
}

/// Power squelch for `C32` and `CQ15`.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, SquelchBlock, SquelchConfig};
/// use sdrk::{Rate, C32};
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = SquelchConfig { tau_s: 0.001, ..SquelchConfig::default() };
/// let mut block = SquelchBlock::<C32, _>::new(&cfg, Rate::hz(48_000)?, &mut arena)?;
/// let loud = [C32::new(0.02, 0.0); 480]; // -34 dBFS
/// let mut out = [C32::default(); 480];
/// block.process(&loud, &mut out)?;
/// assert_eq!(out[0], C32::new(0.0, 0.0)); // closed at the start
/// assert_eq!(out[479], loud[479]); // open after the average has settled
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct SquelchBlock<T: BlockSample, S: Storage> {
    config: SquelchConfig,
    state: SquelchState<T>,
    _storage: PhantomData<fn() -> S>,
}

impl<T: BlockSample, S: Storage> SquelchBlock<T, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &SquelchConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        Ok(Self {
            config: *config,
            state: SquelchState::new(
                config.threshold_db,
                config.hysteresis_db,
                config.tau_s,
                rate_in,
            )?,
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &SquelchConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    pub fn rate_out(config: &SquelchConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }

    /// True while the gate is open.
    pub fn is_open(&self) -> bool {
        self.state.is_open()
    }
}

impl<T: BlockSample, S: Storage> Block for SquelchBlock<T, S>
where
    Sdrk: Squelch<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as Squelch<T>>::squelch(
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
            "threshold_db" => {
                let v = value.as_f64()?;
                self.state.set_thresholds(v, self.config.hysteresis_db)?;
                self.config.threshold_db = v;
                Ok(())
            }
            "hysteresis_db" => {
                let v = value.as_f64()?;
                self.state.set_thresholds(self.config.threshold_db, v)?;
                self.config.hysteresis_db = v;
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
