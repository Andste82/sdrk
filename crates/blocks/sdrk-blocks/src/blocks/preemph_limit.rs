//! `preemph_limit`: pre-emphasis and peak limiter for the audio of an FM transmitter.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::preemph_limit::{PreemphLimit, PreemphLimitState};

use super::{head, head_mut, hz, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::{PREEMPHASIS_MIN_RATE_HZ, preemphasis_coefs};
use crate::error::{Error, Result};
use crate::sample::BlockReal;
use crate::storage::{Plan, Storage};

/// Description of the `preemph_limit` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "preemph_limit",
    summary: "Pre-emphasis and peak limiter for the audio of an FM transmitter.",
    details: "Applies a first-order pre-emphasis (gain 1 at DC, +6 dB per octave above \
        1 / (2 pi tau), a pole at one 2.4th of the sample rate) and then limits the peak of the \
        result to limit times full scale, in one pass. The limiter has an instant attack, so no \
        sample exceeds the limit; it holds the gain for hold_s and then recovers it with the \
        time constant release_s. A pre-emphasis of 0 switches the filter off and leaves the \
        limiter. Non-finite f32 input counts as zero. Place it directly before fm_mod, at the \
        rate of the modulator: a filter or resampler between the limiter and the modulator \
        raises the peaks again. Measured with overdriven noise: limiting at 48 kHz and then \
        resampling to 240 kHz reached 1.41 times the limit, limiting at 240 kHz kept it at \
        1.0. When a channel filter must follow (the voice band of narrow-band FM), set limit \
        below 1 to leave room for its overshoot. limit 1.0 keeps the peak deviation at the \
        deviation_hz of fm_mod. Pitfalls: a pre-emphasis needs an input rate of at least \
        40 kHz; 530 us amplifies by 36 dB at 48 kHz Nyquist, so the limiter works hard on \
        unfiltered audio.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32, PortType::Q15],
        description: "Real audio at the rate of the modulator.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32, PortType::Q15],
        description: "Pre-emphasised and limited audio; same type and rate as the input.",
    }],
    params: &[
        ParamDesc {
            name: "preemphasis_us",
            unit: Unit::Microseconds,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 1000.0,
                default: Some(50.0),
            },
            typical: "50 (Europe), 75 (North America), 530 for two-way radio, 0 for none",
            mutability: Mutability::Build,
            description: "Time constant of the pre-emphasis in microseconds; 0 switches the \
                filter off.",
        },
        ParamDesc {
            name: "limit",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.1,
                max: 1.0,
                default: Some(1.0),
            },
            typical: "1.0 before fm_mod, 0.85 when a filter follows",
            mutability: Mutability::Runtime,
            description: "Peak of the output as a fraction of full scale.",
        },
        ParamDesc {
            name: "hold_s",
            unit: Unit::Seconds,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 1.0,
                default: Some(0.02),
            },
            typical: "0.02 (covers 50 Hz)",
            mutability: Mutability::Build,
            description: "Time the gain stays constant after the last peak.",
        },
        ParamDesc {
            name: "release_s",
            unit: Unit::Seconds,
            kind: ParamKind::F64 {
                min: 0.001,
                max: 10.0,
                default: Some(0.2),
            },
            typical: "0.2",
            mutability: Mutability::Build,
            description: "Time constant of the gain recovery after the hold.",
        },
    ],
    required: &[],
};

/// Configuration of [`PreemphLimitBlock`].
///
/// ```
/// use sdrk_blocks::PreemphLimitConfig;
/// let cfg = PreemphLimitConfig::default();
/// assert_eq!(cfg.preemphasis_us, 50.0);
/// assert!(cfg.validate(sdrk::Rate::hz(240_000)?).is_ok());
/// // A pre-emphasis needs at least 40 kHz; without it any rate is fine.
/// assert!(cfg.validate(sdrk::Rate::hz(32_000)?).is_err());
/// assert!(PreemphLimitConfig { preemphasis_us: 0.0, ..cfg }.validate(sdrk::Rate::hz(8_000)?).is_ok());
/// # Ok::<(), sdrk::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreemphLimitConfig {
    /// Pre-emphasis time constant in microseconds (0 to 1000, 0 = off).
    pub preemphasis_us: f64,
    /// Peak of the output as a fraction of full scale (0.1 to 1).
    pub limit: f64,
    /// Hold time of the gain in seconds (0 to 1).
    pub hold_s: f64,
    /// Release time constant in seconds (0.001 to 10).
    pub release_s: f64,
}

impl Default for PreemphLimitConfig {
    fn default() -> Self {
        Self {
            preemphasis_us: 50.0,
            limit: 1.0,
            hold_s: 0.02,
            release_s: 0.2,
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

impl PreemphLimitConfig {
    /// The coefficients `[b0, b1, a1]` for the input rate (`[1, 0, 0]` without pre-emphasis).
    fn coefs(&self, rate_in: Rate) -> Result<[f64; 3]> {
        if self.preemphasis_us == 0.0 {
            return Ok([1.0, 0.0, 0.0]);
        }
        if hz(rate_in) < PREEMPHASIS_MIN_RATE_HZ {
            return Err(Error::OutOfRange {
                name: "rate_in",
                min: PREEMPHASIS_MIN_RATE_HZ,
                max: f64::MAX,
                got: hz(rate_in),
            });
        }
        preemphasis_coefs(self.preemphasis_us * 1e-6, hz(rate_in))
    }

    /// Checks the configuration against the input sample rate.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        check_range("preemphasis_us", self.preemphasis_us, 0.0, 1000.0)?;
        check_range("limit", self.limit, 0.1, 1.0)?;
        check_range("hold_s", self.hold_s, 0.0, 1.0)?;
        check_range("release_s", self.release_s, 0.001, 10.0)?;
        // The kernel checks the derived figures (hold in samples, release step).
        PreemphLimitState::<f32>::new(
            self.coefs(rate_in)?,
            self.limit,
            self.hold_s,
            self.release_s,
            rate_in,
        )?;
        PreemphLimitState::<sdrk::Q15>::new(
            self.coefs(rate_in)?,
            self.limit,
            self.hold_s,
            self.release_s,
            rate_in,
        )?;
        Ok(())
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "preemphasis_us" => self.preemphasis_us = value.as_f64()?,
            "limit" => self.limit = value.as_f64()?,
            "hold_s" => self.hold_s = value.as_f64()?,
            "release_s" => self.release_s = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "preemphasis_us" => Ok(ParamValue::F64(self.preemphasis_us)),
            "limit" => Ok(ParamValue::F64(self.limit)),
            "hold_s" => Ok(ParamValue::F64(self.hold_s)),
            "release_s" => Ok(ParamValue::F64(self.release_s)),
            _ => unknown(),
        }
    }
}

/// Pre-emphasis and peak limiter for `f32` and `Q15`.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, ParamValue, PreemphLimitBlock, PreemphLimitConfig};
/// use sdrk::Rate;
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = PreemphLimitConfig { preemphasis_us: 0.0, ..PreemphLimitConfig::default() };
/// let mut block = PreemphLimitBlock::<f32, _>::new(&cfg, Rate::hz(240_000)?, &mut arena)?;
/// let loud = [3.0f32, -3.0, 0.5, -0.25];
/// let mut out = [0.0f32; 4];
/// block.process(&loud, &mut out)?;
/// assert_eq!(out[0], 1.0); // limited to the peak
/// assert!(out.iter().all(|v| v.abs() <= 1.0));
/// block.set_param("limit", ParamValue::F64(0.5))?;
/// block.process(&loud, &mut out)?;
/// assert!(out.iter().all(|v| v.abs() <= 0.5));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct PreemphLimitBlock<T: BlockReal, S: Storage> {
    config: PreemphLimitConfig,
    state: PreemphLimitState<T>,
    _storage: PhantomData<fn() -> S>,
}

impl<T: BlockReal, S: Storage> PreemphLimitBlock<T, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &PreemphLimitConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        Ok(Self {
            config: *config,
            state: PreemphLimitState::new(
                config.coefs(rate_in)?,
                config.limit,
                config.hold_s,
                config.release_s,
                rate_in,
            )?,
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &PreemphLimitConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    pub fn rate_out(config: &PreemphLimitConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }

    /// Current gain reduction of the limiter (1.0 while idle), for meters.
    pub fn gain(&self) -> f64 {
        self.state.gain()
    }
}

impl<T: BlockReal, S: Storage> Block for PreemphLimitBlock<T, S>
where
    Sdrk: PreemphLimit<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as PreemphLimit<T>>::preemph_limit(
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
            "limit" => {
                let v = value.as_f64()?;
                self.state.set_limit(v)?;
                self.config.limit = v;
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
