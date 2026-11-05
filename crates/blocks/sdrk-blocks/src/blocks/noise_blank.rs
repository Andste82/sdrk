//! `noise_blank`: removes short impulses from a complex signal by setting them to zero.

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::noise_blank::{NoiseBlank, NoiseBlankState};
use sdrk_core::timing::Timing;

use super::{unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Description of the `noise_blank` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "noise_blank",
    summary: "Noise blanker: sets short strong impulses of a complex signal to zero.",
    details: "Tracks the average power p of the input with a one-pole filter (time constant \
        tau_s). A sample whose power exceeds p by threshold_db is an impulse (ignition noise, \
        power-line spikes): the output is set to zero from lookahead_s before the impulse \
        (the rising edge) until hold_s after its last sample; the power of an impulse never \
        enters the average. Everything else passes unchanged, delayed by the lookahead of \
        round(lookahead_s fs) samples, which also is the group delay; the ring of that many \
        samples needs storage. A signal without impulses passes bit for bit. Place it directly \
        after convert (or after a wide first decimation) and before the ddc: on the wide-band \
        signal an impulse is short and much stronger than the average, after the channel \
        filter it is stretched into a click that the blanker can no longer separate. Remove \
        the block when there is no impulse noise. Pitfalls: a threshold that is too low blanks \
        peaks of the wanted signal (strong modulation, fading); a long hold_s cuts holes into \
        the signal that the channel filter turns into audible gaps.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples of the wide-band signal, before the channel filter.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "The input delayed by the lookahead, impulses set to zero; same type and \
            rate as the input.",
    }],
    params: &[
        ParamDesc {
            name: "threshold_db",
            unit: Unit::Decibel,
            kind: ParamKind::F64 {
                min: 6.0,
                max: 40.0,
                default: Some(15.0),
            },
            typical: "15; lower blanks weaker impulses but also signal peaks",
            mutability: Mutability::Runtime,
            description: "A sample is an impulse when its power exceeds the average power by \
                this many dB; a change applies from the next sample.",
        },
        ParamDesc {
            name: "tau_s",
            unit: Unit::Seconds,
            kind: ParamKind::F64 {
                min: 0.001,
                max: 1.0,
                default: Some(0.01),
            },
            typical: "0.01",
            mutability: Mutability::Build,
            description: "Time constant of the power average in seconds.",
        },
        ParamDesc {
            name: "hold_s",
            unit: Unit::Seconds,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 0.01,
                default: Some(0.0002),
            },
            typical: "0.0002",
            mutability: Mutability::Build,
            description: "Blanking time after the last impulse sample in seconds, rounded to \
                whole samples.",
        },
        ParamDesc {
            name: "lookahead_s",
            unit: Unit::Seconds,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 0.001,
                default: Some(0.00005),
            },
            typical: "0.00005",
            mutability: Mutability::Build,
            description: "Blanking time before an impulse in seconds, rounded to whole \
                samples; it is also the delay of the block.",
        },
    ],
    required: &[],
};

/// Configuration of [`NoiseBlankBlock`].
///
/// ```
/// use sdrk_blocks::NoiseBlankConfig;
/// let cfg = NoiseBlankConfig::default();
/// assert_eq!(cfg.threshold_db, 15.0);
/// let rate = sdrk::Rate::hz(192_000)?;
/// assert_eq!(cfg.lookahead_samples(rate)?, 10);
/// assert!(NoiseBlankConfig { threshold_db: 5.0, ..cfg }.validate(rate).is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseBlankConfig {
    /// Impulse threshold above the average power in dB (6 to 40).
    pub threshold_db: f64,
    /// Time constant of the power average in seconds (0.001 to 1).
    pub tau_s: f64,
    /// Blanking time after the last impulse sample in seconds (0 to 0.01).
    pub hold_s: f64,
    /// Blanking time before an impulse in seconds (0 to 0.001); the delay of the block.
    pub lookahead_s: f64,
}

impl Default for NoiseBlankConfig {
    fn default() -> Self {
        Self {
            threshold_db: 15.0,
            tau_s: 0.01,
            hold_s: 0.0002,
            lookahead_s: 0.00005,
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

impl NoiseBlankConfig {
    /// Checks the configuration against the input sample rate.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        check_range("threshold_db", self.threshold_db, 6.0, 40.0)?;
        check_range("tau_s", self.tau_s, 0.001, 1.0)?;
        check_range("hold_s", self.hold_s, 0.0, 0.01)?;
        check_range("lookahead_s", self.lookahead_s, 0.0, 0.001)?;
        // The kernel checks the smoothing coefficient against the rate; the length of the
        // delay line follows from lookahead_s and cannot exceed the kernel's limit.
        NoiseBlankState::<C32, [C32; 0]>::new(
            self.threshold_db,
            self.tau_s,
            self.hold_s,
            [],
            rate_in,
        )?;
        Ok(())
    }

    /// The lookahead `D = floor(lookahead_s * fs + 0.5)` in samples: the length of the delay
    /// line and the delay of the block (also checks the configuration).
    pub fn lookahead_samples(&self, rate_in: Rate) -> Result<usize> {
        self.validate(rate_in)?;
        Ok(libm::floor(self.lookahead_s * super::hz(rate_in) + 0.5) as usize)
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "threshold_db" => self.threshold_db = value.as_f64()?,
            "tau_s" => self.tau_s = value.as_f64()?,
            "hold_s" => self.hold_s = value.as_f64()?,
            "lookahead_s" => self.lookahead_s = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "threshold_db" => Ok(ParamValue::F64(self.threshold_db)),
            "tau_s" => Ok(ParamValue::F64(self.tau_s)),
            "hold_s" => Ok(ParamValue::F64(self.hold_s)),
            "lookahead_s" => Ok(ParamValue::F64(self.lookahead_s)),
            _ => unknown(),
        }
    }
}

/// Noise blanker for `C32` and `CQ15`.
///
/// ```
/// use sdrk_blocks::{AlignedBytes, ArenaStorage, Block, NoiseBlankBlock, NoiseBlankConfig};
/// use sdrk::{Rate, C32};
/// // Lookahead 2 samples and hold 1 sample at 48 kHz.
/// let cfg = NoiseBlankConfig { hold_s: 1.0 / 48_000.0, lookahead_s: 2.0 / 48_000.0, ..NoiseBlankConfig::default() };
/// let rate = Rate::hz(48_000)?;
/// let bytes = NoiseBlankBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// assert_eq!(bytes, 16);
/// let mut mem = AlignedBytes([0u8; 16]);
/// let mut arena = ArenaStorage::new(&mut mem.0[..bytes]);
/// let mut block = NoiseBlankBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// let mut input = [C32::new(0.01, 0.0); 1000];
/// input[500] = C32::new(1.0, 0.0);
/// let mut out = [C32::default(); 1000];
/// block.process(&input, &mut out)?;
/// // Delayed by 2; the two samples before the impulse, the impulse and one after are zero.
/// assert_eq!(out[2], C32::new(0.01, 0.0));
/// assert!(out[500..504].iter().all(|y| *y == C32::default()));
/// assert_eq!(out[504], C32::new(0.01, 0.0));
/// assert_eq!(block.blanked(), 4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct NoiseBlankBlock<T: BlockSample, S: Storage> {
    config: NoiseBlankConfig,
    state: NoiseBlankState<T, S::Buf<T>>,
}

impl<T: BlockSample, S: Storage> core::fmt::Debug for NoiseBlankBlock<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NoiseBlankBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<T: BlockSample, S: Storage> NoiseBlankBlock<T, S> {
    /// Creates the block; the delay line of `D` samples comes from `storage`.
    pub fn new(config: &NoiseBlankConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        let d = config.lookahead_samples(rate_in)?;
        let history = storage.alloc::<T>(d)?;
        Ok(Self {
            config: *config,
            state: NoiseBlankState::new(
                config.threshold_db,
                config.tau_s,
                config.hold_s,
                history,
                rate_in,
            )?,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address: `D` samples of `T`.
    ///
    /// ```
    /// use sdrk_blocks::{ArenaStorage, NoiseBlankBlock, NoiseBlankConfig};
    /// use sdrk::{Rate, C32, CQ15};
    /// let cfg = NoiseBlankConfig::default();
    /// let rate = Rate::hz(192_000)?; // D = 10
    /// assert_eq!(NoiseBlankBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?, 80);
    /// assert_eq!(NoiseBlankBlock::<CQ15, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?, 40);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn storage_bytes(config: &NoiseBlankConfig, rate_in: Rate) -> Result<usize> {
        let d = config.lookahead_samples(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<T>(d)?;
        Ok(plan.bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    pub fn rate_out(config: &NoiseBlankConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }

    /// The number of output samples set to zero since construction or the last reset.
    pub fn blanked(&self) -> u64 {
        self.state.blanked()
    }

    /// The lookahead `D` in samples (the delay of the block).
    pub fn lookahead(&self) -> usize {
        self.state.lookahead()
    }
}

impl<T: BlockSample, S: Storage> Block for NoiseBlankBlock<T, S>
where
    Sdrk: NoiseBlank<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as NoiseBlank<T>>::noise_blank(
            &Sdrk,
            &mut self.state,
            super::head(input, n)?,
            super::head_mut(output, n)?,
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

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match runtime_param(&DESCRIPTION, name, &value)? {
            "threshold_db" => {
                let v = value.as_f64()?;
                self.state.set_threshold_db(v)?;
                self.config.threshold_db = v;
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
