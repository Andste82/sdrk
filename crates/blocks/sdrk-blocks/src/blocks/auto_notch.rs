//! `auto_notch`: adaptive filter that removes steady tones from real audio.

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::auto_notch::{AutoNotch, AutoNotchState};
use sdrk_core::timing::Timing;

use super::{check_count, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};

/// Smallest and largest number of taps.
const TAPS: (usize, usize) = (8, 256);
/// Smallest and largest delay in samples.
const DELAY: (usize, usize) = (1, 64);
/// Smallest and largest step size.
const MU: (f64, f64) = (0.0001, 0.1);

/// Description of the `auto_notch` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "auto_notch",
    summary: "Automatic notch: removes steady tones (carriers, heterodynes) from real audio.",
    details: "A normalized LMS line enhancer: an adaptive filter of taps weights predicts every \
        sample from the samples delay and more samples back. A steady tone is predictable over \
        that distance, noise and speech are not, so the prediction error, which is the output, \
        keeps the noise and the speech and loses the tones; the filter follows a tone whose \
        frequency drifts and removes several tones at once. Larger mu adapts faster but \
        removes more of the wanted audio. Weights and history need taps + (taps + delay) f32 \
        values of storage. The block has no delay. Place it after the demodulator and before \
        the AGC of the audio, if any. Pitfalls: the notch also removes a wanted steady tone \
        (CW, test tones), so it is off unless placed in the flow; remove it for CW and data \
        modes; a non-finite input sample gives an output of zero.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32],
        description: "Real audio samples (f32 only).",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32],
        description: "The audio without its steady tones; same rate as the input.",
    }],
    params: &[
        ParamDesc {
            name: "taps",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: TAPS.0 as i64,
                max: TAPS.1 as i64,
                default: Some(32),
            },
            typical: "32 at 8 to 12 kHz",
            mutability: Mutability::Build,
            description: "Number of weights of the adaptive filter, 8 to 256.",
        },
        ParamDesc {
            name: "delay",
            unit: Unit::Samples,
            kind: ParamKind::I64 {
                min: DELAY.0 as i64,
                max: DELAY.1 as i64,
                default: Some(8),
            },
            typical: "8",
            mutability: Mutability::Build,
            description: "Distance in samples between a sample and the newest sample of its \
                prediction, 1 to 64; it must exceed the correlation length of the wanted audio.",
        },
        ParamDesc {
            name: "mu",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: MU.0,
                max: MU.1,
                default: Some(0.01),
            },
            typical: "0.01; 0.001 for slow, gentle adaptation",
            mutability: Mutability::Runtime,
            description: "Step size of the normalized LMS update (used as f32); a change \
                applies from the next sample.",
        },
    ],
    required: &[],
};

/// Configuration of [`AutoNotchRealBlock`].
///
/// ```
/// use sdrk_blocks::AutoNotchConfig;
/// let cfg = AutoNotchConfig::default();
/// assert_eq!((cfg.taps, cfg.delay, cfg.mu), (32, 8, 0.01));
/// assert!(cfg.validate(sdrk::Rate::hz(8_000)?).is_ok());
/// assert!(AutoNotchConfig { taps: 4, ..cfg }.validate(sdrk::Rate::hz(8_000)?).is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoNotchConfig {
    /// Number of weights (8 to 256).
    pub taps: usize,
    /// Delay of the prediction in samples (1 to 64).
    pub delay: usize,
    /// Step size of the normalized LMS update (0.0001 to 0.1); the kernel uses it as `f32`.
    pub mu: f64,
}

impl Default for AutoNotchConfig {
    fn default() -> Self {
        Self {
            taps: 32,
            delay: 8,
            mu: 0.01,
        }
    }
}

fn check_mu(mu: f64) -> Result<()> {
    if mu >= MU.0 && mu <= MU.1 {
        Ok(())
    } else {
        Err(Error::OutOfRange {
            name: "mu",
            min: MU.0,
            max: MU.1,
            got: mu,
        })
    }
}

impl AutoNotchConfig {
    /// Checks the configuration (the rate does not matter).
    pub fn validate(&self, _rate_in: Rate) -> Result<()> {
        check_count("taps", self.taps, TAPS.0, TAPS.1)?;
        check_count("delay", self.delay, DELAY.0, DELAY.1)?;
        check_mu(self.mu)
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "taps" => self.taps = value.as_usize()?,
            "delay" => self.delay = value.as_usize()?,
            "mu" => self.mu = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "taps" => Ok(ParamValue::I64(self.taps as i64)),
            "delay" => Ok(ParamValue::I64(self.delay as i64)),
            "mu" => Ok(ParamValue::F64(self.mu)),
            _ => unknown(),
        }
    }
}

/// Automatic notch for `f32` audio.
///
/// ```
/// use sdrk_blocks::{AlignedBytes, ArenaStorage, AutoNotchConfig, AutoNotchRealBlock, Block};
/// use sdrk::Rate;
/// let cfg = AutoNotchConfig { mu: 0.05, ..AutoNotchConfig::default() };
/// let rate = Rate::hz(8_000)?;
/// let bytes = AutoNotchRealBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// assert_eq!(bytes, 4 * (32 + 40));
/// let mut mem = AlignedBytes([0u8; 288]);
/// let mut arena = ArenaStorage::new(&mut mem.0[..bytes]);
/// let mut block = AutoNotchRealBlock::new(&cfg, rate, &mut arena)?;
/// // A 1 kHz whistle is gone after the adaptation.
/// let input: Vec<f32> =
///     (0..8000).map(|n| 0.5 * libm::sinf(2.0 * core::f32::consts::PI * 0.125 * n as f32)).collect();
/// let mut out = vec![0.0f32; 8000];
/// block.process(&input, &mut out)?;
/// assert!(out[7000..].iter().all(|y| y.abs() < 0.01));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct AutoNotchRealBlock<S: Storage> {
    config: AutoNotchConfig,
    state: AutoNotchState<S::Buf<f32>, S::Buf<f32>>,
}

impl<S: Storage> core::fmt::Debug for AutoNotchRealBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AutoNotchRealBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> AutoNotchRealBlock<S> {
    /// Creates the block; the `taps` weights and the history of `taps + delay` samples come
    /// from `storage`, in this order.
    pub fn new(config: &AutoNotchConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let weights = storage.alloc::<f32>(config.taps)?;
        let history = storage.alloc::<f32>(config.taps + config.delay)?;
        Ok(Self {
            config: *config,
            state: AutoNotchState::new(weights, history, config.delay, config.mu as f32)?,
        })
    }

    /// Exact bytes of storage the block needs: `taps + (taps + delay)` values of `f32`.
    pub fn storage_bytes(config: &AutoNotchConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<f32>(config.taps)?;
        plan.add::<f32>(config.taps + config.delay)?;
        Ok(plan.bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    pub fn rate_out(config: &AutoNotchConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }

    /// The weights of the adaptive filter; `w[i]` multiplies the input `delay + i` samples
    /// back.
    pub fn weights(&self) -> &[f32] {
        self.state.weights()
    }
}

impl<S: Storage> Block for AutoNotchRealBlock<S> {
    type In = f32;
    type Out = f32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as AutoNotch>::auto_notch(
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
            "mu" => {
                let v = value.as_f64()?;
                self.state.set_mu(v as f32)?;
                self.config.mu = v;
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
