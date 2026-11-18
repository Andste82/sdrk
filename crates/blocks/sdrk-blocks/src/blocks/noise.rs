//! `noise`: deterministic white Gaussian noise, one sample per input sample.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::noise::{GaussNoise, GaussNoiseState};
use sdrk_core::timing::Timing;

use super::{head_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::sample::{BlockReal, BlockSample};
use crate::storage::{Plan, Storage};

/// Largest `seed` and `stream`: 2^53 - 1, exact as a JSON number and in JavaScript.
const SEED_MAX: i64 = (1 << 53) - 1;

/// Description of the `noise` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "noise",
    summary: "Generates white Gaussian noise of standard deviation sigma, one sample per input \
        sample.",
    details: "A deterministic Gaussian noise source (kernel gauss_noise: PCG32, one draw per \
        real value, mapped by a tabulated inverse normal CDF): for every input sample it outputs \
        one noise sample of the same type. The input values are never read; the input only \
        provides the clock, so a flow needs no source node. Sample n of the output is sample n \
        of the sequence of seed and stream since the construction or the last reset, the same \
        on every platform. sigma is the standard deviation of each real component (the real \
        value, or the real and the imaginary part of a complex value), so a complex sample has \
        the power N0 = 2 sigma^2. GNU Radio's noise_source_c with amplitude A is sigma = A / \
        sqrt(2); liquid-dsp's cawgn with nstd is sigma = nstd / sqrt(2). Every value lies \
        within 6.338 sigma, so the noise is not meant for bit or false-alarm rates below about \
        1e-8. Q15 output saturates at full scale. sigma can change while the block runs and the \
        sequence keeps its position; reset restarts at sample 0. Additive channel noise is an \
        add of the signal and a noise node on the same clock.",
    inputs: &[PortDesc {
        name: "clock",
        types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
        description: "Any stream; only its length matters.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
        description: "Gaussian noise, same type and rate as the input.",
    }],
    params: &[
        ParamDesc {
            name: "sigma",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 16.0,
                default: Some(0.1),
            },
            typical: "0.1; 0.01 for a noise floor 40 dB below full scale",
            mutability: Mutability::Runtime,
            description: "Standard deviation of each real component as a linear amplitude, 0 \
                to 16; 0 gives zeros and keeps the sequence running.",
        },
        ParamDesc {
            name: "seed",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 0,
                max: SEED_MAX,
                default: Some(0),
            },
            typical: "0; any other value gives another sequence",
            mutability: Mutability::Build,
            description: "Seed of the generator, 0 to 2^53 - 1.",
        },
        ParamDesc {
            name: "stream",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 0,
                max: SEED_MAX,
                default: Some(1),
            },
            typical: "1; two nodes with the same seed and different streams are independent",
            mutability: Mutability::Build,
            description: "Stream of the generator, one of many independent sequences of the same \
                seed, 0 to 2^53 - 1.",
        },
    ],
    required: &[],
};

/// Configuration of [`NoiseBlock`] and [`NoiseRealBlock`].
///
/// `sigma` is the standard deviation of each real component (power `N0 = 2 sigma²` per complex
/// sample), as in `sdrk_testkit::channel::Awgn`. GNU Radio's `noise_source_c` with amplitude `A`
/// corresponds to `sigma = A / sqrt(2)` (its `noise_source_f` to `sigma = A`), liquid-dsp's
/// `cawgn` with `nstd` to `sigma = nstd / sqrt(2)`. Their generators differ from ours, so only
/// the statistics agree, not the samples; the same holds for the testkit's `Awgn`.
///
/// **Tail limit.** Every value lies within `6.338 sigma` (`830729 · 2^-17 · sigma`, one 32-bit
/// draw per real value through the inverse normal CDF of the kernel `gauss_noise`). The truncated
/// mass is about 2.3e-10 and the far tail is discrete, so the noise is **not for bit-error or
/// false-alarm studies below about 1e-8** (a BPSK BER at 1e-8 is off by about 1 %).
///
/// ```
/// use sdrk_blocks::NoiseConfig;
/// let cfg = NoiseConfig { sigma: 0.25, ..NoiseConfig::default() };
/// assert!(cfg.validate(sdrk::Rate::hz(48_000)?).is_ok());
/// // sigma is at most 16.
/// assert!(NoiseConfig { sigma: 16.5, ..cfg }.validate(sdrk::Rate::hz(48_000)?).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseConfig {
    /// Standard deviation of each real component, 0 to 16 (the kernel takes it as `f32`).
    pub sigma: f64,
    /// Seed of the generator, 0 to 2^53 - 1.
    pub seed: u64,
    /// Stream of the generator, 0 to 2^53 - 1.
    pub stream: u64,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            sigma: 0.1,
            seed: 0,
            stream: 1,
        }
    }
}

/// `v` as an `I64` parameter value; a value beyond `i64` becomes `i64::MAX`, which is out of
/// range.
fn int_value(v: u64) -> ParamValue {
    ParamValue::I64(i64::try_from(v).unwrap_or(i64::MAX))
}

fn check(name: &str, value: ParamValue) -> Result<()> {
    DESCRIPTION
        .param(name)
        .ok_or(Error::UnknownParameter)?
        .check(&value)
}

impl NoiseConfig {
    /// Checks the configuration (the sample rate does not matter).
    pub fn validate(&self, _rate_in: Rate) -> Result<()> {
        check("sigma", ParamValue::F64(self.sigma))?;
        check("seed", int_value(self.seed))?;
        check("stream", int_value(self.stream))
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "sigma" => self.sigma = value.as_f64()?,
            "seed" => self.seed = u64::try_from(value.as_i64()?).unwrap_or(0),
            "stream" => self.stream = u64::try_from(value.as_i64()?).unwrap_or(0),
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "sigma" => Ok(ParamValue::F64(self.sigma)),
            "seed" => Ok(int_value(self.seed)),
            "stream" => Ok(int_value(self.stream)),
            _ => unknown(),
        }
    }
}

/// State shared by the two noise blocks.
#[derive(Debug)]
struct NoiseCore {
    config: NoiseConfig,
    state: GaussNoiseState,
}

impl NoiseCore {
    fn new(config: &NoiseConfig, rate_in: Rate) -> Result<Self> {
        config.validate(rate_in)?;
        Ok(Self {
            config: *config,
            state: GaussNoiseState::new(config.seed, config.stream, config.sigma as f32)?,
        })
    }

    fn reset(&mut self) {
        self.state.reset();
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match runtime_param(&DESCRIPTION, name, &value)? {
            "sigma" => {
                let s = value.as_f64()?;
                self.state.set_sigma(s as f32)?;
                self.config.sigma = s;
                Ok(())
            }
            _ => unknown(),
        }
    }

    fn generate<T>(&mut self, n: usize, output: &mut [T]) -> Result<Progress>
    where
        Sdrk: GaussNoise<T>,
        T: sdrk_core::Sample,
    {
        let n = n.min(output.len());
        <Sdrk as GaussNoise<T>>::gauss_noise(&Sdrk, &mut self.state, head_mut(output, n)?)?;
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: self.state.ratio(),
            kind: self.state.ratio_kind(),
            delay: self.state.group_delay(),
        }
    }
}

/// Gaussian noise clocked by a complex stream: `C32` to `C32`, `CQ15` to `CQ15`. The values of
/// the input are ignored.
///
/// Sample `n` of the output is sample `n` of the kernel `gauss_noise` with
/// `GaussNoiseState::new(seed, stream, sigma)` since the construction or the last `reset`.
/// `sigma` is per real component, so a complex sample has the power `N0 = 2 sigma²`; GNU
/// Radio's `noise_source_c` with amplitude `A` is `sigma = A / sqrt(2)`, liquid-dsp's `cawgn`
/// with `nstd` is `sigma = nstd / sqrt(2)`. Every component lies within `6.338 sigma`, so the
/// noise is not for bit-error or false-alarm rates below about 1e-8 (see [`NoiseConfig`]). A
/// change of `sigma` takes effect with the next call and keeps the position. For a real clock
/// use [`NoiseRealBlock`]. The block needs no storage.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, NoiseBlock, NoiseConfig, ParamValue};
/// use sdrk::{Rate, C32};
/// let cfg = NoiseConfig { sigma: 0.5, seed: 7, stream: 1 };
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut block = NoiseBlock::<C32, _>::new(&cfg, Rate::hz(48_000)?, &mut arena)?;
/// let mut out = [C32::default(); 4];
/// let p = block.process(&[C32::default(); 4], &mut out)?;
/// assert_eq!((p.consumed, p.produced), (4, 4));
/// // The tail limit: no component exceeds 6.338 sigma.
/// assert!(out.iter().all(|x| x.re.abs() <= 0.5 * 6.338 && x.im.abs() <= 0.5 * 6.338));
/// block.set_param("sigma", ParamValue::F64(0.0))?;
/// block.process(&[C32::default(); 4], &mut out)?;
/// assert!(out.iter().all(|x| *x == C32::default()));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct NoiseBlock<T: BlockSample, S: Storage> {
    core: NoiseCore,
    _marker: PhantomData<fn() -> (T, S)>,
}

impl<T: BlockSample, S: Storage> NoiseBlock<T, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &NoiseConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        Ok(Self {
            core: NoiseCore::new(config, rate_in)?,
            _marker: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &NoiseConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate (equal to the input rate).
    pub fn rate_out(config: &NoiseConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }
}

impl<T: BlockSample, S: Storage> Block for NoiseBlock<T, S>
where
    Sdrk: GaussNoise<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        self.core.generate(input.len(), output)
    }

    fn timing(&self) -> BlockTiming {
        self.core.timing()
    }

    fn reset(&mut self) {
        self.core.reset();
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        self.core.set_param(name, value)
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.core.config.get(name)
    }

    fn min_input(&self) -> usize {
        1
    }
}

/// Gaussian noise clocked by a real stream: `f32` to `f32`, `Q15` to `Q15` (`R` is the input
/// and the output type). The values of the input are ignored; `Q15` saturates at full scale.
///
/// Otherwise like [`NoiseBlock`]: `sigma` is the standard deviation of the real value
/// (GNU Radio's `noise_source_f` with amplitude `A` is `sigma = A`), every value lies within
/// `6.338 sigma`, and the noise is not for bit-error or false-alarm rates below about 1e-8. The
/// block needs no storage.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, NoiseConfig, NoiseRealBlock};
/// use sdrk::{Rate, Q15};
/// let cfg = NoiseConfig { sigma: 0.25, ..NoiseConfig::default() };
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut block = NoiseRealBlock::<Q15, _>::new(&cfg, Rate::hz(8_000)?, &mut arena)?;
/// let mut first = [Q15(0); 3];
/// block.process(&[Q15(0); 3], &mut first)?;
/// // reset starts the sequence again.
/// block.reset();
/// let mut again = [Q15(0); 3];
/// block.process(&[Q15(0); 3], &mut again)?;
/// assert_eq!(first, again);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct NoiseRealBlock<R: BlockReal, S: Storage> {
    core: NoiseCore,
    _marker: PhantomData<fn() -> (R, S)>,
}

impl<R: BlockReal, S: Storage> NoiseRealBlock<R, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &NoiseConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        Ok(Self {
            core: NoiseCore::new(config, rate_in)?,
            _marker: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &NoiseConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate (equal to the input rate).
    pub fn rate_out(config: &NoiseConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }
}

impl<R: BlockReal, S: Storage> Block for NoiseRealBlock<R, S>
where
    Sdrk: GaussNoise<R>,
{
    type In = R;
    type Out = R;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[R], output: &mut [R]) -> Result<Progress> {
        self.core.generate(input.len(), output)
    }

    fn timing(&self) -> BlockTiming {
        self.core.timing()
    }

    fn reset(&mut self) {
        self.core.reset();
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        self.core.set_param(name, value)
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.core.config.get(name)
    }

    fn min_input(&self) -> usize {
        1
    }
}
