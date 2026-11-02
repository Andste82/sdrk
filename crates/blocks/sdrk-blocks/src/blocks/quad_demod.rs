//! `quad_demod`: quadrature (FM) discriminator on its own.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::demod::{QuadDemod, QuadDemodState};
use sdrk_core::timing::Timing;

use super::{HZ_LIMIT, head, head_mut, hz, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Description of the `quad_demod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "quad_demod",
    summary: "Quadrature (FM) discriminator: the phase step between consecutive samples.",
    details: "Computes the angle of x[n] times the conjugate of x[n-1], which is proportional to \
        the instantaneous frequency of the signal. It is the first stage of an FM receiver \
        (after the channel filter, before de-emphasis and audio decimation) and has the same \
        gain as the discriminator inside fm_demod. With C32 the gain is fs / (2 pi deviation), \
        so a carrier at the full deviation reads 1.0. With CQ15 the gain is the largest Q15 \
        value (raw 32767) and the output unit is phase per sample divided by pi; the deviation \
        is not used. The first output after construction or reset looks at a previous sample of \
        zero and reads 0. Pitfall: the input rate should be 2 to 4 times the signal bandwidth, \
        otherwise the phase step wraps.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex baseband samples of the FM signal.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32, PortType::Q15],
        description: "Demodulated signal (f32 for C32 input, Q15 for CQ15 input), one output per input.",
    }],
    params: &[ParamDesc {
        name: "deviation_hz",
        unit: Unit::Hz,
        kind: ParamKind::F64 {
            min: 1e-6,
            max: HZ_LIMIT,
            default: Some(75_000.0),
        },
        typical: "75e3 for broadcast FM, 5e3 for narrow-band voice",
        mutability: Mutability::Build,
        description: "Peak deviation in Hz that reads full scale (used by C32 only).",
    }],
    required: &[],
};

/// Configuration of [`QuadDemodBlock`].
///
/// ```
/// use sdrk_blocks::QuadDemodConfig;
/// let cfg = QuadDemodConfig { deviation_hz: 5_000.0 };
/// assert!(cfg.validate(sdrk::Rate::hz(48_000)?).is_ok());
/// assert!(QuadDemodConfig { deviation_hz: 0.0 }.validate(sdrk::Rate::hz(48_000)?).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadDemodConfig {
    /// Peak deviation in Hz that reads full scale (C32 only).
    pub deviation_hz: f64,
}

impl Default for QuadDemodConfig {
    fn default() -> Self {
        Self {
            deviation_hz: 75_000.0,
        }
    }
}

impl QuadDemodConfig {
    /// Checks the configuration against the input sample rate.
    pub fn validate(&self, _rate_in: Rate) -> Result<()> {
        let v = self.deviation_hz;
        if v > 0.0 && v <= HZ_LIMIT {
            Ok(())
        } else {
            Err(Error::OutOfRange {
                name: "deviation_hz",
                min: 0.0,
                max: HZ_LIMIT,
                got: v,
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

/// Quadrature demodulator: `C32` to `f32` or `CQ15` to `Q15` (`T` is the complex input type).
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, QuadDemodBlock, QuadDemodConfig};
/// use sdrk::{Rate, C32};
/// let rate = Rate::hz(240_000)?;
/// let cfg = QuadDemodConfig::default();
/// assert_eq!(QuadDemodBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?, 0);
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut block = QuadDemodBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// // A carrier 75 kHz above the centre reads +1.0 once the discriminator has two samples.
/// let w = 2.0 * core::f64::consts::PI * 75_000.0 / 240_000.0;
/// let input: Vec<C32> = (0..8).map(|n| C32::new((w * n as f64).cos() as f32, (w * n as f64).sin() as f32)).collect();
/// let mut out = [0.0f32; 8];
/// block.process(&input, &mut out)?;
/// assert!((out[7] - 1.0).abs() < 1e-4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct QuadDemodBlock<T: BlockSample, S: Storage> {
    config: QuadDemodConfig,
    state: QuadDemodState<T>,
    gain: T::Real,
    _storage: PhantomData<fn() -> S>,
}

impl<T: BlockSample, S: Storage> QuadDemodBlock<T, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &QuadDemodConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        Ok(Self {
            config: *config,
            state: QuadDemodState::new(),
            gain: T::demod_gain(config.deviation_hz, hz(rate_in)),
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &QuadDemodConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    pub fn rate_out(config: &QuadDemodConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }
}

impl<T: BlockSample, S: Storage> Block for QuadDemodBlock<T, S>
where
    Sdrk: QuadDemod<T>,
{
    type In = T;
    type Out = T::Real;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T::Real]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as QuadDemod<T>>::quad_demod(
            &Sdrk,
            &mut self.state,
            head(input, n)?,
            head_mut(output, n)?,
            self.gain,
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

    fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
        if DESCRIPTION.param(name).is_some() {
            Err(Error::ReadOnlyParameter)
        } else {
            unknown()
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        1
    }
}
