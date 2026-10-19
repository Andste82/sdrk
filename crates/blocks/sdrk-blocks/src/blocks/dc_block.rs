//! `dc_block`: one-pole DC-blocking high-pass for complex signals.

use core::f64::consts::PI;
use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::RealSample;
use sdrk_core::kernels::demod::{DcBlock, DcBlockState};

use super::{HZ_LIMIT, head, head_mut, hz, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Description of the `dc_block` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "dc_block",
    summary: "Removes the DC offset of a complex signal with a one-pole high-pass.",
    details: "Implements y[n] = x[n] - x[n-1] + r y[n-1] per component with r = exp(-2 pi \
        cutoff / fs). SDR receivers show a spike at 0 Hz (DC offset and LO leakage); this block \
        removes it. A lower cutoff removes less of the signal around DC but settles more \
        slowly. Pitfall: in Q15 the pole is rounded to a multiple of 2^-15, so for a cutoff \
        far below fs / 20000 the effective cutoff is larger than requested.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "The filtered samples, same type and rate as the input.",
    }],
    params: &[ParamDesc {
        name: "cutoff_hz",
        unit: Unit::Hz,
        kind: ParamKind::F64 {
            min: 1e-6,
            max: HZ_LIMIT,
            default: Some(10.0),
        },
        typical: "10 for a 2.4 MHz receiver",
        mutability: Mutability::Build,
        description: "-3 dB corner of the high-pass in Hz; must be below half the sample rate.",
    }],
    required: &[],
};

/// Configuration of [`DcBlockBlock`].
///
/// ```
/// use sdrk_blocks::DcBlockConfig;
/// let cfg = DcBlockConfig::default();
/// assert_eq!(cfg.cutoff_hz, 10.0);
/// assert!(cfg.validate(sdrk::Rate::hz(48_000)?).is_ok());
/// assert!(DcBlockConfig { cutoff_hz: 30_000.0 }.validate(sdrk::Rate::hz(48_000)?).is_err());
/// # Ok::<(), sdrk::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DcBlockConfig {
    /// Corner frequency in Hz (below half the sample rate).
    pub cutoff_hz: f64,
}

impl Default for DcBlockConfig {
    fn default() -> Self {
        Self { cutoff_hz: 10.0 }
    }
}

impl DcBlockConfig {
    /// Checks the configuration against the input sample rate.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        let half = hz(rate_in) / 2.0;
        if self.cutoff_hz > 0.0 && self.cutoff_hz < half {
            Ok(())
        } else {
            Err(Error::OutOfRange {
                name: "cutoff_hz",
                min: 0.0,
                max: half,
                got: self.cutoff_hz,
            })
        }
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "cutoff_hz" => self.cutoff_hz = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "cutoff_hz" => Ok(ParamValue::F64(self.cutoff_hz)),
            _ => unknown(),
        }
    }
}

/// DC blocker for `C32` and `CQ15`.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, DcBlockBlock, DcBlockConfig};
/// use sdrk::{Rate, C32};
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = DcBlockConfig { cutoff_hz: 100.0 };
/// let mut block = DcBlockBlock::<C32, _>::new(&cfg, Rate::hz(48_000)?, &mut arena)?;
/// let input = [C32::new(1.0, 0.0); 2000];
/// let mut out = [C32::default(); 2000];
/// block.process(&input, &mut out)?;
/// assert!(out[1999].re.abs() < 0.01); // a constant offset decays
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct DcBlockBlock<T: BlockSample, S: Storage> {
    config: DcBlockConfig,
    state: DcBlockState<T>,
    _storage: PhantomData<fn() -> S>,
}

impl<T: BlockSample, S: Storage> DcBlockBlock<T, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &DcBlockConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let r = libm::exp(-2.0 * PI * config.cutoff_hz / hz(rate_in));
        Ok(Self {
            config: *config,
            state: DcBlockState::new(T::Real::from_f64_round(r))?,
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &DcBlockConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    pub fn rate_out(config: &DcBlockConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }
}

impl<T: BlockSample, S: Storage> Block for DcBlockBlock<T, S>
where
    Sdrk: DcBlock<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as DcBlock<T>>::dc_block(
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
