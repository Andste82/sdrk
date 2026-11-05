//! `am_mod`: amplitude modulator (real audio to complex baseband with carrier).

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::modulate::AmMod;

use super::{head, head_mut, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Description of the `am_mod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "am_mod",
    summary: "Amplitude-modulates a real signal onto a complex baseband carrier at 0 Hz.",
    details: "Computes out = (carrier (1 + depth x), 0) for every input sample x, with x \
        clamped to [-1, 1] (a NaN input counts as 0, the bare carrier). The output is a double \
        sideband AM signal with its carrier at 0 Hz of the baseband; a duc after the block moves \
        it to the transmit frequency. The peak amplitude is carrier (1 + depth), at most 1 with \
        the defaults (0.5 and 0.8 give 0.9). The modulation depth can change while the block \
        runs. Put a band-limiting low-pass on the audio in front (a biquad low-pass at 4500 Hz \
        for broadcast AM), otherwise the AM signal is as wide as twice the audio band. \
        Pitfalls: depth 1 with a full-scale input lets the envelope touch 0 (100 % \
        modulation); input beyond full scale is clipped, which splatters; am_demod reads the \
        audio back with the level depth carrier x.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32, PortType::Q15],
        description: "Real modulating signal (audio) with full scale 1.0 (f32) or 32767 (Q15).",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex AM baseband signal at the same rate (f32 gives C32, Q15 gives \
            CQ15).",
    }],
    params: &[
        ParamDesc {
            name: "carrier",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 1.0,
                default: Some(0.5),
            },
            typical: "0.5, so that carrier (1 + depth) stays at most 1",
            mutability: Mutability::Build,
            description: "Amplitude of the carrier, 0 to 1 (full scale 1).",
        },
        ParamDesc {
            name: "depth",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 1.0,
                default: Some(0.8),
            },
            typical: "0.8; 0 sends the bare carrier, 1 is 100 % modulation",
            mutability: Mutability::Runtime,
            description: "Modulation index, 0 to 1.",
        },
    ],
    required: &[],
};

/// Configuration of [`AmModBlock`].
///
/// ```
/// use sdrk_blocks::AmModConfig;
/// let cfg = AmModConfig::default();
/// assert_eq!((cfg.carrier, cfg.depth), (0.5, 0.8));
/// assert!(cfg.validate().is_ok());
/// assert!(AmModConfig { depth: 1.5, ..cfg }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmModConfig {
    /// Amplitude of the carrier, 0 to 1.
    pub carrier: f64,
    /// Modulation index, 0 to 1 (can change while the block runs).
    pub depth: f64,
}

impl Default for AmModConfig {
    fn default() -> Self {
        Self {
            carrier: 0.5,
            depth: 0.8,
        }
    }
}

/// Checks `0 <= v <= 1`, otherwise `OutOfRange`.
fn check_unit(name: &'static str, v: f64) -> Result<()> {
    if (0.0..=1.0).contains(&v) {
        Ok(())
    } else {
        Err(Error::OutOfRange {
            name,
            min: 0.0,
            max: 1.0,
            got: v,
        })
    }
}

impl AmModConfig {
    /// Checks the configuration: `carrier` and `depth` both in `0 ..= 1`.
    pub fn validate(&self) -> Result<()> {
        check_unit("carrier", self.carrier)?;
        check_unit("depth", self.depth)
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// ```
    /// use sdrk_blocks::{AmModConfig, Error, ParamValue};
    /// let mut cfg = AmModConfig::default();
    /// cfg.set("depth", ParamValue::F64(1.0))?;
    /// assert_eq!(cfg.depth, 1.0);
    /// assert!(cfg.set("carrier", ParamValue::F64(2.0)).is_err());
    /// # Ok::<(), Error>(())
    /// ```
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "carrier" => self.carrier = value.as_f64()?,
            "depth" => self.depth = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "carrier" => Ok(ParamValue::F64(self.carrier)),
            "depth" => Ok(ParamValue::F64(self.depth)),
            _ => unknown(),
        }
    }
}

/// AM modulator: `f32` to `C32` or `Q15` to `CQ15` (`T` is the complex output type).
///
/// ```
/// use sdrk_blocks::{AmModBlock, AmModConfig, ArenaStorage, Block};
/// use sdrk::{Rate, C32};
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut block = AmModBlock::<C32, _>::new(&AmModConfig::default(), Rate::hz(48_000)?, &mut arena)?;
/// let mut out = [C32::default(); 3];
/// block.process(&[1.0, 0.0, -1.0], &mut out)?;
/// assert_eq!(out[0], C32::new(0.5 * 1.8, 0.0));
/// assert_eq!(out[1], C32::new(0.5, 0.0)); // silence: the bare carrier
/// assert_eq!(out[2], C32::new(0.5 * (1.0 - 0.8), 0.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct AmModBlock<T: BlockSample, S: Storage> {
    config: AmModConfig,
    _marker: PhantomData<fn() -> (T, S)>,
}

impl<T: BlockSample, S: Storage> AmModBlock<T, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &AmModConfig, _rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config: *config,
            _marker: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &AmModConfig, _rate_in: Rate) -> Result<usize> {
        config.validate()?;
        Ok(Plan::default().bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    pub fn rate_out(config: &AmModConfig, rate_in: Rate) -> Result<Rate> {
        config.validate()?;
        Ok(rate_in)
    }
}

impl<T: BlockSample, S: Storage> Block for AmModBlock<T, S>
where
    Sdrk: AmMod<T>,
{
    type In = T::Real;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T::Real], output: &mut [T]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as AmMod<T>>::am_mod(
            &Sdrk,
            self.config.carrier as f32,
            self.config.depth as f32,
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

    fn reset(&mut self) {}

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match runtime_param(&DESCRIPTION, name, &value)? {
            "depth" => {
                let d = value.as_f64()?;
                check_unit("depth", d)?;
                self.config.depth = d;
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
