//! `stereo_pll`: pilot phase-locked loop that regenerates the 38 kHz subcarrier and
//! demodulates the difference signal.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::stereo::{StereoPll, StereoPllState};

use super::{HZ_LIMIT, head, head_mut, lockstep, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::Two;
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Names of the values of the `stereo` parameter.
const MODES: [&str; 2] = ["auto", "mono"];

/// Description of the `stereo_pll` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "stereo_pll",
    summary: "Locks a loop to the 19 kHz stereo pilot and demodulates the 38 kHz difference.",
    details: "The first input is the multiplex signal of an FM broadcast (sum, pilot and the \
        difference around 38 kHz), the second the same signal through a band-pass at the pilot \
        (for example a biquad at 19 kHz). A phase-locked loop follows the pilot; its phase \
        regenerates the 38 kHz carrier, which is multiplied with the multiplex signal. The \
        output is complex: re is the multiplex signal (the sum) and im is the multiplex signal \
        times sin(2 theta), the demodulated difference; both still need an audio low-pass \
        before stereo_matrix. While the pilot is absent or the stereo parameter is mono, im is \
        0. The inputs advance together and must have the same length in a call. This is the \
        loop that fm_stereo_demod uses internally. pilot_level is the nominal amplitude of the \
        pilot in the input signal (0.1 for f32 multiplex normalised to the deviation; for Q15 \
        0.1 times the discriminator scale). Pitfall: the pilot must be below a quarter of the \
        sample rate and natural_hz at most a hundredth of the pilot frequency.",
    inputs: &[
        PortDesc {
            name: "mpx",
            types: &[PortType::F32, PortType::Q15],
            description: "The multiplex signal.",
        },
        PortDesc {
            name: "pilot",
            types: &[PortType::F32, PortType::Q15],
            description: "The multiplex signal through a band-pass at the pilot frequency.",
        },
    ],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples: re is the multiplex signal, im the demodulated \
            difference (f32 gives C32, Q15 gives CQ15).",
    }],
    params: &[
        ParamDesc {
            name: "pilot_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1.0,
                max: HZ_LIMIT,
                default: Some(19_000.0),
            },
            typical: "19e3 for broadcast FM",
            mutability: Mutability::Build,
            description: "Nominal pilot frequency in Hz; below a quarter of the sample rate.",
        },
        ParamDesc {
            name: "natural_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1.0,
                max: 190.0,
                default: Some(20.0),
            },
            typical: "20",
            mutability: Mutability::Build,
            description: "Natural frequency of the loop in Hz, 1 to 190 and at most a hundredth \
                of pilot_hz.",
        },
        ParamDesc {
            name: "pilot_level",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: 1.0,
                default: Some(0.1),
            },
            typical: "0.1; for Q15 0.1 times the scale of the discriminator output",
            mutability: Mutability::Build,
            description: "Nominal pilot amplitude in units of the input signal, above 0 and at \
                most 1; sets the loop gain and the lock thresholds.",
        },
        ParamDesc {
            name: "stereo",
            unit: Unit::None,
            kind: ParamKind::Choice {
                options: &MODES,
                default: 0,
            },
            typical: "auto; mono for a weak station",
            mutability: Mutability::Runtime,
            description: "auto outputs the difference while the pilot is locked; mono forces the \
                difference to 0. Takes effect from the next call.",
        },
    ],
    required: &[],
};

/// Configuration of [`StereoPllBlock`].
///
/// ```
/// use sdrk_blocks::StereoPllConfig;
/// let cfg = StereoPllConfig::default();
/// assert!(cfg.validate(sdrk::Rate::hz(192_000)?).is_ok());
/// // The pilot must be below a quarter of the sample rate.
/// assert!(cfg.validate(sdrk::Rate::hz(48_000)?).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoPllConfig {
    /// Nominal pilot frequency in Hz.
    pub pilot_hz: f64,
    /// Natural frequency of the loop in Hz.
    pub natural_hz: f64,
    /// Nominal pilot amplitude in units of the input signal.
    pub pilot_level: f64,
    /// Forces the difference to 0 (`stereo` = `mono`).
    pub mono: bool,
}

impl Default for StereoPllConfig {
    fn default() -> Self {
        Self {
            pilot_hz: 19_000.0,
            natural_hz: 20.0,
            pilot_level: 0.1,
            mono: false,
        }
    }
}

impl StereoPllConfig {
    /// Checks the ranges of the parameters and the pilot and loop frequency against the sample
    /// rate. (`Q15` has extra requirements on the loop coefficients; the block constructor
    /// checks them.)
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        for (name, v) in [
            ("pilot_hz", self.pilot_hz),
            ("natural_hz", self.natural_hz),
            ("pilot_level", self.pilot_level),
        ] {
            DESCRIPTION
                .param(name)
                .ok_or(Error::UnknownParameter)?
                .check(&ParamValue::F64(v))?;
        }
        StereoPllState::<f32>::new(self.pilot_hz, self.natural_hz, self.pilot_level, rate_in)
            .map(|_| ())
            .map_err(Error::from)
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "pilot_hz" => self.pilot_hz = value.as_f64()?,
            "natural_hz" => self.natural_hz = value.as_f64()?,
            "pilot_level" => self.pilot_level = value.as_f64()?,
            "stereo" => self.mono = value.as_choice()? == 1,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "pilot_hz" => Ok(ParamValue::F64(self.pilot_hz)),
            "natural_hz" => Ok(ParamValue::F64(self.natural_hz)),
            "pilot_level" => Ok(ParamValue::F64(self.pilot_level)),
            "stereo" => Ok(ParamValue::Choice(u16::from(self.mono))),
            _ => unknown(),
        }
    }
}

/// Pilot loop and difference demodulator: `f32` pairs to `C32`, `Q15` pairs to `CQ15` (`T` is
/// the complex output type).
///
/// The inputs are a [`Two`] port: `mpx` and `pilot` have the same length, otherwise `process`
/// returns `LengthMismatch` and changes nothing. The output equals the kernel `stereo_pll` and
/// the internal loop of `fm_stereo_demod` bit for bit on the same input. The block needs no
/// storage.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, StereoPllBlock, StereoPllConfig};
/// use sdrk::{Rate, C32};
/// let rate = Rate::hz(192_000)?;
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut block = StereoPllBlock::<C32, _>::new(&StereoPllConfig::default(), rate, &mut arena)?;
/// // Without a pilot the loop stays unlocked and the difference is 0.
/// let mpx = [0.5f32; 64];
/// let pilot = [0.0f32; 64];
/// let mut out = [C32::new(9.0, 9.0); 64];
/// let p = block.process((&mpx, &pilot), &mut out)?;
/// assert_eq!((p.consumed, p.produced), (64, 64));
/// assert_eq!(out[63], C32::new(0.5, 0.0));
/// assert!(!block.pilot_locked());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct StereoPllBlock<T: BlockSample, S: Storage> {
    config: StereoPllConfig,
    state: StereoPllState<T::Real>,
    _marker: PhantomData<fn() -> (T, S)>,
}

impl<T: BlockSample, S: Storage> core::fmt::Debug for StereoPllBlock<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StereoPllBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<T: BlockSample, S: Storage> StereoPllBlock<T, S> {
    fn make_state(config: &StereoPllConfig, rate_in: Rate) -> Result<StereoPllState<T::Real>> {
        config.validate(rate_in)?;
        Ok(StereoPllState::new(
            config.pilot_hz,
            config.natural_hz,
            config.pilot_level,
            rate_in,
        )?)
    }

    /// Creates the block; it needs no storage.
    pub fn new(config: &StereoPllConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        let mut state = Self::make_state(config, rate_in)?;
        state.set_enabled(!config.mono);
        Ok(Self {
            config: *config,
            state,
            _marker: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &StereoPllConfig, rate_in: Rate) -> Result<usize> {
        Self::make_state(config, rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate (equal to the input rate).
    pub fn rate_out(config: &StereoPllConfig, rate_in: Rate) -> Result<Rate> {
        Self::make_state(config, rate_in)?;
        Ok(rate_in)
    }

    /// Whether the loop has locked to the pilot (`is_stereo()` of the loop state); the
    /// decision does not depend on the `stereo` parameter, which only silences the difference.
    pub fn pilot_locked(&self) -> bool {
        self.state.is_stereo()
    }
}

impl<T: BlockSample, S: Storage> Block for StereoPllBlock<T, S>
where
    Sdrk: StereoPll<T::Real>,
{
    type In = Two<T::Real, T::Real>;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: (&[T::Real], &[T::Real]), output: &mut [T]) -> Result<Progress> {
        let (mpx, pilot) = input;
        let n = lockstep(mpx, pilot)?.min(output.len());
        <Sdrk as StereoPll<T::Real>>::stereo_pll(
            &Sdrk,
            &mut self.state,
            head(mpx, n)?,
            head(pilot, n)?,
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
            "stereo" => {
                let mono = value.as_choice()? == 1;
                self.config.mono = mono;
                self.state.set_enabled(!mono);
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
