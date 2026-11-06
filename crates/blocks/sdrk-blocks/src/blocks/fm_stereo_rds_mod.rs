//! `fm_stereo_rds_mod`: stereo audio and the RDS baseband to the FM multiplex signal (sum,
//! pilot, difference on the 38 kHz subcarrier, RDS on the 57 kHz subcarrier).

use core::marker::PhantomData;

use sdrk::{C32, Progress, Rate, Sdrk};
use sdrk_core::kernels::stereo::{StereoEncodeRds, StereoEncodeRdsState};

use super::{head, head_mut, hz, lockstep, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::Two;
use crate::storage::{Plan, Storage};

/// Lowest input rate of `fm_stereo_rds_mod` in Hz: the RDS band reaches 59.4 kHz, the
/// difference band 53 kHz (the limit of `fm_stereo_mod`).
const MIN_RATE_HZ: f64 = 120_000.0;

/// Description of the `fm_stereo_rds_mod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "fm_stereo_rds_mod",
    summary: "Stereo encoder with RDS: stereo audio and the RDS baseband to the FM multiplex \
        signal.",
    details: "Port a is stereo audio (c32, left channel in re, right channel in im, full \
        scale 1.0), port b the RDS baseband of rds_encode (f32, peak about 1.06) at the same \
        rate. The output is the real multiplex signal main (L + R) / 2 + main (L - R) / 2 \
        sin(2 phi) + pilot sin(phi) + rds b sin(3 phi), the RDS term added last, where phi is \
        the phase of the pilot oscillator at pilot_hz: the 57 kHz carrier of RDS is locked to \
        three times the pilot, as receivers expect (the kind rds_demod). With the defaults \
        (main 0.87, pilot 0.09, rds 0.04) and audio within full scale the peak is at most \
        about 1.002, so the following fm_mod with deviation_hz 75000 stays within about \
        75 kHz. Place it after preemph_limit and the up-sampling resample (port a) and \
        rds_encode clocked by the same stream (port b), at 192 kHz or more, and follow it \
        with fm_mod. pilot 0 turns the pilot off and rds 0 \
        turns RDS off at runtime. Pitfalls: the input rate must be at least 120 kHz; the \
        audio must be band-limited to 15 kHz, otherwise it overlaps the pilot and the \
        subcarriers.",
    inputs: &[
        PortDesc {
            name: "audio",
            types: &[PortType::C32],
            description: "Stereo audio (re left, im right), pre-emphasised and limited, at \
                120 kHz or more.",
        },
        PortDesc {
            name: "rds",
            types: &[PortType::F32],
            description: "The RDS baseband of rds_encode at the same rate.",
        },
    ],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32],
        description: "The real multiplex signal at the same rate, for fm_mod.",
    }],
    params: &[
        ParamDesc {
            name: "pilot_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 18_000.0,
                max: 20_000.0,
                default: Some(19_000.0),
            },
            typical: "19000 (broadcast)",
            mutability: Mutability::Build,
            description: "Frequency of the pilot; the subcarriers are at two and three times \
                this frequency.",
        },
        ParamDesc {
            name: "main",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 1.0,
                default: Some(0.87),
            },
            typical: "0.87 (with pilot 0.09 and rds 0.04: sum 1.0, peak about 1.002)",
            mutability: Mutability::Build,
            description: "Gain of the sum and of the difference signal.",
        },
        ParamDesc {
            name: "pilot",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 0.2,
                default: Some(0.09),
            },
            typical: "0.09 (9 % of the deviation), 0 for mono",
            mutability: Mutability::Runtime,
            description: "Amplitude of the pilot; 0 turns it off.",
        },
        ParamDesc {
            name: "rds",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 0.2,
                default: Some(0.04),
            },
            typical: "0.04 (4 % of the deviation, about 3 kHz)",
            mutability: Mutability::Runtime,
            description: "Gain of the RDS baseband on the 57 kHz subcarrier; 0 turns RDS off.",
        },
    ],
    required: &[],
};

/// Configuration of [`FmStereoRdsModBlock`].
///
/// ```
/// use sdrk_blocks::FmStereoRdsModConfig;
/// let cfg = FmStereoRdsModConfig::default();
/// assert_eq!((cfg.pilot_hz, cfg.main, cfg.pilot, cfg.rds), (19_000.0, 0.87, 0.09, 0.04));
/// assert!(cfg.validate(sdrk::Rate::hz(192_000)?).is_ok());
/// assert!(cfg.validate(sdrk::Rate::hz(96_000)?).is_err());
/// # Ok::<(), sdrk::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FmStereoRdsModConfig {
    /// Pilot frequency in Hz (18000 to 20000).
    pub pilot_hz: f64,
    /// Gain of the sum and difference signals (0 to 1).
    pub main: f64,
    /// Pilot amplitude (0 to 0.2).
    pub pilot: f64,
    /// Gain of the RDS baseband (0 to 0.2).
    pub rds: f64,
}

impl Default for FmStereoRdsModConfig {
    fn default() -> Self {
        Self {
            pilot_hz: 19_000.0,
            main: 0.87,
            pilot: 0.09,
            rds: 0.04,
        }
    }
}

impl FmStereoRdsModConfig {
    /// Checks the configuration against the input sample rate.
    ///
    /// # Errors
    /// `OutOfRange` for a parameter outside its range or a rate below 120 kHz.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        for p in DESCRIPTION.params {
            p.check(&self.get(p.name)?)?;
        }
        if hz(rate_in) < MIN_RATE_HZ {
            return Err(Error::OutOfRange {
                name: "rate_in",
                min: MIN_RATE_HZ,
                max: f64::MAX,
                got: hz(rate_in),
            });
        }
        self.state(rate_in)?;
        Ok(())
    }

    /// The kernel state of the configuration.
    fn state(&self, rate_in: Rate) -> Result<StereoEncodeRdsState> {
        Ok(StereoEncodeRdsState::new(
            self.pilot_hz,
            self.main as f32,
            self.pilot as f32,
            self.rds as f32,
            rate_in,
        )?)
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "pilot_hz" => self.pilot_hz = value.as_f64()?,
            "main" => self.main = value.as_f64()?,
            "pilot" => self.pilot = value.as_f64()?,
            "rds" => self.rds = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    ///
    /// # Errors
    /// `UnknownParameter`.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "pilot_hz" => Ok(ParamValue::F64(self.pilot_hz)),
            "main" => Ok(ParamValue::F64(self.main)),
            "pilot" => Ok(ParamValue::F64(self.pilot)),
            "rds" => Ok(ParamValue::F64(self.rds)),
            _ => unknown(),
        }
    }
}

/// FM stereo encoder with RDS: stereo audio (`C32`, `re` left, `im` right) and the RDS baseband
/// (`f32`) in lockstep ([`Two`]) to the real multiplex signal (`f32`) through the kernel
/// `stereo_encode_rds`.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, FmStereoRdsModBlock, FmStereoRdsModConfig, ParamValue};
/// use sdrk::{Rate, C32};
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = FmStereoRdsModConfig::default();
/// let mut block = FmStereoRdsModBlock::<_>::new(&cfg, Rate::hz(192_000)?, &mut arena)?;
/// block.set_param("pilot", ParamValue::F64(0.0))?;
/// block.set_param("rds", ParamValue::F64(0.0))?;
/// // Mono audio without pilot and RDS is the sum signal alone: 0.87 * 0.5.
/// let mut out = [0.0f32; 8];
/// block.process((&[C32::new(0.5, 0.5); 8], &[1.0; 8]), &mut out)?;
/// assert!(out.iter().all(|&y| y == 0.87f32 * 0.5));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct FmStereoRdsModBlock<S: Storage> {
    config: FmStereoRdsModConfig,
    state: StereoEncodeRdsState,
    _storage: PhantomData<fn() -> S>,
}

impl<S: Storage> FmStereoRdsModBlock<S> {
    /// Creates the block; it needs no storage.
    ///
    /// # Errors
    /// The errors of [`FmStereoRdsModConfig::validate`].
    pub fn new(config: &FmStereoRdsModConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        Ok(Self {
            config: *config,
            state: config.state(rate_in)?,
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    ///
    /// # Errors
    /// The errors of [`FmStereoRdsModConfig::validate`].
    pub fn storage_bytes(config: &FmStereoRdsModConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    ///
    /// # Errors
    /// The errors of [`FmStereoRdsModConfig::validate`].
    pub fn rate_out(config: &FmStereoRdsModConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }

    /// The pilot phase before the next sample, in units of `2^-32` turn.
    pub fn phase(&self) -> u32 {
        self.state.phase()
    }
}

impl<S: Storage> Block for FmStereoRdsModBlock<S> {
    type In = Two<C32, f32>;
    type Out = f32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: (&[C32], &[f32]), output: &mut [f32]) -> Result<Progress> {
        let (audio, rds) = input;
        let n = lockstep(audio, rds)?.min(output.len());
        <Sdrk as StereoEncodeRds>::stereo_encode_rds(
            &Sdrk,
            &mut self.state,
            head(audio, n)?,
            head(rds, n)?,
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
            "pilot" => {
                let v = value.as_f64()?;
                self.state.set_pilot(v as f32)?;
                self.config.pilot = v;
                Ok(())
            }
            "rds" => {
                let v = value.as_f64()?;
                self.state.set_rds(v as f32)?;
                self.config.rds = v;
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
