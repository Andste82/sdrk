//! `fm_demod`: FM discriminator, de-emphasis and audio decimation.

use sdrk::{Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::ComplexSample;
use sdrk_core::kernels::decim::{ResampleReal, ResampleRealState};
use sdrk_core::kernels::demod::{DeemphasisState, QuadDemod, QuadDemodState};
use sdrk_core::kernels::fused::FmDemod;
use sdrk_core::timing::Timing;

use super::{HZ_LIMIT, check_count, divide_rate, head, head_mut, hz, tail, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::DesignSpec;
use crate::error::{Error, Result};
use crate::sample::{BlockSample, TunedElem};
use crate::storage::{Plan, Storage};

/// The audio decimator of a block.
type AudioState<T, S> = ResampleRealState<
    <T as ComplexSample>::Real,
    <S as Storage>::Buf<<T as ComplexSample>::Real>,
    <S as Storage>::Buf<<T as ComplexSample>::Real>,
>;

/// Samples per piece the block processes at a time (the size of its two work buffers).
pub const FM_DEMOD_CHUNK: usize = 256;

/// Largest tap count of the audio filter.
const AUDIO_MAX_TAPS: usize = 1023;

/// Description of the `fm_demod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "fm_demod",
    summary: "Demodulates an FM signal to audio with de-emphasis and decimation.",
    details: "A quadrature discriminator (phase difference of consecutive samples), an optional \
        one-pole de-emphasis and a low-pass with decimation to the audio rate. With C32 the \
        discriminator gain is fs / (2 pi deviation), so a carrier at the full deviation reads \
        1.0. With CQ15 the gain is the largest Q15 value (raw 32767) and the output unit is \
        phase per sample divided by pi (the full deviation of 75 kHz at 240 kHz reads about \
        0.625). The audio filter has a transition width of a quarter of audio_cutoff_hz and 60 \
        dB attenuation; with audio_decimation 1 there is no audio filter. The input rate \
        should be 2 to 4 times the signal bandwidth (240 kHz for broadcast FM). Pitfall: the \
        output rate (input rate divided by audio_decimation) must be above twice \
        audio_cutoff_hz plus the transition.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex baseband samples of the FM signal.",
    }],
    outputs: &[PortDesc {
        name: "audio",
        types: &[PortType::F32, PortType::Q15],
        description: "Demodulated audio (f32 for C32 input, Q15 for CQ15 input).",
    }],
    params: &[
        ParamDesc {
            name: "deviation_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1.0,
                max: HZ_LIMIT,
                default: Some(75_000.0),
            },
            typical: "75e3 for broadcast FM, 5e3 for narrow-band voice",
            mutability: Mutability::Build,
            description: "Peak deviation in Hz that reads full scale (used by C32 only).",
        },
        ParamDesc {
            name: "deemphasis_us",
            unit: Unit::Microseconds,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 1e6,
                default: Some(50.0),
            },
            typical: "50 in Europe, 75 in North America, 0 to switch off",
            mutability: Mutability::Build,
            description: "Time constant of the de-emphasis in microseconds; 0 switches it off.",
        },
        ParamDesc {
            name: "audio_cutoff_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1.0,
                max: HZ_LIMIT,
                default: Some(15_000.0),
            },
            typical: "15e3 for broadcast FM",
            mutability: Mutability::Build,
            description: "Cutoff of the audio low-pass in Hz (ignored when audio_decimation is 1).",
        },
        ParamDesc {
            name: "audio_decimation",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: 64,
                default: Some(1),
            },
            typical: "5 to go from 240 kHz to 48 kHz",
            mutability: Mutability::Build,
            description: "Decimation factor of the audio, 1 to 64.",
        },
    ],
    required: &[],
};

/// Configuration of [`FmDemodBlock`].
///
/// ```
/// use sdrk_blocks::FmDemodConfig;
/// let cfg = FmDemodConfig { audio_decimation: 5, ..FmDemodConfig::default() };
/// assert!(cfg.validate(sdrk::Rate::hz(240_000)?).is_ok());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FmDemodConfig {
    /// Peak deviation in Hz that reads full scale (C32 only).
    pub deviation_hz: f64,
    /// De-emphasis time constant in microseconds; 0 is off.
    pub deemphasis_us: f64,
    /// Cutoff of the audio low-pass in Hz.
    pub audio_cutoff_hz: f64,
    /// Decimation factor of the audio, 1 to 64.
    pub audio_decimation: usize,
}

impl Default for FmDemodConfig {
    fn default() -> Self {
        Self {
            deviation_hz: 75_000.0,
            deemphasis_us: 50.0,
            audio_cutoff_hz: 15_000.0,
            audio_decimation: 1,
        }
    }
}

impl FmDemodConfig {
    fn audio_spec(&self) -> DesignSpec {
        DesignSpec {
            cutoff_hz: self.audio_cutoff_hz,
            transition_hz: self.audio_cutoff_hz / 4.0,
            attenuation_db: 60.0,
        }
    }

    /// Number of taps of the audio filter (0 if `audio_decimation` is 1); also checks the whole
    /// configuration.
    pub fn audio_taps(&self, rate_in: Rate) -> Result<usize> {
        check_count("audio_decimation", self.audio_decimation, 1, 64)?;
        for (name, v, min, max) in [
            ("deviation_hz", self.deviation_hz, 1.0, HZ_LIMIT),
            ("deemphasis_us", self.deemphasis_us, 0.0, 1e6),
            ("audio_cutoff_hz", self.audio_cutoff_hz, 1.0, HZ_LIMIT),
        ] {
            if !(v >= min && v <= max) {
                return Err(Error::OutOfRange {
                    name,
                    min,
                    max,
                    got: v,
                });
            }
        }
        if self.audio_decimation == 1 {
            return Ok(0);
        }
        let fs = hz(rate_in);
        let spec = self.audio_spec();
        // The stop band must start below the output Nyquist frequency, otherwise the
        // wanted band aliases.
        let out_half = fs / self.audio_decimation as f64 / 2.0;
        if spec.cutoff_hz + spec.transition_hz / 2.0 > out_half * (1.0 + 1e-9) {
            return Err(Error::OutOfRange {
                name: "audio_cutoff_hz",
                min: 1.0,
                max: out_half / 1.125,
                got: self.audio_cutoff_hz,
            });
        }
        spec.num_taps(fs, AUDIO_MAX_TAPS)
    }

    /// Checks the configuration against the input sample rate.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        self.audio_taps(rate_in).map(|_| ())
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "deviation_hz" => self.deviation_hz = value.as_f64()?,
            "deemphasis_us" => self.deemphasis_us = value.as_f64()?,
            "audio_cutoff_hz" => self.audio_cutoff_hz = value.as_f64()?,
            "audio_decimation" => self.audio_decimation = value.as_usize()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "deviation_hz" => Ok(ParamValue::F64(self.deviation_hz)),
            "deemphasis_us" => Ok(ParamValue::F64(self.deemphasis_us)),
            "audio_cutoff_hz" => Ok(ParamValue::F64(self.audio_cutoff_hz)),
            "audio_decimation" => Ok(ParamValue::I64(self.audio_decimation as i64)),
            _ => unknown(),
        }
    }
}

/// FM demodulator: `C32` to `f32` or `CQ15` to `Q15` (`T` is the complex input type).
///
/// ```
/// use sdrk_blocks::{AlignedBytes, ArenaStorage, Block, FmDemodBlock, FmDemodConfig};
/// use sdrk::{Rate, C32};
/// let cfg = FmDemodConfig { deemphasis_us: 0.0, ..FmDemodConfig::default() };
/// let rate = Rate::hz(240_000)?;
/// let bytes = FmDemodBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = AlignedBytes([0u8; 4096]);
/// let mut arena = ArenaStorage::new(&mut mem.0[..bytes]);
/// let mut block = FmDemodBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// // A carrier 75 kHz above the centre reads +1.0 once the discriminator has two samples.
/// let w = 2.0 * core::f64::consts::PI * 75_000.0 / 240_000.0;
/// let input: Vec<C32> = (0..8).map(|n| C32::new((w * n as f64).cos() as f32, (w * n as f64).sin() as f32)).collect();
/// let mut audio = [0.0f32; 8];
/// block.process(&input, &mut audio)?;
/// assert!((audio[7] - 1.0).abs() < 1e-4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct FmDemodBlock<T: BlockSample, S: Storage> {
    config: FmDemodConfig,
    rate_in: Rate,
    quad: QuadDemodState<T>,
    deemph: Option<DeemphasisState<T::Real>>,
    gain: T::Real,
    audio: Option<AudioState<T, S>>,
    demod: S::Buf<T::Real>,
    scratch: S::Buf<T::Real>,
}

impl<T: BlockSample, S: Storage> core::fmt::Debug for FmDemodBlock<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FmDemodBlock")
            .field("config", &self.config)
            .field("rate_in", &self.rate_in)
            .finish_non_exhaustive()
    }
}

impl<T: BlockSample, S: Storage> FmDemodBlock<T, S> {
    /// Designs the audio filter and creates the block; filter and work buffers come from
    /// `storage`.
    pub fn new(config: &FmDemodConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        let n = config.audio_taps(rate_in)?;
        let fs = hz(rate_in);
        let audio = if config.audio_decimation > 1 {
            let lp = config.audio_spec().lowpass(fs, AUDIO_MAX_TAPS)?;
            let mut taps = storage.alloc::<T::Real>(n)?;
            for (i, t) in taps.as_mut().iter_mut().enumerate() {
                *t = <T as TunedElem>::tap(lp.tap(i))?;
            }
            let hist = storage.alloc::<T::Real>(n)?;
            Some(ResampleRealState::new(
                taps,
                hist,
                1,
                config.audio_decimation,
            )?)
        } else {
            None
        };
        let demod = storage.alloc::<T::Real>(FM_DEMOD_CHUNK)?;
        let scratch = storage.alloc::<T::Real>(FM_DEMOD_CHUNK)?;
        let deemph = if config.deemphasis_us > 0.0 {
            Some(DeemphasisState::new(config.deemphasis_us * 1e-6, rate_in)?)
        } else {
            None
        };
        Ok(Self {
            config: *config,
            rate_in,
            quad: QuadDemodState::new(),
            deemph,
            gain: T::demod_gain(config.deviation_hz, fs),
            audio,
            demod,
            scratch,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    pub fn storage_bytes(config: &FmDemodConfig, rate_in: Rate) -> Result<usize> {
        let n = config.audio_taps(rate_in)?;
        let mut plan = Plan::default();
        if config.audio_decimation > 1 {
            plan.add::<T::Real>(n)?;
            plan.add::<T::Real>(n)?;
        }
        plan.add::<T::Real>(FM_DEMOD_CHUNK)?;
        plan.add::<T::Real>(FM_DEMOD_CHUNK)?;
        Ok(plan.bytes())
    }

    /// Output rate: the input rate divided by `audio_decimation`.
    pub fn rate_out(config: &FmDemodConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        divide_rate(rate_in, config.audio_decimation)
    }
}

impl<T: BlockSample, S: Storage> Block for FmDemodBlock<T, S>
where
    Sdrk: FmDemod<T> + QuadDemod<T> + ResampleReal<T::Real>,
{
    type In = T;
    type Out = T::Real;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T::Real]) -> Result<Progress> {
        let dec = self.config.audio_decimation;
        let (mut consumed, mut produced) = (0usize, 0usize);
        loop {
            let room = output.len() - produced;
            let m = (input.len() - consumed)
                .min(FM_DEMOD_CHUNK)
                .min(room.saturating_mul(dec));
            if m == 0 {
                break;
            }
            let piece = head(tail(input, consumed)?, m)?;
            // Without an audio filter the demodulated samples go straight to the output.
            let direct;
            let target: &mut [T::Real] = if self.audio.is_some() {
                direct = false;
                head_mut(self.demod.as_mut(), m)?
            } else {
                direct = true;
                head_mut(super::tail_mut(output, produced)?, m)?
            };
            match &mut self.deemph {
                Some(d) => <Sdrk as FmDemod<T>>::fm_demod(
                    &Sdrk,
                    &mut self.quad,
                    d,
                    piece,
                    target,
                    self.scratch.as_mut(),
                    self.gain,
                )?,
                None => <Sdrk as QuadDemod<T>>::quad_demod(
                    &Sdrk,
                    &mut self.quad,
                    piece,
                    target,
                    self.gain,
                )?,
            }
            if direct {
                produced += m;
            } else if let Some(audio) = &mut self.audio {
                let p = <Sdrk as ResampleReal<T::Real>>::resample_real(
                    &Sdrk,
                    audio,
                    head(self.demod.as_ref(), m)?,
                    super::tail_mut(output, produced)?,
                )?;
                if p.consumed != m {
                    return Err(Error::Core(sdrk_core::Error::Internal(
                        "audio resampler did not take the whole piece",
                    )));
                }
                produced += p.produced;
            }
            consumed += m;
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        let ratio = Ratio::new(1, self.config.audio_decimation as u64).unwrap_or(Ratio::ONE);
        let quad = self.quad.group_delay();
        let delay = match &self.audio {
            Some(a) => quad.add(a.group_delay()).unwrap_or(quad),
            None => quad,
        };
        BlockTiming {
            ratio,
            kind: RatioKind::Constant,
            delay,
        }
    }

    fn reset(&mut self) {
        self.quad.reset();
        if let Some(d) = &mut self.deemph {
            d.reset();
        }
        if let Some(a) = &mut self.audio {
            a.reset();
        }
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
        self.config.audio_decimation
    }
}
