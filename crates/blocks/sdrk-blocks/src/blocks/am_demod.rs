//! `am_demod`: envelope detector, DC removal and audio decimation.

use core::f64::consts::PI;

use sdrk::{Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::kernels::decim::{ResampleReal, ResampleRealState};
use sdrk_core::kernels::demod::{DcBlockReal, DcBlockRealState};
use sdrk_core::kernels::spectrum::Mag;
use sdrk_core::timing::Timing;
use sdrk_core::{ComplexSample, Q15, RealSample};

use super::{HZ_LIMIT, check_count, divide_rate, head, head_mut, hz, tail, tail_mut, unknown};
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
pub const AM_CHUNK: usize = 256;

/// Largest tap count of the audio filter.
const AUDIO_MAX_TAPS: usize = 1023;

/// Description of the `am_demod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "am_demod",
    summary: "Demodulates an AM signal to audio: envelope, DC removal and decimation.",
    details: "The envelope detector takes the magnitude of every complex sample, a one-pole \
        high-pass (pole exp(-2 pi dc_cutoff_hz / fs)) removes the carrier, which shows up as a \
        constant offset, and a low-pass with decimation brings the audio to its rate. The \
        audio filter has a transition width of a quarter of audio_cutoff_hz and 60 dB \
        attenuation; with audio_decimation 1 there is no audio filter. The audio level is the \
        envelope: a carrier of amplitude A modulated with index m reads m A at the output, \
        so put an agc in front (the example flows do) to make the level independent of the \
        signal strength. AM needs a slow AGC, otherwise it follows the modulation. Pitfalls: \
        the envelope detector cannot tell the carrier from the signal, so a channel without \
        a carrier produces noise; with CQ15 the magnitude saturates at full scale, so keep \
        the input below 0.7 per component; the output rate (input rate divided by \
        audio_decimation) must be above twice audio_cutoff_hz plus the transition; with \
        CQ15 a dc_cutoff_hz far below fs / 20000 rounds to a pole of 1 and is rejected; an \
        input sample of exactly zero (a closed squelch) gives an envelope of exactly zero, which \
        the block passes as zero audio and which clears the DC blocker, so a gated channel is \
        silent at once (with audio_decimation above 1 the audio filter still rings out for its \
        length).",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex baseband samples of the AM channel.",
    }],
    outputs: &[PortDesc {
        name: "audio",
        types: &[PortType::F32, PortType::Q15],
        description: "Demodulated audio (f32 for C32 input, Q15 for CQ15 input).",
    }],
    params: &[
        ParamDesc {
            name: "dc_cutoff_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 0.0,
                max: HZ_LIMIT,
                default: Some(30.0),
            },
            typical: "30 for broadcast, 0 to switch off",
            mutability: Mutability::Build,
            description: "-3 dB corner of the DC-blocking high-pass in Hz; 0 switches it off; \
                must be below half the input rate.",
        },
        ParamDesc {
            name: "audio_cutoff_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1.0,
                max: HZ_LIMIT,
                default: Some(5_000.0),
            },
            typical: "5e3 for broadcast, 2.5e3 for airband voice",
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
            typical: "1 at 48 kHz, 6 with audio_cutoff_hz 2500 to go to 8 kHz",
            mutability: Mutability::Build,
            description: "Decimation factor of the audio, 1 to 64.",
        },
    ],
    required: &[],
};

/// Configuration of [`AmDemodBlock`].
///
/// ```
/// use sdrk_blocks::AmDemodConfig;
/// let cfg = AmDemodConfig { audio_decimation: 6, audio_cutoff_hz: 2_500.0, ..AmDemodConfig::default() };
/// assert!(cfg.validate(sdrk::Rate::hz(48_000)?).is_ok());
/// assert!(AmDemodConfig { dc_cutoff_hz: 30_000.0, ..cfg }.validate(sdrk::Rate::hz(48_000)?).is_err());
/// # Ok::<(), sdrk::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmDemodConfig {
    /// Corner of the DC-blocking high-pass in Hz; 0 is off.
    pub dc_cutoff_hz: f64,
    /// Cutoff of the audio low-pass in Hz.
    pub audio_cutoff_hz: f64,
    /// Decimation factor of the audio, 1 to 64.
    pub audio_decimation: usize,
}

impl Default for AmDemodConfig {
    fn default() -> Self {
        Self {
            dc_cutoff_hz: 30.0,
            audio_cutoff_hz: 5_000.0,
            audio_decimation: 1,
        }
    }
}

/// The pole of the DC blocker for a corner frequency.
fn dc_pole(cutoff_hz: f64, fs: f64) -> f64 {
    libm::exp(-2.0 * PI * cutoff_hz / fs)
}

impl AmDemodConfig {
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
        let fs = hz(rate_in);
        for (name, v, min, max) in [
            ("dc_cutoff_hz", self.dc_cutoff_hz, 0.0, HZ_LIMIT),
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
        if self.dc_cutoff_hz > 0.0 {
            if self.dc_cutoff_hz >= fs / 2.0 {
                return Err(Error::OutOfRange {
                    name: "dc_cutoff_hz",
                    min: 0.0,
                    max: fs / 2.0,
                    got: self.dc_cutoff_hz,
                });
            }
            // The pole must stay inside (0, 1) in both sample types.
            let r = dc_pole(self.dc_cutoff_hz, fs);
            DcBlockRealState::<f32>::new(<f32 as RealSample>::from_f64_round(r))?;
            DcBlockRealState::<Q15>::new(<Q15 as RealSample>::from_f64_round(r))?;
        }
        if self.audio_decimation == 1 {
            return Ok(0);
        }
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
            "dc_cutoff_hz" => self.dc_cutoff_hz = value.as_f64()?,
            "audio_cutoff_hz" => self.audio_cutoff_hz = value.as_f64()?,
            "audio_decimation" => self.audio_decimation = value.as_usize()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "dc_cutoff_hz" => Ok(ParamValue::F64(self.dc_cutoff_hz)),
            "audio_cutoff_hz" => Ok(ParamValue::F64(self.audio_cutoff_hz)),
            "audio_decimation" => Ok(ParamValue::I64(self.audio_decimation as i64)),
            _ => unknown(),
        }
    }
}

/// AM demodulator: `C32` to `f32` or `CQ15` to `Q15` (`T` is the complex input type).
///
/// ```
/// use sdrk_blocks::{AlignedBytes, AmDemodBlock, AmDemodConfig, ArenaStorage, Block};
/// use sdrk::{Rate, C32};
/// let cfg = AmDemodConfig { dc_cutoff_hz: 0.0, ..AmDemodConfig::default() };
/// let rate = Rate::hz(48_000)?;
/// let bytes = AmDemodBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = AlignedBytes([0u8; 2048]);
/// let mut arena = ArenaStorage::new(&mut mem.0[..bytes]);
/// let mut block = AmDemodBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// // Without the DC blocker the audio is the magnitude of the input.
/// let input = [C32::new(3.0, 4.0), C32::new(0.0, -2.0)];
/// let mut audio = [0.0f32; 2];
/// block.process(&input, &mut audio)?;
/// assert_eq!(audio, [5.0, 2.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct AmDemodBlock<T: BlockSample, S: Storage> {
    config: AmDemodConfig,
    rate_in: Rate,
    dc: Option<DcBlockRealState<T::Real>>,
    audio: Option<AudioState<T, S>>,
    mag: S::Buf<T::Real>,
    scratch: S::Buf<T::Real>,
}

impl<T: BlockSample, S: Storage> core::fmt::Debug for AmDemodBlock<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AmDemodBlock")
            .field("config", &self.config)
            .field("rate_in", &self.rate_in)
            .finish_non_exhaustive()
    }
}

impl<T: BlockSample, S: Storage> AmDemodBlock<T, S> {
    /// Designs the audio filter and creates the block; filter and work buffers come from
    /// `storage`.
    pub fn new(config: &AmDemodConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
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
        let mag = storage.alloc::<T::Real>(AM_CHUNK)?;
        let scratch = storage.alloc::<T::Real>(AM_CHUNK)?;
        let dc = if config.dc_cutoff_hz > 0.0 {
            Some(DcBlockRealState::new(T::Real::from_f64_round(dc_pole(
                config.dc_cutoff_hz,
                fs,
            )))?)
        } else {
            None
        };
        Ok(Self {
            config: *config,
            rate_in,
            dc,
            audio,
            mag,
            scratch,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    pub fn storage_bytes(config: &AmDemodConfig, rate_in: Rate) -> Result<usize> {
        let n = config.audio_taps(rate_in)?;
        let mut plan = Plan::default();
        if config.audio_decimation > 1 {
            plan.add::<T::Real>(n)?;
            plan.add::<T::Real>(n)?;
        }
        plan.add::<T::Real>(AM_CHUNK)?;
        plan.add::<T::Real>(AM_CHUNK)?;
        Ok(plan.bytes())
    }

    /// Output rate: the input rate divided by `audio_decimation`.
    pub fn rate_out(config: &AmDemodConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        divide_rate(rate_in, config.audio_decimation)
    }
}

/// The DC blocker with the silence rule of the block: an envelope sample of exactly zero (a
/// closed squelch, digital silence) is written as zero and clears the blocker, so the audio
/// of a gated channel is exactly zero instead of a decaying tail. The runs of non-zero
/// samples between the zeros go through the `dc_block_real` kernel.
fn dc_block_gated<R: RealSample>(
    state: &mut DcBlockRealState<R>,
    env: &[R],
    out: &mut [R],
) -> Result<()>
where
    Sdrk: DcBlockReal<R>,
{
    let mut pos = 0;
    while pos < env.len() {
        let rest = tail(env, pos)?;
        let run = rest
            .iter()
            .position(|&v| v == R::ZERO)
            .unwrap_or(rest.len());
        if run > 0 {
            <Sdrk as DcBlockReal<R>>::dc_block_real(
                &Sdrk,
                state,
                head(rest, run)?,
                head_mut(tail_mut(out, pos)?, run)?,
            )?;
            pos += run;
            continue;
        }
        let zeros = rest.iter().take_while(|&&v| v == R::ZERO).count();
        head_mut(tail_mut(out, pos)?, zeros)?.fill(R::ZERO);
        state.reset();
        pos += zeros;
    }
    Ok(())
}

impl<T: BlockSample, S: Storage> AmDemodBlock<T, S>
where
    Sdrk: Mag<T> + DcBlockReal<T::Real> + ResampleReal<T::Real>,
{
    /// One piece without a decimator: the last stage writes straight to the output.
    fn piece_direct(&mut self, piece: &[T], out: &mut [T::Real]) -> Result<usize> {
        let m = piece.len();
        let out = head_mut(out, m)?;
        match &mut self.dc {
            Some(d) => {
                let env = head_mut(self.mag.as_mut(), m)?;
                <Sdrk as Mag<T>>::mag(&Sdrk, piece, env)?;
                dc_block_gated(d, head(self.mag.as_ref(), m)?, out)?;
            }
            None => <Sdrk as Mag<T>>::mag(&Sdrk, piece, out)?,
        }
        Ok(m)
    }

    /// One piece through the audio filter with decimation; returns the samples produced.
    fn piece_decimated(&mut self, piece: &[T], out: &mut [T::Real]) -> Result<usize> {
        let m = piece.len();
        <Sdrk as Mag<T>>::mag(&Sdrk, piece, head_mut(self.mag.as_mut(), m)?)?;
        let src: &[T::Real] = match &mut self.dc {
            Some(d) => {
                dc_block_gated(
                    d,
                    head(self.mag.as_ref(), m)?,
                    head_mut(self.scratch.as_mut(), m)?,
                )?;
                head(self.scratch.as_ref(), m)?
            }
            None => head(self.mag.as_ref(), m)?,
        };
        let Some(audio) = &mut self.audio else {
            return Ok(0);
        };
        let p = <Sdrk as ResampleReal<T::Real>>::resample_real(&Sdrk, audio, src, out)?;
        if p.consumed != m {
            return Err(Error::Core(sdrk_core::Error::Internal(
                "audio resampler did not take the whole piece",
            )));
        }
        Ok(p.produced)
    }
}

impl<T: BlockSample, S: Storage> Block for AmDemodBlock<T, S>
where
    Sdrk: Mag<T> + DcBlockReal<T::Real> + ResampleReal<T::Real>,
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
                .min(AM_CHUNK)
                .min(room.saturating_mul(dec));
            if m == 0 {
                break;
            }
            let piece = head(tail(input, consumed)?, m)?;
            let out = tail_mut(output, produced)?;
            produced += if dec == 1 {
                self.piece_direct(piece, out)?
            } else {
                self.piece_decimated(piece, out)?
            };
            consumed += m;
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        let ratio = Ratio::new(1, self.config.audio_decimation as u64).unwrap_or(Ratio::ONE);
        let delay = match &self.audio {
            Some(a) => a.group_delay(),
            None => sdrk::Delay::ZERO,
        };
        BlockTiming {
            ratio,
            kind: RatioKind::Constant,
            delay,
        }
    }

    fn reset(&mut self) {
        if let Some(d) = &mut self.dc {
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
