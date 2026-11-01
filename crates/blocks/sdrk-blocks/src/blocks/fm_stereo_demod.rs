//! `fm_stereo_demod`: FM stereo decoder (pilot PLL, 38 kHz demodulation, matrix, de-emphasis).
//!
//! Processing order per piece of at most [`FM_STEREO_DEMOD_CHUNK`] input samples:
//! `quad_demod`, `biquad_real` band-pass at 19 kHz, `stereo_pll`, `fir_decimate` with the audio
//! low-pass, `stereo_matrix`, and the optional de-emphasis `biquad`.

use sdrk::{Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::ComplexSample;
use sdrk_core::kernels::biquad::{Biquad, BiquadReal, BiquadRealState, BiquadSample, BiquadState};
use sdrk_core::kernels::decim::{FirDecimState, FirDecimate};
use sdrk_core::kernels::demod::{QuadDemod, QuadDemodState};
use sdrk_core::kernels::stereo::{StereoMatrix, StereoPll, StereoPllState};
use sdrk_core::timing::Timing;

use super::biquad::BiquadElem;
use super::{HZ_LIMIT, check_count, divide_rate, head, head_mut, hz, tail, tail_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::{BiquadResponse, DesignSpec, biquad_coefs};
use crate::error::{Error, Result};
use crate::sample::{BlockSample, TunedElem};
use crate::storage::{Plan, Storage};

/// Samples per piece the block processes at a time.
pub const FM_STEREO_DEMOD_CHUNK: usize = 256;

/// Largest tap count of the audio filter.
const AUDIO_MAX_TAPS: usize = 1023;

/// Smallest input sample rate in Hz: the multiplex signal reaches 53 kHz and the products of
/// the 38 kHz demodulation reach 91 kHz.
pub const FM_STEREO_DEMOD_MIN_RATE_HZ: f64 = 120_000.0;

/// Centre frequency of the pilot band-pass and the pilot of the loop in Hz.
const PILOT_HZ: f64 = 19_000.0;

/// Quality factor of the pilot band-pass.
const PILOT_Q: f64 = 5.0;

/// Pilot amplitude in the multiplex signal (nominal).
const PILOT_AMPLITUDE: f64 = 0.1;

/// Names of the values of the `stereo` parameter.
const MODES: [&str; 2] = ["auto", "mono"];

/// Description of the `fm_stereo_demod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "fm_stereo_demod",
    summary: "Demodulates a broadcast FM signal to stereo audio (left in re, right in im).",
    details: "A quadrature discriminator gives the multiplex signal (MPX: sum L+R up to 15 kHz, \
        a 19 kHz pilot and the difference L-R around 38 kHz). A band-pass at 19 kHz isolates the \
        pilot, a phase-locked loop follows its phase and regenerates the 38 kHz carrier, the \
        difference is demodulated with it, both are low-pass filtered with decimation, and a \
        matrix forms left and right; an optional de-emphasis follows. The output is stereo \
        audio as complex samples: re is the left channel and im the right channel (C32 for C32 \
        input, CQ15 for CQ15 input), which is the interleaved two-channel layout of sound cards \
        and WAV files. Without a pilot the output is mono (left and right equal). The input \
        rate must be 120 kHz or more; a channel filter narrower than the FM signal (about \
        2 x (75 + 53) kHz) costs channel separation, so use at least 480 kHz with a ddc \
        cutoff of 130 kHz in front of it. The stereo parameter auto decodes stereo when the \
        pilot is present; mono forces mono output, for weak stations whose difference signal \
        is noisy. With C32 the discriminator gain is fs / (2 pi deviation); with CQ15 the \
        gain is the largest Q15 value. Pitfall: the output rate (input rate divided by \
        audio_decimation) must be above twice audio_cutoff_hz plus the transition.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex baseband samples of the FM signal, at least 120 kHz.",
    }],
    outputs: &[PortDesc {
        name: "audio",
        types: &[PortType::C32, PortType::CQ15],
        description: "Stereo audio: re is the left channel, im the right channel (C32 for C32 \
            input, CQ15 for CQ15 input).",
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
            typical: "75e3 for broadcast FM",
            mutability: Mutability::Build,
            description: "Peak deviation in Hz that reads full scale (C32 and the CQ15 pilot \
                level).",
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
                max: 16_000.0,
                default: Some(15_000.0),
            },
            typical: "15e3 for broadcast FM",
            mutability: Mutability::Build,
            description: "Cutoff of the audio low-pass in Hz, 1 to 16000.",
        },
        ParamDesc {
            name: "audio_decimation",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: 64,
                default: Some(10),
            },
            typical: "10 to go from 480 kHz to 48 kHz",
            mutability: Mutability::Build,
            description: "Decimation factor of the audio, 1 to 64.",
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
            description: "auto decodes stereo while the 19 kHz pilot is present; mono forces \
                mono output (left and right equal). Takes effect from the next call.",
        },
        ParamDesc {
            name: "pll_natural_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1.0,
                max: 190.0,
                default: Some(20.0),
            },
            typical: "20",
            mutability: Mutability::Build,
            description: "Natural frequency of the pilot loop in Hz, 1 to 190.",
        },
    ],
    required: &[],
};

/// Configuration of [`FmStereoDemodBlock`].
///
/// ```
/// use sdrk_blocks::FmStereoDemodConfig;
/// let cfg = FmStereoDemodConfig::default();
/// assert!(cfg.validate(sdrk::Rate::hz(480_000)?).is_ok());
/// // The multiplex signal does not fit into a lower rate.
/// assert!(cfg.validate(sdrk::Rate::hz(96_000)?).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FmStereoDemodConfig {
    /// Peak deviation in Hz that reads full scale.
    pub deviation_hz: f64,
    /// De-emphasis time constant in microseconds; 0 is off.
    pub deemphasis_us: f64,
    /// Cutoff of the audio low-pass in Hz.
    pub audio_cutoff_hz: f64,
    /// Decimation factor of the audio, 1 to 64.
    pub audio_decimation: usize,
    /// Forces mono output (`stereo` = `mono`).
    pub mono: bool,
    /// Natural frequency of the pilot loop in Hz.
    pub pll_natural_hz: f64,
}

impl Default for FmStereoDemodConfig {
    fn default() -> Self {
        Self {
            deviation_hz: 75_000.0,
            deemphasis_us: 50.0,
            audio_cutoff_hz: 15_000.0,
            audio_decimation: 10,
            mono: false,
            pll_natural_hz: 20.0,
        }
    }
}

impl FmStereoDemodConfig {
    fn audio_spec(&self) -> DesignSpec {
        DesignSpec {
            cutoff_hz: self.audio_cutoff_hz,
            transition_hz: self.audio_cutoff_hz / 4.0,
            attenuation_db: 60.0,
        }
    }

    /// Number of taps of the audio filter; also checks the whole configuration.
    pub fn audio_taps(&self, rate_in: Rate) -> Result<usize> {
        check_count("audio_decimation", self.audio_decimation, 1, 64)?;
        for (name, v, min, max) in [
            ("deviation_hz", self.deviation_hz, 1.0, HZ_LIMIT),
            ("deemphasis_us", self.deemphasis_us, 0.0, 1e6),
            ("audio_cutoff_hz", self.audio_cutoff_hz, 1.0, 16_000.0),
            ("pll_natural_hz", self.pll_natural_hz, 1.0, 190.0),
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
        let fs = hz(rate_in);
        if fs < FM_STEREO_DEMOD_MIN_RATE_HZ {
            return Err(Error::OutOfRange {
                name: "sample_rate",
                min: FM_STEREO_DEMOD_MIN_RATE_HZ,
                max: f64::INFINITY,
                got: fs,
            });
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
            "deviation_hz" => self.deviation_hz = value.as_f64()?,
            "deemphasis_us" => self.deemphasis_us = value.as_f64()?,
            "audio_cutoff_hz" => self.audio_cutoff_hz = value.as_f64()?,
            "audio_decimation" => self.audio_decimation = value.as_usize()?,
            "stereo" => self.mono = value.as_choice()? == 1,
            "pll_natural_hz" => self.pll_natural_hz = value.as_f64()?,
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
            "stereo" => Ok(ParamValue::Choice(u16::from(self.mono))),
            "pll_natural_hz" => Ok(ParamValue::F64(self.pll_natural_hz)),
            _ => unknown(),
        }
    }
}

/// Coefficient type of the real band-pass.
type RCoef<T> = <<T as ComplexSample>::Real as BiquadSample>::Coef;

/// The state of the pilot band-pass (real, one section).
type BandpassState<T, S> = BiquadRealState<
    <T as ComplexSample>::Real,
    <S as Storage>::Buf<[RCoef<T>; 5]>,
    <S as Storage>::Buf<<T as ComplexSample>::Real>,
>;

/// The state of the de-emphasis (complex, one section).
type DeemphState<T, S> =
    BiquadState<T, <S as Storage>::Buf<[<T as BiquadSample>::Coef; 5]>, <S as Storage>::Buf<T>>;

/// Most output samples of one piece (`ceil(FM_STEREO_DEMOD_CHUNK / audio_decimation)`).
fn dec_len(decimation: usize) -> usize {
    FM_STEREO_DEMOD_CHUNK / decimation + 1
}

/// FM stereo decoder: `C32` to `C32` or `CQ15` to `CQ15` (`T` is the complex type; `re` of the
/// output is the left channel, `im` the right channel).
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, FmStereoDemodBlock, FmStereoDemodConfig};
/// use sdrk::{Rate, C32};
/// let cfg = FmStereoDemodConfig::default();
/// let rate = Rate::hz(480_000)?;
/// let bytes = FmStereoDemodBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut block = FmStereoDemodBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// // A carrier at the centre is silence; without a pilot the block outputs mono.
/// let input = [C32::new(1.0, 0.0); 480];
/// let mut audio = [C32::new(9.0, 9.0); 48];
/// let p = block.process(&input, &mut audio)?;
/// assert_eq!((p.consumed, p.produced), (480, 48));
/// assert!(!block.pilot_locked());
/// assert_eq!(audio[47].re, audio[47].im);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct FmStereoDemodBlock<T: BlockSample + BiquadElem, S: Storage>
where
    T::Real: BiquadElem,
{
    config: FmStereoDemodConfig,
    quad: QuadDemodState<T>,
    gain: T::Real,
    pilot_bp: BandpassState<T, S>,
    pll: StereoPllState<T::Real>,
    fir: FirDecimState<T, S::Buf<T::Real>, S::Buf<T>>,
    deemph: Option<DeemphState<T, S>>,
    mpx: S::Buf<T::Real>,
    pilot: S::Buf<T::Real>,
    work: S::Buf<T>,
    dec: S::Buf<T>,
}

impl<T: BlockSample + BiquadElem, S: Storage> core::fmt::Debug for FmStereoDemodBlock<T, S>
where
    T::Real: BiquadElem,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FmStereoDemodBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// `[alpha, 0, 0, -(1 - alpha), 0]` of the one-pole de-emphasis as a biquad section.
fn deemphasis_section(tau_us: f64, fs_audio: f64) -> [f64; 5] {
    let alpha = 1.0 - libm::exp(-1.0 / (fs_audio * tau_us * 1e-6));
    [alpha, 0.0, 0.0, -(1.0 - alpha), 0.0]
}

impl<T: BlockSample + BiquadElem, S: Storage> FmStereoDemodBlock<T, S>
where
    T::Real: BiquadElem,
{
    /// Designs the filters and creates the block; filters and work buffers come from
    /// `storage`.
    #[allow(
        clippy::large_stack_frames,
        reason = "clippy counts the temporary of every `?` separately; the real frame is small"
    )]
    pub fn new(config: &FmStereoDemodConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        let n = config.audio_taps(rate_in)?;
        let fs = hz(rate_in);
        let lp = config.audio_spec().lowpass(fs, AUDIO_MAX_TAPS)?;
        let mut taps = storage.alloc::<T::Real>(n)?;
        for (i, t) in taps.as_mut().iter_mut().enumerate() {
            *t = <T as TunedElem>::tap(lp.tap(i))?;
        }
        let hist = storage.alloc::<T>(n - 1)?;
        let fir = FirDecimState::new(taps, hist, config.audio_decimation)?;

        // Pilot band-pass: one RBJ section at 19 kHz.
        let bp = biquad_coefs(BiquadResponse::Bandpass, PILOT_HZ, PILOT_Q, fs)?;
        let mut bp_coefs = storage.alloc::<[RCoef<T>; 5]>(1)?;
        <T::Real as BiquadElem>::convert(&[bp], bp_coefs.as_mut())?;
        let bp_hist = storage.alloc::<T::Real>(4)?;
        let pilot_bp = BiquadRealState::new(bp_coefs, bp_hist)?;

        let mut pll = Self::new_pll(config, rate_in)?;
        pll.set_enabled(!config.mono);

        let mpx = storage.alloc::<T::Real>(FM_STEREO_DEMOD_CHUNK)?;
        let pilot = storage.alloc::<T::Real>(FM_STEREO_DEMOD_CHUNK)?;
        let work = storage.alloc::<T>(FM_STEREO_DEMOD_CHUNK)?;
        let dec = storage.alloc::<T>(dec_len(config.audio_decimation))?;

        let deemph = if config.deemphasis_us > 0.0 {
            let fs_audio = fs / config.audio_decimation as f64;
            let sec = deemphasis_section(config.deemphasis_us, fs_audio);
            let mut coefs = storage.alloc::<[T::Coef; 5]>(1)?;
            <T as BiquadElem>::convert(&[sec], coefs.as_mut())?;
            let hist = storage.alloc::<T>(4)?;
            Some(BiquadState::new(coefs, hist)?)
        } else {
            None
        };
        Ok(Self {
            config: *config,
            quad: QuadDemodState::new(),
            gain: T::demod_gain(config.deviation_hz, fs),
            pilot_bp,
            pll,
            fir,
            deemph,
            mpx,
            pilot,
            work,
            dec,
        })
    }

    /// The pilot loop for `config`. With `CQ15` the nominal pilot level is
    /// `0.1 * 2 * deviation_hz / fs`, which must lie in `(0, 1]` and give non-zero loop
    /// coefficients; `rate_out` and `storage_bytes` run this check on a scratch state, so a
    /// deviation that `new` would reject is already a diagnostic when the flow is validated.
    fn new_pll(config: &FmStereoDemodConfig, rate_in: Rate) -> Result<StereoPllState<T::Real>> {
        let fs = hz(rate_in);
        let level = if <T as BiquadElem>::Q15 {
            // The discriminator output of a deviation df is 2 df / fs in units of pi.
            let level = PILOT_AMPLITUDE * 2.0 * config.deviation_hz / fs;
            if !(level > 0.0 && level <= 1.0) {
                return Err(Error::OutOfRange {
                    name: "deviation_hz",
                    min: 1.0,
                    max: fs / (2.0 * PILOT_AMPLITUDE),
                    got: config.deviation_hz,
                });
            }
            level
        } else {
            PILOT_AMPLITUDE
        };
        Ok(StereoPllState::new(
            PILOT_HZ,
            config.pll_natural_hz,
            level,
            rate_in,
        )?)
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    pub fn storage_bytes(config: &FmStereoDemodConfig, rate_in: Rate) -> Result<usize> {
        let n = config.audio_taps(rate_in)?;
        Self::new_pll(config, rate_in)?;
        let mut plan = Plan::default();
        plan.add::<T::Real>(n)?;
        plan.add::<T>(n - 1)?;
        plan.add::<[RCoef<T>; 5]>(1)?;
        plan.add::<T::Real>(4)?;
        plan.add::<T::Real>(FM_STEREO_DEMOD_CHUNK)?;
        plan.add::<T::Real>(FM_STEREO_DEMOD_CHUNK)?;
        plan.add::<T>(FM_STEREO_DEMOD_CHUNK)?;
        plan.add::<T>(dec_len(config.audio_decimation))?;
        if config.deemphasis_us > 0.0 {
            plan.add::<[T::Coef; 5]>(1)?;
            plan.add::<T>(4)?;
        }
        Ok(plan.bytes())
    }

    /// Output rate: the input rate divided by `audio_decimation`.
    pub fn rate_out(config: &FmStereoDemodConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Self::new_pll(config, rate_in)?;
        divide_rate(rate_in, config.audio_decimation)
    }

    /// Number of taps of the audio filter.
    pub fn num_taps(&self) -> usize {
        self.fir.taps().len()
    }

    /// Whether the pilot loop has locked and stereo is being decoded (`is_stereo()` of the
    /// loop state); the decision does not depend on the `stereo` parameter, which only silences the
    /// difference channel.
    pub fn pilot_locked(&self) -> bool {
        self.pll.is_stereo()
    }
}

impl<T: BlockSample + BiquadElem, S: Storage> Block for FmStereoDemodBlock<T, S>
where
    T::Real: BiquadElem,
    Sdrk: QuadDemod<T>
        + BiquadReal<T::Real>
        + StereoPll<T::Real>
        + FirDecimate<T>
        + StereoMatrix<T>
        + Biquad<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    #[allow(
        clippy::large_stack_frames,
        reason = "clippy counts the temporary of every `?` separately; the real frame is small"
    )]
    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        let dec_cap = dec_len(self.config.audio_decimation);
        let (mut consumed, mut produced) = (0usize, 0usize);
        loop {
            let room = (output.len() - produced).min(dec_cap);
            let m = (input.len() - consumed)
                .min(FM_STEREO_DEMOD_CHUNK)
                .min(self.fir.max_input_for_output(room));
            if m == 0 {
                break;
            }
            let piece = head(tail(input, consumed)?, m)?;
            let mpx = head_mut(self.mpx.as_mut(), m)?;
            <Sdrk as QuadDemod<T>>::quad_demod(&Sdrk, &mut self.quad, piece, mpx, self.gain)?;
            let pilot = head_mut(self.pilot.as_mut(), m)?;
            <Sdrk as BiquadReal<T::Real>>::biquad_real(
                &Sdrk,
                &mut self.pilot_bp,
                head(self.mpx.as_ref(), m)?,
                pilot,
            )?;
            <Sdrk as StereoPll<T::Real>>::stereo_pll(
                &Sdrk,
                &mut self.pll,
                head(self.mpx.as_ref(), m)?,
                head(self.pilot.as_ref(), m)?,
                head_mut(self.work.as_mut(), m)?,
            )?;
            let p = <Sdrk as FirDecimate<T>>::fir_decimate(
                &Sdrk,
                &mut self.fir,
                head(self.work.as_ref(), m)?,
                head_mut(self.dec.as_mut(), room)?,
            )?;
            if p.consumed != m {
                return Err(Error::Core(sdrk_core::Error::Internal(
                    "stereo filter did not take the whole piece",
                )));
            }
            let k = p.produced;
            let out = head_mut(tail_mut(output, produced)?, k)?;
            match &mut self.deemph {
                Some(d) => {
                    <Sdrk as StereoMatrix<T>>::stereo_matrix(
                        &Sdrk,
                        head(self.dec.as_ref(), k)?,
                        head_mut(self.work.as_mut(), k)?,
                    )?;
                    <Sdrk as Biquad<T>>::biquad(&Sdrk, d, head(self.work.as_ref(), k)?, out)?;
                }
                None => {
                    <Sdrk as StereoMatrix<T>>::stereo_matrix(
                        &Sdrk,
                        head(self.dec.as_ref(), k)?,
                        out,
                    )?;
                }
            }
            consumed += m;
            produced += k;
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        let ratio = Ratio::new(1, self.config.audio_decimation as u64).unwrap_or(Ratio::ONE);
        let quad = self.quad.group_delay();
        let delay = quad.add(self.fir.group_delay()).unwrap_or(quad);
        BlockTiming {
            ratio,
            kind: RatioKind::Constant,
            delay,
        }
    }

    fn reset(&mut self) {
        self.quad.reset();
        self.pilot_bp.reset();
        self.pll.reset();
        self.fir.reset();
        if let Some(d) = &mut self.deemph {
            d.reset();
        }
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match runtime_param(&DESCRIPTION, name, &value)? {
            "stereo" => {
                let mono = value.as_choice()? == 1;
                self.config.mono = mono;
                self.pll.set_enabled(!mono);
                Ok(())
            }
            _ => unknown(),
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        self.config.audio_decimation
    }
}
