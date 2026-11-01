//! End-to-end FM stereo receive (E2E-STEREO of M24), built from blocks.
//!
//! The signal is a stereo broadcast at 2.4 MS/s: the carrier at +600 kHz, deviation 75 kHz, the
//! multiplex signal of [`stereo_mpx_tones`] (`L` = 1 kHz, `R` = 3 kHz, amplitude 1 each, pilot
//! 0.1), scaled to 0.9, plus complex noise (sigma 0.003), encoded to `CU8` with the independent
//! testkit encoder. The receiver is the chain of the example flow `fm_stereo_receiver`:
//! `ConvertBlock` (`cu8`), `DdcBlock` (+600 kHz, decimation 5, cutoff 130 kHz, transition
//! 30 kHz) and `FmStereoDemodBlock` (defaults: 480 kHz to 48 kHz).
//!
//! [`measure_stereo`] streams the signal in blocks of [`CHUNK`] samples and keeps only the
//! normal equations of a joint least-squares fit of both tones and the pilot per channel
//! ([`MultitoneAccumulator`]), so the arena of the microcontroller suite is enough. It reports the
//! channel separation, the SINAD per channel, the level, the pilot residue, when the block
//! reported the pilot (`pilot_locked()`, polled after every block) and whether left and right were
//! bit-identical while the block was mono.

// See `e2e.rs`: the lint counts every `Result<_, Failure>` temporary separately.
#![allow(
    clippy::large_stack_frames,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use core::f64::consts::PI;

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_blocks::{
    ARENA_ALIGN, ArenaStorage, BiquadElem, Block, BlockSample, ConvertBlock, ConvertConfig,
    DdcBlock, DdcConfig, FmStereoDemodBlock, FmStereoDemodConfig,
};
use sdrk_core::kernels::biquad::{Biquad, BiquadReal};
use sdrk_core::kernels::convert::Convert;
use sdrk_core::kernels::decim::FirDecimate;
use sdrk_core::kernels::demod::QuadDemod;
use sdrk_core::kernels::fused::Ddc;
use sdrk_core::kernels::stereo::{StereoMatrix, StereoPll};
use sdrk_core::{ComplexSample, Rate, RealSample};
use sdrk_device::formats;

use crate::arena::Arena;
pub use crate::e2e::CHUNK;
use crate::e2e::{
    CARRIER_OFFSET_HZ, DEEMPHASIS_TAU_S, DEVIATION_HZ, E2eSample, IF_RATE_HZ, RX_SIGNAL_AMPLITUDE,
};
use crate::encode::encode;
use crate::fm::{PILOT_HZ, fm_multitone_baseband, stereo_mpx_tones};
use crate::meters::MultitoneAccumulator;
use crate::rng::Pcg32;
use crate::suite::Failure;

const CASE: &str = "e2e_fm_stereo";

/// Sample rate after the digital down-converter in Hz.
pub const STEREO_BASEBAND_HZ: u64 = 480_000;
/// Audio sample rate in Hz.
pub const STEREO_AUDIO_HZ: u64 = 48_000;
/// Frequency of the tone in the left channel in Hz.
pub const LEFT_HZ: f64 = 1_000.0;
/// Frequency of the tone in the right channel in Hz.
pub const RIGHT_HZ: f64 = 3_000.0;
/// Pilot amplitude relative to the full deviation.
pub const PILOT_AMPLITUDE: f64 = 0.1;
/// Standard deviation of the complex noise (per component).
pub const NOISE_SIGMA: f64 = 0.003;
/// Audio before this time is not measured: lock and filter start-up, in seconds.
pub const MEASURE_FROM_S: f64 = 0.25;
/// Quick variant: 0.5 s at 2.4 MS/s.
pub const QUICK_SAMPLES: usize = 1_200_000;
/// Full variant: 2.0 s at 2.4 MS/s.
pub const FULL_SAMPLES: usize = 4_800_000;
/// Input samples per second.
pub const RATE: f64 = IF_RATE_HZ as f64;

/// Channel separation thresholds in dB (`f32`, `Q15`).
pub const MIN_SEPARATION_DB: (f64, f64) = (40.0, 35.0);
/// SINAD thresholds per channel in dB (`f32`, `Q15`).
pub const MIN_SINAD_DB: (f64, f64) = (55.0, 45.0);
/// Pilot residue ceilings in dBFS (`f32`, `Q15`).
pub const MAX_PILOT_DBFS: (f64, f64) = (-70.0, -60.0);
/// Tolerance of the level in dB.
pub const LEVEL_TOLERANCE_DB: f64 = 0.5;
/// The pilot must be reported within this time of the start, in seconds.
pub const MAX_LOCK_S: f64 = 0.1;
/// Time left and right must be bit-identical after the loop reported mono, in seconds. The
/// guard covers the filter delay and the de-emphasis memory.
const MONO_GUARD_S: f64 = 0.005;

fn fail(msg: &'static str) -> Failure {
    Failure::new(CASE, msg)
}

/// The transmitted signal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoSignal {
    /// Amplitude and frequency of the left tone.
    pub left: (f64, f64),
    /// Amplitude and frequency of the right tone.
    pub right: (f64, f64),
    /// Pilot amplitude (0 for a mono transmission).
    pub pilot: f64,
    /// If set, the pilot is present only for the input samples `start .. end`. The window edges
    /// should be whole pilot periods (times that are multiples of 1 ms hold exactly 19 periods
    /// per millisecond), so the multiplex signal stays continuous at the edges.
    pub pilot_window: Option<(u64, u64)>,
    /// Whether the carrier is present (false: noise only).
    pub carrier: bool,
    /// Standard deviation of the complex noise per component.
    pub noise_sigma: f64,
    /// Seed of the noise generator.
    pub seed: u64,
}

impl StereoSignal {
    /// The signal of E2E-STEREO: `L` 1 kHz, `R` 3 kHz, pilot 0.1, noise 0.003.
    pub const STEREO: Self = Self {
        left: (1.0, LEFT_HZ),
        right: (1.0, RIGHT_HZ),
        pilot: PILOT_AMPLITUDE,
        pilot_window: None,
        carrier: true,
        noise_sigma: NOISE_SIGMA,
        seed: 2400,
    };

    /// A mono transmission: `L = R` = 1 kHz, no pilot.
    pub const MONO: Self = Self {
        left: (1.0, LEFT_HZ),
        right: (1.0, LEFT_HZ),
        pilot: 0.0,
        pilot_window: None,
        carrier: true,
        noise_sigma: NOISE_SIGMA,
        seed: 2400,
    };

    /// Complex noise only (sigma 0.3 per component), no carrier.
    pub const NOISE: Self = Self {
        left: (0.0, LEFT_HZ),
        right: (0.0, RIGHT_HZ),
        pilot: 0.0,
        pilot_window: None,
        carrier: false,
        noise_sigma: 0.3,
        seed: 1,
    };

    /// The stereo signal with the pilot present from `start_s` to `end_s` seconds only (both
    /// times must be multiples of 1 ms).
    pub fn switching(start_s: f64, end_s: f64) -> Self {
        Self {
            pilot_window: Some(((start_s * RATE + 0.5) as u64, (end_s * RATE + 0.5) as u64)),
            ..Self::STEREO
        }
    }
}

/// Source of the received bytes: the signal as `CU8`, generated chunk by chunk. The samples
/// depend on the index only (and on the noise generator, which runs along), so `fill` must be
/// called for consecutive blocks.
#[derive(Debug)]
pub struct StereoSource {
    signal: StereoSignal,
    rng: Pcg32,
}

impl StereoSource {
    /// A source for `signal`; the noise generator is seeded with `seed`.
    pub fn new(signal: StereoSignal, seed: u64) -> Self {
        Self {
            signal,
            rng: Pcg32::new(seed, 1),
        }
    }

    /// Writes the samples `start ..` of the signal into `scratch` (before encoding).
    pub fn samples(&mut self, scratch: &mut [Complex64], start: u64) -> Result<(), Failure> {
        let s = self.signal;
        if s.carrier {
            let gated = s.pilot_window.is_some();
            let tones = stereo_mpx_tones(s.left, s.right, if gated { 0.0 } else { s.pilot });
            fm_multitone_baseband(
                scratch,
                RATE,
                CARRIER_OFFSET_HZ,
                DEVIATION_HZ,
                &tones,
                start,
            )
            .map_err(|_| fail("a tone has the frequency 0"))?;
            if let Some((n0, n1)) = s.pilot_window {
                // The pilot term of the phase, integrated over the window only:
                // dev a / f (cos(w t0) - cos(w t_c)) with t_c clamped to the window.
                let k = DEVIATION_HZ * s.pilot / PILOT_HZ;
                let turn = |n: u64| 2.0 * PI * frac(PILOT_HZ * n as f64 / RATE);
                let base = libm::cos(turn(n0));
                for (i, o) in scratch.iter_mut().enumerate() {
                    let n = (start + i as u64).clamp(n0, n1);
                    let extra = k * (base - libm::cos(turn(n)));
                    *o *= Complex64::new(libm::cos(extra), libm::sin(extra));
                }
            }
            for o in scratch.iter_mut() {
                *o *= RX_SIGNAL_AMPLITUDE;
            }
        } else {
            scratch.fill(Complex64::new(0.0, 0.0));
        }
        if s.noise_sigma > 0.0 {
            for o in scratch.iter_mut() {
                *o += Complex64::new(
                    s.noise_sigma * self.rng.normal(),
                    s.noise_sigma * self.rng.normal(),
                );
            }
        }
        Ok(())
    }

    /// Writes the next block of raw `CU8` bytes (`start` is the index of the first sample).
    pub fn fill(
        &mut self,
        scratch: &mut [Complex64],
        start: u64,
        bytes: &mut [u8],
    ) -> Result<(), Failure> {
        self.samples(scratch, start)?;
        encode(&formats::CU8, scratch, bytes)?;
        Ok(())
    }
}

fn frac(x: f64) -> f64 {
    x - libm::floor(x)
}

/// Magnitude of the one-pole de-emphasis `y = a x + (1 - a) y[-1]` at `freq` for the rate
/// `rate` and the time constant [`DEEMPHASIS_TAU_S`].
pub fn deemphasis_gain(freq: f64, rate: f64) -> f64 {
    let alpha = 1.0 - libm::exp(-1.0 / (rate * DEEMPHASIS_TAU_S));
    let w = 2.0 * PI * freq / rate;
    let re = 1.0 - (1.0 - alpha) * libm::cos(w);
    let im = (1.0 - alpha) * libm::sin(w);
    alpha / libm::sqrt(re * re + im * im)
}

/// What [`measure_stereo`] found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoMeasurement {
    /// The 1 kHz component of the left channel over the one of the right channel in dB.
    pub separation_left_db: f64,
    /// The 3 kHz component of the right channel over the one of the left channel in dB.
    pub separation_right_db: f64,
    /// SINAD of the left channel (1 kHz, both tones removed) in dB.
    pub sinad_left_db: f64,
    /// SINAD of the right channel (3 kHz, both tones removed) in dB.
    pub sinad_right_db: f64,
    /// Level of the 1 kHz tone in the left channel against `0.9 |H_de|` in dB.
    pub level_left_db: f64,
    /// Level of the 3 kHz tone in the right channel against `0.9 |H_de|` in dB.
    pub level_right_db: f64,
    /// The 19 kHz component of the left channel in dBFS.
    pub pilot_left_dbfs: f64,
    /// The 19 kHz component of the right channel in dBFS.
    pub pilot_right_dbfs: f64,
    /// Peak amplitude of the 1 kHz tone in the left channel.
    pub amplitude_left: f64,
    /// Peak amplitude of the 3 kHz tone in the right channel.
    pub amplitude_right: f64,
    /// Time of the first report of the pilot in seconds, if any.
    pub first_lock_s: Option<f64>,
    /// Times of the changes of `pilot_locked()` in seconds (the first four).
    pub changes_s: [f64; 4],
    /// Number of changes of `pilot_locked()`.
    pub changes: usize,
    /// `pilot_locked()` after the last block.
    pub locked_at_end: bool,
    /// Left and right were bit-identical in every sample produced while the block had been mono
    /// for at least 5 ms.
    pub mono_identical: bool,
    /// Number of mono samples that were checked for that.
    pub mono_samples: usize,
    /// Number of audio samples (stereo pairs) the chain produced.
    pub audio_samples: usize,
}

/// Peak amplitude to dBFS (full scale 1.0); a floor of -400 dB for zero.
fn dbfs(a: f64) -> f64 {
    20.0 * libm::log10(libm::fmax(a, 1e-20))
}

/// Per-sample scale of the audio of the sample type: `C32` reads the full deviation as 1.0, `CQ15`
/// as `2 deviation / fs` (the unit is phase / pi).
fn audio_scale<T: BiquadElem>() -> f64 {
    if T::Q15 {
        2.0 * DEVIATION_HZ / STEREO_BASEBAND_HZ as f64
    } else {
        1.0
    }
}

/// Runs E2E-STEREO over `samples` input samples (a multiple of [`CHUNK`]) of `signal` and
/// measures the audio from [`MEASURE_FROM_S`] on. Needs about 40 KiB of the arena (no sample is
/// stored).
///
/// # Errors
/// A [`Failure`] if the arena is exhausted, a block fails or `samples` is not a multiple of
/// [`CHUNK`].
pub fn measure_stereo<T>(
    arena: &mut Arena<'_>,
    samples: usize,
    signal: &StereoSignal,
) -> Result<StereoMeasurement, Failure>
where
    T: E2eSample + BlockSample + BiquadElem,
    <T as ComplexSample>::Real: BiquadElem,
    Sdrk: Convert<T>
        + Ddc<T>
        + QuadDemod<T>
        + BiquadReal<<T as ComplexSample>::Real>
        + StereoPll<<T as ComplexSample>::Real>
        + FirDecimate<T>
        + StereoMatrix<T>
        + Biquad<T>,
{
    if samples == 0 || !samples.is_multiple_of(CHUNK) {
        return Err(fail("sample count must be a multiple of the block size"));
    }
    let if_rate = Rate::hz(IF_RATE_HZ)?;
    let bb_rate = Rate::hz(STEREO_BASEBAND_HZ)?;
    let ddc_cfg = DdcConfig {
        frequency_hz: CARRIER_OFFSET_HZ,
        decimation: 5,
        cutoff_hz: Some(130_000.0),
        transition_hz: Some(30_000.0),
        max_chunk: CHUNK,
        ..DdcConfig::default()
    };
    let stereo_cfg = FmStereoDemodConfig::default();
    let mut c = arena.carver();
    let ddc_bytes =
        DdcBlock::<T, ArenaStorage<'_>>::storage_bytes(&ddc_cfg, if_rate)? + ARENA_ALIGN;
    let stereo_bytes =
        FmStereoDemodBlock::<T, ArenaStorage<'_>>::storage_bytes(&stereo_cfg, bb_rate)?
            + ARENA_ALIGN;
    let mut ddc_store = ArenaStorage::new(c.take::<u8>(ddc_bytes)?);
    let mut stereo_store = ArenaStorage::new(c.take::<u8>(stereo_bytes)?);
    let mut convert = ConvertBlock::<T, _>::new(
        &ConvertConfig {
            format: formats::CU8,
        },
        if_rate,
        &mut ArenaStorage::new(&mut []),
    )?;
    let mut ddc = DdcBlock::<T, _>::new(&ddc_cfg, if_rate, &mut ddc_store)?;
    let mut stereo = FmStereoDemodBlock::<T, _>::new(&stereo_cfg, bb_rate, &mut stereo_store)?;

    let bytes = c.take::<u8>(formats::CU8.bytes_for_samples(CHUNK)?)?;
    let iq = c.take::<T>(CHUNK)?;
    let low = c.take::<T>(CHUNK / 5 + 4)?;
    let out = c.take::<T>(CHUNK / 50 + 4)?;
    let scratch = c.take::<Complex64>(CHUNK)?;

    let rate = STEREO_AUDIO_HZ as f64;
    let freqs = [LEFT_HZ, RIGHT_HZ, PILOT_HZ];
    let mut acc_l = MultitoneAccumulator::new(rate, &freqs);
    let mut acc_r = MultitoneAccumulator::new(rate, &freqs);
    let skip = (MEASURE_FROM_S * rate) as usize;
    let guard = (MONO_GUARD_S * IF_RATE_HZ as f64) as usize;
    let scale = audio_scale::<T>();

    let mut source = StereoSource::new(*signal, signal.seed);
    let mut produced_total = 0usize;
    let mut locked = false;
    let mut first_lock_s = None;
    let mut changes = 0usize;
    let mut changes_s = [0.0f64; 4];
    // Input sample index at which the loop last reported mono (None while stereo).
    let mut mono_since: Option<usize> = Some(0);
    let mut mono_identical = true;
    let mut mono_samples = 0usize;
    for chunk in 0..samples / CHUNK {
        source.fill(scratch, (chunk * CHUNK) as u64, bytes)?;
        let p = convert.process(bytes, iq)?;
        if p.consumed != bytes.len() || p.produced != CHUNK {
            return Err(fail("convert did not take the whole block"));
        }
        let q = ddc.process(iq, low)?;
        if q.consumed != CHUNK {
            return Err(fail("ddc did not take the whole block"));
        }
        let mut rest = low.get(..q.produced).unwrap_or_default();
        while !rest.is_empty() {
            let r = stereo.process(rest, out)?;
            if r.consumed == 0 && r.produced == 0 {
                return Err(fail("fm_stereo made no progress"));
            }
            // The state of the loop is polled after the block took the piece, which is the
            // precision of the lock time (one block of 0.2 ms).
            let now = (chunk + 1) * CHUNK;
            let state = stereo.pilot_locked();
            if state != locked {
                locked = state;
                if let Some(slot) = changes_s.get_mut(changes) {
                    *slot = now as f64 / RATE;
                }
                changes += 1;
                if locked && first_lock_s.is_none() {
                    first_lock_s = Some(now as f64 / RATE);
                }
                mono_since = if locked { None } else { Some(now) };
            }
            let checked = mono_since.is_some_and(|m| (chunk * CHUNK) >= m + guard);
            for v in out.iter().take(r.produced) {
                let idx = produced_total;
                produced_total += 1;
                let (l, rr) = (v.re(), v.im());
                if checked {
                    mono_samples += 1;
                    if l.to_f64().to_bits() != rr.to_f64().to_bits() {
                        mono_identical = false;
                    }
                }
                if idx >= skip {
                    acc_l.push(l.to_f64());
                    acc_r.push(rr.to_f64());
                }
            }
            rest = rest.get(r.consumed..).unwrap_or_default();
        }
    }
    let a_l = acc_l.amplitudes();
    let a_r = acc_r.amplitudes();
    let ratio_db =
        |num: f64, den: f64| 20.0 * libm::log10(libm::fmax(num, 1e-20) / libm::fmax(den, 1e-20));
    let expect = |f: f64| 0.9 * deemphasis_gain(f, rate) * scale;
    Ok(StereoMeasurement {
        separation_left_db: ratio_db(a_l[0], a_r[0]),
        separation_right_db: ratio_db(a_r[1], a_l[1]),
        sinad_left_db: acc_l.sinad_db(0),
        sinad_right_db: acc_r.sinad_db(1),
        level_left_db: ratio_db(a_l[0], expect(LEFT_HZ)),
        level_right_db: ratio_db(a_r[1], expect(RIGHT_HZ)),
        pilot_left_dbfs: dbfs(a_l[2]),
        pilot_right_dbfs: dbfs(a_r[2]),
        amplitude_left: a_l[0],
        amplitude_right: a_r[1],
        first_lock_s,
        changes_s,
        changes,
        locked_at_end: locked,
        mono_identical,
        mono_samples,
        audio_samples: produced_total,
    })
}

fn ensure(cond: bool, case: &'static str, msg: &'static str, value: f64) -> Result<(), Failure> {
    if cond {
        Ok(())
    } else {
        Err(Failure::new(case, msg).detail(format_args!("{value}")))
    }
}

/// Checks a measurement of [`StereoSignal::STEREO`] against the thresholds of the milestone:
/// separation, SINAD, level, pilot residue, and a lock within [`MAX_LOCK_S`] that stays.
///
/// # Errors
/// A [`Failure`] naming the first threshold that is missed.
pub fn check_stereo(case: &'static str, m: &StereoMeasurement, q15: bool) -> Result<(), Failure> {
    let pick = |t: (f64, f64)| if q15 { t.1 } else { t.0 };
    let (sep, sinad, pilot) = (
        pick(MIN_SEPARATION_DB),
        pick(MIN_SINAD_DB),
        pick(MAX_PILOT_DBFS),
    );
    ensure(
        m.separation_left_db >= sep,
        case,
        "separation 1 kHz in L over R too low (dB)",
        m.separation_left_db,
    )?;
    ensure(
        m.separation_right_db >= sep,
        case,
        "separation 3 kHz in R over L too low (dB)",
        m.separation_right_db,
    )?;
    ensure(
        m.sinad_left_db >= sinad,
        case,
        "SINAD of L too low (dB)",
        m.sinad_left_db,
    )?;
    ensure(
        m.sinad_right_db >= sinad,
        case,
        "SINAD of R too low (dB)",
        m.sinad_right_db,
    )?;
    ensure(
        libm::fabs(m.level_left_db) <= LEVEL_TOLERANCE_DB,
        case,
        "level of L off (dB)",
        m.level_left_db,
    )?;
    ensure(
        libm::fabs(m.level_right_db) <= LEVEL_TOLERANCE_DB,
        case,
        "level of R off (dB)",
        m.level_right_db,
    )?;
    ensure(
        m.pilot_left_dbfs <= pilot,
        case,
        "pilot residue in L too high (dBFS)",
        m.pilot_left_dbfs,
    )?;
    ensure(
        m.pilot_right_dbfs <= pilot,
        case,
        "pilot residue in R too high (dBFS)",
        m.pilot_right_dbfs,
    )?;
    ensure(
        m.first_lock_s.is_some_and(|t| t <= MAX_LOCK_S),
        case,
        "no lock within 0.1 s (s, 0 = never)",
        m.first_lock_s.unwrap_or(0.0),
    )?;
    ensure(
        m.changes == 1 && m.locked_at_end,
        case,
        "the lock did not stay (number of changes)",
        m.changes as f64,
    )
}

/// Checks a measurement of a mono transmission or of noise: the block never reports the pilot
/// and left and right are bit-identical.
///
/// # Errors
/// A [`Failure`] naming what differs.
pub fn check_mono(case: &'static str, m: &StereoMeasurement) -> Result<(), Failure> {
    ensure(
        m.changes == 0,
        case,
        "stereo was reported (number of changes)",
        m.changes as f64,
    )?;
    ensure(
        m.mono_identical && m.mono_samples > 0,
        case,
        "left and right differ (checked samples)",
        m.mono_samples as f64,
    )
}

/// Checks a measurement of [`StereoSignal::switching`] for a pilot from 0.3 s to 0.7 s: stereo
/// within 0.1 s after the pilot appears, mono within 0.2 s after it disappears, and left and
/// right bit-identical while mono.
///
/// # Errors
/// A [`Failure`] naming what is missed.
pub fn check_switching(case: &'static str, m: &StereoMeasurement) -> Result<(), Failure> {
    ensure(
        m.changes == 2,
        case,
        "wrong number of changes of the stereo state",
        m.changes as f64,
    )?;
    let [on, off, ..] = m.changes_s;
    ensure(
        (0.3..=0.4).contains(&on),
        case,
        "stereo not within 0.1 s after the pilot appeared (s)",
        on,
    )?;
    ensure(
        (0.7..=0.9).contains(&off),
        case,
        "mono not within 0.2 s after the pilot vanished (s)",
        off,
    )?;
    ensure(
        m.mono_identical && m.mono_samples > 0,
        case,
        "left and right differ while mono (checked samples)",
        m.mono_samples as f64,
    )
}

/// Input samples per block of the block-level separation case (a multiple of the decimation 10).
const BLOCK_CHUNK: usize = 480;
/// Length of the block-level separation case in input samples: 0.2 s at 480 kHz.
pub const BLOCK_SAMPLES: usize = 96_000;

/// The block case of M24: 0.2 s of the multiplex signal of E2E-STEREO as FM baseband at 480 kHz
/// (without the RF part) through `FmStereoDemodBlock` alone. Returns the separation in dB of the
/// 1 kHz tone in L over R and of the 3 kHz tone in R over L, measured from 0.1 s on (4800 audio
/// samples). Needs about 12 KiB of the arena.
///
/// # Errors
/// A [`Failure`] if the arena is exhausted or the block fails.
pub fn measure_block_separation<T>(arena: &mut Arena<'_>) -> Result<(f64, f64), Failure>
where
    T: E2eSample + BlockSample + BiquadElem,
    <T as ComplexSample>::Real: BiquadElem,
    Sdrk: QuadDemod<T>
        + BiquadReal<<T as ComplexSample>::Real>
        + StereoPll<<T as ComplexSample>::Real>
        + FirDecimate<T>
        + StereoMatrix<T>
        + Biquad<T>,
{
    let bb_rate = Rate::hz(STEREO_BASEBAND_HZ)?;
    let cfg = FmStereoDemodConfig::default();
    let mut c = arena.carver();
    let bytes =
        FmStereoDemodBlock::<T, ArenaStorage<'_>>::storage_bytes(&cfg, bb_rate)? + ARENA_ALIGN;
    let mut store = ArenaStorage::new(c.take::<u8>(bytes)?);
    let mut block = FmStereoDemodBlock::<T, _>::new(&cfg, bb_rate, &mut store)?;
    let scratch = c.take::<Complex64>(BLOCK_CHUNK)?;
    let input = c.take::<T>(BLOCK_CHUNK)?;
    let out = c.take::<T>(BLOCK_CHUNK / 10 + 2)?;
    let tones = stereo_mpx_tones((1.0, LEFT_HZ), (1.0, RIGHT_HZ), PILOT_AMPLITUDE);
    let rate = STEREO_AUDIO_HZ as f64;
    let mut acc_l = MultitoneAccumulator::new(rate, &[LEFT_HZ, RIGHT_HZ]);
    let mut acc_r = acc_l;
    let skip = BLOCK_SAMPLES / 10 / 2;
    let mut produced_total = 0usize;
    for chunk in 0..BLOCK_SAMPLES / BLOCK_CHUNK {
        fm_multitone_baseband(
            scratch,
            STEREO_BASEBAND_HZ as f64,
            0.0,
            DEVIATION_HZ,
            &tones,
            (chunk * BLOCK_CHUNK) as u64,
        )
        .map_err(|_| fail("a tone has the frequency 0"))?;
        for (i, s) in input.iter_mut().zip(scratch.iter()) {
            *i = T::from_c64_round(*s * RX_SIGNAL_AMPLITUDE);
        }
        let p = block.process(input, out)?;
        if p.consumed != BLOCK_CHUNK {
            return Err(fail("fm_stereo did not take the whole block"));
        }
        for v in out.iter().take(p.produced) {
            if produced_total >= skip {
                acc_l.push(v.re().to_f64());
                acc_r.push(v.im().to_f64());
            }
            produced_total += 1;
        }
    }
    let (a_l, a_r) = (acc_l.amplitudes(), acc_r.amplitudes());
    let ratio_db =
        |num: f64, den: f64| 20.0 * libm::log10(libm::fmax(num, 1e-20) / libm::fmax(den, 1e-20));
    Ok((ratio_db(a_l[0], a_r[0]), ratio_db(a_r[1], a_l[1])))
}

/// The mono checks of M24 at the block level (the RF part left out, so that they fit the quick
/// profile of the microcontrollers): 0.1 s of a mono transmission (`L = R` = 1 kHz, no pilot) and
/// 0.1 s of complex noise (sigma 0.3) at 480 kHz through `FmStereoDemodBlock` alone. Returns the
/// number of samples in which left and right differ and whether the block reported the pilot at
/// any time. Needs about 12 KiB of the arena.
///
/// # Errors
/// A [`Failure`] if the arena is exhausted or the block fails.
pub fn measure_block_mono<T>(arena: &mut Arena<'_>) -> Result<(usize, bool), Failure>
where
    T: E2eSample + BlockSample + BiquadElem,
    <T as ComplexSample>::Real: BiquadElem,
    Sdrk: QuadDemod<T>
        + BiquadReal<<T as ComplexSample>::Real>
        + StereoPll<<T as ComplexSample>::Real>
        + FirDecimate<T>
        + StereoMatrix<T>
        + Biquad<T>,
{
    let bb_rate = Rate::hz(STEREO_BASEBAND_HZ)?;
    let cfg = FmStereoDemodConfig::default();
    let mut c = arena.carver();
    let bytes =
        FmStereoDemodBlock::<T, ArenaStorage<'_>>::storage_bytes(&cfg, bb_rate)? + ARENA_ALIGN;
    let mut store = ArenaStorage::new(c.take::<u8>(bytes)?);
    let mut block = FmStereoDemodBlock::<T, _>::new(&cfg, bb_rate, &mut store)?;
    let scratch = c.take::<Complex64>(BLOCK_CHUNK)?;
    let input = c.take::<T>(BLOCK_CHUNK)?;
    let out = c.take::<T>(BLOCK_CHUNK / 10 + 2)?;
    let tones = stereo_mpx_tones((1.0, LEFT_HZ), (1.0, LEFT_HZ), 0.0);
    let mut rng = Pcg32::new(7, 1);
    let mut differing = 0usize;
    let mut ever_locked = false;
    for noise in [false, true] {
        block.reset();
        for chunk in 0..BLOCK_SAMPLES / 2 / BLOCK_CHUNK {
            if noise {
                for s in scratch.iter_mut() {
                    *s = Complex64::new(0.3 * rng.normal(), 0.3 * rng.normal());
                }
            } else {
                fm_multitone_baseband(
                    scratch,
                    STEREO_BASEBAND_HZ as f64,
                    0.0,
                    DEVIATION_HZ,
                    &tones,
                    (chunk * BLOCK_CHUNK) as u64,
                )
                .map_err(|_| fail("a tone has the frequency 0"))?;
            }
            for (i, s) in input.iter_mut().zip(scratch.iter()) {
                *i = T::from_c64_round(*s * RX_SIGNAL_AMPLITUDE);
            }
            let p = block.process(input, out)?;
            if p.consumed != BLOCK_CHUNK {
                return Err(fail("fm_stereo did not take the whole block"));
            }
            ever_locked |= block.pilot_locked();
            for v in out.iter().take(p.produced) {
                if v.re().to_f64().to_bits() != v.im().to_f64().to_bits() {
                    differing += 1;
                }
            }
        }
    }
    Ok((differing, ever_locked))
}
