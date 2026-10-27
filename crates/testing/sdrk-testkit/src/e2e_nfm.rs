//! End-to-end narrowband FM receive with squelch (E2E-NFM of M19), built from blocks.
//!
//! The signal at 2.4 MS/s is a narrowband FM transmission (offset +300 kHz, deviation 5 kHz,
//! 1 kHz tone, amplitude 0.5) that is present only in the middle half of the run, plus complex
//! Gaussian noise (sigma 0.003 per component) over the whole run, encoded to `CU8` with the
//! independent testkit encoder. The receiver is the chain of the example flow `nfm_receiver`:
//! `ConvertBlock`, `DdcBlock` (300 kHz, decimation 50, cutoff 8 kHz, transition 4 kHz, up to
//! 4096 taps), `SquelchBlock` (-40 dB, 3 dB, 0.01 s) and `FmDemodBlock` (5 kHz, no
//! de-emphasis, audio cutoff 4 kHz).
//!
//! [`measure_nfm`] streams the signal through the chain in blocks of [`CHUNK`]
//! samples and keeps no audio buffer: the SINAD is accumulated while the audio is produced, the
//! gate is observed after every squelch sample and the exactly-zero checks run on the fly. It
//! needs about 50 KiB of the arena for `C32` and about 30 KiB for `CQ15`.
//!
//! [`measure_nfm_ctcss`] (M22) adds an 88.5 Hz sub-tone at 10 % of the deviation to the signal and
//! the `ctcss` node of the updated flow (a fourth-order Butterworth high-pass at 300 Hz,
//! `BiquadRealBlock`) behind the demodulator.

// See `e2e.rs`: the lint counts every `Result<_, Failure>` temporary separately.
#![allow(
    clippy::large_stack_frames,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::items_after_statements
)]

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_blocks::{
    ARENA_ALIGN, ArenaStorage, BiquadConfig, BiquadElem, BiquadRealBlock, BiquadResponse, Block,
    BlockReal, BlockSample, ConvertBlock, ConvertConfig, DdcBlock, DdcConfig, FmDemodBlock,
    FmDemodConfig, SquelchBlock, SquelchConfig,
};
use sdrk_core::kernels::biquad::BiquadReal;
use sdrk_core::kernels::convert::Convert;
use sdrk_core::kernels::decim::ResampleReal;
use sdrk_core::kernels::demod::QuadDemod;
use sdrk_core::kernels::fused::{Ddc, FmDemod};
use sdrk_core::kernels::squelch::Squelch;
use sdrk_core::{ComplexSample, Rate, RealSample};
use sdrk_device::formats;

use crate::arena::{Arena, Carver};
use crate::e2e::{CHUNK, E2eSample, IF_RATE_HZ};
use crate::encode::encode;
use crate::fm::{fm_baseband, fm_baseband_two_tones};
use crate::meters::ToneAccumulator;
use crate::rng::Pcg32;
use crate::suite::Failure;

const CASE: &str = "e2e_nfm_rx";

/// Offset of the carrier from the tuner centre in Hz.
pub const NFM_OFFSET_HZ: f64 = 300_000.0;
/// Peak frequency deviation in Hz (a 25 kHz channel).
pub const NFM_DEVIATION_HZ: f64 = 5_000.0;
/// Audio tone in Hz.
pub const NFM_AUDIO_HZ: f64 = 1_000.0;
/// Amplitude of the transmission relative to full scale.
pub const NFM_AMPLITUDE: f64 = 0.5;
/// Standard deviation of the noise per component.
pub const NFM_NOISE_SIGMA: f64 = 0.003;
/// Decimation of the channel filter (2.4 MS/s to 48 kHz).
pub const NFM_DECIMATION: usize = 50;
/// Audio sample rate in Hz.
pub const NFM_AUDIO_RATE_HZ: f64 = 48_000.0;
/// Quick variant: 0.4 s at 2.4 MS/s.
pub const NFM_QUICK_SAMPLES: usize = 960_000;
/// Full variant: 2.0 s at 2.4 MS/s.
pub const NFM_FULL_SAMPLES: usize = 4_800_000;
/// SINAD threshold for `C32` in dB.
pub const NFM_MIN_SINAD_F32_DB: f64 = 30.0;
/// SINAD threshold for `CQ15` in dB.
pub const NFM_MIN_SINAD_Q15_DB: f64 = 25.0;
/// The gate must open at most this long after the transmission starts, in seconds.
pub const NFM_MAX_OPEN_DELAY_S: f64 = 0.001;
/// The gate must close at most this long after the transmission ends, in seconds.
pub const NFM_MAX_CLOSE_DELAY_S: f64 = 0.1;
/// Frequency of the CTCSS sub-tone of [`measure_nfm_ctcss`] in Hz.
pub const NFM_SUBTONE_HZ: f64 = 88.5;
/// Deviation of the sub-tone relative to the deviation of the 1 kHz tone.
pub const NFM_SUBTONE_RATIO: f64 = 0.1;
/// Cutoff of the `ctcss` high-pass in Hz.
pub const NFM_CTCSS_CUTOFF_HZ: f64 = 300.0;
/// Minimum attenuation of the sub-tone against the 1 kHz tone with the `ctcss` node, `f32`, in dB.
pub const NFM_MIN_SUBTONE_DB_F32: f64 = 50.0;
/// Minimum attenuation of the sub-tone against the 1 kHz tone with the `ctcss` node, Q15, in dB.
pub const NFM_MIN_SUBTONE_DB_Q15: f64 = 40.0;
/// The SINAD with the `ctcss` node may be this much lower than without it, `f32`, in dB.
pub const NFM_MAX_SINAD_LOSS_DB_F32: f64 = 1.0;
/// The SINAD with the `ctcss` node may be this much lower than without it, Q15, in dB (the
/// separate Q15 threshold approved by the maintainer, decision D23; measured 1.07 dB).
pub const NFM_MAX_SINAD_LOSS_DB_Q15: f64 = 1.1;
/// Audio samples after the gate closed that are exempt from the zero check: the tail of the
/// audio filter of the demodulator.
const FILTER_TAIL: usize = 256;

/// What [`measure_nfm`] found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NfmMeasurement {
    /// SINAD at 1 kHz over the middle 80 % of the transmission in dB.
    pub sinad_db: f64,
    /// Seconds from the start of the transmission to the opening of the gate (infinite if the
    /// gate never opened).
    pub open_delay_s: f64,
    /// Seconds from the end of the transmission to the closing of the gate (infinite if the
    /// gate stayed open).
    pub close_delay_s: f64,
    /// True if the gate was open at any time before the transmission started.
    pub opened_early: bool,
    /// Audio samples that must be exactly zero and were checked: all before the transmission,
    /// all from 0.1 s after its end, and all from `FILTER_TAIL` samples after the gate closed.
    pub zero_checked: usize,
    /// How many of the checked samples were not exactly zero (+0.0).
    pub zero_violations: usize,
    /// Audio samples produced in total.
    pub audio_samples: usize,
    /// Level of the 88.5 Hz component below the 1 kHz tone in dB (Hann-windowed fits over the
    /// same window as the SINAD).
    pub subtone_below_db: f64,
}

/// A Hann-windowed single-tone fit over a window of known length.
struct WindowedTone {
    rate: f64,
    freq: f64,
    len: usize,
    n: usize,
    re: f64,
    im: f64,
    weight: f64,
}

impl WindowedTone {
    fn new(rate: f64, freq: f64, len: usize) -> Self {
        Self {
            rate,
            freq,
            len,
            n: 0,
            re: 0.0,
            im: 0.0,
            weight: 0.0,
        }
    }

    fn push(&mut self, x: f64) {
        let k = self.n as f64;
        self.n += 1;
        let w = 0.5 - 0.5 * libm::cos(core::f64::consts::TAU * k / self.len as f64);
        let phase = core::f64::consts::TAU * self.freq * k / self.rate;
        self.re += w * x * libm::cos(phase);
        self.im -= w * x * libm::sin(phase);
        self.weight += w;
    }

    /// Amplitude of the tone.
    fn amplitude(&self) -> f64 {
        if self.weight == 0.0 {
            return 0.0;
        }
        2.0 * libm::sqrt(self.re * self.re + self.im * self.im) / self.weight
    }
}

/// Runs E2E-NFM over `samples` input samples (a multiple of `4 * CHUNK`; the transmission fills
/// the middle half) with the sample type `T`.
///
/// # Errors
/// A [`Failure`] if the arena is exhausted, a block fails or `samples` is not a multiple of
/// `4 * CHUNK`.
pub fn measure_nfm<T>(arena: &mut Arena<'_>, samples: usize) -> Result<NfmMeasurement, Failure>
where
    T: E2eSample + BlockSample,
    Sdrk: Convert<T>
        + Ddc<T>
        + FmDemod<T>
        + QuadDemod<T>
        + Squelch<T>
        + ResampleReal<<T as ComplexSample>::Real>,
{
    run_nfm::<T>(&mut arena.carver(), samples, false, None)
}

/// Runs E2E-NFM with CTCSS (M22): the signal of [`measure_nfm`] with an added sub-tone of
/// [`NFM_SUBTONE_HZ`] at [`NFM_SUBTONE_RATIO`] of the deviation, through the chain with the
/// `ctcss` node (`BiquadRealBlock`, high-pass, order 4, [`NFM_CTCSS_CUTOFF_HZ`]) behind
/// `fm_demod`. The zero check covers the samples before the transmission only: the filter of
/// the `ctcss` node keeps its state for 0.2 s after the gate closed (f32) or may keep a small
/// offset (Q15), see `docs/kernels.md`.
///
/// # Errors
/// As [`measure_nfm`].
pub fn measure_nfm_ctcss<T>(
    arena: &mut Arena<'_>,
    samples: usize,
) -> Result<NfmMeasurement, Failure>
where
    T: E2eSample + BlockSample,
    <T as ComplexSample>::Real: BiquadElem + BlockReal,
    Sdrk: Convert<T>
        + Ddc<T>
        + FmDemod<T>
        + QuadDemod<T>
        + Squelch<T>
        + ResampleReal<<T as ComplexSample>::Real>
        + BiquadReal<<T as ComplexSample>::Real>,
{
    let mut c = arena.carver();
    let audio_rate = Rate::hz(48_000)?;
    let cfg = BiquadConfig {
        response: BiquadResponse::Highpass,
        cutoff_hz: NFM_CTCSS_CUTOFF_HZ,
        order: 4,
        ..BiquadConfig::default()
    };
    let bytes = BiquadRealBlock::<<T as ComplexSample>::Real, ArenaStorage<'_>>::storage_bytes(
        &cfg, audio_rate,
    )? + ARENA_ALIGN;
    let mut store = ArenaStorage::new(c.take::<u8>(bytes)?);
    let mut ctcss =
        BiquadRealBlock::<<T as ComplexSample>::Real, _>::new(&cfg, audio_rate, &mut store)?;
    let mut post = |input: &[<T as ComplexSample>::Real],
                    output: &mut [<T as ComplexSample>::Real]|
     -> Result<(), Failure> {
        ctcss.process(input, output)?;
        Ok(())
    };
    run_nfm::<T>(&mut c, samples, true, Some(&mut post))
}

/// The `ctcss` stage: filters a block of audio.
type Post<'p, R> = &'p mut dyn FnMut(&[R], &mut [R]) -> Result<(), Failure>;

fn run_nfm<T>(
    c: &mut Carver<'_>,
    samples: usize,
    subtone: bool,
    mut post: Option<Post<'_, <T as ComplexSample>::Real>>,
) -> Result<NfmMeasurement, Failure>
where
    T: E2eSample + BlockSample,
    Sdrk: Convert<T>
        + Ddc<T>
        + FmDemod<T>
        + QuadDemod<T>
        + Squelch<T>
        + ResampleReal<<T as ComplexSample>::Real>,
{
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    if samples == 0 || !samples.is_multiple_of(4 * CHUNK) {
        return Err(fail("sample count must be a multiple of four blocks"));
    }
    let if_rate = Rate::hz(IF_RATE_HZ)?;
    let audio_rate = Rate::hz(48_000)?;
    let ddc_cfg = DdcConfig {
        frequency_hz: NFM_OFFSET_HZ,
        decimation: NFM_DECIMATION,
        cutoff_hz: Some(8_000.0),
        transition_hz: Some(4_000.0),
        max_taps: 4096,
        max_chunk: CHUNK,
        ..DdcConfig::default()
    };
    let squelch_cfg = SquelchConfig::default();
    let demod_cfg = FmDemodConfig {
        deviation_hz: NFM_DEVIATION_HZ,
        deemphasis_us: 0.0,
        audio_cutoff_hz: 4_000.0,
        audio_decimation: 1,
    };
    let ddc_bytes =
        DdcBlock::<T, ArenaStorage<'_>>::storage_bytes(&ddc_cfg, if_rate)? + ARENA_ALIGN;
    let demod_bytes =
        FmDemodBlock::<T, ArenaStorage<'_>>::storage_bytes(&demod_cfg, audio_rate)? + ARENA_ALIGN;
    let mut ddc_store = ArenaStorage::new(c.take::<u8>(ddc_bytes)?);
    let mut demod_store = ArenaStorage::new(c.take::<u8>(demod_bytes)?);
    let mut convert = ConvertBlock::<T, _>::new(
        &ConvertConfig {
            format: formats::CU8,
        },
        if_rate,
        &mut ArenaStorage::new(&mut []),
    )?;
    let mut ddc = DdcBlock::<T, _>::new(&ddc_cfg, if_rate, &mut ddc_store)?;
    let mut gate =
        SquelchBlock::<T, _>::new(&squelch_cfg, audio_rate, &mut ArenaStorage::new(&mut []))?;
    let mut demod = FmDemodBlock::<T, _>::new(&demod_cfg, audio_rate, &mut demod_store)?;

    const PIECE: usize = 100;
    let scratch = c.take::<Complex64>(PIECE)?;
    let bytes = c.take::<u8>(formats::CU8.bytes_for_samples(CHUNK)?)?;
    let iq = c.take::<T>(CHUNK)?;
    let low = c.take::<T>(CHUNK / NFM_DECIMATION + 4)?;
    let gated = c.take::<T>(CHUNK / NFM_DECIMATION + 4)?;
    let out = c.take::<<T as ComplexSample>::Real>(CHUNK / NFM_DECIMATION + 4)?;
    let filtered = c.take::<<T as ComplexSample>::Real>(CHUNK / NFM_DECIMATION + 4)?;

    let tx_start = samples / 4;
    let tx_end = 3 * samples / 4;
    let audio_start = tx_start / NFM_DECIMATION;
    let audio_end = tx_end / NFM_DECIMATION;
    let window = (audio_end - audio_start) / 10;
    let (win_lo, win_hi) = (audio_start + window, audio_end - window);
    let after_end = audio_end + (0.1 * NFM_AUDIO_RATE_HZ) as usize;

    let mut rng = Pcg32::new(1900, 1);
    let mut sinad = ToneAccumulator::new(NFM_AUDIO_RATE_HZ, NFM_AUDIO_HZ);
    let mut main_tone = WindowedTone::new(NFM_AUDIO_RATE_HZ, NFM_AUDIO_HZ, win_hi - win_lo);
    let mut low_tone = WindowedTone::new(NFM_AUDIO_RATE_HZ, NFM_SUBTONE_HZ, win_hi - win_lo);
    let (mut gate_index, mut audio_index) = (0usize, 0usize);
    let mut opened_at: Option<usize> = None;
    let mut closed_at: Option<usize> = None;
    let mut opened_early = false;
    let (mut zero_checked, mut zero_violations) = (0usize, 0usize);

    for chunk in 0..samples / CHUNK {
        // Source: the transmission (middle half only) plus noise, encoded to CU8.
        for piece in 0..CHUNK / PIECE {
            let start = chunk * CHUNK + piece * PIECE;
            let block = &mut *scratch;
            if (tx_start..tx_end).contains(&start) {
                if subtone {
                    fm_baseband_two_tones(
                        block,
                        IF_RATE_HZ as f64,
                        NFM_OFFSET_HZ,
                        [
                            (NFM_DEVIATION_HZ, NFM_AUDIO_HZ),
                            (NFM_SUBTONE_RATIO * NFM_DEVIATION_HZ, NFM_SUBTONE_HZ),
                        ],
                        start as u64,
                    );
                } else {
                    fm_baseband(
                        block,
                        IF_RATE_HZ as f64,
                        NFM_OFFSET_HZ,
                        NFM_DEVIATION_HZ,
                        NFM_AUDIO_HZ,
                        1.0,
                        start as u64,
                    );
                }
                for s in block.iter_mut() {
                    *s *= NFM_AMPLITUDE;
                }
            } else {
                block.fill(Complex64::new(0.0, 0.0));
            }
            for s in block.iter_mut() {
                s.re += NFM_NOISE_SIGMA * rng.normal();
                s.im += NFM_NOISE_SIGMA * rng.normal();
            }
            let dst = bytes
                .get_mut(2 * piece * PIECE..2 * (piece + 1) * PIECE)
                .ok_or_else(|| fail("byte slice"))?;
            encode(&formats::CU8, block, dst)?;
        }
        let p = convert.process(bytes, iq)?;
        if p.consumed != bytes.len() || p.produced != CHUNK {
            return Err(fail("convert did not take the whole block"));
        }
        let mut rest: &[T] = iq;
        while !rest.is_empty() {
            let p = ddc.process(rest, low)?;
            if p.consumed == 0 && p.produced == 0 {
                return Err(fail("ddc made no progress"));
            }
            rest = rest.get(p.consumed..).unwrap_or_default();
            let produced = low.get(..p.produced).unwrap_or_default();
            // The squelch runs sample by sample so that the switching instants are exact.
            for (i, s) in produced.iter().enumerate() {
                let o = gated.get_mut(i..=i).ok_or_else(|| fail("gate slice"))?;
                gate.process(core::slice::from_ref(s), o)?;
                let g = gate_index;
                gate_index += 1;
                if gate.is_open() {
                    if g < audio_start {
                        opened_early = true;
                    }
                    if opened_at.is_none() && g >= audio_start {
                        opened_at = Some(g);
                    }
                } else if opened_at.is_some() && closed_at.is_none() && g >= audio_end {
                    closed_at = Some(g);
                }
            }
            let mut g_rest = gated.get(..p.produced).unwrap_or_default();
            while !g_rest.is_empty() {
                let q = demod.process(g_rest, out)?;
                if q.consumed == 0 && q.produced == 0 {
                    return Err(fail("fm_demod made no progress"));
                }
                let demodulated = out.get(..q.produced).unwrap_or_default();
                let audio: &[<T as ComplexSample>::Real] = match post.as_mut() {
                    Some(stage) => {
                        let dst = filtered.get_mut(..q.produced).unwrap_or_default();
                        stage(demodulated, dst)?;
                        dst
                    }
                    None => demodulated,
                };
                for r in audio {
                    let a = audio_index;
                    audio_index += 1;
                    let v = r.to_f64();
                    if (win_lo..win_hi).contains(&a) {
                        sinad.push(v);
                        main_tone.push(v);
                        low_tone.push(v);
                    }
                    let must_be_zero = a < audio_start
                        || (post.is_none()
                            && (a >= after_end || closed_at.is_some_and(|c| a >= c + FILTER_TAIL)));
                    if must_be_zero {
                        zero_checked += 1;
                        if v != 0.0 {
                            zero_violations += 1;
                        }
                    }
                }
                g_rest = g_rest.get(q.consumed..).unwrap_or_default();
            }
        }
    }
    let delay = |at: Option<usize>, from: usize| match at {
        Some(g) => (g - from) as f64 / NFM_AUDIO_RATE_HZ,
        None => f64::INFINITY,
    };
    Ok(NfmMeasurement {
        sinad_db: sinad.sinad_db(),
        open_delay_s: delay(opened_at, audio_start),
        close_delay_s: delay(closed_at, audio_end),
        opened_early,
        zero_checked,
        zero_violations,
        audio_samples: audio_index,
        subtone_below_db: 20.0
            * libm::log10(main_tone.amplitude() / low_tone.amplitude().max(f64::MIN_POSITIVE)),
    })
}
