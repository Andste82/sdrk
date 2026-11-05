//! Loop tests of the M35 transmitters, built from blocks: LOOP-AM (`am_mod` into `am_demod`)
//! and LOOP-SSB (`ssb_mod` into `ssb_demod`), in `f32` and Q15. The measured values are
//! printed (run with `--nocapture` to see them).
//!
//! Both loops run at 48 kHz with a 1 kHz audio tone. The first `SETTLE` samples of every
//! output are skipped: they hold the delay of the FIR filters (436 samples each for the SSB
//! defaults) and the settling of the DC blocker of `am_demod` (time constant about 5 ms at
//! 30 Hz).
//!
//! The file needs the feature `std` (for `VecStorage`).

#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::print_stderr,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::large_stack_frames
)]

use core::f64::consts::PI;

use num_complex::Complex64;
use sdrk::{Rate, Sdrk};
use sdrk_blocks::{
    AmDemodBlock, AmDemodConfig, AmModBlock, AmModConfig, Block, BlockSample, SsbDemodBlock,
    SsbDemodConfig, SsbModBlock, SsbModConfig, SsbSideband, StreamElem,
};
use sdrk_core::kernels::decim::{FirDecimate, ResampleReal};
use sdrk_core::kernels::demod::DcBlockReal;
use sdrk_core::kernels::filter::Fir;
use sdrk_core::kernels::math::Interleave;
use sdrk_core::kernels::mix::Rotate;
use sdrk_core::kernels::modulate::AmMod;
use sdrk_core::kernels::spectrum::Mag;
use sdrk_core::{C32, CQ15, RealSample};
use sdrk_testkit::VecStorage;
use sdrk_testkit::meters::ToneAccumulator;

/// Sample rate of both loops in Hz.
const FS: u64 = 48_000;
/// Audio tone in Hz.
const TONE_HZ: f64 = 1_000.0;
/// Amplitude of the audio tone.
const TONE_AMPLITUDE: f64 = 0.5;
/// Samples skipped at the start of every output (0.1 s).
const SETTLE: usize = 4_800;
/// Samples measured after `SETTLE` (1 s).
const MEASURE: usize = 48_000;
/// Length of the DFT for the bin checks: 10 Hz per bin, so 1 kHz is bin 100 exactly.
const DFT_LEN: usize = 4_800;
/// The bin of the audio tone.
const TONE_BIN: usize = 100;

/// LOOP-AM: SINAD thresholds of the spec in dB.
const AM_MIN_SINAD_F32_DB: f64 = 40.0;
const AM_MIN_SINAD_Q15_DB: f64 = 30.0;
/// LOOP-SSB: largest level error, smallest opposite-sideband rejection and smallest mirror
/// suppression of the spec in dB.
const SSB_MAX_LEVEL_ERROR_DB: f64 = 0.5;
const SSB_MIN_REJECTION_DB: f64 = 50.0;
const SSB_MIN_MIRROR_DB: f64 = 50.0;

/// Runs a block over `input` and resubmits what it did not consume.
fn pump<I: StreamElem, O: StreamElem + Default, B: Block<In = I, Out = O>>(
    block: &mut B,
    input: &[I],
) -> Vec<O> {
    let mut out = Vec::with_capacity(input.len());
    let mut room = vec![O::default(); 4096];
    let mut rest = input;
    loop {
        let p = block.process(rest, &mut room).unwrap();
        out.extend_from_slice(&room[..p.produced]);
        rest = &rest[p.consumed..];
        if rest.is_empty() && p.produced < room.len() {
            break;
        }
        assert!(p.consumed > 0 || p.produced > 0, "no progress");
    }
    out
}

/// The audio tone, `SETTLE + MEASURE` samples, rounded to the real sample type.
fn tone<R: RealSample>() -> Vec<R> {
    (0..SETTLE + MEASURE)
        .map(|n| {
            let cycles = TONE_HZ * n as f64 / FS as f64;
            R::from_f64_round(TONE_AMPLITUDE * (2.0 * PI * (cycles - cycles.floor())).cos())
        })
        .collect()
}

fn rate() -> Rate {
    Rate::hz(FS).unwrap()
}

/// Fits the 1 kHz tone to the measured part of `audio`: `(amplitude, SINAD in dB)`.
fn fit<R: RealSample>(audio: &[R]) -> (f64, f64) {
    let mut acc = ToneAccumulator::new(FS as f64, TONE_HZ);
    for v in &audio[SETTLE..SETTLE + MEASURE] {
        acc.push(v.to_f64());
    }
    (acc.amplitude(), acc.sinad_db())
}

/// RMS of the measured part of `audio`.
fn rms<R: RealSample>(audio: &[R]) -> f64 {
    let part = &audio[SETTLE..SETTLE + MEASURE];
    (part.iter().map(|v| v.to_f64() * v.to_f64()).sum::<f64>() / part.len() as f64).sqrt()
}

/// Magnitudes of the DFT (`DFT_LEN` points, rectangular window, scaled so that a complex tone
/// of amplitude A on a bin reads A) of the `DFT_LEN` samples after `SETTLE`.
fn dft(x: &[Complex64]) -> Vec<f64> {
    let part = &x[SETTLE..SETTLE + DFT_LEN];
    let twiddle: Vec<Complex64> = (0..DFT_LEN)
        .map(|k| {
            let w = -2.0 * PI * k as f64 / DFT_LEN as f64;
            Complex64::new(w.cos(), w.sin())
        })
        .collect();
    (0..DFT_LEN)
        .map(|k| {
            let mut acc = Complex64::new(0.0, 0.0);
            for (n, v) in part.iter().enumerate() {
                acc += v * twiddle[(n * k) % DFT_LEN];
            }
            acc.re.hypot(acc.im) / DFT_LEN as f64
        })
        .collect()
}

/// Index of the largest value.
fn peak(mags: &[f64]) -> usize {
    mags.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

fn db(ratio: f64) -> f64 {
    20.0 * ratio.log10()
}

// ---- LOOP-AM -----------------------------------------------------------------------------

/// `am_mod` (the defaults: carrier 0.5, depth 0.8) into `am_demod` (the defaults); the audio
/// is the envelope, so the tone comes back with amplitude carrier * depth * 0.5 = 0.2.
fn loop_am<T: BlockSample>(name: &str, min_sinad_db: f64)
where
    Sdrk: AmMod<T> + Mag<T> + DcBlockReal<T::Real> + ResampleReal<T::Real>,
{
    let cfg = AmModConfig::default();
    assert_eq!((cfg.carrier, cfg.depth), (0.5, 0.8));
    let s = &mut VecStorage;
    let mut modulator = AmModBlock::<T, VecStorage>::new(&cfg, rate(), s).unwrap();
    let mut demod =
        AmDemodBlock::<T, VecStorage>::new(&AmDemodConfig::default(), rate(), s).unwrap();
    let rf: Vec<T> = pump(&mut modulator, &tone::<T::Real>());
    let audio: Vec<T::Real> = pump(&mut demod, &rf);
    assert_eq!(audio.len(), SETTLE + MEASURE);
    let (amplitude, sinad) = fit(&audio);
    let expected = cfg.carrier * cfg.depth * TONE_AMPLITUDE;
    eprintln!(
        "{name}: SINAD at 1 kHz {sinad:.2} dB (need {min_sinad_db}), amplitude {amplitude:.6} \
         (carrier * depth * tone = {expected}, {:+.3} dB)",
        db(amplitude / expected)
    );
    assert!(
        sinad >= min_sinad_db,
        "{name}: SINAD {sinad:.2} dB, need {min_sinad_db} dB"
    );
}

#[test]
fn loop_am_f32() {
    loop_am::<C32>("loop_am_f32", AM_MIN_SINAD_F32_DB);
}

#[test]
fn loop_am_q15() {
    loop_am::<CQ15>("loop_am_q15", AM_MIN_SINAD_Q15_DB);
}

// ---- LOOP-SSB ----------------------------------------------------------------------------

fn ssb_mod_config(sideband: SsbSideband) -> SsbModConfig {
    SsbModConfig {
        sideband,
        ..SsbModConfig::default()
    }
}

fn ssb_demod_config(sideband: SsbSideband) -> SsbDemodConfig {
    SsbDemodConfig {
        sideband,
        ..SsbDemodConfig::default()
    }
}

/// `ssb_mod` with `sideband` (the defaults otherwise, band 300 ... 3000 Hz) into `ssb_demod`
/// with the same band, once with the same and once with the opposite sideband.
fn loop_ssb<T: BlockSample>(name: &str, sideband: SsbSideband, opposite: SsbSideband)
where
    Sdrk: Interleave<T> + Rotate<T> + Fir<T> + FirDecimate<T>,
{
    let s = &mut VecStorage;
    let mut modulator =
        SsbModBlock::<T, VecStorage>::new(&ssb_mod_config(sideband), rate(), s).unwrap();
    let rf: Vec<T> = pump(&mut modulator, &tone::<T::Real>());
    assert_eq!(rf.len(), SETTLE + MEASURE);

    // The spectrum of the transmitted signal: the wanted tone at +1 kHz (USB) or -1 kHz (LSB),
    // the mirror on the other side.
    let rf64: Vec<Complex64> = rf.iter().map(|v| v.to_c64()).collect();
    let mags = dft(&rf64);
    let (wanted_bin, mirror_bin) = if sideband == SsbSideband::Usb {
        (TONE_BIN, DFT_LEN - TONE_BIN)
    } else {
        (DFT_LEN - TONE_BIN, TONE_BIN)
    };
    let wanted = mags[wanted_bin];
    let mirror_db = db(mags[mirror_bin] / wanted);
    eprintln!(
        "{name}: ssb_mod output: peak bin {} (want {wanted_bin}), amplitude {wanted:.6}, mirror \
         {mirror_db:.2} dB",
        peak(&mags)
    );
    assert_eq!(peak(&mags), wanted_bin, "{name}: the peak of the output");
    assert!(
        mirror_db <= -SSB_MIN_MIRROR_DB,
        "{name}: mirror at {mirror_db:.2} dB, need -{SSB_MIN_MIRROR_DB} dB or less"
    );

    // Same sideband: the tone comes back at the input level and on its bin.
    let mut same =
        SsbDemodBlock::<T, VecStorage>::new(&ssb_demod_config(sideband), rate(), s).unwrap();
    let audio: Vec<T::Real> = pump(&mut same, &rf);
    assert_eq!(audio.len(), SETTLE + MEASURE);
    let (amplitude, sinad) = fit(&audio);
    let level_db = db(amplitude / TONE_AMPLITUDE);
    let audio64: Vec<Complex64> = audio
        .iter()
        .map(|v| Complex64::new(v.to_f64(), 0.0))
        .collect();
    let audio_mags = dft(&audio64);
    let audio_peak = peak(&audio_mags[..DFT_LEN / 2]);
    let same_rms = rms(&audio);

    // Opposite sideband: the level drops.
    let mut other =
        SsbDemodBlock::<T, VecStorage>::new(&ssb_demod_config(opposite), rate(), s).unwrap();
    let rejected: Vec<T::Real> = pump(&mut other, &rf);
    let (rejected_amplitude, _) = fit(&rejected);
    let rejection_db = db(same_rms / rms(&rejected));
    eprintln!(
        "{name}: ssb_demod same sideband: amplitude {amplitude:.6} ({level_db:+.4} dB), \
         SINAD {sinad:.2} dB, audio peak bin {audio_peak} (want {TONE_BIN}); opposite \
         sideband: level {rejection_db:.2} dB lower (RMS), 1 kHz tone {:.2} dB lower",
        db(amplitude / rejected_amplitude)
    );
    assert!(
        level_db.abs() <= SSB_MAX_LEVEL_ERROR_DB,
        "{name}: level {level_db:.4} dB from the input"
    );
    assert_eq!(audio_peak, TONE_BIN, "{name}: the audio is not at 1 kHz");
    assert!(
        rejection_db >= SSB_MIN_REJECTION_DB,
        "{name}: the opposite sideband is only {rejection_db:.2} dB lower"
    );
}

#[test]
fn loop_ssb_usb_f32() {
    loop_ssb::<C32>("loop_ssb_usb_f32", SsbSideband::Usb, SsbSideband::Lsb);
}

#[test]
fn loop_ssb_lsb_f32() {
    loop_ssb::<C32>("loop_ssb_lsb_f32", SsbSideband::Lsb, SsbSideband::Usb);
}

#[test]
fn loop_ssb_usb_q15() {
    loop_ssb::<CQ15>("loop_ssb_usb_q15", SsbSideband::Usb, SsbSideband::Lsb);
}

#[test]
fn loop_ssb_lsb_q15() {
    loop_ssb::<CQ15>("loop_ssb_lsb_q15", SsbSideband::Lsb, SsbSideband::Usb);
}
