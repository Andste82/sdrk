//! End-to-end FM stereo receive (M24, E2E-STEREO): the numbers behind the suite cases
//! `e2e_fm_stereo_*`, for 0.5 s (quick) and 2.0 s (full) of signal in `f32` and Q15, the
//! switching of the pilot and the mono transmissions.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::result_large_err,
    clippy::print_stderr
)]

use sdrk_core::{C32, CQ15};
use sdrk_testkit::Arena;
use sdrk_testkit::e2e_stereo::{
    FULL_SAMPLES, QUICK_SAMPLES, RATE, StereoMeasurement, StereoSignal, check_mono, check_stereo,
    check_switching, measure_stereo,
};

fn run32(samples: usize, signal: &StereoSignal) -> StereoMeasurement {
    let mut mem = vec![0u8; 80 * 1024];
    let m = measure_stereo::<C32>(&mut Arena::new(&mut mem), samples, signal).unwrap();
    eprintln!("f32 {m:?}");
    m
}

fn run15(samples: usize, signal: &StereoSignal) -> StereoMeasurement {
    let mut mem = vec![0u8; 80 * 1024];
    let m = measure_stereo::<CQ15>(&mut Arena::new(&mut mem), samples, signal).unwrap();
    eprintln!("q15 {m:?}");
    m
}

#[test]
fn stereo_f32_quick() {
    let m = run32(QUICK_SAMPLES, &StereoSignal::STEREO);
    assert_eq!(m.audio_samples, QUICK_SAMPLES / 50);
    check_stereo("e2e_fm_stereo_f32", &m, false).unwrap();
}

#[test]
fn stereo_q15_quick() {
    let m = run15(QUICK_SAMPLES, &StereoSignal::STEREO);
    check_stereo("e2e_fm_stereo_q15", &m, true).unwrap();
}

#[test]
fn stereo_f32_full() {
    let m = run32(FULL_SAMPLES, &StereoSignal::STEREO);
    check_stereo("e2e_fm_stereo_f32_2s", &m, false).unwrap();
}

#[test]
fn stereo_q15_full() {
    let m = run15(FULL_SAMPLES, &StereoSignal::STEREO);
    check_stereo("e2e_fm_stereo_q15_2s", &m, true).unwrap();
}

#[test]
fn the_pilot_switches_the_decoder_between_stereo_and_mono() {
    let samples = RATE as usize;
    let m = run32(samples, &StereoSignal::switching(0.3, 0.7));
    check_switching("e2e_fm_stereo_switch_f32", &m).unwrap();
}

#[test]
fn a_mono_transmission_stays_mono() {
    let m = run32(QUICK_SAMPLES / 2, &StereoSignal::MONO);
    check_mono("e2e_fm_stereo_mono", &m).unwrap();
    let m = run15(QUICK_SAMPLES / 2, &StereoSignal::MONO);
    check_mono("e2e_fm_stereo_mono", &m).unwrap();
}

#[test]
fn noise_without_a_carrier_never_reports_stereo() {
    let m = run32(RATE as usize, &StereoSignal::NOISE);
    check_mono("e2e_fm_stereo_noise", &m).unwrap();
    let m = run15(RATE as usize, &StereoSignal::NOISE);
    check_mono("e2e_fm_stereo_noise", &m).unwrap();
}
