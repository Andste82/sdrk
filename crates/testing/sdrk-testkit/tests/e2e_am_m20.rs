//! End-to-end AM receive (M20): the numbers behind the suite cases `e2e_am_rx_*`, for 0.2 s
//! (quick) and 1.0 s (full) of signal in `f32` and Q15.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::result_large_err,
    clippy::print_stderr
)]

use sdrk::Sdrk;
use sdrk_blocks::BlockSample;
use sdrk_core::kernels::agc::Agc;
use sdrk_core::kernels::convert::Convert;
use sdrk_core::kernels::decim::ResampleReal;
use sdrk_core::kernels::demod::DcBlockReal;
use sdrk_core::kernels::fused::Ddc;
use sdrk_core::kernels::spectrum::Mag;
use sdrk_core::{C32, CQ15, ComplexSample};
use sdrk_testkit::e2e::E2eSample;
use sdrk_testkit::e2e_am::{
    AM_FULL_SAMPLES, AM_MAX_LEVEL_ERROR_DB, AM_MIN_SINAD_F32_DB, AM_MIN_SINAD_Q15_DB,
    AM_QUICK_SAMPLES, AmMeasurement, measure_am,
};
use sdrk_testkit::{Arena, Failure};

fn run<T>(samples: usize, arena_bytes: usize) -> Result<AmMeasurement, Failure>
where
    T: E2eSample + BlockSample,
    Sdrk: Convert<T>
        + Ddc<T>
        + Agc<T>
        + Mag<T>
        + DcBlockReal<<T as ComplexSample>::Real>
        + ResampleReal<<T as ComplexSample>::Real>,
{
    let mut mem = vec![0u8; arena_bytes];
    measure_am::<T>(&mut Arena::new(&mut mem), samples)
}

fn check(m: &AmMeasurement, samples: usize, min_db: f64) {
    assert_eq!(m.audio_samples, samples / 50);
    assert!(
        m.sinad_db >= min_db,
        "SINAD {} dB, need {min_db}",
        m.sinad_db
    );
    assert!(
        m.level_error_db.abs() <= AM_MAX_LEVEL_ERROR_DB,
        "audio level {} dB from reference * index (amplitude {}, expected {})",
        m.level_error_db,
        m.tone_amplitude,
        m.expected_amplitude
    );
}

const QUICK_ARENA: usize = 64 * 1024;

#[test]
fn am_quick_f32_meets_the_checks() {
    check(
        &run::<C32>(AM_QUICK_SAMPLES, QUICK_ARENA).unwrap(),
        AM_QUICK_SAMPLES,
        AM_MIN_SINAD_F32_DB,
    );
}

#[test]
fn am_quick_q15_meets_the_checks() {
    check(
        &run::<CQ15>(AM_QUICK_SAMPLES, QUICK_ARENA).unwrap(),
        AM_QUICK_SAMPLES,
        AM_MIN_SINAD_Q15_DB,
    );
}

#[test]
fn bad_sample_counts_are_rejected() {
    assert!(run::<C32>(0, QUICK_ARENA).is_err());
    assert!(run::<C32>(1001, QUICK_ARENA).is_err());
    assert!(run::<C32>(AM_QUICK_SAMPLES, 1024).is_err());
}

#[test]
fn am_full_f32_meets_the_checks() {
    check(
        &run::<C32>(AM_FULL_SAMPLES, QUICK_ARENA).unwrap(),
        AM_FULL_SAMPLES,
        AM_MIN_SINAD_F32_DB,
    );
}

#[test]
fn am_full_q15_meets_the_checks() {
    check(
        &run::<CQ15>(AM_FULL_SAMPLES, QUICK_ARENA).unwrap(),
        AM_FULL_SAMPLES,
        AM_MIN_SINAD_Q15_DB,
    );
}

/// The bias of the level comes from the attack and release ratio: the loop weighs an
/// overshoot against an undershoot differently, so the gain settles below the level of the
/// mean envelope with a slow release (0.1 s and 0.5 s: about -1 dB, the AM gain of the
/// specified flow) and above it with equal time constants (about +1.2 dB).
#[test]
fn the_level_bias_comes_from_the_attack_release_ratio() {
    use sdrk_blocks::{AgcBlock, AgcConfig, ArenaStorage, Block};
    use sdrk_core::Rate;
    // The envelope of the E2E signal at 48 kHz: 0.5 (1 + 0.5 cos).
    let envelope: Vec<C32> = (0..96_000)
        .map(|n| {
            let p = std::f64::consts::TAU * 1000.0 * f64::from(n) / 48_000.0;
            C32::new((0.5 + 0.25 * p.cos()) as f32, 0.0)
        })
        .collect();
    let run_agc = |attack_s: f64, release_s: f64| -> f64 {
        let cfg = AgcConfig {
            attack_s,
            release_s,
            ..AgcConfig::default()
        };
        let mut block = AgcBlock::<C32, _>::new(
            &cfg,
            Rate::hz(48_000).unwrap(),
            &mut ArenaStorage::new(&mut []),
        )
        .unwrap();
        let mut out = vec![C32::default(); envelope.len()];
        block.process(&envelope, &mut out).unwrap();
        block.gain_db()
    };
    let (equal, slow) = (run_agc(0.05, 0.05), run_agc(0.1, 0.5));
    assert!((0.5..2.0).contains(&equal), "{equal}");
    assert!((-1.5..-0.5).contains(&slow), "{slow}");
}
