//! End-to-end narrowband FM receive with squelch (M19): the numbers behind the suite cases
//! `e2e_nfm_rx_*`, for 0.4 s (quick) and 2.0 s (full) of signal in `f32` and Q15.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::result_large_err
)]

use sdrk::Sdrk;
use sdrk_blocks::BlockSample;
use sdrk_core::kernels::convert::Convert;
use sdrk_core::kernels::decim::ResampleReal;
use sdrk_core::kernels::demod::QuadDemod;
use sdrk_core::kernels::fused::{Ddc, FmDemod};
use sdrk_core::kernels::squelch::Squelch;
use sdrk_core::{C32, CQ15, ComplexSample};
use sdrk_testkit::e2e::E2eSample;
use sdrk_testkit::e2e_nfm::{
    NFM_FULL_SAMPLES, NFM_MAX_CLOSE_DELAY_S, NFM_MAX_OPEN_DELAY_S, NFM_MIN_SINAD_F32_DB,
    NFM_MIN_SINAD_Q15_DB, NFM_QUICK_SAMPLES, NfmMeasurement, measure_nfm,
};
use sdrk_testkit::{Arena, Failure};

fn run<T>(samples: usize, arena_bytes: usize) -> Result<NfmMeasurement, Failure>
where
    T: E2eSample + BlockSample,
    Sdrk: Convert<T>
        + Ddc<T>
        + FmDemod<T>
        + QuadDemod<T>
        + Squelch<T>
        + ResampleReal<<T as ComplexSample>::Real>,
{
    let mut mem = vec![0u8; arena_bytes];
    measure_nfm::<T>(&mut Arena::new(&mut mem), samples)
}

fn check(m: &NfmMeasurement, samples: usize, min_db: f64) {
    assert_eq!(m.audio_samples, samples / 50);
    assert!(
        m.sinad_db >= min_db,
        "SINAD {} dB, need {min_db}",
        m.sinad_db
    );
    assert!(!m.opened_early, "the gate opened before the transmission");
    assert!(
        m.open_delay_s <= NFM_MAX_OPEN_DELAY_S,
        "gate opens after {} s",
        m.open_delay_s
    );
    assert!(
        m.close_delay_s <= NFM_MAX_CLOSE_DELAY_S,
        "gate closes after {} s",
        m.close_delay_s
    );
    assert!(m.zero_checked > 0);
    assert_eq!(
        m.zero_violations, 0,
        "audio not zero outside the transmission"
    );
}

const QUICK_ARENA: usize = 64 * 1024;

#[test]
fn nfm_quick_f32_fits_the_quick_arena_and_meets_the_checks() {
    check(
        &run::<C32>(NFM_QUICK_SAMPLES, QUICK_ARENA).unwrap(),
        NFM_QUICK_SAMPLES,
        NFM_MIN_SINAD_F32_DB,
    );
}

#[test]
fn nfm_quick_q15_fits_the_quick_arena_and_meets_the_checks() {
    check(
        &run::<CQ15>(NFM_QUICK_SAMPLES, QUICK_ARENA).unwrap(),
        NFM_QUICK_SAMPLES,
        NFM_MIN_SINAD_Q15_DB,
    );
}

#[test]
fn nfm_full_f32_meets_the_checks() {
    check(
        &run::<C32>(NFM_FULL_SAMPLES, QUICK_ARENA).unwrap(),
        NFM_FULL_SAMPLES,
        NFM_MIN_SINAD_F32_DB,
    );
}

#[test]
fn nfm_full_q15_meets_the_checks() {
    check(
        &run::<CQ15>(NFM_FULL_SAMPLES, QUICK_ARENA).unwrap(),
        NFM_FULL_SAMPLES,
        NFM_MIN_SINAD_Q15_DB,
    );
}

#[test]
fn bad_sample_counts_are_rejected() {
    assert!(run::<C32>(0, QUICK_ARENA).is_err());
    assert!(run::<C32>(1000, QUICK_ARENA).is_err());
    assert!(run::<C32>(NFM_QUICK_SAMPLES, 1024).is_err());
}
