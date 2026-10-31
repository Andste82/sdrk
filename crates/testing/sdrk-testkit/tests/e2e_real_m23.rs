//! End-to-end FM receive from a real 16-bit input (M23, E2E-REAL): the numbers behind the suite
//! cases `e2e_real_fm_rx_*`, for 0.1 s (quick) and 1.0 s (full) of signal in `f32` and Q15.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::result_large_err,
    clippy::print_stderr
)]

use sdrk::Sdrk;
use sdrk_blocks::{BlockReal, BlockSample};
use sdrk_core::kernels::convert::ConvertReal;
use sdrk_core::kernels::decim::ResampleReal;
use sdrk_core::kernels::demod::QuadDemod;
use sdrk_core::kernels::fused::{Ddc, FmDemod};
use sdrk_core::kernels::halfband::RealToComplex;
use sdrk_core::{C32, CQ15, ComplexSample};
use sdrk_testkit::e2e::{AUDIO_HZ, E2eSample, Measurement};
use sdrk_testkit::e2e_real::{
    REAL_FULL_SAMPLES, REAL_MIN_SINAD_F32_DB, REAL_MIN_SINAD_Q15_DB, REAL_QUICK_SAMPLES,
    measure_real_rx,
};
use sdrk_testkit::{Arena, Failure};

fn run<T>(samples: usize, arena_bytes: usize) -> Result<Measurement, Failure>
where
    T: E2eSample + BlockSample,
    <T as ComplexSample>::Real: BlockReal,
    Sdrk: ConvertReal<<T as ComplexSample>::Real>
        + RealToComplex<T>
        + Ddc<T>
        + FmDemod<T>
        + QuadDemod<T>
        + ResampleReal<<T as ComplexSample>::Real>,
{
    let mut mem = vec![0u8; arena_bytes];
    measure_real_rx::<T>(&mut Arena::new(&mut mem), samples)
}

fn check(m: &Measurement, samples: usize, min_sinad_db: f64) {
    eprintln!("{m:?}");
    assert_eq!(m.audio_samples, samples / 100);
    assert!(
        m.sinad_db >= min_sinad_db,
        "SINAD {} dB, need {min_sinad_db}",
        m.sinad_db
    );
    assert!((m.tone_hz - AUDIO_HZ).abs() <= 0.5, "tone {} Hz", m.tone_hz);
    assert!(m.amplitude > 0.1, "{m:?}");
}

/// Arena of the quick tier.
const QUICK_ARENA: usize = 64 * 1024;
/// The full variant stores the audio of its second half (24 000 values of 8 bytes).
const FULL_ARENA: usize = 256 * 1024;

#[test]
fn real_quick_f32_meets_the_threshold() {
    let m = run::<C32>(REAL_QUICK_SAMPLES, QUICK_ARENA).unwrap();
    check(&m, REAL_QUICK_SAMPLES, REAL_MIN_SINAD_F32_DB);
}

#[test]
fn real_quick_q15_meets_the_threshold() {
    let m = run::<CQ15>(REAL_QUICK_SAMPLES, QUICK_ARENA).unwrap();
    check(&m, REAL_QUICK_SAMPLES, REAL_MIN_SINAD_Q15_DB);
}

#[test]
fn real_full_f32_meets_the_threshold() {
    let m = run::<C32>(REAL_FULL_SAMPLES, FULL_ARENA).unwrap();
    check(&m, REAL_FULL_SAMPLES, REAL_MIN_SINAD_F32_DB);
}

#[test]
fn real_full_q15_meets_the_threshold() {
    let m = run::<CQ15>(REAL_FULL_SAMPLES, FULL_ARENA).unwrap();
    check(&m, REAL_FULL_SAMPLES, REAL_MIN_SINAD_Q15_DB);
}

#[test]
fn bad_sample_counts_and_small_arenas_are_rejected() {
    assert!(run::<C32>(0, QUICK_ARENA).is_err());
    assert!(run::<C32>(1001, QUICK_ARENA).is_err());
    assert!(run::<C32>(REAL_QUICK_SAMPLES, 1024).is_err());
}
