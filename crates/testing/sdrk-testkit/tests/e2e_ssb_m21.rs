//! End-to-end SSB receive (M21): the numbers behind the suite cases `e2e_ssb_rx_*`, for 0.2 s
//! (quick) and 1.0 s (full) of signal in `f32` and Q15, receiving the upper and the lower
//! sideband.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::result_large_err,
    clippy::print_stderr
)]

use sdrk::Sdrk;
use sdrk_blocks::{BlockSample, SsbSideband};
use sdrk_core::kernels::agc::Agc;
use sdrk_core::kernels::convert::Convert;
use sdrk_core::kernels::decim::FirDecimate;
use sdrk_core::kernels::fused::Ddc;
use sdrk_core::kernels::mix::Rotate;
use sdrk_core::kernels::squelch::Squelch;
use sdrk_core::{C32, CQ15};
use sdrk_testkit::e2e::E2eSample;
use sdrk_testkit::e2e_ssb::{
    SSB_AMPLITUDE, SSB_FULL_SAMPLES, SSB_MIN_REJECTION_F32_DB, SSB_MIN_REJECTION_Q15_DB,
    SSB_MIN_SINAD_F32_DB, SSB_MIN_SINAD_Q15_DB, SSB_QUICK_SAMPLES, SsbMeasurement, measure_ssb,
};
use sdrk_testkit::{Arena, Failure};

fn run<T>(samples: usize, arena_bytes: usize, side: SsbSideband) -> Result<SsbMeasurement, Failure>
where
    T: E2eSample + BlockSample,
    Sdrk: Convert<T> + Ddc<T> + Squelch<T> + Agc<T> + Rotate<T> + FirDecimate<T>,
{
    let mut mem = vec![0u8; arena_bytes];
    measure_ssb::<T>(&mut Arena::new(&mut mem), samples, side)
}

fn check(m: &SsbMeasurement, samples: usize, min_sinad_db: f64, min_rejection_db: f64) {
    eprintln!("{m:?}");
    assert_eq!(m.audio_samples, samples / 50);
    assert!(
        m.sinad_db >= min_sinad_db,
        "SINAD {} dB, need {min_sinad_db}",
        m.sinad_db
    );
    assert!(
        m.rejection_db >= min_rejection_db,
        "the opposite sideband is only {} dB below the tone, need {min_rejection_db}",
        m.rejection_db
    );
    // The AGC sets the level; the tone must be there at a sensible amplitude.
    assert!(m.tone_amplitude > 0.1 * SSB_AMPLITUDE, "{m:?}");
}

/// Both chains fit the 80 KiB arena of the quick tier; the f32 chain needs about 75 KiB, more
/// than 64 KiB (the channel filter of 4400 taps and the audio filter of 870 taps are twice as
/// large as in Q15).
const QUICK_ARENA: usize = 80 * 1024;
const SIDES: [SsbSideband; 2] = [SsbSideband::Usb, SsbSideband::Lsb];

#[test]
fn ssb_quick_f32_meets_the_checks() {
    for side in SIDES {
        let m = run::<C32>(SSB_QUICK_SAMPLES, QUICK_ARENA, side).unwrap();
        check(
            &m,
            SSB_QUICK_SAMPLES,
            SSB_MIN_SINAD_F32_DB,
            SSB_MIN_REJECTION_F32_DB,
        );
    }
}

#[test]
fn ssb_quick_q15_meets_the_checks() {
    for side in SIDES {
        let m = run::<CQ15>(SSB_QUICK_SAMPLES, QUICK_ARENA, side).unwrap();
        check(
            &m,
            SSB_QUICK_SAMPLES,
            SSB_MIN_SINAD_Q15_DB,
            SSB_MIN_REJECTION_Q15_DB,
        );
    }
}

#[test]
fn bad_sample_counts_are_rejected() {
    assert!(run::<C32>(0, QUICK_ARENA, SsbSideband::Usb).is_err());
    assert!(run::<C32>(1001, QUICK_ARENA, SsbSideband::Usb).is_err());
    assert!(run::<C32>(SSB_QUICK_SAMPLES, 1024, SsbSideband::Usb).is_err());
    // The f32 chain does not fit 64 KiB, but it fits the arena of the quick tier.
    assert!(run::<C32>(SSB_QUICK_SAMPLES, 64 * 1024, SsbSideband::Usb).is_err());
    assert!(run::<C32>(SSB_QUICK_SAMPLES, QUICK_ARENA, SsbSideband::Usb).is_ok());
    assert!(run::<CQ15>(SSB_QUICK_SAMPLES, QUICK_ARENA, SsbSideband::Usb).is_ok());
}

#[test]
fn ssb_full_f32_meets_the_checks() {
    for side in SIDES {
        let m = run::<C32>(SSB_FULL_SAMPLES, QUICK_ARENA, side).unwrap();
        check(
            &m,
            SSB_FULL_SAMPLES,
            SSB_MIN_SINAD_F32_DB,
            SSB_MIN_REJECTION_F32_DB,
        );
    }
}

#[test]
fn ssb_full_q15_meets_the_checks() {
    for side in SIDES {
        let m = run::<CQ15>(SSB_FULL_SAMPLES, QUICK_ARENA, side).unwrap();
        check(
            &m,
            SSB_FULL_SAMPLES,
            SSB_MIN_SINAD_Q15_DB,
            SSB_MIN_REJECTION_Q15_DB,
        );
    }
}
