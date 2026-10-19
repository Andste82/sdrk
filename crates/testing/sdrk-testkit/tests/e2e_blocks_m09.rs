//! The FM receive chain built from blocks (M09), compared with the kernel chain of M08.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::result_large_err
)]

use sdrk::Sdrk;
use sdrk_core::{C32, CQ15};
use sdrk_testkit::Arena;
use sdrk_testkit::e2e::{Measurement, measure_rx, measure_rx_blocks};

const QUICK: usize = 240_000;
const FULL: usize = 2_400_000;

fn blocks<T>(samples: usize) -> Measurement
where
    T: sdrk_testkit::e2e::E2eSample + sdrk_blocks::BlockSample,
    Sdrk: sdrk_core::kernels::convert::Convert<T>
        + sdrk_core::kernels::fused::Ddc<T>
        + sdrk_core::kernels::fused::FmDemod<T>
        + sdrk_core::kernels::demod::QuadDemod<T>
        + sdrk_core::kernels::decim::ResampleReal<T::Real>,
{
    let mut mem = vec![0u8; 1 << 20];
    measure_rx_blocks::<T>(&mut Arena::new(&mut mem), samples).unwrap()
}

#[test]
fn block_chain_meets_the_thresholds_of_the_kernel_chain() {
    for (samples, label) in [(QUICK, "0.1 s"), (FULL, "1.0 s")] {
        let m = blocks::<C32>(samples);
        assert_eq!(m.audio_samples, samples / 50);
        assert!(m.sinad_db >= 40.0, "f32 {label}: {} dB", m.sinad_db);
        assert!((m.tone_hz - 1000.0).abs() <= 0.1, "{}", m.tone_hz);
        assert!(m.amplitude > 0.9 && m.amplitude < 1.0, "{}", m.amplitude);
        let m = blocks::<CQ15>(samples);
        assert_eq!(m.audio_samples, samples / 50);
        assert!(m.sinad_db >= 35.0, "q15 {label}: {} dB", m.sinad_db);
        assert!((m.tone_hz - 1000.0).abs() <= 0.1, "{}", m.tone_hz);
        // Q15: the unit is phase / pi, the full deviation reads 0.625 (less the filter loss).
        assert!(m.amplitude > 0.55 && m.amplitude < 0.625, "{}", m.amplitude);
    }
}

#[test]
fn block_chain_and_kernel_chain_agree_on_the_tone() {
    // Both chains receive the same signal; the audio levels agree within a few percent
    // (the channel and audio filters differ in design).
    let mut mem = vec![0u8; 1 << 20];
    let kernels = measure_rx::<C32, Sdrk>(&Sdrk, &mut Arena::new(&mut mem), QUICK).unwrap();
    let blocks = blocks::<C32>(QUICK);
    assert!((kernels.amplitude - blocks.amplitude).abs() < 0.05);
    assert_eq!(kernels.audio_samples, blocks.audio_samples);
}
