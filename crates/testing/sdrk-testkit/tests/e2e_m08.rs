//! End-to-end kernel chains of M08: FM receive from raw bytes and FM transmit then receive,
//! in `f32` and Q15, for 0.1 s (quick) and 1.0 s (full) of signal.
//!
//! The same chains run as suite cases (`e2e_fm_*`); these tests add the numbers (thresholds,
//! tone frequency, amplitude relations), negative controls and a look at the transmit spectrum.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::result_large_err
)]

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_core::{C32, CQ15};
use sdrk_device::formats;
use sdrk_testkit::e2e::{
    CHUNK, E2eSample, FmTx, IF_RATE_HZ, Measurement, RxKernels, TxKernels, fm_rx, measure_rx,
    measure_txrx,
};
use sdrk_testkit::encode::decode_reference;
use sdrk_testkit::meters::{estimate_frequency, spectral_mask};
use sdrk_testkit::{Arena, Failure};

const QUICK: usize = 240_000;
const FULL: usize = 2_400_000;

fn arena_mem() -> Vec<u8> {
    vec![0u8; 1 << 20]
}

fn rx<T: E2eSample>(samples: usize) -> Measurement
where
    Sdrk: RxKernels<T>,
{
    let mut mem = arena_mem();
    measure_rx::<T, Sdrk>(&Sdrk, &mut Arena::new(&mut mem), samples).unwrap()
}

fn txrx<T: E2eSample>(samples: usize) -> Measurement
where
    Sdrk: TxKernels<T>,
{
    let mut mem = arena_mem();
    measure_txrx::<T, Sdrk>(&Sdrk, &mut Arena::new(&mut mem), samples).unwrap()
}

fn check(m: &Measurement, samples: usize, min_db: f64) {
    assert_eq!(m.audio_samples, samples / 50);
    assert!(
        m.sinad_db >= min_db,
        "SINAD {} dB, need {min_db}",
        m.sinad_db
    );
    assert!(
        (m.tone_hz - 1000.0).abs() <= 0.1,
        "tone at {} Hz",
        m.tone_hz
    );
}

#[test]
fn rx_f32_quick() {
    let m = rx::<C32>(QUICK);
    check(&m, QUICK, 40.0);
    // Full deviation maps to about +-1.0, de-emphasis and the audio filter cost a few percent.
    assert!(
        m.amplitude > 0.9 && m.amplitude < 1.0,
        "amplitude {}",
        m.amplitude
    );
}

#[test]
fn rx_q15_quick() {
    let m = rx::<CQ15>(QUICK);
    check(&m, QUICK, 35.0);
    // The Q15 output unit is phase / pi: the full deviation of 75 kHz at 240 kHz is 0.625.
    let f = rx::<C32>(QUICK);
    let ratio = m.amplitude / f.amplitude;
    assert!((ratio - 0.625).abs() < 0.02, "amplitude ratio {ratio}");
}

#[test]
fn txrx_f32_quick() {
    let m = txrx::<C32>(QUICK);
    check(&m, QUICK, 35.0);
    assert!(
        m.amplitude > 0.9 && m.amplitude < 1.0,
        "amplitude {}",
        m.amplitude
    );
}

#[test]
fn txrx_q15_quick() {
    let m = txrx::<CQ15>(QUICK);
    check(&m, QUICK, 30.0);
    let f = txrx::<C32>(QUICK);
    assert!((m.amplitude / f.amplitude - 0.625).abs() < 0.02);
}

#[test]
fn rx_f32_full() {
    check(&rx::<C32>(FULL), FULL, 40.0);
}

#[test]
fn rx_q15_full() {
    check(&rx::<CQ15>(FULL), FULL, 35.0);
}

#[test]
fn txrx_f32_full() {
    check(&txrx::<C32>(FULL), FULL, 35.0);
}

#[test]
fn txrx_q15_full() {
    check(&txrx::<CQ15>(FULL), FULL, 30.0);
}

#[test]
fn the_chains_are_deterministic() {
    assert_eq!(rx::<CQ15>(QUICK), rx::<CQ15>(QUICK));
    assert_eq!(txrx::<C32>(QUICK), txrx::<C32>(QUICK));
}

#[test]
fn a_silent_carrier_has_no_audio() {
    // Negative control: without modulation the receiver finds no tone at 1 kHz.
    use num_complex::Complex64;
    use sdrk_testkit::fm::fm_baseband;
    let mut mem = arena_mem();
    let mut arena = Arena::new(&mut mem);
    let mut c = arena.carver();
    let samples = QUICK;
    let mut audio = vec![0.0f64; samples / 100];
    let mut scratch = vec![Complex64::new(0.0, 0.0); CHUNK];
    let mut source = |start: u64, bytes: &mut [u8]| -> Result<(), Failure> {
        fm_baseband(&mut scratch, 2.4e6, 600e3, 75e3, 1e3, 0.0, start);
        for s in &mut scratch {
            *s *= 0.9;
        }
        sdrk_testkit::encode::encode(&formats::CU8, &scratch, bytes)?;
        Ok(())
    };
    let produced = fm_rx::<C32, Sdrk>(
        &Sdrk,
        &mut c,
        &formats::CU8,
        &mut source,
        samples,
        samples / 100,
        &mut audio,
    )
    .unwrap();
    assert_eq!(produced, samples / 50);
    let sinad = sdrk_testkit::meters::sinad_db(&audio, 48_000.0, 1000.0);
    // A backend whose arithmetic leaves a constant carrier exactly constant produces exact
    // silence, for which the meter reports an infinite SINAD: there is no tone either way.
    let energy: f64 = audio.iter().map(|x| x * x).sum();
    assert!(
        sinad < 15.0 || energy < 1e-12,
        "SINAD {sinad} dB for an unmodulated carrier"
    );
}

#[test]
fn argument_errors_are_reported() {
    let mut mem = arena_mem();
    // A sample count that is not a multiple of the block size.
    let e = measure_rx::<C32, Sdrk>(&Sdrk, &mut Arena::new(&mut mem), QUICK + 1).unwrap_err();
    assert!(e.msg.contains("multiple"), "{e}");
    // Too little memory is a failure, not a panic.
    let mut tiny = vec![0u8; 1024];
    assert!(measure_rx::<C32, Sdrk>(&Sdrk, &mut Arena::new(&mut tiny), QUICK).is_err());
    assert!(measure_txrx::<CQ15, Sdrk>(&Sdrk, &mut Arena::new(&mut tiny), QUICK).is_err());
    // The transmitter wants consecutive blocks.
    let mut arena = Arena::new(&mut mem);
    let mut c = arena.carver();
    let mut tx = FmTx::<C32, Sdrk>::new(&Sdrk, &mut c, formats::CS8, 1.0).unwrap();
    let mut bytes = vec![0u8; 2 * CHUNK];
    tx.fill(0, &mut bytes).unwrap();
    assert!(tx.fill(0, &mut bytes).is_err());
    assert!(tx.fill(2 * CHUNK as u64, &mut bytes).is_err());
    tx.fill(CHUNK as u64, &mut bytes).unwrap();
    // A byte buffer of the wrong size is reported by the encoder.
    let mut short = vec![0u8; 10];
    assert!(tx.fill(2 * CHUNK as u64, &mut short).is_err());
    assert!(format!("{tx:?}").contains("FmTx"));
}

#[test]
fn the_transmit_spectrum_is_clean() {
    for q15 in [false, true] {
        let mut mem = arena_mem();
        let mut arena = Arena::new(&mut mem);
        let mut c = arena.carver();
        let mut bytes = vec![0u8; 16 * 2 * CHUNK];
        if q15 {
            let mut tx = FmTx::<CQ15, Sdrk>::new(&Sdrk, &mut c, formats::CS8, 1.0).unwrap();
            for (b, chunk) in bytes.chunks_mut(2 * CHUNK).enumerate() {
                tx.fill((b * CHUNK) as u64, chunk).unwrap();
            }
        } else {
            let mut tx = FmTx::<C32, Sdrk>::new(&Sdrk, &mut c, formats::CS8, 1.0).unwrap();
            for (b, chunk) in bytes.chunks_mut(2 * CHUNK).enumerate() {
                tx.fill((b * CHUNK) as u64, chunk).unwrap();
            }
        }
        let mut iq = vec![Complex64::new(0.0, 0.0); 16 * CHUNK];
        decode_reference(&formats::CS8, &bytes, &mut iq).unwrap();
        let rate = IF_RATE_HZ as f64;
        // The carrier sits at +600 kHz (the strongest line is within the deviation).
        assert!((estimate_frequency(&iq, rate) - 600e3).abs() < 80e3);
        // Everything outside 400 to 800 kHz (images of the interpolation, quantisation noise and
        // the DC offset of the 8-bit encoder) is at least 45 dB below the strongest bin.
        let clean = [(-1.2e6, 400e3, -45.0), (800e3, 1.19e6, -45.0)];
        assert_eq!(spectral_mask(&iq, rate, &clean), Ok(()), "q15 = {q15}");
        // The signal itself is there: no bin of the band is 30 dB down everywhere.
        assert!(spectral_mask(&iq, rate, &[(550e3, 650e3, -30.0)]).is_err());
        // The mask is not vacuous: a limit of -80 dB is violated outside the band.
        assert!(spectral_mask(&iq, rate, &[(-1.2e6, 400e3, -80.0)]).is_err());
    }
}
