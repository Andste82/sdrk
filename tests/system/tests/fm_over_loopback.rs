//! A 1 kHz FM transmission through the loopback channel: the transmitter flow in pull mode, a
//! delayed, attenuated, frequency-shifted and noisy channel, the receiver flow in push mode.
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::doc_markdown,
    clippy::unreadable_literal
)]

use sdrk_device::{Rate, TxStream, formats};
use sdrk_pipeline::{FormatName, Precision};
use sdrk_system_tests::{
    AUDIO_RATE, IF_RATE, ToneAudio, build, cs8_power, flow, run_receiver, set_input_format,
    transmit,
};
use sdrk_testkit::channel::{Loopback, LoopbackParams};
use sdrk_testkit::meters::sinad_db;

const DELAY_SAMPLES: f64 = 1234.37;
const ATTENUATION_DB: f64 = 6.0;
const FREQ_OFFSET_HZ: f64 = 1500.0;
/// Carrier to noise ratio in a 200 kHz channel.
const CNR_DB: f64 = 30.0;
const AUDIO_SECONDS: f64 = 0.5;

fn fm_over_loopback(precision: Precision, min_sinad_db: f64) -> f64 {
    // Transmitter: a 1 kHz tone through fm_transmitter.json, CS8 at 2.4 MS/s.
    let tx_spec = flow("fm_transmitter");
    let total = (AUDIO_RATE * AUDIO_SECONDS) as u64;
    let mut tone = ToneAudio::new(1000.0, AUDIO_RATE, 0.8, total);
    let blocks = transmit(&tx_spec, "rf", &mut tone, 1 << 16);
    assert!(blocks.len() > 3);
    let tx_samples: usize = blocks.iter().map(|(b, _)| b.len() / 2).sum();
    assert!(tx_samples > (total as usize - 100) * 50);

    // Channel: the noise follows from the power of what was transmitted.
    let all: Vec<u8> = blocks.iter().flat_map(|(b, _)| b.iter().copied()).collect();
    let p_tx = cs8_power(&all);
    let p_carrier = p_tx * 10f64.powf(-ATTENUATION_DB / 10.0);
    // Complex noise power 2 sigma^2 spread over 2.4 MHz; 200 kHz of it count for the CNR.
    let sigma = (p_carrier / 10f64.powf(CNR_DB / 10.0) * (IF_RATE as f64 / 200_000.0) / 2.0).sqrt();
    let mut params = LoopbackParams::new(Rate::hz(IF_RATE).unwrap(), formats::CS8);
    params.t0_ns = 1_700_000_000_000_000_000;
    params.delay_samples = DELAY_SAMPLES;
    params.attenuation_db = ATTENUATION_DB;
    params.freq_offset_hz = FREQ_OFFSET_HZ;
    params.noise_sigma = sigma;
    params.seed = 12345;
    let (mut tx, mut rx) = Loopback::new(params).unwrap().split();
    for (bytes, meta) in &blocks {
        assert_eq!(tx.write(bytes, meta).unwrap(), bytes.len());
    }

    // Receiver: fm_receiver.json with the input format replaced by cs8.
    let mut rx_spec = flow("fm_receiver");
    set_input_format(&mut rx_spec, "rf", FormatName::Cs8);
    rx_spec.settings.precision = precision;
    let mut p = build(&rx_spec);
    let audio = run_receiver(
        &mut p,
        &mut rx,
        "rf",
        "audio",
        tx_samples + DELAY_SAMPLES as usize,
        precision,
    );

    let half: Vec<f64> = audio[audio.len() / 2..]
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let sinad = sinad_db(&half, AUDIO_RATE, 1000.0);
    assert!(
        sinad >= min_sinad_db,
        "{precision:?}: SINAD {sinad:.1} dB, required {min_sinad_db} dB (CNR {CNR_DB} dB, sigma {sigma:.4})"
    );
    sinad
}

#[test]
fn fm_over_loopback_f32() {
    let sinad = fm_over_loopback(Precision::F32, 30.0);
    println!("f32 SINAD {sinad:.1} dB");
}

#[test]
fn fm_over_loopback_q15() {
    let sinad = fm_over_loopback(Precision::Q15, 25.0);
    println!("q15 SINAD {sinad:.1} dB");
}
