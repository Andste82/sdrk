//! The `nfm_receiver` example flow: it validates, builds and runs on a narrow-band FM burst and
//! the squelch keeps the audio exactly silent outside the transmission.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::assert_is_empty,
    clippy::unreadable_literal,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::redundant_closure_for_method_calls
)]

mod common;

use num_complex::Complex64;
use sdrk_device::formats;
use sdrk_pipeline::{BuildOptions, InputData, Pipeline, Registry};
use sdrk_testkit::{Pcg32, encode::encode, fm::fm_baseband, meters::sinad_db};
use serde_json::json;

use common::{build, flow, tone_power};

const RATE: f64 = 2_400_000.0;
const SAMPLES: usize = 1_440_000; // 0.6 s
/// The transmission runs from 0.1 s to 0.3 s; the rest of the run leaves room for the gate to
/// close (about 85 ms) and for the `ctcss` filter to settle.
const TX_FROM: usize = 240_000;
const TX_TO: usize = 720_000;
const OFFSET_HZ: f64 = 300_000.0;
const AUDIO_RATE: f64 = 48_000.0;

/// CU8 bytes: noise over the whole run, a 5 kHz deviation transmission at 1 kHz audio and
/// amplitude 0.5 from 0.1 s to 0.3 s.
fn recording() -> Vec<u8> {
    let (from, to) = (TX_FROM, TX_TO);
    let mut x = vec![Complex64::new(0.0, 0.0); SAMPLES];
    fm_baseband(
        &mut x[from..to],
        RATE,
        OFFSET_HZ,
        5_000.0,
        1_000.0,
        1.0,
        from as u64,
    );
    let mut rng = Pcg32::new(190, 3);
    for (i, s) in x.iter_mut().enumerate() {
        if (from..to).contains(&i) {
            *s *= 0.5;
        }
        *s += Complex64::new(rng.normal(), rng.normal()) * 0.003;
    }
    let mut bytes = vec![0u8; 2 * SAMPLES];
    encode(&formats::CU8, &x, &mut bytes).unwrap();
    bytes
}

fn run(p: &mut Pipeline, bytes: &[u8], chunk: usize) -> Vec<f32> {
    let mut audio = Vec::new();
    let mut buf = vec![0f32; 4096];
    for piece in bytes.chunks(chunk) {
        p.push("rf", InputData::Raw(piece)).unwrap();
        loop {
            let got = p.read_stream("audio", &mut buf).unwrap();
            if got == 0 {
                break;
            }
            audio.extend_from_slice(&buf[..got]);
        }
    }
    audio
}

fn receiver() -> Pipeline {
    let mut p = build(&flow("nfm_receiver"));
    p.set_param("tune", "frequency_hz", json!(OFFSET_HZ))
        .unwrap();
    p
}

#[test]
fn the_flow_validates_without_warnings() {
    let report =
        Registry::standard().validate_json(&flow("nfm_receiver"), &BuildOptions::default());
    assert!(report.is_ok(), "{}", report.to_json());
}

#[test]
fn the_squelch_gates_the_transmission() {
    let bytes = recording();
    let audio = run(&mut receiver(), &bytes, 8192);
    let per_sample = (RATE / AUDIO_RATE) as usize;
    assert!(audio.len() >= SAMPLES / per_sample - 2);
    let from = TX_FROM / per_sample; // 4800
    let to = TX_TO / per_sample; // 14400
    // Before the transmission the audio is exactly zero.
    assert!(audio[..from].iter().all(|v| *v == 0.0), "silence before");
    // After the transmission the gate closes within about 85 ms and the `ctcss` filter (f32,
    // DF2T) lets its state decay to zero and flushes it: from 0.2 s (9600 samples) after the
    // end of the transmission the audio is exactly zero (amends M19 test 7, M22 section 4).
    assert!(audio.len() > to + 9600);
    assert!(
        audio[to + 9600..].iter().all(|v| *v == 0.0),
        "silence after"
    );
    // The transmission carries the 1 kHz tone.
    let mid = &audio[from + 1200..to - 1200];
    let on = tone_power(mid, 1000.0, AUDIO_RATE);
    let off = tone_power(mid, 3300.0, AUDIO_RATE);
    assert!(on > 1000.0 * off, "1 kHz tone: {on} vs {off}");
    // The 530 us de-emphasis lowers a 1 kHz tone by about 11 dB: peak 0.286 measured (0.5
    // before M25, when the demodulator did not de-emphasise); decision D187.
    assert!(mid.iter().any(|v| v.abs() > 0.2));

    // Opening: the first non-zero audio sample comes at most 1 ms (48 samples) plus the
    // delay of the channel and audio filters after the start of the transmission, and not
    // before it.
    let first = audio.iter().position(|v| *v != 0.0).unwrap();
    assert!(
        (from..from + 48 + 100).contains(&first),
        "first non-zero audio sample {first}, transmission starts at {from}"
    );
    // Closing: the last non-zero sample is at most 0.2 s (9600 samples) after the end and not
    // before it (the signal passes up to its end). The gate closes after about 85 ms; the
    // rest is the decay of the `ctcss` state down to the flush limit of 1e-30.
    let last = audio.iter().rposition(|v| *v != 0.0).unwrap();
    assert!(
        (to - 100..to + 9600).contains(&last),
        "last non-zero audio sample {last}, transmission ends at {to}"
    );
    // The SINAD threshold of E2E-NFM for f32 (milestone section 5, check 1).
    let (lo, hi) = (from + (to - from) / 10, to - (to - from) / 10);
    let window: Vec<f64> = audio[lo..hi].iter().map(|v| f64::from(*v)).collect();
    let sinad = sinad_db(&window, AUDIO_RATE, 1000.0);
    assert!(sinad >= 30.0, "pipeline SINAD {sinad} dB, need 30 dB");
}

#[test]
fn the_gate_must_sit_between_the_channel_filter_and_the_demodulator() {
    // squelch needs complex samples: behind the demodulator (real audio) the flow is invalid.
    let mut spec: serde_json::Value = serde_json::from_str(&flow("nfm_receiver")).unwrap();
    spec["edges"] = json!([
        {"from": "rf", "to": "tune"},
        {"from": "tune", "to": "demod"},
        {"from": "demod", "to": "gate"},
        {"from": "gate", "to": "audio"},
    ]);
    let report = Registry::standard().validate_json(&spec.to_string(), &BuildOptions::default());
    assert!(!report.is_ok(), "{}", report.to_json());
}

#[test]
fn chunking_does_not_change_the_output() {
    let bytes = recording();
    let whole = run(&mut receiver(), &bytes, 8192);
    let other = run(&mut receiver(), &bytes, 4097);
    // The f32 `deemphasis` of the SIMD backend is a tolerance kernel whose bits depend on the
    // split (docs/kernels.md, decision of 2026-10-29), so since the demodulator de-emphasises
    // (M25) the chunked run is compared by SNR (97 dB measured, at least 90 dB required) and by
    // the positions of the exact zeros, which must be identical (decision D187).
    assert_eq!(whole.len(), other.len());
    let zeros = |x: &[f32]| x.iter().map(|v| *v == 0.0).collect::<Vec<_>>();
    assert_eq!(zeros(&whole), zeros(&other));
    let (mut signal, mut error) = (0f64, 0f64);
    for (a, b) in whole.iter().zip(&other) {
        signal += f64::from(*a).powi(2);
        error += (f64::from(*a) - f64::from(*b)).powi(2);
    }
    let snr = 10.0 * (signal / error).log10();
    assert!(snr >= 90.0, "chunked run differs: SNR {snr} dB");
}

#[test]
fn the_threshold_can_be_moved_while_running() {
    let bytes = recording();
    // A threshold above the signal power (0.25 = -6 dBFS) keeps the gate closed all the time.
    let mut p = receiver();
    p.set_param("gate", "threshold_db", json!(0)).unwrap();
    let audio = run(&mut p, &bytes, 8192);
    assert!(!audio.is_empty());
    assert!(audio.iter().all(|v| *v == 0.0));
    // tau_s is a build parameter.
    assert!(p.set_param("gate", "tau_s", json!(0.1)).is_err());
}

/// The bound for the output of the Q15 `ctcss` filter (DF1, high-pass 300 Hz at 48 kHz, order 4)
/// after the squelch closed, in LSB. Measured: a constant DC rest of 321 LSB (about -40 dBFS,
/// inaudible), not a cycle. The 16 LSB of test 6 hold for the low-pass `butter4_lp` at 0.1 fs;
/// the rounding of a high-pass with a cutoff of fs / 160 is amplified by `1 / |1 + a1 + a2|`.
/// The maintainer approved the measured value as the Q15 bound (decision D22, M22 section 4);
/// the `f32` flow stays exactly zero (`the_squelch_gates_the_transmission`).
const CTCSS_Q15_MAX_REST_LSB: i32 = 321;

/// The bound for the output of the Q15 `fm_demod` (with the 530 us `deemphasis`) after the
/// squelch closed, in LSB (M25 section 4). The dead band of the Q15 `deemphasis` keeps the
/// output at a small constant: -12 LSB measured.
const Q15_DEMOD_MAX_REST_LSB: i32 = 16;

#[test]
fn the_q15_flow_keeps_at_most_the_approved_rest_after_the_squelch() {
    q15_flow_silence(CTCSS_Q15_MAX_REST_LSB);
}

fn q15_flow_silence(limit_lsb: i32) {
    // A Q15 build of the flow with a second output behind `fm_demod` (fan-out).
    let mut spec: serde_json::Value = serde_json::from_str(&flow("nfm_receiver")).unwrap();
    spec["settings"]["precision"] = json!("q15");
    spec["edges"]
        .as_array_mut()
        .unwrap()
        .push(json!({"from": "demod", "to": "raw"}));
    spec["outputs"]["raw"] = json!({"type": "stream", "capacity": 16384, "drop": "error"});
    let mut p = build(&spec.to_string());
    p.set_param("tune", "frequency_hz", json!(OFFSET_HZ))
        .unwrap();

    let bytes = recording();
    let (mut audio, mut raw) = (Vec::new(), Vec::new());
    let mut buf = vec![sdrk::Q15(0); 4096];
    for piece in bytes.chunks(8192) {
        p.push("rf", InputData::Raw(piece)).unwrap();
        for (name, sink) in [("audio", &mut audio), ("raw", &mut raw)] {
            loop {
                let got = p.read_stream(name, &mut buf).unwrap();
                if got == 0 {
                    break;
                }
                sink.extend(buf[..got].iter().map(|q| q.0));
            }
        }
    }
    let per_sample = (RATE / AUDIO_RATE) as usize;
    let (from, to) = (TX_FROM / per_sample, TX_TO / per_sample);
    assert_eq!(audio.len(), raw.len());
    assert!(audio.len() > to + 9600);
    // Before the transmission the demodulator output is exactly zero (the squelch writes
    // zeros); from 0.2 s after its end the Q15 `deemphasis` (530 us) sits in its dead band at a
    // constant of at most 16 LSB (amends M19/M22, decision D187; -12 LSB measured, analytically
    // +-12 LSB for alpha_q15 = 1265, docs/kernels.md) ...
    assert!(raw[..from].iter().all(|v| *v == 0), "demod silence before");
    let demod_rest = raw[to + 9600..]
        .iter()
        .map(|v| i32::from(*v).abs())
        .max()
        .unwrap();
    assert!(
        demod_rest <= Q15_DEMOD_MAX_REST_LSB,
        "demod rest {demod_rest} LSB"
    );
    // ... and the output of `ctcss` (Q15, DF1) stays within the approved DC rest (D22).
    let worst = |x: &[i16]| x.iter().map(|v| i32::from(*v).abs()).max().unwrap();
    assert_eq!(worst(&audio[..from]), 0, "before");
    let after = worst(&audio[to + 9600..]);
    assert!(after <= limit_lsb, "ctcss after: {after} LSB");

    // The transmission is audible.
    let peak = worst(&audio[from + 1200..to - 1200]);
    // 1 kHz at full deviation reads 2000 LSB after the 530 us de-emphasis (1967 at the
    // demodulator; 4000 or more before M25), decision D187.
    assert!(peak > 1500, "peak {peak}");
}
