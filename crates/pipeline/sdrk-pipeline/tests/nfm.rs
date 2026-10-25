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
const SAMPLES: usize = 960_000; // 0.4 s
const OFFSET_HZ: f64 = 300_000.0;
const AUDIO_RATE: f64 = 48_000.0;

/// CU8 bytes: noise over the whole run, a 5 kHz deviation transmission at 1 kHz audio and
/// amplitude 0.5 in the middle half.
fn recording() -> Vec<u8> {
    let (from, to) = (SAMPLES / 4, 3 * SAMPLES / 4);
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
    let from = SAMPLES / 4 / per_sample; // 4800
    let to = 3 * SAMPLES / 4 / per_sample; // 14400
    // Before the transmission the audio is exactly zero.
    assert!(audio[..from].iter().all(|v| *v == 0.0), "silence before");
    // More than 0.1 s cannot fit after the end of a 0.4 s run, but the gate closes after
    // about 85 ms: the last 400 samples (8 ms) are exactly zero.
    assert!(
        audio[audio.len() - 400..].iter().all(|v| *v == 0.0),
        "silence after"
    );
    // The transmission carries the 1 kHz tone.
    let mid = &audio[from + 1200..to - 1200];
    let on = tone_power(mid, 1000.0, AUDIO_RATE);
    let off = tone_power(mid, 3300.0, AUDIO_RATE);
    assert!(on > 1000.0 * off, "1 kHz tone: {on} vs {off}");
    assert!(mid.iter().any(|v| v.abs() > 0.5));

    // Opening: the first non-zero audio sample comes at most 1 ms (48 samples) plus the
    // delay of the channel and audio filters after the start of the transmission, and not
    // before it.
    let first = audio.iter().position(|v| *v != 0.0).unwrap();
    assert!(
        (from..from + 48 + 100).contains(&first),
        "first non-zero audio sample {first}, transmission starts at {from}"
    );
    // Closing: the last non-zero sample is at most 0.1 s (4800 samples) after the end and not
    // before it (the signal passes up to its end).
    let last = audio.iter().rposition(|v| *v != 0.0).unwrap();
    assert!(
        (to - 100..to + 4800).contains(&last),
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
    assert_eq!(common::bits(&whole), common::bits(&other));
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
