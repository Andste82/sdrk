//! The kinds `halfband` and `real_to_complex`, the real variant of `convert` (explicit and
//! automatic) and the `real_input_fm_receiver` example flow: plan figures, the cost formula,
//! validation, and a run on the E2E-REAL signal.
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
use sdrk::{Delay, Rate};
use sdrk_blocks::design::halfband_len;
use sdrk_device::formats;
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Pipeline, Registry};
use sdrk_testkit::e2e::{AUDIO_HZ, DEVIATION_HZ};
use sdrk_testkit::e2e_real::{REAL_IF_HZ, REAL_NOISE_SIGMA, REAL_RATE_HZ, REAL_SIGNAL_AMPLITUDE};
use sdrk_testkit::meters::sinad_db;
use sdrk_testkit::{Pcg32, encode::encode, fm::fm_baseband};
use serde_json::{Value, json};

use common::{build, flow};

/// A one-node flow on a 48 kHz input of type `ty`.
fn single(kind: &str, ty: &str, precision: &str, params: Value) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": precision, "max_chunk": 1024 },
        "inputs": { "in": { "type": ty, "rate_hz": 48000 } },
        "nodes": { "n": { "kind": kind, "params": params } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    })
}

#[test]
fn the_kinds_are_registered_with_their_kernels() {
    let registry = Registry::standard();
    let h = registry.factory("halfband").unwrap();
    assert_eq!(h.kernels(), ["halfband_decimate", "halfband_decimate_real"]);
    assert_eq!(h.description().kind, "halfband");
    let r = registry.factory("real_to_complex").unwrap();
    assert_eq!(r.kernels(), ["real_to_complex"]);
    assert_eq!(registry.descriptions().count(), 18);
}

#[test]
fn the_plan_follows_the_cost_formula_and_halves_the_rate() {
    let registry = Registry::standard();
    for (kind, ty, precision) in [
        ("halfband", "c32", "f32"),
        ("halfband", "cq15", "q15"),
        ("halfband", "f32", "f32"),
        ("halfband", "q15", "q15"),
        ("real_to_complex", "f32", "f32"),
        ("real_to_complex", "q15", "q15"),
    ] {
        for (transition, att) in [(None, 60.0), (Some(2400.0), 80.0), (Some(480.0), 40.0)] {
            let mut params = json!({ "attenuation_db": att });
            if let Some(t) = transition {
                params["transition_hz"] = json!(t);
            }
            let doc = single(kind, ty, precision, params);
            let report = registry.validate_json(&doc.to_string(), &BuildOptions::default());
            assert!(report.is_ok(), "{kind} {ty}: {}", report.to_json());
            let taps = halfband_len(transition.unwrap_or(4800.0), att, 48_000.0).unwrap();
            assert_eq!(taps % 4, 3);
            // (N + 3) / 4 per input sample.
            assert_eq!(
                report.cost_per_input_sample,
                (taps + 3) as f64 / 4.0,
                "{kind} {ty} {transition:?} {att}"
            );
            let node = &report.nodes[0];
            assert_eq!(node.rate_in, Rate::hz(48_000).unwrap());
            assert_eq!(node.rate_out, Rate::hz(24_000).unwrap());
            assert_eq!(
                node.delay_samples_in.reduced(),
                Delay::new(taps as u64 - 1, 2).unwrap().reduced()
            );
            assert!(report.memory_bytes > 0);
        }
    }
}

#[test]
fn the_estimate_matches_the_built_pipeline() {
    let registry = Registry::standard();
    let doc = single(
        "real_to_complex",
        "q15",
        "q15",
        json!({ "attenuation_db": 70 }),
    );
    let report = registry.validate_json(&doc.to_string(), &BuildOptions::default());
    assert!(report.is_ok(), "{}", report.to_json());
    let built = registry
        .build_json(&doc.to_string(), &BuildOptions::default())
        .unwrap();
    let built = built.report();
    assert!(built.memory_bytes.abs_diff(report.memory_bytes) <= report.memory_bytes / 20);
    assert_eq!(built.cost_per_input_sample, report.cost_per_input_sample);
}

#[test]
fn types_and_parameters_are_checked() {
    let registry = Registry::standard();
    let invalid = |kind: &str, ty: &str, precision: &str, params: Value| {
        let doc = single(kind, ty, precision, params);
        !registry
            .validate_json(&doc.to_string(), &BuildOptions::default())
            .is_ok()
    };
    // real_to_complex takes real samples only.
    assert!(invalid("real_to_complex", "c32", "f32", json!({})));
    assert!(invalid("real_to_complex", "cq15", "q15", json!({})));
    // A transition at or above half the input rate, an attenuation below 8 dB, too few taps.
    assert!(invalid(
        "halfband",
        "c32",
        "f32",
        json!({ "transition_hz": 24000 })
    ));
    assert!(invalid(
        "halfband",
        "c32",
        "f32",
        json!({ "attenuation_db": 7 })
    ));
    assert!(invalid(
        "halfband",
        "c32",
        "f32",
        json!({ "transition_hz": 100, "max_taps": 31 })
    ));
    assert!(invalid("halfband", "f32", "f32", json!({ "nope": 1 })));
    // Each is fine with its defaults.
    for (kind, ty, precision) in [
        ("halfband", "c32", "f32"),
        ("halfband", "q15", "q15"),
        ("real_to_complex", "q15", "q15"),
    ] {
        assert!(!invalid(kind, ty, precision, json!({})), "{kind} {ty}");
    }
}

/// The real flow with `precision` and the input format.
fn real_flow(precision: &str) -> Value {
    let mut doc: Value = serde_json::from_str(&flow("real_input_fm_receiver")).unwrap();
    doc["settings"]["precision"] = json!(precision);
    doc
}

#[test]
fn a_real_raw_input_gets_the_real_automatic_conversion() {
    let registry = Registry::standard();
    for (precision, port) in [("f32", "F32"), ("q15", "Q15")] {
        let report =
            registry.validate_json(&real_flow(precision).to_string(), &BuildOptions::default());
        assert!(report.is_ok(), "{}", report.to_json());
        let first = &report.nodes[0];
        assert_eq!(
            (first.name.as_str(), first.kind.as_str()),
            ("rf__convert", "convert")
        );
        assert!(first.auto);
        // Same rate in and out; the chain goes on to real_to_complex at half the rate.
        assert_eq!(first.rate_in, first.rate_out);
        let names: Vec<_> = report.nodes.iter().map(|n| n.kind.as_str()).collect();
        assert_eq!(
            names,
            ["convert", "real_to_complex", "ddc", "fm_demod"],
            "{port}"
        );
        let r2c = report
            .nodes
            .iter()
            .find(|n| n.kind == "real_to_complex")
            .unwrap();
        assert_eq!(r2c.rate_in, Rate::hz(REAL_RATE_HZ).unwrap());
        assert_eq!(r2c.rate_out, Rate::hz(REAL_RATE_HZ / 2).unwrap());
    }
}

#[test]
fn a_real_stream_cannot_feed_a_complex_block() {
    // The real converter output is real: a ddc directly behind it is a type error.
    let mut doc = real_flow("f32");
    doc["edges"] = json!([
        { "from": "rf", "to": "tune" },
        { "from": "tune", "to": "demod" },
        { "from": "demod", "to": "audio" }
    ]);
    doc["nodes"].as_object_mut().unwrap().remove("iq");
    let report = Registry::standard().validate_json(&doc.to_string(), &BuildOptions::default());
    assert!(!report.is_ok());
    assert!(
        report
            .errors
            .iter()
            .any(|d| d.code == DiagCode::TypeMismatch),
        "{}",
        report.to_json()
    );
}

/// RS16 bytes of the E2E-REAL signal (the same as the testkit case).
fn rs16_recording(samples: usize) -> Vec<u8> {
    let chunk = 1000usize;
    let mut rng = Pcg32::new(2300, 1);
    let mut bytes = vec![0u8; 2 * samples];
    let mut block = vec![Complex64::new(0.0, 0.0); chunk];
    let mut start = 0usize;
    while start < samples {
        let n = chunk.min(samples - start);
        fm_baseband(
            &mut block[..n],
            REAL_RATE_HZ as f64,
            REAL_IF_HZ,
            DEVIATION_HZ,
            AUDIO_HZ,
            1.0,
            start as u64,
        );
        for s in &mut block[..n] {
            *s = Complex64::new(
                REAL_SIGNAL_AMPLITUDE * s.re + REAL_NOISE_SIGMA * rng.normal(),
                0.0,
            );
        }
        encode(
            &formats::RS16,
            &block[..n],
            &mut bytes[2 * start..2 * (start + n)],
        )
        .unwrap();
        start += n;
    }
    bytes
}

/// Pushes `bytes` in pieces and collects the audio as `f64` (`T` is the audio sample type).
fn run_audio<T: sdrk::RealSample>(p: &mut Pipeline, bytes: &[u8], piece: usize) -> Vec<f64> {
    let mut audio = Vec::new();
    let mut buf = vec![T::default(); 4096];
    for part in bytes.chunks(piece) {
        p.push("rf", InputData::Raw(part)).unwrap();
        loop {
            let got = p.read_stream("audio", &mut buf).unwrap();
            if got == 0 {
                break;
            }
            audio.extend(buf[..got].iter().map(|v| v.to_f64()));
        }
    }
    audio
}

fn sinad_of(audio: &[f64]) -> f64 {
    sinad_db(&audio[audio.len() / 2..], 48_000.0, AUDIO_HZ)
}

#[test]
fn the_real_flow_receives_the_e2e_real_signal() {
    let samples = 480_000; // 0.1 s
    let bytes = rs16_recording(samples);
    let f32_audio = run_audio::<f32>(&mut build(&real_flow("f32").to_string()), &bytes, 8192);
    let q15_audio = run_audio::<sdrk::Q15>(&mut build(&real_flow("q15").to_string()), &bytes, 8192);
    for (precision, audio, min_db) in [("f32", f32_audio, 40.0), ("q15", q15_audio, 35.0)] {
        assert_eq!(audio.len(), samples / 100, "{precision}");
        let db = sinad_of(&audio);
        assert!(db >= min_db, "{precision}: SINAD {db} dB, need {min_db}");
    }
}

#[test]
fn the_real_flow_is_independent_of_the_push_size() {
    let samples = 96_000;
    let bytes = rs16_recording(samples);
    let doc = real_flow("q15").to_string();
    // 4096 samples (8192 bytes) is the largest push (`max_chunk`).
    let whole = run_audio::<sdrk::Q15>(&mut build(&doc), &bytes, 8192);
    // Odd byte counts split the 16-bit samples: the pipeline carries the partial packing group
    // of the real convert node over to the next push.
    for piece in [1, 2, 3, 999, 1000, 1002, 4097, 4098, 8191] {
        let got = run_audio::<sdrk::Q15>(&mut build(&doc), &bytes, piece);
        assert_eq!(got, whole, "piece {piece}");
    }
}

#[cfg(feature = "yaml")]
#[test]
fn the_yaml_twin_describes_the_same_flow() {
    let yaml = std::fs::read_to_string(common::workspace_file(
        "examples/flows/real_input_fm_receiver.yaml",
    ))
    .unwrap();
    let from_yaml = sdrk_pipeline::PipelineSpec::from_yaml(&yaml).unwrap();
    let from_json =
        sdrk_pipeline::PipelineSpec::from_json(&flow("real_input_fm_receiver")).unwrap();
    assert_eq!(from_yaml, from_json);
}
