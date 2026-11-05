//! The kinds `noise_blank` and `auto_notch` (M33): registration, costs, delays, type checks,
//! runtime parameters, and the example flow
//! `hf_am_receiver` (validates, builds and runs in push mode).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::needless_pass_by_value,
    clippy::large_stack_frames
)]

mod common;

use num_complex::Complex64;
use sdrk::{Delay, Rate};
use sdrk_device::formats;
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Registry};
use sdrk_testkit::{Pcg32, encode::encode};
use serde_json::{Value, json};

use common::{build, flow};

/// A one-node flow on an input of type `ty` at `rate` Hz.
fn single(kind: &str, params: Value, ty: &str, precision: &str, rate: u64) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": precision, "max_chunk": 1024 },
        "inputs": { "in": { "type": ty, "rate_hz": rate } },
        "nodes": { "n": { "kind": kind, "params": params } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    })
}

fn validate(doc: &Value) -> sdrk_pipeline::Report {
    Registry::standard().validate_json(&doc.to_string(), &BuildOptions::default())
}

#[test]
fn the_kinds_are_registered_with_their_kernels() {
    let registry = Registry::standard();
    for kind in ["noise_blank", "auto_notch"] {
        let f = registry.factory(kind).unwrap();
        assert_eq!(f.kernels(), [kind]);
        assert_eq!(f.description().kind, kind);
    }
}

#[test]
fn costs_and_delays_follow_the_table() {
    // noise_blank: cost 8, delay D = round(lookahead_s fs).
    for (ty, precision) in [("c32", "f32"), ("cq15", "q15")] {
        for (params, d) in [
            (json!({}), 10),
            (json!({ "lookahead_s": 0.0, "hold_s": 0.0 }), 0),
            (json!({ "lookahead_s": 0.001 }), 192),
        ] {
            let report = validate(&single(
                "noise_blank",
                params.clone(),
                ty,
                precision,
                192_000,
            ));
            assert!(report.is_ok(), "{params}: {}", report.to_json());
            assert_eq!(report.cost_per_input_sample, 8.0);
            let node = &report.nodes[0];
            assert_eq!(node.rate_out, node.rate_in);
            assert_eq!(
                node.delay_samples_in.reduced(),
                Delay::new(d, 1).unwrap().reduced()
            );
        }
    }
    // auto_notch: cost 2 taps + 4, no delay.
    for (params, taps) in [
        (json!({}), 32),
        (json!({ "taps": 8, "delay": 1 }), 8),
        (json!({ "taps": 256, "delay": 64, "mu": 0.0001 }), 256),
    ] {
        let report = validate(&single("auto_notch", params.clone(), "f32", "f32", 12_000));
        assert!(report.is_ok(), "{params}: {}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 2.0 * taps as f64 + 4.0);
        let node = &report.nodes[0];
        assert_eq!(node.rate_out, node.rate_in);
        assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO.reduced());
    }
}

#[test]
fn types_and_parameters_are_checked() {
    let code = |doc: &Value| validate(doc).errors.first().map(|d| d.code);
    // auto_notch takes f32 only: Q15 and complex input are a type mismatch.
    assert_eq!(
        code(&single("auto_notch", json!({}), "q15", "q15", 12_000)),
        Some(DiagCode::TypeMismatch)
    );
    assert_eq!(
        code(&single("auto_notch", json!({}), "c32", "f32", 12_000)),
        Some(DiagCode::TypeMismatch)
    );
    // noise_blank takes complex input only.
    for (ty, precision) in [("f32", "f32"), ("q15", "q15")] {
        assert_eq!(
            code(&single("noise_blank", json!({}), ty, precision, 48_000)),
            Some(DiagCode::TypeMismatch)
        );
    }
    // Ranges.
    for (kind, params, ty) in [
        ("noise_blank", json!({ "threshold_db": 5 }), "c32"),
        ("noise_blank", json!({ "lookahead_s": 0.002 }), "c32"),
        ("noise_blank", json!({ "hold_s": 0.02 }), "c32"),
        ("noise_blank", json!({ "tau_s": 2 }), "c32"),
        ("auto_notch", json!({ "taps": 4 }), "f32"),
        ("auto_notch", json!({ "delay": 65 }), "f32"),
        ("auto_notch", json!({ "mu": 0.5 }), "f32"),
    ] {
        assert!(
            !validate(&single(kind, params.clone(), ty, "f32", 48_000)).is_ok(),
            "{kind} {params}"
        );
    }
    // A time constant shorter than 1.45 samples is rejected for the rate.
    assert!(
        !validate(&single(
            "noise_blank",
            json!({ "tau_s": 0.001 }),
            "c32",
            "f32",
            1_000
        ))
        .is_ok()
    );
}

/// The runtime parameters (`threshold_db`, `mu`) can be set on a running pipeline; the build
/// parameters cannot.
#[test]
fn runtime_parameters_can_be_set_on_the_pipeline() {
    let mut p = build(&flow("hf_am_receiver"));
    p.set_param("nb", "threshold_db", json!(20.0)).unwrap();
    p.set_param("notch", "mu", json!(0.002)).unwrap();
    assert!(p.set_param("nb", "threshold_db", json!(50.0)).is_err());
    assert!(p.set_param("nb", "hold_s", json!(0.001)).is_err());
    assert!(p.set_param("notch", "taps", json!(16)).is_err());
}

#[test]
fn the_hf_am_receiver_validates_without_warnings() {
    let report =
        Registry::standard().validate_json(&flow("hf_am_receiver"), &BuildOptions::default());
    assert!(report.is_ok(), "{}", report.to_json());
    assert!(report.warnings.is_empty(), "{}", report.to_json());
    let node = |name: &str| report.nodes.iter().find(|n| n.name == name).unwrap();
    assert_eq!(node("nb").rate_in, Rate::hz(192_000).unwrap());
    assert_eq!(node("notch").rate_in, Rate::hz(12_000).unwrap());
    assert_eq!(node("notch").rate_out, Rate::hz(12_000).unwrap());
}

/// Pushes 0.5 s of an AM station (1 kHz tone, index 0.5, carrier 0.04, at the tuner centre) with
/// impulses of 20 times the carrier as `cs16` in pieces and reads the audio: 12 kHz, finite,
/// with the tone, and the blanker has blanked.
#[test]
fn the_hf_am_receiver_runs_in_push_mode() {
    let fs = 192_000.0;
    let n = 96_000;
    let mut rng = Pcg32::new(476, 9);
    let mut x: Vec<Complex64> = (0..n)
        .map(|k| {
            let env = 0.04 * (1.0 + 0.5 * (std::f64::consts::TAU * 1000.0 * k as f64 / fs).cos());
            Complex64::new(env + 0.0008 * rng.normal(), 0.0008 * rng.normal())
        })
        .collect();
    for start in (0..n).step_by(1920) {
        for v in &mut x[start + 700..start + 703] {
            *v = Complex64::new(0.0, 0.8);
        }
    }
    let mut bytes = vec![0u8; formats::CS16.bytes_for_samples(n).unwrap()];
    encode(&formats::CS16, &x, &mut bytes).unwrap();
    let mut p = build(&flow("hf_am_receiver"));
    let mut audio = Vec::new();
    let mut buf = vec![0f32; 4096];
    for piece in bytes.chunks(4 * 1000) {
        p.push("rf", InputData::Raw(piece)).unwrap();
        loop {
            let got = p.read_stream("audio", &mut buf).unwrap();
            if got == 0 {
                break;
            }
            audio.extend_from_slice(&buf[..got]);
        }
    }
    // 96000 samples at 192 kHz decimated by 16 (less what the filters still hold).
    assert!(
        audio.len() > 5_900 && audio.len() <= 6_000,
        "{}",
        audio.len()
    );
    assert!(audio.iter().all(|v| v.is_finite()));
    let tail = &audio[3_000..];
    let peak = tail.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(peak > 0.01, "the audio is silent: peak {peak}");
}
