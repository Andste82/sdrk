//! The seven primitive kinds (`quad_demod`, `deemphasis`, `rotate`, `decimate`, `interpolate`,
//! `mag`, `delay`): registration, plan figures and cost formulas, validation, the estimate
//! against the built pipeline, and the primitive FM receiver chain against `fm_receiver`.
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

use sdrk::{Delay, Rate};
use sdrk_blocks::{DecimateConfig, InterpolateConfig};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Registry};
use serde_json::{Value, json};

use common::{build, cu8_recording, flow, run_fm, same_audio};

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

fn validate(doc: &Value) -> sdrk_pipeline::Report {
    Registry::standard().validate_json(&doc.to_string(), &BuildOptions::default())
}

#[test]
fn the_kinds_are_registered_with_their_kernels() {
    let registry = Registry::standard();
    for (kind, kernels) in [
        ("quad_demod", &["quad_demod"][..]),
        ("deemphasis", &["deemphasis", "biquad"]),
        ("rotate", &["rotate"]),
        ("decimate", &["fir_decimate", "resample_real"]),
        ("interpolate", &["fir_interpolate", "fir_interpolate_real"]),
        ("mag", &["mag"]),
        ("delay", &[]),
    ] {
        let f = registry.factory(kind).unwrap();
        assert_eq!(f.kernels(), kernels, "{kind}");
        assert_eq!(f.description().kind, kind);
    }
    assert_eq!(registry.descriptions().count(), 65);
}

#[test]
fn fixed_costs_and_rates() {
    for (kind, ty, precision, params, cost, out_rate) in [
        ("quad_demod", "c32", "f32", json!({}), 12.0, 48_000),
        ("quad_demod", "cq15", "q15", json!({}), 12.0, 48_000),
        ("deemphasis", "f32", "f32", json!({}), 3.0, 48_000),
        ("deemphasis", "c32", "f32", json!({}), 10.0, 48_000),
        ("deemphasis", "cq15", "q15", json!({}), 10.0, 48_000),
        (
            "deemphasis",
            "q15",
            "q15",
            json!({ "tau_us": 75 }),
            3.0,
            48_000,
        ),
        (
            "rotate",
            "c32",
            "f32",
            json!({ "frequency_hz": 5000 }),
            6.0,
            48_000,
        ),
        (
            "rotate",
            "cq15",
            "q15",
            json!({ "frequency_hz": -5000 }),
            6.0,
            48_000,
        ),
        ("mag", "c32", "f32", json!({}), 4.0, 48_000),
        ("mag", "cq15", "q15", json!({}), 4.0, 48_000),
        (
            "delay",
            "c32",
            "f32",
            json!({ "samples": 100 }),
            1.0,
            48_000,
        ),
        ("delay", "q15", "q15", json!({ "samples": 0 }), 1.0, 48_000),
    ] {
        let report = validate(&single(kind, ty, precision, params));
        assert!(report.is_ok(), "{kind} {ty}: {}", report.to_json());
        assert_eq!(report.cost_per_input_sample, cost, "{kind} {ty}");
        assert_eq!(report.nodes[0].rate_out, Rate::hz(out_rate).unwrap());
    }
}

#[test]
fn delay_reports_its_samples() {
    for ty in ["c32", "f32"] {
        let report = validate(&single("delay", ty, "f32", json!({ "samples": 37 })));
        assert!(report.is_ok());
        assert_eq!(
            report.nodes[0].delay_samples_in.reduced(),
            Delay::new(37, 1).unwrap()
        );
    }
}

#[test]
fn decimate_and_interpolate_follow_the_cost_formulas() {
    for (ty, precision) in [
        ("c32", "f32"),
        ("cq15", "q15"),
        ("f32", "f32"),
        ("q15", "q15"),
    ] {
        let doc = single("decimate", ty, precision, json!({ "factor": 4 }));
        let report = validate(&doc);
        assert!(report.is_ok(), "{ty}: {}", report.to_json());
        let taps = DecimateConfig {
            factor: 4,
            ..DecimateConfig::default()
        }
        .num_taps(Rate::hz(48_000).unwrap())
        .unwrap();
        assert!(taps > 1 && taps % 2 == 1, "{taps}");
        assert_eq!(report.cost_per_input_sample, taps as f64 / 4.0);
        assert_eq!(report.nodes[0].rate_out, Rate::hz(12_000).unwrap());
        assert_eq!(
            report.nodes[0].delay_samples_in.reduced(),
            Delay::new(taps as u64 - 1, 2).unwrap().reduced()
        );

        let doc = single("interpolate", ty, precision, json!({ "factor": 3 }));
        let report = validate(&doc);
        assert!(report.is_ok(), "{ty}: {}", report.to_json());
        let taps = InterpolateConfig {
            factor: 3,
            ..InterpolateConfig::default()
        }
        .num_taps(Rate::hz(48_000).unwrap())
        .unwrap();
        assert_eq!(report.cost_per_input_sample, taps as f64);
        assert_eq!(report.nodes[0].rate_out, Rate::hz(144_000).unwrap());
        assert_eq!(
            report.nodes[0].delay_samples_in.reduced(),
            Delay::new(taps as u64 - 1, 6).unwrap().reduced()
        );
    }
}

#[test]
fn the_estimate_matches_the_built_pipeline() {
    let registry = Registry::standard();
    for (kind, ty, precision, params) in [
        ("decimate", "q15", "q15", json!({ "factor": 4 })),
        ("interpolate", "c32", "f32", json!({ "factor": 3 })),
        ("delay", "cq15", "q15", json!({ "samples": 500 })),
        ("rotate", "c32", "f32", json!({ "frequency_hz": 1000 })),
    ] {
        let doc = single(kind, ty, precision, params);
        let report = validate(&doc);
        assert!(report.is_ok(), "{kind}: {}", report.to_json());
        let built = registry
            .build_json(&doc.to_string(), &BuildOptions::default())
            .unwrap();
        let built = built.report();
        assert!(
            built.memory_bytes.abs_diff(report.memory_bytes) <= report.memory_bytes / 20,
            "{kind}"
        );
        assert_eq!(built.cost_per_input_sample, report.cost_per_input_sample);
    }
}

#[test]
fn types_and_parameters_are_checked() {
    let invalid = |kind: &str, ty: &str, precision: &str, params: Value| {
        !validate(&single(kind, ty, precision, params)).is_ok()
    };
    // Complex-only and real-only kinds.
    assert!(invalid("quad_demod", "f32", "f32", json!({})));
    assert!(invalid("mag", "q15", "q15", json!({})));
    assert!(invalid(
        "rotate",
        "f32",
        "f32",
        json!({ "frequency_hz": 100 })
    ));
    // Complex samples (stereo audio) are accepted since I34-1 (D396); raw bytes are not.
    assert!(!invalid("deemphasis", "c32", "f32", json!({})));
    assert!(!invalid("deemphasis", "cq15", "q15", json!({})));
    // Required parameters and ranges.
    assert!(invalid("rotate", "c32", "f32", json!({})));
    assert!(invalid(
        "rotate",
        "c32",
        "f32",
        json!({ "frequency_hz": 24000 })
    ));
    assert!(invalid("decimate", "c32", "f32", json!({})));
    assert!(invalid("decimate", "c32", "f32", json!({ "factor": 0 })));
    assert!(invalid("interpolate", "c32", "f32", json!({})));
    assert!(invalid("delay", "c32", "f32", json!({})));
    assert!(invalid("delay", "c32", "f32", json!({ "samples": 65537 })));
    assert!(invalid("deemphasis", "f32", "f32", json!({ "tau_us": 0 })));
    assert!(invalid(
        "quad_demod",
        "c32",
        "f32",
        json!({ "deviation_hz": 0 })
    ));
    // A stop band above the output Nyquist frequency is a rate problem.
    let report = validate(&single(
        "decimate",
        "c32",
        "f32",
        json!({ "factor": 4, "cutoff_hz": 5900, "transition_hz": 1000 }),
    ));
    assert!(!report.is_ok());
    assert!(
        report
            .errors
            .iter()
            .any(|d| d.code == DiagCode::RateUnsupported),
        "{}",
        report.to_json()
    );
}

/// The flow `rf (cu8) -> rotate -> decimate -> quad_demod -> deemphasis -> decimate (real)`.
fn primitive_receiver() -> Value {
    json!({
        "sdrk": 1, "name": "primitive-fm-receiver",
        "settings": { "precision": "f32", "max_chunk": 4096 },
        "inputs": { "rf": { "format": "cu8", "rate_hz": 2400000 } },
        "nodes": {
            "shift": { "kind": "rotate", "params": { "frequency_hz": -600000 } },
            "chan": { "kind": "decimate", "params": { "factor": 10, "cutoff_hz": 100000 } },
            "demod": { "kind": "quad_demod", "params": { "deviation_hz": 75000 } },
            "deemph": { "kind": "deemphasis", "params": { "tau_us": 50 } },
            "audio_lp": { "kind": "decimate", "params": {
                "factor": 5, "cutoff_hz": 15000, "transition_hz": 3750, "attenuation_db": 60 } }
        },
        "edges": [
            { "from": "rf", "to": "shift" }, { "from": "shift", "to": "chan" },
            { "from": "chan", "to": "demod" }, { "from": "demod", "to": "deemph" },
            { "from": "deemph", "to": "audio_lp" }, { "from": "audio_lp", "to": "audio" }
        ],
        "outputs": { "audio": { "type": "stream", "capacity": 16384, "drop": "error" } }
    })
}

#[test]
fn the_primitive_chain_equals_fm_receiver() {
    let doc = primitive_receiver();
    let report = validate(&doc);
    assert!(report.is_ok(), "{}", report.to_json());
    let bytes = cu8_recording(240_000);
    let mut primitive = build(&doc.to_string());
    let mut got = Vec::new();
    let mut buf = vec![0f32; 4096];
    for piece in bytes.chunks(8_000) {
        primitive.push("rf", InputData::Raw(piece)).unwrap();
        loop {
            let n = primitive.read_stream("audio", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            got.extend_from_slice(&buf[..n]);
        }
    }
    let mut reference = build(&flow("fm_receiver"));
    let (want, _) = run_fm(&mut reference, &bytes, 8_000);
    assert!(!want.is_empty());
    assert!(same_audio(&want, &got), "{} / {}", want.len(), got.len());
}

#[test]
fn rotate_frequency_changes_at_run_time_with_continuous_phase() {
    use sdrk::C32;
    let mut p = build(&single("rotate", "c32", "f32", json!({ "frequency_hz": 4800 })).to_string());
    let ones = vec![C32::new(1.0, 0.0); 100];
    let mut out = vec![C32::new(0.0, 0.0); 256];
    let mut got = Vec::new();
    for (i, hz) in [4800.0, -2400.0].into_iter().enumerate() {
        if i > 0 {
            p.set_param("n", "frequency_hz", json!(hz)).unwrap();
        }
        p.push("in", InputData::C32(&ones)).unwrap();
        loop {
            let n = p.read_stream("out", &mut out).unwrap();
            if n == 0 {
                break;
            }
            got.extend_from_slice(&out[..n]);
        }
    }
    assert_eq!(got.len(), 200);
    let step = |a: C32, b: C32| (b * a.conj()).im.atan2((b * a.conj()).re);
    let w1 = 2.0 * core::f32::consts::PI * 4800.0 / 48000.0;
    let w2 = -2.0 * core::f32::consts::PI * 2400.0 / 48000.0;
    for k in 1..200 {
        let d = step(got[k - 1], got[k]);
        let want = if k < 100 { w1 } else { w2 };
        // The step across the switch is either the old or the new increment.
        if k == 100 {
            assert!(
                (d - w1).abs() < 1e-3 || (d - w2).abs() < 1e-3,
                "switch step {d}"
            );
        } else {
            assert!((d - want).abs() < 1e-3, "step {k}: {d} vs {want}");
        }
    }
}

#[test]
fn the_seven_kinds_build_into_send_pipelines() {
    fn assert_send<T: Send>() {}
    assert_send::<sdrk_pipeline::Pipeline>();
    assert_send::<Registry>();
}
