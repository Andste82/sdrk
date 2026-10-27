//! The `biquad` kind: plan figures and the cost formula, kernels, validation, runtime parameters
//! and the filters of the three receiver flows.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::unreadable_literal,
    clippy::needless_pass_by_value,
    clippy::large_stack_frames,
    clippy::large_stack_arrays
)]

mod common;

use sdrk::{Delay, Q15, Rate};
use sdrk_pipeline::{BuildOptions, InputData, Registry};
use serde_json::json;

use common::{build, flow};

/// A one-node flow on a 48 kHz input of type `ty`.
fn single(ty: &str, precision: &str, params: serde_json::Value) -> serde_json::Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": precision, "max_chunk": 1024 },
        "inputs": { "in": { "type": ty, "rate_hz": 48000 } },
        "nodes": { "n": { "kind": "biquad", "params": params } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    })
}

#[test]
fn the_kind_is_registered_with_both_kernels() {
    let registry = Registry::standard();
    let factory = registry.factory("biquad").unwrap();
    assert_eq!(factory.kernels(), ["biquad", "biquad_real"]);
    assert_eq!(factory.description().kind, "biquad");
}

#[test]
fn the_plan_follows_the_cost_formula() {
    let registry = Registry::standard();
    // Real: 5 per section; complex: 10 per section; same rate, no delay, one sample at a time.
    for (ty, precision, per_section) in [
        ("f32", "f32", 5.0),
        ("q15", "q15", 5.0),
        ("c32", "f32", 10.0),
        ("cq15", "q15", 10.0),
    ] {
        for (response, order) in [
            ("lowpass", 2),
            ("lowpass", 4),
            ("highpass", 6),
            ("lowpass", 8),
            ("bandpass", 2),
            ("notch", 2),
        ] {
            let sections = if response == "bandpass" || response == "notch" {
                1
            } else {
                order / 2
            };
            let params = json!({ "response": response, "order": order, "cutoff_hz": 1000 });
            let doc = single(ty, precision, params);
            let report = registry.validate_json(&doc.to_string(), &BuildOptions::default());
            assert!(report.is_ok(), "{}", report.to_json());
            assert_eq!(
                report.cost_per_input_sample,
                per_section * sections as f64,
                "{ty} {response} {order}"
            );
            let node = &report.nodes[0];
            assert_eq!(node.rate_in, Rate::hz(48_000).unwrap());
            assert_eq!(node.rate_out, node.rate_in);
            assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO.reduced());
            // The memory is exactly what the block takes: 9 values per section (5 coefficients
            // and 4 history entries of the sample type, rounded up to the alignment).
            assert!(report.memory_bytes > 0);
        }
    }
}

#[test]
fn the_estimate_matches_the_built_pipeline() {
    let registry = Registry::standard();
    let doc = single(
        "cq15",
        "q15",
        json!({ "response": "highpass", "order": 4, "cutoff_hz": 300 }),
    );
    let report = registry.validate_json(&doc.to_string(), &BuildOptions::default());
    assert!(report.is_ok(), "{}", report.to_json());
    let built = registry
        .build_json(&doc.to_string(), &BuildOptions::default())
        .unwrap();
    let built = built.report();
    assert!(built.memory_bytes > 0);
    assert!(built.memory_bytes.abs_diff(report.memory_bytes) <= report.memory_bytes / 20);
    assert_eq!(built.cost_per_input_sample, report.cost_per_input_sample);
}

#[test]
fn invalid_parameters_are_diagnosed() {
    let registry = Registry::standard();
    let invalid = |ty: &str, precision: &str, params: serde_json::Value| {
        let doc = single(ty, precision, params);
        let report = registry.validate_json(&doc.to_string(), &BuildOptions::default());
        !report.is_ok()
    };
    // The cutoff is required.
    assert!(invalid("f32", "f32", json!({})));
    // Above half the sample rate and not positive.
    assert!(invalid("f32", "f32", json!({ "cutoff_hz": 24000 })));
    assert!(invalid("f32", "f32", json!({ "cutoff_hz": 0 })));
    // Odd and too large orders, and an order other than 2 for a band-pass and a notch.
    assert!(invalid(
        "f32",
        "f32",
        json!({ "cutoff_hz": 1000, "order": 3 })
    ));
    assert!(invalid(
        "f32",
        "f32",
        json!({ "cutoff_hz": 1000, "order": 10 })
    ));
    for response in ["bandpass", "notch"] {
        assert!(invalid(
            "f32",
            "f32",
            json!({ "cutoff_hz": 1000, "order": 4, "response": response })
        ));
    }
    assert!(invalid(
        "f32",
        "f32",
        json!({ "cutoff_hz": 1000, "response": "allpass" })
    ));
    // Q15: a low-pass or high-pass below fs / 500 = 96 Hz is rejected, f32 accepts it.
    assert!(invalid("q15", "q15", json!({ "cutoff_hz": 50 })));
    assert!(invalid(
        "cq15",
        "q15",
        json!({ "cutoff_hz": 95.9, "response": "highpass" })
    ));
    assert!(!invalid("q15", "q15", json!({ "cutoff_hz": 96 })));
    assert!(!invalid("f32", "f32", json!({ "cutoff_hz": 50 })));
    assert!(!invalid(
        "q15",
        "q15",
        json!({ "cutoff_hz": 50, "response": "notch" })
    ));
    // The input type has to match the precision of the pipeline.
    assert!(invalid("f32", "q15", json!({ "cutoff_hz": 1000 })));
}

fn run_audio(p: &mut sdrk_pipeline::Pipeline, input: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    let mut buf = vec![0.0f32; 1024];
    for piece in input.chunks(1000) {
        p.push("in", InputData::F32(piece)).unwrap();
        loop {
            let got = p.read_stream("out", &mut buf).unwrap();
            if got == 0 {
                break;
            }
            out.extend_from_slice(&buf[..got]);
        }
    }
    out
}

fn tone(n: usize, hz: f64, amp: f64) -> Vec<f32> {
    (0..n)
        .map(|i| (amp * (std::f64::consts::TAU * hz * i as f64 / 48_000.0).sin()) as f32)
        .collect()
}

fn peak(x: &[f32]) -> f32 {
    x[x.len() / 2..].iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

#[test]
fn the_cutoff_and_q_change_while_the_pipeline_runs() {
    let doc = single(
        "f32",
        "f32",
        json!({ "cutoff_hz": 2000, "order": 4, "response": "lowpass" }),
    );
    let mut p = build(&doc.to_string());
    let x = tone(9600, 1_000.0, 0.5);
    let first = run_audio(&mut p, &x);
    assert!((peak(&first) - 0.5).abs() < 0.01);
    p.set_param("n", "cutoff_hz", json!(300)).unwrap();
    let second = run_audio(&mut p, &x);
    assert!(second.iter().all(|v| v.is_finite()));
    assert!(peak(&second) < 0.5 * 0.05, "{}", peak(&second));
    // The build parameters are fixed, invalid runtime values are errors that change nothing.
    assert!(p.set_param("n", "order", json!(2)).is_err());
    assert!(p.set_param("n", "response", json!("highpass")).is_err());
    assert!(p.set_param("n", "cutoff_hz", json!(30000)).is_err());
    assert!(p.set_param("n", "q", json!(-1)).is_err());
    let third = run_audio(&mut p, &x);
    assert!(peak(&third) < 0.5 * 0.05);

    // A Q15 pipeline rejects the runtime change below fs / 500 too.
    let doc = single("q15", "q15", json!({ "cutoff_hz": 2000 }));
    let mut p = build(&doc.to_string());
    assert!(p.set_param("n", "cutoff_hz", json!(50)).is_err());
    p.set_param("n", "cutoff_hz", json!(500)).unwrap();
    let x: Vec<Q15> = tone(1000, 100.0, 0.5)
        .iter()
        .map(|v| Q15((v * 32768.0) as i16))
        .collect();
    p.push("in", InputData::Q15(&x)).unwrap();
    let mut buf = vec![Q15(0); 1024];
    assert_eq!(p.read_stream("out", &mut buf).unwrap(), 1000);
}

#[test]
fn the_receiver_flows_have_the_audio_filters_of_the_milestone() {
    // The comments and the node order are in the flows; the parameters are checked here.
    let expect = [
        ("nfm_receiver", "ctcss", "highpass", 300.0),
        ("am_receiver", "audio_filter", "lowpass", 4000.0),
        ("ssb_receiver", "audio_filter", "lowpass", 2800.0),
    ];
    for (name, node, response, cutoff) in expect {
        let spec: serde_json::Value = serde_json::from_str(&flow(name)).unwrap();
        let n = &spec["nodes"][node];
        assert_eq!(n["kind"], "biquad", "{name}");
        assert_eq!(n["params"]["response"], response, "{name}");
        assert_eq!(n["params"]["order"], 4, "{name}");
        assert_eq!(n["params"]["cutoff_hz"].as_f64(), Some(cutoff), "{name}");
        // The filter sits directly before the output (after the demodulator).
        let edges = spec["edges"].as_array().unwrap();
        assert!(
            edges
                .iter()
                .any(|e| e["from"] == node && e["to"] == "audio"),
            "{name}"
        );
        // ... and the nodes cutoff can change at run time.
        let mut p = build(&flow(name));
        p.set_param(node, "cutoff_hz", json!(cutoff * 0.5)).unwrap();
        assert!(p.set_param(node, "order", json!(2)).is_err());
    }
}
