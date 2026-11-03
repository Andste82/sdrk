//! The kinds `to_complex`, `real_part`, `imag_part`, `nco` and `stereo_matrix`: registration,
//! costs, type checks, and runs through the pipeline.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::large_stack_frames,
    clippy::cast_precision_loss
)]

mod common;

use sdrk::{C32, CQ15, Q15, Rate};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Registry};
use serde_json::{Value, json};

use common::build;

fn flow(kinds: &[(&str, Value)], ty: &str, precision: &str) -> Value {
    let mut nodes = serde_json::Map::new();
    let mut edges = Vec::new();
    let mut prev = "in".to_string();
    for (i, (kind, params)) in kinds.iter().enumerate() {
        let name = format!("n{i}");
        nodes.insert(name.clone(), json!({ "kind": kind, "params": params }));
        edges.push(json!({ "from": prev, "to": name }));
        prev = name;
    }
    edges.push(json!({ "from": prev, "to": "out" }));
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": precision, "max_chunk": 1024 },
        "inputs": { "in": { "type": ty, "rate_hz": 48000 } },
        "nodes": nodes, "edges": edges,
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
        ("to_complex", ["interleave"]),
        ("real_part", ["deinterleave"]),
        ("imag_part", ["deinterleave"]),
        ("nco", ["oscillate"]),
        ("stereo_matrix", ["stereo_matrix"]),
    ] {
        let f = registry.factory(kind).unwrap();
        assert_eq!(f.kernels(), kernels, "{kind}");
        assert_eq!(f.description().kind, kind);
    }
    // The blocks with two inputs wait for M34.
    for kind in ["add", "mul", "make_complex", "stereo_pll"] {
        assert!(registry.factory(kind).is_none(), "{kind}");
    }
}

#[test]
fn the_costs_follow_the_table() {
    for (kind, params, ty, precision, cost) in [
        ("to_complex", json!({}), "f32", "f32", 1.0),
        ("to_complex", json!({}), "q15", "q15", 1.0),
        ("real_part", json!({}), "c32", "f32", 1.0),
        ("imag_part", json!({}), "cq15", "q15", 1.0),
        ("nco", json!({ "frequency_hz": 1000.0 }), "f32", "f32", 6.0),
        ("nco", json!({ "frequency_hz": 1000.0 }), "cq15", "q15", 6.0),
        ("stereo_matrix", json!({}), "c32", "f32", 2.0),
        ("stereo_matrix", json!({}), "cq15", "q15", 2.0),
    ] {
        let report = validate(&flow(&[(kind, params)], ty, precision));
        assert!(report.is_ok(), "{kind}: {}", report.to_json());
        assert_eq!(report.cost_per_input_sample, cost, "{kind} {ty}");
        let node = &report.nodes[0];
        assert_eq!(node.rate_in, Rate::hz(48_000).unwrap());
        assert_eq!(node.rate_out, Rate::hz(48_000).unwrap());
    }
}

#[test]
fn types_and_parameters_are_checked() {
    let bad = |kind: &str, params: Value, ty: &str, precision: &str| {
        !validate(&flow(&[(kind, params)], ty, precision)).is_ok()
    };
    assert!(bad("to_complex", json!({}), "c32", "f32"));
    assert!(bad("real_part", json!({}), "f32", "f32"));
    assert!(bad("imag_part", json!({}), "q15", "q15"));
    assert!(bad("stereo_matrix", json!({}), "f32", "f32"));
    // frequency_hz is required and must be below half the rate.
    assert!(bad("nco", json!({}), "f32", "f32"));
    assert!(bad("nco", json!({ "frequency_hz": 24000.0 }), "f32", "f32"));
    assert!(bad(
        "nco",
        json!({ "frequency_hz": 1.0, "harmonic": 9 }),
        "f32",
        "f32"
    ));
    // A parameter on a block without parameters.
    let report = validate(&flow(&[("real_part", json!({ "x": 1 }))], "c32", "f32"));
    assert!(
        report
            .errors
            .iter()
            .any(|d| d.code == DiagCode::UnknownParam)
    );
}

#[test]
fn real_part_of_to_complex_returns_the_input() {
    let mut p = build(
        &flow(
            &[("to_complex", json!({})), ("real_part", json!({}))],
            "f32",
            "f32",
        )
        .to_string(),
    );
    let input: Vec<f32> = (0..500).map(|i| (i as f32 * 0.37).sin()).collect();
    p.push("in", InputData::F32(&input)).unwrap();
    let mut out = vec![0.0f32; 600];
    let n = p.read_stream("out", &mut out).unwrap();
    assert_eq!(n, 500);
    assert_eq!(&out[..n], &input[..]);

    let mut p = build(
        &flow(
            &[("to_complex", json!({})), ("imag_part", json!({}))],
            "q15",
            "q15",
        )
        .to_string(),
    );
    let input: Vec<Q15> = (0..300).map(|i| Q15(i * 50 - 7000)).collect();
    p.push("in", InputData::Q15(&input)).unwrap();
    let mut out = vec![Q15(1); 400];
    let n = p.read_stream("out", &mut out).unwrap();
    assert_eq!(n, 300);
    // The imaginary part of a real-to-complex conversion is zero.
    assert!(out[..n].iter().all(|v| v.0 == 0));
}

#[test]
fn nco_clocked_by_audio_makes_a_complex_tone() {
    let mut p =
        build(&flow(&[("nco", json!({ "frequency_hz": 1000.0 }))], "f32", "f32").to_string());
    let audio: Vec<f32> = (0..4800)
        .map(|i| 0.5 * (core::f32::consts::TAU * 1000.0 * i as f32 / 48000.0).sin())
        .collect();
    let mut out = vec![C32::default(); 5000];
    let mut n = 0;
    for part in audio.chunks(1000) {
        p.push("in", InputData::F32(part)).unwrap();
        n += p.read_stream("out", &mut out[n..]).unwrap();
    }
    assert_eq!(n, 4800);
    for (i, v) in out[..n].iter().enumerate() {
        let ph = core::f64::consts::TAU * 1000.0 * i as f64 / 48000.0;
        assert!((f64::from(v.re) - ph.cos()).abs() < 1e-3, "{i}");
        assert!((f64::from(v.im) - ph.sin()).abs() < 1e-3, "{i}");
    }
    // The runtime parameter changes the frequency and keeps the phase.
    p.set_param("n0", "frequency_hz", json!(2000.0)).unwrap();
    p.push("in", InputData::F32(&audio[..64])).unwrap();
    let m = p.read_stream("out", &mut out).unwrap();
    assert_eq!(m, 64);
}

#[test]
fn nco_runs_in_q15_on_a_complex_clock() {
    let mut p = build(
        &flow(
            &[("nco", json!({ "frequency_hz": 12000.0 }))],
            "cq15",
            "q15",
        )
        .to_string(),
    );
    let clock = vec![CQ15::default(); 8];
    p.push("in", InputData::CQ15(&clock)).unwrap();
    let mut out = vec![CQ15::default(); 16];
    let n = p.read_stream("out", &mut out).unwrap();
    assert_eq!(n, 8);
    // A quarter turn per sample: 1, j, -1, -j.
    assert!(out[0].re.0 > 32000 && out[0].im.0.abs() < 200);
    assert!(out[1].im.0 > 32000 && out[1].re.0.abs() < 200);
}

#[test]
fn stereo_matrix_passes_through_the_pipeline() {
    let mut p = build(&flow(&[("stereo_matrix", json!({}))], "c32", "f32").to_string());
    let input = vec![C32::new(0.5, 0.25); 100];
    p.push("in", InputData::C32(&input)).unwrap();
    let mut out = vec![C32::default(); 128];
    let n = p.read_stream("out", &mut out).unwrap();
    assert_eq!(n, 100);
    assert!(
        out[..n]
            .iter()
            .all(|v| v.re.is_finite() && v.im.is_finite())
    );
}
