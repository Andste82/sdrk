//! The `dc_block` kind (M32): the complex variant for `c32`/`cq15` input and the real variant
//! (`DcBlockRealBlock`, kernel `dc_block_real`) for `f32`/`q15` input, plan figures, validation
//! and the output of a real flow against the block.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::redundant_closure
)]

mod common;

use sdrk::{Delay, Q15, Rate};
use sdrk_blocks::{Block, DcBlockConfig, DcBlockRealBlock};
use sdrk_pipeline::{BuildOptions, InputData, Registry};
use sdrk_testkit::VecStorage;
use serde_json::json;

use common::{build, same_audio};

/// A one-node `dc_block` flow on a 48 kHz input of type `ty`.
fn single(ty: &str, precision: &str, cutoff_hz: f64) -> serde_json::Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": precision, "max_chunk": 1024 },
        "inputs": { "in": { "type": ty, "rate_hz": 48000 } },
        "nodes": { "n": { "kind": "dc_block", "params": { "cutoff_hz": cutoff_hz } } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    })
}

/// A tone on a DC offset.
fn offset_tone(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.3 + 0.4 * (std::f64::consts::TAU * 1_000.0 * i as f64 / 48_000.0).sin()) as f32)
        .collect()
}

/// Pushes `input` in pieces of 1000 samples and reads the stream output.
fn run<T: sdrk::Sample + Default>(
    p: &mut sdrk_pipeline::Pipeline,
    input: &[T],
    data: impl Fn(&[T]) -> InputData<'_>,
) -> Vec<T> {
    let mut out = Vec::new();
    let mut buf = vec![T::default(); 1024];
    for piece in input.chunks(1000) {
        p.push("in", data(piece)).unwrap();
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

#[test]
fn the_kind_is_registered_with_both_kernels() {
    let registry = Registry::standard();
    let factory = registry.factory("dc_block").unwrap();
    assert_eq!(factory.kernels(), ["dc_block", "dc_block_real"]);
    assert_eq!(factory.description().kind, "dc_block");
}

#[test]
fn every_input_type_costs_three_at_the_input_rate() {
    let registry = Registry::standard();
    for (ty, precision) in [
        ("c32", "f32"),
        ("cq15", "q15"),
        ("f32", "f32"),
        ("q15", "q15"),
    ] {
        let report = registry.validate_json(
            &single(ty, precision, 10.0).to_string(),
            &BuildOptions::default(),
        );
        assert!(report.is_ok(), "{ty}: {}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 3.0, "{ty}");
        let node = &report.nodes[0];
        assert_eq!(node.rate_in, Rate::hz(48_000).unwrap());
        assert_eq!(node.rate_out, node.rate_in);
        assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO.reduced());
    }
}

#[test]
fn a_real_input_is_validated_like_a_complex_one() {
    let registry = Registry::standard();
    let ok = |ty: &str, precision: &str, cutoff_hz: f64| {
        registry
            .validate_json(
                &single(ty, precision, cutoff_hz).to_string(),
                &BuildOptions::default(),
            )
            .is_ok()
    };
    for (ty, precision) in [("f32", "f32"), ("q15", "q15")] {
        assert!(ok(ty, precision, 23_999.0), "{ty}");
        assert!(!ok(ty, precision, 24_000.0), "{ty}");
        assert!(!ok(ty, precision, 0.0), "{ty}");
    }
    // The input type has to match the precision of the pipeline.
    assert!(!ok("f32", "q15", 10.0));
}

#[test]
fn a_real_flow_runs_the_real_block() {
    let cfg = DcBlockConfig { cutoff_hz: 50.0 };
    let rate = Rate::hz(48_000).unwrap();
    let x = offset_tone(9_600);

    // f32: the SIMD `dc_block_real` is a tolerance kernel whose bits may depend on the split.
    let mut p = build(&single("f32", "f32", 50.0).to_string());
    let got = run(&mut p, &x, |s: &[f32]| InputData::F32(s));
    let mut block = DcBlockRealBlock::<f32, _>::new(&cfg, rate, &mut VecStorage).unwrap();
    let mut want = vec![0.0f32; x.len()];
    block.process(&x, &mut want).unwrap();
    assert!(same_audio(&want, &got));
    // The offset is gone at the end, the tone passes.
    let tail = &got[got.len() - 960..];
    let mean = tail.iter().map(|v| f64::from(*v)).sum::<f64>() / tail.len() as f64;
    assert!(mean.abs() < 0.01, "mean {mean}");

    // Q15: bit for bit.
    let xq: Vec<Q15> = x.iter().map(|v| Q15((v * 32768.0) as i16)).collect();
    let mut p = build(&single("q15", "q15", 50.0).to_string());
    let got = run(&mut p, &xq, |s: &[Q15]| InputData::Q15(s));
    let mut block = DcBlockRealBlock::<Q15, _>::new(&cfg, rate, &mut VecStorage).unwrap();
    let mut want = vec![Q15(0); xq.len()];
    block.process(&xq, &mut want).unwrap();
    assert_eq!(want, got);
}
