//! The kind `noise` (M48 §6): registration with the kernel `gauss_noise`, the costs 12 per real
//! component (24 complex, four times `nco` as measured), the rates and delay 0, the output type equal to the input type,
//! `out_of_range` and `wrong_type` at the parameter, the schema, and push-mode flows (an `nco`
//! clock into `noise`, a real input straight into `noise`, a runtime change of `sigma`) that
//! equal the blocks bit for bit in pieces of 1 to 1000 samples (seed 689).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::needless_pass_by_value,
    clippy::large_stack_frames,
    clippy::many_single_char_names
)]

mod common;

use sdrk::{C32, CQ15, Delay, Q15, Rate};
use sdrk_blocks::{
    Block, NcoConfig, NcoRealBlock, NoiseBlock, NoiseConfig, NoiseRealBlock, ParamValue,
};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Registry, VecStorage};
use sdrk_testkit::Pcg32;
use serde_json::{Value, json};

use common::build;

/// The seed of the generator and of the pieces.
const SEED: u64 = 689;

/// A one-node flow of `kind` on an input of type `ty` at `rate` Hz.
fn single(kind: &str, ty: &str, precision: &str, rate: u64, params: Value) -> Value {
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

fn first_error(doc: &Value) -> (String, DiagCode) {
    let report = validate(doc);
    let e = report.errors.first().expect("an error");
    (e.path.clone(), e.code)
}

#[test]
fn the_kind_is_registered_with_its_kernel() {
    let registry = Registry::standard();
    let f = registry.factory("noise").unwrap();
    assert_eq!(f.kernels(), ["gauss_noise"]);
    assert_eq!(f.description().kind, "noise");
    assert_eq!(registry.descriptions().count(), 85);
    for kernel in f.kernels() {
        for ty in [
            sdrk::SampleType::C32,
            sdrk::SampleType::CQ15,
            sdrk::SampleType::F32,
            sdrk::SampleType::Q15,
        ] {
            // The level that runs the kernel is pinned by the facade test; here it only has
            // to be a real implementation, the same for every type.
            let level = sdrk::backend_for(kernel, ty);
            assert!(matches!(level, "simd" | "fallback"), "{ty:?}: {level}");
            assert_eq!(level, sdrk::backend_for(kernel, sdrk::SampleType::C32));
        }
    }
}

#[test]
fn the_cost_is_twelve_per_real_component() {
    for (ty, precision, cost) in [
        ("c32", "f32", 24.0),
        ("cq15", "q15", 24.0),
        ("f32", "f32", 12.0),
        ("q15", "q15", 12.0),
    ] {
        let report = validate(&single(
            "noise",
            ty,
            precision,
            48_000,
            json!({ "sigma": 0.5 }),
        ));
        assert!(report.is_ok(), "{ty}: {}", report.to_json());
        assert_eq!(report.cost_per_input_sample, cost, "{ty}");
        let node = &report.nodes[0];
        assert_eq!(node.rate_out, Rate::hz(48_000).unwrap(), "{ty}");
        assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO, "{ty}");
    }
    // Every parameter has a default.
    assert!(validate(&single("noise", "c32", "f32", 8_000, json!({}))).is_ok());
}

#[test]
fn invalid_parameters_are_errors_at_their_parameter() {
    for (params, path, code) in [
        (
            json!({ "sigma": -1.0 }),
            "nodes.n.params.sigma",
            DiagCode::OutOfRange,
        ),
        (
            json!({ "sigma": -0.001 }),
            "nodes.n.params.sigma",
            DiagCode::OutOfRange,
        ),
        (
            json!({ "sigma": 16.5 }),
            "nodes.n.params.sigma",
            DiagCode::OutOfRange,
        ),
        (
            json!({ "sigma": 17 }),
            "nodes.n.params.sigma",
            DiagCode::OutOfRange,
        ),
        (
            json!({ "sigma": "loud" }),
            "nodes.n.params.sigma",
            DiagCode::WrongType,
        ),
        (
            json!({ "seed": -1 }),
            "nodes.n.params.seed",
            DiagCode::OutOfRange,
        ),
        (
            json!({ "seed": 9_007_199_254_740_992_u64 }),
            "nodes.n.params.seed",
            DiagCode::OutOfRange,
        ),
        (
            json!({ "seed": "1" }),
            "nodes.n.params.seed",
            DiagCode::WrongType,
        ),
        (
            json!({ "stream": -1 }),
            "nodes.n.params.stream",
            DiagCode::OutOfRange,
        ),
        (
            json!({ "stream": 9_007_199_254_740_992_u64 }),
            "nodes.n.params.stream",
            DiagCode::OutOfRange,
        ),
    ] {
        let (p, c) = first_error(&single("noise", "c32", "f32", 48_000, params.clone()));
        assert_eq!((p.as_str(), c), (path, code), "{params}");
    }
    // The limits themselves are valid.
    assert!(
        validate(&single(
            "noise",
            "q15",
            "q15",
            48_000,
            json!({ "sigma": 16, "seed": 9_007_199_254_740_991_u64, "stream": 0 }),
        ))
        .is_ok()
    );
    assert!(
        validate(&single(
            "noise",
            "f32",
            "f32",
            48_000,
            json!({ "sigma": 0 })
        ))
        .is_ok()
    );
}

#[test]
fn the_schema_has_the_kind() {
    let schema = Registry::standard().json_schema_text();
    assert!(schema.contains("\"const\": \"noise\""));
}

/// Reads all `C32` output after each push of a random piece (1 to 1000 samples).
fn push_c32(p: &mut sdrk_pipeline::Pipeline, input: &[f32], rng: &mut Pcg32) -> Vec<C32> {
    let mut got = Vec::new();
    let mut buf = vec![C32::default(); 4096];
    let mut rest = input;
    while !rest.is_empty() {
        let n = (1 + rng.next_u32() as usize % 1000).min(rest.len());
        p.push("in", InputData::F32(&rest[..n])).unwrap();
        rest = &rest[n..];
        let k = p.read_stream::<C32>("out", &mut buf).unwrap();
        got.extend_from_slice(&buf[..k]);
    }
    got
}

fn same_bits_c32(got: &[C32], want: &[C32]) {
    assert_eq!(got.len(), want.len());
    assert!(
        got.iter()
            .zip(want)
            .all(|(a, b)| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()),
        "pipeline differs from the blocks"
    );
}

#[test]
fn an_nco_clock_into_noise_equals_the_blocks() {
    let n = 20_000;
    let r = Rate::hz(48_000).unwrap();
    let mut rng = Pcg32::new(SEED, 1);
    let input: Vec<f32> = (0..n).map(|_| rng.normal() as f32).collect();
    // The blocks: the nco output is only the clock of the noise block.
    let nco_cfg = NcoConfig {
        frequency_hz: 1_000.0,
        ..NcoConfig::default()
    };
    let mut nco = NcoRealBlock::<C32, VecStorage>::new(&nco_cfg, r, &mut VecStorage).unwrap();
    let mut tone = vec![C32::default(); n];
    nco.process(&input, &mut tone).unwrap();
    let cfg = NoiseConfig {
        sigma: 0.3,
        seed: SEED,
        stream: 4,
    };
    let mut noise = NoiseBlock::<C32, VecStorage>::new(&cfg, r, &mut VecStorage).unwrap();
    let mut want = vec![C32::default(); n];
    noise.process(&tone, &mut want).unwrap();

    let doc = json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
        "inputs": { "in": { "type": "f32", "rate_hz": 48000 } },
        "nodes": {
            "clk": { "kind": "nco", "params": { "frequency_hz": 1000.0 } },
            "n": { "kind": "noise", "params": { "sigma": 0.3, "seed": SEED, "stream": 4 } }
        },
        "edges": [
            { "from": "in", "to": "clk" },
            { "from": "clk", "to": "n" },
            { "from": "n", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    });
    let mut p = build(&doc.to_string());
    let got = push_c32(&mut p, &input, &mut rng);
    same_bits_c32(&got, &want);
}

#[test]
fn a_runtime_sigma_change_equals_the_block() {
    let r = Rate::hz(48_000).unwrap();
    let mut rng = Pcg32::new(SEED, 2);
    let (first, second) = (3_000, 2_500);
    let input: Vec<f32> = (0..first + second).map(|_| rng.normal() as f32).collect();
    let cfg = NoiseConfig {
        sigma: 0.1,
        seed: SEED,
        stream: 1,
    };
    let mut block = NoiseRealBlock::<f32, VecStorage>::new(&cfg, r, &mut VecStorage).unwrap();
    let mut want = vec![0.0f32; first + second];
    block.process(&input[..first], &mut want[..first]).unwrap();
    block.set_param("sigma", ParamValue::F64(0.5)).unwrap();
    block.process(&input[first..], &mut want[first..]).unwrap();

    let mut p = build(
        &single(
            "noise",
            "f32",
            "f32",
            48_000,
            json!({ "sigma": 0.1, "seed": SEED }),
        )
        .to_string(),
    );
    let mut got = Vec::new();
    let mut buf = vec![0.0f32; 4096];
    for (k, part) in [&input[..first], &input[first..]].into_iter().enumerate() {
        if k == 1 {
            p.set_param("n", "sigma", json!(0.5)).unwrap();
            assert!(p.set_param("n", "sigma", json!(17.0)).is_err());
            assert!(p.set_param("n", "seed", json!(3)).is_err());
        }
        let mut rest = part;
        while !rest.is_empty() {
            let n = (1 + rng.next_u32() as usize % 1000).min(rest.len());
            p.push("in", InputData::F32(&rest[..n])).unwrap();
            rest = &rest[n..];
            let k = p.read_stream::<f32>("out", &mut buf).unwrap();
            got.extend_from_slice(&buf[..k]);
        }
    }
    assert_eq!(got.len(), want.len());
    assert!(
        got.iter()
            .zip(&want)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "pipeline differs from the block"
    );
}

#[test]
fn the_q15_kinds_equal_the_blocks() {
    let r = Rate::hz(8_000).unwrap();
    let n = 5_000;
    let cfg = NoiseConfig {
        sigma: 0.6,
        seed: SEED,
        stream: 3,
    };
    let params = json!({ "sigma": 0.6, "seed": SEED, "stream": 3 });
    let mut rng = Pcg32::new(SEED, 3);

    // Real: q15 in, q15 out.
    let input = vec![Q15(0); n];
    let mut block = NoiseRealBlock::<Q15, VecStorage>::new(&cfg, r, &mut VecStorage).unwrap();
    let mut want = vec![Q15(0); n];
    block.process(&input, &mut want).unwrap();
    let mut p = build(&single("noise", "q15", "q15", 8_000, params.clone()).to_string());
    let mut got = Vec::new();
    let mut buf = vec![Q15(0); 4096];
    let mut rest = &input[..];
    while !rest.is_empty() {
        let m = (1 + rng.next_u32() as usize % 1000).min(rest.len());
        p.push("in", InputData::Q15(&rest[..m])).unwrap();
        rest = &rest[m..];
        let k = p.read_stream::<Q15>("out", &mut buf).unwrap();
        got.extend_from_slice(&buf[..k]);
    }
    assert_eq!(got, want);

    // Complex: cq15 in, cq15 out.
    let input = vec![CQ15::default(); n];
    let mut block = NoiseBlock::<CQ15, VecStorage>::new(&cfg, r, &mut VecStorage).unwrap();
    let mut want = vec![CQ15::default(); n];
    block.process(&input, &mut want).unwrap();
    let mut p = build(&single("noise", "cq15", "q15", 8_000, params).to_string());
    let mut got = Vec::new();
    let mut buf = vec![CQ15::default(); 4096];
    let mut rest = &input[..];
    while !rest.is_empty() {
        let m = (1 + rng.next_u32() as usize % 1000).min(rest.len());
        p.push("in", InputData::CQ15(&rest[..m])).unwrap();
        rest = &rest[m..];
        let k = p.read_stream::<CQ15>("out", &mut buf).unwrap();
        got.extend_from_slice(&buf[..k]);
    }
    assert_eq!(got, want);
}
