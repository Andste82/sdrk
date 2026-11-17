//! The kinds `pulse_shape` and `pulse_match` (M44.1.1 §3.1, §3.2; seed 863): registration with
//! their kernels (`fir_interpolate`, `fir`), the cost of one tap per input sample, the rates and
//! delays, the required parameters, the type checks (`c32` only), `rate_unsupported` for a
//! symbol rate outside the rate rule of `pulse_match` and `out_of_range` for its ranges, and the
//! push-mode chain `pulse_shape` -> `pulse_match` that equals the blocks in pieces of 1 to 1000
//! samples.
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
    clippy::large_stack_frames
)]

mod common;

use sdrk::{C32, Delay, Rate};
use sdrk_blocks::{Block, PulseMatchBlock, PulseMatchConfig, PulseShapeBlock, PulseShapeConfig};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Registry, VecStorage};
use sdrk_testkit::Pcg32;
use serde_json::{Value, json};

use common::build;

/// The seed of the signal and of the pieces.
const SEED: u64 = 863;

/// A one-node flow of `kind` on an input of type `ty` at `rate` Hz.
fn single(kind: &str, ty: &str, rate: u64, params: Value) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
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
fn the_kinds_are_registered_with_their_kernels() {
    let registry = Registry::standard();
    for (kind, kernel) in [("pulse_shape", "fir_interpolate"), ("pulse_match", "fir")] {
        let f = registry.factory(kind).unwrap();
        assert_eq!(f.kernels(), [kernel], "{kind}");
        assert_eq!(f.description().kind, kind);
    }
}

#[test]
fn pulse_shape_costs_its_taps_per_symbol() {
    for (sps, span) in [(8u64, 10u64), (2, 4), (64, 32)] {
        let report = validate(&single(
            "pulse_shape",
            "c32",
            25_000,
            json!({ "samples_per_symbol": sps, "span": span }),
        ));
        assert!(report.is_ok(), "{}", report.to_json());
        let taps = span * sps + 1;
        assert_eq!(report.cost_per_input_sample, taps as f64);
        let node = &report.nodes[0];
        assert_eq!(node.rate_out, Rate::hz(25_000 * sps).unwrap());
        assert_eq!(
            node.delay_samples_in.reduced(),
            Delay::new(span, 2).unwrap().reduced()
        );
    }
    // Required, 2 to 64.
    let (p, code) = first_error(&single("pulse_shape", "c32", 25_000, json!({})));
    assert_eq!(
        (p.as_str(), code),
        ("nodes.n.params.samples_per_symbol", DiagCode::MissingParam)
    );
    for sps in [1, 65] {
        let (p, code) = first_error(&single(
            "pulse_shape",
            "c32",
            25_000,
            json!({ "samples_per_symbol": sps }),
        ));
        assert_eq!(
            (p.as_str(), code),
            ("nodes.n.params.samples_per_symbol", DiagCode::OutOfRange)
        );
    }
    for ty in ["f32", "cq15"] {
        let (_, code) = first_error(&single(
            "pulse_shape",
            ty,
            25_000,
            json!({ "samples_per_symbol": 8 }),
        ));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
}

#[test]
fn pulse_match_costs_its_taps_and_keeps_the_rate_rule() {
    let report = validate(&single(
        "pulse_match",
        "c32",
        200_000,
        json!({ "symbol_rate_hz": 25_000 }),
    ));
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.cost_per_input_sample, 81.0);
    let node = &report.nodes[0];
    assert_eq!(node.rate_out, Rate::hz(200_000).unwrap());
    assert_eq!(node.delay_samples_in.reduced(), Delay::new(40, 1).unwrap());
    // 7.5 and 1.5 samples per symbol, and 2000: the rate rule, with the hint to resample.
    for (fs, sr) in [
        (187_500u64, 25_000.0),
        (37_500, 25_000.0),
        (2_000_000, 1000.0),
    ] {
        let (p, code) = first_error(&single(
            "pulse_match",
            "c32",
            fs,
            json!({ "symbol_rate_hz": sr }),
        ));
        assert_eq!(
            (p.as_str(), code),
            ("nodes.n.params.symbol_rate_hz", DiagCode::RateUnsupported),
            "{fs} {sr}"
        );
    }
    let (p, code) = first_error(&single("pulse_match", "c32", 200_000, json!({})));
    assert_eq!(
        (p.as_str(), code),
        ("nodes.n.params.symbol_rate_hz", DiagCode::MissingParam)
    );
    let (p, code) = first_error(&single(
        "pulse_match",
        "c32",
        200_000,
        json!({ "symbol_rate_hz": 25_000, "rolloff": 0.01 }),
    ));
    assert_eq!(
        (p.as_str(), code),
        ("nodes.n.params.rolloff", DiagCode::OutOfRange)
    );
    for ty in ["f32", "cq15"] {
        let (_, code) = first_error(&single(
            "pulse_match",
            ty,
            200_000,
            json!({ "symbol_rate_hz": 25_000 }),
        ));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
}

#[test]
fn the_chain_equals_the_blocks() {
    let mut rng = Pcg32::new(SEED, 1);
    let symbols: Vec<C32> = (0..6000)
        .map(|_| {
            let mut u = || (f64::from(rng.next_u32()) / 2f64.powi(31) - 1.0) as f32;
            C32::new(u(), u())
        })
        .collect();
    let shape_cfg = PulseShapeConfig {
        samples_per_symbol: 4,
        rolloff: 0.25,
        span: 8,
    };
    let match_cfg = PulseMatchConfig {
        symbol_rate_hz: 2400.0,
        rolloff: 0.25,
        span: 8,
    };
    let mut shape =
        PulseShapeBlock::<VecStorage>::new(&shape_cfg, Rate::hz(2400).unwrap(), &mut VecStorage)
            .unwrap();
    let mut shaped = vec![C32::default(); symbols.len() * 4];
    assert_eq!(
        shape.process(&symbols, &mut shaped).unwrap().produced,
        shaped.len()
    );
    let mut matched =
        PulseMatchBlock::<VecStorage>::new(&match_cfg, Rate::hz(9600).unwrap(), &mut VecStorage)
            .unwrap();
    let mut want = vec![C32::default(); shaped.len()];
    matched.process(&shaped, &mut want).unwrap();
    let doc = json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
        "inputs": { "in": { "type": "c32", "rate_hz": 2400 } },
        "nodes": {
            "shape": {
                "kind": "pulse_shape",
                "params": { "samples_per_symbol": 4, "rolloff": 0.25, "span": 8 }
            },
            "match": {
                "kind": "pulse_match",
                "params": { "symbol_rate_hz": 2400, "rolloff": 0.25, "span": 8 }
            }
        },
        "edges": [
            { "from": "in", "to": "shape" },
            { "from": "shape", "to": "match" },
            { "from": "match", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream", "capacity": 65536 } }
    });
    let mut p = build(&doc.to_string());
    let mut got = Vec::new();
    let mut buf = vec![C32::default(); 8192];
    let mut rest = &symbols[..];
    while !rest.is_empty() {
        let n = (1 + rng.next_u32() as usize % 1000).min(rest.len());
        p.push("in", InputData::C32(&rest[..n])).unwrap();
        rest = &rest[n..];
        loop {
            let k = p.read_stream::<C32>("out", &mut buf).unwrap();
            if k == 0 {
                break;
            }
            got.extend_from_slice(&buf[..k]);
        }
    }
    assert_eq!(got.len(), want.len());
    assert!(
        got.iter()
            .zip(&want)
            .all(|(a, b)| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()),
        "pipeline differs from the blocks"
    );
}
