//! The kinds `bit_scramble` and `bit_descramble` (M44.2, the maintainer's decision on I44.2-6):
//! registration without kernels, the cost 2 per bit, the rate unchanged, `f32` input
//! only, the choices `polynomial` and `constellation`, and a flow `bit_scramble` ->
//! `bit_descramble` that gives its input back bit for bit, pushed in pieces.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::needless_pass_by_value
)]

mod common;

use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Registry};
use sdrk_testkit::Pcg32;
use serde_json::{Value, json};

use common::build;

fn flow(ty: &str, nodes: Value, edges: Value) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1000 },
        "inputs": { "in": { "type": ty, "rate_hz": 9600 } },
        "nodes": nodes,
        "edges": edges,
        "outputs": { "out": { "type": "stream", "capacity": 65536 } }
    })
}

fn one(kind: &str, ty: &str, params: Value) -> Value {
    flow(
        ty,
        json!({ "n": { "kind": kind, "params": params } }),
        json!([ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ]),
    )
}

fn validate(doc: &Value) -> sdrk_pipeline::Report {
    Registry::standard().validate_json(&doc.to_string(), &BuildOptions::default())
}

#[test]
fn the_kinds_are_registered_without_kernels() {
    let registry = Registry::standard();
    for kind in ["bit_scramble", "bit_descramble"] {
        assert!(
            registry.factory(kind).unwrap().kernels().is_empty(),
            "{kind}"
        );
        let report = validate(&one(
            kind,
            "f32",
            json!({ "polynomial": "x58_x39", "constellation": "qam1024" }),
        ));
        assert!(report.is_ok(), "{}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 2.0, "{kind}");
        assert_eq!(report.nodes[0].rate_out, sdrk::Rate::hz(9600).unwrap());
        // `f32` soft bits only
        let report = validate(&one(kind, "c32", json!({})));
        assert_eq!(report.errors[0].code, DiagCode::TypeMismatch, "{kind}");
        for (params, path) in [
            (
                json!({ "polynomial": "x7_x4" }),
                "nodes.n.params.polynomial",
            ),
            (
                json!({ "constellation": "qam32" }),
                "nodes.n.params.constellation",
            ),
        ] {
            let report = validate(&one(kind, "f32", params.clone()));
            assert_eq!(report.errors[0].path, path, "{kind} {params}");
        }
        let report = validate(&one(kind, "f32", json!({ "taps": 23 })));
        assert_eq!(report.errors[0].code, DiagCode::UnknownParam, "{kind}");
    }
}

#[test]
fn a_flow_of_both_gives_its_input_back() {
    let doc = flow(
        "f32",
        json!({
            "s": { "kind": "bit_scramble", "params": { "constellation": "qam256" } },
            "d": { "kind": "bit_descramble", "params": { "constellation": "qam256" } }
        }),
        json!([
            { "from": "in", "to": "s" }, { "from": "s", "to": "d" }, { "from": "d", "to": "out" }
        ]),
    );
    let mut p = build(&doc.to_string());
    let mut rng = Pcg32::new(882, 3);
    let x: Vec<f32> = (0..20_000)
        .map(|_| (rng.next_f64() as f32 + 0.01) * if rng.below(2) == 1 { 1.0 } else { -1.0 })
        .collect();
    let mut out = Vec::new();
    let mut buf = vec![0.0f32; 65536];
    let mut rest = &x[..];
    while !rest.is_empty() {
        let n = (1 + rng.below(997) as usize).min(rest.len());
        p.push("in", InputData::F32(&rest[..n])).unwrap();
        rest = &rest[n..];
        loop {
            let k = p.read_stream::<f32>("out", &mut buf).unwrap();
            if k == 0 {
                break;
            }
            out.extend_from_slice(&buf[..k]);
        }
    }
    assert_eq!(out, x);
}
