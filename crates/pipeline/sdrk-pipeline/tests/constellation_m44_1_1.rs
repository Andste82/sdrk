//! Test 4 of M44.1.1: `constellation` instead of `order` in `qam_mod`, `qam_demod`,
//! `qam_frame_encode`, `qam_frame_decode`, `psk_mod` and `psk_demod`: every valid and invalid
//! value of the six kinds, the combinations with `differential` and `variant`, and
//! `unknown_param` for the old name `order` in a flow.

#![allow(
    clippy::needless_pass_by_value,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::large_stack_frames
)]

use sdrk_pipeline::{BuildOptions, DiagCode, Registry};
use serde_json::{Value, json};

/// The names of the choice, in index order.
const ALL: [&str; 7] = [
    "bpsk", "qpsk", "8psk", "qam16", "qam64", "qam256", "qam1024",
];

/// A flow with one node of `kind` and the input it takes.
fn flow(kind: &str, params: Value) -> Value {
    let (input, output) = match kind {
        "qam_mod" | "psk_mod" => (json!({ "type": "f32", "rate_hz": 96_000 }), "stream"),
        "qam_demod" | "psk_demod" => (json!({ "type": "c32", "rate_hz": 200_000 }), "stream"),
        "qam_frame_encode" => (json!({ "type": "packets", "rate_hz": 100_000 }), "stream"),
        _ => (json!({ "type": "c32", "rate_hz": 200_000 }), "packets"),
    };
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
        "inputs": { "in": input },
        "nodes": { "n": { "kind": kind, "params": params } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": output } }
    })
}

/// The required parameters of `kind` with `extra` added.
fn params(kind: &str, extra: Value) -> Value {
    let mut p = if kind.ends_with("_mod") || kind == "qam_frame_encode" {
        json!({ "samples_per_symbol": 8 })
    } else {
        json!({ "symbol_rate_hz": 25_000 })
    };
    for (k, v) in extra.as_object().unwrap() {
        p[k] = v.clone();
    }
    p
}

/// The errors of a flow as (path, code).
fn errors(doc: &Value) -> Vec<(String, DiagCode)> {
    Registry::standard()
        .validate_json(&doc.to_string(), &BuildOptions::default())
        .errors
        .iter()
        .map(|e| (e.path.clone(), e.code))
        .collect()
}

#[test]
fn every_value_of_the_constellation_of_the_six_kinds() {
    let qam = [
        "qam_mod",
        "qam_demod",
        "qam_frame_encode",
        "qam_frame_decode",
    ];
    for kind in qam.iter().chain(&["psk_mod", "psk_demod"]) {
        let valid: &[&str] = if kind.starts_with("qam") {
            &["qam16", "qam64"]
        } else {
            &["bpsk", "qpsk", "8psk"]
        };
        for name in ALL {
            let got = errors(&flow(kind, params(kind, json!({ "constellation": name }))));
            if valid.contains(&name) {
                assert!(got.is_empty(), "{kind} {name}: {got:?}");
            } else {
                assert_eq!(
                    got.first(),
                    Some(&(
                        "nodes.n.params.constellation".to_string(),
                        DiagCode::InvalidValue
                    )),
                    "{kind} {name}"
                );
            }
        }
        // A name outside the choice, the wrong type and the default.
        for bad in [
            json!("qam32"),
            json!("qam4"),
            json!("QAM16"),
            json!(16),
            json!(true),
        ] {
            let got = errors(&flow(kind, params(kind, json!({ "constellation": bad }))));
            assert_eq!(
                got.first().map(|(p, _)| p.as_str()),
                Some("nodes.n.params.constellation"),
                "{kind} {bad}"
            );
        }
        assert!(
            errors(&flow(kind, params(kind, json!({})))).is_empty(),
            "{kind}"
        );
        // The old name is gone (no alias): `unknown_param`, with every value it had.
        for order in [2, 4, 8, 16, 64] {
            let got = errors(&flow(kind, params(kind, json!({ "order": order }))));
            assert_eq!(
                got.first(),
                Some(&("nodes.n.params.order".to_string(), DiagCode::UnknownParam)),
                "{kind} order {order}"
            );
        }
    }
}

#[test]
fn the_constellation_combines_with_differential_and_variant() {
    // (constellation, differential, variant, samples per symbol, valid): the rules of M40.1 §4
    // (D648) on the constellation instead of the order.
    for (c, differential, variant, sps, ok) in [
        ("bpsk", false, "plain", 8, true),
        ("bpsk", true, "plain", 8, true),
        ("qpsk", true, "plain", 8, true),
        ("8psk", true, "plain", 8, true),
        ("qpsk", true, "pi4", 8, true),
        ("qpsk", false, "pi4", 8, false),
        ("bpsk", true, "pi4", 8, false),
        ("8psk", true, "pi4", 8, false),
        ("qpsk", false, "offset", 8, true),
        ("qpsk", true, "offset", 8, false),
        ("qpsk", false, "offset", 7, false),
        ("bpsk", false, "offset", 8, false),
        ("8psk", false, "offset", 8, false),
    ] {
        let m = json!({
            "constellation": c, "samples_per_symbol": sps, "differential": differential,
            "variant": variant
        });
        // `psk_demod` at 8 or 7 samples per symbol of 25 kBd.
        let d = json!({
            "constellation": c, "symbol_rate_hz": 25_000, "differential": differential,
            "variant": variant
        });
        let mut demod = flow("psk_demod", d);
        demod["inputs"]["in"]["rate_hz"] = json!(25_000 * sps);
        for (kind, doc) in [("psk_mod", flow("psk_mod", m)), ("psk_demod", demod)] {
            let got = errors(&doc);
            if ok {
                assert!(
                    got.is_empty(),
                    "{kind} {c} {differential} {variant} {sps}: {got:?}"
                );
            } else {
                assert_eq!(
                    got.first(),
                    Some(&("nodes.n.params.variant".to_string(), DiagCode::InvalidValue)),
                    "{kind} {c} {differential} {variant} {sps}"
                );
            }
        }
    }
}
