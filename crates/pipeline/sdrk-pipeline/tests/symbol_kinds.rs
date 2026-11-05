//! The kinds `symbol_sync` and `slice` (M38): registration, the cost `2 + 24 / sps` and 1, the
//! figures (nominal rate, delay), `rate_unsupported` below 2 samples per symbol, the type checks
//! (`symbol_sync` and `slice` take `c32` or `f32`, the constellation decides which), the
//! run-time parameter `scale`, and a push-mode chain `symbol_sync` -> `slice` that equals the
//! blocks.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::needless_pass_by_value,
    clippy::large_stack_frames,
    clippy::assert_is_empty,
    clippy::many_single_char_names
)]

mod common;

use sdrk::{C32, Delay, Rate};
use sdrk_blocks::{
    Block, SliceBlock, SliceConfig, SliceConstellation, SymbolSyncBlock, SymbolSyncConfig,
};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Registry, VecStorage};
use sdrk_testkit::Pcg32;
use serde_json::{Value, json};

use common::build;

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

/// `symbol_sync` then `slice` on an input of type `ty`.
fn chain(ty: &str, rate: u64, sync: Value, slice: Value) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
        "inputs": { "in": { "type": ty, "rate_hz": rate } },
        "nodes": {
            "sync": { "kind": "symbol_sync", "params": sync },
            "slice": { "kind": "slice", "params": slice }
        },
        "edges": [
            { "from": "in", "to": "sync" },
            { "from": "sync", "to": "slice" },
            { "from": "slice", "to": "bits" }
        ],
        "outputs": { "bits": { "type": "stream" } }
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
fn the_kinds_are_registered() {
    let registry = Registry::standard();
    let f = registry.factory("symbol_sync").unwrap();
    assert_eq!(f.kernels(), ["symbol_sync"]);
    assert_eq!(f.description().kind, "symbol_sync");
    let f = registry.factory("slice").unwrap();
    assert!(f.kernels().is_empty());
    assert_eq!(f.description().kind, "slice");
    assert_eq!(registry.descriptions().count(), 51);
}

#[test]
fn symbol_sync_costs_two_plus_24_per_symbol_and_runs_at_the_symbol_rate() {
    for (ty, rate, sym, sps) in [
        ("c32", 4100u64, 1000.0, 4.1),
        ("f32", 19_000, 1187.5, 16.0),
        ("c32", 48_000, 24_000.0, 2.0),
    ] {
        let report = validate(&single(
            "symbol_sync",
            ty,
            rate,
            json!({ "symbol_rate_hz": sym }),
        ));
        assert!(report.is_ok(), "{ty} {rate}: {}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 2.0 + 24.0 / sps);
        let node = &report.nodes[0];
        assert_eq!(node.rate_in, Rate::hz(rate).unwrap());
        assert_eq!(
            node.rate_out,
            Rate::new((sym * 1000.0) as u64, 1000).unwrap()
        );
        assert_eq!(node.delay_samples_in.reduced(), Delay::new(1, 1).unwrap());
    }
}

#[test]
fn symbol_sync_needs_the_symbol_rate_and_two_samples_per_symbol() {
    // Required parameter.
    assert!(!validate(&single("symbol_sync", "c32", 48_000, json!({}))).is_ok());
    // Exactly 2 samples per symbol is fine, fewer is rate_unsupported.
    assert!(
        validate(&single(
            "symbol_sync",
            "c32",
            48_000,
            json!({ "symbol_rate_hz": 24_000 })
        ))
        .is_ok()
    );
    let (path, code) = first_error(&single(
        "symbol_sync",
        "c32",
        48_000,
        json!({ "symbol_rate_hz": 24_001 }),
    ));
    assert_eq!(code, DiagCode::RateUnsupported);
    assert_eq!(path, "nodes.n.params.symbol_rate_hz");
    // Ranges of the description.
    for bad in [
        json!({ "symbol_rate_hz": 1000, "loop_bw": 0.2 }),
        json!({ "symbol_rate_hz": 1000, "damping": 5 }),
        json!({ "symbol_rate_hz": 1000, "max_deviation": 0.1 }),
        json!({ "symbol_rate_hz": 1000, "ted_gain": 0 }),
        json!({ "symbol_rate_hz": 1000, "ted": "costas" }),
        json!({ "symbol_rate_hz": 0 }),
    ] {
        assert!(
            !validate(&single("symbol_sync", "c32", 48_000, bad.clone())).is_ok(),
            "{bad} must be rejected"
        );
    }
    // Only c32 and f32.
    for ty in ["cq15", "q15"] {
        let (_, code) = first_error(&single(
            "symbol_sync",
            ty,
            48_000,
            json!({ "symbol_rate_hz": 1000 }),
        ));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
    assert!(
        validate(&single(
            "symbol_sync",
            "f32",
            48_000,
            json!({ "symbol_rate_hz": 1000, "ted": "mm" })
        ))
        .is_ok()
    );
}

#[test]
fn slice_checks_the_type_per_constellation() {
    for (ty, constellation, bits) in [
        ("f32", "real", 1u64),
        ("c32", "bpsk", 1),
        ("c32", "qpsk", 2),
    ] {
        let report = validate(&single(
            "slice",
            ty,
            1200,
            json!({ "constellation": constellation }),
        ));
        assert!(report.is_ok(), "{constellation}: {}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 1.0);
        let node = &report.nodes[0];
        assert_eq!(node.rate_out, Rate::hz(1200 * bits).unwrap());
        assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO.reduced());
    }
    // The default is bpsk.
    assert!(validate(&single("slice", "c32", 1200, json!({}))).is_ok());
    for (ty, constellation) in [("c32", "real"), ("f32", "bpsk"), ("f32", "qpsk")] {
        let (path, code) = first_error(&single(
            "slice",
            ty,
            1200,
            json!({ "constellation": constellation }),
        ));
        assert_eq!(code, DiagCode::InvalidValue, "{ty} {constellation}");
        assert_eq!(path, "nodes.n.params.constellation");
    }
    for ty in ["cq15", "q15"] {
        let (_, code) = first_error(&single("slice", ty, 1200, json!({})));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
    assert!(!validate(&single("slice", "c32", 1200, json!({ "scale": 0 }))).is_ok());
}

/// Shaped BPSK (half-sine pulses, overlapping by half a symbol) with noise.
fn bpsk(n: usize, sps: f64) -> Vec<C32> {
    let mut rng = Pcg32::new(493, 50);
    let symbols: Vec<f64> = (0..(n as f64 / sps) as usize + 4)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect();
    (0..n)
        .map(|i| {
            let t = i as f64 / sps;
            let k = t.floor() as usize;
            let frac = t - t.floor();
            // Raised-cosine transition between neighbouring symbols.
            let w = 0.5 - 0.5 * (std::f64::consts::PI * frac).cos();
            let v = symbols[k] * (1.0 - w) + symbols[k + 1] * w;
            C32::new(
                (v + 0.05 * rng.normal()) as f32,
                (0.05 * rng.normal()) as f32,
            )
        })
        .collect()
}

#[test]
fn the_chain_equals_the_blocks_and_scale_changes_at_run_time() {
    let sync_cfg = SymbolSyncConfig {
        symbol_rate_hz: 1000.0,
        ..SymbolSyncConfig::default()
    };
    let r = Rate::hz(4100).unwrap();
    let input = bpsk(20_000, 4.1);
    // The blocks directly.
    let mut sync = SymbolSyncBlock::<C32, VecStorage>::new(&sync_cfg, r, &mut VecStorage).unwrap();
    let mut symbols = vec![C32::default(); input.len()];
    let p = sync.process(&input, &mut symbols).unwrap();
    symbols.truncate(p.produced);
    let slice_cfg = SliceConfig {
        constellation: SliceConstellation::Qpsk,
        scale: 2.0,
    };
    let mut slice = SliceBlock::<C32, VecStorage>::new(&slice_cfg, r, &mut VecStorage).unwrap();
    let mut want = vec![0.0f32; 2 * symbols.len()];
    slice.process(&symbols, &mut want).unwrap();
    // The pipeline in pieces of up to 1000 samples.
    let doc = chain(
        "c32",
        4100,
        json!({ "symbol_rate_hz": 1000 }),
        json!({ "constellation": "qpsk", "scale": 2.0 }),
    );
    let mut p = build(&doc.to_string());
    let mut got = Vec::new();
    let mut buf = vec![0.0f32; 4096];
    for piece in input.chunks(1000) {
        p.push("in", InputData::C32(piece)).unwrap();
        let n = p.read_stream::<f32>("bits", &mut buf).unwrap();
        got.extend_from_slice(&buf[..n]);
    }
    assert_eq!(got.len(), want.len());
    assert!(
        got.iter()
            .zip(&want)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "pipeline differs from the blocks"
    );
    // The tail of the soft bits of re are about +-2 (locked), and scale is a runtime parameter.
    assert!(
        got[got.len() - 200..]
            .iter()
            .step_by(2)
            .all(|v| v.abs() > 1.0)
    );
    assert!(p.set_param("slice", "scale", json!(4.0)).is_ok());
    assert!(
        p.set_param("slice", "constellation", json!("bpsk"))
            .is_err()
    );
    assert!(p.set_param("sync", "loop_bw", json!(0.02)).is_err());
}
