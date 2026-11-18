//! The kinds `symbol_map`, `iq_offset` and `qam_acquire` (M44.1 §2.1, §2.2, §2.4): registration
//! (without kernels; `qam_acquire` with `fft`), the costs `1 / bits`, 1 and 20 or 40, the rates,
//! the type checks (`symbol_map` takes `f32` soft bits, `iq_offset` and `qam_acquire` `c32`),
//! `invalid_value` for the invalid combinations of `symbol_map` and an odd `samples_per_symbol`
//! of `iq_offset` (its range: `out_of_range`; `samples_per_symbol` since M44.1.1 §3.3), and
//! push-mode chains `symbol_map` -> `iq_offset` and `symbol_map` -> `qam_acquire` that equal the
//! blocks in pieces of 1 to 1000 samples (seed 852).
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
use sdrk_blocks::{
    Block, IqOffsetBlock, IqOffsetConfig, IqPart, QamAcquireBlock, QamAcquireConfig,
    SymbolMapBlock, SymbolMapConfig, SymbolMapConstellation, SymbolMapVariant,
};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Registry, VecStorage};
use sdrk_testkit::Pcg32;
use serde_json::{Value, json};

use common::build;

/// The seed of the soft bits and of the pieces.
const SEED: u64 = 852;

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
fn the_kinds_are_registered_without_kernels() {
    let registry = Registry::standard();
    for kind in ["symbol_map", "iq_offset"] {
        let f = registry.factory(kind).unwrap();
        assert!(f.kernels().is_empty(), "{kind}");
        assert_eq!(f.description().kind, kind);
    }
    assert_eq!(registry.descriptions().count(), 80);
}

#[test]
fn symbol_map_costs_one_lookup_per_symbol() {
    for (name, bits) in [
        ("bpsk", 1u64),
        ("qpsk", 2),
        ("8psk", 3),
        ("qam16", 4),
        ("qam64", 6),
    ] {
        let report = validate(&single(
            "symbol_map",
            "f32",
            9600,
            json!({ "constellation": name }),
        ));
        assert!(report.is_ok(), "{name}: {}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 1.0 / bits as f64, "{name}");
        let node = &report.nodes[0];
        assert_eq!(node.rate_out, Rate::new(9600, bits).unwrap(), "{name}");
        assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO);
    }
}

#[test]
fn symbol_map_checks_its_combinations_and_its_input() {
    for (params, path) in [
        (
            json!({ "constellation": "qam16", "differential": true }),
            "nodes.n.params.differential",
        ),
        (
            json!({ "constellation": "qpsk", "variant": "pi4" }),
            "nodes.n.params.variant",
        ),
        (
            json!({ "constellation": "8psk", "differential": true, "variant": "pi4" }),
            "nodes.n.params.variant",
        ),
    ] {
        let (p, code) = first_error(&single("symbol_map", "f32", 9600, params.clone()));
        assert_eq!(
            (p.as_str(), code),
            (path, DiagCode::InvalidValue),
            "{params}"
        );
    }
    assert!(
        validate(&single(
            "symbol_map",
            "f32",
            9600,
            json!({ "constellation": "qpsk", "differential": true, "variant": "pi4" })
        ))
        .is_ok()
    );
    // The names of `symbol_demap` without a transmit mapping are not choices.
    for name in ["real", "fsk4"] {
        assert!(
            !validate(&single(
                "symbol_map",
                "f32",
                9600,
                json!({ "constellation": name })
            ))
            .is_ok()
        );
    }
    for ty in ["c32", "cq15", "q15"] {
        let (_, code) = first_error(&single("symbol_map", ty, 9600, json!({})));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
}

#[test]
fn iq_offset_costs_one_and_needs_its_samples_per_symbol() {
    let report = validate(&single(
        "iq_offset",
        "c32",
        9600,
        json!({ "samples_per_symbol": 8 }),
    ));
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.cost_per_input_sample, 1.0);
    let node = &report.nodes[0];
    assert_eq!(node.rate_out, Rate::hz(9600).unwrap());
    assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO);
    // Required, even, 2 to 1024; the old name `samples` is unknown.
    assert!(!validate(&single("iq_offset", "c32", 9600, json!({}))).is_ok());
    for (sps, code) in [
        (0, DiagCode::OutOfRange),
        (1026, DiagCode::OutOfRange),
        (3, DiagCode::InvalidValue),
        (1023, DiagCode::InvalidValue),
    ] {
        let (p, c) = first_error(&single(
            "iq_offset",
            "c32",
            9600,
            json!({ "samples_per_symbol": sps }),
        ));
        assert_eq!(
            (p.as_str(), c),
            ("nodes.n.params.samples_per_symbol", code),
            "{sps}"
        );
    }
    let report = validate(&single("iq_offset", "c32", 9600, json!({ "samples": 4 })));
    assert!(
        report.errors.iter().any(
            |e| (e.path.as_str(), e.code) == ("nodes.n.params.samples", DiagCode::UnknownParam)
        ),
        "{}",
        report.to_json()
    );
    assert!(
        validate(&single(
            "iq_offset",
            "c32",
            9600,
            json!({ "samples_per_symbol": 1024, "part": "real" })
        ))
        .is_ok()
    );
    for ty in ["f32", "cq15"] {
        let (_, code) = first_error(&single(
            "iq_offset",
            ty,
            9600,
            json!({ "samples_per_symbol": 4 }),
        ));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
}

#[test]
fn the_chain_equals_the_blocks() {
    let mut rng = Pcg32::new(SEED, 1);
    let bits: Vec<f32> = (0..20_000)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect();
    let r = Rate::hz(9600).unwrap();
    // The blocks directly.
    let map_cfg = SymbolMapConfig {
        constellation: SymbolMapConstellation::Qpsk,
        differential: true,
        variant: SymbolMapVariant::Pi4,
    };
    let mut map = SymbolMapBlock::<VecStorage>::new(&map_cfg, r, &mut VecStorage).unwrap();
    let mut points = vec![C32::default(); bits.len() / 2];
    assert_eq!(
        map.process(&bits, &mut points).unwrap().produced,
        points.len()
    );
    let off_cfg = IqOffsetConfig {
        samples_per_symbol: 6,
        part: IqPart::Real,
    };
    let mut off = IqOffsetBlock::<VecStorage>::new(&off_cfg, r, &mut VecStorage).unwrap();
    let mut want = vec![C32::default(); points.len()];
    off.process(&points, &mut want).unwrap();
    // The pipeline in pieces of 1 to 1000 samples.
    let doc = json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
        "inputs": { "in": { "type": "f32", "rate_hz": 9600 } },
        "nodes": {
            "map": {
                "kind": "symbol_map",
                "params": { "constellation": "qpsk", "differential": true, "variant": "pi4" }
            },
            "offset": {
                "kind": "iq_offset",
                "params": { "samples_per_symbol": 6, "part": "real" }
            }
        },
        "edges": [
            { "from": "in", "to": "map" },
            { "from": "map", "to": "offset" },
            { "from": "offset", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    });
    let mut p = build(&doc.to_string());
    let mut got = Vec::new();
    let mut buf = vec![C32::default(); 4096];
    let mut rest = &bits[..];
    while !rest.is_empty() {
        let n = (1 + rng.next_u32() as usize % 1000).min(rest.len());
        p.push("in", InputData::F32(&rest[..n])).unwrap();
        rest = &rest[n..];
        let k = p.read_stream::<C32>("out", &mut buf).unwrap();
        got.extend_from_slice(&buf[..k]);
    }
    assert_eq!(got.len(), want.len());
    assert!(
        got.iter()
            .zip(&want)
            .all(|(a, b)| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()),
        "pipeline differs from the blocks"
    );
}

#[test]
fn qam_acquire_needs_the_fft_and_c32() {
    let registry = Registry::standard();
    let f = registry.factory("qam_acquire").unwrap();
    assert_eq!(f.kernels(), ["fft"]);
    for (correct, cost) in [(false, 20.0), (true, 40.0)] {
        let report = validate(&single(
            "qam_acquire",
            "c32",
            25_000,
            json!({ "correct": correct }),
        ));
        assert!(report.is_ok(), "{}", report.to_json());
        assert_eq!(report.cost_per_input_sample, cost);
        let node = &report.nodes[0];
        assert_eq!(node.rate_out, Rate::hz(25_000).unwrap());
        assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO);
    }
    for ty in ["f32", "cq15"] {
        let (_, code) = first_error(&single("qam_acquire", ty, 9600, json!({})));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
    let (p, code) = first_error(&single("qam_acquire", "c32", 9600, json!({ "correct": 1 })));
    assert_eq!(
        (p.as_str(), code),
        ("nodes.n.params.correct", DiagCode::WrongType)
    );
}

#[test]
fn the_acquire_chain_equals_the_blocks() {
    let mut rng = Pcg32::new(SEED, 2);
    let bits: Vec<f32> = (0..24_000)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect();
    let r = Rate::hz(9600).unwrap();
    let map_cfg = SymbolMapConfig {
        constellation: SymbolMapConstellation::Qam16,
        ..SymbolMapConfig::default()
    };
    let mut map = SymbolMapBlock::<VecStorage>::new(&map_cfg, r, &mut VecStorage).unwrap();
    let mut points = vec![C32::default(); bits.len() / 4];
    map.process(&bits, &mut points).unwrap();
    // A carrier offset of 0.003 cycles per symbol for the acquisition to find.
    for (k, v) in points.iter_mut().enumerate() {
        let t = core::f64::consts::TAU * 0.003 * k as f64;
        let (s, c) = (t.sin() as f32, t.cos() as f32);
        *v = C32::new(v.re * c - v.im * s, v.re * s + v.im * c);
    }
    let mut acq = QamAcquireBlock::<VecStorage>::new(
        &QamAcquireConfig { correct: true },
        Rate::hz(2400).unwrap(),
        &mut VecStorage,
    )
    .unwrap();
    let mut want = vec![C32::default(); points.len()];
    acq.process(&points, &mut want).unwrap();
    assert!((acq.estimate().unwrap() - 0.003).abs() < 6e-5);
    let doc = json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
        "inputs": { "in": { "type": "f32", "rate_hz": 9600 } },
        "nodes": {
            "map": { "kind": "symbol_map", "params": { "constellation": "qam16" } },
            "acq": { "kind": "qam_acquire", "params": { "correct": true } }
        },
        "edges": [
            { "from": "in", "to": "map" },
            { "from": "map", "to": "acq" },
            { "from": "acq", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    });
    // The pipeline sees the points without the offset: compare against the block on them.
    let mut plain = vec![C32::default(); bits.len() / 4];
    let mut map = SymbolMapBlock::<VecStorage>::new(&map_cfg, r, &mut VecStorage).unwrap();
    map.process(&bits, &mut plain).unwrap();
    let mut acq = QamAcquireBlock::<VecStorage>::new(
        &QamAcquireConfig { correct: true },
        Rate::hz(2400).unwrap(),
        &mut VecStorage,
    )
    .unwrap();
    let mut want = vec![C32::default(); plain.len()];
    acq.process(&plain, &mut want).unwrap();
    let mut p = build(&doc.to_string());
    let mut got = Vec::new();
    let mut buf = vec![C32::default(); 4096];
    let mut rest = &bits[..];
    while !rest.is_empty() {
        let n = (1 + rng.next_u32() as usize % 1000).min(rest.len());
        p.push("in", InputData::F32(&rest[..n])).unwrap();
        rest = &rest[n..];
        let k = p.read_stream::<C32>("out", &mut buf).unwrap();
        got.extend_from_slice(&buf[..k]);
    }
    assert_eq!(got.len(), want.len());
    assert!(
        got.iter()
            .zip(&want)
            .all(|(a, b)| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()),
        "pipeline differs from the blocks"
    );
}
