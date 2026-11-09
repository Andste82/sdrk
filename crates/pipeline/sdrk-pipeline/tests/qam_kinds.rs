//! The kinds of M44 §4: `qam_mod` and `qam_demod` (registration with their kernels, the costs
//! `4 span` and `4 span + (55 + 10 or 18) / sps + 8 equalizer_taps / sps` of D992, the rates and
//! delays, the required parameters, `rate_unsupported` with the hint to resample,
//! `invalid_value` for the orders 17 to 63 and an even `equalizer_taps`, the type checks, the
//! kernels without `equalize` for `equalizer_taps` 0), `frame_decode` `rotations` `four` (cost
//! 4, D993), and push-mode `qam_mod` and `qam_demod` in pieces of 1 to 1024 samples (seed 607)
//! equal to the blocks bit for bit.
#![allow(
    clippy::needless_pass_by_value,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::large_stack_frames
)]

mod common;

use sdrk::{C32, Delay, Rate};
use sdrk_blocks::{Block, QamDemodBlock, QamDemodConfig, QamModBlock, QamModConfig};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Pipeline, Registry, VecStorage};
use sdrk_testkit::Pcg32;
use serde_json::{Value, json};

use common::build;

/// Seed of the chunk plans and the random data of M44 (test 8).
const SEED: u64 = 607;

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

fn first_error(doc: &Value) -> (String, DiagCode, String) {
    let report = validate(doc);
    let e = report.errors.first().expect("an error");
    (e.path.clone(), e.code, e.message.clone())
}

fn soft_bits(n: usize, stream: u64) -> Vec<f32> {
    let mut rng = Pcg32::new(SEED, stream);
    (0..n)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect()
}

#[test]
fn the_qam_kinds_are_registered_with_their_kernels() {
    let registry = Registry::standard();
    for (kind, kernels) in [
        ("qam_mod", &["fir_interpolate"][..]),
        (
            "qam_demod",
            &["fir", "symbol_sync", "equalize", "costas"][..],
        ),
    ] {
        let f = registry.factory(kind).unwrap();
        assert_eq!(f.kernels(), kernels, "{kind}");
        assert_eq!(f.description().kind, kind);
    }
    let f = registry.factory("qam_demod").unwrap();
    let raw = |v: Value| v.as_object().unwrap().clone();
    assert_eq!(
        f.kernels_for(&raw(
            json!({ "symbol_rate_hz": 25_000, "equalizer_taps": 0 })
        )),
        ["fir", "symbol_sync", "costas"]
    );
    for params in [
        json!({ "symbol_rate_hz": 25_000 }),
        json!({ "symbol_rate_hz": 25_000, "equalizer_taps": 15 }),
        json!({ "symbol_rate_hz": 25_000, "equalizer_taps": "x" }),
    ] {
        assert_eq!(
            f.kernels_for(&raw(params.clone())),
            ["fir", "symbol_sync", "equalize", "costas"],
            "{params}"
        );
    }
    assert_eq!(registry.descriptions().count(), 68);
}

#[test]
fn qam_mod_costs_4_span_and_multiplies_the_rate() {
    // 48000 bits per second: `48000 sps / bits` samples per second
    for (order, sps, span, rate_out) in [
        (16, 8, 10, 96_000u64),
        (64, 8, 10, 64_000),
        (16, 2, 32, 24_000),
    ] {
        let report = validate(&single(
            "qam_mod",
            "f32",
            48_000,
            json!({ "order": order, "samples_per_symbol": sps, "span": span }),
        ));
        assert!(report.is_ok(), "{}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 4.0 * span as f64);
        let node = &report.nodes[0];
        assert_eq!(node.rate_out, Rate::hz(rate_out).unwrap());
        let bps = if order == 64 { 6 } else { 4 };
        assert_eq!(
            node.delay_samples_in.reduced(),
            Delay::new(span * bps, 2).unwrap().reduced()
        );
    }
    // samples_per_symbol is required; only f32 soft bits
    assert!(!validate(&single("qam_mod", "f32", 48_000, json!({}))).is_ok());
    for ty in ["c32", "cq15", "q15"] {
        let (_, code, _) = first_error(&single(
            "qam_mod",
            ty,
            1000,
            json!({ "samples_per_symbol": 4 }),
        ));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
    let (path, code, _) = first_error(&single(
        "qam_mod",
        "f32",
        1000,
        json!({ "samples_per_symbol": 4, "order": 32 }),
    ));
    assert_eq!(
        (path.as_str(), code),
        ("nodes.n.params.order", DiagCode::InvalidValue)
    );
    for bad in [
        json!({ "samples_per_symbol": 65 }),
        json!({ "samples_per_symbol": 4, "order": 4 }),
        json!({ "samples_per_symbol": 4, "span": 3 }),
        json!({ "samples_per_symbol": 4, "rolloff": 1.5 }),
    ] {
        assert!(
            !validate(&single("qam_mod", "f32", 1000, bad.clone())).is_ok(),
            "{bad}"
        );
    }
}

#[test]
fn qam_demod_costs_and_rate_rule() {
    for (order, fs, sym, span, taps) in [
        (16, 200_000u64, 25_000.0, 10u64, 11u64),
        (64, 48_000, 1200.0, 16, 15),
        (16, 9600, 4800.0, 4, 0),
    ] {
        let report = validate(&single(
            "qam_demod",
            "c32",
            fs,
            json!({ "order": order, "symbol_rate_hz": sym, "span": span, "equalizer_taps": taps }),
        ));
        assert!(report.is_ok(), "{}", report.to_json());
        let sps = fs as f64 / sym;
        let slice = if order == 64 { 18.0 } else { 10.0 };
        assert_eq!(
            report.cost_per_input_sample,
            4.0 * span as f64 + (55.0 + slice) / sps + 8.0 * taps as f64 / sps
        );
        let node = &report.nodes[0];
        let bps = if order == 64 { 6 } else { 4 };
        assert_eq!(node.rate_out, Rate::hz(sym as u64 * bps).unwrap());
        let n = sps as u64;
        assert_eq!(
            node.delay_samples_in.reduced(),
            Delay::new(span * n + 2 + taps.saturating_sub(1) * n, 2)
                .unwrap()
                .reduced()
        );
    }
    // the symbol rate is required
    assert!(!validate(&single("qam_demod", "c32", 200_000, json!({}))).is_ok());
    for sym in [26_666.0, 120_000.0, 150_000.0, 200_000.0 / 1.99] {
        let (path, code, message) = first_error(&single(
            "qam_demod",
            "c32",
            200_000,
            json!({ "symbol_rate_hz": sym }),
        ));
        assert_eq!(code, DiagCode::RateUnsupported, "{sym}");
        assert_eq!(path, "nodes.n.params.symbol_rate_hz");
        assert!(message.contains("resample"), "{message}");
    }
    for (params, name) in [
        (json!({ "symbol_rate_hz": 25_000, "order": 20 }), "order"),
        (
            json!({ "symbol_rate_hz": 25_000, "equalizer_taps": 12 }),
            "equalizer_taps",
        ),
    ] {
        let (path, code, _) = first_error(&single("qam_demod", "c32", 200_000, params));
        assert_eq!(
            (path, code),
            (format!("nodes.n.params.{name}"), DiagCode::InvalidValue)
        );
    }
    for ty in ["cq15", "f32", "q15"] {
        let (_, code, _) = first_error(&single(
            "qam_demod",
            ty,
            200_000,
            json!({ "symbol_rate_hz": 25_000 }),
        ));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
    for bad in [
        json!({ "symbol_rate_hz": 25_000, "carrier_bw": 0.3 }),
        json!({ "symbol_rate_hz": 25_000, "sync_bw": 0.2 }),
        json!({ "symbol_rate_hz": 25_000, "equalizer_taps": 65 }),
        json!({ "symbol_rate_hz": 25_000, "max_offset": 0.1 }),
    ] {
        assert!(
            !validate(&single("qam_demod", "c32", 200_000, bad.clone())).is_ok(),
            "{bad}"
        );
    }
}

#[test]
fn frame_decode_four_costs_4() {
    let doc = |params: Value| {
        json!({
            "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
            "inputs": { "in": { "type": "f32", "rate_hz": 100_000 } },
            "nodes": { "n": { "kind": "frame_decode", "params": params } },
            "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
            "outputs": { "out": { "type": "packets" } }
        })
    };
    for (params, cost) in [
        (json!({}), 1.0),
        (json!({ "rotations": "both" }), 1.0),
        (json!({ "rotations": "four", "bits_per_symbol": 4 }), 4.0),
    ] {
        let report = validate(&doc(params.clone()));
        assert!(report.is_ok(), "{params}: {}", report.to_json());
        assert_eq!(report.cost_per_input_sample, cost, "{params}");
    }
    let (path, code, _) = first_error(&doc(json!({ "rotations": "four", "bits_per_symbol": 3 })));
    assert_eq!(
        (path.as_str(), code),
        ("nodes.n.params.bits_per_symbol", DiagCode::InvalidValue)
    );
}

/// Pushes `input` in pieces of 1 to 1024 samples (seed 607, stream `stream`) and reads the
/// stream output `out` after every push.
fn push_all<I, O: Copy + Default + sdrk::Sample>(
    p: &mut Pipeline,
    input: &[I],
    wrap: for<'a> fn(&'a [I]) -> InputData<'a>,
    stream: u64,
) -> Vec<O> {
    let mut rng = Pcg32::new(SEED, stream);
    let mut got = Vec::new();
    let mut buf = vec![O::default(); 1 << 16];
    let mut at = 0;
    while at < input.len() {
        let n = (1 + rng.below(1024) as usize).min(input.len() - at);
        p.push("in", wrap(&input[at..at + n])).unwrap();
        at += n;
        loop {
            let k = p.read_stream::<O>("out", &mut buf).unwrap();
            if k == 0 {
                break;
            }
            got.extend_from_slice(&buf[..k]);
        }
    }
    got
}

#[test]
fn qam_mod_and_qam_demod_kinds_equal_the_blocks() {
    for (order, stream) in [(16u32, 1u64), (64, 5)] {
        let m = QamModConfig {
            order,
            samples_per_symbol: 8,
            ..QamModConfig::default()
        };
        let k = m.bits_per_symbol();
        let bits = soft_bits(6000 * k, stream);
        let bit_rate = Rate::hz(25_000 * k as u64).unwrap();
        let mut block = QamModBlock::<VecStorage>::new(&m, bit_rate, &mut VecStorage).unwrap();
        let mut want = vec![C32::default(); bits.len() / k * 8];
        let p = block.process(&bits, &mut want).unwrap();
        assert_eq!(p.consumed, bits.len());
        let mut tx = build(
            &single(
                "qam_mod",
                "f32",
                25_000 * k as u64,
                json!({ "order": order, "samples_per_symbol": 8 }),
            )
            .to_string(),
        );
        let sig: Vec<C32> = push_all(&mut tx, &bits, |x| InputData::F32(x), stream + 1);
        assert_eq!(
            sig, want,
            "{order}: qam_mod pipeline differs from the block"
        );

        let d = QamDemodConfig {
            order,
            symbol_rate_hz: 25_000.0,
            ..QamDemodConfig::default()
        };
        let fs = Rate::hz(200_000).unwrap();
        let mut rng = Pcg32::new(SEED, stream + 2);
        let noisy: Vec<C32> = sig
            .iter()
            .map(|x| {
                C32::new(
                    x.re + 0.02 * rng.normal() as f32,
                    x.im + 0.02 * rng.normal() as f32,
                )
            })
            .collect();
        let mut block = QamDemodBlock::<VecStorage>::new(&d, fs, &mut VecStorage).unwrap();
        let mut want = vec![0.0f32; noisy.len()];
        let p = block.process(&noisy, &mut want).unwrap();
        assert_eq!(p.consumed, noisy.len());
        want.truncate(p.produced);
        let mut rx = build(
            &single(
                "qam_demod",
                "c32",
                200_000,
                json!({ "order": order, "symbol_rate_hz": 25_000 }),
            )
            .to_string(),
        );
        let got: Vec<f32> = push_all(&mut rx, &noisy, |x| InputData::C32(x), stream + 3);
        assert_eq!(got.len(), want.len(), "{order}");
        assert!(
            got.iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{order}: qam_demod pipeline differs from the block"
        );
        assert!(rx.set_param("n", "carrier_bw", json!(0.01)).is_err());
    }
}
