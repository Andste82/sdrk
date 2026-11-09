//! The kinds of M40: `costas`, `psk_mod` and `psk_demod` (registration with their kernels, the
//! costs 30, `4 span` and `4 span + 40 / sps`, the rates and delays, the required parameters,
//! `rate_unsupported` with the hint to resample, `invalid_value` for the order 3, the type
//! checks), the parameter `rotations` of `frame_decode` (M44, formerly `polarity`), a
//! push-mode `psk_mod` and `psk_demod` that equal the blocks, and packets through
//! `frame_encode` -> `psk_mod` -> an inverted channel -> `psk_demod` -> `frame_decode` with
//! `rotations` both. M40.1: `order` 8, the
//! `variant` (`invalid_value` for the combinations that do not fit, the costs and delays) and
//! push-mode 8PSK, pi/4-DQPSK and OQPSK in pieces of seed 564 equal to the blocks.
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
use sdrk_blocks::{Block, PskDemodBlock, PskDemodConfig, PskModBlock, PskModConfig, PskVariant};
use sdrk_pipeline::{
    BuildOptions, DiagCode, InputData, Packet, PacketFlags, Pipeline, Registry, VecStorage,
};
use sdrk_testkit::Pcg32;
use serde_json::{Value, json};

use common::build;

/// Seed of the random data of the M40 pipeline tests (the chunk-invariance seed of test 6).
const SEED: u64 = 559;

/// Seed of the chunk plans and the random data of the M40.1 pipeline tests (test 5).
const M40_1_SEED: u64 = 564;

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
fn the_kinds_are_registered_with_their_kernels() {
    let registry = Registry::standard();
    for (kind, kernels) in [
        ("costas", &["costas"][..]),
        ("psk_mod", &["fir_interpolate"][..]),
        ("psk_demod", &["fir", "symbol_sync", "costas"][..]),
    ] {
        let f = registry.factory(kind).unwrap();
        assert_eq!(f.kernels(), kernels, "{kind}");
        assert_eq!(f.description().kind, kind);
    }
    assert_eq!(registry.descriptions().count(), 68);
}

/// A `psk_demod` node of 8 samples per symbol at 200 kHz with `params` and `backend`.
fn psk_demod_with_backend(params: Value, backend: Value) -> Value {
    let mut doc = single("psk_demod", "c32", 200_000, params);
    doc["nodes"]["n"]["backend"] = backend;
    doc
}

/// The messages of the `backend_unavailable` diagnostics (errors and warnings) of `doc`.
fn backend_messages(doc: &Value) -> Vec<String> {
    let report = validate(doc);
    report
        .errors
        .iter()
        .chain(report.warnings.iter())
        .filter(|d| d.code == DiagCode::BackendUnavailable)
        .map(|d| d.message.clone())
        .collect()
}

/// D689: the backend check of `psk_demod` `offset` looks at the kernels it runs (`rotate`
/// instead of `costas`, D686); the other variants keep the list of the kind.
#[test]
fn psk_demod_offset_backend_check_uses_rotate_not_costas() {
    let registry = Registry::standard();
    let factory = registry.factory("psk_demod").unwrap();
    let raw = |v: Value| v.as_object().unwrap().clone();
    let offset = raw(json!({ "order": 4, "symbol_rate_hz": 25_000, "variant": "offset" }));
    assert_eq!(
        factory.kernels_for(&offset),
        ["fir", "symbol_sync", "rotate"]
    );
    for params in [
        json!({ "order": 4, "symbol_rate_hz": 25_000 }),
        json!({ "order": 4, "symbol_rate_hz": 25_000, "variant": "plain" }),
        json!({ "order": 4, "symbol_rate_hz": 25_000, "variant": "pi4", "differential": true }),
        json!({ "order": 8, "symbol_rate_hz": 25_000 }),
        json!({ "order": 4, "symbol_rate_hz": 25_000, "variant": 3 }),
    ] {
        assert_eq!(
            factory.kernels_for(&raw(params.clone())),
            ["fir", "symbol_sync", "costas"],
            "{params}"
        );
    }

    let params = json!({ "order": 4, "symbol_rate_hz": 25_000, "variant": "offset" });
    let rotate = sdrk::backend_for("rotate", sdrk::SampleType::C32);
    for key in ["require", "prefer"] {
        // Requiring the backend of `costas` reports `rotate` whenever `rotate` runs elsewhere.
        let costas = sdrk::backend_for("costas", sdrk::SampleType::C32);
        let messages = backend_messages(&psk_demod_with_backend(
            params.clone(),
            json!({ key: costas }),
        ));
        let rotate_reported = messages.iter().any(|m| m.contains("rotate ("));
        assert_eq!(rotate_reported, rotate != costas, "{key}: {messages:?}");
        // `costas` never appears for `offset`, whatever is required.
        for wanted in ["simd", "fallback"] {
            let messages = backend_messages(&psk_demod_with_backend(
                params.clone(),
                json!({ key: wanted }),
            ));
            assert!(
                messages.iter().all(|m| !m.contains("costas")),
                "{key} {wanted}: {messages:?}"
            );
        }
    }
    // Plain QPSK still reports `costas` when another backend is required.
    let other = if sdrk::backend_for("costas", sdrk::SampleType::C32) == "simd" {
        "fallback"
    } else {
        "simd"
    };
    let messages = backend_messages(&psk_demod_with_backend(
        json!({ "order": 4, "symbol_rate_hz": 25_000 }),
        json!({ "require": other }),
    ));
    assert!(
        messages.iter().any(|m| m.contains("costas (")),
        "{messages:?}"
    );
}

#[test]
fn costas_costs_30_and_keeps_the_rate() {
    let report = validate(&single(
        "costas",
        "c32",
        25_000,
        json!({ "constellation": "qpsk" }),
    ));
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.cost_per_input_sample, 30.0);
    let node = &report.nodes[0];
    assert_eq!(node.rate_out, Rate::hz(25_000).unwrap());
    assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO.reduced());
    for ty in ["cq15", "f32", "q15"] {
        let (_, code, _) = first_error(&single("costas", ty, 1000, json!({})));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
    // M44 §3: the QAM constellations cost 45 per symbol (D1206), the PSK ones 30
    for (c, cost) in [
        ("bpsk", 30.0),
        ("8psk", 30.0),
        ("qam16", 45.0),
        ("qam64", 45.0),
    ] {
        let report = validate(&single(
            "costas",
            "c32",
            25_000,
            json!({ "constellation": c }),
        ));
        assert!(report.is_ok(), "{c}: {}", report.to_json());
        assert_eq!(report.cost_per_input_sample, cost, "{c}");
    }
    // the former `order` is gone (M44 §3, a hard break), `constellation` is a choice
    assert!(!validate(&single("costas", "c32", 1000, json!({ "order": 4 }))).is_ok());
    let (path, _, _) = first_error(&single(
        "costas",
        "c32",
        1000,
        json!({ "constellation": "qam256" }),
    ));
    assert_eq!(path.as_str(), "nodes.n.params.constellation");
    for bad in [
        json!({ "constellation": 4 }),
        json!({ "loop_bw": 0.3 }),
        json!({ "damping": 0 }),
        json!({ "max_offset": 0.3 }),
    ] {
        assert!(
            !validate(&single("costas", "c32", 1000, bad.clone())).is_ok(),
            "{bad}"
        );
    }
}

#[test]
fn psk_mod_costs_4_span_and_multiplies_the_rate() {
    for (order, sps, span, rate_out) in [
        (2, 8, 10, 400_000u64),
        (4, 8, 10, 200_000),
        (4, 2, 32, 50_000),
    ] {
        let report = validate(&single(
            "psk_mod",
            "f32",
            50_000,
            json!({ "order": order, "samples_per_symbol": sps, "span": span }),
        ));
        assert!(report.is_ok(), "{}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 4.0 * span as f64);
        let node = &report.nodes[0];
        assert_eq!(node.rate_out, Rate::hz(rate_out).unwrap());
        let bps = if order == 4 { 2 } else { 1 };
        assert_eq!(
            node.delay_samples_in.reduced(),
            Delay::new(span * bps, 2).unwrap().reduced()
        );
    }
    // samples_per_symbol is required; only f32 soft bits
    assert!(!validate(&single("psk_mod", "f32", 50_000, json!({}))).is_ok());
    for ty in ["c32", "cq15", "q15"] {
        let (_, code, _) = first_error(&single(
            "psk_mod",
            ty,
            1000,
            json!({ "samples_per_symbol": 4 }),
        ));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
    let (path, code, _) = first_error(&single(
        "psk_mod",
        "f32",
        1000,
        json!({ "samples_per_symbol": 4, "order": 3 }),
    ));
    assert_eq!(
        (path.as_str(), code),
        ("nodes.n.params.order", DiagCode::InvalidValue)
    );
    for bad in [
        json!({ "samples_per_symbol": 65 }),
        json!({ "samples_per_symbol": 4, "span": 3 }),
        json!({ "samples_per_symbol": 4, "rolloff": 1.5 }),
        json!({ "samples_per_symbol": 4, "differential": 1 }),
    ] {
        assert!(
            !validate(&single("psk_mod", "f32", 1000, bad.clone())).is_ok(),
            "{bad}"
        );
    }
}

#[test]
fn psk_demod_costs_and_rate_rule() {
    for (order, fs, sym, span) in [
        (4, 200_000u64, 25_000.0, 10u64),
        (2, 48_000, 1200.0, 16),
        (4, 9600, 4800.0, 4),
    ] {
        let report = validate(&single(
            "psk_demod",
            "c32",
            fs,
            json!({ "order": order, "symbol_rate_hz": sym, "span": span }),
        ));
        assert!(report.is_ok(), "{}", report.to_json());
        let sps = fs as f64 / sym;
        assert_eq!(report.cost_per_input_sample, 4.0 * span as f64 + 40.0 / sps);
        let node = &report.nodes[0];
        let bps = if order == 4 { 2 } else { 1 };
        assert_eq!(node.rate_out, Rate::hz(sym as u64 * bps).unwrap());
        assert_eq!(
            node.delay_samples_in.reduced(),
            Delay::new(span * sps as u64 + 2, 2).unwrap().reduced()
        );
    }
    // the symbol rate is required
    assert!(!validate(&single("psk_demod", "c32", 200_000, json!({}))).is_ok());
    // not within 1 % of an integer number of samples per symbol: rate_unsupported, with the
    // hint to resample; also within 1 % of 2 but below 2 samples per symbol (1.99, D625)
    for sym in [26_666.0, 120_000.0, 150_000.0, 200_000.0 / 1.99] {
        let (path, code, message) = first_error(&single(
            "psk_demod",
            "c32",
            200_000,
            json!({ "symbol_rate_hz": sym }),
        ));
        assert_eq!(code, DiagCode::RateUnsupported, "{sym}");
        assert_eq!(path, "nodes.n.params.symbol_rate_hz");
        assert!(message.contains("resample"), "{message}");
    }
    let (path, code, _) = first_error(&single(
        "psk_demod",
        "c32",
        200_000,
        json!({ "symbol_rate_hz": 25_000, "order": 3 }),
    ));
    assert_eq!(
        (path.as_str(), code),
        ("nodes.n.params.order", DiagCode::InvalidValue)
    );
    for ty in ["cq15", "f32", "q15"] {
        let (_, code, _) = first_error(&single(
            "psk_demod",
            ty,
            200_000,
            json!({ "symbol_rate_hz": 25_000 }),
        ));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
}

#[test]
fn frame_decode_has_the_rotations_choice() {
    let doc = |rotations: &str, bits: i64| {
        json!({
            "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
            "inputs": { "in": { "type": "f32", "rate_hz": 50_000 } },
            "nodes": { "n": { "kind": "frame_decode", "params": {
                "rotations": rotations, "bits_per_symbol": bits
            } } },
            "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
            "outputs": { "out": { "type": "packets" } }
        })
    };
    for (rotations, bits) in [
        ("normal", 2),
        ("both", 1),
        ("both", 3),
        ("four", 2),
        ("four", 6),
    ] {
        let report = validate(&doc(rotations, bits));
        assert!(report.is_ok(), "{rotations} {bits}: {}", report.to_json());
    }
    assert!(!validate(&doc("inverted", 2)).is_ok());
    // four needs an even count; 7 is outside 1 to 6
    for (rotations, bits) in [("four", 3), ("normal", 7), ("four", 0)] {
        assert!(
            !validate(&doc(rotations, bits)).is_ok(),
            "{rotations} {bits}"
        );
    }
    // the former name is gone (M44, no alias)
    let old = json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
        "inputs": { "in": { "type": "f32", "rate_hz": 50_000 } },
        "nodes": { "n": { "kind": "frame_decode", "params": { "polarity": "both" } } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": "packets" } }
    });
    assert!(!validate(&old).is_ok());
}

/// Pushes `input` in pieces of 1 to 1024 samples (seed 559, stream `stream`) and reads the
/// stream output `out` after every push.
fn push_all<I, O: Copy + Default + sdrk::Sample>(
    p: &mut Pipeline,
    input: &[I],
    wrap: for<'a> fn(&'a [I]) -> InputData<'a>,
    stream: u64,
) -> Vec<O> {
    push_all_seed(p, input, wrap, SEED, stream)
}

/// As `push_all` with the seed of the piece sizes.
fn push_all_seed<I, O: Copy + Default + sdrk::Sample>(
    p: &mut Pipeline,
    input: &[I],
    wrap: for<'a> fn(&'a [I]) -> InputData<'a>,
    seed: u64,
    stream: u64,
) -> Vec<O> {
    let mut rng = Pcg32::new(seed, stream);
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
fn psk_mod_and_psk_demod_kinds_equal_the_blocks() {
    let m = PskModConfig {
        order: 4,
        samples_per_symbol: 8,
        ..PskModConfig::default()
    };
    let bits = soft_bits(20_000, 1);
    let mut block =
        PskModBlock::<VecStorage>::new(&m, Rate::hz(50_000).unwrap(), &mut VecStorage).unwrap();
    let mut want = vec![C32::default(); bits.len() * 4];
    let p = block.process(&bits, &mut want).unwrap();
    assert_eq!(p.consumed, bits.len());
    let mut tx = build(
        &single(
            "psk_mod",
            "f32",
            50_000,
            json!({ "order": 4, "samples_per_symbol": 8 }),
        )
        .to_string(),
    );
    let sig: Vec<C32> = push_all(&mut tx, &bits, |x| InputData::F32(x), 2);
    assert_eq!(sig, want, "psk_mod pipeline differs from the block");

    let d = PskDemodConfig {
        order: 4,
        symbol_rate_hz: 25_000.0,
        ..PskDemodConfig::default()
    };
    let fs = Rate::hz(200_000).unwrap();
    let mut rng = Pcg32::new(SEED, 3);
    let noisy: Vec<C32> = sig
        .iter()
        .map(|x| {
            C32::new(
                x.re + 0.05 * rng.normal() as f32,
                x.im + 0.05 * rng.normal() as f32,
            )
        })
        .collect();
    let mut block = PskDemodBlock::<VecStorage>::new(&d, fs, &mut VecStorage).unwrap();
    let mut want = vec![0.0f32; noisy.len()];
    let p = block.process(&noisy, &mut want).unwrap();
    assert_eq!(p.consumed, noisy.len());
    want.truncate(p.produced);
    let mut rx = build(
        &single(
            "psk_demod",
            "c32",
            200_000,
            json!({ "order": 4, "symbol_rate_hz": 25_000 }),
        )
        .to_string(),
    );
    let soft: Vec<f32> = push_all(&mut rx, &noisy, |x| InputData::C32(x), 4);
    assert_eq!(soft.len(), want.len());
    assert!(
        soft.iter()
            .zip(&want)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "psk_demod pipeline differs from the block"
    );
    assert!(rx.set_param("n", "carrier_bw", json!(0.01)).is_err());
}

#[test]
fn inverted_bpsk_frames_are_found_with_rotations_both() {
    // transmitter: packets -> frame_encode -> psk_mod (BPSK, 4 samples per symbol)
    let tx_doc = json!({
        "sdrk": 1, "name": "tx", "settings": { "precision": "f32", "max_chunk": 4096 },
        "inputs": { "msg": { "type": "packets", "rate_hz": 10_000 } },
        "nodes": {
            "enc": { "kind": "frame_encode", "params": { "preamble_bytes": 32 } },
            "mod": { "kind": "psk_mod", "params": { "order": 2, "samples_per_symbol": 4 } }
        },
        "edges": [
            { "from": "msg", "to": "enc" }, { "from": "enc", "to": "mod" },
            { "from": "mod", "to": "rf" }
        ],
        "outputs": { "rf": { "type": "stream" } }
    });
    let mut tx = build(&tx_doc.to_string());
    let mut rng = Pcg32::new(SEED, 5);
    // 21 packets: the last one only pushes the pulse tail of the 20th through the transmit
    // filter (push mode has no idle signal)
    let packets: Vec<Packet> = (0..21)
        .map(|_| {
            let len = 1 + rng.below(60) as usize;
            let data: Vec<u8> = (0..len).map(|_| rng.next_u32() as u8).collect();
            Packet::new(0, PacketFlags::NONE, &data).unwrap()
        })
        .collect();
    let mut sig = Vec::new();
    let mut buf = vec![C32::default(); 1 << 16];
    for p in &packets {
        tx.push_packets("msg", core::slice::from_ref(p)).unwrap();
        loop {
            let n = tx.read_stream::<C32>("rf", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            sig.extend_from_slice(&buf[..n]);
        }
    }
    // the channel turns the carrier by 180 degrees; a tail flushes the filters
    let mut rx_in: Vec<C32> = sig.iter().map(|x| C32::new(-x.re, -x.im)).collect();
    rx_in.extend(std::iter::repeat_n(C32::default(), 400));
    for (rotations, expected) in [("normal", 0usize), ("both", 20)] {
        let rx_doc = json!({
            "sdrk": 1, "name": "rx", "settings": { "precision": "f32", "max_chunk": 4096 },
            "inputs": { "in": { "type": "c32", "rate_hz": 40_000 } },
            "nodes": {
                "demod": { "kind": "psk_demod", "params": { "order": 2, "symbol_rate_hz": 10_000 } },
                "dec": { "kind": "frame_decode", "params": { "rotations": rotations } }
            },
            "edges": [
                { "from": "in", "to": "demod" }, { "from": "demod", "to": "dec" },
                { "from": "dec", "to": "out" }
            ],
            "outputs": { "out": { "type": "packets", "queue": 256 } }
        });
        let mut rx = build(&rx_doc.to_string());
        let mut got = Vec::new();
        let mut pk = vec![Packet::default(); 64];
        for chunk in rx_in.chunks(4096) {
            rx.push("in", InputData::C32(chunk)).unwrap();
            loop {
                let n = rx.read_packets("out", &mut pk).unwrap();
                if n == 0 {
                    break;
                }
                got.extend_from_slice(&pk[..n]);
            }
        }
        // the 21st frame may or may not survive its truncated tail
        if expected == 0 {
            assert!(got.is_empty(), "rotations {rotations}");
        } else {
            assert!(
                got.len() >= expected,
                "rotations {rotations}: {}",
                got.len()
            );
        }
        for (g, s) in got.iter().zip(&packets) {
            assert_eq!(g.data(), s.data());
            assert_eq!(g.flags, PacketFlags::CRC_OK);
        }
    }
}

/// The constellation `8psk` of `slice` (M40.1): `c32` only, three soft bits per symbol, cost 16
/// (D644); `f32` is `invalid_value` at `params.constellation`.
#[test]
fn slice_takes_8psk() {
    let report = validate(&single(
        "slice",
        "c32",
        4800,
        json!({ "constellation": "8psk" }),
    ));
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.nodes[0].rate_out, Rate::hz(14_400).unwrap());
    assert_eq!(report.cost_per_input_sample, 16.0);
    let (path, code, _) = first_error(&single(
        "slice",
        "f32",
        4800,
        json!({ "constellation": "8psk" }),
    ));
    assert_eq!(code, DiagCode::InvalidValue);
    assert_eq!(path, "nodes.n.params.constellation");
}

/// M40.1 §4: `order` 8 on all three kinds (8PSK: `psk_mod` ratio `sps / 3`, `psk_demod` three
/// soft bits per symbol and the cost `4 span + 56 / sps`, D651), the `variant` of `psk_mod`
/// and `psk_demod` (`offset` adds half a symbol to the delay) and `invalid_value` at
/// `params.variant` for the combinations that do not fit (D648).
#[test]
fn psk_kinds_take_order_8_and_the_variants() {
    let report = validate(&single(
        "costas",
        "c32",
        25_000,
        json!({ "constellation": "8psk" }),
    ));
    assert!(report.is_ok(), "{}", report.to_json());
    // psk_mod: rate, delay and cost
    for (params, rate_out, delay) in [
        (
            json!({ "order": 8, "samples_per_symbol": 8 }),
            25_600u64,
            Delay::new(30, 2),
        ),
        (
            json!({ "order": 4, "samples_per_symbol": 8, "variant": "offset" }),
            38_400,
            Delay::new(22, 2),
        ),
        (
            json!({ "order": 4, "samples_per_symbol": 8, "differential": true, "variant": "pi4" }),
            38_400,
            Delay::new(20, 2),
        ),
    ] {
        let report = validate(&single("psk_mod", "f32", 9600, params.clone()));
        assert!(report.is_ok(), "{params}: {}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 40.0, "{params}");
        let node = &report.nodes[0];
        assert_eq!(node.rate_out, Rate::hz(rate_out).unwrap(), "{params}");
        assert_eq!(
            node.delay_samples_in.reduced(),
            delay.unwrap().reduced(),
            "{params}"
        );
    }
    // psk_demod: 8PSK cost and rate; OQPSK delay
    let report = validate(&single(
        "psk_demod",
        "c32",
        200_000,
        json!({ "order": 8, "symbol_rate_hz": 25_000 }),
    ));
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.cost_per_input_sample, 40.0 + 56.0 / 8.0);
    assert_eq!(report.nodes[0].rate_out, Rate::hz(75_000).unwrap());
    let report = validate(&single(
        "psk_demod",
        "c32",
        200_000,
        json!({ "order": 4, "symbol_rate_hz": 25_000, "variant": "offset" }),
    ));
    assert!(report.is_ok(), "{}", report.to_json());
    // the derotation of the OQPSK carrier loop adds 6 per sample (D686)
    assert_eq!(report.cost_per_input_sample, 40.0 + 40.0 / 8.0 + 6.0);
    assert_eq!(
        report.nodes[0].delay_samples_in.reduced(),
        Delay::new(82, 2).unwrap().reduced() // as plain QPSK (D658)
    );
    // the combinations that do not fit: invalid_value at params.variant
    for params in [
        json!({ "order": 4, "samples_per_symbol": 8, "variant": "pi4" }),
        json!({ "order": 2, "samples_per_symbol": 8, "differential": true, "variant": "pi4" }),
        json!({ "order": 8, "samples_per_symbol": 8, "differential": true, "variant": "pi4" }),
        json!({ "order": 4, "samples_per_symbol": 7, "variant": "offset" }),
        json!({ "order": 4, "samples_per_symbol": 8, "differential": true, "variant": "offset" }),
        json!({ "order": 8, "samples_per_symbol": 8, "variant": "offset" }),
    ] {
        let (path, code, _) = first_error(&single("psk_mod", "f32", 9600, params.clone()));
        assert_eq!(
            (path.as_str(), code),
            ("nodes.n.params.variant", DiagCode::InvalidValue),
            "{params}"
        );
    }
    for (fs, params) in [
        (
            200_000,
            json!({ "order": 4, "symbol_rate_hz": 25_000, "variant": "pi4" }),
        ),
        (
            200_000,
            json!({ "order": 8, "symbol_rate_hz": 25_000, "differential": true, "variant": "pi4" }),
        ),
        // 7 samples per symbol: the rate is fine, the odd count is the variant's error
        (
            175_000,
            json!({ "order": 4, "symbol_rate_hz": 25_000, "variant": "offset" }),
        ),
        (
            200_000,
            json!({ "order": 4, "symbol_rate_hz": 25_000, "differential": true, "variant": "offset" }),
        ),
    ] {
        let (path, code, _) = first_error(&single("psk_demod", "c32", fs, params.clone()));
        assert_eq!(
            (path.as_str(), code),
            ("nodes.n.params.variant", DiagCode::InvalidValue),
            "{params}"
        );
    }
    assert!(
        !validate(&single(
            "psk_mod",
            "f32",
            9600,
            json!({ "samples_per_symbol": 8, "variant": "oqpsk" })
        ))
        .is_ok()
    );
}

/// Test 5 of M40.1 in the pipeline: push-mode `psk_mod` and `psk_demod` with 8PSK, pi/4-DQPSK
/// and OQPSK in pieces of 1 to 1024 samples (seed 564) equal the blocks bit for bit.
#[test]
fn psk_variant_kinds_equal_the_blocks_in_pieces() {
    for (i, (order, differential, variant, name)) in [
        (8u32, false, PskVariant::Plain, "plain"),
        (8, true, PskVariant::Plain, "plain"),
        (4, true, PskVariant::Pi4, "pi4"),
        (4, false, PskVariant::Offset, "offset"),
    ]
    .into_iter()
    .enumerate()
    {
        let m = PskModConfig {
            order,
            samples_per_symbol: 8,
            differential,
            variant,
            ..PskModConfig::default()
        };
        let bps = m.bits_per_symbol();
        let mut rng = Pcg32::new(M40_1_SEED, 10 + i as u64);
        let bits: Vec<f32> = (0..6000 * bps)
            .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
            .collect();
        let bit_rate = 25_000 * bps as u64;
        let mut block =
            PskModBlock::<VecStorage>::new(&m, Rate::hz(bit_rate).unwrap(), &mut VecStorage)
                .unwrap();
        let mut want = vec![C32::default(); 6000 * 8];
        let p = block.process(&bits, &mut want).unwrap();
        assert_eq!(p.consumed, bits.len());
        let params = json!({
            "order": order, "samples_per_symbol": 8, "differential": differential,
            "variant": name
        });
        let mut tx = build(&single("psk_mod", "f32", bit_rate, params).to_string());
        let sig: Vec<C32> = push_all_seed(
            &mut tx,
            &bits,
            |x| InputData::F32(x),
            M40_1_SEED,
            20 + i as u64,
        );
        assert_eq!(sig, want, "psk_mod {order} {name}");

        let d = PskDemodConfig {
            order,
            symbol_rate_hz: 25_000.0,
            differential,
            variant,
            ..PskDemodConfig::default()
        };
        let fs = Rate::hz(200_000).unwrap();
        let noisy: Vec<C32> = sig
            .iter()
            .map(|x| {
                C32::new(
                    x.re + 0.05 * rng.normal() as f32,
                    x.im + 0.05 * rng.normal() as f32,
                )
            })
            .collect();
        let mut block = PskDemodBlock::<VecStorage>::new(&d, fs, &mut VecStorage).unwrap();
        let mut want = vec![0.0f32; noisy.len()];
        let p = block.process(&noisy, &mut want).unwrap();
        assert_eq!(p.consumed, noisy.len());
        want.truncate(p.produced);
        assert!(want.len() > 5900 * bps, "{order} {name}");
        let params = json!({
            "order": order, "symbol_rate_hz": 25_000, "differential": differential,
            "variant": name
        });
        let mut rx = build(&single("psk_demod", "c32", 200_000, params).to_string());
        let soft: Vec<f32> = push_all_seed(
            &mut rx,
            &noisy,
            |x| InputData::C32(x),
            M40_1_SEED,
            30 + i as u64,
        );
        assert_eq!(soft.len(), want.len(), "{order} {name}");
        assert!(
            soft.iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "psk_demod {order} {name}: pipeline differs from the block"
        );
    }
}
