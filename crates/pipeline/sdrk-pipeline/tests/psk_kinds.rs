//! The kinds of M40: `costas`, `psk_mod` and `psk_demod` (registration with their kernels, the
//! costs 30, `4 span` and `4 span + 40 / sps`, the rates and delays, the required parameters,
//! `rate_unsupported` with the hint to resample, `invalid_value` for the order 3, the type
//! checks), the parameter `polarity` of `frame_decode`, a push-mode `psk_mod` and `psk_demod`
//! that equal the blocks, and packets through `frame_encode` -> `psk_mod` -> an inverted
//! channel -> `psk_demod` -> `frame_decode` with `polarity` both.
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
use sdrk_blocks::{Block, PskDemodBlock, PskDemodConfig, PskModBlock, PskModConfig};
use sdrk_pipeline::{
    BuildOptions, DiagCode, InputData, Packet, PacketFlags, Pipeline, Registry, VecStorage,
};
use sdrk_testkit::Pcg32;
use serde_json::{Value, json};

use common::build;

/// Seed of the random data of the M40 pipeline tests (the chunk-invariance seed of test 6).
const SEED: u64 = 559;

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
    assert_eq!(registry.descriptions().count(), 48);
}

#[test]
fn costas_costs_30_and_keeps_the_rate() {
    let report = validate(&single("costas", "c32", 25_000, json!({ "order": 4 })));
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.cost_per_input_sample, 30.0);
    let node = &report.nodes[0];
    assert_eq!(node.rate_out, Rate::hz(25_000).unwrap());
    assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO.reduced());
    for ty in ["cq15", "f32", "q15"] {
        let (_, code, _) = first_error(&single("costas", ty, 1000, json!({})));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
    let (path, code, _) = first_error(&single("costas", "c32", 1000, json!({ "order": 3 })));
    assert_eq!(
        (path.as_str(), code),
        ("nodes.n.params.order", DiagCode::InvalidValue)
    );
    for bad in [
        json!({ "order": 8 }),
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
fn frame_decode_has_the_polarity_choice() {
    let doc = |polarity: &str| {
        json!({
            "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
            "inputs": { "in": { "type": "f32", "rate_hz": 50_000 } },
            "nodes": { "n": { "kind": "frame_decode", "params": { "polarity": polarity } } },
            "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
            "outputs": { "out": { "type": "packets" } }
        })
    };
    for polarity in ["normal", "both"] {
        let report = validate(&doc(polarity));
        assert!(report.is_ok(), "{}", report.to_json());
    }
    assert!(!validate(&doc("inverted")).is_ok());
}

/// Pushes `input` in pieces of 1 to 1024 samples (seed 559, stream `stream`) and reads the
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
fn inverted_bpsk_frames_are_found_with_polarity_both() {
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
    for (polarity, expected) in [("normal", 0usize), ("both", 20)] {
        let rx_doc = json!({
            "sdrk": 1, "name": "rx", "settings": { "precision": "f32", "max_chunk": 4096 },
            "inputs": { "in": { "type": "c32", "rate_hz": 40_000 } },
            "nodes": {
                "demod": { "kind": "psk_demod", "params": { "order": 2, "symbol_rate_hz": 10_000 } },
                "dec": { "kind": "frame_decode", "params": { "polarity": polarity } }
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
            assert!(got.is_empty(), "polarity {polarity}");
        } else {
            assert!(got.len() >= expected, "polarity {polarity}: {}", got.len());
        }
        for (g, s) in got.iter().zip(&packets) {
            assert_eq!(g.data(), s.data());
            assert_eq!(g.flags, PacketFlags::CRC_OK);
        }
    }
}
