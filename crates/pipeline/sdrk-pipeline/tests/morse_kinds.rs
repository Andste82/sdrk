//! The kinds of M47: `morse_encode` and `morse_decode` (registration without kernels, the
//! costs, rates, type checks, the parameter ranges, `rate_unsupported` below 8 samples per
//! unit) and a Morse loop of packets through both kinds in push mode that equals the blocks.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::large_stack_frames,
    clippy::needless_pass_by_value
)]

mod common;

use sdrk::{Delay, Rate};
use sdrk_blocks::{
    ArenaStorage, Block, MorseDecodeBlock, MorseDecodeConfig, MorseEncodeBlock, MorseEncodeConfig,
    Packet, PacketFlags,
};
use sdrk_pipeline::{BuildOptions, DiagCode, Registry};
use serde_json::{Value, json};

use common::build;

/// The costs of the two kinds in operations per sample (D909).
const ENCODE_COST: f64 = 20.0;
const DECODE_COST: f64 = 69.0;

/// A one-node flow of `kind` on an input of type `ty` at `rate` Hz; `out` is a stream or
/// packets output.
fn single(kind: &str, ty: &str, rate: u64, params: Value, out: &str) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
        "inputs": { "in": { "type": ty, "rate_hz": rate } },
        "nodes": { "n": { "kind": kind, "params": params } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": out } }
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
    for kind in ["morse_encode", "morse_decode"] {
        let f = registry.factory(kind).unwrap();
        assert!(f.kernels().is_empty(), "{kind}");
        assert_eq!(f.description().kind, kind);
    }
    assert_eq!(registry.descriptions().count(), 71);
}

#[test]
fn morse_encode_costs_and_rates() {
    let report = validate(&single(
        "morse_encode",
        "packets",
        8000,
        json!({}),
        "stream",
    ));
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.cost_per_input_sample, ENCODE_COST);
    let node = &report.nodes[0];
    // The output rate is the rate_hz of the packet input; there is no delay.
    assert_eq!(node.rate_out, Rate::hz(8000).unwrap());
    assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO.reduced());
    // Ranges.
    for ok in [
        json!({ "wpm": 5, "rise_ms": 0 }),
        json!({ "wpm": 60 }),
        json!({ "wpm": 10, "char_wpm": 18 }),
        json!({ "rise_ms": 50 }),
    ] {
        assert!(
            validate(&single(
                "morse_encode",
                "packets",
                8000,
                ok.clone(),
                "stream"
            ))
            .is_ok(),
            "{ok}"
        );
    }
    for bad in [
        json!({ "wpm": 4 }),
        json!({ "wpm": 61 }),
        json!({ "wpm": 20, "char_wpm": 19 }),
        json!({ "char_wpm": 101 }),
        json!({ "rise_ms": -1 }),
        json!({ "rise_ms": 51 }),
        json!({ "wpm": "fast" }),
        json!({ "speed": 20 }),
    ] {
        assert!(
            !validate(&single(
                "morse_encode",
                "packets",
                8000,
                bad.clone(),
                "stream"
            ))
            .is_ok(),
            "{bad}"
        );
    }
    // Fewer than 8 samples per unit is `rate_unsupported`.
    let low = single("morse_encode", "packets", 100, json!({}), "stream");
    let report = validate(&low);
    assert!(!report.is_ok());
    assert!(
        report.to_json().contains("rate_unsupported"),
        "{}",
        report.to_json()
    );
    // Only packets go in.
    for ty in ["f32", "c32", "q15", "cq15"] {
        let (_, code) = first_error(&single("morse_encode", ty, 8000, json!({}), "stream"));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
    let (_, code) = first_error(&single(
        "morse_encode",
        "packets",
        8000,
        json!({}),
        "packets",
    ));
    assert_eq!(code, DiagCode::TypeMismatch);
}

#[test]
fn morse_decode_costs_and_rates() {
    let report = validate(&single("morse_decode", "f32", 8000, json!({}), "packets"));
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.cost_per_input_sample, DECODE_COST);
    let node = &report.nodes[0];
    // The packet stream has the nominal rate rate / (6 * glitch): 80 samples at 8 kHz and 40
    // words per minute.
    assert_eq!(node.rate_out, Rate::new(8000, 6 * 80).unwrap());
    assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO.reduced());
    for bad in [
        json!({ "wpm": 4 }),
        json!({ "wpm": 61 }),
        json!({ "wpm_max": 19 }),
        json!({ "wpm_max": 101 }),
        json!({ "threshold": -0.1 }),
        json!({ "wpm": "slow" }),
    ] {
        assert!(
            !validate(&single("morse_decode", "f32", 8000, bad.clone(), "packets")).is_ok(),
            "{bad}"
        );
    }
    assert!(
        validate(&single(
            "morse_decode",
            "f32",
            8000,
            json!({ "threshold": 0.5 }),
            "packets"
        ))
        .is_ok()
    );
    // 40 words per minute needs 8 samples per unit: 267 Hz.
    let low = single("morse_decode", "f32", 200, json!({}), "packets");
    assert!(validate(&low).to_json().contains("rate_unsupported"));
    assert!(validate(&single("morse_decode", "f32", 267, json!({}), "packets")).is_ok());
    for ty in ["c32", "q15", "cq15"] {
        let (_, code) = first_error(&single("morse_decode", ty, 8000, json!({}), "packets"));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
    let (_, code) = first_error(&single("morse_decode", "f32", 8000, json!({}), "stream"));
    assert_eq!(code, DiagCode::TypeMismatch);
}

/// The Morse loop: packets at 8000 samples per second through `morse_encode` and `morse_decode`.
fn loop_spec(enc: Value, dec: Value) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 4096 },
        "inputs": { "msg": { "type": "packets", "rate_hz": 8000 } },
        "nodes": {
            "enc": { "kind": "morse_encode", "params": enc },
            "dec": { "kind": "morse_decode", "params": dec }
        },
        "edges": [
            { "from": "msg", "to": "enc" },
            { "from": "enc", "to": "dec" },
            { "from": "dec", "to": "out" }
        ],
        "outputs": { "out": { "type": "packets", "queue": 4096 } }
    })
}

fn read_all(p: &mut sdrk_pipeline::Pipeline, into: &mut Vec<Packet>) {
    let mut buf = vec![Packet::default(); 64];
    loop {
        let n = p.read_packets("out", &mut buf).unwrap();
        if n == 0 {
            return;
        }
        into.extend_from_slice(&buf[..n]);
    }
}

/// The words of `packets` through the blocks, as the pipeline connects them.
fn through_blocks(
    packets: &[Packet],
    enc: &MorseEncodeConfig,
    dec: &MorseDecodeConfig,
) -> Vec<Packet> {
    let rate = Rate::hz(8000).unwrap();
    let mut e = MorseEncodeBlock::new(enc, rate, &mut ArenaStorage::new(&mut [])).unwrap();
    let mut d = MorseDecodeBlock::new(dec, rate, &mut ArenaStorage::new(&mut [])).unwrap();
    let mut keys = vec![0f32; 4_000_000];
    let p = e.process(packets, &mut keys).unwrap();
    assert_eq!(p.consumed, packets.len());
    let mut out = vec![Packet::default(); 256];
    let q = d.process(&keys[..p.produced], &mut out).unwrap();
    assert_eq!(q.consumed, p.produced);
    out[..q.produced].to_vec()
}

#[test]
fn the_morse_loop_returns_the_words_and_equals_the_blocks() {
    let texts: [&[u8]; 5] = [b"", b"VVV CQ DE W1AW", b"TEST 599", b"73 SK", b"K"];
    let packets: Vec<Packet> = texts
        .iter()
        .map(|t| Packet::new(0, PacketFlags::NONE, t).unwrap())
        .collect();
    let want = through_blocks(
        &packets,
        &MorseEncodeConfig::default(),
        &MorseDecodeConfig::default(),
    );
    let words: Vec<String> = want
        .iter()
        .map(|p| String::from_utf8(p.data().to_vec()).unwrap())
        .collect();
    // The empty first packet is the silence the decoder learns the noise from.
    assert_eq!(
        words,
        ["VVV", "CQ", "DE", "W1AW", "TEST", "599", "73", "SK", "K"]
    );
    let spec = loop_spec(json!({}), json!({}));
    let report = validate(&spec);
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.cost_per_input_sample, ENCODE_COST + DECODE_COST);
    let kinds: Vec<&str> = report.nodes.iter().map(|n| n.kind.as_str()).collect();
    assert_eq!(kinds, ["morse_encode", "morse_decode"]);
    // Pushed at once and in pieces of 1 to 3 packets: the same packets (the packets of the
    // pipeline carry the index of the decoder block).
    for cut in [packets.len(), 1, 2, 3] {
        let mut p = build(&spec.to_string());
        let mut got = Vec::new();
        for piece in packets.chunks(cut) {
            p.push_packets("msg", piece).unwrap();
            read_all(&mut p, &mut got);
        }
        assert_eq!(got.len(), want.len(), "cut {cut}");
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(g.data(), w.data(), "cut {cut}");
        }
        assert!(got.windows(2).all(|w| w[1].index > w[0].index));
    }
}
