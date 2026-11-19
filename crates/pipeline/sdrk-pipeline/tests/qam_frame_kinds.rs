//! The kinds `qam_frame_encode` and `qam_frame_decode` (M44.1 §7): registration with the kernels
//! of `qam_mod` and `qam_demod`, the flat parameters, the costs of the pairs, the rates, the type
//! checks, `invalid_value` and `rate_unsupported` from the inner blocks, and a push-mode flow
//! `qam_frame_encode` -> `qam_frame_decode` that delivers its packets.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::large_stack_frames,
    clippy::needless_pass_by_value
)]

mod common;

use sdrk::{C32, Rate};
use sdrk_blocks::{Block, FrameIdle, QamFrameEncodeBlock, QamFrameEncodeConfig};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Packet, PacketFlags, Registry, VecStorage};
use serde_json::{Value, json};

use common::build;

fn validate(doc: &Value) -> sdrk_pipeline::Report {
    Registry::standard().validate_json(&doc.to_string(), &BuildOptions::default())
}

fn first_error(doc: &Value) -> (String, DiagCode) {
    let report = validate(doc);
    let e = report.errors.first().expect("an error");
    (e.path.clone(), e.code)
}

fn encoder(params: Value) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 16 },
        "inputs": { "msg": { "type": "packets", "rate_hz": 100_000 } },
        "nodes": { "n": { "kind": "qam_frame_encode", "params": params } },
        "edges": [ { "from": "msg", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": "stream", "capacity": 65536 } }
    })
}

fn decoder(ty: &str, params: Value) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 4096 },
        "inputs": { "in": { "type": ty, "rate_hz": 200_000 } },
        "nodes": { "n": { "kind": "qam_frame_decode", "params": params } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": "packets", "queue": 256 } }
    })
}

#[test]
fn the_kinds_are_registered_with_the_kernels_of_their_parts() {
    let registry = Registry::standard();
    let enc = registry.factory("qam_frame_encode").unwrap();
    assert_eq!(enc.kernels(), &["fir_interpolate"]);
    let dec = registry.factory("qam_frame_decode").unwrap();
    assert_eq!(
        dec.kernels(),
        registry.factory("qam_demod").unwrap().kernels()
    );
    assert_eq!(registry.descriptions().count(), 83);
}

#[test]
fn qam_frame_encode_plans_as_the_pair() {
    let report = validate(&encoder(
        json!({ "constellation": "qam16", "samples_per_symbol": 8, "idle": "pn9", "preamble_bytes": 4 }),
    ));
    assert!(report.is_ok(), "{}", report.to_json());
    let node = &report.nodes[0];
    assert_eq!(node.rate_out, Rate::hz(200_000).unwrap());
    // `frame_encode` 1, `bit_scramble` 2 (the default `scrambler`, M44.2) and `qam_mod` 4 span
    // per bit
    assert_eq!(report.cost_per_input_sample, 43.0);
    // without the scrambler, the pair of M44.1
    let report = validate(&encoder(
        json!({ "constellation": "qam16", "samples_per_symbol": 8, "scrambler": "none" }),
    ));
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.cost_per_input_sample, 41.0);
    let report = validate(&encoder(
        json!({ "constellation": "qam64", "samples_per_symbol": 4 }),
    ));
    assert!(report.is_ok(), "{}", report.to_json());
    assert!((report.nodes[0].rate_out.as_f64() - 400_000.0 / 6.0).abs() < 1e-6);
    // required, invalid values and the input type
    let (p, code) = first_error(&encoder(json!({})));
    assert_eq!(p, "nodes.n.params.samples_per_symbol");
    assert_eq!(code, DiagCode::MissingParam);
    let (p, code) = first_error(&encoder(
        json!({ "constellation": "8psk", "samples_per_symbol": 8 }),
    ));
    assert_eq!(
        (p.as_str(), code),
        ("nodes.n.params.constellation", DiagCode::InvalidValue)
    );
    let (p, _) = first_error(&encoder(
        json!({ "preamble_bytes": 33, "samples_per_symbol": 8 }),
    ));
    assert_eq!(p, "nodes.n.params.preamble_bytes");
    let (p, _) = first_error(&encoder(
        json!({ "rotations": "four", "samples_per_symbol": 8 }),
    ));
    assert_eq!(p, "nodes.n.params.rotations");
}

#[test]
fn qam_frame_decode_plans_as_the_pair() {
    let params =
        json!({ "constellation": "qam16", "symbol_rate_hz": 25_000, "equalizer_taps": 15 });
    let report = validate(&decoder("c32", params.clone()));
    assert!(report.is_ok(), "{}", report.to_json());
    let mut pair = decoder("c32", json!({}));
    // the chain of M44.2: the demodulator, the descrambler (the default `scrambler`) and the
    // decoder
    pair["nodes"] = json!({
        "d": { "kind": "qam_demod", "params": params },
        "s": { "kind": "bit_descramble", "params": { "constellation": "qam16" } },
        "f": { "kind": "frame_decode", "params": { "rotations": "four", "constellation": "qam16" } }
    });
    pair["edges"] = json!([
        { "from": "in", "to": "d" }, { "from": "d", "to": "s" }, { "from": "s", "to": "f" },
        { "from": "f", "to": "out" }
    ]);
    let pair = validate(&pair);
    assert!(pair.is_ok(), "{}", pair.to_json());
    assert!(
        (report.cost_per_input_sample - pair.cost_per_input_sample).abs() < 1e-9,
        "{} against {}",
        report.cost_per_input_sample,
        pair.cost_per_input_sample
    );
    for (params, path, code) in [
        (
            json!({ "symbol_rate_hz": 23_000 }),
            "nodes.n.params.symbol_rate_hz",
            DiagCode::RateUnsupported,
        ),
        (
            json!({ "symbol_rate_hz": 25_000, "equalizer_taps": 4 }),
            "nodes.n.params.equalizer_taps",
            DiagCode::InvalidValue,
        ),
        (
            json!({ "symbol_rate_hz": 25_000, "frame_timeout": -1 }),
            "nodes.n.params.frame_timeout",
            DiagCode::OutOfRange,
        ),
        (
            json!({ "symbol_rate_hz": 25_000, "rotations": "both" }),
            "nodes.n.params.rotations",
            DiagCode::UnknownParam,
        ),
    ] {
        let (p, c) = first_error(&decoder("c32", params.clone()));
        assert_eq!((p.as_str(), c), (path, code), "{params}");
    }
    let (_, code) = first_error(&decoder("f32", json!({ "symbol_rate_hz": 25_000 })));
    assert_eq!(code, DiagCode::TypeMismatch);
}

#[test]
fn a_push_flow_of_the_two_composites_delivers_its_packets() {
    let params = json!({ "constellation": "qam16", "samples_per_symbol": 8, "idle": "pn9" });
    let mut tx = build(&encoder(params).to_string());
    let mut cfg = QamFrameEncodeConfig::default();
    cfg.frame.idle = FrameIdle::Pn9;
    let rate = Rate::hz(100_000).unwrap();
    // the flow in push mode equals a block that gets the same calls
    let mut same = QamFrameEncodeBlock::new(&cfg, rate, &mut VecStorage).unwrap();
    // the signal of the receiver: idle symbols for the acquisition, then the 20 packets
    let mut block = QamFrameEncodeBlock::new(&cfg, rate, &mut VecStorage).unwrap();
    let mut signal = vec![C32::default(); 8 * 6000];
    assert_eq!(
        block.process(&[], &mut signal).unwrap().produced,
        signal.len()
    );
    let sent: Vec<Packet> = (0..20u8)
        .map(|k| Packet::new(0, PacketFlags::NONE, &[k; 16]).unwrap())
        .collect();
    let mut buf = vec![C32::default(); 1 << 16];
    for p in sent.chunks(4) {
        tx.push("msg", InputData::Packets(p)).unwrap();
        let mut flow = Vec::new();
        loop {
            let n = tx.read_stream("out", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            flow.extend_from_slice(&buf[..n]);
        }
        let q = same.process(p, &mut buf).unwrap();
        assert_eq!(q.consumed, p.len());
        let k = flow
            .iter()
            .zip(&buf[..q.produced])
            .position(|(a, b)| a != b);
        assert_eq!((flow.len(), k), (q.produced, None));
        let q = block.process(p, &mut buf).unwrap();
        assert_eq!(q.consumed, p.len());
        signal.extend_from_slice(&buf[..q.produced]);
    }
    // idle symbols that carry the last frame out of the pulse filter
    let q = block.process(&[], &mut buf[..8 * 400]).unwrap();
    signal.extend_from_slice(&buf[..q.produced]);
    let rx = decoder(
        "c32",
        json!({ "constellation": "qam16", "symbol_rate_hz": 25_000 }),
    );
    let mut rx = build(&rx.to_string());
    for piece in signal.chunks(4096) {
        rx.push("in", InputData::C32(piece)).unwrap();
    }
    let mut got = vec![Packet::default(); 64];
    let n = rx.read_packets("out", &mut got).unwrap();
    let ok: Vec<&Packet> = got[..n]
        .iter()
        .filter(|p| p.flags.contains(PacketFlags::CRC_OK))
        .collect();
    assert_eq!(ok.len(), sent.len());
    for (a, b) in ok.iter().zip(&sent) {
        assert_eq!(a.data(), b.data());
    }
}
