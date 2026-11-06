//! The kinds of M41: `rds_demod` and `rds_decode` (registration with their kernels, the costs
//! `6 + 60 / d` and 2, the rates, delays and packet rate, `rate_unsupported` below 128 kHz, the
//! type checks and the parameter ranges), and the two kinds in one flow on the reference
//! multiplex: push-mode output equal to the blocks in pieces of seed 575, the groups of the
//! carousel as packets.
#![allow(
    clippy::needless_pass_by_value,
    clippy::large_stack_frames,
    clippy::many_single_char_names,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp
)]

mod common;

use sdrk::{Delay, Rate};
use sdrk_blocks::{Block, RdsDecodeBlock, RdsDecodeConfig, RdsDemodBlock, RdsDemodConfig};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Packet, PacketFlags, Registry, VecStorage};
use sdrk_proto::rds::Group;
use sdrk_testkit::Pcg32;
use sdrk_testkit::rds::{RdsReference, TABLE_LEN, carousel_group};
use serde_json::{Value, json};

use common::build;

/// Seed of the piece sizes (the chunk-invariance seed of M41, test 5).
const SEED: u64 = 575;

/// A one-node flow of `kind` on an input of type `ty` at `rate` Hz, output `out` of `output`.
fn single(kind: &str, ty: &str, rate: u64, params: Value, output: &str) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
        "inputs": { "in": { "type": ty, "rate_hz": rate } },
        "nodes": { "n": { "kind": kind, "params": params } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": output } }
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
    for (kind, kernels) in [
        (
            "rds_demod",
            &[
                "interleave",
                "rotate",
                "fir_decimate",
                "symbol_sync",
                "costas",
            ][..],
        ),
        ("rds_decode", &[][..]),
    ] {
        let f = registry.factory(kind).unwrap();
        assert_eq!(f.kernels(), kernels, "{kind}");
        assert_eq!(f.description().kind, kind);
    }
    assert_eq!(registry.descriptions().count(), 58);
}

#[test]
fn rds_demod_costs_6_plus_60_over_d_and_outputs_1187_5_hz() {
    for (rate, d) in [(240_000u64, 12u64), (128_000, 6)] {
        let taps = RdsDemodConfig::default()
            .num_taps(Rate::hz(rate).unwrap())
            .unwrap();
        let report = validate(&single("rds_demod", "f32", rate, json!({}), "stream"));
        assert!(report.is_ok(), "{}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 6.0 + 60.0 / d as f64);
        let node = &report.nodes[0];
        assert_eq!(node.rate_out, Rate::new(2375, 2).unwrap());
        assert_eq!(
            node.delay_samples_in.reduced(),
            Delay::new(taps as u64 - 1 + 2 * d, 2).unwrap().reduced()
        );
    }
    // below 128 kHz the rate is unsupported
    for rate in [127_999u64, 48_000] {
        let (path, code) = first_error(&single("rds_demod", "f32", rate, json!({}), "stream"));
        assert_eq!(
            (path.as_str(), code),
            ("nodes.n.params.sample_rate", DiagCode::RateUnsupported),
            "{rate}"
        );
    }
    for ty in ["c32", "cq15", "q15"] {
        let (_, code) = first_error(&single("rds_demod", ty, 240_000, json!({}), "stream"));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
    for (bad, code) in [
        (json!({ "sync_bw": 0.2 }), DiagCode::OutOfRange),
        (json!({ "carrier_bw": 0.3 }), DiagCode::OutOfRange),
        (json!({ "sync_bw": "fast" }), DiagCode::WrongType),
        (json!({ "loop_bw": 0.01 }), DiagCode::UnknownParam),
    ] {
        let (_, got) = first_error(&single("rds_demod", "f32", 240_000, bad.clone(), "stream"));
        assert_eq!(got, code, "{bad}");
    }
    let ok = validate(&single(
        "rds_demod",
        "f32",
        240_000,
        json!({ "sync_bw": 0.02, "carrier_bw": 0.005 }),
        "stream",
    ));
    assert!(ok.is_ok(), "{}", ok.to_json());
}

#[test]
fn rds_decode_costs_2_and_feeds_a_packet_output() {
    let report = validate(&single("rds_decode", "f32", 1187, json!({}), "packets"));
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.cost_per_input_sample, 2.0);
    assert_eq!(
        report.nodes[0].delay_samples_in.reduced(),
        Delay::ZERO.reduced()
    );
    for ty in ["c32", "cq15", "q15"] {
        let (_, code) = first_error(&single("rds_decode", ty, 1187, json!({}), "packets"));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
    for (bad, code) in [
        (json!({ "max_bad": 51 }), DiagCode::OutOfRange),
        (json!({ "max_bad": -1 }), DiagCode::OutOfRange),
        (json!({ "emit_bad": 1 }), DiagCode::WrongType),
    ] {
        let (_, got) = first_error(&single("rds_decode", "f32", 1187, bad.clone(), "packets"));
        assert_eq!(got, code, "{bad}");
    }
    let ok = validate(&single(
        "rds_decode",
        "f32",
        1187,
        json!({ "emit_bad": true, "max_bad": 50 }),
        "packets",
    ));
    assert!(ok.is_ok(), "{}", ok.to_json());
}

/// `rds_demod` -> `rds_decode` in one flow on 2 s of the reference multiplex, pushed in pieces
/// of 1 to 4096 samples (seed 575): the soft bits of a stream output equal those of the block,
/// and the packets are the groups of the carousel with `CRC_OK`, at the same indices as the
/// blocks give them.
#[test]
fn the_kinds_in_one_flow_equal_the_blocks() {
    let fs = Rate::hz(240_000).unwrap();
    let reference = RdsReference::new(vec![0.0; TABLE_LEN], 0.04).unwrap();
    let mpx: Vec<f32> = (0..480_000u64).map(|m| reference.mpx(m) as f32).collect();
    let mut demod =
        RdsDemodBlock::<VecStorage>::new(&RdsDemodConfig::default(), fs, &mut VecStorage).unwrap();
    let mut want_soft = vec![0.0f32; 4000];
    let p = demod.process(&mpx, &mut want_soft).unwrap();
    assert_eq!(p.consumed, mpx.len());
    want_soft.truncate(p.produced);
    let mut decode = RdsDecodeBlock::<VecStorage>::new(
        &RdsDecodeConfig::default(),
        Rate::new(2375, 2).unwrap(),
        &mut VecStorage,
    )
    .unwrap();
    let mut want = vec![Packet::default(); 64];
    let q = decode.process(&want_soft, &mut want).unwrap();
    want.truncate(q.produced);
    assert!(want.len() >= 20, "{}", want.len());

    let doc = json!({
        "sdrk": 1, "name": "rds", "settings": { "precision": "f32", "max_chunk": 4096 },
        "inputs": { "in": { "type": "f32", "rate_hz": 240_000 } },
        "nodes": {
            "subcarrier": { "kind": "rds_demod", "params": {} },
            "groups": { "kind": "rds_decode", "params": {} }
        },
        "edges": [
            { "from": "in", "to": "subcarrier" }, { "from": "subcarrier", "to": "groups" },
            { "from": "subcarrier", "to": "soft" }, { "from": "groups", "to": "rds" }
        ],
        "outputs": { "soft": { "type": "stream" }, "rds": { "type": "packets", "queue": 64 } }
    });
    let mut p = build(&doc.to_string());
    let mut rng = Pcg32::new(SEED, 1);
    let (mut soft, mut packets) = (Vec::new(), Vec::new());
    let mut buf = vec![0.0f32; 8192];
    let mut pk = vec![Packet::default(); 16];
    let mut at = 0;
    while at < mpx.len() {
        let n = (1 + rng.below(4096) as usize).min(mpx.len() - at);
        p.push("in", InputData::F32(&mpx[at..at + n])).unwrap();
        at += n;
        loop {
            let k = p.read_stream::<f32>("soft", &mut buf).unwrap();
            let j = p.read_packets("rds", &mut pk).unwrap();
            soft.extend_from_slice(&buf[..k]);
            packets.extend_from_slice(&pk[..j]);
            if k == 0 && j == 0 {
                break;
            }
        }
    }
    assert_eq!(soft.len(), want_soft.len());
    assert!(
        soft.iter()
            .zip(&want_soft)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the soft bits of the flow differ from the block"
    );
    assert_eq!(packets.len(), want.len());
    for (got, w) in packets.iter().zip(&want) {
        assert_eq!(got.data(), w.data());
        assert_eq!(got.flags, w.flags);
        assert_eq!(got.index, w.index);
        assert!(got.flags.contains(PacketFlags::CRC_OK));
        let g = Group::from_packet(got).unwrap();
        assert_eq!(got.index % 104, 0);
        assert_eq!(g.blocks, carousel_group(got.index / 104));
    }
}
