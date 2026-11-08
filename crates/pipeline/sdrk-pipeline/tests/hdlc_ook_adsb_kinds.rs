//! The kinds of M39.1: `hdlc_encode`, `hdlc_decode`, `ook_mod`, `ook_demod` and `adsb_decode`
//! (registration with their kernels, the costs 2, 2, `1 + 2 sps`, `5 + round(sps) + 28 / sps`
//! and 10, rates and delays, required parameters, type checks, `rate_unsupported` below 4
//! samples per symbol and away from 2 MHz), an HDLC loop of packets through the pipeline in
//! push mode, and push-mode chains `ook_demod` -> `frame_decode` and `adsb_decode` that equal
//! the blocks.
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

use sdrk::{C32, Delay, Rate};
use sdrk_blocks::{
    AdsbDecodeBlock, AdsbDecodeConfig, Block, FrameDecodeBlock, FrameDecodeConfig,
    FrameEncodeBlock, FrameEncodeConfig, OokDemodBlock, OokDemodConfig, OokModBlock, OokModConfig,
    Packet, PacketFlags,
};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Registry, VecStorage};
use sdrk_testkit::Pcg32;
use serde_json::{Value, json};

use common::build;

/// Seed of M39.1 test 6 (seed table of docs/milestones/README.md).
const SEED: u64 = 546;

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

fn read_all(p: &mut sdrk_pipeline::Pipeline, output: &str, into: &mut Vec<Packet>) {
    let mut buf = vec![Packet::default(); 64];
    loop {
        let n = p.read_packets(output, &mut buf).unwrap();
        if n == 0 {
            return;
        }
        into.extend_from_slice(&buf[..n]);
    }
}

#[test]
fn the_kinds_are_registered_with_their_kernels() {
    let registry = Registry::standard();
    for (kind, kernels) in [
        ("hdlc_encode", &[][..]),
        ("hdlc_decode", &[][..]),
        ("ook_mod", &["fir_interpolate_real", "interleave"][..]),
        ("ook_demod", &["mag", "resample_real", "symbol_sync"][..]),
        ("adsb_decode", &["mag"][..]),
    ] {
        let f = registry.factory(kind).unwrap();
        assert_eq!(f.kernels(), kernels, "{kind}");
        assert_eq!(f.description().kind, kind);
    }
    assert_eq!(registry.descriptions().count(), 66);
}

/// The HDLC loop: packets at 1200 bit/s through `hdlc_encode` and `hdlc_decode`.
fn hdlc_loop(enc: Value, dec: Value) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 4096 },
        "inputs": { "msg": { "type": "packets", "rate_hz": 1200 } },
        "nodes": {
            "enc": { "kind": "hdlc_encode", "params": enc },
            "dec": { "kind": "hdlc_decode", "params": dec }
        },
        "edges": [
            { "from": "msg", "to": "enc" },
            { "from": "enc", "to": "dec" },
            { "from": "dec", "to": "out" }
        ],
        "outputs": { "out": { "type": "packets", "queue": 4096 } }
    })
}

#[test]
fn hdlc_kinds_cost_2_each_and_loop_packets_in_push_mode() {
    let spec = hdlc_loop(json!({}), json!({}));
    let report = validate(&spec);
    assert!(report.is_ok(), "{}", report.to_json());
    assert!((report.cost_per_input_sample - 4.0).abs() < 1e-9);
    let kinds: Vec<&str> = report.nodes.iter().map(|n| n.kind.as_str()).collect();
    assert_eq!(kinds, ["hdlc_encode", "hdlc_decode"]);
    assert_eq!(report.nodes[0].rate_out, Rate::hz(1200).unwrap());
    assert_eq!(report.nodes[1].rate_out, Rate::new(1200, 40).unwrap());
    let mut rng = Pcg32::new(SEED, 30);
    let packets: Vec<Packet> = (0..100)
        .map(|_| {
            let len = 2 + rng.below(255) as usize;
            let data: Vec<u8> = (0..len).map(|_| rng.next_u32() as u8).collect();
            Packet::new(0, PacketFlags::NONE, &data).unwrap()
        })
        .collect();
    for (enc, dec) in [
        (json!({}), json!({})),
        (
            json!({ "nrzi": false, "preamble_flags": 1, "tail_flags": 1 }),
            json!({ "nrzi": false, "emit_bad": true }),
        ),
    ] {
        let mut p = build(&hdlc_loop(enc.clone(), dec).to_string());
        let mut got = Vec::new();
        let mut at = 0;
        while at < packets.len() {
            let n = (1 + rng.below(9) as usize).min(packets.len() - at);
            p.push_packets("msg", &packets[at..at + n]).unwrap();
            read_all(&mut p, "out", &mut got);
            at += n;
        }
        assert_eq!(got.len(), packets.len(), "{enc}");
        for (g, s) in got.iter().zip(&packets) {
            assert_eq!(g.data(), s.data());
            assert_eq!(g.flags, PacketFlags::CRC_OK);
        }
        // Indices grow by a whole frame each.
        assert!(got.windows(2).all(|w| w[1].index > w[0].index));
    }
    // Ranges and types.
    for bad in [
        json!({ "preamble_flags": 0 }),
        json!({ "preamble_flags": 1025 }),
        json!({ "tail_flags": 65 }),
    ] {
        assert!(
            !validate(&hdlc_loop(bad.clone(), json!({}))).is_ok(),
            "{bad}"
        );
    }
    let (_, code) = first_error(&single("hdlc_decode", "c32", 1200, json!({}), "packets"));
    assert_eq!(code, DiagCode::TypeMismatch);
    let (_, code) = first_error(&single("hdlc_encode", "f32", 1200, json!({}), "stream"));
    assert_eq!(code, DiagCode::TypeMismatch);
}

#[test]
fn ook_mod_costs_1_plus_2_sps() {
    for (sps, rate_out) in [(500u64, 1_000_000u64), (8, 16_000), (1, 2000)] {
        let report = validate(&single(
            "ook_mod",
            "f32",
            2000,
            json!({ "samples_per_symbol": sps }),
            "stream",
        ));
        assert!(report.is_ok(), "{}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 1.0 + 2.0 * sps as f64);
        let node = &report.nodes[0];
        assert_eq!(node.rate_out, Rate::hz(rate_out).unwrap());
        assert_eq!(
            node.delay_samples_in.reduced(),
            Delay::new(sps - 1, 2 * sps).unwrap().reduced()
        );
    }
    assert!(!validate(&single("ook_mod", "f32", 2000, json!({}), "stream")).is_ok());
    for bad in [
        json!({ "samples_per_symbol": 0 }),
        json!({ "samples_per_symbol": 4097 }),
        json!({ "samples_per_symbol": 8, "level": 1.5 }),
    ] {
        assert!(
            !validate(&single("ook_mod", "f32", 2000, bad.clone(), "stream")).is_ok(),
            "{bad}"
        );
    }
    for ty in ["c32", "q15", "cq15"] {
        let (_, code) = first_error(&single(
            "ook_mod",
            ty,
            2000,
            json!({ "samples_per_symbol": 8 }),
            "stream",
        ));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
}

#[test]
fn ook_demod_costs_4_plus_round_sps_plus_28_per_symbol_and_needs_4_samples_per_symbol() {
    for (fs, sym, sps) in [
        (250_000u64, 2000.0, 125.0f64),
        (8000, 2000.0, 4.0),
        (10_100, 1000.0, 10.1),
    ] {
        let report = validate(&single(
            "ook_demod",
            "c32",
            fs,
            json!({ "symbol_rate_hz": sym }),
            "stream",
        ));
        assert!(report.is_ok(), "{}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 5.0 + sps.round() + 28.0 / sps);
        let node = &report.nodes[0];
        assert_eq!(
            node.rate_out,
            Rate::new((sym * 1000.0) as u64, 1000).unwrap()
        );
        let taps = sps.round() as u64;
        assert_eq!(
            node.delay_samples_in.reduced(),
            Delay::new(taps + 1, 2).unwrap().reduced()
        );
    }
    let (path, code) = first_error(&single(
        "ook_demod",
        "c32",
        7999,
        json!({ "symbol_rate_hz": 2000 }),
        "stream",
    ));
    assert_eq!(code, DiagCode::RateUnsupported);
    assert_eq!(path, "nodes.n.params.symbol_rate_hz");
    assert!(!validate(&single("ook_demod", "c32", 8000, json!({}), "stream")).is_ok());
    for ty in ["f32", "cq15"] {
        let (_, code) = first_error(&single(
            "ook_demod",
            ty,
            8000,
            json!({ "symbol_rate_hz": 1000 }),
            "stream",
        ));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
}

#[test]
fn adsb_decode_costs_10_and_needs_exactly_2_mhz() {
    let report = validate(&single(
        "adsb_decode",
        "c32",
        2_000_000,
        json!({}),
        "packets",
    ));
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.cost_per_input_sample, 10.0);
    assert_eq!(report.nodes[0].rate_out, Rate::new(2_000_000, 128).unwrap());
    for fs in [2_400_000u64, 1_999_999, 1_000_000] {
        let (_, code) = first_error(&single("adsb_decode", "c32", fs, json!({}), "packets"));
        assert_eq!(code, DiagCode::RateUnsupported, "{fs}");
    }
    let (_, code) = first_error(&single(
        "adsb_decode",
        "f32",
        2_000_000,
        json!({}),
        "packets",
    ));
    assert_eq!(code, DiagCode::TypeMismatch);
    let (path, code) = first_error(&single(
        "adsb_decode",
        "c32",
        2_000_000,
        json!({ "min_snr_db": 61 }),
        "packets",
    ));
    assert_eq!(code, DiagCode::OutOfRange);
    assert_eq!(path, "nodes.n.params.min_snr_db");
}

/// The OOK receive chain `ook_demod` -> `frame_decode` at 16 kHz (2000 symbols per second).
fn ook_chain() -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
        "inputs": { "in": { "type": "c32", "rate_hz": 16_000 } },
        "nodes": {
            "demod": { "kind": "ook_demod", "params": { "symbol_rate_hz": 2000 } },
            "dec": { "kind": "frame_decode", "params": {} }
        },
        "edges": [
            { "from": "in", "to": "demod" },
            { "from": "demod", "to": "dec" },
            { "from": "dec", "to": "out" }
        ],
        "outputs": { "out": { "type": "packets", "queue": 1024 } }
    })
}

#[test]
fn the_ook_chain_equals_the_blocks() {
    let mut rng = Pcg32::new(SEED, 31);
    let packets: Vec<Packet> = (0..20)
        .map(|_| {
            let data: Vec<u8> = (0..16).map(|_| rng.next_u32() as u8).collect();
            Packet::new(0, PacketFlags::NONE, &data).unwrap()
        })
        .collect();
    let bit_rate = Rate::hz(2000).unwrap();
    let enc_cfg = FrameEncodeConfig {
        preamble_bytes: 8,
        ..FrameEncodeConfig::default()
    };
    let mut enc = FrameEncodeBlock::new(&enc_cfg, bit_rate, &mut VecStorage).unwrap();
    let mut bits = Vec::new();
    let mut room = vec![0.0f32; 4096];
    for p in &packets {
        let r = enc.process(std::slice::from_ref(p), &mut room).unwrap();
        bits.extend_from_slice(&room[..r.produced]);
        bits.extend(std::iter::repeat_n(-1.0f32, 16));
    }
    let mod_cfg = OokModConfig {
        samples_per_symbol: 8,
        ..OokModConfig::default()
    };
    let mut ook = OokModBlock::new(&mod_cfg, bit_rate, &mut VecStorage).unwrap();
    let mut signal = vec![C32::default(); bits.len() * 8];
    ook.process(&bits, &mut signal).unwrap();
    // A carrier offset of 300 Hz and a little noise.
    for (i, s) in signal.iter_mut().enumerate() {
        let ph = std::f64::consts::TAU * 300.0 * i as f64 / 16_000.0;
        let (sin, cos) = ph.sin_cos();
        let re = f64::from(s.re);
        *s = C32::new(
            (re * cos + 0.02 * rng.normal()) as f32,
            (re * sin + 0.02 * rng.normal()) as f32,
        );
    }
    // The blocks directly.
    let fs = Rate::hz(16_000).unwrap();
    let demod_cfg = OokDemodConfig {
        symbol_rate_hz: 2000.0,
        ..OokDemodConfig::default()
    };
    let mut demod = OokDemodBlock::new(&demod_cfg, fs, &mut VecStorage).unwrap();
    let mut soft = vec![0.0f32; bits.len() + 16];
    let p = demod.process(&signal, &mut soft).unwrap();
    assert_eq!(p.consumed, signal.len());
    soft.truncate(p.produced);
    let rate = demod.timing().ratio;
    let mut dec = FrameDecodeBlock::new(
        &FrameDecodeConfig::default(),
        fs.scale(rate).unwrap(),
        &mut VecStorage,
    )
    .unwrap();
    let mut want = vec![Packet::default(); 64];
    let p = dec.process(&soft, &mut want).unwrap();
    want.truncate(p.produced);
    assert_eq!(want.len(), packets.len());
    for (w, s) in want.iter().zip(&packets) {
        assert_eq!(w.data(), s.data());
    }
    // The pipeline in pieces of up to 1000 samples.
    let mut pipe = build(&ook_chain().to_string());
    let mut got = Vec::new();
    for piece in signal.chunks(1000) {
        pipe.push("in", InputData::C32(piece)).unwrap();
        read_all(&mut pipe, "out", &mut got);
    }
    assert_eq!(got, want);
    assert!(pipe.set_param("demod", "track_symbols", json!(8)).is_err());
}

/// CRC-24 of Mode S (polynomial 0xFFF409, init 0, most significant bit first), bit by bit.
fn crc24(data: &[u8]) -> u32 {
    let mut crc = 0u32;
    for &b in data {
        crc ^= u32::from(b) << 16;
        for _ in 0..8 {
            crc <<= 1;
            if crc & 0x100_0000 != 0 {
                crc ^= 0x1FF_F409;
            }
        }
    }
    crc & 0xFF_FFFF
}

#[test]
fn the_adsb_flow_equals_the_block() {
    assert_eq!(
        crc24(&[
            0x8D, 0x48, 0x40, 0xD6, 0x20, 0x2C, 0xC3, 0x71, 0xC3, 0x2C, 0xE0
        ]),
        0x57_6098
    );
    let fs = Rate::hz(2_000_000).unwrap();
    let mut rng = Pcg32::new(SEED, 32);
    let mut x: Vec<C32> = (0..60_000)
        .map(|_| C32::new((0.01 * rng.normal()) as f32, (0.01 * rng.normal()) as f32))
        .collect();
    let mut p = 500usize;
    while p + 300 < x.len() {
        let mut f = [0u8; 14];
        f[0] = 0x8D;
        for b in &mut f[1..11] {
            *b = rng.next_u32() as u8;
        }
        let crc = crc24(&f[..11]);
        f[11..14].copy_from_slice(&crc.to_be_bytes()[1..]);
        let a = 0.1 + 0.3 * rng.next_f64() as f32;
        for k in [0, 2, 7, 9] {
            x[p + k] += C32::new(a, 0.0);
        }
        for k in 0..112 {
            let bit = (f[k / 8] >> (7 - k % 8)) & 1;
            x[p + 16 + 2 * k + usize::from(bit == 0)] += C32::new(a, 0.0);
        }
        p += 240 + rng.below(3000) as usize;
    }
    let mut block =
        AdsbDecodeBlock::new(&AdsbDecodeConfig::default(), fs, &mut VecStorage).unwrap();
    let mut want = vec![Packet::default(); 256];
    let r = block.process(&x, &mut want).unwrap();
    assert_eq!(r.consumed, x.len());
    want.truncate(r.produced);
    assert!(want.len() > 20);
    let spec = single("adsb_decode", "c32", 2_000_000, json!({}), "packets");
    let mut pipe = build(&spec.to_string());
    let mut got = Vec::new();
    for piece in x.chunks(777) {
        pipe.push("in", InputData::C32(piece)).unwrap();
        read_all(&mut pipe, "out", &mut got);
    }
    assert_eq!(got, want);
}
