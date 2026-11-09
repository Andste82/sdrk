//! The kinds of M42: `rds_encode` (a clocked encoder: a clock stream of any type at port `a`,
//! control packets at port `b`, cost 12) and `fm_stereo_rds_mod` (stereo audio `c32` at port
//! `a` and the RDS baseband `f32` at port `b`, cost 14): registration with their kernels, the
//! figures, the type checks at both ports (the two ports of `fm_stereo_rds_mod` take different
//! types), the rate and parameter checks, the runtime parameters, the reading
//! `ignored_packets` in the statistics, and a pipeline `audio` -> (`rds`, `mpx`) that equals the
//! blocks bit for bit with packets pushed between the audio pieces.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_pass_by_value,
    clippy::large_stack_frames
)]

mod common;

use sdrk::{C32, Delay, Rate};
use sdrk_blocks::{
    Block, FmStereoRdsModBlock, FmStereoRdsModConfig, Packet, PacketFlags, ParamValue,
    RdsEncodeBlock, RdsEncodeConfig,
};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Registry};
use sdrk_testkit::{Pcg32, VecStorage};
use serde_json::{Value, json};

use common::{bits, build};

const FS: u64 = 192_000;

/// The encoder alone: the clock input `clock` of type `ty` at `rate` Hz and the packet input
/// `text`.
fn encoder(ty: &str, rate: u64, params: Value) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 4096 },
        "inputs": {
            "clock": { "type": ty, "rate_hz": rate },
            "text": { "type": "packets" }
        },
        "nodes": { "rds": { "kind": "rds_encode", "params": params } },
        "edges": [
            { "from": "clock", "to": "rds" },
            { "from": "text", "to": "rds", "port": "b" },
            { "from": "rds", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    })
}

/// The transmitter core: the audio clocks the encoder, both feed the multiplex.
fn transmitter(rate: u64, rds: Value, mpx: Value) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 4096 },
        "inputs": {
            "audio": { "type": "c32", "rate_hz": rate },
            "text": { "type": "packets" }
        },
        "nodes": {
            "rds": { "kind": "rds_encode", "params": rds },
            "mpx": { "kind": "fm_stereo_rds_mod", "params": mpx }
        },
        "edges": [
            { "from": "audio", "to": "mpx" },
            { "from": "audio", "to": "rds" },
            { "from": "text", "to": "rds", "port": "b" },
            { "from": "rds", "to": "mpx", "port": "b" },
            { "from": "mpx", "to": "out" }
        ],
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

fn pi() -> Value {
    json!({ "pi": 53539, "pty": 10 })
}

#[test]
fn the_kinds_are_registered_with_their_kernels() {
    let registry = Registry::standard();
    for (kind, kernels) in [
        ("rds_encode", ["rds_shape"]),
        ("fm_stereo_rds_mod", ["stereo_encode_rds"]),
    ] {
        let f = registry.factory(kind).unwrap();
        assert_eq!(f.kernels(), kernels, "{kind}");
        assert_eq!(f.description().kind, kind);
    }
    assert_eq!(registry.descriptions().count(), 69);
}

/// `rds_encode` costs 12 per sample at every clock type, keeps the rate, reports no delay and
/// takes its pulse table (4100 bytes); with `fm_stereo_rds_mod` (14) the transmitter core
/// costs 26 per audio sample.
#[test]
fn costs_rates_and_delays() {
    for (ty, precision) in [
        ("c32", "f32"),
        ("cq15", "q15"),
        ("f32", "f32"),
        ("q15", "q15"),
    ] {
        let mut doc = encoder(ty, FS, pi());
        doc["settings"]["precision"] = json!(precision);
        let report = validate(&doc);
        assert!(report.is_ok(), "{ty}: {}", report.to_json());
        assert!(report.warnings.is_empty(), "{}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 12.0);
        let node = &report.nodes[0];
        assert_eq!(node.rate_out, Rate::hz(FS).unwrap());
        assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO.reduced());
        assert_eq!(node.clock_domain, "clock");
        assert!(report.memory_bytes >= 4100);
    }
    for rate in [120_000, FS, 240_000] {
        let report = validate(&transmitter(rate, pi(), json!({})));
        assert!(report.is_ok(), "{}", report.to_json());
        assert!(report.warnings.is_empty(), "{}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 26.0);
        for node in &report.nodes {
            assert_eq!(node.rate_out, Rate::hz(rate).unwrap());
            assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO.reduced());
            assert_eq!(node.clock_domain, "audio");
        }
    }
}

#[test]
fn types_rates_and_parameters_are_checked() {
    // pi is required; pty and the ranges are checked.
    for (params, code) in [
        (json!({}), None),
        (json!({ "pi": 65_536 }), Some(DiagCode::OutOfRange)),
        (json!({ "pi": -1 }), Some(DiagCode::OutOfRange)),
        (json!({ "pi": 1, "pty": 32 }), Some(DiagCode::OutOfRange)),
        (json!({ "pi": 1, "tp": 1 }), Some(DiagCode::WrongType)),
        (json!({ "pi": 1, "ps": "x" }), Some(DiagCode::UnknownParam)),
    ] {
        let report = validate(&encoder("f32", FS, params.clone()));
        assert!(!report.is_ok(), "{params}");
        if let Some(code) = code {
            assert_eq!(report.errors[0].code, code, "{params}");
        }
    }
    // Below 4750 Hz the pulse does not fit.
    let (_, code) = first_error(&encoder("f32", 4_000, pi()));
    assert_eq!(code, DiagCode::RateUnsupported);
    assert!(validate(&encoder("f32", 4_750, pi())).is_ok());
    // fm_stereo_rds_mod needs 120 kHz.
    let (_, code) = first_error(&transmitter(96_000, pi(), json!({})));
    assert_eq!(code, DiagCode::RateUnsupported);
    for params in [
        json!({ "rds": 0.21 }),
        json!({ "rds": -0.01 }),
        json!({ "main": 1.1 }),
        json!({ "pilot_hz": 17_999 }),
    ] {
        assert!(
            !validate(&transmitter(FS, pi(), params.clone())).is_ok(),
            "{params}"
        );
    }
    // Port b of rds_encode takes packets only (a stream is a type mismatch at the edge).
    let mut doc = encoder("f32", FS, pi());
    doc["inputs"]["text"] = json!({ "type": "f32", "rate_hz": FS });
    let (path, code) = first_error(&doc);
    assert_eq!(
        (path.as_str(), code),
        ("edges[1].to", DiagCode::TypeMismatch)
    );
    // Port a of rds_encode takes a sample stream, not packets.
    let mut doc = encoder("f32", FS, pi());
    doc["inputs"]["clock"] = json!({ "type": "packets", "rate_hz": FS });
    assert!(!validate(&doc).is_ok());
    // Port b of fm_stereo_rds_mod takes f32 only: the audio itself is a type mismatch.
    let has = |doc: &Value, path: &str| {
        validate(doc)
            .errors
            .iter()
            .any(|d| d.path == path && d.code == DiagCode::TypeMismatch)
    };
    let mut doc = transmitter(FS, pi(), json!({}));
    doc["edges"][3] = json!({ "from": "audio", "to": "mpx", "port": "b" });
    doc["edges"]
        .as_array_mut()
        .unwrap()
        .push(json!({ "from": "rds", "to": "spare" }));
    doc["outputs"]["spare"] = json!({ "type": "stream" });
    assert!(has(&doc, "edges[3].to"), "{}", validate(&doc).to_json());
    // Port a takes c32 only: the RDS baseband there is a type mismatch.
    let mut doc = transmitter(FS, pi(), json!({}));
    doc["edges"][0] = json!({ "from": "rds", "to": "mpx" });
    assert!(has(&doc, "edges[0].to"), "{}", validate(&doc).to_json());
    // A Q15 flow cannot feed fm_stereo_rds_mod (port a takes c32).
    let mut doc = transmitter(FS, pi(), json!({}));
    doc["settings"]["precision"] = json!("q15");
    doc["inputs"]["audio"]["type"] = json!("cq15");
    assert!(!validate(&doc).is_ok());
}

/// `rds` and `pilot` change while running; the build parameters do not.
#[test]
fn runtime_parameters_can_be_set_on_the_pipeline() {
    let mut p = build(&transmitter(FS, pi(), json!({})).to_string());
    p.set_param("mpx", "rds", json!(0.0)).unwrap();
    p.set_param("mpx", "rds", json!(0.2)).unwrap();
    assert!(p.set_param("mpx", "rds", json!(0.3)).is_err());
    p.set_param("mpx", "pilot", json!(0.0)).unwrap();
    assert!(p.set_param("mpx", "main", json!(0.5)).is_err());
    assert!(p.set_param("rds", "pi", json!(1)).is_err());
    assert!(p.set_param("rds", "pty", json!(1)).is_err());
}

fn packet(data: &[u8]) -> Packet {
    Packet::new(0, PacketFlags::NONE, data).unwrap()
}

fn stereo_audio(n: usize, seed: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(seed, 3);
    (0..n)
        .map(|i| {
            let t = i as f64 / FS as f64;
            let l = 0.6 * (std::f64::consts::TAU * 1000.0 * t).sin() + 0.1 * rng.normal();
            let r = 0.4 * (std::f64::consts::TAU * 3000.0 * t).sin();
            C32::new(l as f32, r as f32)
        })
        .collect()
}

/// The pipeline equals `RdsEncodeBlock` and `FmStereoRdsModBlock` run directly, bit for bit:
/// the texts before the first audio, 20 raw groups and an unknown packet before the audio
/// sample 30000 (four groups wait in the packet buffer of the pipeline), a new radiotext and a
/// change of `rds` before sample 300000. The statistics carry the reading `ignored_packets`.
#[test]
fn the_pipeline_equals_the_blocks() {
    const N: usize = 420_000;
    const PIECE: usize = 1500;
    let rate = Rate::hz(FS).unwrap();
    let audio = stereo_audio(N, 586);
    let mut rng = Pcg32::new(586, 7);
    let mut raw: Vec<Packet> = (0..20)
        .map(|_| {
            let mut d = vec![0x03];
            d.extend((0..8).map(|_| rng.next_u32() as u8));
            packet(&d)
        })
        .collect();
    raw.push(packet(b"\x09junk"));
    let schedule: Vec<(usize, Vec<Packet>)> = vec![
        (
            0,
            vec![packet(b"\x01SDRK TST"), packet(b"\x02Hello from sdrk!")],
        ),
        (30_000, raw),
        (300_000, vec![packet(b"\x02Another text")]),
    ];
    let change_at = 300_000;

    // The blocks, cut at the same pieces.
    let cfg = RdsEncodeConfig {
        pi: 53539,
        pty: 10,
        tp: false,
    };
    let mut enc = RdsEncodeBlock::<C32, _>::new(&cfg, rate, &mut VecStorage).unwrap();
    let mut mpx =
        FmStereoRdsModBlock::new(&FmStereoRdsModConfig::default(), rate, &mut VecStorage).unwrap();
    let mut want = Vec::with_capacity(N);
    let mut pending: Vec<Packet> = Vec::new();
    for (k, piece) in audio.chunks(PIECE).enumerate() {
        let at = k * PIECE;
        for (_, packets) in schedule.iter().filter(|e| e.0 == at) {
            pending.extend(packets.iter().copied());
        }
        if at == change_at {
            mpx.set_param("rds", ParamValue::F64(0.1)).unwrap();
        }
        let mut b = vec![0.0f32; piece.len()];
        enc.process((piece, &pending), &mut b).unwrap();
        pending.drain(..enc.consumed_packets());
        let mut y = vec![0.0f32; piece.len()];
        mpx.process((piece, &b), &mut y).unwrap();
        want.extend(y);
    }
    assert_eq!(enc.ignored_packets(), 1);

    let mut p = build(&transmitter(FS, pi(), json!({})).to_string());
    let names: Vec<_> = p.stats().iter().map(|s| s.name.clone()).collect();
    let rds_index = names.iter().position(|n| n == "rds").unwrap();
    assert_eq!(
        p.stats()[rds_index].readings,
        vec![("ignored_packets", 0.0)]
    );
    let mut got = Vec::with_capacity(N);
    let mut buf = vec![0f32; 8192];
    for (k, piece) in audio.chunks(PIECE).enumerate() {
        let at = k * PIECE;
        for (_, packets) in schedule.iter().filter(|e| e.0 == at) {
            p.push_packets("text", packets).unwrap();
        }
        if at == change_at {
            p.set_param("mpx", "rds", json!(0.1)).unwrap();
        }
        p.push("audio", InputData::C32(piece)).unwrap();
        loop {
            let n = p.read_stream::<f32>("out", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            got.extend_from_slice(&buf[..n]);
        }
    }
    assert_eq!(got.len(), want.len());
    assert_eq!(bits(&got), bits(&want));
    assert_eq!(
        p.stats()[rds_index].readings,
        vec![("ignored_packets", 1.0)]
    );
    // The defaults keep the multiplex within main + pilot + rds = 1 for this audio.
    assert!(want.iter().all(|y| y.abs() <= 1.0 + 1e-6));
}
