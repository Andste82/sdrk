//! Packet inputs and outputs in the engine behind the JavaScript API (M37 §4): layouts, the
//! packet objects, push and read, pull mode with an idle source and the errors.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::large_stack_arrays,
    clippy::large_stack_frames
)]

use sdrk_pipeline::{Packet, PacketFlags};
use sdrk_testkit::Pcg32;
use sdrk_wasm::engine::{
    Dtype, Dyn, Engine, Lanes, LanesMut, Layout, MAX_PACKET_CHUNK, OutLayout, SourceBuf, WasmError,
    packet_dyn, packet_from,
};
use serde_json::json;

const FRAME_LOOP: &str = include_str!("../../../examples/flows/frame_loop.json");

fn random_packets(n: usize, stream: u64) -> Vec<Packet> {
    let mut rng = Pcg32::new(479, stream);
    (0..n)
        .map(|_| {
            let len = rng.below(254) as usize;
            let data: Vec<u8> = (0..len).map(|_| rng.below(256) as u8).collect();
            Packet::new(0, PacketFlags::NONE, &data).unwrap()
        })
        .collect()
}

fn tx_flow() -> String {
    json!({
        "sdrk": 1, "name": "tx",
        "inputs": { "msg": { "type": "packets", "rate_hz": 48000 } },
        "nodes": { "enc": { "kind": "frame_encode" } },
        "edges": [ { "from": "msg", "to": "enc" }, { "from": "enc", "to": "tx" } ],
        "outputs": { "tx": { "type": "stream" } }
    })
    .to_string()
}

fn rx_flow() -> String {
    json!({
        "sdrk": 1, "name": "rx",
        "settings": { "max_chunk": 4096 },
        "inputs": { "bits": { "type": "f32", "rate_hz": 48000 } },
        "nodes": { "dec": { "kind": "frame_decode" } },
        "edges": [ { "from": "bits", "to": "dec" }, { "from": "dec", "to": "out" } ],
        "outputs": { "out": { "type": "packets", "queue": 1024 } }
    })
    .to_string()
}

#[test]
fn packet_ports_have_the_packet_layout() {
    let e = Engine::from_json(FRAME_LOOP).unwrap();
    let packets = Layout {
        dtype: Dtype::Packets,
        lanes: 1,
    };
    assert_eq!(e.input_layout("msg").unwrap(), packets);
    assert_eq!(e.output_layout("out").unwrap(), OutLayout::Stream(packets));
    assert_eq!(Dtype::Packets.array_name(), "Array");
    // `max_chunk` packets per push, at most `MAX_PACKET_CHUNK` per read.
    assert_eq!(e.input_capacity("msg").unwrap(), 4096);
    assert_eq!(e.chunk("out").unwrap(), 4096.min(MAX_PACKET_CHUNK));
}

#[test]
fn packet_objects_use_hex_and_a_bigint_index() {
    let p = Packet::new(
        u64::MAX,
        PacketFlags::CRC_OK | PacketFlags::CORRECTED,
        &[0, 0xab],
    )
    .unwrap();
    assert_eq!(
        packet_dyn(&p),
        Dyn::Obj(vec![
            ("index", Dyn::U64(u64::MAX)),
            ("flags", Dyn::Num(5.0)),
            ("data", Dyn::Str("00ab".into())),
        ])
    );
    assert_eq!(packet_from(u64::MAX, 5, "00ab").unwrap(), p);
    for bad in ["AB", "abc", "zz", " ab", &"00".repeat(257)] {
        let e = packet_from(0, 0, bad).unwrap_err();
        assert_eq!(e.kind, "InvalidArgument", "{bad}");
        assert!(e.message.starts_with("packet data: "), "{}", e.message);
    }
    assert_eq!(packet_from(0, 0, &"ff".repeat(256)).unwrap().len(), 256);
}

#[test]
fn frame_loop_round_trip_through_the_engine() {
    let mut e = Engine::from_json(FRAME_LOOP).unwrap();
    let packets = random_packets(300, 1);
    let mut got: Vec<Packet> = Vec::new();
    let mut rng = Pcg32::new(479, 2);
    let mut at = 0;
    while at < packets.len() {
        let n = (1 + rng.below(16) as usize).min(packets.len() - at);
        e.push_packets("msg", &packets[at..at + n]).unwrap();
        at += n;
        // Read in small steps: a read moves at most `max` packets.
        loop {
            let chunk = e.read_packets("out", 3).unwrap();
            assert!(chunk.len() <= 3);
            if chunk.is_empty() {
                break;
            }
            got.extend_from_slice(chunk);
        }
    }
    assert_eq!(got.len(), packets.len());
    let mut last = None;
    for (g, p) in got.iter().zip(&packets) {
        assert_eq!(g.data(), p.data());
        assert_eq!(g.flags, PacketFlags::CRC_OK);
        assert!(last.is_none_or(|l| g.index > l));
        last = Some(g.index);
    }
    assert_eq!(e.available("out").unwrap(), 0);
    assert_eq!(e.dropped("out").unwrap(), 0);
}

#[test]
fn pull_mode_with_packets_and_idle_gaps() {
    let packets = random_packets(40, 3);
    let mut tx = Engine::from_json(&tx_flow()).unwrap();
    let mut rx = Engine::from_json(&rx_flow()).unwrap();
    let mut next = 0usize;
    let mut calls = 0usize;
    let mut source = |input: &str, buf: SourceBuf<'_>| -> Result<usize, WasmError> {
        assert_eq!(input, "msg");
        assert_eq!(buf.layout.dtype, Dtype::Packets);
        let LanesMut::Packets(dst) = buf.lanes else {
            panic!("a packet input gets a packet buffer");
        };
        calls += 1;
        // Every third call has nothing to send: the encoder sends its idle signal.
        if calls.is_multiple_of(3) || next >= packets.len() {
            return Ok(0);
        }
        let n = 2.min(buf.max_samples).min(packets.len() - next);
        dst[..n].copy_from_slice(&packets[next..next + n]);
        next += n;
        Ok(n)
    };
    let mut got: Vec<Packet> = Vec::new();
    let mut samples = 0usize;
    // Enough samples for every frame (at most 2112 samples each) plus idle gaps.
    while samples < 40 * 2112 + 200_000 {
        let pulled = tx.pull("tx", 4096, &mut source).unwrap();
        assert_eq!(pulled.samples, 4096, "an idle source never ends the stream");
        let Lanes::F32(bits) = pulled.data else {
            panic!("the encoder output is f32");
        };
        samples += bits.len();
        let bits = bits.to_vec();
        rx.push_f32("bits", &bits, None).unwrap();
        got.extend_from_slice(rx.read_packets("out", 1024).unwrap());
    }
    assert_eq!(next, packets.len());
    assert_eq!(got.len(), packets.len());
    for (g, p) in got.iter().zip(&packets) {
        assert_eq!(g.data(), p.data());
        assert_eq!(g.flags, PacketFlags::CRC_OK);
    }
}

#[test]
fn wrong_types_and_limits_are_errors() {
    let mut e = Engine::from_json(FRAME_LOOP).unwrap();
    // Typed arrays do not go into a packet input.
    let err = e.push_f32("msg", &[0.0; 4], None).unwrap_err();
    assert_eq!(err.kind, "InvalidArgument");
    assert!(err.message.contains("carries packets"), "{}", err.message);
    // Packets do not go into a sample input, and samples are not read as packets.
    let mut rx = Engine::from_json(&rx_flow()).unwrap();
    let err = rx.push_packets("bits", &[Packet::EMPTY]).unwrap_err();
    assert_eq!(err.kind, "InvalidArgument");
    assert!(err.message.contains("Float32Array"), "{}", err.message);
    let mut tx = Engine::from_json(&tx_flow()).unwrap();
    assert_eq!(
        tx.read_packets("tx", 4).unwrap_err().kind,
        "InvalidArgument"
    );
    // A packet output is neither a sample stream nor a transmit output.
    let mut none = |_: &str, _: SourceBuf<'_>| Ok(0);
    assert_eq!(
        e.pull("out", 16, &mut none).unwrap_err().kind,
        "InvalidArgument"
    );
    // More packets than `max_chunk` per push.
    let many = vec![Packet::EMPTY; 4097];
    assert_eq!(
        e.push_packets("msg", &many).unwrap_err().kind,
        "LengthMismatch"
    );
    // A payload the encoder cannot frame (254 bytes).
    let long = Packet::new(0, PacketFlags::NONE, &[0; 254]).unwrap();
    let err = e.push_packets("msg", &[long]).unwrap_err();
    assert_ne!(err.kind, "Internal");
    assert_eq!(e.read_packets("out", 8).unwrap().len(), 0);
    assert_eq!(
        e.read_packets("nope", 8).unwrap_err().kind,
        "InvalidArgument"
    );
    assert_eq!(
        e.push_packets("nope", &[]).unwrap_err().kind,
        "InvalidArgument"
    );
}

#[test]
fn reset_discards_queued_packets() {
    let mut e = Engine::from_json(FRAME_LOOP).unwrap();
    e.push_packets("msg", &random_packets(5, 4)).unwrap();
    assert_eq!(e.available("out").unwrap(), 5);
    e.reset();
    assert_eq!(e.available("out").unwrap(), 0);
    assert_eq!(e.read_packets("out", 8).unwrap().len(), 0);
}

/// The payload limit of the JSON Lines files (`sdrk-io`, which does not depend on
/// `sdrk-proto`) is the capacity of a packet.
#[test]
fn the_jsonl_limit_is_the_packet_capacity() {
    let max = sdrk_io::packets::MAX_DATA_BYTES;
    assert!(Packet::new(0, PacketFlags::NONE, &vec![0; max]).is_ok());
    assert!(Packet::new(0, PacketFlags::NONE, &vec![0; max + 1]).is_err());
    let hex = "ab".repeat(max);
    assert_eq!(packet_from(1, 1, &hex).unwrap().data().len(), max);
    assert_eq!(
        packet_from(1, 1, &format!("{hex}ab")).unwrap_err().kind,
        "InvalidArgument"
    );
}
