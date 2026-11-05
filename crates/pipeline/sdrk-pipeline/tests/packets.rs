//! Packet inputs, outputs and edges (M37): validation, push and pull mode, drop policies, the
//! proof flow `frame_loop` (also with the PN9 whitening of M39).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::items_after_statements,
    clippy::float_cmp,
    clippy::doc_markdown
)]

mod common;

use sdrk_blocks::PortType;
use sdrk_device::BlockFlags;
use sdrk_pipeline::{
    BuildOptions, DiagCode, Error, InputData, Packet, PacketFlags, PipelineSpec, PortKind,
    PullError, PullSource, Registry, Report,
};
use sdrk_testkit::Pcg32;
use serde_json::{Value, json};

use common::flow;

/// The one seed of M37.
const SEED: u64 = 479;

/// Preamble bytes of the encoder in `frame_loop`.
const PREAMBLE: usize = 4;

/// Bits (samples) of the frame of a payload of `len` bytes: preamble, sync word, length,
/// payload, CRC.
fn frame_samples(len: usize) -> usize {
    (PREAMBLE + 4 + 1 + len + 2) * 8
}

/// The bits of the sync word 0x1ACFFC1D as samples, most significant bit first.
fn sync_samples() -> Vec<f32> {
    (0..32)
        .map(|k| {
            if (0x1ACF_FC1D_u32 >> (31 - k)) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// `count` random packets with 0 to 253 bytes; the index and the flags are 0.
fn random_packets(count: usize, stream: u64) -> Vec<Packet> {
    let mut rng = Pcg32::new(SEED, stream);
    (0..count)
        .map(|_| {
            let len = rng.below(254) as usize;
            let data: Vec<u8> = (0..len).map(|_| rng.next_u32() as u8).collect();
            Packet::new(0, PacketFlags::NONE, &data).unwrap()
        })
        .collect()
}

fn validate(spec: &Value) -> Report {
    Registry::standard().validate_json(&spec.to_string(), &BuildOptions::default())
}

fn build(spec: &Value) -> sdrk_pipeline::Pipeline {
    Registry::standard()
        .build_json(&spec.to_string(), &BuildOptions::default())
        .unwrap_or_else(|e| panic!("build failed: {e}"))
}

fn has(report: &Report, code: DiagCode, path: &str) -> bool {
    report
        .errors
        .iter()
        .any(|d| d.code == code && d.path == path)
}

fn read_all(p: &mut sdrk_pipeline::Pipeline, output: &str, into: &mut Vec<Packet>) {
    let mut buf = [Packet::default(); 64];
    loop {
        let n = p.read_packets(output, &mut buf).unwrap();
        if n == 0 {
            return;
        }
        into.extend_from_slice(&buf[..n]);
    }
}

/// The loop flow `frame_loop` with other output settings.
fn frame_loop_with(output: Value) -> Value {
    let mut spec: Value = serde_json::from_str(&flow("frame_loop")).unwrap();
    spec["outputs"]["out"] = output;
    spec
}

// ------------------------------------------------------------------ the proof flow

#[test]
fn frame_loop_is_valid_and_sized_as_specified() {
    let text = flow("frame_loop");
    let report = Registry::standard().validate_json(&text, &BuildOptions::default());
    assert!(report.is_ok(), "{}", report.to_json());
    assert!(report.warnings.is_empty(), "{}", report.to_json());
    let p = Registry::standard()
        .build_json(&text, &BuildOptions::default())
        .unwrap();
    assert_eq!(p.input_port("msg").unwrap(), (PortType::Packets, 4096));
    assert_eq!(
        p.output_port("out").unwrap(),
        PortKind::Stream(PortType::Packets)
    );
    assert_eq!(p.input_rate("msg").unwrap().num(), 48_000);
    // Buffers: the packet input holds max_chunk + 4 packets, the encoder output
    // max_chunk + min_input + 64 samples, the decoder output ceil(4096 / 56) + 4 packets, the
    // output queue 4096 packets; both blocks need no storage.
    let packet = core::mem::size_of::<Packet>();
    let expected = (4096 + 4) * packet + (4096 + 1 + 64) * 4 + (74 + 4) * packet + 4096 * packet;
    assert_eq!(p.report().memory_bytes, expected);
    // The estimate of `validate` is the same figure.
    assert_eq!(report.memory_bytes, expected);
    // Cost 1 per sample for each of the two nodes, per sample of the encoder output.
    assert!((report.cost_per_input_sample - 2.0).abs() < 1e-9);
    let kinds: Vec<&str> = report.nodes.iter().map(|n| n.kind.as_str()).collect();
    assert_eq!(kinds, ["frame_encode", "frame_decode"]);
    // The packet input is the clock domain of both nodes.
    assert!(report.nodes.iter().all(|n| n.clock_domain == "msg"));
}

#[test]
fn frame_loop_exact() {
    let packets = random_packets(1000, 1);
    let mut p = common::build(&flow("frame_loop"));
    let mut rng = Pcg32::new(SEED, 2);
    let mut got = Vec::new();
    let mut at = 0;
    while at < packets.len() {
        let n = (1 + rng.below(64) as usize).min(packets.len() - at);
        p.push_packets("msg", &packets[at..at + n]).unwrap();
        at += n;
        read_all(&mut p, "out", &mut got);
    }
    assert_eq!(got.len(), packets.len());
    let mut start = 0;
    for (k, (sent, out)) in packets.iter().zip(&got).enumerate() {
        assert_eq!(out.data(), sent.data(), "packet {k}");
        assert_eq!(out.flags, PacketFlags::CRC_OK, "packet {k}");
        // The frames follow each other without a gap; the sync word follows the preamble.
        assert_eq!(out.index, (start + PREAMBLE * 8) as u64, "packet {k}");
        start += frame_samples(sent.len());
    }
    assert_eq!(p.dropped("out").unwrap(), 0);
    let stats = p.stats();
    assert_eq!(stats[0].samples_in, 1000);
    assert_eq!(stats[0].samples_out, start as u64);
    assert_eq!(stats[1].samples_in, start as u64);
    assert_eq!(stats[1].samples_out, 1000);
}

/// `frame_loop` with the `whitening` of the encoder and the decoder set (M39).
fn frame_loop_whitened(enc: &str, dec: &str) -> Value {
    let mut spec: Value = serde_json::from_str(&flow("frame_loop")).unwrap();
    spec["nodes"]["enc"]["params"]["whitening"] = json!(enc);
    spec["nodes"]["dec"]["params"]["whitening"] = json!(dec);
    spec
}

#[test]
fn frame_loop_with_pn9_whitening() {
    let packets = random_packets(200, 3);
    let mut p = build(&frame_loop_whitened("pn9", "pn9"));
    p.push_packets("msg", &packets).unwrap();
    let mut got = Vec::new();
    read_all(&mut p, "out", &mut got);
    assert_eq!(got.len(), packets.len());
    let mut start = 0;
    for (k, (sent, out)) in packets.iter().zip(&got).enumerate() {
        assert_eq!(out.data(), sent.data(), "packet {k}");
        assert_eq!(out.flags, PacketFlags::CRC_OK, "packet {k}");
        assert_eq!(out.index, (start + PREAMBLE * 8) as u64, "packet {k}");
        start += frame_samples(sent.len());
    }
    // Whitening on one side only: no packet passes the CRC.
    for (enc, dec) in [("pn9", "none"), ("none", "pn9")] {
        let mut p = build(&frame_loop_whitened(enc, dec));
        p.push_packets("msg", &packets).unwrap();
        let mut got = Vec::new();
        read_all(&mut p, "out", &mut got);
        assert!(got.is_empty(), "enc {enc} dec {dec}: {} packets", got.len());
    }
    // An unknown whitening is rejected at its parameter path.
    let report = validate(&frame_loop_whitened("pn15", "none"));
    assert!(
        report
            .errors
            .iter()
            .any(|d| d.path == "nodes.enc.params.whitening"),
        "{}",
        report.to_json()
    );
}

/// Sends the packets of a list in random small groups and, in between, nothing (idle).
struct Sender {
    packets: Vec<Packet>,
    next: usize,
    rng: Pcg32,
    calls: usize,
    idle_calls: usize,
}

impl PullSource for Sender {
    fn provide(&mut self, input: &str, max: usize) -> Result<InputData<'_>, PullError> {
        assert_eq!(input, "msg");
        assert!(max >= 1);
        self.calls += 1;
        if self.rng.below(2) == 0 || self.next == self.packets.len() {
            self.idle_calls += 1;
            return Ok(InputData::Packets(&[]));
        }
        let n = (1 + self.rng.below(3) as usize)
            .min(max)
            .min(self.packets.len() - self.next);
        let from = self.next;
        self.next += n;
        Ok(InputData::Packets(&self.packets[from..from + n]))
    }
}

/// The encoder half of `frame_loop` with a stream output, for pull mode.
fn tx_flow(idle: &str) -> Value {
    json!({
        "sdrk": 1, "name": "tx",
        "inputs": { "msg": { "type": "packets", "rate_hz": 48000 } },
        "nodes": { "enc": { "kind": "frame_encode", "params": { "idle": idle } } },
        "edges": [ { "from": "msg", "to": "enc" }, { "from": "enc", "to": "tx" } ],
        "outputs": { "tx": { "type": "stream" } }
    })
}

/// The decoder half of `frame_loop` with an `f32` input.
fn rx_flow() -> Value {
    json!({
        "sdrk": 1, "name": "rx",
        "inputs": { "bits": { "type": "f32", "rate_hz": 48000 } },
        "nodes": { "dec": { "kind": "frame_decode" } },
        "edges": [ { "from": "bits", "to": "dec" }, { "from": "dec", "to": "out" } ],
        "outputs": { "out": { "type": "packets", "queue": 1024 } }
    })
}

#[test]
fn frame_loop_exact_in_pull_mode_with_an_idle_source() {
    for idle in ["zero", "preamble"] {
        let packets = random_packets(1000, 1);
        let mut tx = build(&tx_flow(idle));
        let mut rx = build(&rx_flow());
        let mut source = Sender {
            packets: packets.clone(),
            next: 0,
            rng: Pcg32::new(SEED, 3),
            calls: 0,
            idle_calls: 0,
        };
        let mut sizes = Pcg32::new(SEED, 4);
        let mut stream: Vec<f32> = Vec::new();
        let mut got = Vec::new();
        let mut buf = vec![0f32; 4096];
        while got.len() < packets.len() {
            assert!(stream.len() < 5_000_000, "{idle}: the loop does not end");
            let want = 1 + sizes.below(4096) as usize;
            let (n, meta) = tx
                .pull_samples::<f32>("tx", &mut buf[..want], &mut source)
                .unwrap();
            // An idle encoder never underruns: every pull is complete.
            assert_eq!(n, want, "{idle}");
            assert!(!meta.flags.contains(BlockFlags::UNDERRUN));
            assert_eq!(meta.start.0, stream.len() as u64);
            stream.extend_from_slice(&buf[..n]);
            for chunk in buf[..n].chunks(4096) {
                rx.push("bits", InputData::F32(chunk)).unwrap();
            }
            read_all(&mut rx, "out", &mut got);
        }
        assert!(source.idle_calls > 0, "{idle}: the source was never idle");
        assert_eq!(got.len(), packets.len(), "{idle}");
        let sync = sync_samples();
        let mut last = None;
        for (k, (sent, out)) in packets.iter().zip(&got).enumerate() {
            assert_eq!(out.data(), sent.data(), "{idle}: packet {k}");
            assert_eq!(out.flags, PacketFlags::CRC_OK, "{idle}: packet {k}");
            // The index is the sample position of the sync word in the pulled stream.
            let i = out.index as usize;
            assert_eq!(&stream[i..i + 32], &sync[..], "{idle}: packet {k}");
            assert!(last.is_none_or(|l| i > l), "{idle}: packet {k} in order");
            last = Some(i);
        }
        if idle == "zero" {
            // Idle samples are 0.0 between frames.
            assert!(stream.contains(&0.0));
        } else {
            assert!(stream.iter().all(|&x| x == 1.0 || x == -1.0));
        }
    }
}

#[test]
fn bursts_behind_a_packet_input() {
    let mut tx = build(&tx_flow("zero"));
    let packets = random_packets(3, 5);
    let mut source = Sender {
        packets,
        next: 0,
        rng: Pcg32::new(SEED, 6),
        calls: 0,
        idle_calls: 0,
    };
    tx.begin_burst("tx", None).unwrap();
    let mut buf = vec![0f32; 1000];
    let (n, meta) = tx.pull_samples::<f32>("tx", &mut buf, &mut source).unwrap();
    assert_eq!(n, 1000);
    assert!(meta.flags.contains(BlockFlags::BURST_START));
    tx.end_burst("tx").unwrap();
    // The burst ends after what the encoder had produced (a frame is written whole when the
    // buffer has room, so a few samples may follow the 1000 pulled ones).
    let (tail, meta) = tx.pull_samples::<f32>("tx", &mut buf, &mut source).unwrap();
    assert!(tail < 1000);
    assert!(meta.flags.contains(BlockFlags::BURST_END));
    // Between bursts nothing is produced.
    let (n, _) = tx.pull_samples::<f32>("tx", &mut buf, &mut source).unwrap();
    assert_eq!(n, 0);
}

// ------------------------------------------------------------------ push mode details

#[test]
fn an_encoder_without_packets_produces_nothing_in_push_mode() {
    let mut tx = build(&tx_flow("preamble"));
    tx.push_packets("msg", &[]).unwrap();
    assert_eq!(tx.available("tx").unwrap(), 0);
    let msg = Packet::new(0, PacketFlags::CRC_OK, b"abc").unwrap();
    tx.push_packets("msg", &[msg]).unwrap();
    assert_eq!(tx.available("tx").unwrap(), frame_samples(3));
    let mut out = vec![0f32; frame_samples(3)];
    assert_eq!(tx.read_stream("tx", &mut out).unwrap(), out.len());
    assert_eq!(&out[32..64], &sync_samples()[..]);
    tx.push_packets("msg", &[]).unwrap();
    assert_eq!(tx.available("tx").unwrap(), 0);
}

#[test]
fn drop_oldest_on_a_full_packet_queue() {
    let packets = random_packets(10, 7);
    let mut p = build(&frame_loop_with(
        json!({ "type": "packets", "queue": 4, "drop": "drop_oldest" }),
    ));
    p.push_packets("msg", &packets).unwrap();
    assert_eq!(p.available("out").unwrap(), 4);
    assert_eq!(p.dropped("out").unwrap(), 6);
    let mut got = Vec::new();
    read_all(&mut p, "out", &mut got);
    let kept: Vec<&[u8]> = got.iter().map(Packet::data).collect();
    let newest: Vec<&[u8]> = packets[6..].iter().map(Packet::data).collect();
    assert_eq!(kept, newest);
}

#[test]
fn drop_newest_and_error_on_a_full_packet_queue() {
    let packets = random_packets(10, 8);
    let mut p = build(&frame_loop_with(
        json!({ "type": "packets", "queue": 4, "drop": "drop_newest" }),
    ));
    p.push_packets("msg", &packets).unwrap();
    assert_eq!(p.dropped("out").unwrap(), 6);
    let mut got = Vec::new();
    read_all(&mut p, "out", &mut got);
    assert_eq!(got.len(), 4);
    assert_eq!(got[0].data(), packets[0].data());

    // `error` (the default) with the default queue of 64 packets: a push is accepted only
    // when the queue has room for the worst case (2112 samples per packet, at most one packet
    // per 56 samples), limited to the capacity; nothing is taken otherwise.
    let mut p = build(&frame_loop_with(json!({ "type": "packets" })));
    p.push_packets("msg", &packets).unwrap();
    assert_eq!(p.available("out").unwrap(), 10);
    // 10 more packets could give 378 packets: more than the 54 free slots.
    let err = p.push_packets("msg", &packets).unwrap_err();
    assert!(
        matches!(err, Error::OutputFull(ref o) if o == "out"),
        "{err}"
    );
    assert_eq!(p.available("out").unwrap(), 10);
    let mut got = Vec::new();
    read_all(&mut p, "out", &mut got);
    p.push_packets("msg", &packets[..1]).unwrap();
    assert_eq!(p.available("out").unwrap(), 1);
}

#[test]
fn packet_sources_fan_out() {
    // One packet input read by two encoders with different preambles and copied to an output
    // directly; every reader gets every packet.
    let spec = json!({
        "sdrk": 1, "name": "fan",
        "inputs": { "msg": { "type": "packets", "rate_hz": 9600 } },
        "nodes": {
            "enc_a": { "kind": "frame_encode" },
            "enc_b": { "kind": "frame_encode", "params": { "preamble_bytes": 1 } },
            "dec_a": { "kind": "frame_decode" },
            "dec_b": { "kind": "frame_decode" }
        },
        "edges": [
            { "from": "msg", "to": "enc_a" }, { "from": "msg", "to": "enc_b" },
            { "from": "msg", "to": "copy" },
            { "from": "enc_a", "to": "dec_a" }, { "from": "enc_b", "to": "dec_b" },
            { "from": "dec_a", "to": "out_a" }, { "from": "dec_b", "to": "out_b" },
            { "from": "dec_b", "to": "out_c" }
        ],
        "outputs": {
            "copy": { "type": "packets" },
            "out_a": { "type": "packets", "queue": 1024 },
            "out_b": { "type": "packets", "queue": 1024 },
            "out_c": { "type": "packets", "queue": 1024 }
        }
    });
    let mut p = build(&spec);
    let packets: Vec<Packet> = random_packets(5, 9)
        .into_iter()
        .enumerate()
        .map(|(k, q)| Packet::new(k as u64, PacketFlags::from_bits(1 << 16), q.data()).unwrap())
        .collect();
    p.push_packets("msg", &packets).unwrap();
    let mut copy = Vec::new();
    read_all(&mut p, "copy", &mut copy);
    // The direct copy is the input, index and flags included.
    assert_eq!(copy, packets);
    for (out, preamble) in [("out_a", 4), ("out_b", 1), ("out_c", 1)] {
        let mut got = Vec::new();
        read_all(&mut p, out, &mut got);
        assert_eq!(got.len(), 5, "{out}");
        let mut start = 0;
        for (sent, g) in packets.iter().zip(&got) {
            assert_eq!(g.data(), sent.data(), "{out}");
            assert_eq!(g.flags, PacketFlags::CRC_OK, "{out}");
            assert_eq!(g.index, (start + preamble * 8) as u64, "{out}");
            start += (preamble + 7 + sent.len()) * 8;
        }
    }
}

#[test]
fn wrong_calls_on_packet_ports() {
    let mut p = common::build(&flow("frame_loop"));
    let err = p.push("msg", InputData::F32(&[0.0; 4])).unwrap_err();
    assert!(
        matches!(
            err,
            Error::WrongType {
                expected: "packets",
                got: "f32",
                ..
            }
        ),
        "{err}"
    );
    let mut out = [0f32; 4];
    let err = p.read_stream("out", &mut out).unwrap_err();
    assert!(
        matches!(
            err,
            Error::WrongType {
                expected: "packets",
                got: "f32",
                ..
            }
        ),
        "{err}"
    );
    let too_many = vec![Packet::default(); 4097];
    let err = p.push_packets("msg", &too_many).unwrap_err();
    assert!(
        matches!(
            err,
            Error::ChunkTooLarge {
                len: 4097,
                max: 4096
            }
        ),
        "{err}"
    );
    let meta = sdrk_device::BlockMeta::default();
    let err = p
        .push_with_meta("msg", InputData::Packets(&[]), &meta)
        .unwrap_err();
    assert!(matches!(err, Error::WrongType { .. }), "{err}");
    // A payload that the length byte cannot hold.
    let long = Packet::new(0, PacketFlags::NONE, &[7u8; 254]).unwrap();
    let err = p.push_packets("msg", &[long]).unwrap_err();
    assert!(
        matches!(err, Error::Block { ref node, .. } if node == "enc"),
        "{err}"
    );

    let mut rx = build(&rx_flow());
    let err = rx.push_packets("bits", &[Packet::default()]).unwrap_err();
    assert!(
        matches!(
            err,
            Error::WrongType {
                expected: "f32",
                got: "packets",
                ..
            }
        ),
        "{err}"
    );
    let mut tx = build(&tx_flow("zero"));
    let mut packets = [Packet::default(); 2];
    let err = tx.read_packets("tx", &mut packets).unwrap_err();
    assert!(
        matches!(
            err,
            Error::WrongType {
                expected: "f32",
                got: "packets",
                ..
            }
        ),
        "{err}"
    );
}

#[test]
fn reset_restarts_the_packet_indices() {
    let packets = random_packets(3, 10);
    let mut p = common::build(&flow("frame_loop"));
    p.push_packets("msg", &packets).unwrap();
    let mut first = Vec::new();
    read_all(&mut p, "out", &mut first);
    p.reset();
    p.push_packets("msg", &packets).unwrap();
    let mut again = Vec::new();
    read_all(&mut p, "out", &mut again);
    assert_eq!(first, again);
}

// ------------------------------------------------------------------ validation

fn loop_spec() -> Value {
    serde_json::from_str(&flow("frame_loop")).unwrap()
}

#[test]
fn packet_edges_connect_only_packet_ports() {
    // A stream into the encoder.
    let mut spec = loop_spec();
    spec["inputs"]["msg"] = json!({ "type": "f32", "rate_hz": 48000 });
    let report = validate(&spec);
    assert!(
        has(&report, DiagCode::TypeMismatch, "edges[0].to"),
        "{}",
        report.to_json()
    );
    let message = &report.errors[0].message;
    assert!(message.contains("takes packets"), "{message}");

    // Packets into a stream node.
    let mut spec = loop_spec();
    spec["nodes"]["enc"] = json!({ "kind": "scale" });
    let report = validate(&spec);
    assert!(
        has(&report, DiagCode::TypeMismatch, "edges[0].to"),
        "{}",
        report.to_json()
    );
    assert!(report.errors[0].message.contains("delivers packets"));

    // A packet stream into a stream output and a sample stream into a packet output.
    let mut spec = loop_spec();
    spec["outputs"]["out"] = json!({ "type": "stream" });
    let report = validate(&spec);
    assert!(has(&report, DiagCode::TypeMismatch, "outputs.out.type"));
    assert!(report.errors[0].message.contains("type `packets`"));
    let mut spec = tx_flow("zero");
    spec["outputs"]["tx"] = json!({ "type": "packets" });
    let report = validate(&spec);
    assert!(has(&report, DiagCode::TypeMismatch, "outputs.tx.type"));
    assert!(
        report
            .errors
            .iter()
            .any(|d| d.message.contains("type `stream`"))
    );
    // Frames into a packet output.
    let spec = json!({
        "sdrk": 1, "name": "t",
        "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
        "nodes": { "fft": { "kind": "spectrum", "params": { "fft_size": 64 } } },
        "edges": [ { "from": "iq", "to": "fft" }, { "from": "fft", "to": "out" } ],
        "outputs": { "out": { "type": "packets" } }
    });
    assert!(has(
        &validate(&spec),
        DiagCode::TypeMismatch,
        "outputs.out.type"
    ));
}

#[test]
fn an_encoder_reads_packets_only_from_a_packet_input() {
    // A relay: the decoder's packets have a nominal sizing rate and the clock of `bits`, so
    // they cannot give the encoder its output rate (D526).
    let spec = json!({
        "sdrk": 1, "name": "relay",
        "inputs": { "bits": { "type": "f32", "rate_hz": 48000 } },
        "nodes": {
            "dec": { "kind": "frame_decode" },
            "enc": { "kind": "frame_encode" }
        },
        "edges": [
            { "from": "bits", "to": "dec" },
            { "from": "dec", "to": "enc" },
            { "from": "enc", "to": "tx" }
        ],
        "outputs": { "tx": { "type": "stream" } }
    });
    let report = validate(&spec);
    assert!(
        has(&report, DiagCode::TypeMismatch, "edges[1].to"),
        "{}",
        report.to_json()
    );
    assert_eq!(report.errors.len(), 1, "{}", report.to_json());
    assert!(
        report.errors[0]
            .message
            .contains("`dec` is not a pipeline input")
    );
    // The encoder straight behind a packet input stays valid.
    let report = validate(&tx_flow("zero"));
    assert!(report.errors.is_empty(), "{}", report.to_json());
}

#[test]
fn packet_outputs_take_a_queue_only() {
    for (output, code, path) in [
        (
            json!({ "type": "packets", "queue": 0 }),
            DiagCode::OutOfRange,
            "outputs.out.queue",
        ),
        (
            json!({ "type": "packets", "queue": 65_537 }),
            DiagCode::OutOfRange,
            "outputs.out.queue",
        ),
        (
            json!({ "type": "packets", "capacity": 8 }),
            DiagCode::InvalidValue,
            "outputs.out.capacity",
        ),
        (
            json!({ "type": "packets", "format": "cu8" }),
            DiagCode::InvalidValue,
            "outputs.out.format",
        ),
        (
            json!({ "type": "packet" }),
            DiagCode::InvalidValue,
            "outputs.out.type",
        ),
    ] {
        let report = validate(&frame_loop_with(output.clone()));
        assert!(has(&report, code, path), "{output}: {}", report.to_json());
        assert_eq!(report.errors.len(), 1, "{output}: {}", report.to_json());
    }
    let report = validate(&frame_loop_with(
        json!({ "type": "packets", "queue": 65_536 }),
    ));
    assert!(report.is_ok(), "{}", report.to_json());
}

#[test]
fn a_packet_input_without_rate_hz() {
    // Read by an encoder: `missing_param`.
    let mut spec = loop_spec();
    spec["inputs"]["msg"] = json!({ "type": "packets" });
    let report = validate(&spec);
    assert!(
        has(&report, DiagCode::MissingParam, "inputs.msg.rate_hz"),
        "{}",
        report.to_json()
    );
    assert_eq!(report.errors.len(), 1);
    assert!(report.errors[0].message.contains("encoder"));
    // A sample input still needs one.
    let mut spec = loop_spec();
    spec["inputs"]["bits"] = json!({ "type": "f32" });
    spec["edges"]
        .as_array_mut()
        .unwrap()
        .push(json!({ "from": "bits", "to": "out2" }));
    spec["outputs"]["out2"] = json!({ "type": "stream" });
    let report = validate(&spec);
    assert!(
        has(&report, DiagCode::MissingParam, "inputs.bits.rate_hz"),
        "{}",
        report.to_json()
    );
    // Read by nothing but an output: valid, and the input has no rate.
    let spec = json!({
        "sdrk": 1, "name": "relay",
        "inputs": { "msg": { "type": "packets" } },
        "edges": [ { "from": "msg", "to": "out" } ],
        "outputs": { "out": { "type": "packets" } }
    });
    let report = validate(&spec);
    assert!(report.is_ok(), "{}", report.to_json());
    let mut p = build(&spec);
    assert!(matches!(p.input_rate("msg"), Err(Error::InvalidState(_))));
    let msg = Packet::new(9, PacketFlags::CRC_FAIL, b"x").unwrap();
    p.push_packets("msg", &[msg, msg]).unwrap();
    let mut got = [Packet::default(); 4];
    assert_eq!(p.read_packets("out", &mut got).unwrap(), 2);
    assert_eq!(got[1], msg);
    // The model keeps `rate_hz` out of the JSON when there is none.
    let model = PipelineSpec::from_json(&spec.to_string()).unwrap();
    assert!(!model.to_json().contains("rate_hz"));
}

#[test]
fn a_packet_input_is_its_own_clock_domain() {
    // The encoded stream and a sample input of the same rate meet at `add`.
    let spec = json!({
        "sdrk": 1, "name": "mix",
        "inputs": {
            "msg": { "type": "packets", "rate_hz": 48000 },
            "noise": { "type": "f32", "rate_hz": 48000 }
        },
        "nodes": {
            "enc": { "kind": "frame_encode" },
            "sum": { "kind": "add" }
        },
        "edges": [
            { "from": "msg", "to": "enc" },
            { "from": "enc", "to": "sum" },
            { "from": "noise", "to": "sum", "port": "b" },
            { "from": "sum", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    });
    let report = validate(&spec);
    assert!(
        has(&report, DiagCode::ClockDomain, "edges[2].to"),
        "{}",
        report.to_json()
    );
    let message = &report.errors[0].message;
    assert!(
        message.contains("`msg`") && message.contains("`noise`"),
        "{message}"
    );
    // Two encoders of the same packet input are one clock: they can be added.
    let spec = json!({
        "sdrk": 1, "name": "twice",
        "inputs": { "msg": { "type": "packets", "rate_hz": 48000 } },
        "nodes": {
            "enc_a": { "kind": "frame_encode" },
            "enc_b": { "kind": "frame_encode" },
            "sum": { "kind": "add" }
        },
        "edges": [
            { "from": "msg", "to": "enc_a" },
            { "from": "msg", "to": "enc_b" },
            { "from": "enc_a", "to": "sum" },
            { "from": "enc_b", "to": "sum", "port": "b" },
            { "from": "sum", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    });
    let report = validate(&spec);
    assert!(report.is_ok(), "{}", report.to_json());
    let mut p = build(&spec);
    p.push_packets("msg", &[Packet::new(0, PacketFlags::NONE, b"z").unwrap()])
        .unwrap();
    let mut out = vec![0f32; frame_samples(1)];
    assert_eq!(p.read_stream("out", &mut out).unwrap(), frame_samples(1));
    assert!(out.iter().all(|&x| x == 2.0 || x == -2.0));
}

#[test]
fn packet_edges_cannot_cross_groups() {
    let mut spec = loop_spec();
    spec["groups"] = json!({ "rx": {} });
    spec["nodes"]["dec"]["group"] = json!("rx");
    // The sample edge `enc -> dec` crosses groups and is split.
    let spec_ok: PipelineSpec = serde_json::from_value(spec.clone()).unwrap();
    let parts = sdrk_pipeline::split_by_groups(&spec_ok, &Registry::standard()).unwrap();
    assert_eq!(parts.len(), 2);
    // The packet edge `dec -> enc2` cannot be split.
    let spec = json!({
        "sdrk": 1, "name": "t",
        "groups": { "tx": {} },
        "inputs": { "msg": { "type": "packets", "rate_hz": 48000 } },
        "nodes": {
            "enc": { "kind": "frame_encode" },
            "dec": { "kind": "frame_decode" },
            "enc2": { "kind": "frame_encode", "group": "tx" }
        },
        "edges": [
            { "from": "msg", "to": "enc" },
            { "from": "enc", "to": "dec" },
            { "from": "dec", "to": "enc2" },
            { "from": "enc2", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    });
    let spec: PipelineSpec = serde_json::from_value(spec).unwrap();
    let report = sdrk_pipeline::split_by_groups(&spec, &Registry::standard()).unwrap_err();
    assert!(
        report.errors.iter().any(|d| d.message.contains("packets")),
        "{}",
        report.to_json()
    );
}

#[test]
fn the_new_public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    use sdrk_blocks::{FrameDecodeBlock, FrameEncodeBlock};
    use sdrk_pipeline::{PacketInNode, PacketOutNode, VecStorage};
    assert_send_sync::<Packet>();
    assert_send_sync::<PacketFlags>();
    assert_send_sync::<PacketInNode<FrameEncodeBlock<VecStorage>>>();
    assert_send_sync::<PacketOutNode<FrameDecodeBlock<VecStorage>>>();
    assert_send_sync::<InputData<'static>>();
}
