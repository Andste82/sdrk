//! M38 test 7: the test flow `tests/flows/bpsk_sync_test.json` (`symbol_sync` -> `slice bpsk`
//! -> `frame_decode`) decodes 100 frames that a `frame_encode` pipeline made and a test-side
//! raised-cosine modulator sent.
//!
//! The transmitter: `frame_encode` (4 preamble bytes) turns the packets into one `+1`/`-1`
//! sample per bit; the test maps every sample to a BPSK symbol (`> 0` is bit 1) and shapes the
//! symbols with the raised-cosine pulse (roll-off 0.35) of `PskTest` at 4.1 samples per symbol,
//! after 1024 alternating training symbols (lock of the synchronizer) and before 64 more
//! (they flush the last frame through the receiver). The channel: 50 ppm clock drift and white
//! Gaussian noise at Eb/N0 = 20 dB (seed 493, D559). The flow uses the block defaults of
//! `symbol_sync` (D565).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::print_stderr,
    clippy::large_stack_frames,
    clippy::large_stack_arrays
)]

mod common;

use num_complex::Complex64;
use sdrk::C32;
use sdrk_pipeline::{InputData, Packet, PacketFlags, Pipeline};
use sdrk_testkit::Pcg32;
use sdrk_testkit::channel::{Awgn, ChannelChain, ClockDrift, Impairment};
use sdrk_testkit::r#gen::{Psk, PskTest};
use serde_json::json;

use common::{build, workspace_file};

/// Seed of the test (payloads: stream 60; noise: the `Awgn` of the seed; D559).
const SEED: u64 = 493;
/// Frames sent.
const FRAMES: usize = 100;
/// Preamble bytes of the encoder.
const PREAMBLE: usize = 4;
/// Symbol rate and sample rate of the flow in Hz (4.1 samples per symbol).
const SYMBOL_RATE: u64 = 10_000;
const SAMPLE_RATE: f64 = 41_000.0;
/// Training symbols before and after the frames.
const LEAD: usize = 1024;
const TAIL: usize = 64;

/// Bits of the frame of a payload of `len` bytes: preamble, sync word, length, payload, CRC.
fn frame_bits(len: usize) -> usize {
    (PREAMBLE + 4 + 1 + len + 2) * 8
}

/// `FRAMES` random packets with 0 to 253 bytes.
fn random_packets() -> Vec<Packet> {
    let mut rng = Pcg32::new(SEED, 60);
    (0..FRAMES)
        .map(|_| {
            let len = rng.below(254) as usize;
            let data: Vec<u8> = (0..len).map(|_| rng.next_u32() as u8).collect();
            Packet::new(0, PacketFlags::NONE, &data).unwrap()
        })
        .collect()
}

/// The encoder: packets in, one `+1`/`-1` sample per bit out.
fn encode(packets: &[Packet]) -> Vec<f32> {
    let doc = json!({
        "sdrk": 1, "name": "tx", "settings": { "precision": "f32", "max_chunk": 4096 },
        "inputs": { "msg": { "type": "packets", "rate_hz": SYMBOL_RATE } },
        "nodes": { "enc": { "kind": "frame_encode", "params": { "preamble_bytes": PREAMBLE } } },
        "edges": [ { "from": "msg", "to": "enc" }, { "from": "enc", "to": "tx" } ],
        "outputs": { "tx": { "type": "stream" } }
    });
    let mut tx = build(&doc.to_string());
    let mut stream = Vec::new();
    let mut buf = vec![0f32; 4096];
    for p in packets {
        tx.push_packets("msg", core::slice::from_ref(p)).unwrap();
        loop {
            let n = tx.read_stream::<f32>("tx", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            stream.extend_from_slice(&buf[..n]);
        }
    }
    stream
}

fn read_all(p: &mut Pipeline, into: &mut Vec<Packet>) {
    let mut buf = [Packet::default(); 64];
    loop {
        let n = p.read_packets("out", &mut buf).unwrap();
        if n == 0 {
            return;
        }
        into.extend_from_slice(&buf[..n]);
    }
}

#[test]
fn bpsk_sync_test_decodes_100_frames() {
    let packets = random_packets();
    let frames = encode(&packets);
    let expected: usize = packets.iter().map(|p| frame_bits(p.data().len())).sum();
    assert_eq!(frames.len(), expected);
    assert!(frames.iter().all(|&x| x == 1.0 || x == -1.0));

    // Training, frames, training as bits (a sample above 0 is bit 1).
    let mut bits: Vec<u8> = (0..LEAD).map(|k| (k % 2) as u8).collect();
    bits.extend(frames.iter().map(|&x| u8::from(x > 0.0)));
    bits.extend((0..TAIL).map(|k| (k % 2) as u8));
    let g = PskTest::new(Psk::Bpsk, SAMPLE_RATE / SYMBOL_RATE as f64, 0.35);
    let mut tx = vec![Complex64::new(0.0, 0.0); g.samples(&bits)];
    g.generate(&bits, 0, &mut tx);
    let power = tx.iter().map(Complex64::norm_sqr).sum::<f64>() / tx.len() as f64;
    let mut channel = ChannelChain::new()
        .with(ClockDrift::new(50.0).unwrap())
        .with(Awgn::from_ebn0(20.0, 1, g.sps, power, SEED).unwrap());
    let mut rx = Vec::new();
    channel.process(&tx, &mut rx);
    let rx: Vec<C32> = rx
        .iter()
        .map(|v| C32::new(v.re as f32, v.im as f32))
        .collect();

    let text = std::fs::read_to_string(workspace_file(
        "crates/pipeline/sdrk-pipeline/tests/flows/bpsk_sync_test.json",
    ))
    .unwrap();
    let mut p = build(&text);
    let mut got = Vec::new();
    for chunk in rx.chunks(4096) {
        p.push("rx", InputData::C32(chunk)).unwrap();
        read_all(&mut p, &mut got);
    }
    eprintln!(
        "bpsk_sync_test: {} of {FRAMES} frames decoded from {} samples",
        got.len(),
        rx.len()
    );
    assert_eq!(got.len(), FRAMES);
    for (k, (sent, out)) in packets.iter().zip(&got).enumerate() {
        assert_eq!(out.data(), sent.data(), "packet {k}");
        assert_eq!(out.flags, PacketFlags::CRC_OK, "packet {k}");
    }
    // The frames follow each other without a gap in the soft bits: no symbol was lost or
    // inserted between the first and the last sync word.
    for k in 1..FRAMES {
        let step = got[k].index - got[k - 1].index;
        assert_eq!(
            step,
            frame_bits(packets[k - 1].data().len()) as u64,
            "packet {k}"
        );
    }
    assert_eq!(p.dropped("out").unwrap(), 0);
}
