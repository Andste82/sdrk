//! M40 test 5: the example flows `qpsk_transmitter` -> channel -> `qpsk_receiver` (D623).
//!
//! The transmitter runs in pull mode: the packet source is idle for the first
//! [`LEAD_SYMBOLS`] symbols (the encoder sends its idle preamble while the receiver's loops
//! pull in), then hands over the [`PACKETS`] packets, three per call, and is idle again for
//! [`TAIL_SYMBOLS`] symbols after the last call (the frames still queued in the transmitter
//! and the delay of the receiver). The `cs8` bytes are read as
//! complex samples, shifted by 2 kHz and given white Gaussian noise at Eb/N0 = 10 dB (both
//! from `sdrk_testkit::channel`, seed 558), scaled and rounded to the `cs16` input of the
//! receiver.
//!
//! Coherent QPSK leaves a carrier phase ambiguity of 90 degrees; `frame_decode` with
//! `polarity` both resolves 180 degrees only. The receiver flow is exactly equivariant to a
//! rotation of its input by 90 degrees (the filters are real, the AGC, Gardner and the Costas
//! detector are symmetric under the swaps and negations), so the test runs it on the channel
//! output rotated by 0, 90, 180 and 270 degrees: the two rotations that put the lock on 0 or
//! 180 degrees deliver at least 995 of the 1000 packets with `CRC_OK` and their payload
//! unchanged, and the same packets; the other two deliver none (the 90 degree hypothesis is
//! the `four` option of `frame_decode`, M44).
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
use sdrk::Rate;
use sdrk_pipeline::{
    BuildOptions, InputData, Packet, PacketFlags, Pipeline, PullError, PullSource, Registry,
};
use sdrk_testkit::Pcg32;
use sdrk_testkit::channel::{Awgn, ChannelChain, FrequencyOffset, Impairment};

use common::{build, flow};

/// Seed of the test (payloads: stream 61; noise: the `Awgn` of the seed).
const SEED: u64 = 558;
/// Packets sent.
const PACKETS: usize = 1000;
/// Packets that must arrive with `CRC_OK` and the payload unchanged.
const REQUIRED: usize = 995;
/// Sample rate of the transmitter output and of the receiver input in Hz.
const FS: f64 = 1_000_000.0;
/// Samples per symbol at `FS` (25 kBd).
const SPS: usize = 40;
/// Carrier offset of the channel in Hz (0.08 cycles per symbol).
const OFFSET_HZ: f64 = 2000.0;
/// Eb/N0 of the channel in dB.
const EBN0_DB: f64 = 10.0;
/// Idle symbols before the first packet: the Costas loop pulls in from 0.08 cycles per symbol
/// within about 10^4 symbols (D622).
const LEAD_SYMBOLS: usize = 20_000;
/// Symbols pulled after the source handed over the last packet (the frames still queued in
/// the encoder, then idle symbols that flush the filters and the decoder).
const TAIL_SYMBOLS: usize = 2000;
/// Bytes per pull of the transmitter (`cs8`: two bytes per sample).
const PULL_BYTES: usize = 8000;
/// Samples per push into the receiver (its `max_chunk`).
const PUSH_SAMPLES: usize = 4096;
/// RMS amplitude of the receiver input relative to the `cs16` full scale.
const RX_RMS: f64 = 0.1;

/// The packet source of the transmitter: idle until `open`, then up to three packets per call.
struct Sender {
    packets: Vec<Packet>,
    next: usize,
    open: bool,
}

impl PullSource for Sender {
    fn provide(&mut self, _input: &str, max: usize) -> Result<InputData<'_>, PullError> {
        if !self.open || self.next == self.packets.len() {
            return Ok(InputData::Packets(&[]));
        }
        let n = 3.min(max).min(self.packets.len() - self.next);
        let from = self.next;
        self.next += n;
        Ok(InputData::Packets(&self.packets[from..from + n]))
    }
}

/// `PACKETS` random packets with 0 to 63 bytes.
fn random_packets() -> Vec<Packet> {
    let mut rng = Pcg32::new(SEED, 61);
    (0..PACKETS)
        .map(|_| {
            let len = rng.below(64) as usize;
            let data: Vec<u8> = (0..len).map(|_| rng.next_u32() as u8).collect();
            Packet::new(0, PacketFlags::NONE, &data).unwrap()
        })
        .collect()
}

/// Runs the transmitter until every packet is sent and the tail is out; returns the complex
/// baseband (the `cs8` bytes divided by 128).
fn transmit(packets: &[Packet]) -> Vec<Complex64> {
    let mut tx = build(&flow("qpsk_transmitter"));
    let mut source = Sender {
        packets: packets.to_vec(),
        next: 0,
        open: false,
    };
    let mut out = Vec::new();
    let mut bytes = vec![0u8; PULL_BYTES];
    let mut tail_from = None;
    loop {
        source.open = out.len() >= LEAD_SYMBOLS * SPS;
        let (n, _) = tx.pull("rf", &mut bytes, &mut source).unwrap();
        assert_eq!(n, PULL_BYTES, "a packet input never underruns");
        out.extend(bytes.chunks_exact(2).map(|iq| {
            Complex64::new(
                f64::from(iq[0].cast_signed()) / 128.0,
                f64::from(iq[1].cast_signed()) / 128.0,
            )
        }));
        if source.next == packets.len() && tail_from.is_none() {
            tail_from = Some(out.len());
        }
        if tail_from.is_some_and(|t| out.len() >= t + TAIL_SYMBOLS * SPS) {
            return out;
        }
        assert!(out.len() < 50_000_000, "the transmitter does not finish");
    }
}

/// Rotates the `cs16` sample pairs by `quarter` times 90 degrees (exact: swaps and negations)
/// and returns their bytes.
fn rotate(iq: &[[i16; 2]], quarter: usize) -> Vec<u8> {
    iq.iter()
        .flat_map(|&[re, im]| {
            let [re, im] = match quarter {
                0 => [re, im],
                1 => [-im, re],
                2 => [-re, -im],
                _ => [im, -re],
            };
            let mut b = [0u8; 4];
            b[..2].copy_from_slice(&re.to_le_bytes());
            b[2..].copy_from_slice(&im.to_le_bytes());
            b
        })
        .collect()
}

fn read_all(p: &mut Pipeline, into: &mut Vec<Packet>) {
    let mut buf = [Packet::default(); 64];
    loop {
        let n = p.read_packets("packets", &mut buf).unwrap();
        if n == 0 {
            return;
        }
        into.extend_from_slice(&buf[..n]);
    }
}

/// Runs the receiver flow on `bytes`; returns the positions (in `packets`) of the sent packets
/// that came out with `CRC_OK` and unchanged, matched in order, and the number of packets out.
fn receive(bytes: &[u8], packets: &[Packet]) -> (Vec<usize>, usize) {
    let mut p = build(&flow("qpsk_receiver"));
    let mut got = Vec::new();
    for chunk in bytes.chunks(PUSH_SAMPLES * 4) {
        p.push("rf", InputData::Raw(chunk)).unwrap();
        read_all(&mut p, &mut got);
    }
    assert_eq!(p.dropped("packets").unwrap(), 0);
    let mut next = 0;
    let mut ok = Vec::new();
    for out in &got {
        if out.flags != PacketFlags::CRC_OK {
            continue;
        }
        if let Some(k) = (next..packets.len()).find(|&k| packets[k].data() == out.data()) {
            ok.push(k);
            next = k + 1;
        }
    }
    (ok, got.len())
}

#[test]
fn the_qpsk_flows_validate_without_warnings() {
    for (name, kinds) in [
        (
            "qpsk_transmitter",
            &["convert_out", "duc", "frame_encode", "psk_mod", "scale"][..],
        ),
        (
            "qpsk_receiver",
            &["agc", "convert", "ddc", "frame_decode", "psk_demod"][..],
        ),
    ] {
        let report = Registry::standard().validate_json(&flow(name), &BuildOptions::default());
        assert!(report.is_ok(), "{name}: {}", report.to_json());
        assert!(report.warnings.is_empty(), "{name}: {}", report.to_json());
        let mut got: Vec<_> = report.nodes.iter().map(|n| n.kind.as_str()).collect();
        got.sort_unstable();
        assert_eq!(got, kinds, "{name}");
        let rate = |kind: &str| {
            report
                .nodes
                .iter()
                .find(|n| n.kind == kind)
                .map(|n| n.rate_out)
                .unwrap()
        };
        if name == "qpsk_transmitter" {
            // 50 kbit/s, 8 samples per QPSK symbol: 200 kHz, then 1 MHz.
            assert_eq!(rate("psk_mod"), Rate::hz(200_000).unwrap());
            assert_eq!(rate("duc"), Rate::hz(1_000_000).unwrap());
        } else {
            // 200 kHz into the demodulator, 50 kbit/s of soft bits out.
            assert_eq!(rate("ddc"), Rate::hz(200_000).unwrap());
            assert_eq!(rate("psk_demod"), Rate::hz(50_000).unwrap());
        }
        eprintln!(
            "{name}: cost {:.1} per input sample, {} bytes",
            report.cost_per_input_sample, report.memory_bytes
        );
    }
}

#[test]
fn the_idle_transmitter_answers_pulls_of_one_sample() {
    // An idle source from the start. A QPSK `psk_mod` takes whole symbols of two bits, so one
    // idle bit of the encoder is not enough for one output sample; the pull adds idle bits
    // until the output moves (D620), and the stream does not depend on the pull sizes.
    let mut source = Sender {
        packets: Vec::new(),
        next: 0,
        open: false,
    };
    let mut whole = build(&flow("qpsk_transmitter"));
    let mut expected = vec![0u8; 2 * 4000];
    let (n, _) = whole.pull("rf", &mut expected, &mut source).unwrap();
    assert_eq!(n, expected.len());
    let mut single = build(&flow("qpsk_transmitter"));
    let mut got = Vec::new();
    let mut one = [0u8; 2];
    while got.len() < expected.len() {
        let (n, _) = single.pull("rf", &mut one, &mut source).unwrap();
        assert_eq!(n, 2);
        got.extend_from_slice(&one);
    }
    assert_eq!(got, expected);
    assert!(expected.iter().any(|&b| b != 0));
}

#[test]
fn qpsk_flows_carry_1000_packets_through_noise_and_a_carrier_offset() {
    let packets = random_packets();
    let tx = transmit(&packets);
    let power = tx.iter().map(Complex64::norm_sqr).sum::<f64>() / tx.len() as f64;
    let peak = tx
        .iter()
        .map(|v| v.re.abs().max(v.im.abs()))
        .fold(0.0, f64::max);
    // The level of the transmitter keeps the peaks inside the cs8 range.
    assert!(peak < 127.0 / 128.0, "peak {peak}");

    let mut channel = ChannelChain::new()
        .with(FrequencyOffset::new(OFFSET_HZ, FS).unwrap())
        .with(Awgn::from_ebn0(EBN0_DB, 2, SPS as f64, power, SEED).unwrap());
    let mut rx = Vec::new();
    channel.process(&tx, &mut rx);
    let rms = (rx.iter().map(Complex64::norm_sqr).sum::<f64>() / rx.len() as f64).sqrt();
    let scale = RX_RMS / rms * 32768.0;
    // A symmetric range, so that the rotations by 90 degrees are exact.
    let to_i16 = |x: f64| (x * scale).round().clamp(-32767.0, 32767.0) as i16;
    let iq: Vec<[i16; 2]> = rx.iter().map(|v| [to_i16(v.re), to_i16(v.im)]).collect();

    let results: Vec<(Vec<usize>, usize)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..4)
            .map(|q| {
                let (iq, packets) = (&iq, &packets);
                s.spawn(move || receive(&rotate(iq, q), packets))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let summary: Vec<(usize, usize)> = results.iter().map(|r| (r.0.len(), r.1)).collect();
    let missing: Vec<usize> = results
        .iter()
        .max_by_key(|r| r.0.len())
        .map(|r| (0..PACKETS).filter(|k| !r.0.contains(k)).collect())
        .unwrap();
    eprintln!(
        "qpsk_loop: {} samples, power {power:.4}, peak {peak:.4}; (correct, out) at 0, 90, \
         180, 270 degrees: {summary:?}; missing packets {missing:?}",
        tx.len()
    );
    // `polarity` both: the rotations by 180 degrees give the same packets.
    assert_eq!(results[0], results[2]);
    assert_eq!(results[1], results[3]);
    let (good, bad) = if results[0].0.len() >= results[1].0.len() {
        (&results[0], &results[1])
    } else {
        (&results[1], &results[0])
    };
    assert!(
        good.0.len() >= REQUIRED,
        "{} of {PACKETS} packets with CRC_OK",
        good.0.len()
    );
    assert!(bad.0.is_empty() && bad.1 == 0, "{summary:?}");
}
