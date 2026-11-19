//! M44.2 §5: the example flows `qam256_transmitter` and `qam256_receiver` validate without
//! warnings, with the rates of 256-QAM at 25 kBd (200 kbit/s), and test 7 of M44.2: the flows
//! `qam256_transmitter` -> channel -> `qam256_receiver`.
//!
//! As the 16-QAM loop of M44 (`qam_loop.rs`): the transmitter runs in pull mode, the packet
//! source is idle for the first [`LEAD_SYMBOLS`] symbols (the encoder sends the PN9 idle
//! signal), then hands over the [`PACKETS`] packets, three per call, and is idle again for
//! [`TAIL_SYMBOLS`] symbols. Both flows use the scrambler of the QAM frame composites
//! (`scrambler` `multiplicative`, `polynomial` `x23_x18`, the maintainer's decision on I44.2-6):
//! the whole bit stream, the PN9 idle signal, the preamble and the sync word included, looks
//! random on the air, on which the blind equalizer of 256-QAM converges (without it, it does not:
//! D1523). The `cs16` samples go through `Multipath::mild` (at 40 samples per symbol), a carrier
//! offset of 1 kHz (0.04 cycles per symbol at 25 kBd) and white Gaussian noise at [`EBN0_DB`]
//! (all from `sdrk_testkit::channel`), then are scaled and rounded to the `cs16` input of the
//! receiver, on the seeds 834, 835 and 836 (payloads: stream 71 of the seed; noise: the `Awgn` of
//! the seed). The receiver flow is equivariant to a rotation of its input by 90 degrees, and the
//! descrambler runs over whole symbols so that the quarter turns pass through it (D1530): every
//! seed runs at the four rotations, and each of the 12 receivers delivers at least [`REQUIRED`]
//! of the 1000 packets with `CRC_OK` and their payload unchanged. Test 7 is the acceptance test
//! of M44.3 (D1541); it passes since the `agc` of `qam256_receiver` holds the unit power that
//! `qam_demod` expects with time constants of 100 ms (D2413: 999, 1000 and 999 packets on the
//! three seeds at every rotation; 930, 933 and 954 with the `agc` of M44.2).
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
use sdrk_proto::frame::rotation_of;
use sdrk_testkit::Pcg32;
use sdrk_testkit::channel::{Awgn, ChannelChain, FrequencyOffset, Impairment, Multipath};

use common::{build, flow};

/// The seeds of the test (M44.2 test 7).
const SEEDS: [u64; 3] = [834, 835, 836];
/// Packets counted.
const PACKETS: usize = 1000;
/// Packets that must arrive with `CRC_OK` and the payload unchanged (M44.2 test 7).
const REQUIRED: usize = 990;
/// Sample rate of the transmitter output and of the receiver input in Hz.
const FS: f64 = 1_000_000.0;
/// Samples per symbol at `FS` (25 kBd).
const SPS: usize = 40;
/// Bits per 256-QAM symbol.
const BITS_PER_SYMBOL: usize = 8;
/// Carrier offset of the channel in Hz (0.04 cycles per symbol).
const OFFSET_HZ: f64 = 1000.0;
/// Eb/N0 of the channel in dB (M44.2 test 7, D1522).
const EBN0_DB: f64 = 26.0;
/// Idle symbols before the first packet: the frequency acquisition, the 8000 blind symbols of the
/// equalizer of 256-QAM, its decision mode and the first check at 12288 symbols after the
/// estimate, up to the lock of the carrier loop.
const LEAD_SYMBOLS: usize = 30_000;
/// Symbols pulled after the source handed over the last packet.
const TAIL_SYMBOLS: usize = 2000;
/// Bytes per pull of the transmitter (`cs16`: four bytes per sample).
const PULL_BYTES: usize = 16_000;
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

/// `PACKETS` random packets with 0 to 63 bytes (stream 71 of the seed).
fn random_packets(seed: u64) -> Vec<Packet> {
    let mut rng = Pcg32::new(seed, 71);
    (0..PACKETS)
        .map(|_| {
            let len = rng.below(64) as usize;
            let data: Vec<u8> = (0..len).map(|_| rng.next_u32() as u8).collect();
            Packet::new(0, PacketFlags::NONE, &data).unwrap()
        })
        .collect()
}

/// Runs the flow `qam256_transmitter` until every counted packet is sent and the tail is out; returns the
/// complex baseband (the `cs16` samples divided by 32768).
fn transmit(packets: &[Packet]) -> Vec<Complex64> {
    transmit_pull(&flow("qam256_transmitter"), packets, PULL_BYTES)
}

/// The transmitter flow `spec` with `pull` bytes per pull.
fn transmit_pull(spec: &str, packets: &[Packet], pull: usize) -> Vec<Complex64> {
    let mut tx = build(spec);
    let mut source = Sender {
        packets: packets.to_vec(),
        next: 0,
        open: false,
    };
    let mut out = Vec::new();
    let mut bytes = vec![0u8; pull];
    let mut tail_from = None;
    loop {
        source.open = out.len() >= LEAD_SYMBOLS * SPS;
        let (n, _) = tx.pull("rf", &mut bytes, &mut source).unwrap();
        assert_eq!(n, pull, "a packet input never underruns");
        out.extend(bytes.chunks_exact(4).map(|iq| {
            Complex64::new(
                f64::from(i16::from_le_bytes([iq[0], iq[1]])) / 32768.0,
                f64::from(i16::from_le_bytes([iq[2], iq[3]])) / 32768.0,
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

/// What one receiver delivered.
#[derive(Debug, PartialEq, Eq)]
struct Received {
    /// The positions (in `packets`) of the counted packets that came out with `CRC_OK` and
    /// unchanged, matched in order.
    ok: Vec<usize>,
    /// The quarter turns the flags of these packets report (one entry per value seen).
    rotations: Vec<u32>,
    /// Packets out (with `CRC_OK` or not).
    out: usize,
    /// The longest interval in symbols between the sync words of two successive packets with
    /// `CRC_OK` after the first (the measure of the default `frame_timeout`, D1575).
    longest: u64,
}

/// Runs the receiver flow on `bytes`.
fn receive(bytes: &[u8], packets: &[Packet]) -> Received {
    let mut p = build(&flow("qam256_receiver"));
    let mut got = Vec::new();
    for chunk in bytes.chunks(PUSH_SAMPLES * 4) {
        p.push("rf", InputData::Raw(chunk)).unwrap();
        read_all(&mut p, &mut got);
    }
    assert_eq!(p.dropped("packets").unwrap(), 0);
    let mut next = 0;
    let mut ok = Vec::new();
    let mut rotations = Vec::new();
    for out in &got {
        if !out.flags.contains(PacketFlags::CRC_OK) {
            continue;
        }
        if let Some(k) = (next..packets.len()).find(|&k| packets[k].data() == out.data()) {
            ok.push(k);
            next = k + 1;
            let r = rotation_of(out.flags);
            if !rotations.contains(&r) {
                rotations.push(r);
            }
        }
    }
    // The packet index counts the 200 kHz input samples of `qam_frame_decode`, 8 per symbol.
    let starts: Vec<u64> = got
        .iter()
        .filter(|p| p.flags.contains(PacketFlags::CRC_OK))
        .map(|p| p.index / 8)
        .collect();
    let longest = starts.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
    Received {
        ok,
        rotations,
        out: got.len(),
        longest,
    }
}

#[test]
fn the_qam256_flows_validate_without_warnings() {
    for (name, kinds) in [
        (
            "qam256_transmitter",
            &["convert_out", "duc", "qam_frame_encode", "scale"][..],
        ),
        (
            "qam256_receiver",
            &["agc", "convert", "ddc", "qam_frame_decode"][..],
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
        if name == "qam256_transmitter" {
            // 200 kbit/s, 8 samples per 256-QAM symbol of 8 bits: 200 kHz, then 1 MHz.
            assert_eq!(rate("qam_frame_encode"), Rate::hz(200_000).unwrap());
            assert_eq!(rate("duc"), Rate::hz(1_000_000).unwrap());
        } else {
            // 200 kHz into the demodulator, 200 kbit/s of soft bits out of it.
            assert_eq!(rate("ddc"), Rate::hz(200_000).unwrap());
        }
        eprintln!(
            "{name}: cost {:.1} per input sample, {} bytes",
            report.cost_per_input_sample, report.memory_bytes
        );
    }
}

/// The channel of the test on the seed `seed`: `Multipath::mild`, the carrier offset and the
/// noise, scaled and rounded to `cs16` pairs.
fn channel(tx: &[Complex64], seed: u64) -> Vec<[i16; 2]> {
    let mut multipath = Multipath::mild(SPS as f64).unwrap();
    let mut faded = Vec::new();
    multipath.process(tx, &mut faded);
    let faded_power = faded.iter().map(Complex64::norm_sqr).sum::<f64>() / faded.len() as f64;
    let mut channel = ChannelChain::new()
        .with(FrequencyOffset::new(OFFSET_HZ, FS).unwrap())
        .with(
            Awgn::from_ebn0(
                EBN0_DB,
                BITS_PER_SYMBOL as u32,
                SPS as f64,
                faded_power,
                seed,
            )
            .unwrap(),
        );
    let mut rx = Vec::new();
    channel.process(&faded, &mut rx);
    let rms = (rx.iter().map(Complex64::norm_sqr).sum::<f64>() / rx.len() as f64).sqrt();
    let scale = RX_RMS / rms * 32768.0;
    // A symmetric range, so that the rotations by 90 degrees are exact.
    let to_i16 = |x: f64| (x * scale).round().clamp(-32767.0, 32767.0) as i16;
    rx.iter().map(|v| [to_i16(v.re), to_i16(v.im)]).collect()
}

#[test]
fn qam256_flows_carry_1000_packets_through_multipath_noise_and_a_carrier_offset() {
    for seed in SEEDS {
        let packets = random_packets(seed);
        let tx = transmit(&packets);
        let power = tx.iter().map(Complex64::norm_sqr).sum::<f64>() / tx.len() as f64;
        let peak = tx
            .iter()
            .map(|v| v.re.abs().max(v.im.abs()))
            .fold(0.0, f64::max);
        // The level of the transmitter keeps the peaks inside the cs16 range.
        assert!(peak < 32767.0 / 32768.0, "peak {peak}");
        let iq = channel(&tx, seed);
        let results: Vec<Received> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..4)
                .map(|q| {
                    let (iq, packets) = (&iq, &packets);
                    s.spawn(move || receive(&rotate(iq, q), packets))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let summary: Vec<(usize, &[u32], usize, u64)> = results
            .iter()
            .map(|r| (r.ok.len(), &r.rotations[..], r.out, r.longest))
            .collect();
        eprintln!(
            "qam256_loop seed {seed}: {} samples, power {power:.4}, peak {peak:.4}; (correct, \
             rotations, out, longest interval) at 0, 90, 180, 270 degrees: {summary:?}",
            tx.len()
        );
        for (q, r) in results.iter().enumerate() {
            assert!(
                r.ok.len() >= REQUIRED,
                "seed {seed}: {} of {PACKETS} packets with CRC_OK at {} degrees",
                r.ok.len(),
                90 * q
            );
        }
    }
}
