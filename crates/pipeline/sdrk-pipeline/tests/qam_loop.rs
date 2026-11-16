//! M44 test 7 and M44.1 test 9: the example flows `qam16_transmitter` -> channel ->
//! `qam16_receiver`, since M44.1 on the composites `qam_frame_encode` and `qam_frame_decode`
//! (the transmitter's bytes and the receiver's soft bits keep the baseline of the pairs they
//! replace).
//!
//! The transmitter runs in pull mode: the packet source is idle for the first
//! [`LEAD_SYMBOLS`] symbols (the encoder sends its idle signal, the PN9 sequence, while the
//! receiver's loops, its frequency acquisition and its blind equalizer train), then hands over
//! the [`PACKETS`] packets, three per call, and is idle again for [`TAIL_SYMBOLS`] symbols
//! after the last call (the frames still queued and the delay of the receiver). Whenever the
//! encoder runs ahead of the source it fills the gap with PN9 idle bits (I44-5); random-looking
//! 16-QAM symbols keep the timing loop and the equalizer trained, which the constant symbol of
//! the idle preamble 0x55 did not (D1218: filler packets before I44-5; D1408, D1410). The
//! `cs16` samples are read as complex samples, passed through `Multipath::mild` (at 40 samples
//! per symbol), shifted by 1 kHz (0.04 cycles per symbol at 25 kBd) and given white Gaussian
//! noise at Eb/N0 = 16 dB (all from `sdrk_testkit::channel`, seed 606), then scaled and rounded
//! to the `cs16` input of the receiver.
//!
//! Coherent 16-QAM leaves a carrier phase ambiguity of 90 degrees, which `frame_decode` with
//! `rotations` four resolves at the sync word. The receiver flow is exactly equivariant to a
//! rotation of its input by 90 degrees (the filters are real, the AGC, the Gardner detector,
//! the equalizer and the decision-directed carrier loop are symmetric under the swaps and
//! negations), so the test runs it on the channel output rotated by 0, 90, 180 and 270
//! degrees: every rotation delivers at least [`REQUIRED`] of the 1000 packets with `CRC_OK` and
//! their payload unchanged, so `rotations` four resolves every lock position. The quarter turns
//! reported in the packet flags are printed (they step with the rotation of the input; the
//! test does not assert it, because a decision exactly on a boundary takes the upper level,
//! D1202, which breaks the symmetry in rare cases).
//!
//! The baseline of M44.1 (spec §5, test 5; `sdrk_testkit::baseline`, table `qam_loop` of
//! `tests/baseline/m44_1`): the `cs16` bytes of the transmitter and the soft bits of the
//! `qam_demod` node of the receiver flow (the flow up to `demod`, its output read as a stream) at
//! the four rotations keep their bits, per SIMD level.
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
use sdrk_testkit::baseline::{self, Baseline, Digest};
use sdrk_testkit::channel::{Awgn, ChannelChain, FrequencyOffset, Impairment, Multipath};

use common::{build, flow};

/// Seed of the test (payloads: stream 71; noise: the `Awgn` of the seed).
const SEED: u64 = 606;
/// Packets counted.
const PACKETS: usize = 1000;
/// Packets that must arrive with `CRC_OK` and the payload unchanged: M44 test 7 asks for 990;
/// the test guards 995, since the frequency acquisition of `qam_demod` lets the flow run the
/// narrow default `carrier_bw` (I44-4); with the PN9 idle signal measured 1000 at every
/// rotation on the seeds 606 and 770 to 774 (D1410), with the loops of D1622 (`carrier_bw`
/// 0.0025, `mu_dd` 0.005, the narrow tracker of the equalizer) 1000, and 999 on seed 770.
const REQUIRED: usize = 995;
/// Sample rate of the transmitter output and of the receiver input in Hz.
const FS: f64 = 1_000_000.0;
/// Samples per symbol at `FS` (25 kBd).
const SPS: usize = 40;
/// Bits per 16-QAM symbol.
const BITS_PER_SYMBOL: usize = 4;
/// Carrier offset of the channel in Hz (0.04 cycles per symbol; the frequency acquisition of
/// `qam_demod` finds it, and the loops run at the default `carrier_bw` 0.0025).
const OFFSET_HZ: f64 = 1000.0;
/// Eb/N0 of the channel in dB.
const EBN0_DB: f64 = 16.0;
/// Idle symbols before the first packet: the frequency acquisition, the 1000 CMA symbols of
/// the equalizer and the pull-in of the carrier loop (as the 20000 idle symbols of
/// `qpsk_loop`).
const LEAD_SYMBOLS: usize = 20_000;
/// Symbols pulled after the source handed over the last packet (the frames still queued in
/// the encoder, then idle symbols that flush the filters and the decoder).
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

/// `PACKETS` random packets with 0 to 63 bytes.
fn random_packets() -> Vec<Packet> {
    let mut rng = Pcg32::new(SEED, 71);
    (0..PACKETS)
        .map(|_| {
            let len = rng.below(64) as usize;
            let data: Vec<u8> = (0..len).map(|_| rng.next_u32() as u8).collect();
            Packet::new(0, PacketFlags::NONE, &data).unwrap()
        })
        .collect()
}

/// Runs the transmitter until every counted packet is sent and the tail is out; returns the
/// complex baseband (the `cs16` samples divided by 32768).
fn transmit(packets: &[Packet]) -> Vec<Complex64> {
    transmit_with(&flow("qam16_transmitter"), packets)
}

/// [`transmit`] with the transmitter flow `spec`.
fn transmit_with(spec: &str, packets: &[Packet]) -> Vec<Complex64> {
    let mut tx = build(spec);
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
    let mut p = build(&flow("qam16_receiver"));
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
fn the_qam16_flows_validate_without_warnings() {
    for (name, kinds) in [
        (
            "qam16_transmitter",
            &["convert_out", "duc", "qam_frame_encode", "scale"][..],
        ),
        (
            "qam16_receiver",
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
        if name == "qam16_transmitter" {
            // 100 kbit/s, 8 samples per 16-QAM symbol of 4 bits: 200 kHz, then 1 MHz.
            assert_eq!(rate("qam_frame_encode"), Rate::hz(200_000).unwrap());
            assert_eq!(rate("duc"), Rate::hz(1_000_000).unwrap());
        } else {
            // 200 kHz into the demodulator.
            assert_eq!(rate("ddc"), Rate::hz(200_000).unwrap());
        }
        eprintln!(
            "{name}: cost {:.1} per input sample, {} bytes",
            report.cost_per_input_sample, report.memory_bytes
        );
    }
}

/// The channel of the test: `Multipath::mild`, the carrier offset and the noise, scaled and
/// rounded to `cs16` pairs.
fn channel(tx: &[Complex64]) -> Vec<[i16; 2]> {
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
                SEED,
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
fn qam16_flows_carry_1000_packets_through_multipath_noise_and_a_carrier_offset() {
    let packets = random_packets();
    let tx = transmit(&packets);
    let power = tx.iter().map(Complex64::norm_sqr).sum::<f64>() / tx.len() as f64;
    let peak = tx
        .iter()
        .map(|v| v.re.abs().max(v.im.abs()))
        .fold(0.0, f64::max);
    // The level of the transmitter keeps the peaks inside the cs16 range.
    assert!(peak < 32767.0 / 32768.0, "peak {peak}");
    let iq = channel(&tx);

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
    let missing: Vec<Vec<usize>> = results
        .iter()
        .map(|r| (0..PACKETS).filter(|k| !r.ok.contains(k)).collect())
        .collect();
    eprintln!(
        "qam_loop: {} samples, power {power:.4}, peak {peak:.4}; (correct, rotations, out) at \
         0, 90, 180, 270 degrees (and the longest interval in symbols between packets with CRC_OK): \
         {summary:?}; missing packets {missing:?}",
        tx.len()
    );
    for (q, r) in results.iter().enumerate() {
        assert!(
            r.ok.len() >= REQUIRED,
            "{} of {PACKETS} packets with CRC_OK at {} degrees",
            r.ok.len(),
            90 * q
        );
    }
}

/// The transmitter flow before M44.1: the pair `frame_encode` -> `qam_mod` with the parameters of
/// the `qam_frame_encode` node.
fn pair_transmitter() -> String {
    let mut spec: serde_json::Value = serde_json::from_str(&flow("qam16_transmitter")).unwrap();
    let nodes = spec["nodes"].as_object_mut().unwrap();
    let tx = nodes.remove("tx").unwrap();
    let p = &tx["params"];
    nodes.insert(
        "enc".to_string(),
        serde_json::json!({ "kind": "frame_encode", "params": {
            "idle": p["idle"], "preamble_bytes": p["preamble_bytes"] } }),
    );
    nodes.insert(
        "mod".to_string(),
        serde_json::json!({ "kind": "qam_mod", "params": {
            "order": p["order"], "rolloff": p["rolloff"],
            "samples_per_symbol": p["samples_per_symbol"], "span": p["span"] } }),
    );
    spec["edges"] = serde_json::json!([
        { "from": "msg", "to": "enc" },
        { "from": "enc", "to": "mod" },
        { "from": "mod", "to": "level" },
        { "from": "level", "to": "duc" },
        { "from": "duc", "to": "rf" }
    ]);
    spec.to_string()
}

/// The receiver flow with the `qam_demod` of its `qam_frame_decode` node (the parameters of the
/// demodulator), whose soft bits go to the stream output `soft`.
fn soft_receiver() -> String {
    let mut spec: serde_json::Value = serde_json::from_str(&flow("qam16_receiver")).unwrap();
    let nodes = spec["nodes"].as_object_mut().unwrap();
    let mut demod = nodes.remove("rx").unwrap();
    demod["kind"] = serde_json::json!("qam_demod");
    let params = demod["params"].as_object_mut().unwrap();
    for name in ["frame_timeout", "max_sync_errors", "emit_bad", "whitening"] {
        params.remove(name);
    }
    nodes.insert("demod".to_string(), demod);
    spec["edges"] = serde_json::json!([
        { "from": "rf", "to": "tune" },
        { "from": "tune", "to": "agc" },
        { "from": "agc", "to": "demod" },
        { "from": "demod", "to": "soft" }
    ]);
    spec["outputs"] = serde_json::json!({ "soft": { "type": "stream", "capacity": 65536 } });
    spec.to_string()
}

/// The digest of the soft bits of [`soft_receiver`] on `bytes`, pushed as in [`receive`].
fn soft_digest(bytes: &[u8]) -> Digest {
    let mut p = build(&soft_receiver());
    let mut digest = Digest::new();
    let mut buf = vec![0.0f32; 65536];
    for chunk in bytes.chunks(PUSH_SAMPLES * 4) {
        p.push("rf", InputData::Raw(chunk)).unwrap();
        loop {
            let n = p.read_stream("soft", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            digest.f32s(&buf[..n]);
        }
    }
    assert_eq!(p.dropped("soft").unwrap(), 0);
    digest
}

#[test]
fn the_qam16_flows_keep_the_baseline_of_m44_1() {
    let table = Baseline::load(
        "m44_1",
        "qam_loop",
        baseline::level(sdrk_backend_simd::level_name()),
    )
    .unwrap();
    let packets = random_packets();
    // the `cs16` bytes of the transmitter of the pair (the samples of `transmit` are exactly the
    // `cs16` values over 32768): in pull mode the idle symbols between the frames follow the room
    // the pipeline gives the encoder, which for `qam_frame_encode` counts samples, not bits, so
    // the bytes of the composite flow are checked by the receiver below and by the equality of
    // the blocks (`qam_frame_blocks`, D1578)
    let pair = transmit_with(&pair_transmitter(), &packets);
    let mut digest = Digest::new();
    for v in &pair {
        for x in [v.re, v.im] {
            digest.bytes(&((x * 32768.0) as i16).to_le_bytes());
        }
    }
    table.check("tx_cs16", &digest).unwrap();
    let iq = channel(&pair);
    let digests: Vec<Digest> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..4)
            .map(|q| {
                let iq = &iq;
                s.spawn(move || soft_digest(&rotate(iq, q)))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (q, d) in digests.iter().enumerate() {
        table.check(&format!("soft_turn{q}"), d).unwrap();
    }
}
