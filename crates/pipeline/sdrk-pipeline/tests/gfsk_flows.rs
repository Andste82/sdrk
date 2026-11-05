//! The flows `gfsk_receiver` and `gfsk_transmitter` (M39 §6) and test 5 (the flow loop).
//!
//! - Both flows validate without warnings and have the parameters of §6; the transmitter in
//!   pull mode gives the same bytes as in push mode.
//! - **Test 5:** `gfsk_transmitter` in pull mode (a packet source that is idle in a quarter of
//!   the calls, so the encoder sends its idle preamble) -> testkit channel (30 ppm clock drift,
//!   a 3 kHz frequency offset, white Gaussian noise at Eb/N0 14 dB; seed 536) ->
//!   `gfsk_receiver`: of 1000 packets with 1 to 64 random bytes at least 990 come out with
//!   `CRC_OK` and the payload sent. The transmitted `cs8` samples are scaled by 0.2 before the
//!   channel, so that signal and noise fit the `cs16` input of the receiver without clipping
//!   (D584).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stderr,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::too_many_lines
)]

mod common;

use num_complex::Complex64;
use sdrk_device::formats;
use sdrk_pipeline::{
    BuildOptions, InputData, Packet, PacketFlags, Pipeline, PullError, PullSource, Registry,
};
use sdrk_testkit::Pcg32;
use sdrk_testkit::channel::{Awgn, ChannelChain, ClockDrift, FrequencyOffset, Impairment};
use sdrk_testkit::encode::{decode_reference, encode};
use serde_json::Value;

use common::{build, flow};

const FLOWS: [&str; 2] = ["gfsk_receiver", "gfsk_transmitter"];

/// Seed of test 5 (payloads: stream 1; idle calls: stream 2; noise: the `Awgn` of the seed).
const SEED: u64 = 536;

/// `count` packets with 1 to 64 random bytes (index and flags 0).
fn random_packets(count: usize, stream: u64) -> Vec<Packet> {
    let mut rng = Pcg32::new(SEED, stream);
    (0..count)
        .map(|_| {
            let len = 1 + rng.below(64) as usize;
            let data: Vec<u8> = (0..len).map(|_| rng.next_u32() as u8).collect();
            Packet::new(0, PacketFlags::NONE, &data).unwrap()
        })
        .collect()
}

/// A packet source for pull mode: one packet per call, and with `idle` a quarter of the calls
/// (chosen by the generator) and every call after the last packet give nothing, so the
/// encoder sends its idle signal.
struct Sender {
    packets: Vec<Packet>,
    next: usize,
    rng: Pcg32,
    idle: bool,
}

impl PullSource for Sender {
    fn provide(&mut self, input: &str, max: usize) -> Result<InputData<'_>, PullError> {
        assert_eq!(input, "msg");
        if (self.idle && self.rng.below(4) == 0) || self.next == self.packets.len() || max == 0 {
            return Ok(InputData::Packets(&[]));
        }
        let from = self.next;
        self.next += 1;
        Ok(InputData::Packets(&self.packets[from..self.next]))
    }
}

#[test]
fn both_flows_validate_without_warnings() {
    let registry = Registry::standard();
    for name in FLOWS {
        let report = registry.validate_json(&flow(name), &BuildOptions::default());
        assert!(report.is_ok(), "{name}: {}", report.to_json());
        assert!(report.warnings.is_empty(), "{name}: {}", report.to_json());
        assert!(report.memory_bytes > 0, "{name}");
        build(&flow(name));
    }
}

#[test]
fn the_flows_have_the_parameters_of_the_spec() {
    let rx: Value = serde_json::from_str(&flow("gfsk_receiver")).unwrap();
    assert_eq!(rx["inputs"]["rf"]["format"], "cs16");
    assert_eq!(rx["inputs"]["rf"]["rate_hz"], 1_000_000);
    let n = &rx["nodes"];
    assert_eq!(n["tune"]["kind"], "ddc");
    assert_eq!(n["tune"]["params"]["decimation"], 5);
    assert_eq!(n["tune"]["params"]["cutoff_hz"], 60_000);
    assert_eq!(n["afc"]["kind"], "afc");
    assert_eq!(n["afc"]["params"]["max_offset_hz"], 20_000);
    assert_eq!(n["demod"]["kind"], "fsk_demod");
    assert_eq!(n["demod"]["params"]["symbol_rate_hz"], 50_000);
    assert_eq!(n["demod"]["params"]["deviation_hz"], 25_000);
    assert_eq!(n["dec"]["kind"], "frame_decode");
    assert_eq!(n["dec"]["params"]["whitening"], "pn9");
    assert_eq!(rx["outputs"]["packets"]["type"], "packets");

    let tx: Value = serde_json::from_str(&flow("gfsk_transmitter")).unwrap();
    assert_eq!(tx["inputs"]["msg"]["type"], "packets");
    assert_eq!(tx["inputs"]["msg"]["rate_hz"], 50_000);
    let n = &tx["nodes"];
    assert_eq!(n["enc"]["kind"], "frame_encode");
    assert_eq!(n["enc"]["params"]["whitening"], "pn9");
    assert_eq!(n["mod"]["kind"], "fsk_mod");
    assert_eq!(n["mod"]["params"]["samples_per_symbol"], 4);
    assert_eq!(n["mod"]["params"]["deviation_hz"], 25_000);
    assert_eq!(n["mod"]["params"]["bt"], 0.5);
    assert_eq!(n["duc"]["kind"], "duc");
    assert_eq!(n["duc"]["params"]["interpolation"], 5);
    assert_eq!(tx["outputs"]["rf"]["format"], "cs8");

    // The rates of the chain: 50 kbit/s -> 200 kHz -> 1 MS/s, and 1 MS/s -> 200 kHz -> 50 kbit/s
    // in the receiver.
    let rates = |name: &str| -> Vec<(String, u64, u64)> {
        let report = Registry::standard().validate_json(&flow(name), &BuildOptions::default());
        report
            .nodes
            .iter()
            .filter(|n| !n.auto)
            .map(|n| {
                (
                    n.name.clone(),
                    n.rate_in.as_f64().round() as u64,
                    n.rate_out.as_f64().round() as u64,
                )
            })
            .collect()
    };
    let tx_rates = rates("gfsk_transmitter");
    assert!(
        tx_rates.contains(&("mod".into(), 50_000, 200_000)),
        "{tx_rates:?}"
    );
    assert!(
        tx_rates.contains(&("duc".into(), 200_000, 1_000_000)),
        "{tx_rates:?}"
    );
    let rx_rates = rates("gfsk_receiver");
    assert!(
        rx_rates.contains(&("tune".into(), 1_000_000, 200_000)),
        "{rx_rates:?}"
    );
    assert!(
        rx_rates.contains(&("afc".into(), 200_000, 200_000)),
        "{rx_rates:?}"
    );
    assert!(
        rx_rates.contains(&("demod".into(), 200_000, 50_000)),
        "{rx_rates:?}"
    );
}

/// The bytes of `gfsk_transmitter` in push mode: all packets pushed at once.
fn push_run(packets: &[Packet]) -> Vec<u8> {
    let mut p = build(&flow("gfsk_transmitter"));
    p.push_packets("msg", packets).unwrap();
    let mut out = Vec::new();
    let mut buf = vec![0u8; 8192];
    loop {
        let n = p.read_raw("rf", &mut buf).unwrap();
        if n == 0 {
            break;
        }
        out.extend_from_slice(&buf[..n]);
    }
    out
}

#[test]
fn the_transmitter_gives_the_same_bytes_in_pull_and_push_mode() {
    let packets = random_packets(100, 3);
    let pushed = push_run(&packets);
    // A source that always has a packet: the pulled stream is the pushed one.
    let mut p = build(&flow("gfsk_transmitter"));
    let mut source = Sender {
        packets,
        next: 0,
        rng: Pcg32::new(SEED, 4),
        idle: false,
    };
    let want = 400_000;
    assert!(pushed.len() > want, "{} bytes pushed", pushed.len());
    let mut sizes = Pcg32::new(SEED, 5);
    let mut pulled = Vec::new();
    let mut buf = vec![0u8; 16_384];
    while pulled.len() < want {
        let n = (2 * (1 + sizes.below(8_000) as usize)).min(want - pulled.len());
        let (got, _) = p.pull("rf", &mut buf[..n], &mut source).unwrap();
        assert_eq!(got, n);
        pulled.extend_from_slice(&buf[..got]);
    }
    assert!(pulled == pushed[..want], "pull differs from push");
    assert!(pulled.iter().any(|&b| b != 0));
}

// ---- test 5 ------------------------------------------------------------------------------

/// Packets of test 5.
const PACKETS: usize = 1_000;
/// Packets that must arrive with `CRC_OK` and the payload sent.
const MIN_OK: usize = 990;
/// Eb/N0 of the channel in dB.
const EBN0_DB: f64 = 14.0;
/// Carrier offset of the channel in Hz.
const OFFSET_HZ: f64 = 3_000.0;
/// Sample-clock drift of the channel in ppm.
const DRIFT_PPM: f64 = 30.0;
/// Amplitude of the transmitted signal in the channel.
const AMPLITUDE: f64 = 0.2;
/// Samples per bit at 1 MS/s (50 kbit/s).
const SAMPLES_PER_BIT: f64 = 20.0;
/// Bytes pulled per call (2048 `cs8` samples).
const PULL_BYTES: usize = 4_096;

fn read_packets(p: &mut Pipeline, into: &mut Vec<Packet>) {
    let mut buf = [Packet::default(); 64];
    loop {
        let n = p.read_packets("packets", &mut buf).unwrap();
        if n == 0 {
            return;
        }
        into.extend_from_slice(&buf[..n]);
    }
}

#[test]
fn flow_loop_1000_packets_through_the_channel() {
    let packets = random_packets(PACKETS, 1);
    let mut tx = build(&flow("gfsk_transmitter"));
    let mut rx = build(&flow("gfsk_receiver"));
    let mut source = Sender {
        packets: packets.clone(),
        next: 0,
        rng: Pcg32::new(SEED, 2),
        idle: true,
    };
    let mut channel = ChannelChain::new()
        .with(ClockDrift::new(DRIFT_PPM).unwrap())
        .with(FrequencyOffset::new(OFFSET_HZ, 1e6).unwrap())
        .with(Awgn::from_ebn0(EBN0_DB, 1, SAMPLES_PER_BIT, AMPLITUDE * AMPLITUDE, SEED).unwrap());
    let mut raw = vec![0u8; PULL_BYTES];
    let mut samples = vec![Complex64::new(0.0, 0.0); PULL_BYTES / 2];
    let mut noisy = Vec::new();
    let mut cs16 = vec![0u8; 2 * PULL_BYTES];
    let mut got = Vec::new();
    let mut total = 0usize;
    // After the last packet, 0.1 s of idle signal carries it through the receiver.
    let mut tail = 0usize;
    while tail < 100_000 {
        let (n, _) = tx.pull("rf", &mut raw, &mut source).unwrap();
        assert_eq!(n, PULL_BYTES);
        let m = decode_reference(&formats::CS8, &raw, &mut samples).unwrap();
        for s in &mut samples[..m] {
            *s *= AMPLITUDE;
        }
        noisy.clear();
        channel.process(&samples[..m], &mut noisy);
        let k = encode(&formats::CS16, &noisy, &mut cs16).unwrap();
        rx.push("rf", InputData::Raw(&cs16[..k])).unwrap();
        read_packets(&mut rx, &mut got);
        total += noisy.len();
        if source.next == PACKETS {
            tail += noisy.len();
        }
    }
    assert!(got.iter().all(|p| p.flags == PacketFlags::CRC_OK));
    // In order: every decoded payload is matched with the next sent packet that has it.
    let mut next = 0;
    let mut ok = 0;
    for g in &got {
        if let Some(k) = packets[next..].iter().position(|s| s.data() == g.data()) {
            ok += 1;
            next += k + 1;
        }
    }
    let afc = rx.stats().iter().find(|s| s.name == "afc").unwrap();
    eprintln!(
        "flow loop (Eb/N0 {EBN0_DB} dB, offset {OFFSET_HZ} Hz, drift {DRIFT_PPM} ppm): \
         {PACKETS} packets sent, {} decoded with CRC_OK, {ok} with the payload sent (PER \
         {:.4}); {total} samples at 1 MS/s; AFC readings at the end {:?}",
        got.len(),
        1.0 - ok as f64 / PACKETS as f64,
        afc.readings
    );
    assert!(ok >= MIN_OK, "only {ok} of {PACKETS} packets");
}
