//! M41 tests 2 and 3: the example flow `fm_rds_receiver` on the RDS reference signal of M41 §5.
//!
//! The input is the reference signal re-synthesized by `sdrk_testkit::rds::RdsReference` (the
//! Rust model of `tests/golden/sdrk_golden/cases/rds.py`, cross-checked against the committed
//! case `rds/reference_short`, D663) with complex white Gaussian noise from
//! `sdrk_testkit::channel::Awgn`, rounded to `cs16` at 480 kHz and pushed in pieces of 4096
//! samples. The SNR is per sample over the 480 kHz band, `0.25 / (2 sigma^2)` (D666).
//!
//! What is measured (D681 to D683):
//! - PS and RT: the `CRC_OK` packets of the output `rds` go through `sdrk_proto::rds::RdsState`
//!   in order; the time of "complete" is the RF time pushed when `ps_ready` (`rt_ready`) first
//!   holds, which includes the delay of the flow (conservative).
//! - The groups: a `CRC_OK` packet at the soft-bit index `i` is the carousel group
//!   `g = floor((i + 52) / 104)` and must equal it (no wrong group passes); the share of valid
//!   groups is the number of distinct groups received with `CRC_OK` among those sent after
//!   the first second (`g >= 12`: group 12 starts at 1.05 s) up to the last group that ends
//!   at least 20 bits before the end of the signal.
//! - The stereo audio: the SINAD of the left channel at 1 kHz and of the right channel at
//!   2 kHz (`MultitoneAccumulator`, both tones and the offset fitted in each channel) over the
//!   audio from 1 s to 3 s, with and without RDS in the multiplex, the same noise; the two
//!   must agree within 0.5 dB.
//!
//! Test 3 runs the flow with a 2 kHz carrier offset in the RF and `tune.frequency_hz` 2000,
//! with 100 ppm clock drift (`ClockDrift`, seed 572 for the noise) and at 15 dB SNR (seed 573).
//! The test `python_reference_file_through_the_flow` (ignored) takes the file of
//! `generate.py --emit rds_reference --emit-out PATH` named by `SDRK_RDS_REFERENCE`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::print_stderr,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::too_many_lines,
    clippy::float_cmp
)]

mod common;

use std::collections::BTreeSet;

use num_complex::Complex64;
use sdrk::{C32, Rate};
use sdrk_pipeline::{BuildOptions, InputData, Packet, PacketFlags, Pipeline, Registry};
use sdrk_proto::rds::{Group, RdsState};
use sdrk_testkit::channel::{Awgn, ChannelChain, ClockDrift, FrequencyOffset, Impairment};
use sdrk_testkit::meters::MultitoneAccumulator;
use sdrk_testkit::rds::{RDS_LEVEL, RF_RATE_HZ, RdsReference, TABLE_LEN, carousel_group, to_cs16};
use serde_json::Value;

use common::{build, flow};

/// Seed of E2E-RDS (test 2) and of the carrier-offset run of test 3.
const SEED_E2E: u64 = 571;
/// Seed of the clock-drift run of test 3.
const SEED_DRIFT: u64 = 572;
/// Seed of the 15 dB run of test 3.
const SEED_15DB: u64 = 573;
/// RF samples per push (the `max_chunk` of the flow).
const PUSH: usize = 4096;
/// RF sample rate.
const FS: f64 = RF_RATE_HZ as f64;
/// The RDS bit rate.
const BIT_RATE: f64 = 1187.5;
/// Bits per group.
const GROUP_BITS: u64 = 104;
/// Audio sample rate of the flow.
const AUDIO_HZ: f64 = 48_000.0;
/// The audio window of the SINAD comparison in seconds of audio output.
const AUDIO_WINDOW: (f64, f64) = (1.0, 3.0);
/// Largest difference of the SINAD with and without RDS (D682).
const MAX_SINAD_DIFF_DB: f64 = 0.5;

/// One run of the flow.
#[derive(Clone, Copy, Debug)]
struct Run {
    seconds: f64,
    snr_db: f64,
    seed: u64,
    offset_hz: f64,
    ppm: f64,
    rds_level: f64,
    /// Measure the audio over [`AUDIO_WINDOW`].
    audio: bool,
}

impl Run {
    fn new(seconds: f64, snr_db: f64, seed: u64) -> Self {
        Self {
            seconds,
            snr_db,
            seed,
            offset_hz: 0.0,
            ppm: 0.0,
            rds_level: RDS_LEVEL,
            audio: false,
        }
    }
}

/// What a run delivered.
#[derive(Debug)]
struct Outcome {
    /// RF time pushed when PS (RT) was complete, in seconds.
    ps_at: Option<f64>,
    rt_at: Option<f64>,
    ps: String,
    rt: String,
    packets: Vec<Packet>,
    /// SINAD of left (1 kHz) and right (2 kHz) over the audio window.
    sinad_db: Option<(f64, f64)>,
    /// RF samples pushed.
    samples: u64,
}

/// The flow with `tune.frequency_hz` set to `tune_hz`.
fn receiver(tune_hz: f64) -> Pipeline {
    let mut doc: Value = serde_json::from_str(&flow("fm_rds_receiver")).unwrap();
    doc["nodes"]["tune"]["params"]["frequency_hz"] = serde_json::json!(tune_hz);
    build(&doc.to_string())
}

/// Complex noise of `snr_db` per sample against the carrier power 0.25 (D666).
fn noise(snr_db: f64, seed: u64) -> Awgn {
    let sigma = (0.25 / (2.0 * 10f64.powf(snr_db / 10.0))).sqrt();
    Awgn::new(sigma, seed).unwrap()
}

fn cs16_bytes(x: &[Complex64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 * x.len());
    for v in x {
        let q = to_cs16(*v);
        out.extend_from_slice(&q.re.0.to_le_bytes());
        out.extend_from_slice(&q.im.0.to_le_bytes());
    }
    out
}

/// The receiving side of a run: pushes `cs16` bytes, reads both outputs after every push,
/// follows PS and RT and measures the audio.
struct Receiver {
    p: Pipeline,
    state: RdsState,
    out: Outcome,
    audio_buf: Vec<C32>,
    packet_buf: Vec<Packet>,
    audio_n: usize,
    audio: bool,
    left: MultitoneAccumulator,
    right: MultitoneAccumulator,
}

impl Receiver {
    fn new(tune_hz: f64, audio: bool) -> Self {
        Self {
            p: receiver(tune_hz),
            state: RdsState::new(),
            out: Outcome {
                ps_at: None,
                rt_at: None,
                ps: String::new(),
                rt: String::new(),
                packets: Vec::new(),
                sinad_db: None,
                samples: 0,
            },
            audio_buf: vec![C32::new(0.0, 0.0); 4096],
            packet_buf: vec![Packet::default(); 64],
            audio_n: 0,
            audio,
            left: MultitoneAccumulator::new(AUDIO_HZ, &[1000.0, 2000.0]),
            right: MultitoneAccumulator::new(AUDIO_HZ, &[1000.0, 2000.0]),
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        for piece in bytes.chunks(4 * PUSH) {
            self.p.push("rf", InputData::Raw(piece)).unwrap();
            self.out.samples += (piece.len() / 4) as u64;
            self.drain();
        }
    }

    fn drain(&mut self) {
        let now = self.out.samples as f64 / FS;
        loop {
            let n = self
                .p
                .read_stream::<C32>("audio", &mut self.audio_buf)
                .unwrap();
            if self.audio {
                let (a, b) = (
                    (AUDIO_WINDOW.0 * AUDIO_HZ) as usize,
                    (AUDIO_WINDOW.1 * AUDIO_HZ) as usize,
                );
                for (i, v) in self.audio_buf[..n].iter().enumerate() {
                    if (a..b).contains(&(self.audio_n + i)) {
                        self.left.push(f64::from(v.re));
                        self.right.push(f64::from(v.im));
                    }
                }
            }
            self.audio_n += n;
            let k = self.p.read_packets("rds", &mut self.packet_buf).unwrap();
            for pk in &self.packet_buf[..k] {
                self.out.packets.push(*pk);
                if pk.flags.contains(PacketFlags::CRC_OK) {
                    self.state.update(&Group::from_packet(pk).unwrap());
                }
            }
            if self.out.ps_at.is_none() && self.state.ps_ready() {
                self.out.ps_at = Some(now);
                self.out.ps = self.state.ps().to_string();
            }
            if self.out.rt_at.is_none() && self.state.rt_ready() {
                self.out.rt_at = Some(now);
                self.out.rt = self.state.rt().to_string();
            }
            if n == 0 && k == 0 {
                return;
            }
        }
    }

    fn finish(mut self) -> Outcome {
        if self.audio {
            let b = (AUDIO_WINDOW.1 * AUDIO_HZ) as usize;
            assert!(
                self.audio_n >= b,
                "audio window not reached: {}",
                self.audio_n
            );
            self.out.sinad_db = Some((self.left.sinad_db(0), self.right.sinad_db(1)));
        }
        assert_eq!(self.p.dropped("rds").unwrap(), 0);
        self.out
    }
}

/// Runs the reference signal through the channel of `run` and the flow tuned to `tune_hz`.
fn run(r: Run, tune_hz: f64) -> Outcome {
    let mut sig = RdsReference::new(vec![0.0; TABLE_LEN], r.rds_level).unwrap();
    let mut channel = ChannelChain::new();
    if r.ppm != 0.0 {
        channel.push(ClockDrift::new(r.ppm).unwrap());
    }
    if r.offset_hz != 0.0 {
        channel.push(FrequencyOffset::new(r.offset_hz, FS).unwrap());
    }
    channel.push(noise(r.snr_db, r.seed));
    let mut rx = Receiver::new(tune_hz, r.audio);
    let total = (r.seconds * FS) as u64;
    let mut clean = vec![Complex64::new(0.0, 0.0); PUSH];
    let mut noisy = Vec::with_capacity(PUSH + 64);
    let mut at = 0u64;
    while at < total {
        let n = PUSH.min((total - at) as usize);
        sig.fill(at, &mut clean[..n]).unwrap();
        at += n as u64;
        noisy.clear();
        channel.process(&clean[..n], &mut noisy);
        rx.push(&cs16_bytes(&noisy));
    }
    rx.finish()
}

/// The group accounting of the module documentation: panics on a `CRC_OK` packet that is not
/// the transmitted group; returns (valid groups, groups sent) after the first second.
fn score(o: &Outcome) -> (usize, usize) {
    let mut ok = BTreeSet::new();
    for pk in &o.packets {
        if !pk.flags.contains(PacketFlags::CRC_OK) {
            continue;
        }
        let g = (pk.index + GROUP_BITS / 2) / GROUP_BITS;
        let got = Group::from_packet(pk).unwrap();
        assert_eq!(
            got.blocks,
            carousel_group(g),
            "a CRC_OK packet at index {} is not group {g}",
            pk.index
        );
        assert!(ok.insert(g), "group {g} delivered twice");
    }
    let first = (BIT_RATE / GROUP_BITS as f64).ceil() as u64; // 12
    let bits = o.samples as f64 / FS * BIT_RATE;
    let last = ((bits - 20.0) / GROUP_BITS as f64).floor() as u64 - 1;
    let sent = (last + 1 - first) as usize;
    let valid = ok.range(first..=last).count();
    (valid, sent)
}

fn report(name: &str, o: &Outcome, valid: usize, sent: usize) {
    eprintln!(
        "{name}: PS {:?} at {:?} s, RT {:?} at {:?} s, {valid} of {sent} groups after 1 s CRC_OK \
         ({:.2} %), {} packets, the first CRC_OK group {:?}",
        o.ps,
        o.ps_at,
        o.rt,
        o.rt_at,
        100.0 * valid as f64 / sent as f64,
        o.packets.len(),
        o.packets
            .iter()
            .find(|p| p.flags.contains(PacketFlags::CRC_OK))
            .map(|p| (p.index + GROUP_BITS / 2) / GROUP_BITS)
    );
}

fn assert_text(o: &Outcome, ps_by: f64, rt_by: f64) {
    assert_eq!(o.ps, "SDRK TST");
    assert_eq!(o.rt, "Hello from sdrk!");
    assert!(o.ps_at.unwrap() <= ps_by, "PS at {:?} s", o.ps_at);
    assert!(o.rt_at.unwrap() <= rt_by, "RT at {:?} s", o.rt_at);
}

#[test]
fn the_flow_validates_with_the_parameters_of_the_milestone() {
    let registry = Registry::standard();
    let report = registry.validate_json(&flow("fm_rds_receiver"), &BuildOptions::default());
    assert!(report.is_ok(), "{}", report.to_json());
    assert!(report.warnings.is_empty(), "{}", report.to_json());
    let doc: Value = serde_json::from_str(&flow("fm_rds_receiver")).unwrap();
    let n = &doc["nodes"];
    assert_eq!(n["tune"]["kind"], "ddc");
    assert_eq!(n["tune"]["params"]["decimation"], 2);
    assert_eq!(n["tune"]["params"]["cutoff_hz"], 100_000);
    assert_eq!(n["stereo"]["kind"], "fm_stereo_demod");
    assert_eq!(n["stereo"]["params"]["audio_decimation"], 5);
    assert_eq!(n["quad"]["kind"], "quad_demod");
    assert_eq!(n["quad"]["params"]["deviation_hz"], 75_000);
    assert_eq!(n["subcarrier"]["kind"], "rds_demod");
    assert_eq!(n["groups"]["kind"], "rds_decode");
    assert_eq!(doc["inputs"]["rf"]["format"], "cs16");
    assert_eq!(doc["outputs"]["rds"]["type"], "packets");
    let p = build(&flow("fm_rds_receiver"));
    assert_eq!(p.input_rate("rf").unwrap(), Rate::hz(480_000).unwrap());
    assert_eq!(p.output_rate("audio").unwrap(), Rate::hz(48_000).unwrap());
    assert_eq!(
        p.output_rate("rds").unwrap(),
        Rate::new(2375, 2 * 104).unwrap()
    );
}

/// E2E-RDS (test 2): 6.5 s at 30 dB (seed 571), and the same signal without RDS for the audio.
#[test]
fn e2e_rds_reference() {
    let mut r = Run::new(6.5, 30.0, SEED_E2E);
    r.audio = true;
    let with = run(r, 0.0);
    let (valid, sent) = score(&with);
    report("e2e_rds_reference", &with, valid, sent);
    assert_text(&with, 2.0, 6.0);
    assert!(100 * valid >= 99 * sent, "{valid} of {sent}");

    // The audio without RDS: the same noise, only the window of the comparison.
    let mut plain = r;
    plain.rds_level = 0.0;
    plain.seconds = AUDIO_WINDOW.1 + 0.2;
    let without = run(plain, 0.0);
    assert!(without.packets.is_empty(), "packets without RDS");
    let (l1, r1) = with.sinad_db.unwrap();
    let (l0, r0) = without.sinad_db.unwrap();
    eprintln!(
        "audio SINAD with RDS: L {l1:.2} dB, R {r1:.2} dB; without: L {l0:.2} dB, R {r0:.2} dB"
    );
    assert!((l1 - l0).abs() <= MAX_SINAD_DIFF_DB, "left {l1} vs {l0}");
    assert!((r1 - r0).abs() <= MAX_SINAD_DIFF_DB, "right {r1} vs {r0}");
}

/// Test 3: a carrier 2 kHz off, fixed by tuning the `ddc` to it (30 dB, seed 571).
#[test]
fn carrier_offset_fixed_by_tuning() {
    let mut r = Run::new(3.5, 30.0, SEED_E2E);
    r.offset_hz = 2000.0;
    let o = run(r, 2000.0);
    let (valid, sent) = score(&o);
    report("offset 2 kHz", &o, valid, sent);
    assert_text(&o, 2.0, 3.5);
    assert!(100 * valid >= 99 * sent, "{valid} of {sent}");
}

/// Test 3: 100 ppm clock drift (30 dB, seed 572).
#[test]
fn clock_drift_100_ppm() {
    let mut r = Run::new(3.5, 30.0, SEED_DRIFT);
    r.ppm = 100.0;
    let o = run(r, 0.0);
    let (valid, sent) = score(&o);
    report("drift 100 ppm", &o, valid, sent);
    assert_text(&o, 2.0, 3.5);
    assert!(100 * valid >= 99 * sent, "{valid} of {sent}");
}

/// Test 3: 15 dB SNR (seed 573): at least 90 % valid groups.
#[test]
fn snr_15_db() {
    let o = run(Run::new(3.5, 15.0, SEED_15DB), 0.0);
    let (valid, sent) = score(&o);
    report("15 dB", &o, valid, sent);
    assert!(10 * valid >= 9 * sent, "{valid} of {sent}");
}

/// The file of `generate.py --emit rds_reference --emit-out PATH` (the Python modulator with
/// its own noise, seed 570) through the flow: PS, RT and the groups as in E2E-RDS.
#[test]
#[ignore = "needs SDRK_RDS_REFERENCE=<path of generate.py --emit rds_reference>"]
fn python_reference_file_through_the_flow() {
    let path = std::env::var("SDRK_RDS_REFERENCE").expect("SDRK_RDS_REFERENCE is not set");
    let raw = std::fs::read(&path).unwrap();
    let meta: Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{path}.json")).unwrap()).unwrap();
    assert_eq!(meta["rate_hz"].as_u64().unwrap(), RF_RATE_HZ);
    let mut rx = Receiver::new(0.0, false);
    rx.push(&raw[..raw.len() / 4 * 4]);
    let o = rx.finish();
    let (valid, sent) = score(&o);
    report(&path, &o, valid, sent);
    assert_text(&o, 2.0, 6.0);
    assert!(100 * valid >= 99 * sent, "{valid} of {sent}");
}
