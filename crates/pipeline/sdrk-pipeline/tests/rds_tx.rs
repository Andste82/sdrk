//! M42 §5 and test 3: the flow `fm_stereo_rds_transmitter` and the loop test `e2e_rds_loop`.
//!
//! The flow validates without warnings, and its pull output equals the push output with control
//! packets pushed between the pull calls (the pull source is asked for the audio only: the
//! output `rf` has its origin in `audio`, the encoder takes its clock from it).
//!
//! The loop (D768): the flow without `duc`, the FM signal (`c32`, amplitude 1) at 192 kHz, plus
//! complex white Gaussian noise (`sdrk_testkit::channel::Awgn`; the SNR is per sample over the
//! 192 kHz band, `1 / (2 sigma^2)`), into the receiver chain of `fm_rds_receiver` with its `ddc`
//! replaced by a `resample` to 240 kHz: `quad_demod` -> `rds_demod` -> `rds_decode`, and
//! `fm_stereo_demod` for the audio. The station name and the radiotext are pushed as packets
//! before the first audio sample, one raw group at 3 s. What is measured:
//! - PS and RT: the `CRC_OK` packets go through `RdsState` in order; the time of "complete" is
//!   the audio time pushed into the transmitter when `ps_ready` (`rt_ready`) first holds, which
//!   includes the delay of both pipelines (conservative, as D683).
//! - The groups: a `CRC_OK` packet at the soft-bit index `i` is group `g = floor((i + 52) /
//!   104)` (as D681); the raw group is found by its blocks, and the expected sequence is a
//!   `Carousel` with the same texts and the raw group queued before group `g_raw`. Every
//!   `CRC_OK` packet must equal its expected group; the share of valid groups is the number of
//!   distinct groups received among those sent after the first second (`g >= 12`) up to the
//!   last group that ends at least 20 bits before the end of the signal.
//! - The stereo audio: left 1 kHz 0.5, right 3 kHz 0.3 as in E2E-STEREO-TX (M36); the
//!   separation over the last second, at least 30 dB in both channels.
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
    clippy::too_many_lines,
    clippy::float_cmp
)]

mod common;

use std::collections::BTreeMap;

use num_complex::Complex64;
use sdrk::C32;
use sdrk_pipeline::{
    BuildOptions, InputData, Packet, PacketFlags, Pipeline, PullError, PullSource, Registry,
};
use sdrk_proto::rds::{Carousel, Group, RdsState};
use sdrk_testkit::Pcg32;
use sdrk_testkit::channel::{Awgn, Impairment};
use sdrk_testkit::meters::MultitoneAccumulator;
use serde_json::{Value, json};

use common::{build, flow};

/// Seed of the 40 dB loop.
const SEED_40DB: u64 = 583;
/// Seed of the 25 dB loop.
const SEED_25DB: u64 = 584;
/// Audio rate of the flow.
const AUDIO_HZ: f64 = 48_000.0;
/// Rate of the FM signal (the flow without `duc`).
const FS: f64 = 192_000.0;
/// Audio samples per push (the `max_chunk` of the flow).
const PUSH: usize = 4800;
/// The RDS bit rate.
const BIT_RATE: f64 = 1187.5;
/// Bits per group.
const GROUP_BITS: u64 = 104;
/// The settings of the flow.
const PI: u16 = 0xD123;
const PTY: u8 = 10;
const PS: &[u8; 8] = b"SDRK TST";
const RT: &[u8] = b"Hello from sdrk!";
/// The raw group sent at [`RAW_AT_S`] (group type 8A, a TMC-like payload).
const RAW: [u16; 4] = [PI, 0x8000 | ((PTY as u16) << 5) | 0x05, 0x1234, 0xABCD];
const RAW_AT_S: f64 = 3.0;

fn packet(data: &[u8]) -> Packet {
    Packet::new(0, PacketFlags::NONE, data).unwrap()
}

fn ps_packet(ps: [u8; 8]) -> Packet {
    let mut d = vec![0x01];
    d.extend_from_slice(&ps);
    packet(&d)
}

fn rt_packet(rt: &[u8]) -> Packet {
    let mut d = vec![0x02];
    d.extend_from_slice(rt);
    packet(&d)
}

fn raw_packet(g: [u16; 4]) -> Packet {
    let mut d = vec![0x03];
    for w in g {
        d.extend_from_slice(&w.to_be_bytes());
    }
    packet(&d)
}

/// `n` samples of stereo audio at 48 kHz: left 1 kHz 0.5, right 3 kHz 0.3, plus independent
/// Gaussian noise of 0.001 in both channels (seeded), the audio of E2E-STEREO-TX.
fn loop_audio(n: usize, seed: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(seed, 0);
    (0..n)
        .map(|i| {
            let t = i as f64 / AUDIO_HZ;
            let l = 0.5 * (std::f64::consts::TAU * 1000.0 * t).sin() + 0.001 * rng.normal();
            let r = 0.3 * (std::f64::consts::TAU * 3000.0 * t).sin() + 0.001 * rng.normal();
            C32::new(l as f32, r as f32)
        })
        .collect()
}

// ---- The flow ----

#[test]
fn the_flow_validates_with_the_parameters_of_the_milestone() {
    let text = flow("fm_stereo_rds_transmitter");
    let report = Registry::standard().validate_json(&text, &BuildOptions::default());
    assert!(report.is_ok(), "{}", report.to_json());
    assert!(report.warnings.is_empty(), "{}", report.to_json());
    let mut kinds: Vec<_> = report.nodes.iter().map(|n| n.kind.as_str()).collect();
    kinds.sort_unstable();
    assert_eq!(
        kinds,
        [
            "convert_out",
            "duc",
            "fm_mod",
            "fm_stereo_rds_mod",
            "preemph_limit",
            "rds_encode",
            "resample"
        ]
    );
    let v: Value = serde_json::from_str(&text).unwrap();
    let n = &v["nodes"];
    assert_eq!(v["settings"]["max_chunk"], 4800);
    assert_eq!(v["inputs"]["audio"]["type"], "c32");
    assert_eq!(v["inputs"]["audio"]["rate_hz"], 48_000);
    assert_eq!(v["inputs"]["text"]["type"], "packets");
    assert_eq!(n["up"]["params"]["rate_out_hz"], 192_000);
    assert_eq!(n["rds"]["kind"], "rds_encode");
    assert_eq!(n["rds"]["params"]["pi"], 0xD123);
    assert_eq!(n["mpx"]["params"]["main"], 0.87);
    assert_eq!(n["mpx"]["params"]["rds"], 0.04);
    assert_eq!(n["mod"]["params"]["deviation_hz"], 75_000);
    assert_eq!(n["duc"]["params"]["interpolation"], 10);
    assert_eq!(n["duc"]["params"]["frequency_hz"], 600_000);
    let edges = v["edges"].as_array().unwrap();
    for e in [
        json!({ "from": "pre", "to": "mpx" }),
        json!({ "from": "pre", "to": "rds" }),
        json!({ "from": "text", "to": "rds", "port": "b" }),
        json!({ "from": "rds", "to": "mpx", "port": "b" }),
    ] {
        assert!(edges.contains(&e), "{e}");
    }
    let p = build(&text);
    assert_eq!(p.output_rate("rf").unwrap().as_f64(), 1_920_000.0);
    // A rate_hz on the packet input is not used: the validation warns.
    let mut with_rate = v.clone();
    with_rate["inputs"]["text"]["rate_hz"] = json!(192_000);
    let report =
        Registry::standard().validate_json(&with_rate.to_string(), &BuildOptions::default());
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.warnings.len(), 1, "{}", report.to_json());
    assert_eq!(report.warnings[0].path, "inputs.text.rate_hz");
}

/// Hands out stereo audio in pieces and counts what it handed out.
struct StereoPullSource {
    data: Vec<C32>,
    pos: usize,
}

impl PullSource for StereoPullSource {
    fn provide(&mut self, input: &str, max_samples: usize) -> Result<InputData<'_>, PullError> {
        assert_eq!(
            input, "audio",
            "the pull source is asked for the audio only"
        );
        assert!(max_samples > 0 && max_samples <= PUSH, "{max_samples}");
        let n = max_samples.min(self.data.len() - self.pos);
        let from = self.pos;
        self.pos += n;
        Ok(InputData::C32(&self.data[from..from + n]))
    }
}

/// Push mode: `schedule` lists (audio sample, packets pushed before it); the audio goes in
/// pieces of `piece` cut at the schedule positions; all bytes of `rf` are read.
fn transmitter_push(data: &[C32], piece: usize, schedule: &[(usize, Vec<Packet>)]) -> Vec<u8> {
    let mut p = build(&flow("fm_stereo_rds_transmitter"));
    let mut out = Vec::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut pos = 0;
    while pos < data.len() {
        for (_, packets) in schedule.iter().filter(|e| e.0 == pos) {
            p.push_packets("text", packets).unwrap();
        }
        let next = schedule
            .iter()
            .map(|e| e.0)
            .filter(|&at| at > pos)
            .min()
            .unwrap_or(usize::MAX);
        let end = (pos + piece).min(data.len()).min(next);
        p.push("audio", InputData::C32(&data[pos..end])).unwrap();
        pos = end;
        loop {
            let n = p.read_raw("rf", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buf[..n]);
        }
    }
    out
}

/// Pull equals push: the texts before the first pull, a new name, a raw group and an unknown
/// packet between two pull calls; the push run gets the packets at the audio position the pull
/// source had reached then. The `rf` bytes are identical.
#[test]
fn pull_equals_push_with_packets_between_the_pulls() {
    let data = loop_audio(30_000, 583);
    let first = vec![ps_packet(*PS), rt_packet(RT)];
    let later = vec![
        ps_packet(*b"NEW NAME"),
        raw_packet(RAW),
        packet(b"\x7fignored"),
    ];

    let mut p = build(&flow("fm_stereo_rds_transmitter"));
    let mut source = StereoPullSource { data, pos: 0 };
    p.push_packets("text", &first).unwrap();
    let mut pulled = vec![0u8; 1_000_000];
    let (n, _) = p.pull("rf", &mut pulled[..400_000], &mut source).unwrap();
    assert_eq!(n, 400_000);
    let at = source.pos;
    p.push_packets("text", &later).unwrap();
    let (n, _) = p.pull("rf", &mut pulled[400_000..], &mut source).unwrap();
    assert_eq!(n, 600_000);
    let rds = p.stats().iter().find(|s| s.name == "rds").unwrap();
    assert_eq!(rds.readings, vec![("ignored_packets", 1.0)]);

    let schedule = vec![(0, first), (at, later)];
    for piece in [PUSH, 1000, 777] {
        let pushed = transmitter_push(&source.data, piece, &schedule);
        assert!(pushed.len() > 1_000_000, "{}", pushed.len());
        assert_eq!(pushed[..1_000_000], pulled[..], "pieces of {piece}");
    }
    // The packets change the signal: without the later packets the bytes differ after `at`.
    let plain = transmitter_push(&source.data, PUSH, &schedule[..1]);
    assert_eq!(plain[..400_000], pulled[..400_000]);
    assert_ne!(plain[..1_000_000], pulled[..]);
}

// ---- The loop ----

/// The transmitter: the flow without `duc`, its FM signal at 192 kHz on the output `iq`.
fn loop_transmitter() -> Pipeline {
    let mut v: Value = serde_json::from_str(&flow("fm_stereo_rds_transmitter")).unwrap();
    assert!(v["nodes"].as_object_mut().unwrap().remove("duc").is_some());
    let edges: Vec<Value> = v["edges"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["to"] != "duc" && e["from"] != "duc")
        .cloned()
        .chain([json!({ "from": "mod", "to": "iq" })])
        .collect();
    assert_eq!(edges.len(), 8);
    v["edges"] = json!(edges);
    v["outputs"] = json!({ "iq": { "type": "stream" } });
    let p = build(&v.to_string());
    assert_eq!(p.output_rate("iq").unwrap().as_f64(), FS);
    p
}

/// The receiver: `fm_rds_receiver` with its `ddc` replaced by a `resample` from 192 kHz to
/// 240 kHz (the rate of the receiver after its `ddc`).
fn loop_receiver() -> Pipeline {
    let mut v: Value = serde_json::from_str(&flow("fm_rds_receiver")).unwrap();
    let nodes = v["nodes"].as_object_mut().unwrap();
    assert!(nodes.remove("tune").is_some());
    nodes.insert(
        "tune".into(),
        json!({ "kind": "resample", "params": { "rate_out_hz": 240_000 } }),
    );
    v["inputs"] = json!({ "iq": { "type": "c32", "rate_hz": 192_000 } });
    v["settings"]["max_chunk"] = json!(4 * PUSH);
    for e in v["edges"].as_array_mut().unwrap() {
        if e["from"] == "rf" {
            e["from"] = json!("iq");
        }
    }
    let p = build(&v.to_string());
    assert_eq!(p.output_rate("audio").unwrap().as_f64(), AUDIO_HZ);
    p
}

/// What a loop run delivered.
#[derive(Debug, Default)]
struct Outcome {
    /// Audio time pushed into the transmitter when PS (RT) was complete, in seconds.
    ps_at: Option<f64>,
    rt_at: Option<f64>,
    ps: String,
    rt: String,
    packets: Vec<Packet>,
    /// Amplitudes of 1 kHz and 3 kHz in left and right over the last second.
    tones: ([f64; 2], [f64; 2]),
    /// Samples of the FM signal sent.
    samples: u64,
}

/// One loop run of `seconds` at `snr_db` (seed `seed` for the audio noise and the channel).
fn run_loop(seconds: f64, snr_db: f64, seed: u64) -> Outcome {
    let mut tx = loop_transmitter();
    let mut rx = loop_receiver();
    let sigma = (1.0 / (2.0 * 10f64.powf(snr_db / 10.0))).sqrt();
    let mut channel = Awgn::new(sigma, seed).unwrap();
    let total = (seconds * AUDIO_HZ) as usize;
    let audio = loop_audio(total, seed);
    let raw_at = (RAW_AT_S * AUDIO_HZ) as usize;
    let tone_from = total - AUDIO_HZ as usize;

    let mut out = Outcome::default();
    let mut state = RdsState::new();
    let mut left = MultitoneAccumulator::new(AUDIO_HZ, &[1000.0, 3000.0]);
    let mut right = left;
    let mut iq = vec![C32::default(); 1 << 16];
    let mut clean = Vec::new();
    let mut noisy = Vec::new();
    let mut rx_in = Vec::new();
    let mut stereo = vec![C32::default(); 1 << 14];
    let mut packets = vec![Packet::default(); 64];
    let mut audio_n = 0usize;

    tx.push_packets("text", &[ps_packet(*PS), rt_packet(RT)])
        .unwrap();
    for (k, piece) in audio.chunks(PUSH).enumerate() {
        let at = k * PUSH;
        if at == raw_at {
            tx.push_packets("text", &[raw_packet(RAW)]).unwrap();
        }
        tx.push("audio", InputData::C32(piece)).unwrap();
        let now = (at + piece.len()) as f64 / AUDIO_HZ;
        loop {
            let n = tx.read_stream::<C32>("iq", &mut iq).unwrap();
            if n == 0 {
                break;
            }
            out.samples += n as u64;
            clean.clear();
            clean.extend(
                iq[..n]
                    .iter()
                    .map(|v| Complex64::new(f64::from(v.re), f64::from(v.im))),
            );
            noisy.clear();
            channel.process(&clean, &mut noisy);
            rx_in.clear();
            rx_in.extend(noisy.iter().map(|v| C32::new(v.re as f32, v.im as f32)));
            for part in rx_in.chunks(4 * PUSH) {
                rx.push("iq", InputData::C32(part)).unwrap();
                loop {
                    let a = rx.read_stream::<C32>("audio", &mut stereo).unwrap();
                    for (i, v) in stereo[..a].iter().enumerate() {
                        if audio_n + i >= tone_from {
                            left.push(f64::from(v.re));
                            right.push(f64::from(v.im));
                        }
                    }
                    audio_n += a;
                    let m = rx.read_packets("rds", &mut packets).unwrap();
                    for pk in &packets[..m] {
                        out.packets.push(*pk);
                        if pk.flags.contains(PacketFlags::CRC_OK) {
                            state.update(&Group::from_packet(pk).unwrap());
                        }
                    }
                    if out.ps_at.is_none() && state.ps_ready() {
                        out.ps_at = Some(now);
                        out.ps = state.ps().to_string();
                    }
                    if out.rt_at.is_none() && state.rt_ready() {
                        out.rt_at = Some(now);
                        out.rt = state.rt().to_string();
                    }
                    if a == 0 && m == 0 {
                        break;
                    }
                }
            }
        }
    }
    assert!(audio_n > tone_from + AUDIO_HZ as usize / 2, "{audio_n}");
    assert_eq!(rx.dropped("rds").unwrap(), 0);
    let (al, ar) = (left.amplitudes(), right.amplitudes());
    out.tones = ([al[0], al[1]], [ar[0], ar[1]]);
    out
}

/// The expected groups and the share of valid groups (see the module documentation): panics
/// on a `CRC_OK` packet that is not the expected group. Returns (valid, sent, group of the
/// raw group).
fn score(o: &Outcome) -> (usize, usize, u64) {
    let mut got = BTreeMap::new();
    for pk in o
        .packets
        .iter()
        .filter(|p| p.flags.contains(PacketFlags::CRC_OK))
    {
        let g = (pk.index + GROUP_BITS / 2) / GROUP_BITS;
        let blocks = Group::from_packet(pk).unwrap().blocks;
        assert!(
            got.insert(g, (pk.index, blocks)).is_none(),
            "group {g} delivered twice"
        );
    }
    let raw: Vec<u64> = got
        .iter()
        .filter(|(_, (_, b))| *b == RAW)
        .map(|(g, _)| *g)
        .collect();
    assert_eq!(raw.len(), 1, "the raw group arrives once: {raw:?}");
    let g_raw = raw[0];
    let mut carousel = Carousel::new(PI, PTY, false).unwrap();
    carousel.set_ps(PS);
    carousel.set_rt(RT).unwrap();
    let last_g = *got.keys().last().unwrap();
    let expected: Vec<[u16; 4]> = (0..=last_g.max(g_raw))
        .map(|g| {
            if g == g_raw {
                carousel.push_raw(Group::new(RAW)).unwrap();
            }
            carousel.next_group()
        })
        .collect();
    for (g, (index, blocks)) in &got {
        assert_eq!(
            *blocks, expected[*g as usize],
            "a CRC_OK packet at index {index} is not group {g}"
        );
    }
    let first = (BIT_RATE / GROUP_BITS as f64).ceil() as u64; // 12
    let bits = o.samples as f64 / FS * BIT_RATE;
    let last = ((bits - 20.0) / GROUP_BITS as f64).floor() as u64 - 1;
    let sent = (last + 1 - first) as usize;
    let valid = got.range(first..=last).count();
    (valid, sent, g_raw)
}

fn db(x: f64) -> f64 {
    20.0 * x.log10()
}

/// Checks a loop run: PS and RT, the groups, the raw group, and returns (valid, sent).
fn check(name: &str, o: &Outcome, seconds: f64) -> (usize, usize) {
    let (valid, sent, g_raw) = score(o);
    let indices: Vec<i64> = o
        .packets
        .iter()
        .filter(|p| p.flags.contains(PacketFlags::CRC_OK))
        .take(3)
        .map(|p| p.index as i64 - ((p.index + GROUP_BITS / 2) / GROUP_BITS * GROUP_BITS) as i64)
        .collect();
    eprintln!(
        "{name}: PS {:?} at {:?} s, RT {:?} at {:?} s, {valid} of {sent} groups after 1 s \
         CRC_OK ({:.2} %), {} packets, the raw group is group {g_raw}, index offsets {indices:?}",
        o.ps,
        o.ps_at,
        o.rt,
        o.rt_at,
        100.0 * valid as f64 / sent as f64,
        o.packets.len(),
    );
    assert_eq!(o.ps, "SDRK TST");
    assert_eq!(o.rt, "Hello from sdrk!");
    assert!(o.ps_at.unwrap() <= 2.0, "PS at {:?} s", o.ps_at);
    assert!(o.rt_at.unwrap() <= 6.0, "RT at {:?} s", o.rt_at);
    // The raw group follows the group that runs when it is pushed (3 s is in group 34).
    let pushed_in = (RAW_AT_S * BIT_RATE / GROUP_BITS as f64).floor() as u64;
    assert!(
        (pushed_in..=pushed_in + 2).contains(&g_raw),
        "raw group sent as group {g_raw}"
    );
    assert!(o.samples as f64 >= (seconds - 0.1) * FS, "{}", o.samples);
    (valid, sent)
}

/// Test 3 at 40 dB (seed 583): PS within 2 s, RT within 6 s, every group after the first
/// second valid, the raw group unchanged, and the stereo separation of E2E-STEREO-TX (at least
/// 30 dB in both channels).
#[test]
fn e2e_rds_loop() {
    let seconds = 6.5;
    let o = run_loop(seconds, 40.0, SEED_40DB);
    let (valid, sent) = check("e2e_rds_loop 40 dB", &o, seconds);
    assert_eq!(valid, sent, "{valid} of {sent}");

    let (l, r) = o.tones;
    let (level_l, level_r) = (db(l[0] / 0.5), db(r[1] / 0.3));
    let (sep_l, sep_r) = (db(l[0] / l[1]), db(r[1] / r[0]));
    eprintln!(
        "e2e_rds_loop 40 dB: level L {level_l:.3} dB, R {level_r:.3} dB; separation L \
         {sep_l:.1} dB, R {sep_r:.1} dB"
    );
    assert!(sep_l >= 30.0, "3 kHz in L only {sep_l} dB below 1 kHz");
    assert!(sep_r >= 30.0, "1 kHz in R only {sep_r} dB below 3 kHz");
}

/// Test 3 at 25 dB (seed 584): PS within 2 s, RT within 6 s, at least 99 % valid groups after
/// the first second, the raw group unchanged.
#[test]
fn e2e_rds_loop_25_db() {
    let seconds = 6.5;
    let o = run_loop(seconds, 25.0, SEED_25DB);
    let (valid, sent) = check("e2e_rds_loop 25 dB", &o, seconds);
    assert!(100 * valid >= 99 * sent, "{valid} of {sent}");
}
