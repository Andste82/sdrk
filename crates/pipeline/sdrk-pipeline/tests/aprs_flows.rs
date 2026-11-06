//! The flows `aprs_receiver` and `aprs_transmitter` (M39.1 §3) and tests 2 and 3.
//!
//! - Both flows validate without warnings, build and have the parameters of §3; the
//!   transmitter in pull mode gives the same audio as in push mode.
//! - **Test 2 (APRS loop):** `aprs_transmitter` in pull mode (a packet source that is idle in
//!   a quarter of the calls, so the encoder sends flags) -> white Gaussian noise on the audio at
//!   20 dB SNR (seed 540) -> `aprs_receiver`: of 100 AX.25 UI frames at least 99 come out with
//!   `CRC_OK` and the bytes sent; the same with a sound-card clock offset of 200 ppm (testkit
//!   `ClockDrift`, seed 541). The SNR is the power of the audio over the noise power in the
//!   whole 48 kHz band (D714).
//! - **Test 3 (APRS reference signal):** the synthetic Bell 202 signal of the golden case
//!   `flows/aprs_bell202_20_frames` (seed 542, made by the Python generator independently of
//!   `fsk_mod`) gives all 20 frames.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stderr,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::too_many_lines
)]

mod common;

use num_complex::Complex64;
use sdrk_pipeline::{
    BuildOptions, InputData, Packet, PacketFlags, Pipeline, PullError, PullSource, Registry,
};
use sdrk_testkit::Pcg32;
use sdrk_testkit::channel::{Awgn, ChannelChain, ClockDrift, Impairment};
use sdrk_testkit::golden::{self, Array};
use serde_json::Value;

use common::{build, flow};

const FLOWS: [&str; 2] = ["aprs_receiver", "aprs_transmitter"];

/// Seed of the frames and the noise of test 2 without clock offset.
const SEED: u64 = 540;
/// Seed of test 2 with the clock offset.
const SEED_DRIFT: u64 = 541;

/// The 7-byte AX.25 address field of `call` (up to 6 characters) and `ssid`: the characters
/// shifted left by one, padded with spaces, then the SSID byte with both reserved bits set,
/// the C or H bit `ch` and the extension bit `last`.
fn address(call: &[u8], ssid: u8, ch: bool, last: bool) -> [u8; 7] {
    let mut a = [b' ' << 1; 7];
    for (d, c) in a.iter_mut().zip(call) {
        *d = c << 1;
    }
    a[6] = 0x60 | (ssid << 1) | (u8::from(ch) << 7) | u8::from(last);
    a
}

/// A random callsign of 4 to 6 characters (two letters, a digit, letters).
fn callsign(rng: &mut Pcg32) -> Vec<u8> {
    let letter = |rng: &mut Pcg32| b'A' + rng.below(26) as u8;
    let len = 4 + rng.below(3) as usize;
    let mut call = vec![letter(rng), letter(rng), b'0' + rng.below(10) as u8];
    while call.len() < len {
        call.push(letter(rng));
    }
    call
}

/// `count` AX.25 UI frames as APRS sends them: destination `APRS`, a random source callsign
/// and SSID, the path `WIDE1-1` (half of them also `WIDE2-1`), control 0x03, PID 0xF0 and 10 to
/// 80 random printable characters of information (without FCS).
fn aprs_frames(count: usize, seed: u64, stream: u64) -> Vec<Packet> {
    let mut rng = Pcg32::new(seed, stream);
    (0..count)
        .map(|_| {
            let two = rng.below(2) == 1;
            let mut f = Vec::new();
            f.extend_from_slice(&address(b"APRS", 0, true, false));
            let src = callsign(&mut rng);
            f.extend_from_slice(&address(&src, rng.below(16) as u8, false, false));
            f.extend_from_slice(&address(b"WIDE1", 1, false, !two));
            if two {
                f.extend_from_slice(&address(b"WIDE2", 1, false, true));
            }
            f.extend_from_slice(&[0x03, 0xF0]);
            let info = 10 + rng.below(71) as usize;
            f.extend((0..info).map(|_| b' ' + rng.below(95) as u8));
            Packet::new(0, PacketFlags::NONE, &f).unwrap()
        })
        .collect()
}

/// A packet source for pull mode: one packet per call, and with `idle` a quarter of the calls
/// (chosen by the generator) and every call after the last packet give nothing, so the
/// encoder sends flags.
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

/// The (name, input rate, output rate) of every node of a flow, rounded to hertz.
fn rates(name: &str) -> Vec<(String, u64, u64)> {
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
}

#[test]
fn the_flows_have_the_parameters_of_the_spec() {
    let rx: Value = serde_json::from_str(&flow("aprs_receiver")).unwrap();
    assert_eq!(rx["inputs"]["audio"]["type"], "f32");
    assert_eq!(rx["inputs"]["audio"]["rate_hz"], 48_000);
    let n = &rx["nodes"];
    assert_eq!(n["cplx"]["kind"], "to_complex");
    assert_eq!(n["shift"]["kind"], "rotate");
    assert_eq!(n["shift"]["params"]["frequency_hz"], -1700);
    assert_eq!(n["lpf"]["kind"], "decimate");
    assert_eq!(n["lpf"]["params"]["factor"], 4);
    assert_eq!(n["lpf"]["params"]["cutoff_hz"], 1800);
    assert_eq!(n["demod"]["kind"], "fsk_demod");
    assert_eq!(n["demod"]["params"]["symbol_rate_hz"], 1200);
    assert_eq!(n["demod"]["params"]["deviation_hz"], 500);
    assert_eq!(n["dec"]["kind"], "hdlc_decode");
    assert_eq!(rx["outputs"]["packets"]["type"], "packets");

    let tx: Value = serde_json::from_str(&flow("aprs_transmitter")).unwrap();
    assert_eq!(tx["inputs"]["msg"]["type"], "packets");
    assert_eq!(tx["inputs"]["msg"]["rate_hz"], 1200);
    let n = &tx["nodes"];
    assert_eq!(n["enc"]["kind"], "hdlc_encode");
    assert_eq!(n["mod"]["kind"], "fsk_mod");
    assert_eq!(n["mod"]["params"]["samples_per_symbol"], 10);
    assert_eq!(n["mod"]["params"]["deviation_hz"], 500);
    assert_eq!(n["shift"]["kind"], "rotate");
    assert_eq!(n["shift"]["params"]["frequency_hz"], 1700);
    assert_eq!(n["re"]["kind"], "real_part");
    assert_eq!(n["resamp"]["kind"], "resample");
    assert_eq!(n["resamp"]["params"]["rate_out_hz"], 48_000);
    assert_eq!(tx["outputs"]["audio"]["type"], "stream");

    // The rates of the chains: 1200 bit/s -> 12 kHz -> 48 kHz, and 48 kHz -> 12 kHz -> 1200 Bd.
    let tx_rates = rates("aprs_transmitter");
    for want in [
        ("mod", 1200, 12_000),
        ("shift", 12_000, 12_000),
        ("re", 12_000, 12_000),
        ("resamp", 12_000, 48_000),
    ] {
        assert!(
            tx_rates.contains(&(want.0.into(), want.1, want.2)),
            "{tx_rates:?}"
        );
    }
    let rx_rates = rates("aprs_receiver");
    for want in [
        ("cplx", 48_000, 48_000),
        ("shift", 48_000, 48_000),
        ("lpf", 48_000, 12_000),
        ("demod", 12_000, 1200),
    ] {
        assert!(
            rx_rates.contains(&(want.0.into(), want.1, want.2)),
            "{rx_rates:?}"
        );
    }
}

/// The audio of `aprs_transmitter` in push mode: all packets pushed at once.
fn push_run(packets: &[Packet]) -> Vec<f32> {
    let mut p = build(&flow("aprs_transmitter"));
    p.push_packets("msg", packets).unwrap();
    let mut out = Vec::new();
    let mut buf = vec![0f32; 8192];
    loop {
        let n = p.read_stream("audio", &mut buf).unwrap();
        if n == 0 {
            break;
        }
        out.extend_from_slice(&buf[..n]);
    }
    out
}

#[test]
fn the_transmitter_gives_the_same_audio_in_pull_and_push_mode() {
    let packets = aprs_frames(10, SEED, 3);
    let pushed = push_run(&packets);
    // A source that always has a packet: the pulled stream is the pushed one.
    let mut p = build(&flow("aprs_transmitter"));
    let mut source = Sender {
        packets,
        next: 0,
        rng: Pcg32::new(SEED, 4),
        idle: false,
    };
    let want = 200_000;
    assert!(pushed.len() > want, "{} samples pushed", pushed.len());
    let mut sizes = Pcg32::new(SEED, 5);
    let mut pulled = Vec::new();
    let mut buf = vec![0f32; 8192];
    while pulled.len() < want {
        let n = (1 + sizes.below(8_000) as usize).min(want - pulled.len());
        let (got, _) = p.pull_samples("audio", &mut buf[..n], &mut source).unwrap();
        assert_eq!(got, n);
        pulled.extend_from_slice(&buf[..got]);
    }
    assert!(pulled == pushed[..want], "pull differs from push");
    // The audio is the AFSK tone with the amplitude 1 (the resampler rings a little).
    let peak = pulled.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(peak > 0.9 && peak < 1.1, "peak {peak}");
}

// ---- test 2 ------------------------------------------------------------------------------

/// Frames of test 2.
const FRAMES: usize = 100;
/// Frames that must arrive with `CRC_OK` and the bytes sent.
const MIN_OK: usize = 99;
/// SNR of the audio in dB over the whole 48 kHz band.
const SNR_DB: f64 = 20.0;
/// Sound-card clock offset of the second run in ppm.
const DRIFT_PPM: f64 = 200.0;
/// Audio samples pulled per call.
const PULL: usize = 4_096;

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

/// The decoded frames that match the sent ones, in order: every decoded frame is matched with
/// the next sent frame that has its bytes.
fn matched(sent: &[Packet], got: &[Packet]) -> usize {
    let mut next = 0;
    let mut ok = 0;
    for g in got {
        if let Some(k) = sent[next..].iter().position(|s| s.data() == g.data()) {
            ok += 1;
            next += k + 1;
        }
    }
    ok
}

/// Test 2: `FRAMES` frames through the transmitter, the channel (optional clock drift, then
/// noise at `SNR_DB`) and the receiver; returns (decoded with `CRC_OK`, matched).
fn aprs_loop(seed: u64, drift_ppm: Option<f64>) -> (usize, usize) {
    let frames = aprs_frames(FRAMES, seed, 1);
    let mut tx = build(&flow("aprs_transmitter"));
    let mut rx = build(&flow("aprs_receiver"));
    let mut source = Sender {
        packets: frames.clone(),
        next: 0,
        rng: Pcg32::new(seed, 2),
        idle: true,
    };
    // The AFSK audio has the amplitude 1, so the power 1/2; the noise is real (the real part
    // of the complex noise of `Awgn`, sigma per component).
    let signal_power = 0.5;
    let sigma = (signal_power / 10f64.powf(SNR_DB / 10.0)).sqrt();
    let mut channel = ChannelChain::new();
    if let Some(ppm) = drift_ppm {
        channel.push(ClockDrift::new(ppm).unwrap());
    }
    channel.push(Awgn::new(sigma, seed).unwrap());
    let mut audio = vec![0f32; PULL];
    let mut cplx = vec![Complex64::new(0.0, 0.0); PULL];
    let mut noisy = Vec::new();
    let mut real = Vec::new();
    let mut got = Vec::new();
    let mut signal = 0.0;
    let mut noise_energy = 0.0;
    // After the last frame, 1 s of flags carries it through the receiver.
    let mut tail = 0usize;
    while tail < 48_000 {
        let (n, _) = tx.pull_samples("audio", &mut audio, &mut source).unwrap();
        assert_eq!(n, PULL);
        for (c, a) in cplx.iter_mut().zip(&audio) {
            *c = Complex64::new(f64::from(*a), 0.0);
            signal += f64::from(*a) * f64::from(*a);
        }
        noisy.clear();
        channel.process(&cplx, &mut noisy);
        real.clear();
        real.extend(noisy.iter().map(|c| c.re as f32));
        rx.push("audio", InputData::F32(&real)).unwrap();
        read_packets(&mut rx, &mut got);
        if drift_ppm.is_none() {
            for (r, a) in real.iter().zip(&audio) {
                noise_energy += f64::from(r - a) * f64::from(r - a);
            }
        }
        if source.next == FRAMES {
            tail += real.len();
        }
    }
    assert!(got.iter().all(|p| p.flags == PacketFlags::CRC_OK));
    let ok = matched(&frames, &got);
    let measured = if drift_ppm.is_none() {
        format!("{:.2} dB", 10.0 * (signal / noise_energy).log10())
    } else {
        "-".into()
    };
    eprintln!(
        "APRS loop (SNR {SNR_DB} dB over 48 kHz, measured {measured}, clock offset {drift_ppm:?} \
         ppm, seed {seed}): {FRAMES} frames sent, {} decoded with CRC_OK, {ok} with the bytes \
         sent",
        got.len()
    );
    (got.len(), ok)
}

#[test]
fn aprs_loop_100_frames_at_20_db() {
    let (_, ok) = aprs_loop(SEED, None);
    assert!(ok >= MIN_OK, "only {ok} of {FRAMES} frames");
}

#[test]
fn aprs_loop_100_frames_at_20_db_with_200_ppm_clock_offset() {
    let (_, ok) = aprs_loop(SEED_DRIFT, Some(DRIFT_PPM));
    assert!(ok >= MIN_OK, "only {ok} of {FRAMES} frames");
}

// ---- test 3 ------------------------------------------------------------------------------

#[test]
fn the_reference_signal_decodes_completely() {
    let case = golden::load("flows", "aprs_bell202_20_frames").unwrap();
    let Array::I8(raw) = case.array("audio").unwrap() else {
        panic!("audio is i8")
    };
    let Array::U8(bytes) = case.array("frames").unwrap() else {
        panic!("frames is u8")
    };
    let Array::I32(lengths) = case.array("lengths").unwrap() else {
        panic!("lengths is i32")
    };
    let scale = case.param_f64("scale").unwrap() as f32;
    let audio: Vec<f32> = raw.iter().map(|&v| f32::from(v) * scale).collect();
    let mut want = Vec::new();
    let mut at = 0;
    for &len in lengths {
        let len = usize::try_from(len).unwrap();
        want.push(bytes[at..at + len].to_vec());
        at += len;
    }
    assert_eq!(want.len(), 20);
    assert_eq!(at, bytes.len());

    let mut rx = build(&flow("aprs_receiver"));
    let mut got = Vec::new();
    for piece in audio.chunks(4096) {
        rx.push("audio", InputData::F32(piece)).unwrap();
        read_packets(&mut rx, &mut got);
    }
    let decoded: Vec<Vec<u8>> = got.iter().map(|p| p.data().to_vec()).collect();
    eprintln!(
        "APRS reference signal ({} samples at 48 kHz): {} of {} frames decoded",
        audio.len(),
        decoded.len(),
        want.len()
    );
    assert!(got.iter().all(|p| p.flags == PacketFlags::CRC_OK));
    assert_eq!(decoded, want);
}
