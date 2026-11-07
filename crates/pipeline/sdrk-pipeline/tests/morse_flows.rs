//! The flows `cw_transmitter`, `cw_receiver`, `cw_tone_transmitter` and `cw_tone_receiver`
//! (M47 §5) and tests 2, 3, 5 and 9.
//!
//! - All four flows validate without warnings, build and have the parameters of §5.
//! - **Test 2 (round trip, IQ):** `cw_transmitter` in pull mode (cs8) -> `cw_receiver` (the
//!   bytes as cu8): the text comes out word for word.
//! - **Test 9 (tone variants):** `cw_tone_transmitter` -> `cw_tone_receiver` at 700 Hz and
//!   450 Hz (the pitch is the transmitter's `frequency_hz` and the receiver's band-pass centre,
//!   both set with `set_param`): the text comes out word for word at 20 dB.
//! - **Test 3 (noise):** the tone loop with white Gaussian noise (seed 652): the character
//!   error rate (edit distance over the characters sent) at 6, 9, 12 and 15 dB, SNR as the
//!   power of the keyed tone over the noise power in 500 Hz.
//! - **Test 5 (frequency offset):** the transmitter at 650, 700 and 750 Hz against a receiver
//!   at 700 Hz at 15 dB.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stderr,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::too_many_lines,
    clippy::needless_range_loop
)]

mod common;

use sdrk_pipeline::{
    BuildOptions, InputData, Packet, PacketFlags, Pipeline, PullError, PullSource, Registry,
};
use sdrk_testkit::Pcg32;
use serde_json::json;

use common::{build, flow};

const FLOWS: [&str; 4] = [
    "cw_transmitter",
    "cw_receiver",
    "cw_tone_transmitter",
    "cw_tone_receiver",
];

/// Seed of the noise and the texts of test 3.
const SEED: u64 = 652;

/// A packet source for pull mode: nothing until `go` is set, then one packet per call.
struct Sender {
    packets: Vec<Packet>,
    next: usize,
    go: bool,
}

impl PullSource for Sender {
    fn provide(&mut self, input: &str, max: usize) -> Result<InputData<'_>, PullError> {
        assert_eq!(input, "msg");
        if !self.go || self.next == self.packets.len() || max == 0 {
            return Ok(InputData::Packets(&[]));
        }
        let from = self.next;
        self.next += 1;
        Ok(InputData::Packets(&self.packets[from..self.next]))
    }
}

fn packets_of(texts: &[&[u8]]) -> Vec<Packet> {
    texts
        .iter()
        .map(|t| Packet::new(0, PacketFlags::NONE, t).unwrap())
        .collect()
}

fn read_words(rx: &mut Pipeline, into: &mut Vec<String>) {
    let mut buf = vec![Packet::default(); 64];
    loop {
        let n = rx.read_packets("words", &mut buf).unwrap();
        if n == 0 {
            return;
        }
        into.extend(
            buf[..n]
                .iter()
                .map(|p| String::from_utf8(p.data().to_vec()).unwrap()),
        );
    }
}

#[test]
fn the_flows_validate_without_warnings() {
    let registry = Registry::standard();
    for name in FLOWS {
        let report = registry.validate_json(&flow(name), &BuildOptions::default());
        assert!(report.is_ok(), "{name}: {}", report.to_json());
        assert!(report.warnings.is_empty(), "{name}: {}", report.to_json());
        assert!(report.memory_bytes > 0, "{name}");
        build(&flow(name));
    }
}

/// The (node, kind, input rate, output rate) of every node of a flow, rounded to hertz.
fn nodes(name: &str) -> Vec<(String, String, u64, u64)> {
    let report = Registry::standard().validate_json(&flow(name), &BuildOptions::default());
    report
        .nodes
        .iter()
        .filter(|n| !n.auto)
        .map(|n| {
            (
                n.name.clone(),
                n.kind.clone(),
                n.rate_in.as_f64().round() as u64,
                n.rate_out.as_f64().round() as u64,
            )
        })
        .collect()
}

#[test]
fn the_flows_have_the_nodes_and_rates_of_the_spec() {
    let tx = nodes("cw_transmitter");
    assert!(
        tx.contains(&("keys".into(), "morse_encode".into(), 8000, 8000)),
        "{tx:?}"
    );
    assert!(
        tx.contains(&("up".into(), "duc".into(), 8000, 240_000)),
        "{tx:?}"
    );
    let rx = nodes("cw_receiver");
    assert!(
        rx.contains(&("tune".into(), "ddc".into(), 240_000, 8000)),
        "{rx:?}"
    );
    assert!(
        rx.iter().any(|n| n.1 == "morse_decode" && n.2 == 8000),
        "{rx:?}"
    );
    let tt = nodes("cw_tone_transmitter");
    assert!(
        tt.iter()
            .any(|n| n.1 == "nco" && n.2 == 8000 && n.3 == 8000),
        "{tt:?}"
    );
    let tr = nodes("cw_tone_receiver");
    assert!(
        tr.iter().any(|n| n.1 == "morse_decode" && n.2 == 8000),
        "{tr:?}"
    );
    // The decoder needs 8 samples per unit at 40 words per minute: 267 Hz.
    let slow = flow("cw_tone_receiver").replace("8000", "200");
    let report = Registry::standard().validate_json(&slow, &BuildOptions::default());
    assert!(!report.is_ok());
}

/// The audio of the tone transmitter: `lead` samples of silence, the texts, and `tail`
/// samples of silence after the last tone.
fn tone_audio(texts: &[&[u8]], tone_hz: f64, lead: usize, tail: usize) -> Vec<f32> {
    let mut tx = build(&flow("cw_tone_transmitter"));
    tx.set_param("tone", "frequency_hz", json!(tone_hz))
        .unwrap();
    let mut source = Sender {
        packets: packets_of(texts),
        next: 0,
        go: false,
    };
    let mut out = Vec::new();
    let mut buf = vec![0f32; 4096];
    let mut pull = |tx: &mut Pipeline, source: &mut Sender, n: usize, out: &mut Vec<f32>| {
        let (got, _) = tx.pull_samples("audio", &mut buf[..n], source).unwrap();
        assert_eq!(got, n);
        out.extend_from_slice(&buf[..got]);
    };
    for _ in 0..lead.div_ceil(4096) {
        pull(&mut tx, &mut source, 4096, &mut out);
    }
    source.go = true;
    // Pull until the tone has been heard and then `tail` samples of silence have come.
    let mut silent = 0;
    let mut heard = false;
    while !heard || silent < tail {
        let before = out.len();
        pull(&mut tx, &mut source, 4096, &mut out);
        for &x in &out[before..] {
            if x.abs() > 1e-4 {
                heard = true;
                silent = 0;
            } else {
                silent += 1;
            }
        }
        assert!(out.len() < 20_000_000, "the transmitter does not stop");
    }
    out
}

/// The words the tone receiver returns for `audio`, centred at `centre_hz`, in pieces of 997
/// samples.
fn tone_words(audio: &[f32], centre_hz: f64) -> Vec<String> {
    let mut rx = build(&flow("cw_tone_receiver"));
    rx.set_param("tone", "cutoff_hz", json!(centre_hz)).unwrap();
    let mut got = Vec::new();
    for piece in audio.chunks(997) {
        rx.push("audio", InputData::F32(piece)).unwrap();
        read_words(&mut rx, &mut got);
    }
    got
}

#[test]
fn the_tone_loop_returns_the_text() {
    let texts: [&[u8]; 3] = [b"VVV CQ CQ DE DL1ABC", b"TEST 599 73", b"K"];
    for tone in [700.0, 450.0] {
        let audio = tone_audio(&texts, tone, 4000, 8000);
        let got = tone_words(&audio, tone);
        let want: Vec<String> = texts
            .iter()
            .flat_map(|t| {
                String::from_utf8(t.to_vec())
                    .unwrap()
                    .split(' ')
                    .map(String::from)
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(got, want, "tone {tone} Hz");
    }
}

#[test]
fn the_iq_loop_returns_the_text() {
    let mut tx = build(&flow("cw_transmitter"));
    let mut source = Sender {
        packets: packets_of(&[b"VVV TEST"]),
        next: 0,
        go: false,
    };
    let mut bytes = Vec::new();
    let mut buf = vec![0u8; 16384];
    let mut pull = |tx: &mut Pipeline, source: &mut Sender, bytes: &mut Vec<u8>| {
        let (got, _) = tx.pull("rf", &mut buf, source).unwrap();
        assert_eq!(got, buf.len());
        bytes.extend_from_slice(&buf[..got]);
    };
    // 0.25 s of silence, then the text, then 0.5 s of silence after the last carrier.
    while bytes.len() < 2 * 240_000 / 4 {
        pull(&mut tx, &mut source, &mut bytes);
    }
    source.go = true;
    let mut silent = 0;
    let mut heard = false;
    while !heard || silent < 2 * 240_000 / 2 {
        let before = bytes.len();
        pull(&mut tx, &mut source, &mut bytes);
        for &b in &bytes[before..] {
            // The up converter rings a little at the edges; anything below 4 of 127 is silence.
            if (b as i8).unsigned_abs() > 4 {
                heard = true;
                silent = 0;
            } else {
                silent += 1;
            }
        }
        assert!(bytes.len() < 40_000_000, "the transmitter does not stop");
    }
    // Signed 8-bit IQ to unsigned 8-bit IQ.
    for b in &mut bytes {
        *b ^= 0x80;
    }
    let mut rx = build(&flow("cw_receiver"));
    let mut got = Vec::new();
    for piece in bytes.chunks(2 * 4096) {
        rx.push("rf", InputData::Raw(piece)).unwrap();
        read_words(&mut rx, &mut got);
    }
    assert_eq!(got, ["VVV", "TEST"]);
}

/// The edit distance (insertions, deletions and substitutions) between two byte strings.
fn edit_distance(a: &[u8], b: &[u8]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, &x) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, &y) in b.iter().enumerate() {
            cur.push(
                (prev[j] + usize::from(x != y))
                    .min(prev[j + 1] + 1)
                    .min(cur[j] + 1),
            );
        }
        prev = cur;
    }
    prev[b.len()]
}

/// About 150 random characters in words of 2 to 6, after the preamble `VVV`.
fn random_text(rng: &mut Pcg32) -> Vec<u8> {
    let alphabet = b"ETAOINSHRDLUCMWFGYPBVKJXQZ0123456789";
    let mut text = b"VVV ".to_vec();
    let mut chars = 0;
    while chars < 150 {
        let len = 2 + rng.below(5) as usize;
        for _ in 0..len {
            text.push(alphabet[rng.below(alphabet.len() as u32) as usize]);
        }
        text.push(b' ');
        chars += len;
    }
    text.pop();
    text
}

/// The character error rate of the tone loop at `snr_db` (the power of the keyed tone over
/// the noise power in 500 Hz), the transmitter at `tone_hz`, the receiver at `centre_hz`.
fn tone_cer(snr_db: f64, tone_hz: f64, centre_hz: f64, seed: u64) -> f64 {
    let mut rng = Pcg32::new(seed, 7);
    let text = random_text(&mut rng);
    let audio = tone_audio(&[&text], tone_hz, 8000, 16000);
    // The keyed tone has the power 1/2 while the key is down. White noise of variance s^2 over
    // the 4 kHz band has s^2 / 8 in 500 Hz.
    let snr = 10f64.powf(snr_db / 10.0);
    let sigma = (0.5 * 8.0 / snr).sqrt();
    let mut rng_noise = Pcg32::new(seed, 11);
    let noisy: Vec<f32> = audio
        .iter()
        .map(|&a| (f64::from(a) + sigma * rng_noise.normal()) as f32)
        .collect();
    let got = tone_words(&noisy, centre_hz).join(" ");
    let want = String::from_utf8(text.clone()).unwrap();
    // Compare without the preamble, whose first characters set the estimate.
    let sent: Vec<u8> = want.bytes().skip(4).collect();
    let rest: Vec<u8> = got
        .bytes()
        .skip(got.find(' ').map_or(0, |i| i + 1))
        .collect();
    edit_distance(&sent, &rest) as f64 / sent.len() as f64
}

#[test]
fn the_character_error_rate_in_noise() {
    // Three texts and noises (seeds 652 to 654): at 12 dB and above the error rate stays at
    // 1 % or below, at 6 dB at 5 % or below (the starting values of the milestone, which the
    // measurement meets; the numbers are in the decision log).
    for seed in SEED..SEED + 3 {
        for (snr, limit) in [
            (6.0, 0.05),
            (9.0, 0.05),
            (12.0, 0.01),
            (15.0, 0.01),
            (20.0, 0.01),
        ] {
            let cer = tone_cer(snr, 700.0, 700.0, seed);
            eprintln!(
                "CW tone loop, SNR {snr} dB in 500 Hz (seed {seed}): CER {:.2} %",
                100.0 * cer
            );
            assert!(cer <= limit, "SNR {snr} dB, seed {seed}: CER {cer}");
        }
    }
}

#[test]
fn a_tone_off_the_band_pass_centre_is_decoded() {
    for off in [-50.0, 0.0, 50.0] {
        let cer = tone_cer(15.0, 700.0 + off, 700.0, SEED + 3);
        eprintln!(
            "CW tone loop, offset {off} Hz at 15 dB: CER {:.2} %",
            100.0 * cer
        );
        assert!(cer <= 0.01, "offset {off} Hz: CER {cer}");
    }
}
