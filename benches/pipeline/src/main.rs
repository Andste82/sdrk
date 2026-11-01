//! The pipeline benchmark of `cargo xtask count --pipeline`: the receive chain of
//! `examples/flows/fm_receiver.json` and the SSB chain (`ddc`, `agc`, `ssb_demod` at the rates
//! of the end-to-end SSB test) at 2.4 MHz, and the whole blocks `fm_demod` (`fm480`) and
//! `fm_stereo_demod` (`stereo480`) on a `cf32` input at 480 kHz (M24).
//!
//! ```text
//! pipeline run <k> [ssb|fm480|stereo480]    one warm-up block, then k blocks of 4096 input samples
//! pipeline info [ssb|fm480|stereo480]       latency of the audio output, as a line of JSON
//! ```
//!
//! The instruction counter measures the whole process; the differential method of `xtask`
//! (`k = 0` against `k = K`) leaves the instructions of the `K` blocks.

use sdrk_pipeline::{BuildOptions, InputData, Registry};

/// Input samples per block (`max_chunk` of the flow).
const BLOCK: usize = 4096;

/// The SSB chain of the milestone M21: the channel filter of the example flow `ssb_receiver`
/// tuned to the carrier of the end-to-end test, the AGC with its defaults and the upper
/// sideband demodulator with its defaults.
const SSB_FLOW: &str = r#"{
  "sdrk": 1,
  "name": "ssb-chain",
  "settings": { "precision": "f32", "max_chunk": 4096 },
  "inputs": { "rf": { "format": "cu8", "rate_hz": 2400000 } },
  "nodes": {
    "tune": { "kind": "ddc", "params": { "frequency_hz": 300000, "cutoff_hz": 4000, "decimation": 50, "max_taps": 8192, "transition_hz": 2000 } },
    "agc": { "kind": "agc" },
    "demod": { "kind": "ssb_demod", "params": { "sideband": "usb" } }
  },
  "edges": [
    { "from": "rf", "to": "tune" },
    { "from": "tune", "to": "agc" },
    { "from": "agc", "to": "demod" },
    { "from": "demod", "to": "audio" }
  ],
  "outputs": { "audio": { "type": "stream", "capacity": 16384, "drop": "error" } }
}"#;

/// The whole `fm_demod` block on a `cf32` input at 480 kHz (the rate of the stereo flow) with
/// the audio decimation of `fm_stereo_demod` (48 kHz output).
const FM480_FLOW: &str = r#"{
  "sdrk": 1,
  "name": "fm-480k",
  "settings": { "precision": "f32", "max_chunk": 4096 },
  "inputs": { "rf": { "format": "cf32", "rate_hz": 480000 } },
  "nodes": { "demod": { "kind": "fm_demod", "params": { "audio_decimation": 10 } } },
  "edges": [ { "from": "rf", "to": "demod" }, { "from": "demod", "to": "audio" } ],
  "outputs": { "audio": { "type": "stream", "capacity": 16384, "drop": "error" } }
}"#;

/// The whole `fm_stereo_demod` block on a `cf32` input at 480 kHz with its defaults.
const STEREO480_FLOW: &str = r#"{
  "sdrk": 1,
  "name": "fm-stereo-demod-480k",
  "settings": { "precision": "f32", "max_chunk": 4096 },
  "inputs": { "rf": { "format": "cf32", "rate_hz": 480000 } },
  "nodes": { "demod": { "kind": "fm_stereo_demod", "params": {} } },
  "edges": [ { "from": "rf", "to": "demod" }, { "from": "demod", "to": "audio" } ],
  "outputs": { "audio": { "type": "stream", "capacity": 16384, "drop": "error" } }
}"#;

/// The measured chain, from the second argument of the command line.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Chain {
    Fm,
    Ssb,
    Fm480,
    Stereo480,
}

impl Chain {
    fn parse(arg: Option<&String>) -> Self {
        match arg.map(String::as_str) {
            Some("ssb") => Self::Ssb,
            Some("fm480") => Self::Fm480,
            Some("stereo480") => Self::Stereo480,
            _ => Self::Fm,
        }
    }
}

fn build(chain: Chain) -> sdrk_pipeline::Pipeline {
    let json = match chain {
        Chain::Ssb => SSB_FLOW,
        Chain::Fm480 => FM480_FLOW,
        Chain::Stereo480 => STEREO480_FLOW,
        Chain::Fm => include_str!("../../../examples/flows/fm_receiver.json"),
    };
    Registry::standard()
        .build_json(json, &BuildOptions::default())
        .expect("the example flow builds")
}

/// One block of `cu8` samples: a carrier 600 kHz off the centre, frequency-modulated by a
/// 1 kHz tone (the flow tunes to +600 kHz).
fn block(index: usize, chain: Chain) -> Vec<u8> {
    match chain {
        Chain::Ssb => return ssb_block(index),
        Chain::Fm480 | Chain::Stereo480 => return stereo_block(index),
        Chain::Fm => {}
    }
    let mut out = Vec::with_capacity(2 * BLOCK);
    for n in 0..BLOCK {
        let t = (index * BLOCK + n) as f64 / 2_400_000.0;
        let phase = 2.0 * std::f64::consts::PI * 600_000.0 * t
            + 75_000.0 / 1_000.0 * (2.0 * std::f64::consts::PI * 1_000.0 * t).sin();
        out.push((127.5 + 100.0 * phase.cos()) as u8);
        out.push((127.5 + 100.0 * phase.sin()) as u8);
    }
    out
}

/// One block of `cf32` samples at 480 kHz: an FM stereo multiplex (left 1 kHz, right 3 kHz,
/// pilot 0.1 at 19 kHz, 75 kHz deviation), the E2E-STEREO signal without noise.
fn stereo_block(index: usize) -> Vec<u8> {
    let tau = 2.0 * std::f64::consts::PI;
    let mut out = Vec::with_capacity(8 * BLOCK);
    for n in 0..BLOCK {
        let t = (index * BLOCK + n) as f64 / 480_000.0;
        // The integral of the multiplex 0.45 sin(1 k) + 0.45 sin(3 k) + 0.225 (cos(37 k) -
        // cos(39 k)) - 0.225 (cos(35 k) - cos(41 k)) + 0.1 sin(19 k), term by term.
        let sin_int = |amp: f64, f: f64| amp * (1.0 - (tau * f * t).cos()) / (tau * f);
        let cos_int = |amp: f64, f: f64| amp * (tau * f * t).sin() / (tau * f);
        let integral = sin_int(0.45, 1_000.0)
            + sin_int(0.45, 3_000.0)
            + cos_int(0.225, 37_000.0)
            - cos_int(0.225, 39_000.0)
            - cos_int(0.225, 35_000.0)
            + cos_int(0.225, 41_000.0)
            + sin_int(0.1, 19_000.0);
        let phase = tau * 75_000.0 * integral;
        out.extend_from_slice(&(phase.cos() as f32).to_le_bytes());
        out.extend_from_slice(&(phase.sin() as f32).to_le_bytes());
    }
    out
}

/// One block of `cu8` samples: a suppressed carrier 300 kHz off the centre with a tone 1 kHz
/// above it and one 1.5 kHz below it (each amplitude 0.3).
fn ssb_block(index: usize) -> Vec<u8> {
    let tau = 2.0 * std::f64::consts::PI;
    let mut out = Vec::with_capacity(2 * BLOCK);
    for n in 0..BLOCK {
        let t = (index * BLOCK + n) as f64 / 2_400_000.0;
        let (si, sq) = (
            0.3 * (tau * 301_000.0 * t).cos() + 0.3 * (tau * 298_500.0 * t).cos(),
            0.3 * (tau * 301_000.0 * t).sin() + 0.3 * (tau * 298_500.0 * t).sin(),
        );
        out.push((127.5 + 127.0 * si) as u8);
        out.push((127.5 + 127.0 * sq) as u8);
    }
    out
}

fn drain(
    p: &mut sdrk_pipeline::Pipeline,
    chain: Chain,
    audio: &mut [f32],
    stereo: &mut [sdrk::C32],
    frame: &mut [f32],
) {
    if chain == Chain::Stereo480 {
        while p.read_stream::<sdrk::C32>("audio", stereo).unwrap_or(0) > 0 {}
    } else {
        while p.read_stream::<f32>("audio", audio).unwrap_or(0) > 0 {}
    }
    while let Ok(Some(_)) = p.read_frame("spectrum", frame) {}
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => {
            let k: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            let chain = Chain::parse(args.get(2));
            let mut p = build(chain);
            let mut audio = vec![0f32; 4096];
            let mut stereo = vec![sdrk::C32::new(0.0, 0.0); 4096];
            let mut frame = vec![0f32; 1024];
            let warm = block(0, chain);
            p.push("rf", InputData::Raw(&warm)).expect("push");
            drain(&mut p, chain, &mut audio, &mut stereo, &mut frame);
            let blocks: Vec<Vec<u8>> = (1..=k).map(|i| block(i, chain)).collect();
            for b in &blocks {
                p.push("rf", InputData::Raw(b)).expect("push");
                drain(&mut p, chain, &mut audio, &mut stereo, &mut frame);
            }
            std::hint::black_box((&audio, &stereo));
        }
        Some("info") => {
            let p = build(Chain::parse(args.get(1)));
            let rate = p.output_rate("audio").expect("rate");
            let lat = p.latency("audio").expect("latency");
            let samples = lat.group_delay.as_f64() + lat.max_buffered as f64;
            println!(
                "{{\"output\":\"audio\",\"rate_hz\":{},\"group_delay_samples\":{:.3},\"max_buffered_samples\":{},\"latency_samples\":{:.3},\"latency_us\":{:.1}}}",
                rate.as_f64(),
                lat.group_delay.as_f64(),
                lat.max_buffered,
                samples,
                samples / rate.as_f64() * 1e6
            );
        }
        _ => {
            eprintln!("usage: pipeline run <k> [ssb|fm480|stereo480] | info [ssb|fm480|stereo480]");
            std::process::exit(2);
        }
    }
}
