//! The pipeline benchmark of `cargo xtask count --pipeline`: the receive chain of
//! `examples/flows/fm_receiver.json` and the SSB chain (`ddc`, `agc`, `ssb_demod` at the rates
//! of the end-to-end SSB test) at 2.4 MHz.
//!
//! ```text
//! pipeline run <k> [ssb]    one warm-up block, then k blocks of 4096 input samples
//! pipeline info [ssb]       latency of the audio output, as a line of JSON
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

fn build(ssb: bool) -> sdrk_pipeline::Pipeline {
    let json = if ssb {
        SSB_FLOW
    } else {
        include_str!("../../../examples/flows/fm_receiver.json")
    };
    Registry::standard()
        .build_json(json, &BuildOptions::default())
        .expect("the example flow builds")
}

/// One block of `cu8` samples: a carrier 600 kHz off the centre, frequency-modulated by a
/// 1 kHz tone (the flow tunes to +600 kHz).
fn block(index: usize, ssb: bool) -> Vec<u8> {
    if ssb {
        return ssb_block(index);
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

fn drain(p: &mut sdrk_pipeline::Pipeline, audio: &mut [f32], frame: &mut [f32]) {
    while p.read_stream::<f32>("audio", audio).unwrap_or(0) > 0 {}
    while let Ok(Some(_)) = p.read_frame("spectrum", frame) {}
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => {
            let k: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            let ssb = args.get(2).map(String::as_str) == Some("ssb");
            let mut p = build(ssb);
            let mut audio = vec![0f32; 4096];
            let mut frame = vec![0f32; 1024];
            let warm = block(0, ssb);
            p.push("rf", InputData::Raw(&warm)).expect("push");
            drain(&mut p, &mut audio, &mut frame);
            let blocks: Vec<Vec<u8>> = (1..=k).map(|i| block(i, ssb)).collect();
            for b in &blocks {
                p.push("rf", InputData::Raw(b)).expect("push");
                drain(&mut p, &mut audio, &mut frame);
            }
            std::hint::black_box(&audio);
        }
        Some("info") => {
            let p = build(args.get(1).map(String::as_str) == Some("ssb"));
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
            eprintln!("usage: pipeline run <k> [ssb] | info [ssb]");
            std::process::exit(2);
        }
    }
}
