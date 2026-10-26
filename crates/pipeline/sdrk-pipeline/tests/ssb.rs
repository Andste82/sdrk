//! The kind `ssb_demod` and the `ssb_receiver` example flow: cost, rates and delay of the
//! plan, validation, a run on an SSB signal and the run-time switch of the sideband.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::assert_is_empty,
    clippy::unreadable_literal,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::redundant_closure_for_method_calls
)]

mod common;

use num_complex::Complex64;
use sdrk::{C32, Rate};
use sdrk_blocks::SsbDemodConfig;
use sdrk_device::formats;
use sdrk_pipeline::{BuildOptions, InputData, Pipeline, Registry};
use sdrk_testkit::{
    Pcg32,
    encode::encode,
    r#gen::{Sideband, ssb_tone},
    meters::ToneAccumulator,
};
use serde_json::json;

use common::{build, flow};

const RATE: f64 = 2_400_000.0;
const AUDIO_RATE: f64 = 48_000.0;
const OFFSET_HZ: f64 = 300_000.0;
const PIECE: usize = 4096;

/// CU8 bytes of a tone `audio_hz` from a suppressed carrier at `OFFSET_HZ`, plus noise.
fn recording(samples: usize, audio_hz: f64, side: Sideband) -> Vec<u8> {
    let rate = Rate::hz(RATE as u64).unwrap();
    let mut rng = Pcg32::new(2100, 1);
    let mut bytes = vec![0u8; 2 * samples];
    let mut tone = vec![C32::default(); PIECE];
    let mut block = vec![Complex64::new(0.0, 0.0); PIECE];
    let mut start = 0usize;
    while start < samples {
        let n = PIECE.min(samples - start);
        ssb_tone(&mut tone[..n], rate, audio_hz, side, 0.3, start as u64);
        for i in 0..n {
            let k = start + i;
            let cycles = OFFSET_HZ * k as f64 / RATE;
            let w = std::f64::consts::TAU * (cycles - cycles.floor());
            let s = Complex64::new(f64::from(tone[i].re), f64::from(tone[i].im));
            block[i] = s * Complex64::new(w.cos(), w.sin())
                + Complex64::new(rng.normal(), rng.normal()) * 0.003;
        }
        encode(
            &formats::CU8,
            &block[..n],
            &mut bytes[2 * start..2 * (start + n)],
        )
        .unwrap();
        start += n;
    }
    bytes
}

/// CU8 bytes of the E2E-SSB signal: a tone 1 kHz on the wanted side and one 1.5 kHz on the other
/// side of a suppressed carrier at `OFFSET_HZ` (each 0.3), plus noise (sigma 0.003).
fn two_sided_recording(samples: usize, wanted: Sideband) -> Vec<u8> {
    let rate = Rate::hz(RATE as u64).unwrap();
    let other = match wanted {
        Sideband::Usb => Sideband::Lsb,
        Sideband::Lsb => Sideband::Usb,
    };
    let mut rng = Pcg32::new(2100, 1);
    let mut bytes = vec![0u8; 2 * samples];
    let mut a = vec![C32::default(); PIECE];
    let mut b = vec![C32::default(); PIECE];
    let mut block = vec![Complex64::new(0.0, 0.0); PIECE];
    let mut start = 0usize;
    while start < samples {
        let n = PIECE.min(samples - start);
        ssb_tone(&mut a[..n], rate, 1000.0, wanted, 0.3, start as u64);
        ssb_tone(&mut b[..n], rate, 1500.0, other, 0.3, start as u64);
        for i in 0..n {
            let k = start + i;
            let cycles = OFFSET_HZ * k as f64 / RATE;
            let w = std::f64::consts::TAU * (cycles - cycles.floor());
            let s = Complex64::new(
                f64::from(a[i].re) + f64::from(b[i].re),
                f64::from(a[i].im) + f64::from(b[i].im),
            );
            block[i] = s * Complex64::new(w.cos(), w.sin())
                + Complex64::new(rng.normal(), rng.normal()) * 0.003;
        }
        encode(
            &formats::CU8,
            &block[..n],
            &mut bytes[2 * start..2 * (start + n)],
        )
        .unwrap();
        start += n;
    }
    bytes
}

fn run(p: &mut Pipeline, bytes: &[u8], chunk: usize) -> Vec<f32> {
    let mut audio = Vec::new();
    let mut buf = vec![0f32; 4096];
    for piece in bytes.chunks(chunk) {
        p.push("rf", InputData::Raw(piece)).unwrap();
        loop {
            let got = p.read_stream("audio", &mut buf).unwrap();
            if got == 0 {
                break;
            }
            audio.extend_from_slice(&buf[..got]);
        }
    }
    audio
}

/// Like `run` for a `q15` flow: the audio as `f32` in [-1, 1).
fn run_q15(p: &mut Pipeline, bytes: &[u8], chunk: usize) -> Vec<f32> {
    let mut audio = Vec::new();
    let mut buf = vec![sdrk::Q15(0); 4096];
    for piece in bytes.chunks(chunk) {
        p.push("rf", InputData::Raw(piece)).unwrap();
        loop {
            let got = p.read_stream("audio", &mut buf).unwrap();
            if got == 0 {
                break;
            }
            audio.extend(buf[..got].iter().map(|q| f32::from(q.0) / 32768.0));
        }
    }
    audio
}

/// The flow tuned to the carrier. The AGC reference is set to the level of the test tone
/// (0.3), so that the gain starts settled: its release time constant of 0.2 s would otherwise
/// still be ramping the level during these short runs and modulate the tone.
fn receiver() -> Pipeline {
    let mut p = build(&flow("ssb_receiver"));
    p.set_param("tune", "frequency_hz", json!(OFFSET_HZ))
        .unwrap();
    p.set_param("agc", "reference_db", json!(20.0 * 0.3f64.log10()))
        .unwrap();
    p
}

fn sinad(audio: &[f32], tone_hz: f64) -> f64 {
    let mut meter = ToneAccumulator::new(AUDIO_RATE, tone_hz);
    for v in audio {
        meter.push(f64::from(*v));
    }
    meter.sinad_db()
}

fn level(audio: &[f32]) -> f32 {
    audio.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

#[test]
fn the_flow_validates_without_warnings() {
    let report =
        Registry::standard().validate_json(&flow("ssb_receiver"), &BuildOptions::default());
    assert!(report.is_ok(), "{}", report.to_json());
}

/// A one-node flow on a 48 kHz complex input.
fn single(params: serde_json::Value, precision: &str) -> serde_json::Value {
    let ty = if precision == "q15" { "cq15" } else { "c32" };
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": precision, "max_chunk": 1024 },
        "inputs": { "in": { "type": ty, "rate_hz": 48000 } },
        "nodes": { "n": { "kind": "ssb_demod", "params": params } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    })
}

#[test]
fn the_plan_follows_the_cost_formula() {
    let registry = Registry::standard();
    for precision in ["f32", "q15"] {
        for decimation in [1usize, 2, 4] {
            let doc = single(json!({ "audio_decimation": decimation }), precision);
            let report = registry.validate_json(&doc.to_string(), &BuildOptions::default());
            assert!(report.is_ok(), "{}", report.to_json());
            let cfg = SsbDemodConfig {
                audio_decimation: decimation,
                ..SsbDemodConfig::default()
            };
            let taps = cfg.num_taps(Rate::hz(48_000).unwrap()).unwrap();
            assert!(taps > 100);
            // ssb_demod: 8 + 4 * taps / audio_decimation per input sample.
            assert_eq!(
                report.cost_per_input_sample,
                8.0 + 4.0 * taps as f64 / decimation as f64,
                "{precision}, decimation {decimation}"
            );
            assert_eq!(
                report.nodes[0].rate_out,
                Rate::hz(48_000 / decimation as u64).unwrap()
            );
            assert_eq!(
                report.nodes[0].delay_samples_in.reduced(),
                sdrk::Delay::new(taps as u64 - 1, 2).unwrap().reduced()
            );
        }
    }
}

#[test]
fn the_kind_lists_its_kernels_and_checks_the_band() {
    let registry = Registry::standard();
    // A band that does not fit below half the output rate is out of range.
    let doc = single(json!({ "high_hz": 23_950 }), "f32");
    let report = registry.validate_json(&doc.to_string(), &BuildOptions::default());
    assert!(!report.is_ok());
    let doc = single(json!({ "low_hz": 3000, "high_hz": 3000 }), "f32");
    let report = registry.validate_json(&doc.to_string(), &BuildOptions::default());
    assert!(!report.is_ok());
    // An unknown sideband name is rejected.
    let doc = single(json!({ "sideband": "both" }), "f32");
    let report = registry.validate_json(&doc.to_string(), &BuildOptions::default());
    assert!(!report.is_ok());
    // Real audio behind the demodulator is not complex input.
    let mut spec: serde_json::Value = serde_json::from_str(&flow("ssb_receiver")).unwrap();
    spec["nodes"]["again"] = json!({"kind": "ssb_demod"});
    spec["edges"] = json!([
        {"from": "rf", "to": "tune"},
        {"from": "tune", "to": "gate"},
        {"from": "gate", "to": "agc"},
        {"from": "agc", "to": "demod"},
        {"from": "demod", "to": "again"},
        {"from": "again", "to": "audio"},
    ]);
    let report = registry.validate_json(&spec.to_string(), &BuildOptions::default());
    assert!(!report.is_ok(), "{}", report.to_json());
}

#[test]
fn the_receiver_demodulates_a_usb_tone() {
    let bytes = recording(480_000, 1000.0, Sideband::Usb);
    let audio = run(&mut receiver(), &bytes, 8192);
    assert!(audio.len() >= 480_000 / 50 - 2);
    let tail = &audio[audio.len() / 2..];
    assert!(
        sinad(tail, 1000.0) >= 35.0,
        "SINAD {} dB",
        sinad(tail, 1000.0)
    );
}

#[test]
fn the_sideband_can_be_switched_while_the_flow_runs() {
    let bytes = recording(960_000, 1000.0, Sideband::Usb);
    let half = bytes.len() / 2;
    let mut p = receiver();
    let first = run(&mut p, &bytes[..half], 8192);
    // The signal is in the upper sideband: after the switch to lsb it is rejected.
    p.set_param("demod", "sideband", json!("lsb")).unwrap();
    let second = run(&mut p, &bytes[half..], 8192);
    let (a, b) = (&first[first.len() / 2..], &second[second.len() / 2..]);
    assert!(level(a) > 0.29, "{}", level(a));
    assert!(level(b) < 0.02 * level(a), "{} vs {}", level(b), level(a));
    // Build parameters cannot change while the flow runs, and the sideband has two names.
    assert!(p.set_param("demod", "high_hz", json!(2500)).is_err());
    assert!(p.set_param("demod", "sideband", json!("both")).is_err());
    p.set_param("demod", "sideband", json!("usb")).unwrap();
}

#[test]
fn chunking_does_not_change_the_output() {
    let bytes = recording(480_000, 1500.0, Sideband::Lsb);
    let mut a = receiver();
    a.set_param("demod", "sideband", json!("lsb")).unwrap();
    let whole = run(&mut a, &bytes, 8192);
    let mut b = receiver();
    b.set_param("demod", "sideband", json!("lsb")).unwrap();
    let other = run(&mut b, &bytes, 4097);
    assert_eq!(common::bits(&whole), common::bits(&other));
    let s = sinad(&whole[whole.len() / 2..], 1500.0);
    assert!(s >= 35.0, "SINAD {s} dB");
}

/// The shipped flow, with its default parameters (AGC included), on the two-sided E2E-SSB
/// signal in both precisions and both sidebands: SINAD of the wanted 1 kHz tone and rejection of
/// the 1.5 kHz tone on the other side, with the thresholds of the milestone (35 dB / 50 dB for
/// `f32`, 30 dB / 40 dB for `q15`), measured over the last half of the audio.
#[test]
fn the_flow_meets_the_e2e_checks_in_both_sidebands_and_precisions() {
    let samples = 480_000;
    for (precision, min_sinad, min_rejection) in [("f32", 35.0, 50.0), ("q15", 30.0, 40.0)] {
        for (side, name) in [(Sideband::Usb, "usb"), (Sideband::Lsb, "lsb")] {
            let json = flow("ssb_receiver").replace(
                "\"precision\": \"f32\"",
                &format!("\"precision\": \"{precision}\""),
            );
            let mut p = build(&json);
            p.set_param("tune", "frequency_hz", json!(OFFSET_HZ))
                .unwrap();
            p.set_param("demod", "sideband", json!(name)).unwrap();
            let bytes = two_sided_recording(samples, side);
            let audio = if precision == "q15" {
                run_q15(&mut p, &bytes, 8192)
            } else {
                run(&mut p, &bytes, 8192)
            };
            assert!(audio.len() >= samples / 50 - 2);
            let tail = &audio[audio.len() / 2..];
            let mut wanted = ToneAccumulator::new(AUDIO_RATE, 1000.0);
            let mut other = ToneAccumulator::new(AUDIO_RATE, 1500.0);
            for v in tail {
                wanted.push(f64::from(*v));
                other.push(f64::from(*v));
            }
            let sinad = wanted.sinad_db();
            let rejection = 20.0 * (wanted.amplitude() / other.amplitude()).log10();
            assert!(
                sinad >= min_sinad,
                "{precision} {name}: SINAD {sinad} dB (limit {min_sinad})"
            );
            assert!(
                rejection >= min_rejection,
                "{precision} {name}: rejection {rejection} dB (limit {min_rejection})"
            );
        }
    }
}
