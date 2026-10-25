//! The `am_receiver` and `airband_receiver` example flows: they validate, build and run on the
//! E2E-AM signal; the airband flow also on a burst, where the squelch keeps the audio silent
//! outside the transmission (exactly zero before the burst and again once the gate has closed
//! after it).
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
use sdrk_device::formats;
use sdrk_pipeline::{BuildOptions, InputData, Pipeline, Registry};
use sdrk_testkit::e2e_am::{
    AM_AUDIO_HZ, AM_CARRIER, AM_INDEX, AM_MAX_LEVEL_ERROR_DB, AM_MIN_SINAD_F32_DB, AM_NOISE_SIGMA,
    AM_OFFSET_HZ,
};
use sdrk_testkit::{Pcg32, encode::encode, r#gen::am_tone, meters::ToneAccumulator};
use serde_json::json;

use common::{build, flow, tone_power};

const RATE: f64 = 2_400_000.0;
const AUDIO_RATE: f64 = 48_000.0;
const PIECE: usize = 4096;

/// CU8 bytes: the AM signal at the offset in `[from, to)` (all of the run when `None`) plus
/// noise over the whole run.
fn recording(samples: usize, burst: Option<(usize, usize)>) -> Vec<u8> {
    let (from, to) = burst.unwrap_or((0, samples));
    let rate = Rate::hz(RATE as u64).unwrap();
    let mut rng = Pcg32::new(2000, 1);
    let mut bytes = vec![0u8; 2 * samples];
    let mut tone = vec![C32::default(); PIECE];
    let mut block = vec![Complex64::new(0.0, 0.0); PIECE];
    let mut start = 0usize;
    while start < samples {
        let n = PIECE.min(samples - start);
        am_tone(
            &mut tone[..n],
            rate,
            AM_AUDIO_HZ,
            AM_INDEX,
            AM_CARRIER,
            0.0,
            start as u64,
        );
        for i in 0..n {
            let k = start + i;
            let cycles = AM_OFFSET_HZ * k as f64 / RATE;
            let w = std::f64::consts::TAU * (cycles - cycles.floor());
            let s = Complex64::new(f64::from(tone[i].re), f64::from(tone[i].im));
            let on = (from..to).contains(&k);
            block[i] = if on {
                s * Complex64::new(w.cos(), w.sin())
            } else {
                Complex64::new(0.0, 0.0)
            };
            block[i] += Complex64::new(rng.normal(), rng.normal()) * AM_NOISE_SIGMA;
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

fn receiver(name: &str) -> Pipeline {
    let mut p = build(&flow(name));
    p.set_param("tune", "frequency_hz", json!(AM_OFFSET_HZ))
        .unwrap();
    p
}

#[test]
fn both_flows_validate_without_warnings() {
    for name in ["am_receiver", "airband_receiver"] {
        let report = Registry::standard().validate_json(&flow(name), &BuildOptions::default());
        assert!(report.is_ok(), "{name}: {}", report.to_json());
    }
}

/// A one-node flow on a 48 kHz complex input.
fn single(kind: &str, params: serde_json::Value, precision: &str) -> serde_json::Value {
    let ty = if precision == "q15" { "cq15" } else { "c32" };
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": precision, "max_chunk": 1024 },
        "inputs": { "in": { "type": ty, "rate_hz": 48000 } },
        "nodes": { "n": { "kind": kind, "params": params } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    })
}

#[test]
fn the_cost_of_the_new_kinds_follows_the_table() {
    use sdrk_blocks::AmDemodConfig;
    let registry = Registry::standard();
    for precision in ["f32", "q15"] {
        // agc: 6 per input sample, no storage, same rate and no delay.
        let doc = single("agc", json!({}), precision);
        let report = registry.validate_json(&doc.to_string(), &BuildOptions::default());
        assert!(report.is_ok(), "{}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 6.0);
        assert_eq!(report.nodes[0].rate_out, report.nodes[0].rate_in);
        assert_eq!(
            report.nodes[0].delay_samples_in.reduced(),
            sdrk::Delay::ZERO.reduced()
        );
        // am_demod: 4 + taps / decimation, taps of the audio filter (0 without it).
        for (decimation, cutoff) in [(1, 5000.0), (2, 5000.0), (4, 2000.0)] {
            let doc = single(
                "am_demod",
                json!({ "audio_decimation": decimation, "audio_cutoff_hz": cutoff }),
                precision,
            );
            let report = registry.validate_json(&doc.to_string(), &BuildOptions::default());
            assert!(report.is_ok(), "{}", report.to_json());
            let cfg = AmDemodConfig {
                audio_decimation: decimation,
                audio_cutoff_hz: cutoff,
                ..AmDemodConfig::default()
            };
            let taps = cfg.audio_taps(Rate::hz(48_000).unwrap()).unwrap();
            assert_eq!(
                report.cost_per_input_sample,
                4.0 + taps as f64 / decimation as f64,
                "decimation {decimation}"
            );
            assert_eq!(
                report.nodes[0].rate_out,
                Rate::hz(48_000 / decimation as u64).unwrap()
            );
            if decimation > 1 {
                assert!(taps > 3);
            }
        }
    }
}

#[test]
fn the_am_receiver_demodulates_the_tone_at_the_agc_level() {
    let samples = 960_000; // 0.4 s
    let bytes = recording(samples, None);
    let audio = run(&mut receiver("am_receiver"), &bytes, 8192);
    assert!(audio.len() >= samples / 50 - 2);
    let window: Vec<f64> = audio[audio.len() / 2..]
        .iter()
        .map(|v| f64::from(*v))
        .collect();
    let mut meter = ToneAccumulator::new(AUDIO_RATE, AM_AUDIO_HZ);
    for v in &window {
        meter.push(*v);
    }
    assert!(
        meter.sinad_db() >= AM_MIN_SINAD_F32_DB,
        "pipeline SINAD {} dB, need {AM_MIN_SINAD_F32_DB} dB",
        meter.sinad_db()
    );
    // The level is the modulation depth times the AGC reference (-6 dBFS).
    let expected = 10f64.powf(-6.0 / 20.0) * AM_INDEX;
    let error_db = 20.0 * (meter.amplitude() / expected).log10();
    assert!(
        error_db.abs() <= AM_MAX_LEVEL_ERROR_DB,
        "audio level {error_db} dB from reference * index"
    );
    // The carrier is removed: the mean of the audio is close to zero.
    let mean = window.iter().sum::<f64>() / window.len() as f64;
    assert!(mean.abs() < 0.02, "DC {mean}");
}

#[test]
fn the_squelch_gates_the_airband_burst() {
    let samples = 2_400_000; // 1.0 s
    let (from, to) = (240_000, 960_000); // 0.1 s to 0.4 s
    let bytes = recording(samples, Some((from, to)));
    let audio = run(&mut receiver("airband_receiver"), &bytes, 8192);
    let per_sample = (RATE / AUDIO_RATE) as usize;
    assert!(audio.len() >= samples / per_sample - 2);
    let (a_from, a_to) = (from / per_sample, to / per_sample); // 4800 and 19200

    // Before the transmission the audio is exactly zero.
    assert!(audio[..a_from].iter().all(|v| *v == 0.0), "silence before");
    // The first non-zero audio sample comes at most 1 ms plus the channel filter delay after
    // the start of the transmission, and not before it.
    let first = audio.iter().position(|v| *v != 0.0).unwrap();
    assert!(
        (a_from..a_from + 48 + 100).contains(&first),
        "first non-zero audio sample {first}, transmission starts at {a_from}"
    );
    // The transmission carries the 1 kHz tone.
    let mid = &audio[a_from + 4800..a_to - 1200];
    let on = tone_power(mid, 1000.0, AUDIO_RATE);
    let off = tone_power(mid, 3300.0, AUDIO_RATE);
    assert!(on > 1000.0 * off, "1 kHz tone: {on} vs {off}");
    assert!(mid.iter().any(|v| v.abs() > 0.1));
    // After the gate closed (at most 0.1 s after the end) the audio is exactly zero, as in
    // E2E-NFM: the squelch writes exact zeros, the AGC holds its gain on them and the
    // demodulator writes zero audio and clears its DC blocker (no decaying tail).
    let closed = a_to + 4800;
    assert!(
        audio[closed..].iter().all(|v| *v == 0.0),
        "silence after the gate closed"
    );
    assert!(
        audio[a_to..closed].iter().any(|v| *v != 0.0),
        "the gate is still open for a while after the end of the burst"
    );
}

#[test]
fn chunking_does_not_change_the_output() {
    let samples = 480_000;
    for (name, burst) in [
        ("am_receiver", None),
        ("airband_receiver", Some((120_000, 360_000))),
    ] {
        let bytes = recording(samples, burst);
        let whole = run(&mut receiver(name), &bytes, 8192);
        let other = run(&mut receiver(name), &bytes, 4097);
        assert_eq!(common::bits(&whole), common::bits(&other), "{name}");
    }
}

#[test]
fn run_time_parameters_of_the_agc_and_the_squelch() {
    let bytes = recording(480_000, None);
    let level = |audio: &[f32]| {
        let tail = &audio[audio.len() * 3 / 4..];
        tail.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    };
    // A lower reference lowers the audio level (here by 12 dB, within the AGC time constant).
    let base = run(&mut receiver("am_receiver"), &bytes, 8192);
    let mut p = receiver("am_receiver");
    p.set_param("agc", "reference_db", json!(-18)).unwrap();
    let quiet = run(&mut p, &bytes, 8192);
    assert!(level(&quiet) < 0.5 * level(&base));
    // Build parameters cannot change while the flow runs.
    assert!(p.set_param("agc", "initial_gain_db", json!(3)).is_err());
    assert!(p.set_param("demod", "dc_cutoff_hz", json!(10)).is_err());
    // A squelch threshold above the signal power (0.25 = -6 dBFS) keeps the audio silent.
    let mut p = receiver("airband_receiver");
    p.set_param("gate", "threshold_db", json!(0)).unwrap();
    let silent = run(&mut p, &bytes, 8192);
    assert!(!silent.is_empty());
    assert!(silent.iter().all(|v| *v == 0.0));
}

#[test]
fn the_demodulator_needs_complex_input() {
    // am_demod behind itself (its output is real audio) is invalid.
    let mut spec: serde_json::Value = serde_json::from_str(&flow("am_receiver")).unwrap();
    spec["nodes"]["again"] = json!({"kind": "am_demod"});
    spec["edges"] = json!([
        {"from": "rf", "to": "tune"},
        {"from": "tune", "to": "agc"},
        {"from": "agc", "to": "demod"},
        {"from": "demod", "to": "again"},
        {"from": "again", "to": "audio"},
    ]);
    let report = Registry::standard().validate_json(&spec.to_string(), &BuildOptions::default());
    assert!(!report.is_ok(), "{}", report.to_json());
    // So is an agc behind the demodulator.
    let mut spec: serde_json::Value = serde_json::from_str(&flow("am_receiver")).unwrap();
    spec["edges"] = json!([
        {"from": "rf", "to": "tune"},
        {"from": "tune", "to": "demod"},
        {"from": "demod", "to": "agc"},
        {"from": "agc", "to": "audio"},
    ]);
    let report = Registry::standard().validate_json(&spec.to_string(), &BuildOptions::default());
    assert!(!report.is_ok(), "{}", report.to_json());
}
