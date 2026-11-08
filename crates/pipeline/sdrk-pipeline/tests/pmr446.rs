//! The flow `pmr446_monitor` (M45 tests 7 to 10) on NFM channel scenes of the testkit
//! (`sdrk_testkit::nfm::ChannelScene`).
//!
//! - **Test 7 (adjacent-channel rejection, seed 667):** a 1 kHz tone at full deviation on
//!   channel 5 and noise-modulated NFM on channels 4 and 6, 10 dB stronger and at the same
//!   level; the SINAD of the demodulated audio of channel 5 (the output of `demod5`, tapped by a
//!   second output).
//! - **Test 8 (eight channels, seeds 668 and 669):** eight tones from 300 to 2400 Hz on channels
//!   1 to 8, one channel at a time and all at once, each tone in the audio of the flow; with
//!   noise only the audio is exactly silent (no squelch opens).
//! - **Test 9 (cost):** one `channelize` node with eight `channel_select` readers against eight
//!   `ddc` nodes for the same channels, both from 200 kS/s: the report cost of both (the cycles
//!   per sample are the count entries `channel_bank_channelize` and `channel_bank_ddc`,
//!   `docs/performance.md`).
//! - **Test 10 (flow, seeds 670 and 671):** the flow validates, builds and runs on a scene file;
//!   the YAML twin parses to the same spec; chunk invariance with 20 plans.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::needless_pass_by_value,
    clippy::unreadable_literal,
    clippy::large_stack_frames,
    clippy::print_stdout
)]

mod common;

use num_complex::Complex64;
use sdrk_device::formats;
use sdrk_pipeline::{BuildOptions, InputData, Pipeline, Registry};
use sdrk_testkit::nfm::{Audio, ChannelScene, SceneChannel};
use sdrk_testkit::{ChunkPlan, encode::encode, meters::sinad_db};
use serde_json::{Value, json};

use common::{build, flow, tone_power};

/// The input rate of the flow.
const RATE: f64 = 2_400_000.0;
/// The channel spacing of PMR446.
const SPACING: f64 = 12_500.0;
/// The rate of a channel (and of the audio of `demodN`).
const CHANNEL_RATE: f64 = 25_000.0;
/// The audio rate of the flow output.
const AUDIO_RATE: f64 = 48_000.0;
/// Peak deviation of the PMR446 transmitters.
const DEVIATION: f64 = 2_500.0;
/// The band noise of the scenes (complex Gaussian over 2.4 MHz): about -53 dBFS per channel.
const NOISE_DB: f64 = -30.0;
/// The level of a wanted transmitter.
const LEVEL_DB: f64 = -26.0;
/// The tones of channels 1 to 8.
const TONES: [f64; 8] = [300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0, 2100.0, 2400.0];

/// The minimum SINAD of channel 5 with the neighbours 10 dB stronger (test 7).
const MIN_SINAD_STRONG_DB: f64 = 20.0;
/// The minimum SINAD of channel 5 with the neighbours at the same level (test 7).
const MIN_SINAD_EQUAL_DB: f64 = 30.0;

/// The offset index of PMR446 channel `n` from the centre (channel 9).
fn lane(n: i32) -> i32 {
    n - 9
}

fn transmitter(n: i32, audio: Audio, level_db: f64) -> SceneChannel {
    SceneChannel {
        offset_index: lane(n),
        audio,
        deviation_hz: DEVIATION,
        level_db,
    }
}

/// CU8 bytes of `samples` samples of a scene at 2.4 MS/s, encoded with the testkit encoder.
fn scene_cu8(channels: &[SceneChannel], seed: u64, samples: usize) -> Vec<u8> {
    let scene = ChannelScene {
        spacing_hz: SPACING,
        channels,
        fs: RATE,
        noise_db: NOISE_DB,
        seed,
    };
    let mut r = scene.renderer().unwrap();
    let mut bytes = vec![0u8; 2 * samples];
    let mut x = vec![Complex64::new(0.0, 0.0); 4096];
    for piece in bytes.chunks_mut(2 * 4096) {
        let n = piece.len() / 2;
        r.render(&mut x[..n]);
        encode(&formats::CU8, &x[..n], piece).unwrap();
    }
    bytes
}

/// Pushes `bytes` in pieces of the given numbers of samples and reads the stream outputs
/// `names` after every push; returns their samples in the order of `names`.
fn run(
    p: &mut Pipeline,
    bytes: &[u8],
    sizes: &mut dyn Iterator<Item = usize>,
    names: &[&str],
) -> Vec<Vec<f32>> {
    let mut outs = vec![Vec::new(); names.len()];
    let mut buf = vec![0f32; 8192];
    let mut at = 0;
    while at < bytes.len() {
        let n = (2 * sizes.next().unwrap()).min(bytes.len() - at);
        p.push("rf", InputData::Raw(&bytes[at..at + n])).unwrap();
        at += n;
        for (name, out) in names.iter().zip(outs.iter_mut()) {
            loop {
                let got = p.read_stream(name, &mut buf).unwrap();
                if got == 0 {
                    break;
                }
                out.extend_from_slice(&buf[..got]);
            }
        }
    }
    outs
}

fn monitor() -> Pipeline {
    build(&flow("pmr446_monitor"))
}

/// The flow with a second output `tap`: the audio of `node` (a demodulator) through a copy of
/// the resampler `out` (the speech-band low-pass to 48 kHz), so that one channel can be
/// measured as the flow outputs it, without the other channels of the sum.
fn monitor_with_tap(node: &str) -> Pipeline {
    let mut doc: Value = serde_json::from_str(&flow("pmr446_monitor")).unwrap();
    doc["nodes"]["tap_out"] = doc["nodes"]["out"].clone();
    let edges = doc["edges"].as_array_mut().unwrap();
    edges.push(json!({ "from": node, "to": "tap_out" }));
    edges.push(json!({ "from": "tap_out", "to": "tap" }));
    doc["outputs"]["tap"] = json!({ "type": "stream", "capacity": 65536, "drop": "error" });
    build(&doc.to_string())
}

fn window(x: &[f32], from_s: f64, to_s: f64, rate: f64) -> Vec<f64> {
    let (a, b) = ((from_s * rate) as usize, (to_s * rate) as usize);
    x[a..b.min(x.len())].iter().map(|v| f64::from(*v)).collect()
}

// ------------------------------------------------------------------ test 7

/// The SINAD of the 1 kHz tone of channel 5 over 0.1 s to 0.5 s with noise-modulated
/// neighbours on channels 4 and 6 at `neighbour_db` relative to channel 5: the audio of
/// `demod5` behind a copy of the flow's resampler `out` (48 kHz, the speech-band low-pass,
/// [`monitor_with_tap`]), and for information the audio of `demod5` itself (25 kS/s, the whole
/// channel up to 12.5 kHz).
fn adjacent_sinad(neighbour_db: f64) -> (f64, f64) {
    let level = LEVEL_DB + neighbour_db;
    let channels = [
        transmitter(4, Audio::Noise, level),
        transmitter(5, Audio::Tone { freq_hz: 1000.0 }, LEVEL_DB),
        transmitter(6, Audio::Noise, level),
    ];
    let bytes = scene_cu8(&channels, 667, (0.5 * RATE) as usize);
    let mut p = monitor_with_tap("demod5");
    let mut doc: Value = serde_json::from_str(&flow("pmr446_monitor")).unwrap();
    let edges = doc["edges"].as_array_mut().unwrap();
    edges.push(json!({ "from": "demod5", "to": "wide" }));
    doc["outputs"]["wide"] = json!({ "type": "stream", "capacity": 65536, "drop": "error" });
    let mut wide = build(&doc.to_string());
    let tap = run(
        &mut p,
        &bytes,
        &mut std::iter::repeat(4096),
        &["audio", "tap"],
    )
    .remove(1);
    let demod5 = run(
        &mut wide,
        &bytes,
        &mut std::iter::repeat(4096),
        &["audio", "wide"],
    )
    .remove(1);
    assert!(tap.len() as f64 >= 0.49 * AUDIO_RATE);
    assert!(demod5.len() as f64 >= 0.49 * CHANNEL_RATE);
    (
        sinad_db(&window(&tap, 0.1, 0.5, AUDIO_RATE), AUDIO_RATE, 1000.0),
        sinad_db(
            &window(&demod5, 0.1, 0.5, CHANNEL_RATE),
            CHANNEL_RATE,
            1000.0,
        ),
    )
}

#[test]
fn channel_5_is_clean_next_to_neighbours_10_db_stronger() {
    let (sinad, wide) = adjacent_sinad(10.0);
    println!("channel 5 with neighbours +10 dB: SINAD {sinad:.2} dB (demod5: {wide:.2} dB)");
    assert!(
        sinad >= MIN_SINAD_STRONG_DB,
        "SINAD {sinad:.2} dB, need {MIN_SINAD_STRONG_DB} dB"
    );
}

#[test]
fn channel_5_is_clean_next_to_neighbours_at_the_same_level() {
    let (sinad, wide) = adjacent_sinad(0.0);
    println!("channel 5 with neighbours at 0 dB: SINAD {sinad:.2} dB (demod5: {wide:.2} dB)");
    assert!(
        sinad >= MIN_SINAD_EQUAL_DB,
        "SINAD {sinad:.2} dB, need {MIN_SINAD_EQUAL_DB} dB"
    );
}

// ------------------------------------------------------------------ test 8

/// The power at each of the eight tones over a window of the flow audio.
fn tone_powers(x: &[f32]) -> [f64; 8] {
    TONES.map(|f| tone_power(x, f, AUDIO_RATE))
}

#[test]
fn every_channel_reaches_the_audio_alone() {
    // One channel at a time: eight scenes of 0.3 s one after another.
    let slot = (0.3 * RATE) as usize;
    let mut bytes = Vec::new();
    for n in 1..=8 {
        let ch = [transmitter(
            n,
            Audio::Tone {
                freq_hz: TONES[n as usize - 1],
            },
            LEVEL_DB,
        )];
        bytes.extend(scene_cu8(&ch, 668 + 1000 * n as u64, slot));
    }
    let audio = run(
        &mut monitor(),
        &bytes,
        &mut std::iter::repeat(4096),
        &["audio"],
    )
    .remove(0);
    for k in 0..8 {
        // The second half of slot k: the gate of the previous channel has closed.
        let from = 0.3 * k as f64 + 0.15;
        let w: Vec<f32> = window(&audio, from, from + 0.15, AUDIO_RATE)
            .iter()
            .map(|v| *v as f32)
            .collect();
        let p = tone_powers(&w);
        let own = p[k];
        let others = p
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != k)
            .map(|(_, v)| *v)
            .fold(0.0, f64::max);
        println!(
            "channel {}: {} Hz at {:.1} dB above the strongest other tone",
            k + 1,
            TONES[k],
            10.0 * (own / others).log10()
        );
        // The other tones include the harmonics of the own tone (300 Hz to 2400 Hz in steps of
        // 300 Hz), so this bounds the distortion as well.
        assert!(own > 100.0 * others, "channel {}: {p:?}", k + 1);
        assert!(own > 1e-3, "channel {}: tone power {own}", k + 1);
    }
}

#[test]
fn every_channel_reaches_the_audio_together() {
    let channels: Vec<SceneChannel> = (1..=8)
        .map(|n| {
            transmitter(
                n,
                Audio::Tone {
                    freq_hz: TONES[n as usize - 1],
                },
                LEVEL_DB,
            )
        })
        .collect();
    let bytes = scene_cu8(&channels, 668, (0.4 * RATE) as usize);
    let audio = run(
        &mut monitor(),
        &bytes,
        &mut std::iter::repeat(4096),
        &["audio"],
    )
    .remove(0);
    let w: Vec<f32> = window(&audio, 0.1, 0.4, AUDIO_RATE)
        .iter()
        .map(|v| *v as f32)
        .collect();
    let p = tone_powers(&w);
    // A frequency between the tones as the floor.
    let floor = tone_power(&w, 3300.0, AUDIO_RATE);
    for (k, own) in p.iter().enumerate() {
        println!(
            "channel {}: {} Hz at {:.1} dB above 3.3 kHz",
            k + 1,
            TONES[k],
            10.0 * (own / floor).log10()
        );
        assert!(*own > 1000.0 * floor, "channel {}: {p:?} {floor}", k + 1);
    }
}

/// Seconds of noise in [`noise_keeps_every_squelch_closed`].
const NOISE_SECONDS: f64 = 2.0;

/// Pushes `seconds` of band noise only (seed 669) and returns the number of non-zero audio
/// samples and the number of audio samples.
fn noise_only(seconds: f64) -> (usize, usize) {
    let scene = ChannelScene {
        spacing_hz: SPACING,
        channels: &[],
        fs: RATE,
        noise_db: NOISE_DB,
        seed: 669,
    };
    let mut r = scene.renderer().unwrap();
    let mut p = monitor();
    let mut x = vec![Complex64::new(0.0, 0.0); 4096];
    let mut bytes = vec![0u8; 2 * 4096];
    let mut buf = vec![0f32; 8192];
    let (mut nonzero, mut total) = (0, 0);
    let mut left = (seconds * RATE) as usize;
    while left > 0 {
        let n = left.min(4096);
        r.render(&mut x[..n]);
        encode(&formats::CU8, &x[..n], &mut bytes[..2 * n]).unwrap();
        p.push("rf", InputData::Raw(&bytes[..2 * n])).unwrap();
        left -= n;
        loop {
            let got = p.read_stream("audio", &mut buf).unwrap();
            if got == 0 {
                break;
            }
            nonzero += buf[..got].iter().filter(|v| **v != 0.0).count();
            total += got;
        }
    }
    (nonzero, total)
}

#[test]
fn noise_keeps_every_squelch_closed() {
    let (nonzero, total) = noise_only(NOISE_SECONDS);
    assert!(total as f64 >= (NOISE_SECONDS - 0.01) * AUDIO_RATE);
    assert_eq!(
        nonzero, 0,
        "{nonzero} of {total} audio samples are not zero"
    );
}

/// The 60 s of the spec (to be confirmed): long in a debug build.
#[test]
#[ignore = "60 s of noise through the flow: run with --profile emulated -- --ignored"]
fn sixty_seconds_of_noise_keep_every_squelch_closed() {
    let (nonzero, total) = noise_only(60.0);
    assert!(total as f64 >= 59.99 * AUDIO_RATE);
    assert_eq!(
        nonzero, 0,
        "{nonzero} of {total} audio samples are not zero"
    );
}

// ------------------------------------------------------------------ test 9

/// One `channelize` node with eight `channel_select` readers (channels 1 to 8, lanes -8 to -1)
/// or eight `ddc` nodes for the same channels, from a `c32` input at 200 kS/s.
fn bank(channelizer: bool) -> String {
    let mut nodes = serde_json::Map::new();
    let mut edges = Vec::new();
    let mut outputs = serde_json::Map::new();
    if channelizer {
        nodes.insert(
            "chan".into(),
            json!({ "kind": "channelize", "params": {
                "channels": 16, "passband": 0.4, "stopband": 0.6, "taps_per_branch": 24 } }),
        );
        edges.push(json!({ "from": "x", "to": "chan" }));
    }
    for n in 1..=8 {
        let name = format!("ch{n}");
        if channelizer {
            nodes.insert(
                name.clone(),
                json!({ "kind": "channel_select", "params": { "channel": lane(n) } }),
            );
            edges.push(json!({ "from": "chan", "to": name }));
        } else {
            // The channel filter of the channelizer as a `ddc`: flat to 0.4 of the spacing,
            // 60 dB from 0.6 (the -6 dB point at 6.25 kHz, the transition of 2.5 kHz centred
            // on it), decimation 8 to 25 kS/s.
            nodes.insert(
                name.clone(),
                json!({ "kind": "ddc", "params": {
                    "frequency_hz": f64::from(lane(n)) * SPACING, "decimation": 8,
                    "cutoff_hz": 6250, "transition_hz": 2500 } }),
            );
            edges.push(json!({ "from": "x", "to": name }));
        }
        edges.push(json!({ "from": name, "to": format!("out{n}") }));
        outputs.insert(format!("out{n}"), json!({ "type": "stream" }));
    }
    json!({
        "sdrk": 1, "name": "bank", "settings": { "max_chunk": 4096 },
        "inputs": { "x": { "type": "c32", "rate_hz": 200000 } },
        "nodes": nodes, "edges": edges, "outputs": outputs
    })
    .to_string()
}

#[test]
fn the_cost_of_one_channelizer_against_eight_ddc() {
    let registry = Registry::standard();
    let opts = BuildOptions::default();
    let chan = registry.validate_json(&bank(true), &opts);
    let ddc = registry.validate_json(&bank(false), &opts);
    assert!(chan.is_ok(), "{}", chan.to_json());
    assert!(ddc.is_ok(), "{}", ddc.to_json());
    let (c, d) = (chan.cost_per_input_sample, ddc.cost_per_input_sample);
    println!("report cost per input sample: channelize + 8 channel_select {c}, 8 ddc {d}");
    // channelize: L + log2(M) 5/2 = 24 + 10, eight channel_select at one per step (1/8 of the
    // input rate each).
    assert!((c - 35.0).abs() < 1e-9, "{c}");
    // ddc: 4 + 4 taps / decimation each, with the taps of the Kaiser estimate (cutoff 6.25 kHz,
    // transition 2.5 kHz at 200 kS/s).
    assert_eq!(ddc.nodes.iter().filter(|n| n.kind == "ddc").count(), 8);
    assert!(d > 20.0 * c, "{d} against {c}");
    // Both banks give eight channels at 25 kS/s.
    for r in [&chan, &ddc] {
        let rates: Vec<_> = r
            .nodes
            .iter()
            .filter(|n| n.name.starts_with("ch") && n.name != "chan")
            .map(|n| n.rate_out)
            .collect();
        assert_eq!(rates.len(), 8);
        assert!(rates.iter().all(|x| *x == rates[0]));
    }
}

// ------------------------------------------------------------------ test 10

#[test]
fn the_flow_validates_without_warnings_and_reports_its_lanes() {
    let report =
        Registry::standard().validate_json(&flow("pmr446_monitor"), &BuildOptions::default());
    assert!(report.is_ok(), "{}", report.to_json());
    assert!(report.warnings.is_empty(), "{}", report.to_json());
    let text = report.to_json();
    assert!(text.contains("\"lanes\": 16"), "{text}");
    let names: Vec<&str> = report.nodes.iter().map(|n| n.name.as_str()).collect();
    for n in 1..=8 {
        for prefix in ["sel", "gate", "demod"] {
            assert!(
                names.contains(&format!("{prefix}{n}").as_str()),
                "{prefix}{n}"
            );
        }
    }
    for n in 1..=7 {
        assert!(names.contains(&format!("mix{n}").as_str()), "mix{n}");
    }
}

#[cfg(feature = "yaml")]
#[test]
fn the_yaml_twin_parses_to_the_same_spec() {
    let read = |ext: &str| {
        std::fs::read_to_string(common::workspace_file(&format!(
            "examples/flows/pmr446_monitor.{ext}"
        )))
        .unwrap()
    };
    let yaml = sdrk_pipeline::PipelineSpec::from_yaml(&read("yaml")).unwrap();
    let json = sdrk_pipeline::PipelineSpec::from_json(&read("json")).unwrap();
    assert_eq!(yaml, json);
    assert_eq!(yaml.to_json() + "\n", read("json"));
}

/// The scene of test 10 (seed 670): channels 2, 5 and 7, `seconds` long.
fn scene_bytes(seconds: f64) -> Vec<u8> {
    let channels = [
        transmitter(2, Audio::Tone { freq_hz: 600.0 }, LEVEL_DB),
        transmitter(5, Audio::Tone { freq_hz: 1500.0 }, LEVEL_DB - 6.0),
        transmitter(7, Audio::Noise, LEVEL_DB),
    ];
    scene_cu8(&channels, 670, (seconds * RATE) as usize)
}

#[test]
fn the_flow_runs_on_a_scene_file() {
    let path = std::env::temp_dir().join(format!("sdrk-pmr446-{}.cu8", std::process::id()));
    std::fs::write(&path, scene_bytes(0.25)).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let audio = run(
        &mut monitor(),
        &bytes,
        &mut std::iter::repeat(4096),
        &["audio"],
    )
    .remove(0);
    // 0.25 s at 48 kHz, less the samples still inside the filters.
    assert!(audio.len() as f64 >= 0.24 * AUDIO_RATE, "{}", audio.len());
    let w: Vec<f32> = window(&audio, 0.08, 0.25, AUDIO_RATE)
        .iter()
        .map(|v| *v as f32)
        .collect();
    let floor = tone_power(&w, 3300.0, AUDIO_RATE);
    for f in [600.0, 1500.0] {
        let p = tone_power(&w, f, AUDIO_RATE);
        assert!(p > 100.0 * floor, "{f} Hz: {p} against {floor}");
    }
}

#[test]
fn the_flow_is_chunk_invariant_with_20_plans() {
    // 0.1 s: the audio of the open channels starts after about 20 ms.
    let bytes = scene_bytes(0.1);
    let whole = run(
        &mut monitor(),
        &bytes,
        &mut std::iter::repeat(4096),
        &["audio"],
    )
    .remove(0);
    assert!(whole.iter().any(|v| *v != 0.0));
    let zeros = |x: &[f32]| x.iter().map(|v| *v == 0.0).collect::<Vec<_>>();
    for k in 0..20 {
        let plan = ChunkPlan::new(671, 1 + 200 * k);
        let got = run(&mut monitor(), &bytes, &mut plan.sizes(), &["audio"]).remove(0);
        assert_eq!(got.len(), whole.len(), "chunk plan {k}");
        // The f32 de-emphasis of the SIMD backend is a tolerance kernel whose bits depend on
        // the split (as in nfm.rs, D187): the runs agree within 90 dB and in the positions of
        // the exact zeros.
        assert_eq!(zeros(&got), zeros(&whole), "chunk plan {k}");
        let (mut signal, mut error) = (0f64, 0f64);
        for (a, b) in whole.iter().zip(&got) {
            signal += f64::from(*a).powi(2);
            error += (f64::from(*a) - f64::from(*b)).powi(2);
        }
        let snr = if error == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (signal / error).log10()
        };
        assert!(snr >= 90.0, "chunk plan {k}: SNR {snr} dB");
    }
}
