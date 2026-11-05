//! The kinds of the FM stereo transmitter (M36): `preemph_limit` with `c32` input (the stereo
//! variant, cost 12) and `fm_stereo_mod` (cost 10): registration, figures, type and rate checks,
//! the runtime parameters, and a pipeline that equals the blocks; the flow
//! `fm_stereo_transmitter` (validates, builds, pull equals push), the loop test E2E-STEREO-TX
//! (`e2e_fm_stereo_loop_f32`: the flow without `duc` into `fm_stereo_demod`) and the limiter test
//! (a full-scale 15 kHz burst keeps the multiplex within 0.99 and the deviation within 75 kHz;
//! the flow runs the limiter after the resampler, D487).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::needless_pass_by_value,
    clippy::large_stack_frames,
    clippy::many_single_char_names,
    clippy::print_stderr
)]

mod common;

use sdrk::{C32, Delay, Rate};
use sdrk_blocks::{
    Block, FmStereoModBlock, FmStereoModConfig, PreemphLimitBlock, PreemphLimitConfig,
};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Pipeline, PullError, PullSource, Registry};
use sdrk_testkit::meters::MultitoneAccumulator;
use sdrk_testkit::{Pcg32, VecStorage};
use serde_json::{Value, json};

use common::{bits, build, flow};

/// A one-node flow on an input of type `ty` at `rate` Hz.
fn single(kind: &str, params: Value, ty: &str, precision: &str, rate: u64) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": precision, "max_chunk": 1024 },
        "inputs": { "in": { "type": ty, "rate_hz": rate } },
        "nodes": { "n": { "kind": kind, "params": params } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    })
}

/// `preemph_limit` (stereo) followed by `fm_stereo_mod` at `rate` Hz.
fn chain(rate: u64, pre: Value, mpx: Value) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
        "inputs": { "in": { "type": "c32", "rate_hz": rate } },
        "nodes": {
            "pre": { "kind": "preemph_limit", "params": pre },
            "mpx": { "kind": "fm_stereo_mod", "params": mpx }
        },
        "edges": [
            { "from": "in", "to": "pre" }, { "from": "pre", "to": "mpx" },
            { "from": "mpx", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    })
}

fn validate(doc: &Value) -> sdrk_pipeline::Report {
    Registry::standard().validate_json(&doc.to_string(), &BuildOptions::default())
}

fn code(doc: &Value) -> Option<DiagCode> {
    validate(doc).errors.first().map(|d| d.code)
}

#[test]
fn the_kinds_are_registered_with_their_kernels() {
    let registry = Registry::standard();
    let f = registry.factory("fm_stereo_mod").unwrap();
    assert_eq!(f.kernels(), ["stereo_encode"]);
    assert_eq!(f.description().kind, "fm_stereo_mod");
    let f = registry.factory("preemph_limit").unwrap();
    assert_eq!(f.kernels(), ["preemph_limit", "preemph_limit_stereo"]);
    assert_eq!(registry.descriptions().count(), 48);
}

#[test]
fn costs_rates_and_delays_follow_the_table() {
    for rate in [48_000u64, 192_000] {
        let report = validate(&single("preemph_limit", json!({}), "c32", "f32", rate));
        assert!(report.is_ok(), "{}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 12.0);
        let node = &report.nodes[0];
        assert_eq!(node.rate_out, Rate::hz(rate).unwrap());
        assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO.reduced());
    }
    for rate in [120_000u64, 192_000, 240_000] {
        let report = validate(&single("fm_stereo_mod", json!({}), "c32", "f32", rate));
        assert!(report.is_ok(), "{}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 10.0);
        let node = &report.nodes[0];
        assert_eq!(node.rate_out, Rate::hz(rate).unwrap());
        assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO.reduced());
    }
    // The real variants keep the cost of M25.
    let report = validate(&single("preemph_limit", json!({}), "f32", "f32", 48_000));
    assert_eq!(report.cost_per_input_sample, 6.0);
}

#[test]
fn types_and_rates_are_checked() {
    // fm_stereo_mod takes c32 only.
    for (ty, precision) in [("q15", "q15"), ("cq15", "q15"), ("f32", "f32")] {
        assert_eq!(
            code(&single("fm_stereo_mod", json!({}), ty, precision, 192_000)),
            Some(DiagCode::TypeMismatch),
            "{ty}"
        );
    }
    // preemph_limit rejects cq15 (backlog).
    assert_eq!(
        code(&single("preemph_limit", json!({}), "cq15", "q15", 48_000)),
        Some(DiagCode::TypeMismatch)
    );
    // The subcarrier reaches 53 kHz: below 120 kHz the rate is unsupported.
    for rate in [96_000u64, 119_999] {
        let report = validate(&single("fm_stereo_mod", json!({}), "c32", "f32", rate));
        let e = report.errors.first().unwrap();
        assert_eq!(e.code, DiagCode::RateUnsupported, "{rate}");
    }
    // The 40 kHz rule of the pre-emphasis holds for stereo as well.
    assert!(!validate(&single("preemph_limit", json!({}), "c32", "f32", 32_000)).is_ok());
    // Ranges.
    for params in [
        json!({ "pilot_hz": 17_999 }),
        json!({ "pilot_hz": 20_001 }),
        json!({ "main": 1.1 }),
        json!({ "main": -0.1 }),
        json!({ "pilot": 0.21 }),
        json!({ "pilot": -0.01 }),
        json!({ "nope": 1 }),
    ] {
        assert!(
            !validate(&single(
                "fm_stereo_mod",
                params.clone(),
                "c32",
                "f32",
                192_000
            ))
            .is_ok(),
            "{params}"
        );
    }
}

/// `pilot` (`fm_stereo_mod`) and `limit` (`preemph_limit`) change while running; the build
/// parameters do not.
#[test]
fn runtime_parameters_can_be_set_on_the_pipeline() {
    let mut p = build(&chain(192_000, json!({}), json!({})).to_string());
    p.set_param("mpx", "pilot", json!(0.0)).unwrap();
    p.set_param("mpx", "pilot", json!(0.2)).unwrap();
    assert!(p.set_param("mpx", "pilot", json!(0.3)).is_err());
    assert!(p.set_param("mpx", "main", json!(0.5)).is_err());
    assert!(p.set_param("mpx", "pilot_hz", json!(19_500)).is_err());
    p.set_param("pre", "limit", json!(0.5)).unwrap();
    assert!(p.set_param("pre", "hold_s", json!(0.1)).is_err());
}

fn stereo_audio(n: usize, seed: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(seed, 3);
    (0..n)
        .map(|i| {
            let t = i as f64 / 192_000.0;
            let l = 0.6 * (std::f64::consts::TAU * 1000.0 * t).sin() + 0.4 * rng.normal();
            let r = 0.4 * (std::f64::consts::TAU * 3000.0 * t).sin();
            C32::new(l as f32, r as f32)
        })
        .collect()
}

/// The pipeline `preemph_limit` -> `fm_stereo_mod` equals the two blocks run directly, bit for
/// bit, with a change of `pilot` and `limit` in the middle (from the next pushed sample).
#[test]
fn the_pipeline_equals_the_blocks() {
    let rate = Rate::hz(192_000).unwrap();
    let input = stereo_audio(6000, 488);
    let at = 3000;

    let mut pre =
        PreemphLimitBlock::<C32, _>::new(&PreemphLimitConfig::default(), rate, &mut VecStorage)
            .unwrap();
    let mut mpx =
        FmStereoModBlock::new(&FmStereoModConfig::default(), rate, &mut VecStorage).unwrap();
    let mut want = Vec::new();
    for (k, part) in [&input[..at], &input[at..]].into_iter().enumerate() {
        if k == 1 {
            pre.set_param("limit", sdrk_blocks::ParamValue::F64(0.7))
                .unwrap();
            mpx.set_param("pilot", sdrk_blocks::ParamValue::F64(0.0))
                .unwrap();
        }
        let mut a = vec![C32::default(); part.len()];
        let mut b = vec![0f32; part.len()];
        pre.process(part, &mut a).unwrap();
        mpx.process(&a, &mut b).unwrap();
        want.extend(b);
    }

    let mut p = build(&chain(192_000, json!({}), json!({})).to_string());
    let mut got = Vec::new();
    let mut buf = vec![0f32; 4096];
    for (k, part) in [&input[..at], &input[at..]].into_iter().enumerate() {
        if k == 1 {
            p.set_param("pre", "limit", json!(0.7)).unwrap();
            p.set_param("mpx", "pilot", json!(0.0)).unwrap();
        }
        for piece in part.chunks(700) {
            p.push("in", InputData::C32(piece)).unwrap();
            loop {
                let n = p.read_stream::<f32>("out", &mut buf).unwrap();
                if n == 0 {
                    break;
                }
                got.extend_from_slice(&buf[..n]);
            }
        }
    }
    assert_eq!(got.len(), want.len());
    assert_eq!(bits(&got), bits(&want));
    // The default multiplex stays within main + pilot for audio within the limit.
    assert!(want[..at].iter().all(|y| y.abs() <= 0.99 + 1e-6));
}

// ---- The flow `fm_stereo_transmitter` ----

const AUDIO_HZ: f64 = 48_000.0;

/// `n` samples of stereo audio at 48 kHz: left 1 kHz 0.5, right 3 kHz 0.3, plus independent
/// Gaussian noise of `sigma` in both channels (seeded).
fn loop_audio(n: usize, sigma: f64, seed: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(seed, 0);
    (0..n)
        .map(|i| {
            let t = i as f64 / AUDIO_HZ;
            let l = 0.5 * (std::f64::consts::TAU * 1000.0 * t).sin() + sigma * rng.normal();
            let r = 0.3 * (std::f64::consts::TAU * 3000.0 * t).sin() + sigma * rng.normal();
            C32::new(l as f32, r as f32)
        })
        .collect()
}

/// Hands out stereo audio in pieces.
struct StereoPullSource {
    data: Vec<C32>,
    pos: usize,
}

impl PullSource for StereoPullSource {
    fn provide(&mut self, input: &str, max_samples: usize) -> Result<InputData<'_>, PullError> {
        assert_eq!(input, "audio");
        assert!(max_samples > 0 && max_samples <= 4800, "{max_samples}");
        let n = max_samples.min(self.data.len() - self.pos);
        let from = self.pos;
        self.pos += n;
        Ok(InputData::C32(&self.data[from..from + n]))
    }
}

/// The push-mode run of the flow: `data` pushed in pieces of `piece`, all bytes of `rf` read.
fn transmitter_push(data: &[C32], piece: usize) -> Vec<u8> {
    let mut p = build(&flow("fm_stereo_transmitter"));
    let mut out = Vec::new();
    let mut buf = vec![0u8; 1 << 16];
    for chunk in data.chunks(piece) {
        p.push("audio", InputData::C32(chunk)).unwrap();
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

#[test]
fn the_stereo_transmitter_flow_validates_and_pull_equals_push() {
    let text = flow("fm_stereo_transmitter");
    let report = Registry::standard().validate_json(&text, &BuildOptions::default());
    assert!(report.is_ok(), "{}", report.to_json());
    assert!(report.warnings.is_empty(), "{}", report.to_json());
    assert!(report.cost_per_input_sample > 0.0);
    let mut kinds: Vec<_> = report.nodes.iter().map(|n| n.kind.as_str()).collect();
    kinds.sort_unstable();
    assert_eq!(
        kinds,
        [
            "convert_out",
            "duc",
            "fm_mod",
            "fm_stereo_mod",
            "preemph_limit",
            "resample"
        ]
    );
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["settings"]["max_chunk"], 4800);
    assert_eq!(v["inputs"]["audio"]["type"], "c32");
    assert_eq!(v["nodes"]["mod"]["params"]["deviation_hz"], 75000);
    assert_eq!(v["nodes"]["up"]["params"]["rate_out_hz"], 192_000);
    let p = build(&text);
    assert_eq!(p.output_rate("rf").unwrap().as_f64(), 1_920_000.0);

    let data = loop_audio(6000, 0.001, 488);
    let pushed = transmitter_push(&data, 1000);
    // 40 output samples of two bytes per audio sample.
    assert!(pushed.len() > 400_000, "{}", pushed.len());
    assert_eq!(transmitter_push(&data, 777)[..300_000], pushed[..300_000]);
    let mut p = build(&text);
    let mut source = StereoPullSource { data, pos: 0 };
    let mut buf = vec![0u8; 300_000];
    let (n, _) = p.pull("rf", &mut buf, &mut source).unwrap();
    assert_eq!(n, buf.len());
    assert_eq!(buf[..], pushed[..300_000]);
    assert!(pushed.iter().any(|b| *b != 0));
}

/// The flow without `duc` (the FM signal at 192 kHz), received by `fm_stereo_demod` (50 us
/// de-emphasis, 192 kHz to 48 kHz); the multiplex signal and the instantaneous frequency of the
/// FM signal (`quad_demod`, 1.0 = 75 kHz) are outputs, too. The front of the chain is that of the
/// flow: `audio`, `up`, `pre` (the limiter after the resampler, D487), `mpx`.
fn loop_pipeline() -> Pipeline {
    let mut v: Value = serde_json::from_str(&flow("fm_stereo_transmitter")).unwrap();
    let nodes = v["nodes"].as_object_mut().unwrap();
    assert!(nodes.remove("duc").is_some());
    nodes.insert(
        "demod".into(),
        json!({
            "kind": "fm_stereo_demod",
            "params": { "audio_decimation": 4, "deemphasis_us": 50 }
        }),
    );
    nodes.insert(
        "fq".into(),
        json!({ "kind": "quad_demod", "params": { "deviation_hz": 75000 } }),
    );
    let front = json!([
        { "from": "audio", "to": "up" },
        { "from": "up", "to": "pre" },
        { "from": "pre", "to": "mpx" }
    ]);
    let edges = v["edges"].as_array().unwrap();
    for e in front.as_array().unwrap() {
        assert!(edges.contains(e), "the flow has the edge {e}");
    }
    v["edges"] = json!([
        front[0],
        front[1],
        front[2],
        { "from": "mpx", "to": "mod" },
        { "from": "mpx", "to": "multiplex" },
        { "from": "mod", "to": "demod" },
        { "from": "mod", "to": "fq" },
        { "from": "demod", "to": "stereo" },
        { "from": "fq", "to": "deviation" }
    ]);
    v["outputs"] = json!({
        "stereo": { "type": "stream" },
        "multiplex": { "type": "stream" },
        "deviation": { "type": "stream" }
    });
    let p = build(&v.to_string());
    assert_eq!(p.output_rate("stereo").unwrap().as_f64(), AUDIO_HZ);
    assert_eq!(p.output_rate("multiplex").unwrap().as_f64(), 192_000.0);
    p
}

/// What the loop pipeline produced.
#[derive(Default)]
struct LoopRun {
    stereo: Vec<C32>,
    multiplex: Vec<f32>,
    deviation: Vec<f32>,
}

/// Pushes `input` in pieces of 4800 samples; `pilot_off_at` sets `pilot` to 0 before the
/// sample with that index.
fn run_loop(input: &[C32], pilot_off_at: Option<usize>) -> LoopRun {
    let mut p = loop_pipeline();
    let mut out = LoopRun::default();
    let mut c = vec![C32::default(); 1 << 14];
    let mut f = vec![0f32; 1 << 16];
    let mut pos = 0;
    while pos < input.len() {
        let mut end = (pos + 4800).min(input.len());
        if let Some(at) = pilot_off_at {
            if pos == at {
                p.set_param("mpx", "pilot", json!(0.0)).unwrap();
            } else if pos < at && at < end {
                end = at;
            }
        }
        p.push("audio", InputData::C32(&input[pos..end])).unwrap();
        pos = end;
        loop {
            let a = p.read_stream::<C32>("stereo", &mut c).unwrap();
            out.stereo.extend_from_slice(&c[..a]);
            let b = p.read_stream::<f32>("multiplex", &mut f).unwrap();
            out.multiplex.extend_from_slice(&f[..b]);
            let d = p.read_stream::<f32>("deviation", &mut f).unwrap();
            out.deviation.extend_from_slice(&f[..d]);
            if a == 0 && b == 0 && d == 0 {
                break;
            }
        }
    }
    out
}

/// Whether the receiver decodes stereo in a window: left and right differ in at least one
/// sample (an unlocked `fm_stereo_demod` gives bit-identical channels, M24).
fn is_stereo(w: &[C32]) -> bool {
    w.iter().any(|v| v.re.to_bits() != v.im.to_bits())
}

/// Amplitudes of 1 kHz and 3 kHz in left and right.
fn tones(audio: &[C32]) -> ([f64; 2], [f64; 2]) {
    let mut l = MultitoneAccumulator::new(AUDIO_HZ, &[1000.0, 3000.0]);
    let mut r = l;
    for v in audio {
        l.push(f64::from(v.re));
        r.push(f64::from(v.im));
    }
    let (al, ar) = (l.amplitudes(), r.amplitudes());
    ([al[0], al[1]], [ar[0], ar[1]])
}

fn db(x: f64) -> f64 {
    20.0 * x.log10()
}

/// E2E-STEREO-TX (`e2e_fm_stereo_loop_f32`): left 1 kHz 0.5, right 3 kHz 0.3, noise 0.001
/// (seed 488) through the transmitter flow (without `duc`) and `fm_stereo_demod`: the pilot is
/// locked after 0.5 s, the last second of the stereo part has each tone within 1 dB of its
/// input level and the other tone at least 30 dB below it, and with `pilot` 0 at runtime the
/// receiver falls back to mono within 0.5 s.
#[test]
fn e2e_fm_stereo_loop_f32() {
    let rate = AUDIO_HZ as usize;
    let stereo_end = 2 * rate;
    let input = loop_audio(3 * rate, 0.001, 488);
    let run = run_loop(&input, Some(stereo_end));
    let audio = &run.stereo;
    assert!(audio.len() + 64 >= input.len(), "{}", audio.len());

    // Lock: every 10 ms window from 0.5 s to the switch decodes stereo.
    let win = rate / 100;
    let windows: Vec<bool> = audio.chunks_exact(win).map(is_stereo).collect();
    let last_mono = windows[..stereo_end / win]
        .iter()
        .rposition(|s| !s)
        .map_or(0.0, |k| (k + 1) as f64 * 0.01);
    eprintln!("e2e_fm_stereo_loop_f32: stereo from {last_mono:.2} s on");
    assert!(
        windows[rate / 2 / win..stereo_end / win].iter().all(|s| *s),
        "pilot not locked after 0.5 s (stereo from {last_mono} s)"
    );

    // Levels and separation in the last second of the stereo part.
    let (l, r) = tones(&audio[stereo_end - rate..stereo_end]);
    let (level_l, level_r) = (db(l[0] / 0.5), db(r[1] / 0.3));
    let (sep_l, sep_r) = (db(l[0] / l[1]), db(r[1] / r[0]));
    eprintln!(
        "e2e_fm_stereo_loop_f32: level L {level_l:.3} dB, R {level_r:.3} dB; \
         separation L {sep_l:.1} dB, R {sep_r:.1} dB"
    );
    assert!(level_l.abs() <= 1.0, "level of L {level_l} dB");
    assert!(level_r.abs() <= 1.0, "level of R {level_r} dB");
    assert!(sep_l >= 30.0, "3 kHz in L only {sep_l} dB below 1 kHz");
    assert!(sep_r >= 30.0, "1 kHz in R only {sep_r} dB below 3 kHz");

    // pilot 0: mono (left and right bit-identical) within 0.5 s.
    let mono_from = audio[..]
        .iter()
        .rposition(|v| v.re.to_bits() != v.im.to_bits())
        .map_or(0, |k| k + 1);
    eprintln!(
        "e2e_fm_stereo_loop_f32: mono {:.3} s after pilot 0",
        mono_from.saturating_sub(stereo_end) as f64 / AUDIO_HZ
    );
    assert!(mono_from > stereo_end, "stereo until the switch");
    assert!(
        mono_from <= stereo_end + rate / 2,
        "not mono within 0.5 s after pilot 0 ({mono_from})"
    );
    assert!(audio.len() > stereo_end + rate / 2 + rate / 4);
}

/// Peak of the multiplex signal and peak deviation in Hz (`quad_demod`) for a full-scale 15 kHz
/// burst in both channels (0.1 s to 0.5 s of 0.6 s; seed 489 draws the phases): `same` gives
/// both channels the same phase (sum only), otherwise the phases are independent (sum and
/// difference).
fn burst_peaks(same: bool) -> (f32, f64) {
    let rate = AUDIO_HZ as usize;
    let mut rng = Pcg32::new(489, 0);
    let pl = std::f64::consts::TAU * rng.next_f64();
    let pr = if same {
        pl
    } else {
        std::f64::consts::TAU * rng.next_f64()
    };
    let (start, end) = (rate / 10, rate / 2);
    let input: Vec<C32> = (0..rate * 6 / 10)
        .map(|i| {
            if (start..end).contains(&i) {
                let w = std::f64::consts::TAU * 15_000.0 * i as f64 / AUDIO_HZ;
                C32::new((w + pl).sin() as f32, (w + pr).sin() as f32)
            } else {
                C32::default()
            }
        })
        .collect();
    let run = run_loop(&input, None);
    assert!(run.multiplex.len() > 4 * rate / 2);
    let peak_mpx = run.multiplex.iter().fold(0f32, |m, x| m.max(x.abs()));
    let peak_dev = run.deviation.iter().fold(0f32, |m, x| m.max(x.abs()));
    let dev_hz = f64::from(peak_dev) * 75_000.0;
    eprintln!(
        "limiter ({}): peak multiplex {peak_mpx:.5}, peak deviation {dev_hz:.1} Hz",
        if same {
            "same phase"
        } else {
            "independent phases"
        }
    );
    (peak_mpx, dev_hz)
}

/// The limiter test of the spec (seed 489) on the flow: a full-scale 15 kHz burst in both
/// channels keeps the multiplex signal within +-0.99 and the deviation of `fm_mod` within 75 kHz
/// (measured by `quad_demod`). The limiter runs after the resampler (D487): at 48 kHz before it
/// the resampler rebuilt the peaks between the limited samples and rang at the gain steps of the
/// attack (1.112 and 83.4 kHz with the same phase in both channels). Both phase cases are
/// measured and printed before the thresholds are checked.
#[test]
fn a_full_scale_burst_stays_within_the_deviation() {
    let peaks = [
        ("same phase", burst_peaks(true)),
        ("independent phases", burst_peaks(false)),
    ];
    let failures: Vec<String> = peaks
        .iter()
        .filter(|(_, (mpx, dev_hz))| *mpx > 0.99 || *dev_hz > 75_000.0)
        .map(|(case, (mpx, dev_hz))| format!("{case}: multiplex peak {mpx}, deviation {dev_hz} Hz"))
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("; "));
}
