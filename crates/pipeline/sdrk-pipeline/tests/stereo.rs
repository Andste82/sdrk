//! The kind `fm_stereo_demod` and the example flow `fm_stereo_receiver`: registration, the cost
//! formula, validation, the run on the E2E-STEREO signal (equal to the hand-wired blocks bit for
//! bit), the runtime parameter `stereo`, a tunable resampler behind the decoder, and the YAML
//! twin.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::float_cmp,
    clippy::needless_pass_by_value
)]

mod common;

use num_complex::Complex64;
use sdrk::{C32, CQ15, Delay, Rate};
use sdrk_blocks::{
    Block, ConvertBlock, ConvertConfig, DdcBlock, DdcConfig, FmStereoDemodBlock,
    FmStereoDemodConfig,
};
use sdrk_device::formats;
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Pipeline, Registry, VecStorage};
use sdrk_testkit::e2e_stereo::{StereoSignal, StereoSource};
use sdrk_testkit::meters::MultitoneAccumulator;
use serde_json::{Value, json};

use common::{build, flow};

const CHUNK: usize = 500;

/// A one-node flow on a complex input of type `ty` at `rate` Hz.
fn single(ty: &str, precision: &str, rate: u64, params: Value) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": precision, "max_chunk": 1024 },
        "inputs": { "in": { "type": ty, "rate_hz": rate } },
        "nodes": { "n": { "kind": "fm_stereo_demod", "params": params } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    })
}

fn validate(doc: &Value) -> sdrk_pipeline::Report {
    Registry::standard().validate_json(&doc.to_string(), &BuildOptions::default())
}

#[test]
fn the_kind_is_registered_with_its_kernels() {
    let registry = Registry::standard();
    let f = registry.factory("fm_stereo_demod").unwrap();
    assert_eq!(
        f.kernels(),
        [
            "quad_demod",
            "biquad_real",
            "stereo_pll",
            "fir_decimate",
            "stereo_matrix",
            "biquad"
        ]
    );
    assert_eq!(f.description().kind, "fm_stereo_demod");
    assert_eq!(registry.descriptions().count(), 20);
}

#[test]
fn the_plan_follows_the_cost_formula() {
    for (ty, precision) in [("c32", "f32"), ("cq15", "q15")] {
        for decimation in [1, 5, 10, 20] {
            let doc = single(
                ty,
                precision,
                480_000,
                json!({ "audio_decimation": decimation }),
            );
            let report = validate(&doc);
            if decimation == 20 {
                // 24 kHz output is below 2 x 15 kHz + the transition.
                assert!(!report.is_ok());
                continue;
            }
            assert!(report.is_ok(), "{ty} {decimation}: {}", report.to_json());
            let cfg = FmStereoDemodConfig {
                audio_decimation: decimation,
                ..FmStereoDemodConfig::default()
            };
            let taps = cfg.audio_taps(Rate::hz(480_000).unwrap()).unwrap();
            assert_eq!(
                report.cost_per_input_sample,
                25.0 + (2 * taps + 12) as f64 / decimation as f64,
                "{ty} {decimation}"
            );
            let node = &report.nodes[0];
            assert_eq!(node.rate_in, Rate::hz(480_000).unwrap());
            assert_eq!(
                node.rate_out,
                Rate::hz(480_000 / decimation as u64).unwrap()
            );
            assert_eq!(
                node.delay_samples_in.reduced(),
                Delay::new(taps as u64, 2).unwrap().reduced()
            );
            assert!(report.memory_bytes > 0);
        }
    }
}

#[test]
fn the_input_rate_and_the_parameters_are_checked() {
    // Below 120 kHz the rate is unsupported.
    let report = validate(&single(
        "c32",
        "f32",
        96_000,
        json!({ "audio_decimation": 2 }),
    ));
    assert!(!report.is_ok());
    assert!(
        report
            .errors
            .iter()
            .any(|d| d.code == DiagCode::RateUnsupported),
        "{}",
        report.to_json()
    );
    // A real input is a type error.
    assert!(!validate(&single("f32", "f32", 480_000, json!({}))).is_ok());
    // Range checks of the parameters.
    for bad in [
        json!({ "audio_cutoff_hz": 16001 }),
        json!({ "audio_decimation": 0 }),
        json!({ "audio_decimation": 65 }),
        json!({ "pll_natural_hz": 191 }),
        json!({ "pll_natural_hz": 0.5 }),
        json!({ "stereo": "left" }),
        json!({ "nope": 1 }),
    ] {
        assert!(
            !validate(&single("c32", "f32", 480_000, bad.clone())).is_ok(),
            "{bad}"
        );
    }
    // The pilot level of the Q15 loop is 0.1 * 2 * deviation / fs: a deviation outside
    // (0, fs / 0.2] is a validation diagnostic, not a failure of the build.
    let report = validate(&single(
        "cq15",
        "q15",
        480_000,
        json!({ "deviation_hz": 3.0e6 }),
    ));
    assert!(!report.is_ok(), "{}", report.to_json());
    assert!(
        validate(&single(
            "c32",
            "f32",
            480_000,
            json!({ "deviation_hz": 3.0e6 })
        ))
        .is_ok()
    );
    // The Choice is given by name.
    for good in [
        json!({}),
        json!({ "stereo": "auto" }),
        json!({ "stereo": "mono", "deemphasis_us": 75, "pll_natural_hz": 30 }),
    ] {
        let r = validate(&single("c32", "f32", 480_000, good.clone()));
        assert!(r.is_ok(), "{good}: {}", r.to_json());
    }
}

/// Pushes the E2E-STEREO signal (`samples` input samples) in pieces of `piece` samples and
/// collects the stereo audio.
fn run(p: &mut Pipeline, signal: StereoSignal, samples: usize, piece: usize) -> Vec<C32> {
    let mut source = StereoSource::new(signal, signal.seed);
    let mut scratch = vec![Complex64::new(0.0, 0.0); CHUNK];
    let mut bytes = vec![0u8; 2 * CHUNK];
    let mut all = Vec::with_capacity(samples * 2);
    for chunk in 0..samples / CHUNK {
        source
            .fill(&mut scratch, (chunk * CHUNK) as u64, &mut bytes)
            .unwrap();
        all.extend_from_slice(&bytes);
    }
    let mut out = Vec::new();
    let mut buf = vec![C32::default(); 4096];
    for part in all.chunks(2 * piece) {
        p.push("rf", InputData::Raw(part)).unwrap();
        loop {
            let got = p.read_stream("audio", &mut buf).unwrap();
            if got == 0 {
                break;
            }
            out.extend_from_slice(&buf[..got]);
        }
    }
    out
}

fn stereo_flow() -> Value {
    serde_json::from_str(&flow("fm_stereo_receiver")).unwrap()
}

/// The same chain wired by hand from the blocks.
fn hand_wired<T>(signal: StereoSignal, samples: usize) -> Vec<T>
where
    T: sdrk_blocks::BlockSample + sdrk_blocks::BiquadElem,
    T::Real: sdrk_blocks::BiquadElem,
    sdrk::Sdrk: sdrk_core::kernels::convert::Convert<T>
        + sdrk_core::kernels::fused::Ddc<T>
        + sdrk_core::kernels::demod::QuadDemod<T>
        + sdrk_core::kernels::biquad::BiquadReal<T::Real>
        + sdrk_core::kernels::stereo::StereoPll<T::Real>
        + sdrk_core::kernels::decim::FirDecimate<T>
        + sdrk_core::kernels::stereo::StereoMatrix<T>
        + sdrk_core::kernels::biquad::Biquad<T>,
{
    let if_rate = Rate::hz(2_400_000).unwrap();
    let ddc_cfg = DdcConfig {
        frequency_hz: 600_000.0,
        decimation: 5,
        cutoff_hz: Some(130_000.0),
        transition_hz: Some(30_000.0),
        max_chunk: 4096,
        ..DdcConfig::default()
    };
    let mut convert = ConvertBlock::<T, _>::new(
        &ConvertConfig {
            format: formats::CU8,
        },
        if_rate,
        &mut VecStorage,
    )
    .unwrap();
    let mut ddc = DdcBlock::<T, _>::new(&ddc_cfg, if_rate, &mut VecStorage).unwrap();
    let mut stereo = FmStereoDemodBlock::<T, _>::new(
        &FmStereoDemodConfig::default(),
        Rate::hz(480_000).unwrap(),
        &mut VecStorage,
    )
    .unwrap();
    let mut source = StereoSource::new(signal, signal.seed);
    let mut scratch = vec![Complex64::new(0.0, 0.0); CHUNK];
    let mut bytes = vec![0u8; 2 * CHUNK];
    let mut iq = vec![T::default(); CHUNK];
    let mut low = vec![T::default(); CHUNK];
    let mut out = vec![T::default(); CHUNK];
    let mut all = Vec::new();
    for chunk in 0..samples / CHUNK {
        source
            .fill(&mut scratch, (chunk * CHUNK) as u64, &mut bytes)
            .unwrap();
        convert.process(&bytes, &mut iq).unwrap();
        let q = ddc.process(&iq, &mut low).unwrap();
        let mut rest = &low[..q.produced];
        while !rest.is_empty() {
            let r = stereo.process(rest, &mut out).unwrap();
            all.extend_from_slice(&out[..r.produced]);
            rest = &rest[r.consumed..];
        }
    }
    all
}

fn separation(audio: &[C32], skip: usize, rate: f64) -> (f64, f64, f64, f64) {
    let mut l = MultitoneAccumulator::new(rate, &[1000.0, 3000.0]);
    let mut r = l;
    for v in &audio[skip..] {
        l.push(f64::from(v.re));
        r.push(f64::from(v.im));
    }
    let (al, ar) = (l.amplitudes(), r.amplitudes());
    (
        20.0 * (al[0] / ar[0]).log10(),
        20.0 * (ar[1] / al[1]).log10(),
        l.sinad_db(0),
        r.sinad_db(1),
    )
}

#[test]
fn the_flow_receives_the_stereo_signal_like_the_hand_wired_blocks() {
    let samples = 600_000; // 0.25 s; the full thresholds are measured by the testkit
    let doc = stereo_flow();
    let report = validate(&doc);
    assert!(report.is_ok(), "{}", report.to_json());
    let mut names: Vec<_> = report.nodes.iter().map(|n| n.kind.as_str()).collect();
    names.sort_unstable();
    assert_eq!(names, ["convert", "ddc", "fm_stereo_demod", "spectrum"]);
    let mut p = build(&doc.to_string());
    assert_eq!(p.output_rate("audio").unwrap().as_f64(), 48_000.0);
    let got = run(&mut p, StereoSignal::STEREO, samples, 4096);
    assert_eq!(got.len(), samples / 50);
    let want = hand_wired::<C32>(StereoSignal::STEREO, samples);
    assert_eq!(got.len(), want.len());
    assert!(
        got.iter()
            .zip(&want)
            .all(|(a, b)| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()),
        "the pipeline differs from the hand-wired blocks"
    );
    // The same thresholds as E2E-STEREO, over the second half of the 0.25 s.
    let (sl, sr, nl, nr) = separation(&got, got.len() / 3, 48_000.0);
    assert!(sl >= 40.0 && sr >= 40.0, "separation {sl} dB, {sr} dB");
    assert!(nl >= 55.0 && nr >= 55.0, "SINAD {nl} dB, {nr} dB");
}

#[test]
fn the_q15_flow_equals_the_hand_wired_blocks() {
    let samples = 200_000;
    let mut doc = stereo_flow();
    doc["settings"]["precision"] = json!("q15");
    let mut p = build(&doc.to_string());
    let mut source = StereoSource::new(StereoSignal::STEREO, 2400);
    let mut scratch = vec![Complex64::new(0.0, 0.0); CHUNK];
    let mut bytes = vec![0u8; 2 * CHUNK];
    let mut got = Vec::new();
    let mut buf = vec![CQ15::default(); 4096];
    for chunk in 0..samples / CHUNK {
        source
            .fill(&mut scratch, (chunk * CHUNK) as u64, &mut bytes)
            .unwrap();
        p.push("rf", InputData::Raw(&bytes)).unwrap();
        loop {
            let n = p.read_stream("audio", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            got.extend_from_slice(&buf[..n]);
        }
    }
    let want = hand_wired::<CQ15>(StereoSignal::STEREO, samples);
    assert_eq!(got, want);
}

#[test]
fn the_output_does_not_depend_on_the_push_size() {
    let samples = 60_000;
    let whole = run(
        &mut build(&stereo_flow().to_string()),
        StereoSignal::STEREO,
        samples,
        2048,
    );
    for piece in [500, 1000, 1500] {
        let got = run(
            &mut build(&stereo_flow().to_string()),
            StereoSignal::STEREO,
            samples,
            piece,
        );
        assert_eq!(got, whole, "piece {piece}");
    }
}

#[test]
fn the_stereo_parameter_can_change_while_the_flow_runs() {
    let samples = 240_000;
    let mut p = build(&stereo_flow().to_string());
    let mut source = StereoSource::new(StereoSignal::STEREO, 2400);
    let mut scratch = vec![Complex64::new(0.0, 0.0); CHUNK];
    let mut bytes = vec![0u8; 2 * CHUNK];
    let mut buf = vec![C32::default(); 4096];
    let mut first = Vec::new();
    let mut second = Vec::new();
    for chunk in 0..samples / CHUNK {
        if chunk == samples / CHUNK / 2 {
            p.set_param("demod", "stereo", json!("mono")).unwrap();
        }
        source
            .fill(&mut scratch, (chunk * CHUNK) as u64, &mut bytes)
            .unwrap();
        p.push("rf", InputData::Raw(&bytes)).unwrap();
        loop {
            let n = p.read_stream("audio", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            if chunk < samples / CHUNK / 2 {
                first.extend_from_slice(&buf[..n]);
            } else {
                second.extend_from_slice(&buf[..n]);
            }
        }
    }
    assert!(first[2000..].iter().any(|v| v.re != v.im));
    // 5 ms after the switch (240 samples) left and right are bit-identical.
    assert!(
        second[240..]
            .iter()
            .all(|v| v.re.to_bits() == v.im.to_bits())
    );
    // Build parameters stay read-only; a wrong value is rejected.
    assert!(p.set_param("demod", "audio_decimation", json!(5)).is_err());
    assert!(p.set_param("demod", "stereo", json!("both")).is_err());
}

#[test]
fn a_tunable_resampler_works_on_the_stereo_stream() {
    let mut doc = stereo_flow();
    doc["nodes"]["rs"] = json!({
        "kind": "resample",
        "params": { "rate_out_hz": 44100, "tunable": true }
    });
    doc["edges"] = json!([
        { "from": "rf", "to": "tune" },
        { "from": "rf", "to": "fft" },
        { "from": "tune", "to": "demod" },
        { "from": "demod", "to": "rs" },
        { "from": "rs", "to": "audio" },
        { "from": "fft", "to": "spectrum" }
    ]);
    let report = validate(&doc);
    assert!(report.is_ok(), "{}", report.to_json());
    let mut p = build(&doc.to_string());
    assert_eq!(p.output_rate("audio").unwrap().as_f64(), 44_100.0);
    let samples = 600_000;
    let got = run(&mut p, StereoSignal::STEREO, samples, 4096);
    assert!(got.len().abs_diff(samples / 50 * 441 / 480) <= 2);
    // Both channels keep their tones: the resampler treats re and im alike.
    let (sl, sr, _, _) = separation(&got, got.len() / 3, 44_100.0);
    assert!(sl >= 35.0 && sr >= 35.0, "separation {sl} dB, {sr} dB");
}

#[cfg(feature = "yaml")]
#[test]
fn the_yaml_twin_describes_the_same_flow() {
    let yaml = std::fs::read_to_string(common::workspace_file(
        "examples/flows/fm_stereo_receiver.yaml",
    ))
    .unwrap();
    let from_yaml = sdrk_pipeline::PipelineSpec::from_yaml(&yaml).unwrap();
    let from_json = sdrk_pipeline::PipelineSpec::from_json(&flow("fm_stereo_receiver")).unwrap();
    assert_eq!(from_yaml, from_json);
}
