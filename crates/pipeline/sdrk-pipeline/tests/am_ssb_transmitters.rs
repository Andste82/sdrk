//! The flows `am_transmitter` and `ssb_transmitter` (M35, Test 6): they validate without
//! warnings, build in `f32` and Q15, a pull of 10^6 bytes gives the same bytes as a push run of
//! the same audio (in two push sizes), and the parameters of the flows are those of the spec.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp
)]

mod common;

use sdrk::Q15;
use sdrk_pipeline::{BuildOptions, InputData, PullError, PullSource, Registry};
use serde_json::{Value, json};

use common::{build, flow};

const FLOWS: [&str; 2] = ["am_transmitter", "ssb_transmitter"];

/// Bytes pulled from the output `rf` (Test 6).
const PULL_BYTES: usize = 1_000_000;

/// 0.25 s of audio at 48 kHz: 12000 samples give 600000 complex samples (1.2 * 10^6 bytes of
/// `cs8`) after the interpolation by 50, more than the pull asks for.
const AUDIO_SAMPLES: usize = 12_000;

/// Two tones (700 Hz 0.4 and 1900 Hz 0.3) inside the voice band of both flows.
fn audio(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f64 / 48_000.0;
            let w = 2.0 * std::f64::consts::PI;
            (0.4 * (w * 700.0 * t).sin() + 0.3 * (w * 1900.0 * t).cos()) as f32
        })
        .collect()
}

struct Source {
    data: Vec<f32>,
    pos: usize,
}

impl PullSource for Source {
    fn provide(&mut self, _input: &str, max_samples: usize) -> Result<InputData<'_>, PullError> {
        let n = max_samples.min(self.data.len() - self.pos);
        let from = self.pos;
        self.pos += n;
        Ok(InputData::F32(&self.data[from..from + n]))
    }
}

fn push_run(name: &str, data: &[f32], chunk: usize) -> Vec<u8> {
    let mut p = build(&flow(name));
    let mut out = Vec::new();
    let mut buf = vec![0u8; 8192];
    for piece in data.chunks(chunk) {
        p.push("audio", InputData::F32(piece)).unwrap();
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

fn pull_run(name: &str, data: Vec<f32>, want: usize) -> Vec<u8> {
    let mut p = build(&flow(name));
    let mut source = Source { data, pos: 0 };
    let mut buf = vec![0u8; want];
    let (n, _) = p.pull("rf", &mut buf, &mut source).unwrap();
    assert_eq!(n, want);
    buf
}

#[test]
fn both_transmitters_validate_without_warnings() {
    for name in FLOWS {
        let report = Registry::standard().validate_json(&flow(name), &BuildOptions::default());
        assert!(report.is_ok(), "{name}: {}", report.to_json());
        assert!(report.warnings.is_empty(), "{name}: {}", report.to_json());
        assert!(report.cost_per_input_sample > 0.0);
        // 48 kHz audio, 2.4 MS/s at the output.
        let duc = report.nodes.iter().find(|n| n.name == "duc").unwrap();
        assert_eq!(duc.rate_out, sdrk::Rate::hz(2_400_000).unwrap());
    }
}

#[test]
fn pull_mode_equals_push_mode() {
    for name in FLOWS {
        let data = audio(AUDIO_SAMPLES);
        let pushed = push_run(name, &data, 1000);
        assert!(pushed.len() >= PULL_BYTES, "{name}: {}", pushed.len());
        // Another push size gives the same bytes.
        let other = push_run(name, &data, 777);
        assert_eq!(other[..PULL_BYTES], pushed[..PULL_BYTES], "{name}");
        // Pull mode gives the same bytes.
        let pulled = pull_run(name, data, PULL_BYTES);
        assert!(
            pulled == pushed[..PULL_BYTES],
            "{name}: pull differs from push"
        );
        // The signal is not silence.
        assert!(pulled.iter().any(|b| *b != 0), "{name}");
    }
}

#[test]
fn the_flows_run_in_q15() {
    for name in FLOWS {
        let mut v: Value = serde_json::from_str(&flow(name)).unwrap();
        v["settings"]["precision"] = json!("q15");
        v["inputs"]["audio"]["type"] = json!("q15");
        let mut p = build(&v.to_string());
        let q: Vec<Q15> = audio(4800)
            .iter()
            .map(|x| Q15((f64::from(*x) * 32768.0) as i16))
            .collect();
        p.push("audio", InputData::Q15(&q)).unwrap();
        let mut buf = vec![0u8; 1 << 16];
        let mut total = 0;
        let mut loud = false;
        loop {
            let n = p.read_raw("rf", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            loud |= buf[..n].iter().any(|b| *b != 0);
            total += n;
        }
        assert!(total > 400_000, "{name}: {total}");
        assert!(loud, "{name}");
    }
}

#[test]
fn the_flows_have_the_parameters_of_the_spec() {
    let v: Value = serde_json::from_str(&flow("am_transmitter")).unwrap();
    assert_eq!(v["nodes"]["lpf"]["kind"], "biquad");
    assert_eq!(v["nodes"]["lpf"]["params"]["response"], "lowpass");
    assert_eq!(v["nodes"]["lpf"]["params"]["cutoff_hz"], 4500);
    assert_eq!(v["nodes"]["mod"]["kind"], "am_mod");
    assert_eq!(v["nodes"]["duc"]["params"]["interpolation"], 50);
    assert_eq!(v["nodes"]["duc"]["params"]["frequency_hz"], 600_000);
    assert_eq!(v["outputs"]["rf"]["format"], "cs8");
    let v: Value = serde_json::from_str(&flow("ssb_transmitter")).unwrap();
    assert_eq!(v["nodes"]["mod"]["kind"], "ssb_mod");
    assert_eq!(v["nodes"]["duc"]["params"]["interpolation"], 50);
    assert_eq!(v["nodes"]["duc"]["params"]["frequency_hz"], 0);
    assert_eq!(v["outputs"]["rf"]["format"], "cs8");
    // The runtime parameters of the running flows.
    let mut p = build(&flow("am_transmitter"));
    p.set_param("mod", "depth", json!(0.5)).unwrap();
    assert!(p.set_param("mod", "carrier", json!(0.3)).is_err());
    let mut p = build(&flow("ssb_transmitter"));
    p.set_param("mod", "sideband", json!("lsb")).unwrap();
    assert!(p.set_param("mod", "low_hz", json!(200)).is_err());
}
