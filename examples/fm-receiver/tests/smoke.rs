//! Smoke tests of the `fm-receiver` binary on generated files.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_lossless,
    clippy::too_many_lines
)]

use std::path::Path;
use std::process::{Command, Output};

use num_complex::Complex64;
use sdrk_device::{BlockMeta, Rate, RxStream, SampleIndex, TxStream, formats};
use sdrk_io::{Annotation, SigmfWriter, WavReader};
use sdrk_testkit::encode::encode;
use sdrk_testkit::fm::fm_baseband;
use sdrk_testkit::meters::sinad_db;

const BIN: &str = env!("CARGO_BIN_EXE_fm-receiver");
const RATE: u64 = 2_400_000;

fn run(args: &[&str]) -> Output {
    Command::new(BIN).args(args).output().unwrap()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn p(path: &Path) -> &str {
    path.to_str().unwrap()
}

/// CU8 bytes of one FM station (`offset_hz` from the centre, 1 kHz tone, amplitude 0.9).
fn station_cu8(offset_hz: f64, samples: usize) -> Vec<u8> {
    station(&formats::CU8, offset_hz, samples)
}

fn station(format: &sdrk_device::SampleFormat, offset_hz: f64, samples: usize) -> Vec<u8> {
    let mut iq = vec![Complex64::new(0.0, 0.0); samples];
    fm_baseband(&mut iq, RATE as f64, offset_hz, 75_000.0, 1000.0, 1.0, 0);
    for v in &mut iq {
        *v *= 0.9;
    }
    let mut bytes = vec![0u8; samples * format.bytes_for_samples(1).unwrap()];
    encode(format, &iq, &mut bytes).unwrap();
    bytes
}

/// The samples of a mono f32 WAV file and its rate.
fn read_wav(path: &Path) -> (Vec<f32>, u64) {
    let file = std::fs::File::open(path).unwrap();
    let mut r = WavReader::new(std::io::BufReader::new(file), false, 4096).unwrap();
    assert_eq!(r.format(), formats::RF32);
    let rate = r.sample_rate().num();
    let mut out = Vec::new();
    let mut buf = vec![0u8; 16384];
    loop {
        let (n, _) = r.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        out.extend(
            buf[..n]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );
    }
    (out, rate)
}

fn sinad_of(audio: &[f32]) -> f64 {
    let tail: Vec<f64> = audio[audio.len() / 2..]
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    sinad_db(&tail, 48_000.0, 1000.0)
}

#[test]
fn demodulates_a_raw_recording() {
    let dir = tempfile::tempdir().unwrap();
    let rec = dir.path().join("rec.cu8");
    let out = dir.path().join("audio.wav");
    std::fs::write(&rec, station_cu8(600_000.0, 720_000)).unwrap();
    let o = run(&[
        "--input",
        p(&rec),
        "--format",
        "cu8",
        "--rate",
        "2400000",
        "--output",
        p(&out),
    ]);
    assert!(o.status.success(), "{}", stderr(&o));
    let (audio, rate) = read_wav(&out);
    assert_eq!(rate, 48_000);
    assert_eq!(audio.len(), 720_000 / 50);
    let sinad = sinad_of(&audio);
    assert!(sinad > 35.0, "SINAD {sinad} dB");
    let log = stderr(&o);
    assert!(log.contains("input samples:  720000"), "{log}");
    assert!(log.contains("audio samples:  14400 at 48000 Hz"), "{log}");
    assert!(log.contains("node tune"), "{log}");
    assert!(log.contains("latency of `audio`"), "{log}");
}

#[test]
fn set_overrides_a_node_parameter() {
    let dir = tempfile::tempdir().unwrap();
    let rec = dir.path().join("rec.cu8");
    std::fs::write(&rec, station_cu8(-200_000.0, 480_000)).unwrap();
    let args = |out: &Path, extra: &'static [&'static str]| {
        let mut a = vec![
            "--input".to_string(),
            p(&rec).to_string(),
            "--format".into(),
            "cu8".into(),
            "--rate".into(),
            "2400000".into(),
            "--output".into(),
            p(out).to_string(),
        ];
        a.extend(extra.iter().map(|s| (*s).to_string()));
        a
    };
    let run_with = |out: &Path, extra: &'static [&'static str]| {
        let a = args(out, extra);
        let a: Vec<&str> = a.iter().map(String::as_str).collect();
        run(&a)
    };
    // default tuning (+600 kHz): the station is not in the channel
    let off = dir.path().join("off.wav");
    assert!(run_with(&off, &[]).status.success());
    // tuned with --set (also checks the `=` form and the JSON value)
    let on = dir.path().join("on.wav");
    let o = run_with(&on, &["--set", "tune.frequency_hz=-200000"]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert!(sinad_of(&read_wav(&on).0) > 35.0);
    assert!(sinad_of(&read_wav(&off).0) < 10.0);
    let on2 = dir.path().join("on2.wav");
    assert!(
        run_with(&on2, &["--set=tune.frequency_hz=-2e5"])
            .status
            .success()
    );
    assert_eq!(read_wav(&on2).0, read_wav(&on).0);
}

#[test]
fn reads_sigmf_with_gaps_and_prints_events() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("rec");
    let total = 480_000usize;
    let bytes = station_cu8(600_000.0, total);
    let mut w = SigmfWriter::create(&base, formats::CU8, Rate::hz(RATE).unwrap()).unwrap();
    // 200k samples, a gap of 10k samples, the rest
    w.write(&bytes[..2 * 200_000], &BlockMeta::default())
        .unwrap();
    let after = BlockMeta {
        start: SampleIndex(210_000),
        ..BlockMeta::default()
    };
    w.write(&bytes[2 * 200_000..], &after).unwrap();
    w.annotate(Annotation {
        sample_start: 100,
        sample_count: 50,
        label: Some("x".into()),
        ..Annotation::default()
    })
    .unwrap();
    w.finish().unwrap();
    let out = dir.path().join("audio.wav");
    let meta = dir.path().join("rec.sigmf-meta");
    let o = run(&["--input", p(&meta), "--output", p(&out)]);
    assert!(o.status.success(), "{}", stderr(&o));
    let log = stderr(&o);
    assert!(log.contains("rec.sigmf-meta (2400000 Hz, rf)"), "{log}");
    assert!(
        log.contains("event: audio @ 4200: Gap { missing: 10000 }"),
        "{log}"
    );
    let (audio, _) = read_wav(&out);
    // the gap carries no samples: 480k samples of data give 9600 audio samples
    assert_eq!(audio.len(), 480_000 / 50);
}

#[test]
fn reads_a_signed_recording_in_the_format_of_the_file() {
    // The format of the flow input (cu8) is replaced by the format of the recording (cs8).
    let dir = tempfile::tempdir().unwrap();
    let rec = dir.path().join("rec.cs8");
    let out = dir.path().join("audio.wav");
    std::fs::write(&rec, station(&formats::CS8, 600_000.0, 480_000)).unwrap();
    let o = run(&[
        "--input",
        p(&rec),
        "--format",
        "cs8",
        "--rate",
        "2400000",
        "--output",
        p(&out),
    ]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert!(sinad_of(&read_wav(&out).0) > 35.0);
}

#[test]
fn accepts_a_yaml_flow() {
    let dir = tempfile::tempdir().unwrap();
    let rec = dir.path().join("rec.cu8");
    let out = dir.path().join("audio.wav");
    std::fs::write(&rec, station_cu8(600_000.0, 480_000)).unwrap();
    let flow = concat!(env!("CARGO_MANIFEST_DIR"), "/../flows/fm_receiver.yaml");
    let o = run(&[
        "--input",
        p(&rec),
        "--format",
        "cu8",
        "--rate",
        "2400000",
        "--flow",
        flow,
        "--output",
        p(&out),
    ]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert!(sinad_of(&read_wav(&out).0) > 35.0);
}

#[test]
fn an_invalid_flow_exits_with_one_and_diagnostics_json() {
    let dir = tempfile::tempdir().unwrap();
    let rec = dir.path().join("rec.cu8");
    std::fs::write(&rec, [128u8; 1000]).unwrap();
    let out = dir.path().join("audio.wav");
    let flow = dir.path().join("bad.json");
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../flows/fm_receiver.json"
    ))
    .unwrap();
    std::fs::write(&flow, text.replace("\"fm_demod\"", "\"fm_demodulator\"")).unwrap();
    let base = [
        "--input",
        p(&rec),
        "--format",
        "cu8",
        "--rate",
        "2400000",
        "--output",
        p(&out),
    ];
    let mut args = base.to_vec();
    args.extend(["--flow", p(&flow)]);
    let o = run(&args);
    assert_eq!(o.status.code(), Some(1));
    let report: serde_json::Value = serde_json::from_str(&stderr(&o)).unwrap();
    let errors = report["errors"].as_array().unwrap();
    assert_eq!(errors[0]["code"], "unknown_kind");
    assert_eq!(errors[0]["path"], "nodes.demod.kind");
    assert!(!out.exists());

    // a syntax error
    std::fs::write(&flow, "{ not json").unwrap();
    let o = run(&args);
    assert_eq!(o.status.code(), Some(1));
    let report: serde_json::Value = serde_json::from_str(&stderr(&o)).unwrap();
    assert_eq!(report["errors"][0]["code"], "syntax");

    // a --set that makes the flow invalid is found by the build
    let mut args = base.to_vec();
    args.extend(["--set", "tune.decimation=0"]);
    let o = run(&args);
    assert_eq!(o.status.code(), Some(1));
    let report: serde_json::Value = serde_json::from_str(&stderr(&o)).unwrap();
    assert_ne!(report["errors"].as_array().unwrap().len(), 0);
}

#[test]
fn usage_errors_exit_with_two() {
    let dir = tempfile::tempdir().unwrap();
    let rec = dir.path().join("rec.cu8");
    std::fs::write(&rec, [128u8; 100]).unwrap();
    let out = dir.path().join("audio.wav");
    let (r, o) = (p(&rec), p(&out));
    let cases: Vec<Vec<&str>> = vec![
        vec![],
        vec!["--input", r],
        vec!["--output", o],
        vec!["--input", r, "--output", o], // raw without format and rate
        vec!["--input", r, "--format", "cu8", "--output", o], // no rate
        vec![
            "--input", r, "--format", "nope", "--rate", "1", "--output", o,
        ],
        vec![
            "--input", r, "--format", "cu8", "--rate", "fast", "--output", o,
        ],
        vec![
            "--input", r, "--format", "cu8", "--rate", "2400000", "--output", o, "--bogus", "1",
        ],
        vec![
            "--input", r, "--format", "cu8", "--rate", "2400000", "--output", o, "stray",
        ],
        vec![
            "--input", r, "--format", "cu8", "--rate", "2400000", "--output", o, "--set", "nodot",
        ],
        vec![
            "--input",
            r,
            "--format",
            "cu8",
            "--rate",
            "2400000",
            "--output",
            o,
            "--set",
            "nonode.x=1",
        ],
        vec![
            "--input",
            r,
            "--format",
            "cu8",
            "--rate",
            "2400000",
            "--output",
            o,
            "--input-name",
            "zz",
        ],
        vec![
            "--input", r, "--format", "cu8", "--rate", "2400000", "--output",
        ],
    ];
    for args in cases {
        let res = run(&args);
        assert_eq!(res.status.code(), Some(2), "{args:?}: {}", stderr(&res));
        assert!(stderr(&res).contains("usage: fm-receiver"), "{args:?}");
    }
    let help = run(&["--help"]);
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("usage: fm-receiver"));
}

#[test]
fn run_errors_exit_with_one() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("audio.wav");
    let o = run(&[
        "--input",
        p(&dir.path().join("absent.cu8")),
        "--format",
        "cu8",
        "--rate",
        "2400000",
        "--output",
        p(&out),
    ]);
    assert_eq!(o.status.code(), Some(1));
    assert!(stderr(&o).contains("cannot open"));
    let o = run(&[
        "--input",
        p(&dir.path().join("absent")),
        "--output",
        "x.wav",
        "--flow",
        "/no/such/flow.json",
        "--format",
        "cu8",
        "--rate",
        "1",
    ]);
    assert_eq!(o.status.code(), Some(1));
    assert!(stderr(&o).contains("cannot read"));
    let o = run(&[
        "--input",
        p(&dir.path().join("none.sigmf-meta")),
        "--output",
        p(&out),
    ]);
    assert_eq!(o.status.code(), Some(1));
    assert!(stderr(&o).contains("i/o error"), "{}", stderr(&o));
}
