//! Smoke tests of the `fm-transmitter` binary on generated files. The baseband that it writes
//! is demodulated with the pipeline and the receiver example flow.
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

use sdrk_device::{BlockMeta, Rate, RxStream, TxStream, formats};
use sdrk_io::{SigmfReader, WavWriter};
use sdrk_pipeline::{BuildOptions, InputData, Registry};
use sdrk_testkit::meters::sinad_db;

const BIN: &str = env!("CARGO_BIN_EXE_fm-transmitter");

fn run(args: &[&str]) -> Output {
    Command::new(BIN).args(args).output().unwrap()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn p(path: &Path) -> &str {
    path.to_str().unwrap()
}

/// Writes a mono WAV file with a 1 kHz tone of `seconds` at `rate` (f32 or 16-bit PCM).
fn write_tone(path: &Path, rate: u64, seconds: f64, pcm16: bool) -> usize {
    let n = (rate as f64 * seconds) as usize;
    let format = if pcm16 { formats::RS16 } else { formats::RF32 };
    let mut bytes = Vec::new();
    for i in 0..n {
        let v = 0.8 * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / rate as f64).sin();
        if pcm16 {
            bytes.extend_from_slice(&((v * 32767.0).round() as i16).to_le_bytes());
        } else {
            bytes.extend_from_slice(&(v as f32).to_le_bytes());
        }
    }
    let file = std::fs::File::create(path).unwrap();
    let mut w = WavWriter::new(
        std::io::BufWriter::new(file),
        format,
        Rate::hz(rate).unwrap(),
    )
    .unwrap();
    w.write(&bytes, &BlockMeta::default()).unwrap();
    w.finish().unwrap();
    n
}

/// Demodulates cs8 baseband bytes with the receiver example flow and returns the SINAD of the
/// 1 kHz tone over the second half of the audio.
fn sinad_of_baseband(bytes: &[u8]) -> f64 {
    let registry = Registry::standard();
    let json = include_str!("../../flows/fm_receiver.json").replace("\"cu8\"", "\"cs8\"");
    let mut p = registry
        .build_json(&json, &BuildOptions::default())
        .unwrap();
    let mut audio = Vec::new();
    let mut buf = vec![0f32; 4096];
    for piece in bytes.chunks(8192) {
        p.push("rf", InputData::Raw(piece)).unwrap();
        loop {
            let n = p.read_stream("audio", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            audio.extend(buf[..n].iter().map(|&v| f64::from(v)));
        }
    }
    sinad_db(&audio[audio.len() / 2..], 48_000.0, 1000.0)
}

#[test]
fn modulates_a_wav_file_to_a_raw_baseband_file() {
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("audio.wav");
    let out = dir.path().join("out.cs8");
    let n = write_tone(&wav, 48_000, 0.5, false);
    let o = run(&["--input", p(&wav), "--output", p(&out)]);
    assert!(o.status.success(), "{}", stderr(&o));
    let bytes = std::fs::read(&out).unwrap();
    // 50 output samples (two bytes each) per audio sample; the tail of the filters is not flushed
    let samples = bytes.len() / 2;
    assert_eq!(bytes.len() % 2, 0);
    assert!(
        samples <= n * 50 && samples > (n - 100) * 50,
        "{samples} samples for {n} audio samples"
    );
    let log = stderr(&o);
    assert!(log.contains("output: "), "{log}");
    assert!(log.contains("node up"), "{log}");
    assert!(sinad_of_baseband(&bytes) > 35.0);
}

#[test]
fn writes_a_sigmf_recording() {
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("audio.wav");
    let out = dir.path().join("tx.sigmf-meta");
    write_tone(&wav, 48_000, 0.3, true); // 16-bit PCM input
    let o = run(&["--input", p(&wav), "--output", p(&out)]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert!(dir.path().join("tx.sigmf-data").exists());
    let mut r = SigmfReader::open(dir.path().join("tx")).unwrap();
    assert_eq!(r.format(), formats::CS8);
    assert_eq!(r.sample_rate(), Rate::hz(2_400_000).unwrap());
    let mut bytes = Vec::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let (n, meta) = r.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        assert_eq!(meta.start.0 as usize, bytes.len() / 2);
        bytes.extend_from_slice(&buf[..n]);
    }
    assert!(sinad_of_baseband(&bytes) > 35.0);
    // the same modulation to a raw file gives the same bytes
    let raw = dir.path().join("tx.cs8");
    assert!(
        run(&["--input", p(&wav), "--output", p(&raw)])
            .status
            .success()
    );
    assert_eq!(std::fs::read(&raw).unwrap(), bytes);
}

#[test]
fn another_input_rate_replaces_the_rate_of_the_flow() {
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("audio.wav");
    let out = dir.path().join("out.cs8");
    write_tone(&wav, 96_000, 0.2, false);
    let o = run(&["--input", p(&wav), "--output", p(&out)]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert!(stderr(&o).contains("input: ") && stderr(&o).contains("96000 Hz"));
    // 0.2 s at 2.4 MS/s, minus the tail that is not flushed
    let samples = std::fs::metadata(&out).unwrap().len() / 2;
    assert!((450_000..=480_000).contains(&samples), "{samples}");

    // 44.1 kHz would need a polyphase filter with more taps than the resampler allows: the
    // build reports that as a diagnostic.
    let odd = dir.path().join("odd.wav");
    write_tone(&odd, 44_100, 0.05, false);
    let o = run(&["--input", p(&odd), "--output", p(&out)]);
    assert_eq!(o.status.code(), Some(1));
    let report: serde_json::Value = serde_json::from_str(&stderr(&o)).unwrap();
    assert_eq!(report["errors"][0]["code"], "rate_unsupported");
    assert_eq!(report["errors"][0]["path"], "nodes.up.params.max_taps");
}

#[test]
fn set_overrides_a_node_parameter() {
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("audio.wav");
    write_tone(&wav, 48_000, 0.3, false);
    let wide = dir.path().join("wide.cs8");
    let narrow = dir.path().join("narrow.cs8");
    assert!(
        run(&["--input", p(&wav), "--output", p(&wide)])
            .status
            .success()
    );
    let o = run(&[
        "--input",
        p(&wav),
        "--output",
        p(&narrow),
        "--set",
        "mod.deviation_hz=25000",
    ]);
    assert!(o.status.success(), "{}", stderr(&o));
    let (a, b) = (
        std::fs::read(&wide).unwrap(),
        std::fs::read(&narrow).unwrap(),
    );
    assert_eq!(a.len(), b.len());
    assert_ne!(a, b);
}

#[test]
fn an_invalid_flow_exits_with_one_and_diagnostics_json() {
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("audio.wav");
    write_tone(&wav, 48_000, 0.05, false);
    let out = dir.path().join("out.cs8");
    let flow = dir.path().join("bad.json");
    let text = include_str!("../../flows/fm_transmitter.json");
    std::fs::write(
        &flow,
        text.replace("\"deviation_hz\": 75000", "\"deviation_hz\": -5"),
    )
    .unwrap();
    let o = run(&["--input", p(&wav), "--output", p(&out), "--flow", p(&flow)]);
    assert_eq!(o.status.code(), Some(1));
    let report: serde_json::Value = serde_json::from_str(&stderr(&o)).unwrap();
    assert_ne!(report["errors"].as_array().unwrap().len(), 0);
    assert!(!out.exists());
    std::fs::write(&flow, "{ not json").unwrap();
    let o = run(&["--input", p(&wav), "--output", p(&out), "--flow", p(&flow)]);
    assert_eq!(o.status.code(), Some(1));
    let report: serde_json::Value = serde_json::from_str(&stderr(&o)).unwrap();
    assert_eq!(report["errors"][0]["code"], "syntax");
}

#[test]
fn usage_and_run_errors() {
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("audio.wav");
    write_tone(&wav, 48_000, 0.05, false);
    let out = dir.path().join("out.cs8");
    let (w, o) = (p(&wav), p(&out));
    for args in [
        vec![],
        vec!["--input", w],
        vec!["--output", o],
        vec!["--input", w, "--output", o, "--bogus", "1"],
        vec!["--input", w, "--output", o, "--set", "x"],
        vec!["--input", w, "--output", o, "--set", "nonode.x=1"],
        vec!["--input", w, "--output", o, "--rf-output", "nope"],
        vec!["--input", w, "--output", o, "--input-name", "nope"],
    ] {
        let res = run(&args);
        assert_eq!(res.status.code(), Some(2), "{args:?}: {}", stderr(&res));
        assert!(stderr(&res).contains("usage: fm-transmitter"));
    }
    let res = run(&["--input", p(&dir.path().join("absent.wav")), "--output", o]);
    assert_eq!(res.status.code(), Some(1));
    assert!(stderr(&res).contains("cannot open"));
    // not a WAV file
    let junk = dir.path().join("junk.wav");
    std::fs::write(&junk, b"hello world, this is not a wav file").unwrap();
    let res = run(&["--input", p(&junk), "--output", o]);
    assert_eq!(res.status.code(), Some(1));
    assert!(stderr(&res).contains("cannot parse"), "{}", stderr(&res));
    let help = run(&["--help"]);
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("usage: fm-transmitter"));
}

#[test]
fn the_narrow_band_flow_modulates_a_wav_file_for_the_nfm_receiver() {
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("audio.wav");
    let out = dir.path().join("nfm.cs8");
    // The tone of the helper has amplitude 0.8: the pre-emphasis lifts it above the limit of
    // 0.85 and the limiter works; the receiver must still decode the tone cleanly.
    let n = write_tone(&wav, 48_000, 0.5, false);
    let flow = Path::new(env!("CARGO_MANIFEST_DIR")).join("../flows/nfm_transmitter.json");
    let o = run(&["--input", p(&wav), "--output", p(&out), "--flow", p(&flow)]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert!(stderr(&o).contains("node shape"), "{}", stderr(&o));
    let bytes = std::fs::read(&out).unwrap();
    let samples = bytes.len() / 2;
    assert!(
        samples <= n * 50 && samples > (n - 100) * 50,
        "{samples} samples for {n} audio samples"
    );
    // Decode with the NFM receiver flow: cs8 input, channel at +200 kHz.
    let json = include_str!("../../flows/nfm_receiver.json").replace("\"cu8\"", "\"cs8\"");
    let mut p = Registry::standard()
        .build_json(&json, &BuildOptions::default())
        .unwrap();
    p.set_param("tune", "frequency_hz", serde_json::json!(200_000))
        .unwrap();
    let mut audio = Vec::new();
    let mut buf = vec![0f32; 4096];
    for piece in bytes.chunks(8192) {
        p.push("rf", InputData::Raw(piece)).unwrap();
        loop {
            let got = p.read_stream("audio", &mut buf).unwrap();
            if got == 0 {
                break;
            }
            audio.extend(buf[..got].iter().map(|&v| f64::from(v)));
        }
    }
    assert!(sinad_db(&audio[audio.len() / 2..], 48_000.0, 1000.0) > 30.0);
}
