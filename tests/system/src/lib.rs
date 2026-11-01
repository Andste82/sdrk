//! Helpers of the system tests: example flows, signal generation, the transmit run and the
//! receive run over a loopback channel.
//!
//! The system tests live in `tests/*.rs`; each one is a scenario of phase 3 of the plan
//! (FM over a simulated channel, burst timing, two stations, a long run without drift, exact
//! event propagation, a SigMF round trip through the pipeline).
#![allow(clippy::print_stdout, clippy::print_stderr)]
// The helpers are test support: they may unwrap, index and panic with a message.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_lossless,
    clippy::too_many_lines,
    clippy::missing_panics_doc
)]

use std::path::PathBuf;

use num_complex::Complex64;
use sdrk_device::{BlockMeta, RxStream, formats};
use sdrk_pipeline::{
    BuildOptions, Event, FormatName, FormatSpec, InputData, Pipeline, PipelineSpec, Precision,
    PullError, PullSource, Registry, UnderrunPolicy,
};
use sdrk_testkit::channel::LoopbackRx;
use sdrk_testkit::encode::{decode_reference, encode};
use sdrk_testkit::fm::fm_baseband;

/// Sample rate of the intermediate frequency in the FM flows.
pub const IF_RATE: u64 = 2_400_000;
/// Audio sample rate of the FM flows.
pub const AUDIO_RATE: f64 = 48_000.0;

/// Path of a file below the workspace root.
pub fn workspace_file(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(rel)
}

/// An example flow of `examples/flows`, parsed.
pub fn flow(name: &str) -> PipelineSpec {
    let text = std::fs::read_to_string(workspace_file(&format!("examples/flows/{name}.json")))
        .unwrap_or_else(|e| panic!("cannot read flow {name}: {e}"));
    PipelineSpec::from_json(&text).unwrap_or_else(|e| panic!("cannot parse flow {name}: {e}"))
}

/// Builds a description with the default options.
pub fn build(spec: &PipelineSpec) -> Pipeline {
    Registry::standard()
        .build(spec, &BuildOptions::default())
        .unwrap_or_else(|report| panic!("build failed: {}", report.to_json()))
}

/// Sets the raw format of an input of the description.
pub fn set_input_format(spec: &mut PipelineSpec, input: &str, format: FormatName) {
    let entry = spec.inputs.get_mut(input).expect("input exists");
    entry.format = Some(FormatSpec::Name(format));
    entry.sample_type = None;
}

/// Moves every event of the pipeline into `events`.
pub fn drain_events(p: &mut Pipeline, events: &mut Vec<Event>) {
    while let Some(e) = p.poll_event() {
        events.push(e);
    }
}

/// A pull source of a sine wave: `total` samples of `amp * sin(2 pi freq n / rate)` as `f32`
/// audio, then nothing.
#[derive(Debug)]
pub struct ToneAudio {
    /// Frequency in hertz.
    pub freq: f64,
    /// Sample rate in hertz.
    pub rate: f64,
    /// Amplitude.
    pub amp: f64,
    /// Number of samples to deliver.
    pub total: u64,
    /// Number of samples delivered.
    pub pos: u64,
    buf: Vec<f32>,
}

impl ToneAudio {
    /// A tone source.
    pub fn new(freq: f64, rate: f64, amp: f64, total: u64) -> Self {
        Self {
            freq,
            rate,
            amp,
            total,
            pos: 0,
            buf: Vec::new(),
        }
    }
}

impl PullSource for ToneAudio {
    fn provide(&mut self, _input: &str, max_samples: usize) -> Result<InputData<'_>, PullError> {
        let n = (self.total - self.pos).min(max_samples as u64) as usize;
        self.buf.clear();
        for i in 0..n {
            let t = (self.pos + i as u64) as f64 * self.freq / self.rate;
            let phase = 2.0 * std::f64::consts::PI * (t - t.floor());
            self.buf.push((self.amp * phase.sin()) as f32);
        }
        self.pos += n as u64;
        Ok(InputData::F32(&self.buf))
    }
}

/// Runs the transmit flow in pull mode until the source is exhausted and returns the blocks
/// that `pull` produced. The `underrun` policy of the output is set to `short`.
pub fn transmit(
    spec: &PipelineSpec,
    output: &str,
    source: &mut dyn PullSource,
    block_bytes: usize,
) -> Vec<(Vec<u8>, BlockMeta)> {
    let mut spec = spec.clone();
    spec.outputs
        .get_mut(output)
        .expect("output exists")
        .underrun = Some(UnderrunPolicy::Short);
    let mut p = build(&spec);
    let mut blocks = Vec::new();
    let mut buf = vec![0u8; block_bytes];
    loop {
        let (n, meta) = p.pull(output, &mut buf, source).expect("pull");
        if n == 0 {
            break;
        }
        blocks.push((buf[..n].to_vec(), meta));
    }
    blocks
}

/// Mean power `E|x|^2` of CS8 bytes (full scale 1.0).
pub fn cs8_power(bytes: &[u8]) -> f64 {
    let mut iq = vec![Complex64::new(0.0, 0.0); bytes.len() / 2];
    decode_reference(&formats::CS8, bytes, &mut iq).expect("decode");
    iq.iter().map(Complex64::norm_sqr).sum::<f64>() / iq.len().max(1) as f64
}

/// Reads `samples` samples from a loopback receiver and pushes them with their metadata into
/// the input `input` of `p`; returns all audio of the stream output `output`.
pub fn run_receiver(
    p: &mut Pipeline,
    rx: &mut LoopbackRx,
    input: &str,
    output: &str,
    samples: usize,
    precision: Precision,
) -> Vec<f32> {
    let sample_bytes = 2;
    let mut buf = vec![0u8; 4096 * sample_bytes];
    let mut audio = Vec::new();
    let mut scratch = vec![0f32; 4096];
    let mut got = 0usize;
    let mut events = Vec::new();
    while got < samples {
        let want = (samples - got).min(4096) * sample_bytes;
        let (n, meta) = rx.read(&mut buf[..want]).expect("loopback read");
        p.push_with_meta(input, InputData::Raw(&buf[..n]), &meta)
            .expect("push");
        got += n / sample_bytes;
        read_audio(p, output, precision, &mut scratch, &mut audio);
        drain_events(p, &mut events);
    }
    audio
}

/// Appends everything that is waiting in the real stream output `output` to `audio` (Q15
/// samples are converted to `f32`).
pub fn read_audio(
    p: &mut Pipeline,
    output: &str,
    precision: Precision,
    scratch: &mut [f32],
    audio: &mut Vec<f32>,
) {
    loop {
        let k = match precision {
            Precision::F32 => {
                let k = p.read_stream::<f32>(output, scratch).expect("read");
                audio.extend_from_slice(&scratch[..k]);
                k
            }
            Precision::Q15 => {
                let mut raw = vec![sdrk::Q15::default(); scratch.len()];
                let k = p.read_stream::<sdrk::Q15>(output, &mut raw).expect("read");
                audio.extend(raw[..k].iter().map(|q| f32::from(q.0) / 32768.0));
                k
            }
        };
        if k == 0 {
            break;
        }
    }
}

/// `samples` samples of an FM station (`offset_hz` from the tuner centre, `audio_hz` tone,
/// full deviation 75 kHz) at `amp`, from sample `start`, as complex `f64`.
pub fn fm_station(
    offset_hz: f64,
    audio_hz: f64,
    amp: f64,
    start: u64,
    samples: usize,
) -> Vec<Complex64> {
    let mut iq = vec![Complex64::new(0.0, 0.0); samples];
    fm_baseband(
        &mut iq,
        IF_RATE as f64,
        offset_hz,
        75_000.0,
        audio_hz,
        1.0,
        start,
    );
    for v in &mut iq {
        *v *= amp;
    }
    iq
}

/// Encodes complex samples into the raw bytes of `format`.
pub fn encode_bytes(format: &sdrk_device::SampleFormat, iq: &[Complex64]) -> Vec<u8> {
    let mut bytes = vec![0u8; iq.len() * format.bytes_for_samples(1).unwrap()];
    encode(format, iq, &mut bytes).unwrap();
    bytes
}

// ---------------------------------------------------------------- loop tests of M25

/// An example flow as a JSON value (to change a node or an edge before it is built).
pub fn flow_value(name: &str) -> serde_json::Value {
    let text = std::fs::read_to_string(workspace_file(&format!("examples/flows/{name}.json")))
        .unwrap_or_else(|e| panic!("cannot read flow {name}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("cannot parse flow {name}: {e}"))
}

/// A description from a JSON value.
pub fn spec_from_value(value: &serde_json::Value) -> PipelineSpec {
    PipelineSpec::from_json(&value.to_string())
        .unwrap_or_else(|e| panic!("cannot parse the changed flow: {e}"))
}

/// Sets the precision of a flow; a real `audio` input of the flow gets the matching type (the
/// input type must fit the precision).
pub fn set_precision(value: &mut serde_json::Value, precision: Precision) {
    let (p, t) = match precision {
        Precision::F32 => ("f32", "f32"),
        Precision::Q15 => ("q15", "q15"),
    };
    value["settings"]["precision"] = serde_json::json!(p);
    if value["inputs"]["audio"]["type"].is_string() {
        value["inputs"]["audio"]["type"] = serde_json::json!(t);
    }
}

/// Removes the node `node` from a flow and connects its predecessor to its successor.
pub fn remove_node(value: &mut serde_json::Value, node: &str) {
    let edges = value["edges"].as_array().expect("edges").clone();
    let from = edges
        .iter()
        .find(|e| e["to"] == node)
        .map(|e| e["from"].clone())
        .expect("edge into the node");
    let to = edges
        .iter()
        .find(|e| e["from"] == node)
        .map(|e| e["to"].clone())
        .expect("edge out of the node");
    let mut kept: Vec<serde_json::Value> = edges
        .into_iter()
        .filter(|e| e["from"] != node && e["to"] != node)
        .collect();
    kept.push(serde_json::json!({"from": from, "to": to}));
    value["edges"] = serde_json::Value::Array(kept);
    value["nodes"].as_object_mut().expect("nodes").remove(node);
}

/// A pull source over a vector of audio samples (`f32` or, converted with saturation, Q15).
#[derive(Debug)]
pub struct VecAudio {
    data: Vec<f32>,
    pos: usize,
    q15: bool,
    buf_q15: Vec<sdrk::Q15>,
}

impl VecAudio {
    /// A source that delivers `data` once and then nothing.
    pub fn new(data: Vec<f32>, precision: Precision) -> Self {
        Self {
            data,
            pos: 0,
            q15: precision == Precision::Q15,
            buf_q15: Vec::new(),
        }
    }
}

impl PullSource for VecAudio {
    fn provide(&mut self, _input: &str, max_samples: usize) -> Result<InputData<'_>, PullError> {
        let n = (self.data.len() - self.pos).min(max_samples);
        let part = &self.data[self.pos..self.pos + n];
        self.pos += n;
        if self.q15 {
            self.buf_q15.clear();
            self.buf_q15.extend(part.iter().map(|v| {
                sdrk::Q15(
                    (f64::from(*v) * 32768.0 + 0.5)
                        .floor()
                        .clamp(-32768.0, 32767.0) as i16,
                )
            }));
            Ok(InputData::Q15(&self.buf_q15))
        } else {
            Ok(InputData::F32(part))
        }
    }
}

/// Runs a transmitter flow over `audio` in pull mode and returns the `cs8` bytes of `rf`.
pub fn transmit_bytes(spec: &PipelineSpec, audio: Vec<f32>, precision: Precision) -> Vec<u8> {
    let mut source = VecAudio::new(audio, precision);
    let blocks = transmit(spec, "rf", &mut source, 1 << 16);
    blocks.into_iter().flat_map(|(b, _)| b).collect()
}

/// Pushes `bytes` (without a channel) into a receiver flow whose input `rf` has a raw format,
/// and returns all audio of the stream output `audio` (Q15 converted to `f32`).
pub fn receive_bytes(p: &mut Pipeline, bytes: &[u8], precision: Precision) -> Vec<f32> {
    let mut audio = Vec::new();
    let mut scratch = vec![0f32; 4096];
    // 4096 samples of cs8 per push: the largest chunk of the receiver flows.
    for piece in bytes.chunks(8192) {
        p.push("rf", InputData::Raw(piece)).expect("push");
        read_audio(p, "audio", precision, &mut scratch, &mut audio);
    }
    audio
}

/// The highest absolute instantaneous frequency of `cs8` bytes, in units of `deviation_hz`:
/// a `ddc` from 2.4 MS/s to 240 kS/s (centre `frequency_hz`, low-pass `cutoff_hz`) and
/// `fm_demod` without de-emphasis, i.e. `quad_demod` with the gain `240 kHz / (2 pi
/// deviation_hz)`; the first `skip` output samples (filter settling) are ignored.
#[allow(clippy::large_stack_frames)]
pub fn peak_deviation(
    bytes: &[u8],
    frequency_hz: f64,
    cutoff_hz: f64,
    transition_hz: f64,
    deviation_hz: f64,
    skip: usize,
) -> f64 {
    let spec = serde_json::json!({
        "sdrk": 1,
        "name": "peak-meter",
        "settings": {"precision": "f32", "max_chunk": 4096},
        "inputs": {"rf": {"format": "cs8", "rate_hz": IF_RATE}},
        "nodes": {
            "tune": {"kind": "ddc", "params": {
                "frequency_hz": frequency_hz, "decimation": 10, "cutoff_hz": cutoff_hz,
                "transition_hz": transition_hz, "max_taps": 4096}},
            "demod": {"kind": "fm_demod", "params": {
                "deviation_hz": deviation_hz, "deemphasis_us": 0, "audio_decimation": 1}}
        },
        "edges": [
            {"from": "rf", "to": "tune"},
            {"from": "tune", "to": "demod"},
            {"from": "demod", "to": "audio"}
        ],
        "outputs": {"audio": {"type": "stream", "capacity": 16384, "drop": "error"}}
    });
    let mut p = build(&spec_from_value(&spec));
    let audio = receive_bytes(&mut p, bytes, Precision::F32);
    assert!(audio.len() > skip + 1000, "{} samples", audio.len());
    audio[skip..]
        .iter()
        .map(|v| f64::from(v.abs()))
        .fold(0.0, f64::max)
}

/// `samples` samples of white noise (`Pcg32::new(seed, 1)`), filtered by a fourth-order
/// Butterworth low-pass at `cutoff_hz` (two biquads in `f64`), scaled to the standard
/// deviation `sigma` after the filter.
pub fn band_limited_noise(
    seed: u64,
    samples: usize,
    rate: f64,
    cutoff_hz: f64,
    sigma: f64,
) -> Vec<f32> {
    let mut rng = sdrk_testkit::Pcg32::new(seed, 1);
    let mut x: Vec<f64> = (0..samples).map(|_| rng.normal()).collect();
    let w0 = 2.0 * std::f64::consts::PI * cutoff_hz / rate;
    // Q of the two sections of a fourth-order Butterworth filter: 1 / (2 cos(k pi / 8)), k = 1, 3.
    for k in [1.0, 3.0] {
        let q = 1.0 / (2.0 * (k * std::f64::consts::PI / 8.0).cos());
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        let b0 = (1.0 - w0.cos()) / 2.0 / a0;
        let b1 = (1.0 - w0.cos()) / a0;
        let a1 = -2.0 * w0.cos() / a0;
        let a2 = (1.0 - alpha) / a0;
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        for v in &mut x {
            let y = b0 * *v + b1 * x1 + b0 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = *v;
            y2 = y1;
            y1 = y;
            *v = y;
        }
    }
    let power = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let scale = sigma / power.sqrt();
    x.iter().map(|v| (v * scale) as f32).collect()
}
