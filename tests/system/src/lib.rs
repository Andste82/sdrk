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
