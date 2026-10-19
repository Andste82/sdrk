//! `fm-transmitter`: modulates a WAV file with a pipeline flow (pull mode) and writes the
//! baseband as a raw file or a SigMF recording.
//!
//! ```text
//! fm-transmitter --input audio.wav --output out.cs8
//! fm-transmitter --input audio.wav --output out.sigmf-meta \
//!                [--flow examples/flows/fm_transmitter.json] [--set mod.deviation_hz=50000 ...]
//! ```
//!
//! The summary goes to standard error. Exit code 1 with the diagnostics JSON on standard error
//! for an invalid flow, 2 for a wrong command line.
#![allow(clippy::print_stdout, clippy::print_stderr)]

#[path = "../../common/cli.rs"]
mod cli;

use std::fs::File;
use std::path::Path;
use std::process::ExitCode;

use sdrk_device::{BlockMeta, RxStream, TxStream};
use sdrk_io::{RawWriter, SigmfWriter, WavReader};
use sdrk_pipeline::{
    BuildOptions, InputData, PullError, PullSource, RateHz, Registry, UnderrunPolicy,
};

use cli::{Failure, Options};

const USAGE: &str = "usage: fm-transmitter --input <audio.wav> --output <out.cs8 | out.sigmf-meta>
                      [--flow <flow.json | flow.yaml>] [--set <node.param=value>]...
                      [--input-name <name>] [--rf-output <name>]

Reads a mono WAV file (16-bit PCM or 32-bit float), pulls the modulated baseband from the
raw stream output `rf` (default) of the flow and writes it to a raw file (any extension) or, for
a name ending in .sigmf-meta or .sigmf-data, to a SigMF recording. The sample rate of the
WAV file replaces the rate of the flow input. The output stops with the end of the audio
(underrun policy `short`). Without --flow the example flow examples/flows/fm_transmitter.json
is used.";

const DEFAULT_FLOW: &str = include_str!("../../flows/fm_transmitter.json");
/// Bytes requested from `pull` per call.
const PULL_BYTES: usize = 1 << 16;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(f) => cli::report_failure(&f, USAGE),
    }
}

/// Supplies the WAV samples as `f32` audio.
struct WavSource {
    reader: WavReader<std::io::BufReader<File>>,
    bytes: Vec<u8>,
    samples: Vec<f32>,
    pcm16: bool,
}

impl PullSource for WavSource {
    fn provide(&mut self, _input: &str, max_samples: usize) -> Result<InputData<'_>, PullError> {
        let width = if self.pcm16 { 2 } else { 4 };
        let want = max_samples.min(self.samples.capacity()).max(1);
        let buf = self
            .bytes
            .get_mut(..want * width)
            .ok_or(PullError::Internal("wav buffer"))?;
        let (n, _) = self
            .reader
            .read(buf)
            .map_err(|e| PullError::InvalidState(format!("wav source: {e}")))?;
        self.samples.clear();
        if self.pcm16 {
            for c in buf.get(..n).unwrap_or(&[]).chunks_exact(2) {
                let raw = <[u8; 2]>::try_from(c).unwrap_or_default();
                self.samples
                    .push(f32::from(i16::from_le_bytes(raw)) / 32768.0);
            }
        } else {
            for c in buf.get(..n).unwrap_or(&[]).chunks_exact(4) {
                let raw = <[u8; 4]>::try_from(c).unwrap_or_default();
                self.samples.push(f32::from_le_bytes(raw));
            }
        }
        Ok(InputData::F32(&self.samples))
    }
}

#[allow(
    clippy::large_stack_frames,
    clippy::too_many_lines,
    reason = "clippy counts the temporary of every `?` separately; the real frame is small"
)]
fn run(args: &[String]) -> Result<(), Failure> {
    let o = Options::parse(
        args,
        &["input", "flow", "set", "output", "input-name", "rf-output"],
    )?;
    let input_path = o.required("input")?;
    let output_path = o.required("output")?;
    let registry = Registry::standard();
    let opts = BuildOptions::default();
    let mut spec = cli::load_spec(&registry, o.get("flow"), DEFAULT_FLOW, &opts)?;
    for assignment in o.all("set") {
        cli::apply_set(&mut spec, assignment)?;
    }

    let input_name = cli::input_name(&spec, o.get("input-name"))?;
    let rf_name = o.get("rf-output").unwrap_or("rf").to_string();

    let file = File::open(input_path)
        .map_err(|e| Failure::Run(format!("cannot open `{input_path}`: {e}")))?;
    let chunk = spec.settings.max_chunk;
    let reader = WavReader::new(std::io::BufReader::new(file), false, chunk)?;
    let wav_format = reader.format();
    let wav_rate = reader.sample_rate();
    let pcm16 = wav_format == sdrk_device::formats::RS16;
    let entry = spec
        .inputs
        .get_mut(&input_name)
        .ok_or_else(|| Failure::Usage(format!("the flow has no input `{input_name}`")))?;
    entry.rate_hz = RateHz(wav_rate);

    // A file has an end; `zeros` would pad the output with silence forever.
    let out_spec = spec
        .outputs
        .get_mut(&rf_name)
        .ok_or_else(|| Failure::Usage(format!("the flow has no output `{rf_name}`")))?;
    out_spec.underrun = Some(UnderrunPolicy::Short);
    let rf_format = out_spec
        .format
        .ok_or_else(|| Failure::Run(format!("the output `{rf_name}` has no raw `format`")))?
        .to_format()
        .map_err(Failure::Run)?;

    let mut pipeline = cli::build(&registry, &spec, &opts)?;
    let rf_rate = pipeline.output_rate(&rf_name)?;
    let mut source = WavSource {
        reader,
        bytes: vec![0u8; chunk * 4],
        samples: Vec::with_capacity(chunk),
        pcm16,
    };

    let sigmf = output_path.ends_with(".sigmf-meta") || output_path.ends_with(".sigmf-data");
    let mut sink: Box<dyn TxStream<Error = sdrk_io::Error>>;
    let mut sigmf_writer = None;
    if sigmf {
        sigmf_writer = Some(SigmfWriter::create(
            Path::new(output_path),
            rf_format,
            rf_rate,
        )?);
        sink = Box::new(RawWriter::new(std::io::sink(), rf_format, rf_rate));
    } else {
        let f = File::create(output_path)
            .map_err(|e| Failure::Run(format!("cannot create `{output_path}`: {e}")))?;
        sink = Box::new(RawWriter::new(
            std::io::BufWriter::new(f),
            rf_format,
            rf_rate,
        ));
    }

    let granule = cli::granule_samples(&rf_format);
    let granule_bytes = rf_format.bytes_for_samples(granule)?;
    let mut buf = vec![0u8; PULL_BYTES - PULL_BYTES % granule_bytes];
    let mut total: u64 = 0;
    let mut blocks = 0u64;
    loop {
        let (n, meta) = pipeline.pull(&rf_name, &mut buf, &mut source)?;
        if n == 0 {
            break;
        }
        let bytes = buf.get(..n).unwrap_or(&[]);
        match sigmf_writer.as_mut() {
            Some(w) => w.write(bytes, &meta)?,
            None => sink.write(bytes, &meta)?,
        };
        total += rf_format.samples_in_bytes(n) as u64;
        blocks += 1;
    }
    drop(sink);
    if let Some(w) = sigmf_writer {
        w.finish()?;
    }
    eprintln!("input: {input_path} ({} Hz)", wav_rate.as_f64());
    eprintln!(
        "output: {total} samples in {blocks} blocks at {} Hz ({:.3} s) to {output_path}",
        rf_rate.as_f64(),
        total as f64 / rf_rate.as_f64()
    );
    for s in pipeline.stats() {
        eprintln!(
            "node {:<12} calls {:>6}  in {:>10}  out {:>10}",
            s.name, s.calls, s.samples_in, s.samples_out
        );
    }
    while let Some(e) = pipeline.poll_event() {
        eprintln!(
            "event: {} @ {}: {:?}",
            pipeline.output_name(e.output),
            e.index,
            e.kind
        );
    }
    let _ = BlockMeta::default();
    Ok(())
}
