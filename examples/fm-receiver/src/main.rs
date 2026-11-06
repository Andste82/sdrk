//! `fm-receiver`: demodulates an IQ recording with a pipeline flow and writes the audio as a
//! 48 kHz WAV file.
//!
//! ```text
//! fm-receiver --input rec.sigmf-meta --output audio.wav
//! fm-receiver --input rec.cu8 --format cu8 --rate 2400000 --output audio.wav \
//!             [--flow examples/flows/fm_receiver.json] [--set tune.frequency_hz=600000 ...]
//! ```
//!
//! The audio output decides the file: a real stream (`f32`, as in `fm_receiver.json`) gives a
//! mono WAV file, a complex stream (`C32`, stereo audio with the left channel in `re` and the
//! right channel in `im`, as in `fm_stereo_receiver.json`) a two-channel WAV file.
//!
//! RDS (M41): when the flow has a packets output `rds` (as in `fm_rds_receiver.json`; another
//! name with `--rds-output`), its groups go through `sdrk_proto::rds::RdsState`, and the
//! programme service name and the radiotext are printed to standard error once they are
//! complete and whenever they change:
//!
//! ```text
//! fm-receiver --input rds.cs16 --format cs16 --rate 480000 \
//!             --flow examples/flows/fm_rds_receiver.json --output audio.wav
//! rds: PI 0xD123 PTY 10 PS "SDRK TST"
//! rds: RT "Hello from sdrk!"
//! ```
//!
//! The report summary and the events go to standard error. Exit code 1 with the diagnostics
//! JSON on standard error for an invalid flow, 2 for a wrong command line.
#![allow(clippy::print_stdout, clippy::print_stderr)]

#[path = "../../common/cli.rs"]
mod cli;

use std::fs::File;
use std::path::Path;
use std::process::ExitCode;

use sdrk::C32;
use sdrk_blocks::PortType;
use sdrk_device::{BlockMeta, Rate, RxStream, SampleIndex, formats};
use sdrk_io::{RawReader, SigmfReader, WavWriter, sigmf_paths};
use sdrk_pipeline::{
    BuildOptions, EventKind, InputData, Packet, PacketFlags, Pipeline, PortKind, RateHz, Registry,
};
use sdrk_proto::rds::{Group, RdsState};

use cli::{Failure, Options};

const USAGE: &str = "usage: fm-receiver --input <rec.sigmf-meta | rec.cu8 ...> --output <audio.wav>
                   [--format <name> --rate <hz>]   (raw input only)
                   [--flow <flow.json | flow.yaml>] [--set <node.param=value>]...
                   [--input-name <name>] [--audio-output <name>] [--rds-output <name>]

Reads an IQ recording (SigMF, or a raw file with --format and --rate), pushes it through the
flow in max_chunk pieces and writes the stream output `audio` (default) as a mono f32 WAV file.
Without --flow the example flow examples/flows/fm_receiver.json is used. When the flow has the
packets output `rds` (default name; examples/flows/fm_rds_receiver.json), the RDS programme
service name and radiotext are printed to standard error.";

const DEFAULT_FLOW: &str = include_str!("../../flows/fm_receiver.json");
/// Events printed to standard error before the rest is only counted.
const MAX_PRINTED_EVENTS: usize = 200;

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

#[allow(
    clippy::large_stack_frames,
    clippy::too_many_lines,
    reason = "clippy counts the temporary of every `?` separately; the real frame is small"
)]
fn run(args: &[String]) -> Result<(), Failure> {
    let o = Options::parse(
        args,
        &[
            "input",
            "format",
            "rate",
            "flow",
            "set",
            "output",
            "input-name",
            "audio-output",
            "rds-output",
        ],
    )?;
    let input_path = o.required("input")?;
    let output_path = o.required("output")?;
    let registry = Registry::standard();
    let opts = BuildOptions::default();
    let mut spec = cli::load_spec(&registry, o.get("flow"), DEFAULT_FLOW, &opts)?;
    for assignment in o.all("set") {
        cli::apply_set(&mut spec, assignment)?;
    }

    // The input stream
    let input_name = cli::input_name(&spec, o.get("input-name"))?;
    let chunk = spec.settings.max_chunk;
    let mut reader = open_input(&o, input_path, chunk)?;
    let format = reader.format();
    let rate = reader.sample_rate();
    let entry = spec
        .inputs
        .get_mut(&input_name)
        .ok_or_else(|| Failure::Usage(format!("the flow has no input `{input_name}`")))?;
    entry.format = Some(cli::format_spec_of(&format).ok_or_else(|| {
        Failure::Run("the sample format of the file has no name in the flow model".into())
    })?);
    entry.sample_type = None;
    entry.rate_hz = Some(RateHz(rate));

    let audio_name = o.get("audio-output").unwrap_or("audio").to_string();
    let mut pipeline = cli::build(&registry, &spec, &opts)?;
    let audio_rate = pipeline.output_rate(&audio_name)?;

    // A complex stream is stereo audio (left in re, right in im); a real one is mono.
    let stereo = match pipeline.output_port(&audio_name)? {
        PortKind::Stream(PortType::C32) => true,
        PortKind::Stream(PortType::F32) => false,
        _ => {
            return Err(Failure::Usage(format!(
                "output `{audio_name}` is neither an f32 stream nor a C32 stereo stream"
            )));
        }
    };
    // The RDS groups, when the flow has a packets output of that name.
    let rds_name = o.get("rds-output").unwrap_or("rds").to_string();
    let mut rds = match pipeline.output_port(&rds_name) {
        Ok(PortKind::Stream(PortType::Packets)) => Some(Rds::new(rds_name)),
        Ok(_) => {
            return Err(Failure::Usage(format!(
                "output `{rds_name}` is not a packets output"
            )));
        }
        Err(e) if o.get("rds-output").is_some() => return Err(e.into()),
        Err(_) => None,
    };
    let wav_file = File::create(output_path)
        .map_err(|e| Failure::Run(format!("cannot create `{output_path}`: {e}")))?;
    let wav_format = if stereo { formats::CF32 } else { formats::RF32 };
    let mut wav = WavWriter::new(std::io::BufWriter::new(wav_file), wav_format, audio_rate)?;

    let granule = cli::granule_samples(&format);
    let chunk_samples = chunk - chunk % granule;
    let mut buf = vec![0u8; format.bytes_for_samples(chunk_samples.max(granule))?];
    let mut audio = vec![0f32; 4096];
    let mut stereo_audio = vec![C32::new(0.0, 0.0); 2048];
    let mut scratch = Scratch {
        mono: &mut audio,
        stereo: &mut stereo_audio,
        is_stereo: stereo,
    };
    let mut stats = Stats::default();
    loop {
        let (n, meta) = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        let bytes = buf
            .get(..n)
            .ok_or_else(|| Failure::Run("internal: block longer than the buffer".into()))?;
        pipeline.push_with_meta(&input_name, InputData::Raw(bytes), &meta)?;
        stats.samples_in += format.samples_in_bytes(n) as u64;
        drain(
            &mut pipeline,
            &audio_name,
            &mut scratch,
            &mut wav,
            &mut stats,
        )?;
        if let Some(r) = rds.as_mut() {
            r.drain(&mut pipeline)?;
        }
        events(&mut pipeline, &mut stats);
    }
    let out = wav.finish()?;
    drop(out);
    let shown = if input_path.ends_with(".sigmf-meta") || input_path.ends_with(".sigmf-data") {
        sigmf_paths(Path::new(input_path)).0.display().to_string()
    } else {
        input_path.to_string()
    };
    eprintln!("input: {shown} ({} Hz, {})", rate.as_f64(), input_name);
    summary(&pipeline, &stats, &input_name, &audio_name, audio_rate);
    if let Some(r) = &rds {
        r.summary();
    }
    Ok(())
}

/// The RDS display: the groups of a packets output through [`RdsState`].
struct Rds {
    output: String,
    state: RdsState,
    buf: Vec<Packet>,
    groups: u64,
    bad: u64,
    shown_ps: String,
    shown_rt: String,
}

impl Rds {
    fn new(output: String) -> Self {
        Self {
            output,
            state: RdsState::new(),
            buf: vec![Packet::default(); 64],
            groups: 0,
            bad: 0,
            shown_ps: String::new(),
            shown_rt: String::new(),
        }
    }

    /// Reads every waiting group, updates the state and prints PS and RT when they change.
    fn drain(&mut self, p: &mut Pipeline) -> Result<(), Failure> {
        loop {
            let n = p.read_packets(&self.output, &mut self.buf)?;
            for pk in self.buf.iter().take(n) {
                // Only groups with four valid blocks (a flow with `emit_bad` also delivers the
                // others, marked CRC_FAIL).
                match Group::from_packet(pk) {
                    Some(g) if pk.flags.contains(PacketFlags::CRC_OK) => {
                        self.groups += 1;
                        self.state.update(&g);
                    }
                    _ => self.bad += 1,
                }
            }
            self.show();
            if n == 0 {
                return Ok(());
            }
        }
    }

    fn show(&mut self) {
        if self.state.ps_ready() && self.state.ps() != self.shown_ps {
            self.shown_ps = self.state.ps().to_string();
            eprintln!(
                "rds: PI 0x{:04X} PTY {} PS \"{}\"",
                self.state.pi().unwrap_or(0),
                self.state.pty().unwrap_or(0),
                self.shown_ps
            );
        }
        if self.state.rt_ready() && self.state.rt() != self.shown_rt {
            self.shown_rt = self.state.rt().to_string();
            eprintln!("rds: RT \"{}\"", self.shown_rt);
        }
    }

    fn summary(&self) {
        eprintln!(
            "rds groups on `{}`: {} valid, {} not valid",
            self.output, self.groups, self.bad
        );
    }
}

type Input = Box<dyn RxStream<Error = sdrk_io::Error>>;

fn open_input(o: &Options, path: &str, chunk: usize) -> Result<Input, Failure> {
    if path.ends_with(".sigmf-meta") || path.ends_with(".sigmf-data") {
        open_sigmf(path, chunk)
    } else {
        open_raw(o, path, chunk)
    }
}

// Kept out of line so each reader's stack frame stays small (clippy::large_stack_frames).
#[inline(never)]
fn open_sigmf(path: &str, chunk: usize) -> Result<Input, Failure> {
    let r = SigmfReader::open(Path::new(path))?.with_chunk_samples(chunk)?;
    Ok(Box::new(r))
}

#[inline(never)]
fn open_raw(o: &Options, path: &str, chunk: usize) -> Result<Input, Failure> {
    let name = o.required("format")?;
    let format = sdrk_pipeline::FormatName::from_name(name)
        .ok_or_else(|| Failure::Usage(format!("unknown format `{name}`")))?
        .format();
    let hz: u64 = o
        .required("rate")?
        .parse()
        .map_err(|_| Failure::Usage("--rate must be a whole number of hertz".into()))?;
    let rate = Rate::hz(hz).map_err(|e| Failure::Usage(format!("--rate: {e}")))?;
    let file = File::open(path).map_err(|e| Failure::Run(format!("cannot open `{path}`: {e}")))?;
    let r = RawReader::new(
        std::io::BufReader::new(file),
        format,
        rate,
        SampleIndex(0),
        chunk,
    )?;
    Ok(Box::new(r))
}

#[derive(Default)]
struct Stats {
    samples_in: u64,
    samples_out: u64,
    events: usize,
    anchors: usize,
    printed: usize,
}

/// The read buffers of the audio output: real samples for a mono file, complex ones for stereo.
struct Scratch<'a> {
    mono: &'a mut [f32],
    stereo: &'a mut [C32],
    is_stereo: bool,
}

fn drain(
    p: &mut Pipeline,
    name: &str,
    scratch: &mut Scratch<'_>,
    wav: &mut WavWriter<std::io::BufWriter<File>>,
    stats: &mut Stats,
) -> Result<(), Failure> {
    use sdrk_device::TxStream;
    loop {
        let mut bytes = Vec::new();
        let n = if scratch.is_stereo {
            let n = p.read_stream::<C32>(name, scratch.stereo)?;
            // C32 is `re, im` = left, right: the interleaved frames of a stereo WAV file.
            for v in scratch.stereo.iter().take(n) {
                bytes.extend_from_slice(&v.re.to_le_bytes());
                bytes.extend_from_slice(&v.im.to_le_bytes());
            }
            n
        } else {
            let n = p.read_stream::<f32>(name, scratch.mono)?;
            for v in scratch.mono.iter().take(n) {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            n
        };
        if n == 0 {
            return Ok(());
        }
        wav.write(&bytes, &BlockMeta::default())?;
        stats.samples_out += n as u64;
    }
}

fn events(p: &mut Pipeline, stats: &mut Stats) {
    while let Some(e) = p.poll_event() {
        stats.events += 1;
        if matches!(e.kind, EventKind::Anchor(_)) {
            stats.anchors += 1;
            continue;
        }
        if stats.printed < MAX_PRINTED_EVENTS {
            stats.printed += 1;
            eprintln!(
                "event: {} @ {}: {:?}",
                p.output_name(e.output),
                e.index,
                e.kind
            );
        }
    }
}

fn summary(p: &Pipeline, stats: &Stats, input: &str, output: &str, audio_rate: Rate) {
    eprintln!("input samples:  {}", stats.samples_in);
    eprintln!(
        "audio samples:  {} at {} Hz ({:.3} s)",
        stats.samples_out,
        audio_rate.as_f64(),
        stats.samples_out as f64 / audio_rate.as_f64()
    );
    for s in p.stats() {
        eprintln!(
            "node {:<12} calls {:>6}  in {:>10}  out {:>10}  frames {}",
            s.name, s.calls, s.samples_in, s.samples_out, s.frames
        );
    }
    if let Ok(l) = p.latency(output) {
        eprintln!(
            "latency of `{output}`: {:.2} samples group delay, {} buffered at most",
            l.group_delay.as_f64(),
            l.max_buffered
        );
    }
    if let Ok(d) = p.dropped(output) {
        eprintln!("dropped on `{output}`: {d}");
    }
    eprintln!(
        "events: {} ({} anchors not printed), input `{input}`",
        stats.events, stats.anchors
    );
}
