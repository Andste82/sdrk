//! Two FM stations at -400 kHz and +400 kHz in one recording, received by one flow with a
//! fan-out of the input to two down-converter / demodulator branches.
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::doc_markdown,
    clippy::unreadable_literal
)]

use num_complex::Complex64;
use sdrk_device::formats;
use sdrk_pipeline::{EdgeSpec, FormatName, InputData, NodeSpec, OutputSpec, OutputType};
use sdrk_system_tests::{AUDIO_RATE, build, encode_bytes, flow, fm_station, set_input_format};
use sdrk_testkit::meters::{ToneMeter, sinad_db};
use serde_json::json;

const STATIONS: [(f64, f64); 2] = [(-400_000.0, 1000.0), (400_000.0, 1700.0)];
/// Samples per station: 0.5 s.
const SAMPLES: usize = 1_200_000;

fn stream_output() -> OutputSpec {
    OutputSpec {
        output_type: OutputType::Stream,
        format: None,
        capacity: Some(16384),
        queue: None,
        drop: None,
        underrun: None,
        description: None,
    }
}

#[test]
fn two_stations_no_crosstalk() {
    // One flow, two branches: rf -> tune_a -> demod_a -> audio_a and rf -> tune_b -> demod_b -> audio_b.
    let mut spec = flow("fm_receiver");
    set_input_format(&mut spec, "rf", FormatName::Cs8);
    let tune = spec.nodes["tune"].clone();
    let demod = spec.nodes["demod"].clone();
    spec.nodes.clear();
    spec.edges.clear();
    spec.outputs.clear();
    for (name, (offset, _)) in ["a", "b"].into_iter().zip(STATIONS) {
        let mut t: NodeSpec = tune.clone();
        t.params.insert("frequency_hz".into(), json!(offset));
        spec.nodes.insert(format!("tune_{name}"), t);
        spec.nodes.insert(format!("demod_{name}"), demod.clone());
        for (from, to) in [
            ("rf".to_string(), format!("tune_{name}")),
            (format!("tune_{name}"), format!("demod_{name}")),
            (format!("demod_{name}"), format!("audio_{name}")),
        ] {
            spec.edges.push(EdgeSpec { from, to });
        }
        spec.outputs
            .insert(format!("audio_{name}"), stream_output());
    }
    let mut p = build(&spec);

    // The recording: both stations at the same level, CS8.
    let mut bytes = Vec::with_capacity(2 * SAMPLES);
    let mut start = 0usize;
    while start < SAMPLES {
        let n = 50_000.min(SAMPLES - start);
        let mut iq = vec![Complex64::new(0.0, 0.0); n];
        for (offset, tone) in STATIONS {
            for (sum, v) in iq
                .iter_mut()
                .zip(fm_station(offset, tone, 0.4, start as u64, n))
            {
                *sum += v;
            }
        }
        bytes.extend(encode_bytes(&formats::CS8, &iq));
        start += n;
    }

    let mut audio = [Vec::new(), Vec::new()];
    let mut buf = vec![0f32; 4096];
    for piece in bytes.chunks(8192) {
        p.push("rf", InputData::Raw(piece)).unwrap();
        for (k, name) in ["audio_a", "audio_b"].into_iter().enumerate() {
            loop {
                let n = p.read_stream::<f32>(name, &mut buf).unwrap();
                if n == 0 {
                    break;
                }
                audio[k].extend(buf[..n].iter().map(|&v| f64::from(v)));
            }
        }
    }

    for (k, (_, wanted)) in STATIONS.into_iter().enumerate() {
        let other = STATIONS[1 - k].1;
        let x = &audio[k];
        assert!(x.len() > 20_000);
        let tail = &x[x.len() / 2..];
        let sinad = sinad_db(tail, AUDIO_RATE, wanted);
        let a_wanted = ToneMeter::measure(tail, AUDIO_RATE, wanted).amplitude;
        let a_other = ToneMeter::measure(tail, AUDIO_RATE, other).amplitude;
        let crosstalk_db = 20.0 * (a_other / a_wanted).log10();
        println!("output {k}: SINAD {sinad:.1} dB, crosstalk {crosstalk_db:.1} dB");
        assert!(sinad >= 35.0, "output {k}: SINAD {sinad} dB");
        assert!(
            crosstalk_db <= -50.0,
            "output {k}: crosstalk {crosstalk_db} dB"
        );
    }
}
