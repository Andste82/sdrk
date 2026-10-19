//! The push runner: errors, drop policies, output kinds, run-time parameters, config, stats.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::assert_is_empty,
    clippy::unreadable_literal,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::redundant_closure_for_method_calls
)]

mod common;

use sdrk::{C32, CQ15, Q15};
use sdrk_blocks::{FrameMeta, ParamValue};
use sdrk_device::{ErrorKind, SdrkError};
use sdrk_pipeline::{DropPolicy, Error, InputData, Pipeline, PipelineSpec};
use serde_json::{Value, json};

use common::{bits, build, flow};

/// c32 in, a 0 dB gain (exactly 1.0 in f32), one stream output.
fn passthrough(capacity: usize, drop: &str, max_chunk: usize) -> Pipeline {
    build(
        &json!({
            "sdrk": 1, "name": "pass", "settings": { "precision": "f32", "max_chunk": max_chunk },
            "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
            "nodes": { "g": { "kind": "scale" } },
            "edges": [ { "from": "iq", "to": "g" }, { "from": "g", "to": "out" } ],
            "outputs": { "out": { "type": "stream", "capacity": capacity, "drop": drop } }
        })
        .to_string(),
    )
}

fn ramp(from: usize, n: usize) -> Vec<C32> {
    (from..from + n)
        .map(|i| C32::new(i as f32, -(i as f32)))
        .collect()
}

fn read_all(p: &mut Pipeline, out: &str) -> Vec<C32> {
    let mut all = Vec::new();
    let mut buf = vec![C32::default(); 37];
    loop {
        let n = p.read_stream(out, &mut buf).unwrap();
        if n == 0 {
            return all;
        }
        all.extend_from_slice(&buf[..n]);
    }
}

#[test]
fn a_gain_node_passes_the_samples() {
    let mut p = passthrough(1000, "error", 64);
    p.push("iq", InputData::C32(&ramp(0, 10))).unwrap();
    assert_eq!(p.available("out").unwrap(), 10);
    let got = read_all(&mut p, "out");
    assert_eq!(got, ramp(0, 10));
    assert_eq!(p.available("out").unwrap(), 0);
    // An empty push is fine.
    p.push("iq", InputData::C32(&[])).unwrap();
}

#[test]
fn the_error_policy_refuses_a_push_and_consumes_nothing() {
    let mut p = passthrough(100, "error", 64);
    p.push("iq", InputData::C32(&ramp(0, 64))).unwrap();
    let before: Vec<_> = p.stats().to_vec();
    let err = p.push("iq", InputData::C32(&ramp(64, 64))).unwrap_err();
    assert_eq!(err, Error::OutputFull("out".into()));
    assert_eq!(err.kind(), ErrorKind::CapacityExceeded);
    assert_eq!(p.stats(), &before[..], "nothing was processed");
    assert_eq!(
        p.available("out").unwrap(),
        64,
        "nothing was consumed or lost"
    );
    // Reading makes room; the same push now works and nothing is missing.
    let mut first = read_all(&mut p, "out");
    p.push("iq", InputData::C32(&ramp(64, 64))).unwrap();
    first.extend(read_all(&mut p, "out"));
    assert_eq!(first, ramp(0, 128));
    assert_eq!(p.dropped("out").unwrap(), 0);
}

#[test]
fn drop_newest_keeps_the_oldest_data() {
    let mut p = passthrough(100, "drop_newest", 64);
    for k in 0..3 {
        p.push("iq", InputData::C32(&ramp(64 * k, 64))).unwrap();
    }
    assert_eq!(p.available("out").unwrap(), 100);
    assert_eq!(p.dropped("out").unwrap(), 192 - 100);
    assert_eq!(read_all(&mut p, "out"), ramp(0, 100));
}

#[test]
fn drop_oldest_keeps_the_newest_data() {
    let mut p = passthrough(100, "drop_oldest", 64);
    for k in 0..3 {
        p.push("iq", InputData::C32(&ramp(64 * k, 64))).unwrap();
    }
    assert_eq!(p.available("out").unwrap(), 100);
    assert_eq!(p.dropped("out").unwrap(), 92);
    assert_eq!(read_all(&mut p, "out"), ramp(92, 100));
    // A single push larger than the queue keeps its newest samples.
    let mut p = passthrough(10, "drop_oldest", 64);
    p.push("iq", InputData::C32(&ramp(0, 64))).unwrap();
    assert_eq!(read_all(&mut p, "out"), ramp(54, 10));
    assert_eq!(p.dropped("out").unwrap(), 54);
}

#[test]
fn drop_policies_of_frame_outputs() {
    let frames_flow = |drop: &str| {
        build(
            &json!({
                "sdrk": 1, "name": "spec", "settings": { "max_chunk": 256 },
                "inputs": { "iq": { "type": "c32", "rate_hz": 1000 } },
                "nodes": { "f": { "kind": "spectrum", "params": { "fft_size": 16 } } },
                "edges": [ { "from": "iq", "to": "f" }, { "from": "f", "to": "spectrum" } ],
                "outputs": { "spectrum": { "type": "frames", "queue": 2, "drop": drop } }
            })
            .to_string(),
        )
    };
    let tone: Vec<C32> = (0..160)
        .map(|i| C32::new((i as f32 * 0.3).cos(), (i as f32 * 0.3).sin()))
        .collect();
    let starts = |p: &mut Pipeline| -> Vec<u64> {
        let mut frame = [0f32; 16];
        let mut v = Vec::new();
        while let Some(FrameMeta { first_sample, .. }) =
            p.read_frame("spectrum", &mut frame).unwrap()
        {
            v.push(first_sample);
        }
        v
    };
    // drop_oldest: 10 frames made, the last two stay
    let mut p = frames_flow("drop_oldest");
    p.push("iq", InputData::C32(&tone)).unwrap();
    assert_eq!(p.available("spectrum").unwrap(), 2);
    assert_eq!(p.dropped("spectrum").unwrap(), 8);
    assert_eq!(starts(&mut p), vec![128, 144]);
    // drop_newest: the first two stay
    let mut p = frames_flow("drop_newest");
    p.push("iq", InputData::C32(&tone)).unwrap();
    assert_eq!(p.dropped("spectrum").unwrap(), 8);
    assert_eq!(starts(&mut p), vec![0, 16]);
    // error: a full queue refuses the push, nothing is consumed
    let mut p = frames_flow("error");
    p.push("iq", InputData::C32(&tone[..32])).unwrap();
    assert_eq!(p.available("spectrum").unwrap(), 2);
    let calls = p.stats()[0].calls;
    let err = p.push("iq", InputData::C32(&tone[32..64])).unwrap_err();
    assert_eq!(err, Error::OutputFull("spectrum".into()));
    assert_eq!(p.stats()[0].calls, calls);
    assert_eq!(starts(&mut p), vec![0, 16]);
    p.push("iq", InputData::C32(&tone[32..64])).unwrap();
    assert_eq!(starts(&mut p), vec![32, 48]);
}

#[test]
fn a_stall_with_the_error_policy_loses_nothing() {
    // A decimating chain whose worst-case estimate is too optimistic still never loses data:
    // the blocked nodes keep it until the reader makes room.
    let mut p = build(
        &json!({
            "sdrk": 1, "name": "stall", "settings": { "max_chunk": 64 },
            "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
            "nodes": { "r": { "kind": "resampler", "params": { "rate_out_hz": 96000, "max_taps": 4096 } } },
            "edges": [ { "from": "iq", "to": "r" }, { "from": "r", "to": "out" } ],
            "outputs": { "out": { "type": "stream", "capacity": 150, "drop": "error" } }
        })
        .to_string(),
    );
    let input = ramp(0, 640);
    let mut got = Vec::new();
    for chunk in input.chunks(64) {
        loop {
            match p.push("iq", InputData::C32(chunk)) {
                Ok(()) => break,
                Err(Error::OutputFull(_)) => got.extend(read_all(&mut p, "out")),
                Err(e) => panic!("{e}"),
            }
        }
    }
    got.extend(read_all(&mut p, "out"));
    // Same samples as an unconstrained run.
    let mut q = build(
        &json!({
            "sdrk": 1, "name": "stall", "settings": { "max_chunk": 640 },
            "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
            "nodes": { "r": { "kind": "resampler", "params": { "rate_out_hz": 96000, "max_taps": 4096 } } },
            "edges": [ { "from": "iq", "to": "r" }, { "from": "r", "to": "out" } ],
            "outputs": { "out": { "type": "stream", "capacity": 4000, "drop": "error" } }
        })
        .to_string(),
    );
    q.push("iq", InputData::C32(&input)).unwrap();
    let expected = read_all(&mut q, "out");
    assert_eq!(got.len(), expected.len());
    assert_eq!(got, expected);
}

#[test]
fn argument_errors() {
    let mut p = passthrough(100, "error", 64);
    let mut out = [C32::default(); 4];
    assert_eq!(
        p.push("nope", InputData::C32(&[])).unwrap_err(),
        Error::UnknownInput("nope".into())
    );
    assert_eq!(
        p.read_stream("nope", &mut out).unwrap_err(),
        Error::UnknownOutput("nope".into())
    );
    assert_eq!(
        p.available("nope").unwrap_err(),
        Error::UnknownOutput("nope".into())
    );
    assert_eq!(
        p.set_param("nope", "gain_db", json!(1)).unwrap_err(),
        Error::UnknownNode("nope".into())
    );
    // Wrong sample type on push and on read.
    let err = p.push("iq", InputData::F32(&[0.0])).unwrap_err();
    assert!(matches!(err, Error::WrongType { .. }), "{err}");
    assert_eq!(err.kind(), ErrorKind::InvalidArgument);
    let err = p.push("iq", InputData::Raw(&[0, 0])).unwrap_err();
    assert!(matches!(err, Error::WrongType { .. }));
    let mut wrong = [0f32; 4];
    assert!(matches!(
        p.read_stream("out", &mut wrong).unwrap_err(),
        Error::WrongType { .. }
    ));
    let mut cq = [CQ15::default(); 4];
    assert!(matches!(
        p.read_stream("out", &mut cq).unwrap_err(),
        Error::WrongType { .. }
    ));
    assert!(matches!(
        p.read_raw("out", &mut [0u8; 4]).unwrap_err(),
        Error::WrongType { .. }
    ));
    assert!(matches!(
        p.read_frame("out", &mut [0f32; 4]).unwrap_err(),
        Error::WrongType { .. }
    ));
    // Too many samples.
    let err = p.push("iq", InputData::C32(&ramp(0, 65))).unwrap_err();
    assert_eq!(err, Error::ChunkTooLarge { len: 65, max: 64 });
    assert_eq!(err.kind(), ErrorKind::LengthMismatch);
    assert_eq!(p.stats()[0].calls, 0, "nothing was processed");
    assert!(err.to_string().contains("65"));
}

#[test]
fn outputs_of_every_kind() {
    let mut p = build(&flow("fm_receiver"));
    let mut frame = [0f32; 1024];
    assert!(matches!(
        p.read_stream("spectrum", &mut [0f32; 4]).unwrap_err(),
        Error::WrongType { .. }
    ));
    assert!(matches!(
        p.read_frame("audio", &mut frame).unwrap_err(),
        Error::WrongType { .. }
    ));
    assert_eq!(
        p.read_frame("spectrum", &mut [0f32; 3]).unwrap_err(),
        Error::LengthMismatch {
            expected: 1024,
            got: 3
        }
    );
    assert_eq!(p.read_frame("spectrum", &mut frame).unwrap(), None);
    assert!(matches!(
        p.read_stream::<Q15>("audio", &mut [Q15(0); 4]).unwrap_err(),
        Error::WrongType { .. }
    ));
    // Raw input of a wrong type.
    assert!(matches!(
        p.push("rf", InputData::F32(&[])).unwrap_err(),
        Error::WrongType { .. }
    ));
}

#[test]
fn a_raw_output_carries_converted_bytes() {
    let mut p = build(
        &json!({
            "sdrk": 1, "name": "raw", "settings": { "max_chunk": 64 },
            "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
            "nodes": { "g": { "kind": "scale" } },
            "edges": [ { "from": "iq", "to": "g" }, { "from": "g", "to": "out" } ],
            "outputs": { "out": { "type": "stream", "format": "cs8", "capacity": 64 } }
        })
        .to_string(),
    );
    p.push(
        "iq",
        InputData::C32(&[C32::new(0.5, -0.5), C32::new(0.0, 1.0)]),
    )
    .unwrap();
    assert_eq!(p.available("out").unwrap(), 4);
    let mut bytes = [0u8; 8];
    assert_eq!(p.read_raw("out", &mut bytes).unwrap(), 4);
    assert_eq!(&bytes[..4], &[64, 192, 0, 127]);
    assert!(matches!(
        p.read_stream("out", &mut [C32::default(); 2]).unwrap_err(),
        Error::WrongType { .. }
    ));
}

#[test]
fn the_q15_pipeline_runs_and_outputs_q15() {
    let json = flow("fm_receiver").replace("\"precision\": \"f32\"", "\"precision\": \"q15\"");
    let mut p = build(&json);
    let bytes = common::cu8_recording(100_000);
    let mut audio = Vec::new();
    let mut buf = vec![Q15(0); 4096];
    for piece in bytes.chunks(8000) {
        p.push("rf", InputData::Raw(piece)).unwrap();
        loop {
            let n = p.read_stream("audio", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            audio.extend_from_slice(&buf[..n]);
        }
    }
    assert_eq!(audio.len(), 100_000 / 50);
    let f: Vec<f32> = audio
        .iter()
        .skip(1000)
        .map(|q| f32::from(q.0) / 32768.0)
        .collect();
    assert!(
        common::tone_power(&f, 1000.0, 48_000.0) > 1e-4,
        "the tone is there"
    );
    let mut wrong = [0f32; 4];
    assert!(matches!(
        p.read_stream("audio", &mut wrong).unwrap_err(),
        Error::WrongType { .. }
    ));
}

fn two_readers() -> Pipeline {
    // Two readers of the same input stream with different history needs (1 and 7 samples).
    build(
        &json!({
            "sdrk": 1, "name": "fan", "settings": { "max_chunk": 700 },
            "inputs": { "iq": { "type": "c32", "rate_hz": 70000 } },
            "nodes": {
                "a": { "kind": "ddc", "params": { "decimation": 10, "max_taps": 4096 } },
                "b": { "kind": "ddc", "params": { "decimation": 7, "frequency_hz": 5000, "max_taps": 4096 } },
                "c": { "kind": "scale", "params": { "gain_db": -3 } }
            },
            "edges": [ { "from": "iq", "to": "a" }, { "from": "iq", "to": "b" }, { "from": "iq", "to": "c" },
                       { "from": "a", "to": "oa" }, { "from": "b", "to": "ob" }, { "from": "c", "to": "oc" } ],
            "outputs": { "oa": { "type": "stream" }, "ob": { "type": "stream" }, "oc": { "type": "stream" } }
        })
        .to_string(),
    )
}

#[test]
fn readers_with_different_leftovers_share_one_input_buffer() {
    let input: Vec<C32> = (0..2000)
        .map(|i| C32::new((i as f32 * 0.01).sin(), (i as f32 * 0.017).cos()))
        .collect();
    let mut whole = two_readers();
    for chunk in input.chunks(700) {
        whole.push("iq", InputData::C32(chunk)).unwrap();
    }
    let expected: Vec<Vec<C32>> = ["oa", "ob", "oc"]
        .iter()
        .map(|o| read_all(&mut whole, o))
        .collect();
    assert!(expected.iter().all(|e| !e.is_empty()));
    for step in [1usize, 2, 3, 5, 7, 11, 64, 333] {
        let mut p = two_readers();
        for chunk in input.chunks(step) {
            p.push("iq", InputData::C32(chunk)).unwrap();
        }
        for (o, want) in ["oa", "ob", "oc"].iter().zip(&expected) {
            let got = read_all(&mut p, o);
            assert_eq!(got.len(), want.len(), "{o} step {step}");
            let g: Vec<u32> = got
                .iter()
                .flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
                .collect();
            let w: Vec<u32> = want
                .iter()
                .flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
                .collect();
            assert_eq!(g, w, "{o} step {step}");
        }
    }
}

#[test]
fn two_inputs_run_independently() {
    let mut p = build(
        &json!({
            "sdrk": 1, "name": "two", "settings": { "max_chunk": 128 },
            "inputs": { "i1": { "type": "c32", "rate_hz": 48000 }, "i2": { "type": "f32", "rate_hz": 8000 } },
            "nodes": { "g": { "kind": "scale", "params": { "gain_db": -6.0206 } },
                       "r": { "kind": "resampler", "params": { "rate_out_hz": 16000 } } },
            "edges": [ { "from": "i1", "to": "g" }, { "from": "g", "to": "o1" }, { "from": "i2", "to": "r" }, { "from": "r", "to": "o2" } ],
            "outputs": { "o1": { "type": "stream" }, "o2": { "type": "stream" } }
        })
        .to_string(),
    );
    p.push("i2", InputData::F32(&vec![1.0; 100])).unwrap();
    assert_eq!(p.available("o1").unwrap(), 0);
    assert!(p.available("o2").unwrap() > 150);
    p.push("i1", InputData::C32(&ramp(0, 5))).unwrap();
    assert_eq!(p.available("o1").unwrap(), 5);
    assert_eq!(
        p.stats()
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>(),
        vec!["g", "r"]
    );
    assert_eq!((p.stats()[0].samples_in, p.stats()[0].samples_out), (5, 5));
    assert_eq!(
        (p.stats()[1].samples_in, p.stats()[1].samples_out),
        (100, p.stats()[1].samples_out)
    );
}

#[test]
fn stats_count_calls_samples_and_frames() {
    let mut p = build(&flow("fm_receiver"));
    let bytes = common::cu8_recording(50_000);
    for piece in bytes.chunks(8000) {
        p.push("rf", InputData::Raw(piece)).unwrap();
        let mut buf = [0f32; 4096];
        p.read_stream("audio", &mut buf).unwrap();
        let mut frame = [0f32; 1024];
        while p.read_frame("spectrum", &mut frame).unwrap().is_some() {}
    }
    let stats = p.stats();
    let by = |n: &str| stats.iter().find(|s| s.name == n).unwrap();
    assert_eq!(by("rf__convert").samples_in, 100_000, "bytes in");
    assert_eq!(by("rf__convert").samples_out, 50_000);
    assert_eq!(by("tune").samples_in, 50_000);
    assert_eq!(by("tune").samples_out, 5_000);
    assert_eq!(by("demod").samples_out, 1_000);
    assert_eq!(by("fft").samples_in, 50_000);
    assert_eq!(by("fft").frames, 12, "50 000 / (1024 * 4)");
    assert!(by("tune").calls >= 7);
}

#[test]
fn reset_empties_everything_and_restarts_the_blocks() {
    let bytes = common::cu8_recording(60_000);
    let run = |p: &mut Pipeline| -> (Vec<f32>, usize) {
        let (a, f) = common::run_fm(p, &bytes, 8000);
        (a, f.len())
    };
    let mut fresh = build(&flow("fm_receiver"));
    let (expected, frames) = run(&mut fresh);
    let mut p = build(&flow("fm_receiver"));
    // Leave data in the buffers and the queues, then reset.
    p.push("rf", InputData::Raw(&bytes[..3001])).unwrap();
    assert!(p.available("audio").unwrap() > 0);
    let calls = p.stats()[0].calls;
    p.reset();
    assert_eq!(p.available("audio").unwrap(), 0);
    assert_eq!(p.available("spectrum").unwrap(), 0);
    assert_eq!(p.stats()[0].calls, calls, "statistics are kept");
    let (again, frames_again) = run(&mut p);
    assert_eq!(bits(&again), bits(&expected));
    assert_eq!(frames_again, frames);
}

// ------------------------------------------------------------ set_param / config

fn text_of(spec: &PipelineSpec) -> String {
    spec.to_json()
}

#[test]
fn config_of_an_unchanged_pipeline_equals_the_input() {
    for name in ["fm_receiver", "spectrum_only", "resample_audio"] {
        let text = flow(name);
        let p = build(&text);
        let expected = PipelineSpec::from_json(&text).unwrap().to_json();
        assert_eq!(text_of(&p.config()), expected, "{name}");
        assert!(
            !p.config().nodes.keys().any(|n| n.contains("__")),
            "no automatic nodes"
        );
    }
}

#[test]
fn set_param_changes_the_value_and_the_readout() {
    let mut p = build(&flow("fm_receiver"));
    p.set_param("tune", "frequency_hz", json!(123_456.5))
        .unwrap();
    p.set_param("fft", "averaging", json!(2)).unwrap();
    p.set_param("fft", "interval_samples", json!(512)).unwrap();
    let cfg = p.config();
    let params = |n: &str| cfg.nodes[n].params.clone();
    assert_eq!(params("tune")["frequency_hz"], json!(123_456.5));
    assert_eq!(
        params("tune")["decimation"],
        json!(10),
        "build parameters are untouched"
    );
    assert_eq!(params("fft")["averaging"], json!(2));
    assert_eq!(params("fft")["interval_samples"], json!(512));
    assert_eq!(params("fft")["fft_size"], json!(1024));
    // The readout is itself a valid description that builds the same pipeline state.
    let again = build(&cfg.to_json());
    assert_eq!(again.config(), cfg);
    // A run-time parameter that the description did not mention appears once it changed.
    assert!(
        !PipelineSpec::from_json(&flow("fm_receiver")).unwrap().nodes["fft"]
            .params
            .contains_key("interval_samples")
    );
}

#[test]
fn set_param_errors() {
    let mut p = build(&flow("fm_receiver"));
    let source = |e: Error| match e {
        Error::Block { source, node } => (node, source),
        other => panic!("not a block error: {other}"),
    };
    // A build parameter cannot change.
    let (node, e) = source(p.set_param("tune", "decimation", json!(5)).unwrap_err());
    assert_eq!(
        (node.as_str(), e),
        ("tune", sdrk_blocks::Error::ReadOnlyParameter)
    );
    // Unknown parameter.
    let (_, e) = source(p.set_param("tune", "nope", json!(5)).unwrap_err());
    assert_eq!(e, sdrk_blocks::Error::UnknownParameter);
    let (_, e) = source(p.set_param("fft", "window", json!("kaiser")).unwrap_err());
    assert_eq!(e, sdrk_blocks::Error::ReadOnlyParameter);
    // Out of range (the block's own limit).
    let (_, e) = source(
        p.set_param("tune", "frequency_hz", json!(1.0e9))
            .unwrap_err(),
    );
    assert!(matches!(e, sdrk_blocks::Error::OutOfRange { .. }), "{e}");
    let err = p
        .set_param("tune", "frequency_hz", json!(1.0e9))
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::OutOfRange);
    // Wrong JSON type, bad option name, non-integer.
    for (node, name, value) in [
        ("tune", "frequency_hz", json!("fast")),
        ("tune", "frequency_hz", Value::Null),
        ("fft", "averaging", json!("2")),
        ("fft", "interval_samples", json!([1])),
        ("fft", "averaging", json!(2.5)),
        ("fft", "averaging", json!(true)),
    ] {
        let err = p.set_param(node, name, value.clone()).unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { .. }),
            "{node}.{name} = {value}: {err}"
        );
        assert_eq!(err.kind(), ErrorKind::InvalidArgument);
    }
    // Integers written as floats are accepted.
    p.set_param("fft", "averaging", json!(3.0)).unwrap();
    // The failed calls changed nothing.
    let cfg = p.config();
    assert_eq!(cfg.nodes["tune"].params["frequency_hz"], json!(600_000));
    assert_eq!(cfg.nodes["fft"].params["averaging"], json!(3));
    assert!(!cfg.nodes["fft"].params.contains_key("interval_samples"));
}

#[test]
fn a_gain_change_applies_from_the_next_chunk() {
    let mut p = build(
        &json!({
            "sdrk": 1, "name": "gain",
            "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
            "nodes": { "g": { "kind": "scale", "params": { "gain_db": 0 } } },
            "edges": [ { "from": "iq", "to": "g" }, { "from": "g", "to": "out" } ],
            "outputs": { "out": { "type": "stream" } }
        })
        .to_string(),
    );
    let one = [C32::new(1.0, 0.0); 4];
    p.push("iq", InputData::C32(&one)).unwrap();
    p.set_param("g", "gain_db", json!(-20.0)).unwrap();
    p.push("iq", InputData::C32(&one)).unwrap();
    let out = read_all(&mut p, "out");
    assert_eq!(out.len(), 8);
    assert!(out[..4].iter().all(|c| (c.re - 1.0).abs() < 1e-6));
    assert!(out[4..].iter().all(|c| (c.re - 0.1).abs() < 1e-6));
}

#[test]
fn the_error_kinds_are_mapped() {
    let cases: Vec<(Error, ErrorKind)> = vec![
        (Error::UnknownInput("x".into()), ErrorKind::InvalidArgument),
        (Error::UnknownOutput("x".into()), ErrorKind::InvalidArgument),
        (Error::UnknownNode("x".into()), ErrorKind::InvalidArgument),
        (
            Error::WrongType {
                name: "x".into(),
                expected: "a",
                got: "b",
            },
            ErrorKind::InvalidArgument,
        ),
        (
            Error::ChunkTooLarge { len: 2, max: 1 },
            ErrorKind::LengthMismatch,
        ),
        (Error::OutputFull("x".into()), ErrorKind::CapacityExceeded),
        (
            Error::LengthMismatch {
                expected: 1,
                got: 2,
            },
            ErrorKind::LengthMismatch,
        ),
        (Error::Json("x".into()), ErrorKind::Parse),
        (
            Error::Build(sdrk_pipeline::Report::default()),
            ErrorKind::Validation,
        ),
        (Error::Internal("x"), ErrorKind::Internal),
        (
            Error::Block {
                node: "n".into(),
                source: sdrk_blocks::Error::ReadOnlyParameter,
            },
            ErrorKind::InvalidState,
        ),
    ];
    for (e, k) in cases {
        assert_eq!(e.kind(), k, "{e}");
        let text = e.to_string();
        assert!(!text.is_empty() && !text.ends_with('.'), "{text}");
    }
    let block = Error::Block {
        node: "n".into(),
        source: sdrk_blocks::Error::UnknownParameter,
    };
    assert!(std::error::Error::source(&block).is_some());
}

#[test]
fn public_types_are_send() {
    fn is_send<T: Send>() {}
    is_send::<Pipeline>();
    is_send::<sdrk_pipeline::Registry>();
    is_send::<sdrk_pipeline::Report>();
    is_send::<sdrk_pipeline::Diagnostic>();
    is_send::<sdrk_pipeline::PipelineSpec>();
    is_send::<sdrk_pipeline::Error>();
    is_send::<sdrk_pipeline::Checked>();
    is_send::<sdrk_pipeline::VecStorage>();
    is_send::<Box<dyn sdrk_pipeline::Node>>();
    is_send::<sdrk_pipeline::NodeStats>();
    is_send::<sdrk_pipeline::BuildOptions>();
    let _ = ParamValue::F64(0.0);
    let _ = DropPolicy::Error;
}

#[test]
fn an_input_can_feed_an_output_directly() {
    let mut p = build(
        &json!({
            "sdrk": 1, "name": "wire", "settings": { "max_chunk": 64 },
            "inputs": { "iq": { "type": "c32", "rate_hz": 48000 }, "rf": { "format": "cs16", "rate_hz": 48000 } },
            "edges": [ { "from": "iq", "to": "o1" }, { "from": "rf", "to": "o2" } ],
            "outputs": { "o1": { "type": "stream" }, "o2": { "type": "stream" } }
        })
        .to_string(),
    );
    p.push("iq", InputData::C32(&ramp(0, 5))).unwrap();
    assert_eq!(read_all(&mut p, "o1"), ramp(0, 5));
    // cs16 bytes: I = 16384 (0.5), Q = -16384 (-0.5), then a partial sample.
    p.push("rf", InputData::Raw(&[0x00, 0x40, 0x00, 0xC0, 0x01]))
        .unwrap();
    assert_eq!(read_all(&mut p, "o2"), vec![C32::new(0.5, -0.5)]);
    p.push("rf", InputData::Raw(&[0x00, 0x00, 0x00])).unwrap();
    assert_eq!(
        read_all(&mut p, "o2").len(),
        1,
        "the split sample is completed by the next push"
    );
}

#[test]
fn explicit_converter_nodes_chain_with_raw_streams() {
    let mut p = build(
        &json!({
            "sdrk": 1, "name": "roundtrip", "settings": { "max_chunk": 64 },
            "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
            "nodes": {
                "co": { "kind": "convert_out", "params": { "format": "cs16" } },
                "ci": { "kind": "convert", "params": { "format": "cs16" } }
            },
            "edges": [ { "from": "iq", "to": "co" }, { "from": "co", "to": "ci" }, { "from": "ci", "to": "out" },
                       { "from": "iq", "to": "o2" } ],
            "outputs": { "out": { "type": "stream" }, "o2": { "type": "stream" } }
        })
        .to_string(),
    );
    let input: Vec<C32> = (0..40)
        .map(|i| C32::new(i as f32 / 100.0, -(i as f32) / 100.0))
        .collect();
    p.push("iq", InputData::C32(&input)).unwrap();
    let out = read_all(&mut p, "out");
    assert_eq!(out.len(), 40);
    for (a, b) in input.iter().zip(&out) {
        assert!((a.re - b.re).abs() < 1e-4 && (a.im - b.im).abs() < 1e-4);
    }
    assert_eq!(read_all(&mut p, "o2"), input);
}

#[test]
fn a_raw_stream_output_of_an_explicit_converter_is_readable_as_bytes() {
    let mut p = build(
        &json!({
            "sdrk": 1, "name": "bytes", "settings": { "max_chunk": 64 },
            "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
            "nodes": { "co": { "kind": "convert_out", "params": { "format": "cu8" } } },
            "edges": [ { "from": "iq", "to": "co" }, { "from": "co", "to": "out" } ],
            "outputs": { "out": { "type": "stream", "capacity": 16 } }
        })
        .to_string(),
    );
    p.push(
        "iq",
        InputData::C32(&[C32::new(0.0, 0.0), C32::new(1.0, -1.0)]),
    )
    .unwrap();
    let mut bytes = [0u8; 8];
    assert_eq!(p.read_raw("out", &mut bytes).unwrap(), 4);
    assert_eq!(&bytes[..4], &[128, 128, 255, 0]);
}
