//! The engine behind the JavaScript API, tested on the host: layouts, the one-copy data path,
//! errors, events, timed parameters, pull mode and bursts.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::large_stack_arrays,
    clippy::large_stack_frames
)]

use sdrk_device::{BlockFlags, BlockMeta, SampleIndex, TimeAnchor, TimeSource};
use sdrk_pipeline::ParamTime;
use sdrk_wasm::engine::{
    Dtype, Dyn, Engine, Lanes, LanesMut, Layout, OutLayout, SourceBuf, WasmError, flags_from,
    meta_dyn, time_source_from, wrong_dtype,
};
use serde_json::json;

const FM_RECEIVER: &str = include_str!("../../../examples/flows/fm_receiver.json");
const FM_TRANSMITTER: &str = include_str!("../../../examples/flows/fm_transmitter.json");

fn q15_flow() -> String {
    json!({
        "sdrk": 1, "name": "q15",
        "settings": { "precision": "q15", "max_chunk": 256 },
        "inputs": { "iq": { "type": "cq15", "rate_hz": 48000 } },
        "nodes": { "g": { "kind": "scale", "params": { "gain_db": 0 } } },
        "edges": [ { "from": "iq", "to": "g" }, { "from": "g", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    })
    .to_string()
}

fn c32_flow() -> String {
    json!({
        "sdrk": 1, "name": "c32",
        "settings": { "precision": "f32", "max_chunk": 256 },
        "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
        "nodes": { "g": { "kind": "scale", "params": { "gain_db": 0 } } },
        "edges": [ { "from": "iq", "to": "g" }, { "from": "g", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    })
    .to_string()
}

#[test]
fn layouts_follow_the_stream_types() {
    let e = Engine::from_json(FM_RECEIVER).unwrap();
    assert_eq!(
        e.input_layout("rf").unwrap(),
        Layout {
            dtype: Dtype::U8,
            lanes: 1
        }
    );
    assert_eq!(
        e.output_layout("audio").unwrap(),
        OutLayout::Stream(Layout {
            dtype: Dtype::F32,
            lanes: 1
        })
    );
    assert_eq!(
        e.output_layout("spectrum").unwrap(),
        OutLayout::Frames(1024)
    );
    assert_eq!(e.chunk("audio").unwrap(), 4096);

    let e = Engine::from_json(&q15_flow()).unwrap();
    assert_eq!(
        e.input_layout("iq").unwrap(),
        Layout {
            dtype: Dtype::I16,
            lanes: 2
        }
    );
    let e = Engine::from_json(&c32_flow()).unwrap();
    assert_eq!(
        e.output_layout("out").unwrap(),
        OutLayout::Stream(Layout {
            dtype: Dtype::F32,
            lanes: 2
        })
    );
    // A raw output counts bytes.
    let e = Engine::from_json(FM_TRANSMITTER).unwrap();
    assert_eq!(
        e.output_layout("rf").unwrap(),
        OutLayout::Stream(Layout {
            dtype: Dtype::U8,
            lanes: 1
        })
    );
    assert_eq!(e.input_capacity("audio").unwrap(), 4800);
}

#[test]
fn the_input_buffer_is_allocated_once_and_reused() {
    let mut e = Engine::from_json(FM_RECEIVER).unwrap();
    let capacity = e.input_capacity("rf").unwrap();
    assert_eq!(capacity, 8192);
    let first = match e.stage("rf", 100).unwrap() {
        LanesMut::U8(s) => s.as_ptr(),
        other => panic!("{other:?}"),
    };
    e.push_staged("rf", None).unwrap();
    let second = match e.stage("rf", 200).unwrap() {
        LanesMut::U8(s) => s.as_ptr(),
        other => panic!("{other:?}"),
    };
    assert_eq!(first, second, "the same buffer");
    assert_eq!(e.input_capacity("rf").unwrap(), capacity);
}

#[test]
fn staging_checks_the_length() {
    let mut e = Engine::from_json(&q15_flow()).unwrap();
    // 3 array elements are one and a half complex samples.
    let error = e.stage("iq", 3).unwrap_err();
    assert_eq!(error.kind, "LengthMismatch");
    // max_chunk is 256 samples = 512 array elements.
    assert!(e.stage("iq", 512).is_ok());
    let error = e.stage("iq", 514).unwrap_err();
    assert_eq!(error.kind, "LengthMismatch");
    assert!(error.message.contains("max_chunk"), "{}", error.message);
    let error = e.stage("nope", 2).unwrap_err();
    assert_eq!(error.kind, "InvalidArgument");
    assert!(e.input_layout("nope").is_err());
    assert!(e.output_layout("nope").is_err());
    assert!(e.chunk("nope").is_err());
}

#[test]
fn the_dtype_must_match_the_stream() {
    let mut e = Engine::from_json(FM_RECEIVER).unwrap();
    assert_eq!(
        e.push_f32("rf", &[0.0; 4], None).unwrap_err().kind,
        "InvalidArgument"
    );
    assert_eq!(
        e.push_i16("rf", &[0; 4], None).unwrap_err().kind,
        "InvalidArgument"
    );
    e.push_u8("rf", &[128; 4], None).unwrap();
    let message = wrong_dtype("rf", Dtype::U8).message;
    assert!(message.contains("Uint8Array") && message.contains("rf"));

    let mut e = Engine::from_json(&q15_flow()).unwrap();
    assert_eq!(
        e.push_f32("iq", &[0.0; 4], None).unwrap_err().kind,
        "InvalidArgument"
    );
    e.push_i16("iq", &[100, 200, 300, 400], None).unwrap();
    let mut e = Engine::from_json(&c32_flow()).unwrap();
    assert_eq!(
        e.push_i16("iq", &[0; 4], None).unwrap_err().kind,
        "InvalidArgument"
    );
    e.push_f32("iq", &[0.25, -0.25, 0.5, -0.5], None).unwrap();
    match e.read_chunk("out", 10).unwrap() {
        Lanes::F32(s) => {
            assert_eq!(s.len(), 4, "two complex samples are four floats");
            assert!((s[0] - 0.25).abs() < 1e-3 && (s[3] + 0.5).abs() < 1e-3);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_failed_build_carries_the_diagnostics() {
    let bad = FM_RECEIVER
        .replace("\"decimation\": 10", "\"decimation\": 0")
        .replace("\"kind\": \"fm_demod\"", "\"kind\": \"nope\"");
    let error = Engine::from_json(&bad).unwrap_err();
    assert_eq!(error.kind, "Validation");
    assert_eq!(error.diagnostics.len(), 2);
    assert_eq!(error.code, Some(error.diagnostics[0].code));
    assert_eq!(
        error.path.as_deref(),
        Some(error.diagnostics[0].path.as_str())
    );
    assert!(
        error
            .diagnostics
            .iter()
            .any(|d| d.code == "unknown_kind" && d.path == "nodes.demod.kind")
    );
    assert!(error.to_string().starts_with("Validation: "));
    // A syntax error has a diagnostic, too.
    let error = Engine::from_json("{").unwrap_err();
    assert_eq!(error.code, Some("syntax"));
    // Other errors have no code.
    let e = Engine::from_json(FM_RECEIVER).unwrap();
    let error = e.available("nope").unwrap_err();
    assert_eq!(
        (error.kind.as_str(), error.code, error.path),
        ("InvalidArgument", None, None)
    );
}

#[test]
fn validate_schema_and_backends() {
    assert!(Engine::validate(FM_RECEIVER).is_ok());
    assert!(!Engine::validate("{").is_ok());
    let schema: serde_json::Value = serde_json::from_str(&Engine::schema()).unwrap();
    assert!(
        schema["$schema"]
            .as_str()
            .unwrap()
            .contains("json-schema.org")
    );
    let backends = Engine::backends();
    let names: Vec<&str> = backends.iter().map(|(k, _)| *k).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(names, sorted, "sorted by kernel");
    for kernel in [
        "rotate",
        "fir_decimate",
        "mag_sq",
        "fft",
        "ddc",
        "fm_demod",
        "rfft",
        "scale",
        "halfband_decimate",
        "halfband_decimate_real",
        "real_to_complex",
        "stereo_matrix",
        "stereo_pll",
    ] {
        assert!(names.contains(&kernel), "{kernel}");
    }
    assert!(backends.iter().all(|(_, b)| *b != "none"), "{backends:?}");
    let backend = |kernel: &str| backends.iter().find(|(k, _)| *k == kernel).unwrap().1;
    assert_eq!(backend("ddc"), "chained");
    assert_eq!(backend("squelch"), "fallback");
    assert_eq!(backend("stereo_pll"), "fallback");
    assert_eq!(
        backend("rotate"),
        if sdrk::SIMD_BACKEND {
            "simd"
        } else {
            "fallback"
        }
    );
    #[cfg(sdrk_fallback_only)]
    assert!(backends.iter().all(|(_, b)| *b != "simd"));
}

#[test]
fn frames_stats_latency_and_config() {
    let mut e = Engine::from_json(FM_RECEIVER).unwrap();
    assert!(e.read_frame("spectrum").unwrap().is_none());
    for _ in 0..2 {
        e.push_u8("rf", &[128; 8192], None).unwrap();
    }
    assert!(e.available("spectrum").unwrap() >= 1);
    let frame = e.read_frame("spectrum").unwrap().unwrap();
    assert_eq!(frame.data.len(), 1024);
    assert_eq!(frame.meta.averaged, 4);
    // A stream is no frames output and the other way round.
    assert_eq!(e.read_frame("audio").unwrap_err().kind, "InvalidArgument");
    assert_eq!(
        e.read_chunk("spectrum", 10).unwrap_err().kind,
        "InvalidArgument"
    );

    let stats = e.stats();
    assert!(stats.len() >= 4);
    assert!(
        stats
            .iter()
            .any(|s| s.get("name") == Some(&Dyn::Str("tune".into())))
    );
    assert!(stats[0].get("calls").is_some());

    let (delay, buffered) = e.latency("audio").unwrap();
    assert!(delay > 0.0 && buffered > 0);
    assert!(e.latency("nope").is_err());
    assert_eq!(e.dropped("audio").unwrap(), 0);
    assert_eq!(
        e.config().trim_end(),
        FM_RECEIVER.trim_end(),
        "an unchanged pipeline"
    );
    assert!(e.report().is_ok());
}

#[test]
fn timed_parameters_gaps_and_anchors_become_events() {
    let mut e = Engine::from_json(FM_RECEIVER).unwrap();
    assert_eq!(e.poll_event(), None);
    e.set_param(
        "tune",
        "frequency_hz",
        json!(610_000.0),
        ParamTime::InputIndex {
            input: "rf".into(),
            index: 1000,
        },
    )
    .unwrap();
    e.push_u8("rf", &[128; 4000], None).unwrap();
    let ack = e.poll_event().unwrap();
    assert_eq!(ack.get("type"), Some(&Dyn::Str("paramApplied".into())));
    assert_eq!(ack.get("node"), Some(&Dyn::Str("tune".into())));
    assert_eq!(ack.get("param"), Some(&Dyn::Str("frequency_hz".into())));
    assert_eq!(ack.get("value"), Some(&Dyn::Num(610_000.0)));
    assert_eq!(ack.get("late"), Some(&Dyn::Bool(false)));
    assert_eq!(ack.get("output"), Some(&Dyn::Str("audio".into())));
    assert_eq!(ack.get("index"), Some(&Dyn::U64(20)));
    assert_eq!(e.poll_event(), None);

    // A gap and an anchor in the metadata of a block.
    assert_eq!(e.input_index("rf").unwrap(), 2000);
    let meta = BlockMeta {
        start: SampleIndex(6000),
        flags: BlockFlags::OVERFLOW,
        anchor: Some(TimeAnchor {
            index: SampleIndex(6000),
            unix_ns: 1_700_000_000_000_000_000,
            source: TimeSource::Gps,
        }),
        tx_time_ns: None,
    };
    e.push_u8("rf", &[128; 2000], Some(&meta)).unwrap();
    let mut seen = Vec::new();
    while let Some(event) = e.poll_event() {
        seen.push(event);
    }
    let kind = |event: &Dyn| match event.get("type") {
        Some(Dyn::Str(s)) => s.clone(),
        other => panic!("{other:?}"),
    };
    let kinds: Vec<String> = seen.iter().map(kind).collect();
    assert!(kinds.contains(&"gap".to_string()), "{kinds:?}");
    assert!(kinds.contains(&"overflow".to_string()), "{kinds:?}");
    let gap = seen.iter().find(|ev| kind(ev) == "gap").unwrap();
    assert_eq!(gap.get("missing"), Some(&Dyn::U64(4000)));
    let anchor = seen.iter().find(|ev| kind(ev) == "anchor").unwrap();
    let a = anchor.get("anchor").unwrap();
    // The time of the anchored sample at the output, compensated for the group delay.
    let Some(Dyn::I64(unix_ns)) = a.get("unixNs") else {
        panic!("{a:?}")
    };
    assert!(
        (unix_ns - 1_700_000_000_000_000_000).abs() < 10_000_000,
        "{unix_ns}"
    );
    assert_eq!(a.get("source"), Some(&Dyn::Str("gps".into())));
    assert_eq!(e.input_index("rf").unwrap(), 7000);

    // The start of a block may not go back.
    let back = BlockMeta {
        start: SampleIndex(1),
        ..BlockMeta::default()
    };
    assert_eq!(
        e.push_u8("rf", &[128; 2], Some(&back)).unwrap_err().kind,
        "InvalidState"
    );
}

#[test]
fn set_param_reports_its_errors() {
    let mut e = Engine::from_json(FM_RECEIVER).unwrap();
    let now = || ParamTime::Immediate;
    e.set_param("tune", "frequency_hz", json!(620_000), now())
        .unwrap();
    assert!(e.config().contains("620000"));
    assert_eq!(
        e.set_param("nope", "x", json!(1), now()).unwrap_err().kind,
        "InvalidArgument"
    );
    // `decimation` is fixed at build time.
    assert_eq!(
        e.set_param("tune", "decimation", json!(5), now())
            .unwrap_err()
            .kind,
        "InvalidState"
    );
    assert_eq!(
        e.set_param("tune", "frequency_hz", json!("fast"), now())
            .unwrap_err()
            .kind,
        "InvalidArgument"
    );
    let at = ParamTime::InputIndex {
        input: "nope".into(),
        index: 1,
    };
    assert_eq!(
        e.set_param("tune", "frequency_hz", json!(1.0), at)
            .unwrap_err()
            .kind,
        "InvalidArgument"
    );
    // Absolute time needs an anchor.
    let error = e
        .set_param("tune", "frequency_hz", json!(1.0), ParamTime::UnixNs(5))
        .unwrap_err();
    assert_eq!(error.kind, "InvalidState");
}

/// A source that supplies `total` audio samples in blocks of 480.
fn tone_source(total: usize) -> impl FnMut(&str, SourceBuf<'_>) -> Result<usize, WasmError> {
    let mut remaining = total;
    move |input, buf| {
        assert_eq!(input, "audio");
        assert_eq!(buf.layout.dtype, Dtype::F32);
        let n = buf.max_samples.min(remaining).min(480);
        remaining -= n;
        if let LanesMut::F32(dst) = buf.lanes {
            for (i, v) in dst.iter_mut().take(n).enumerate() {
                *v = 0.5 * (i as f32 * 0.1).sin();
            }
        }
        Ok(n)
    }
}

#[test]
fn pull_mode_bursts_and_their_events() {
    let flow = FM_TRANSMITTER.replace("\"underrun\": \"zeros\"", "\"underrun\": \"short\"");
    let mut e = Engine::from_json(&flow).unwrap();
    let mut source = tone_source(960);
    e.begin_burst("rf", Some(1_700_000_000_000_000_000))
        .unwrap();
    let (first_len, meta) = {
        let pulled = e.pull("rf", 4000, &mut source).unwrap();
        assert_eq!(pulled.samples, 4000);
        assert!(
            matches!(pulled.data, Lanes::U8(b) if b.len() == 4000 && b.iter().any(|x| *x != 0))
        );
        (pulled.samples, pulled.meta)
    };
    assert_eq!(meta.start, SampleIndex(0));
    assert!(meta.flags.contains(BlockFlags::BURST_START));
    assert_eq!(meta.tx_time_ns, Some(1_700_000_000_000_000_000));
    let shape = meta_dyn(&meta);
    assert_eq!(shape.get("start"), Some(&Dyn::U64(0)));
    assert_eq!(shape.get("flags"), Some(&Dyn::Num(8.0)));
    assert_eq!(
        shape.get("txTimeNs"),
        Some(&Dyn::I64(1_700_000_000_000_000_000))
    );

    let mut total = first_len;
    loop {
        let n = e.pull("rf", 4000, &mut source).unwrap().samples;
        if n == 0 {
            break;
        }
        total += n;
    }
    e.end_burst("rf").unwrap();
    loop {
        let n = e.pull("rf", 4000, &mut source).unwrap().samples;
        if n == 0 {
            break;
        }
        total += n;
    }
    assert!((90_000..110_000).contains(&total), "{total}");
    let mut kinds = Vec::new();
    while let Some(event) = e.poll_event() {
        kinds.push(format!("{:?}", event.get("type").unwrap()));
    }
    assert!(kinds.iter().any(|k| k.contains("burstStart")), "{kinds:?}");
    assert!(kinds.iter().any(|k| k.contains("burstEnd")), "{kinds:?}");
    // A second begin while a burst is active is an error.
    e.begin_burst("rf", None).unwrap();
    assert_eq!(e.begin_burst("rf", None).unwrap_err().kind, "InvalidState");
}

#[test]
fn a_failing_source_returns_its_own_error() {
    let mut e = Engine::from_json(FM_TRANSMITTER).unwrap();
    let mut failing = |_: &str, _: SourceBuf<'_>| Err(WasmError::invalid("the source says no"));
    let error = e.pull("rf", 100, &mut failing).unwrap_err();
    assert_eq!(
        (error.kind.as_str(), error.message.as_str()),
        ("InvalidArgument", "the source says no")
    );

    // A source that is empty from the start: silence fills the raw output (policy `zeros`).
    let mut empty = |_: &str, _: SourceBuf<'_>| Ok(0);
    let pulled = e.pull("rf", 100, &mut empty).unwrap();
    assert_eq!(pulled.samples, 100);
    assert!(matches!(pulled.data, Lanes::U8(b) if b.iter().all(|x| *x == 0)));

    // The source may not answer with more than it was asked for; the engine caps the count.
    let mut greedy = |_: &str, b: SourceBuf<'_>| Ok(b.max_samples + 1000);
    assert!(e.pull("rf", 100, &mut greedy).is_ok());
    assert_eq!(
        e.pull("nope", 1, &mut empty).unwrap_err().kind,
        "InvalidArgument"
    );
}

#[test]
fn reads_move_at_most_one_buffer_and_continue_where_they_stopped() {
    let mut e = Engine::from_json(FM_RECEIVER).unwrap();
    for _ in 0..3 {
        e.push_u8("rf", &[128; 8192], None).unwrap();
    }
    let available = e.available("audio").unwrap();
    assert!(available > 100);
    let mut got = 0;
    while let Lanes::F32(s) = e.read_chunk("audio", 100).unwrap() {
        if s.is_empty() {
            break;
        }
        assert!(s.len() <= 100);
        got += s.len();
    }
    assert_eq!(got, available);
    assert_eq!(e.available("audio").unwrap(), 0);
}

#[test]
fn reset_clears_buffers_and_events() {
    let mut e = Engine::from_json(FM_RECEIVER).unwrap();
    e.push_u8("rf", &[128; 4000], Some(&BlockMeta::default()))
        .unwrap();
    e.push_u8(
        "rf",
        &[128; 100],
        Some(&BlockMeta {
            start: SampleIndex(9000),
            ..BlockMeta::default()
        }),
    )
    .unwrap();
    assert!(e.poll_event().is_some());
    e.reset();
    assert_eq!(e.poll_event(), None);
    assert_eq!(e.available("audio").unwrap(), 0);
    assert_eq!(e.input_index("rf").unwrap(), 0);
    e.push_u8("rf", &[128; 4000], None).unwrap();
}

#[test]
fn small_helpers() {
    assert_eq!(time_source_from("gps"), Some(TimeSource::Gps));
    assert_eq!(time_source_from("device"), Some(TimeSource::Device));
    assert_eq!(time_source_from("host"), Some(TimeSource::Host));
    assert_eq!(time_source_from("unknown"), Some(TimeSource::Unknown));
    assert_eq!(time_source_from("sun"), None);
    assert_eq!(flags_from(0xFFFF).bits(), 0x7F, "unknown bits are dropped");
    assert_eq!(Dtype::F32.array_name(), "Float32Array");
    assert_eq!(Dtype::I16.array_name(), "Int16Array");
    assert!(Lanes::U8(&[]).is_empty());
    assert_eq!(Lanes::I16(&[1, 2]).len(), 2);
    let meta = BlockMeta {
        start: SampleIndex(5),
        flags: flags_from(2),
        anchor: Some(TimeAnchor {
            index: SampleIndex(5),
            unix_ns: -7,
            source: TimeSource::Host,
        }),
        tx_time_ns: None,
    };
    let shape = meta_dyn(&meta);
    assert_eq!(
        shape.get("anchor").unwrap().get("unixNs"),
        Some(&Dyn::I64(-7))
    );
    assert_eq!(shape.get("txTimeNs"), None);
}

#[test]
fn public_types_are_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Engine>();
    assert_send::<WasmError>();
    assert_send::<Dyn>();
    assert_send::<sdrk_wasm::engine::Diag>();
    assert_send::<Dtype>();
    assert_send::<Layout>();
    assert_send::<OutLayout>();
    assert_send::<sdrk_wasm::Pipeline>();
}
