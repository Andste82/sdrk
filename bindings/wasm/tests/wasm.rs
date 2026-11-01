//! The JavaScript API, called from Rust under `wasm-bindgen-test` in Node, Chrome and Firefox.
//!
//! Build and run: `cargo xtask wasm-test` (see `docs/wasm.md`). The test build decides what the
//! receive test checks: with `--cfg sdrk_fallback_only` (the scalar build) the audio must match
//! the committed fixture bit for bit, otherwise the SINAD must stay within 0.5 dB of it.
#![cfg(target_arch = "wasm32")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::large_stack_frames,
    clippy::too_many_lines,
    clippy::cast_possible_truncation
)]

mod common;

use common::{FM_RECEIVER, PUSH_SAMPLES, audio_sinad, f32_bytes, fixture, fm_rx_cu8, sha256_hex};
use js_sys::{BigInt, Float32Array, Function, Int16Array, JSON, Object, Reflect, Uint8Array};
use sdrk_wasm::{Pipeline, backends, error_class, schema, validate, version};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::wasm_bindgen_test;

const FM_RECEIVER_GROUPED: &str = include_str!("../../../examples/flows/fm_receiver_grouped.json");
const NFM_RECEIVER: &str = include_str!("../../../examples/flows/nfm_receiver.json");
const AM_RECEIVER: &str = include_str!("../../../examples/flows/am_receiver.json");
const AIRBAND_RECEIVER: &str = include_str!("../../../examples/flows/airband_receiver.json");
const SSB_RECEIVER: &str = include_str!("../../../examples/flows/ssb_receiver.json");
const REAL_INPUT_FM_RECEIVER: &str =
    include_str!("../../../examples/flows/real_input_fm_receiver.json");
const FM_STEREO_RECEIVER: &str = include_str!("../../../examples/flows/fm_stereo_receiver.json");
const FM_TRANSMITTER: &str = include_str!("../../../examples/flows/fm_transmitter.json");
const NFM_TRANSMITTER: &str = include_str!("../../../examples/flows/nfm_transmitter.json");
const RESAMPLE_AUDIO: &str = include_str!("../../../examples/flows/resample_audio.json");
const SPECTRUM_ONLY: &str = include_str!("../../../examples/flows/spectrum_only.json");

fn get(value: &JsValue, key: &str) -> JsValue {
    Reflect::get(value, &JsValue::from_str(key)).unwrap()
}

fn string(value: &JsValue, key: &str) -> String {
    get(value, key).as_string().unwrap()
}

fn number(value: &JsValue, key: &str) -> f64 {
    get(value, key).as_f64().unwrap()
}

fn big(value: &JsValue) -> i128 {
    assert!(value.is_bigint(), "{value:?} is not a bigint");
    i128::try_from(value.clone()).unwrap()
}

/// `n` bytes of CU8 silence.
fn silence(n: usize) -> Uint8Array {
    Uint8Array::from(vec![128u8; n].as_slice())
}

fn object(fields: &[(&str, JsValue)]) -> JsValue {
    let o = Object::new();
    for (k, v) in fields {
        Reflect::set(&o, &JsValue::from_str(k), v).unwrap();
    }
    o.into()
}

fn array_len(value: &JsValue, key: &str) -> u32 {
    js_sys::Array::from(&get(value, key)).length()
}

/// Asserts that `result` failed with an `SdrkError` of this kind and returns the error.
fn expect_error<T>(result: Result<T, JsValue>, kind: &str) -> JsValue {
    let Err(error) = result else {
        panic!("the call should have failed");
    };
    assert!(error.is_instance_of::<js_sys::Error>(), "{error:?}");
    let class = error_class();
    let is_instance = Function::new_with_args("error, ctor", "return error instanceof ctor;");
    let same = is_instance.call2(&JsValue::NULL, &error, &class).unwrap();
    assert_eq!(same.as_bool(), Some(true), "not an SdrkError: {error:?}");
    assert_eq!(string(&error, "name"), "SdrkError");
    assert_eq!(
        string(&error, "kind"),
        kind,
        "{}",
        string(&error, "message")
    );
    error
}

/// Pushes the CU8 signal and returns all audio samples.
fn receive(pipeline: &mut Pipeline, cu8: &[u8]) -> Vec<f32> {
    let mut audio = Vec::new();
    let out = Float32Array::new_with_length(8192);
    for chunk in cu8.chunks(2 * PUSH_SAMPLES) {
        let data = Uint8Array::from(chunk);
        pipeline
            .push("rf", &data, &JsValue::UNDEFINED)
            .map_err(|e| string(&e, "message"))
            .unwrap();
        loop {
            let n = pipeline.read_stream("audio", &out).unwrap() as usize;
            if n == 0 {
                break;
            }
            let mut block = vec![0f32; n];
            out.subarray(0, n as u32).copy_to(&mut block);
            audio.extend_from_slice(&block);
        }
    }
    audio
}

#[wasm_bindgen_test]
fn version_and_schema() {
    assert_eq!(version(), env!("CARGO_PKG_VERSION"));
    let parsed = JSON::parse(&schema()).unwrap();
    assert!(string(&parsed, "$schema").contains("json-schema.org"));
    assert!(get(&parsed, "properties").is_object());
}

#[wasm_bindgen_test]
fn every_example_flow_validates_clean() {
    for (name, flow) in [
        ("fm_receiver", FM_RECEIVER),
        ("fm_receiver_grouped", FM_RECEIVER_GROUPED),
        ("fm_transmitter", FM_TRANSMITTER),
        ("nfm_transmitter", NFM_TRANSMITTER),
        ("nfm_receiver", NFM_RECEIVER),
        ("am_receiver", AM_RECEIVER),
        ("airband_receiver", AIRBAND_RECEIVER),
        ("ssb_receiver", SSB_RECEIVER),
        ("real_input_fm_receiver", REAL_INPUT_FM_RECEIVER),
        ("fm_stereo_receiver", FM_STEREO_RECEIVER),
        ("resample_audio", RESAMPLE_AUDIO),
        ("spectrum_only", SPECTRUM_ONLY),
    ] {
        let report = validate(flow).unwrap();
        assert_eq!(array_len(&report, "errors"), 0, "{name}");
        assert!(number(&report, "memory_bytes") > 0.0, "{name}");
    }
}

#[wasm_bindgen_test]
fn validate_reports_invalid_flows_without_throwing() {
    // A value out of range: the code and the path of the problem.
    let bad = FM_RECEIVER.replace("\"decimation\": 10", "\"decimation\": 0");
    let report = validate(&bad).unwrap();
    assert_eq!(array_len(&report, "errors"), 1);
    let first = js_sys::Array::from(&get(&report, "errors")).get(0);
    assert_eq!(string(&first, "code"), "out_of_range");
    assert_eq!(string(&first, "path"), "nodes.tune.params.decimation");

    // A syntax error is a diagnostic, too.
    let report = validate("{").unwrap();
    let first = js_sys::Array::from(&get(&report, "errors")).get(0);
    assert_eq!(string(&first, "code"), "syntax");

    // An unknown kind.
    let bad = FM_RECEIVER.replace("\"kind\": \"ddc\"", "\"kind\": \"nope\"");
    let report = validate(&bad).unwrap();
    let first = js_sys::Array::from(&get(&report, "errors")).get(0);
    assert_eq!(string(&first, "code"), "unknown_kind");
    assert_eq!(string(&first, "path"), "nodes.tune.kind");
}

#[wasm_bindgen_test]
fn building_an_invalid_flow_throws_an_sdrk_error_with_diagnostics() {
    let bad = FM_RECEIVER.replace("\"decimation\": 10", "\"decimation\": 0");
    let error = expect_error(Pipeline::from_json(&bad), "Validation");
    assert_eq!(string(&error, "code"), "out_of_range");
    assert_eq!(string(&error, "path"), "nodes.tune.params.decimation");
    assert_eq!(array_len(&error, "diagnostics"), 1);
    expect_error(Pipeline::from_json("{"), "Validation");
}

#[wasm_bindgen_test]
fn backends_names_the_backend_of_every_kernel() {
    let b = backends();
    let rotate = string(&b, "rotate");
    if cfg!(any(sdrk_fallback_only, not(target_feature = "simd128"))) {
        assert_eq!(rotate, "fallback");
        assert_eq!(string(&b, "fir"), "fallback");
    } else {
        assert_eq!(rotate, "simd");
        assert_eq!(string(&b, "fir_decimate"), "simd");
    }
    assert_eq!(string(&b, "ddc"), "chained");
    assert_eq!(string(&b, "squelch"), "fallback");
    assert_eq!(string(&b, "stereo_pll"), "fallback");
    assert_eq!(string(&b, "preemph_limit"), "fallback");
}

#[wasm_bindgen_test]
fn fm_receive_meets_sinad_and_the_reproducibility_rule() {
    let cu8 = fm_rx_cu8();
    let fixture = fixture();
    assert_eq!(sha256_hex(&cu8), fixture.input_sha256, "the input signal");
    let mut pipeline = Pipeline::from_json(FM_RECEIVER).unwrap();
    let audio = receive(&mut pipeline, &cu8);
    let sinad = audio_sinad(&audio);
    assert!(sinad >= 40.0, "SINAD {sinad} dB");
    if cfg!(sdrk_fallback_only) {
        // Fallback everywhere: the result is identical on every platform.
        assert_eq!(audio.len(), fixture.audio_samples);
        assert_eq!(sha256_hex(&f32_bytes(&audio)), fixture.audio_sha256);
    } else {
        assert!(
            sinad >= fixture.sinad_db - 0.5,
            "SINAD {sinad} dB against the fixture {} dB",
            fixture.sinad_db
        );
    }
}

#[wasm_bindgen_test]
fn spectrum_frames_and_stats_and_latency() {
    let mut pipeline = Pipeline::from_json(FM_RECEIVER).unwrap();
    let cu8 = fm_rx_cu8();
    for chunk in cu8.chunks(2 * PUSH_SAMPLES).take(2) {
        pipeline
            .push("rf", &Uint8Array::from(chunk), &JsValue::UNDEFINED)
            .unwrap();
    }
    assert!(pipeline.available("spectrum").unwrap() >= 1.0);
    let frame = Float32Array::new_with_length(1024);
    let meta = pipeline.read_frame("spectrum", &frame).unwrap();
    assert!(meta.is_object());
    assert!(get(&meta, "firstSample").is_bigint());
    assert_eq!(number(&meta, "averaged"), 4.0);
    let mut values = vec![0f32; 1024];
    frame.copy_to(&mut values);
    assert!(values.iter().any(|v| *v != 0.0));
    // A frame of the wrong length, and a stream read from a frames output.
    expect_error(
        pipeline.read_frame("spectrum", &Float32Array::new_with_length(10)),
        "LengthMismatch",
    );
    expect_error(
        pipeline.read_stream("spectrum", &Float32Array::new_with_length(10)),
        "InvalidArgument",
    );

    let stats = js_sys::Array::from(&pipeline.stats());
    assert!(stats.length() >= 4);
    let first = stats.get(0);
    assert!(get(&first, "calls").is_bigint());
    assert_ne!(string(&first, "name"), "");

    let latency = pipeline.latency("audio").unwrap();
    assert!(number(&latency, "groupDelay") > 0.0);
    assert!(number(&latency, "maxBuffered") > 0.0);

    let config = JSON::parse(&pipeline.config()).unwrap();
    assert_eq!(string(&config, "name"), "fm-receiver");
    let report = pipeline.report().unwrap();
    assert_eq!(array_len(&report, "errors"), 0);
}

#[wasm_bindgen_test]
fn wrong_typed_arrays_throw_invalid_argument() {
    let mut pipeline = Pipeline::from_json(FM_RECEIVER).unwrap();
    // The input `rf` takes raw bytes.
    let error = expect_error(
        pipeline.push("rf", &Float32Array::new_with_length(8), &JsValue::UNDEFINED),
        "InvalidArgument",
    );
    assert!(string(&error, "message").contains("Uint8Array"));
    expect_error(
        pipeline.push("rf", &Int16Array::new_with_length(8), &JsValue::UNDEFINED),
        "InvalidArgument",
    );
    expect_error(
        pipeline.push("rf", &JsValue::from_str("bytes"), &JsValue::UNDEFINED),
        "InvalidArgument",
    );
    // The output `audio` gives f32.
    expect_error(
        pipeline.read_stream("audio", &Int16Array::new_with_length(8)),
        "InvalidArgument",
    );
    // Unknown names, a chunk that is too large, an odd number of bytes in a complex pair.
    expect_error(
        pipeline.push("nope", &Uint8Array::new_with_length(2), &JsValue::UNDEFINED),
        "InvalidArgument",
    );
    expect_error(
        pipeline.push(
            "rf",
            &Uint8Array::new_with_length(100_000),
            &JsValue::UNDEFINED,
        ),
        "LengthMismatch",
    );
    expect_error(pipeline.available("nope"), "InvalidArgument");
}

#[wasm_bindgen_test]
fn a_complex_stream_is_interleaved_and_a_q15_flow_uses_int16() {
    // The spectrum-only flow takes converted samples; build a flow with a complex output.
    let flow = r#"{
      "sdrk": 1, "name": "t",
      "settings": { "precision": "q15", "max_chunk": 256 },
      "inputs": { "iq": { "type": "cq15", "rate_hz": 48000 } },
      "nodes": { "g": { "kind": "scale", "params": { "gain_db": 0 } } },
      "edges": [ { "from": "iq", "to": "g" }, { "from": "g", "to": "out" } ],
      "outputs": { "out": { "type": "stream" } }
    }"#;
    let mut pipeline = Pipeline::from_json(flow).unwrap();
    let input = Int16Array::from(&[1000i16, -2000, 3000, -4000][..]);
    pipeline.push("iq", &input, &JsValue::UNDEFINED).unwrap();
    assert_eq!(pipeline.available("out").unwrap(), 2.0);
    let out = Int16Array::new_with_length(8);
    let n = pipeline.read_stream("out", &out).unwrap();
    assert_eq!(n, 2.0);
    let mut values = [0i16; 8];
    out.copy_to(&mut values);
    assert!((i32::from(values[0]) - 1000).abs() <= 1 && (i32::from(values[3]) + 4000).abs() <= 1);
    // An odd number of array elements is not a whole number of complex samples.
    expect_error(
        pipeline.push("iq", &Int16Array::new_with_length(3), &JsValue::UNDEFINED),
        "LengthMismatch",
    );
    // A Float32Array is the wrong type for a Q15 input.
    expect_error(
        pipeline.push("iq", &Float32Array::new_with_length(4), &JsValue::UNDEFINED),
        "InvalidArgument",
    );
}

#[wasm_bindgen_test]
fn events_gaps_and_timed_parameters() {
    let mut pipeline = Pipeline::from_json(FM_RECEIVER).unwrap();
    assert!(pipeline.poll_event().is_null());

    // A timed parameter change: takes effect at input index 1000 = audio index 20.
    let at = object(&[
        ("input", JsValue::from_str("rf")),
        ("index", BigInt::from(1000u32).into()),
    ]);
    pipeline
        .set_param("tune", "frequency_hz", &JsValue::from_f64(610_000.0), &at)
        .unwrap();
    pipeline
        .push("rf", &silence(4000), &JsValue::UNDEFINED)
        .unwrap();
    let ack = pipeline.poll_event();
    assert_eq!(string(&ack, "type"), "paramApplied");
    assert_eq!(string(&ack, "node"), "tune");
    assert_eq!(string(&ack, "param"), "frequency_hz");
    assert_eq!(number(&ack, "value"), 610_000.0);
    assert_eq!(get(&ack, "late").as_bool(), Some(false));
    assert_eq!(string(&ack, "output"), "audio");
    assert_eq!(big(&get(&ack, "index")), 20);
    assert!(pipeline.poll_event().is_null());

    // An immediate change is visible in the configuration.
    pipeline
        .set_param(
            "tune",
            "frequency_hz",
            &JsValue::from_f64(620_000.0),
            &JsValue::UNDEFINED,
        )
        .unwrap();
    let config = JSON::parse(&pipeline.config()).unwrap();
    let tune = get(&get(&config, "nodes"), "tune");
    assert_eq!(number(&get(&tune, "params"), "frequency_hz"), 620_000.0);

    // Errors of set_param: unknown node, a build-time parameter, a value of the wrong kind.
    expect_error(
        pipeline.set_param(
            "nope",
            "frequency_hz",
            &JsValue::from_f64(1.0),
            &JsValue::UNDEFINED,
        ),
        "InvalidArgument",
    );
    expect_error(
        pipeline.set_param(
            "tune",
            "decimation",
            &JsValue::from_f64(5.0),
            &JsValue::UNDEFINED,
        ),
        "InvalidState",
    );
    expect_error(
        pipeline.set_param(
            "tune",
            "frequency_hz",
            &JsValue::from_str("fast"),
            &JsValue::UNDEFINED,
        ),
        "InvalidArgument",
    );
    expect_error(
        pipeline.set_param(
            "tune",
            "frequency_hz",
            &JsValue::from_f64(f64::NAN),
            &JsValue::UNDEFINED,
        ),
        "InvalidArgument",
    );
    expect_error(
        pipeline.set_param(
            "tune",
            "frequency_hz",
            &JsValue::from_f64(1.0),
            &object(&[("when", JsValue::from_f64(1.0))]),
        ),
        "InvalidArgument",
    );

    // A gap: the block starts 4000 samples after the end of the previous one.
    let meta = object(&[("start", BigInt::from(6000u32).into())]);
    pipeline.push("rf", &silence(2000), &meta).unwrap();
    let mut kinds = Vec::new();
    loop {
        let event = pipeline.poll_event();
        if event.is_null() {
            break;
        }
        kinds.push(string(&event, "type"));
        if string(&event, "type") == "gap" {
            assert_eq!(big(&get(&event, "missing")), 4000);
        }
    }
    assert!(kinds.contains(&"gap".to_string()), "{kinds:?}");
    // Without `start` a block follows the previous one: an anchor at the next index.
    let anchor = object(&[
        ("index", BigInt::from(7000u32).into()),
        ("unixNs", BigInt::from(1_700_000_000_000_000_000u64).into()),
        ("source", JsValue::from_str("gps")),
    ]);
    pipeline
        .push("rf", &silence(200), &object(&[("anchor", anchor)]))
        .unwrap();
    let mut found = false;
    loop {
        let event = pipeline.poll_event();
        if event.is_null() {
            break;
        }
        if string(&event, "type") == "anchor" {
            found = true;
            let a = get(&event, "anchor");
            assert_eq!(string(&a, "source"), "gps");
            assert!(get(&a, "unixNs").is_bigint());
        }
    }
    assert!(found, "no anchor event");
    // A source that is not one of the four names is refused.
    let bad = object(&[(
        "anchor",
        object(&[
            ("index", BigInt::from(0u32).into()),
            ("unixNs", BigInt::from(0u32).into()),
            ("source", JsValue::from_str("sun")),
        ]),
    )]);
    expect_error(pipeline.push("rf", &silence(2), &bad), "InvalidArgument");
    // The start index may not go back.
    let back = object(&[("start", BigInt::from(1u32).into())]);
    expect_error(pipeline.push("rf", &silence(2), &back), "InvalidState");

    pipeline.reset();
    assert!(pipeline.poll_event().is_null());
}

fn tone_source(
    calls: std::rc::Rc<std::cell::RefCell<Vec<(String, u32)>>>,
    total: usize,
) -> Function {
    let remaining = std::rc::Rc::new(std::cell::Cell::new(total));
    let closure = wasm_bindgen::closure::Closure::<dyn FnMut(String, u32) -> JsValue>::new(
        move |input: String, max: u32| {
            calls.borrow_mut().push((input, max));
            let n = (max as usize).min(remaining.get()).min(480);
            if n == 0 {
                return JsValue::NULL;
            }
            remaining.set(remaining.get() - n);
            let samples: Vec<f32> = (0..n).map(|i| 0.5 * ((i as f32) * 0.1).sin()).collect();
            Float32Array::from(&samples[..]).into()
        },
    );
    let function: Function = closure.as_ref().unchecked_ref::<Function>().clone();
    closure.forget();
    function
}

#[wasm_bindgen_test]
fn pull_mode_with_a_callback_source_and_bursts() {
    // `short`: the pull ends when the source is exhausted (the example fills with zeros).
    let flow = FM_TRANSMITTER.replace("\"underrun\": \"zeros\"", "\"underrun\": \"short\"");
    let mut pipeline = Pipeline::from_json(&flow).unwrap();
    let calls = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let source = tone_source(calls.clone(), 960);
    let out = Uint8Array::new_with_length(4000);

    // A burst with a transmit time: the first block carries the flag and the time.
    pipeline
        .begin_burst("rf", Some(1_700_000_000_000_000_000))
        .unwrap();
    let result = pipeline.pull("rf", &out, &source).unwrap();
    assert_eq!(number(&result, "samples"), 4000.0);
    let meta = get(&result, "meta");
    assert_eq!(big(&get(&meta, "start")), 0);
    assert_eq!(
        number(&meta, "flags") as u32 & 0b1000,
        0b1000,
        "BURST_START"
    );
    assert_eq!(big(&get(&meta, "txTimeNs")), 1_700_000_000_000_000_000);
    {
        let calls = calls.borrow();
        assert!(!calls.is_empty());
        assert_eq!(calls[0].0, "audio");
        assert!(calls[0].1 > 0);
    }
    let mut bytes = vec![0u8; 4000];
    out.copy_to(&mut bytes);
    assert!(bytes.iter().any(|b| *b != 0));

    // Drain the burst: the source is exhausted after 960 audio samples.
    let mut total = 4000.0;
    loop {
        let r = pipeline.pull("rf", &out, &source).unwrap();
        let n = number(&r, "samples");
        total += n;
        if n == 0.0 {
            break;
        }
        assert!(total < 1_000_000.0);
    }
    pipeline.end_burst("rf").unwrap();
    loop {
        let r = pipeline.pull("rf", &out, &source).unwrap();
        let n = number(&r, "samples");
        if n == 0.0 {
            break;
        }
        total += n;
    }
    // 960 audio samples * 50 = 48000 CS8 samples = 96000 bytes; the end of the burst flushes the
    // filters, which adds their tail.
    assert!(total > 90_000.0 && total < 110_000.0, "{total}");

    let mut kinds = Vec::new();
    loop {
        let event = pipeline.poll_event();
        if event.is_null() {
            break;
        }
        kinds.push(string(&event, "type"));
    }
    assert!(kinds.contains(&"burstStart".to_string()), "{kinds:?}");
    assert!(kinds.contains(&"burstEnd".to_string()), "{kinds:?}");

    // A second begin while a burst is active is an error.
    pipeline.begin_burst("rf", None).unwrap();
    expect_error(pipeline.begin_burst("rf", None), "InvalidState");
}

#[wasm_bindgen_test]
fn pull_reports_errors_of_the_source_and_of_the_arguments() {
    let mut pipeline = Pipeline::from_json(FM_TRANSMITTER).unwrap();
    let out = Uint8Array::new_with_length(1000);

    // A source that throws: the exception comes through unchanged.
    let thrower = Function::new_with_args("input, max", "throw new RangeError('source broke');");
    let error = pipeline.pull("rf", &out, &thrower).unwrap_err();
    assert_eq!(string(&error, "name"), "RangeError");
    assert_eq!(string(&error, "message"), "source broke");

    // A source that returns the wrong typed array.
    let wrong = Function::new_with_args("input, max", "return new Int16Array(4);");
    expect_error(pipeline.pull("rf", &out, &wrong), "InvalidArgument");

    // A source that returns more than asked for.
    let greedy = Function::new_with_args("input, max", "return new Float32Array(max + 1);");
    expect_error(pipeline.pull("rf", &out, &greedy), "LengthMismatch");

    // A source that is exhausted at once: zeros fill the output (underrun policy of the flow).
    let empty = Function::new_with_args("input, max", "return null;");
    let result = pipeline.pull("rf", &out, &empty).unwrap();
    assert_eq!(number(&result, "samples"), 1000.0);

    // The output array must be a Uint8Array for a raw output; a stream read is no pull.
    expect_error(
        pipeline.pull("rf", &Float32Array::new_with_length(10), &empty),
        "InvalidArgument",
    );
    expect_error(pipeline.pull("nope", &out, &empty), "InvalidArgument");
}

#[wasm_bindgen_test]
fn a_pipeline_is_freed_and_rebuilt() {
    let pipeline = Pipeline::from_json(FM_RECEIVER).unwrap();
    drop(pipeline);
    let again = Pipeline::from_json(FM_RECEIVER).unwrap();
    assert_eq!(again.chunk_size("audio").unwrap(), 4096.0);
}
