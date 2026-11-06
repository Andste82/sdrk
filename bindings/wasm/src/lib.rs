//! SDRk WebAssembly bindings: the pipeline API for JavaScript.
//!
//! The bindings only *configure*: they build a pipeline from a JSON description, set
//! parameters, push and pull data and read results and events. All signal processing stays in
//! Rust, and data crosses the boundary once per block (`push` copies the typed array into a
//! buffer that was allocated when the pipeline was built, `readStream` copies out of one).
//!
//! The module has two layers. [`engine`] is plain Rust (no JavaScript types) and is tested on
//! the host; this file converts between JavaScript values and the engine. The JavaScript
//! package (loader, builder and TypeScript declarations) is in `pkg-template/`; `cargo xtask
//! wasm-pkg` assembles it. See `docs/wasm.md`.
//!
//! The crate does not forbid `unsafe_code`: the expansion of the `wasm-bindgen` macros needs it
//! (it is allowed at the module level below, and nowhere else).
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::indexing_slicing
)]

pub mod engine;

use js_sys::{
    Array, BigInt, Float32Array, Function, Int16Array, JSON, Object, Reflect, Uint8Array,
};
use sdrk_device::{BlockMeta, ErrorKind, SampleIndex, TimeAnchor};
use sdrk_pipeline::{Packet, ParamTime};
use serde_json::Value;
use wasm_bindgen::prelude::*;

use engine::{Dtype, Dyn, Engine, Lanes, LanesMut, Layout, OutLayout, SourceBuf, WasmError};

#[wasm_bindgen(module = "/js/error.js")]
extern "C" {
    #[wasm_bindgen(js_name = makeError)]
    fn make_error(
        message: &str,
        kind: &str,
        code: Option<String>,
        path: Option<String>,
        diagnostics: JsValue,
    ) -> JsValue;

    #[wasm_bindgen(js_name = errorClass)]
    fn error_class_js() -> JsValue;
}

/// The `SdrkError` class (a subclass of `Error`) that every failing call throws.
#[wasm_bindgen(js_name = errorClass)]
#[must_use]
pub fn error_class() -> JsValue {
    error_class_js()
}

/// Called when the module is instantiated: debug builds log Rust panics to `console.error`.
#[wasm_bindgen(start)]
fn start() {
    #[cfg(debug_assertions)]
    console_error_panic_hook::set_once();
}

// ------------------------------------------------------------------ conversions

fn set(target: &Object, key: &str, value: &JsValue) {
    // `Reflect::set` on a plain object cannot fail.
    let _ = Reflect::set(target, &JsValue::from_str(key), value);
}

fn get(target: &JsValue, key: &str) -> JsValue {
    Reflect::get(target, &JsValue::from_str(key)).unwrap_or(JsValue::UNDEFINED)
}

fn dyn_to_js(value: &Dyn) -> JsValue {
    match value {
        Dyn::Null => JsValue::NULL,
        Dyn::Bool(b) => JsValue::from_bool(*b),
        Dyn::Num(n) => JsValue::from_f64(*n),
        Dyn::U64(n) => JsValue::from(*n),
        Dyn::I64(n) => JsValue::from(*n),
        Dyn::Str(s) => JsValue::from_str(s),
        Dyn::Obj(fields) => {
            let object = Object::new();
            for (key, field) in fields {
                set(&object, key, &dyn_to_js(field));
            }
            object.into()
        }
        Dyn::Arr(items) => items.iter().map(dyn_to_js).collect::<Array>().into(),
    }
}

fn error_to_js(error: &WasmError) -> JsValue {
    let diagnostics: Array = error
        .diagnostics
        .iter()
        .map(|d| {
            let o = Object::new();
            set(&o, "path", &JsValue::from_str(&d.path));
            set(&o, "code", &JsValue::from_str(d.code));
            set(&o, "message", &JsValue::from_str(&d.message));
            JsValue::from(o)
        })
        .collect();
    make_error(
        &error.message,
        &error.kind,
        error.code.map(str::to_string),
        error.path.clone(),
        diagnostics.into(),
    )
}

fn fail(error: &WasmError) -> JsValue {
    error_to_js(error)
}

fn invalid(message: impl Into<String>) -> JsValue {
    fail(&WasmError::invalid(message))
}

/// A JSON text as a JavaScript value.
fn json_to_js(text: &str) -> Result<JsValue, JsValue> {
    JSON::parse(text).map_err(|_| fail(&WasmError::new(ErrorKind::Internal, "invalid JSON")))
}

/// An unsigned 64-bit value from a `bigint` (or an integral, non-negative `number`).
fn to_u64(value: &JsValue, what: &str) -> Result<u64, JsValue> {
    if let Some(n) = value.as_f64() {
        if n.fract() == 0.0 && (0.0..9_007_199_254_740_992.0).contains(&n) {
            return Ok(n as u64);
        }
        return Err(invalid(format!("{what} must be a non-negative integer")));
    }
    u64::try_from(value.clone()).map_err(|_| invalid(format!("{what} must be a bigint")))
}

/// A signed 64-bit value from a `bigint` (or an integral `number`).
fn to_i64(value: &JsValue, what: &str) -> Result<i64, JsValue> {
    if let Some(n) = value.as_f64() {
        if n.fract() == 0.0 && n.abs() < 9_007_199_254_740_992.0 {
            return Ok(n as i64);
        }
        return Err(invalid(format!("{what} must be an integer")));
    }
    i64::try_from(value.clone()).map_err(|_| invalid(format!("{what} must be a bigint")))
}

fn present(value: &JsValue) -> bool {
    !value.is_undefined() && !value.is_null()
}

/// `{ start?, flags?, anchor?, txTimeNs? }` as a `BlockMeta`; `None` if `value` is absent. The
/// flag says whether `start` was given.
fn parse_meta(value: &JsValue) -> Result<Option<(BlockMeta, bool)>, JsValue> {
    if !present(value) {
        return Ok(None);
    }
    if !value.is_object() {
        return Err(invalid("meta must be an object"));
    }
    let mut meta = BlockMeta::default();
    let start = get(value, "start");
    let start_given = present(&start);
    if start_given {
        meta.start = SampleIndex(to_u64(&start, "meta.start")?);
    }
    let flags = get(value, "flags");
    if present(&flags) {
        let bits = flags
            .as_f64()
            .filter(|f| f.fract() == 0.0 && (0.0..=65535.0).contains(f))
            .ok_or_else(|| invalid("meta.flags must be an integer"))?;
        meta.flags = engine::flags_from(bits as u16);
    }
    let anchor = get(value, "anchor");
    if present(&anchor) {
        let source = get(&anchor, "source");
        let source = if present(&source) {
            source
                .as_string()
                .and_then(|s| engine::time_source_from(&s))
                .ok_or_else(|| invalid("meta.anchor.source must be device, host, gps or unknown"))?
        } else {
            sdrk_device::TimeSource::Unknown
        };
        meta.anchor = Some(TimeAnchor {
            index: SampleIndex(to_u64(&get(&anchor, "index"), "meta.anchor.index")?),
            unix_ns: to_i64(&get(&anchor, "unixNs"), "meta.anchor.unixNs")?,
            source,
        });
    }
    let tx = get(value, "txTimeNs");
    if present(&tx) {
        meta.tx_time_ns = Some(to_i64(&tx, "meta.txTimeNs")?);
    }
    Ok(Some((meta, start_given)))
}

/// `undefined` (now), `{ input, index }` or `{ unixNs }` as a `ParamTime`.
fn parse_time(value: &JsValue) -> Result<ParamTime, JsValue> {
    if !present(value) {
        return Ok(ParamTime::Immediate);
    }
    let unix = get(value, "unixNs");
    if present(&unix) {
        return Ok(ParamTime::UnixNs(to_i64(&unix, "at.unixNs")?));
    }
    let input = get(value, "input");
    let index = get(value, "index");
    match (input.as_string(), present(&index)) {
        (Some(input), true) => Ok(ParamTime::InputIndex {
            input,
            index: to_u64(&index, "at.index")?,
        }),
        _ => Err(invalid(
            "at must be { input, index } or { unixNs } (or undefined)",
        )),
    }
}

/// A parameter value as JSON: `number`, `boolean`, `string` (or an integral `bigint`).
fn parse_value(value: &JsValue) -> Result<Value, JsValue> {
    if let Some(n) = value.as_f64() {
        return serde_json::Number::from_f64(n)
            .map(Value::Number)
            .ok_or_else(|| invalid("the value must be a finite number"));
    }
    if let Some(b) = value.as_bool() {
        return Ok(Value::Bool(b));
    }
    if let Some(s) = value.as_string() {
        return Ok(Value::String(s));
    }
    if value.is_bigint() {
        return Ok(Value::from(to_i64(value, "the value")?));
    }
    Err(invalid("the value must be a number, boolean or string"))
}

/// The length of `value` if it is a typed array of type `want`, else a `InvalidArgument`
/// error naming `what`.
fn array_len(value: &JsValue, want: Dtype, what: &str) -> Result<usize, JsValue> {
    let len = match want {
        Dtype::U8 => value.dyn_ref::<Uint8Array>().map(Uint8Array::length),
        Dtype::I16 => value.dyn_ref::<Int16Array>().map(Int16Array::length),
        Dtype::F32 => value.dyn_ref::<Float32Array>().map(Float32Array::length),
        // Packets go through `pushPackets` and `readPackets`, never through a typed array.
        Dtype::Packets => None,
    };
    len.map(|n| n as usize)
        .ok_or_else(|| fail(&engine::wrong_dtype(what, want)))
}

/// A packet from a JavaScript object `{ index?, flags?, data }`: `index` a `bigint` or an
/// integral `number` (default 0), `flags` an integer 0 … 2^32 - 1 (default 0), `data` the
/// payload as lower-case hex. Other fields are ignored.
fn parse_packet(value: &JsValue, i: usize) -> Result<Packet, WasmError> {
    // The name is formatted only for an error: a valid packet costs no allocation here.
    let what = Fmt(i);
    let bad = |message: String| WasmError::invalid(message);
    if !value.is_object() {
        return Err(bad(format!(
            "{what} must be an object {{ index, flags, data }}"
        )));
    }
    let index = get(value, "index");
    let index = if present(&index) {
        if let Some(n) = index.as_f64() {
            if n.fract() != 0.0 || !(0.0..9_007_199_254_740_992.0).contains(&n) {
                return Err(bad(format!("{what}.index must be a non-negative integer")));
            }
            n as u64
        } else {
            u64::try_from(index).map_err(|_| bad(format!("{what}.index must be a bigint")))?
        }
    } else {
        0
    };
    let flags = get(value, "flags");
    let flags = if present(&flags) {
        flags
            .as_f64()
            .filter(|f| f.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(f))
            .ok_or_else(|| bad(format!("{what}.flags must be an integer 0 … 4294967295")))?
            as u32
    } else {
        0
    };
    let data = get(value, "data")
        .as_string()
        .ok_or_else(|| bad(format!("{what}.data must be a hex string")))?;
    engine::packet_from(index, flags, &data).map_err(|e| bad(format!("{what}: {}", e.message)))
}

/// Displays as `packets[i]`.
struct Fmt(usize);

impl std::fmt::Display for Fmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "packets[{}]", self.0)
    }
}

/// Parses the packet objects of the JavaScript array `src` into `dst` (same length).
fn packets_in(src: &Array, dst: &mut [Packet]) -> Result<(), WasmError> {
    for (i, slot) in dst.iter_mut().enumerate() {
        *slot = parse_packet(&src.get(i as u32), i)?;
    }
    Ok(())
}

/// Copies `src` into the typed array `dst` (of the matching type) from element `at` on.
fn copy_out(dst: &JsValue, at: usize, src: Lanes<'_>) {
    let (from, to) = (at as u32, (at + src.len()) as u32);
    match src {
        Lanes::U8(s) => {
            if let Some(a) = dst.dyn_ref::<Uint8Array>() {
                a.subarray(from, to).copy_from(s);
            }
        }
        Lanes::I16(s) => {
            if let Some(a) = dst.dyn_ref::<Int16Array>() {
                a.subarray(from, to).copy_from(s);
            }
        }
        Lanes::F32(s) => {
            if let Some(a) = dst.dyn_ref::<Float32Array>() {
                a.subarray(from, to).copy_from(s);
            }
        }
        // Packets leave through `readPackets` as objects.
        Lanes::Packets(_) => {}
    }
}

/// Copies the typed array `src` into `dst`; the lengths are equal (the caller checked).
fn copy_in(src: &JsValue, dst: LanesMut<'_>) {
    match dst {
        LanesMut::U8(d) => {
            if let Some(a) = src.dyn_ref::<Uint8Array>() {
                a.copy_to(d);
            }
        }
        LanesMut::I16(d) => {
            if let Some(a) = src.dyn_ref::<Int16Array>() {
                a.copy_to(d);
            }
        }
        LanesMut::F32(d) => {
            if let Some(a) = src.dyn_ref::<Float32Array>() {
                a.copy_to(d);
            }
        }
        // Packets enter through `packets_in`.
        LanesMut::Packets(_) => {}
    }
}

// ------------------------------------------------------------------ free functions

/// The version of the library.
///
/// ```js
/// console.log(version());
/// ```
#[wasm_bindgen]
#[must_use]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// The JSON Schema (draft 2020-12) of the pipeline description, as text.
#[wasm_bindgen]
#[must_use]
pub fn schema() -> String {
    Engine::schema()
}

/// Validates a pipeline description. Never throws on an invalid description: syntax errors
/// and every other problem are diagnostics of the returned report.
///
/// ```js
/// const report = validate(flowJson);
/// if (report.errors.length > 0) console.log(report.errors[0].path, report.errors[0].code);
/// ```
#[wasm_bindgen]
pub fn validate(flow_json: &str) -> Result<JsValue, JsValue> {
    json_to_js(&Engine::validate(flow_json).to_json())
}

/// The backend of every kernel in this build, for example `{ rotate: "simd" }` (`"fallback"`
/// in the scalar build).
///
/// ```js
/// console.log(backends().rotate); // "simd" or "fallback"
/// ```
#[wasm_bindgen]
#[must_use]
pub fn backends() -> JsValue {
    let object = Object::new();
    for (kernel, backend) in Engine::backends() {
        set(&object, kernel, &JsValue::from_str(backend));
    }
    object.into()
}

// ------------------------------------------------------------------ Pipeline

/// A built pipeline. Not shared between workers: use one pipeline per worker.
#[wasm_bindgen]
pub struct Pipeline {
    engine: Engine,
}

#[wasm_bindgen]
impl Pipeline {
    /// Builds a pipeline from a JSON description. Throws an `SdrkError` with the diagnostics
    /// if the description is invalid.
    ///
    /// ```js
    /// try {
    ///   const pipeline = Pipeline.fromJson(flowJson);
    /// } catch (error) {
    ///   console.log(error.kind, error.code, error.path, error.diagnostics);
    /// }
    /// ```
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(flow_json: &str) -> Result<Pipeline, JsValue> {
        Engine::from_json(flow_json)
            .map(|engine| Pipeline { engine })
            .map_err(|e| fail(&e))
    }

    /// Pushes samples into an input: raw bytes (`Uint8Array`), converted `Int16Array` (Q15) or
    /// `Float32Array` samples; complex samples are interleaved (re, im). The array type must
    /// match the input. `meta` is an optional `{ start, flags, anchor, txTimeNs }`.
    ///
    /// ```js
    /// pipeline.push("rf", new Uint8Array(8192)); // raw bytes
    /// pipeline.push("iq", new Float32Array(2048), { start: 4096n }); // 1024 complex samples
    /// ```
    pub fn push(&mut self, input: &str, data: &JsValue, meta: &JsValue) -> Result<(), JsValue> {
        let layout = self.engine.input_layout(input).map_err(|e| fail(&e))?;
        let len = array_len(data, layout.dtype, input)?;
        let meta = match parse_meta(meta)? {
            Some((mut meta, start_given)) => {
                if !start_given {
                    // Without `start` the block follows the previous one.
                    let next = self.engine.input_index(input).map_err(|e| fail(&e))?;
                    meta.start = SampleIndex(next);
                }
                Some(meta)
            }
            None => None,
        };
        let dst = self.engine.stage(input, len).map_err(|e| fail(&e))?;
        copy_in(data, dst);
        self.engine
            .push_staged(input, meta.as_ref())
            .map_err(|e| fail(&e))
    }

    /// Pushes packets into a packet input: an `Array` of `{ index, flags, data }` objects
    /// (`data` the payload as lower-case hex, at most 256 bytes; `index` and `flags` are
    /// optional and default to 0). All packets are taken, or the call throws before anything is
    /// taken.
    ///
    /// ```js
    /// pipeline.pushPackets("msg", [{ data: "68656c6c6f" }, { index: 0n, flags: 0, data: "" }]);
    /// ```
    #[wasm_bindgen(js_name = pushPackets)]
    pub fn push_packets(&mut self, input: &str, packets: &JsValue) -> Result<(), JsValue> {
        let layout = self.engine.input_layout(input).map_err(|e| fail(&e))?;
        if layout.dtype != Dtype::Packets {
            return Err(fail(&engine::wrong_dtype(input, layout.dtype)));
        }
        let array = packets
            .dyn_ref::<Array>()
            .ok_or_else(|| invalid("packets must be an Array of { index, flags, data }"))?;
        let dst = self
            .engine
            .stage(input, array.length() as usize)
            .map_err(|e| fail(&e))?;
        if let LanesMut::Packets(dst) = dst {
            packets_in(array, dst).map_err(|e| fail(&e))?;
        }
        self.engine.push_staged(input, None).map_err(|e| fail(&e))
    }

    /// Reads every packet waiting in a packet output, oldest first: an `Array` of
    /// `{ index, flags, data }` objects (`index` a `bigint`, `data` lower-case hex).
    ///
    /// ```js
    /// for (const p of pipeline.readPackets("out")) console.log(p.index, p.flags, p.data);
    /// ```
    #[wasm_bindgen(js_name = readPackets)]
    pub fn read_packets(&mut self, output: &str) -> Result<Array, JsValue> {
        let all = Array::new();
        loop {
            let got = self
                .engine
                .read_packets(output, engine::MAX_PACKET_CHUNK)
                .map_err(|e| fail(&e))?;
            if got.is_empty() {
                break;
            }
            for p in got {
                all.push(&dyn_to_js(&engine::packet_dyn(p)));
            }
        }
        Ok(all)
    }

    /// Produces data for a transmit output by asking `source(input, max)` for input data. The
    /// source returns a typed array of at most `max` samples, or `null` when it is exhausted.
    /// For a packet input it returns an `Array` of at most `max` packet objects; `null` or an
    /// empty array means "nothing to send now" and the encoder sends its idle signal.
    /// At most `chunkSize(output)` samples are produced per call. Returns
    /// `{ samples, meta }`.
    ///
    /// ```js
    /// const out = new Uint8Array(4800);
    /// const { samples, meta } = pipeline.pull("rf", out, (input, max) => nextAudio(max)); // a Float32Array, or null at the end
    /// ```
    pub fn pull(
        &mut self,
        output: &str,
        out: &JsValue,
        source: &Function,
    ) -> Result<JsValue, JsValue> {
        let layout = match self.engine.output_layout(output).map_err(|e| fail(&e))? {
            OutLayout::Stream(layout) => layout,
            OutLayout::Frames(_) => return Err(invalid(format!("`{output}` is a frames output"))),
        };
        let len = array_len(out, layout.dtype, output)?;
        let mut thrown: Option<JsValue> = None;
        let result = {
            let mut call = |input: &str, buf: SourceBuf<'_>| -> engine::Result<usize> {
                call_source(source, input, buf, &mut thrown)
            };
            self.engine
                .pull(output, len / layout.lanes, &mut call)
                .map(|pulled| {
                    copy_out(out, 0, pulled.data);
                    (pulled.samples, pulled.meta)
                })
        };
        match result {
            Ok((samples, meta)) => {
                let object = Object::new();
                set(&object, "samples", &JsValue::from_f64(samples as f64));
                set(&object, "meta", &dyn_to_js(&engine::meta_dyn(&meta)));
                Ok(object.into())
            }
            Err(e) => Err(thrown.unwrap_or_else(|| fail(&e))),
        }
    }

    /// Reads samples of a stream output into `out`; complex samples are interleaved (re, im).
    /// Returns the number of samples (bytes for a raw output) read.
    ///
    /// ```js
    /// const audio = new Float32Array(4096);
    /// const n = pipeline.readStream("audio", audio);
    /// ```
    #[wasm_bindgen(js_name = readStream)]
    pub fn read_stream(&mut self, output: &str, out: &JsValue) -> Result<f64, JsValue> {
        let layout = match self.engine.output_layout(output).map_err(|e| fail(&e))? {
            OutLayout::Stream(layout) => layout,
            OutLayout::Frames(_) => return Err(invalid(format!("`{output}` is a frames output"))),
        };
        let wanted = array_len(out, layout.dtype, output)? / layout.lanes;
        let mut done = 0usize;
        while done < wanted {
            let chunk = self
                .engine
                .read_chunk(output, wanted - done)
                .map_err(|e| fail(&e))?;
            if chunk.is_empty() {
                break;
            }
            let samples = chunk.len() / layout.lanes;
            copy_out(out, done * layout.lanes, chunk);
            done += samples;
        }
        Ok(done as f64)
    }

    /// Reads the oldest frame of a frames output into `frame` (a `Float32Array` of the frame
    /// length). Returns `{ firstSample, averaged, exponent }`, or `null` if there is no frame.
    #[wasm_bindgen(js_name = readFrame)]
    pub fn read_frame(&mut self, output: &str, frame: &JsValue) -> Result<JsValue, JsValue> {
        let OutLayout::Frames(frame_len) =
            self.engine.output_layout(output).map_err(|e| fail(&e))?
        else {
            return Err(invalid(format!("`{output}` is a stream output")));
        };
        let len = array_len(frame, Dtype::F32, output)?;
        if len != frame_len {
            return Err(fail(&WasmError::new(
                ErrorKind::LengthMismatch,
                format!("buffer length mismatch: expected {frame_len}, got {len}"),
            )));
        }
        let Some(read) = self.engine.read_frame(output).map_err(|e| fail(&e))? else {
            return Ok(JsValue::NULL);
        };
        copy_out(frame, 0, Lanes::F32(read.data));
        let meta = Dyn::Obj(vec![
            ("firstSample", Dyn::U64(read.meta.first_sample)),
            ("averaged", Dyn::Num(f64::from(read.meta.averaged))),
            ("exponent", Dyn::Num(f64::from(read.meta.exponent))),
        ]);
        Ok(dyn_to_js(&meta))
    }

    /// Samples (bytes for a raw output, frames for a frames output) waiting in an output.
    pub fn available(&self, output: &str) -> Result<f64, JsValue> {
        self.engine
            .available(output)
            .map(|n| n as f64)
            .map_err(|e| fail(&e))
    }

    /// Number of samples one `readStream` or `pull` call moves at most for an output.
    #[wasm_bindgen(js_name = chunkSize)]
    pub fn chunk_size(&self, output: &str) -> Result<f64, JsValue> {
        self.engine
            .chunk(output)
            .map(|n| n as f64)
            .map_err(|e| fail(&e))
    }

    /// Number of samples (frames) the drop policy of an output has discarded.
    pub fn dropped(&self, output: &str) -> Result<BigInt, JsValue> {
        let n = self.engine.dropped(output).map_err(|e| fail(&e))?;
        Ok(BigInt::from(n))
    }

    /// Changes a run-time parameter of a node. `at` is `undefined` (at the next sample),
    /// `{ input, index }` (at a sample index of an input) or `{ unixNs }` (at an absolute time).
    ///
    /// ```js
    /// pipeline.setParam("tune", "frequency_hz", 610_000);
    /// pipeline.setParam("tune", "frequency_hz", 620_000, { input: "rf", index: 1_000_000n });
    /// ```
    #[wasm_bindgen(js_name = setParam)]
    pub fn set_param(
        &mut self,
        node: &str,
        name: &str,
        value: &JsValue,
        at: &JsValue,
    ) -> Result<(), JsValue> {
        let value = parse_value(value)?;
        let at = parse_time(at)?;
        self.engine
            .set_param(node, name, value, at)
            .map_err(|e| fail(&e))
    }

    /// The oldest event, or `null`.
    ///
    /// ```js
    /// for (let e; (e = pipeline.pollEvent()); ) console.log(e.type, e.output, e.index);
    /// ```
    #[wasm_bindgen(js_name = pollEvent)]
    pub fn poll_event(&mut self) -> JsValue {
        self.engine
            .poll_event()
            .map_or(JsValue::NULL, |event| dyn_to_js(&event))
    }

    /// Marks the start of a burst on a transmit output, optionally with a transmit time.
    #[wasm_bindgen(js_name = beginBurst)]
    pub fn begin_burst(&mut self, output: &str, tx_time_ns: Option<i64>) -> Result<(), JsValue> {
        self.engine
            .begin_burst(output, tx_time_ns)
            .map_err(|e| fail(&e))
    }

    /// Marks the end of a burst.
    #[wasm_bindgen(js_name = endBurst)]
    pub fn end_burst(&mut self, output: &str) -> Result<(), JsValue> {
        self.engine.end_burst(output).map_err(|e| fail(&e))
    }

    /// `{ groupDelay, maxBuffered }` of an output, in samples of the output.
    pub fn latency(&self, output: &str) -> Result<JsValue, JsValue> {
        let (delay, buffered) = self.engine.latency(output).map_err(|e| fail(&e))?;
        let object = Object::new();
        set(&object, "groupDelay", &JsValue::from_f64(delay));
        set(&object, "maxBuffered", &JsValue::from_f64(buffered as f64));
        Ok(object.into())
    }

    /// The configuration as canonical JSON text (run-time parameters at their current values).
    #[must_use]
    pub fn config(&self) -> String {
        self.engine.config()
    }

    /// The report of the build (figures and diagnostics).
    pub fn report(&self) -> Result<JsValue, JsValue> {
        json_to_js(&self.engine.report().to_json())
    }

    /// Statistics of every node.
    #[must_use]
    pub fn stats(&self) -> JsValue {
        let items: Array = self.engine.stats().iter().map(dyn_to_js).collect();
        items.into()
    }

    /// Resets the state of all nodes and discards queued data and events.
    pub fn reset(&mut self) {
        self.engine.reset();
    }
}

/// Calls the JavaScript source for one input and copies what it returns into `buf`.
fn call_source(
    source: &Function,
    input: &str,
    buf: SourceBuf<'_>,
    thrown: &mut Option<JsValue>,
) -> engine::Result<usize> {
    let returned = source
        .call2(
            &JsValue::NULL,
            &JsValue::from_str(input),
            &JsValue::from_f64(buf.max_samples as f64),
        )
        .map_err(|e| {
            *thrown = Some(e);
            WasmError::new(ErrorKind::InvalidState, "the source function threw")
        })?;
    if !present(&returned) {
        return Ok(0);
    }
    let Layout { dtype, lanes } = buf.layout;
    if let LanesMut::Packets(dst) = buf.lanes {
        let array = returned.dyn_ref::<Array>().ok_or_else(|| {
            WasmError::invalid(format!(
                "the source must return an Array of packets (or null) for `{input}`"
            ))
        })?;
        let len = array.length() as usize;
        let dst = dst.get_mut(..len).ok_or_else(|| {
            WasmError::new(
                ErrorKind::LengthMismatch,
                format!(
                    "the source returned {len} packets for `{input}`; at most {} were asked for",
                    buf.max_samples
                ),
            )
        })?;
        packets_in(array, dst)?;
        return Ok(len);
    }
    let len = match dtype {
        Dtype::U8 => returned.dyn_ref::<Uint8Array>().map(Uint8Array::length),
        Dtype::I16 => returned.dyn_ref::<Int16Array>().map(Int16Array::length),
        Dtype::F32 => returned.dyn_ref::<Float32Array>().map(Float32Array::length),
        Dtype::Packets => None,
    }
    .ok_or_else(|| engine::wrong_dtype(input, dtype))? as usize;
    if !len.is_multiple_of(lanes) || len / lanes > buf.max_samples {
        return Err(WasmError::new(
            ErrorKind::LengthMismatch,
            format!(
                "the source returned {len} array elements for `{input}`; at most {} samples \
                 ({} elements) were asked for",
                buf.max_samples,
                buf.max_samples * lanes
            ),
        ));
    }
    match buf.lanes {
        LanesMut::U8(d) => copy_in_prefix(&returned, LanesMut::U8(d), len),
        LanesMut::I16(d) => copy_in_prefix(&returned, LanesMut::I16(d), len),
        LanesMut::F32(d) => copy_in_prefix(&returned, LanesMut::F32(d), len),
        LanesMut::Packets(_) => {}
    }
    Ok(len / lanes)
}

/// Copies the typed array `src` of `len` elements into the first `len` elements of `dst`.
fn copy_in_prefix(src: &JsValue, dst: LanesMut<'_>, len: usize) {
    match dst {
        LanesMut::U8(d) => {
            if let Some(d) = d.get_mut(..len) {
                copy_in(src, LanesMut::U8(d));
            }
        }
        LanesMut::I16(d) => {
            if let Some(d) = d.get_mut(..len) {
                copy_in(src, LanesMut::I16(d));
            }
        }
        LanesMut::F32(d) => {
            if let Some(d) = d.get_mut(..len) {
                copy_in(src, LanesMut::F32(d));
            }
        }
        LanesMut::Packets(_) => {}
    }
}

/// Character set argument of the RDS helpers: `"rds"` (default), `"utf8"` or `"auto"`.
fn rds_charset(name: Option<&str>) -> Result<sdrk_proto::rds::Charset, JsValue> {
    engine::charset(name).map_err(|e| fail(&e))
}

/// RDS bytes as text: `charset` `"rds"` (the RDS table of IEC 62106, default), `"utf8"` or
/// `"auto"` (UTF-8 when the bytes are valid UTF-8 with a multi-byte sequence).
///
/// ```js
/// rdsDecodeText(new Uint8Array([0x57, 0x99, 0x72]), "rds"); // "Wür"
/// ```
#[wasm_bindgen(js_name = rdsDecodeText)]
#[allow(
    clippy::needless_pass_by_value,
    reason = "wasm-bindgen hands an optional JavaScript string over as an owned value"
)]
pub fn rds_decode_text(bytes: &[u8], charset: Option<String>) -> Result<String, JsValue> {
    Ok(engine::rds_decode_text(
        bytes,
        rds_charset(charset.as_deref())?,
    ))
}

/// Text as RDS bytes (the RDS table of IEC 62106); throws for a character without a code.
///
/// ```js
/// rdsEncodeText("Württemberg"); // Uint8Array [0x57, 0x99, ...]
/// ```
#[wasm_bindgen(js_name = rdsEncodeText)]
pub fn rds_encode_text(text: &str) -> Result<Vec<u8>, JsValue> {
    engine::rds_encode_text(text).map_err(|e| fail(&e))
}

/// Station data from the RDS groups of an `rds_decode` output: `sdrk_proto::rds::RdsState`
/// (PI, PTY, TP, the programme service name and the radiotext once complete).
///
/// ```js
/// const rds = new RdsDecoder();
/// rds.push(pipeline.readPackets("rds"));
/// console.log(rds.ps("auto"), rds.rt("auto"), rds.pi());
/// ```
#[wasm_bindgen]
#[derive(Default)]
pub struct RdsDecoder {
    inner: engine::RdsDecoder,
}

#[wasm_bindgen]
impl RdsDecoder {
    /// An empty decoder.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> RdsDecoder {
        RdsDecoder::default()
    }

    /// Takes an `Array` of packets (`{ index, flags, data }`, as `readPackets` returns them);
    /// packets that are no RDS group are ignored. Returns the number of packets taken.
    pub fn push(&mut self, packets: &JsValue) -> Result<f64, JsValue> {
        let array = packets
            .dyn_ref::<Array>()
            .ok_or_else(|| invalid("packets must be an Array of { index, flags, data }"))?;
        for i in 0..array.length() {
            let p = parse_packet(&array.get(i), i as usize).map_err(|e| fail(&e))?;
            self.inner.push(&p);
        }
        Ok(f64::from(array.length()))
    }

    /// Forgets the station and the counts (call it after tuning to another station).
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// The programme identification, or `undefined`.
    pub fn pi(&self) -> Option<u16> {
        self.inner.state().pi()
    }

    /// The programme type 0 … 31, or `undefined`.
    pub fn pty(&self) -> Option<u8> {
        self.inner.state().pty()
    }

    /// The traffic programme flag, or `undefined`.
    pub fn tp(&self) -> Option<bool> {
        self.inner.state().tp()
    }

    /// The programme service name once all four segments arrived, else `undefined`.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "wasm-bindgen hands an optional JavaScript string over as an owned value"
    )]
    pub fn ps(&self, charset: Option<String>) -> Result<Option<String>, JsValue> {
        Ok(self.inner.ps(rds_charset(charset.as_deref())?))
    }

    /// The radiotext once complete, else `undefined`.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "wasm-bindgen hands an optional JavaScript string over as an owned value"
    )]
    pub fn rt(&self, charset: Option<String>) -> Result<Option<String>, JsValue> {
        Ok(self.inner.rt(rds_charset(charset.as_deref())?))
    }

    /// The programme service name as far as it arrived, with spaces for the segments not
    /// received yet, for a display that fills while they come in; `undefined` before the first.
    #[wasm_bindgen(js_name = psPartial)]
    #[allow(
        clippy::needless_pass_by_value,
        reason = "wasm-bindgen hands an optional JavaScript string over as an owned value"
    )]
    pub fn ps_partial(&self, charset: Option<String>) -> Result<Option<String>, JsValue> {
        Ok(self.inner.ps_partial(rds_charset(charset.as_deref())?))
    }

    /// The radiotext as far as it arrived (up to a received carriage return), with spaces for
    /// the segments not received yet; `undefined` before the first.
    #[wasm_bindgen(js_name = rtPartial)]
    #[allow(
        clippy::needless_pass_by_value,
        reason = "wasm-bindgen hands an optional JavaScript string over as an owned value"
    )]
    pub fn rt_partial(&self, charset: Option<String>) -> Result<Option<String>, JsValue> {
        Ok(self.inner.rt_partial(rds_charset(charset.as_deref())?))
    }

    /// Groups taken.
    pub fn groups(&self) -> f64 {
        self.inner.groups() as f64
    }

    /// Groups taken with `CRC_OK`.
    #[wasm_bindgen(js_name = okGroups)]
    pub fn ok_groups(&self) -> f64 {
        self.inner.ok_groups() as f64
    }
}
