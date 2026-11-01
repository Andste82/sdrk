//! The JavaScript-free core of the bindings.
//!
//! [`Engine`] wraps a [`Pipeline`] together with the buffers that the JavaScript layer copies
//! typed arrays through: one reusable buffer per input and per output, allocated once when the
//! pipeline is built. Nothing here touches `wasm-bindgen`, so the whole data path runs (and is
//! tested) on the host as well; `lib.rs` only converts between JavaScript values and these
//! types.

use std::collections::BTreeMap;

use sdrk::{C32, CQ15, Q15};
use sdrk_blocks::{FrameMeta, ParamKind, ParamValue, PortType};
use sdrk_device::{BlockFlags, BlockMeta, ErrorKind, SdrkError, TimeAnchor, TimeSource};
use sdrk_pipeline::{
    BuildOptions, Event, EventKind, InputData, ParamTime, Pipeline, PortKind, PullSource, Registry,
    Report,
};
use serde_json::Value;

/// Largest number of elements one `readStream` or `pull` call moves through the output buffer.
pub const MAX_OUT_CHUNK: usize = 65_536;

/// One problem of a description, as reported to JavaScript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diag {
    /// Location, for example `nodes.tune.params.frequency_hz`.
    pub path: String,
    /// The machine-readable code (`DiagCode`), for example `out_of_range`.
    pub code: &'static str,
    /// Human-readable explanation.
    pub message: String,
}

/// An error of the bindings: the fields of the JavaScript `SdrkError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmError {
    /// Name of the `ErrorKind`, for example `"InvalidArgument"`.
    pub kind: String,
    /// `DiagCode` of the first problem of a failed build.
    pub code: Option<&'static str>,
    /// Path of the first problem of a failed build.
    pub path: Option<String>,
    /// Human-readable message.
    pub message: String,
    /// All problems of a failed build.
    pub diagnostics: Vec<Diag>,
}

impl WasmError {
    /// An error of the given kind without diagnostics.
    ///
    /// ```
    /// use sdrk_device::ErrorKind;
    /// let error = sdrk_wasm::engine::WasmError::new(ErrorKind::InvalidState, "no burst");
    /// assert_eq!((error.kind.as_str(), error.code), ("InvalidState", None));
    /// assert_eq!(error.to_string(), "InvalidState: no burst");
    /// ```
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind: format!("{kind:?}"),
            code: None,
            path: None,
            message: message.into(),
            diagnostics: Vec::new(),
        }
    }

    /// A wrong argument (a typed array of the wrong type, for example).
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidArgument, message)
    }
}

impl std::fmt::Display for WasmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl std::error::Error for WasmError {}

impl From<sdrk_pipeline::Error> for WasmError {
    fn from(error: sdrk_pipeline::Error) -> Self {
        let mut out = Self::new(error.kind(), error.to_string());
        if let sdrk_pipeline::Error::Build(report) = &error {
            out.diagnostics = report
                .errors
                .iter()
                .map(|d| Diag {
                    path: d.path.clone(),
                    code: d.code.as_str(),
                    message: d.message.clone(),
                })
                .collect();
            if let Some(first) = report.errors.first() {
                out.code = Some(first.code.as_str());
                out.path = Some(first.path.clone());
            }
        }
        out
    }
}

/// Result alias of the bindings.
pub type Result<T, E = WasmError> = core::result::Result<T, E>;

/// The element type of the JavaScript typed array that carries a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// `Uint8Array`: raw bytes.
    U8,
    /// `Int16Array`: Q15 samples (complex ones interleaved).
    I16,
    /// `Float32Array`: `f32` samples (complex ones interleaved).
    F32,
}

impl Dtype {
    /// The name of the JavaScript class.
    pub const fn array_name(self) -> &'static str {
        match self {
            Self::U8 => "Uint8Array",
            Self::I16 => "Int16Array",
            Self::F32 => "Float32Array",
        }
    }
}

/// The typed-array layout of a stream port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    /// The typed array that carries the port.
    pub dtype: Dtype,
    /// Array elements per sample: 2 for complex samples (re, im), else 1.
    pub lanes: usize,
}

/// What an output delivers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutLayout {
    /// A stream.
    Stream(Layout),
    /// Frames of this many `f32` values.
    Frames(usize),
}

fn layout_of(ty: PortType) -> Option<Layout> {
    let (dtype, lanes) = match ty {
        PortType::Raw => (Dtype::U8, 1),
        PortType::C32 => (Dtype::F32, 2),
        PortType::CQ15 => (Dtype::I16, 2),
        PortType::F32 => (Dtype::F32, 1),
        PortType::Q15 => (Dtype::I16, 1),
        _ => return None,
    };
    Some(Layout { dtype, lanes })
}

/// A mutable view of typed-array elements.
#[derive(Debug)]
pub enum LanesMut<'a> {
    /// Bytes.
    U8(&'a mut [u8]),
    /// 16-bit integers.
    I16(&'a mut [i16]),
    /// Single-precision floats.
    F32(&'a mut [f32]),
}

/// A shared view of typed-array elements.
#[derive(Debug, Clone, Copy)]
pub enum Lanes<'a> {
    /// Bytes.
    U8(&'a [u8]),
    /// 16-bit integers.
    I16(&'a [i16]),
    /// Single-precision floats.
    F32(&'a [f32]),
}

impl Lanes<'_> {
    /// Number of array elements.
    pub fn len(&self) -> usize {
        match self {
            Self::U8(s) => s.len(),
            Self::I16(s) => s.len(),
            Self::F32(s) => s.len(),
        }
    }

    /// True if there are none.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A typed buffer of samples.
enum Buf {
    Raw(Vec<u8>),
    C32(Vec<C32>),
    CQ15(Vec<CQ15>),
    F32(Vec<f32>),
    Q15(Vec<Q15>),
}

impl Buf {
    fn new(ty: PortType, elems: usize) -> Option<Self> {
        Some(match ty {
            PortType::Raw => Self::Raw(vec![0; elems]),
            PortType::C32 => Self::C32(vec![C32::default(); elems]),
            PortType::CQ15 => Self::CQ15(vec![CQ15::default(); elems]),
            PortType::F32 => Self::F32(vec![0.0; elems]),
            PortType::Q15 => Self::Q15(vec![Q15::default(); elems]),
            _ => return None,
        })
    }

    /// The first `elems` samples as array elements; `None` if the buffer is shorter.
    fn lanes_mut(&mut self, elems: usize) -> Option<LanesMut<'_>> {
        Some(match self {
            Self::Raw(v) => LanesMut::U8(v.get_mut(..elems)?),
            Self::C32(v) => LanesMut::F32(bytemuck::cast_slice_mut(v.get_mut(..elems)?)),
            Self::CQ15(v) => LanesMut::I16(bytemuck::cast_slice_mut(v.get_mut(..elems)?)),
            Self::F32(v) => LanesMut::F32(v.get_mut(..elems)?),
            Self::Q15(v) => LanesMut::I16(bytemuck::cast_slice_mut(v.get_mut(..elems)?)),
        })
    }

    fn lanes(&self, elems: usize) -> Option<Lanes<'_>> {
        Some(match self {
            Self::Raw(v) => Lanes::U8(v.get(..elems)?),
            Self::C32(v) => Lanes::F32(bytemuck::cast_slice(v.get(..elems)?)),
            Self::CQ15(v) => Lanes::I16(bytemuck::cast_slice(v.get(..elems)?)),
            Self::F32(v) => Lanes::F32(v.get(..elems)?),
            Self::Q15(v) => Lanes::I16(bytemuck::cast_slice(v.get(..elems)?)),
        })
    }

    fn input(&self, elems: usize) -> Option<InputData<'_>> {
        Some(match self {
            Self::Raw(v) => InputData::Raw(v.get(..elems)?),
            Self::C32(v) => InputData::C32(v.get(..elems)?),
            Self::CQ15(v) => InputData::CQ15(v.get(..elems)?),
            Self::F32(v) => InputData::F32(v.get(..elems)?),
            Self::Q15(v) => InputData::Q15(v.get(..elems)?),
        })
    }

    fn len(&self) -> usize {
        match self {
            Self::Raw(v) => v.len(),
            Self::C32(v) => v.len(),
            Self::CQ15(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::Q15(v) => v.len(),
        }
    }
}

struct InputBuf {
    layout: Layout,
    max_len: usize,
    buf: Buf,
    /// Samples staged by the last `stage`.
    staged: usize,
}

struct OutputBuf {
    layout: OutLayout,
    /// Stream: samples (bytes for a raw output) per call. Frames: unused.
    chunk: usize,
    buf: Buf,
}

/// A dynamically typed value for the results that JavaScript receives as objects.
#[derive(Debug, Clone, PartialEq)]
pub enum Dyn {
    /// `null`.
    Null,
    /// A boolean.
    Bool(bool),
    /// A JavaScript `number`.
    Num(f64),
    /// A JavaScript `bigint` from an unsigned 64-bit value.
    U64(u64),
    /// A JavaScript `bigint` from a signed 64-bit value.
    I64(i64),
    /// A string.
    Str(String),
    /// An object with ordered keys.
    Obj(Vec<(&'static str, Dyn)>),
    /// An array.
    Arr(Vec<Dyn>),
}

impl Dyn {
    /// The field `key` of an object.
    ///
    /// ```
    /// use sdrk_wasm::engine::Dyn;
    /// let value = Dyn::Obj(vec![("index", Dyn::U64(7))]);
    /// assert_eq!(value.get("index"), Some(&Dyn::U64(7)));
    /// assert_eq!(value.get("other"), None);
    /// ```
    pub fn get(&self, key: &str) -> Option<&Dyn> {
        match self {
            Self::Obj(fields) => fields.iter().find(|(k, _)| *k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

fn time_source_name(source: TimeSource) -> &'static str {
    match source {
        TimeSource::Device => "device",
        TimeSource::Host => "host",
        TimeSource::Gps => "gps",
        _ => "unknown",
    }
}

/// The time source named `name` (`"device"`, `"host"`, `"gps"` or `"unknown"`).
///
/// ```
/// use sdrk_device::TimeSource;
/// assert_eq!(sdrk_wasm::engine::time_source_from("gps"), Some(TimeSource::Gps));
/// assert_eq!(sdrk_wasm::engine::time_source_from("moon"), None);
/// ```
pub fn time_source_from(name: &str) -> Option<TimeSource> {
    Some(match name {
        "device" => TimeSource::Device,
        "host" => TimeSource::Host,
        "gps" => TimeSource::Gps,
        "unknown" => TimeSource::Unknown,
        _ => return None,
    })
}

fn anchor_dyn(a: &TimeAnchor) -> Dyn {
    Dyn::Obj(vec![
        ("index", Dyn::U64(a.index.0)),
        ("unixNs", Dyn::I64(a.unix_ns)),
        ("source", Dyn::Str(time_source_name(a.source).to_string())),
    ])
}

/// A `BlockMeta` as the JavaScript object `{ start, flags, anchor?, txTimeNs? }`.
///
/// ```
/// use sdrk_device::{BlockMeta, SampleIndex};
/// use sdrk_wasm::engine::{meta_dyn, Dyn};
/// let meta = BlockMeta { start: SampleIndex(5), ..BlockMeta::default() };
/// assert_eq!(meta_dyn(&meta).get("start"), Some(&Dyn::U64(5)));
/// ```
pub fn meta_dyn(meta: &BlockMeta) -> Dyn {
    let mut fields = vec![
        ("start", Dyn::U64(meta.start.0)),
        ("flags", Dyn::Num(f64::from(meta.flags.bits()))),
    ];
    if let Some(a) = &meta.anchor {
        fields.push(("anchor", anchor_dyn(a)));
    }
    if let Some(t) = meta.tx_time_ns {
        fields.push(("txTimeNs", Dyn::I64(t)));
    }
    Dyn::Obj(fields)
}

/// The `BlockFlags` of a JavaScript number (unknown bits are dropped).
///
/// ```
/// assert_eq!(sdrk_wasm::engine::flags_from(0xFFFF).bits(), 0x7F);
/// ```
pub fn flags_from(bits: u16) -> BlockFlags {
    BlockFlags::from_bits_truncate(bits)
}

/// What a `pull` call delivered.
#[derive(Debug)]
pub struct Pulled<'a> {
    /// The samples produced (a prefix of the output buffer).
    pub data: Lanes<'a>,
    /// Number of samples (bytes for a raw output).
    pub samples: usize,
    /// Metadata of the first sample.
    pub meta: BlockMeta,
}

/// A frame that was read.
#[derive(Debug)]
pub struct FrameRead<'a> {
    /// The values of the frame.
    pub data: &'a [f32],
    /// Where the frame came from.
    pub meta: FrameMeta,
}

/// The buffer a pull source fills.
#[derive(Debug)]
pub struct SourceBuf<'a> {
    /// The typed-array layout of the input.
    pub layout: Layout,
    /// Most samples the source may supply.
    pub max_samples: usize,
    /// Room for `max_samples` samples.
    pub lanes: LanesMut<'a>,
}

/// The source of a pull call: fills the buffer for input `name` and returns the number of
/// samples written; 0 means exhausted.
pub type SourceFn<'a> = dyn FnMut(&str, SourceBuf<'_>) -> Result<usize> + 'a;

struct Adapter<'a, 'b> {
    inputs: &'a mut BTreeMap<String, InputBuf>,
    call: &'a mut SourceFn<'b>,
    failure: &'a mut Option<WasmError>,
}

impl PullSource for Adapter<'_, '_> {
    fn provide(
        &mut self,
        input: &str,
        max: usize,
    ) -> core::result::Result<InputData<'_>, sdrk_pipeline::PullError> {
        let slot = self
            .inputs
            .get_mut(input)
            .ok_or_else(|| sdrk_pipeline::Error::UnknownInput(input.to_string()))?;
        let max = max.min(slot.max_len);
        let filled = {
            let lanes = slot
                .buf
                .lanes_mut(max)
                .ok_or(sdrk_pipeline::Error::Internal("source buffer"))?;
            (self.call)(
                input,
                SourceBuf {
                    layout: slot.layout,
                    max_samples: max,
                    lanes,
                },
            )
        };
        let n = match filled {
            Ok(n) => n.min(max),
            Err(e) => {
                let message = e.message.clone();
                *self.failure = Some(e);
                return Err(sdrk_pipeline::Error::InvalidState(message));
            }
        };
        slot.buf
            .input(n)
            .ok_or(sdrk_pipeline::Error::Internal("source buffer"))
    }
}

/// A pipeline with the buffers of the bindings.
pub struct Engine {
    pipeline: Pipeline,
    registry: Registry,
    inputs: BTreeMap<String, InputBuf>,
    outputs: BTreeMap<String, OutputBuf>,
    /// Kind of every user node, to name the parameters of `ParamApplied` events.
    kinds: BTreeMap<String, String>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("pipeline", &self.pipeline)
            .finish_non_exhaustive()
    }
}

fn internal(what: &'static str) -> WasmError {
    WasmError::new(ErrorKind::Internal, what)
}

/// One buffer per input, sized for the largest push.
fn input_buffers<'a>(
    pipeline: &Pipeline,
    names: impl Iterator<Item = &'a String>,
) -> Result<BTreeMap<String, InputBuf>> {
    let mut inputs = BTreeMap::new();
    for name in names {
        let (ty, max_len) = pipeline.input_port(name)?;
        let layout = layout_of(ty).ok_or_else(|| internal("input type"))?;
        let buf = Buf::new(ty, max_len).ok_or_else(|| internal("input type"))?;
        let slot = InputBuf {
            layout,
            max_len,
            buf,
            staged: 0,
        };
        inputs.insert(name.clone(), slot);
    }
    Ok(inputs)
}

/// One buffer per output: `out_chunk` samples for a stream, one frame for a frames output.
fn output_buffers<'a>(
    pipeline: &Pipeline,
    names: impl Iterator<Item = &'a String>,
    out_chunk: usize,
) -> Result<BTreeMap<String, OutputBuf>> {
    let mut outputs = BTreeMap::new();
    for name in names {
        let slot = match pipeline.output_port(name)? {
            PortKind::Stream(ty) => {
                let layout = layout_of(ty).ok_or_else(|| internal("output type"))?;
                // A raw output counts bytes: leave room for a few bytes per sample.
                let chunk = if ty == PortType::Raw {
                    out_chunk.saturating_mul(4)
                } else {
                    out_chunk
                };
                OutputBuf {
                    layout: OutLayout::Stream(layout),
                    chunk,
                    buf: Buf::new(ty, chunk).ok_or_else(|| internal("output type"))?,
                }
            }
            PortKind::Frames { len } => OutputBuf {
                layout: OutLayout::Frames(len),
                chunk: 0,
                buf: Buf::F32(vec![0.0; len]),
            },
        };
        outputs.insert(name.clone(), slot);
    }
    Ok(outputs)
}

/// Reads up to `n` samples of the stream output into `buf`.
fn read_into(
    pipeline: &mut Pipeline,
    output: &str,
    buf: &mut Buf,
    n: usize,
) -> core::result::Result<usize, sdrk_pipeline::Error> {
    match buf {
        Buf::Raw(v) => pipeline.read_raw(output, v.get_mut(..n).unwrap_or_default()),
        Buf::C32(v) => pipeline.read_stream(output, v.get_mut(..n).unwrap_or_default()),
        Buf::CQ15(v) => pipeline.read_stream(output, v.get_mut(..n).unwrap_or_default()),
        Buf::F32(v) => pipeline.read_stream(output, v.get_mut(..n).unwrap_or_default()),
        Buf::Q15(v) => pipeline.read_stream(output, v.get_mut(..n).unwrap_or_default()),
    }
}

/// Pulls up to `n` samples of the stream output into `buf`.
fn pull_into(
    pipeline: &mut Pipeline,
    output: &str,
    buf: &mut Buf,
    n: usize,
    source: &mut dyn PullSource,
) -> core::result::Result<(usize, BlockMeta), sdrk_pipeline::Error> {
    match buf {
        Buf::Raw(v) => pipeline.pull(output, v.get_mut(..n).unwrap_or_default(), source),
        Buf::C32(v) => pipeline.pull_samples(output, v.get_mut(..n).unwrap_or_default(), source),
        Buf::CQ15(v) => pipeline.pull_samples(output, v.get_mut(..n).unwrap_or_default(), source),
        Buf::F32(v) => pipeline.pull_samples(output, v.get_mut(..n).unwrap_or_default(), source),
        Buf::Q15(v) => pipeline.pull_samples(output, v.get_mut(..n).unwrap_or_default(), source),
    }
}

impl Engine {
    /// Builds a pipeline from a JSON description and allocates the buffers of the data path.
    ///
    /// ```
    /// use sdrk_wasm::engine::Engine;
    /// let flow = include_str!("../../../examples/flows/fm_receiver.json");
    /// let engine = Engine::from_json(flow)?;
    /// assert_eq!(engine.chunk("audio")?, 4096);
    /// assert_eq!(Engine::from_json("{").unwrap_err().kind, "Validation");
    /// # Ok::<(), sdrk_wasm::engine::WasmError>(())
    /// ```
    ///
    /// # Errors
    /// A [`WasmError`] of kind `Validation` with the diagnostics if the description is invalid.
    #[allow(
        clippy::large_stack_frames,
        reason = "clippy sums the unoptimised temporaries of every `?` (the pipeline is 640 bytes); the real frame is small"
    )]
    pub fn from_json(json: &str) -> Result<Self> {
        let registry = Registry::standard();
        let pipeline = registry.build_json(json, &BuildOptions::default())?;
        let spec = pipeline.config();
        let out_chunk = spec.settings.max_chunk.clamp(1, MAX_OUT_CHUNK);
        let inputs = input_buffers(&pipeline, spec.inputs.keys())?;
        let outputs = output_buffers(&pipeline, spec.outputs.keys(), out_chunk)?;
        let kinds = spec
            .nodes
            .iter()
            .map(|(name, node)| (name.clone(), node.kind.clone()))
            .collect();
        Ok(Self {
            pipeline,
            registry,
            inputs,
            outputs,
            kinds,
        })
    }

    /// Validates a description; never fails (syntax errors become diagnostics).
    ///
    /// ```
    /// use sdrk_wasm::engine::Engine;
    /// assert!(Engine::validate(include_str!("../../../examples/flows/fm_receiver.json")).is_ok());
    /// assert_eq!(Engine::validate("{").errors[0].code.as_str(), "syntax");
    /// ```
    pub fn validate(json: &str) -> Report {
        Registry::standard().validate_json(json, &BuildOptions::default())
    }

    /// The JSON Schema of the description, as text.
    ///
    /// ```
    /// let schema: serde_json::Value = serde_json::from_str(&sdrk_wasm::engine::Engine::schema())?;
    /// assert!(schema["properties"].is_object());
    /// # Ok::<(), serde_json::Error>(())
    /// ```
    pub fn schema() -> String {
        serde_json::to_string_pretty(&Registry::standard().json_schema()).unwrap_or_default()
    }

    /// The backend of every kernel in this build, sorted by kernel name.
    ///
    /// ```
    /// let backends = sdrk_wasm::engine::Engine::backends();
    /// assert!(backends.contains(&("ddc", "chained")));
    /// assert!(backends.iter().any(|(kernel, _)| *kernel == "rotate"));
    /// ```
    pub fn backends() -> Vec<(&'static str, &'static str)> {
        use sdrk::SampleType::{C32, F32};
        const KERNELS: &[(&str, sdrk::SampleType)] = &[
            ("agc", C32),
            ("convert", C32),
            ("convert_out", C32),
            ("convert_out_real", F32),
            ("convert_real", F32),
            ("dc_block", C32),
            ("dc_block_real", F32),
            ("ddc", C32),
            ("deemphasis", F32),
            ("dot", C32),
            ("dot_real", F32),
            ("duc", C32),
            ("fft", C32),
            ("fir", C32),
            ("fir_decimate", C32),
            ("fir_interpolate", C32),
            ("fir_interpolate_real", F32),
            ("fm_demod", C32),
            ("fm_mod", C32),
            ("halfband_decimate", C32),
            ("halfband_decimate_real", F32),
            ("irfft", F32),
            ("mag", C32),
            ("mag_sq", C32),
            ("power_db", F32),
            ("quad_demod", C32),
            ("real_to_complex", C32),
            ("resample", C32),
            ("resample_real", F32),
            ("rfft", F32),
            ("rotate", C32),
            ("scale", C32),
            ("squelch", C32),
            ("stereo_matrix", C32),
            ("stereo_pll", F32),
            ("window", C32),
        ];
        KERNELS
            .iter()
            .map(|&(kernel, ty)| (kernel, sdrk::backend_for(kernel, ty)))
            .collect()
    }

    // ------------------------------------------------------------------ data path

    fn input_slot(&self, input: &str) -> Result<&InputBuf> {
        self.inputs
            .get(input)
            .ok_or_else(|| sdrk_pipeline::Error::UnknownInput(input.to_string()).into())
    }

    fn output_slot(&self, output: &str) -> Result<&OutputBuf> {
        self.outputs
            .get(output)
            .ok_or_else(|| sdrk_pipeline::Error::UnknownOutput(output.to_string()).into())
    }

    /// The typed-array layout of an input.
    ///
    /// # Errors
    /// `InvalidArgument` for an unknown input.
    pub fn input_layout(&self, input: &str) -> Result<Layout> {
        Ok(self.input_slot(input)?.layout)
    }

    /// The layout of an output.
    ///
    /// # Errors
    /// `InvalidArgument` for an unknown output.
    pub fn output_layout(&self, output: &str) -> Result<OutLayout> {
        Ok(self.output_slot(output)?.layout)
    }

    /// Hands out the buffer of an input for `lanes` array elements. Fill it, then call
    /// [`Engine::push_staged`]. This is the one copy of the data path: no allocation.
    ///
    /// ```
    /// use sdrk_wasm::engine::{Engine, LanesMut};
    /// let mut engine = Engine::from_json(include_str!("../../../examples/flows/fm_receiver.json"))?;
    /// if let LanesMut::U8(bytes) = engine.stage("rf", 4)? {
    ///     bytes.copy_from_slice(&[128, 128, 128, 128]); // the one copy of the data path
    /// }
    /// engine.push_staged("rf", None)?;
    /// assert_eq!(engine.input_index("rf")?, 2);
    /// # Ok::<(), sdrk_wasm::engine::WasmError>(())
    /// ```
    ///
    /// # Errors
    /// `LengthMismatch` if `lanes` is not a whole number of samples or exceeds `max_chunk`.
    pub fn stage(&mut self, input: &str, lanes: usize) -> Result<LanesMut<'_>> {
        let slot = self
            .inputs
            .get_mut(input)
            .ok_or_else(|| sdrk_pipeline::Error::UnknownInput(input.to_string()))?;
        if !lanes.is_multiple_of(slot.layout.lanes) {
            return Err(WasmError::new(
                ErrorKind::LengthMismatch,
                format!(
                    "{lanes} array elements are not a whole number of samples ({} per sample)",
                    slot.layout.lanes
                ),
            ));
        }
        let samples = lanes / slot.layout.lanes;
        if samples > slot.max_len {
            return Err(sdrk_pipeline::Error::ChunkTooLarge {
                len: samples,
                max: slot.max_len,
            }
            .into());
        }
        slot.staged = samples;
        slot.buf
            .lanes_mut(samples)
            .ok_or_else(|| internal("input buffer"))
    }

    /// Pushes what [`Engine::stage`] handed out.
    ///
    /// # Errors
    /// The errors of `Pipeline::push` and `Pipeline::push_with_meta`.
    pub fn push_staged(&mut self, input: &str, meta: Option<&BlockMeta>) -> Result<()> {
        let slot = self
            .inputs
            .get(input)
            .ok_or_else(|| sdrk_pipeline::Error::UnknownInput(input.to_string()))?;
        let data = slot
            .buf
            .input(slot.staged)
            .ok_or_else(|| internal("input buffer"))?;
        match meta {
            Some(m) => self.pipeline.push_with_meta(input, data, m)?,
            None => self.pipeline.push(input, data)?,
        }
        Ok(())
    }

    /// Copies `data` (raw bytes) into the input buffer and pushes it.
    ///
    /// ```
    /// use sdrk_wasm::engine::Engine;
    /// let mut engine = Engine::from_json(include_str!("../../../examples/flows/fm_receiver.json"))?;
    /// engine.push_u8("rf", &[128; 4000], None)?;
    /// assert!(engine.available("audio")? > 0);
    /// assert_eq!(engine.push_f32("rf", &[0.0; 2], None).unwrap_err().kind, "InvalidArgument");
    /// # Ok::<(), sdrk_wasm::engine::WasmError>(())
    /// ```
    ///
    /// # Errors
    /// `InvalidArgument` if the input does not take bytes, else as [`Engine::push_staged`].
    pub fn push_u8(&mut self, input: &str, data: &[u8], meta: Option<&BlockMeta>) -> Result<()> {
        match self.stage(input, data.len())? {
            LanesMut::U8(dst) => dst.copy_from_slice(data),
            _ => return Err(wrong_dtype(input, Dtype::U8)),
        }
        self.push_staged(input, meta)
    }

    /// Copies `data` (Q15 values, complex ones interleaved) into the input buffer and pushes it.
    ///
    /// # Errors
    /// `InvalidArgument` if the input does not take `Int16Array`, else as
    /// [`Engine::push_staged`].
    pub fn push_i16(&mut self, input: &str, data: &[i16], meta: Option<&BlockMeta>) -> Result<()> {
        match self.stage(input, data.len())? {
            LanesMut::I16(dst) => dst.copy_from_slice(data),
            _ => return Err(wrong_dtype(input, Dtype::I16)),
        }
        self.push_staged(input, meta)
    }

    /// Copies `data` (`f32` values, complex ones interleaved) into the input buffer and pushes
    /// it.
    ///
    /// # Errors
    /// `InvalidArgument` if the input does not take `Float32Array`, else as
    /// [`Engine::push_staged`].
    pub fn push_f32(&mut self, input: &str, data: &[f32], meta: Option<&BlockMeta>) -> Result<()> {
        match self.stage(input, data.len())? {
            LanesMut::F32(dst) => dst.copy_from_slice(data),
            _ => return Err(wrong_dtype(input, Dtype::F32)),
        }
        self.push_staged(input, meta)
    }

    /// Reads up to `max_samples` samples of a stream output through the output buffer (at
    /// most one buffer full per call).
    ///
    /// ```
    /// use sdrk_wasm::engine::{Engine, Lanes};
    /// let mut engine = Engine::from_json(include_str!("../../../examples/flows/fm_receiver.json"))?;
    /// engine.push_u8("rf", &[128; 8192], None)?;
    /// engine.push_u8("rf", &[128; 8192], None)?;
    /// let Lanes::F32(audio) = engine.read_chunk("audio", 100)? else { unreachable!() };
    /// assert_eq!(audio.len(), 100);
    /// # Ok::<(), sdrk_wasm::engine::WasmError>(())
    /// ```
    ///
    /// # Errors
    /// `InvalidArgument` for a frames output or an unknown output.
    pub fn read_chunk(&mut self, output: &str, max_samples: usize) -> Result<Lanes<'_>> {
        let slot = self
            .outputs
            .get_mut(output)
            .ok_or_else(|| sdrk_pipeline::Error::UnknownOutput(output.to_string()))?;
        if matches!(slot.layout, OutLayout::Frames(_)) {
            return Err(sdrk_pipeline::Error::WrongType {
                name: output.to_string(),
                expected: "frames",
                got: "a stream",
            }
            .into());
        }
        let n = max_samples.min(slot.chunk);
        let got = read_into(&mut self.pipeline, output, &mut slot.buf, n)?;
        slot.buf.lanes(got).ok_or_else(|| internal("output buffer"))
    }

    /// Reads the oldest frame of a frames output.
    ///
    /// ```
    /// use sdrk_wasm::engine::Engine;
    /// let mut engine = Engine::from_json(include_str!("../../../examples/flows/fm_receiver.json"))?;
    /// assert!(engine.read_frame("spectrum")?.is_none());
    /// for _ in 0..2 {
    ///     engine.push_u8("rf", &[128; 8192], None)?;
    /// }
    /// let frame = engine.read_frame("spectrum")?.expect("a frame");
    /// assert_eq!(frame.data.len(), 1024);
    /// # Ok::<(), sdrk_wasm::engine::WasmError>(())
    /// ```
    ///
    /// # Errors
    /// `InvalidArgument` for a stream output or an unknown output.
    pub fn read_frame(&mut self, output: &str) -> Result<Option<FrameRead<'_>>> {
        let slot = self
            .outputs
            .get_mut(output)
            .ok_or_else(|| sdrk_pipeline::Error::UnknownOutput(output.to_string()))?;
        let Buf::F32(frame) = &mut slot.buf else {
            return Err(sdrk_pipeline::Error::WrongType {
                name: output.to_string(),
                expected: "a stream",
                got: "frames",
            }
            .into());
        };
        if !matches!(slot.layout, OutLayout::Frames(_)) {
            return Err(sdrk_pipeline::Error::WrongType {
                name: output.to_string(),
                expected: "a stream",
                got: "frames",
            }
            .into());
        }
        Ok(self
            .pipeline
            .read_frame(output, frame)?
            .map(|meta| FrameRead { data: frame, meta }))
    }

    /// Pulls up to `max_samples` samples (bytes for a raw output) from a transmit output,
    /// calling `source` for input data (see `Pipeline::pull`). At most one buffer full per call.
    ///
    /// ```
    /// use sdrk_wasm::engine::{Engine, Lanes, LanesMut};
    /// let mut engine = Engine::from_json(include_str!("../../../examples/flows/fm_transmitter.json"))?;
    /// let mut silence = |_input: &str, buf: sdrk_wasm::engine::SourceBuf<'_>| {
    ///     if let LanesMut::F32(samples) = buf.lanes {
    ///         samples.fill(0.0);
    ///     }
    ///     Ok(buf.max_samples)
    /// };
    /// let pulled = engine.pull("rf", 1000, &mut silence)?;
    /// assert_eq!(pulled.samples, 1000);
    /// assert!(matches!(pulled.data, Lanes::U8(bytes) if bytes.len() == 1000));
    /// # Ok::<(), sdrk_wasm::engine::WasmError>(())
    /// ```
    ///
    /// # Errors
    /// The errors of `Pipeline::pull`; an error returned by `source` is returned unchanged.
    pub fn pull(
        &mut self,
        output: &str,
        max_samples: usize,
        source: &mut SourceFn<'_>,
    ) -> Result<Pulled<'_>> {
        let slot = self
            .outputs
            .get_mut(output)
            .ok_or_else(|| sdrk_pipeline::Error::UnknownOutput(output.to_string()))?;
        let n = max_samples.min(slot.chunk);
        let mut failure = None;
        let mut adapter = Adapter {
            inputs: &mut self.inputs,
            call: source,
            failure: &mut failure,
        };
        let result = pull_into(&mut self.pipeline, output, &mut slot.buf, n, &mut adapter);
        let (samples, meta) = match result {
            Ok(r) => r,
            Err(e) => return Err(failure.unwrap_or_else(|| e.into())),
        };
        let data = slot
            .buf
            .lanes(samples)
            .ok_or_else(|| internal("output buffer"))?;
        Ok(Pulled {
            data,
            samples,
            meta,
        })
    }

    /// Samples (bytes for a raw output, frames for a frames output) waiting in an output.
    ///
    /// # Errors
    /// `InvalidArgument` for an unknown output.
    pub fn available(&self, output: &str) -> Result<usize> {
        Ok(self.pipeline.available(output)?)
    }

    /// Index of the next sample the pipeline expects at an input.
    ///
    /// # Errors
    /// `InvalidArgument` for an unknown input.
    pub fn input_index(&self, input: &str) -> Result<u64> {
        Ok(self.pipeline.input_index(input)?)
    }

    /// Number of samples the output buffer moves per `readStream` or `pull` call.
    ///
    /// # Errors
    /// `InvalidArgument` for an unknown output.
    pub fn chunk(&self, output: &str) -> Result<usize> {
        Ok(self.output_slot(output)?.chunk)
    }

    /// Number of elements the drop policy of an output has discarded.
    ///
    /// # Errors
    /// `InvalidArgument` for an unknown output.
    pub fn dropped(&self, output: &str) -> Result<u64> {
        Ok(self.pipeline.dropped(output)?)
    }

    // ------------------------------------------------------------------ control

    /// Changes a run-time parameter, now or at a given time.
    ///
    /// ```
    /// use sdrk_pipeline::ParamTime;
    /// use sdrk_wasm::engine::{Dyn, Engine};
    /// let mut engine = Engine::from_json(include_str!("../../../examples/flows/fm_receiver.json"))?;
    /// let at = ParamTime::InputIndex { input: "rf".into(), index: 1000 };
    /// engine.set_param("tune", "frequency_hz", serde_json::json!(610_000.0), at)?;
    /// engine.push_u8("rf", &[128; 4000], None)?;
    /// let event = engine.poll_event().expect("the change took effect");
    /// assert_eq!(event.get("type"), Some(&Dyn::Str("paramApplied".into())));
    /// assert_eq!(event.get("index"), Some(&Dyn::U64(20)));
    /// # Ok::<(), sdrk_wasm::engine::WasmError>(())
    /// ```
    ///
    /// # Errors
    /// The errors of `Pipeline::set_param_at`.
    pub fn set_param(&mut self, node: &str, name: &str, value: Value, at: ParamTime) -> Result<()> {
        Ok(self.pipeline.set_param_at(node, name, value, at)?)
    }

    /// The oldest event, described for JavaScript; `None` if there is none.
    pub fn poll_event(&mut self) -> Option<Dyn> {
        let event = self.pipeline.poll_event()?;
        Some(self.describe_event(&event))
    }

    fn param_dyn(&self, node: &str, param: u16, value: ParamValue) -> (Dyn, Dyn) {
        let desc = self
            .kinds
            .get(node)
            .and_then(|kind| self.registry.factory(kind))
            .and_then(|f| f.description().params.get(usize::from(param)));
        let name = desc.map_or_else(|| Dyn::Null, |d| Dyn::Str(d.name.to_string()));
        let value = match value {
            ParamValue::F64(v) => Dyn::Num(v),
            ParamValue::I64(v) => Dyn::Num(v as f64),
            ParamValue::Bool(v) => Dyn::Bool(v),
            ParamValue::Choice(i) => match desc.map(|d| d.kind) {
                Some(ParamKind::Choice { options, .. }) => options
                    .get(usize::from(i))
                    .map_or(Dyn::Num(f64::from(i)), |o| Dyn::Str((*o).to_string())),
                _ => Dyn::Num(f64::from(i)),
            },
            other => Dyn::Str(format!("{other:?}")),
        };
        (name, value)
    }

    fn describe_event(&self, event: &Event) -> Dyn {
        let mut fields = vec![("type", Dyn::Null)];
        let tag = |name: &'static str, fields: &mut Vec<(&'static str, Dyn)>| {
            if let Some(slot) = fields.first_mut() {
                slot.1 = Dyn::Str(name.to_string());
            }
        };
        fields.push((
            "output",
            Dyn::Str(self.pipeline.output_name(event.output).to_string()),
        ));
        fields.push(("index", Dyn::U64(event.index)));
        match &event.kind {
            EventKind::Gap { missing } => {
                tag("gap", &mut fields);
                fields.push(("missing", Dyn::U64(*missing)));
            }
            EventKind::Overflow => tag("overflow", &mut fields),
            EventKind::Underrun => tag("underrun", &mut fields),
            EventKind::Anchor(a) => {
                tag("anchor", &mut fields);
                fields.push(("anchor", anchor_dyn(a)));
            }
            EventKind::Retuned => tag("retuned", &mut fields),
            EventKind::BurstStart { tx_time_ns } => {
                tag("burstStart", &mut fields);
                if let Some(t) = tx_time_ns {
                    fields.push(("txTimeNs", Dyn::I64(*t)));
                }
            }
            EventKind::BurstEnd => tag("burstEnd", &mut fields),
            EventKind::Late => tag("late", &mut fields),
            EventKind::ParamApplied {
                node,
                param,
                value,
                late,
            } => {
                tag("paramApplied", &mut fields);
                let node_name = self.pipeline.node_name(*node);
                let (name, value) = self.param_dyn(node_name, *param, *value);
                fields.push(("node", Dyn::Str(node_name.to_string())));
                fields.push(("param", name));
                fields.push(("value", value));
                fields.push(("late", Dyn::Bool(*late)));
            }
            EventKind::EventsLost { count } => {
                tag("eventsLost", &mut fields);
                fields.push(("count", Dyn::U64(*count)));
            }
            _ => tag("unknown", &mut fields),
        }
        Dyn::Obj(fields)
    }

    /// Marks the start of a burst on a transmit output.
    ///
    /// # Errors
    /// The errors of `Pipeline::begin_burst`.
    pub fn begin_burst(&mut self, output: &str, tx_time_ns: Option<i64>) -> Result<()> {
        Ok(self.pipeline.begin_burst(output, tx_time_ns)?)
    }

    /// Marks the end of a burst.
    ///
    /// # Errors
    /// The errors of `Pipeline::end_burst`.
    pub fn end_burst(&mut self, output: &str) -> Result<()> {
        Ok(self.pipeline.end_burst(output)?)
    }

    /// `(group delay in samples of the output, worst-case buffered samples)`.
    ///
    /// ```
    /// let engine = sdrk_wasm::engine::Engine::from_json(include_str!("../../../examples/flows/fm_receiver.json"))?;
    /// let (delay, buffered) = engine.latency("audio")?;
    /// assert!(delay > 0.0 && buffered > 0);
    /// # Ok::<(), sdrk_wasm::engine::WasmError>(())
    /// ```
    ///
    /// # Errors
    /// `InvalidArgument` for an unknown output.
    pub fn latency(&self, output: &str) -> Result<(f64, u64)> {
        let l = self.pipeline.latency(output)?;
        Ok((l.group_delay.as_f64(), l.max_buffered))
    }

    /// The configuration as canonical JSON (run-time parameters at their current values).
    pub fn config(&self) -> String {
        self.pipeline.config().to_json()
    }

    /// The report of the build.
    pub fn report(&self) -> &Report {
        self.pipeline.report()
    }

    /// Statistics of every node.
    pub fn stats(&self) -> Vec<Dyn> {
        self.pipeline
            .stats()
            .iter()
            .map(|s| {
                Dyn::Obj(vec![
                    ("name", Dyn::Str(s.name.clone())),
                    ("calls", Dyn::U64(s.calls)),
                    ("samplesIn", Dyn::U64(s.samples_in)),
                    ("samplesOut", Dyn::U64(s.samples_out)),
                    ("frames", Dyn::U64(s.frames)),
                ])
            })
            .collect()
    }

    /// Resets the state of all nodes and discards queued data and events.
    pub fn reset(&mut self) {
        self.pipeline.reset();
        for slot in self.inputs.values_mut() {
            slot.staged = 0;
        }
    }

    /// Number of samples the input buffer holds, for tests.
    pub fn input_capacity(&self, input: &str) -> Result<usize> {
        Ok(self.input_slot(input)?.buf.len())
    }
}

/// The error for a typed array of the wrong type.
///
/// ```
/// use sdrk_wasm::engine::{wrong_dtype, Dtype};
/// assert!(wrong_dtype("rf", Dtype::U8).message.contains("Uint8Array"));
/// ```
pub fn wrong_dtype(name: &str, expected: Dtype) -> WasmError {
    WasmError::invalid(format!(
        "`{name}` needs a {}, not another typed array",
        expected.array_name()
    ))
}
