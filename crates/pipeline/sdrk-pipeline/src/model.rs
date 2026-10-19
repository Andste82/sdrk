//! The data model of a pipeline description (JSON, `format_version` 1).
//!
//! JSON is the canonical format. The structs here are what `serde` reads and writes; the keys
//! come out in the order of the fields, and maps are sorted by key (`BTreeMap`). Unknown fields
//! are errors.

use std::borrow::Cow;
use std::collections::BTreeMap;

use schemars::{JsonSchema, Schema, SchemaGenerator};
use sdrk::Rate;
use sdrk_device::{Align, Container, Endian, FullScale, Layout, SampleFormat, formats};
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::error::{Error, Result};

/// The format version this crate reads and writes.
pub const FORMAT_VERSION: u32 = 1;

/// Largest value of `settings.max_chunk`.
pub const MAX_CHUNK_LIMIT: usize = 1_048_576;

/// Default of `settings.max_chunk`.
pub const DEFAULT_MAX_CHUNK: usize = 4096;

/// Longest `description` in characters.
pub const MAX_DESCRIPTION_CHARS: usize = 2000;

/// Pattern of every name of an input, a node or an output.
pub const NAME_PATTERN: &str = "^[a-z][a-z0-9_]{0,63}$";

/// Checks a name against [`NAME_PATTERN`].
///
/// ```
/// use sdrk_pipeline::is_valid_name;
/// assert!(is_valid_name("fft_1"));
/// assert!(!is_valid_name("1fft"));
/// assert!(!is_valid_name("Fft"));
/// assert!(!is_valid_name(""));
/// ```
pub fn is_valid_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    let first_ok = matches!(bytes.next(), Some(b'a'..=b'z'));
    first_ok && name.len() <= 64 && bytes.all(|b| matches!(b, b'a'..=b'z' | b'0'..=b'9' | b'_'))
}

/// A sample rate in the JSON model: a positive integer or an exact fraction `"num/den"`.
///
/// The value is reduced on load; an integral rate is written as a number, anything else as the
/// reduced fraction.
///
/// ```
/// use sdrk_pipeline::RateHz;
/// let r: RateHz = serde_json::from_str("\"96000/2\"")?;
/// assert_eq!(serde_json::to_string(&r)?, "48000");
/// let r: RateHz = serde_json::from_str("\"1000/3\"")?;
/// assert_eq!(serde_json::to_string(&r)?, "\"1000/3\"");
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateHz(pub Rate);

/// Parses `"num/den"` (or a plain integer string) into a rate.
pub(crate) fn parse_rate_text(text: &str) -> core::result::Result<Rate, String> {
    let (num, den) = match text.split_once('/') {
        Some((n, d)) => (n, d),
        None => (text, "1"),
    };
    let number = |s: &str| -> core::result::Result<u64, String> {
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
            return Err(format!(
                "`{text}` is not a rate: expected an integer or \"num/den\""
            ));
        }
        s.parse::<u64>()
            .map_err(|_| format!("`{text}` does not fit into 64 bits"))
    };
    let (n, d) = (number(num)?, number(den)?);
    if n == 0 || d == 0 {
        return Err(format!("`{text}` is not positive"));
    }
    Rate::new(n, d).map_err(|e| e.to_string())
}

impl Serialize for RateHz {
    fn serialize<S: Serializer>(&self, serializer: S) -> core::result::Result<S::Ok, S::Error> {
        if self.0.den() == 1 {
            serializer.serialize_u64(self.0.num())
        } else {
            serializer.serialize_str(&format!("{}/{}", self.0.num(), self.0.den()))
        }
    }
}

impl<'de> Deserialize<'de> for RateHz {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> core::result::Result<Self, D::Error> {
        struct V;
        impl Visitor<'_> for V {
            type Value = RateHz;

            fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str("a positive integer or a string \"num/den\"")
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> core::result::Result<RateHz, E> {
                if v == 0 {
                    return Err(E::custom("the rate must be positive"));
                }
                Rate::new(v, 1).map(RateHz).map_err(E::custom)
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> core::result::Result<RateHz, E> {
                u64::try_from(v)
                    .map_err(|_| E::custom("the rate must be positive"))
                    .and_then(|u| self.visit_u64(u))
            }

            fn visit_str<E: de::Error>(self, v: &str) -> core::result::Result<RateHz, E> {
                parse_rate_text(v).map(RateHz).map_err(E::custom)
            }
        }
        deserializer.deserialize_any(V)
    }
}

impl JsonSchema for RateHz {
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("RateHz")
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> Schema {
        schemars::json_schema!({
            "description": "Sample rate in Hz: a positive integer or an exact fraction \"num/den\".",
            "oneOf": [
                { "type": "integer", "minimum": 1 },
                { "type": "string", "pattern": "^[1-9][0-9]*(/[1-9][0-9]*)?$" }
            ]
        })
    }
}

/// Numeric precision of the whole pipeline.
///
/// ```
/// use sdrk_pipeline::Precision;
/// assert_eq!(serde_json::to_string(&Precision::Q15)?, "\"q15\"");
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// Complex `C32` and real `f32`.
    #[default]
    F32,
    /// Complex `CQ15` and real `Q15`.
    Q15,
}

/// What happens when an output is full.
///
/// ```
/// use sdrk_pipeline::DropPolicy;
/// assert_eq!(DropPolicy::default(), DropPolicy::Error);
/// assert_eq!(serde_json::to_string(&DropPolicy::DropOldest)?, "\"drop_oldest\"");
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DropPolicy {
    /// `push` fails with `OutputFull` before it consumes anything.
    #[default]
    Error,
    /// The oldest data of the output is discarded to make room.
    DropOldest,
    /// New data that does not fit is discarded.
    DropNewest,
}

/// The `settings` object.
///
/// ```
/// use sdrk_pipeline::{Precision, Settings};
/// let s: Settings = serde_json::from_str("{}")?;
/// assert_eq!((s.precision, s.max_chunk), (Precision::F32, 4096));
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
pub struct Settings {
    /// `"f32"` (complex `C32`, real `f32`) or `"q15"` (`CQ15`, `Q15`); one per pipeline.
    pub precision: Precision,
    /// Largest number of samples accepted per `push` and input (1 to 1 048 576).
    #[schemars(range(min = 1, max = 1_048_576))]
    pub max_chunk: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            precision: Precision::F32,
            max_chunk: DEFAULT_MAX_CHUNK,
        }
    }
}

/// Container of a custom raw format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ContainerSpec {
    /// Unsigned 8-bit.
    U8,
    /// Signed 8-bit.
    I8,
    /// Signed 16-bit.
    I16,
    /// Signed 32-bit.
    I32,
    /// 32-bit float.
    F32,
    /// Two 12-bit values in 3 bytes.
    Packed12,
    /// Four 10-bit values in 5 bytes.
    Packed10,
    /// Eight 1-bit values per byte.
    Bits1,
}

/// Bit position of a custom raw format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum AlignSpec {
    /// Valid bits are the low bits.
    Lsb,
    /// Valid bits are the high bits.
    Msb,
}

/// Component layout of a custom raw format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum LayoutSpec {
    /// I and Q alternate.
    Interleaved,
    /// All I values, then all Q values.
    Planar,
    /// One real value per sample.
    Real,
}

/// Byte order of a custom raw format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum EndianSpec {
    /// Little endian.
    #[serde(rename = "le")]
    Little,
    /// Big endian.
    #[serde(rename = "be")]
    Big,
}

/// What raw value corresponds to amplitude 1.0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FullScaleSpec {
    /// `2^(bits-1)`.
    PowerOfTwo,
    /// An explicit integer, e.g. 2048.
    Explicit(u32),
}

/// A custom raw format, mapped to `sdrk_device::SampleFormat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CustomFormat {
    /// Storage container of one component.
    pub container: ContainerSpec,
    /// Number of valid bits.
    pub bits: u8,
    /// Position of the valid bits inside the container.
    pub align: AlignSpec,
    /// Arrangement of the components.
    pub layout: LayoutSpec,
    /// Byte order of multi-byte containers.
    pub endian: EndianSpec,
    /// Raw value of amplitude 1.0 (default `power_of_two`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_scale: Option<FullScaleSpec>,
    /// If true, the first value of each pair is Q (default false).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swap_iq: Option<bool>,
}

/// A raw sample format in the model: a name (`"cu8"`) or a [`CustomFormat`].
///
/// ```
/// use sdrk_pipeline::FormatSpec;
/// let f: FormatSpec = serde_json::from_str("\"cs16\"")?;
/// assert_eq!(f.to_format().unwrap(), sdrk_device::formats::CS16);
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum FormatSpec {
    /// One of the well-known formats, see [`FORMAT_NAMES`].
    Name(FormatName),
    /// A custom format.
    Custom(CustomFormat),
}

/// Names of the well-known raw formats.
pub const FORMAT_NAMES: [&str; 12] = [
    "cu8",
    "cs8",
    "cs16",
    "cs12_lsb16",
    "sc16_q11",
    "cs16_planar",
    "packed12",
    "packed10",
    "i2s32_msb",
    "cf32",
    "rf32",
    "rs16",
];

/// A well-known raw format by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FormatName {
    /// RTL-SDR, unsigned 8-bit.
    Cu8,
    /// HackRF, signed 8-bit.
    Cs8,
    /// Signed 16-bit.
    Cs16,
    /// 12-bit values sign-extended in 16 bits.
    #[serde(rename = "cs12_lsb16")]
    Cs12Lsb16,
    /// bladeRF, 12 bits in 16, full scale 2048.
    #[serde(rename = "sc16_q11")]
    Sc16Q11,
    /// Planar signed 16-bit.
    Cs16Planar,
    /// Two 12-bit values in 3 bytes.
    Packed12,
    /// Four 10-bit values in 5 bytes.
    Packed10,
    /// 24 valid bits MSB-aligned in 32.
    #[serde(rename = "i2s32_msb")]
    I2s32Msb,
    /// Interleaved 32-bit float.
    Cf32,
    /// Real 32-bit float.
    Rf32,
    /// Real signed 16-bit.
    Rs16,
}

impl FormatName {
    /// Parses a format name.
    ///
    /// ```
    /// use sdrk_pipeline::FormatName;
    /// assert_eq!(FormatName::from_name("cu8"), Some(FormatName::Cu8));
    /// assert_eq!(FormatName::from_name("nope"), None);
    /// ```
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "cu8" => Self::Cu8,
            "cs8" => Self::Cs8,
            "cs16" => Self::Cs16,
            "cs12_lsb16" => Self::Cs12Lsb16,
            "sc16_q11" => Self::Sc16Q11,
            "cs16_planar" => Self::Cs16Planar,
            "packed12" => Self::Packed12,
            "packed10" => Self::Packed10,
            "i2s32_msb" => Self::I2s32Msb,
            "cf32" => Self::Cf32,
            "rf32" => Self::Rf32,
            "rs16" => Self::Rs16,
            _ => return None,
        })
    }

    /// The format constant of `sdrk-device`.
    pub fn format(self) -> SampleFormat {
        match self {
            Self::Cu8 => formats::CU8,
            Self::Cs8 => formats::CS8,
            Self::Cs16 => formats::CS16,
            Self::Cs12Lsb16 => formats::CS12_LSB16,
            Self::Sc16Q11 => formats::SC16_Q11,
            Self::Cs16Planar => formats::CS16_PLANAR,
            Self::Packed12 => formats::PACKED12,
            Self::Packed10 => formats::PACKED10,
            Self::I2s32Msb => formats::I2S32_MSB,
            Self::Cf32 => formats::CF32,
            Self::Rf32 => formats::RF32,
            Self::Rs16 => formats::RS16,
        }
    }
}

impl FormatSpec {
    /// Maps the model entry to a validated `SampleFormat`.
    ///
    /// # Errors
    /// The text of the `sdrk-device` error if the custom format is inconsistent.
    pub fn to_format(&self) -> core::result::Result<SampleFormat, String> {
        match self {
            Self::Name(n) => Ok(n.format()),
            Self::Custom(c) => {
                let container = match c.container {
                    ContainerSpec::U8 => Container::U8,
                    ContainerSpec::I8 => Container::I8,
                    ContainerSpec::I16 => Container::I16,
                    ContainerSpec::I32 => Container::I32,
                    ContainerSpec::F32 => Container::F32,
                    ContainerSpec::Packed12 => Container::Packed12,
                    ContainerSpec::Packed10 => Container::Packed10,
                    ContainerSpec::Bits1 => Container::Bits1,
                };
                let signed = c.container != ContainerSpec::U8;
                let layout = match c.layout {
                    LayoutSpec::Interleaved => Layout::Interleaved,
                    LayoutSpec::Planar => Layout::Planar,
                    LayoutSpec::Real => Layout::Real,
                };
                let align = match c.align {
                    AlignSpec::Lsb => Align::Lsb,
                    AlignSpec::Msb => Align::Msb,
                };
                let endian = match c.endian {
                    EndianSpec::Little => Endian::Little,
                    EndianSpec::Big => Endian::Big,
                };
                let mut format =
                    SampleFormat::new(container, c.bits, signed, layout, align, endian)
                        .map_err(|e| e.to_string())?
                        .with_swap_iq(c.swap_iq.unwrap_or(false));
                if let Some(FullScaleSpec::Explicit(v)) = c.full_scale {
                    format = format
                        .with_full_scale(FullScale::Explicit(v))
                        .map_err(|e| e.to_string())?;
                }
                Ok(format)
            }
        }
    }
}

/// The type of an input that delivers already converted samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum InputType {
    /// Complex `f32`.
    C32,
    /// Complex Q15.
    Cq15,
    /// Real `f32`.
    F32,
    /// Real Q15.
    Q15,
}

/// An input of the pipeline: raw bytes in a `format` or converted samples of a `type`.
///
/// ```
/// use sdrk_pipeline::InputSpec;
/// let i: InputSpec = serde_json::from_str(r#"{"format":"cu8","rate_hz":2400000}"#)?;
/// assert_eq!(i.rate_hz.0.num(), 2_400_000);
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InputSpec {
    /// Raw sample format; an automatic `convert` node follows the input. Exactly one of
    /// `format` and `type` must be given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<FormatSpec>,
    /// Type of already converted samples. Exactly one of `format` and `type` must be given.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub sample_type: Option<InputType>,
    /// Sample rate in Hz.
    pub rate_hz: RateHz,
    /// Free text (at most 2000 characters).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// A processing node: a block kind and its parameters.
///
/// ```
/// use sdrk_pipeline::NodeSpec;
/// let n: NodeSpec = serde_json::from_str(r#"{"kind":"ddc","params":{"decimation":10}}"#)?;
/// assert_eq!(n.kind, "ddc");
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NodeSpec {
    /// The block kind, e.g. `"ddc"`.
    pub kind: String,
    /// Parameters by name; missing ones take the default of the block.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub params: BTreeMap<String, Value>,
    /// Free text (at most 2000 characters).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// A connection from an input or node to a node or output.
///
/// ```
/// use sdrk_pipeline::EdgeSpec;
/// let e: EdgeSpec = serde_json::from_str(r#"{"from":"rf","to":"tune"}"#)?;
/// assert_eq!((e.from.as_str(), e.to.as_str()), ("rf", "tune"));
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EdgeSpec {
    /// Name of the source (an input or a node).
    pub from: String,
    /// Name of the destination (a node or an output).
    pub to: String,
}

/// The kind of an output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum OutputType {
    /// A stream of samples from a `Block`.
    Stream,
    /// Frames of `f32` values from a `FrameBlock`.
    Frames,
}

/// An output of the pipeline.
///
/// ```
/// use sdrk_pipeline::{DropPolicy, OutputSpec};
/// let o: OutputSpec = serde_json::from_str(r#"{"type":"frames","queue":2}"#)?;
/// assert_eq!(o.drop_policy(), DropPolicy::Error);
/// assert_eq!(o.queue_frames(), 2);
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OutputSpec {
    /// `"stream"` or `"frames"`.
    #[serde(rename = "type")]
    pub output_type: OutputType,
    /// Stream outputs only: convert to this raw format (an automatic `convert_out` node).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<FormatSpec>,
    /// Stream outputs only: capacity in samples (default `4 * max_chunk`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capacity: Option<usize>,
    /// Frame outputs only: queue length in frames (default 4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue: Option<usize>,
    /// What happens when the output is full (default `"error"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drop: Option<DropPolicy>,
    /// Free text (at most 2000 characters).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Default queue length of a frame output.
pub const DEFAULT_QUEUE: usize = 4;

impl OutputSpec {
    /// The drop policy with the default applied.
    pub fn drop_policy(&self) -> DropPolicy {
        self.drop.unwrap_or_default()
    }

    /// The queue length in frames with the default applied.
    pub fn queue_frames(&self) -> usize {
        self.queue.unwrap_or(DEFAULT_QUEUE)
    }

    /// The capacity in samples with the default `4 * max_chunk` applied.
    pub fn capacity_samples(&self, max_chunk: usize) -> usize {
        self.capacity.unwrap_or_else(|| max_chunk.saturating_mul(4))
    }
}

/// A complete pipeline description.
///
/// ```
/// use sdrk_pipeline::PipelineSpec;
/// let json = r#"{
///   "sdrk": 1,
///   "name": "tone",
///   "settings": { "precision": "f32", "max_chunk": 4096 },
///   "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
///   "nodes": { "gain": { "kind": "scale", "params": { "gain_db": -6 } } },
///   "edges": [ { "from": "iq", "to": "gain" }, { "from": "gain", "to": "out" } ],
///   "outputs": { "out": { "type": "stream" } }
/// }"#;
/// let spec = PipelineSpec::from_json(json)?;
/// assert_eq!(spec.nodes.len(), 1);
/// assert_eq!(PipelineSpec::from_json(&spec.to_json())?, spec);
/// # Ok::<(), sdrk_pipeline::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PipelineSpec {
    /// Format version; must be 1.
    pub sdrk: u32,
    /// Name of the pipeline.
    pub name: String,
    /// Free text (at most 2000 characters).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Precision and chunk size.
    #[serde(default)]
    pub settings: Settings,
    /// The inputs by name.
    #[serde(default)]
    pub inputs: BTreeMap<String, InputSpec>,
    /// The processing nodes by name.
    #[serde(default)]
    pub nodes: BTreeMap<String, NodeSpec>,
    /// The connections.
    #[serde(default)]
    pub edges: Vec<EdgeSpec>,
    /// The outputs by name.
    #[serde(default)]
    pub outputs: BTreeMap<String, OutputSpec>,
}

impl PipelineSpec {
    /// Parses a description. Unknown fields are errors.
    ///
    /// # Errors
    /// [`Error::Json`] with the position of the problem. This is only the syntax and type
    /// level; use [`Registry::validate_json`](crate::Registry::validate_json) for the full check
    /// with machine-readable diagnostics.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| Error::Json(e.to_string()))
    }

    /// Writes the canonical JSON: pretty-printed, keys in field order, maps sorted by key.
    pub fn to_json(&self) -> String {
        // Serializing plain data structures with string keys cannot fail.
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}
