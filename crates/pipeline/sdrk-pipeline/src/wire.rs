//! The versioned wire format for split pipelines (`docs/wire-format.md`).
//!
//! A message is a 32-byte header and a payload, all little-endian. It carries samples, frames,
//! events, the description of a stream and timed parameter changes between the parts of a
//! pipeline that run on different threads, cores, devices or machines.
//!
//! ```
//! use sdrk_device::BlockFlags;
//! use sdrk_pipeline::wire::{Decoded, Message, decode, encode};
//!
//! let msg = Message::samples_f32(7, 1000, BlockFlags::empty(), &[0.5, -0.5]);
//! let mut bytes = Vec::new();
//! encode(&msg, &mut bytes);
//! assert_eq!(&bytes[..4], b"SDRK");
//! match decode(&bytes)? {
//!     Decoded::Message(back, used) => {
//!         assert_eq!(back, msg);
//!         assert_eq!(used, bytes.len());
//!     }
//!     other => panic!("{other:?}"),
//! }
//! # Ok::<(), sdrk_pipeline::Error>(())
//! ```

use sdrk::{C32, CQ15, Q15};
use sdrk_blocks::{FrameMeta, ParamValue};
use sdrk_device::{BlockFlags, SampleIndex, TimeAnchor, TimeSource};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::model::{FORMAT_NAMES, FormatName};
use crate::runner::InputData;
use crate::time::{Event, EventKind, NodeId, OutputId, ParamTime};

/// The first four bytes of every message.
pub const MAGIC: [u8; 4] = *b"SDRK";

/// The format version this module reads and writes.
pub const VERSION: u8 = 1;

/// Size of the header in bytes.
pub const HEADER_LEN: usize = 32;

/// Largest payload accepted by `decode` (64 MiB).
pub const MAX_PAYLOAD: usize = 64 * 1024 * 1024;

/// The kind of a message (header byte 5).
///
/// ```
/// use sdrk_pipeline::wire::MessageType;
/// assert_eq!(MessageType::Event.code(), 3);
/// assert_eq!(MessageType::from_code(4), Some(MessageType::Hello));
/// assert_eq!(MessageType::from_code(99), None);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MessageType {
    /// A block of samples (code 1).
    Samples,
    /// One frame of `f32` values (code 2).
    Frame,
    /// An event (code 3).
    Event,
    /// The description of a stream (code 4).
    Hello,
    /// A timed parameter change (code 5).
    ParamSet,
    /// The acknowledgement of a parameter change (code 6).
    ParamAck,
}

impl MessageType {
    /// The code in the header.
    pub const fn code(self) -> u8 {
        match self {
            Self::Samples => 1,
            Self::Frame => 2,
            Self::Event => 3,
            Self::Hello => 4,
            Self::ParamSet => 5,
            Self::ParamAck => 6,
        }
    }

    /// The type for a header code, `None` for an unknown one.
    pub const fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            1 => Self::Samples,
            2 => Self::Frame,
            3 => Self::Event,
            4 => Self::Hello,
            5 => Self::ParamSet,
            6 => Self::ParamAck,
            _ => return None,
        })
    }
}

/// The encoding of the payload (header byte 10).
///
/// ```
/// use sdrk_pipeline::wire::PayloadType;
/// assert_eq!(PayloadType::Json.code(), 5);
/// assert_eq!(PayloadType::Cq15.element_size(), Some(4));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PayloadType {
    /// Raw bytes in the format named by the hello of the stream (code 0).
    Raw,
    /// Interleaved `f32` I and Q (code 1).
    C32,
    /// Interleaved `i16` I and Q (code 2).
    Cq15,
    /// `f32` (code 3).
    F32,
    /// `i16` (code 4).
    Q15,
    /// UTF-8 JSON (code 5).
    Json,
}

impl PayloadType {
    /// The code in the header.
    pub const fn code(self) -> u8 {
        match self {
            Self::Raw => 0,
            Self::C32 => 1,
            Self::Cq15 => 2,
            Self::F32 => 3,
            Self::Q15 => 4,
            Self::Json => 5,
        }
    }

    /// The type for a header code, `None` for an unknown one.
    pub const fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => Self::Raw,
            1 => Self::C32,
            2 => Self::Cq15,
            3 => Self::F32,
            4 => Self::Q15,
            5 => Self::Json,
            _ => return None,
        })
    }

    /// Bytes of one element for the sample types; `None` for raw bytes and JSON.
    pub const fn element_size(self) -> Option<usize> {
        match self {
            Self::C32 => Some(8),
            Self::Cq15 | Self::F32 => Some(4),
            Self::Q15 => Some(2),
            Self::Raw | Self::Json => None,
        }
    }
}

/// One message.
///
/// ```
/// use sdrk_device::BlockFlags;
/// use sdrk_pipeline::wire::{MessageType, PayloadType, Message};
/// let m = Message::samples_q15(1, 0, BlockFlags::empty(), &[sdrk::Q15(1), sdrk::Q15(-1)]);
/// assert_eq!((m.kind, m.payload_type, m.count), (MessageType::Samples, PayloadType::Q15, 2));
/// assert_eq!(m.payload, [1, 0, 255, 255]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    /// The kind of message.
    pub kind: MessageType,
    /// Flags of the block (`BlockFlags` bits).
    pub flags: BlockFlags,
    /// The stream the message belongs to.
    pub stream: u16,
    /// Encoding of the payload.
    pub payload_type: PayloadType,
    /// Samples: index of the first sample. Frames: `first_sample`. Otherwise 0 or an index
    /// the message refers to.
    pub start: u64,
    /// Number of elements in the payload.
    pub count: u32,
    /// Frames: number of averaged windows. Otherwise 0.
    pub averaged: u32,
    /// The payload bytes.
    pub payload: Vec<u8>,
}

fn count_of(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

impl Message {
    fn samples(
        stream: u16,
        start: u64,
        flags: BlockFlags,
        payload_type: PayloadType,
        count: usize,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            kind: MessageType::Samples,
            flags,
            stream,
            payload_type,
            start,
            count: count_of(count),
            averaged: 0,
            payload,
        }
    }

    /// A block of raw bytes holding `samples` samples.
    pub fn samples_raw(
        stream: u16,
        start: u64,
        flags: BlockFlags,
        samples: usize,
        bytes: &[u8],
    ) -> Self {
        Self::samples(
            stream,
            start,
            flags,
            PayloadType::Raw,
            samples,
            bytes.to_vec(),
        )
    }

    /// A block of complex `f32` samples.
    pub fn samples_c32(stream: u16, start: u64, flags: BlockFlags, data: &[C32]) -> Self {
        let bytes = bytemuck::cast_slice(data).to_vec();
        Self::samples(stream, start, flags, PayloadType::C32, data.len(), bytes)
    }

    /// A block of complex Q15 samples.
    pub fn samples_cq15(stream: u16, start: u64, flags: BlockFlags, data: &[CQ15]) -> Self {
        let bytes = bytemuck::cast_slice(data).to_vec();
        Self::samples(stream, start, flags, PayloadType::Cq15, data.len(), bytes)
    }

    /// A block of real `f32` samples.
    pub fn samples_f32(stream: u16, start: u64, flags: BlockFlags, data: &[f32]) -> Self {
        let bytes = bytemuck::cast_slice(data).to_vec();
        Self::samples(stream, start, flags, PayloadType::F32, data.len(), bytes)
    }

    /// A block of real Q15 samples.
    pub fn samples_q15(stream: u16, start: u64, flags: BlockFlags, data: &[Q15]) -> Self {
        let bytes = bytemuck::cast_slice(data).to_vec();
        Self::samples(stream, start, flags, PayloadType::Q15, data.len(), bytes)
    }

    /// A block of samples from the data type of `push`; `samples` is the number of samples (for
    /// raw bytes the data does not tell it).
    pub fn from_input(
        stream: u16,
        start: u64,
        flags: BlockFlags,
        samples: usize,
        data: &InputData<'_>,
    ) -> Self {
        match data {
            InputData::Raw(b) => Self::samples_raw(stream, start, flags, samples, b),
            InputData::C32(d) => Self::samples_c32(stream, start, flags, d),
            InputData::CQ15(d) => Self::samples_cq15(stream, start, flags, d),
            InputData::F32(d) => Self::samples_f32(stream, start, flags, d),
            InputData::Q15(d) => Self::samples_q15(stream, start, flags, d),
        }
    }

    /// A frame of `f32` values.
    pub fn frame(stream: u16, meta: &FrameMeta, values: &[f32]) -> Self {
        Self {
            kind: MessageType::Frame,
            flags: BlockFlags::empty(),
            stream,
            payload_type: PayloadType::F32,
            start: meta.first_sample,
            count: count_of(values.len()),
            averaged: meta.averaged,
            payload: bytemuck::cast_slice(values).to_vec(),
        }
    }

    fn json<T: Serialize>(kind: MessageType, stream: u16, start: u64, value: &T) -> Self {
        // Serializing plain data structures with string keys cannot fail.
        let payload = serde_json::to_vec(value).unwrap_or_default();
        Self {
            kind,
            flags: BlockFlags::empty(),
            stream,
            payload_type: PayloadType::Json,
            start,
            count: count_of(payload.len()),
            averaged: 0,
            payload,
        }
    }

    /// The description of a stream.
    pub fn hello(hello: &Hello) -> Self {
        Self::json(MessageType::Hello, hello.stream, 0, hello)
    }

    /// An event of the stream `stream`; `start` of the message is the index of the event.
    pub fn event(stream: u16, event: &Event) -> Self {
        let wire = WireEvent::from(event);
        Self::json(MessageType::Event, stream, event.index, &wire)
    }

    /// A timed parameter change.
    pub fn param_set(stream: u16, change: &WireParamSet) -> Self {
        Self::json(MessageType::ParamSet, stream, 0, change)
    }

    /// The acknowledgement of a parameter change.
    pub fn param_ack(stream: u16, ack: &WireParamAck) -> Self {
        Self::json(MessageType::ParamAck, stream, ack.index, ack)
    }

    fn parse_json<T: for<'de> Deserialize<'de>>(&self, expected: MessageType) -> Result<T> {
        if self.kind != expected {
            return Err(Error::Wire(format!(
                "expected a {expected:?} message, got {:?}",
                self.kind
            )));
        }
        serde_json::from_slice(&self.payload).map_err(|e| Error::Wire(e.to_string()))
    }

    /// The stream description of a hello message.
    ///
    /// # Errors
    /// `Wire` for another message type or a malformed payload.
    pub fn to_hello(&self) -> Result<Hello> {
        self.parse_json(MessageType::Hello)
    }

    /// The event of an event message, attributed to `output`.
    ///
    /// # Errors
    /// `Wire` for another message type or a malformed payload.
    pub fn to_event(&self, output: OutputId) -> Result<Event> {
        let wire: WireEvent = self.parse_json(MessageType::Event)?;
        wire.into_event(output)
    }

    /// The change of a parameter-set message.
    ///
    /// # Errors
    /// `Wire` for another message type or a malformed payload.
    pub fn to_param_set(&self) -> Result<WireParamSet> {
        self.parse_json(MessageType::ParamSet)
    }

    /// The acknowledgement of a parameter-ack message.
    ///
    /// # Errors
    /// `Wire` for another message type or a malformed payload.
    pub fn to_param_ack(&self) -> Result<WireParamAck> {
        self.parse_json(MessageType::ParamAck)
    }

    /// The samples of a samples message as owned data.
    ///
    /// # Errors
    /// `Wire` for another message type or a payload that is not a whole number of elements.
    pub fn to_data(&self) -> Result<OwnedData> {
        if self.kind != MessageType::Samples {
            return Err(Error::Wire(format!(
                "expected a samples message, got {:?}",
                self.kind
            )));
        }
        let bad = || Error::Wire("the payload is not a whole number of elements".into());
        Ok(match self.payload_type {
            PayloadType::Raw => OwnedData::Raw(self.payload.clone()),
            PayloadType::C32 => OwnedData::C32(read_vec::<C32>(&self.payload).ok_or_else(bad)?),
            PayloadType::Cq15 => OwnedData::CQ15(read_vec::<CQ15>(&self.payload).ok_or_else(bad)?),
            PayloadType::F32 => OwnedData::F32(read_vec::<f32>(&self.payload).ok_or_else(bad)?),
            PayloadType::Q15 => OwnedData::Q15(read_vec::<Q15>(&self.payload).ok_or_else(bad)?),
            _ => return Err(Error::Wire("a samples message carries no JSON".into())),
        })
    }

    /// The values and metadata of a frame message.
    ///
    /// # Errors
    /// `Wire` for another message type or a malformed payload.
    pub fn to_frame(&self) -> Result<(FrameMeta, Vec<f32>)> {
        if self.kind != MessageType::Frame || self.payload_type != PayloadType::F32 {
            return Err(Error::Wire(
                "expected a frame message with f32 values".into(),
            ));
        }
        let values = read_vec::<f32>(&self.payload)
            .ok_or_else(|| Error::Wire("the payload is not a whole number of values".into()))?;
        Ok((
            FrameMeta {
                first_sample: self.start,
                averaged: self.averaged,
                exponent: 0,
            },
            values,
        ))
    }
}

/// Reads a payload as a vector of `T` (any alignment); `None` if it is not a whole number.
fn read_vec<T: bytemuck::Pod>(bytes: &[u8]) -> Option<Vec<T>> {
    let size = core::mem::size_of::<T>();
    if size == 0 || !bytes.len().is_multiple_of(size) {
        return None;
    }
    Some(
        bytes
            .chunks_exact(size)
            .map(bytemuck::pod_read_unaligned::<T>)
            .collect(),
    )
}

/// Owned samples decoded from a message.
///
/// ```
/// use sdrk_pipeline::wire::OwnedData;
/// let d = OwnedData::F32(vec![1.0, 2.0]);
/// assert_eq!(d.as_input().len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum OwnedData {
    /// Raw bytes.
    Raw(Vec<u8>),
    /// Complex `f32`.
    C32(Vec<C32>),
    /// Complex Q15.
    CQ15(Vec<CQ15>),
    /// Real `f32`.
    F32(Vec<f32>),
    /// Real Q15.
    Q15(Vec<Q15>),
}

impl OwnedData {
    /// The data as input for `push`.
    pub fn as_input(&self) -> InputData<'_> {
        match self {
            Self::Raw(v) => InputData::Raw(v),
            Self::C32(v) => InputData::C32(v),
            Self::CQ15(v) => InputData::CQ15(v),
            Self::F32(v) => InputData::F32(v),
            Self::Q15(v) => InputData::Q15(v),
        }
    }
}

/// Appends the encoding of `msg` to `out`.
///
/// ```
/// use sdrk_device::BlockFlags;
/// use sdrk_pipeline::wire::{HEADER_LEN, Message, encode};
/// let mut out = Vec::new();
/// encode(&Message::samples_f32(1, 0, BlockFlags::empty(), &[1.0]), &mut out);
/// assert_eq!(out.len(), HEADER_LEN + 4);
/// ```
pub fn encode(msg: &Message, out: &mut Vec<u8>) {
    out.reserve(HEADER_LEN + msg.payload.len());
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(msg.kind.code());
    out.extend_from_slice(&msg.flags.bits().to_le_bytes());
    out.extend_from_slice(&msg.stream.to_le_bytes());
    out.push(msg.payload_type.code());
    out.push(0);
    out.extend_from_slice(&msg.start.to_le_bytes());
    out.extend_from_slice(&msg.count.to_le_bytes());
    out.extend_from_slice(&count_of(msg.payload.len()).to_le_bytes());
    let averaged = if msg.kind == MessageType::Frame {
        msg.averaged
    } else {
        0
    };
    out.extend_from_slice(&averaged.to_le_bytes());
    out.extend_from_slice(&msg.payload);
}

/// The result of decoding the start of a buffer.
///
/// ```
/// use sdrk_pipeline::wire::{Decoded, decode};
/// assert!(matches!(decode(b"SD")?, Decoded::NeedMore(_)));
/// # Ok::<(), sdrk_pipeline::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decoded {
    /// A message and the number of bytes it used.
    Message(Message, usize),
    /// The buffer ends inside a message: at least this many more bytes are needed.
    NeedMore(usize),
    /// A message of an unknown type was skipped; the number of bytes it used.
    Skipped(usize),
}

fn le16(b: &[u8], at: usize) -> u16 {
    let mut a = [0u8; 2];
    a.copy_from_slice(b.get(at..at + 2).unwrap_or_default());
    u16::from_le_bytes(a)
}

fn le32(b: &[u8], at: usize) -> u32 {
    let mut a = [0u8; 4];
    a.copy_from_slice(b.get(at..at + 4).unwrap_or_default());
    u32::from_le_bytes(a)
}

fn le64(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(b.get(at..at + 8).unwrap_or_default());
    u64::from_le_bytes(a)
}

/// Decodes the message at the start of `buf`.
///
/// # Errors
/// `Wire` for a wrong magic or version, a payload that is too large, a payload type that
/// does not exist, or a payload length that does not match the element count.
pub fn decode(buf: &[u8]) -> Result<Decoded> {
    let seen = buf.len().min(MAGIC.len());
    if buf.get(..seen) != MAGIC.get(..seen) {
        return Err(Error::Wire("bad magic".into()));
    }
    if buf.len() < HEADER_LEN {
        if buf.len() > 4 && buf.get(4) != Some(&VERSION) {
            return Err(Error::Wire(format!(
                "unsupported version {}",
                buf.get(4).copied().unwrap_or(0)
            )));
        }
        return Ok(Decoded::NeedMore(HEADER_LEN - buf.len()));
    }
    let version = buf.get(4).copied().unwrap_or(0);
    if version != VERSION {
        return Err(Error::Wire(format!("unsupported version {version}")));
    }
    let payload_len = le32(buf, 24) as usize;
    if payload_len > MAX_PAYLOAD {
        return Err(Error::Wire(format!(
            "payload of {payload_len} bytes exceeds the limit of {MAX_PAYLOAD}"
        )));
    }
    let total = HEADER_LEN + payload_len;
    let code = buf.get(5).copied().unwrap_or(0);
    let Some(kind) = MessageType::from_code(code) else {
        // Forward compatibility: skip what this version does not know.
        return Ok(if buf.len() < total {
            Decoded::NeedMore(total - buf.len())
        } else {
            Decoded::Skipped(total)
        });
    };
    let payload_code = buf.get(10).copied().unwrap_or(0);
    let payload_type = PayloadType::from_code(payload_code)
        .ok_or_else(|| Error::Wire(format!("unknown payload type {payload_code}")))?;
    let count = le32(buf, 20);
    if matches!(kind, MessageType::Samples | MessageType::Frame)
        && let Some(size) = payload_type.element_size()
        && (count as usize).checked_mul(size) != Some(payload_len)
    {
        return Err(Error::Wire(format!(
            "{count} elements of {size} bytes do not fill a payload of {payload_len} bytes"
        )));
    }
    if buf.len() < total {
        return Ok(Decoded::NeedMore(total - buf.len()));
    }
    let message = Message {
        kind,
        flags: BlockFlags::from_bits_truncate(le16(buf, 6)),
        stream: le16(buf, 8),
        payload_type,
        start: le64(buf, 12),
        count,
        averaged: if kind == MessageType::Frame {
            le32(buf, 28)
        } else {
            0
        },
        payload: buf.get(HEADER_LEN..total).unwrap_or_default().to_vec(),
    };
    Ok(Decoded::Message(message, total))
}

/// Decodes all complete messages in `buf`, skipping unknown types; returns the messages and
/// the number of bytes consumed. The bytes of an incomplete last message are not consumed.
///
/// ```
/// use sdrk_device::BlockFlags;
/// use sdrk_pipeline::wire::{Message, decode_stream, encode};
/// let mut bytes = Vec::new();
/// encode(&Message::samples_f32(1, 0, BlockFlags::empty(), &[1.0]), &mut bytes);
/// encode(&Message::samples_f32(1, 1, BlockFlags::empty(), &[2.0]), &mut bytes);
/// let used = bytes.len();
/// bytes.extend_from_slice(b"SDRK");
/// let (messages, consumed) = decode_stream(&bytes)?;
/// assert_eq!((messages.len(), consumed), (2, used));
/// # Ok::<(), sdrk_pipeline::Error>(())
/// ```
///
/// # Errors
/// As [`decode`].
pub fn decode_stream(buf: &[u8]) -> Result<(Vec<Message>, usize)> {
    let mut messages = Vec::new();
    let mut at = 0usize;
    while let Some(rest) = buf.get(at..).filter(|r| !r.is_empty()) {
        match decode(rest)? {
            Decoded::Message(m, used) => {
                messages.push(m);
                at += used;
            }
            Decoded::Skipped(used) => at += used,
            Decoded::NeedMore(_) => break,
        }
    }
    Ok((messages, at))
}

// ------------------------------------------------------------------ JSON mirror types

/// A time anchor on the wire.
///
/// ```
/// use sdrk_device::{SampleIndex, TimeAnchor, TimeSource};
/// use sdrk_pipeline::wire::WireAnchor;
/// let a = TimeAnchor { index: SampleIndex(5), unix_ns: 9, source: TimeSource::Gps };
/// assert_eq!(TimeAnchor::try_from(WireAnchor::from(a)).unwrap(), a);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireAnchor {
    /// Index of the anchored sample.
    pub index: u64,
    /// Absolute time of that sample in unix nanoseconds.
    pub unix_ns: i64,
    /// `device`, `host`, `gps` or `unknown`.
    pub source: String,
}

impl From<TimeAnchor> for WireAnchor {
    fn from(a: TimeAnchor) -> Self {
        let source = match a.source {
            TimeSource::Device => "device",
            TimeSource::Host => "host",
            TimeSource::Gps => "gps",
            _ => "unknown",
        };
        Self {
            index: a.index.0,
            unix_ns: a.unix_ns,
            source: source.to_string(),
        }
    }
}

impl TryFrom<WireAnchor> for TimeAnchor {
    type Error = Error;

    fn try_from(w: WireAnchor) -> Result<Self> {
        let source = match w.source.as_str() {
            "device" => TimeSource::Device,
            "host" => TimeSource::Host,
            "gps" => TimeSource::Gps,
            "unknown" => TimeSource::Unknown,
            other => return Err(Error::Wire(format!("unknown time source `{other}`"))),
        };
        Ok(Self {
            index: SampleIndex(w.index),
            unix_ns: w.unix_ns,
            source,
        })
    }
}

/// A parameter value on the wire.
///
/// ```
/// use sdrk_blocks::ParamValue;
/// use sdrk_pipeline::wire::WireParamValue;
/// let w = WireParamValue::from(ParamValue::F64(1.5));
/// assert_eq!(serde_json::to_string(&w)?, r#"{"type":"f64","value":1.5}"#);
/// assert_eq!(ParamValue::try_from(w).unwrap(), ParamValue::F64(1.5));
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum WireParamValue {
    /// A floating-point number.
    F64(f64),
    /// An integer.
    I64(i64),
    /// A flag.
    Bool(bool),
    /// A choice index.
    Choice(u16),
    /// A well-known sample format by name, or `custom`.
    Format(String),
}

impl From<ParamValue> for WireParamValue {
    fn from(v: ParamValue) -> Self {
        match v {
            ParamValue::F64(x) => Self::F64(x),
            ParamValue::I64(x) => Self::I64(x),
            ParamValue::Bool(x) => Self::Bool(x),
            ParamValue::Choice(x) => Self::Choice(x),
            ParamValue::Format(f) => Self::Format(
                FORMAT_NAMES
                    .iter()
                    .find(|n| FormatName::from_name(n).is_some_and(|k| k.format() == f))
                    .map_or("custom", |n| *n)
                    .to_string(),
            ),
            _ => Self::Format("custom".to_string()),
        }
    }
}

impl TryFrom<WireParamValue> for ParamValue {
    type Error = Error;

    fn try_from(w: WireParamValue) -> Result<Self> {
        Ok(match w {
            WireParamValue::F64(x) => Self::F64(x),
            WireParamValue::I64(x) => Self::I64(x),
            WireParamValue::Bool(x) => Self::Bool(x),
            WireParamValue::Choice(x) => Self::Choice(x),
            WireParamValue::Format(name) => Self::Format(
                FormatName::from_name(&name)
                    .ok_or_else(|| Error::Wire(format!("unknown format `{name}`")))?
                    .format(),
            ),
        })
    }
}

/// An event on the wire: the index and the kind with a `type` tag in `snake_case`.
///
/// ```
/// use sdrk_pipeline::wire::{WireEvent, WireEventKind};
/// let e = WireEvent { index: 10, kind: WireEventKind::Gap { missing: 3 } };
/// assert_eq!(serde_json::to_string(&e)?, r#"{"index":10,"kind":{"type":"gap","missing":3}}"#);
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireEvent {
    /// Sample index on the stream.
    pub index: u64,
    /// What happened.
    pub kind: WireEventKind,
}

/// The kinds of [`WireEvent`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireEventKind {
    /// Samples are missing before this index.
    Gap {
        /// Number of missing samples.
        missing: u64,
    },
    /// The receiver dropped samples.
    Overflow,
    /// The transmitter ran out of samples.
    Underrun,
    /// An absolute time for a sample.
    Anchor {
        /// The anchor.
        anchor: WireAnchor,
    },
    /// Frequency, rate or gain changed.
    Retuned,
    /// First sample of a burst.
    BurstStart {
        /// Requested transmit time.
        tx_time_ns: Option<i64>,
    },
    /// Last sample of a burst.
    BurstEnd,
    /// A timed burst arrived too late.
    Late,
    /// A timed parameter change took effect.
    ParamApplied {
        /// Index of the node.
        node: u32,
        /// Index of the parameter in the description.
        param: u16,
        /// The new value.
        value: WireParamValue,
        /// True if the requested time had passed.
        late: bool,
    },
    /// Events were dropped.
    EventsLost {
        /// Number of events lost.
        count: u64,
    },
}

impl From<&Event> for WireEvent {
    fn from(e: &Event) -> Self {
        let kind = match e.kind {
            EventKind::Gap { missing } => WireEventKind::Gap { missing },
            EventKind::Overflow => WireEventKind::Overflow,
            EventKind::Underrun => WireEventKind::Underrun,
            EventKind::Anchor(a) => WireEventKind::Anchor { anchor: a.into() },
            EventKind::Retuned => WireEventKind::Retuned,
            EventKind::BurstStart { tx_time_ns } => WireEventKind::BurstStart { tx_time_ns },
            EventKind::BurstEnd => WireEventKind::BurstEnd,
            EventKind::Late => WireEventKind::Late,
            EventKind::ParamApplied {
                node,
                param,
                value,
                late,
            } => WireEventKind::ParamApplied {
                node: u32::try_from(node.index()).unwrap_or(u32::MAX),
                param,
                value: value.into(),
                late,
            },
            EventKind::EventsLost { count } => WireEventKind::EventsLost { count },
        };
        Self {
            index: e.index,
            kind,
        }
    }
}

impl WireEvent {
    /// The event on `output` of the receiving pipeline.
    ///
    /// # Errors
    /// `Wire` for an anchor with an unknown source or a parameter value that cannot be
    /// converted.
    pub fn into_event(self, output: OutputId) -> Result<Event> {
        let kind = match self.kind {
            WireEventKind::Gap { missing } => EventKind::Gap { missing },
            WireEventKind::Overflow => EventKind::Overflow,
            WireEventKind::Underrun => EventKind::Underrun,
            WireEventKind::Anchor { anchor } => EventKind::Anchor(anchor.try_into()?),
            WireEventKind::Retuned => EventKind::Retuned,
            WireEventKind::BurstStart { tx_time_ns } => EventKind::BurstStart { tx_time_ns },
            WireEventKind::BurstEnd => EventKind::BurstEnd,
            WireEventKind::Late => EventKind::Late,
            WireEventKind::ParamApplied {
                node,
                param,
                value,
                late,
            } => EventKind::ParamApplied {
                node: NodeId::new(node as usize),
                param,
                value: value.try_into()?,
                late,
            },
            WireEventKind::EventsLost { count } => EventKind::EventsLost { count },
        };
        Ok(Event {
            output,
            index: self.index,
            kind,
        })
    }
}

/// The payload of a hello message: what a stream carries.
///
/// ```
/// use sdrk_pipeline::wire::Hello;
/// let h = Hello::samples(3, "audio", "f32", 48000, 1);
/// assert_eq!(serde_json::to_string(&h)?, r#"{"stream":3,"name":"audio","type":"f32","rate_hz":"48000/1","anchor":null}"#);
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hello {
    /// The id of the stream in the messages that follow.
    pub stream: u16,
    /// Name of the stream.
    pub name: String,
    /// Sample type (`c32`, `cq15`, `f32`, `q15`) of a stream of converted samples.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub sample_type: Option<String>,
    /// Name of the raw format of a stream of raw bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Sample rate as `"num/den"`.
    pub rate_hz: String,
    /// The latest anchor of the stream, if any.
    pub anchor: Option<WireAnchor>,
}

impl Hello {
    /// A stream of converted samples with a rate `num/den`.
    pub fn samples(stream: u16, name: &str, sample_type: &str, num: u64, den: u64) -> Self {
        Self {
            stream,
            name: name.to_string(),
            sample_type: Some(sample_type.to_string()),
            format: None,
            rate_hz: format!("{num}/{den}"),
            anchor: None,
        }
    }

    /// A stream of raw bytes in a format with a rate `num/den`.
    pub fn raw(stream: u16, name: &str, format: &str, num: u64, den: u64) -> Self {
        Self {
            stream,
            name: name.to_string(),
            sample_type: None,
            format: Some(format.to_string()),
            rate_hz: format!("{num}/{den}"),
            anchor: None,
        }
    }
}

/// When a timed change takes effect, on the wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireParamTime {
    /// At the next sample.
    Immediate,
    /// At a sample index of a pipeline input.
    InputIndex {
        /// Name of the input.
        input: String,
        /// The sample index.
        index: u64,
    },
    /// At an absolute time.
    UnixNs {
        /// Unix nanoseconds.
        ns: i64,
    },
}

impl From<&ParamTime> for WireParamTime {
    fn from(t: &ParamTime) -> Self {
        match t {
            ParamTime::Immediate => Self::Immediate,
            ParamTime::InputIndex { input, index } => Self::InputIndex {
                input: input.clone(),
                index: *index,
            },
            ParamTime::UnixNs(ns) => Self::UnixNs { ns: *ns },
        }
    }
}

impl From<WireParamTime> for ParamTime {
    fn from(t: WireParamTime) -> Self {
        match t {
            WireParamTime::Immediate => Self::Immediate,
            WireParamTime::InputIndex { input, index } => Self::InputIndex { input, index },
            WireParamTime::UnixNs { ns } => Self::UnixNs(ns),
        }
    }
}

/// The payload of a `param_set` message.
///
/// ```
/// use sdrk_pipeline::wire::{WireParamSet, WireParamTime};
/// let p = WireParamSet {
///     node: "tune".into(),
///     name: "frequency_hz".into(),
///     value: serde_json::json!(610000),
///     at: WireParamTime::Immediate,
/// };
/// assert!(serde_json::to_string(&p)?.starts_with(r#"{"node":"tune""#));
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireParamSet {
    /// Node name.
    pub node: String,
    /// Parameter name.
    pub name: String,
    /// The new value as JSON.
    pub value: Value,
    /// When the change takes effect.
    pub at: WireParamTime,
}

/// The payload of a `param_ack` message.
///
/// ```
/// use sdrk_pipeline::wire::WireParamAck;
/// let a = WireParamAck { node: "tune".into(), name: "frequency_hz".into(), index: 100, late: false };
/// assert_eq!(serde_json::to_string(&a)?, r#"{"node":"tune","name":"frequency_hz","index":100,"late":false}"#);
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireParamAck {
    /// Node name.
    pub node: String,
    /// Parameter name.
    pub name: String,
    /// Sample index at which the change took effect.
    pub index: u64,
    /// True if the requested time had passed.
    pub late: bool,
}
