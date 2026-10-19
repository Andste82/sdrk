//! SigMF recordings (spec 1.x, `core` namespace): a `<base>.sigmf-meta` JSON file and a
//! `<base>.sigmf-data` file of raw samples.
//!
//! # Mapping to blocks
//!
//! A *capture* segment of the metadata becomes one or more blocks; a block never spans two
//! segments. For a capture with `core:sample_start` `s` and `core:global_index` `g` (default
//! `s`), the sample at data position `p` has the stream index `g + (p - s)`, and that is
//! [`BlockMeta::start`](sdrk_device::BlockMeta). A capture whose `g` lies above the end of the
//! previous capture starts a gap: its first block carries `GAP_BEFORE`. A changed
//! `core:frequency` sets `RETUNED`. `core:datetime` becomes a [`TimeAnchor`] with
//! `index = g` and source `Host` on the first block of the capture.
//!
//! The writer starts a new capture whenever a block has a gap (start index not equal to the
//! end of the previous block), an anchor, the `RETUNED` flag, or the frequency set with
//! [`SigmfWriter::set_frequency`] changed. Other flags (`OVERFLOW`, `UNDERRUN`, bursts, ...)
//! have no representation in SigMF and are not stored.
//!
//! Annotation positions (`sample_start`) are positions in the data file, as in the SigMF
//! specification, not stream indices.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sdrk_device::{
    BlockFlags, BlockMeta, Rate, RxStream, SampleFormat, SampleIndex, TimeAnchor, TimeSource,
    TxStream, formats,
};
use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::iso8601::{format_iso8601_ns, parse_iso8601_ns};
use crate::raw::{FlagCounts, check_whole, read_full};

/// Version written to `core:version`.
pub const SIGMF_VERSION: &str = "1.2.0";
const META_EXT: &str = ".sigmf-meta";
const DATA_EXT: &str = ".sigmf-data";

/// The `core:datatype` strings and the formats they stand for.
const DATATYPES: [(&str, SampleFormat); 6] = [
    ("cu8", formats::CU8),
    ("ci8", formats::CS8),
    ("ci16_le", formats::CS16),
    ("cf32_le", formats::CF32),
    ("ri16_le", formats::RS16),
    ("rf32_le", formats::RF32),
];

/// The format for a `core:datatype`.
///
/// ```
/// use sdrk_device::formats;
/// assert_eq!(sdrk_io::format_of_datatype("ci8")?, formats::CS8);
/// assert!(sdrk_io::format_of_datatype("cf32_be").is_err());
/// # Ok::<(), sdrk_io::Error>(())
/// ```
///
/// # Errors
/// `Unsupported` for every datatype outside the six that map to a format.
pub fn format_of_datatype(datatype: &str) -> Result<SampleFormat> {
    DATATYPES
        .iter()
        .find(|(name, _)| *name == datatype)
        .map(|(_, f)| *f)
        .ok_or(Error::Unsupported(
            "sigmf datatype (supported: cu8, ci8, ci16_le, cf32_le, ri16_le, rf32_le)",
        ))
}

/// The `core:datatype` for a format.
///
/// ```
/// use sdrk_device::formats;
/// assert_eq!(sdrk_io::datatype_of_format(&formats::CS16)?, "ci16_le");
/// # Ok::<(), sdrk_io::Error>(())
/// ```
///
/// # Errors
/// `Unsupported` for formats without a SigMF datatype.
pub fn datatype_of_format(format: &SampleFormat) -> Result<&'static str> {
    DATATYPES
        .iter()
        .find(|(_, f)| f == format)
        .map(|(name, _)| *name)
        .ok_or(Error::Unsupported("sample format without a sigmf datatype"))
}

/// A region of interest in the data file.
///
/// ```
/// use sdrk_io::Annotation;
/// let a = Annotation { sample_start: 10, sample_count: 5, label: Some("burst".into()), ..Annotation::default() };
/// assert_eq!(a.freq_lower_edge, None);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Annotation {
    /// Position of the first sample in the data file.
    pub sample_start: u64,
    /// Number of samples (0 when the file omits it).
    pub sample_count: u64,
    /// Lower edge of the feature in hertz.
    pub freq_lower_edge: Option<f64>,
    /// Upper edge of the feature in hertz.
    pub freq_upper_edge: Option<f64>,
    /// Short label.
    pub label: Option<String>,
    /// Free-text comment.
    pub comment: Option<String>,
}

/// One entry of the `captures` array.
///
/// ```
/// use sdrk_io::Capture;
/// let c = Capture { sample_start: 0, global_index: 0, datetime_ns: None, frequency_hz: Some(1e8) };
/// assert_eq!(c.global_index, 0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Capture {
    /// Position of the first sample of the segment in the data file.
    pub sample_start: u64,
    /// Stream index of that sample (`core:global_index`, default `sample_start`).
    pub global_index: u64,
    /// Time of that sample in unix nanoseconds (`core:datetime`).
    pub datetime_ns: Option<i64>,
    /// Centre frequency in hertz (`core:frequency`).
    pub frequency_hz: Option<f64>,
}

// ------------------------------------------------------------------------------------ JSON

/// Pretty-prints a JSON value with 2-space indentation, sorted keys and a final LF.
fn to_json_text(value: &Value) -> String {
    fn scalar(v: &Value) -> String {
        serde_json::to_string(v).unwrap_or_else(|_| "null".to_string())
    }
    fn emit(v: &Value, depth: usize, out: &mut String) {
        let pad = |n: usize| "  ".repeat(n);
        match v {
            Value::Object(map) if !map.is_empty() => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                out.push_str("{\n");
                for (i, k) in keys.iter().enumerate() {
                    out.push_str(&pad(depth + 1));
                    out.push_str(&scalar(&Value::String((*k).clone())));
                    out.push_str(": ");
                    if let Some(child) = map.get(*k) {
                        emit(child, depth + 1, out);
                    }
                    out.push_str(if i + 1 < keys.len() { ",\n" } else { "\n" });
                }
                out.push_str(&pad(depth));
                out.push('}');
            }
            Value::Array(items) if !items.is_empty() => {
                out.push_str("[\n");
                for (i, child) in items.iter().enumerate() {
                    out.push_str(&pad(depth + 1));
                    emit(child, depth + 1, out);
                    out.push_str(if i + 1 < items.len() { ",\n" } else { "\n" });
                }
                out.push_str(&pad(depth));
                out.push(']');
            }
            Value::Object(_) => out.push_str("{}"),
            Value::Array(_) => out.push_str("[]"),
            other => out.push_str(&scalar(other)),
        }
    }
    let mut out = String::new();
    emit(value, 0, &mut out);
    out.push('\n');
    out
}

fn num(v: f64) -> Value {
    serde_json::Number::from_f64(v).map_or(Value::Null, Value::Number)
}

/// The exact `Rate` of a `core:sample_rate` value.
///
/// An integer, or a decimal with at most six fractional digits (`num / 10^k`, reduced).
fn rate_of_number(v: f64) -> Result<Rate> {
    let bad = || {
        Error::invalid(
            "core:sample_rate must be an integer or a decimal with at most 6 fractional digits",
        )
    };
    if !(v.is_finite() && v > 0.0) {
        return Err(bad());
    }
    let text = format!("{v}");
    let (int, frac) = text.split_once('.').unwrap_or((text.as_str(), ""));
    if frac.len() > 6 {
        return Err(bad());
    }
    let scale = 10u64.pow(frac.len() as u32);
    let whole: u64 = int.parse().map_err(|_| bad())?;
    let frac_value: u64 = if frac.is_empty() {
        0
    } else {
        frac.parse().map_err(|_| bad())?
    };
    let num = whole
        .checked_mul(scale)
        .and_then(|w| w.checked_add(frac_value))
        .ok_or_else(bad)?;
    Ok(Rate::new(num, scale)?)
}

/// `core:sample_rate` for a rate; fails unless it has at most six fractional decimal digits.
fn number_of_rate(rate: Rate) -> Result<Value> {
    if rate.den() == 1 {
        return Ok(Value::from(rate.num()));
    }
    if 1_000_000 % rate.den() != 0 {
        return Err(Error::invalid(
            "the rate has more than 6 fractional decimal digits and cannot be written to sigmf",
        ));
    }
    Ok(num(rate.num() as f64 / rate.den() as f64))
}

// ------------------------------------------------------------------------------------ reader

fn field_u64(obj: &Map<String, Value>, key: &str, name: &str) -> Result<Option<u64>> {
    match obj.get(key) {
        None => Ok(None),
        Some(v) => v
            .as_u64()
            .map(Some)
            .ok_or_else(|| Error::parse(name, format!("{key} must be a non-negative integer"))),
    }
}

fn field_f64(obj: &Map<String, Value>, key: &str, name: &str) -> Result<Option<f64>> {
    match obj.get(key) {
        None => Ok(None),
        Some(v) => v
            .as_f64()
            .map(Some)
            .ok_or_else(|| Error::parse(name, format!("{key} must be a number"))),
    }
}

fn field_str(obj: &Map<String, Value>, key: &str, name: &str) -> Result<Option<String>> {
    match obj.get(key) {
        None => Ok(None),
        Some(v) => v
            .as_str()
            .map(|s| Some(s.to_string()))
            .ok_or_else(|| Error::parse(name, format!("{key} must be a string"))),
    }
}

fn object<'a>(v: &'a Value, what: &str, name: &str) -> Result<&'a Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| Error::parse(name, format!("{what} must be an object")))
}

fn parse_captures(root: &Map<String, Value>, name: &str) -> Result<Vec<Capture>> {
    let mut captures = Vec::new();
    if let Some(list) = root.get("captures") {
        let list = list
            .as_array()
            .ok_or_else(|| Error::parse(name, "`captures` must be an array"))?;
        for item in list {
            let c = object(item, "a capture", name)?;
            if field_u64(c, "core:header_bytes", name)?.is_some_and(|b| b != 0) {
                return Err(Error::Unsupported("sigmf captures with header bytes"));
            }
            let sample_start = field_u64(c, "core:sample_start", name)?.unwrap_or(0);
            let global_index = field_u64(c, "core:global_index", name)?.unwrap_or(sample_start);
            let datetime_ns = match field_str(c, "core:datetime", name)? {
                Some(text) => Some(parse_iso8601_ns(&text)?),
                None => None,
            };
            let frequency_hz = field_f64(c, "core:frequency", name)?;
            captures.push(Capture {
                sample_start,
                global_index,
                datetime_ns,
                frequency_hz,
            });
        }
    }
    if captures.is_empty() {
        captures.push(Capture {
            sample_start: 0,
            global_index: 0,
            datetime_ns: None,
            frequency_hz: None,
        });
    }
    if captures.first().is_some_and(|c| c.sample_start != 0) {
        return Err(Error::parse(
            name,
            "the first capture must start at sample 0",
        ));
    }
    if captures
        .iter()
        .zip(captures.iter().skip(1))
        .any(|(a, b)| a.sample_start > b.sample_start)
    {
        return Err(Error::parse(
            name,
            "captures are not sorted by `core:sample_start`",
        ));
    }
    Ok(captures)
}

fn parse_annotations(root: &Map<String, Value>, name: &str) -> Result<Vec<Annotation>> {
    let mut annotations = Vec::new();
    if let Some(list) = root.get("annotations") {
        let list = list
            .as_array()
            .ok_or_else(|| Error::parse(name, "`annotations` must be an array"))?;
        for item in list {
            let a = object(item, "an annotation", name)?;
            annotations.push(Annotation {
                sample_start: field_u64(a, "core:sample_start", name)?.unwrap_or(0),
                sample_count: field_u64(a, "core:sample_count", name)?.unwrap_or(0),
                freq_lower_edge: field_f64(a, "core:freq_lower_edge", name)?,
                freq_upper_edge: field_f64(a, "core:freq_upper_edge", name)?,
                label: field_str(a, "core:label", name)?,
                comment: field_str(a, "core:comment", name)?,
            });
        }
    }
    Ok(annotations)
}

/// Reads a SigMF recording as an RX stream. The data can come from any reader.
///
/// ```
/// use sdrk_device::{formats, BlockFlags, RxStream};
/// use sdrk_io::SigmfReader;
/// let meta = r#"{
///   "global": { "core:datatype": "cu8", "core:sample_rate": 2000000, "core:version": "1.0.0" },
///   "captures": [
///     { "core:sample_start": 0 },
///     { "core:sample_start": 2, "core:global_index": 12 }
///   ],
///   "annotations": []
/// }"#;
/// let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
/// let mut r = SigmfReader::new(meta, &data[..])?;
/// assert_eq!(r.format(), formats::CU8);
/// let mut buf = [0u8; 64];
/// let (n, first) = r.read(&mut buf)?;
/// assert_eq!((n, first.start.0), (4, 0));
/// let (n, second) = r.read(&mut buf)?;
/// assert_eq!((n, second.start.0), (4, 12));
/// assert!(second.flags.contains(BlockFlags::GAP_BEFORE));
/// # Ok::<(), sdrk_io::Error>(())
/// ```
#[derive(Debug)]
pub struct SigmfReader<R: Read = File> {
    data: R,
    format: SampleFormat,
    rate: Rate,
    captures: Vec<Capture>,
    annotations: Vec<Annotation>,
    description: Option<String>,
    sample_bytes: usize,
    chunk_bytes: usize,
    pos: u64,
    seg: usize,
    seg_started: bool,
    last_end: Option<u64>,
    last_freq: Option<f64>,
    eof: bool,
}

impl SigmfReader<File> {
    /// Opens `<base>.sigmf-meta` and `<base>.sigmf-data`. `base` may be given with or
    /// without either extension.
    ///
    /// # Errors
    /// `Io` if a file cannot be read, and the errors of [`SigmfReader::new`].
    pub fn open(base: impl AsRef<Path>) -> Result<Self> {
        let (meta_path, data_path) = sigmf_paths(base.as_ref());
        let meta = std::fs::read_to_string(&meta_path)
            .map_err(|e| Error::Io(format!("{}: {e}", meta_path.display())))?;
        let data = File::open(&data_path)
            .map_err(|e| Error::Io(format!("{}: {e}", data_path.display())))?;
        Self::new_named(&meta_path.display().to_string(), &meta, data)
    }
}

/// The meta and data paths for a base path with or without extension.
pub fn sigmf_paths(base: &Path) -> (PathBuf, PathBuf) {
    let text = base.to_string_lossy();
    let stem = text
        .strip_suffix(META_EXT)
        .or_else(|| text.strip_suffix(DATA_EXT))
        .unwrap_or(&text)
        .to_string();
    (
        PathBuf::from(format!("{stem}{META_EXT}")),
        PathBuf::from(format!("{stem}{DATA_EXT}")),
    )
}

impl<R: Read> SigmfReader<R> {
    /// Parses the metadata text; `data` supplies the samples.
    ///
    /// # Errors
    /// `Parse` for malformed metadata, `Unsupported` for other datatypes, a major version
    /// other than 1, non-conforming datasets or header bytes, and `InvalidArgument` for a
    /// sample rate that is not an exact decimal with at most six fractional digits.
    pub fn new(meta_json: &str, data: R) -> Result<Self> {
        Self::new_named("sigmf-meta", meta_json, data)
    }

    #[allow(
        clippy::large_stack_frames,
        reason = "clippy counts the temporary of every `?` separately; the real frame is small"
    )]
    fn new_named(name: &str, meta_json: &str, data: R) -> Result<Self> {
        let root: Value =
            serde_json::from_str(meta_json).map_err(|e| Error::parse(name, e.to_string()))?;
        let root = object(&root, "the document", name)?;
        let global = object(
            root.get("global")
                .ok_or_else(|| Error::parse(name, "missing `global`"))?,
            "`global`",
            name,
        )?;
        let version = field_str(global, "core:version", name)?
            .ok_or_else(|| Error::parse(name, "missing `core:version`"))?;
        if !version.starts_with("1.") {
            return Err(Error::Unsupported("sigmf version (only 1.x)"));
        }
        if global.contains_key("core:dataset")
            || global
                .get("core:metadata_only")
                .is_some_and(|v| v == &Value::Bool(true))
        {
            return Err(Error::Unsupported(
                "sigmf non-conforming or metadata-only datasets",
            ));
        }
        let datatype = field_str(global, "core:datatype", name)?
            .ok_or_else(|| Error::parse(name, "missing `core:datatype`"))?;
        let format = format_of_datatype(&datatype)?;
        let rate_value = field_f64(global, "core:sample_rate", name)?
            .ok_or_else(|| Error::parse(name, "missing `core:sample_rate`"))?;
        let rate = rate_of_number(rate_value)?;
        let description = field_str(global, "core:description", name)?;

        let captures = parse_captures(root, name)?;
        let annotations = parse_annotations(root, name)?;
        let sample_bytes = format.bytes_for_samples(1)?;
        Ok(Self {
            data,
            format,
            rate,
            captures,
            annotations,
            description,
            sample_bytes,
            chunk_bytes: usize::MAX,
            pos: 0,
            seg: 0,
            seg_started: false,
            last_end: None,
            last_freq: None,
            eof: false,
        })
    }

    /// Limits the number of samples per `read`.
    ///
    /// # Errors
    /// `InvalidArgument` if `chunk_samples` is zero.
    pub fn with_chunk_samples(mut self, chunk_samples: usize) -> Result<Self> {
        if chunk_samples == 0 {
            return Err(Error::invalid("chunk size must be at least one sample"));
        }
        self.chunk_bytes = chunk_samples.saturating_mul(self.sample_bytes);
        Ok(self)
    }

    /// The annotations of the recording, in file order.
    pub fn annotations(&self) -> &[Annotation] {
        &self.annotations
    }

    /// The captures of the recording.
    pub fn captures(&self) -> &[Capture] {
        &self.captures
    }

    /// `core:description`, if present.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

impl<R: Read> RxStream for SigmfReader<R> {
    type Error = Error;

    fn format(&self) -> SampleFormat {
        self.format
    }

    fn sample_rate(&self) -> Rate {
        self.rate
    }

    fn read(&mut self, buf: &mut [u8]) -> Result<(usize, BlockMeta)> {
        while self
            .captures
            .get(self.seg + 1)
            .is_some_and(|next| next.sample_start <= self.pos)
        {
            self.seg += 1;
            self.seg_started = false;
        }
        let cap = self.captures.get(self.seg).cloned().ok_or(Error::Device(
            sdrk_device::Error::InvalidFormat("no capture"),
        ))?;
        let start_index = cap.global_index.saturating_add(self.pos - cap.sample_start);
        let mut want = buf.len().min(self.chunk_bytes);
        if let Some(next) = self.captures.get(self.seg + 1) {
            let room = usize::try_from(
                (next.sample_start - self.pos).saturating_mul(self.sample_bytes as u64),
            )
            .unwrap_or(usize::MAX);
            want = want.min(room);
        }
        want -= want % self.sample_bytes;
        if want == 0 {
            return Err(Error::Device(sdrk_device::Error::BufferTooSmall {
                needed: self.sample_bytes,
                got: buf.len(),
            }));
        }
        let mut got = 0;
        if !self.eof {
            let target = buf
                .get_mut(..want)
                .ok_or_else(|| Error::invalid("buffer shorter than requested"))?;
            got = read_full(&mut self.data, target)?;
            if got < want {
                self.eof = true;
            }
            got -= got % self.sample_bytes;
        }
        let mut meta = BlockMeta {
            start: SampleIndex(start_index),
            ..BlockMeta::default()
        };
        if got == 0 {
            return Ok((0, meta));
        }
        if !self.seg_started {
            self.seg_started = true;
            if let Some(end) = self.last_end {
                if cap.global_index > end {
                    meta.flags.insert(BlockFlags::GAP_BEFORE);
                } else if cap.global_index < end {
                    return Err(Error::parse(
                        "sigmf-meta",
                        "`core:global_index` of a capture lies before the end of the previous one",
                    ));
                }
            }
            if let (Some(prev), Some(now)) = (self.last_freq, cap.frequency_hz)
                && prev != now
            {
                meta.flags.insert(BlockFlags::RETUNED);
            }
            if let Some(ns) = cap.datetime_ns {
                meta.anchor = Some(TimeAnchor {
                    index: SampleIndex(cap.global_index),
                    unix_ns: ns,
                    source: TimeSource::Host,
                });
            }
        }
        if cap.frequency_hz.is_some() {
            self.last_freq = cap.frequency_hz;
        }
        let samples = (got / self.sample_bytes) as u64;
        self.pos += samples;
        self.last_end = Some(start_index.saturating_add(samples));
        Ok((got, meta))
    }
}

// ------------------------------------------------------------------------------------ writer

/// Writes a SigMF recording.
///
/// `write` appends samples to the data stream and opens a new capture where needed (see the
/// module documentation); [`SigmfWriter::annotate`] collects annotations; [`SigmfWriter::finish`]
/// produces the metadata. The writer implements [`TxStream`] so that it can be driven like a
/// transmitter.
///
/// ```
/// use sdrk_device::{formats, BlockMeta, Rate, SampleIndex, TxStream};
/// use sdrk_io::{Annotation, SigmfWriter};
/// let mut w = SigmfWriter::new(Vec::new(), formats::CU8, Rate::hz(2_000_000)?)?;
/// w.write(&[1, 2, 3, 4], &BlockMeta::default())?;
/// let gap = BlockMeta { start: SampleIndex(10), ..BlockMeta::default() };
/// w.write(&[5, 6], &gap)?; // 8 samples are missing: a new capture
/// w.annotate(Annotation { sample_start: 0, sample_count: 3, ..Annotation::default() })?;
/// let (data, meta) = w.finish()?;
/// assert_eq!(data, [1, 2, 3, 4, 5, 6]);
/// assert!(meta.contains("\"core:global_index\": 10"));
/// # Ok::<(), sdrk_io::Error>(())
/// ```
#[derive(Debug)]
pub struct SigmfWriter<W: Write = File> {
    data: W,
    meta_path: Option<PathBuf>,
    format: SampleFormat,
    rate: Rate,
    datatype: &'static str,
    position: u64,
    expected: Option<u64>,
    frequency: Option<f64>,
    open_frequency: Option<f64>,
    captures: Vec<Capture>,
    annotations: Vec<Annotation>,
    description: Option<String>,
    counts: FlagCounts,
}

impl SigmfWriter<File> {
    /// Creates `<base>.sigmf-data` now; [`SigmfWriter::finish`] writes `<base>.sigmf-meta`.
    ///
    /// # Errors
    /// `Io`, and the errors of [`SigmfWriter::new`].
    pub fn create(base: impl AsRef<Path>, format: SampleFormat, rate: Rate) -> Result<Self> {
        let (meta_path, data_path) = sigmf_paths(base.as_ref());
        let file = File::create(&data_path)
            .map_err(|e| Error::Io(format!("{}: {e}", data_path.display())))?;
        let mut w = Self::new(file, format, rate)?;
        w.meta_path = Some(meta_path);
        Ok(w)
    }
}

impl<W: Write> SigmfWriter<W> {
    /// Creates a writer around any data sink.
    ///
    /// # Errors
    /// `Unsupported` for a format without a SigMF datatype, `InvalidArgument` for a rate with
    /// more than six fractional decimal digits.
    pub fn new(data: W, format: SampleFormat, rate: Rate) -> Result<Self> {
        let datatype = datatype_of_format(&format)?;
        number_of_rate(rate)?;
        Ok(Self {
            data,
            meta_path: None,
            format,
            rate,
            datatype,
            position: 0,
            expected: None,
            frequency: None,
            open_frequency: None,
            captures: Vec::new(),
            annotations: Vec::new(),
            description: None,
            counts: FlagCounts::default(),
        })
    }

    /// Sets `core:description`.
    pub fn set_description(&mut self, text: &str) {
        self.description = Some(text.to_string());
    }

    /// Sets the centre frequency stored in the captures that start from now on. A change
    /// opens a new capture at the next `write`.
    ///
    /// # Errors
    /// `InvalidArgument` if `hz` is not finite.
    pub fn set_frequency(&mut self, hz: f64) -> Result<()> {
        if !hz.is_finite() {
            return Err(Error::invalid("the frequency must be finite"));
        }
        self.frequency = Some(hz);
        Ok(())
    }

    /// Adds an annotation.
    ///
    /// # Errors
    /// `InvalidArgument` if a frequency edge is not finite.
    pub fn annotate(&mut self, annotation: Annotation) -> Result<()> {
        let finite = |v: Option<f64>| v.is_none_or(f64::is_finite);
        if !finite(annotation.freq_lower_edge) || !finite(annotation.freq_upper_edge) {
            return Err(Error::invalid("annotation frequency edges must be finite"));
        }
        self.annotations.push(annotation);
        Ok(())
    }

    /// The flags seen so far, counted per flag.
    pub fn flag_counts(&self) -> FlagCounts {
        self.counts
    }

    /// Number of samples written to the data stream.
    pub fn samples_written(&self) -> u64 {
        self.position
    }

    fn metadata(&self) -> Result<Value> {
        let mut global = Map::new();
        global.insert("core:datatype".into(), Value::from(self.datatype));
        global.insert("core:sample_rate".into(), number_of_rate(self.rate)?);
        global.insert("core:version".into(), Value::from(SIGMF_VERSION));
        if let Some(d) = &self.description {
            global.insert("core:description".into(), Value::from(d.as_str()));
        }
        let captures: Vec<Value> = self
            .captures
            .iter()
            .map(|c| {
                let mut m = Map::new();
                m.insert("core:sample_start".into(), Value::from(c.sample_start));
                if c.global_index != c.sample_start {
                    m.insert("core:global_index".into(), Value::from(c.global_index));
                }
                if let Some(ns) = c.datetime_ns {
                    m.insert("core:datetime".into(), Value::from(format_iso8601_ns(ns)));
                }
                if let Some(f) = c.frequency_hz {
                    m.insert("core:frequency".into(), num(f));
                }
                Value::Object(m)
            })
            .collect();
        let annotations: Vec<Value> = self
            .annotations
            .iter()
            .map(|a| {
                let mut m = Map::new();
                m.insert("core:sample_start".into(), Value::from(a.sample_start));
                m.insert("core:sample_count".into(), Value::from(a.sample_count));
                if let Some(f) = a.freq_lower_edge {
                    m.insert("core:freq_lower_edge".into(), num(f));
                }
                if let Some(f) = a.freq_upper_edge {
                    m.insert("core:freq_upper_edge".into(), num(f));
                }
                if let Some(s) = &a.label {
                    m.insert("core:label".into(), Value::from(s.as_str()));
                }
                if let Some(s) = &a.comment {
                    m.insert("core:comment".into(), Value::from(s.as_str()));
                }
                Value::Object(m)
            })
            .collect();
        let mut root = Map::new();
        root.insert("global".into(), Value::Object(global));
        root.insert("captures".into(), Value::Array(captures));
        root.insert("annotations".into(), Value::Array(annotations));
        Ok(Value::Object(root))
    }

    /// Flushes the data and produces the metadata. Returns the data sink and the text of the
    /// `.sigmf-meta` file (sorted keys, 2-space indent, LF); a writer made by
    /// [`SigmfWriter::create`] also writes that text to `<base>.sigmf-meta`.
    ///
    /// # Errors
    /// `Io`.
    pub fn finish(mut self) -> Result<(W, String)> {
        self.data.flush()?;
        let text = to_json_text(&self.metadata()?);
        if let Some(path) = &self.meta_path {
            std::fs::write(path, text.as_bytes())
                .map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
        }
        Ok((self.data, text))
    }
}

impl<W: Write> TxStream for SigmfWriter<W> {
    type Error = Error;

    fn format(&self) -> SampleFormat {
        self.format
    }

    fn sample_rate(&self) -> Rate {
        self.rate
    }

    fn write(&mut self, buf: &[u8], meta: &BlockMeta) -> Result<usize> {
        let samples = check_whole(&self.format, buf.len())? as u64;
        if buf.is_empty() {
            return Ok(0);
        }
        let start = meta.start.0;
        if self.expected.is_some_and(|e| start < e) {
            return Err(Error::invalid(
                "the block starts before the end of the previous block",
            ));
        }
        let new_capture = self.captures.is_empty()
            || self.expected != Some(start)
            || meta.flags.contains(BlockFlags::GAP_BEFORE)
            || meta.flags.contains(BlockFlags::RETUNED)
            || meta.anchor.is_some()
            || self.frequency != self.open_frequency;
        if new_capture {
            let datetime_ns = match meta.anchor {
                Some(a) => Some(a.time_of(meta.start, self.rate)?),
                None => None,
            };
            self.captures.push(Capture {
                sample_start: self.position,
                global_index: start,
                datetime_ns,
                frequency_hz: self.frequency,
            });
            self.open_frequency = self.frequency;
        }
        self.data.write_all(buf)?;
        self.position += samples;
        self.expected = Some(start.saturating_add(samples));
        self.counts.record(meta.flags);
        Ok(buf.len())
    }

    fn poll_status(&mut self) -> Result<Option<BlockMeta>> {
        Ok(None)
    }
}
