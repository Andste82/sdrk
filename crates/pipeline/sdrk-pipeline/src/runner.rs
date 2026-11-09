//! The push-mode runner: linear buffers with zero-copy fan-out, output queues, statistics.
//!
//! Everything is allocated by `build`. The hot path (`push`, `read_*`, `set_param`) allocates
//! nothing unless it returns an error.

use std::collections::BTreeMap;
use std::ops::Range;

use sdrk::{C32, CQ15, Delay, Progress, Q15, Rate, Sample, SampleType};
use sdrk_blocks::{FrameMeta, Packet, ParamValue, PortType};
use sdrk_device::TimeAnchor;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::model::{DropPolicy, PipelineSpec, UnderrunPolicy};
use crate::node::{Node, NodeElem, NodeInput, NodeOutput};
use crate::params::{to_json, to_param};
use crate::time::EventQueue;
use crate::validate::Report;

/// Data pushed into one input of a pipeline.
///
/// The variant must match the input: `Raw` for inputs with a `format`, the sample type for
/// inputs with a `type`, `Packets` for packet inputs (see also [`Pipeline::push_packets`]).
///
/// ```
/// use sdrk::C32;
/// use sdrk_pipeline::InputData;
/// let samples = [C32::new(1.0, 0.0); 3];
/// assert_eq!(InputData::C32(&samples).len(), 3);
/// assert!(InputData::Raw(&[]).is_empty());
/// ```
#[derive(Debug, Clone, Copy)]
pub enum InputData<'a> {
    /// Raw bytes in the format of the input.
    Raw(&'a [u8]),
    /// Complex `f32` samples.
    C32(&'a [C32]),
    /// Complex Q15 samples.
    CQ15(&'a [CQ15]),
    /// Real `f32` samples.
    F32(&'a [f32]),
    /// Real Q15 samples.
    Q15(&'a [Q15]),
    /// Packets. In pull mode an empty slice from a packet input means "nothing to send now":
    /// the encoder fills the output with its idle signal.
    Packets(&'a [Packet]),
}

impl InputData<'_> {
    /// Number of elements (bytes for `Raw`, packets for `Packets`).
    pub fn len(&self) -> usize {
        match self {
            Self::Raw(s) => s.len(),
            Self::C32(s) => s.len(),
            Self::CQ15(s) => s.len(),
            Self::F32(s) => s.len(),
            Self::Q15(s) => s.len(),
            Self::Packets(s) => s.len(),
        }
    }

    /// True if there is no data.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn port_type(&self) -> PortType {
        match self {
            Self::Raw(_) => PortType::Raw,
            Self::C32(_) => PortType::C32,
            Self::CQ15(_) => PortType::CQ15,
            Self::F32(_) => PortType::F32,
            Self::Q15(_) => PortType::Q15,
            Self::Packets(_) => PortType::Packets,
        }
    }

    /// `limit` elements from element `from` on, as a node input.
    fn view(&self, from: usize, limit: usize) -> Option<NodeInput<'_>> {
        let to = from.checked_add(limit)?;
        Some(match self {
            Self::Raw(s) => NodeInput::Raw(s.get(from..to)?),
            Self::C32(s) => NodeInput::C32(s.get(from..to)?),
            Self::CQ15(s) => NodeInput::CQ15(s.get(from..to)?),
            Self::F32(s) => NodeInput::F32(s.get(from..to)?),
            Self::Q15(s) => NodeInput::Q15(s.get(from..to)?),
            Self::Packets(s) => NodeInput::Packets(s.get(from..to)?),
        })
    }
}

/// Statistics of one node (automatic nodes included).
///
/// ```
/// use sdrk_pipeline::NodeStats;
/// let s = NodeStats {
///     name: "afc".into(),
///     calls: 0,
///     samples_in: 0,
///     samples_out: 0,
///     frames: 0,
///     readings: vec![("offset_hz", 0.0)],
/// };
/// assert_eq!(s.calls, 0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct NodeStats {
    /// Node name.
    pub name: String,
    /// Number of `process` calls.
    pub calls: u64,
    /// Elements consumed (bytes for `convert`).
    pub samples_in: u64,
    /// Elements produced (bytes for `convert_out`).
    pub samples_out: u64,
    /// Frames produced by a frame node.
    pub frames: u64,
    /// The readings of the block (`Block::reading`, for example `offset_hz` and `offset_ppm`
    /// of `afc`) as of its last call or `reset`; empty for blocks without readings.
    pub readings: Vec<(&'static str, f64)>,
}

/// Copies the current readings of a node into its statistics (no allocation: the names and
/// their number are fixed at build time).
pub(crate) fn update_readings(node: &dyn Node, st: &mut NodeStats) {
    for (i, r) in st.readings.iter_mut().enumerate() {
        if let Some(v) = node.reading(i) {
            *r = v;
        }
    }
}

pub(crate) fn name_of(t: PortType) -> &'static str {
    match t {
        PortType::C32 => "c32",
        PortType::CQ15 => "cq15",
        PortType::F32 => "f32",
        PortType::Q15 => "q15",
        PortType::Raw => "raw",
        PortType::Packets => "packets",
        PortType::Lanes { .. } => "lanes",
        _ => "frames",
    }
}

pub(crate) fn sample_port(t: SampleType) -> PortType {
    match t {
        SampleType::C32 => PortType::C32,
        SampleType::CQ15 => PortType::CQ15,
        SampleType::F32 => PortType::F32,
        _ => PortType::Q15,
    }
}

pub(crate) fn internal(m: &'static str) -> Error {
    Error::Internal(m)
}

pub(crate) fn at<T>(v: &[T], i: usize) -> Result<&T> {
    v.get(i).ok_or_else(|| internal("index out of range"))
}

pub(crate) fn at_mut<T>(v: &mut [T], i: usize) -> Result<&mut T> {
    v.get_mut(i).ok_or_else(|| internal("index out of range"))
}

/// Two distinct elements of a slice, mutably, in the order of the indices.
fn pair_mut<T>(v: &mut [T], first: usize, second: usize) -> Result<(&mut T, &mut T)> {
    let bad = || internal("stream pair");
    match first.cmp(&second) {
        core::cmp::Ordering::Less => {
            let (head, tail) = v.split_at_mut_checked(second).ok_or_else(bad)?;
            Ok((
                head.get_mut(first).ok_or_else(bad)?,
                tail.first_mut().ok_or_else(bad)?,
            ))
        }
        core::cmp::Ordering::Greater => {
            let (head, tail) = v.split_at_mut_checked(first).ok_or_else(bad)?;
            Ok((
                tail.first_mut().ok_or_else(bad)?,
                head.get_mut(second).ok_or_else(bad)?,
            ))
        }
        core::cmp::Ordering::Equal => Err(bad()),
    }
}

// ------------------------------------------------------------------ buffers

/// A typed linear buffer.
///
/// A lane buffer holds whole time steps of `width` values; every position, length and range
/// of it is in steps (the value `i` of step `s` is `data[s * width + i]`), so the stream logic
/// of the runner is the same for lanes.
pub(crate) enum Buffer {
    Raw(Vec<u8>),
    C32(Vec<C32>),
    CQ15(Vec<CQ15>),
    F32(Vec<f32>),
    Q15(Vec<Q15>),
    Packets(Vec<Packet>),
    Lanes { data: Vec<C32>, width: usize },
}

macro_rules! each {
    ($b:expr, $v:ident => $e:expr, $data:ident, $width:ident => $lanes:expr) => {
        match $b {
            Buffer::Raw($v) => $e,
            Buffer::C32($v) => $e,
            Buffer::CQ15($v) => $e,
            Buffer::F32($v) => $e,
            Buffer::Q15($v) => $e,
            Buffer::Packets($v) => $e,
            Buffer::Lanes {
                data: $data,
                width: $width,
            } => $lanes,
        }
    };
}

/// The values of the steps `r` of a lane buffer of `width` lanes.
fn lane_values(r: Range<usize>, width: usize) -> Option<Range<usize>> {
    Some(r.start.checked_mul(width)?..r.end.checked_mul(width)?)
}

/// `dst[dst_at..dst_at + n] = src[from..from + n]`.
fn copy_slice<T: Copy>(dst: &mut [T], dst_at: usize, src: &[T], from: usize, n: usize) -> bool {
    match (
        dst_at.checked_add(n).and_then(|e| dst.get_mut(dst_at..e)),
        from.checked_add(n).and_then(|e| src.get(from..e)),
    ) {
        (Some(d), Some(s)) => {
            d.copy_from_slice(s);
            true
        }
        _ => false,
    }
}

impl Buffer {
    pub(crate) fn new(ty: PortType, len: usize) -> Result<Self, sdrk_blocks::Error> {
        use crate::storage::alloc_vec;
        Ok(match ty {
            PortType::Raw => Self::Raw(alloc_vec(len)?),
            PortType::C32 => Self::C32(alloc_vec(len)?),
            PortType::CQ15 => Self::CQ15(alloc_vec(len)?),
            PortType::F32 => Self::F32(alloc_vec(len)?),
            PortType::Q15 => Self::Q15(alloc_vec(len)?),
            PortType::Packets => Self::Packets(alloc_vec(len)?),
            _ => {
                return Err(sdrk_blocks::Error::InvalidParameter {
                    name: "buffer",
                    reason: "frames and lanes have no stream buffer",
                });
            }
        })
    }

    /// A lane buffer of `steps` time steps of `width` `C32` values.
    pub(crate) fn lanes(width: usize, steps: usize) -> Result<Self, sdrk_blocks::Error> {
        let values = steps
            .checked_mul(width)
            .ok_or(sdrk_blocks::Error::Storage {
                needed: usize::MAX,
                available: 0,
            })?;
        Ok(Self::Lanes {
            data: crate::storage::alloc_vec(values)?,
            width,
        })
    }

    /// Number of elements (time steps of a lane buffer).
    pub(crate) fn len(&self) -> usize {
        each!(self, v => v.len(), data, width => data.len() / (*width).max(1))
    }

    pub(crate) fn view(&self, r: Range<usize>) -> Option<NodeInput<'_>> {
        Some(match self {
            Self::Raw(v) => NodeInput::Raw(v.get(r)?),
            Self::C32(v) => NodeInput::C32(v.get(r)?),
            Self::CQ15(v) => NodeInput::CQ15(v.get(r)?),
            Self::F32(v) => NodeInput::F32(v.get(r)?),
            Self::Q15(v) => NodeInput::Q15(v.get(r)?),
            Self::Packets(v) => NodeInput::Packets(v.get(r)?),
            Self::Lanes { data, width } => NodeInput::Lanes {
                data: data.get(lane_values(r, *width)?)?,
                width: *width,
            },
        })
    }

    /// The packets in `r` (`None` unless this is a packet buffer).
    pub(crate) fn packets(&self, r: Range<usize>) -> Option<&[Packet]> {
        match self {
            Self::Packets(v) => v.get(r),
            _ => None,
        }
    }

    /// The same view as [`Buffer::view`], for one port of a two-input node.
    pub(crate) fn elem_view(&self, r: Range<usize>) -> Option<NodeElem<'_>> {
        Some(match self {
            Self::Raw(v) => NodeElem::Raw(v.get(r)?),
            Self::C32(v) => NodeElem::C32(v.get(r)?),
            Self::CQ15(v) => NodeElem::CQ15(v.get(r)?),
            Self::F32(v) => NodeElem::F32(v.get(r)?),
            Self::Q15(v) => NodeElem::Q15(v.get(r)?),
            // Packets and lanes are never one stream of a lockstep pair.
            Self::Packets(_) | Self::Lanes { .. } => return None,
        })
    }

    pub(crate) fn view_mut(&mut self, r: Range<usize>) -> Option<NodeOutput<'_>> {
        Some(match self {
            Self::Raw(v) => NodeOutput::Raw(v.get_mut(r)?),
            Self::C32(v) => NodeOutput::C32(v.get_mut(r)?),
            Self::CQ15(v) => NodeOutput::CQ15(v.get_mut(r)?),
            Self::F32(v) => NodeOutput::F32(v.get_mut(r)?),
            Self::Q15(v) => NodeOutput::Q15(v.get_mut(r)?),
            Self::Packets(v) => NodeOutput::Packets(v.get_mut(r)?),
            Self::Lanes { data, width } => NodeOutput::Lanes {
                data: data.get_mut(lane_values(r, *width)?)?,
                width: *width,
            },
        })
    }

    /// Moves `r` to the front at `dst`.
    pub(crate) fn move_within(&mut self, r: Range<usize>, dst: usize) {
        fn within<T: Copy>(v: &mut [T], r: Range<usize>, dst: usize) {
            let n = r.end.saturating_sub(r.start);
            if r.end <= v.len() && r.start <= r.end && dst.saturating_add(n) <= v.len() {
                v.copy_within(r, dst);
            }
        }
        each!(self, v => within(v, r, dst), data, width => {
            if let (Some(values), Some(at)) = (lane_values(r, *width), dst.checked_mul(*width)) {
                within(data, values, at);
            }
        });
    }

    /// Copies `n` elements of `src` starting at `from` to `at` of this buffer.
    pub(crate) fn copy_from(&mut self, at: usize, src: &Buffer, from: usize, n: usize) -> bool {
        match (self, src) {
            (Self::Raw(d), Self::Raw(s)) => copy_slice(d, at, s, from, n),
            (Self::C32(d), Self::C32(s)) => copy_slice(d, at, s, from, n),
            (Self::CQ15(d), Self::CQ15(s)) => copy_slice(d, at, s, from, n),
            (Self::F32(d), Self::F32(s)) => copy_slice(d, at, s, from, n),
            (Self::Q15(d), Self::Q15(s)) => copy_slice(d, at, s, from, n),
            (Self::Packets(d), Self::Packets(s)) => copy_slice(d, at, s, from, n),
            (Self::Lanes { data: d, width: w }, Self::Lanes { data: s, width: ws }) if w == ws => {
                match (at.checked_mul(*w), from.checked_mul(*w), n.checked_mul(*w)) {
                    (Some(at), Some(from), Some(n)) => copy_slice(d, at, s, from, n),
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Copies `n` elements of the pushed data starting at `from` to `at`.
    pub(crate) fn copy_from_input(
        &mut self,
        at: usize,
        src: &InputData<'_>,
        from: usize,
        n: usize,
    ) -> bool {
        match (self, src) {
            (Self::Raw(d), InputData::Raw(s)) => copy_slice(d, at, s, from, n),
            (Self::C32(d), InputData::C32(s)) => copy_slice(d, at, s, from, n),
            (Self::CQ15(d), InputData::CQ15(s)) => copy_slice(d, at, s, from, n),
            (Self::F32(d), InputData::F32(s)) => copy_slice(d, at, s, from, n),
            (Self::Q15(d), InputData::Q15(s)) => copy_slice(d, at, s, from, n),
            (Self::Packets(d), InputData::Packets(s)) => copy_slice(d, at, s, from, n),
            _ => false,
        }
    }

    /// Copies `n` elements starting at `from` into `out`, which has the same element type.
    pub(crate) fn copy_to<T: bytemuck::Pod>(&self, from: usize, n: usize, out: &mut [T]) -> bool {
        fn bytes_of<E: bytemuck::Pod>(v: &[E], from: usize, n: usize) -> Option<&[u8]> {
            let end = from.checked_add(n)?;
            Some(bytemuck::cast_slice(v.get(from..end)?))
        }
        let size = core::mem::size_of::<T>();
        let bytes = match self {
            Self::Raw(v) => bytes_of(v, from, n),
            Self::C32(v) => bytes_of(v, from, n),
            Self::CQ15(v) => bytes_of(v, from, n),
            Self::F32(v) => bytes_of(v, from, n),
            Self::Q15(v) => bytes_of(v, from, n),
            // Packets are not plain bytes (see `copy_packets_to`); no output reads lanes.
            Self::Packets(_) | Self::Lanes { .. } => None,
        };
        let dst: &mut [u8] = bytemuck::cast_slice_mut(out);
        match (bytes, dst.get_mut(..n.saturating_mul(size))) {
            (Some(s), Some(d)) if s.len() == d.len() => {
                d.copy_from_slice(s);
                true
            }
            _ => false,
        }
    }

    /// Copies `n` packets starting at `from` into `out`.
    pub(crate) fn copy_packets_to(&self, from: usize, n: usize, out: &mut [Packet]) -> bool {
        match self {
            Self::Packets(v) => copy_slice(out, 0, v, from, n),
            _ => false,
        }
    }

    /// Adds `offset` to the index of the packets in `r` (a decoder's packets move from the
    /// block's count to the index domain of the node's input).
    fn shift_packets(&mut self, r: Range<usize>, offset: u64) {
        if let Self::Packets(v) = self
            && let Some(packets) = v.get_mut(r)
        {
            for p in packets {
                p.index = p.index.saturating_add(offset);
            }
        }
    }
}

/// A linear buffer written by one node and read by any number of readers, each at its own
/// position.
pub(crate) struct Stream {
    pub(crate) buf: Buffer,
    pub(crate) end: usize,
    pub(crate) readers: Vec<usize>,
    /// Bytes and samples of one packing group (`(1, 1)` unless the stream is raw).
    pub(crate) elems: (u64, u64),
    /// Sample index of the element counter value `origin_el`.
    pub(crate) origin_idx: u64,
    pub(crate) origin_el: u64,
    /// Elements written since the build.
    pub(crate) total_el: u64,
}

/// Samples in `el` elements of a stream with `elems` (bytes, samples) per packing group.
pub(crate) fn samples_of(el: u64, elems: (u64, u64)) -> u64 {
    let v = u128::from(el) * u128::from(elems.1) / u128::from(elems.0.max(1));
    u64::try_from(v).unwrap_or(u64::MAX)
}

/// Elements that hold `samples` samples (rounded up to whole elements).
pub(crate) fn elems_of(samples: u64, elems: (u64, u64)) -> u64 {
    let v = (u128::from(samples) * u128::from(elems.0)).div_ceil(u128::from(elems.1.max(1)));
    u64::try_from(v).unwrap_or(u64::MAX)
}

impl Stream {
    pub(crate) fn cap(&self) -> usize {
        self.buf.len()
    }

    /// Sample index of the element with the absolute counter value `total`.
    pub(crate) fn idx_of(&self, total: u64) -> u64 {
        self.origin_idx
            .saturating_add(samples_of(total.saturating_sub(self.origin_el), self.elems))
    }

    /// Sample index of the next element that will be written.
    pub(crate) fn next_idx(&self) -> u64 {
        self.idx_of(self.total_el)
    }

    /// Sample index of the element at buffer position `pos`.
    pub(crate) fn idx_of_pos(&self, pos: usize) -> u64 {
        let behind = self.end.saturating_sub(pos) as u64;
        self.idx_of(self.total_el.saturating_sub(behind))
    }

    /// Empties the buffer and starts a new index origin at `idx`.
    pub(crate) fn rebase(&mut self, idx: u64) {
        self.clear();
        self.origin_idx = idx;
        self.origin_el = self.total_el;
    }

    /// Moves the unread data to the front of the buffer.
    fn compact(&mut self) {
        let base = self.readers.iter().copied().min().unwrap_or(self.end);
        if base == 0 {
            return;
        }
        self.buf.move_within(base..self.end, 0);
        self.end = self.end.saturating_sub(base);
        for r in &mut self.readers {
            *r = r.saturating_sub(base);
        }
    }

    /// Restarts at the front when every reader has caught up.
    fn normalize(&mut self) {
        if self.end != 0 && self.readers.iter().all(|r| *r >= self.end) {
            self.end = 0;
            self.readers.iter_mut().for_each(|r| *r = 0);
        }
    }

    fn clear(&mut self) {
        self.end = 0;
        self.readers.iter_mut().for_each(|r| *r = 0);
    }
}

/// The queue behind a stream output.
pub(crate) struct Queue {
    pub(crate) buf: Buffer,
    pub(crate) start: usize,
    pub(crate) end: usize,
    /// Bytes and samples of one packing group.
    pub(crate) elems: (u64, u64),
    /// Elements appended since the build.
    pub(crate) total_in: u64,
    /// Elements removed from the front (read or dropped) since the build.
    pub(crate) total_out: u64,
    /// `(element counter, sample index)` where the index sequence is not contiguous with the
    /// previous element: the start, every gap and every drop.
    pub(crate) marks: Vec<(u64, u64)>,
}

/// Room for index marks of one queue.
pub(crate) const QUEUE_MARKS: usize = 64;

impl Queue {
    pub(crate) fn cap(&self) -> usize {
        self.buf.len()
    }

    /// The sample index of the element with counter value `at`.
    pub(crate) fn idx_at(&self, at: u64) -> u64 {
        self.marks
            .iter()
            .rev()
            .find(|(pos, _)| *pos <= at)
            .or_else(|| self.marks.first())
            .map_or(0, |(pos, idx)| {
                idx.saturating_add(samples_of(at.saturating_sub(*pos), self.elems))
            })
    }

    /// Sample index of the first queued element.
    pub(crate) fn first_idx(&self) -> u64 {
        self.idx_at(self.total_out)
    }

    /// Records that the next appended element has the sample index `idx`.
    fn note_next(&mut self, idx: u64) {
        let contiguous = self.marks.last().is_some_and(|(pos, i)| {
            i.saturating_add(samples_of(self.total_in - pos, self.elems)) == idx
        });
        if contiguous {
            return;
        }
        if self.marks.len() >= QUEUE_MARKS {
            self.marks.remove(0);
        }
        self.marks.push((self.total_in, idx));
    }

    /// Removes elements from the front for the bookkeeping of indices.
    pub(crate) fn advance(&mut self, n: usize) {
        self.start += n;
        self.total_out += n as u64;
        while self.marks.len() > 1
            && self
                .marks
                .get(1)
                .is_some_and(|(pos, _)| *pos <= self.total_out)
        {
            self.marks.remove(0);
        }
        if self.start == self.end {
            self.start = 0;
            self.end = 0;
        }
    }

    /// Drops all queued elements; the index sequence continues.
    pub(crate) fn clear(&mut self) {
        self.start = 0;
        self.end = 0;
        self.total_out = self.total_in;
    }

    /// Drops all queued elements and restarts the index sequence at `idx`.
    pub(crate) fn reset_to(&mut self, idx: u64) {
        self.clear();
        self.marks.clear();
        self.marks.push((self.total_in, idx));
    }

    pub(crate) fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub(crate) fn free(&self) -> usize {
        self.cap().saturating_sub(self.len())
    }

    fn compact(&mut self) {
        if self.start > 0 {
            self.buf.move_within(self.start..self.end, 0);
            self.end -= self.start;
            self.start = 0;
        }
    }

    /// Appends `n` elements of `src` (the caller has checked that they fit).
    fn append(&mut self, src: &Buffer, from: usize, n: usize) -> bool {
        if self.cap().saturating_sub(self.end) < n {
            self.compact();
        }
        let ok = self.buf.copy_from(self.end, src, from, n);
        if ok {
            self.end += n;
            self.total_in += n as u64;
        }
        ok
    }
}

/// The queue behind a frame output.
pub(crate) struct FrameQueue {
    pub(crate) len: usize,
    pub(crate) cap: usize,
    pub(crate) data: Vec<f32>,
    pub(crate) metas: Vec<FrameMeta>,
    pub(crate) scratch: Vec<f32>,
    pub(crate) head: usize,
    pub(crate) count: usize,
}

impl FrameQueue {
    /// Stores the frame in `scratch` according to the policy; returns the number of frames lost.
    fn push(&mut self, meta: FrameMeta, policy: DropPolicy) -> u64 {
        let mut lost = 0;
        if self.count == self.cap {
            match policy {
                DropPolicy::DropNewest | DropPolicy::Error => return 1,
                DropPolicy::DropOldest => {
                    self.head = (self.head + 1) % self.cap.max(1);
                    self.count -= 1;
                    lost = 1;
                }
            }
        }
        let slot = (self.head + self.count) % self.cap.max(1);
        let at = slot * self.len;
        if let Some(dst) = self.data.get_mut(at..at + self.len) {
            dst.copy_from_slice(&self.scratch);
        }
        if let Some(m) = self.metas.get_mut(slot) {
            *m = meta;
        }
        self.count += 1;
        lost
    }
}

pub(crate) enum OutKind {
    Stream {
        src: usize,
        slot: usize,
        ty: PortType,
        queue: Queue,
    },
    Frames(FrameQueue),
}

/// The state of the burst machine of an output (pull mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Burst {
    /// `begin_burst` was never called: samples are produced all the time.
    Continuous,
    /// Between bursts: nothing is produced.
    Idle,
    /// `begin_burst` was called; the sample with index `at` starts the burst.
    Armed { at: u64, tx_time_ns: Option<i64> },
    /// Inside a burst.
    Active,
    /// `end_burst` was called; the burst ends before the sample with index `end`.
    Ending { end: u64 },
}

pub(crate) struct OutRt {
    pub(crate) name: String,
    pub(crate) policy: DropPolicy,
    pub(crate) dropped: u64,
    pub(crate) kind: OutKind,
    /// The pipeline input this output depends on.
    pub(crate) input: usize,
    /// The nodes from the input to the output in processing order (a frame node comes last),
    /// following port `a` of every two-input node.
    pub(crate) path: Vec<usize>,
    /// The branches that join the path at port `b` of a two-input node.
    pub(crate) sides: Vec<Side>,
    /// Every node that the output depends on (the path and the branches), in processing order.
    pub(crate) tree: Vec<usize>,
    /// Rate of the samples of this output (frames: rate at the input of the frame node).
    pub(crate) rate: Rate,
    /// The latest anchor of this output (delay-compensated).
    pub(crate) anchor: Option<TimeAnchor>,
    /// Group delay in samples of the output.
    pub(crate) lead: Delay,
    /// Group delay of the path in samples of the pipeline input.
    pub(crate) lead_in: Delay,
    /// Worst-case buffered samples, in samples of the output.
    pub(crate) max_buffered: u64,
    pub(crate) underrun: UnderrunPolicy,
    pub(crate) burst: Burst,
    /// One packing group of silence for a raw output (empty otherwise).
    pub(crate) silence: Vec<u8>,
}

/// A branch that joins the path of an output at port `b` of the two-input node `join`.
pub(crate) struct Side {
    /// The two-input node on the path of the output.
    pub(crate) join: usize,
    /// The nodes from the input to port `b` of `join` in processing order, following port `a`
    /// of every node on the way.
    pub(crate) chain: Vec<usize>,
}

impl OutRt {
    pub(crate) fn free(&self) -> usize {
        match &self.kind {
            OutKind::Stream { queue, .. } => queue.free(),
            OutKind::Frames(f) => f.cap - f.count,
        }
    }

    pub(crate) fn capacity(&self) -> usize {
        match &self.kind {
            OutKind::Stream { queue, .. } => queue.cap(),
            OutKind::Frames(f) => f.cap,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Dst {
    Stream(usize),
    Frames(usize),
}

pub(crate) struct NodeRt {
    pub(crate) name: String,
    pub(crate) node: Box<dyn Node>,
    /// The stream of port `a`, and the reader slot of the node in it.
    pub(crate) src: usize,
    pub(crate) slot: usize,
    /// The stream of port `b` of a two-input node, and the reader slot of the node in it.
    pub(crate) src_b: Option<usize>,
    pub(crate) slot_b: usize,
    /// True for a clocked node: port `b` carries packets and lockstep applies to port `a`
    /// only.
    pub(crate) clocked: bool,
    /// True for a node that reads a packet stream (`sensor_decode`): the packets it writes
    /// carry the index of the packet it read, which is already in the index domain of the
    /// upstream node, so `n0_in` is not added to it and the index maps to itself (M46).
    pub(crate) packets_in: bool,
    pub(crate) dst: Dst,
    /// Output elements per input element (`num`, `den`).
    pub(crate) ratio: (u64, u64),
    /// Bit `i` is set when the run-time parameter `i` of the description was changed.
    pub(crate) changed: u64,
    /// Elements consumed since the origin of the node.
    pub(crate) consumed: u64,
    /// The origin pair of the index mapping: the input index and the output index at which the
    /// node started (build, reset or gap).
    pub(crate) n0_in: u64,
    pub(crate) n0_out: u64,
    /// Sample rate at the input of the node.
    pub(crate) rate_in: Rate,
    /// Sample rate at the output of the node.
    pub(crate) rate_out: Rate,
    /// Group delay in input samples.
    pub(crate) delay: Delay,
    /// The pipeline input the node depends on.
    pub(crate) input: usize,
    /// The nodes between the input and this node, in processing order.
    pub(crate) upstream: Vec<usize>,
    /// Timed parameter changes, ordered by target index.
    pub(crate) pending: Vec<Pending>,
}

/// A timed parameter change that has not taken effect yet.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Pending {
    /// Sample index in the input stream of the node.
    pub(crate) at: u64,
    /// Index of the parameter in the description.
    pub(crate) param: u16,
    pub(crate) value: ParamValue,
}

/// How an output depends on an input: output elements (or frames) per input element.
#[derive(Clone, Copy)]
pub(crate) struct Reach {
    pub(crate) output: usize,
    pub(crate) num: u128,
    pub(crate) den: u128,
}

pub(crate) struct InRt {
    pub(crate) name: String,
    pub(crate) stream: usize,
    pub(crate) ty: PortType,
    pub(crate) rate: Rate,
    /// False for a packet input without `rate_hz`.
    pub(crate) has_rate: bool,
    pub(crate) max_len: usize,
    /// Zero samples for the flush of `end_burst`, allocated once in `build`.
    pub(crate) zeros: crate::pull::ZeroInput,
    /// Smallest chunk (in elements) that lets the readers produce output.
    pub(crate) min_input: usize,
    /// Nodes that read the input stream, in processing order.
    pub(crate) readers: Vec<usize>,
    pub(crate) reach: Vec<Reach>,
    /// Every node that depends on the input, in processing order.
    pub(crate) tree_nodes: Vec<usize>,
    /// Every stream that depends on the input, the input stream first.
    pub(crate) tree_streams: Vec<usize>,
    /// The outputs that depend on the input.
    pub(crate) outputs: Vec<usize>,
    /// The latest anchor pushed to the input.
    pub(crate) anchor: Option<TimeAnchor>,
    /// True until the first data or metadata arrives: the first block fixes the origin.
    pub(crate) fresh: bool,
}

/// A built pipeline: owns every buffer and block; processing does not allocate.
///
/// ```
/// use sdrk::C32;
/// use sdrk_pipeline::{BuildOptions, InputData, PipelineSpec, Registry};
///
/// let spec = PipelineSpec::from_json(r#"{
///   "sdrk": 1, "name": "gain",
///   "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
///   "nodes": { "g": { "kind": "scale", "params": { "gain_db": -6.0206 } } },
///   "edges": [ { "from": "iq", "to": "g" }, { "from": "g", "to": "out" } ],
///   "outputs": { "out": { "type": "stream" } }
/// }"#)?;
/// let mut pipeline = Registry::standard().build(&spec, &BuildOptions::default()).map_err(|r| r.to_json())?;
/// pipeline.push("iq", InputData::C32(&[C32::new(2.0, 0.0); 8]))?;
/// let mut out = [C32::default(); 8];
/// assert_eq!(pipeline.read_stream("out", &mut out)?, 8);
/// assert!((out[0].re - 1.0).abs() < 1e-3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Pipeline {
    pub(crate) spec: PipelineSpec,
    pub(crate) report: Report,
    pub(crate) inputs: Vec<InRt>,
    pub(crate) input_index: BTreeMap<String, usize>,
    pub(crate) nodes: Vec<NodeRt>,
    pub(crate) node_index: BTreeMap<String, usize>,
    pub(crate) streams: Vec<Stream>,
    pub(crate) outputs: Vec<OutRt>,
    pub(crate) output_index: BTreeMap<String, usize>,
    pub(crate) stats: Vec<NodeStats>,
    pub(crate) events: EventQueue,
    pub(crate) pending_cap: usize,
    /// Scratch space of the gap handling: missing samples per stream.
    pub(crate) gap_scratch: Vec<u64>,
}

impl core::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Pipeline")
            .field("name", &self.spec.name)
            .field("nodes", &self.nodes.len())
            .finish_non_exhaustive()
    }
}

fn block_error(name: &str, source: sdrk_blocks::Error) -> Error {
    Error::Block {
        node: name.to_string(),
        source,
    }
}

/// Runs a stream node once. `None` if its output buffer has no room.
#[allow(
    clippy::large_stack_frames,
    reason = "clippy counts the temporary of every `?` separately; the real frame is small"
)]
fn step_stream(
    n: &mut NodeRt,
    streams: &mut [Stream],
    d: usize,
    avail: usize,
    direct: Option<(&InputData<'_>, usize)>,
) -> Result<Option<(Progress, usize)>> {
    let needed = (avail as u128)
        .saturating_mul(u128::from(n.ratio.0))
        .div_ceil(u128::from(n.ratio.1.max(1)))
        .saturating_add(4);
    let needed = usize::try_from(needed).unwrap_or(usize::MAX);
    {
        let ds = at_mut(streams, d)?;
        if ds.cap().saturating_sub(ds.end) < needed {
            ds.compact();
        }
        if ds.end >= ds.cap() {
            return Ok(None);
        }
    }
    let mut meta: Option<FrameMeta> = None;
    let shift = if n.packets_in { 0 } else { n.n0_in };
    let p = if let Some((data, off)) = direct {
        let ds = at_mut(streams, d)?;
        let input = data
            .view(off, avail)
            .ok_or_else(|| internal("input view"))?;
        let out = ds
            .buf
            .view_mut(ds.end..ds.cap())
            .ok_or_else(|| internal("output view"))?;
        let p = n
            .node
            .process(input, out, &mut meta)
            .map_err(|e| block_error(&n.name, e))?;
        if p.consumed > avail || ds.end + p.produced > ds.cap() {
            return Err(internal("block reported impossible progress"));
        }
        ds.buf.shift_packets(ds.end..ds.end + p.produced, shift);
        ds.end += p.produced;
        ds.total_el += p.produced as u64;
        n.consumed += p.consumed as u64;
        p
    } else {
        let (ss, ds) = pair_mut(streams, n.src, d)?;
        let pos = *at(&ss.readers, n.slot)?;
        let input = ss
            .buf
            .view(pos..pos + avail)
            .ok_or_else(|| internal("input view"))?;
        let out = ds
            .buf
            .view_mut(ds.end..ds.cap())
            .ok_or_else(|| internal("output view"))?;
        let p = n
            .node
            .process(input, out, &mut meta)
            .map_err(|e| block_error(&n.name, e))?;
        if p.consumed > avail || ds.end + p.produced > ds.cap() {
            return Err(internal("block reported impossible progress"));
        }
        ds.buf.shift_packets(ds.end..ds.end + p.produced, shift);
        ds.end += p.produced;
        ds.total_el += p.produced as u64;
        n.consumed += p.consumed as u64;
        *at_mut(&mut ss.readers, n.slot)? += p.consumed;
        p
    };
    Ok(Some((p, 0)))
}

/// Runs a two-input node once on `avail` samples of each port. `None` if its output buffer has
/// no room.
#[allow(
    clippy::large_stack_frames,
    reason = "clippy counts the temporary of every `?` separately; the real frame is small"
)]
fn step_pair(
    n: &mut NodeRt,
    streams: &mut [Stream],
    d: usize,
    avail: usize,
) -> Result<Option<(Progress, usize)>> {
    let sb = n
        .src_b
        .ok_or_else(|| internal("pair node without a second source"))?;
    let needed = (avail as u128)
        .saturating_mul(u128::from(n.ratio.0))
        .div_ceil(u128::from(n.ratio.1.max(1)))
        .saturating_add(4);
    let needed = usize::try_from(needed).unwrap_or(usize::MAX);
    {
        let ds = at_mut(streams, d)?;
        if ds.cap().saturating_sub(ds.end) < needed {
            ds.compact();
        }
        if ds.end >= ds.cap() {
            return Ok(None);
        }
    }
    // The sources are written before the destination, so they lie in front of it.
    let (head, tail) = streams
        .split_at_mut_checked(d)
        .ok_or_else(|| internal("stream split"))?;
    let ds = tail
        .first_mut()
        .ok_or_else(|| internal("destination stream"))?;
    let mut meta: Option<FrameMeta> = None;
    let p = {
        let (sa, sbs) = (at(head, n.src)?, at(head, sb)?);
        let pos_a = *at(&sa.readers, n.slot)?;
        let pos_b = *at(&sbs.readers, n.slot_b)?;
        let a = sa
            .buf
            .elem_view(pos_a..pos_a + avail)
            .ok_or_else(|| internal("input view"))?;
        let b = sbs
            .buf
            .elem_view(pos_b..pos_b + avail)
            .ok_or_else(|| internal("input view"))?;
        let out = ds
            .buf
            .view_mut(ds.end..ds.cap())
            .ok_or_else(|| internal("output view"))?;
        n.node
            .process(NodeInput::Two { a, b }, out, &mut meta)
            .map_err(|e| block_error(&n.name, e))?
    };
    if p.consumed > avail || ds.end + p.produced > ds.cap() {
        return Err(internal("block reported impossible progress"));
    }
    ds.end += p.produced;
    ds.total_el += p.produced as u64;
    n.consumed += p.consumed as u64;
    *at_mut(&mut at_mut(head, n.src)?.readers, n.slot)? += p.consumed;
    *at_mut(&mut at_mut(head, sb)?.readers, n.slot_b)? += p.consumed;
    Ok(Some((p, 0)))
}

/// Runs a clocked node once on `avail` clock samples of port `a` and every packet queued at
/// port `b`. `None` if its output buffer has no room.
#[allow(
    clippy::large_stack_frames,
    reason = "clippy counts the temporary of every `?` separately; the real frame is small"
)]
fn step_clocked(
    n: &mut NodeRt,
    streams: &mut [Stream],
    d: usize,
    avail: usize,
) -> Result<Option<(Progress, usize)>> {
    let sb = n
        .src_b
        .ok_or_else(|| internal("clocked node without a packet source"))?;
    let needed = (avail as u128)
        .saturating_mul(u128::from(n.ratio.0))
        .div_ceil(u128::from(n.ratio.1.max(1)))
        .saturating_add(4);
    let needed = usize::try_from(needed).unwrap_or(usize::MAX);
    {
        let ds = at_mut(streams, d)?;
        if ds.cap().saturating_sub(ds.end) < needed {
            ds.compact();
        }
        if ds.end >= ds.cap() {
            return Ok(None);
        }
    }
    let (head, tail) = streams
        .split_at_mut_checked(d)
        .ok_or_else(|| internal("stream split"))?;
    let ds = tail
        .first_mut()
        .ok_or_else(|| internal("destination stream"))?;
    let mut meta: Option<FrameMeta> = None;
    let (p, queued) = {
        let (sa, sbs) = (at(head, n.src)?, at(head, sb)?);
        let pos_a = *at(&sa.readers, n.slot)?;
        let pos_b = *at(&sbs.readers, n.slot_b)?;
        let a = sa
            .buf
            .elem_view(pos_a..pos_a + avail)
            .ok_or_else(|| internal("input view"))?;
        let packets = sbs
            .buf
            .packets(pos_b..sbs.end.max(pos_b))
            .ok_or_else(|| internal("packet view"))?;
        let out = ds
            .buf
            .view_mut(ds.end..ds.cap())
            .ok_or_else(|| internal("output view"))?;
        let p = n
            .node
            .process(NodeInput::Clocked { a, packets }, out, &mut meta)
            .map_err(|e| block_error(&n.name, e))?;
        (p, packets.len())
    };
    let taken = n.node.consumed_packets();
    if p.consumed > avail || taken > queued || ds.end + p.produced > ds.cap() {
        return Err(internal("block reported impossible progress"));
    }
    ds.end += p.produced;
    ds.total_el += p.produced as u64;
    n.consumed += p.consumed as u64;
    *at_mut(&mut at_mut(head, n.src)?.readers, n.slot)? += p.consumed;
    *at_mut(&mut at_mut(head, sb)?.readers, n.slot_b)? += taken;
    Ok(Some((p, 0)))
}

/// Runs a frame node once. `None` if the frame queue is full and the policy is `error`.
#[allow(
    clippy::large_stack_frames,
    reason = "clippy counts the temporary of every `?` separately; the real frame is small"
)]
fn step_frames(
    n: &mut NodeRt,
    streams: &mut [Stream],
    out: &mut OutRt,
    avail: usize,
    direct: Option<(&InputData<'_>, usize)>,
) -> Result<Option<(Progress, usize)>> {
    let policy = out.policy;
    let OutKind::Frames(fq) = &mut out.kind else {
        return Err(internal("frame output"));
    };
    if policy == DropPolicy::Error && fq.count == fq.cap {
        return Ok(None);
    }
    let mut meta: Option<FrameMeta> = None;
    let origin = at(streams, n.src)?.origin_idx;
    let p = if let Some((data, off)) = direct {
        let input = data
            .view(off, avail)
            .ok_or_else(|| internal("input view"))?;
        n.node
            .process(input, NodeOutput::Frame(&mut fq.scratch), &mut meta)
            .map_err(|e| block_error(&n.name, e))?
    } else {
        let s = at_mut(streams, n.src)?;
        let pos = *at(&s.readers, n.slot)?;
        let input = s
            .buf
            .view(pos..pos + avail)
            .ok_or_else(|| internal("input view"))?;
        let p = n
            .node
            .process(input, NodeOutput::Frame(&mut fq.scratch), &mut meta)
            .map_err(|e| block_error(&n.name, e))?;
        if p.consumed > avail {
            return Err(internal("block reported impossible progress"));
        }
        *at_mut(&mut s.readers, n.slot)? += p.consumed;
        p
    };
    n.consumed += p.consumed as u64;
    let frames = usize::from(meta.is_some());
    if let Some(m) = meta {
        let m = FrameMeta {
            first_sample: m.first_sample.saturating_add(origin),
            ..m
        };
        out.dropped += fq.push(m, policy);
    }
    Ok(Some((
        Progress {
            consumed: p.consumed,
            produced: frames,
        },
        frames,
    )))
}

impl Pipeline {
    /// Feeds up to `max_chunk` samples to one input and runs the graph until no node can make
    /// progress. All of `data` is consumed, or an error is returned before anything is
    /// consumed.
    ///
    /// # Errors
    /// `UnknownInput`, `WrongType` (the variant does not match the input), `ChunkTooLarge`,
    /// `OutputFull` (an output with the drop policy `error` could not take the worst-case
    /// result; nothing was consumed) and `Block` (a block failed).
    pub fn push(&mut self, input: &str, data: InputData<'_>) -> Result<()> {
        let ii = self.input_pos(input)?;
        self.check_push(ii, &data)?;
        self.ingest(ii, &data)?;
        self.run()
    }

    pub(crate) fn input_pos(&self, input: &str) -> Result<usize> {
        self.input_index
            .get(input)
            .copied()
            .ok_or_else(|| Error::UnknownInput(input.to_string()))
    }

    /// The checks of `push` that must pass before anything is consumed.
    pub(crate) fn check_push(&self, ii: usize, data: &InputData<'_>) -> Result<()> {
        let inp = at(&self.inputs, ii)?;
        if data.port_type() != inp.ty {
            return Err(Error::WrongType {
                name: inp.name.clone(),
                expected: name_of(inp.ty),
                got: name_of(data.port_type()),
            });
        }
        if data.len() > inp.max_len {
            return Err(Error::ChunkTooLarge {
                len: data.len(),
                max: inp.max_len,
            });
        }
        // Worst-case production must fit into every `error` output.
        for reach in &inp.reach {
            let out = at(&self.outputs, reach.output)?;
            if out.policy != DropPolicy::Error {
                continue;
            }
            let bound = (data.len() as u128)
                .saturating_mul(reach.num)
                .div_ceil(reach.den.max(1))
                .saturating_add(1);
            let bound = usize::try_from(bound)
                .unwrap_or(usize::MAX)
                .min(out.capacity());
            if bound > out.free() {
                return Err(Error::OutputFull(out.name.clone()));
            }
        }
        Ok(())
    }

    /// Hands the data to the first readers directly and keeps what they did not take.
    pub(crate) fn ingest(&mut self, ii: usize, data: &InputData<'_>) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        at_mut(&mut self.inputs, ii)?.fresh = false;
        let sid = at(&self.inputs, ii)?.stream;
        if at(&self.streams, sid)?.end == 0 {
            self.ingest_direct(ii, sid, data)
        } else {
            self.ingest_append(ii, sid, data)
        }
    }

    /// The input buffer is empty: the readers process the caller's slice in place and only the
    /// part that all of them left is copied into the buffer.
    #[allow(
        clippy::large_stack_frames,
        reason = "clippy counts the temporary of every `?` separately; the real frame is small"
    )]
    fn ingest_direct(&mut self, ii: usize, sid: usize, data: &InputData<'_>) -> Result<()> {
        let len = data.len();
        at_mut(&mut self.streams, sid)?.total_el += len as u64;
        let reader_count = at(&self.inputs, ii)?.readers.len();
        for k in 0..reader_count {
            let ni = *at(&at(&self.inputs, ii)?.readers, k)?;
            let consumed = self.run_direct(ni, data)?;
            let slot = at(&self.nodes, ni)?.slot;
            *at_mut(&mut at_mut(&mut self.streams, sid)?.readers, slot)? = consumed;
        }
        let s = at_mut(&mut self.streams, sid)?;
        let keep_from = s.readers.iter().copied().min().unwrap_or(len).min(len);
        let rest = len - keep_from;
        if rest > 0 && !s.buf.copy_from_input(0, data, keep_from, rest) {
            return Err(internal("input buffer too small"));
        }
        s.end = rest;
        for r in &mut s.readers {
            *r = r.saturating_sub(keep_from);
        }
        s.normalize();
        Ok(())
    }

    /// Data is pending in the input buffer: the new data is appended after it.
    fn ingest_append(&mut self, ii: usize, sid: usize, data: &InputData<'_>) -> Result<()> {
        let len = data.len();
        let s = at_mut(&mut self.streams, sid)?;
        if s.cap().saturating_sub(s.end) < len {
            s.compact();
        }
        let at_end = s.end;
        if s.cap().saturating_sub(at_end) < len || !s.buf.copy_from_input(at_end, data, 0, len) {
            return Err(Error::OutputFull(at(&self.inputs, ii)?.name.clone()));
        }
        s.end += len;
        s.total_el += len as u64;
        Ok(())
    }

    /// Runs node `ni` on the caller's data until it can do no more; returns the number of
    /// elements consumed.
    fn run_direct(&mut self, ni: usize, data: &InputData<'_>) -> Result<usize> {
        let mut offset = 0;
        while offset < data.len() {
            match self.step(ni, Some((data, offset)))? {
                Some(p) if p.consumed > 0 || p.produced > 0 => offset += p.consumed,
                _ => break,
            }
        }
        Ok(offset)
    }

    /// One `process` call of node `ni`, reading from its stream or from `direct` (data and
    /// offset). `None` if the node cannot run (no input or no room).
    #[allow(
        clippy::large_stack_frames,
        reason = "clippy counts the temporary of every `?` separately; the real frame is small"
    )]
    fn step(
        &mut self,
        ni: usize,
        direct: Option<(&InputData<'_>, usize)>,
    ) -> Result<Option<Progress>> {
        let avail = {
            let n = at(&self.nodes, ni)?;
            if let Some((data, off)) = direct {
                data.len().saturating_sub(off)
            } else {
                let s = at(&self.streams, n.src)?;
                let a = s.end.saturating_sub(*at(&s.readers, n.slot)?);
                match n.src_b {
                    // A clocked node advances with its clock; the packets are optional.
                    Some(_) if n.clocked => a,
                    // The two ports advance together: as much as both hold.
                    Some(sb) => {
                        let sbs = at(&self.streams, sb)?;
                        let b = sbs.end.saturating_sub(*at(&sbs.readers, n.slot_b)?);
                        debug_assert!(
                            a.abs_diff(b) <= s.cap().max(sbs.cap()),
                            "the buffers of the two ports of `{}` drifted apart",
                            n.name
                        );
                        a.min(b)
                    }
                    None => a,
                }
            }
        };
        if avail == 0 {
            return Ok(None);
        }
        // Timed parameter changes split the call at the exact sample.
        let avail = self.apply_pending(ni, avail)?;
        let Self {
            nodes,
            streams,
            outputs,
            stats,
            ..
        } = self;
        let n = at_mut(nodes, ni)?;
        let st = at_mut(stats, ni)?;
        let progress = match n.dst {
            Dst::Stream(d) if n.clocked => step_clocked(n, streams, d, avail)?,
            Dst::Stream(d) if n.src_b.is_some() => step_pair(n, streams, d, avail)?,
            Dst::Stream(d) => step_stream(n, streams, d, avail, direct)?,
            Dst::Frames(oi) => step_frames(n, streams, at_mut(outputs, oi)?, avail, direct)?,
        };
        if let Some((p, frames)) = progress {
            update_readings(&*n.node, st);
            st.calls += 1;
            st.samples_in += p.consumed as u64;
            st.samples_out += (p.produced - frames) as u64;
            st.frames += frames as u64;
            return Ok(Some(p));
        }
        Ok(None)
    }

    /// One idle call of the packet encoder `ni` (pull mode): an empty packet slice and room for
    /// at most `samples` output samples, which the encoder fills with its idle signal. Returns
    /// the number of samples produced (0 when the output buffer is full).
    pub(crate) fn step_idle(&mut self, ni: usize, samples: usize) -> Result<usize> {
        let Self {
            nodes,
            streams,
            stats,
            ..
        } = self;
        let n = at_mut(nodes, ni)?;
        let Dst::Stream(d) = n.dst else {
            return Err(internal("idle call of a frame node"));
        };
        let ds = at_mut(streams, d)?;
        if ds.cap().saturating_sub(ds.end) < samples {
            ds.compact();
        }
        let end = ds.end.saturating_add(samples).min(ds.cap());
        if end <= ds.end {
            return Ok(0);
        }
        let out = ds
            .buf
            .view_mut(ds.end..end)
            .ok_or_else(|| internal("output view"))?;
        let mut meta: Option<FrameMeta> = None;
        let p = n
            .node
            .process(NodeInput::Packets(&[]), out, &mut meta)
            .map_err(|e| block_error(&n.name, e))?;
        if p.consumed > 0 || ds.end + p.produced > end {
            return Err(internal("block reported impossible progress"));
        }
        ds.end += p.produced;
        ds.total_el += p.produced as u64;
        let st = at_mut(stats, ni)?;
        st.calls += 1;
        st.samples_out += p.produced as u64;
        Ok(p.produced)
    }

    /// Moves data from the source stream of output `oi` into its queue, applying the drop
    /// policy. Returns whether anything moved.
    fn drain(&mut self, oi: usize) -> Result<bool> {
        let Self {
            streams, outputs, ..
        } = self;
        let out = at_mut(outputs, oi)?;
        let policy = out.policy;
        let OutKind::Stream {
            src, slot, queue, ..
        } = &mut out.kind
        else {
            return Ok(false);
        };
        let s = at_mut(streams, *src)?;
        let pos = *at(&s.readers, *slot)?;
        let avail = s.end.saturating_sub(pos);
        if avail == 0 {
            return Ok(false);
        }
        let cap = queue.cap();
        let free = queue.free();
        let (skip, take) = match policy {
            DropPolicy::Error => (0, avail.min(free)),
            DropPolicy::DropNewest => {
                let n = avail.min(free);
                out.dropped += (avail - n) as u64;
                (0, n)
            }
            DropPolicy::DropOldest => {
                if avail >= cap {
                    out.dropped += (avail - cap + queue.len()) as u64;
                    queue.advance(queue.len());
                    (avail - cap, cap)
                } else {
                    let lost = avail.saturating_sub(free);
                    queue.advance(lost);
                    out.dropped += lost as u64;
                    (0, avail)
                }
            }
        };
        if take > 0 {
            queue.note_next(s.idx_of_pos(pos + skip));
            if !queue.append(&s.buf, pos + skip, take) {
                return Err(internal("output queue copy"));
            }
        }
        let consumed = match policy {
            DropPolicy::Error => take,
            _ => avail,
        };
        *at_mut(&mut s.readers, *slot)? += consumed;
        Ok(consumed > 0)
    }

    /// Repeats passes over the nodes in topological order until nothing moves.
    pub(crate) fn run(&mut self) -> Result<()> {
        loop {
            let mut progress = false;
            for ni in 0..self.nodes.len() {
                while let Some(p) = self.step(ni, None)? {
                    if p.consumed == 0 && p.produced == 0 {
                        break;
                    }
                    progress = true;
                }
            }
            for oi in 0..self.outputs.len() {
                progress |= self.drain(oi)?;
            }
            for s in &mut self.streams {
                s.normalize();
            }
            if !progress {
                return Ok(());
            }
        }
    }

    pub(crate) fn output_pos(&self, name: &str) -> Result<usize> {
        self.output_index
            .get(name)
            .copied()
            .ok_or_else(|| Error::UnknownOutput(name.to_string()))
    }

    /// Reads up to `out.len()` samples of a stream output; returns the number read.
    ///
    /// # Errors
    /// `UnknownOutput`, `WrongType` (`T` differs from the type of the output, or the output
    /// is a frame output).
    pub fn read_stream<T: Sample>(&mut self, output: &str, out: &mut [T]) -> Result<usize> {
        let oi = self.output_pos(output)?;
        let o = at_mut(&mut self.outputs, oi)?;
        let policy = o.policy;
        let OutKind::Stream { ty, queue, .. } = &mut o.kind else {
            return Err(Error::WrongType {
                name: output.to_string(),
                expected: "frames",
                got: name_of(sample_port(T::TYPE)),
            });
        };
        if *ty != sample_port(T::TYPE) {
            return Err(Error::WrongType {
                name: output.to_string(),
                expected: name_of(*ty),
                got: name_of(sample_port(T::TYPE)),
            });
        }
        let n = out.len().min(queue.len());
        if !queue.buf.copy_to(queue.start, n, out) {
            return Err(internal("output queue read"));
        }
        queue.advance(n);
        if n > 0 && policy == DropPolicy::Error {
            self.run()?;
        }
        Ok(n)
    }

    /// Reads up to `out.len()` bytes of a raw stream output (one with a `format`, or fed by a
    /// `convert_out` node); returns the number of bytes read.
    ///
    /// # Errors
    /// `UnknownOutput`, `WrongType` (the output does not carry raw bytes).
    pub fn read_raw(&mut self, output: &str, out: &mut [u8]) -> Result<usize> {
        let oi = self.output_pos(output)?;
        let o = at_mut(&mut self.outputs, oi)?;
        let policy = o.policy;
        let OutKind::Stream { ty, queue, .. } = &mut o.kind else {
            return Err(Error::WrongType {
                name: output.to_string(),
                expected: "frames",
                got: "raw",
            });
        };
        if *ty != PortType::Raw {
            return Err(Error::WrongType {
                name: output.to_string(),
                expected: name_of(*ty),
                got: "raw",
            });
        }
        let n = out.len().min(queue.len());
        if !queue.buf.copy_to(queue.start, n, out) {
            return Err(internal("output queue read"));
        }
        queue.advance(n);
        if n > 0 && policy == DropPolicy::Error {
            self.run()?;
        }
        Ok(n)
    }

    /// Reads the oldest frame of a frame output into `frame`; `None` if there is none.
    ///
    /// # Errors
    /// `UnknownOutput`, `WrongType` (a stream output), `LengthMismatch` (`frame` has not the
    /// frame length of the output).
    pub fn read_frame(&mut self, output: &str, frame: &mut [f32]) -> Result<Option<FrameMeta>> {
        let oi = self.output_pos(output)?;
        let o = at_mut(&mut self.outputs, oi)?;
        let policy = o.policy;
        let OutKind::Frames(fq) = &mut o.kind else {
            return Err(Error::WrongType {
                name: output.to_string(),
                expected: "a stream",
                got: "frames",
            });
        };
        if frame.len() != fq.len {
            return Err(Error::LengthMismatch {
                expected: fq.len,
                got: frame.len(),
            });
        }
        if fq.count == 0 {
            return Ok(None);
        }
        let at_frame = fq.head * fq.len;
        let src = fq
            .data
            .get(at_frame..at_frame + fq.len)
            .ok_or_else(|| internal("frame queue"))?;
        frame.copy_from_slice(src);
        let meta = *at(&fq.metas, fq.head)?;
        fq.head = (fq.head + 1) % fq.cap.max(1);
        fq.count -= 1;
        if policy == DropPolicy::Error {
            self.run()?;
        }
        Ok(Some(meta))
    }

    /// Pushes packets into a packet input: the same as `push(input, InputData::Packets(..))`.
    /// All packets are taken, or an error is returned before anything is taken. A pure
    /// encoder (`frame_encode`) produces the samples of the packets it has; with no packet it
    /// produces nothing (in push mode there is no idle signal).
    ///
    /// ```
    /// use sdrk_pipeline::{BuildOptions, Packet, PacketFlags, Registry};
    /// let mut p = Registry::standard().build_json(
    ///     include_str!("../../../../examples/flows/frame_loop.json"),
    ///     &BuildOptions::default(),
    /// )?;
    /// let msg = Packet::new(0, PacketFlags::NONE, b"hello").map_err(|e| e.to_string())?;
    /// p.push_packets("msg", &[msg])?;
    /// assert_eq!(p.available("out")?, 1);
    /// let mut got = [Packet::default(); 4];
    /// assert_eq!(p.read_packets("out", &mut got)?, 1);
    /// assert_eq!(got[0].data(), b"hello");
    /// assert!(got[0].flags.contains(PacketFlags::CRC_OK));
    /// assert_eq!(got[0].index, 32); // after the 4 preamble bytes
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    /// As `push`: `UnknownInput`, `WrongType` (not a packet input), `ChunkTooLarge` (more than
    /// `settings.max_chunk` packets), `OutputFull`, `Block` (e.g. a payload that the encoder
    /// cannot frame).
    pub fn push_packets(&mut self, input: &str, packets: &[Packet]) -> Result<()> {
        self.push(input, InputData::Packets(packets))
    }

    /// Reads up to `out.len()` packets of a packet output, oldest first; returns the number
    /// read.
    ///
    /// # Errors
    /// `UnknownOutput`, `WrongType` (the output does not carry packets).
    pub fn read_packets(&mut self, output: &str, out: &mut [Packet]) -> Result<usize> {
        let oi = self.output_pos(output)?;
        let o = at_mut(&mut self.outputs, oi)?;
        let policy = o.policy;
        let wrong = |expected| Error::WrongType {
            name: output.to_string(),
            expected,
            got: "packets",
        };
        let OutKind::Stream { ty, queue, .. } = &mut o.kind else {
            return Err(wrong("frames"));
        };
        if *ty != PortType::Packets {
            return Err(wrong(name_of(*ty)));
        }
        let n = out.len().min(queue.len());
        if !queue.buf.copy_packets_to(queue.start, n, out) {
            return Err(internal("output queue read"));
        }
        queue.advance(n);
        if n > 0 && policy == DropPolicy::Error {
            self.run()?;
        }
        Ok(n)
    }

    /// Elements waiting in an output: samples (bytes for a raw output) of a stream output,
    /// frames of a frame output or packets of a packet output.
    ///
    /// # Errors
    /// `UnknownOutput`.
    pub fn available(&self, output: &str) -> Result<usize> {
        let o = at(&self.outputs, self.output_pos(output)?)?;
        Ok(match &o.kind {
            OutKind::Stream { queue, .. } => queue.len(),
            OutKind::Frames(f) => f.count,
        })
    }

    /// Number of elements (frames, packets) that the drop policy discarded in an output since the
    /// build.
    ///
    /// # Errors
    /// `UnknownOutput`.
    pub fn dropped(&self, output: &str) -> Result<u64> {
        Ok(at(&self.outputs, self.output_pos(output)?)?.dropped)
    }

    /// Changes a run-time parameter of a node; it applies from the next sample the node
    /// processes. The call does not allocate when it succeeds.
    ///
    /// # Errors
    /// `UnknownNode`, `Block` (`UnknownParameter`; `ReadOnlyParameter` for a build parameter;
    /// `OutOfRange`), `InvalidParameter` (the JSON value does not fit the parameter).
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the signature takes the JSON value by value"
    )]
    pub fn set_param(&mut self, node: &str, name: &str, value: Value) -> Result<()> {
        let ni = *self
            .node_index
            .get(node)
            .ok_or_else(|| Error::UnknownNode(node.to_string()))?;
        let n = at_mut(&mut self.nodes, ni)?;
        let desc = n.node.description();
        let block_error = |source| Error::Block {
            node: node.to_string(),
            source,
        };
        let Some(index) = desc.params.iter().position(|p| p.name == name) else {
            return Err(block_error(sdrk_blocks::Error::UnknownParameter));
        };
        let p = at(desc.params, index)?;
        if p.mutability != sdrk_blocks::Mutability::Runtime {
            return Err(block_error(sdrk_blocks::Error::ReadOnlyParameter));
        }
        let pv = to_param(p, &value).map_err(|issue| match issue {
            crate::params::ParamIssue::Range(e) => block_error(e),
            other => Error::InvalidParameter {
                node: node.to_string(),
                name: name.to_string(),
                reason: other.message(&value),
            },
        })?;
        n.node.set_param(p.name, pv).map_err(block_error)?;
        if index < 64 {
            n.changed |= 1 << index;
        }
        Ok(())
    }

    /// The configuration: the user's description with the current values of the run-time
    /// parameters that were changed. For an unchanged pipeline `to_json(config)` equals
    /// `to_json(from_json(input))`; the automatic nodes do not appear.
    pub fn config(&self) -> PipelineSpec {
        let mut spec = self.spec.clone();
        for n in &self.nodes {
            if n.changed == 0 {
                continue;
            }
            let desc = n.node.description();
            let Some(node) = spec.nodes.get_mut(&n.name) else {
                continue;
            };
            for (i, p) in desc.params.iter().enumerate() {
                if i < 64
                    && n.changed & (1 << i) != 0
                    && let Ok(v) = n.node.param(p.name)
                {
                    node.params.insert(p.name.to_string(), to_json(p, v));
                }
            }
        }
        spec
    }

    /// Statistics of every node, automatic ones included, in processing order.
    pub fn stats(&self) -> &[NodeStats] {
        &self.stats
    }

    /// Resets all blocks, empties all buffers, queues and events and starts every index at 0
    /// again; the statistics are kept.
    pub fn reset(&mut self) {
        for (n, st) in self.nodes.iter_mut().zip(self.stats.iter_mut()) {
            n.node.reset();
            update_readings(&*n.node, st);
        }
        for n in &mut self.nodes {
            n.consumed = 0;
            n.n0_in = 0;
            n.n0_out = 0;
            n.pending.clear();
        }
        for s in &mut self.streams {
            s.rebase(0);
        }
        for i in &mut self.inputs {
            i.anchor = None;
            i.fresh = true;
        }
        for o in &mut self.outputs {
            o.anchor = None;
            o.burst = if o.burst == Burst::Continuous {
                Burst::Continuous
            } else {
                Burst::Idle
            };
            match &mut o.kind {
                OutKind::Stream { queue, .. } => queue.reset_to(0),
                OutKind::Frames(f) => {
                    f.head = 0;
                    f.count = 0;
                }
            }
        }
        self.events.clear();
    }

    /// The validation report, with the delays reported by the built blocks.
    pub fn report(&self) -> &Report {
        &self.report
    }
}
