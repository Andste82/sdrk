//! Time, events and latency: the public types of the sample-index time base.
//!
//! Every stream has its own sample index (`u64`) at its own rate. Events travel with the
//! indices of the output they are reported on; `docs/timing.md` explains the mapping rule.

use std::collections::VecDeque;

use sdrk::Delay;
use sdrk_blocks::ParamValue;
use sdrk_device::TimeAnchor;

/// Identifies a node of a built pipeline (an index; automatic nodes included).
///
/// ```
/// use sdrk_pipeline::{BuildOptions, Registry};
/// let mut p = Registry::standard()
///     .build_json(include_str!("../../../../examples/flows/fm_receiver.json"), &BuildOptions::default())?;
/// let id = p.node_id("tune")?;
/// assert_eq!(p.node_name(id), "tune");
/// # Ok::<(), sdrk_pipeline::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub(crate) u32);

impl NodeId {
    /// Position of the node in processing order.
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub(crate) fn new(index: usize) -> Self {
        Self(u32::try_from(index).unwrap_or(u32::MAX))
    }
}

/// Identifies an output of a built pipeline (an index into the outputs, sorted by name).
///
/// ```
/// use sdrk_pipeline::{BuildOptions, Registry};
/// let p = Registry::standard()
///     .build_json(include_str!("../../../../examples/flows/fm_receiver.json"), &BuildOptions::default())?;
/// let id = p.output_id("audio")?;
/// assert_eq!(p.output_name(id), "audio");
/// # Ok::<(), sdrk_pipeline::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OutputId(pub(crate) u32);

impl OutputId {
    /// Position of the output (outputs are sorted by name).
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub(crate) fn new(index: usize) -> Self {
        Self(u32::try_from(index).unwrap_or(u32::MAX))
    }
}

/// What happened at an output.
///
/// ```
/// use sdrk_pipeline::EventKind;
/// assert_eq!(EventKind::Gap { missing: 3 }, EventKind::Gap { missing: 3 });
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum EventKind {
    /// Samples are missing before this index (`missing` samples at the rate of the input).
    Gap {
        /// Number of missing samples at the input.
        missing: u64,
    },
    /// The receiver hardware dropped samples.
    Overflow,
    /// The transmit hardware ran out of samples, or `pull` filled in silence.
    Underrun,
    /// The sample `index` has an absolute time (delay-compensated for this output).
    Anchor(TimeAnchor),
    /// Frequency, rate or gain changed at this index.
    Retuned,
    /// First sample of a burst.
    BurstStart {
        /// Requested transmit time (unix nanoseconds), if any.
        tx_time_ns: Option<i64>,
    },
    /// Last sample of a burst.
    BurstEnd,
    /// A timed burst arrived too late.
    Late,
    /// A timed parameter change took effect.
    ParamApplied {
        /// The node whose parameter changed.
        node: NodeId,
        /// Index of the parameter in the description of the block.
        param: u16,
        /// The new value.
        value: ParamValue,
        /// True if the requested time had already passed when the change was requested.
        late: bool,
    },
    /// Events were dropped because the queue was full.
    EventsLost {
        /// Number of events lost since the last report.
        count: u64,
    },
}

/// An event at an index of an output.
///
/// ```
/// use sdrk_pipeline::{BuildOptions, Event, EventKind, InputData, Registry};
/// use sdrk_device::{BlockMeta, SampleIndex};
/// let mut p = Registry::standard().build_json(
///     r#"{ "sdrk": 1, "name": "t",
///          "inputs": { "iq": { "type": "c32", "rate_hz": 1000 } },
///          "nodes": { "g": { "kind": "scale" } },
///          "edges": [ { "from": "iq", "to": "g" }, { "from": "g", "to": "out" } ],
///          "outputs": { "out": { "type": "stream" } } }"#,
///     &BuildOptions::default(),
/// )?;
/// p.push("iq", InputData::C32(&[Default::default(); 2]))?;
/// let meta = BlockMeta { start: SampleIndex(10), ..BlockMeta::default() };
/// p.push_with_meta("iq", InputData::C32(&[]), &meta)?;
/// let Some(Event { index, kind: EventKind::Gap { missing }, .. }) = p.poll_event() else { panic!() };
/// assert_eq!((index, missing), (10, 8));
/// # Ok::<(), sdrk_pipeline::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Event {
    /// The output the event belongs to.
    pub output: OutputId,
    /// Sample index on that output (frame outputs: index in the input of the frame node).
    pub index: u64,
    /// What happened.
    pub kind: EventKind,
}

/// When a timed parameter change takes effect.
///
/// ```
/// use sdrk_pipeline::ParamTime;
/// let t = ParamTime::InputIndex { input: "rf".into(), index: 1000 };
/// assert_ne!(t, ParamTime::Immediate);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamTime {
    /// At the next sample the node processes.
    Immediate,
    /// At the sample `index` of the stream of the pipeline input `input`.
    InputIndex {
        /// Name of the pipeline input.
        input: String,
        /// Sample index at that input.
        index: u64,
    },
    /// At the sample of the input that was taken at this absolute time (needs an anchor).
    UnixNs(i64),
}

/// Latency of an output, see `docs/timing.md`.
///
/// ```
/// use sdrk_pipeline::{BuildOptions, Registry};
/// let p = Registry::standard()
///     .build_json(include_str!("../../../../examples/flows/fm_receiver.json"), &BuildOptions::default())?;
/// let l = p.latency("audio")?;
/// assert!(l.group_delay.as_f64() > 0.0);
/// # Ok::<(), sdrk_pipeline::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Latency {
    /// Group delay from the input to the output in samples of the output.
    pub group_delay: Delay,
    /// Worst-case number of samples that the nodes along the path hold back, in samples of the
    /// output.
    pub max_buffered: u64,
}

/// The queue of events. Allocated at build; `push` and `pop` never allocate.
pub(crate) struct EventQueue {
    queue: VecDeque<Event>,
    cap: usize,
    lost: u64,
    first_lost: Option<(OutputId, u64)>,
}

impl EventQueue {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(cap),
            cap,
            lost: 0,
            first_lost: None,
        }
    }

    /// Appends an event; the newest event is dropped (and counted) if the queue is full.
    pub(crate) fn push(&mut self, event: Event) {
        if self.queue.len() < self.cap {
            self.queue.push_back(event);
        } else {
            self.lost += 1;
            self.first_lost.get_or_insert((event.output, event.index));
        }
    }

    /// The oldest event; once the queue is empty, the report of the events lost.
    pub(crate) fn pop(&mut self) -> Option<Event> {
        if let Some(e) = self.queue.pop_front() {
            return Some(e);
        }
        if self.lost > 0 {
            let (output, index) = self.first_lost.take().unwrap_or((OutputId(0), 0));
            let count = core::mem::take(&mut self.lost);
            return Some(Event {
                output,
                index,
                kind: EventKind::EventsLost { count },
            });
        }
        None
    }

    pub(crate) fn clear(&mut self) {
        self.queue.clear();
        self.lost = 0;
        self.first_lost = None;
    }
}
