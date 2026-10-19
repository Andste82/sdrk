//! Index domains, metadata input, events, timed parameters and latency of the runner.

use sdrk::Delay;
use sdrk_blocks::{Mutability, ParamValue};
use sdrk_device::{BlockFlags, BlockMeta, SampleIndex, TimeAnchor};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::params::to_param;
use crate::runner::{
    Dst, InputData, NodeRt, Pending, Pipeline, Stream, at, at_mut, elems_of, internal, samples_of,
};
use crate::time::{Event, EventKind, Latency, NodeId, OutputId, ParamTime};

/// Output samples of a node for `rel` input samples since its origin. The blocks count in
/// elements (a raw stream has one element per byte), so the counts are converted at the edges.
pub(crate) fn map_rel(n: &NodeRt, src: &Stream, dst: &Stream, rel: u64) -> u64 {
    let rel_el =
        u64::try_from(u128::from(rel) * u128::from(src.elems.0) / u128::from(src.elems.1.max(1)))
            .unwrap_or(u64::MAX);
    samples_of(n.node.map_index(rel_el), dst.elems)
}

/// Maps `idx`, an index in the domain of the input of `path[0]`, to the domain after the last
/// node of the path (§1 of the milestone: `m = n0_out + map_index(n - n0_in)`, node by node).
pub(crate) fn map_along(nodes: &[NodeRt], streams: &[Stream], path: &[usize], idx: u64) -> u64 {
    let mut idx = idx;
    for k in path {
        let Some(n) = nodes.get(*k) else {
            continue;
        };
        let Dst::Stream(d) = n.dst else {
            // A frame node maps every index to itself.
            continue;
        };
        let (Some(src), Some(dst)) = (streams.get(n.src), streams.get(d)) else {
            continue;
        };
        idx = n
            .n0_out
            .saturating_add(map_rel(n, src, dst, idx.saturating_sub(n.n0_in)));
    }
    idx
}

/// `ceil(a / b)` for a signed numerator and a positive denominator.
fn div_ceil_signed(a: i128, b: i128) -> i128 {
    -((-a).div_euclid(b))
}

/// Nanoseconds that the signal of output sample `out_rel` (counted from the origin of the node)
/// lies after the input anchor, relative to the anchored input sample `in_rel` (also from the
/// origin): `(out_rel * M / L - in_rel - D) / rate_in`, rounded half up.
///
/// `L / M` is the ratio of the sample rates and `D` the group delay in input samples. The
/// arithmetic is exact in `i128`; `None` on overflow.
fn signal_offset_ns(node: &NodeRt, out_rel: i128, in_rel: i128) -> Option<i128> {
    let up = i128::from(node.rate_out.num()) * i128::from(node.rate_in.den());
    let down = i128::from(node.rate_out.den()) * i128::from(node.rate_in.num());
    let delay_num = i128::from(node.delay.num());
    let delay_den = i128::from(node.delay.den());
    // The offset in input samples as a fraction over `up * delay_den`.
    let top = out_rel
        .checked_mul(down)?
        .checked_mul(delay_den)?
        .checked_sub(delay_num.checked_mul(up)?)?
        .checked_sub(in_rel.checked_mul(up)?.checked_mul(delay_den)?)?;
    let bottom = up
        .checked_mul(delay_den)?
        .checked_mul(i128::from(node.rate_in.num()))?;
    let scaled = top
        .checked_mul(1_000_000_000)?
        .checked_mul(i128::from(node.rate_in.den()))?;
    Some(
        scaled
            .checked_mul(2)?
            .checked_add(bottom)?
            .div_euclid(bottom.checked_mul(2)?),
    )
}

/// Carries an anchor through the nodes of `path` (delay-compensated, see `docs/timing.md`).
pub(crate) fn anchor_along(
    nodes: &[NodeRt],
    streams: &[Stream],
    path: &[usize],
    anchor: TimeAnchor,
) -> Option<TimeAnchor> {
    let mut index = anchor.index.0;
    let mut time = i128::from(anchor.unix_ns);
    for k in path {
        let node = nodes.get(*k)?;
        let Dst::Stream(dst_stream) = node.dst else {
            continue;
        };
        let (src, dst) = (streams.get(node.src)?, streams.get(dst_stream)?);
        let in_rel = i128::from(index) - i128::from(node.n0_in);
        let mapped =
            node.n0_out
                .saturating_add(map_rel(node, src, dst, index.saturating_sub(node.n0_in)));
        let out_rel = i128::from(mapped - node.n0_out);
        time = time.checked_add(signal_offset_ns(node, out_rel, in_rel)?)?;
        index = mapped;
    }
    Some(TimeAnchor {
        index: SampleIndex(index),
        unix_ns: i64::try_from(time).ok()?,
        source: anchor.source,
    })
}

impl Pipeline {
    // ------------------------------------------------------------------ names and ids

    /// The id of a node (automatic nodes included).
    ///
    /// # Errors
    /// `UnknownNode`.
    pub fn node_id(&self, name: &str) -> Result<NodeId> {
        self.nodes
            .iter()
            .position(|n| n.name == name)
            .map(NodeId::new)
            .ok_or_else(|| Error::UnknownNode(name.to_string()))
    }

    /// The id of an output.
    ///
    /// # Errors
    /// `UnknownOutput`.
    pub fn output_id(&self, name: &str) -> Result<OutputId> {
        self.output_pos(name).map(OutputId::new)
    }

    /// The name of a node; empty for an id that does not belong to this pipeline.
    pub fn node_name(&self, id: NodeId) -> &str {
        self.nodes.get(id.index()).map_or("", |n| n.name.as_str())
    }

    /// The name of an output; empty for an id that does not belong to this pipeline.
    pub fn output_name(&self, id: OutputId) -> &str {
        self.outputs.get(id.index()).map_or("", |o| o.name.as_str())
    }

    /// The oldest event, or `None`. Never allocates.
    pub fn poll_event(&mut self) -> Option<Event> {
        self.events.pop()
    }

    // ------------------------------------------------------------------ indices

    /// Sample index of the next sample that the pipeline expects at an input.
    ///
    /// ```
    /// use sdrk_pipeline::{BuildOptions, InputData, Registry};
    /// let mut p = Registry::standard().build_json(
    ///     include_str!("../../../../examples/flows/fm_receiver.json"),
    ///     &BuildOptions::default(),
    /// )?;
    /// p.push("rf", InputData::Raw(&[128u8; 2000]))?;
    /// assert_eq!(p.input_index("rf")?, 1000);
    /// # Ok::<(), sdrk_pipeline::Error>(())
    /// ```
    ///
    /// # Errors
    /// `UnknownInput`.
    pub fn input_index(&self, input: &str) -> Result<u64> {
        let ii = self.input_pos(input)?;
        let sid = at(&self.inputs, ii)?.stream;
        Ok(at(&self.streams, sid)?.next_idx())
    }

    /// Sample index of the first sample waiting in a stream output (the next one `read_stream`
    /// returns), or of the next sample to come when the output is empty.
    ///
    /// ```
    /// use sdrk_pipeline::{BuildOptions, InputData, Registry};
    /// let mut p = Registry::standard().build_json(
    ///     include_str!("../../../../examples/flows/fm_receiver.json"),
    ///     &BuildOptions::default(),
    /// )?;
    /// p.push("rf", InputData::Raw(&[128u8; 4000]))?;
    /// let mut audio = [0f32; 16];
    /// assert_eq!(p.read_index("audio")?, 0);
    /// let n = p.read_stream("audio", &mut audio)?;
    /// assert_eq!(p.read_index("audio")?, n as u64);
    /// # Ok::<(), sdrk_pipeline::Error>(())
    /// ```
    ///
    /// # Errors
    /// `UnknownOutput`, `WrongType` (a frames output).
    pub fn read_index(&self, output: &str) -> Result<u64> {
        let o = at(&self.outputs, self.output_pos(output)?)?;
        match &o.kind {
            crate::runner::OutKind::Stream { queue, .. } => Ok(queue.first_idx()),
            crate::runner::OutKind::Frames(_) => Err(Error::WrongType {
                name: output.to_string(),
                expected: "frames",
                got: "a stream",
            }),
        }
    }

    /// Absolute time (unix nanoseconds) of sample `index` of an output, from the latest anchor
    /// of that output; `None` if no anchor was reported yet.
    ///
    /// Anchors are delay-compensated: the time is that of the signal in the sample, not of
    /// its arrival.
    ///
    /// ```
    /// use sdrk_device::{BlockMeta, SampleIndex, TimeAnchor, TimeSource};
    /// use sdrk_pipeline::{BuildOptions, InputData, Registry};
    /// let mut p = Registry::standard().build_json(
    ///     include_str!("../../../../examples/flows/fm_receiver.json"),
    ///     &BuildOptions::default(),
    /// )?;
    /// assert_eq!(p.time_of("audio", 0)?, None);
    /// let anchor = TimeAnchor { index: SampleIndex(0), unix_ns: 1_000_000, source: TimeSource::Host };
    /// let meta = BlockMeta { anchor: Some(anchor), ..BlockMeta::default() };
    /// p.push_with_meta("rf", InputData::Raw(&[128u8; 2000]), &meta)?;
    /// // The filters delay the signal, so audio sample 0 shows the signal of earlier input.
    /// assert!(p.time_of("audio", 0)?.unwrap() < 1_000_000);
    /// # Ok::<(), sdrk_pipeline::Error>(())
    /// ```
    ///
    /// # Errors
    /// `UnknownOutput`, `InvalidState` (the time does not fit).
    pub fn time_of(&self, output: &str, index: u64) -> Result<Option<i64>> {
        let o = at(&self.outputs, self.output_pos(output)?)?;
        match o.anchor {
            None => Ok(None),
            Some(a) => a
                .time_of(SampleIndex(index), o.rate)
                .map(Some)
                .map_err(|e| Error::InvalidState(e.to_string())),
        }
    }

    /// Group delay and buffering from the input to an output, see `docs/timing.md`.
    ///
    /// # Errors
    /// `UnknownOutput`.
    pub fn latency(&self, output: &str) -> Result<Latency> {
        let o = at(&self.outputs, self.output_pos(output)?)?;
        Ok(Latency {
            group_delay: o.lead,
            max_buffered: o.max_buffered,
        })
    }

    /// Index of the next sample node `ni` reads, in the domain of its input.
    pub(crate) fn node_in_idx(&self, ni: usize) -> Result<u64> {
        let n = at(&self.nodes, ni)?;
        let s = at(&self.streams, n.src)?;
        Ok(n.n0_in.saturating_add(samples_of(n.consumed, s.elems)))
    }

    // ------------------------------------------------------------------ metadata input

    /// Like `push`, with the metadata of the block: its start index, flags and anchor.
    ///
    /// A start index above the expected one is a gap: the nodes that depend on the input are
    /// reset and a `Gap` event is reported on every output reachable from the input (the
    /// result equals two separately reset runs). A start index below the expected one fails
    /// because time cannot go backwards. Flags and the anchor become events at the mapped
    /// positions of every reachable output. Raw data must hold whole packing groups.
    ///
    /// ```
    /// use sdrk_device::{BlockFlags, BlockMeta, SampleIndex};
    /// use sdrk_pipeline::{BuildOptions, EventKind, InputData, Registry};
    /// let mut p = Registry::standard().build_json(
    ///     include_str!("../../../../examples/flows/fm_receiver.json"),
    ///     &BuildOptions::default(),
    /// )?;
    /// let bytes = [128u8; 2000];
    /// let first = BlockMeta { start: SampleIndex(5000), ..BlockMeta::default() };
    /// p.push_with_meta("rf", InputData::Raw(&bytes), &first)?; // fixes the origin
    /// let next = BlockMeta {
    ///     start: SampleIndex(6000),
    ///     flags: BlockFlags::OVERFLOW,
    ///     ..BlockMeta::default()
    /// };
    /// p.push_with_meta("rf", InputData::Raw(&bytes), &next)?;
    /// let event = p.poll_event().unwrap();
    /// assert_eq!(event.kind, EventKind::Overflow);
    /// assert_eq!(p.output_name(event.output), "audio");
    /// assert_eq!(event.index, 120); // the origin 5000 / 10 / 5 = 100, plus 1000 / 10 / 5 = 20
    /// # Ok::<(), sdrk_pipeline::Error>(())
    /// ```
    ///
    /// # Errors
    /// The errors of `push`, `InvalidState` (the start index lies before the expected one) and
    /// `LengthMismatch` (raw data that is not a whole number of packing groups).
    #[allow(
        clippy::large_stack_frames,
        reason = "clippy counts the temporary of every `?` separately; the real frame is small"
    )]
    pub fn push_with_meta(
        &mut self,
        input: &str,
        data: InputData<'_>,
        meta: &BlockMeta,
    ) -> Result<()> {
        let ii = self.input_pos(input)?;
        self.check_push(ii, &data)?;
        let sid = at(&self.inputs, ii)?.stream;
        let elems = at(&self.streams, sid)?.elems;
        if matches!(data, InputData::Raw(_)) && data.len() as u64 % elems.0.max(1) != 0 {
            return Err(Error::LengthMismatch {
                expected: usize::try_from(elems.0).unwrap_or(usize::MAX),
                got: data.len(),
            });
        }
        let expected = at(&self.streams, sid)?.next_idx();
        let start = meta.start.0;
        if start < expected {
            return Err(Error::InvalidState(format!(
                "the block starts at sample {start} of input `{input}`, which is before the expected sample {expected}"
            )));
        }
        let samples = samples_of(data.len() as u64, elems);
        let fresh = core::mem::replace(&mut at_mut(&mut self.inputs, ii)?.fresh, false);
        if start > expected {
            // The first block of an input only fixes the origin; later blocks make a gap.
            self.gap(ii, expected, start, !fresh)?;
        }
        self.meta_events(ii, meta, samples)?;
        self.ingest(ii, &data)?;
        self.run()
    }

    /// Resets everything that depends on input `ii` and moves the origins by the gap.
    #[allow(
        clippy::large_stack_frames,
        reason = "clippy counts the temporary of every `?` separately; the real frame is small"
    )]
    fn gap(&mut self, ii: usize, expected: u64, start: u64, report: bool) -> Result<()> {
        let missing = start - expected;
        let tree_nodes = at(&self.inputs, ii)?.tree_nodes.len();
        let tree_streams = at(&self.inputs, ii)?.tree_streams.len();
        // Missing samples per stream: the gap at the input, mapped with the nominal ratios.
        let input_stream = at(&self.inputs, ii)?.stream;
        *at_mut(&mut self.gap_scratch, input_stream)? = missing;
        for k in 0..tree_nodes {
            let ni = *at(&at(&self.inputs, ii)?.tree_nodes, k)?;
            let n = at(&self.nodes, ni)?;
            if let Dst::Stream(d) = n.dst {
                let miss = *at(&self.gap_scratch, n.src)?;
                let mapped = map_rel(n, at(&self.streams, n.src)?, at(&self.streams, d)?, miss);
                *at_mut(&mut self.gap_scratch, d)? = mapped;
            }
        }
        for k in 0..tree_streams {
            let si = *at(&at(&self.inputs, ii)?.tree_streams, k)?;
            let miss = *at(&self.gap_scratch, si)?;
            let s = at_mut(&mut self.streams, si)?;
            let origin = s.next_idx().saturating_add(miss);
            s.rebase(origin);
        }
        for k in 0..tree_nodes {
            let ni = *at(&at(&self.inputs, ii)?.tree_nodes, k)?;
            let n = at_mut(&mut self.nodes, ni)?;
            n.node.reset();
            n.consumed = 0;
            n.n0_in = at(&self.streams, n.src)?.origin_idx;
            if let Dst::Stream(d) = n.dst {
                n.n0_out = at(&self.streams, d)?.origin_idx;
            }
        }
        if !report {
            return Ok(());
        }
        let outs = at(&self.inputs, ii)?.outputs.len();
        for k in 0..outs {
            let oi = *at(&at(&self.inputs, ii)?.outputs, k)?;
            let o = at(&self.outputs, oi)?;
            let index = map_along(&self.nodes, &self.streams, &o.path, start);
            self.events.push(Event {
                output: OutputId::new(oi),
                index,
                kind: EventKind::Gap { missing },
            });
        }
        Ok(())
    }

    /// Events for the flags and the anchor of a block.
    #[allow(
        clippy::large_stack_frames,
        reason = "clippy counts the temporary of every `?` separately; the real frame is small"
    )]
    fn meta_events(&mut self, ii: usize, meta: &BlockMeta, samples: u64) -> Result<()> {
        let start = meta.start.0;
        let last = start.saturating_add(samples.saturating_sub(1));
        let flags = meta.flags;
        if let Some(a) = meta.anchor {
            at_mut(&mut self.inputs, ii)?.anchor = Some(a);
        }
        let outs = at(&self.inputs, ii)?.outputs.len();
        for k in 0..outs {
            let oi = *at(&at(&self.inputs, ii)?.outputs, k)?;
            let Self {
                nodes,
                streams,
                outputs,
                events,
                ..
            } = self;
            let o = at_mut(outputs, oi)?;
            let id = OutputId::new(oi);
            let simple = [
                (BlockFlags::OVERFLOW, EventKind::Overflow),
                (BlockFlags::UNDERRUN, EventKind::Underrun),
                (BlockFlags::RETUNED, EventKind::Retuned),
                (BlockFlags::LATE, EventKind::Late),
                (
                    BlockFlags::BURST_START,
                    EventKind::BurstStart {
                        tx_time_ns: meta.tx_time_ns,
                    },
                ),
            ];
            for (flag, kind) in simple {
                if flags.contains(flag) {
                    events.push(Event {
                        output: id,
                        index: map_along(nodes, streams, &o.path, start),
                        kind,
                    });
                }
            }
            if flags.contains(BlockFlags::BURST_END) {
                events.push(Event {
                    output: id,
                    index: map_along(nodes, streams, &o.path, last),
                    kind: EventKind::BurstEnd,
                });
            }
            if let Some(a) = meta.anchor {
                if let Some(mapped) = anchor_along(nodes, streams, &o.path, a) {
                    o.anchor = Some(mapped);
                    events.push(Event {
                        output: id,
                        index: mapped.index.0,
                        kind: EventKind::Anchor(mapped),
                    });
                }
            }
        }
        Ok(())
    }

    /// Reports the status that a transmit device returned from `TxStream::poll_status` as
    /// events of an output: the flags and the anchor of `meta` at the index `meta.start` of
    /// the output.
    ///
    /// ```
    /// use sdrk_device::{BlockFlags, BlockMeta, SampleIndex};
    /// use sdrk_pipeline::{BuildOptions, EventKind, Registry};
    /// let mut p = Registry::standard().build_json(
    ///     include_str!("../../../../examples/flows/fm_transmitter.json"),
    ///     &BuildOptions::default(),
    /// )?;
    /// let status = BlockMeta { start: SampleIndex(900), flags: BlockFlags::UNDERRUN, ..BlockMeta::default() };
    /// p.device_status("rf", &status)?;
    /// let event = p.poll_event().unwrap();
    /// assert_eq!((event.index, event.kind), (900, EventKind::Underrun));
    /// # Ok::<(), sdrk_pipeline::Error>(())
    /// ```
    ///
    /// # Errors
    /// `UnknownOutput`.
    pub fn device_status(&mut self, output: &str, meta: &BlockMeta) -> Result<()> {
        let oi = self.output_pos(output)?;
        let id = OutputId::new(oi);
        let index = meta.start.0;
        let flags = meta.flags;
        let table = [
            (BlockFlags::OVERFLOW, EventKind::Overflow),
            (BlockFlags::UNDERRUN, EventKind::Underrun),
            (BlockFlags::RETUNED, EventKind::Retuned),
            (BlockFlags::LATE, EventKind::Late),
            (
                BlockFlags::BURST_START,
                EventKind::BurstStart {
                    tx_time_ns: meta.tx_time_ns,
                },
            ),
            (BlockFlags::BURST_END, EventKind::BurstEnd),
        ];
        for (flag, kind) in table {
            if flags.contains(flag) {
                self.events.push(Event {
                    output: id,
                    index,
                    kind,
                });
            }
        }
        if let Some(a) = meta.anchor {
            at_mut(&mut self.outputs, oi)?.anchor = Some(a);
            self.events.push(Event {
                output: id,
                index: a.index.0,
                kind: EventKind::Anchor(a),
            });
        }
        Ok(())
    }

    // ------------------------------------------------------------------ timed parameters

    /// Looks up a run-time parameter and converts the JSON value.
    fn prepare_param(
        &self,
        ni: usize,
        node: &str,
        name: &str,
        value: &Value,
    ) -> Result<(u16, ParamValue)> {
        let n = at(&self.nodes, ni)?;
        let desc = n.node.description();
        let block_error = |source| Error::Block {
            node: node.to_string(),
            source,
        };
        let Some(index) = desc.params.iter().position(|p| p.name == name) else {
            return Err(block_error(sdrk_blocks::Error::UnknownParameter));
        };
        let p = at(desc.params, index)?;
        if p.mutability != Mutability::Runtime {
            return Err(block_error(sdrk_blocks::Error::ReadOnlyParameter));
        }
        let pv = to_param(p, value).map_err(|issue| match issue {
            crate::params::ParamIssue::Range(e) => block_error(e),
            other => Error::InvalidParameter {
                node: node.to_string(),
                name: name.to_string(),
                reason: other.message(value),
            },
        })?;
        let index = u16::try_from(index).map_err(|_| internal("parameter index"))?;
        Ok((index, pv))
    }

    /// Sets the parameter now and reports it on every output that depends on the node.
    fn apply_param(&mut self, ni: usize, param: u16, value: ParamValue, late: bool) -> Result<()> {
        let cur = self.node_in_idx(ni)?;
        let n = at_mut(&mut self.nodes, ni)?;
        let desc = n.node.description();
        let p = at(desc.params, usize::from(param))?;
        let applied = n.node.set_param(p.name, value);
        applied.map_err(|source| Error::Block {
            node: n.name.clone(),
            source,
        })?;
        if param < 64 {
            n.changed |= 1 << param;
        }
        let Self {
            nodes,
            streams,
            outputs,
            events,
            ..
        } = self;
        for (oi, o) in outputs.iter().enumerate() {
            let Some(pos) = o.path.iter().position(|k| *k == ni) else {
                continue;
            };
            let tail = o.path.get(pos..).unwrap_or_default();
            events.push(Event {
                output: OutputId::new(oi),
                index: map_along(nodes, streams, tail, cur),
                kind: EventKind::ParamApplied {
                    node: NodeId::new(ni),
                    param,
                    value,
                    late,
                },
            });
        }
        Ok(())
    }

    /// Applies the changes of node `ni` that are due and limits `avail` so that the next
    /// call of the node ends exactly at the next pending change.
    pub(crate) fn apply_pending(&mut self, ni: usize, avail: usize) -> Result<usize> {
        loop {
            let first = match at(&self.nodes, ni)?.pending.first() {
                None => return Ok(avail),
                Some(p) => *p,
            };
            let cur = self.node_in_idx(ni)?;
            if first.at <= cur {
                at_mut(&mut self.nodes, ni)?.pending.remove(0);
                self.apply_param(ni, first.param, first.value, false)?;
                continue;
            }
            let elems = at(&self.streams, at(&self.nodes, ni)?.src)?.elems;
            let until = usize::try_from(elems_of(first.at - cur, elems)).unwrap_or(usize::MAX);
            return Ok(avail.min(until.max(1)));
        }
    }

    /// Changes a run-time parameter at an exact sample.
    ///
    /// The target is mapped to the input domain of the node. The node's next `process` call
    /// ends exactly at that sample, the parameter is set, and the change is visible at the first
    /// output sample computed after it. A `ParamApplied` event is reported on every output
    /// that depends on the node, at the mapped index of the effective sample. A target in the
    /// past is applied at once with `late: true`.
    ///
    /// ```
    /// use sdrk_pipeline::{BuildOptions, EventKind, InputData, ParamTime, Registry};
    /// let mut p = Registry::standard().build_json(
    ///     include_str!("../../../../examples/flows/fm_receiver.json"),
    ///     &BuildOptions::default(),
    /// )?;
    /// p.set_param_at(
    ///     "tune",
    ///     "frequency_hz",
    ///     serde_json::json!(610_000.0),
    ///     ParamTime::InputIndex { input: "rf".into(), index: 1000 },
    /// )?;
    /// p.push("rf", InputData::Raw(&[128u8; 4000]))?;
    /// let ack = p.poll_event().unwrap();
    /// assert!(matches!(ack.kind, EventKind::ParamApplied { late: false, .. }));
    /// assert_eq!(ack.index, 20); // 1000 / 10 / 5 on the audio output
    /// # Ok::<(), sdrk_pipeline::Error>(())
    /// ```
    ///
    /// # Errors
    /// The errors of `set_param`, `UnknownInput`, `InvalidState` (the node is not fed by the
    /// input, or `UnixNs` without an anchor) and `PendingFull`.
    #[allow(
        clippy::large_stack_frames,
        reason = "clippy counts the temporary of every `?` separately; the real frame is small"
    )]
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the signature takes the JSON value and the time by value, like `set_param`"
    )]
    pub fn set_param_at(
        &mut self,
        node: &str,
        name: &str,
        value: Value,
        at_time: ParamTime,
    ) -> Result<()> {
        let ni = *self
            .node_index
            .get(node)
            .ok_or_else(|| Error::UnknownNode(node.to_string()))?;
        let (param, pv) = self.prepare_param(ni, node, name, &value)?;
        let target = match at_time {
            ParamTime::Immediate => {
                return self.apply_param(ni, param, pv, false);
            }
            ParamTime::InputIndex { input, index } => {
                let ii = self.input_pos(&input)?;
                let n = at(&self.nodes, ni)?;
                if n.input != ii {
                    return Err(Error::InvalidState(format!(
                        "node `{node}` is not fed by input `{input}`"
                    )));
                }
                map_along(&self.nodes, &self.streams, &n.upstream, index)
            }
            ParamTime::UnixNs(t) => {
                let n = at(&self.nodes, ni)?;
                let inp = at(&self.inputs, n.input)?;
                let a = inp.anchor.ok_or_else(|| {
                    Error::InvalidState(format!(
                        "input `{}` has no anchor; a time in nanoseconds needs one",
                        inp.name
                    ))
                })?;
                let delta = i128::from(t) - i128::from(a.unix_ns);
                let top = delta
                    .checked_mul(i128::from(inp.rate.num()))
                    .ok_or_else(|| Error::InvalidState("time arithmetic overflows".into()))?;
                let bottom = 1_000_000_000i128 * i128::from(inp.rate.den());
                let samples = div_ceil_signed(top, bottom);
                let index = (i128::from(a.index.0) + samples).max(0);
                let index = u64::try_from(index).unwrap_or(u64::MAX);
                map_along(&self.nodes, &self.streams, &n.upstream, index)
            }
        };
        let cur = self.node_in_idx(ni)?;
        if target < cur {
            return self.apply_param(ni, param, pv, true);
        }
        let cap = self.pending_cap;
        let n = at_mut(&mut self.nodes, ni)?;
        if n.pending.len() >= cap {
            return Err(Error::PendingFull(node.to_string()));
        }
        let pos = n.pending.partition_point(|p| p.at <= target);
        n.pending.insert(
            pos,
            Pending {
                at: target,
                param,
                value: pv,
            },
        );
        Ok(())
    }

    /// Group delay in output samples of the path of an output (the lead time of a transmit
    /// path).
    ///
    /// ```
    /// use sdrk_pipeline::{BuildOptions, Registry};
    /// let p = Registry::standard().build_json(
    ///     include_str!("../../../../examples/flows/fm_transmitter.json"),
    ///     &BuildOptions::default(),
    /// )?;
    /// assert!(p.lead_time("rf")?.as_f64() > 100.0);
    /// # Ok::<(), sdrk_pipeline::Error>(())
    /// ```
    ///
    /// # Errors
    /// `UnknownOutput`.
    pub fn lead_time(&self, output: &str) -> Result<Delay> {
        Ok(at(&self.outputs, self.output_pos(output)?)?.lead)
    }
}
