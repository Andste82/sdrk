//! Turns a plan into a runnable pipeline: builds the nodes, allocates every buffer.

use std::collections::BTreeMap;

use sdrk_blocks::{FrameMeta, ParamValue, PortType};

use sdrk::{Delay, Rate};

use crate::analysis::{
    OutPath, OutSource, Plan, PlanDst, PlanInput, PlanNode, PlanOutput, PlanStream, elem_size,
    output_paths, output_unit_rate,
};
use crate::latency::{PathNode, path_latency};
use crate::model::{PipelineSpec, UnderrunPolicy};
use crate::node::{NodeInput, NodeOutput};
use crate::registry::Registry;
use crate::runner::{
    Buffer, Burst, Dst, FrameQueue, InRt, NodeRt, NodeStats, OutKind, OutRt, Pipeline, QUEUE_MARKS,
    Queue, Reach, Stream,
};
use crate::storage::{VecStorage, alloc_vec};
use crate::structure::join;
use crate::time::EventQueue;
use crate::validate::{DiagCode, Diagnostic, Report};

/// A problem found while building: it becomes an error of the report.
type Failure = Diagnostic;

fn limit(message: impl Into<String>) -> Failure {
    Diagnostic {
        path: "nodes".to_string(),
        code: DiagCode::LimitExceeded,
        message: message.into(),
    }
}

/// Elements of `stream` per element of its input stream: `(num, den)`.
fn gain(stream: &PlanStream, origin: &PlanStream) -> Option<(u128, u128)> {
    let big = u128::from;
    let num = big(stream.rate.num())
        .checked_mul(big(origin.rate.den()))?
        .checked_mul(big(stream.elems.0))?
        .checked_mul(big(origin.elems.1))?;
    let den = big(stream.rate.den())
        .checked_mul(big(origin.rate.num()))?
        .checked_mul(big(stream.elems.1))?
        .checked_mul(big(origin.elems.0))?;
    Some((num, den))
}

/// Builds the blocks of all nodes (in processing order).
fn build_nodes(
    registry: &Registry,
    plan_nodes: &[PlanNode],
    streams: &mut [PlanStream],
    report: &mut Report,
    pending_cap: usize,
) -> Result<Vec<NodeRt>, Failure> {
    let mut storage = VecStorage;
    let mut nodes: Vec<NodeRt> = Vec::with_capacity(plan_nodes.len());
    for (i, pn) in plan_nodes.iter().enumerate() {
        let factory = registry.factory(&pn.kind).ok_or_else(|| Diagnostic {
            path: "nodes".to_string(),
            code: DiagCode::UnknownKind,
            message: format!("unknown kind `{}`", pn.kind),
        })?;
        let node = factory.build(&pn.checked, &mut storage).map_err(|d| {
            let path = if pn.auto {
                "nodes".to_string()
            } else {
                join("nodes", &pn.name)
            };
            Diagnostic {
                path: if d.path.is_empty() {
                    path
                } else {
                    join(&path, &d.path)
                },
                ..d
            }
        })?;
        let ratio = node
            .timing()
            .map_or((1, 1), |t| (t.ratio.num(), t.ratio.den()));
        let delay = node.timing().map_or(Delay::ZERO, |t| t.delay);
        if let (Some(t), Some(r)) = (node.timing(), report.nodes.get_mut(i)) {
            r.delay_samples_in = t.delay;
        }
        if let Some(s) = streams.get_mut(pn.src) {
            s.min_input = s.min_input.max(node.min_input());
        }
        nodes.push(NodeRt {
            name: pn.name.clone(),
            node,
            src: pn.src,
            slot: 0,
            dst: match pn.dst {
                PlanDst::Stream(s) => Dst::Stream(s),
                PlanDst::Frames { output } => Dst::Frames(output),
            },
            ratio,
            changed: 0,
            consumed: 0,
            n0_in: 0,
            n0_out: 0,
            rate_in: pn.rate_in,
            rate_out: pn.checked.rate_out,
            delay,
            input: streams.get(pn.src).map_or(0, |s| s.origin),
            upstream: Vec::new(),
            pending: Vec::with_capacity(pending_cap),
        });
    }
    Ok(nodes)
}

/// Allocates the buffer of every stream and registers the node readers.
fn build_streams(
    streams: &[PlanStream],
    inputs: &[PlanInput],
    nodes: &mut [NodeRt],
    max_chunk: usize,
    memory: &mut usize,
) -> Result<Vec<Stream>, Failure> {
    let mut rt: Vec<Stream> = Vec::with_capacity(streams.len());
    for s in streams {
        let origin_rate = inputs.get(s.origin).map_or(s.rate, |i| i.rate);
        let cap = s.capacity(max_chunk, origin_rate, s.min_input);
        let buf = Buffer::new(s.ty, cap).map_err(|e| limit(e.to_string()))?;
        *memory = memory.saturating_add(cap.saturating_mul(elem_size(s.ty)));
        rt.push(Stream {
            buf,
            end: 0,
            readers: Vec::new(),
            elems: s.elems,
            origin_idx: 0,
            origin_el: 0,
            total_el: 0,
        });
    }
    for n in nodes {
        if let Some(s) = rt.get_mut(n.src) {
            n.slot = s.readers.len();
            s.readers.push(0);
        }
    }
    Ok(rt)
}

fn frame_queue(len: usize, capacity: usize) -> Result<FrameQueue, sdrk_blocks::Error> {
    let mut metas = Vec::new();
    metas
        .try_reserve_exact(capacity)
        .map_err(|_| sdrk_blocks::Error::Storage {
            needed: capacity,
            available: 0,
        })?;
    metas.resize(
        capacity,
        FrameMeta {
            first_sample: 0,
            averaged: 0,
            exponent: 0,
        },
    );
    Ok(FrameQueue {
        len,
        cap: capacity,
        data: alloc_vec(capacity.saturating_mul(len))?,
        metas,
        scratch: alloc_vec(len)?,
        head: 0,
        count: 0,
    })
}

/// Allocates the queue of one output.
fn build_output(
    po: &PlanOutput,
    rate: Rate,
    streams: &[PlanStream],
    rt_streams: &mut [Stream],
    memory: &mut usize,
) -> Result<OutRt, Failure> {
    let kind = match po.source {
        OutSource::Stream(sid) => {
            let ty = streams.get(sid).map_or(PortType::Raw, |s| s.ty);
            let buf = Buffer::new(ty, po.capacity).map_err(|e| limit(e.to_string()))?;
            *memory = memory.saturating_add(po.capacity.saturating_mul(elem_size(ty)));
            let elems = streams.get(sid).map_or((1, 1), |s| s.elems);
            let slot = rt_streams.get(sid).map_or(0, |s| s.readers.len());
            if let Some(s) = rt_streams.get_mut(sid) {
                s.readers.push(0);
            }
            OutKind::Stream {
                src: sid,
                slot,
                ty,
                queue: Queue {
                    buf,
                    start: 0,
                    end: 0,
                    elems,
                    total_in: 0,
                    total_out: 0,
                    marks: Vec::with_capacity(QUEUE_MARKS),
                },
            }
        }
        OutSource::Frames { len } => {
            let q = frame_queue(len, po.capacity).map_err(|e| limit(e.to_string()))?;
            let per_frame = len.saturating_mul(4).saturating_add(24);
            *memory = memory
                .saturating_add(po.capacity.saturating_mul(per_frame))
                .saturating_add(len.saturating_mul(4));
            OutKind::Frames(q)
        }
    };
    let raw = matches!(
        &kind,
        OutKind::Stream {
            ty: PortType::Raw,
            ..
        }
    );
    Ok(OutRt {
        name: po.name.clone(),
        policy: po.spec.drop_policy(),
        dropped: 0,
        kind,
        input: 0,
        path: Vec::new(),
        rate,
        anchor: None,
        lead: Delay::ZERO,
        lead_in: Delay::ZERO,
        max_buffered: 0,
        underrun: po.spec.underrun.unwrap_or(if raw {
            UnderrunPolicy::Zeros
        } else {
            UnderrunPolicy::Short
        }),
        burst: Burst::Continuous,
        silence: Vec::new(),
    })
}

/// The stream an output depends on, and the number of input elements per frame hop.
fn output_source(
    oi: usize,
    po: &PlanOutput,
    nodes: &[NodeRt],
    plan_nodes: &[PlanNode],
) -> Option<(usize, u128)> {
    match po.source {
        OutSource::Stream(s) => Some((s, 1)),
        OutSource::Frames { .. } => {
            let (n, pn) = nodes
                .iter()
                .zip(plan_nodes)
                .find(|(_, pn)| pn.dst == PlanDst::Frames { output: oi })?;
            let hop = match n.node.param("interval_samples") {
                Ok(ParamValue::I64(h)) => u128::try_from(h.max(1)).unwrap_or(1),
                _ => 1,
            };
            Some((pn.src, hop))
        }
    }
}

/// The runtime form of the inputs, with the outputs that depend on each of them.
fn build_inputs(
    plan: (&[PlanInput], &[PlanNode], &[PlanOutput]),
    streams: &[PlanStream],
    nodes: &[NodeRt],
    max_chunk: usize,
) -> Vec<InRt> {
    let (plan_inputs, plan_nodes, plan_outputs) = plan;
    let mut inputs: Vec<InRt> = Vec::with_capacity(plan_inputs.len());
    for (ii, pi) in plan_inputs.iter().enumerate() {
        let elems = streams.get(pi.stream).map_or((1, 1), |s| s.elems);
        let max_len = if pi.ty == PortType::Raw {
            let n = (max_chunk as u128 * u128::from(elems.0)).div_ceil(u128::from(elems.1.max(1)));
            usize::try_from(n).unwrap_or(usize::MAX)
        } else {
            max_chunk
        };
        let readers: Vec<usize> = plan_nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.src == pi.stream)
            .map(|(i, _)| i)
            .collect();
        let mut reach = Vec::new();
        let origin = streams.get(pi.stream);
        for (oi, po) in plan_outputs.iter().enumerate() {
            let Some((sid, hop)) = output_source(oi, po, nodes, plan_nodes) else {
                continue;
            };
            let Some(s) = streams.get(sid).filter(|s| s.origin == ii) else {
                continue;
            };
            if let Some((num, den)) = origin.and_then(|o| gain(s, o)) {
                reach.push(Reach {
                    output: oi,
                    num,
                    den: den.saturating_mul(hop),
                });
            }
        }
        let tree_nodes: Vec<usize> = plan_nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| streams.get(n.src).is_some_and(|s| s.origin == ii))
            .map(|(i, _)| i)
            .collect();
        let tree_streams: Vec<usize> = streams
            .iter()
            .enumerate()
            .filter(|(_, s)| s.origin == ii)
            .map(|(i, _)| i)
            .collect();
        inputs.push(InRt {
            name: pi.name.clone(),
            stream: pi.stream,
            ty: pi.ty,
            rate: pi.rate,
            max_len,
            zeros: crate::pull::zero_input(
                pi.ty,
                if pi.ty == PortType::Raw {
                    usize::try_from(crate::runner::elems_of(max_len as u64, elems))
                        .unwrap_or(max_len)
                } else {
                    max_len
                },
            ),
            min_input: origin.map_or(1, |s| s.min_input),
            readers,
            reach,
            tree_nodes,
            tree_streams,
            outputs: Vec::new(),
            anchor: None,
            fresh: true,
        });
    }
    inputs
}

pub(crate) fn build(
    registry: &Registry,
    spec: &PipelineSpec,
    plan: Plan,
    mut report: Report,
) -> Result<Pipeline, Report> {
    assemble(registry, spec, plan, &mut report).map_err(|d| {
        report.errors.push(d);
        report
    })
}

#[allow(
    clippy::large_stack_frames,
    reason = "clippy counts the temporary of every `?` separately; the real frame is small"
)]
fn assemble(
    registry: &Registry,
    spec: &PipelineSpec,
    plan: Plan,
    report: &mut Report,
) -> Result<Pipeline, Failure> {
    let Plan {
        max_chunk,
        inputs: plan_inputs,
        mut streams,
        nodes: plan_nodes,
        outputs: plan_outputs,
    } = plan;
    let pending_cap = spec.settings.pending_params_len();
    let mut nodes = build_nodes(registry, &plan_nodes, &mut streams, report, pending_cap)?;
    let mut memory: usize = plan_nodes.iter().map(|p| p.checked.memory_bytes).sum();
    let mut rt_streams = build_streams(&streams, &plan_inputs, &mut nodes, max_chunk, &mut memory)?;
    let paths = output_paths(&plan_nodes, &streams, &plan_outputs);
    let mut outputs: Vec<OutRt> = Vec::with_capacity(plan_outputs.len());
    let mut output_index = BTreeMap::new();
    for (oi, (po, path)) in plan_outputs.iter().zip(&paths).enumerate() {
        let rate = output_unit_rate(po, path, &plan_nodes, &streams)
            .ok_or_else(|| limit("output without a source"))?;
        outputs.push(build_output(
            po,
            rate,
            &streams,
            &mut rt_streams,
            &mut memory,
        )?);
        output_index.insert(po.name.clone(), oi);
    }
    let mut inputs = build_inputs(
        (&plan_inputs, &plan_nodes, &plan_outputs),
        &streams,
        &nodes,
        max_chunk,
    );
    connect_paths(
        registry,
        (&plan_nodes, &plan_inputs, &plan_outputs),
        &paths,
        &streams,
        &mut nodes,
        &mut outputs,
        &mut inputs,
        report,
    );
    let input_index = plan_inputs
        .iter()
        .enumerate()
        .map(|(i, p)| (p.name.clone(), i))
        .collect();
    let node_index = plan_nodes
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.auto)
        .map(|(i, p)| (p.name.clone(), i))
        .collect();
    let stats = plan_nodes
        .iter()
        .map(|n| NodeStats {
            name: n.name.clone(),
            calls: 0,
            samples_in: 0,
            samples_out: 0,
            frames: 0,
        })
        .collect();
    report.memory_bytes = memory;
    Ok(Pipeline {
        spec: spec.clone(),
        report: std::mem::take(report),
        inputs,
        input_index,
        nodes,
        node_index,
        streams: rt_streams,
        outputs,
        output_index,
        stats,
        events: EventQueue::new(spec.settings.event_queue_len()),
        pending_cap,
        gap_scratch: vec![0; streams.len()],
    })
}

/// One packing group of silence for a raw output fed by a `convert_out` node: the node
/// converts zero samples.
fn silence_group(
    registry: &Registry,
    plan_nodes: &[PlanNode],
    streams: &[PlanStream],
    sid: usize,
) -> Vec<u8> {
    let Some(stream) = streams.get(sid) else {
        return Vec::new();
    };
    let Some(pn) = plan_nodes.iter().find(|n| n.dst == PlanDst::Stream(sid)) else {
        return Vec::new();
    };
    let Some(factory) = registry.factory(&pn.kind) else {
        return Vec::new();
    };
    let Ok(mut node) = factory.build(&pn.checked, &mut VecStorage) else {
        return Vec::new();
    };
    let (bytes, samples) = (stream.elems.0 as usize, stream.elems.1 as usize);
    let mut out = vec![0u8; bytes];
    let mut meta = None;
    let progress = match streams.get(pn.src).map(|s| s.ty) {
        Some(PortType::C32) => {
            let zeros = vec![sdrk::C32::default(); samples];
            node.process(NodeInput::C32(&zeros), NodeOutput::Raw(&mut out), &mut meta)
        }
        Some(PortType::CQ15) => {
            let zeros = vec![sdrk::CQ15::default(); samples];
            node.process(
                NodeInput::CQ15(&zeros),
                NodeOutput::Raw(&mut out),
                &mut meta,
            )
        }
        _ => return Vec::new(),
    };
    match progress {
        Ok(p) if p.produced == bytes => out,
        _ => Vec::new(),
    }
}

/// Fills the path, delay and silence fields that need the whole plan.
#[allow(clippy::too_many_arguments)]
fn connect_paths(
    registry: &Registry,
    plan: (&[PlanNode], &[PlanInput], &[PlanOutput]),
    paths: &[OutPath],
    streams: &[PlanStream],
    nodes: &mut [NodeRt],
    outputs: &mut [OutRt],
    inputs: &mut [InRt],
    report: &mut Report,
) {
    let (plan_nodes, plan_inputs, plan_outputs) = plan;
    // The nodes upstream of every node.
    for k in 0..nodes.len() {
        let mut up = Vec::new();
        let mut cur = nodes.get(k).map(|n| n.src);
        while let Some(sid) = cur {
            match plan_nodes
                .iter()
                .position(|n| n.dst == PlanDst::Stream(sid))
            {
                Some(p) => {
                    up.push(p);
                    cur = plan_nodes.get(p).map(|n| n.src);
                }
                None => break,
            }
        }
        up.reverse();
        if let Some(n) = nodes.get_mut(k) {
            n.upstream = up;
        }
    }
    for (oi, ((out, path), po)) in outputs.iter_mut().zip(paths).zip(plan_outputs).enumerate() {
        out.input = path.input;
        out.path.clone_from(&path.nodes);
        if let Some(i) = inputs.get_mut(path.input) {
            i.outputs.push(oi);
        }
        let stream_nodes: Vec<PathNode> = path
            .nodes
            .iter()
            .filter_map(|k| nodes.get(*k).zip(plan_nodes.get(*k)))
            .filter(|(n, _)| matches!(n.dst, Dst::Stream(_)))
            .map(|(n, pn)| PathNode {
                rate_in: n.rate_in,
                delay: n.delay,
                min_input: pn.checked.min_input.max(n.node.min_input()),
            })
            .collect();
        if let Some((lead, buffered)) = path_latency(&stream_nodes, out.rate) {
            out.lead = lead;
            out.max_buffered = buffered;
        }
        if let Some(rate_in) = plan_inputs.get(path.input).map(|i| i.rate)
            && let Some((lead_in, _)) = path_latency(&stream_nodes, rate_in)
        {
            out.lead_in = lead_in;
        }
        if let Some(r) = report.outputs.get_mut(oi) {
            r.group_delay = out.lead;
            r.max_buffered = out.max_buffered;
        }
        if let OutSource::Stream(sid) = po.source
            && streams.get(sid).is_some_and(|s| s.ty == PortType::Raw)
        {
            out.silence = silence_group(registry, plan_nodes, streams, sid);
        }
    }
}
