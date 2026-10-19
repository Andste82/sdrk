//! Turns a plan into a runnable pipeline: builds the nodes, allocates every buffer.

use std::collections::BTreeMap;

use sdrk_blocks::{FrameMeta, ParamValue, PortType};

use crate::analysis::{
    OutSource, Plan, PlanDst, PlanInput, PlanNode, PlanOutput, PlanStream, elem_size,
};
use crate::model::PipelineSpec;
use crate::registry::Registry;
use crate::runner::{
    Buffer, Dst, FrameQueue, InRt, NodeRt, NodeStats, OutKind, OutRt, Pipeline, Queue, Reach,
    Stream,
};
use crate::storage::{VecStorage, alloc_vec};
use crate::structure::join;
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
    streams: &[PlanStream],
    rt_streams: &mut [Stream],
    memory: &mut usize,
) -> Result<OutRt, Failure> {
    let kind = match po.source {
        OutSource::Stream(sid) => {
            let ty = streams.get(sid).map_or(PortType::Raw, |s| s.ty);
            let buf = Buffer::new(ty, po.capacity).map_err(|e| limit(e.to_string()))?;
            *memory = memory.saturating_add(po.capacity.saturating_mul(elem_size(ty)));
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
    Ok(OutRt {
        name: po.name.clone(),
        policy: po.spec.drop_policy(),
        dropped: 0,
        kind,
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
        inputs.push(InRt {
            name: pi.name.clone(),
            stream: pi.stream,
            ty: pi.ty,
            max_len,
            readers,
            reach,
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
    let mut nodes = build_nodes(registry, &plan_nodes, &mut streams, report)?;
    let mut memory: usize = plan_nodes.iter().map(|p| p.checked.memory_bytes).sum();
    let mut rt_streams = build_streams(&streams, &plan_inputs, &mut nodes, max_chunk, &mut memory)?;
    let mut outputs: Vec<OutRt> = Vec::with_capacity(plan_outputs.len());
    let mut output_index = BTreeMap::new();
    for (oi, po) in plan_outputs.iter().enumerate() {
        outputs.push(build_output(po, &streams, &mut rt_streams, &mut memory)?);
        output_index.insert(po.name.clone(), oi);
    }
    let inputs = build_inputs(
        (&plan_inputs, &plan_nodes, &plan_outputs),
        &streams,
        &nodes,
        max_chunk,
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
    })
}
