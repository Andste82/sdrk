//! Validation of a typed description and the plan that `build` executes.
//!
//! Stage 1 checks the structure of the model (versions, names, kinds, parameters, output
//! attributes). Stage 2 checks the graph (endpoints, inputs, cycles) and propagates types and
//! rates in topological order, calling the factory of every node. Stage 2 only runs when
//! stage 1 found no error, so that a single mistake does not produce a flood of follow-up
//! diagnostics.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

use sdrk::{Delay, Rate};
use sdrk_blocks::PortType;
use sdrk_device::SampleFormat;
use serde_json::{Map, Value};

use crate::latency::{
    Frac, PathNode, delay_seconds, frac_abs_diff, frac_add, frac_cmp, path_latency,
    seconds_to_samples,
};
use crate::model::{
    AlignMode, BackendSpec, InputType, MAIN_GROUP, MAX_CHUNK_LIMIT, MAX_DESCRIPTION_CHARS,
    OutputSpec, OutputType, PipelineSpec, PortName, Precision, is_valid_name,
};
use crate::params::check_params;
use crate::registry::{BlockFactory, BuildOptions, Checked, PortInfo, PortKind, Registry};
use crate::structure::join;
use crate::validate::{DiagCode, Diagnostic, GroupEdge, NodeReport, OutputReport, Report};

/// Largest `capacity` of a stream output in samples.
const MAX_CAPACITY: usize = 1 << 28;
/// Largest `queue` of a frame or packet output.
const MAX_QUEUE: usize = 1 << 16;
/// Slack added to every packet buffer.
pub(crate) const PACKET_SLACK: usize = 4;
/// Slack added to every buffer.
pub(crate) const BUFFER_SLACK: usize = 64;

/// The result of an analysis: always a report, and a plan if there are no errors.
pub(crate) struct Analysis {
    pub report: Report,
    pub plan: Option<Plan>,
}

/// Where a node writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlanDst {
    /// Into the stream with this index.
    Stream(usize),
    /// Frames into the frame queue of this output.
    Frames { output: usize },
}

/// A node in processing order.
pub(crate) struct PlanNode {
    pub name: String,
    pub kind: String,
    pub auto: bool,
    pub checked: Checked,
    /// The stream of port `a`.
    pub src: usize,
    /// The stream of port `b` of a two-input node.
    pub src_b: Option<usize>,
    pub dst: PlanDst,
    pub rate_in: Rate,
}

/// A linear buffer shared by the readers of one source.
#[derive(Debug, Clone)]
pub(crate) struct PlanStream {
    pub ty: PortType,
    pub rate: Rate,
    pub origin: usize,
    /// Elements per sample: bytes per sample for raw streams, 1 otherwise (`num`, `den`).
    pub elems: (u64, u64),
    /// Largest `min_input` of the readers.
    pub min_input: usize,
    /// Group delay from the pipeline input to this stream, in seconds (exact).
    pub delay_s: Frac,
}

/// An input of the pipeline.
pub(crate) struct PlanInput {
    pub name: String,
    pub stream: usize,
    pub ty: PortType,
    /// The sample rate; for a packet input the `rate_hz` of the encoder output (1 Hz as a
    /// placeholder when it has none).
    pub rate: Rate,
    /// False for a packet input without `rate_hz`.
    pub has_rate: bool,
}

/// What feeds an output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutSource {
    /// A stream (possibly the output of an automatic converter).
    Stream(usize),
    /// Frames of this length from a frame node.
    Frames { len: usize },
}

/// An output of the pipeline.
pub(crate) struct PlanOutput {
    pub name: String,
    pub spec: OutputSpec,
    pub source: OutSource,
    /// Capacity in elements (stream) or frames (frames).
    pub capacity: usize,
}

/// Everything `build` needs.
pub(crate) struct Plan {
    pub max_chunk: usize,
    pub inputs: Vec<PlanInput>,
    pub streams: Vec<PlanStream>,
    pub nodes: Vec<PlanNode>,
    pub outputs: Vec<PlanOutput>,
}

/// `ceil(a * b / c)` in `u128` with overflow checks.
fn ceil_mul_div(a: u128, b: u128, c: u128) -> Option<u128> {
    if c == 0 {
        return None;
    }
    a.checked_mul(b).map(|p| p.div_ceil(c))
}

impl PlanStream {
    /// Elements of this stream produced by `samples` samples at the pipeline input
    /// (rounded up).
    pub(crate) fn elements_for(&self, samples: usize, origin_rate: Rate) -> usize {
        // samples * (rate / origin_rate) * elems
        let big = u128::from;
        let num = big(self.rate.num())
            .checked_mul(big(origin_rate.den()))
            .and_then(|v| v.checked_mul(big(self.elems.0)));
        let den = big(self.rate.den())
            .checked_mul(big(origin_rate.num()))
            .and_then(|v| v.checked_mul(big(self.elems.1)));
        match (num, den) {
            (Some(n), Some(d)) => ceil_mul_div(samples as u128, n, d)
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(usize::MAX),
            _ => usize::MAX,
        }
    }

    /// Capacity of the buffer: `ceil(max_chunk * R) + min_input + 64`; a packet buffer holds
    /// `ceil(max_chunk * R) + 4` packets (`R` is `max_per_input` behind a decoder, 1 for the
    /// buffer of a packet input).
    pub(crate) fn capacity(&self, max_chunk: usize, origin_rate: Rate, min_input: usize) -> usize {
        if self.ty == PortType::Packets {
            return self
                .elements_for(max_chunk, origin_rate)
                .saturating_add(PACKET_SLACK);
        }
        self.elements_for(max_chunk, origin_rate)
            .saturating_add(min_input)
            .saturating_add(BUFFER_SLACK)
    }
}

/// Bytes of one element of a port type.
pub(crate) fn elem_size(t: PortType) -> usize {
    match t {
        PortType::C32 => 8,
        PortType::CQ15 | PortType::F32 => 4,
        PortType::Q15 => 2,
        PortType::Packets => core::mem::size_of::<sdrk_blocks::Packet>(),
        _ => 1,
    }
}

/// Samples and bytes of the smallest packing group of a streaming format.
pub(crate) fn packing_group(format: &SampleFormat) -> (usize, usize) {
    (1..=64usize)
        .find_map(|n| format.bytes_for_samples(n).ok().map(|b| (n, b)))
        .unwrap_or((1, 1))
}

fn prefixed(prefix: &str, d: Diagnostic) -> Diagnostic {
    let path = match (prefix.is_empty(), d.path.is_empty()) {
        (_, true) => prefix.to_string(),
        (true, false) => d.path,
        (false, false) => join(prefix, &d.path),
    };
    Diagnostic { path, ..d }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Input,
    Node,
    Output,
}

/// What a name produces during stage 2.
#[derive(Debug, Clone, Copy)]
enum Produced {
    Stream(usize),
    Frames { len: usize, node: usize },
    Failed,
}

fn input_port(t: InputType) -> PortType {
    match t {
        InputType::C32 => PortType::C32,
        InputType::Cq15 => PortType::CQ15,
        InputType::F32 => PortType::F32,
        InputType::Q15 => PortType::Q15,
        InputType::Packets => PortType::Packets,
    }
}

fn port_name(t: PortType) -> &'static str {
    match t {
        PortType::C32 => "c32",
        PortType::CQ15 => "cq15",
        PortType::F32 => "f32",
        PortType::Q15 => "q15",
        PortType::Raw => "raw bytes",
        PortType::Packets => "packets",
        _ => "frames",
    }
}

fn describe_types(types: &[PortType]) -> String {
    types
        .iter()
        .map(|t| port_name(*t))
        .collect::<Vec<_>>()
        .join(", ")
}

/// A rate for messages: an integer or `num/den`.
fn rate_text(rate: Rate) -> String {
    if rate.den() == 1 {
        rate.num().to_string()
    } else {
        format!("{}/{}", rate.num(), rate.den())
    }
}

/// A non-negative fraction for messages: an integer or `num/den`.
fn format_delay(value: Frac) -> String {
    if value.1 == 1 {
        value.0.to_string()
    } else {
        format!("{}/{}", value.0, value.1)
    }
}

/// Runs both stages.
pub(crate) fn analyze(registry: &Registry, spec: &PipelineSpec, opts: &BuildOptions) -> Analysis {
    let mut report = Report::default();
    stage1(registry, spec, &mut report);
    if !report.errors.is_empty() {
        return Analysis { report, plan: None };
    }
    let plan = Stage2::new(registry, spec, opts, &mut report).run();
    let plan = if report.errors.is_empty() { plan } else { None };
    Analysis { report, plan }
}

// ------------------------------------------------------------------ stage 1

fn check_description(description: Option<&String>, path: &str, report: &mut Report) {
    if let Some(d) = description
        && d.chars().count() > MAX_DESCRIPTION_CHARS
    {
        report.error(
            join(path, "description"),
            DiagCode::OutOfRange,
            format!("a description has at most {MAX_DESCRIPTION_CHARS} characters"),
        );
    }
}

fn stage1(registry: &Registry, spec: &PipelineSpec, report: &mut Report) {
    if spec.sdrk != crate::model::FORMAT_VERSION {
        report.error(
            "sdrk",
            DiagCode::SchemaVersion,
            format!(
                "unsupported format version {}; this library reads version {}",
                spec.sdrk,
                crate::model::FORMAT_VERSION
            ),
        );
    }
    check_description(spec.description.as_ref(), "", report);
    if !(1..=MAX_CHUNK_LIMIT).contains(&spec.settings.max_chunk) {
        report.error(
            "settings.max_chunk",
            DiagCode::OutOfRange,
            format!(
                "max_chunk is 1 to {MAX_CHUNK_LIMIT}, got {}",
                spec.settings.max_chunk
            ),
        );
    }
    check_names(registry, spec, report);
    check_groups(spec, report);
    check_inputs(registry, spec, report);
    check_nodes(registry, spec, report);
    check_outputs(spec, report);
}

/// The names of the automatic nodes with the path of the part that owns them: the converters of
/// the inputs (`<input>__convert`) and outputs (`<output>__convert_out`) and, for the nodes with
/// two inputs, the delay that aligns them (`<node>__align`).
fn automatic_names(registry: &Registry, spec: &PipelineSpec) -> Vec<(String, String)> {
    let inputs = spec
        .inputs
        .iter()
        .filter(|(_, i)| i.format.is_some())
        .map(|(n, _)| (format!("{n}__convert"), join("inputs", n)));
    let outputs = spec
        .outputs
        .iter()
        .filter(|(_, o)| o.format.is_some() && o.output_type == OutputType::Stream)
        .map(|(n, _)| (format!("{n}__convert_out"), join("outputs", n)));
    let nodes = spec
        .nodes
        .iter()
        .filter(|(_, n)| {
            registry
                .factory(&n.kind)
                .is_some_and(|f| f.inputs().len() > 1)
        })
        .map(|(n, _)| (format!("{n}__align"), join("nodes", n)));
    inputs.chain(outputs).chain(nodes).collect()
}

/// Name pattern, then uniqueness across the three sections and the automatic nodes.
fn check_names(registry: &Registry, spec: &PipelineSpec, report: &mut Report) {
    let mut seen: BTreeMap<&str, &'static str> = BTreeMap::new();
    let sections: [(&'static str, Vec<&String>); 3] = [
        ("inputs", spec.inputs.keys().collect()),
        ("nodes", spec.nodes.keys().collect()),
        ("outputs", spec.outputs.keys().collect()),
    ];
    for (section, names) in sections {
        for name in names {
            let path = join(section, name);
            if !is_valid_name(name) {
                report.error(
                    path,
                    DiagCode::InvalidName,
                    format!("`{name}` does not match {}", crate::model::NAME_PATTERN),
                );
            } else if let Some(first) = seen.insert(name.as_str(), section) {
                report.error(
                    path,
                    DiagCode::DuplicateName,
                    format!(
                        "`{name}` is already used in `{first}`; names are shared by inputs, nodes and outputs"
                    ),
                );
            }
        }
    }
    for (auto, owner) in automatic_names(registry, spec) {
        if let Some(section) = seen.get(auto.as_str()) {
            report.error(
                join(section, &auto),
                DiagCode::DuplicateName,
                format!("`{auto}` collides with the automatic node of `{owner}`"),
            );
        }
    }
}

/// Group names and the references of the nodes to them.
fn check_groups(spec: &PipelineSpec, report: &mut Report) {
    for (name, group) in &spec.groups {
        let path = join("groups", name);
        if !is_valid_name(name) {
            report.error(
                path.clone(),
                DiagCode::InvalidName,
                format!("`{name}` does not match {}", crate::model::NAME_PATTERN),
            );
        }
        check_description(group.description.as_ref(), &path, report);
    }
    for (name, node) in &spec.nodes {
        if let Some(group) = &node.group
            && group != MAIN_GROUP
            && !spec.groups.contains_key(group)
        {
            report.error(
                    join(&join("nodes", name), "group"),
                    DiagCode::UnknownGroup,
                    format!(
                        "the group `{group}` is not declared in `groups` (nodes without a group are in `{MAIN_GROUP}`)"
                    ),
                );
        }
    }
}

/// How the nodes read a packet input: `(port a, port b)` readers. Port `a` of a node whose
/// first input takes packets is an encoder of kind `Source`; port `b` is the packet port of a
/// clocked encoder.
fn packet_readers(registry: &Registry, spec: &PipelineSpec, input: &str) -> (bool, bool) {
    let mut readers = (false, false);
    for edge in spec.edges.iter().filter(|e| e.from == input) {
        let Some(factory) = spec
            .nodes
            .get(&edge.to)
            .and_then(|n| registry.factory(&n.kind))
        else {
            continue;
        };
        let port = edge.port_or_default();
        let takes_packets = factory
            .description()
            .inputs
            .get(port.index())
            .is_some_and(|p| p.types.contains(&PortType::Packets));
        match (port, takes_packets) {
            (PortName::A, true) => readers.0 = true,
            (PortName::B, true) => readers.1 = true,
            _ => {}
        }
    }
    readers
}

fn check_inputs(registry: &Registry, spec: &PipelineSpec, report: &mut Report) {
    for (name, input) in &spec.inputs {
        let path = join("inputs", name);
        check_description(input.description.as_ref(), &path, report);
        if input.sample_type == Some(InputType::Packets) {
            let (port_a, port_b) = packet_readers(registry, spec, name);
            if port_a && input.rate_hz.is_none() {
                report.error(
                    join(&path, "rate_hz"),
                    DiagCode::MissingParam,
                    format!(
                        "an encoder reads the packet input `{name}` and needs its `rate_hz`: the sample rate of the stream it produces"
                    ),
                );
            } else if !port_a && port_b && input.rate_hz.is_some() {
                report.warning(
                    join(&path, "rate_hz"),
                    DiagCode::InvalidValue,
                    format!(
                        "only port `b` of clocked encoders reads the packet input `{name}`; they take their rate from their clock, so `rate_hz` is ignored"
                    ),
                );
            }
        } else if input.rate_hz.is_none() {
            report.error(
                join(&path, "rate_hz"),
                DiagCode::MissingParam,
                "the field `rate_hz` is required",
            );
        }
        match (&input.format, &input.sample_type) {
            (Some(_), Some(_)) | (None, None) => report.error(
                &path,
                DiagCode::InvalidValue,
                "exactly one of `format` (raw bytes) and `type` (converted samples) must be given",
            ),
            (Some(f), None) => {
                if let Err(m) = f.to_format() {
                    report.error(join(&path, "format"), DiagCode::InvalidValue, m);
                }
            }
            (None, Some(_)) => {}
        }
    }
}

fn check_nodes(registry: &Registry, spec: &PipelineSpec, report: &mut Report) {
    for (name, node) in &spec.nodes {
        let path = join("nodes", name);
        check_description(node.description.as_ref(), &path, report);
        match registry.factory(&node.kind) {
            None => report.error(
                join(&path, "kind"),
                DiagCode::UnknownKind,
                format!(
                    "unknown kind `{}`; the registered kinds are: {}",
                    node.kind,
                    registry
                        .descriptions()
                        .map(|d| d.kind)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ),
            Some(factory) => {
                let mut diags = Vec::new();
                check_params(
                    factory.description(),
                    node.params.iter(),
                    &join(&path, "params"),
                    &mut diags,
                );
                report.errors.extend(diags);
            }
        }
    }
}

fn check_outputs(spec: &PipelineSpec, report: &mut Report) {
    for (name, output) in &spec.outputs {
        let path = join("outputs", name);
        check_description(output.description.as_ref(), &path, report);
        if let Some(f) = &output.format {
            if output.output_type != OutputType::Stream {
                report.error(
                    join(&path, "format"),
                    DiagCode::InvalidValue,
                    "only a stream output has a format",
                );
            } else if let Err(m) = f.to_format() {
                report.error(join(&path, "format"), DiagCode::InvalidValue, m);
            }
        }
        match output.output_type {
            OutputType::Stream => {
                if output.queue.is_some() {
                    report.error(
                        join(&path, "queue"),
                        DiagCode::InvalidValue,
                        "`queue` belongs to frames outputs; a stream output has a `capacity`",
                    );
                }
                check_range(output.capacity, MAX_CAPACITY, "capacity", &path, report);
            }
            OutputType::Frames | OutputType::Packets => {
                if output.capacity.is_some() {
                    report.error(
                        join(&path, "capacity"),
                        DiagCode::InvalidValue,
                        "`capacity` belongs to stream outputs; a frames or packets output has a `queue`",
                    );
                }
                check_range(output.queue, MAX_QUEUE, "queue", &path, report);
            }
        }
    }
}

fn check_range(value: Option<usize>, max: usize, key: &str, path: &str, report: &mut Report) {
    if let Some(v) = value
        && !(1..=max).contains(&v)
    {
        report.error(
            join(path, key),
            DiagCode::OutOfRange,
            format!("{key} is 1 to {max}, got {v}"),
        );
    }
}

// ------------------------------------------------------------------ groups, paths, latency

/// The group of a node (`main` without one).
pub(crate) fn group_of<'a>(spec: &'a PipelineSpec, node: &str) -> &'a str {
    spec.nodes
        .get(node)
        .and_then(|n| n.group.as_deref())
        .unwrap_or(MAIN_GROUP)
}

/// Edges between nodes of different groups, in the order of `edges`.
pub(crate) fn group_edges(spec: &PipelineSpec) -> Vec<GroupEdge> {
    spec.edges
        .iter()
        .filter(|e| spec.nodes.contains_key(&e.from) && spec.nodes.contains_key(&e.to))
        .filter_map(|e| {
            let (a, b) = (group_of(spec, &e.from), group_of(spec, &e.to));
            (a != b).then(|| GroupEdge {
                from: e.from.clone(),
                to: e.to.clone(),
                from_group: a.to_string(),
                to_group: b.to_string(),
            })
        })
        .collect()
}

/// The way from a pipeline input to an output: the input and the nodes in processing order.
pub(crate) struct OutPath {
    pub input: usize,
    pub nodes: Vec<usize>,
}

/// The node that writes the stream `sid`.
pub(crate) fn producer_of(nodes: &[PlanNode], sid: usize) -> Option<usize> {
    nodes.iter().position(|n| n.dst == PlanDst::Stream(sid))
}

/// The nodes from the pipeline input to the stream `sid` in processing order, following port
/// `a` of every node.
pub(crate) fn chain_to(nodes: &[PlanNode], sid: usize) -> Vec<usize> {
    let mut chain = Vec::new();
    let mut cur = Some(sid);
    while let Some(s) = cur {
        match producer_of(nodes, s) {
            Some(k) => {
                chain.push(k);
                cur = nodes.get(k).map(|n| n.src);
            }
            None => break,
        }
    }
    chain.reverse();
    chain
}

/// Adds every node that the stream `sid` depends on (both ports of the two-input nodes).
fn collect_tree(nodes: &[PlanNode], sid: usize, tree: &mut BTreeSet<usize>) {
    if let Some(k) = producer_of(nodes, sid)
        && tree.insert(k)
        && let Some(n) = nodes.get(k)
    {
        collect_tree(nodes, n.src, tree);
        if let Some(sb) = n.src_b {
            collect_tree(nodes, sb, tree);
        }
    }
}

/// Every node that output `oi` depends on, in processing order.
pub(crate) fn output_tree(nodes: &[PlanNode], out: &PlanOutput, oi: usize) -> Vec<usize> {
    let mut tree = BTreeSet::new();
    match out.source {
        OutSource::Stream(sid) => collect_tree(nodes, sid, &mut tree),
        OutSource::Frames { .. } => {
            if let Some(k) = nodes
                .iter()
                .position(|n| n.dst == PlanDst::Frames { output: oi })
            {
                tree.insert(k);
                if let Some(n) = nodes.get(k) {
                    collect_tree(nodes, n.src, &mut tree);
                    if let Some(sb) = n.src_b {
                        collect_tree(nodes, sb, &mut tree);
                    }
                }
            }
        }
    }
    tree.into_iter().collect()
}

/// The path of every output.
pub(crate) fn output_paths(
    nodes: &[PlanNode],
    streams: &[PlanStream],
    outputs: &[PlanOutput],
) -> Vec<OutPath> {
    let mut paths = Vec::with_capacity(outputs.len());
    for (oi, out) in outputs.iter().enumerate() {
        let mut chain: Vec<usize> = Vec::new();
        let mut cur = match out.source {
            OutSource::Stream(sid) => Some(sid),
            OutSource::Frames { .. } => {
                let k = nodes
                    .iter()
                    .position(|n| n.dst == PlanDst::Frames { output: oi });
                if let Some(k) = k {
                    chain.push(k);
                }
                k.and_then(|k| nodes.get(k)).map(|n| n.src)
            }
        };
        while let Some(sid) = cur {
            match nodes.iter().position(|n| n.dst == PlanDst::Stream(sid)) {
                Some(k) => {
                    chain.push(k);
                    cur = nodes.get(k).map(|n| n.src);
                }
                None => break,
            }
        }
        chain.reverse();
        let input = chain
            .first()
            .and_then(|k| nodes.get(*k))
            .and_then(|n| streams.get(n.src))
            .map_or(0, |s| s.origin);
        paths.push(OutPath {
            input,
            nodes: chain,
        });
    }
    paths
}

/// The rate of the samples an output is counted in: its stream, or the input of its frame node.
pub(crate) fn output_unit_rate(
    out: &PlanOutput,
    path: &OutPath,
    nodes: &[PlanNode],
    streams: &[PlanStream],
) -> Option<Rate> {
    match out.source {
        OutSource::Stream(sid) => streams.get(sid).map(|s| s.rate),
        OutSource::Frames { .. } => path
            .nodes
            .last()
            .and_then(|k| nodes.get(*k))
            .map(|n| n.rate_in),
    }
}

/// Group delay and buffering of every output from the figures of the plan.
fn latency_report(
    paths: &[OutPath],
    nodes: &[PlanNode],
    streams: &[PlanStream],
    outputs: &[PlanOutput],
) -> Vec<OutputReport> {
    let mut reports = Vec::with_capacity(outputs.len());
    for (out, path) in outputs.iter().zip(paths) {
        let stream_nodes: Vec<PathNode> = path
            .nodes
            .iter()
            .filter_map(|k| nodes.get(*k))
            .filter(|n| matches!(n.dst, PlanDst::Stream(_)))
            .map(|n| PathNode {
                rate_in: n.rate_in,
                delay: n.checked.delay,
                min_input: n.checked.min_input,
            })
            .collect();
        let (group_delay, max_buffered) = output_unit_rate(out, path, nodes, streams)
            .and_then(|unit| path_latency(&stream_nodes, unit))
            .unwrap_or((Delay::ZERO, 0));
        reports.push(OutputReport {
            name: out.name.clone(),
            group_delay,
            max_buffered,
        });
    }
    reports
}

/// One node of each cycle among the `leftover` nodes (those that Kahn's algorithm could not
/// order): the first by `prio`, then by name. Walks the feeders depth first; a node behind a
/// cycle finds the cycle in front of it.
fn cycle_nodes<'a>(
    parents: &BTreeMap<&'a str, Vec<&'a str>>,
    leftover: &BTreeSet<&'a str>,
    prio: &dyn Fn(&str) -> usize,
) -> Vec<&'a str> {
    let mut state: BTreeMap<&str, u8> = BTreeMap::new();
    let mut found: Vec<&str> = Vec::new();
    for &start in leftover {
        if state.contains_key(start) {
            continue;
        }
        state.insert(start, 1);
        let mut stack: Vec<(&str, usize)> = vec![(start, 0)];
        while let Some(&(n, i)) = stack.last() {
            let feeder = parents.get(n).and_then(|f| f.get(i)).copied();
            let Some(feeder) = feeder else {
                state.insert(n, 2);
                stack.pop();
                continue;
            };
            if let Some(top) = stack.last_mut() {
                top.1 += 1;
            }
            if !leftover.contains(feeder) {
                continue;
            }
            match state.get(feeder) {
                None => {
                    state.insert(feeder, 1);
                    stack.push((feeder, 0));
                }
                Some(1) => {
                    // A cycle: the walk from `feeder` onwards.
                    if let Some(pos) = stack.iter().position(|(w, _)| *w == feeder)
                        && let Some(best) = stack
                            .iter()
                            .skip(pos)
                            .map(|(w, _)| *w)
                            .min_by_key(|w| (prio(w), *w))
                    {
                        found.push(best);
                    }
                }
                Some(_) => {}
            }
        }
    }
    found.sort_unstable();
    found.dedup();
    found
}

// ------------------------------------------------------------------ stage 2

struct Stage2<'a> {
    registry: &'a Registry,
    spec: &'a PipelineSpec,
    opts: &'a BuildOptions,
    report: &'a mut Report,
    streams: Vec<PlanStream>,
    nodes: Vec<PlanNode>,
    inputs: Vec<PlanInput>,
    outputs: Vec<PlanOutput>,
    /// Index of the edge that feeds each input port of each node, and each output (port `a`).
    incoming: BTreeMap<(&'a str, PortName), usize>,
    produced: BTreeMap<String, Produced>,
}

impl<'a> Stage2<'a> {
    fn new(
        registry: &'a Registry,
        spec: &'a PipelineSpec,
        opts: &'a BuildOptions,
        report: &'a mut Report,
    ) -> Self {
        Self {
            registry,
            spec,
            opts,
            report,
            streams: Vec::new(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            incoming: BTreeMap::new(),
            produced: BTreeMap::new(),
        }
    }

    fn role(&self, name: &str) -> Option<Role> {
        if self.spec.inputs.contains_key(name) {
            Some(Role::Input)
        } else if self.spec.nodes.contains_key(name) {
            Some(Role::Node)
        } else if self.spec.outputs.contains_key(name) {
            Some(Role::Output)
        } else {
            None
        }
    }

    fn run(mut self) -> Option<Plan> {
        self.graph();
        if !self.report.errors.is_empty() {
            return None;
        }
        let order = self.topological_order();
        if !self.report.errors.is_empty() {
            return None;
        }
        self.sources();
        for name in &order {
            self.node(name);
        }
        self.attach_outputs();
        if !self.report.errors.is_empty() {
            return None;
        }
        Some(self.finish())
    }

    /// Reports an edge end that cannot be used on that side; true if it can.
    fn endpoint_ok(&mut self, k: usize, side: &'static str, name: &str) -> bool {
        let why = match (side, self.role(name)) {
            ("from", Some(Role::Input | Role::Node)) | ("to", Some(Role::Node | Role::Output)) => {
                return true;
            }
            ("from", Some(Role::Output)) => "is an output and cannot be the source of an edge",
            ("to", Some(Role::Input)) => "is an input and cannot be the destination of an edge",
            ("from", _) => "is neither an input nor a node",
            _ => "is neither a node nor an output",
        };
        self.report.error(
            format!("edges[{k}].{side}"),
            DiagCode::UnknownEndpoint,
            format!("`{name}` {why}"),
        );
        false
    }

    /// Endpoints, incoming edges, dangling outputs.
    fn graph(&mut self) {
        let spec = self.spec;
        let mut outgoing: BTreeSet<&str> = BTreeSet::new();
        let mut extra: Vec<(usize, &str, PortName)> = Vec::new();
        for (k, edge) in spec.edges.iter().enumerate() {
            let from_ok = self.endpoint_ok(k, "from", &edge.from);
            let to_ok = self.endpoint_ok(k, "to", &edge.to);
            if from_ok {
                outgoing.insert(edge.from.as_str());
            }
            if from_ok && to_ok {
                let port = self.edge_port(k, edge);
                match self.incoming.entry((edge.to.as_str(), port)) {
                    Entry::Occupied(_) => extra.push((k, edge.to.as_str(), port)),
                    Entry::Vacant(slot) => {
                        slot.insert(k);
                    }
                }
            }
        }
        for (k, to, port) in extra {
            // Port `b` has its own path; `a` and the outputs keep `edges[k].to`.
            let path = match port {
                PortName::A => format!("edges[{k}].to"),
                PortName::B => format!("edges[{k}].port"),
            };
            self.report.error(
                path,
                DiagCode::MultipleInputs,
                format!(
                    "the port `{}` of `{to}` already has an incoming edge; an input port of a node or an output has exactly one",
                    port.as_str()
                ),
            );
        }
        self.unconnected(&outgoing);
    }

    /// The port an edge feeds. An edge to a port that the destination does not have is reported
    /// (`unknown_port`) and counts for port `a` so that it does not also show up as a missing
    /// input.
    fn edge_port(&mut self, k: usize, edge: &'a crate::model::EdgeSpec) -> PortName {
        let spec = self.spec;
        let port = edge.port_or_default();
        let ports = spec
            .nodes
            .get(&edge.to)
            .and_then(|n| self.registry.factory(&n.kind))
            .map(|f| f.inputs().len());
        let bad = match (edge.port, ports) {
            (Some(_), None) => Some(format!(
                "`{}` is an output and has no input ports; leave `port` out",
                edge.to
            )),
            (Some(_), Some(count)) if port.index() >= count => Some(format!(
                "`{}` has no input port `{}`; its kind has {count} input{}",
                edge.to,
                port.as_str(),
                if count == 1 { "" } else { "s" }
            )),
            _ => None,
        };
        match bad {
            Some(message) => {
                self.report
                    .error(format!("edges[{k}].port"), DiagCode::UnknownPort, message);
                PortName::A
            }
            None => port,
        }
    }

    /// Ports of a node (or the one port of an output) without an incoming edge, nodes and
    /// inputs that go nowhere.
    fn unconnected(&mut self, outgoing: &BTreeSet<&str>) {
        let spec = self.spec;
        for (name, node) in &spec.nodes {
            let count = self
                .registry
                .factory(&node.kind)
                .map_or(1, |f| f.inputs().len());
            for port in [PortName::A, PortName::B].into_iter().take(count) {
                if !self.incoming.contains_key(&(name.as_str(), port)) {
                    let (path, message) = match port {
                        PortName::A => (
                            join("nodes", name),
                            format!("`{name}` has no incoming edge"),
                        ),
                        PortName::B => (
                            join(&join("nodes", name), "ports.b"),
                            format!("the port `b` of `{name}` has no incoming edge"),
                        ),
                    };
                    self.report.error(path, DiagCode::MissingInput, message);
                }
            }
        }
        for name in spec.outputs.keys() {
            if !self.incoming.contains_key(&(name.as_str(), PortName::A)) {
                self.report.error(
                    join("outputs", name),
                    DiagCode::MissingInput,
                    format!("`{name}` has no incoming edge"),
                );
            }
        }
        for name in spec.nodes.keys() {
            if !outgoing.contains(name.as_str()) {
                self.report.error(
                    join("nodes", name),
                    DiagCode::DanglingOutput,
                    format!("the output of `{name}` is not connected to a node or an output"),
                );
            }
        }
        for name in spec.inputs.keys() {
            if !outgoing.contains(name.as_str()) {
                self.report.warning(
                    join("inputs", name),
                    DiagCode::DanglingOutput,
                    format!("the input `{name}` is not connected; its data is ignored"),
                );
            }
        }
    }

    /// Kahn's algorithm; ties are broken by the first appearance in `edges`. Reports one
    /// node of each cycle.
    fn topological_order(&mut self) -> Vec<String> {
        let spec = self.spec;
        let mut first_seen: BTreeMap<&str, usize> = BTreeMap::new();
        for edge in &spec.edges {
            let next = first_seen.len();
            first_seen.entry(edge.from.as_str()).or_insert(next);
            let next = first_seen.len();
            first_seen.entry(edge.to.as_str()).or_insert(next);
        }
        let prio = |n: &str| first_seen.get(n).copied().unwrap_or(usize::MAX);
        // parents[node] = the nodes feeding it (one per port, in the order of the ports)
        let mut parents: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        let mut children: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for name in spec.nodes.keys() {
            for port in [PortName::A, PortName::B] {
                if let Some(&k) = self.incoming.get(&(name.as_str(), port))
                    && let Some(edge) = spec.edges.get(k)
                    && spec.nodes.contains_key(&edge.from)
                {
                    parents
                        .entry(name.as_str())
                        .or_default()
                        .push(edge.from.as_str());
                    children
                        .entry(edge.from.as_str())
                        .or_default()
                        .push(name.as_str());
                }
            }
        }
        // A node is ready when every edge from a node into it has been passed.
        let mut waiting: BTreeMap<&str, usize> = parents
            .iter()
            .map(|(n, feeders)| (*n, feeders.len()))
            .collect();
        let mut ready: BTreeSet<(usize, &str)> = spec
            .nodes
            .keys()
            .filter(|n| !waiting.contains_key(n.as_str()))
            .map(|n| (prio(n), n.as_str()))
            .collect();
        let mut order = Vec::with_capacity(spec.nodes.len());
        let mut done: BTreeSet<&str> = BTreeSet::new();
        while let Some(&(p, n)) = ready.iter().next() {
            ready.remove(&(p, n));
            order.push(n.to_string());
            done.insert(n);
            if let Some(kids) = children.get(n) {
                for kid in kids {
                    if let Some(left) = waiting.get_mut(kid) {
                        *left = left.saturating_sub(1);
                        if *left == 0 {
                            ready.insert((prio(kid), kid));
                        }
                    }
                }
            }
        }
        if order.len() < spec.nodes.len() {
            let leftover: BTreeSet<&str> = spec
                .nodes
                .keys()
                .map(String::as_str)
                .filter(|n| !done.contains(n))
                .collect();
            for n in cycle_nodes(&parents, &leftover, &prio) {
                self.report.error(
                    join("nodes", n),
                    DiagCode::Cycle,
                    format!("`{n}` is part of a cycle; the graph must be acyclic"),
                );
            }
        }
        order
    }

    fn new_stream(&mut self, ty: PortType, rate: Rate, origin: usize, elems: (u64, u64)) -> usize {
        self.streams.push(PlanStream {
            ty,
            rate,
            origin,
            elems,
            min_input: 0,
            delay_s: (0, 1),
        });
        self.streams.len() - 1
    }

    fn params_map(params: &BTreeMap<String, Value>) -> Map<String, Value> {
        params.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    /// Calls the factory and records the node; `None` after reporting the diagnostics.
    #[allow(clippy::too_many_arguments)]
    fn check_node(
        &mut self,
        name: &str,
        kind: &str,
        auto: bool,
        params: &Map<String, Value>,
        src: usize,
        path: &str,
        remap: &dyn Fn(&str) -> String,
    ) -> Option<usize> {
        let factory: &dyn BlockFactory = self.registry.factory(kind)?;
        let (ty, rate_in, origin) = {
            let s = self.streams.get(src)?;
            (s.ty, s.rate, s.origin)
        };
        let precision = self.spec.settings.precision;
        match factory.check(params, ty, rate_in, precision) {
            Ok(checked) => {
                let min_input = checked.min_input;
                if let Some(s) = self.streams.get_mut(src) {
                    s.min_input = s.min_input.max(min_input);
                }
                let _ = origin;
                self.nodes.push(PlanNode {
                    name: name.to_string(),
                    kind: kind.to_string(),
                    auto,
                    checked,
                    src,
                    src_b: None,
                    dst: PlanDst::Stream(usize::MAX),
                    rate_in,
                });
                Some(self.nodes.len() - 1)
            }
            Err(diags) => {
                for d in diags {
                    let relative = remap(&d.path);
                    let d = Diagnostic {
                        path: relative,
                        ..d
                    };
                    self.report.errors.push(if path.is_empty() {
                        d
                    } else {
                        prefixed(path, d)
                    });
                }
                None
            }
        }
    }

    /// Sets the destination of node `idx` from its `Checked` output; returns what it produces.
    fn finish_node(&mut self, idx: usize) -> Option<Produced> {
        let (output, rate_out, rate_in, src) = {
            let n = self.nodes.get(idx)?;
            (n.checked.output, n.checked.rate_out, n.rate_in, n.src)
        };
        let origin = self.streams.get(src)?.origin;
        // The group delay from the input: that of the source (port `a`) plus the node's own.
        let before = self.streams.get(src)?.delay_s;
        let own = self
            .nodes
            .get(idx)
            .and_then(|n| delay_seconds(n.checked.delay, rate_in));
        let delay_s = own.and_then(|d| frac_add(before, d)).unwrap_or(before);
        match output {
            PortKind::Stream(ty) => {
                let elems = if ty == PortType::Raw {
                    self.raw_elems(idx)
                } else {
                    (1, 1)
                };
                let s = self.new_stream(ty, rate_out, origin, elems);
                if let Some(stream) = self.streams.get_mut(s) {
                    stream.delay_s = delay_s;
                }
                self.nodes.get_mut(idx)?.dst = PlanDst::Stream(s);
                Some(Produced::Stream(s))
            }
            PortKind::Frames { len } => Some(Produced::Frames { len, node: idx }),
        }
    }

    /// Bytes per sample of the raw output of a `convert_out` node.
    fn raw_elems(&self, idx: usize) -> (u64, u64) {
        let params = self.nodes.get(idx).and_then(|n| {
            if n.auto {
                self.spec
                    .outputs
                    .iter()
                    .find(|(o, _)| format!("{o}__convert_out") == n.name)
                    .and_then(|(_, o)| o.format.as_ref())
                    .and_then(|f| f.to_format().ok())
            } else {
                self.spec
                    .nodes
                    .get(&n.name)
                    .and_then(|node| node.params.get("format"))
                    .and_then(|v| match v {
                        Value::String(s) => crate::model::FormatName::from_name(s)
                            .map(crate::model::FormatName::format),
                        other => serde_json::from_value::<crate::model::FormatSpec>(other.clone())
                            .ok()
                            .and_then(|f| f.to_format().ok()),
                    })
            }
        });
        match params {
            Some(f) => {
                let (samples, bytes) = packing_group(&f);
                (bytes as u64, samples as u64)
            }
            None => (1, 1),
        }
    }

    /// Streams of the inputs, with the automatic `convert` nodes.
    fn sources(&mut self) {
        let spec = self.spec;
        let precision = spec.settings.precision;
        for (name, input) in &spec.inputs {
            // Stage 1 reported a sample input without a rate; a packet input without one has
            // a placeholder (only port `b` of clocked encoders reads it, or nothing).
            let Some(rate) = input.rate_hz.map(|r| r.0).or_else(|| Rate::hz(1).ok()) else {
                self.produced.insert(name.clone(), Produced::Failed);
                continue;
            };
            let path = join("inputs", name);
            let input_index = self.inputs.len();
            if let Some(format) = &input.format {
                let Ok(f) = format.to_format() else {
                    self.produced.insert(name.clone(), Produced::Failed);
                    continue;
                };
                let (samples, bytes) = packing_group(&f);
                let raw = self.new_stream(
                    PortType::Raw,
                    rate,
                    input_index,
                    (bytes as u64, samples as u64),
                );
                self.inputs.push(PlanInput {
                    name: name.clone(),
                    stream: raw,
                    ty: PortType::Raw,
                    rate,
                    has_rate: true,
                });
                let mut params = Map::new();
                if let Ok(v) = serde_json::to_value(format) {
                    params.insert("format".to_string(), v);
                }
                let auto_name = format!("{name}__convert");
                let input_path = path.clone();
                let remap = move |p: &str| {
                    if p == "params.format" {
                        join(&input_path, "format")
                    } else {
                        input_path.clone()
                    }
                };
                let idx = self.check_node(&auto_name, "convert", true, &params, raw, "", &remap);
                match idx.and_then(|i| self.finish_node(i)) {
                    Some(p) => {
                        self.produced.insert(name.clone(), p);
                    }
                    None => {
                        self.produced.insert(name.clone(), Produced::Failed);
                    }
                }
            } else if input.sample_type == Some(InputType::Packets) {
                // A packet input is its own clock domain (`origin`); its stream holds packets.
                let s = self.new_stream(PortType::Packets, rate, input_index, (1, 1));
                self.inputs.push(PlanInput {
                    name: name.clone(),
                    stream: s,
                    ty: PortType::Packets,
                    rate,
                    has_rate: input.rate_hz.is_some(),
                });
                self.produced.insert(name.clone(), Produced::Stream(s));
            } else if let Some(t) = input.sample_type {
                let ty = input_port(t);
                let wanted = match precision {
                    Precision::F32 => [PortType::C32, PortType::F32],
                    Precision::Q15 => [PortType::CQ15, PortType::Q15],
                };
                if !wanted.contains(&ty) {
                    self.report.error(
                        join(&path, "type"),
                        DiagCode::TypeMismatch,
                        format!(
                            "the input type {} does not fit the pipeline precision {:?}; use {}",
                            port_name(ty),
                            precision,
                            describe_types(&wanted)
                        ),
                    );
                    self.produced.insert(name.clone(), Produced::Failed);
                    continue;
                }
                let s = self.new_stream(ty, rate, input_index, (1, 1));
                self.inputs.push(PlanInput {
                    name: name.clone(),
                    stream: s,
                    ty,
                    rate,
                    has_rate: true,
                });
                self.produced.insert(name.clone(), Produced::Stream(s));
            } else {
                self.produced.insert(name.clone(), Produced::Failed);
            }
        }
    }

    /// The stream that arrives at a port of a node, with the checks that belong to the edge:
    /// the producer must not have failed or produce frames, and the type must be accepted.
    /// `None` after reporting (or when a failure was reported earlier).
    fn port_stream(
        &mut self,
        node: &str,
        kind: &str,
        k: usize,
        accepted: &[PortType],
    ) -> Option<usize> {
        let spec = self.spec;
        let edge = spec.edges.get(k)?;
        match self
            .produced
            .get(&edge.from)
            .copied()
            .unwrap_or(Produced::Failed)
        {
            Produced::Failed => None,
            Produced::Frames { .. } => {
                self.report.error(
                    format!("edges[{k}].to"),
                    DiagCode::TypeMismatch,
                    format!(
                        "`{}` produces frames, which only an output can take; `{node}` needs a stream",
                        edge.from
                    ),
                );
                None
            }
            Produced::Stream(s) => {
                let ty = self.streams.get(s).map_or(PortType::Raw, |x| x.ty);
                if accepted.contains(&ty) {
                    Some(s)
                } else {
                    self.report.error(
                        format!("edges[{k}].to"),
                        DiagCode::TypeMismatch,
                        format!(
                            "`{node}` ({kind}) takes {}, but `{}` delivers {}",
                            describe_types(accepted),
                            edge.from,
                            port_name(ty)
                        ),
                    );
                    None
                }
            }
        }
    }

    /// The rate, clock and type checks of the two inputs of a node; true if they pass. The
    /// diagnostics sit at the `to` of the edge of port `b`.
    fn pair_ok(&mut self, name: &str, kb: usize, a: usize, b: usize) -> bool {
        let (Some(sa), Some(sb)) = (self.streams.get(a), self.streams.get(b)) else {
            return false;
        };
        let (ta, tb, ra, rb, oa, ob) = (sa.ty, sb.ty, sa.rate, sb.rate, sa.origin, sb.origin);
        let path = format!("edges[{kb}].to");
        let mut ok = true;
        if ta != tb {
            self.report.error(
                path.clone(),
                DiagCode::TypeMismatch,
                format!(
                    "the ports of `{name}` take the same type, but port `a` has {} and port `b` has {}",
                    port_name(ta),
                    port_name(tb)
                ),
            );
            ok = false;
        }
        if ra != rb {
            self.report.error(
                path.clone(),
                DiagCode::RateMismatch,
                format!(
                    "the ports of `{name}` need the same sample rate, but port `a` has {} Hz and port `b` has {} Hz; resample or decimate one path",
                    rate_text(ra),
                    rate_text(rb)
                ),
            );
            ok = false;
        }
        if oa != ob {
            let (na, nb) = (self.input_name(oa), self.input_name(ob));
            self.report.error(
                path,
                DiagCode::ClockDomain,
                format!(
                    "the inputs `{na}` and `{nb}` of `{name}` are different clocks; resample one with a tunable `resample` node driven by a drift controller, or use one input"
                ),
            );
            ok = false;
        }
        ok
    }

    fn input_name(&self, origin: usize) -> String {
        self.inputs
            .get(origin)
            .map_or_else(String::new, |i| i.name.clone())
    }

    /// Lines up the two paths of a node with a `delay` node in front of the faster one.
    /// Returns the sources of the ports after the alignment, `None` after an error.
    fn align(&mut self, name: &str, mode: AlignMode, a: usize, b: usize) -> Option<(usize, usize)> {
        let (Some(sa), Some(sb)) = (self.streams.get(a), self.streams.get(b)) else {
            return None;
        };
        let (da, db, rate) = (sa.delay_s, sb.delay_s, sa.rate);
        let path = join("nodes", name);
        let overflow = || "the group delays are too large to compare".to_string();
        let Some(order) = frac_cmp(da, db) else {
            self.report
                .error(&path, DiagCode::MisalignedInputs, overflow());
            return None;
        };
        if order == core::cmp::Ordering::Equal {
            return Some((a, b));
        }
        let Some(samples) = frac_abs_diff(da, db).and_then(|d| seconds_to_samples(d, rate)) else {
            self.report
                .error(&path, DiagCode::MisalignedInputs, overflow());
            return None;
        };
        let text = format_delay(samples);
        let (first, second) = (
            format_delay(seconds_to_samples(da, rate).unwrap_or((0, 1))),
            format_delay(seconds_to_samples(db, rate).unwrap_or((0, 1))),
        );
        let whole = samples.1 == 1;
        if mode == AlignMode::None {
            self.report.warning(
                &path,
                DiagCode::MisalignedInputs,
                format!(
                    "the paths into `{name}` differ by {text} samples (port `a` {first}, port `b` {second}); `align` is `none`, so nothing is inserted"
                ),
            );
            return Some((a, b));
        }
        if !whole {
            self.report.error(
                &path,
                DiagCode::MisalignedInputs,
                format!(
                    "the group delays of the paths into `{name}` are {first} samples (port `a`) and {second} samples (port `b`); they differ by {text} samples, which is not a whole number. Change a filter so that the delays fit"
                ),
            );
            return None;
        }
        let k = usize::try_from(samples.0).unwrap_or(usize::MAX);
        let config = sdrk_blocks::DelayConfig { samples: k };
        if config.validate(rate).is_err() {
            self.report.error(
                &path,
                DiagCode::MisalignedInputs,
                format!(
                    "the paths into `{name}` differ by {text} samples; a delay that long cannot be inserted automatically"
                ),
            );
            return None;
        }
        // The path with the smaller delay is the faster one and gets the delay.
        let a_is_faster = order == core::cmp::Ordering::Less;
        let slow_src = if a_is_faster { a } else { b };
        let auto_name = format!("{name}__align");
        let mut params = Map::new();
        params.insert("samples".to_string(), Value::from(k));
        let node_path = path.clone();
        let remap = move |_: &str| node_path.clone();
        let idx = self.check_node(&auto_name, "delay", true, &params, slow_src, "", &remap)?;
        let Some(Produced::Stream(aligned)) = self.finish_node(idx) else {
            return None;
        };
        Some(if a_is_faster {
            (aligned, b)
        } else {
            (a, aligned)
        })
    }

    /// An encoder of kind `Source` (packets at port `a`) takes its output rate from the
    /// `rate_hz` of a packet input and starts that input's clock domain (D526). A packet stream
    /// of a decoder has a nominal rate for buffer sizing only (D513) and the clock of the
    /// decoder's input, so it cannot feed such an encoder: `type_mismatch` at the edge.
    fn source_packets_ok(
        &mut self,
        name: &str,
        ports: &[PortInfo],
        first: Option<(usize, usize)>,
    ) -> bool {
        let Some((k, s)) = first else {
            return true;
        };
        let takes_packets = ports
            .first()
            .is_some_and(|p| p.accepts.contains(&PortType::Packets));
        let is_packets = self
            .streams
            .get(s)
            .is_some_and(|x| x.ty == PortType::Packets);
        if !takes_packets || !is_packets || self.inputs.iter().any(|i| i.stream == s) {
            return true;
        }
        let from = self.spec.edges.get(k).map_or("", |e| e.from.as_str());
        self.report.error(
            format!("edges[{k}].to"),
            DiagCode::TypeMismatch,
            format!(
                "`{name}` encodes packets at the `rate_hz` of a packet input, but `{from}` is not a pipeline input; its packets have no sample rate. Read the packets out and push them into a packet input with `rate_hz`"
            ),
        );
        false
    }

    /// Type, rate, clock and alignment checks of one node, then the node itself.
    fn node(&mut self, name: &str) {
        let spec = self.spec;
        let Some(node) = spec.nodes.get(name) else {
            return;
        };
        let Some(factory) = self.registry.factory(&node.kind) else {
            return;
        };
        let ports = factory.inputs();
        let path = join("nodes", name);
        let mut feeds: Vec<(usize, usize)> = Vec::with_capacity(ports.len());
        let mut failed = false;
        for port in &ports {
            // A missing edge was reported by the graph check.
            let Some(&k) = self.incoming.get(&(name, port.name)) else {
                return;
            };
            match self.port_stream(name, &node.kind, k, port.accepts) {
                Some(s) => feeds.push((k, s)),
                None => failed = true,
            }
        }
        if failed {
            self.produced.insert(name.to_string(), Produced::Failed);
            return;
        }
        let (Some(&(_, mut src)), second) = (feeds.first(), feeds.get(1).copied()) else {
            return;
        };
        if !self.source_packets_ok(name, &ports, feeds.first().copied()) {
            self.produced.insert(name.to_string(), Produced::Failed);
            return;
        }
        let mut src_b = None;
        let clocked = ports
            .get(1)
            .is_some_and(|p| p.accepts.contains(&PortType::Packets));
        if let Some((_, b)) = second.filter(|_| clocked) {
            // Port `b` of a clocked encoder: packets have no rate, clock or group delay, so
            // the pair checks and the alignment apply to stream ports only.
            src_b = Some(b);
        } else if let Some((kb, b)) = second {
            if !self.pair_ok(name, kb, src, b) {
                self.produced.insert(name.to_string(), Produced::Failed);
                return;
            }
            let mode = node.align.unwrap_or_default();
            let Some((a2, b2)) = self.align(name, mode, src, b) else {
                self.produced.insert(name.to_string(), Produced::Failed);
                return;
            };
            src = a2;
            src_b = Some(b2);
        }
        if let Some(backend) = &node.backend {
            self.check_backend(&path, factory, backend);
        }
        let params = Self::params_map(&node.params);
        let id = |p: &str| p.to_string();
        let idx = self.check_node(name, &node.kind, false, &params, src, &path, &id);
        if let (Some(i), Some(b)) = (idx, src_b) {
            let min_input = self.nodes.get(i).map_or(0, |n| n.checked.min_input);
            if let Some(n) = self.nodes.get_mut(i) {
                n.src_b = Some(b);
            }
            if let Some(s) = self.streams.get_mut(b) {
                s.min_input = s.min_input.max(min_input);
            }
        }
        let result = idx.and_then(|i| self.finish_node(i));
        self.produced
            .insert(name.to_string(), result.unwrap_or(Produced::Failed));
    }

    /// Compares the backend that the facade uses for the kernels of the node with the
    /// preference: `require` is an error, `prefer` a warning.
    ///
    /// A kernel exists for the complex or for the real sample type of the precision, whichever
    /// the facade lists; a fused kernel (`chained`) is built from the single kernels of the
    /// list and is not judged itself.
    fn check_backend(&mut self, path: &str, factory: &dyn BlockFactory, backend: &BackendSpec) {
        let (complex, real) = match self.spec.settings.precision {
            Precision::F32 => (sdrk::SampleType::C32, sdrk::SampleType::F32),
            Precision::Q15 => (sdrk::SampleType::CQ15, sdrk::SampleType::Q15),
        };
        for (wanted, required) in [(&backend.require, true), (&backend.prefer, false)] {
            let Some(wanted) = wanted else {
                continue;
            };
            let missing: Vec<String> = factory
                .kernels()
                .iter()
                .filter_map(|k| {
                    let c = sdrk::backend_for(k, complex);
                    let actual = if c == "none" {
                        sdrk::backend_for(k, real)
                    } else {
                        c
                    };
                    (actual != "chained" && actual != wanted).then(|| format!("{k} ({actual})"))
                })
                .collect();
            if missing.is_empty() {
                continue;
            }
            let key = if required { "require" } else { "prefer" };
            let path = join(&join(path, "backend"), key);
            let message = format!(
                "the backend `{wanted}` does not implement: {}",
                missing.join(", ")
            );
            if required {
                self.report
                    .error(path, DiagCode::BackendUnavailable, message);
            } else {
                self.report
                    .warning(path, DiagCode::BackendUnavailable, message);
            }
        }
    }

    fn attach_outputs(&mut self) {
        let spec = self.spec;
        for (name, output) in &spec.outputs {
            let Some(edge) = self
                .incoming
                .get(&(name.as_str(), PortName::A))
                .and_then(|&k| spec.edges.get(k))
            else {
                continue;
            };
            let path = join("outputs", name);
            let produced = self
                .produced
                .get(&edge.from)
                .copied()
                .unwrap_or(Produced::Failed);
            let packets = match produced {
                Produced::Stream(s) => self
                    .streams
                    .get(s)
                    .is_some_and(|x| x.ty == PortType::Packets),
                _ => false,
            };
            let wanted = match (produced, packets) {
                (Produced::Frames { .. }, _) => "frames",
                (_, true) => "packets",
                _ => "stream",
            };
            match (output.output_type, produced) {
                (_, Produced::Failed) => {}
                (OutputType::Stream, Produced::Stream(_)) if packets => {
                    self.type_error(&path, &edge.from, wanted);
                }
                (OutputType::Packets, Produced::Stream(s)) if packets => {
                    self.outputs.push(PlanOutput {
                        name: name.clone(),
                        spec: output.clone(),
                        source: OutSource::Stream(s),
                        capacity: output.queue_packets(),
                    });
                }
                (OutputType::Stream, Produced::Frames { .. })
                | (OutputType::Frames | OutputType::Packets, Produced::Stream(_))
                | (OutputType::Packets, Produced::Frames { .. }) => {
                    self.type_error(&path, &edge.from, wanted);
                }
                (OutputType::Frames, Produced::Frames { len, node }) => {
                    let idx = self.outputs.len();
                    if let Some(n) = self.nodes.get_mut(node) {
                        n.dst = PlanDst::Frames { output: idx };
                    }
                    self.outputs.push(PlanOutput {
                        name: name.clone(),
                        spec: output.clone(),
                        source: OutSource::Frames { len },
                        capacity: output.queue_frames(),
                    });
                }
                (OutputType::Stream, Produced::Stream(s)) => {
                    self.attach_stream_output(name, output, s);
                }
            }
        }
    }

    /// An output whose type does not fit what its source produces (`wanted`: `stream`,
    /// `frames` or `packets`).
    fn type_error(&mut self, path: &str, from: &str, wanted: &str) {
        let what = match wanted {
            "frames" => "frames",
            "packets" => "packets",
            _ => "a stream",
        };
        self.report.error(
            join(path, "type"),
            DiagCode::TypeMismatch,
            format!("`{from}` produces {what}; declare the output with type `{wanted}`"),
        );
    }

    /// A stream output, with the automatic `convert_out` node if it has a format.
    fn attach_stream_output(&mut self, name: &str, output: &OutputSpec, source: usize) {
        let path = join("outputs", name);
        let mut s = source;
        if let Some(format) = &output.format {
            let ty = self.streams.get(s).map(|x| x.ty);
            if !matches!(ty, Some(PortType::C32 | PortType::CQ15)) {
                self.report.error(
                    join(&path, "format"),
                    DiagCode::InvalidValue,
                    "a raw output format needs a complex stream; real streams cannot be converted",
                );
                return;
            }
            let mut params = Map::new();
            if let Ok(v) = serde_json::to_value(format) {
                params.insert("format".to_string(), v);
            }
            let auto_name = format!("{name}__convert_out");
            let out_path = path.clone();
            let remap = move |p: &str| {
                if p == "params.format" {
                    join(&out_path, "format")
                } else {
                    out_path.clone()
                }
            };
            let idx = self.check_node(&auto_name, "convert_out", true, &params, s, "", &remap);
            match idx.and_then(|i| self.finish_node(i)) {
                Some(Produced::Stream(ns)) => s = ns,
                _ => return,
            }
        }
        let Some(stream) = self.streams.get(s) else {
            return;
        };
        let capacity = if stream.ty == PortType::Raw {
            let (bytes, samples) = stream.elems;
            let c = output.capacity_samples(self.spec.settings.max_chunk) as u128;
            ceil_mul_div(c, u128::from(bytes), u128::from(samples))
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(usize::MAX)
        } else {
            output.capacity_samples(self.spec.settings.max_chunk)
        };
        self.outputs.push(PlanOutput {
            name: name.to_string(),
            spec: output.clone(),
            source: OutSource::Stream(s),
            capacity,
        });
    }

    /// Estimated memory in bytes of the buffers, queues and block storage.
    fn memory(&self, input_rates: &[Rate]) -> usize {
        let max_chunk = self.spec.settings.max_chunk;
        let mut memory = 0usize;
        for n in &self.nodes {
            memory = memory.saturating_add(n.checked.memory_bytes);
        }
        for s in &self.streams {
            if let Some(&r) = input_rates.get(s.origin) {
                let cap = s.capacity(max_chunk, r, s.min_input);
                memory = memory.saturating_add(cap.saturating_mul(elem_size(s.ty)));
            }
        }
        for o in &self.outputs {
            let bytes = match o.source {
                OutSource::Stream(s) => {
                    let ty = self.streams.get(s).map_or(PortType::Raw, |x| x.ty);
                    o.capacity.saturating_mul(elem_size(ty))
                }
                OutSource::Frames { len } => o
                    .capacity
                    .saturating_mul(len.saturating_mul(4).saturating_add(24))
                    .saturating_add(len.saturating_mul(4)),
            };
            memory = memory.saturating_add(bytes);
        }
        memory
    }

    /// Estimated multiply-accumulates per input sample: the sum over the nodes of their cost
    /// times the samples per second they see, divided by the input samples per second.
    fn cost(&self, input_rates: &[Rate]) -> f64 {
        // A packet input without `rate_hz` carries no samples.
        let total_rate: f64 = input_rates
            .iter()
            .zip(&self.inputs)
            .filter(|(_, i)| i.has_rate)
            .map(|(r, _)| r.as_f64())
            .sum();
        let macs_per_second: f64 = self
            .nodes
            .iter()
            .map(|n| n.checked.cost * n.rate_in.as_f64())
            .sum();
        if total_rate > 0.0 {
            macs_per_second / total_rate
        } else {
            0.0
        }
    }

    fn check_limits(&mut self, memory: usize, cost: f64) {
        let limits = self.opts.limits;
        if self.nodes.len() > limits.max_nodes {
            self.report.error(
                "nodes",
                DiagCode::LimitExceeded,
                format!(
                    "the pipeline has {} nodes (automatic ones included); the limit is {}",
                    self.nodes.len(),
                    limits.max_nodes
                ),
            );
        }
        for n in &self.nodes {
            if n.checked.taps > limits.max_taps_per_node {
                let path = if n.auto {
                    "nodes".to_string()
                } else {
                    join("nodes", &n.name)
                };
                self.report.error(
                    path,
                    DiagCode::LimitExceeded,
                    format!(
                        "`{}` needs {} taps; the limit is {}",
                        n.name, n.checked.taps, limits.max_taps_per_node
                    ),
                );
            }
        }
        if memory > limits.max_memory_bytes {
            self.report.error(
                "nodes",
                DiagCode::LimitExceeded,
                format!(
                    "the pipeline needs about {memory} bytes; the limit is {}",
                    limits.max_memory_bytes
                ),
            );
        }
        if cost > limits.max_cost_per_input_sample {
            self.report.error(
                "nodes",
                DiagCode::LimitExceeded,
                format!(
                    "the estimated cost is {cost:.1} multiply-accumulates per input sample; the limit is {}",
                    limits.max_cost_per_input_sample
                ),
            );
        }
    }

    /// Limits, totals, the report; returns the plan.
    fn finish(mut self) -> Plan {
        let input_rates: Vec<Rate> = self.inputs.iter().map(|i| i.rate).collect();
        let memory = self.memory(&input_rates);
        let cost = self.cost(&input_rates);
        self.check_limits(memory, cost);
        self.report.memory_bytes = memory;
        self.report.cost_per_input_sample = cost;
        self.report.nodes = self
            .nodes
            .iter()
            .map(|n| NodeReport {
                name: n.name.clone(),
                kind: n.kind.clone(),
                rate_in: n.rate_in,
                rate_out: n.checked.rate_out,
                delay_samples_in: n.checked.delay,
                auto: n.auto,
                clock_domain: self
                    .streams
                    .get(n.src)
                    .map_or_else(String::new, |s| self.input_name(s.origin)),
            })
            .collect();
        self.report.group_edges = group_edges(self.spec).into_boxed_slice();
        let plan_view = (&self.nodes, &self.streams, &self.inputs, &self.outputs);
        let paths = output_paths(plan_view.0, plan_view.1, plan_view.3);
        self.report.outputs =
            latency_report(&paths, plan_view.0, plan_view.1, plan_view.3).into_boxed_slice();
        Plan {
            max_chunk: self.spec.settings.max_chunk,
            inputs: self.inputs,
            streams: self.streams,
            nodes: self.nodes,
            outputs: self.outputs,
        }
    }
}
