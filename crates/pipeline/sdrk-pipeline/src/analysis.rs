//! Validation of a typed description and the plan that `build` executes.
//!
//! Stage 1 checks the structure of the model (versions, names, kinds, parameters, output
//! attributes). Stage 2 checks the graph (endpoints, inputs, cycles) and propagates types and
//! rates in topological order, calling the factory of every node. Stage 2 only runs when
//! stage 1 found no error, so that a single mistake does not produce a flood of follow-up
//! diagnostics.

use std::collections::{BTreeMap, BTreeSet};

use sdrk::{Delay, Rate};
use sdrk_blocks::PortType;
use sdrk_device::SampleFormat;
use serde_json::{Map, Value};

use crate::latency::{PathNode, path_latency};
use crate::model::{
    BackendSpec, InputType, MAIN_GROUP, MAX_CHUNK_LIMIT, MAX_DESCRIPTION_CHARS, OutputSpec,
    OutputType, PipelineSpec, Precision, is_valid_name,
};
use crate::params::check_params;
use crate::registry::{BlockFactory, BuildOptions, Checked, PortKind, Registry};
use crate::structure::join;
use crate::validate::{DiagCode, Diagnostic, GroupEdge, NodeReport, OutputReport, Report};

/// Largest `capacity` of a stream output in samples.
const MAX_CAPACITY: usize = 1 << 28;
/// Largest `queue` of a frame output.
const MAX_QUEUE: usize = 1 << 16;
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
    pub src: usize,
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
}

/// An input of the pipeline.
pub(crate) struct PlanInput {
    pub name: String,
    pub stream: usize,
    pub ty: PortType,
    pub rate: Rate,
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

    /// Capacity of the buffer: `ceil(max_chunk * R) + min_input + 64`.
    pub(crate) fn capacity(&self, max_chunk: usize, origin_rate: Rate, min_input: usize) -> usize {
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
    }
}

fn port_name(t: PortType) -> &'static str {
    match t {
        PortType::C32 => "c32",
        PortType::CQ15 => "cq15",
        PortType::F32 => "f32",
        PortType::Q15 => "q15",
        PortType::Raw => "raw bytes",
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
    check_names(spec, report);
    check_groups(spec, report);
    check_inputs(spec, report);
    check_nodes(registry, spec, report);
    check_outputs(spec, report);
}

/// Name pattern, then uniqueness across the three sections and the automatic nodes.
fn check_names(spec: &PipelineSpec, report: &mut Report) {
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
    // Automatic nodes take the names `<input>__convert` and `<output>__convert_out`.
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
    for (auto, owner) in inputs.chain(outputs) {
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

fn check_inputs(spec: &PipelineSpec, report: &mut Report) {
    for (name, input) in &spec.inputs {
        let path = join("inputs", name);
        check_description(input.description.as_ref(), &path, report);
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
            if output.output_type == OutputType::Frames {
                report.error(
                    join(&path, "format"),
                    DiagCode::InvalidValue,
                    "a frames output has no format",
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
            OutputType::Frames => {
                if output.capacity.is_some() {
                    report.error(
                        join(&path, "capacity"),
                        DiagCode::InvalidValue,
                        "`capacity` belongs to stream outputs; a frames output has a `queue`",
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
    /// Index of the edge that feeds each node or output.
    incoming: BTreeMap<&'a str, usize>,
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
        let mut extra: Vec<(usize, &str)> = Vec::new();
        for (k, edge) in spec.edges.iter().enumerate() {
            let from_ok = self.endpoint_ok(k, "from", &edge.from);
            let to_ok = self.endpoint_ok(k, "to", &edge.to);
            if from_ok {
                outgoing.insert(edge.from.as_str());
            }
            if from_ok && to_ok {
                if self.incoming.contains_key(edge.to.as_str()) {
                    extra.push((k, edge.to.as_str()));
                } else {
                    self.incoming.insert(edge.to.as_str(), k);
                }
            }
        }
        for (k, to) in extra {
            self.report.error(
                format!("edges[{k}].to"),
                DiagCode::MultipleInputs,
                format!("`{to}` already has an incoming edge; a node or output has exactly one"),
            );
        }
        self.unconnected(&outgoing);
    }

    /// Nodes and outputs without an incoming edge, nodes and inputs that go nowhere.
    fn unconnected(&mut self, outgoing: &BTreeSet<&str>) {
        let spec = self.spec;
        let sections: [(&str, Vec<&String>); 2] = [
            ("nodes", spec.nodes.keys().collect()),
            ("outputs", spec.outputs.keys().collect()),
        ];
        for (section, names) in sections {
            for name in names {
                if !self.incoming.contains_key(name.as_str()) {
                    self.report.error(
                        join(section, name),
                        DiagCode::MissingInput,
                        format!("`{name}` has no incoming edge"),
                    );
                }
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
        // parent[node] = the node feeding it, if the source is a node
        let mut parent: BTreeMap<&str, &str> = BTreeMap::new();
        let mut children: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for name in spec.nodes.keys() {
            if let Some(&k) = self.incoming.get(name.as_str())
                && let Some(edge) = spec.edges.get(k)
                && spec.nodes.contains_key(&edge.from)
            {
                parent.insert(name.as_str(), edge.from.as_str());
                children
                    .entry(edge.from.as_str())
                    .or_default()
                    .push(name.as_str());
            }
        }
        let mut ready: BTreeSet<(usize, &str)> = spec
            .nodes
            .keys()
            .filter(|n| !parent.contains_key(n.as_str()))
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
                    ready.insert((prio(kid), kid));
                }
            }
        }
        if order.len() < spec.nodes.len() {
            // Leftover nodes lie on a cycle or behind one. Follow the parent pointers.
            let mut state: BTreeMap<&str, u8> = BTreeMap::new();
            let leftover: Vec<&str> = spec
                .nodes
                .keys()
                .map(String::as_str)
                .filter(|n| !done.contains(n))
                .collect();
            let mut found: Vec<&str> = Vec::new();
            for &start in &leftover {
                let mut walk: Vec<&str> = Vec::new();
                let mut cur = Some(start);
                while let Some(n) = cur {
                    match state.get(n) {
                        Some(1) => {
                            // A cycle: the walk from `n` onwards.
                            if let Some(pos) = walk.iter().position(|w| *w == n)
                                && let Some(best) =
                                    walk.iter().skip(pos).min_by_key(|w| (prio(w), **w))
                            {
                                found.push(best);
                            }
                            break;
                        }
                        Some(_) => break,
                        None => {
                            state.insert(n, 1);
                            walk.push(n);
                            cur = parent.get(n).copied();
                        }
                    }
                }
                for w in walk {
                    state.insert(w, 2);
                }
            }
            for n in found {
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
        let _ = rate_in;
        match output {
            PortKind::Stream(ty) => {
                let elems = if ty == PortType::Raw {
                    self.raw_elems(idx)
                } else {
                    (1, 1)
                };
                let s = self.new_stream(ty, rate_out, origin, elems);
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
            let rate = input.rate_hz.0;
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
                });
                self.produced.insert(name.clone(), Produced::Stream(s));
            } else {
                self.produced.insert(name.clone(), Produced::Failed);
            }
        }
    }

    /// Type and rate check of one node.
    fn node(&mut self, name: &str) {
        let spec = self.spec;
        let (Some(node), Some(&k)) = (spec.nodes.get(name), self.incoming.get(name)) else {
            return;
        };
        let Some(edge) = spec.edges.get(k) else {
            return;
        };
        let path = join("nodes", name);
        let produced = self
            .produced
            .get(&edge.from)
            .copied()
            .unwrap_or(Produced::Failed);
        let src = match produced {
            Produced::Failed => {
                self.produced.insert(name.to_string(), Produced::Failed);
                return;
            }
            Produced::Frames { .. } => {
                self.report.error(
                    format!("edges[{k}].to"),
                    DiagCode::TypeMismatch,
                    format!(
                        "`{}` produces frames, which only an output can take; `{name}` needs a stream",
                        edge.from
                    ),
                );
                self.produced.insert(name.to_string(), Produced::Failed);
                return;
            }
            Produced::Stream(s) => s,
        };
        let Some(factory) = self.registry.factory(&node.kind) else {
            return;
        };
        let ty = self.streams.get(src).map_or(PortType::Raw, |s| s.ty);
        let accepted = factory
            .description()
            .inputs
            .first()
            .map_or(&[][..], |p| p.types);
        if !accepted.contains(&ty) {
            self.report.error(
                format!("edges[{k}].to"),
                DiagCode::TypeMismatch,
                format!(
                    "`{name}` ({}) takes {}, but `{}` delivers {}",
                    node.kind,
                    describe_types(accepted),
                    edge.from,
                    port_name(ty)
                ),
            );
            self.produced.insert(name.to_string(), Produced::Failed);
            return;
        }
        if let Some(backend) = &node.backend {
            self.check_backend(&path, factory, backend);
        }
        let params = Self::params_map(&node.params);
        let id = |p: &str| p.to_string();
        let idx = self.check_node(name, &node.kind, false, &params, src, &path, &id);
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
                .get(name.as_str())
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
            match (output.output_type, produced) {
                (_, Produced::Failed) => {}
                (OutputType::Stream, Produced::Frames { .. }) => self.report.error(
                    join(&path, "type"),
                    DiagCode::TypeMismatch,
                    format!(
                        "`{}` produces frames; declare the output with type `frames`",
                        edge.from
                    ),
                ),
                (OutputType::Frames, Produced::Stream(_)) => self.report.error(
                    join(&path, "type"),
                    DiagCode::TypeMismatch,
                    format!(
                        "`{}` produces a stream; declare the output with type `stream`",
                        edge.from
                    ),
                ),
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
        let total_rate: f64 = input_rates.iter().map(Rate::as_f64).sum();
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
