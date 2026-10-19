//! Node groups: splitting a description into one description per group.

use std::collections::BTreeMap;

use sdrk_blocks::PortType;

use crate::analysis::{PlanNode, analyze, group_of};
use crate::model::{
    EdgeSpec, InputSpec, InputType, MAIN_GROUP, OutputSpec, OutputType, PipelineSpec, RateHz,
};
use crate::registry::{BuildOptions, PortKind, Registry};
use crate::validate::{DiagCode, Report};

/// The name of the output and input that carry the edge `from -> to` between two groups.
fn edge_name(from: &str, to: &str) -> String {
    format!("edge__{from}__{to}")
}

fn input_type(ty: PortType) -> Option<InputType> {
    match ty {
        PortType::C32 => Some(InputType::C32),
        PortType::CQ15 => Some(InputType::Cq15),
        PortType::F32 => Some(InputType::F32),
        PortType::Q15 => Some(InputType::Q15),
        _ => None,
    }
}

/// Splits a description into one description per group.
///
/// The result is sorted by group name. Nodes without a group are in `"main"`. Every edge
/// between nodes of different groups `a -> b` becomes the output `edge__a__b` (a stream of the
/// type of `a`) in the group of `a` and the input `edge__a__b` (the same type and the exact
/// rate) in the group of `b`. Pipeline inputs are copied into every group whose nodes read
/// them; an output belongs to the group of the node that feeds it. The descriptions carry no
/// `groups` and no `group` keys. Groups have no runtime effect yet.
///
/// ```
/// use sdrk_pipeline::{BuildOptions, PipelineSpec, Registry, split_by_groups};
/// let spec = PipelineSpec::from_json(include_str!("../../../../examples/flows/fm_receiver_grouped.json"))?;
/// let registry = Registry::standard();
/// let parts = split_by_groups(&spec, &registry).unwrap();
/// let names: Vec<&str> = parts.iter().map(|(g, _)| g.as_str()).collect();
/// assert_eq!(names, ["audio", "front"]);
/// for (_, part) in &parts {
///     assert!(registry.validate(part, &BuildOptions::default()).is_ok());
/// }
/// # Ok::<(), sdrk_pipeline::Error>(())
/// ```
///
/// # Errors
/// The report of the description if it is not valid, or an `invalid_value` error for an edge
/// that cannot be carried (frames or raw bytes) or a name that is too long.
#[allow(
    clippy::large_stack_frames,
    reason = "clippy counts the temporary of every `?` separately; the real frame is small"
)]
pub fn split_by_groups(
    spec: &PipelineSpec,
    registry: &Registry,
) -> Result<Vec<(String, PipelineSpec)>, Report> {
    let analysis = analyze(registry, spec, &BuildOptions::default());
    if !analysis.report.is_ok() {
        return Err(analysis.report);
    }
    let Some(plan) = analysis.plan else {
        return Err(analysis.report);
    };
    let mut split = Splitter {
        spec,
        plan_nodes: &plan.nodes,
        parts: BTreeMap::new(),
        fail: Report::default(),
    };
    for (name, node) in &spec.nodes {
        let mut node = node.clone();
        node.group = None;
        split
            .part(group_of(spec, name))
            .nodes
            .insert(name.clone(), node);
    }
    for edge in &spec.edges {
        match (
            spec.nodes.contains_key(&edge.from),
            spec.nodes.contains_key(&edge.to),
        ) {
            (false, true) => split.input_edge(edge),
            (true, true) => split.node_edge(edge),
            _ => split.output_edge(edge),
        }
    }
    if !split.fail.errors.is_empty() {
        return Err(split.fail);
    }
    let result: Vec<(String, PipelineSpec)> = split.parts.into_iter().collect();
    for (group, part) in &result {
        let check = registry.validate(part, &BuildOptions::default());
        if !check.is_ok() {
            let mut report = Report::default();
            report.error(
                format!("groups.{group}"),
                DiagCode::InvalidValue,
                format!(
                    "the part for group `{group}` is not valid: {}",
                    check.to_json()
                ),
            );
            return Err(report);
        }
    }
    Ok(result)
}

/// The descriptions being built.
struct Splitter<'a> {
    spec: &'a PipelineSpec,
    plan_nodes: &'a [PlanNode],
    parts: BTreeMap<String, PipelineSpec>,
    fail: Report,
}

impl Splitter<'_> {
    /// The description of a group, created empty on first use.
    fn part(&mut self, group: &str) -> &mut PipelineSpec {
        let spec = self.spec;
        self.parts
            .entry(group.to_string())
            .or_insert_with(|| PipelineSpec {
                sdrk: spec.sdrk,
                name: format!("{}.{group}", spec.name),
                description: spec.description.clone(),
                settings: spec.settings,
                inputs: BTreeMap::new(),
                groups: BTreeMap::new(),
                nodes: BTreeMap::new(),
                edges: Vec::new(),
                outputs: BTreeMap::new(),
            })
    }

    /// An input read by a node: the input is copied into the group of the node.
    fn input_edge(&mut self, edge: &EdgeSpec) {
        let group = group_of(self.spec, &edge.to).to_string();
        if let Some(input) = self.spec.inputs.get(&edge.from) {
            let part = self.part(&group);
            part.inputs.insert(edge.from.clone(), input.clone());
            part.edges.push(edge.clone());
        }
    }

    /// An edge between two nodes: kept inside a group, or cut into an output and an input.
    fn node_edge(&mut self, edge: &EdgeSpec) {
        let spec = self.spec;
        let (from_group, to_group) = (group_of(spec, &edge.from), group_of(spec, &edge.to));
        if from_group == to_group {
            self.part(from_group).edges.push(edge.clone());
            return;
        }
        let name = edge_name(&edge.from, &edge.to);
        let Some(source) = self.plan_nodes.iter().find(|n| n.name == edge.from) else {
            return;
        };
        let sample_type = match source.checked.output {
            PortKind::Stream(ty) => input_type(ty),
            PortKind::Frames { .. } => None,
        };
        let path = format!("edges.{}", edge.from);
        let Some(sample_type) = sample_type else {
            self.fail.error(
                path,
                DiagCode::InvalidValue,
                format!(
                    "the edge `{} -> {}` crosses groups but carries frames or raw bytes",
                    edge.from, edge.to
                ),
            );
            return;
        };
        if name.len() > 64 {
            self.fail.error(
                path,
                DiagCode::InvalidName,
                format!("the name `{name}` of the group edge is longer than 64 characters"),
            );
            return;
        }
        let rate = source.checked.rate_out;
        self.part(from_group).outputs.insert(
            name.clone(),
            OutputSpec {
                output_type: OutputType::Stream,
                format: None,
                capacity: None,
                queue: None,
                drop: None,
                underrun: None,
                description: Some(format!("group edge to `{}`", edge.to)),
            },
        );
        self.part(from_group).edges.push(EdgeSpec {
            from: edge.from.clone(),
            to: name.clone(),
        });
        let consumer = self.part(to_group);
        consumer.inputs.insert(
            name.clone(),
            InputSpec {
                format: None,
                sample_type: Some(sample_type),
                rate_hz: RateHz(rate),
                description: Some(format!("group edge from `{}`", edge.from)),
            },
        );
        consumer.edges.push(EdgeSpec {
            from: name,
            to: edge.to.clone(),
        });
    }

    /// An edge to an output: the output belongs to the group of its source node (`main` for an
    /// input).
    fn output_edge(&mut self, edge: &EdgeSpec) {
        let spec = self.spec;
        let from_node = spec.nodes.contains_key(&edge.from);
        let group = if from_node {
            group_of(spec, &edge.from)
        } else {
            MAIN_GROUP
        };
        let Some(output) = spec.outputs.get(&edge.to) else {
            return;
        };
        let part = self.part(group);
        part.outputs.insert(edge.to.clone(), output.clone());
        if !from_node {
            if let Some(input) = spec.inputs.get(&edge.from) {
                part.inputs.insert(edge.from.clone(), input.clone());
            }
        }
        part.edges.push(edge.clone());
    }
}
