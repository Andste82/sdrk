//! Diagnostics and the validation report.

use sdrk::{Delay, Rate};
use serde::{Serialize, Serializer};

/// Machine-readable class of a problem. Serializes as `snake_case`.
///
/// | Code | Meaning |
/// | --- | --- |
/// | `Syntax` | the text is not valid JSON, or the document does not fit the model (message carries line and column) |
/// | `SchemaVersion` | `sdrk` is missing or not 1 |
/// | `UnknownField` | a key that the model does not have |
/// | `InvalidName` | a name that does not match `^[a-z][a-z0-9_]{0,63}$` |
/// | `DuplicateName` | a name used twice among inputs, nodes and outputs (automatic nodes included) |
/// | `UnknownKind` | a node kind that is not registered |
/// | `UnknownParam` | a parameter the block does not have |
/// | `WrongType` | a JSON value of the wrong type |
/// | `OutOfRange` | a number outside the limits of the parameter or field |
/// | `MissingParam` | a required field or parameter is absent |
/// | `InvalidValue` | the right type but not a possible value (unknown name, non-integer, inconsistent combination) |
/// | `UnknownEndpoint` | an edge names something that does not exist or cannot be used there |
/// | `MultipleInputs` | a node or output with more than one incoming edge |
/// | `MissingInput` | a node or output without an incoming edge |
/// | `DanglingOutput` | a node output that goes nowhere (as a warning: an input nobody reads) |
/// | `Cycle` | a node that lies on a cycle |
/// | `TypeMismatch` | the port types of source and destination do not fit |
/// | `RateUnsupported` | the block rejects its configuration for the sample rate at its input |
/// | `LimitExceeded` | a resource limit is exceeded |
/// | `UnknownGroup` | a node names a group that `groups` does not declare |
/// | `BackendUnavailable` | a backend preference names a backend that does not implement the kernels of the node (an error for `require`, a warning for `prefer`) |
///
/// ```
/// use sdrk_pipeline::DiagCode;
/// assert_eq!(DiagCode::UnknownField.as_str(), "unknown_field");
/// assert_eq!(serde_json::to_string(&DiagCode::LimitExceeded)?, "\"limit_exceeded\"");
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DiagCode {
    /// Invalid JSON or a document that does not fit the model.
    Syntax,
    /// `sdrk` is missing or not 1.
    SchemaVersion,
    /// A key that the model does not have.
    UnknownField,
    /// A name that does not match the name pattern.
    InvalidName,
    /// A name used twice.
    DuplicateName,
    /// A node kind that is not registered.
    UnknownKind,
    /// A parameter the block does not have.
    UnknownParam,
    /// A JSON value of the wrong type.
    WrongType,
    /// A number outside its limits.
    OutOfRange,
    /// A required field or parameter is absent.
    MissingParam,
    /// The right type but not a possible value.
    InvalidValue,
    /// An edge names something that does not exist or cannot be used there.
    UnknownEndpoint,
    /// More than one incoming edge.
    MultipleInputs,
    /// No incoming edge.
    MissingInput,
    /// A node output that goes nowhere.
    DanglingOutput,
    /// A node on a cycle.
    Cycle,
    /// Port types that do not fit.
    TypeMismatch,
    /// The block rejects its configuration for the sample rate.
    RateUnsupported,
    /// A resource limit is exceeded.
    LimitExceeded,
    /// A node names a group that is not declared.
    UnknownGroup,
    /// A required (or, as a warning, preferred) backend does not implement the kernels.
    BackendUnavailable,
}

impl DiagCode {
    /// The `snake_case` name used in the JSON report.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Syntax => "syntax",
            Self::SchemaVersion => "schema_version",
            Self::UnknownField => "unknown_field",
            Self::InvalidName => "invalid_name",
            Self::DuplicateName => "duplicate_name",
            Self::UnknownKind => "unknown_kind",
            Self::UnknownParam => "unknown_param",
            Self::WrongType => "wrong_type",
            Self::OutOfRange => "out_of_range",
            Self::MissingParam => "missing_param",
            Self::InvalidValue => "invalid_value",
            Self::UnknownEndpoint => "unknown_endpoint",
            Self::MultipleInputs => "multiple_inputs",
            Self::MissingInput => "missing_input",
            Self::DanglingOutput => "dangling_output",
            Self::Cycle => "cycle",
            Self::TypeMismatch => "type_mismatch",
            Self::RateUnsupported => "rate_unsupported",
            Self::LimitExceeded => "limit_exceeded",
            Self::UnknownGroup => "unknown_group",
            Self::BackendUnavailable => "backend_unavailable",
        }
    }
}

/// One problem: where (`path`, with dots and the JSON names), what (`code`) and why
/// (`message`, for humans).
///
/// ```
/// use sdrk_pipeline::{DiagCode, Diagnostic};
/// let d = Diagnostic { path: "nodes.tune.params.decimation".into(), code: DiagCode::OutOfRange, message: "0 is not in 1 to 1024".into() };
/// assert_eq!(serde_json::to_value(&d)?["code"], "out_of_range");
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Diagnostic {
    /// Location, e.g. `nodes.tune.params.frequency_hz`, `edges[2].to`, `inputs.rf.format`.
    pub path: String,
    /// Machine-readable class.
    pub code: DiagCode,
    /// Human-readable explanation.
    pub message: String,
}

/// What the validation found out about one node.
///
/// ```
/// # use sdrk_pipeline::{BuildOptions, Registry};
/// let registry = Registry::standard();
/// let report = registry.validate_json(include_str!("../../../../examples/flows/fm_receiver.json"), &BuildOptions::default());
/// let tune = report.nodes.iter().find(|n| n.name == "tune").unwrap();
/// assert_eq!(tune.kind, "ddc");
/// assert!(!tune.auto);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NodeReport {
    /// Node name (automatic nodes: `<input>__convert`, `<output>__convert_out`).
    pub name: String,
    /// Block kind.
    pub kind: String,
    /// Sample rate at the input of the node.
    #[serde(serialize_with = "serialize_rate")]
    pub rate_in: Rate,
    /// Sample rate at the output of the node.
    #[serde(serialize_with = "serialize_rate")]
    pub rate_out: Rate,
    /// Group delay of the node in its input samples.
    #[serde(serialize_with = "serialize_delay")]
    pub delay_samples_in: Delay,
    /// Whether the pipeline inserted the node (conversion).
    pub auto: bool,
}

/// An edge between two nodes of different groups: an interface of the partitioning.
///
/// ```
/// use sdrk_pipeline::{BuildOptions, Registry};
/// let json = include_str!("../../../../examples/flows/fm_receiver_grouped.json");
/// let report = Registry::standard().validate_json(json, &BuildOptions::default());
/// assert_eq!(report.group_edges.len(), 1);
/// assert_eq!((report.group_edges[0].from.as_str(), report.group_edges[0].to.as_str()), ("tune", "demod"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GroupEdge {
    /// Source node.
    pub from: String,
    /// Destination node.
    pub to: String,
    /// Group of the source node.
    pub from_group: String,
    /// Group of the destination node.
    pub to_group: String,
}

/// Latency figures of one output, see `docs/timing.md`.
///
/// ```
/// # use sdrk_pipeline::{BuildOptions, Registry};
/// let json = include_str!("../../../../examples/flows/fm_receiver.json");
/// let report = Registry::standard().validate_json(json, &BuildOptions::default());
/// let audio = report.outputs.iter().find(|o| o.name == "audio").unwrap();
/// assert!(audio.group_delay.num() > 0);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OutputReport {
    /// Output name.
    pub name: String,
    /// Group delay from the input to this output in samples of the output (samples at the
    /// input of the last node for a frames output).
    #[serde(serialize_with = "serialize_delay")]
    pub group_delay: Delay,
    /// Worst-case number of samples held back in the buffers along the path, in samples of
    /// the output.
    pub max_buffered: u64,
}

fn serialize_rate<S: Serializer>(rate: &Rate, s: S) -> Result<S::Ok, S::Error> {
    if rate.den() == 1 {
        s.serialize_u64(rate.num())
    } else {
        s.serialize_str(&format!("{}/{}", rate.num(), rate.den()))
    }
}

fn serialize_delay<S: Serializer>(delay: &Delay, s: S) -> Result<S::Ok, S::Error> {
    let d = delay.reduced();
    s.serialize_str(&format!("{}/{}", d.num(), d.den()))
}

/// The result of validation: all errors and warnings, and the figures of the pipeline.
///
/// `memory_bytes` and `cost_per_input_sample` are estimates computed from the configuration
/// (buffers, block storage; multiply-accumulates per input sample). After `build` the report
/// of the pipeline carries the exact delays and buffer sizes.
///
/// ```
/// use sdrk_pipeline::{BuildOptions, Registry};
/// let registry = Registry::standard();
/// let report = registry.validate_json("{", &BuildOptions::default());
/// assert!(!report.is_ok());
/// let json: serde_json::Value = serde_json::from_str(&report.to_json())?;
/// assert_eq!(json["errors"][0]["code"], "syntax");
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct Report {
    /// Problems that make the description unusable.
    pub errors: Vec<Diagnostic>,
    /// Things that are legal but probably not intended.
    pub warnings: Vec<Diagnostic>,
    /// Estimated memory of the pipeline in bytes.
    pub memory_bytes: usize,
    /// Estimated multiply-accumulates per pipeline input sample.
    pub cost_per_input_sample: f64,
    /// The nodes in processing order, automatic ones included.
    pub nodes: Vec<NodeReport>,
    /// Latency of every output. (A boxed slice keeps `Report`, and with it `Error`, small.)
    pub outputs: Box<[OutputReport]>,
    /// Edges between nodes of different groups (empty without groups).
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    pub group_edges: Box<[GroupEdge]>,
}

impl Report {
    /// True if there are no errors.
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// The report as pretty JSON: the machine-readable error format for agents.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    pub(crate) fn error(
        &mut self,
        path: impl Into<String>,
        code: DiagCode,
        message: impl Into<String>,
    ) {
        self.errors.push(Diagnostic {
            path: path.into(),
            code,
            message: message.into(),
        });
    }

    pub(crate) fn warning(
        &mut self,
        path: impl Into<String>,
        code: DiagCode,
        message: impl Into<String>,
    ) {
        self.warnings.push(Diagnostic {
            path: path.into(),
            code,
            message: message.into(),
        });
    }
}
