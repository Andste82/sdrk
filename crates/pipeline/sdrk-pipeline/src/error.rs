//! Error type of `sdrk-pipeline`.

use core::fmt;

use sdrk_device::{ErrorKind, SdrkError};

use crate::validate::Report;

/// Errors of the `sdrk-pipeline` crate.
///
/// Upper layer: the variants may carry `String`s (names of inputs, outputs and nodes). The hot
/// path of the runner never creates an error unless something went wrong.
///
/// ```
/// use sdrk_pipeline::{Error, PipelineSpec};
/// let err = PipelineSpec::from_json("{").unwrap_err();
/// assert!(matches!(err, Error::Json(_)));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// The input name is not declared (kind: `InvalidArgument`).
    UnknownInput(String),
    /// The output name is not declared (kind: `InvalidArgument`).
    UnknownOutput(String),
    /// The node name is not declared (kind: `InvalidArgument`).
    UnknownNode(String),
    /// The data or the sample type does not match the input or output (kind: `InvalidArgument`).
    WrongType {
        /// Name of the input or output.
        name: String,
        /// What the input or output carries.
        expected: &'static str,
        /// What the caller supplied.
        got: &'static str,
    },
    /// A push carries more samples than `settings.max_chunk` (kind: `LengthMismatch`).
    ChunkTooLarge {
        /// Number of elements supplied.
        len: usize,
        /// Largest number accepted.
        max: usize,
    },
    /// An output with the drop policy `error` cannot take the result of a push; nothing was
    /// consumed (kind: `CapacityExceeded`).
    OutputFull(String),
    /// A frame buffer of the wrong length was passed (kind: `LengthMismatch`).
    LengthMismatch {
        /// Length of a frame of this output.
        expected: usize,
        /// Length of the buffer supplied.
        got: usize,
    },
    /// A run-time parameter value is not valid JSON for the parameter (kind: `InvalidArgument`).
    InvalidParameter {
        /// Node name.
        node: String,
        /// Parameter name.
        name: String,
        /// Why the value is invalid.
        reason: String,
    },
    /// A block failed (kind forwarded from the block error).
    Block {
        /// Node name.
        node: String,
        /// The error of the block.
        source: sdrk_blocks::Error,
    },
    /// The description did not pass validation; the report lists all problems (kind:
    /// `Validation`).
    Build(Report),
    /// The JSON text could not be parsed into the model (kind: `Parse`).
    Json(String),
    /// The call does not fit the state of the pipeline, for example metadata that goes back in
    /// time or a timed change without an anchor (kind: `InvalidState`).
    InvalidState(String),
    /// A node has as many pending timed parameter changes as `settings.pending_params` allows
    /// (kind: `CapacityExceeded`).
    PendingFull(String),
    /// A wire message is malformed (kind: `Parse`).
    Wire(String),
    /// The YAML text could not be parsed into the model (kind: `Parse`).
    Yaml(String),
    /// An invariant of the runner was violated; cannot happen after a successful build (kind:
    /// `Internal`).
    Internal(&'static str),
}

/// Result alias of this crate.
pub type Result<T, E = Error> = core::result::Result<T, E>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownInput(n) => write!(f, "unknown input `{n}`"),
            Self::UnknownOutput(n) => write!(f, "unknown output `{n}`"),
            Self::UnknownNode(n) => write!(f, "unknown node `{n}`"),
            Self::WrongType {
                name,
                expected,
                got,
            } => write!(
                f,
                "wrong sample type for `{name}`: expected {expected}, got {got}"
            ),
            Self::ChunkTooLarge { len, max } => {
                write!(f, "chunk of {len} elements exceeds max_chunk ({max})")
            }
            Self::OutputFull(n) => write!(f, "output `{n}` is full"),
            Self::LengthMismatch { expected, got } => {
                write!(f, "buffer length mismatch: expected {expected}, got {got}")
            }
            Self::InvalidParameter { node, name, reason } => {
                write!(f, "invalid parameter {node}.{name}: {reason}")
            }
            Self::Block { node, source } => write!(f, "block `{node}` failed: {source}"),
            Self::Build(report) => write!(
                f,
                "the pipeline description has {} error(s){}",
                report.errors.len(),
                report
                    .errors
                    .first()
                    .map(|d| format!(
                        "; first: {} at `{}`: {}",
                        d.code.as_str(),
                        d.path,
                        d.message
                    ))
                    .unwrap_or_default()
            ),
            Self::Json(m) => write!(f, "invalid pipeline JSON: {m}"),
            Self::Yaml(m) => write!(f, "invalid pipeline YAML: {m}"),
            Self::InvalidState(m) => write!(f, "invalid state: {m}"),
            Self::PendingFull(n) => write!(f, "too many pending parameter changes for `{n}`"),
            Self::Wire(m) => write!(f, "invalid wire message: {m}"),
            Self::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Block { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl SdrkError for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::UnknownInput(_)
            | Self::UnknownOutput(_)
            | Self::UnknownNode(_)
            | Self::WrongType { .. }
            | Self::InvalidParameter { .. } => ErrorKind::InvalidArgument,
            Self::ChunkTooLarge { .. } | Self::LengthMismatch { .. } => ErrorKind::LengthMismatch,
            Self::OutputFull(_) | Self::PendingFull(_) => ErrorKind::CapacityExceeded,
            Self::InvalidState(_) => ErrorKind::InvalidState,
            Self::Block { source, .. } => source.kind(),
            Self::Build(_) => ErrorKind::Validation,
            Self::Json(_) | Self::Wire(_) | Self::Yaml(_) => ErrorKind::Parse,
            Self::Internal(_) => ErrorKind::Internal,
        }
    }
}

impl From<Report> for Error {
    fn from(report: Report) -> Self {
        Self::Build(report)
    }
}
