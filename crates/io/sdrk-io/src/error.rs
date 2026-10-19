//! Error type of `sdrk-io`.

use core::fmt;
use sdrk_device::{ErrorKind, SdrkError};

/// Errors of the `sdrk-io` crate.
///
/// ```
/// use sdrk_device::{ErrorKind, SdrkError};
/// let e = sdrk_io::Error::Parse { file: "rec.sigmf-meta".into(), reason: "no captures".into() };
/// assert_eq!(e.kind(), ErrorKind::Parse);
/// assert_eq!(e.to_string(), "cannot parse rec.sigmf-meta: no captures");
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// An operating-system I/O error (kind: `Io`).
    Io(String),
    /// A file is malformed (kind: `Parse`).
    Parse {
        /// The file (or a description of the input) that failed.
        file: String,
        /// What is wrong.
        reason: String,
    },
    /// The file uses a feature that is not supported (kind: `Unsupported`).
    Unsupported(&'static str),
    /// An argument is invalid (kind: `InvalidArgument`).
    InvalidArgument(String),
    /// An error of the device layer (kind forwarded).
    Device(sdrk_device::Error),
}

/// Result alias of this crate.
pub type Result<T, E = Error> = core::result::Result<T, E>;

impl Error {
    pub(crate) fn parse(file: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Parse {
            file: file.into(),
            reason: reason.into(),
        }
    }

    pub(crate) fn invalid(reason: impl Into<String>) -> Self {
        Self::InvalidArgument(reason.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "i/o error: {msg}"),
            Self::Parse { file, reason } => write!(f, "cannot parse {file}: {reason}"),
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::Device(e) => write!(f, "device error: {e}"),
        }
    }
}

impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Device(e) => Some(e),
            _ => None,
        }
    }
}

impl SdrkError for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Io(_) => ErrorKind::Io,
            Self::Parse { .. } => ErrorKind::Parse,
            Self::Unsupported(_) => ErrorKind::Unsupported,
            Self::InvalidArgument(_) => ErrorKind::InvalidArgument,
            Self::Device(e) => e.kind(),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

impl From<sdrk_device::Error> for Error {
    fn from(e: sdrk_device::Error) -> Self {
        Self::Device(e)
    }
}
