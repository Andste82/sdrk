//! Error type of `sdrk-core`.

use core::fmt;
use sdrk_device::{ErrorKind, SdrkError};

/// Errors of the `sdrk-core` crate. All variants are allocation-free.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// Input and output lengths differ (kind: `LengthMismatch`).
    LengthMismatch {
        /// Expected length.
        expected: usize,
        /// Length that was given.
        got: usize,
    },
    /// An output buffer is too small (kind: `BufferTooSmall`).
    BufferTooSmall {
        /// Elements needed.
        needed: usize,
        /// Elements available.
        got: usize,
    },
    /// A parameter is invalid (kind: `InvalidArgument`).
    InvalidParameter(&'static str),
    /// The requested operation is not supported (kind: `Unsupported`).
    Unsupported(&'static str),
    /// An error of the device layer (kind forwarded).
    Device(sdrk_device::Error),
    /// An internal invariant was violated (kind: `Internal`).
    Internal(&'static str),
}

/// Result alias of this crate.
pub type Result<T, E = Error> = core::result::Result<T, E>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {expected}, got {got}")
            }
            Self::BufferTooSmall { needed, got } => {
                write!(f, "buffer too small: needed {needed}, got {got}")
            }
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Self::Device(e) => write!(f, "device error: {e}"),
            Self::Internal(msg) => write!(f, "internal error: {msg}"),
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
            Self::LengthMismatch { .. } => ErrorKind::LengthMismatch,
            Self::BufferTooSmall { .. } => ErrorKind::BufferTooSmall,
            Self::InvalidParameter(_) => ErrorKind::InvalidArgument,
            Self::Unsupported(_) => ErrorKind::Unsupported,
            Self::Device(e) => e.kind(),
            Self::Internal(_) => ErrorKind::Internal,
        }
    }
}

impl From<sdrk_device::Error> for Error {
    fn from(e: sdrk_device::Error) -> Self {
        Self::Device(e)
    }
}
