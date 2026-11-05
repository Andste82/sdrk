//! Error type of `sdrk-proto`.

use core::fmt;
use sdrk_device::{ErrorKind, SdrkError};

/// Errors of the `sdrk-proto` crate. All variants are allocation-free.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A packet or payload is longer than allowed (kind: `OutOfRange`).
    TooLong {
        /// Length that was given, in bytes.
        len: usize,
        /// Largest permitted length, in bytes.
        max: usize,
    },
    /// An output buffer is too small (kind: `BufferTooSmall`).
    BufferTooSmall {
        /// Elements needed.
        needed: usize,
        /// Elements available.
        got: usize,
    },
    /// An argument has an invalid value (kind: `InvalidArgument`).
    InvalidArgument(&'static str),
}

/// The name the milestone specification uses for [`Error`].
pub type ProtoError = Error;

/// Result alias of this crate.
pub type Result<T, E = Error> = core::result::Result<T, E>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLong { len, max } => {
                write!(f, "too long: {len} bytes, at most {max} allowed")
            }
            Self::BufferTooSmall { needed, got } => {
                write!(f, "buffer too small: needed {needed}, got {got}")
            }
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl core::error::Error for Error {}

impl SdrkError for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::TooLong { .. } => ErrorKind::OutOfRange,
            Self::BufferTooSmall { .. } => ErrorKind::BufferTooSmall,
            Self::InvalidArgument(_) => ErrorKind::InvalidArgument,
        }
    }
}
