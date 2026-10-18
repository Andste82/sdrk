//! Shared error classification and the error type of this crate.

use core::fmt;

/// Coarse classification of every error in the workspace.
///
/// Each crate has its own `Error` enum that maps its variants to one of
/// these kinds through [`SdrkError::kind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// An argument has an invalid value.
    InvalidArgument,
    /// Two lengths that must match differ.
    LengthMismatch,
    /// An output buffer is too small.
    BufferTooSmall,
    /// A value is outside its permitted range.
    OutOfRange,
    /// The requested feature is not supported.
    Unsupported,
    /// A fixed-capacity container is full.
    CapacityExceeded,
    /// The operation is not valid in the current state.
    InvalidState,
    /// A resource limit was hit.
    ResourceLimit,
    /// Input could not be parsed.
    Parse,
    /// A configuration failed validation.
    Validation,
    /// An I/O operation failed.
    Io,
    /// A device reported a failure.
    Device,
    /// An operation timed out.
    Timeout,
    /// The device or peer went away.
    Disconnected,
    /// An internal invariant was violated (a bug).
    Internal,
}

impl ErrorKind {
    /// The `snake_case` name of the kind.
    ///
    /// ```
    /// use sdrk_device::ErrorKind;
    /// assert_eq!(ErrorKind::BufferTooSmall.as_str(), "buffer_too_small");
    /// ```
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::InvalidArgument => "invalid_argument",
            Self::LengthMismatch => "length_mismatch",
            Self::BufferTooSmall => "buffer_too_small",
            Self::OutOfRange => "out_of_range",
            Self::Unsupported => "unsupported",
            Self::CapacityExceeded => "capacity_exceeded",
            Self::InvalidState => "invalid_state",
            Self::ResourceLimit => "resource_limit",
            Self::Parse => "parse",
            Self::Validation => "validation",
            Self::Io => "io",
            Self::Device => "device",
            Self::Timeout => "timeout",
            Self::Disconnected => "disconnected",
            Self::Internal => "internal",
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Implemented by every error type in the workspace.
pub trait SdrkError: core::error::Error {
    /// The classification of this error.
    fn kind(&self) -> ErrorKind;
}

/// Errors of the `sdrk-device` crate.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A sample format is inconsistent (kind: `InvalidArgument`).
    InvalidFormat(&'static str),
    /// A rate or ratio has a zero numerator or denominator (kind: `InvalidArgument`).
    InvalidRate {
        /// Numerator as given.
        num: u64,
        /// Denominator as given.
        den: u64,
    },
    /// Arithmetic overflow (kind: `OutOfRange`).
    Overflow(&'static str),
    /// A buffer is too small (kind: `BufferTooSmall`).
    BufferTooSmall {
        /// Bytes needed.
        needed: usize,
        /// Bytes available.
        got: usize,
    },
}

/// Result alias of this crate.
pub type Result<T, E = Error> = core::result::Result<T, E>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat(msg) => write!(f, "invalid sample format: {msg}"),
            Self::InvalidRate { num, den } => write!(f, "invalid rate {num}/{den}"),
            Self::Overflow(msg) => write!(f, "arithmetic overflow: {msg}"),
            Self::BufferTooSmall { needed, got } => {
                write!(f, "buffer too small: needed {needed} bytes, got {got}")
            }
        }
    }
}

impl core::error::Error for Error {}

impl SdrkError for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::InvalidFormat(_) | Self::InvalidRate { .. } => ErrorKind::InvalidArgument,
            Self::Overflow(_) => ErrorKind::OutOfRange,
            Self::BufferTooSmall { .. } => ErrorKind::BufferTooSmall,
        }
    }
}
