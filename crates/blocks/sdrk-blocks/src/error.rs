//! Error type of `sdrk-blocks`.

use core::fmt;
use sdrk_device::{ErrorKind, SdrkError};

/// Errors of the `sdrk-blocks` crate. All variants are allocation-free.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// An error of a kernel or of a lower layer (kind forwarded).
    Core(sdrk_core::Error),
    /// A parameter value is invalid for a reason other than its range (kind: `InvalidArgument`).
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
        /// Why the value is invalid.
        reason: &'static str,
    },
    /// A numeric parameter is outside its permitted range (kind: `OutOfRange`).
    OutOfRange {
        /// Parameter name.
        name: &'static str,
        /// Smallest permitted value.
        min: f64,
        /// Largest permitted value.
        max: f64,
        /// The value that was given.
        got: f64,
    },
    /// The storage cannot provide a buffer (kind: `CapacityExceeded`).
    Storage {
        /// Bytes needed (including alignment padding).
        needed: usize,
        /// Bytes available.
        available: usize,
    },
    /// The parameter name is not known to the block (kind: `InvalidArgument`).
    UnknownParameter,
    /// The parameter can only be set at construction (kind: `InvalidState`).
    ReadOnlyParameter,
    /// The value has the wrong type for the parameter (kind: `InvalidArgument`).
    WrongParamType,
}

/// Result alias of this crate.
pub type Result<T, E = Error> = core::result::Result<T, E>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(e) => write!(f, "core error: {e}"),
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter {name}: {reason}")
            }
            Self::OutOfRange {
                name,
                min,
                max,
                got,
            } => write!(
                f,
                "parameter {name} out of range: {got} is not in {min} to {max}"
            ),
            Self::Storage { needed, available } => {
                write!(
                    f,
                    "storage exhausted: need {needed} bytes, have {available}"
                )
            }
            Self::UnknownParameter => f.write_str("unknown parameter"),
            Self::ReadOnlyParameter => f.write_str("parameter can only be set at construction"),
            Self::WrongParamType => f.write_str("wrong value type for the parameter"),
        }
    }
}

impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Core(e) => Some(e),
            _ => None,
        }
    }
}

impl SdrkError for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Core(e) => e.kind(),
            Self::InvalidParameter { .. } | Self::UnknownParameter | Self::WrongParamType => {
                ErrorKind::InvalidArgument
            }
            Self::OutOfRange { .. } => ErrorKind::OutOfRange,
            Self::Storage { .. } => ErrorKind::CapacityExceeded,
            Self::ReadOnlyParameter => ErrorKind::InvalidState,
        }
    }
}

impl From<sdrk_core::Error> for Error {
    fn from(e: sdrk_core::Error) -> Self {
        Self::Core(e)
    }
}

impl From<sdrk_device::Error> for Error {
    fn from(e: sdrk_device::Error) -> Self {
        Self::Core(sdrk_core::Error::Device(e))
    }
}
