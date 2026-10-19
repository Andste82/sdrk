//! File I/O of SDRk: raw sample files, SigMF recordings and WAV files as RX and TX streams.
//!
//! Layer `crates/io` (std, heap allowed); depends only on `sdrk-device`. Readers implement
//! [`sdrk_device::RxStream`], writers implement [`sdrk_device::TxStream`], so they plug into
//! the pipeline and the driver conformance suite like a device.
//!
//! ```
//! use sdrk_device::{formats, Rate, RxStream, SampleIndex};
//! use sdrk_io::RawReader;
//! let bytes = [128u8, 128, 255, 0];
//! let mut rx = RawReader::new(&bytes[..], formats::CU8, Rate::hz(2_400_000)?, SampleIndex(0), 1024)?;
//! let mut buf = [0u8; 16];
//! let (n, meta) = rx.read(&mut buf)?;
//! assert_eq!((n, meta.start.0), (4, 0));
//! # Ok::<(), sdrk_io::Error>(())
//! ```
#![forbid(unsafe_code)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::indexing_slicing
)]

mod error;
mod iso8601;
mod raw;
mod sigmf;
mod wav;

pub use error::{Error, Result};
pub use iso8601::{format_iso8601_ns, parse_iso8601_ns};
pub use raw::{FlagCounts, RawReader, RawWriter};
pub use sigmf::{
    Annotation, Capture, SIGMF_VERSION, SigmfReader, SigmfWriter, datatype_of_format,
    format_of_datatype, sigmf_paths,
};
pub use wav::{WAV_HEADER_BYTES, WavReader, WavWriter};
