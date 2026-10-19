//! Device interface of SDRk: sample formats, time types, RX/TX stream traits (synchronous and
//! async) and the optional control capabilities of a device.
//!
//! This is the lowest layer (`crates/core`). It is heap-free, has no
//! dependencies and is meant to reach 1.0 early so that external drivers can
//! rely on it.
//!
//! ```
//! use sdrk_device::{formats, Rate};
//! assert_eq!(formats::CU8.bytes_for_samples(10)?, 20);
//! assert_eq!(Rate::new(96_000, 2)?, Rate::hz(48_000)?);
//! # Ok::<(), sdrk_device::Error>(())
//! ```
#![no_std]
#![forbid(unsafe_code)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::indexing_slicing
)]

#[cfg(target_endian = "big")]
compile_error!("sdrk supports little-endian targets only");

mod asyncio;
mod capabilities;
mod control;
mod error;
mod format;
mod stream;
mod time;

pub use asyncio::{AsyncRxStream, AsyncTxStream};
pub use capabilities::{Capabilities, Direction, Range};
pub use control::{
    BandwidthControl, Command, CommandTime, Device, GainControl, HardwareTime, RateControl,
    TimedCommands, Tuner,
};
pub use error::{Error, ErrorKind, Result, SdrkError};
pub use format::{Align, Container, Endian, FullScale, Layout, SampleFormat, formats};
pub use stream::{BlockFlags, BlockMeta, RxStream, TxStream};
pub use time::{Rate, Ratio, SampleIndex, TimeAnchor, TimeSource};
