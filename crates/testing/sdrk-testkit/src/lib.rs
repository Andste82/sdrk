//! Testkit of SDRk: signal generators, virtual devices, loopback and the
//! on-target test suite.
//!
//! Layer `crates/testing`; `no_std` by default, the `std` feature adds file
//! I/O and host helpers.
//!
//! ```
//! use sdrk_testkit as _;
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

#[cfg(feature = "std")]
extern crate std;
