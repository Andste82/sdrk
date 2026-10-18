//! Device interface of SDRk: sample formats, time types and RX/TX stream traits.
//!
//! This is the lowest layer (`crates/core`). It is heap-free, has no
//! dependencies and is meant to reach 1.0 early so that external drivers can
//! rely on it.
//!
//! ```
//! use sdrk_device as _;
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
