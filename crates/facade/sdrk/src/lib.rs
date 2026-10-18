//! SDRk facade: per kernel, picks the fastest available backend for the
//! platform.
//!
//! Layer `crates/facade`, heap-free; depends on `sdrk-core` and the backends.
//!
//! ```
//! use sdrk as _;
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
