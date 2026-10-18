//! Core of SDRk: sample types, kernel traits and the reference and fallback
//! implementations.
//!
//! Layer `crates/core`, heap-free; depends only on `sdrk-device`.
//!
//! ```
//! use sdrk_core as _;
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
