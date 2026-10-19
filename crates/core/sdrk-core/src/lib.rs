//! Core of SDRk: sample types, kernel traits and the reference and fallback
//! implementations.
//!
//! Layer `crates/core`, heap-free; depends only on `sdrk-device`.
//!
//! ```
//! use sdrk_core::{num::q15_mul, Q15};
//! assert_eq!(q15_mul(Q15(16384), Q15(16384)), Q15(8192));
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

#[cfg(test)]
extern crate std;

mod error;
mod fallback;
pub mod fused;
pub mod kernels;
mod macros;
pub mod num;
#[cfg(feature = "reference")]
mod reference;
pub mod tables;
pub mod timing;
mod types;
pub mod window;

pub use error::{Error, Result};
pub use fallback::Fallback;
#[cfg(feature = "reference")]
pub use reference::{REFERENCE_IN_PLACE_MAX, Reference};
pub use timing::*;
pub use types::*;
