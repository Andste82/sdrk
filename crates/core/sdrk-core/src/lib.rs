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
// `deny`, not `forbid`: the one module of x86 shuffle intrinsics of the `C32` stereo matrix
// fallback allows `unsafe` under `cfg(target_feature)` (the maintainer's decision D3005); no
// other code of the crate may.
#![deny(unsafe_code)]
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
#[cfg(all(test, not(feature = "std")))]
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
// The run-time dispatch of `sdrk-backend-simd` (decision D2840): not part of the public API,
// hidden from the documentation; `Target` has a contract (`hot` runs an `#[inline(always)]`
// closure, `NATIVE` means `hot` is never called) that only that backend's target keeps.
#[doc(hidden)]
pub use fallback::{FallbackAt, Native, Target};
#[cfg(feature = "reference")]
pub use reference::{REFERENCE_IN_PLACE_MAX, Reference};
pub use timing::*;
pub use types::*;
