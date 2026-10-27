//! SDRk SIMD backend: vectorised implementations of the hot kernels for x86-64 (SSE2, AVX2 with
//! FMA, AVX-512), AArch64 (NEON) and wasm32 (SIMD128).
//!
//! Layer `crates/backends`, heap-free; depends on `sdrk-core`, `sdrk-device` and
//! `fearless_simd`. Only the facade `sdrk` knows this crate.
//!
//! # Dispatch
//!
//! Without the feature `std` the SIMD level is fixed at compile time from the enabled target
//! features (for example `-C target-cpu=x86-64-v3` selects AVX2). With the feature `std`
//! every kernel entry asks `fearless_simd` for the best level of the running CPU; the answer
//! is cached by the standard library, so the cost per call is one load and one branch.
//! Targets without a SIMD level (for example Cortex-M) run the portable `fearless_simd`
//! fallback level, which is correct but not fast; the facade uses `sdrk_core::Fallback` there.
//!
//! # Contract
//!
//! Every kernel of this crate gives the same result as the fallback of `sdrk-core` where the
//! numeric contract says bit-exact, and stays within the contract of `docs/kernels.md`
//! otherwise. `docs/backends.md` lists, per kernel, which of the two applies.
//!
//! ```
//! use sdrk_backend_simd::Simd;
//! use sdrk_core::kernels::spectrum::MagSq;
//! use sdrk_core::C32;
//! let mut out = [0.0f32; 2];
//! Simd.mag_sq(&[C32::new(3.0, 4.0), C32::new(0.0, -2.0)], &mut out)?;
//! assert_eq!(out, [25.0, 4.0]);
//! # Ok::<(), sdrk_core::Error>(())
//! ```
#![no_std]
// `#[inline(always)]` is how `fearless_simd` carries the target features of the dispatch point
// into the kernels (without it the vector code is not compiled with the vector instructions).
// The kernels use the one-letter names of the maths (`a`, `b`, `w`, `x`, `y`) and the long
// argument lists of the dispatch closures.
#![allow(
    clippy::inline_always,
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names
)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::indexing_slicing,
    clippy::undocumented_unsafe_blocks
)]

mod agc;
mod biquad;
mod convert;
mod convert_out;
mod demod;
mod fft;
mod fir;
mod level;
mod mix;
mod modulate;
mod spectrum;
mod table;
mod util;

/// The SIMD implementation of the accelerated kernels.
///
/// A unit struct, like [`sdrk_core::Fallback`]; it implements the kernel traits of
/// `sdrk-core` for the variants it accelerates. Everything else stays with the fallback in the
/// facade.
#[derive(Debug, Clone, Copy, Default)]
pub struct Simd;

/// The name of the SIMD level the kernels run at on this machine, for diagnostics and the
/// benchmark reports: `"sse2"`, `"sse4.2"`, `"avx2"`, `"avx512"`, `"neon"`, `"simd128"` or
/// `"scalar"`.
///
/// ```
/// assert!(!sdrk_backend_simd::level_name().is_empty());
/// ```
pub fn level_name() -> &'static str {
    level::name()
}
