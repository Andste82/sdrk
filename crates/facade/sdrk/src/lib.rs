//! SDRk facade: per kernel, picks the fastest available backend for the
//! platform.
//!
//! Layer `crates/facade`, heap-free; depends on `sdrk-core` and the backends.
//!
//! ```
//! use sdrk::{kernels::basic::Scale, Sdrk, C32};
//! let mut out = [C32::default()];
//! Sdrk.scale(&[C32::new(2.0, 4.0)], &mut out, 0.5)?;
//! assert_eq!(out[0], C32::new(1.0, 2.0));
//! # Ok::<(), sdrk::Error>(())
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

mod delegate;
mod delegations;

pub use sdrk_core::{
    C32, CQ15, ComplexSample, Delay, Error, Fallback, Progress, Q15, Rate, Ratio, RatioKind,
    RealSample, Result, Sample, SampleIndex, SampleType, Timing, fused, kernels, num,
};

/// The facade: implements every kernel trait for every supported sample type by
/// delegating to the best available implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sdrk;

/// Human-readable name of a kernel implementation, reported by [`backend_for`].
pub trait BackendName {
    /// Name of the implementation, e.g. `"fallback"`.
    const NAME: &'static str;
}

impl BackendName for Fallback {
    const NAME: &'static str = "fallback";
}

/// Marker of the fused kernels (`ddc`, `duc`, `fm_demod`), which the facade builds by chaining
/// its own single kernels; reported by [`backend_for`] as `"chained"`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Chained;

impl BackendName for Chained {
    const NAME: &'static str = "chained";
}

pub use delegations::backend_for;
