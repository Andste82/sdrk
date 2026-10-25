//! SDRk facade: per kernel, picks the fastest available backend for the
//! platform.
//!
//! Layer `crates/facade`, heap-free; depends on `sdrk-core` and the backends.
//!
//! The accelerated kernels (`convert`, `convert_out`, `rotate`, `fir`, `fir_decimate`,
//! `fir_interpolate`, `resample`, `mag_sq`, `quad_demod`, `fm_mod`, `fft`) run on the SIMD
//! backend (`sdrk-backend-simd`, feature `simd`, default) on x86-64, AArch64 and wasm32 with
//! `simd128`, everything else on `Fallback`. The feature `std` selects the SIMD level at run
//! time; without it the level is fixed by the target features of the build. Building with
//! `RUSTFLAGS="--cfg sdrk_fallback_only"` forces the fallback everywhere (see
//! [`SIMD_BACKEND`]).
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

#[cfg(all(
    feature = "simd",
    not(sdrk_fallback_only),
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )
))]
impl BackendName for sdrk_backend_simd::Simd {
    const NAME: &'static str = "simd";
}

/// Marker of the fused kernels (`ddc`, `duc`, `fm_demod`), which the facade builds by chaining
/// its own single kernels; reported by [`backend_for`] as `"chained"`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Chained;

impl BackendName for Chained {
    const NAME: &'static str = "chained";
}

pub use delegations::backend_for;

/// Whether the accelerated kernels run on the SIMD backend in this build: the feature `simd`
/// is on, the build does not set `--cfg sdrk_fallback_only`, and the target has SIMD
/// instructions (x86-64, AArch64, wasm32 with `simd128`).
///
/// ```
/// // `backend_for` reports the backend the constant announces.
/// let name = sdrk::backend_for("rotate", sdrk::SampleType::C32);
/// assert_eq!(name, if sdrk::SIMD_BACKEND { "simd" } else { "fallback" });
/// ```
pub const SIMD_BACKEND: bool = cfg!(all(
    feature = "simd",
    not(sdrk_fallback_only),
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )
));
