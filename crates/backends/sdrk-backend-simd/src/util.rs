//! Small helpers shared by the kernels.

use fearless_simd::Simd as SimdToken;
#[cfg(all(
    feature = "relaxed-simd",
    target_arch = "wasm32",
    target_feature = "relaxed-simd"
))]
use fearless_simd::prelude::*;
use sdrk_core::{Error, Result};

/// `Ok` when both lengths are equal, else `LengthMismatch { expected, got }`.
#[inline(always)]
pub(crate) fn check_len(expected: usize, got: usize) -> Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::LengthMismatch { expected, got })
    }
}

/// The error for a violated internal invariant.
#[cold]
pub(crate) fn internal(what: &'static str) -> Error {
    Error::Internal(what)
}

/// `acc + a * b` on `f32` vectors for the `C32` accumulations of the FIR family and `mag_sq`.
///
/// With the feature `relaxed-simd` on `wasm32` built with the target feature `relaxed-simd` this
/// is one relaxed multiply-add (`f32x4.relaxed_madd`): fused or not, whichever the host runs
/// faster, so the result is not bit-exact across hosts but within the contract of
/// `docs/kernels.md`. In every other build it is the separate multiplication and addition.
#[inline(always)]
pub(crate) fn mac<S: SimdToken>(a: S::f32s, b: S::f32s, acc: S::f32s) -> S::f32s {
    #[cfg(all(
        feature = "relaxed-simd",
        target_arch = "wasm32",
        target_feature = "relaxed-simd"
    ))]
    {
        a.mul_add(b, acc)
    }
    #[cfg(not(all(
        feature = "relaxed-simd",
        target_arch = "wasm32",
        target_feature = "relaxed-simd"
    )))]
    {
        acc + a * b
    }
}
