//! Dot products: a sample window with real taps (the building block of polyphase interpolators)
//! and complex samples with complex weights (the filter of the adaptive equalizer, M43).

use crate::error::Result;
use crate::types::{ComplexSample, RealSample};

/// Dot product of complex samples with real taps.
///
/// ```
/// use sdrk_core::{kernels::dot::Dot, Fallback, C32};
/// let x = [C32::new(1.0, 2.0), C32::new(3.0, -1.0)];
/// let y = Fallback.dot(&x, &[0.5, 2.0])?;
/// assert_eq!(y, C32::new(6.5, -1.0));
/// assert!(Fallback.dot(&x, &[1.0]).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Dot<T: ComplexSample> {
    /// `Σ x[i] · taps[i]` over `i` ascending. `x.len()` and `taps.len()` must be equal
    /// (`LengthMismatch` otherwise); empty slices give zero. `C32` accumulates each component
    /// in `f32`, ascending, without fused multiply-add. `CQ15` accumulates the Q30 products in
    /// `i64` and applies one `round_shift(…, 15)` and saturation at the end.
    fn dot(&self, x: &[T], taps: &[T::Real]) -> Result<T>;
}

/// Dot product of real samples with real taps.
///
/// ```
/// use sdrk_core::{kernels::dot::DotReal, Fallback, Q15};
/// let x = [Q15(16384), Q15(16384)];
/// let y = Fallback.dot_real(&x, &[Q15(16384), Q15(-8192)])?;
/// assert_eq!(y, Q15(4096)); // 0.5 * 0.5 + 0.5 * -0.25 = 0.125
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait DotReal<T: RealSample> {
    /// `Σ x[i] · taps[i]`, with the accumulation rules of [`Dot`]. The lengths must be equal.
    fn dot_real(&self, x: &[T], taps: &[T]) -> Result<T>;
}

/// The number of partial sums per component of [`DotComplex`] `C32`: element `i` goes to the
/// partial sum `i mod DOT_COMPLEX_PARTIALS`.
///
/// One value for every backend and level, so that the result does not depend on either: eight
/// complex lanes fill one 256-bit vector per component (AVX2, and AVX-512 at that width), two
/// 128-bit vectors per component (SSE2, SSE4.2, NEON, SIMD128) and the 32 single-precision
/// registers of a Cortex-M4F with the operands (decision D801).
///
/// ```
/// assert_eq!(sdrk_core::kernels::dot::DOT_COMPLEX_PARTIALS, 8);
/// ```
pub const DOT_COMPLEX_PARTIALS: usize = 8;

/// Dot product of complex samples with complex weights, `Σ x[i] · w[i]` (not conjugated).
///
/// ```
/// use sdrk_core::{kernels::dot::DotComplex, Fallback, C32};
/// let x = [C32::new(1.0, 2.0), C32::new(3.0, -1.0)];
/// let w = [C32::new(0.5, 0.0), C32::new(0.0, 1.0)];
/// // (1 + 2j) 0.5 + (3 - j) j = 0.5 + j + 1 + 3j
/// assert_eq!(Fallback.dot_complex(&x, &w)?, C32::new(1.5, 4.0));
/// assert!(Fallback.dot_complex(&x, &w[..1]).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait DotComplex<T: ComplexSample> {
    /// `Σ x[i] · w[i]`, the complex product without conjugation. `x.len()` and `w.len()` must
    /// be equal (`LengthMismatch` otherwise); empty slices give `+0`.
    ///
    /// `C32` is bit-exact on every backend and level, in this order (f32, no fused
    /// multiply-add): eight partial sums per component ([`DOT_COMPLEX_PARTIALS`]), each
    /// starting at `+0.0`; for `i` ascending, the product
    /// `p = (x.re w.re - x.im w.im, x.re w.im + x.im w.re)` (each product before the sum, in
    /// that order) is added to partial sum `k = i mod 8` (`acc[k] = acc[k] + p`); at the end,
    /// per component, `s[k] = acc[k] + acc[k + 4]` (`k` 0 to 3), `t[k] = s[k] + s[k + 2]`
    /// (`k` 0 and 1) and the result `t[0] + t[1]`. A NaN result is NaN on every backend; its
    /// sign and payload are not part of the contract.
    fn dot_complex(&self, x: &[T], w: &[T]) -> Result<T>;
}
