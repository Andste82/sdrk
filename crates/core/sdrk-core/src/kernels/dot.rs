//! Dot products of a sample window with real taps (the building block of polyphase interpolators).

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
