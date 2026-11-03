//! Element-wise arithmetic of two streams (`add`, `mul` and their real variants) and the
//! conversion between a complex stream and its two real components (`interleave`,
//! `deinterleave`).
//!
//! All kernels are stateless. Unequal lengths are `LengthMismatch`, checked before anything is
//! written, so a failed call changes nothing.

use crate::error::Result;
use crate::types::{ComplexSample, RealSample};

/// Sum of two complex streams.
///
/// `C32`: `out[n] = a[n] + b[n]` per component in `f32`. `CQ15`: `sat16(a + b)` per component
/// in `i32`. Bit-exact kernel. The three lengths must be equal.
///
/// ```
/// use sdrk_core::{kernels::math::Add, Fallback, C32};
/// let a = [C32::new(1.0, 2.0)];
/// let b = [C32::new(0.5, -1.0)];
/// let mut out = [C32::default()];
/// Fallback.add(&a, &b, &mut out)?;
/// assert_eq!(out[0], C32::new(1.5, 1.0));
/// assert!(Fallback.add(&a, &[], &mut out).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Add<T: ComplexSample> {
    /// `out[n] = a[n] + b[n]`. The lengths must be equal.
    fn add(&self, a: &[T], b: &[T], out: &mut [T]) -> Result<()>;
}

/// Sum of two real streams.
///
/// `f32`: `a + b`. `Q15`: `sat16(a + b)` in `i32`. Bit-exact kernel.
///
/// ```
/// use sdrk_core::{kernels::math::AddReal, Fallback, Q15};
/// let mut out = [Q15(0)];
/// Fallback.add_real(&[Q15(30000)], &[Q15(30000)], &mut out)?;
/// assert_eq!(out[0], Q15(32767));
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait AddReal<T: RealSample> {
    /// `out[n] = a[n] + b[n]`. The lengths must be equal.
    fn add_real(&self, a: &[T], b: &[T], out: &mut [T]) -> Result<()>;
}

/// Complex product of two complex streams.
///
/// `C32`: `re = ar·br − ai·bi`, `im = ar·bi + ai·br`, the products first, then the difference
/// or sum, in `f32` without FMA. `CQ15`: `re = sat16(round_shift(ar·br − ai·bi, 15))`,
/// `im = sat16(round_shift(ar·bi + ai·br, 15))` in `i64` (so `(-1)·(-1) = 32767`). Bit-exact
/// kernel.
///
/// ```
/// use sdrk_core::{kernels::math::Mul, Fallback, C32};
/// let a = [C32::new(0.0, 1.0)];
/// let mut out = [C32::default()];
/// Fallback.mul(&a, &a, &mut out)?;
/// assert_eq!(out[0], C32::new(-1.0, 0.0));
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Mul<T: ComplexSample> {
    /// `out[n] = a[n] · b[n]` (complex product). The lengths must be equal.
    fn mul(&self, a: &[T], b: &[T], out: &mut [T]) -> Result<()>;
}

/// Product of two real streams.
///
/// `f32`: `a·b`. `Q15`: `sat16(round_shift(a·b, 15))`, the multiply of the numeric conventions.
/// Bit-exact kernel.
///
/// ```
/// use sdrk_core::{kernels::math::MulReal, Fallback, Q15};
/// let mut out = [Q15(0)];
/// Fallback.mul_real(&[Q15::MIN], &[Q15::MIN], &mut out)?;
/// assert_eq!(out[0], Q15::MAX);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait MulReal<T: RealSample> {
    /// `out[n] = a[n] · b[n]`. The lengths must be equal.
    fn mul_real(&self, a: &[T], b: &[T], out: &mut [T]) -> Result<()>;
}

/// Joins two real streams into one complex stream and splits a complex stream into two real
/// ones. Bit-exact copies.
///
/// ```
/// use sdrk_core::{kernels::math::Interleave, Fallback, C32};
/// let mut out = [C32::default(); 2];
/// Fallback.interleave(&[1.0, 2.0], Some(&[3.0, 4.0]), &mut out)?;
/// assert_eq!(out, [C32::new(1.0, 3.0), C32::new(2.0, 4.0)]);
/// let mut re = [0.0; 2];
/// Fallback.deinterleave(&out, Some(&mut re), None)?;
/// assert_eq!(re, [1.0, 2.0]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Interleave<T: ComplexSample> {
    /// `out[n] = (re[n], im[n])`; `im = None` gives `im = 0`. The lengths must be equal.
    fn interleave(&self, re: &[T::Real], im: Option<&[T::Real]>, out: &mut [T]) -> Result<()>;

    /// `re[n] = input[n].re`, `im[n] = input[n].im`; either output may be `None`. The lengths
    /// of the outputs that are given must equal the length of `input`.
    fn deinterleave(
        &self,
        input: &[T],
        re: Option<&mut [T::Real]>,
        im: Option<&mut [T::Real]>,
    ) -> Result<()>;
}
