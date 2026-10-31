//! Q15 arithmetic and numeric helpers. All rules are exact (see the numeric conventions).

use crate::types::{CQ15, Q15};

/// Clamps `x` to `[-32768, 32767]`.
///
/// ```
/// assert_eq!(sdrk_core::num::sat16(40_000), 32767);
/// ```
#[inline]
pub const fn sat16(x: i64) -> i16 {
    if x > 32767 {
        32767
    } else if x < -32768 {
        -32768
    } else {
        x as i16
    }
}

/// `s == 0 -> x`, otherwise `(x + (1 << (s - 1))) >> s` (arithmetic shift, round half up).
/// `s` must be below 63; larger shifts saturate to the sign.
///
/// ```
/// assert_eq!(sdrk_core::num::round_shift(3, 1), 2);
/// assert_eq!(sdrk_core::num::round_shift(-3, 1), -1);
/// ```
#[inline]
pub const fn round_shift(x: i64, s: u32) -> i64 {
    if s == 0 {
        x
    } else if s >= 63 {
        if x < 0 { -1 } else { 0 }
    } else {
        x.saturating_add(1 << (s - 1)) >> s
    }
}

/// `sat16(round_shift(a * b, 15))`; `-1 * -1 = 32767`.
///
/// ```
/// use sdrk_core::{num::q15_mul, Q15};
/// assert_eq!(q15_mul(Q15(16384), Q15(16384)), Q15(8192));
/// assert_eq!(q15_mul(Q15::MIN, Q15::MIN), Q15::MAX);
/// ```
#[inline]
pub const fn q15_mul(a: Q15, b: Q15) -> Q15 {
    Q15(sat16(round_shift(a.0 as i64 * b.0 as i64, 15)))
}

/// Saturating addition.
#[inline]
pub const fn q15_add_sat(a: Q15, b: Q15) -> Q15 {
    Q15(a.0.saturating_add(b.0))
}

/// Saturating subtraction.
#[inline]
pub const fn q15_sub_sat(a: Q15, b: Q15) -> Q15 {
    Q15(a.0.saturating_sub(b.0))
}

/// f64 to Q15: NaN gives 0, otherwise `floor(x * 32768 + 0.5)` clamped to the Q15 range.
#[inline]
pub fn q15_from_f64(x: f64) -> Q15 {
    if x.is_nan() {
        return Q15(0);
    }
    let v = libm::floor(x * 32768.0 + 0.5);
    if v >= 32767.0 {
        Q15(32767)
    } else if v <= -32768.0 {
        Q15(-32768)
    } else {
        Q15(v as i16)
    }
}

/// f64 to a Q2.14 coefficient (scale `2^14`): `floor(x * 16384 + 0.5)`, or `None` if the
/// result is outside `-32768 ..= 32767` or `x` is NaN (so `-2.0` is exact and `+2.0` is
/// rejected).
///
/// ```
/// use sdrk_core::num::q14_from_f64;
/// assert_eq!(q14_from_f64(0.5), Some(8192));
/// assert_eq!(q14_from_f64(-2.0), Some(-32768));
/// assert_eq!(q14_from_f64(2.0), None);
/// ```
#[inline]
pub fn q14_from_f64(x: f64) -> Option<i16> {
    let v = libm::floor(x * 16384.0 + 0.5);
    if (-32768.0..=32767.0).contains(&v) {
        Some(v as i16)
    } else {
        None
    }
}

/// f32 to Q15, computed in f64 (`floor(x * 32768 + 0.5)`, NaN gives 0, saturating).
///
/// ```
/// use sdrk_core::{num::q15_from_f32, Q15};
/// assert_eq!(q15_from_f32(1.0), Q15(32767));
/// assert_eq!(q15_from_f32(f32::NAN), Q15(0));
/// ```
#[inline]
pub fn q15_from_f32(x: f32) -> Q15 {
    q15_from_f64(f64::from(x))
}

/// Complex multiplication with a single rounding and saturation per component.
///
/// ```
/// use sdrk_core::{num::cq15_mul, CQ15, Q15};
/// let j = CQ15 { re: Q15(0), im: Q15(32767) };
/// let x = CQ15 { re: Q15(1000), im: Q15(0) };
/// assert_eq!(cq15_mul(x, j).im, Q15(1000));
/// ```
#[inline]
pub const fn cq15_mul(a: CQ15, b: CQ15) -> CQ15 {
    let (ar, ai, br, bi) = (a.re.0 as i64, a.im.0 as i64, b.re.0 as i64, b.im.0 as i64);
    CQ15 {
        re: Q15(sat16(round_shift(ar * br - ai * bi, 15))),
        im: Q15(sat16(round_shift(ar * bi + ai * br, 15))),
    }
}

/// `a * conj(b)` with a single rounding and saturation per component.
#[inline]
pub const fn cq15_mul_conj(a: CQ15, b: CQ15) -> CQ15 {
    let (ar, ai, br, bi) = (a.re.0 as i64, a.im.0 as i64, b.re.0 as i64, b.im.0 as i64);
    CQ15 {
        re: Q15(sat16(round_shift(ar * br + ai * bi, 15))),
        im: Q15(sat16(round_shift(ai * br - ar * bi, 15))),
    }
}

/// Q30 accumulator: products are summed exactly in `i64`, rounded and saturated once.
///
/// ```
/// use sdrk_core::{num::Acc64, Q15};
/// let mut acc = Acc64::default();
/// acc.mac(Q15(16384), Q15(16384));
/// assert_eq!(acc.finish(), Q15(8192));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Acc64(pub i64);

impl Acc64 {
    /// Adds the product `a * b` (Q30).
    #[inline]
    pub const fn mac(&mut self, a: Q15, b: Q15) {
        self.0 = self.0.saturating_add(a.0 as i64 * b.0 as i64);
    }

    /// `sat16(round_shift(acc, 15))`.
    #[inline]
    pub const fn finish(self) -> Q15 {
        Q15(sat16(round_shift(self.0, 15)))
    }
}

/// The correctly rounded square root of `x` in single precision (the same value as
/// `libm::sqrtf`: IEEE 754 rounds the square root correctly, so every conforming instruction or
/// library gives the same bits; a negative or NaN `x` gives a NaN).
///
/// With the feature `std` it is `f32::sqrt`, LLVM's own square root (`llvm.sqrt.f32`): the
/// compiler vectorizes loops over it (`vsqrtps`, `fsqrt .4s`, `f32x4.sqrt`) and emits the
/// scalar instruction on every target that has one. Without `std` there is no stable safe way
/// to it in `core` (`core::f32::math::sqrt` is unstable, the `core::arch` intrinsics of x86 and
/// AArch64 need `unsafe` outside a `#[target_feature]` function): there `libm` computes it,
/// which is inline assembly on x86-64 (`sqrtss`) and AArch64 (`fsqrt`) that the compiler
/// cannot vectorize, and software on wasm32 (189 fuel per `agc` sample instead of 25) and the
/// Cortex-M and RISC-V targets. On wasm32 with `simd128` the vector instruction does it
/// without `std` as well, and the scalar result is its first lane.
///
/// ```
/// assert_eq!(sdrk_core::num::sqrt_f32(2.25), 1.5);
/// ```
#[inline]
pub fn sqrt_f32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(all(
        not(feature = "std"),
        target_arch = "wasm32",
        target_feature = "simd128"
    ))]
    {
        use core::arch::wasm32::{f32x4_extract_lane, f32x4_splat, f32x4_sqrt};
        f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(x)))
    }
    #[cfg(not(any(
        feature = "std",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        libm::sqrtf(x)
    }
}

/// The correctly rounded square root of `x` in double precision (the same value as
/// `libm::sqrt`; `f64::sqrt` with the feature `std`, in hardware on wasm32 with `simd128`, see
/// [`sqrt_f32`]).
///
/// ```
/// assert_eq!(sdrk_core::num::sqrt_f64(2.25), 1.5);
/// ```
#[inline]
pub fn sqrt_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(all(
        not(feature = "std"),
        target_arch = "wasm32",
        target_feature = "simd128"
    ))]
    {
        use core::arch::wasm32::{f64x2_extract_lane, f64x2_splat, f64x2_sqrt};
        f64x2_extract_lane::<0>(f64x2_sqrt(f64x2_splat(x)))
    }
    #[cfg(not(any(
        feature = "std",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        libm::sqrt(x)
    }
}

/// The exact floor of the square root of `x`.
///
/// An f64 estimate is corrected with exact integer products, so the result is right for
/// every `u64`.
///
/// ```
/// use sdrk_core::num::isqrt_u64;
/// assert_eq!(isqrt_u64(0), 0);
/// assert_eq!(isqrt_u64(15), 3);
/// assert_eq!(isqrt_u64(16), 4);
/// assert_eq!(isqrt_u64(u64::MAX), u32::MAX as u64);
/// ```
pub fn isqrt_u64(x: u64) -> u64 {
    let mut r = sqrt_f64(x as f64) as u64;
    // The estimate is within one of the root; `r * r` is computed in u128 so that the
    // correction cannot overflow near `u64::MAX`.
    while u128::from(r) * u128::from(r) > u128::from(x) {
        r -= 1;
    }
    while u128::from(r + 1) * u128::from(r + 1) <= u128::from(x) {
        r += 1;
    }
    r
}

/// The square root of `s >= 0` rounded to the nearest integer (a tie is impossible for
/// integers): `r = isqrt(s)`, then `r + 1` when `s - r^2 > r`. Not saturated. The rule of
/// `mag` `CQ15` as written in the contract; the kernels compute it in faster forms with the same
/// results, and their tests compare them with it.
#[cfg(test)]
pub(crate) fn round_sqrt(s: i64) -> i64 {
    let s = s.max(0) as u64;
    let r = isqrt_u64(s);
    let rem = s - r * r;
    (r + u64::from(rem > r)) as i64
}

/// Returns 0.0 when `|x| < 1e-30` (NaN is unchanged), otherwise `x`.
///
/// ```
/// assert_eq!(sdrk_core::num::flush_denormal(1e-31), 0.0);
/// ```
#[inline]
pub fn flush_denormal(x: f32) -> f32 {
    if libm::fabsf(x) < 1e-30 { 0.0 } else { x }
}

/// f64 variant of [`flush_denormal`].
#[inline]
pub fn flush_denormal_f64(x: f64) -> f64 {
    if libm::fabs(x) < 1e-30 { 0.0 } else { x }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::indexing_slicing)]

    use super::{sqrt_f32, sqrt_f64};
    use std::vec::Vec;

    /// The roots of `x` through a loop over a slice (the form that vectorizes, as in `mag`), each
    /// compared with `libm` and with the standard library: the same bits, or a NaN for all.
    fn check_f32(x: &[f32]) {
        let mut r = [0.0f32; 256];
        for chunk in x.chunks(r.len()) {
            for (o, &v) in r.iter_mut().zip(chunk) {
                *o = sqrt_f32(v);
            }
            for (&v, &o) in chunk.iter().zip(&r) {
                for want in [libm::sqrtf(v), v.sqrt()] {
                    assert!(
                        o.to_bits() == want.to_bits() || (o.is_nan() && want.is_nan()),
                        "sqrt_f32({v:e} = {:#010x}) = {o:e}, expected {want:e}",
                        v.to_bits()
                    );
                }
            }
        }
    }

    /// [`check_f32`] for [`sqrt_f64`].
    fn check_f64(x: &[f64]) {
        let mut r = [0.0f64; 256];
        for chunk in x.chunks(r.len()) {
            for (o, &v) in r.iter_mut().zip(chunk) {
                *o = sqrt_f64(v);
            }
            for (&v, &o) in chunk.iter().zip(&r) {
                for want in [libm::sqrt(v), v.sqrt()] {
                    assert!(
                        o.to_bits() == want.to_bits() || (o.is_nan() && want.is_nan()),
                        "sqrt_f64({v:e} = {:#018x}) = {o:e}, expected {want:e}",
                        v.to_bits()
                    );
                }
            }
        }
    }

    const SPECIAL_F32: [f32; 12] = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::EPSILON,
        -1.0,
        -f32::MIN_POSITIVE,
        -1.0e-45,
    ];

    #[test]
    fn sqrt_f32_is_bit_identical_to_libm() {
        check_f32(&SPECIAL_F32);
        // Every 7th subnormal and every 251st finite positive pattern up to the infinity, a
        // sample of the NaN payloads and of the negative patterns.
        let mut x = Vec::new();
        for (from, to, step) in [
            (0, 0x007f_ffff, 7),
            (0x0080_0000, 0x7f80_0000, 251),
            (0x7f80_0001, 0x7fff_ffff, 65_521),
            (0x8000_0000, 0xffff_ffff, 65_521),
        ] {
            x.extend((from..=to).step_by(step).map(f32::from_bits));
        }
        check_f32(&x);
    }

    #[test]
    fn sqrt_f64_is_bit_identical_to_libm() {
        check_f64(&SPECIAL_F32.map(f64::from));
        check_f64(&[f64::MIN_POSITIVE, f64::MAX, 5.0e-324, -5.0e-324, 1.8e19]);
        // A walk over all bit patterns (Weyl sequence), the subnormals and the integers below
        // 2^53 (the estimates of `isqrt_u64`).
        let mut x = Vec::new();
        let mut b = 0u64;
        for _ in 0..1 << 20 {
            b = b.wrapping_add(0x9e37_79b9_7f4a_7c15);
            x.push(f64::from_bits(b));
            x.push(f64::from_bits(b >> 12));
            x.push((b >> 11) as f64);
        }
        check_f64(&x);
    }
}
