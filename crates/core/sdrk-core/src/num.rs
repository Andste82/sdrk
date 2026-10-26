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
/// `libm::sqrtf`).
///
/// `libm` computes it in software on wasm32 (the hardware instruction of `core` is unstable
/// there, 189 fuel per `agc` sample instead of 25); with `simd128` the vector instruction does
/// it in hardware, and the scalar result is its first lane. Elsewhere `libm` already compiles to
/// the hardware instruction.
///
/// ```
/// assert_eq!(sdrk_core::num::sqrt_f32(2.25), 1.5);
/// ```
#[inline]
pub fn sqrt_f32(x: f32) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        use core::arch::wasm32::{f32x4_extract_lane, f32x4_splat, f32x4_sqrt};
        f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(x)))
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        libm::sqrtf(x)
    }
}

/// The correctly rounded square root of `x` in double precision (the same value as
/// `libm::sqrt`, in hardware on wasm32 with `simd128`, see [`sqrt_f32`]).
///
/// ```
/// assert_eq!(sdrk_core::num::sqrt_f64(2.25), 1.5);
/// ```
#[inline]
pub fn sqrt_f64(x: f64) -> f64 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        use core::arch::wasm32::{f64x2_extract_lane, f64x2_splat, f64x2_sqrt};
        f64x2_extract_lane::<0>(f64x2_sqrt(f64x2_splat(x)))
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
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
/// integers): `r = isqrt(s)`, then `r + 1` when `s - r^2 > r`. Not saturated.
#[inline]
pub(crate) fn round_sqrt(s: i64) -> i64 {
    let s = s.max(0) as u64;
    let r = isqrt_u64(s);
    let rem = s - r * r;
    (r + u64::from(rem > r)) as i64
}

/// An integer within `1/2` of `x`, for `0 <= x < 2^50` (which of the two at a tie is not
/// specified), without the range checks of a saturating `as` conversion where they cost.
///
/// x86-64: `x + 2^52` rounds `x` to the nearest integer (the ulp in `[2^52, 2^53)` is 1), and
/// its bits minus those of `2^52` are that integer; a saturating `as` costs about five
/// instructions more there. Elsewhere `trunc(x + 1/2)` (`x + 1/2` is exact below `2^50`), one
/// instruction on AArch64 (`fcvtzs`) and one operator on wasm32 (`i64.trunc_sat_f64_s`).
#[inline]
pub(crate) fn near_int_f64(x: f64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        const MAGIC: f64 = 4_503_599_627_370_496.0; // 2^52
        (x + MAGIC).to_bits() - MAGIC.to_bits()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (x + 0.5) as i64 as u64
    }
}

/// An integer within `1/2` of `x + 1/2`, for `0 <= x < 2^50`: [`near_int_f64`] of `x + 1/2`
/// on x86-64, `trunc(x) + 1` elsewhere (`x + 1/2 - (trunc(x) + 1)` is `frac(x) - 1/2`).
#[inline]
pub(crate) fn near_int_plus_half_f64(x: f64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        near_int_f64(x + 0.5)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        x as u64 + 1
    }
}

/// [`round_sqrt`] for `0 <= s <= 2^32` with one f64 square root: [`near_int_f64`] of
/// `sqrt(s)`.
///
/// Why it is exact: `s` is exact in f64 and the square root (at most `2^16`) is correctly
/// rounded, with an ulp of at most `2^-36`. The rounded root steps from `n` to `n + 1`
/// between `s = n^2 + n` and `n^2 + n + 1`, where `sqrt(s)` is `n + 1/2 - 0.25 / (n + 1/2 +
/// sqrt(s))` and `n + 1/2 + 0.75 / (n + 1/2 + sqrt(s))`: at least `0.25 / (2^17 + 1) > 2^-20`
/// away from `n + 1/2`, far more than the half ulp (`2^-37`) by which the rounded square root
/// can move. So `fl(sqrt(s))` is never within `2^-21` of a half integer, and the only integer
/// within `1/2` of it is the rounded root. The unit test checks both sides of every step for
/// every `n` up to `2^16` (both functions are monotone in `s`, so that is every `s`).
#[inline]
pub(crate) fn round_sqrt_u32(s: u64) -> i64 {
    // `s` is below 2^63, so the signed conversion (one instruction on x86-64) is exact.
    near_int_f64(sqrt_f64(s as i64 as f64)) as i64
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
