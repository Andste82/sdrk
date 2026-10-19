//! Vector twins of the cosine-table lookup of the `rotate` and `fm_mod` fallbacks.
//!
//! The results are bit-exact with `sdrk_core`'s scalar code: the same table entries, the same
//! order of f32 operations (no fused multiply-add) and the same integer arithmetic.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::tables::{COS_F32, COS_Q15};

/// A quarter turn of the 32-bit phase.
const QUARTER: u32 = 1 << 30;

/// `COS_F32[(i & 4095) + add]` for every lane (the mask keeps the index inside the table).
#[inline(always)]
fn gather_f32<S: SimdToken>(simd: S, idx: S::u32s, add: usize) -> S::f32s {
    let mut it = idx.to_array().into_iter();
    S::f32s::from_fn(simd, |_| {
        let i = it.next().unwrap_or(0) as usize & 0xFFF;
        COS_F32.get(i + add).copied().unwrap_or(0.0)
    })
}

/// `COS_Q15[(i & 4095) + add]` for every lane, widened to `i32`.
#[inline(always)]
fn gather_q15<S: SimdToken>(simd: S, idx: S::u32s, add: usize) -> S::i32s {
    let mut it = idx.to_array().into_iter();
    S::i32s::from_fn(simd, |_| {
        let i = it.next().unwrap_or(0) as usize & 0xFFF;
        i32::from(COS_Q15.get(i + add).copied().unwrap_or(0))
    })
}

/// `(cos, sin)` of the phases `p` with linear interpolation, as `cos_sin_f32` of the fallback.
#[inline(always)]
pub(crate) fn cos_sin_f32<S: SimdToken>(simd: S, p: S::u32s) -> (S::f32s, S::f32s) {
    let frac: S::i32s = (p & 0xF_FFFFu32).bitcast();
    let f = frac.to_float::<S::f32s>() * (1.0 / 1_048_576.0);
    let idx_c = p >> 20u32;
    let idx_s = (p - QUARTER) >> 20u32;
    let ac = gather_f32(simd, idx_c, 0);
    let bc = gather_f32(simd, idx_c, 1);
    let a_s = gather_f32(simd, idx_s, 0);
    let b_s = gather_f32(simd, idx_s, 1);
    (ac + (bc - ac) * f, a_s + (b_s - a_s) * f)
}

/// One interpolated Q15 table value for the index vector `idx` and the fractions `f15`.
#[inline(always)]
fn interp_q15<S: SimdToken>(simd: S, idx: S::u32s, f15: S::i32s) -> S::i32s {
    let a = gather_q15(simd, idx, 0);
    let b = gather_q15(simd, idx, 1);
    a + (((b - a) * f15 + (1 << 14)) >> 15u32)
}

/// `(cos, sin)` of the phases `p` as Q15 values in `i32` lanes, as `cos_sin_q15` of the
/// fallback.
#[inline(always)]
pub(crate) fn cos_sin_q15<S: SimdToken>(simd: S, p: S::u32s) -> (S::i32s, S::i32s) {
    let f15: S::i32s = ((p >> 5u32) & 0x7FFFu32).bitcast();
    let idx_c = p >> 20u32;
    let idx_s = (p - QUARTER) >> 20u32;
    (
        interp_q15::<S>(simd, idx_c, f15),
        interp_q15::<S>(simd, idx_s, f15),
    )
}
