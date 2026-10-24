//! `fm_mod`: frequency modulation of a real baseband signal.
//!
//! The phase steps `floor(x * k + 0.5)` are computed exactly like the fallback (in `f64`,
//! wrapped into `u32`), the running phase is a wrapping prefix sum carried from vector to
//! vector in a register, and the oscillator values come from the same table lookup as `rotate`
//! (see `table.rs`). Bit-exact with the fallback.
//!
//! # Exact steps
//!
//! `floor(v) + 1.5 * 2^52` leaves the step `floor(v)` in two's complement in the low mantissa
//! bits as long as `|v| < 2^51`; the low 32 bits are the wrapped step. The vector path accepts
//! a block of [`BLOCK`] vectors only when every `|x|` is at most a limit derived from `k` once
//! per call, so that `|x * k| <= 2^50`; NaN fails the test. The test is one integer maximum of
//! the absolute values (as bit patterns) per vector, folded into the computation of the block,
//! and one comparison per block: a block that fails is computed anyway, then the phase goes
//! back to the start of the block and the fallback, which handles every input, overwrites it;
//! the vector path continues after the samples of the fallback. A `Q15` input has `|x| <= 1`,
//! so its test is one comparison of `k` per call.
//!
//! # Level-specific paths
//!
//! Every SIMD level has its own path (rules in `level.rs`), with the generic code below as
//! the fallback: the floor is `roundpd` from SSE4.1 on, the add-and-correct of the magic
//! number on plain SSE2, the rounding mode of the addition on AVX-512 and `fcvtms` on NEON;
//! the prefix sum is an in-lane shift-and-add scan with one cross-lane step per 128-bit block
//! instead of whole-vector shifts; the phase of the last lane is broadcast in the register
//! for the next vector. With the feature `nco-tables` the lookups use the entry tables of
//! `table.rs`; without it only SIMD128 keeps a level-specific path for `C32` (it reads
//! `COS_F32` directly), every other level and variant runs the generic code.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::modulate::{FmMod, FmModState};
use sdrk_core::{C32, CQ15, Fallback, Q15, Result};

use crate::Simd;
use crate::level::{dispatch, is_sse2_baseline};
use crate::table::{cos_sin_f32, cos_sin_q15};
use crate::util::{check_len, internal};

/// `1.5 * 2^52`: adding it to an integer-valued `f64` of magnitude below `2^51` leaves the
/// two's-complement integer in the low mantissa bits.
const MAGIC: f64 = 6_755_399_441_055_744.0;
/// The bound `2^50` on `|x * k|` that the input test guarantees: with the `+ 0.5` and the
/// rounding the steps stay well inside the `2^51` of [`MAGIC`].
const STEP_BOUND: f64 = 1_125_899_906_842_624.0;
/// Vectors per block of the input test.
const BLOCK: usize = 16;
/// Samples that the fallback takes over after the vector path stopped at a block that failed
/// the input test or at the partial last vector: at least one block at the widest level.
const SCALAR_RUN: usize = BLOCK * 16;

/// The largest `|x|` that the vector path accepts for the scale `k`, as the bit pattern of an
/// `f32` (non-negative `f32` values and their bit patterns as `i32` have the same order, and
/// NaN lies above every limit); `None` when `k` is not finite.
fn input_limit(k: f64) -> Option<i32> {
    if !k.is_finite() {
        return None;
    }
    let t = (STEP_BOUND / k.abs()).min(f64::from(f32::MAX));
    let mut l = t as f32;
    if f64::from(l) > t {
        l = l.next_down();
    }
    Some(l.to_bits() as i32)
}

/// Runs `block` over the whole vectors of `h` samples of `x` (and `2h` values of `y`) in
/// blocks of [`BLOCK`] vectors (the last one may be shorter) until it returns `false` (the
/// block failed the input test); returns the number of samples of the accepted blocks.
#[inline(always)]
fn blocks<T, U>(
    h: usize,
    x: &[T],
    y: &mut [U],
    mut block: impl FnMut(&[T], &mut [U]) -> bool,
) -> usize {
    let full = x.len() / (BLOCK * h) * (BLOCK * h);
    let rest = (x.len() - full) / h * h;
    let (Some((xf, xr)), Some((yf, yr))) =
        (x.split_at_checked(full), y.split_at_mut_checked(2 * full))
    else {
        return 0;
    };
    for (i, (xb, yb)) in xf
        .chunks_exact(BLOCK * h)
        .zip(yf.chunks_exact_mut(2 * BLOCK * h))
        .enumerate()
    {
        if !block(xb, yb) {
            return i * BLOCK * h;
        }
    }
    match (xr.get(..rest), yr.get_mut(..2 * rest)) {
        (Some(xb), Some(yb)) if rest > 0 => {
            if block(xb, yb) {
                full + rest
            } else {
                full
            }
        }
        _ => full,
    }
}

/// `(floor(v) as i64) as u32` of two vectors of `f64` of magnitude below `2^51` without a
/// vector floor (plain SSE2, where `fearless_simd` falls back to `libm::floor` per lane):
/// `v + MAGIC` rounds to the nearest integer and leaves it in the low mantissa bits; where
/// that rounded up (`(v + MAGIC) - MAGIC > v`) one is subtracted.
#[inline(always)]
fn floor_wrap_magic<S: SimdToken>(simd: S, v_lo: S::f64s, v_hi: S::f64s) -> S::u32s {
    let magic = S::f64s::splat(simd, MAGIC);
    let m_lo = v_lo + magic;
    let m_hi = v_hi + magic;
    let ones = S::f64s::splat(simd, f64::from_bits(u64::MAX));
    let zero = S::f64s::splat(simd, 0.0);
    let up_lo: S::u32s = (m_lo - magic).simd_gt(v_lo).select(ones, zero).bitcast();
    let up_hi: S::u32s = (m_hi - magic).simd_gt(v_hi).select(ones, zero).bitcast();
    let a: S::u32s = m_lo.bitcast();
    let b: S::u32s = m_hi.bitcast();
    // The mask is all ones (-1) where the sum rounded up.
    a.unzip_low(b) + up_lo.unzip_low(up_hi)
}

/// The wrapped phase steps `(floor(x * k + 0.5) as i64) as u32` of `h` f32 values with
/// `|x * k| <= 2^50` (generic code).
#[inline(always)]
fn steps<S: SimdToken>(simd: S, x: S::f32s, k: f64) -> S::u32s {
    let (lo, hi) = x.widen();
    let kk = S::f64s::splat(simd, k);
    let v_lo = lo * kk + 0.5;
    let v_hi = hi * kk + 0.5;
    if is_sse2_baseline(simd) {
        return floor_wrap_magic(simd, v_lo, v_hi);
    }
    let magic = S::f64s::splat(simd, MAGIC);
    let a: S::u32s = (v_lo.floor() + magic).bitcast();
    let b: S::u32s = (v_hi.floor() + magic).bitcast();
    a.unzip_low(b)
}

/// Inclusive wrapping prefix sum over the lanes (generic code).
#[inline(always)]
fn prefix_sum<S: SimdToken>(v: S::u32s) -> S::u32s {
    let h = S::u32s::LEN;
    let mut s = v;
    if h > 1 {
        s = s + s.shift_elements_right::<1>(0);
    }
    if h > 2 {
        s = s + s.shift_elements_right::<2>(0);
    }
    if h > 4 {
        s = s + s.shift_elements_right::<4>(0);
    }
    if h > 8 {
        s = s + s.shift_elements_right::<8>(0);
    }
    s
}

/// Modulates whole vectors of `f32` input while the blocks pass the input test (`limit` from
/// [`input_limit`]); returns the number of samples done and the phase after them.
#[inline(always)]
fn fm_mod_f32<S: SimdToken>(
    simd: S,
    k: f64,
    limit: i32,
    phase: u32,
    x: &[f32],
    y: &mut [f32],
) -> (usize, u32) {
    let level = simd.level();
    #[cfg(all(
        feature = "nco-tables",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        if let Some(t) = level.as_avx512() {
            return x86::f32_avx512(t, k, limit, phase, x, y);
        }
        if let Some(t) = level.as_avx2() {
            return x86::f32_avx2(t, k, limit, phase, x, y);
        }
        if let Some(t) = level.as_sse4_2() {
            return x86::f32_sse4_2(t, k, limit, phase, x, y);
        }
        if let Some(t) = crate::level::sse2_only(level) {
            return x86::f32_sse2(t, k, limit, phase, x, y);
        }
    }
    #[cfg(all(feature = "nco-tables", target_arch = "aarch64"))]
    if let Some(t) = level.as_neon() {
        return neon::f32_neon(t, k, limit, phase, x, y);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return wasm::f32_simd128(t, k, limit, phase, x, y);
    }
    let _ = level;
    let h = S::f32s::LEN;
    let lim = S::f32s::splat(simd, f32::from_bits(limit as u32));
    let mut phase = phase;
    let done = blocks(h, x, y, |xb, yb| {
        let start = phase;
        let mut ok = lim.simd_ge(S::f32s::splat(simd, 0.0));
        for (xv, yv) in xb.chunks_exact(h).zip(yb.chunks_exact_mut(2 * h)) {
            let x = S::f32s::from_slice(simd, xv);
            ok &= x.abs().simd_le(lim);
            let p = S::u32s::splat(simd, phase) + prefix_sum::<S>(steps(simd, x, k));
            let (o0, o1) = cos_sin_f32(simd, p);
            let (d0, d1) = yv.split_at_mut(h);
            o0.store_slice(d0);
            o1.store_slice(d1);
            phase = p.as_slice().last().copied().unwrap_or(phase);
        }
        let ok = ok.all_true();
        if !ok {
            phase = start;
        }
        ok
    });
    (done, phase)
}

/// `CQ15` variant over whole vectors (`k` with `k <= 2^50`); returns the number of samples
/// done and the phase after them.
#[inline(always)]
fn fm_mod_q15<S: SimdToken>(simd: S, k: f64, phase: u32, x: &[i16], y: &mut [i16]) -> (usize, u32) {
    let level = simd.level();
    #[cfg(all(
        feature = "nco-tables",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        let k15 = k * (1.0 / 32768.0);
        if let Some(t) = level.as_avx512() {
            return x86::q15_avx512(t, k15, phase, x, y);
        }
        if let Some(t) = level.as_avx2() {
            return x86::q15_avx2(t, k15, phase, x, y);
        }
        if let Some(t) = level.as_sse4_2() {
            return x86::q15_sse4_2(t, k15, phase, x, y);
        }
        if let Some(t) = crate::level::sse2_only(level) {
            return x86::q15_sse2(t, k15, phase, x, y);
        }
    }
    #[cfg(all(feature = "nco-tables", target_arch = "aarch64"))]
    if let Some(t) = level.as_neon() {
        return neon::q15_neon(t, k, phase, x, y);
    }
    #[cfg(all(
        feature = "nco-tables",
        target_arch = "wasm32",
        target_feature = "simd128"
    ))]
    if let Some(t) = level.as_wasm_simd128() {
        return wasm::q15_simd128(t, k * (1.0 / 32768.0), phase, x, y);
    }
    let _ = level;
    // Two vectors of `i32` lanes per vector of `i16` input.
    let h = S::i32s::LEN;
    let mut phase = phase;
    let done = blocks(2 * h, x, y, |xb, yb| {
        for (xv, yv) in xb.chunks_exact(2 * h).zip(yb.chunks_exact_mut(4 * h)) {
            let (lo, hi) = S::i16s::from_slice(simd, xv).widen();
            for (xi, o) in [lo, hi].into_iter().zip(yv.chunks_exact_mut(2 * h)) {
                // `i16` to `f32` is exact; `q / 32768` in `f64` is exact too.
                let st = steps(simd, xi.to_float::<S::f32s>() * (1.0 / 32768.0), k);
                let p = S::u32s::splat(simd, phase) + prefix_sum::<S>(st);
                let (c, s) = cos_sin_q15(simd, p);
                let packed = (c & 0xFFFFi32) | (s << 16u32);
                packed.bitcast::<S::i16s>().store_slice(o);
                phase = p.as_slice().last().copied().unwrap_or(phase);
            }
        }
        true
    });
    (done, phase)
}

impl FmMod<C32> for Simd {
    fn fm_mod(&self, s: &mut FmModState, input: &[f32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let Some(limit) = input_limit(s.scale()) else {
            return Fallback.fm_mod(s, input, output);
        };
        let mut at = 0;
        while at < input.len() {
            let (Some(x), Some(y)) = (input.get(at..), output.get_mut(at..)) else {
                return Err(internal("fm_mod block"));
            };
            let y: &mut [f32] = bytemuck::cast_slice_mut(y);
            let (k, phase) = s.parts_mut();
            let (done, p) = dispatch!(simd => fm_mod_f32(simd, k, limit, *phase, x, y));
            *phase = p;
            at += done;
            let end = input.len().min(at + SCALAR_RUN);
            match (input.get(at..end), output.get_mut(at..end)) {
                (Some(x), Some(y)) => Fallback.fm_mod(s, x, y)?,
                _ => return Err(internal("fm_mod block")),
            }
            at = end;
        }
        Ok(())
    }
}

impl FmMod<CQ15> for Simd {
    fn fm_mod(&self, s: &mut FmModState, input: &[Q15], output: &mut [CQ15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (k, phase) = s.parts_mut();
        // `|x| <= 1`; the level paths scale the integer input by `k / 2^15`, which must be
        // exact for that to give the same products.
        if !(k.abs() <= STEP_BOUND && k * (1.0 / 32768.0) * 32768.0 == k) {
            return Fallback.fm_mod(s, input, output);
        }
        let x: &[i16] = bytemuck::cast_slice(input);
        let y: &mut [i16] = bytemuck::cast_slice_mut(output);
        let (done, p) = dispatch!(simd => fm_mod_q15(simd, k, *phase, x, y));
        *phase = p;
        match (input.get(done..), output.get_mut(done..)) {
            (Some(x), Some(y)) => Fallback.fm_mod(s, x, y),
            _ => Err(internal("fm_mod tail")),
        }
    }
}

/// The level-specific paths of x86. The helpers carry the target features of the lowest level
/// that uses them and are inlined into the kernels.
#[cfg(all(
    feature = "nco-tables",
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m128, __m128d, __m128i, __m256, __m256d, __m256i, __m512, __m512d, __m512i,
        _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEG_INF, _mm_add_epi32, _mm_add_pd, _mm_and_si128,
        _mm_castpd_ps, _mm_castps_si128, _mm_cmpgt_epi32, _mm_cmplt_epi16, _mm_cmplt_pd,
        _mm_cvtepi16_epi32, _mm_cvtepi32_pd, _mm_cvtps_pd, _mm_cvtsi128_si32, _mm_floor_pd,
        _mm_max_epi16, _mm_max_epi32, _mm_movehl_ps, _mm_movemask_epi8, _mm_mul_pd, _mm_set1_epi16,
        _mm_set1_epi32, _mm_set1_pd, _mm_setzero_si128, _mm_shuffle_epi32, _mm_shuffle_ps,
        _mm_slli_si128, _mm_srai_epi32, _mm_sub_pd, _mm_unpackhi_epi16, _mm_unpackhi_epi64,
        _mm_unpacklo_epi16, _mm256_add_epi32, _mm256_add_pd, _mm256_and_si256, _mm256_castpd_ps,
        _mm256_castpd_si256, _mm256_castps_pd, _mm256_castps_si256, _mm256_castps256_ps128,
        _mm256_cmpgt_epi32, _mm256_cvtepi32_pd, _mm256_cvtps_pd, _mm256_cvtsi256_si32,
        _mm256_extractf128_ps, _mm256_floor_pd, _mm256_max_epi32, _mm256_movemask_epi8,
        _mm256_mul_pd, _mm256_permute2x128_si256, _mm256_permute4x64_pd,
        _mm256_permutevar8x32_epi32, _mm256_set1_epi32, _mm256_set1_pd, _mm256_setzero_si256,
        _mm256_shuffle_epi32, _mm256_shuffle_ps, _mm256_slli_si256, _mm512_add_epi32,
        _mm512_add_pd, _mm512_add_round_pd, _mm512_and_si512, _mm512_bslli_epi128,
        _mm512_castpd_si512, _mm512_castps_pd, _mm512_castps_si512, _mm512_castps512_ps256,
        _mm512_castsi512_si256, _mm512_cmpgt_epi32_mask, _mm512_cvtepi16_epi32, _mm512_cvtepi32_pd,
        _mm512_cvtps_pd, _mm512_cvtsi512_si32, _mm512_extractf64x4_pd, _mm512_extracti64x4_epi64,
        _mm512_maskz_permutexvar_epi32, _mm512_max_epi32, _mm512_mul_pd, _mm512_permutex2var_epi32,
        _mm512_permutexvar_epi32, _mm512_set1_epi32, _mm512_set1_pd, _mm512_setr_epi32,
        _mm512_setzero_si512,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128, __m128d, __m128i, __m256, __m256d, __m256i, __m512, __m512d, __m512i,
        _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEG_INF, _mm_add_epi32, _mm_add_pd, _mm_and_si128,
        _mm_castpd_ps, _mm_castps_si128, _mm_cmpgt_epi32, _mm_cmplt_epi16, _mm_cmplt_pd,
        _mm_cvtepi16_epi32, _mm_cvtepi32_pd, _mm_cvtps_pd, _mm_cvtsi128_si32, _mm_floor_pd,
        _mm_max_epi16, _mm_max_epi32, _mm_movehl_ps, _mm_movemask_epi8, _mm_mul_pd, _mm_set1_epi16,
        _mm_set1_epi32, _mm_set1_pd, _mm_setzero_si128, _mm_shuffle_epi32, _mm_shuffle_ps,
        _mm_slli_si128, _mm_srai_epi32, _mm_sub_pd, _mm_unpackhi_epi16, _mm_unpackhi_epi64,
        _mm_unpacklo_epi16, _mm256_add_epi32, _mm256_add_pd, _mm256_and_si256, _mm256_castpd_ps,
        _mm256_castpd_si256, _mm256_castps_pd, _mm256_castps_si256, _mm256_castps256_ps128,
        _mm256_cmpgt_epi32, _mm256_cvtepi32_pd, _mm256_cvtps_pd, _mm256_cvtsi256_si32,
        _mm256_extractf128_ps, _mm256_floor_pd, _mm256_max_epi32, _mm256_movemask_epi8,
        _mm256_mul_pd, _mm256_permute2x128_si256, _mm256_permute4x64_pd,
        _mm256_permutevar8x32_epi32, _mm256_set1_epi32, _mm256_set1_pd, _mm256_setzero_si256,
        _mm256_shuffle_epi32, _mm256_shuffle_ps, _mm256_slli_si256, _mm512_add_epi32,
        _mm512_add_pd, _mm512_add_round_pd, _mm512_and_si512, _mm512_bslli_epi128,
        _mm512_castpd_si512, _mm512_castps_pd, _mm512_castps_si512, _mm512_castps512_ps256,
        _mm512_castsi512_si256, _mm512_cmpgt_epi32_mask, _mm512_cvtepi16_epi32, _mm512_cvtepi32_pd,
        _mm512_cvtps_pd, _mm512_cvtsi512_si32, _mm512_extractf64x4_pd, _mm512_extracti64x4_epi64,
        _mm512_maskz_permutexvar_epi32, _mm512_max_epi32, _mm512_mul_pd, _mm512_permutex2var_epi32,
        _mm512_permutexvar_epi32, _mm512_set1_epi32, _mm512_set1_pd, _mm512_setr_epi32,
        _mm512_setzero_si512,
    };

    use fearless_simd::{f32x4, f32x8, f32x16, i16x8, i16x16, prelude::*};

    use super::{MAGIC, blocks};
    use crate::table::x86::{
        cos_sin_f32_x4, cos_sin_f32_x8, cos_sin_f32_x16, cos_sin_q15_x4, cos_sin_q15_x4_sse2,
        cos_sin_q15_x8, cos_sin_q15_x16,
    };

    /// The running maximum of the input test on SSE2, which has no 32-bit maximum: the 16-bit
    /// maximum of the high halves of the absolute values (the low halves are cleared).
    #[target_feature(enable = "sse2")]
    #[inline]
    fn max_abs_sse2(m: __m128i, x: __m128) -> __m128i {
        _mm_max_epi16(
            m,
            _mm_and_si128(_mm_castps_si128(x), _mm_set1_epi32(0x7FFF_0000)),
        )
    }

    /// The input test of a block on SSE2: every high half below that of the limit, which is
    /// stricter than `|x| <= limit` (a value just below the limit takes the fallback).
    #[target_feature(enable = "sse2")]
    #[inline]
    fn in_range_sse2(m: __m128i, limit: i32) -> bool {
        let lim = _mm_set1_epi16((limit >> 16) as i16);
        _mm_movemask_epi8(_mm_cmplt_epi16(m, lim)) == 0xFFFF
    }

    /// The running maximum of the absolute values (as bit patterns) of the input test.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn max_abs(m: __m128i, x: __m128) -> __m128i {
        _mm_max_epi32(
            m,
            _mm_and_si128(_mm_castps_si128(x), _mm_set1_epi32(i32::MAX)),
        )
    }

    /// The input test of a block: every `|x| <= limit`.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn in_range(m: __m128i, limit: i32) -> bool {
        _mm_movemask_epi8(_mm_cmpgt_epi32(m, _mm_set1_epi32(limit))) == 0
    }

    /// The low 32 bits of the integer-valued sums `a + MAGIC` and `b + MAGIC`, in lane order.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn low_words(a: __m128d, b: __m128d) -> __m128i {
        _mm_castps_si128(_mm_shuffle_ps::<0x88>(_mm_castpd_ps(a), _mm_castpd_ps(b)))
    }

    /// The wrapped steps of four values `v = x * k + 0.5` of magnitude below `2^51`, floor by
    /// `roundpd`.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn floor_x4(lo: __m128d, hi: __m128d) -> __m128i {
        let magic = _mm_set1_pd(MAGIC);
        low_words(
            _mm_add_pd(_mm_floor_pd(lo), magic),
            _mm_add_pd(_mm_floor_pd(hi), magic),
        )
    }

    /// [`floor_x4`] on SSE2: round to nearest through the magic number, then one less where
    /// that rounded up.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn floor_x4_sse2(lo: __m128d, hi: __m128d) -> __m128i {
        let magic = _mm_set1_pd(MAGIC);
        let (m_lo, m_hi) = (_mm_add_pd(lo, magic), _mm_add_pd(hi, magic));
        let up_lo = _mm_cmplt_pd(lo, _mm_sub_pd(m_lo, magic));
        let up_hi = _mm_cmplt_pd(hi, _mm_sub_pd(m_hi, magic));
        // The mask is all ones (-1) where the sum rounded up.
        _mm_add_epi32(low_words(m_lo, m_hi), low_words(up_lo, up_hi))
    }

    /// `x * k + 0.5` of four `f32` values, as two vectors of `f64`.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn scaled_x4(x: __m128, k: __m128d) -> (__m128d, __m128d) {
        let half = _mm_set1_pd(0.5);
        let lo = _mm_cvtps_pd(x);
        let hi = _mm_cvtps_pd(_mm_movehl_ps(x, x));
        (
            _mm_add_pd(_mm_mul_pd(lo, k), half),
            _mm_add_pd(_mm_mul_pd(hi, k), half),
        )
    }

    /// `q * k15 + 0.5` of four integers.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn scaled_q15_x4(q: __m128i, k15: __m128d) -> (__m128d, __m128d) {
        let half = _mm_set1_pd(0.5);
        let lo = _mm_cvtepi32_pd(q);
        let hi = _mm_cvtepi32_pd(_mm_shuffle_epi32::<0xEE>(q));
        (
            _mm_add_pd(_mm_mul_pd(lo, k15), half),
            _mm_add_pd(_mm_mul_pd(hi, k15), half),
        )
    }

    /// The phases `carry + inclusive prefix sum of st` of four steps.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn scan_x4(st: __m128i, carry: __m128i) -> __m128i {
        let s = _mm_add_epi32(st, _mm_slli_si128::<4>(st));
        let s = _mm_add_epi32(s, _mm_slli_si128::<8>(s));
        _mm_add_epi32(s, carry)
    }

    /// The modulator over vectors of four samples, with the input test and the floor of the
    /// level.
    macro_rules! f32_x4 {
        ($t:ident, $k:ident, $limit:ident, $phase:ident, $x:ident, $y:ident,
         $max_abs:ident, $in_range:ident, $floor:ident) => {{
            let kk = _mm_set1_pd($k);
            let mut carry = _mm_set1_epi32($phase as i32);
            let done = blocks(4, $x, $y, |xb, yb| {
                let start = carry;
                let mut m = _mm_setzero_si128();
                for (xv, yv) in xb.chunks_exact(4).zip(yb.chunks_exact_mut(8)) {
                    let x: __m128 = f32x4::from_slice($t, xv).into();
                    m = $max_abs(m, x);
                    let (lo, hi) = scaled_x4(x, kk);
                    let p = scan_x4($floor(lo, hi), carry);
                    carry = _mm_shuffle_epi32::<0xFF>(p);
                    let (o0, o1) = cos_sin_f32_x4($t, p);
                    let (d0, d1) = yv.split_at_mut(4);
                    f32x4::simd_from($t, o0).store_slice(d0);
                    f32x4::simd_from($t, o1).store_slice(d1);
                }
                let ok = $in_range(m, $limit);
                if !ok {
                    carry = start;
                }
                ok
            });
            (done, _mm_cvtsi128_si32(carry) as u32)
        }};
    }

    /// The `CQ15` modulator over vectors of eight samples (two phase vectors), with the sign
    /// extension, the floor and the lookup of the level.
    macro_rules! q15_x4 {
        ($t:ident, $k15:ident, $phase:ident, $x:ident, $y:ident,
         $widen:ident, $floor:ident, $lookup:ident) => {{
            let kk = _mm_set1_pd($k15);
            let mut carry = _mm_set1_epi32($phase as i32);
            let done = blocks(8, $x, $y, |xb, yb| {
                for (xv, yv) in xb.chunks_exact(8).zip(yb.chunks_exact_mut(16)) {
                    let (lo, hi) = $widen(i16x8::from_slice($t, xv).into());
                    for (q, o) in [lo, hi].into_iter().zip(yv.chunks_exact_mut(8)) {
                        let (a, b) = scaled_q15_x4(q, kk);
                        let p = scan_x4($floor(a, b), carry);
                        carry = _mm_shuffle_epi32::<0xFF>(p);
                        i16x8::simd_from($t, $lookup(p)).store_slice(o);
                    }
                }
                true
            });
            (done, _mm_cvtsi128_si32(carry) as u32)
        }};
    }

    /// Sign extension of eight `i16` to two vectors of four `i32` (SSE2).
    #[target_feature(enable = "sse2")]
    #[inline]
    fn widen_sse2(x: __m128i) -> (__m128i, __m128i) {
        (
            _mm_srai_epi32::<16>(_mm_unpacklo_epi16(x, x)),
            _mm_srai_epi32::<16>(_mm_unpackhi_epi16(x, x)),
        )
    }

    /// Sign extension of eight `i16` to two vectors of four `i32` (SSE4.1).
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn widen_sse4_1(x: __m128i) -> (__m128i, __m128i) {
        (
            _mm_cvtepi16_epi32(x),
            _mm_cvtepi16_epi32(_mm_unpackhi_epi64(x, x)),
        )
    }

    fearless_simd::kernel!(
        pub(super) fn f32_sse2(
            t: Sse2,
            k: f64,
            limit: i32,
            phase: u32,
            x: &[f32],
            y: &mut [f32],
        ) -> (usize, u32) {
            f32_x4!(
                t,
                k,
                limit,
                phase,
                x,
                y,
                max_abs_sse2,
                in_range_sse2,
                floor_x4_sse2
            )
        }
    );

    fearless_simd::kernel!(
        pub(super) fn f32_sse4_2(
            t: Sse4_2,
            k: f64,
            limit: i32,
            phase: u32,
            x: &[f32],
            y: &mut [f32],
        ) -> (usize, u32) {
            f32_x4!(t, k, limit, phase, x, y, max_abs, in_range, floor_x4)
        }
    );

    fearless_simd::kernel!(
        pub(super) fn q15_sse2(
            t: Sse2,
            k15: f64,
            phase: u32,
            x: &[i16],
            y: &mut [i16],
        ) -> (usize, u32) {
            q15_x4!(
                t,
                k15,
                phase,
                x,
                y,
                widen_sse2,
                floor_x4_sse2,
                cos_sin_q15_x4_sse2
            )
        }
    );

    fearless_simd::kernel!(
        pub(super) fn q15_sse4_2(
            t: Sse4_2,
            k15: f64,
            phase: u32,
            x: &[i16],
            y: &mut [i16],
        ) -> (usize, u32) {
            q15_x4!(t, k15, phase, x, y, widen_sse4_1, floor_x4, cos_sin_q15_x4)
        }
    );

    /// The wrapped steps of `v = lo | hi` (eight values of magnitude below `2^51`), in lane
    /// order.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn floor_x8(lo: __m256d, hi: __m256d) -> __m256i {
        let magic = _mm256_set1_pd(MAGIC);
        let a = _mm256_castpd_ps(_mm256_add_pd(_mm256_floor_pd(lo), magic));
        let b = _mm256_castpd_ps(_mm256_add_pd(_mm256_floor_pd(hi), magic));
        // [a0 a1 b0 b1 | a2 a3 b2 b3] in 32-bit lanes, then the middle 64-bit pairs swapped.
        let w = _mm256_castps_pd(_mm256_shuffle_ps::<0x88>(a, b));
        _mm256_castpd_si256(_mm256_permute4x64_pd::<0xD8>(w))
    }

    /// The phases `carry + inclusive prefix sum of st` of eight steps: an in-lane scan, then
    /// the total of the low half added to the high half.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn scan_x8(st: __m256i, carry: __m256i) -> __m256i {
        let s = _mm256_add_epi32(st, _mm256_slli_si256::<4>(st));
        let s = _mm256_add_epi32(s, _mm256_slli_si256::<8>(s));
        let low = _mm256_shuffle_epi32::<0xFF>(s);
        let low = _mm256_permute2x128_si256::<0x08>(low, low);
        _mm256_add_epi32(_mm256_add_epi32(s, low), carry)
    }

    /// `x * k + 0.5` of eight `f32` values, as two vectors of `f64`.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn scaled_x8(x: __m256, k: __m256d) -> (__m256d, __m256d) {
        let half = _mm256_set1_pd(0.5);
        let lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));
        let hi = _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(x));
        (
            _mm256_add_pd(_mm256_mul_pd(lo, k), half),
            _mm256_add_pd(_mm256_mul_pd(hi, k), half),
        )
    }

    fearless_simd::kernel!(
        pub(super) fn f32_avx2(
            t: Avx2,
            k: f64,
            limit: i32,
            phase: u32,
            x: &[f32],
            y: &mut [f32],
        ) -> (usize, u32) {
            let kk = _mm256_set1_pd(k);
            let abs = _mm256_set1_epi32(i32::MAX);
            let last = _mm256_set1_epi32(7);
            let mut carry = _mm256_set1_epi32(phase as i32);
            let done = blocks(8, x, y, |xb, yb| {
                let start = carry;
                let mut m = _mm256_setzero_si256();
                for (xv, yv) in xb.chunks_exact(8).zip(yb.chunks_exact_mut(16)) {
                    let x: __m256 = f32x8::from_slice(t, xv).into();
                    m = _mm256_max_epi32(m, _mm256_and_si256(_mm256_castps_si256(x), abs));
                    let (lo, hi) = scaled_x8(x, kk);
                    let p = scan_x8(floor_x8(lo, hi), carry);
                    carry = _mm256_permutevar8x32_epi32(p, last);
                    let (o0, o1) = cos_sin_f32_x8(t, p);
                    let (d0, d1) = yv.split_at_mut(8);
                    f32x8::simd_from(t, o0).store_slice(d0);
                    f32x8::simd_from(t, o1).store_slice(d1);
                }
                let over = _mm256_cmpgt_epi32(m, _mm256_set1_epi32(limit));
                let ok = _mm256_movemask_epi8(over) == 0;
                if !ok {
                    carry = start;
                }
                ok
            });
            (done, _mm256_cvtsi256_si32(carry) as u32)
        }
    );

    fearless_simd::kernel!(
        pub(super) fn q15_avx2(
            t: Avx2,
            k15: f64,
            phase: u32,
            x: &[i16],
            y: &mut [i16],
        ) -> (usize, u32) {
            let kk = _mm256_set1_pd(k15);
            let half = _mm256_set1_pd(0.5);
            let last = _mm256_set1_epi32(7);
            let mut carry = _mm256_set1_epi32(phase as i32);
            let done = blocks(8, x, y, |xb, yb| {
                for (xv, yv) in xb.chunks_exact(8).zip(yb.chunks_exact_mut(16)) {
                    let q: __m128i = i16x8::from_slice(t, xv).into();
                    let lo = _mm256_cvtepi32_pd(_mm_cvtepi16_epi32(q));
                    let hi = _mm256_cvtepi32_pd(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(q, q)));
                    let lo = _mm256_add_pd(_mm256_mul_pd(lo, kk), half);
                    let hi = _mm256_add_pd(_mm256_mul_pd(hi, kk), half);
                    let p = scan_x8(floor_x8(lo, hi), carry);
                    carry = _mm256_permutevar8x32_epi32(p, last);
                    i16x16::simd_from(t, cos_sin_q15_x8(p)).store_slice(yv);
                }
                true
            });
            (done, _mm256_cvtsi256_si32(carry) as u32)
        }
    );

    /// The wrapped steps of sixteen values `v = lo | hi` of magnitude below `2^51`: the
    /// addition of the magic number rounds toward minus infinity, which gives `floor(v)` at
    /// once.
    #[target_feature(enable = "avx512f")]
    #[inline]
    fn floor_x16(lo: __m512d, hi: __m512d) -> __m512i {
        const DOWN: i32 = _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC;
        let magic = _mm512_set1_pd(MAGIC);
        let a = _mm512_castpd_si512(_mm512_add_round_pd::<DOWN>(lo, magic));
        let b = _mm512_castpd_si512(_mm512_add_round_pd::<DOWN>(hi, magic));
        let even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        _mm512_permutex2var_epi32(a, even, b)
    }

    /// The phases `carry + inclusive prefix sum of st` of sixteen steps: an in-lane scan, then
    /// the totals of the lower 128-bit blocks added in two steps.
    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    fn scan_x16(st: __m512i, carry: __m512i) -> __m512i {
        let s = _mm512_add_epi32(st, _mm512_bslli_epi128::<4>(st));
        let s = _mm512_add_epi32(s, _mm512_bslli_epi128::<8>(s));
        // Blocks 1 to 3 add the last lane of the block before, blocks 2 and 3 then the last
        // lane of the block two before (which already holds the sum of two blocks).
        let one = _mm512_setr_epi32(0, 0, 0, 0, 3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11);
        let s = _mm512_add_epi32(s, _mm512_maskz_permutexvar_epi32(0xFFF0, one, s));
        let two = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 7, 7, 7, 7);
        let s = _mm512_add_epi32(s, _mm512_maskz_permutexvar_epi32(0xFF00, two, s));
        _mm512_add_epi32(s, carry)
    }

    /// `x * k + 0.5` of sixteen `f32` values, as two vectors of `f64`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    fn scaled_x16(x: __m512, k: __m512d) -> (__m512d, __m512d) {
        let half = _mm512_set1_pd(0.5);
        let lo = _mm512_cvtps_pd(_mm512_castps512_ps256(x));
        let hi = _mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(
            _mm512_castps_pd(x),
        )));
        (
            _mm512_add_pd(_mm512_mul_pd(lo, k), half),
            _mm512_add_pd(_mm512_mul_pd(hi, k), half),
        )
    }

    fearless_simd::kernel!(
        pub(super) fn f32_avx512(
            t: Avx512,
            k: f64,
            limit: i32,
            phase: u32,
            x: &[f32],
            y: &mut [f32],
        ) -> (usize, u32) {
            let kk = _mm512_set1_pd(k);
            let abs = _mm512_set1_epi32(i32::MAX);
            let last = _mm512_set1_epi32(15);
            let mut carry = _mm512_set1_epi32(phase as i32);
            let done = blocks(16, x, y, |xb, yb| {
                let start = carry;
                let mut m = _mm512_setzero_si512();
                for (xv, yv) in xb.chunks_exact(16).zip(yb.chunks_exact_mut(32)) {
                    let x: __m512 = f32x16::from_slice(t, xv).into();
                    m = _mm512_max_epi32(m, _mm512_and_si512(_mm512_castps_si512(x), abs));
                    let (lo, hi) = scaled_x16(x, kk);
                    let p = scan_x16(floor_x16(lo, hi), carry);
                    carry = _mm512_permutexvar_epi32(last, p);
                    let (o0, o1) = cos_sin_f32_x16(p);
                    let (d0, d1) = yv.split_at_mut(16);
                    f32x16::simd_from(t, o0).store_slice(d0);
                    f32x16::simd_from(t, o1).store_slice(d1);
                }
                let ok = _mm512_cmpgt_epi32_mask(m, _mm512_set1_epi32(limit)) == 0;
                if !ok {
                    carry = start;
                }
                ok
            });
            (done, _mm512_cvtsi512_si32(carry) as u32)
        }
    );

    fearless_simd::kernel!(
        pub(super) fn q15_avx512(
            t: Avx512,
            k15: f64,
            phase: u32,
            x: &[i16],
            y: &mut [i16],
        ) -> (usize, u32) {
            let kk = _mm512_set1_pd(k15);
            let half = _mm512_set1_pd(0.5);
            let last = _mm512_set1_epi32(15);
            let mut carry = _mm512_set1_epi32(phase as i32);
            let done = blocks(16, x, y, |xb, yb| {
                for (xv, yv) in xb.chunks_exact(16).zip(yb.chunks_exact_mut(32)) {
                    let q = _mm512_cvtepi16_epi32(i16x16::from_slice(t, xv).into());
                    let lo = _mm512_cvtepi32_pd(_mm512_castsi512_si256(q));
                    let hi = _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64::<1>(q));
                    let lo = _mm512_add_pd(_mm512_mul_pd(lo, kk), half);
                    let hi = _mm512_add_pd(_mm512_mul_pd(hi, kk), half);
                    let p = scan_x16(floor_x16(lo, hi), carry);
                    carry = _mm512_permutexvar_epi32(last, p);
                    let v = cos_sin_q15_x16(p);
                    let (d0, d1) = yv.split_at_mut(16);
                    i16x16::simd_from(t, _mm512_castsi512_si256(v)).store_slice(d0);
                    i16x16::simd_from(t, _mm512_extracti64x4_epi64::<1>(v)).store_slice(d1);
                }
                true
            });
            (done, _mm512_cvtsi512_si32(carry) as u32)
        }
    );

    const _: () = assert!(super::BLOCK * 16 <= super::SCALAR_RUN);
}

/// The level-specific paths of NEON (see [`x86`]).
#[cfg(all(feature = "nco-tables", target_arch = "aarch64"))]
mod neon {
    use core::arch::aarch64::{
        float32x4_t, float64x2_t, int16x8_t, uint32x4_t, vaddq_f64, vaddq_u32, vandq_u32,
        vcaleq_f32, vcvt_f64_f32, vcvt_high_f64_f32, vcvtmq_s64_f64, vcvtq_n_f32_s32,
        vdupq_laneq_u32, vdupq_n_f32, vdupq_n_f64, vdupq_n_u32, vextq_u32, vget_low_f32,
        vget_low_s16, vgetq_lane_u32, vminvq_u32, vmovl_high_s16, vmovl_s16, vmulq_f64,
        vreinterpretq_s16_u32, vreinterpretq_u32_s64, vuzp1q_u32,
    };

    use fearless_simd::{f32x4, i16x8, prelude::*};

    use super::blocks;
    use crate::table::neon::{cos_sin_f32_x4, cos_sin_q15_x4};

    /// The wrapped steps `(floor(x * k + 0.5) as i64) as u32` of four values with
    /// `|x * k| <= 2^50`: `fcvtms` converts with the floor, then the low words of the `i64`.
    #[target_feature(enable = "neon")]
    #[inline]
    fn steps_x4(x: float32x4_t, k: float64x2_t) -> uint32x4_t {
        let half = vdupq_n_f64(0.5);
        let lo = vaddq_f64(vmulq_f64(vcvt_f64_f32(vget_low_f32(x)), k), half);
        let hi = vaddq_f64(vmulq_f64(vcvt_high_f64_f32(x), k), half);
        vuzp1q_u32(
            vreinterpretq_u32_s64(vcvtmq_s64_f64(lo)),
            vreinterpretq_u32_s64(vcvtmq_s64_f64(hi)),
        )
    }

    /// The phases `carry + inclusive prefix sum of st` of four steps.
    #[target_feature(enable = "neon")]
    #[inline]
    fn scan_x4(st: uint32x4_t, carry: uint32x4_t) -> uint32x4_t {
        let zero = vdupq_n_u32(0);
        let s = vaddq_u32(st, vextq_u32::<3>(zero, st));
        let s = vaddq_u32(s, vextq_u32::<2>(zero, s));
        vaddq_u32(s, carry)
    }

    fearless_simd::kernel!(
        pub(super) fn f32_neon(
            t: Neon,
            k: f64,
            limit: i32,
            phase: u32,
            x: &[f32],
            y: &mut [f32],
        ) -> (usize, u32) {
            let kk = vdupq_n_f64(k);
            let lim = vdupq_n_f32(f32::from_bits(limit as u32));
            let mut carry = vdupq_n_u32(phase);
            let done = blocks(4, x, y, |xb, yb| {
                let start = carry;
                let mut ok = vdupq_n_u32(u32::MAX);
                for (xv, yv) in xb.chunks_exact(4).zip(yb.chunks_exact_mut(8)) {
                    let x: float32x4_t = f32x4::from_slice(t, xv).into();
                    ok = vandq_u32(ok, vcaleq_f32(x, lim));
                    let p = scan_x4(steps_x4(x, kk), carry);
                    carry = vdupq_laneq_u32::<3>(p);
                    let (o0, o1) = cos_sin_f32_x4(t, p);
                    let (d0, d1) = yv.split_at_mut(4);
                    f32x4::simd_from(t, o0).store_slice(d0);
                    f32x4::simd_from(t, o1).store_slice(d1);
                }
                let ok = vminvq_u32(ok) != 0;
                if !ok {
                    carry = start;
                }
                ok
            });
            (done, vgetq_lane_u32::<0>(carry))
        }
    );

    fearless_simd::kernel!(
        pub(super) fn q15_neon(
            t: Neon,
            k: f64,
            phase: u32,
            x: &[i16],
            y: &mut [i16],
        ) -> (usize, u32) {
            let kk = vdupq_n_f64(k);
            let mut carry = vdupq_n_u32(phase);
            let done = blocks(8, x, y, |xb, yb| {
                for (xv, yv) in xb.chunks_exact(8).zip(yb.chunks_exact_mut(16)) {
                    let q: int16x8_t = i16x8::from_slice(t, xv).into();
                    for (q, o) in [vmovl_s16(vget_low_s16(q)), vmovl_high_s16(q)]
                        .into_iter()
                        .zip(yv.chunks_exact_mut(8))
                    {
                        // `q / 2^15` exactly, as the fallback's input.
                        let p = scan_x4(steps_x4(vcvtq_n_f32_s32::<15>(q), kk), carry);
                        carry = vdupq_laneq_u32::<3>(p);
                        let v = vreinterpretq_s16_u32(cos_sin_q15_x4(p));
                        i16x8::simd_from(t, v).store_slice(o);
                    }
                }
                true
            });
            (done, vgetq_lane_u32::<0>(carry))
        }
    );
}

/// The level-specific paths of SIMD128 (see [`x86`]).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::{
        f64x2_add, f64x2_floor, f64x2_mul, f64x2_promote_low_f32x4, f64x2_splat, i32x4_add,
        i32x4_gt, i32x4_max, i32x4_shuffle, i32x4_splat, i64x2_shuffle, u32x4_extract_lane,
        u32x4_splat, v128, v128_and, v128_any_true,
    };
    #[cfg(feature = "nco-tables")]
    use core::arch::wasm32::{
        f64x2_convert_low_i32x4, i32x4_extend_high_i16x8, i32x4_extend_low_i16x8,
    };

    #[cfg(feature = "nco-tables")]
    use fearless_simd::i16x8;
    use fearless_simd::{f32x4, prelude::*};

    use super::{MAGIC, blocks};
    use crate::table::wasm::cos_sin_f32_x4;
    #[cfg(feature = "nco-tables")]
    use crate::table::wasm::cos_sin_q15_x4;

    /// The wrapped steps of `v = lo | hi` (four values of magnitude below `2^51`).
    #[inline(always)]
    fn floor_x4(lo: v128, hi: v128) -> v128 {
        let magic = f64x2_splat(MAGIC);
        let a = f64x2_add(f64x2_floor(lo), magic);
        let b = f64x2_add(f64x2_floor(hi), magic);
        i32x4_shuffle::<0, 2, 4, 6>(a, b)
    }

    /// The phases `carry + inclusive prefix sum of st` of four steps.
    #[inline(always)]
    fn scan_x4(st: v128, carry: v128) -> v128 {
        let zero = u32x4_splat(0);
        let s = i32x4_add(st, i32x4_shuffle::<4, 0, 1, 2>(st, zero));
        let s = i32x4_add(s, i32x4_shuffle::<4, 5, 0, 1>(s, zero));
        i32x4_add(s, carry)
    }

    fearless_simd::kernel!(
        pub(super) fn f32_simd128(
            t: WasmSimd128,
            k: f64,
            limit: i32,
            phase: u32,
            x: &[f32],
            y: &mut [f32],
        ) -> (usize, u32) {
            let kk = f64x2_splat(k);
            let half = f64x2_splat(0.5);
            let abs = i32x4_splat(i32::MAX);
            let mut carry = u32x4_splat(phase);
            let done = blocks(4, x, y, |xb, yb| {
                let start = carry;
                let mut m = i32x4_splat(0);
                for (xv, yv) in xb.chunks_exact(4).zip(yb.chunks_exact_mut(8)) {
                    let x: v128 = f32x4::from_slice(t, xv).into();
                    m = i32x4_max(m, v128_and(x, abs));
                    let lo = f64x2_promote_low_f32x4(x);
                    let hi = f64x2_promote_low_f32x4(i64x2_shuffle::<1, 1>(x, x));
                    let lo = f64x2_add(f64x2_mul(lo, kk), half);
                    let hi = f64x2_add(f64x2_mul(hi, kk), half);
                    let p = scan_x4(floor_x4(lo, hi), carry);
                    carry = i32x4_shuffle::<3, 3, 3, 3>(p, p);
                    #[cfg(feature = "nco-tables")]
                    let (o0, o1) = cos_sin_f32_x4(t, p);
                    #[cfg(not(feature = "nco-tables"))]
                    let (o0, o1) = cos_sin_f32_x4(p);
                    let (d0, d1) = yv.split_at_mut(4);
                    f32x4::simd_from(t, o0).store_slice(d0);
                    f32x4::simd_from(t, o1).store_slice(d1);
                }
                let ok = !v128_any_true(i32x4_gt(m, i32x4_splat(limit)));
                if !ok {
                    carry = start;
                }
                ok
            });
            (done, u32x4_extract_lane::<0>(carry))
        }
    );

    #[cfg(feature = "nco-tables")]
    fearless_simd::kernel!(
        pub(super) fn q15_simd128(
            t: WasmSimd128,
            k15: f64,
            phase: u32,
            x: &[i16],
            y: &mut [i16],
        ) -> (usize, u32) {
            let kk = f64x2_splat(k15);
            let half = f64x2_splat(0.5);
            let mut carry = u32x4_splat(phase);
            let done = blocks(8, x, y, |xb, yb| {
                for (xv, yv) in xb.chunks_exact(8).zip(yb.chunks_exact_mut(16)) {
                    let q: v128 = i16x8::from_slice(t, xv).into();
                    for (q, o) in [i32x4_extend_low_i16x8(q), i32x4_extend_high_i16x8(q)]
                        .into_iter()
                        .zip(yv.chunks_exact_mut(8))
                    {
                        let lo = f64x2_convert_low_i32x4(q);
                        let hi = f64x2_convert_low_i32x4(i64x2_shuffle::<1, 1>(q, q));
                        let lo = f64x2_add(f64x2_mul(lo, kk), half);
                        let hi = f64x2_add(f64x2_mul(hi, kk), half);
                        let p = scan_x4(floor_x4(lo, hi), carry);
                        carry = i32x4_shuffle::<3, 3, 3, 3>(p, p);
                        i16x8::simd_from(t, cos_sin_q15_x4(p)).store_slice(o);
                    }
                }
                true
            });
            (done, u32x4_extract_lane::<0>(carry))
        }
    );
}
