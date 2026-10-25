//! Vector twins of the cosine-table lookup of the `rotate` and `fm_mod` fallbacks.
//!
//! The results are bit-exact with `sdrk_core`'s scalar code: the same table entries, the same
//! order of f32 operations (no fused multiply-add) and the same integer arithmetic.
//!
//! # Entry tables (feature `nco-tables`)
//!
//! The fallback interpolates `a + (b - a) * f` between `COS_F32[i]` and `COS_F32[i + 1]` for
//! the cosine (`i` = bits 20 to 31 of the phase) and between `COS_F32[j]` and `COS_F32[j + 1]`
//! for the sine, with `j = (i - 1024) mod 4096`: the sine index is that of `phase - 2^30`, and
//! the low 20 bits of `2^30` are zero. So both lookups depend on `i` alone, and
//! [`COS_SIN_F32`] keeps one 16-byte entry `[cos a, sin a, cos b - cos a, sin b - sin a]` per
//! index (64 KiB), built at compile time with the same f32 subtraction the fallback does at run
//! time. One load per sample replaces four, the low half of an entry is already the interleaved
//! output pair and the high half holds the matching differences, so `a + d * f` comes out in
//! output order. The differences are stored times `2^-20` (exact: they are normal numbers far
//! above the subnormal range), so that they multiply the integer fraction `p & 0xF_FFFF` itself:
//! `(d * 2^-20) * F` is the same real number as the fallback's `d * (F * 2^-20)`, so it rounds
//! to the same f32. [`COS_SIN_Q15`] is the same layout in Q15 (8-byte entries, 32 KiB): the
//! fallback's `a + round_shift((b - a) * f15, 15)` with `f15 < 2^15` is exactly the rounding
//! high multiplication of `pmulhrsw` (SSSE3), `sqrdmulh` (NEON) and `i16x8.q15mulr_sat_s`
//! (SIMD128), whose saturation needs `-32768 * -32768` and cannot occur. The interpolated value
//! lies between `a` and `b`, so the wrapping 16-bit addition is exact and the fallback's final
//! saturation never acts.
//!
//! Without the feature (the default of the backend; the facade enables it on x86-64 and
//! AArch64) the lookups read `COS_F32` and `COS_Q15` directly.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::tables::{COS_F32, COS_Q15};

/// A quarter turn of the 32-bit phase.
#[cfg(not(feature = "nco-tables"))]
const QUARTER: u32 = 1 << 30;

/// Entries of the interpolation tables, one per table index `i < 4096`.
#[cfg(feature = "nco-tables")]
const ENTRIES: usize = 4096;

/// `2^-20`: the weight of one unit of the interpolation fraction `p & 0xF_FFFF`.
const SCALE: f32 = 1.0 / 1_048_576.0;

/// The f32 interpolation table: `[cos a, sin a, (cos b - cos a) * 2^-20, (sin b - sin a) * 2^-20]`
/// per index, 16-byte aligned so that every entry is one aligned vector load.
#[cfg(feature = "nco-tables")]
#[repr(C, align(16))]
pub(crate) struct CosSinF32(pub(crate) [[f32; 4]; ENTRIES]);

/// The Q15 interpolation table, with the same layout as [`CosSinF32`] in 8-byte entries.
#[cfg(feature = "nco-tables")]
#[repr(C, align(8))]
pub(crate) struct CosSinQ15(pub(crate) [[i16; 4]; ENTRIES]);

/// The f32 table, built at compile time from `COS_F32` (see the module documentation).
#[cfg(feature = "nco-tables")]
#[allow(
    clippy::large_stack_arrays,
    clippy::indexing_slicing,
    reason = "evaluated at compile time: the array is the static itself, and an index out of \
              range would be a compile error"
)]
pub(crate) static COS_SIN_F32: CosSinF32 = {
    let mut t = [[0.0f32; 4]; ENTRIES];
    let mut i = 0;
    while i < ENTRIES {
        let j = (i + 3 * ENTRIES / 4) % ENTRIES;
        let (c, s) = (COS_F32[i], COS_F32[j]);
        t[i] = [
            c,
            s,
            (COS_F32[i + 1] - c) * SCALE,
            (COS_F32[j + 1] - s) * SCALE,
        ];
        i += 1;
    }
    CosSinF32(t)
};

/// The Q15 table, built at compile time from `COS_Q15`. Neighbouring Q15 cosines differ by
/// at most 51, so every difference fits `i16`.
#[cfg(feature = "nco-tables")]
#[allow(
    clippy::large_stack_arrays,
    clippy::indexing_slicing,
    reason = "evaluated at compile time: the array is the static itself, and an index out of \
              range would be a compile error"
)]
pub(crate) static COS_SIN_Q15: CosSinQ15 = {
    let mut t = [[0i16; 4]; ENTRIES];
    let mut i = 0;
    while i < ENTRIES {
        let j = (i + 3 * ENTRIES / 4) % ENTRIES;
        let (c, s) = (COS_Q15[i], COS_Q15[j]);
        t[i] = [c, s, COS_Q15[i + 1] - c, COS_Q15[j + 1] - s];
        i += 1;
    }
    CosSinQ15(t)
};

/// The f32 entry of the table index `i` (below 4096).
#[cfg(feature = "nco-tables")]
#[inline(always)]
pub(crate) fn entry_f32(i: u32) -> [f32; 4] {
    COS_SIN_F32.0.get(i as usize).copied().unwrap_or_default()
}

/// The f32 entry at the byte offset `o` (`16 * i`, below 64 KiB) of the table.
#[cfg(all(
    feature = "nco-tables",
    any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )
))]
#[inline(always)]
pub(crate) fn entry_f32_at(o: u32) -> [f32; 4] {
    entry_f32(o / 16)
}

/// The Q15 entry of the table index `i` (below 4096).
#[cfg(feature = "nco-tables")]
#[inline(always)]
pub(crate) fn entry_q15(i: u32) -> [i16; 4] {
    COS_SIN_Q15.0.get(i as usize).copied().unwrap_or_default()
}

/// The Q15 entry at the byte offset `o` (`8 * i`, below 32 KiB) of the table.
#[cfg(all(
    feature = "nco-tables",
    any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )
))]
#[inline(always)]
pub(crate) fn entry_q15_at(o: u32) -> [i16; 4] {
    entry_q15(o / 8)
}

/// The interpolation fraction `p & 0xF_FFFF` of the f32 lookup as `f32` (exact).
#[inline(always)]
fn fraction_units<S: SimdToken>(p: S::u32s) -> S::f32s {
    let frac: S::i32s = (p & 0xF_FFFFu32).bitcast();
    frac.to_float::<S::f32s>()
}

/// The interleaved `(cos, sin)` of the phases `p` with linear interpolation, as `cos_sin_f32`
/// of the fallback: the pairs of lanes `0..h/2` and of lanes `h/2..h`.
#[inline(always)]
pub(crate) fn cos_sin_f32<S: SimdToken>(simd: S, p: S::u32s) -> (S::f32s, S::f32s) {
    #[cfg(feature = "nco-tables")]
    {
        let h = S::f32s::LEN;
        let f = fraction_units::<S>(p);
        let (ps, fs) = (p.as_slice(), f.as_slice());
        let pair = |i: usize| {
            let lane = i / 2;
            let e = entry_f32(ps.get(lane).copied().unwrap_or(0) >> 20);
            let (a, d) = if i.is_multiple_of(2) {
                (e[0], e[2])
            } else {
                (e[1], e[3])
            };
            a + d * fs.get(lane).copied().unwrap_or(0.0)
        };
        (
            S::f32s::from_fn(simd, pair),
            S::f32s::from_fn(simd, |i| pair(i + h)),
        )
    }
    #[cfg(not(feature = "nco-tables"))]
    {
        let f = fraction_units::<S>(p) * SCALE;
        let idx_c = p >> 20u32;
        let idx_s = (p - QUARTER) >> 20u32;
        let ac = gather_f32(simd, idx_c, 0);
        let bc = gather_f32(simd, idx_c, 1);
        let a_s = gather_f32(simd, idx_s, 0);
        let b_s = gather_f32(simd, idx_s, 1);
        (ac + (bc - ac) * f).interleave(a_s + (b_s - a_s) * f)
    }
}

/// `COS_F32[(i & 4095) + add]` for every lane (the mask keeps the index inside the table).
#[cfg(not(feature = "nco-tables"))]
#[inline(always)]
fn gather_f32<S: SimdToken>(simd: S, idx: S::u32s, add: usize) -> S::f32s {
    let mut it = idx.to_array().into_iter();
    S::f32s::from_fn(simd, |_| {
        let i = it.next().unwrap_or(0) as usize & 0xFFF;
        COS_F32.get(i + add).copied().unwrap_or(0.0)
    })
}

/// `COS_Q15[(i & 4095) + add]` for every lane, widened to `i32`.
#[cfg(not(feature = "nco-tables"))]
#[inline(always)]
fn gather_q15<S: SimdToken>(simd: S, idx: S::u32s, add: usize) -> S::i32s {
    let mut it = idx.to_array().into_iter();
    S::i32s::from_fn(simd, |_| {
        let i = it.next().unwrap_or(0) as usize & 0xFFF;
        i32::from(COS_Q15.get(i + add).copied().unwrap_or(0))
    })
}

/// One interpolated Q15 table value for the index vector `idx` and the fractions `f15`.
#[cfg(not(feature = "nco-tables"))]
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
    #[cfg(feature = "nco-tables")]
    {
        let v: S::i32s = cos_sin_q15_packed(simd, p).bitcast();
        ((v << 16u32) >> 16u32, v >> 16u32)
    }
    #[cfg(not(feature = "nco-tables"))]
    {
        let f15: S::i32s = ((p >> 5u32) & 0x7FFFu32).bitcast();
        let idx_c = p >> 20u32;
        let idx_s = (p - QUARTER) >> 20u32;
        (
            interp_q15::<S>(simd, idx_c, f15),
            interp_q15::<S>(simd, idx_s, f15),
        )
    }
}

/// `(cos, sin)` of the phases `p` as a `CQ15` pair per 32-bit lane (cosine in the low half),
/// through the level's rounding high multiplication where it has one.
#[cfg(feature = "nco-tables")]
#[inline(always)]
pub(crate) fn cos_sin_q15_packed<S: SimdToken>(simd: S, p: S::u32s) -> S::u32s {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        use fearless_simd::{u32x4, u32x8, u32x16};
        let ps = p.as_slice();
        if let Some(t) = level.as_avx512().filter(|_| ps.len() == 16) {
            let v = x86::q15_avx512(t, u32x16::from_slice(t, ps).into());
            return S::u32s::from_slice(simd, u32x16::simd_from(t, v).as_slice());
        }
        if let Some(t) = level.as_avx2().filter(|_| ps.len() == 8) {
            let v = x86::q15_avx2(t, u32x8::from_slice(t, ps).into());
            return S::u32s::from_slice(simd, u32x8::simd_from(t, v).as_slice());
        }
        if let Some(t) = level.as_sse4_2().filter(|_| ps.len() == 4) {
            let v = x86::q15_sse4_2(t, u32x4::from_slice(t, ps).into());
            return S::u32s::from_slice(simd, u32x4::simd_from(t, v).as_slice());
        }
        if let Some(t) = crate::level::sse2_only(level).filter(|_| ps.len() == 4) {
            let v = x86::q15_sse2(t, u32x4::from_slice(t, ps).into());
            return S::u32s::from_slice(simd, u32x4::simd_from(t, v).as_slice());
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon().filter(|_| S::u32s::LEN == 4) {
        use fearless_simd::u32x4;
        let v = neon::q15_neon(t, u32x4::from_slice(t, p.as_slice()).into());
        return S::u32s::from_slice(simd, u32x4::simd_from(t, v).as_slice());
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128().filter(|_| S::u32s::LEN == 4) {
        use fearless_simd::u32x4;
        let v = wasm::cos_sin_q15_x4(u32x4::from_slice(t, p.as_slice()).into());
        return S::u32s::from_slice(simd, u32x4::simd_from(t, v).as_slice());
    }
    let _ = level;
    let ps = p.as_slice();
    S::u32s::from_fn(simd, |lane| {
        let p = ps.get(lane).copied().unwrap_or(0);
        let [c, s, dc, ds] = entry_q15(p >> 20);
        let f15 = i32::from((p >> 5) as u16 & 0x7FFF);
        let interp =
            |a: i16, d: i16| a.wrapping_add(((i32::from(d) * f15 + (1 << 14)) >> 15) as i16);
        u32::from(interp(c, dc) as u16) | u32::from(interp(s, ds) as u16) << 16
    })
}

/// The level-specific lookups of x86: whole vectors of phases in, interleaved `(cos, sin)`
/// out. Every helper is compiled with the features of its level and inlined into the kernels
/// of that level.
#[cfg(all(feature = "nco-tables", target_arch = "x86_64"))]
#[allow(unsafe_code)]
pub(crate) mod x86 {
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128, __m128i, __m256, __m256d, __m256i, __m512, __m512i, _mm_add_epi16, _mm_add_ps,
        _mm_and_si128, _mm_castpd_ps, _mm_castps_pd, _mm_castps_si128, _mm_castsi128_ps,
        _mm_cvtepi32_ps, _mm_cvtsi128_si32, _mm_extract_epi32, _mm_madd_epi16, _mm_mul_ps,
        _mm_mulhrs_epi16, _mm_or_si128, _mm_packs_epi32, _mm_set_epi64x, _mm_set1_epi16,
        _mm_set1_epi32, _mm_setr_epi8, _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_shuffle_ps,
        _mm_slli_epi32, _mm_srai_epi32, _mm_srli_epi32, _mm_unpackhi_epi16, _mm_unpackhi_epi32,
        _mm_unpackhi_pd, _mm_unpackhi_ps, _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_pd,
        _mm_unpacklo_ps, _mm256_add_epi16, _mm256_add_ps, _mm256_and_si256, _mm256_castpd_ps,
        _mm256_castps_pd, _mm256_castps_si256, _mm256_castps128_ps256, _mm256_castsi128_si256,
        _mm256_castsi256_ps, _mm256_cvtepi32_ps, _mm256_extract_epi32, _mm256_insertf128_ps,
        _mm256_inserti128_si256, _mm256_mul_ps, _mm256_mulhrs_epi16, _mm256_permutevar8x32_ps,
        _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi8, _mm256_setr_epi32,
        _mm256_shuffle_epi8, _mm256_shuffle_ps, _mm256_slli_epi32, _mm256_srli_epi32,
        _mm256_unpackhi_pd, _mm256_unpacklo_pd, _mm512_add_epi16, _mm512_add_ps, _mm512_and_si512,
        _mm512_castpd_ps, _mm512_castsi512_si256, _mm512_cvtepi32_ps, _mm512_extracti64x4_epi64,
        _mm512_i32gather_epi64, _mm512_i32gather_pd, _mm512_mul_ps, _mm512_mulhrs_epi16,
        _mm512_or_si512, _mm512_permutex2var_epi32, _mm512_permutexvar_ps, _mm512_set1_epi32,
        _mm512_setr_epi32, _mm512_slli_epi32, _mm512_srli_epi32,
    };

    use fearless_simd::{Simd as SimdToken, f32x4, prelude::*};

    use super::{COS_SIN_F32, entry_f32_at, entry_q15_at};

    /// The f32 table entry at the byte offset `o` as one vector.
    #[inline(always)]
    fn load_f32<S: SimdToken>(simd: S, o: i32) -> __m128 {
        f32x4::simd_from(simd, entry_f32_at(o as u32)).into()
    }

    /// The byte offsets `16 * (p >> 20)` of the f32 entries of four phases.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn offsets_f32_x4(p: __m128i) -> __m128i {
        _mm_and_si128(_mm_srli_epi32::<16>(p), _mm_set1_epi32(0xFFF0))
    }

    /// The byte offsets `8 * (p >> 20)` of the Q15 entries of four phases.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn offsets_q15_x4(p: __m128i) -> __m128i {
        _mm_and_si128(_mm_srli_epi32::<17>(p), _mm_set1_epi32(0x7FF8))
    }

    /// The Q15 table entries at the byte offsets `a` and `b` as the low and high halves of a
    /// vector.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn load_q15_pair(a: i32, b: i32) -> __m128i {
        let (a, b) = (entry_q15_at(a as u32), entry_q15_at(b as u32));
        _mm_set_epi64x(bytemuck::cast(b), bytemuck::cast(a))
    }

    /// `a + d * f` for the entries `e0` and `e1` (two samples) with the fractions
    /// `[f0, f0, f1, f1]`.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn interp_x2(e0: __m128, e1: __m128, f: __m128) -> __m128 {
        let (e0, e1) = (_mm_castps_pd(e0), _mm_castps_pd(e1));
        let a = _mm_castpd_ps(_mm_unpacklo_pd(e0, e1));
        let d = _mm_castpd_ps(_mm_unpackhi_pd(e0, e1));
        _mm_add_ps(a, _mm_mul_ps(d, f))
    }

    /// The interpolation fractions `p & 0xF_FFFF` of four phases (the weights of the scaled
    /// differences).
    #[target_feature(enable = "sse2")]
    #[inline]
    fn fraction_x4(p: __m128i) -> __m128 {
        _mm_cvtepi32_ps(_mm_and_si128(p, _mm_set1_epi32(0xF_FFFF)))
    }

    /// Interleaved `(cos, sin)` of four phases: samples 0 and 1, samples 2 and 3.
    #[target_feature(enable = "sse2")]
    #[inline]
    pub(crate) fn cos_sin_f32_x4<S: SimdToken>(simd: S, p: __m128i) -> (__m128, __m128) {
        let f = fraction_x4(p);
        let o = offsets_f32_x4(p);
        let e0 = load_f32(simd, _mm_cvtsi128_si32(o));
        let e1 = load_f32(simd, _mm_cvtsi128_si32(_mm_shuffle_epi32::<0x55>(o)));
        let e2 = load_f32(simd, _mm_cvtsi128_si32(_mm_shuffle_epi32::<0xAA>(o)));
        let e3 = load_f32(simd, _mm_cvtsi128_si32(_mm_shuffle_epi32::<0xFF>(o)));
        (
            interp_x2(e0, e1, _mm_unpacklo_ps(f, f)),
            interp_x2(e2, e3, _mm_unpackhi_ps(f, f)),
        )
    }

    /// Interleaved `(cos, sin)` of eight phases: samples 0 to 3, samples 4 to 7.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(crate) fn cos_sin_f32_x8<S: SimdToken>(simd: S, p: __m256i) -> (__m256, __m256) {
        let frac = _mm256_and_si256(p, _mm256_set1_epi32(0xF_FFFF));
        let f = _mm256_cvtepi32_ps(frac);
        let pair = |a: i32, b: i32| {
            let lo = _mm256_castps128_ps256(load_f32(simd, a));
            _mm256_castps_pd(_mm256_insertf128_ps::<1>(lo, load_f32(simd, b)))
        };
        let o = _mm256_and_si256(_mm256_srli_epi32::<16>(p), _mm256_set1_epi32(0xFFF0));
        // [e0 | e2], [e1 | e3]: one in-lane unpack gives samples 0 to 3 in output order.
        let e02 = pair(_mm256_extract_epi32::<0>(o), _mm256_extract_epi32::<2>(o));
        let e13 = pair(_mm256_extract_epi32::<1>(o), _mm256_extract_epi32::<3>(o));
        let e46 = pair(_mm256_extract_epi32::<4>(o), _mm256_extract_epi32::<6>(o));
        let e57 = pair(_mm256_extract_epi32::<5>(o), _mm256_extract_epi32::<7>(o));
        let lo = _mm256_permutevar8x32_ps(f, _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3));
        let hi = _mm256_permutevar8x32_ps(f, _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7));
        let interp = |e0: __m256d, e1: __m256d, f: __m256| {
            let a = _mm256_castpd_ps(_mm256_unpacklo_pd(e0, e1));
            let d = _mm256_castpd_ps(_mm256_unpackhi_pd(e0, e1));
            _mm256_add_ps(a, _mm256_mul_ps(d, f))
        };
        (interp(e02, e13, lo), interp(e46, e57, hi))
    }

    /// Interleaved `(cos, sin)` of sixteen phases: samples 0 to 7, samples 8 to 15. The entry
    /// halves are fetched with 64-bit gathers, as the generic code at this level did with
    /// 32-bit gathers before.
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub(crate) fn cos_sin_f32_x16(p: __m512i) -> (__m512, __m512) {
        let frac = _mm512_and_si512(p, _mm512_set1_epi32(0xF_FFFF));
        let f = _mm512_cvtepi32_ps(frac);
        // Index of the entry in units of 8 bytes: 2 * (p >> 20) < 8192.
        let idx = _mm512_slli_epi32::<1>(_mm512_srli_epi32::<20>(p));
        let base = core::ptr::from_ref(&COS_SIN_F32).cast::<f64>();
        let half = |idx: __m256i, f: __m512| {
            // SAFETY: every index is `2 * (p >> 20)`, so even and at most 8190, and `base`
            // points to the 8192 doubles (64 KiB) of the table; the differences start one
            // double later, and the largest index plus one (8191) is still inside the table.
            let (a, d) = unsafe {
                (
                    _mm512_i32gather_pd::<8>(idx, base),
                    _mm512_i32gather_pd::<8>(idx, base.add(1)),
                )
            };
            _mm512_add_ps(_mm512_castpd_ps(a), _mm512_mul_ps(_mm512_castpd_ps(d), f))
        };
        let dup_lo = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
        let dup_hi = _mm512_setr_epi32(8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15);
        (
            half(
                _mm512_castsi512_si256(idx),
                _mm512_permutexvar_ps(dup_lo, f),
            ),
            half(
                _mm512_extracti64x4_epi64::<1>(idx),
                _mm512_permutexvar_ps(dup_hi, f),
            ),
        )
    }

    /// The Q15 fractions `(p >> 5) & 0x7FFF` of four phases in both 16-bit halves of a lane:
    /// bytes 1 and 2 of `p << 3`, twice. The byte pairs at odd offsets keep the one `pshufb`;
    /// whole 16-bit words of `p >> 5` would be lowered to `pshuflw` and `pshufhw` (one
    /// instruction more) unless the build is tuned for a CPU with a fast `pshufb`.
    #[target_feature(enable = "ssse3")]
    #[inline]
    fn fraction_q15_x4(p: __m128i) -> __m128i {
        let dup = _mm_setr_epi8(1, 2, 1, 2, 5, 6, 5, 6, 9, 10, 9, 10, 13, 14, 13, 14);
        let f = _mm_shuffle_epi8(_mm_slli_epi32::<3>(p), dup);
        _mm_and_si128(f, _mm_set1_epi16(0x7FFF))
    }

    /// `(cos, sin)` as a `CQ15` pair per lane of four phases, with `pmulhrsw`.
    #[target_feature(enable = "sse4.2")]
    #[inline]
    pub(crate) fn cos_sin_q15_x4(p: __m128i) -> __m128i {
        let o = offsets_q15_x4(p);
        let e01 = load_q15_pair(_mm_extract_epi32::<0>(o), _mm_extract_epi32::<1>(o));
        let e23 = load_q15_pair(_mm_extract_epi32::<2>(o), _mm_extract_epi32::<3>(o));
        let (e01, e23) = (_mm_castsi128_ps(e01), _mm_castsi128_ps(e23));
        let a = _mm_castps_si128(_mm_shuffle_ps::<0x88>(e01, e23));
        let d = _mm_castps_si128(_mm_shuffle_ps::<0xDD>(e01, e23));
        _mm_add_epi16(a, _mm_mulhrs_epi16(d, fraction_q15_x4(p)))
    }

    /// The SSE2 form of [`cos_sin_q15_x4`]: the rounding product with `pmaddwd` on the pairs
    /// `(d, 1) * (f15, 2^14)`, an arithmetic shift and a pack (the values fit `i16`).
    #[target_feature(enable = "sse2")]
    #[inline]
    pub(crate) fn cos_sin_q15_x4_sse2(p: __m128i) -> __m128i {
        let o = offsets_q15_x4(p);
        let lane = |k: __m128i| _mm_cvtsi128_si32(k);
        let e01 = load_q15_pair(lane(o), lane(_mm_shuffle_epi32::<0x55>(o)));
        let e23 = load_q15_pair(
            lane(_mm_shuffle_epi32::<0xAA>(o)),
            lane(_mm_shuffle_epi32::<0xFF>(o)),
        );
        let (e01, e23) = (_mm_castsi128_ps(e01), _mm_castsi128_ps(e23));
        let a = _mm_castps_si128(_mm_shuffle_ps::<0x88>(e01, e23));
        let d = _mm_castps_si128(_mm_shuffle_ps::<0xDD>(e01, e23));
        // `f15 | 2^14 << 16` per lane, duplicated for the two components of a sample.
        let f = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi32::<5>(p), _mm_set1_epi32(0x7FFF)),
            _mm_set1_epi32(1 << 30),
        );
        let one = _mm_set1_epi16(1);
        let lo = _mm_madd_epi16(_mm_unpacklo_epi16(d, one), _mm_unpacklo_epi32(f, f));
        let hi = _mm_madd_epi16(_mm_unpackhi_epi16(d, one), _mm_unpackhi_epi32(f, f));
        let r = _mm_packs_epi32(_mm_srai_epi32::<15>(lo), _mm_srai_epi32::<15>(hi));
        _mm_add_epi16(a, r)
    }

    /// `(cos, sin)` as a `CQ15` pair per lane of eight phases, with `vpmulhrsw`.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(crate) fn cos_sin_q15_x8(p: __m256i) -> __m256i {
        let quad = |a: i32, b: i32, c: i32, d: i32| {
            let lo = _mm256_castsi128_si256(load_q15_pair(a, b));
            _mm256_inserti128_si256::<1>(lo, load_q15_pair(c, d))
        };
        // [e0 e1 | e4 e5], [e2 e3 | e6 e7]: one in-lane shuffle gives samples 0 to 7 in order.
        let ex = |k: __m256i| {
            (
                _mm256_extract_epi32::<0>(k),
                _mm256_extract_epi32::<1>(k),
                _mm256_extract_epi32::<2>(k),
                _mm256_extract_epi32::<3>(k),
                _mm256_extract_epi32::<4>(k),
                _mm256_extract_epi32::<5>(k),
                _mm256_extract_epi32::<6>(k),
                _mm256_extract_epi32::<7>(k),
            )
        };
        let o = _mm256_and_si256(_mm256_srli_epi32::<17>(p), _mm256_set1_epi32(0x7FF8));
        let (i0, i1, i2, i3, i4, i5, i6, i7) = ex(o);
        let e0 = _mm256_castsi256_ps(quad(i0, i1, i4, i5));
        let e1 = _mm256_castsi256_ps(quad(i2, i3, i6, i7));
        let a = _mm256_castps_si256(_mm256_shuffle_ps::<0x88>(e0, e1));
        let d = _mm256_castps_si256(_mm256_shuffle_ps::<0xDD>(e0, e1));
        // The fractions as in `fraction_q15_x4`, per 128-bit block.
        let dup = _mm256_setr_epi8(
            1, 2, 1, 2, 5, 6, 5, 6, 9, 10, 9, 10, 13, 14, 13, 14, 1, 2, 1, 2, 5, 6, 5, 6, 9, 10, 9,
            10, 13, 14, 13, 14,
        );
        let f = _mm256_shuffle_epi8(_mm256_slli_epi32::<3>(p), dup);
        let f = _mm256_and_si256(f, _mm256_set1_epi16(0x7FFF));
        _mm256_add_epi16(a, _mm256_mulhrs_epi16(d, f))
    }

    /// `(cos, sin)` as a `CQ15` pair per lane of sixteen phases, with 64-bit gathers of the
    /// whole entries and `vpmulhrsw`.
    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    pub(crate) fn cos_sin_q15_x16(p: __m512i) -> __m512i {
        let idx = _mm512_srli_epi32::<20>(p);
        let base = core::ptr::from_ref(&super::COS_SIN_Q15).cast::<i64>();
        // SAFETY: every index is below 4096 and `base` points to the 4096 entries of 8 bytes
        // of the table.
        let (g0, g1) = unsafe {
            (
                _mm512_i32gather_epi64::<8>(_mm512_castsi512_si256(idx), base),
                _mm512_i32gather_epi64::<8>(_mm512_extracti64x4_epi64::<1>(idx), base),
            )
        };
        let even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let odd = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        let a = _mm512_permutex2var_epi32(g0, even, g1);
        let d = _mm512_permutex2var_epi32(g0, odd, g1);
        let f = _mm512_and_si512(_mm512_srli_epi32::<5>(p), _mm512_set1_epi32(0x7FFF));
        let f = _mm512_or_si512(f, _mm512_slli_epi32::<16>(f));
        _mm512_add_epi16(a, _mm512_mulhrs_epi16(d, f))
    }

    fearless_simd::kernel!(
        /// [`cos_sin_q15_x16`] for the generic code.
        #[inline(always)]
        pub(crate) fn q15_avx512(t: Avx512, p: __m512i) -> __m512i {
            cos_sin_q15_x16(p)
        }
    );

    fearless_simd::kernel!(
        /// [`cos_sin_q15_x8`] for the generic code.
        #[inline(always)]
        pub(crate) fn q15_avx2(t: Avx2, p: __m256i) -> __m256i {
            cos_sin_q15_x8(p)
        }
    );

    fearless_simd::kernel!(
        /// [`cos_sin_q15_x4`] for the generic code.
        #[inline(always)]
        pub(crate) fn q15_sse4_2(t: Sse4_2, p: __m128i) -> __m128i {
            cos_sin_q15_x4(p)
        }
    );

    fearless_simd::kernel!(
        /// [`cos_sin_q15_x4_sse2`] for the generic code.
        #[inline(always)]
        pub(crate) fn q15_sse2(t: Sse2, p: __m128i) -> __m128i {
            cos_sin_q15_x4_sse2(p)
        }
    );
}

/// The level-specific lookups of NEON (see [`x86`]).
#[cfg(all(feature = "nco-tables", target_arch = "aarch64"))]
pub(crate) mod neon {
    use core::arch::aarch64::{
        float32x4_t, float64x2_t, uint32x4_t, vaddq_f32, vaddq_s16, vandq_u32, vcombine_s16,
        vcreate_s16, vcvtq_f32_u32, vdupq_n_u32, vgetq_lane_u32, vmulq_f32, vqrdmulhq_s16,
        vreinterpretq_f32_f64, vreinterpretq_f64_f32, vreinterpretq_s16_u32, vreinterpretq_u32_s16,
        vshrq_n_u32, vsliq_n_u32, vuzp1q_u32, vuzp2q_u32, vzip1q_f32, vzip1q_f64, vzip2q_f32,
        vzip2q_f64,
    };

    use fearless_simd::{Neon, f32x4, prelude::*};

    use super::{entry_f32, entry_q15};

    /// The interpolation fractions `p & 0xF_FFFF` of four phases (the weights of the scaled
    /// differences).
    #[target_feature(enable = "neon")]
    #[inline]
    fn fraction_x4(p: uint32x4_t) -> float32x4_t {
        vcvtq_f32_u32(vandq_u32(p, vdupq_n_u32(0xF_FFFF)))
    }

    /// Interleaved `(cos, sin)` of four phases: samples 0 and 1, samples 2 and 3.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(crate) fn cos_sin_f32_x4(simd: Neon, p: uint32x4_t) -> (float32x4_t, float32x4_t) {
        let f = fraction_x4(p);
        let i = vshrq_n_u32::<20>(p);
        let load = |i: u32| vreinterpretq_f64_f32(f32x4::simd_from(simd, entry_f32(i)).into());
        let e0 = load(vgetq_lane_u32::<0>(i));
        let e1 = load(vgetq_lane_u32::<1>(i));
        let e2 = load(vgetq_lane_u32::<2>(i));
        let e3 = load(vgetq_lane_u32::<3>(i));
        let interp = |e0: float64x2_t, e1: float64x2_t, f: float32x4_t| {
            let a = vreinterpretq_f32_f64(vzip1q_f64(e0, e1));
            let d = vreinterpretq_f32_f64(vzip2q_f64(e0, e1));
            vaddq_f32(a, vmulq_f32(d, f))
        };
        (
            interp(e0, e1, vzip1q_f32(f, f)),
            interp(e2, e3, vzip2q_f32(f, f)),
        )
    }

    /// `(cos, sin)` as a `CQ15` pair per lane of four phases, with `sqrdmulh`.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(crate) fn cos_sin_q15_x4(p: uint32x4_t) -> uint32x4_t {
        let i = vshrq_n_u32::<20>(p);
        let load = |i: u32| vcreate_s16(bytemuck::cast(entry_q15(i)));
        let pair = |a: u32, b: u32| vreinterpretq_u32_s16(vcombine_s16(load(a), load(b)));
        let e01 = pair(vgetq_lane_u32::<0>(i), vgetq_lane_u32::<1>(i));
        let e23 = pair(vgetq_lane_u32::<2>(i), vgetq_lane_u32::<3>(i));
        let a = vreinterpretq_s16_u32(vuzp1q_u32(e01, e23));
        let d = vreinterpretq_s16_u32(vuzp2q_u32(e01, e23));
        let f = vandq_u32(vshrq_n_u32::<5>(p), vdupq_n_u32(0x7FFF));
        let f = vreinterpretq_s16_u32(vsliq_n_u32::<16>(f, f));
        vreinterpretq_u32_s16(vaddq_s16(a, vqrdmulhq_s16(d, f)))
    }

    fearless_simd::kernel!(
        /// [`cos_sin_q15_x4`] for the generic code.
        #[inline(always)]
        pub(crate) fn q15_neon(t: Neon, p: uint32x4_t) -> uint32x4_t {
            cos_sin_q15_x4(p)
        }
    );
}

/// The level-specific lookups of SIMD128 (see [`x86`]). The f32 lookup also exists without the
/// entry table (the default on wasm32): it then reads the two neighbour pairs of `COS_F32` of
/// each sample with two 8-byte loads.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod wasm {
    use core::arch::wasm32::{
        f32x4_add, f32x4_convert_i32x4, f32x4_mul, i32x4_shuffle, u32x4_extract_lane, u32x4_shr,
        u32x4_splat, u64x2, v128, v128_and,
    };
    #[cfg(not(feature = "nco-tables"))]
    use core::arch::wasm32::{f32x4_splat, f32x4_sub, i32x4_sub};
    #[cfg(feature = "nco-tables")]
    use core::arch::wasm32::{i16x8_add, i16x8_q15mulr_sat, i32x4_shl, i64x2_shuffle, v128_or};

    #[cfg(feature = "nco-tables")]
    use fearless_simd::{WasmSimd128, f32x4, prelude::*};

    #[cfg(not(feature = "nco-tables"))]
    use super::COS_F32;
    #[cfg(feature = "nco-tables")]
    use super::{entry_f32_at, entry_q15_at};

    /// The interpolation fractions `p & 0xF_FFFF` of four phases.
    #[inline(always)]
    fn fraction_x4(p: v128) -> v128 {
        f32x4_convert_i32x4(v128_and(p, u32x4_splat(0xF_FFFF)))
    }

    /// Interleaved `(cos, sin)` of four phases: samples 0 and 1, samples 2 and 3.
    #[cfg(feature = "nco-tables")]
    #[inline(always)]
    pub(crate) fn cos_sin_f32_x4(simd: WasmSimd128, p: v128) -> (v128, v128) {
        let f = fraction_x4(p);
        let o = v128_and(u32x4_shr(p, 16), u32x4_splat(0xFFF0));
        let load = |o: u32| -> v128 { f32x4::simd_from(simd, entry_f32_at(o)).into() };
        let e0 = load(u32x4_extract_lane::<0>(o));
        let e1 = load(u32x4_extract_lane::<1>(o));
        let e2 = load(u32x4_extract_lane::<2>(o));
        let e3 = load(u32x4_extract_lane::<3>(o));
        let interp = |e0: v128, e1: v128, f: v128| {
            let a = i64x2_shuffle::<0, 2>(e0, e1);
            let d = i64x2_shuffle::<1, 3>(e0, e1);
            f32x4_add(a, f32x4_mul(d, f))
        };
        (
            interp(e0, e1, i32x4_shuffle::<0, 0, 1, 1>(f, f)),
            interp(e2, e3, i32x4_shuffle::<2, 2, 3, 3>(f, f)),
        )
    }

    /// Interleaved `(cos, sin)` of four phases from `COS_F32` itself.
    #[cfg(not(feature = "nco-tables"))]
    #[inline(always)]
    pub(crate) fn cos_sin_f32_x4(p: v128) -> (v128, v128) {
        let f = f32x4_mul(fraction_x4(p), f32x4_splat(super::SCALE));
        // `[c[i], c[i + 1]]` at the byte offset `o = 4 * i` as one 64-bit value; the index is
        // at most 4095, so both values are inside the 4097 entries of the table.
        let pair = |o: u32| -> u64 {
            let i = (o / 4) as usize;
            match COS_F32.get(i..i + 2) {
                Some(&[a, b]) => u64::from(a.to_bits()) | u64::from(b.to_bits()) << 32,
                _ => 0,
            }
        };
        let offsets = |p: v128| v128_and(u32x4_shr(p, 18), u32x4_splat(0x3FFC));
        let cos = offsets(p);
        let sin = offsets(i32x4_sub(p, u32x4_splat(1 << 30)));
        // [c a, c b, s a, s b] of one sample.
        let sample = |c: u32, s: u32| u64x2(pair(c), pair(s));
        let e0 = sample(u32x4_extract_lane::<0>(cos), u32x4_extract_lane::<0>(sin));
        let e1 = sample(u32x4_extract_lane::<1>(cos), u32x4_extract_lane::<1>(sin));
        let e2 = sample(u32x4_extract_lane::<2>(cos), u32x4_extract_lane::<2>(sin));
        let e3 = sample(u32x4_extract_lane::<3>(cos), u32x4_extract_lane::<3>(sin));
        let interp = |e0: v128, e1: v128, f: v128| {
            let a = i32x4_shuffle::<0, 2, 4, 6>(e0, e1);
            let b = i32x4_shuffle::<1, 3, 5, 7>(e0, e1);
            f32x4_add(a, f32x4_mul(f32x4_sub(b, a), f))
        };
        (
            interp(e0, e1, i32x4_shuffle::<0, 0, 1, 1>(f, f)),
            interp(e2, e3, i32x4_shuffle::<2, 2, 3, 3>(f, f)),
        )
    }

    /// `(cos, sin)` as a `CQ15` pair per lane of four phases, with `i16x8.q15mulr_sat_s`.
    #[cfg(feature = "nco-tables")]
    #[inline(always)]
    pub(crate) fn cos_sin_q15_x4(p: v128) -> v128 {
        let o = v128_and(u32x4_shr(p, 17), u32x4_splat(0x7FF8));
        let pair = |a: u32, b: u32| {
            u64x2(
                bytemuck::cast(entry_q15_at(a)),
                bytemuck::cast(entry_q15_at(b)),
            )
        };
        let e01 = pair(u32x4_extract_lane::<0>(o), u32x4_extract_lane::<1>(o));
        let e23 = pair(u32x4_extract_lane::<2>(o), u32x4_extract_lane::<3>(o));
        let a = i32x4_shuffle::<0, 2, 4, 6>(e01, e23);
        let d = i32x4_shuffle::<1, 3, 5, 7>(e01, e23);
        let f = v128_and(u32x4_shr(p, 5), u32x4_splat(0x7FFF));
        let f = v128_or(f, i32x4_shl(f, 16));
        i16x8_add(a, i16x8_q15mulr_sat(d, f))
    }
}

#[cfg(all(test, feature = "nco-tables"))]
mod tests {
    extern crate std;

    use std::vec;

    use fearless_simd::{Simd as SimdToken, prelude::*};
    use sdrk_core::tables::{COS_F32, COS_Q15};

    use super::{COS_SIN_F32, COS_SIN_Q15, SCALE, cos_sin_q15_packed, entry_q15};
    use crate::level::dispatch;

    #[test]
    fn every_entry_holds_the_neighbours_of_the_fallback_lookup() {
        for i in 0..4096 {
            let j = (i + 3072) % 4096;
            let [c, s, dc, ds] = COS_SIN_F32.0[i];
            assert_eq!(c.to_bits(), COS_F32[i].to_bits(), "{i}");
            assert_eq!(s.to_bits(), COS_F32[j].to_bits(), "{i}");
            // The scaled differences, and the scaling is exact.
            let (d0, d1) = (COS_F32[i + 1] - COS_F32[i], COS_F32[j + 1] - COS_F32[j]);
            assert_eq!(dc.to_bits(), (d0 * SCALE).to_bits(), "{i}");
            assert_eq!(ds.to_bits(), (d1 * SCALE).to_bits(), "{i}");
            assert_eq!((dc * 1_048_576.0).to_bits(), d0.to_bits(), "{i}");
            assert_eq!((ds * 1_048_576.0).to_bits(), d1.to_bits(), "{i}");
            let [c, s, dc, ds] = COS_SIN_Q15.0[i];
            assert_eq!([c, s], [COS_Q15[i], COS_Q15[j]], "{i}");
            assert_eq!(
                i32::from(dc),
                i32::from(COS_Q15[i + 1]) - i32::from(COS_Q15[i])
            );
            assert_eq!(
                i32::from(ds),
                i32::from(COS_Q15[j + 1]) - i32::from(COS_Q15[j])
            );
        }
    }

    /// Compares the lookup of the level with the scalar form, lane by lane.
    fn check_q15_lookup<S: SimdToken>(simd: S, phases: &[u32]) {
        for ps in phases.chunks_exact(S::u32s::LEN) {
            let got = cos_sin_q15_packed(simd, S::u32s::from_slice(simd, ps));
            for (&p, &got) in ps.iter().zip(got.as_slice()) {
                let [c, s, dc, ds] = entry_q15(p >> 20);
                let f15 = i32::from((p >> 5) as u16 & 0x7FFF);
                let interp = |a: i16, d: i16| {
                    a.wrapping_add(((i32::from(d) * f15 + (1 << 14)) >> 15) as i16)
                };
                let want = u32::from(interp(c, dc) as u16) | u32::from(interp(s, ds) as u16) << 16;
                assert_eq!(got, want, "p={p:#x}");
            }
        }
    }

    #[test]
    fn the_q15_lookup_of_the_level_gives_the_bits_of_the_scalar_lookup() {
        // Every index with the extreme fractions, and scattered phases for the bits between.
        let phases: vec::Vec<u32> = (0..4096u32)
            .flat_map(|i| [i << 20, i << 20 | 0xF_FFFF, i << 20 | 0x8_0010])
            .chain((0..1 << 16).map(|k: u32| k.wrapping_mul(0x9E37_79B9)))
            .collect();
        dispatch!(simd => check_q15_lookup(simd, &phases));
    }

    #[test]
    fn the_tables_are_aligned_to_their_entries() {
        assert_eq!(COS_SIN_F32.0.as_ptr() as usize % 16, 0);
        assert_eq!(COS_SIN_Q15.0.as_ptr() as usize % 8, 0);
    }
}
