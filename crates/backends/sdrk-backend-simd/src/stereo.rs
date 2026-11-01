//! `stereo_matrix`: `l = s + 2 d`, `r = s - 2 d` of every `(s, d)` sample, bit-exact with the
//! fallback (D108 to D110: the matrix is the only SIMD kernel of the stereo milestone; the
//! phase of every sample of `stereo_pll` depends on the previous detector output, so the PLL
//! has no SIMD branch).
//!
//! `C32`: two vectors of interleaved floats are unzipped into the `s` and `d` vectors,
//! `d2 = d + d`, `l = s + d2` and `r = s - d2` follow in this order (one rounding each, no
//! fused operation), and the zips write `l r l r ...` back. The operations are those of the
//! fallback on every lane, so NaN, infinities, `-0` and denormals give the same bits. On x86
//! the unzip is `shufps` and the zip `unpcklps`/`unpckhps`, which work within 128-bit lanes
//! and cancel each other (the arithmetic acts on every lane alike), so no cross-lane
//! permutation is needed at AVX2 and AVX-512 (SSE2 included); the other targets use the
//! generic unzip and zip of the token.
//!
//! `CQ15`: `s + 2 d` and `s - 2 d` are exact in 32 bits and the result is `sat16` of them.
//!
//! - SSE2 (so SSE4.2, which adds nothing here), AVX2, AVX-512: `pmaddwd` of the pair `(s, d)`
//!   with the constants `(1, 2)` and `(1, -2)` gives the exact 32-bit `l` and `r`, `packssdw`
//!   is `sat16`, and the unpacks interleave. Both packs and unpacks work within 128-bit lanes
//!   and cancel each other: the output of two input vectors is in order at every width.
//! - SIMD128: no branch (D156). The version with `i32x4.dot_i16x8_s`, `i16x8.narrow_i32x4_s`
//!   and a shuffle took 5.90 instructions per sample against 5.40 for the compiled fallback,
//!   so `CQ15` runs the fallback there, bit for bit the same result.
//! - NEON: `ld2`, the saturating chains `qadd(qadd(s, d), d)` and
//!   `qsub(qsub(s, d), d)`, `st2` (`C32`: `ld2`, the three float operations, `st2`). The chain equals `sat16(s + 2 d)`: if `s + d`
//!   is in range the second step is the exact saturating sum; if it clamps, `d` has the sign
//!   of the clamp and the second step pushes the same way, so the result is the same limit
//!   that `s + 2 d` is beyond. Likewise for the difference. The exhaustive sweep over all
//!   2^32 pairs (`tests/stereo_matrix_sweep.rs`, ignored, nightly) proves it.
//!
//! The rest of a block (fewer than one step) and every level without a path run the fallback.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::stereo::StereoMatrix;
use sdrk_core::{C32, CQ15, Fallback, Result};

use crate::Simd;
use crate::level::dispatch_fn;
use crate::util::check_len;

/// `l = s + (d + d)`, `r = s - (d + d)` of the samples of `input` (interleaved floats) in whole
/// steps of one vector of samples; `done` is the number of floats done.
#[inline(always)]
fn matrix_f32<S: SimdToken>(simd: S, input: &[f32], out: &mut [f32], done: &mut usize) {
    #[cfg(target_arch = "x86_64")]
    {
        let level = simd.level();
        if let Some(t) = level.as_avx512() {
            *done = x86::matrix_f32_avx512(t, input, out);
            return;
        }
        if let Some(t) = level.as_avx2() {
            *done = x86::matrix_f32_avx2(t, input, out);
            return;
        }
        if let Some(t) = level.as_sse2() {
            *done = x86::matrix_f32_sse2(t, input, out);
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = simd.level().as_neon() {
        *done = neon::matrix_f32(t, input, out);
        return;
    }
    let l = S::f32s::LEN;
    let n = input.len().min(out.len()) / (2 * l) * (2 * l);
    let (ib, _) = input.split_at(n);
    let (ob, _) = out.split_at_mut(n);
    for (i, o) in ib.chunks_exact(2 * l).zip(ob.chunks_exact_mut(2 * l)) {
        let (a, b) = i.split_at(l);
        let a = S::f32s::from_slice(simd, a);
        let b = S::f32s::from_slice(simd, b);
        let s = a.unzip_low(b);
        let d = a.unzip_high(b);
        let d2 = d + d;
        let (lv, rv) = (s + d2, s - d2);
        let (oa, ob) = o.split_at_mut(l);
        lv.zip_low(rv).store_slice(oa);
        lv.zip_high(rv).store_slice(ob);
    }
    *done = n;
}

dispatch_fn!(fn run_f32(input: &[f32], out: &mut [f32], done: &mut usize) => matrix_f32);

/// The level-specific `CQ15` path for the current level, if there is one; `done` is the number
/// of leading `i16` done.
#[inline(always)]
fn matrix_q15<S: SimdToken>(simd: S, input: &[i16], out: &mut [i16], done: &mut usize) {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx512() {
            *done = x86::matrix_avx512(t, input, out);
            return;
        }
        if let Some(t) = level.as_avx2() {
            *done = x86::matrix_avx2(t, input, out);
            return;
        }
        if let Some(t) = level.as_sse2() {
            *done = x86::matrix_sse2(t, input, out);
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        *done = neon::matrix(t, input, out);
        return;
    }
    let _ = (level, input, out, done);
}

dispatch_fn!(fn run_q15(input: &[i16], out: &mut [i16], done: &mut usize) => matrix_q15);

impl StereoMatrix<C32> for Simd {
    fn stereo_matrix(&self, input: &[C32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let x: &[f32] = bytemuck::cast_slice(input);
        let y: &mut [f32] = bytemuck::cast_slice_mut(output);
        let mut done = 0;
        run_f32(x, y, &mut done);
        finish(input, output, done / 2)
    }
}

impl StereoMatrix<CQ15> for Simd {
    fn stereo_matrix(&self, input: &[CQ15], output: &mut [CQ15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let x: &[i16] = bytemuck::cast_slice(input);
        let y: &mut [i16] = bytemuck::cast_slice_mut(output);
        let mut done = 0;
        run_q15(x, y, &mut done);
        finish(input, output, done / 2)
    }
}

/// The samples from `done` on, by the fallback.
fn finish<T>(input: &[T], output: &mut [T], done: usize) -> Result<()>
where
    Fallback: StereoMatrix<T>,
    T: sdrk_core::ComplexSample,
{
    match (input.get(done..), output.get_mut(done..)) {
        (Some(i), Some(o)) if !i.is_empty() => Fallback.stereo_matrix(i, o),
        _ => Ok(()),
    }
}

/// `$body` with `$i` and `$o` every block of `$step` values of `$input` and `$out`; evaluates
/// to the number of values done.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
macro_rules! blocks {
    ($input:expr, $out:expr, $step:expr, |$i:ident, $o:ident| $body:block) => {
        blocks!(i16, $input, $out, $step, |$i, $o| $body)
    };
    ($ty:ty, $input:expr, $out:expr, $step:expr, |$i:ident, $o:ident| $body:block) => {{
        crate::util::blocks($input, $out, $step, $step, |$i: &[$ty], $o: &mut [$ty]| {
            $body
        }) * $step
    }};
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::{
        _mm_madd_epi16, _mm_packs_epi32, _mm_set1_epi32, _mm_unpackhi_epi16, _mm_unpacklo_epi16,
        _mm256_madd_epi16, _mm256_packs_epi32, _mm256_set1_epi32, _mm256_unpackhi_epi16,
        _mm256_unpacklo_epi16, _mm512_madd_epi16, _mm512_packs_epi32, _mm512_set1_epi32,
        _mm512_unpackhi_epi16, _mm512_unpacklo_epi16,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{i16x8, i16x16, i16x32};

    /// The pairs `(1, 2)` and `(1, -2)` as 32-bit lanes: `pmaddwd` with `(s, d)` is `s + 2 d`
    /// and `s - 2 d`.
    const PLUS: i32 = 0x0002_0001;
    const MINUS: i32 = 0xFFFE_0001_u32 as i32;

    use core::arch::x86_64::{
        _mm_add_ps, _mm_shuffle_ps, _mm_sub_ps, _mm_unpackhi_ps, _mm_unpacklo_ps, _mm256_add_ps,
        _mm256_shuffle_ps, _mm256_sub_ps, _mm256_unpackhi_ps, _mm256_unpacklo_ps, _mm512_add_ps,
        _mm512_shuffle_ps, _mm512_sub_ps, _mm512_unpackhi_ps, _mm512_unpacklo_ps,
    };
    use fearless_simd::{f32x4, f32x8, f32x16};

    /// `C32`: the shuffles work within 128-bit lanes. `shufps` gives `s` from the even and `d`
    /// from the odd floats of the lane of both inputs, so `s` holds the pairs 0 and 1 of `a` and
    /// of `b` in every lane; the unpacks of `l` and `r` in the same lane order give the pairs of
    /// `a` (low) and of `b` (high) again. No cross-lane permutation is needed, as the arithmetic
    /// acts on every lane alike.
    macro_rules! level_f32 {
        ($name:ident, $tok:ident, $vec:ident, $v:literal, $shuf:ident, $add:ident, $sub:ident,
         $lo:ident, $hi:ident) => {
            fearless_simd::kernel!(
                pub(super) fn $name(t: $tok, input: &[f32], out: &mut [f32]) -> usize {
                    let pair = |i: &[f32], o: &mut [f32]| {
                        let (a, b) = i.split_at($v);
                        let a = $vec::from_slice(t, a).into();
                        let b = $vec::from_slice(t, b).into();
                        let s = $shuf::<0x88>(a, b);
                        let d = $shuf::<0xDD>(a, b);
                        let d2 = $add(d, d);
                        let (l, r) = ($add(s, d2), $sub(s, d2));
                        let (oa, ob) = o.split_at_mut($v);
                        $vec::simd_from(t, $lo(l, r)).store_slice(oa);
                        $vec::simd_from(t, $hi(l, r)).store_slice(ob);
                    };
                    // Two pairs per iteration (the compiler of the run-time dispatch build does
                    // not unroll), then the pair that may remain.
                    let n = blocks!(f32, input, out, 4 * $v, |i, o| {
                        let ((i0, i1), (o0, o1)) = (i.split_at(2 * $v), o.split_at_mut(2 * $v));
                        pair(i0, o0);
                        pair(i1, o1);
                    });
                    let rest = (input.get(n..), out.get_mut(n..));
                    let (Some(ri), Some(ro)) = rest else {
                        return n;
                    };
                    n + blocks!(f32, ri, ro, 2 * $v, |i, o| {
                        pair(i, o);
                    })
                }
            );
        };
    }

    level_f32!(
        matrix_f32_sse2,
        Sse2,
        f32x4,
        4,
        _mm_shuffle_ps,
        _mm_add_ps,
        _mm_sub_ps,
        _mm_unpacklo_ps,
        _mm_unpackhi_ps
    );
    level_f32!(
        matrix_f32_avx2,
        Avx2,
        f32x8,
        8,
        _mm256_shuffle_ps,
        _mm256_add_ps,
        _mm256_sub_ps,
        _mm256_unpacklo_ps,
        _mm256_unpackhi_ps
    );
    level_f32!(
        matrix_f32_avx512,
        Avx512,
        f32x16,
        16,
        _mm512_shuffle_ps,
        _mm512_add_ps,
        _mm512_sub_ps,
        _mm512_unpacklo_ps,
        _mm512_unpackhi_ps
    );

    macro_rules! level {
        ($name:ident, $tok:ident, $vec:ident, $v:literal, $set1:ident, $madd:ident,
         $packs:ident, $lo:ident, $hi:ident) => {
            fearless_simd::kernel!(
                pub(super) fn $name(t: $tok, input: &[i16], out: &mut [i16]) -> usize {
                    let (cp, cm) = ($set1(PLUS), $set1(MINUS));
                    let pair = |i: &[i16], o: &mut [i16]| {
                        let (a, b) = i.split_at($v);
                        let a = $vec::from_slice(t, a).into();
                        let b = $vec::from_slice(t, b).into();
                        let l = $packs($madd(a, cp), $madd(b, cp));
                        let r = $packs($madd(a, cm), $madd(b, cm));
                        let (oa, ob) = o.split_at_mut($v);
                        $vec::simd_from(t, $lo(l, r)).store_slice(oa);
                        $vec::simd_from(t, $hi(l, r)).store_slice(ob);
                    };
                    // Two pairs per iteration, then the pair that may remain (see `level_f32`).
                    let n = blocks!(input, out, 4 * $v, |i, o| {
                        let ((i0, i1), (o0, o1)) = (i.split_at(2 * $v), o.split_at_mut(2 * $v));
                        pair(i0, o0);
                        pair(i1, o1);
                    });
                    let rest = (input.get(n..), out.get_mut(n..));
                    let (Some(ri), Some(ro)) = rest else {
                        return n;
                    };
                    n + blocks!(ri, ro, 2 * $v, |i, o| {
                        pair(i, o);
                    })
                }
            );
        };
    }

    level!(
        matrix_sse2,
        Sse2,
        i16x8,
        8,
        _mm_set1_epi32,
        _mm_madd_epi16,
        _mm_packs_epi32,
        _mm_unpacklo_epi16,
        _mm_unpackhi_epi16
    );
    level!(
        matrix_avx2,
        Avx2,
        i16x16,
        16,
        _mm256_set1_epi32,
        _mm256_madd_epi16,
        _mm256_packs_epi32,
        _mm256_unpacklo_epi16,
        _mm256_unpackhi_epi16
    );
    level!(
        matrix_avx512,
        Avx512,
        i16x32,
        32,
        _mm512_set1_epi32,
        _mm512_madd_epi16,
        _mm512_packs_epi32,
        _mm512_unpacklo_epi16,
        _mm512_unpackhi_epi16
    );
}

/// NEON: `ld2` and `st2` deinterleave and interleave in the load and the store, so a block is
/// one load, the arithmetic and one store per vector of pairs.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon {
    use core::arch::aarch64::{
        float32x4x2_t, int16x8x2_t, vaddq_f32, vld2q_f32, vld2q_s16, vqaddq_s16, vqsubq_s16,
        vst2q_f32, vst2q_s16, vsubq_f32,
    };

    fearless_simd::kernel!(
        pub(super) fn matrix_f32(t: Neon, input: &[f32], out: &mut [f32]) -> usize {
            let _ = t;
            // Four pairs (8 floats) per `ld2`/`st2`; two per iteration, then one that may remain.
            let pair = |i: &[f32], o: &mut [f32]| {
                // SAFETY: `i` and `o` are chunks of exactly 8 floats, which `vld2q_f32` reads
                // and `vst2q_f32` writes.
                unsafe {
                    let float32x4x2_t(s, d) = vld2q_f32(i.as_ptr());
                    let d2 = vaddq_f32(d, d);
                    vst2q_f32(
                        o.as_mut_ptr(),
                        float32x4x2_t(vaddq_f32(s, d2), vsubq_f32(s, d2)),
                    );
                }
            };
            let n = blocks!(f32, input, out, 16, |i, o| {
                let ((i0, i1), (o0, o1)) = (i.split_at(8), o.split_at_mut(8));
                pair(i0, o0);
                pair(i1, o1);
            });
            let rest = (input.get(n..), out.get_mut(n..));
            let (Some(ri), Some(ro)) = rest else {
                return n;
            };
            n + blocks!(f32, ri, ro, 8, |i, o| {
                pair(i, o);
            })
        }
    );

    fearless_simd::kernel!(
        pub(super) fn matrix(t: Neon, input: &[i16], out: &mut [i16]) -> usize {
            let _ = t;
            // Eight pairs (16 values) per `ld2`/`st2`; two per iteration, then one that may
            // remain.
            let pair = |i: &[i16], o: &mut [i16]| {
                // SAFETY: `i` and `o` are chunks of exactly 16 values, which `vld2q_s16` reads
                // and `vst2q_s16` writes.
                unsafe {
                    let int16x8x2_t(s, d) = vld2q_s16(i.as_ptr());
                    let l = vqaddq_s16(vqaddq_s16(s, d), d);
                    let r = vqsubq_s16(vqsubq_s16(s, d), d);
                    vst2q_s16(o.as_mut_ptr(), int16x8x2_t(l, r));
                }
            };
            let n = blocks!(i16, input, out, 32, |i, o| {
                let ((i0, i1), (o0, o1)) = (i.split_at(16), o.split_at_mut(16));
                pair(i0, o0);
                pair(i1, o1);
            });
            let rest = (input.get(n..), out.get_mut(n..));
            let (Some(ri), Some(ro)) = rest else {
                return n;
            };
            n + blocks!(i16, ri, ro, 16, |i, o| {
                pair(i, o);
            })
        }
    );
}
