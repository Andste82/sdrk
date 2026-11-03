//! `add`, `add_real`, `mul`, `mul_real`, `interleave` and `deinterleave`, bit-exact with the
//! fallback (the contract of these kernels is bit-exact, `docs/kernels.md`).
//!
//! `C32` and `f32`: the sum and the product of every float in generic code (one operation per
//! float, no fused operation, so NaN, infinities, `-0` and denormals give the bits of the
//! fallback). The complex product loads the two operands deinterleaved (the [`Split`] of
//! `rotate`: `shufps` and `unpcklps` within the 128-bit blocks on AVX2 and AVX-512, `ld2` and
//! `st2` on NEON, the zips of the token elsewhere) and computes `ar br - ai bi` and
//! `ar bi + ai br` as two multiplications and one subtraction or addition each.
//!
//! `Q15`, sums: `saturating_add` of the token (`paddsw`, `sqadd`, `i16x8.add_sat_s`).
//!
//! `Q15`, products of real streams: the rounding high multiplication (`pmulhrsw`, `sqrdmulh`,
//! `i16x8.q15mulr_sat`) as in `scale`; `pmulhrsw` wraps `-32768 * -32768` to `-32768`, which is
//! the only lane that gives `0x8000`, so a compare and an exclusive or (a masked blend on
//! AVX-512) turn it into `32767`. SSE2 has no rounding high multiplication and runs the
//! fallback.
//!
//! `CQ15` complex product, whose sums of two products reach 2^31:
//!
//! - x86 (SSE4.2 and AVX2; AVX-512 runs the AVX2 path): `pmaddwd` on the `(I, Q)` pairs.
//!   With `a = (ar, ai)` and `b = (br, bi)`, `pmaddwd(a, (br, !bi))` is
//!   `ar br - ai bi - ai` (`!bi = -bi - 1` cannot overflow where `-bi` can), so adding `ai` gives
//!   the real part; the sum wraps modulo 2^32 but the result is in range, so it is exact.
//!   `pmaddwd(a, (bi, br))` is the imaginary part; it overflows only when all four values are
//!   `-32768` (the sum 2^31 wraps to `i32::MIN`, a value no other input gives), which a compare
//!   after the sum turns into `32767` after the pack. The rounding term, the shift by 15 and
//!   `packssdw` are `round_shift` and `sat16`; `pshufb` interleaves the packed parts.
//! - NEON: `ld2`, the widening `smull`/`smlsl`, a saturating `sqadd` for the imaginary part
//!   (the sum 2^31 saturates to a value that `sat16` clamps the same), `sqrshrn` (the rounding
//!   shift with saturation) and `st2`.
//! - SIMD128 and SSE2: no path (the fallback runs).
//!
//! `interleave` and `deinterleave`: the zips and unzips of the token. The rest of a call (fewer
//! samples than one step) and every level without a path run the plain loops of the fallback.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::math::{Add, AddReal, Interleave, Mul, MulReal};
use sdrk_core::kernels::mix::{Nco, Oscillate};
use sdrk_core::{C32, CQ15, Fallback, Q15, Result};

use crate::Simd;
use crate::level::dispatch_fn;
use crate::mix::{Portable, Split};
use crate::table::cos_sin_f32;
use crate::util::check_len;

/// The three lengths of an element-wise call must be equal.
#[inline(always)]
fn check3(a: usize, b: usize, out: usize) -> Result<()> {
    check_len(a, b)?;
    check_len(a, out)
}

// f32 sums and products (also the components of C32 sums)

/// One vector of `a + b` or `a * b`.
#[inline(always)]
fn vop<S: SimdToken, const MUL: bool>(simd: S, a: &[f32], b: &[f32], o: &mut [f32]) {
    let (x, y) = (S::f32s::from_slice(simd, a), S::f32s::from_slice(simd, b));
    if MUL { x * y } else { x + y }.store_slice(o);
}

/// `out = a + b` (`MUL = false`) or `a * b` (`MUL = true`) of every float.
#[inline(always)]
fn zip_f32<S: SimdToken, const MUL: bool>(simd: S, a: &[f32], b: &[f32], out: &mut [f32]) {
    let l = S::f32s::LEN;
    let step = 8 * l;
    let n = a.len().min(b.len()).min(out.len());
    let k = n / step * step;
    let (ah, at) = a.split_at(k);
    let (bh, bt) = b.split_at(k);
    let (oh, ot) = out.split_at_mut(k);
    for ((a, b), o) in ah
        .chunks_exact(step)
        .zip(bh.chunks_exact(step))
        .zip(oh.chunks_exact_mut(step))
    {
        for ((a, b), o) in a
            .chunks_exact(l)
            .zip(b.chunks_exact(l))
            .zip(o.chunks_exact_mut(l))
        {
            vop::<S, MUL>(simd, a, b, o);
        }
    }
    let mut at = at.chunks_exact(l);
    let mut bt = bt.chunks_exact(l);
    let mut ot = ot.chunks_exact_mut(l);
    for ((a, b), o) in (&mut at).zip(&mut bt).zip(&mut ot) {
        vop::<S, MUL>(simd, a, b, o);
    }
    for ((a, b), o) in at
        .remainder()
        .iter()
        .zip(bt.remainder())
        .zip(ot.into_remainder())
    {
        *o = if MUL { *a * *b } else { *a + *b };
    }
}

dispatch_fn!(fn run_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) => zip_f32::<_, false>);
dispatch_fn!(fn run_mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) => zip_f32::<_, true>);

// Q15 sums

#[inline(always)]
fn vadd_i16<S: SimdToken>(simd: S, a: &[i16], b: &[i16], o: &mut [i16]) {
    let (x, y) = (S::i16s::from_slice(simd, a), S::i16s::from_slice(simd, b));
    x.saturating_add(y).store_slice(o);
}

#[inline(always)]
fn add_i16<S: SimdToken>(simd: S, a: &[i16], b: &[i16], out: &mut [i16]) {
    let l = S::i16s::LEN;
    let step = 8 * l;
    let n = a.len().min(b.len()).min(out.len());
    let k = n / step * step;
    let (ah, at) = a.split_at(k);
    let (bh, bt) = b.split_at(k);
    let (oh, ot) = out.split_at_mut(k);
    for ((a, b), o) in ah
        .chunks_exact(step)
        .zip(bh.chunks_exact(step))
        .zip(oh.chunks_exact_mut(step))
    {
        for ((a, b), o) in a
            .chunks_exact(l)
            .zip(b.chunks_exact(l))
            .zip(o.chunks_exact_mut(l))
        {
            vadd_i16(simd, a, b, o);
        }
    }
    let mut at = at.chunks_exact(l);
    let mut bt = bt.chunks_exact(l);
    let mut ot = ot.chunks_exact_mut(l);
    for ((a, b), o) in (&mut at).zip(&mut bt).zip(&mut ot) {
        vadd_i16(simd, a, b, o);
    }
    for ((a, b), o) in at
        .remainder()
        .iter()
        .zip(bt.remainder())
        .zip(ot.into_remainder())
    {
        *o = a.saturating_add(*b);
    }
}

dispatch_fn!(fn run_add_i16(a: &[i16], b: &[i16], out: &mut [i16]) => add_i16);

// C32 product

/// The complex products of `a` and `b` (interleaved floats), `re = ar br - ai bi` and
/// `im = ar bi + ai br` with separate multiplications.
#[inline(always)]
fn mul_split<S: SimdToken, L: Split<S>>(simd: S, split: L, a: &[f32], b: &[f32], out: &mut [f32]) {
    let step = 2 * S::f32s::LEN;
    let n = a.len().min(b.len()).min(out.len());
    let k = n / step * step;
    let (ah, at) = a.split_at(k);
    let (bh, bt) = b.split_at(k);
    let (oh, ot) = out.split_at_mut(k);
    for ((a, b), o) in ah
        .chunks_exact(step)
        .zip(bh.chunks_exact(step))
        .zip(oh.chunks_exact_mut(step))
    {
        let (ar, ai) = split.load(simd, a);
        let (br, bi) = split.load(simd, b);
        let re = ar * br - ai * bi;
        let im = ar * bi + ai * br;
        split.store(simd, re, im, o);
    }
    for ((a, b), o) in at
        .chunks_exact(2)
        .zip(bt.chunks_exact(2))
        .zip(ot.chunks_exact_mut(2))
    {
        if let ([ar, ai], [br, bi], [or, oi]) = (a, b, o) {
            *or = *ar * *br - *ai * *bi;
            *oi = *ar * *bi + *ai * *br;
        }
    }
}

/// [`mul_split`] with the [`Split`] of the level of `simd`.
#[inline(always)]
fn mul_c32<S: SimdToken>(simd: S, a: &[f32], b: &[f32], out: &mut [f32]) {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx512() {
            return mul_split(t, crate::mix::x86::InLane, a, b, out);
        }
        if let Some(t) = level.as_avx2() {
            return mul_split(t, crate::mix::x86::InLane, a, b, out);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return mul_split(t, crate::mix::neon::Ld2, a, b, out);
    }
    let _ = level;
    mul_split(simd, Portable, a, b, out);
}

dispatch_fn!(fn run_mul_c32(a: &[f32], b: &[f32], out: &mut [f32]) => mul_c32);

// Q15 and CQ15 products

/// The level-specific path of the `Q15` product of real streams for the current level: returns
/// the number of leading values done.
#[inline(always)]
fn mul_q15_level<S: SimdToken>(simd: S, a: &[i16], b: &[i16], out: &mut [i16]) -> usize {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx512() {
            return x86::mul_real_avx512(t, a, b, out);
        }
        if let Some(t) = level.as_avx2() {
            return x86::mul_real_avx2(t, a, b, out);
        }
        if let Some(t) = level.as_sse4_2() {
            return x86::mul_real_sse4_2(t, a, b, out);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return neon::mul_real(t, a, b, out);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return wasm::mul_real(t, a, b, out);
    }
    let _ = (level, a, b, out);
    0
}

#[inline(always)]
fn mul_real_q15<S: SimdToken>(simd: S, a: &[i16], b: &[i16], out: &mut [i16], done: &mut usize) {
    *done = mul_q15_level(simd, a, b, out);
}

/// The level-specific path of the `CQ15` complex product for the current level: returns the
/// number of leading values done (a multiple of two).
#[inline(always)]
fn mul_cq15_level<S: SimdToken>(simd: S, a: &[i16], b: &[i16], out: &mut [i16]) -> usize {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx2() {
            return x86::mul_cq15_avx2(t, a, b, out);
        }
        if let Some(t) = level.as_sse4_2() {
            return x86::mul_cq15_sse4_2(t, a, b, out);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return neon::mul_cq15(t, a, b, out);
    }
    let _ = (level, a, b, out);
    0
}

#[inline(always)]
fn mul_cq15<S: SimdToken>(simd: S, a: &[i16], b: &[i16], out: &mut [i16], done: &mut usize) {
    *done = mul_cq15_level(simd, a, b, out);
}

dispatch_fn!(fn run_mul_real_q15(a: &[i16], b: &[i16], out: &mut [i16], done: &mut usize) => mul_real_q15);
dispatch_fn!(fn run_mul_cq15(a: &[i16], b: &[i16], out: &mut [i16], done: &mut usize) => mul_cq15);

/// `$body` with `$x`, `$y` and `$z` the blocks of `$va`, `$vb` and `$vc` values of `$a`, `$b` and
/// `$c`: groups of eight blocks first (unrolled), then single blocks. Evaluates to the number of
/// blocks done. The loops are written like those of `scale`: one closure for both, so that the
/// build with run-time detection compiles the same code as the build for a fixed level.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! blocks3 {
    ($a:expr, $va:literal, $b:expr, $vb:literal, $c:expr, $vc:literal, |$x:ident, $y:ident, $z:ident| $body:block) => {{
        let n = ($a.len() / $va).min($b.len() / $vb).min($c.len() / $vc);
        let (a, _) = $a.split_at(n * $va);
        let (b, _) = $b.split_at(n * $vb);
        let (c, _) = $c.split_at_mut(n * $vc);
        let body = |$x: &[_; $va], $y: &[_; $vb], $z: &mut [_; $vc]| $body;
        let g = n / 8 * 8;
        let (ag, ar) = a.split_at(g * $va);
        let (bg, br) = b.split_at(g * $vb);
        let (cg, cr) = c.split_at_mut(g * $vc);
        let (ag, _) = ag.as_chunks::<{ 8 * $va }>();
        let (bg, _) = bg.as_chunks::<{ 8 * $vb }>();
        let (cg, _) = cg.as_chunks_mut::<{ 8 * $vc }>();
        for ((aq, bq), cq) in ag.iter().zip(bg).zip(cg) {
            let (aq, _) = aq.as_chunks::<$va>();
            let (bq, _) = bq.as_chunks::<$vb>();
            let (cq, _) = cq.as_chunks_mut::<$vc>();
            for ((x, y), z) in aq.iter().zip(bq).zip(cq) {
                body(x, y, z);
            }
        }
        let (ar, _) = ar.as_chunks::<$va>();
        let (br, _) = br.as_chunks::<$vb>();
        let (cr, _) = cr.as_chunks_mut::<$vc>();
        for ((x, y), z) in ar.iter().zip(br).zip(cr) {
            body(x, y, z);
        }
        n
    }};
}

/// [`blocks3`] with the second stream written too (`deinterleave`).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
macro_rules! blocks3m {
    ($a:expr, $va:literal, $b:expr, $vb:literal, $c:expr, $vc:literal, |$x:ident, $y:ident, $z:ident| $body:block) => {{
        let n = ($a.len() / $va).min($b.len() / $vb).min($c.len() / $vc);
        let (a, _) = $a.split_at(n * $va);
        let (b, _) = $b.split_at_mut(n * $vb);
        let (c, _) = $c.split_at_mut(n * $vc);
        let body = |$x: &[_; $va], $y: &mut [_; $vb], $z: &mut [_; $vc]| $body;
        let g = n / 8 * 8;
        let (ag, ar) = a.split_at(g * $va);
        let (bg, br) = b.split_at_mut(g * $vb);
        let (cg, cr) = c.split_at_mut(g * $vc);
        let (ag, _) = ag.as_chunks::<{ 8 * $va }>();
        let (bg, _) = bg.as_chunks_mut::<{ 8 * $vb }>();
        let (cg, _) = cg.as_chunks_mut::<{ 8 * $vc }>();
        for ((aq, bq), cq) in ag.iter().zip(bg).zip(cg) {
            let (aq, _) = aq.as_chunks::<$va>();
            let (bq, _) = bq.as_chunks_mut::<$vb>();
            let (cq, _) = cq.as_chunks_mut::<$vc>();
            for ((x, y), z) in aq.iter().zip(bq).zip(cq) {
                body(x, y, z);
            }
        }
        let (ar, _) = ar.as_chunks::<$va>();
        let (br, _) = br.as_chunks_mut::<$vb>();
        let (cr, _) = cr.as_chunks_mut::<$vc>();
        for ((x, y), z) in ar.iter().zip(br).zip(cr) {
            body(x, y, z);
        }
        n
    }};
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::{
        _mm_add_epi32, _mm_cmpeq_epi16, _mm_cmpeq_epi32, _mm_madd_epi16, _mm_mulhrs_epi16,
        _mm_packs_epi32, _mm_set1_epi16, _mm_set1_epi32, _mm_setr_epi8, _mm_setzero_si128,
        _mm_shuffle_epi8, _mm_srai_epi32, _mm_xor_si128, _mm256_add_epi32, _mm256_cmpeq_epi16,
        _mm256_cmpeq_epi32, _mm256_madd_epi16, _mm256_mulhrs_epi16, _mm256_packs_epi32,
        _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi8, _mm256_setzero_si256,
        _mm256_shuffle_epi8, _mm256_srai_epi32, _mm256_xor_si256, _mm512_cmpeq_epi16_mask,
        _mm512_mask_blend_epi16, _mm512_mulhrs_epi16, _mm512_set1_epi16,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{i16x8, i16x16, i16x32};

    fearless_simd::kernel!(
        pub(super) fn mul_real_sse4_2(t: Sse4_2, a: &[i16], b: &[i16], out: &mut [i16]) -> usize {
            let min = _mm_set1_epi16(i16::MIN);
            blocks3!(a, 8, b, 8, out, 8, |a, b, o| {
                let r = _mm_mulhrs_epi16(
                    i16x8::from_slice(t, a).into(),
                    i16x8::from_slice(t, b).into(),
                );
                i16x8::simd_from(t, _mm_xor_si128(r, _mm_cmpeq_epi16(r, min))).store_slice(o);
            }) * 8
        }
    );

    fearless_simd::kernel!(
        pub(super) fn mul_real_avx2(t: Avx2, a: &[i16], b: &[i16], out: &mut [i16]) -> usize {
            let min = _mm256_set1_epi16(i16::MIN);
            blocks3!(a, 16, b, 16, out, 16, |a, b, o| {
                let r = _mm256_mulhrs_epi16(
                    i16x16::from_slice(t, a).into(),
                    i16x16::from_slice(t, b).into(),
                );
                i16x16::simd_from(t, _mm256_xor_si256(r, _mm256_cmpeq_epi16(r, min)))
                    .store_slice(o);
            }) * 16
        }
    );

    fearless_simd::kernel!(
        pub(super) fn mul_real_avx512(t: Avx512, a: &[i16], b: &[i16], out: &mut [i16]) -> usize {
            let min = _mm512_set1_epi16(i16::MIN);
            let max = _mm512_set1_epi16(i16::MAX);
            blocks3!(a, 32, b, 32, out, 32, |a, b, o| {
                let r = _mm512_mulhrs_epi16(
                    i16x32::from_slice(t, a).into(),
                    i16x32::from_slice(t, b).into(),
                );
                let m = _mm512_cmpeq_epi16_mask(r, min);
                i16x32::simd_from(t, _mm512_mask_blend_epi16(m, r, max)).store_slice(o);
            }) * 32
        }
    );

    fearless_simd::kernel!(
        pub(super) fn mul_cq15_sse4_2(t: Sse4_2, a: &[i16], b: &[i16], out: &mut [i16]) -> usize {
            let not_im = _mm_set1_epi32(!0xFFFF);
            let swap = _mm_setr_epi8(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);
            let weave = _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
            let (min, round, zero) = (
                _mm_set1_epi32(i32::MIN),
                _mm_set1_epi32(1 << 14),
                _mm_setzero_si128(),
            );
            blocks3!(a, 8, b, 8, out, 8, |a, b, o| {
                let x = i16x8::from_slice(t, a).into();
                let y = i16x8::from_slice(t, b).into();
                let re = _mm_add_epi32(
                    _mm_madd_epi16(x, _mm_xor_si128(y, not_im)),
                    _mm_srai_epi32::<16>(x),
                );
                let im = _mm_madd_epi16(x, _mm_shuffle_epi8(y, swap));
                let ovf = _mm_cmpeq_epi32(im, min);
                let re = _mm_srai_epi32::<15>(_mm_add_epi32(re, round));
                let im = _mm_srai_epi32::<15>(_mm_add_epi32(im, round));
                let p = _mm_xor_si128(_mm_packs_epi32(re, im), _mm_packs_epi32(zero, ovf));
                i16x8::simd_from(t, _mm_shuffle_epi8(p, weave)).store_slice(o);
            }) * 8
        }
    );

    fearless_simd::kernel!(
        pub(super) fn mul_cq15_avx2(t: Avx2, a: &[i16], b: &[i16], out: &mut [i16]) -> usize {
            let not_im = _mm256_set1_epi32(!0xFFFF);
            let swap = _mm256_setr_epi8(
                2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10,
                11, 8, 9, 14, 15, 12, 13,
            );
            let weave = _mm256_setr_epi8(
                0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4,
                5, 12, 13, 6, 7, 14, 15,
            );
            let (min, round, zero) = (
                _mm256_set1_epi32(i32::MIN),
                _mm256_set1_epi32(1 << 14),
                _mm256_setzero_si256(),
            );
            blocks3!(a, 16, b, 16, out, 16, |a, b, o| {
                let x = i16x16::from_slice(t, a).into();
                let y = i16x16::from_slice(t, b).into();
                let re = _mm256_add_epi32(
                    _mm256_madd_epi16(x, _mm256_xor_si256(y, not_im)),
                    _mm256_srai_epi32::<16>(x),
                );
                let im = _mm256_madd_epi16(x, _mm256_shuffle_epi8(y, swap));
                let ovf = _mm256_cmpeq_epi32(im, min);
                let re = _mm256_srai_epi32::<15>(_mm256_add_epi32(re, round));
                let im = _mm256_srai_epi32::<15>(_mm256_add_epi32(im, round));
                let p = _mm256_xor_si256(_mm256_packs_epi32(re, im), _mm256_packs_epi32(zero, ovf));
                i16x16::simd_from(t, _mm256_shuffle_epi8(p, weave)).store_slice(o);
            }) * 16
        }
    );
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon {
    use core::arch::aarch64::{
        int16x8x2_t, vcombine_s16, vget_low_s16, vld2q_s16, vmlsl_high_s16, vmlsl_s16,
        vmull_high_s16, vmull_s16, vqaddq_s32, vqrdmulhq_s16, vqrshrn_n_s32, vst2q_s16,
    };
    use fearless_simd::i16x8;
    use fearless_simd::prelude::*;

    fearless_simd::kernel!(
        pub(super) fn mul_real(t: Neon, a: &[i16], b: &[i16], out: &mut [i16]) -> usize {
            blocks3!(a, 8, b, 8, out, 8, |a, b, o| {
                let r = vqrdmulhq_s16(
                    i16x8::from_slice(t, a).into(),
                    i16x8::from_slice(t, b).into(),
                );
                i16x8::simd_from(t, r).store_slice(o);
            }) * 8
        }
    );

    fearless_simd::kernel!(
        pub(super) fn mul_cq15(_t: Neon, a: &[i16], b: &[i16], out: &mut [i16]) -> usize {
            blocks3!(a, 16, b, 16, out, 16, |a, b, o| {
                // SAFETY: `a` and `b` are the 16 values that `vld2q_s16` reads and `o` the 16
                // values that `vst2q_s16` writes.
                unsafe {
                    let (x, y) = (vld2q_s16(a.as_ptr()), vld2q_s16(b.as_ptr()));
                    let (ar, ai, br, bi) = (x.0, x.1, y.0, y.1);
                    let re_lo = vmlsl_s16(
                        vmull_s16(vget_low_s16(ar), vget_low_s16(br)),
                        vget_low_s16(ai),
                        vget_low_s16(bi),
                    );
                    let re_hi = vmlsl_high_s16(vmull_high_s16(ar, br), ai, bi);
                    let im_lo = vqaddq_s32(
                        vmull_s16(vget_low_s16(ar), vget_low_s16(bi)),
                        vmull_s16(vget_low_s16(ai), vget_low_s16(br)),
                    );
                    let im_hi = vqaddq_s32(vmull_high_s16(ar, bi), vmull_high_s16(ai, br));
                    let re = vcombine_s16(vqrshrn_n_s32::<15>(re_lo), vqrshrn_n_s32::<15>(re_hi));
                    let im = vcombine_s16(vqrshrn_n_s32::<15>(im_lo), vqrshrn_n_s32::<15>(im_hi));
                    vst2q_s16(o.as_mut_ptr(), int16x8x2_t(re, im));
                }
            }) * 16
        }
    );
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::i16x8_q15mulr_sat;
    use fearless_simd::i16x8;
    use fearless_simd::prelude::*;

    fearless_simd::kernel!(
        pub(super) fn mul_real(t: WasmSimd128, a: &[i16], b: &[i16], out: &mut [i16]) -> usize {
            blocks3!(a, 8, b, 8, out, 8, |a, b, o| {
                let r = i16x8_q15mulr_sat(
                    i16x8::from_slice(t, a).into(),
                    i16x8::from_slice(t, b).into(),
                );
                i16x8::simd_from(t, r).store_slice(o);
            }) * 8
        }
    );
}

/// The x86 paths of `interleave` and `deinterleave`: the in-lane unpacks and shuffles of the
/// level, with the cross-lane permutation of AVX2 where the lanes of 128 bits have to be put
/// in order (`vpermpd`, one per vector). Each function returns the number of samples done (zero
/// where the generic code of the token is as good).
#[cfg(target_arch = "x86_64")]
mod split_x86 {
    use core::arch::x86_64::{
        __m128i, _mm_setr_epi8, _mm_shuffle_epi8, _mm_unpackhi_epi16, _mm_unpackhi_epi64,
        _mm_unpacklo_epi16, _mm_unpacklo_epi64, _mm256_castpd_ps, _mm256_castps_pd,
        _mm256_permute4x64_pd, _mm256_setr_epi8, _mm256_shuffle_epi8, _mm256_shuffle_ps,
        _mm256_unpackhi_epi16, _mm256_unpackhi_epi64, _mm256_unpackhi_ps, _mm256_unpacklo_epi16,
        _mm256_unpacklo_epi64, _mm256_unpacklo_ps,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{Sse2, Sse4_2, f32x8, i16x8, i16x16};

    pub(super) mod int_f32 {
        #[allow(
            clippy::wildcard_imports,
            reason = "the intrinsics and vector types of the parent"
        )]
        use super::*;

        fearless_simd::kernel!(
            pub(crate) fn avx2(t: Avx2, re: &[f32], im: &[f32], out: &mut [f32]) -> usize {
                blocks3!(re, 8, im, 8, out, 16, |r, i, o| {
                    let r = f32x8::from_slice(t, r).into();
                    let i = f32x8::from_slice(t, i).into();
                    // Samples 0, 1, 4, 5 | 2, 3, 6, 7 of the real parts and of the imaginary
                    // parts, so that the in-lane unpacks give the samples in order.
                    let r = _mm256_castpd_ps(_mm256_permute4x64_pd::<0xD8>(_mm256_castps_pd(r)));
                    let i = _mm256_castpd_ps(_mm256_permute4x64_pd::<0xD8>(_mm256_castps_pd(i)));
                    let (o0, o1) = o.split_at_mut(8);
                    f32x8::simd_from(t, _mm256_unpacklo_ps(r, i)).store_slice(o0);
                    f32x8::simd_from(t, _mm256_unpackhi_ps(r, i)).store_slice(o1);
                }) * 8
            }
        );

        pub(crate) fn sse2(_t: Sse2, _re: &[f32], _im: &[f32], _out: &mut [f32]) -> usize {
            0
        }
    }

    pub(super) mod deint_f32 {
        #[allow(
            clippy::wildcard_imports,
            reason = "the intrinsics and vector types of the parent"
        )]
        use super::*;

        fearless_simd::kernel!(
            pub(crate) fn avx2(t: Avx2, input: &[f32], re: &mut [f32], im: &mut [f32]) -> usize {
                blocks3m!(input, 16, re, 8, im, 8, |c, r, i| {
                    let (a, b) = c.split_at(8);
                    let a = f32x8::from_slice(t, a).into();
                    let b = f32x8::from_slice(t, b).into();
                    // Samples 0, 1, 4, 5 | 2, 3, 6, 7, then the lanes in order.
                    let lo = _mm256_shuffle_ps::<0x88>(a, b);
                    let hi = _mm256_shuffle_ps::<0xDD>(a, b);
                    let lo = _mm256_castpd_ps(_mm256_permute4x64_pd::<0xD8>(_mm256_castps_pd(lo)));
                    let hi = _mm256_castpd_ps(_mm256_permute4x64_pd::<0xD8>(_mm256_castps_pd(hi)));
                    f32x8::simd_from(t, lo).store_slice(r);
                    f32x8::simd_from(t, hi).store_slice(i);
                }) * 8
            }
        );

        pub(crate) fn sse4_2(_t: Sse4_2, _i: &[f32], _re: &mut [f32], _im: &mut [f32]) -> usize {
            0
        }

        pub(crate) fn sse2(_t: Sse2, _i: &[f32], _re: &mut [f32], _im: &mut [f32]) -> usize {
            0
        }
    }

    pub(super) mod int_i16 {
        #[allow(
            clippy::wildcard_imports,
            reason = "the intrinsics and vector types of the parent"
        )]
        use super::*;

        fearless_simd::kernel!(
            pub(crate) fn avx2(t: Avx2, re: &[i16], im: &[i16], out: &mut [i16]) -> usize {
                blocks3!(re, 16, im, 16, out, 32, |r, i, o| {
                    let r = i16x16::from_slice(t, r).into();
                    let i = i16x16::from_slice(t, i).into();
                    let r =
                        _mm256_permute4x64_pd::<0xD8>(core::arch::x86_64::_mm256_castsi256_pd(r));
                    let i =
                        _mm256_permute4x64_pd::<0xD8>(core::arch::x86_64::_mm256_castsi256_pd(i));
                    let (r, i) = (
                        core::arch::x86_64::_mm256_castpd_si256(r),
                        core::arch::x86_64::_mm256_castpd_si256(i),
                    );
                    let (o0, o1) = o.split_at_mut(16);
                    i16x16::simd_from(t, _mm256_unpacklo_epi16(r, i)).store_slice(o0);
                    i16x16::simd_from(t, _mm256_unpackhi_epi16(r, i)).store_slice(o1);
                }) * 16
            }
        );

        pub(crate) fn sse2(_t: Sse2, _re: &[i16], _im: &[i16], _out: &mut [i16]) -> usize {
            0
        }
    }

    pub(super) mod deint_i16 {
        #[allow(
            clippy::wildcard_imports,
            reason = "the intrinsics and vector types of the parent"
        )]
        use super::*;

        fearless_simd::kernel!(
            pub(crate) fn avx2(t: Avx2, input: &[i16], re: &mut [i16], im: &mut [i16]) -> usize {
                let split = _mm256_setr_epi8(
                    0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 0, 1, 4, 5, 8, 9, 12, 13,
                    2, 3, 6, 7, 10, 11, 14, 15,
                );
                blocks3m!(input, 32, re, 16, im, 16, |c, r, i| {
                    let (a, b) = c.split_at(16);
                    // Per lane: four real parts, then four imaginary parts.
                    let a = _mm256_shuffle_epi8(i16x16::from_slice(t, a).into(), split);
                    let b = _mm256_shuffle_epi8(i16x16::from_slice(t, b).into(), split);
                    let lo = core::arch::x86_64::_mm256_castsi256_pd(_mm256_unpacklo_epi64(a, b));
                    let hi = core::arch::x86_64::_mm256_castsi256_pd(_mm256_unpackhi_epi64(a, b));
                    let lo =
                        core::arch::x86_64::_mm256_castpd_si256(_mm256_permute4x64_pd::<0xD8>(lo));
                    let hi =
                        core::arch::x86_64::_mm256_castpd_si256(_mm256_permute4x64_pd::<0xD8>(hi));
                    i16x16::simd_from(t, lo).store_slice(r);
                    i16x16::simd_from(t, hi).store_slice(i);
                }) * 16
            }
        );

        fearless_simd::kernel!(
            pub(crate) fn sse4_2(
                t: Sse4_2,
                input: &[i16],
                re: &mut [i16],
                im: &mut [i16],
            ) -> usize {
                let split = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
                blocks3m!(input, 16, re, 8, im, 8, |c, r, i| {
                    let (a, b) = c.split_at(8);
                    let a = _mm_shuffle_epi8(i16x8::from_slice(t, a).into(), split);
                    let b = _mm_shuffle_epi8(i16x8::from_slice(t, b).into(), split);
                    i16x8::simd_from(t, _mm_unpacklo_epi64(a, b)).store_slice(r);
                    i16x8::simd_from(t, _mm_unpackhi_epi64(a, b)).store_slice(i);
                }) * 8
            }
        );

        fearless_simd::kernel!(
            pub(crate) fn sse2(t: Sse2, input: &[i16], re: &mut [i16], im: &mut [i16]) -> usize {
                blocks3m!(input, 16, re, 8, im, 8, |c, r, i| {
                    let (a, b) = c.split_at(8);
                    let a: __m128i = i16x8::from_slice(t, a).into();
                    let b: __m128i = i16x8::from_slice(t, b).into();
                    // Three rounds of unpacks sort the even and the odd values.
                    let (t0, t1) = (_mm_unpacklo_epi16(a, b), _mm_unpackhi_epi16(a, b));
                    let (u0, u1) = (_mm_unpacklo_epi16(t0, t1), _mm_unpackhi_epi16(t0, t1));
                    i16x8::simd_from(t, _mm_unpacklo_epi16(u0, u1)).store_slice(r);
                    i16x8::simd_from(t, _mm_unpackhi_epi16(u0, u1)).store_slice(i);
                }) * 8
            }
        );
    }
}

/// The NEON paths of `interleave` and `deinterleave`: `ld2` and `st2` deinterleave and
/// interleave while loading and storing.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod split_neon {
    use core::arch::aarch64::{
        float32x4x2_t, int16x8x2_t, vld1q_f32, vld1q_s16, vld2q_f32, vld2q_s16, vst1q_f32,
        vst1q_s16, vst2q_f32, vst2q_s16,
    };

    fearless_simd::kernel!(
        pub(super) fn int_f32(_t: Neon, re: &[f32], im: &[f32], out: &mut [f32]) -> usize {
            blocks3!(re, 4, im, 4, out, 8, |r, i, o| {
                // SAFETY: `r` and `i` are the 4 values that `vld1q_f32` reads and `o` the 8 that
                // `vst2q_f32` writes.
                unsafe {
                    vst2q_f32(
                        o.as_mut_ptr(),
                        float32x4x2_t(vld1q_f32(r.as_ptr()), vld1q_f32(i.as_ptr())),
                    );
                }
            }) * 4
        }
    );

    fearless_simd::kernel!(
        pub(super) fn deint_f32(_t: Neon, input: &[f32], re: &mut [f32], im: &mut [f32]) -> usize {
            blocks3m!(input, 8, re, 4, im, 4, |c, r, i| {
                // SAFETY: `c` is the 8 values that `vld2q_f32` reads, `r` and `i` the 4 that
                // `vst1q_f32` writes.
                unsafe {
                    let v = vld2q_f32(c.as_ptr());
                    vst1q_f32(r.as_mut_ptr(), v.0);
                    vst1q_f32(i.as_mut_ptr(), v.1);
                }
            }) * 4
        }
    );

    fearless_simd::kernel!(
        pub(super) fn int_i16(_t: Neon, re: &[i16], im: &[i16], out: &mut [i16]) -> usize {
            blocks3!(re, 8, im, 8, out, 16, |r, i, o| {
                // SAFETY: `r` and `i` are the 8 values that `vld1q_s16` reads and `o` the 16 that
                // `vst2q_s16` writes.
                unsafe {
                    vst2q_s16(
                        o.as_mut_ptr(),
                        int16x8x2_t(vld1q_s16(r.as_ptr()), vld1q_s16(i.as_ptr())),
                    );
                }
            }) * 8
        }
    );

    fearless_simd::kernel!(
        pub(super) fn deint_i16(_t: Neon, input: &[i16], re: &mut [i16], im: &mut [i16]) -> usize {
            blocks3m!(input, 16, re, 8, im, 8, |c, r, i| {
                // SAFETY: `c` is the 16 values that `vld2q_s16` reads, `r` and `i` the 8 that
                // `vst1q_s16` writes.
                unsafe {
                    let v = vld2q_s16(c.as_ptr());
                    vst1q_s16(r.as_mut_ptr(), v.0);
                    vst1q_s16(i.as_mut_ptr(), v.1);
                }
            }) * 8
        }
    );
}

/// The level-specific path of `interleave` with both inputs, for the current level: returns the
/// number of leading samples done.
macro_rules! level_paths {
    ($int:ident, $deint:ident, $elem:ty, $x86_int:ident, $x86_deint:ident, $neon_int:ident, $neon_deint:ident) => {
        #[inline(always)]
        fn $int<S: SimdToken>(simd: S, re: &[$elem], im: &[$elem], out: &mut [$elem]) -> usize {
            let level = simd.level();
            #[cfg(target_arch = "x86_64")]
            {
                if let Some(t) = level.as_avx2() {
                    return split_x86::$x86_int::avx2(t, re, im, out);
                }
                if let Some(t) = level.as_sse2() {
                    return split_x86::$x86_int::sse2(t, re, im, out);
                }
            }
            #[cfg(target_arch = "aarch64")]
            if let Some(t) = level.as_neon() {
                return split_neon::$neon_int(t, re, im, out);
            }
            let _ = (level, re, im, out);
            0
        }

        #[inline(always)]
        fn $deint<S: SimdToken>(
            simd: S,
            input: &[$elem],
            re: &mut [$elem],
            im: &mut [$elem],
        ) -> usize {
            let level = simd.level();
            #[cfg(target_arch = "x86_64")]
            {
                if let Some(t) = level.as_avx2() {
                    return split_x86::$x86_deint::avx2(t, input, re, im);
                }
                if let Some(t) = level.as_sse4_2() {
                    return split_x86::$x86_deint::sse4_2(t, input, re, im);
                }
                if let Some(t) = level.as_sse2() {
                    return split_x86::$x86_deint::sse2(t, input, re, im);
                }
            }
            #[cfg(target_arch = "aarch64")]
            if let Some(t) = level.as_neon() {
                return split_neon::$neon_deint(t, input, re, im);
            }
            let _ = (level, input, re, im);
            0
        }
    };
}

level_paths!(
    int_f32_level,
    deint_f32_level,
    f32,
    int_f32,
    deint_f32,
    int_f32,
    deint_f32
);
level_paths!(
    int_i16_level,
    deint_i16_level,
    i16,
    int_i16,
    deint_i16,
    int_i16,
    deint_i16
);

/// `out = (re[n], im[n])` of whole vectors of samples; `done` is the number of samples done.
/// `im = None` gives zeros.
macro_rules! interleave_kernel {
    ($name:ident, $vec:ident, $elem:ty, $zero:expr, $level:ident) => {
        #[inline(always)]
        fn $name<S: SimdToken>(
            simd: S,
            re: &[$elem],
            im: Option<&[$elem]>,
            out: &mut [$elem],
            done: &mut usize,
        ) {
            let l = S::$vec::LEN;
            let blocks = re.len().min(out.len() / 2) / l;
            match im {
                Some(im) => {
                    let n = $level(simd, re, im, out);
                    if n > 0 {
                        *done = n;
                        return;
                    }
                    for ((r, i), o) in re
                        .chunks_exact(l)
                        .zip(im.chunks_exact(l))
                        .zip(out.chunks_exact_mut(2 * l))
                    {
                        let (r, i) = (S::$vec::from_slice(simd, r), S::$vec::from_slice(simd, i));
                        let (o0, o1) = o.split_at_mut(l);
                        r.zip_low(i).store_slice(o0);
                        r.zip_high(i).store_slice(o1);
                    }
                }
                None => {
                    let z = S::$vec::splat(simd, $zero);
                    for (r, o) in re.chunks_exact(l).zip(out.chunks_exact_mut(2 * l)) {
                        let r = S::$vec::from_slice(simd, r);
                        let (o0, o1) = o.split_at_mut(l);
                        r.zip_low(z).store_slice(o0);
                        r.zip_high(z).store_slice(o1);
                    }
                }
            }
            *done = blocks * l;
        }
    };
}

interleave_kernel!(interleave_f32, f32s, f32, 0.0, int_f32_level);
interleave_kernel!(interleave_i16, i16s, i16, 0, int_i16_level);

dispatch_fn!(fn run_interleave_f32(re: &[f32], im: Option<&[f32]>, out: &mut [f32], done: &mut usize) => interleave_f32);
dispatch_fn!(fn run_interleave_i16(re: &[i16], im: Option<&[i16]>, out: &mut [i16], done: &mut usize) => interleave_i16);

/// The real and imaginary parts of whole vectors of samples; `done` is the number of samples
/// done.
macro_rules! deinterleave_kernel {
    ($name:ident, $vec:ident, $elem:ty, $level:ident) => {
        #[inline(always)]
        fn $name<S: SimdToken>(
            simd: S,
            input: &[$elem],
            re: Option<&mut [$elem]>,
            im: Option<&mut [$elem]>,
            done: &mut usize,
        ) {
            let l = S::$vec::LEN;
            let n = input.len() / 2;
            let n = match (&re, &im) {
                (Some(r), Some(i)) => n.min(r.len()).min(i.len()),
                (Some(r), None) => n.min(r.len()),
                (None, Some(i)) => n.min(i.len()),
                (None, None) => 0,
            };
            let blocks = n / l;
            let load = |c: &[$elem]| {
                let (c0, c1) = c.split_at(l);
                (S::$vec::from_slice(simd, c0), S::$vec::from_slice(simd, c1))
            };
            match (re, im) {
                (Some(re), Some(im)) => {
                    let k = $level(simd, input, re, im);
                    if k > 0 {
                        *done = k;
                        return;
                    }
                    for ((c, r), i) in input
                        .chunks_exact(2 * l)
                        .zip(re.chunks_exact_mut(l))
                        .zip(im.chunks_exact_mut(l))
                    {
                        let (a, b) = load(c);
                        a.unzip_low(b).store_slice(r);
                        a.unzip_high(b).store_slice(i);
                    }
                }
                (Some(re), None) => {
                    for (c, r) in input.chunks_exact(2 * l).zip(re.chunks_exact_mut(l)) {
                        let (a, b) = load(c);
                        a.unzip_low(b).store_slice(r);
                    }
                }
                (None, Some(im)) => {
                    for (c, i) in input.chunks_exact(2 * l).zip(im.chunks_exact_mut(l)) {
                        let (a, b) = load(c);
                        a.unzip_high(b).store_slice(i);
                    }
                }
                (None, None) => {}
            }
            *done = blocks * l;
        }
    };
}

deinterleave_kernel!(deinterleave_f32, f32s, f32, deint_f32_level);
deinterleave_kernel!(deinterleave_i16, i16s, i16, deint_i16_level);

dispatch_fn!(fn run_deinterleave_f32(input: &[f32], re: Option<&mut [f32]>, im: Option<&mut [f32]>, done: &mut usize) => deinterleave_f32);
dispatch_fn!(fn run_deinterleave_i16(input: &[i16], re: Option<&mut [i16]>, im: Option<&mut [i16]>, done: &mut usize) => deinterleave_i16);

// The kernels

/// The samples from `done` on of an element-wise call, by the fallback.
macro_rules! finish3 {
    ($tr:ident :: $m:ident, $ty:ty, $a:expr, $b:expr, $out:expr, $done:expr) => {{
        let d = $done;
        match ($a.get(d..), $b.get(d..), $out.get_mut(d..)) {
            (Some(a), Some(b), Some(o)) if !a.is_empty() => {
                <Fallback as $tr<$ty>>::$m(&Fallback, a, b, o)
            }
            _ => Ok(()),
        }
    }};
}

impl Add<C32> for Simd {
    fn add(&self, a: &[C32], b: &[C32], out: &mut [C32]) -> Result<()> {
        check3(a.len(), b.len(), out.len())?;
        run_add_f32(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            bytemuck::cast_slice_mut(out),
        );
        Ok(())
    }
}

impl Add<CQ15> for Simd {
    fn add(&self, a: &[CQ15], b: &[CQ15], out: &mut [CQ15]) -> Result<()> {
        check3(a.len(), b.len(), out.len())?;
        run_add_i16(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            bytemuck::cast_slice_mut(out),
        );
        Ok(())
    }
}

impl AddReal<f32> for Simd {
    fn add_real(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        check3(a.len(), b.len(), out.len())?;
        run_add_f32(a, b, out);
        Ok(())
    }
}

impl AddReal<Q15> for Simd {
    fn add_real(&self, a: &[Q15], b: &[Q15], out: &mut [Q15]) -> Result<()> {
        check3(a.len(), b.len(), out.len())?;
        run_add_i16(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            bytemuck::cast_slice_mut(out),
        );
        Ok(())
    }
}

impl Mul<C32> for Simd {
    fn mul(&self, a: &[C32], b: &[C32], out: &mut [C32]) -> Result<()> {
        check3(a.len(), b.len(), out.len())?;
        run_mul_c32(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            bytemuck::cast_slice_mut(out),
        );
        Ok(())
    }
}

impl Mul<CQ15> for Simd {
    fn mul(&self, a: &[CQ15], b: &[CQ15], out: &mut [CQ15]) -> Result<()> {
        check3(a.len(), b.len(), out.len())?;
        let mut done = 0;
        run_mul_cq15(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            bytemuck::cast_slice_mut(out),
            &mut done,
        );
        finish3!(Mul::mul, CQ15, a, b, out, done / 2)
    }
}

impl MulReal<f32> for Simd {
    fn mul_real(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        check3(a.len(), b.len(), out.len())?;
        run_mul_f32(a, b, out);
        Ok(())
    }
}

impl MulReal<Q15> for Simd {
    fn mul_real(&self, a: &[Q15], b: &[Q15], out: &mut [Q15]) -> Result<()> {
        check3(a.len(), b.len(), out.len())?;
        let mut done = 0;
        run_mul_real_q15(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            bytemuck::cast_slice_mut(out),
            &mut done,
        );
        finish3!(MulReal::mul_real, Q15, a, b, out, done)
    }
}

impl Interleave<C32> for Simd {
    fn interleave(&self, re: &[f32], im: Option<&[f32]>, out: &mut [C32]) -> Result<()> {
        check_len(re.len(), out.len())?;
        if let Some(im) = im {
            check_len(re.len(), im.len())?;
        }
        let mut done = 0;
        run_interleave_f32(re, im, bytemuck::cast_slice_mut(out), &mut done);
        <Fallback as Interleave<C32>>::interleave(
            &Fallback,
            re.get(done..).unwrap_or_default(),
            im.map(|im| im.get(done..).unwrap_or_default()),
            out.get_mut(done..).unwrap_or_default(),
        )
    }

    fn deinterleave(
        &self,
        input: &[C32],
        re: Option<&mut [f32]>,
        im: Option<&mut [f32]>,
    ) -> Result<()> {
        if let Some(re) = &re {
            check_len(input.len(), re.len())?;
        }
        if let Some(im) = &im {
            check_len(input.len(), im.len())?;
        }
        let mut done = 0;
        let (mut re, mut im) = (re, im);
        run_deinterleave_f32(
            bytemuck::cast_slice(input),
            re.as_deref_mut(),
            im.as_deref_mut(),
            &mut done,
        );
        <Fallback as Interleave<C32>>::deinterleave(
            &Fallback,
            input.get(done..).unwrap_or_default(),
            re.and_then(|r| r.get_mut(done..)),
            im.and_then(|i| i.get_mut(done..)),
        )
    }
}

impl Interleave<CQ15> for Simd {
    fn interleave(&self, re: &[Q15], im: Option<&[Q15]>, out: &mut [CQ15]) -> Result<()> {
        check_len(re.len(), out.len())?;
        if let Some(im) = im {
            check_len(re.len(), im.len())?;
        }
        let mut done = 0;
        run_interleave_i16(
            bytemuck::cast_slice(re),
            im.map(bytemuck::cast_slice),
            bytemuck::cast_slice_mut(out),
            &mut done,
        );
        <Fallback as Interleave<CQ15>>::interleave(
            &Fallback,
            re.get(done..).unwrap_or_default(),
            im.map(|im| im.get(done..).unwrap_or_default()),
            out.get_mut(done..).unwrap_or_default(),
        )
    }

    fn deinterleave(
        &self,
        input: &[CQ15],
        re: Option<&mut [Q15]>,
        im: Option<&mut [Q15]>,
    ) -> Result<()> {
        if let Some(re) = &re {
            check_len(input.len(), re.len())?;
        }
        if let Some(im) = &im {
            check_len(input.len(), im.len())?;
        }
        let mut done = 0;
        let (mut re, mut im) = (re, im);
        run_deinterleave_i16(
            bytemuck::cast_slice(input),
            re.as_deref_mut().map(bytemuck::cast_slice_mut),
            im.as_deref_mut().map(bytemuck::cast_slice_mut),
            &mut done,
        );
        <Fallback as Interleave<CQ15>>::deinterleave(
            &Fallback,
            input.get(done..).unwrap_or_default(),
            re.and_then(|r| r.get_mut(done..)),
            im.and_then(|i| i.get_mut(done..)),
        )
    }
}

// oscillate

/// `(cos, sin)` of the phases `phase + n step` for whole vectors of samples (the lookups of
/// `table.rs`, bit-exact with the fallback); `done` is the number of samples done.
#[inline(always)]
fn osc_f32<S: SimdToken>(simd: S, phase: u32, step: u32, out: &mut [f32], done: &mut usize) {
    let h = S::u32s::LEN;
    let ramp = S::u32s::from_fn(simd, |i| (i as u32).wrapping_mul(step));
    let block = (h as u32).wrapping_mul(step);
    let mut base = phase;
    let mut n = 0;
    for y in out.chunks_exact_mut(2 * h) {
        let p = S::u32s::splat(simd, base) + ramp;
        let (o0, o1) = cos_sin_f32(simd, p);
        let (d0, d1) = y.split_at_mut(h);
        o0.store_slice(d0);
        o1.store_slice(d1);
        base = base.wrapping_add(block);
        n += h;
    }
    *done = n;
}

/// The `CQ15` form of [`osc_f32`].
#[inline(always)]
fn osc_q15<S: SimdToken>(simd: S, phase: u32, step: u32, out: &mut [i16], done: &mut usize) {
    let h = S::u32s::LEN;
    let ramp = S::u32s::from_fn(simd, |i| (i as u32).wrapping_mul(step));
    let block = (h as u32).wrapping_mul(step);
    let mut base = phase;
    let mut n = 0;
    for y in out.chunks_exact_mut(2 * h) {
        let p = S::u32s::splat(simd, base) + ramp;
        #[cfg(feature = "nco-tables")]
        let packed: S::i32s = crate::table::cos_sin_q15_packed(simd, p).bitcast();
        #[cfg(not(feature = "nco-tables"))]
        let packed: S::i32s = {
            let (c, s) = crate::table::cos_sin_q15(simd, p);
            (c & 0xFFFFi32) | (s << 16u32)
        };
        packed.bitcast::<S::i16s>().store_slice(y);
        base = base.wrapping_add(block);
        n += h;
    }
    *done = n;
}

dispatch_fn!(fn run_osc_f32(phase: u32, step: u32, out: &mut [f32], done: &mut usize) => osc_f32_entry);
dispatch_fn!(fn run_osc_q15(phase: u32, step: u32, out: &mut [i16], done: &mut usize) => osc_q15_entry);

#[inline(always)]
fn osc_f32_entry<S: SimdToken>(simd: S, phase: u32, step: u32, out: &mut [f32], done: &mut usize) {
    osc_f32(simd, phase, step, out, done);
}

#[inline(always)]
fn osc_q15_entry<S: SimdToken>(simd: S, phase: u32, step: u32, out: &mut [i16], done: &mut usize) {
    osc_q15(simd, phase, step, out, done);
}

/// The oscillator state after `done` samples, for the fallback to finish the call.
fn finish_oscillate<T>(
    nco: &mut Nco,
    harmonic: u32,
    out: &mut [T],
    done: usize,
    rest: impl FnOnce(&mut Nco, &mut [T]) -> Result<()>,
) -> Result<()>
where
    T: Copy,
{
    let inc = nco.increment();
    let mut tail = Nco::from_increment(inc);
    tail.set_phase(nco.phase().wrapping_add((done as u32).wrapping_mul(inc)));
    let _ = harmonic;
    let len = out.len();
    if let Some(o) = out.get_mut(done..) {
        rest(&mut tail, o)?;
    }
    nco.set_phase(nco.phase().wrapping_add((len as u32).wrapping_mul(inc)));
    Ok(())
}

impl Oscillate<C32> for Simd {
    fn oscillate(&self, nco: &mut Nco, harmonic: u32, out: &mut [C32]) -> Result<()> {
        if !(1..=8).contains(&harmonic) {
            return Fallback.oscillate(nco, harmonic, out);
        }
        let phase = nco.phase().wrapping_mul(harmonic);
        let step = nco.increment().wrapping_mul(harmonic);
        let mut done = 0;
        run_osc_f32(phase, step, bytemuck::cast_slice_mut(out), &mut done);
        finish_oscillate(nco, harmonic, out, done, |tail, o| {
            Fallback.oscillate(tail, harmonic, o)
        })
    }
}

impl Oscillate<CQ15> for Simd {
    fn oscillate(&self, nco: &mut Nco, harmonic: u32, out: &mut [CQ15]) -> Result<()> {
        if !(1..=8).contains(&harmonic) {
            return Fallback.oscillate(nco, harmonic, out);
        }
        let phase = nco.phase().wrapping_mul(harmonic);
        let step = nco.increment().wrapping_mul(harmonic);
        let mut done = 0;
        run_osc_q15(phase, step, bytemuck::cast_slice_mut(out), &mut done);
        finish_oscillate(nco, harmonic, out, done, |tail, o| {
            Fallback.oscillate(tail, harmonic, o)
        })
    }
}
