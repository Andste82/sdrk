//! `dot` and `dot_real`: the dot product of a window of samples with real taps.
//!
//! - `f32` and `C32`: generic code with [`ACC`] vector accumulators, so that the additions of
//!   one vector do not wait for the previous ones (the fallback sums in the order of the
//!   contract, one dependency chain per component). The levels with fused multiply-add (AVX2,
//!   AVX-512, NEON) use it (decision D5). `C32` multiplies a vector of interleaved samples by
//!   the taps duplicated for the real and the imaginary lane (`zip_low`/`zip_high` of the taps
//!   with themselves). The remainder after the whole vectors is summed in scalar code. The
//!   result is within the contract of `docs/kernels.md` (120 dB) but not bit-identical with the
//!   fallback: the sums are formed in another order (and fused on the levels with FMA), which
//!   makes them more accurate, not less (the error grows with the length of each chain).
//! - `Q15` and `CQ15`: exact integer sums, bit-identical with the fallback. The level-specific
//!   paths (`x86`, `neon`, `wasm` below) multiply with the instructions that multiply 16-bit
//!   lanes and add adjacent products into 32-bit lanes (`pmaddwd`, `i32x4.dot_i16x8_s`) or
//!   widen the products (`smull`); the levels without a path run the fallback, which the
//!   compiler vectorizes.

use fearless_simd::{Simd as SimdToken, f32x4, f32x8, f32x16, prelude::*};
use sdrk_core::kernels::dot::{Dot, DotReal};
use sdrk_core::num::Acc64;
use sdrk_core::{C32, CQ15, Fallback, Q15, Result};

use crate::Simd;
use crate::level::dispatch_fn;
use crate::util::check_len;

/// Vector accumulators of the `f32` and `C32` sums.
const ACC: usize = 8;

/// Whether the level has fused multiply-add for the float sums (decision D5: AVX2, AVX-512
/// and NEON).
#[inline(always)]
fn has_fma<S: SimdToken>(simd: S) -> bool {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        level.as_avx2().is_some()
    }
    #[cfg(target_arch = "aarch64")]
    {
        level.as_neon().is_some()
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = level;
        false
    }
}

/// The sum of the accumulators as a balanced tree (`A` a power of two).
#[inline(always)]
fn tree<S: SimdToken, V: SimdFloat<S, Element = f32>, const A: usize>(acc: [V; A], zero: V) -> V {
    let mut acc = acc;
    let mut n = A;
    while n > 1 {
        n /= 2;
        let (lo, hi) = acc.split_at_mut(n);
        for (a, b) in lo.iter_mut().zip(hi.iter()) {
            *a += *b;
        }
    }
    acc.first().copied().unwrap_or(zero)
}

/// `acc + a * b`, fused where `FMA`.
#[inline(always)]
fn madd<S: SimdToken, V: SimdFloat<S, Element = f32>, const FMA: bool>(a: V, b: V, acc: V) -> V {
    if FMA { a.mul_add(b, acc) } else { acc + a * b }
}

/// `sum x[i] * taps[i]` with [`ACC`] accumulators of the vector type `V`.
#[inline(always)]
fn dot_f32_v<S: SimdToken, V: SimdFloat<S, Element = f32>, const FMA: bool, const ACC: usize>(
    simd: S,
    x: &[f32],
    taps: &[f32],
) -> f32 {
    let l = V::LEN;
    let zero = V::splat(simd, 0.0);
    let mut acc = [zero; ACC];
    let blocks = x.len() / (ACC * l);
    let (xb, xr) = x.split_at(blocks * ACC * l);
    let (tb, tr) = taps.split_at(blocks * ACC * l);
    for (xs, ts) in xb.chunks_exact(ACC * l).zip(tb.chunks_exact(ACC * l)) {
        for (a, (xv, tv)) in acc
            .iter_mut()
            .zip(xs.chunks_exact(l).zip(ts.chunks_exact(l)))
        {
            *a = madd::<S, V, FMA>(V::from_slice(simd, xv), V::from_slice(simd, tv), *a);
        }
    }
    // The remaining whole vectors in one more accumulator (indexing `acc` by a run-time count
    // would keep the accumulators in memory), then the scalar rest.
    let whole = xr.len() / l;
    let (xv, xs) = xr.split_at(whole * l);
    let (tv, ts) = tr.split_at(whole * l);
    let mut rest = zero;
    for (xv, tv) in xv.chunks_exact(l).zip(tv.chunks_exact(l)) {
        rest = madd::<S, V, FMA>(V::from_slice(simd, xv), V::from_slice(simd, tv), rest);
    }
    let mut sum = (tree(acc, zero) + rest).reduce_sum();
    for (&s, &h) in xs.iter().zip(ts) {
        sum += s * h;
    }
    sum
}

/// `sum x[i] * taps[i]` of interleaved complex samples `x` (`2 n` floats) with `n` real taps,
/// with [`ACC`] accumulators of the vector type `V`: one vector of taps covers two vectors of
/// samples.
#[inline(always)]
fn dot_c32_v<S: SimdToken, V: SimdFloat<S, Element = f32>, const FMA: bool, const ACC: usize>(
    simd: S,
    x: &[f32],
    taps: &[f32],
) -> (f32, f32) {
    let l = V::LEN;
    let zero = V::splat(simd, 0.0);
    let mut acc = [zero; ACC];
    let step = ACC / 2 * l;
    let blocks = taps.len() / step;
    let (tb, tr) = taps.split_at(blocks * step);
    let (xb, xr) = x.split_at(2 * blocks * step);
    for (ts, xs) in tb.chunks_exact(step).zip(xb.chunks_exact(2 * step)) {
        for (pair, (tv, xv)) in acc
            .chunks_exact_mut(2)
            .zip(ts.chunks_exact(l).zip(xs.chunks_exact(2 * l)))
        {
            let t = V::from_slice(simd, tv);
            let (x0, x1) = xv.split_at(l);
            if let [a0, a1] = pair {
                *a0 = madd::<S, V, FMA>(V::from_slice(simd, x0), t.zip_low(t), *a0);
                *a1 = madd::<S, V, FMA>(V::from_slice(simd, x1), t.zip_high(t), *a1);
            }
        }
    }
    // The remaining whole vectors of taps in two more accumulators (see `dot_f32_v`), then the
    // scalar rest.
    let whole = tr.len() / l;
    let (tv, ts) = tr.split_at(whole * l);
    let (xv, xs) = xr.split_at(2 * whole * l);
    let (mut r0, mut r1) = (zero, zero);
    for (tv, xv) in tv.chunks_exact(l).zip(xv.chunks_exact(2 * l)) {
        let t = V::from_slice(simd, tv);
        let (x0, x1) = xv.split_at(l);
        r0 = madd::<S, V, FMA>(V::from_slice(simd, x0), t.zip_low(t), r0);
        r1 = madd::<S, V, FMA>(V::from_slice(simd, x1), t.zip_high(t), r1);
    }
    let sum = tree(acc, zero) + (r0 + r1);
    let mut re = sum.unzip_low(zero).reduce_sum();
    let mut im = sum.unzip_high(zero).reduce_sum();
    for (s, &h) in xs.chunks_exact(2).zip(ts) {
        if let [sr, si] = *s {
            re += sr * h;
            im += si * h;
        }
    }
    (re, im)
}

/// `dot_real` `f32` at the native width of the level.
#[inline(always)]
fn dot_f32<S: SimdToken>(simd: S, x: &[f32], taps: &[f32], out: &mut f32) {
    let fma = has_fma(simd);
    *out = match (S::f32s::LEN, fma) {
        (16, _) => dot_f32_v::<S, f32x16<S>, true, ACC>(simd, x, taps),
        (8, _) => dot_f32_v::<S, f32x8<S>, true, ACC>(simd, x, taps),
        (_, true) => dot_f32_v::<S, f32x4<S>, true, ACC>(simd, x, taps),
        _ => dot_f32_v::<S, f32x4<S>, false, ACC>(simd, x, taps),
    };
}

/// `dot` `C32` at the native width of the level.
#[inline(always)]
fn dot_c32<S: SimdToken>(simd: S, x: &[f32], taps: &[f32], out: &mut (f32, f32)) {
    let fma = has_fma(simd);
    *out = match (S::f32s::LEN, fma) {
        (16, _) => dot_c32_v::<S, f32x16<S>, true, ACC>(simd, x, taps),
        (8, _) => dot_c32_v::<S, f32x8<S>, true, ACC>(simd, x, taps),
        (_, true) => dot_c32_v::<S, f32x4<S>, true, ACC>(simd, x, taps),
        _ => dot_c32_v::<S, f32x4<S>, false, ACC>(simd, x, taps),
    };
}

// The entries, one function per level (see `dispatch_fn!`): with the level detected at run time,
// `dispatch!` hands the kernel a closure over references, which costs the `Q15` sums about a
// tenth more instructions per sample at AVX2.
dispatch_fn!(fn run_c32(x: &[f32], taps: &[f32], out: &mut (f32, f32)) => dot_c32);
dispatch_fn!(fn run_f32(x: &[f32], taps: &[f32], out: &mut f32) => dot_f32);
dispatch_fn!(fn run_cq15(x: &[i16], taps: &[i16], out: &mut Option<Partial>) => dot_cq15);
dispatch_fn!(fn run_q15(x: &[i16], taps: &[i16], out: &mut Option<Partial>) => dot_q15);

impl Dot<C32> for Simd {
    fn dot(&self, x: &[C32], taps: &[f32]) -> Result<C32> {
        check_len(x.len(), taps.len())?;
        let xf: &[f32] = bytemuck::cast_slice(x);
        let mut out = (0.0, 0.0);
        run_c32(xf, taps, &mut out);
        Ok(C32::new(out.0, out.1))
    }
}

impl DotReal<f32> for Simd {
    fn dot_real(&self, x: &[f32], taps: &[f32]) -> Result<f32> {
        check_len(x.len(), taps.len())?;
        let mut out = 0.0;
        run_f32(x, taps, &mut out);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------
// Q15 and CQ15

/// The exact sums of a level-specific path: the samples done and their sums (`re`, `im`; the
/// real `dot_real` uses `re`).
#[derive(Clone, Copy, Default)]
struct Partial {
    done: usize,
    re: i64,
    im: i64,
}

/// The level-specific path of `dot` `CQ15` (`x`: `2 n` interleaved values, `taps`: `n`), if the
/// level has one.
#[inline(always)]
fn dot_cq15_level<S: SimdToken>(simd: S, x: &[i16], taps: &[i16]) -> Option<Partial> {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx512() {
            return Some(x86::dot_cq15_avx512(t, x, taps));
        }
        if let Some(t) = level.as_avx2() {
            return Some(x86::dot_cq15_avx2(t, x, taps));
        }
        if let Some(t) = level.as_sse4_2() {
            return Some(x86::dot_cq15_sse4_2(t, x, taps));
        }
        if let Some(t) = crate::level::sse2_only(level) {
            return Some(x86::dot_cq15_sse2(t, x, taps));
        }
    }
    // NEON runs the fallback: the compiler's code of it (`ld2`, `smlal` into 64-bit lanes) takes
    // 0.90 cycles per sample on both core models, the path with `uzp1`/`uzp2`, `smull` and
    // `sadalp` 0.95 (`cortex-a72`) and 1.04 (`cortex-a76`).
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return Some(wasm::dot_cq15(t, x, taps));
    }
    let _ = (level, x, taps);
    None
}

/// The level-specific path of `dot_real` `Q15`, if the level has one.
#[inline(always)]
fn dot_q15_level<S: SimdToken>(simd: S, x: &[i16], taps: &[i16]) -> Option<Partial> {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx2() {
            return Some(x86::dot_q15_avx2(t, x, taps));
        }
        if let Some(t) = level.as_sse4_2() {
            return Some(x86::dot_q15_sse4_2(t, x, taps));
        }
        if let Some(t) = crate::level::sse2_only(level) {
            return Some(x86::dot_q15_sse2(t, x, taps));
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return Some(neon::dot_q15(t, x, taps));
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return Some(wasm::dot_q15(t, x, taps));
    }
    let _ = (level, x, taps);
    None
}

#[inline(always)]
fn dot_cq15<S: SimdToken>(simd: S, x: &[i16], taps: &[i16], out: &mut Option<Partial>) {
    *out = dot_cq15_level(simd, x, taps);
}

#[inline(always)]
fn dot_q15<S: SimdToken>(simd: S, x: &[i16], taps: &[i16], out: &mut Option<Partial>) {
    *out = dot_q15_level(simd, x, taps);
}

/// The samples of a level path: at most this many per call of the path, so that its `i64`
/// sums cannot overflow (`2^30` per product); a longer input is split into such runs, whose
/// sums are added with the saturation of `Acc64`, as the fallback does.
const RUN: usize = 1 << 30;

impl Dot<CQ15> for Simd {
    fn dot(&self, x: &[CQ15], taps: &[Q15]) -> Result<CQ15> {
        check_len(x.len(), taps.len())?;
        let (mut re, mut im) = (Acc64::default(), Acc64::default());
        for (xs, ts) in x.chunks(RUN).zip(taps.chunks(RUN)) {
            let xi: &[i16] = bytemuck::cast_slice(xs);
            let ti: &[i16] = bytemuck::cast_slice(ts);
            let mut part = None;
            run_cq15(xi, ti, &mut part);
            let Some(p) = part else {
                return Fallback.dot(x, taps);
            };
            let (mut r, mut i) = (p.re, p.im);
            for (s, h) in xs.iter().zip(ts).skip(p.done) {
                r += i64::from(i32::from(s.re.0) * i32::from(h.0));
                i += i64::from(i32::from(s.im.0) * i32::from(h.0));
            }
            re.0 = re.0.saturating_add(r);
            im.0 = im.0.saturating_add(i);
        }
        Ok(CQ15 {
            re: re.finish(),
            im: im.finish(),
        })
    }
}

impl DotReal<Q15> for Simd {
    fn dot_real(&self, x: &[Q15], taps: &[Q15]) -> Result<Q15> {
        check_len(x.len(), taps.len())?;
        let mut acc = Acc64::default();
        for (xs, ts) in x.chunks(RUN).zip(taps.chunks(RUN)) {
            let xi: &[i16] = bytemuck::cast_slice(xs);
            let ti: &[i16] = bytemuck::cast_slice(ts);
            let mut part = None;
            run_q15(xi, ti, &mut part);
            let Some(p) = part else {
                return Fallback.dot_real(x, taps);
            };
            let mut r = p.re;
            for (s, h) in xs.iter().zip(ts).skip(p.done) {
                r += i64::from(i32::from(s.0) * i32::from(h.0));
            }
            acc.0 = acc.0.saturating_add(r);
        }
        Ok(acc.finish())
    }
}

/// Vectors (of 16-bit lanes) between two flushes of the 32-bit sums of the split products
/// (see `x86`): a lane of a sum gains at most `2^24` per vector, so 64 vectors stay below
/// `2^30`.
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
const FLUSH: usize = 64;

/// `$body` with `$xv: &[i16; $xs]` and `$tv: &[i16; $ts]`, the corresponding blocks of `$x` and
/// `$t`: in groups of four (unrolled) first, then one at a time. A macro, not a function with a
/// closure: the body must be compiled inside the `#[target_feature]` function of the path.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! pairs {
    ($x:expr, $t:expr, $xs:literal, $ts:literal, |$xv:ident, $tv:ident| $body:block) => {{
        // One closure for both loops: two expansions of the body would exceed the stack frame
        // the lint allows in unoptimized builds. It inherits the target features and is inlined.
        let mut body = |$xv: &[i16; $xs], $tv: &[i16; $ts]| $body;
        let groups = ($x.len() / (4 * $xs)).min($t.len() / (4 * $ts));
        let (xg, xr) = $x.split_at(groups * 4 * $xs);
        let (tg, tr) = $t.split_at(groups * 4 * $ts);
        let (xg, _) = xg.as_chunks::<{ 4 * $xs }>();
        let (tg, _) = tg.as_chunks::<{ 4 * $ts }>();
        for (xq, tq) in xg.iter().zip(tg) {
            let (xq, _) = xq.as_chunks::<$xs>();
            let (tq, _) = tq.as_chunks::<$ts>();
            for (xv, tv) in xq.iter().zip(tq) {
                body(xv, tv);
            }
        }
        let (xr, _) = xr.as_chunks::<$xs>();
        let (tr, _) = tr.as_chunks::<$ts>();
        for (xv, tv) in xr.iter().zip(tr) {
            body(xv, tv);
        }
    }};
}

/// The sum of the 32-bit lanes of `v` at the even positions and at the odd positions.
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn lane_sums(v: &[i32]) -> (i64, i64) {
    let (mut even, mut odd) = (0i64, 0i64);
    for pair in v.chunks_exact(2) {
        if let [a, b] = *pair {
            even += i64::from(a);
            odd += i64::from(b);
        }
    }
    (even, odd)
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    //! `pmaddwd` multiplies 16-bit lanes and adds adjacent products into 32-bit lanes. A pair of
    //! full-range products does not fit (`2 * 2^30`), so every sample is split into its high
    //! byte (signed, `x >> 8`) and its low byte (unsigned, `x & 0xff`), `x = 256 h + l`: a pair
    //! of products with a byte is below `2^24` in magnitude, so [`FLUSH`] vectors of them sum in
    //! 32-bit lanes without overflow before the lanes are added to the `i64` sums as `256 h + l`.
    //!
    //! `CQ15` pairs two consecutive samples of the same component: the samples are reordered
    //! within each 64-bit group (`re0 re1 im0 im1`) and the taps are duplicated per pair of
    //! taps (`h0 h1 h0 h1`), so the 32-bit lanes alternate between the real and the imaginary
    //! sums.
    //!
    //! The constants pass through `core::hint::black_box`: in the build with run-time detection
    //! (generic tuning) the compiler otherwise splits the byte shuffle into two word shuffles and
    //! the lane permutation into two, and reloads the mask in the loop (`dot` `CQ15` at AVX2:
    //! 1.77 instead of 1.65 instructions per sample).

    use core::arch::x86_64::{
        __m128i, __m256i, __m512i, _mm_add_epi32, _mm_and_si128, _mm_madd_epi16, _mm_set1_epi16,
        _mm_setr_epi8, _mm_setzero_si128, _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_shufflehi_epi16,
        _mm_shufflelo_epi16, _mm_srai_epi16, _mm256_add_epi32, _mm256_and_si256,
        _mm256_castsi128_si256, _mm256_madd_epi16, _mm256_permutevar8x32_epi32, _mm256_set1_epi16,
        _mm256_setr_epi8, _mm256_setr_epi32, _mm256_setzero_si256, _mm256_shuffle_epi8,
        _mm256_srai_epi16, _mm512_add_epi32, _mm512_and_si512, _mm512_broadcast_i32x4,
        _mm512_castsi256_si512, _mm512_madd_epi16, _mm512_permutexvar_epi32, _mm512_set1_epi16,
        _mm512_setr_epi32, _mm512_setzero_si512, _mm512_shuffle_epi8, _mm512_srai_epi16,
    };
    use core::hint::black_box;
    use fearless_simd::prelude::*;
    use fearless_simd::{i16x8, i16x16, i16x32, i32x4, i32x8, i32x16};

    use super::{FLUSH, Partial, lane_sums};

    /// Adds the lanes of the high-byte sums `hi` and the low-byte sums `lo` to `p`: the even
    /// lanes to `re`, the odd lanes to `im` (for `dot_real` both belong to `re`).
    #[inline(always)]
    fn add_lanes(p: &mut Partial, hi: &[i32], lo: &[i32]) {
        let (he, ho) = lane_sums(hi);
        let (le, lo) = lane_sums(lo);
        p.re += 256 * he + le;
        p.im += 256 * ho + lo;
    }

    /// `dot_real` with 128-bit vectors (SSE2 and SSE4.2 alike).
    macro_rules! q15_sse {
        ($name:ident, $token:ident) => {
            fearless_simd::kernel!(
                pub(super) fn $name(t: $token, x: &[i16], taps: &[i16]) -> Partial {
                    let mut p = Partial::default();
                    let whole = x.len().min(taps.len()) / 8 * 8;
                    let (xw, tw) = (
                        x.get(..whole).unwrap_or(&[]),
                        taps.get(..whole).unwrap_or(&[]),
                    );
                    let mask = black_box(_mm_set1_epi16(0xff));
                    for (xc, tc) in xw.chunks(8 * FLUSH).zip(tw.chunks(8 * FLUSH)) {
                        let (mut hi, mut lo) = (_mm_setzero_si128(), _mm_setzero_si128());
                        pairs!(xc, tc, 8, 8, |xv, tv| {
                            let xv: __m128i = i16x8::from_slice(t, xv).into();
                            let tv: __m128i = i16x8::from_slice(t, tv).into();
                            hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_srai_epi16::<8>(xv), tv));
                            lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_and_si128(xv, mask), tv));
                        });
                        add_lanes(
                            &mut p,
                            i32x4::simd_from(t, hi).as_slice(),
                            i32x4::simd_from(t, lo).as_slice(),
                        );
                    }
                    p.re += p.im;
                    p.im = 0;
                    p.done = whole;
                    p
                }
            );
        };
    }
    q15_sse!(dot_q15_sse2, Sse2);
    q15_sse!(dot_q15_sse4_2, Sse4_2);

    fearless_simd::kernel!(
        pub(super) fn dot_q15_avx2(t: Avx2, x: &[i16], taps: &[i16]) -> Partial {
            let mut p = Partial::default();
            let whole = x.len().min(taps.len()) / 16 * 16;
            let (xw, tw) = (
                x.get(..whole).unwrap_or(&[]),
                taps.get(..whole).unwrap_or(&[]),
            );
            let mask = black_box(_mm256_set1_epi16(0xff));
            for (xc, tc) in xw.chunks(16 * FLUSH).zip(tw.chunks(16 * FLUSH)) {
                let (mut hi, mut lo) = (_mm256_setzero_si256(), _mm256_setzero_si256());
                pairs!(xc, tc, 16, 16, |xv, tv| {
                    let xv: __m256i = i16x16::from_slice(t, xv).into();
                    let tv: __m256i = i16x16::from_slice(t, tv).into();
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_srai_epi16::<8>(xv), tv));
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_and_si256(xv, mask), tv));
                });
                add_lanes(
                    &mut p,
                    i32x8::simd_from(t, hi).as_slice(),
                    i32x8::simd_from(t, lo).as_slice(),
                );
            }
            p.re += p.im;
            p.im = 0;
            p.done = whole;
            p
        }
    );

    /// `dot` `CQ15` with 128-bit vectors: steps of 8 samples (two vectors of samples, one of
    /// taps), each vector with its own sums. `$pair` reorders a vector of samples to
    /// `re0 re1 im0 im1 re2 re3 im2 im3`.
    macro_rules! cq15_sse {
        ($name:ident, $token:ident, |$v:ident, $order:ident| $pair:expr) => {
            fearless_simd::kernel!(
                pub(super) fn $name(t: $token, x: &[i16], taps: &[i16]) -> Partial {
                    let mut p = Partial::default();
                    let whole = (x.len() / 2).min(taps.len()) / 8 * 8;
                    let (xw, tw) = (
                        x.get(..2 * whole).unwrap_or(&[]),
                        taps.get(..whole).unwrap_or(&[]),
                    );
                    let mask = black_box(_mm_set1_epi16(0xff));
                    let $order = black_box(_mm_setr_epi8(
                        0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15,
                    ));
                    let pair = |$v: __m128i| -> __m128i { $pair };
                    let zero = _mm_setzero_si128();
                    for (xc, tc) in xw.chunks(16 * FLUSH).zip(tw.chunks(8 * FLUSH)) {
                        let mut acc = [zero; 4];
                        pairs!(xc, tc, 16, 8, |xv, tv| {
                            let (x0, x1) = xv.split_at(8);
                            let x0 = pair(i16x8::from_slice(t, x0).into());
                            let x1 = pair(i16x8::from_slice(t, x1).into());
                            let tv: __m128i = i16x8::from_slice(t, tv).into();
                            // `[d0 d0 d1 d1]` and `[d2 d2 d3 d3]`, `d_j` the taps `2j, 2j + 1`.
                            let t0 = _mm_shuffle_epi32::<0b01_01_00_00>(tv);
                            let t1 = _mm_shuffle_epi32::<0b11_11_10_10>(tv);
                            let [h0, l0, h1, l1] = &mut acc;
                            *h0 = _mm_add_epi32(*h0, _mm_madd_epi16(_mm_srai_epi16::<8>(x0), t0));
                            *l0 = _mm_add_epi32(*l0, _mm_madd_epi16(_mm_and_si128(x0, mask), t0));
                            *h1 = _mm_add_epi32(*h1, _mm_madd_epi16(_mm_srai_epi16::<8>(x1), t1));
                            *l1 = _mm_add_epi32(*l1, _mm_madd_epi16(_mm_and_si128(x1, mask), t1));
                        });
                        let [h0, l0, h1, l1] = acc;
                        add_lanes(
                            &mut p,
                            i32x4::simd_from(t, h0).as_slice(),
                            i32x4::simd_from(t, l0).as_slice(),
                        );
                        add_lanes(
                            &mut p,
                            i32x4::simd_from(t, h1).as_slice(),
                            i32x4::simd_from(t, l1).as_slice(),
                        );
                    }
                    p.done = whole;
                    p
                }
            );
        };
    }
    // SSE2 has no byte shuffle: the two middle words of each half swap with two word shuffles.
    cq15_sse!(dot_cq15_sse2, Sse2, |v, _order| _mm_shufflehi_epi16::<
        0b11_01_10_00,
    >(
        _mm_shufflelo_epi16::<
        0b11_01_10_00,
    >(v)
    ));
    cq15_sse!(dot_cq15_sse4_2, Sse4_2, |v, order| _mm_shuffle_epi8(
        v, order
    ));

    fearless_simd::kernel!(
        pub(super) fn dot_cq15_avx2(t: Avx2, x: &[i16], taps: &[i16]) -> Partial {
            let mut p = Partial::default();
            let whole = (x.len() / 2).min(taps.len()) / 8 * 8;
            let (xw, tw) = (
                x.get(..2 * whole).unwrap_or(&[]),
                taps.get(..whole).unwrap_or(&[]),
            );
            let mask = black_box(_mm256_set1_epi16(0xff));
            let order = black_box(_mm256_setr_epi8(
                0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15, 0, 1, 4, 5, 2, 3, 6, 7, 8, 9,
                12, 13, 10, 11, 14, 15,
            ));
            // `[d0 d0 d1 d1 d2 d2 d3 d3]`, `d_j` the taps `2j, 2j + 1`.
            let dup = black_box(_mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3));
            for (xc, tc) in xw.chunks(16 * FLUSH).zip(tw.chunks(8 * FLUSH)) {
                let (mut hi, mut lo) = (_mm256_setzero_si256(), _mm256_setzero_si256());
                pairs!(xc, tc, 16, 8, |xv, tv| {
                    let xv: __m256i = i16x16::from_slice(t, xv).into();
                    let xv = _mm256_shuffle_epi8(xv, order);
                    let tv: __m128i = i16x8::from_slice(t, tv).into();
                    let tv = _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(tv), dup);
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_srai_epi16::<8>(xv), tv));
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_and_si256(xv, mask), tv));
                });
                add_lanes(
                    &mut p,
                    i32x8::simd_from(t, hi).as_slice(),
                    i32x8::simd_from(t, lo).as_slice(),
                );
            }
            p.done = whole;
            p
        }
    );
    fearless_simd::kernel!(
        pub(super) fn dot_cq15_avx512(t: Avx512, x: &[i16], taps: &[i16]) -> Partial {
            let mut p = Partial::default();
            let whole = (x.len() / 2).min(taps.len()) / 16 * 16;
            let (xw, tw) = (
                x.get(..2 * whole).unwrap_or(&[]),
                taps.get(..whole).unwrap_or(&[]),
            );
            let mask = black_box(_mm512_set1_epi16(0xff));
            let order = black_box(_mm512_broadcast_i32x4(_mm_setr_epi8(
                0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15,
            )));
            // `[d0 d0 d1 d1 ... d7 d7]`, `d_j` the taps `2j, 2j + 1`.
            let dup = black_box(_mm512_setr_epi32(
                0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
            ));
            for (xc, tc) in xw.chunks(32 * FLUSH).zip(tw.chunks(16 * FLUSH)) {
                let (mut hi, mut lo) = (_mm512_setzero_si512(), _mm512_setzero_si512());
                pairs!(xc, tc, 32, 16, |xv, tv| {
                    let xv: __m512i = i16x32::from_slice(t, xv).into();
                    let xv = _mm512_shuffle_epi8(xv, order);
                    let tv: __m256i = i16x16::from_slice(t, tv).into();
                    let tv = _mm512_permutexvar_epi32(dup, _mm512_castsi256_si512(tv));
                    hi = _mm512_add_epi32(hi, _mm512_madd_epi16(_mm512_srai_epi16::<8>(xv), tv));
                    lo = _mm512_add_epi32(lo, _mm512_madd_epi16(_mm512_and_si512(xv, mask), tv));
                });
                add_lanes(
                    &mut p,
                    i32x16::simd_from(t, hi).as_slice(),
                    i32x16::simd_from(t, lo).as_slice(),
                );
            }
            p.done = whole;
            p
        }
    );
}

#[cfg(target_arch = "aarch64")]
mod neon {
    //! `dot_real` `Q15`: `smull` multiplies four 16-bit lanes into exact 32-bit products, and
    //! `sadalp` adds adjacent pairs of them to two 64-bit sums, so no product is ever rounded or
    //! limited.

    use core::arch::aarch64::{
        int16x8_t, int64x2_t, vaddq_s64, vdupq_n_s64, vget_low_s16, vgetq_lane_s64, vmull_high_s16,
        vmull_s16, vpadalq_s32,
    };
    use fearless_simd::i16x8;
    use fearless_simd::prelude::*;

    use super::Partial;

    /// `acc + (x * t)` for the eight lanes, as two pairwise `i64` sums. Macros, not functions:
    /// the intrinsics must be called inside the `#[target_feature]` function of the path.
    macro_rules! mac {
        ($acc:expr, $x:expr, $t:expr $(,)?) => {{
            let [a, b] = $acc;
            let (x, t): (int16x8_t, int16x8_t) = ($x, $t);
            [
                vpadalq_s32(a, vmull_s16(vget_low_s16(x), vget_low_s16(t))),
                vpadalq_s32(b, vmull_high_s16(x, t)),
            ]
        }};
    }

    macro_rules! total {
        ($acc:expr) => {{
            let [a, b]: [int64x2_t; 2] = $acc;
            let s = vaddq_s64(a, b);
            vgetq_lane_s64::<0>(s) + vgetq_lane_s64::<1>(s)
        }};
    }

    fearless_simd::kernel!(
        pub(super) fn dot_q15(t: Neon, x: &[i16], taps: &[i16]) -> Partial {
            let whole = x.len().min(taps.len()) / 16 * 16;
            let (xw, tw) = (
                x.get(..whole).unwrap_or(&[]),
                taps.get(..whole).unwrap_or(&[]),
            );
            let zero = vdupq_n_s64(0);
            let (mut a, mut b) = ([zero; 2], [zero; 2]);
            pairs!(xw, tw, 16, 16, |xv, tv| {
                let (x0, x1) = xv.split_at(8);
                let (t0, t1) = tv.split_at(8);
                a = mac!(
                    a,
                    i16x8::from_slice(t, x0).into(),
                    i16x8::from_slice(t, t0).into(),
                );
                b = mac!(
                    b,
                    i16x8::from_slice(t, x1).into(),
                    i16x8::from_slice(t, t1).into(),
                );
            });
            Partial {
                done: whole,
                re: total!(a) + total!(b),
                im: 0,
            }
        }
    );
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    //! `i32x4.dot_i16x8_s` is `pmaddwd`: the paths of `x86` with 128-bit vectors (the split into
    //! bytes, the reordering of `CQ15` with one byte shuffle).

    use core::arch::wasm32::{
        i8x16_shuffle, i16x8_shr, i16x8_splat, i32x4_add, i32x4_dot_i16x8, i32x4_shuffle,
        u32x4_splat, v128, v128_and,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{WasmSimd128, i16x8, i32x4};

    use super::{FLUSH, Partial, lane_sums};

    #[inline(always)]
    fn add_lanes(p: &mut Partial, t: WasmSimd128, hi: v128, lo: v128) {
        let (he, ho) = lane_sums(i32x4::simd_from(t, hi).as_slice());
        let (le, lo) = lane_sums(i32x4::simd_from(t, lo).as_slice());
        p.re += 256 * he + le;
        p.im += 256 * ho + lo;
    }

    fearless_simd::kernel!(
        pub(super) fn dot_q15(t: WasmSimd128, x: &[i16], taps: &[i16]) -> Partial {
            let mut p = Partial::default();
            let whole = x.len().min(taps.len()) / 8 * 8;
            let (xw, tw) = (
                x.get(..whole).unwrap_or(&[]),
                taps.get(..whole).unwrap_or(&[]),
            );
            let mask = i16x8_splat(0xff);
            let zero = u32x4_splat(0);
            for (xc, tc) in xw.chunks(8 * FLUSH).zip(tw.chunks(8 * FLUSH)) {
                let (mut hi, mut lo) = (zero, zero);
                pairs!(xc, tc, 8, 8, |xv, tv| {
                    let xv: v128 = i16x8::from_slice(t, xv).into();
                    let tv: v128 = i16x8::from_slice(t, tv).into();
                    hi = i32x4_add(hi, i32x4_dot_i16x8(i16x8_shr(xv, 8), tv));
                    lo = i32x4_add(lo, i32x4_dot_i16x8(v128_and(xv, mask), tv));
                });
                add_lanes(&mut p, t, hi, lo);
            }
            p.re += p.im;
            p.im = 0;
            p.done = whole;
            p
        }
    );

    fearless_simd::kernel!(
        pub(super) fn dot_cq15(t: WasmSimd128, x: &[i16], taps: &[i16]) -> Partial {
            let mut p = Partial::default();
            let whole = (x.len() / 2).min(taps.len()) / 8 * 8;
            let (xw, tw) = (
                x.get(..2 * whole).unwrap_or(&[]),
                taps.get(..whole).unwrap_or(&[]),
            );
            let mask = i16x8_splat(0xff);
            let zero = u32x4_splat(0);
            let pair = |v: v128| -> v128 {
                i8x16_shuffle::<0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15>(v, v)
            };
            for (xc, tc) in xw.chunks(16 * FLUSH).zip(tw.chunks(8 * FLUSH)) {
                let mut acc = [zero; 4];
                pairs!(xc, tc, 16, 8, |xv, tv| {
                    let (x0, x1) = xv.split_at(8);
                    let x0 = pair(i16x8::from_slice(t, x0).into());
                    let x1 = pair(i16x8::from_slice(t, x1).into());
                    let tv: v128 = i16x8::from_slice(t, tv).into();
                    let t0 = i32x4_shuffle::<0, 0, 1, 1>(tv, tv);
                    let t1 = i32x4_shuffle::<2, 2, 3, 3>(tv, tv);
                    let [h0, l0, h1, l1] = &mut acc;
                    *h0 = i32x4_add(*h0, i32x4_dot_i16x8(i16x8_shr(x0, 8), t0));
                    *l0 = i32x4_add(*l0, i32x4_dot_i16x8(v128_and(x0, mask), t0));
                    *h1 = i32x4_add(*h1, i32x4_dot_i16x8(i16x8_shr(x1, 8), t1));
                    *l1 = i32x4_add(*l1, i32x4_dot_i16x8(v128_and(x1, mask), t1));
                });
                let [h0, l0, h1, l1] = acc;
                add_lanes(&mut p, t, h0, l0);
                add_lanes(&mut p, t, h1, l1);
            }
            p.done = whole;
            p
        }
    );
}
