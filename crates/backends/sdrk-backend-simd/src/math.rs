//! `add`, `add_real`, `mul`, `mul_real`, `interleave` and `deinterleave`, bit-exact with the
//! fallback (the contract of these kernels is bit-exact, `docs/kernels.md`).
//!
//! `C32` and `f32`: the sum and the product of every float in generic code (one operation per
//! float, no fused operation, so NaN, infinities, `-0` and denormals give the bits of the
//! fallback). The complex product loads the two operands deinterleaved (the [`Split`] of
//! `rotate`: `shufps` and `unpcklps` within the 128-bit blocks on AVX-512, `ld2` and
//! `st2` on NEON, the zips of the token elsewhere) and computes `ar br - ai bi` and
//! `ar bi + ai br` as two multiplications and one subtraction or addition each. SSE4.2 and AVX2
//! multiply the interleaved pairs instead: `(ar, ar) (br, bi)` and `(ai, ai) (bi, br)` with
//! `vmovsldup`, `vmovshdup` and `vpermilps`, then `vaddsubps` (the same products and sums in
//! the same operand order, one rounding each; SSE4.2 with the 128-bit forms, D295). SSE4.2,
//! AVX2, AVX-512 and NEON run the whole steps in the unrolled groups of `blocks3` (D288).
//!
//! `Q15`, sums: `saturating_add` of the token (`paddsw`, `sqadd`, `i16x8.add_sat_s`).
//!
//! Sums and real products run in `whole_zip` (D322): groups of eight vectors, single vectors and
//! an overlapping last vector; NEON stores each pair of vectors with one `st1` of two registers.
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
//! `interleave` and `deinterleave`: the paths of `split_x86`, `split_neon` and `split_wasm` do the
//! whole call (an overlapping last step; calls shorter than one step run the plain loops of the
//! fallback) on every level. With one stream absent the zips and unzips of the token run, and
//! the rest of the call the plain loops of the fallback, inside the dispatched function.
//!
//! The element-wise sums and products and the entries of `interleave` and `deinterleave` are
//! `#[inline]` down to the level paths, and the dispatched function does the whole call: a SIMD
//! path must be at least as fast as the fallback in simulated cycles on both core models of its
//! level (D321), and the fallback's loop is inlined into its caller.

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

// f32 sums and products (also the components of C32 sums) and Q15 sums

/// A whole element-wise call `out[i] = f(a[i], b[i])` of `l` lanes per vector (`group` does
/// eight vectors, `vector` one, `scalar` one value): groups of eight vectors with constant
/// offsets from one index, single vectors, and a last vector that overlaps the one before it
/// where the length is not a multiple of `l` (the values it writes again are the same: `out`
/// cannot alias the inputs); calls shorter than one vector run `scalar`, value for value as the
/// fallback. The whole call runs here, so the remainder costs a few compares instead of a loop
/// of its own (D322: the fixed cost per call decided whether the path beat the fallback's loop
/// in simulated cycles).
#[allow(
    clippy::large_stack_frames,
    reason = "the lint counts the slices and closures of the inlined body before optimization; \
              the loops keep them in registers"
)]
#[inline(always)]
fn whole_zip<E: Copy>(
    l: usize,
    a: &[E],
    b: &[E],
    out: &mut [E],
    group: impl Fn(&[E], &[E], &mut [E]),
    vector: impl Fn(&[E], &[E], &mut [E]),
    scalar: impl Fn(E, E) -> E,
) {
    let n = out.len().min(a.len()).min(b.len());
    let ((a, _), (b, _), (out, _)) = (a.split_at(n), b.split_at(n), out.split_at_mut(n));
    if n < l {
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = scalar(x, y);
        }
        return;
    }
    let step = 8 * l;
    let g = n / step * step;
    let k = n / l * l;
    {
        let ((ag, ar), (bg, br), (og, or)) = (a.split_at(g), b.split_at(g), out.split_at_mut(g));
        for ((a, b), o) in ag
            .chunks_exact(step)
            .zip(bg.chunks_exact(step))
            .zip(og.chunks_exact_mut(step))
        {
            group(a, b, o);
        }
        let ((ar, _), (br, _), (or, _)) = (
            ar.split_at(k - g),
            br.split_at(k - g),
            or.split_at_mut(k - g),
        );
        for ((a, b), o) in ar
            .chunks_exact(l)
            .zip(br.chunks_exact(l))
            .zip(or.chunks_exact_mut(l))
        {
            vector(a, b, o);
        }
    }
    if k < n {
        let ((_, a), (_, b), (_, o)) = (
            a.split_at(n - l),
            b.split_at(n - l),
            out.split_at_mut(n - l),
        );
        vector(a, b, o);
    }
}

/// [`whole_zip`] with the vectors of the token: a group is eight single vectors.
#[inline(always)]
fn whole_zip_token<E: Copy>(
    l: usize,
    a: &[E],
    b: &[E],
    out: &mut [E],
    vector: impl Fn(&[E], &[E], &mut [E]),
    scalar: impl Fn(E, E) -> E,
) {
    whole_zip(
        l,
        a,
        b,
        out,
        |a, b, o| {
            for ((a, b), o) in a
                .chunks_exact(l)
                .zip(b.chunks_exact(l))
                .zip(o.chunks_exact_mut(l))
            {
                vector(a, b, o);
            }
        },
        &vector,
        scalar,
    );
}

/// `out = a + b` (`MUL = false`) or `a * b` (`MUL = true`) of every float.
#[inline(always)]
fn zip_f32<S: SimdToken, const MUL: bool>(simd: S, a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = simd.level().as_neon() {
        if MUL {
            zip_neon::mul_f32(t, a, b, out);
        } else {
            zip_neon::add_f32(t, a, b, out);
        }
        return;
    }
    whole_zip_token(
        S::f32s::LEN,
        a,
        b,
        out,
        |a, b, o| {
            let (x, y) = (S::f32s::from_slice(simd, a), S::f32s::from_slice(simd, b));
            if MUL { x * y } else { x + y }.store_slice(o);
        },
        |x, y| if MUL { x * y } else { x + y },
    );
}

dispatch_fn!(#[cfg_attr(not(target_arch = "wasm32"), inline)] fn run_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) => zip_f32::<_, false>);
dispatch_fn!(#[cfg_attr(not(target_arch = "wasm32"), inline)] fn run_mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) => zip_f32::<_, true>);

/// `out = a + b` of every `Q15` value, saturating (`saturating_add` of the token: `paddsw`,
/// `sqadd`, `i16x8.add_sat_s`).
#[inline(always)]
fn add_i16<S: SimdToken>(simd: S, a: &[i16], b: &[i16], out: &mut [i16]) {
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = simd.level().as_neon() {
        zip_neon::add_i16(t, a, b, out);
        return;
    }
    whole_zip_token(
        S::i16s::LEN,
        a,
        b,
        out,
        |a, b, o| {
            let (x, y) = (S::i16s::from_slice(simd, a), S::i16s::from_slice(simd, b));
            x.saturating_add(y).store_slice(o);
        },
        i16::saturating_add,
    );
}

/// The NEON paths of the element-wise sums and products: [`whole_zip`] with the groups taken in
/// pairs of vectors, each operand loaded by two plain loads (which LLVM pairs into one `ldp`)
/// and the pair stored by one `st1` of two registers (`vst1q_f32_x2`, `vst1q_s16_x2`). LLVM stores the token's pairs with
/// single `str` (or `stp`), which the `cortex-a76` model (`neoverse-n1`) rates at four (six)
/// micro-ops against four of `st1`; the plain loop of the fallback was faster there (D322).
/// The same `fadd`, `fmul` and `sqadd` of the same lanes, so the same bits.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod zip_neon {
    use core::arch::aarch64::{
        float32x4x2_t, int16x8x2_t, vaddq_f32, vld1q_f32, vld1q_s16, vmulq_f32, vqaddq_s16,
        vst1q_f32, vst1q_f32_x2, vst1q_s16, vst1q_s16_x2,
    };

    use super::whole_zip;

    /// One kernel: `$op` on `$v` lanes of `$e` loaded by `$ld`, stored by `$st` (one vector)
    /// and `$st2` (two vectors).
    macro_rules! zip_kernel {
        ($name:ident, $e:ty, $v:literal, $ld:ident, $st:ident, $st2:ident, $pair:ident, $op:ident, $scalar:expr) => {
            fearless_simd::kernel!(
                #[inline(always)]
                pub(super) fn $name(_t: Neon, a: &[$e], b: &[$e], out: &mut [$e]) {
                    whole_zip(
                        $v,
                        a,
                        b,
                        out,
                        |a: &[$e], b: &[$e], o: &mut [$e]| {
                            for ((a, b), o) in a
                                .chunks_exact(2 * $v)
                                .zip(b.chunks_exact(2 * $v))
                                .zip(o.chunks_exact_mut(2 * $v))
                            {
                                // SAFETY: `a` and `b` are the `2 $v` values that the four loads
                                // read and `o` the `2 $v` that the store writes (chunks of
                                // exactly that length).
                                unsafe {
                                    let (pa, pb) = (a.as_ptr(), b.as_ptr());
                                    let x0 = $ld(pa);
                                    let x1 = $ld(pa.add($v));
                                    let y0 = $ld(pb);
                                    let y1 = $ld(pb.add($v));
                                    $st2(o.as_mut_ptr(), $pair($op(x0, y0), $op(x1, y1)));
                                }
                            }
                        },
                        |a: &[$e], b: &[$e], o: &mut [$e]| {
                            assert!(a.len() >= $v && b.len() >= $v && o.len() >= $v);
                            // SAFETY: `a` and `b` hold the `$v` values that the loads read and
                            // `o` the `$v` that the store writes (asserted above).
                            unsafe { $st(o.as_mut_ptr(), $op($ld(a.as_ptr()), $ld(b.as_ptr()))) }
                        },
                        $scalar,
                    );
                }
            );
        };
    }

    zip_kernel!(
        add_f32,
        f32,
        4,
        vld1q_f32,
        vst1q_f32,
        vst1q_f32_x2,
        float32x4x2_t,
        vaddq_f32,
        |x: f32, y: f32| x + y
    );
    zip_kernel!(
        mul_f32,
        f32,
        4,
        vld1q_f32,
        vst1q_f32,
        vst1q_f32_x2,
        float32x4x2_t,
        vmulq_f32,
        |x: f32, y: f32| x * y
    );
    zip_kernel!(
        add_i16,
        i16,
        8,
        vld1q_s16,
        vst1q_s16,
        vst1q_s16_x2,
        int16x8x2_t,
        vqaddq_s16,
        i16::saturating_add
    );
}

dispatch_fn!(#[cfg_attr(not(target_arch = "wasm32"), inline)] fn run_add_i16(a: &[i16], b: &[i16], out: &mut [i16]) => add_i16);

// C32 product

/// The most floats of one step of [`mul_split`] (two vectors of AVX-512).
const MAX_STEP: usize = 32;

/// `(ar + j ai) (br + j bi)` for one vector of each part. Without `FMA` (SSE2, SSE4.2,
/// SIMD128) `re = ar br - ai bi` and `im = ar bi + ai br`: the products and the difference or
/// sum of the fallback, one rounding each. With `FMA` (AVX2, AVX-512 and NEON, D326) the complex
/// product of `rotate` ([`crate::mix::cmul`]): on `x86_64` `re = fms(ar, br, ai bi)` and
/// `im = fma(ar, bi, ai br)`, the lanes of `vfmaddsub` in [`x86::mul_c32_avx2`]; on AArch64
/// `re = fnma(ai, bi, ar br)` (`fmls`).
#[inline(always)]
fn cmul<S: SimdToken, const FMA: bool>(
    simd: S,
    ar: S::f32s,
    ai: S::f32s,
    br: S::f32s,
    bi: S::f32s,
) -> (S::f32s, S::f32s) {
    let _ = simd;
    if FMA {
        crate::mix::cmul::<S>(ar, ai, br, bi)
    } else {
        (ar * br - ai * bi, ar * bi + ai * br)
    }
}

/// The complex products of `a` and `b` (interleaved floats) with [`cmul`]. Without `FMA` the
/// floats after the last whole step go through the scalar arithmetic of the fallback (the same
/// bits). With `FMA` they are copied into one zero-padded step instead, as `quad_demod` does with
/// its last partial vector: `core` has no scalar fused multiply-add, and so every sample gets
/// the arithmetic of the vectors whatever its position in the call.
#[allow(
    clippy::large_stack_frames,
    reason = "the lint counts the slices and closures of the inlined body before optimization; \
              the padded step holds three arrays of 128 bytes"
)]
#[inline(always)]
fn mul_split<S: SimdToken, L: Split<S>, const FMA: bool>(
    simd: S,
    split: L,
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
) {
    let step = 2 * S::f32s::LEN;
    let n = a.len().min(b.len()).min(out.len()) / 2 * 2;
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
        let (re, im) = cmul::<S, FMA>(simd, ar, ai, br, bi);
        split.store(simd, re, im, o);
    }
    if FMA {
        let rest = n - k;
        if rest == 0 {
            return;
        }
        let (mut x, mut y, mut z) = ([0f32; MAX_STEP], [0f32; MAX_STEP], [0f32; MAX_STEP]);
        if let (Some(dx), Some(sx), Some(dy), Some(sy)) = (
            x.get_mut(..rest),
            at.get(..rest),
            y.get_mut(..rest),
            bt.get(..rest),
        ) {
            dx.copy_from_slice(sx);
            dy.copy_from_slice(sy);
        }
        if let (Some(x), Some(y), Some(z)) = (x.get(..step), y.get(..step), z.get_mut(..step)) {
            let (ar, ai) = split.load(simd, x);
            let (br, bi) = split.load(simd, y);
            let (re, im) = cmul::<S, FMA>(simd, ar, ai, br, bi);
            split.store(simd, re, im, z);
        }
        if let (Some(d), Some(s)) = (ot.get_mut(..rest), z.get(..rest)) {
            d.copy_from_slice(s);
        }
        return;
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
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
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

/// The complex products of the whole steps of `$w` floats of `$a` and `$b` with the [`Split`]
/// `$split` of the token `$t`, in the unrolled groups of [`blocks3`], fused ([`cmul`] with
/// `FMA`: the macro runs on AVX-512 and NEON only); evaluates to the number of floats done.
/// AVX-512 keeps this split form: the form of [`x86::mul_c32_avx2`] on 512 bits (`vfmaddsub`
/// after `vmovsldup`, `vmovshdup`, `vpermilps`) simulates 0.51 instead of 0.47 cycles per sample
/// on `znver4` and 0.39 on `sapphirerapids` either way (D326).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
macro_rules! mul_split_blocks {
    ($t:ident, $split:expr, $w:literal, $a:ident, $b:ident, $out:ident) => {{
        let split = $split;
        blocks3!($a, $w, $b, $w, $out, $w, |x, y, z| {
            let (ar, ai) = split.load($t, x);
            let (br, bi) = split.load($t, y);
            let (re, im) = cmul::<_, true>($t, ar, ai, br, bi);
            split.store($t, re, im, z);
        }) * $w
    }};
}

/// The `C32` product at the level of `simd`: on SSE4.2 and AVX2 [`x86::mul_c32_sse4_2`] and
/// [`x86::mul_c32_avx2`] (D295, D288), on AVX-512 and NEON
/// [`mul_split`] with the [`Split`] of the level in the unrolled groups of [`blocks3`] (eight
/// steps per iteration, independent of each other); the rest of the call by [`mul_split`]
/// (D288). AVX2, AVX-512 and NEON fuse the products (D326), SSE2, SSE4.2 and SIMD128 do not.
#[inline(always)]
fn mul_c32<S: SimdToken>(simd: S, a: &[f32], b: &[f32], out: &mut [f32]) {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx512() {
            let k = mul_split_blocks!(t, crate::mix::x86::InLane, 32, a, b, out);
            return mul_rest::<_, _, true>(t, crate::mix::x86::InLane, k, a, b, out);
        }
        if let Some(t) = level.as_avx2() {
            let k = x86::mul_c32_avx2(t, a, b, out);
            return mul_rest::<_, _, true>(t, crate::mix::x86::InLane, k, a, b, out);
        }
        if let Some(t) = level.as_sse4_2() {
            let k = x86::mul_c32_sse4_2(t, a, b, out);
            return mul_rest::<_, _, false>(t, Portable, k, a, b, out);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        let k = mul_split_blocks!(t, crate::mix::neon::Ld2, 8, a, b, out);
        return mul_rest::<_, _, true>(t, crate::mix::neon::Ld2, k, a, b, out);
    }
    let _ = level;
    mul_split::<_, _, false>(simd, Portable, a, b, out);
}

/// [`mul_split`] of the floats from `k` on.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn mul_rest<S: SimdToken, L: Split<S>, const FMA: bool>(
    simd: S,
    split: L,
    k: usize,
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
) {
    if let (Some(a), Some(b), Some(out)) = (a.get(k..), b.get(k..), out.get_mut(k..)) {
        mul_split::<S, L, FMA>(simd, split, a, b, out);
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::{
        __m128, __m256, _mm_add_epi32, _mm_addsub_ps, _mm_cmpeq_epi16, _mm_cmpeq_epi32,
        _mm_madd_epi16, _mm_movehdup_ps, _mm_moveldup_ps, _mm_mul_ps, _mm_mulhrs_epi16,
        _mm_packs_epi32, _mm_set1_epi16, _mm_set1_epi32, _mm_setr_epi8, _mm_setzero_si128,
        _mm_shuffle_epi8, _mm_shuffle_ps, _mm_srai_epi32, _mm_xor_si128, _mm256_add_epi32,
        _mm256_cmpeq_epi16, _mm256_cmpeq_epi32, _mm256_fmaddsub_ps, _mm256_madd_epi16,
        _mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_mul_ps, _mm256_mulhrs_epi16,
        _mm256_packs_epi32, _mm256_permute_ps, _mm256_set1_epi16, _mm256_set1_epi32,
        _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_srai_epi32,
        _mm256_xor_si256, _mm512_cmpeq_epi16_mask, _mm512_mask_blend_epi16, _mm512_mulhrs_epi16,
        _mm512_set1_epi16,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4, f32x8, i16x8, i16x16, i16x32};

    fearless_simd::kernel!(
        /// The `C32` product of AVX2 on the interleaved floats, four samples per vector:
        /// `(ai, ai) * (bi, br)` (`vmovshdup`, `vpermilps`), then `vfmaddsub` with
        /// `(ar, ar)` (`vmovsldup`) and `(br, bi)` gives `fms(ar, br, ai bi)` and
        /// `fma(ar, bi, ai br)` (D326, the form of [`super::cmul`] with FMA). In the unrolled
        /// groups of [`blocks3`]; the number of floats done.
        pub(super) fn mul_c32_avx2(t: Avx2, a: &[f32], b: &[f32], out: &mut [f32]) -> usize {
            blocks3!(a, 8, b, 8, out, 8, |x, y, z| {
                let x: __m256 = f32x8::from_slice(t, x).into();
                let y: __m256 = f32x8::from_slice(t, y).into();
                // The products (ar br, ar bi) and (ai bi, ai br) of every sample. The three
                // shuffles come first: written that way LLVM gives the swap of `y` a register of
                // its own; with each product written next to its shuffle it reuses the register
                // of `y`, and `znver3` simulates 16.1 instead of 13.4 cycles per eight steps.
                let ar = _mm256_moveldup_ps(x);
                let ai = _mm256_movehdup_ps(x);
                let ys = _mm256_permute_ps::<0xB1>(y);
                let r = _mm256_fmaddsub_ps(ar, y, _mm256_mul_ps(ai, ys));
                f32x8::simd_from(t, r).store_slice(z);
            }) * 8
        }
    );

    fearless_simd::kernel!(
        /// The `C32` product of SSE4.2 in the form of [`mul_c32_avx2`] (`movsldup`, `movshdup`,
        /// `shufps`, two `mulps`, `addsubps` of SSE3), two samples per vector, in the unrolled
        /// groups of [`blocks3`]; the number of floats done (D295).
        pub(super) fn mul_c32_sse4_2(t: Sse4_2, a: &[f32], b: &[f32], out: &mut [f32]) -> usize {
            blocks3!(a, 4, b, 4, out, 4, |x, y, z| {
                let x: __m128 = f32x4::from_slice(t, x).into();
                let y: __m128 = f32x4::from_slice(t, y).into();
                let ar = _mm_moveldup_ps(x);
                let ai = _mm_movehdup_ps(x);
                let ys = _mm_shuffle_ps::<0xB1>(y, y);
                let r = _mm_addsub_ps(_mm_mul_ps(ar, y), _mm_mul_ps(ai, ys));
                f32x4::simd_from(t, r).store_slice(z);
            }) * 4
        }
    );

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

/// The plain loops of `interleave` and `deinterleave` with both streams, for the calls of the
/// level paths shorter than one step (the fallback's loops, value for value).
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod split_scalar {
    /// `out = (re[n], im[n])`.
    #[inline(always)]
    pub(super) fn int<E: Copy>(re: &[E], im: &[E], out: &mut [E]) {
        let (out, _) = out.as_chunks_mut::<2>();
        for ((o, &r), &i) in out.iter_mut().zip(re).zip(im) {
            *o = [r, i];
        }
    }

    /// `(re[n], im[n]) = input[n]`.
    #[inline(always)]
    pub(super) fn deint<E: Copy>(input: &[E], re: &mut [E], im: &mut [E]) {
        let (input, _) = input.as_chunks::<2>();
        for ((x, r), i) in input.iter().zip(re.iter_mut()).zip(im.iter_mut()) {
            *r = x[0];
            *i = x[1];
        }
    }
}

/// A whole `interleave` call by a level path: `$step` interleaves `$v` samples (`$w = 2 $v`
/// values out). Groups of eight steps, single steps, and a last step that overlaps the one
/// before it where the length is not a multiple of `$v` (the values it writes again are the
/// same: the output is a copy of inputs that it cannot alias); calls shorter than one step run
/// [`split_scalar::int`]. So the path does every sample and the caller has no rest to do.
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! whole_int {
    ($re:expr, $im:expr, $out:expr, $v:literal, $w:literal, $step:expr) => {{
        let n = $re.len().min($im.len()).min($out.len() / 2);
        let ((re, _), (im, _), (out, _)) =
            ($re.split_at(n), $im.split_at(n), $out.split_at_mut(2 * n));
        if n < $v {
            split_scalar::int(re, im, out);
        } else {
            let step = $step;
            blocks3!(re, $v, im, $v, out, $w, |r, i, o| { step(r, i, o) });
            if n % $v != 0 {
                if let (Some(r), Some(i), Some(o)) = (
                    re.last_chunk::<$v>(),
                    im.last_chunk::<$v>(),
                    out.last_chunk_mut::<$w>(),
                ) {
                    step(r, i, o);
                }
            }
        }
    }};
}

/// [`whole_int`] for `deinterleave`: `$step` splits `$w = 2 $v` values into `$v` real and `$v`
/// imaginary parts.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! whole_deint {
    ($input:expr, $re:expr, $im:expr, $v:literal, $w:literal, $step:expr) => {{
        let n = ($input.len() / 2).min($re.len()).min($im.len());
        let ((input, _), (re, _), (im, _)) = (
            $input.split_at(2 * n),
            $re.split_at_mut(n),
            $im.split_at_mut(n),
        );
        if n < $v {
            split_scalar::deint(input, re, im);
        } else {
            let step = $step;
            blocks3m!(input, $w, re, $v, im, $v, |c, r, i| { step(c, r, i) });
            if n % $v != 0 {
                if let (Some(c), Some(r), Some(i)) = (
                    input.last_chunk::<$w>(),
                    re.last_chunk_mut::<$v>(),
                    im.last_chunk_mut::<$v>(),
                ) {
                    step(c, r, i);
                }
            }
        }
    }};
}

/// The x86 paths of `interleave` and `deinterleave`: the in-lane unpacks and shuffles of the
/// level, with the cross-lane permutation of AVX2 where the lanes of 128 bits have to be put
/// in order (`vperm2f128`/`vperm2i128` of two vectors for `interleave`, `vpermpd` of one for
/// `deinterleave`); on AVX-512 one two-source permutation per output vector (`vpermt2ps`,
/// `vpermt2w`). Each function does the whole call ([`whole_int`], [`whole_deint`]); a level
/// without a function here runs the generic code of the token.
#[cfg(target_arch = "x86_64")]
mod split_x86 {
    use core::arch::x86_64::{
        __m128i, _mm_madd_epi16, _mm_packs_epi32, _mm_set1_epi32, _mm_setr_epi8, _mm_shuffle_epi8,
        _mm_shuffle_ps, _mm_srai_epi32, _mm_unpackhi_epi16, _mm_unpackhi_epi64, _mm_unpackhi_ps,
        _mm_unpacklo_epi16, _mm_unpacklo_epi64, _mm_unpacklo_ps, _mm256_blend_epi32,
        _mm256_castpd_ps, _mm256_castps_pd, _mm256_permute2f128_ps, _mm256_permute2x128_si256,
        _mm256_permute4x64_epi64, _mm256_permute4x64_pd, _mm256_setr_epi8, _mm256_shuffle_epi8,
        _mm256_shuffle_ps, _mm256_unpackhi_epi16, _mm256_unpackhi_ps, _mm256_unpacklo_epi16,
        _mm256_unpacklo_ps, _mm512_permutex2var_epi16, _mm512_permutex2var_ps,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4, f32x8, f32x16, i16x8, i16x16, i16x32, i32x16};

    use super::split_scalar;

    pub(super) mod int_f32 {
        #[allow(
            clippy::wildcard_imports,
            reason = "the intrinsics and vector types of the parent"
        )]
        use super::*;

        fearless_simd::kernel!(
            pub(crate) fn avx2(t: Avx2, re: &[f32], im: &[f32], out: &mut [f32]) {
                whole_int!(
                    re,
                    im,
                    out,
                    8,
                    16,
                    |r: &[f32; 8], i: &[f32; 8], o: &mut [f32; 16]| {
                        let r = f32x8::from_slice(t, r).into();
                        let i = f32x8::from_slice(t, i).into();
                        // The in-lane unpacks give samples 0, 1 | 4, 5 and 2, 3 | 6, 7; the lane
                        // permutation of two sources puts them in order (`vperm2f128`, which the
                        // `znver3` model runs faster than a `vpermpd` of each input).
                        let (lo, hi) = (_mm256_unpacklo_ps(r, i), _mm256_unpackhi_ps(r, i));
                        let (o0, o1) = o.split_at_mut(8);
                        f32x8::simd_from(t, _mm256_permute2f128_ps::<0x20>(lo, hi)).store_slice(o0);
                        f32x8::simd_from(t, _mm256_permute2f128_ps::<0x31>(lo, hi)).store_slice(o1);
                    }
                );
            }
        );

        fearless_simd::kernel!(
            pub(crate) fn sse2(t: Sse2, re: &[f32], im: &[f32], out: &mut [f32]) {
                whole_int!(
                    re,
                    im,
                    out,
                    4,
                    8,
                    |r: &[f32; 4], i: &[f32; 4], o: &mut [f32; 8]| {
                        let r = f32x4::from_slice(t, r).into();
                        let i = f32x4::from_slice(t, i).into();
                        let (o0, o1) = o.split_at_mut(4);
                        f32x4::simd_from(t, _mm_unpacklo_ps(r, i)).store_slice(o0);
                        f32x4::simd_from(t, _mm_unpackhi_ps(r, i)).store_slice(o1);
                    }
                );
            }
        );
    }

    pub(super) mod int_f32_512 {
        #[allow(
            clippy::wildcard_imports,
            reason = "the intrinsics and vector types of the parent"
        )]
        use super::*;

        // `vpermt2ps` indices: `r0, i0, r1, i1, ...` (index 16 + k is `i[k]`).
        const LO: [i32; 16] = [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23];
        const HI: [i32; 16] = [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31];

        fearless_simd::kernel!(
            #[allow(
                clippy::large_stack_frames,
                reason = "the lint counts the 64-byte temporaries of the eight inlined steps before \
                          optimization; the loop keeps them in registers"
            )]
            pub(crate) fn avx512(t: Avx512, re: &[f32], im: &[f32], out: &mut [f32]) {
                // One two-source permutation per output vector (`vpermt2ps`): the AVX2 unpacks
                // and lane permutations compiled for AVX-512 become `vpmovzxdq` and a 256-bit
                // `vpermt2ps` per output, which the `znver4` model runs at 2.4 times VOLK's AVX
                // loop.
                let (lo, hi) = (
                    i32x16::from_slice(t, &LO).into(),
                    i32x16::from_slice(t, &HI).into(),
                );
                whole_int!(
                    re,
                    im,
                    out,
                    16,
                    32,
                    |r: &[f32; 16], i: &[f32; 16], o: &mut [f32; 32]| {
                        let r = f32x16::from_slice(t, r).into();
                        let i = f32x16::from_slice(t, i).into();
                        let (o0, o1) = o.split_at_mut(16);
                        f32x16::simd_from(t, _mm512_permutex2var_ps(r, lo, i)).store_slice(o0);
                        f32x16::simd_from(t, _mm512_permutex2var_ps(r, hi, i)).store_slice(o1);
                    }
                );
            }
        );
    }

    pub(super) mod deint_f32_512 {
        #[allow(
            clippy::wildcard_imports,
            reason = "the intrinsics and vector types of the parent"
        )]
        use super::*;

        // `vpermt2ps` indices: the even and the odd values of `a` and then `b` (16 + k is `b[k]`).
        const EVEN: [i32; 16] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30];
        const ODD: [i32; 16] = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31];

        fearless_simd::kernel!(
            #[allow(
                clippy::large_stack_frames,
                reason = "the lint counts the 64-byte temporaries of the eight inlined steps before \
                          optimization; the loop keeps them in registers"
            )]
            pub(crate) fn avx512(t: Avx512, input: &[f32], re: &mut [f32], im: &mut [f32]) {
                let (even, odd) = (
                    i32x16::from_slice(t, &EVEN).into(),
                    i32x16::from_slice(t, &ODD).into(),
                );
                whole_deint!(
                    input,
                    re,
                    im,
                    16,
                    32,
                    |c: &[f32; 32], r: &mut [f32; 16], i: &mut [f32; 16]| {
                        let (a, b) = c.split_at(16);
                        let a = f32x16::from_slice(t, a).into();
                        let b = f32x16::from_slice(t, b).into();
                        f32x16::simd_from(t, _mm512_permutex2var_ps(a, even, b)).store_slice(r);
                        f32x16::simd_from(t, _mm512_permutex2var_ps(a, odd, b)).store_slice(i);
                    }
                );
            }
        );
    }

    pub(super) mod int_i16_512 {
        #[allow(
            clippy::wildcard_imports,
            reason = "the intrinsics and vector types of the parent"
        )]
        use super::*;

        // `vpermt2w` indices, as for `f32` with 32 values per vector.
        const LO: [i16; 32] = [
            0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43,
            12, 44, 13, 45, 14, 46, 15, 47,
        ];
        const HI: [i16; 32] = [
            16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58,
            27, 59, 28, 60, 29, 61, 30, 62, 31, 63,
        ];

        fearless_simd::kernel!(
            #[allow(
                clippy::large_stack_frames,
                reason = "the lint counts the 64-byte temporaries of the eight inlined steps before \
                          optimization; the loop keeps them in registers"
            )]
            pub(crate) fn avx512(t: Avx512, re: &[i16], im: &[i16], out: &mut [i16]) {
                let (lo, hi) = (
                    i16x32::from_slice(t, &LO).into(),
                    i16x32::from_slice(t, &HI).into(),
                );
                whole_int!(
                    re,
                    im,
                    out,
                    32,
                    64,
                    |r: &[i16; 32], i: &[i16; 32], o: &mut [i16; 64]| {
                        let r = i16x32::from_slice(t, r).into();
                        let i = i16x32::from_slice(t, i).into();
                        let (o0, o1) = o.split_at_mut(32);
                        i16x32::simd_from(t, _mm512_permutex2var_epi16(r, lo, i)).store_slice(o0);
                        i16x32::simd_from(t, _mm512_permutex2var_epi16(r, hi, i)).store_slice(o1);
                    }
                );
            }
        );
    }

    pub(super) mod deint_i16_512 {
        #[allow(
            clippy::wildcard_imports,
            reason = "the intrinsics and vector types of the parent"
        )]
        use super::*;

        // `vpermt2w` indices, as for `f32` with 32 values per vector.
        const EVEN: [i16; 32] = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44,
            46, 48, 50, 52, 54, 56, 58, 60, 62,
        ];
        const ODD: [i16; 32] = [
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45,
            47, 49, 51, 53, 55, 57, 59, 61, 63,
        ];

        fearless_simd::kernel!(
            #[allow(
                clippy::large_stack_frames,
                reason = "the lint counts the 64-byte temporaries of the eight inlined steps before \
                          optimization; the loop keeps them in registers"
            )]
            pub(crate) fn avx512(t: Avx512, input: &[i16], re: &mut [i16], im: &mut [i16]) {
                let (even, odd) = (
                    i16x32::from_slice(t, &EVEN).into(),
                    i16x32::from_slice(t, &ODD).into(),
                );
                whole_deint!(
                    input,
                    re,
                    im,
                    32,
                    64,
                    |c: &[i16; 64], r: &mut [i16; 32], i: &mut [i16; 32]| {
                        let (a, b) = c.split_at(32);
                        let a = i16x32::from_slice(t, a).into();
                        let b = i16x32::from_slice(t, b).into();
                        i16x32::simd_from(t, _mm512_permutex2var_epi16(a, even, b)).store_slice(r);
                        i16x32::simd_from(t, _mm512_permutex2var_epi16(a, odd, b)).store_slice(i);
                    }
                );
            }
        );
    }

    pub(super) mod deint_f32 {
        #[allow(
            clippy::wildcard_imports,
            reason = "the intrinsics and vector types of the parent"
        )]
        use super::*;

        fearless_simd::kernel!(
            pub(crate) fn avx2(t: Avx2, input: &[f32], re: &mut [f32], im: &mut [f32]) {
                whole_deint!(
                    input,
                    re,
                    im,
                    8,
                    16,
                    |c: &[f32; 16], r: &mut [f32; 8], i: &mut [f32; 8]| {
                        let (a, b) = c.split_at(8);
                        let a = f32x8::from_slice(t, a).into();
                        let b = f32x8::from_slice(t, b).into();
                        // Samples 0, 1, 4, 5 | 2, 3, 6, 7, then the lanes in order.
                        let lo = _mm256_shuffle_ps::<0x88>(a, b);
                        let hi = _mm256_shuffle_ps::<0xDD>(a, b);
                        let lo =
                            _mm256_castpd_ps(_mm256_permute4x64_pd::<0xD8>(_mm256_castps_pd(lo)));
                        let hi =
                            _mm256_castpd_ps(_mm256_permute4x64_pd::<0xD8>(_mm256_castps_pd(hi)));
                        f32x8::simd_from(t, lo).store_slice(r);
                        f32x8::simd_from(t, hi).store_slice(i);
                    }
                );
            }
        );

        fearless_simd::kernel!(
            pub(crate) fn sse2(t: Sse2, input: &[f32], re: &mut [f32], im: &mut [f32]) {
                whole_deint!(
                    input,
                    re,
                    im,
                    4,
                    8,
                    |c: &[f32; 8], r: &mut [f32; 4], i: &mut [f32; 4]| {
                        let (a, b) = c.split_at(4);
                        let a = f32x4::from_slice(t, a).into();
                        let b = f32x4::from_slice(t, b).into();
                        f32x4::simd_from(t, _mm_shuffle_ps::<0x88>(a, b)).store_slice(r);
                        f32x4::simd_from(t, _mm_shuffle_ps::<0xDD>(a, b)).store_slice(i);
                    }
                );
            }
        );
    }

    pub(super) mod int_i16 {
        #[allow(
            clippy::wildcard_imports,
            reason = "the intrinsics and vector types of the parent"
        )]
        use super::*;

        fearless_simd::kernel!(
            pub(crate) fn avx2(t: Avx2, re: &[i16], im: &[i16], out: &mut [i16]) {
                whole_int!(
                    re,
                    im,
                    out,
                    16,
                    32,
                    |r: &[i16; 16], i: &[i16; 16], o: &mut [i16; 32]| {
                        let r = i16x16::from_slice(t, r).into();
                        let i = i16x16::from_slice(t, i).into();
                        // As `int_f32`: the in-lane unpacks, then `vperm2i128`.
                        let (lo, hi) = (_mm256_unpacklo_epi16(r, i), _mm256_unpackhi_epi16(r, i));
                        let (o0, o1) = o.split_at_mut(16);
                        i16x16::simd_from(t, _mm256_permute2x128_si256::<0x20>(lo, hi))
                            .store_slice(o0);
                        i16x16::simd_from(t, _mm256_permute2x128_si256::<0x31>(lo, hi))
                            .store_slice(o1);
                    }
                );
            }
        );

        fearless_simd::kernel!(
            pub(crate) fn sse2(t: Sse2, re: &[i16], im: &[i16], out: &mut [i16]) {
                whole_int!(
                    re,
                    im,
                    out,
                    8,
                    16,
                    |r: &[i16; 8], i: &[i16; 8], o: &mut [i16; 16]| {
                        let r = i16x8::from_slice(t, r).into();
                        let i = i16x8::from_slice(t, i).into();
                        let (o0, o1) = o.split_at_mut(8);
                        i16x8::simd_from(t, _mm_unpacklo_epi16(r, i)).store_slice(o0);
                        i16x8::simd_from(t, _mm_unpackhi_epi16(r, i)).store_slice(o1);
                    }
                );
            }
        );
    }

    pub(super) mod deint_i16 {
        #[allow(
            clippy::wildcard_imports,
            reason = "the intrinsics and vector types of the parent"
        )]
        use super::*;

        fearless_simd::kernel!(
            pub(crate) fn avx2(t: Avx2, input: &[i16], re: &mut [i16], im: &mut [i16]) {
                // Per lane of the first vector: four real parts, then four imaginary parts; of
                // the second: the imaginary parts first. A blend of the two vectors (`vpblendd`,
                // no shuffle port) then holds the real parts of both and another the imaginary
                // parts, and one `vpermq` each puts the four quarters in order: four shuffles
                // per 16 samples where `vpunpck*qdq` needed six (eight as compiled).
                let split = _mm256_setr_epi8(
                    0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 0, 1, 4, 5, 8, 9, 12, 13,
                    2, 3, 6, 7, 10, 11, 14, 15,
                );
                let swap = _mm256_setr_epi8(
                    2, 3, 6, 7, 10, 11, 14, 15, 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14,
                    15, 0, 1, 4, 5, 8, 9, 12, 13,
                );
                whole_deint!(
                    input,
                    re,
                    im,
                    16,
                    32,
                    |c: &[i16; 32], r: &mut [i16; 16], i: &mut [i16; 16]| {
                        let (a, b) = c.split_at(16);
                        let a = _mm256_shuffle_epi8(i16x16::from_slice(t, a).into(), split);
                        let b = _mm256_shuffle_epi8(i16x16::from_slice(t, b).into(), swap);
                        // Quarters (a re | b re), (a re | b re) and (b im | a im), (b im | a im).
                        let lo = _mm256_blend_epi32::<0xCC>(a, b);
                        let hi = _mm256_blend_epi32::<0x33>(a, b);
                        i16x16::simd_from(t, _mm256_permute4x64_epi64::<0xD8>(lo)).store_slice(r);
                        i16x16::simd_from(t, _mm256_permute4x64_epi64::<0x8D>(hi)).store_slice(i);
                    }
                );
            }
        );

        fearless_simd::kernel!(
            pub(crate) fn sse4_2(t: Sse4_2, input: &[i16], re: &mut [i16], im: &mut [i16]) {
                let split = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
                whole_deint!(
                    input,
                    re,
                    im,
                    8,
                    16,
                    |c: &[i16; 16], r: &mut [i16; 8], i: &mut [i16; 8]| {
                        let (a, b) = c.split_at(8);
                        let a = _mm_shuffle_epi8(i16x8::from_slice(t, a).into(), split);
                        let b = _mm_shuffle_epi8(i16x8::from_slice(t, b).into(), split);
                        i16x8::simd_from(t, _mm_unpacklo_epi64(a, b)).store_slice(r);
                        i16x8::simd_from(t, _mm_unpackhi_epi64(a, b)).store_slice(i);
                    }
                );
            }
        );

        fearless_simd::kernel!(
            pub(crate) fn sse2(t: Sse2, input: &[i16], re: &mut [i16], im: &mut [i16]) {
                // `pmaddwd` with `(1, 0)` per pair is the real part sign-extended to 32 bits
                // (`re * 1 + im * 0`, exact), the arithmetic shift by 16 the imaginary part;
                // `packssdw` packs both back without saturating anything. Two multiplications,
                // two shifts and two packs per eight samples: the three rounds of unpacks, which
                // LLVM turns into shifts and packs of both parts, kept the one shift pipe of
                // `znver2` busy (0.76 cycles per sample against 0.64 of the fallback; D323).
                let real = _mm_set1_epi32(1);
                whole_deint!(
                    input,
                    re,
                    im,
                    8,
                    16,
                    |c: &[i16; 16], r: &mut [i16; 8], i: &mut [i16; 8]| {
                        let (a, b) = c.split_at(8);
                        let a: __m128i = i16x8::from_slice(t, a).into();
                        let b: __m128i = i16x8::from_slice(t, b).into();
                        let (ra, rb) = (_mm_madd_epi16(a, real), _mm_madd_epi16(b, real));
                        i16x8::simd_from(t, _mm_packs_epi32(ra, rb)).store_slice(r);
                        let (ia, ib) = (_mm_srai_epi32::<16>(a), _mm_srai_epi32::<16>(b));
                        i16x8::simd_from(t, _mm_packs_epi32(ia, ib)).store_slice(i);
                    }
                );
            }
        );
    }
}

/// The NEON paths of `interleave` and `deinterleave`: `ld2` and `st2` deinterleave and
/// interleave while loading and storing; each does the whole call as on x86. The groups of eight
/// steps of `interleave` are one `asm!` block of plain loads and `st2` (D324): from the
/// intrinsics LLVM pairs the loads of neighbouring steps into `ldp`, three micro-ops each on the
/// `cortex-a72` model against one per `ldr`, and the group then took 1.06 cycles per sample
/// against 1.01 of the fallback's loop (which reuses its registers, so LLVM pairs nothing there).
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod split_neon {
    use core::arch::aarch64::{
        float32x4x2_t, int16x8x2_t, vld1q_f32, vld1q_s16, vld2q_f32, vld2q_s16, vst1q_f32,
        vst1q_s16, vst2q_f32, vst2q_s16,
    };
    use core::arch::asm;

    use super::split_scalar;

    /// Eight steps of `interleave` in one block: from `$r` and `$i` 128 bytes each (eight
    /// vectors), to `$o` 256 bytes, `$t` the arrangement of the lanes (`4s`, `8h`). Plain `ldr`
    /// with the offset in the instruction, `st2` at the address of the step (one `add` each),
    /// the registers `v16` to `v23` in turn.
    macro_rules! group_st2 {
        ($t:literal, $r:expr, $i:expr, $o:expr) => {
            asm!(
                "ldr q16, [{r}]",
                "ldr q17, [{i}]",
                concat!("st2 {{ v16.", $t, ", v17.", $t, " }}, [{o}]"),
                "ldr q18, [{r}, #16]",
                "ldr q19, [{i}, #16]",
                "add {a}, {o}, #32",
                concat!("st2 {{ v18.", $t, ", v19.", $t, " }}, [{a}]"),
                "ldr q20, [{r}, #32]",
                "ldr q21, [{i}, #32]",
                "add {a}, {o}, #64",
                concat!("st2 {{ v20.", $t, ", v21.", $t, " }}, [{a}]"),
                "ldr q22, [{r}, #48]",
                "ldr q23, [{i}, #48]",
                "add {a}, {o}, #96",
                concat!("st2 {{ v22.", $t, ", v23.", $t, " }}, [{a}]"),
                "ldr q16, [{r}, #64]",
                "ldr q17, [{i}, #64]",
                "add {a}, {o}, #128",
                concat!("st2 {{ v16.", $t, ", v17.", $t, " }}, [{a}]"),
                "ldr q18, [{r}, #80]",
                "ldr q19, [{i}, #80]",
                "add {a}, {o}, #160",
                concat!("st2 {{ v18.", $t, ", v19.", $t, " }}, [{a}]"),
                "ldr q20, [{r}, #96]",
                "ldr q21, [{i}, #96]",
                "add {a}, {o}, #192",
                concat!("st2 {{ v20.", $t, ", v21.", $t, " }}, [{a}]"),
                "ldr q22, [{r}, #112]",
                "ldr q23, [{i}, #112]",
                "add {a}, {o}, #224",
                concat!("st2 {{ v22.", $t, ", v23.", $t, " }}, [{a}]"),
                r = in(reg) $r,
                i = in(reg) $i,
                o = in(reg) $o,
                a = out(reg) _,
                out("v16") _,
                out("v17") _,
                out("v18") _,
                out("v19") _,
                out("v20") _,
                out("v21") _,
                out("v22") _,
                out("v23") _,
                options(nostack, preserves_flags),
            )
        };
    }

    /// `whole_int` with the groups of eight steps of `group_st2`: `$v` samples per step
    /// (16 bytes of each input), `$step` one step by the intrinsics.
    macro_rules! whole_int_st2 {
        ($re:expr, $im:expr, $out:expr, $v:literal, $t:literal, $step:expr) => {{
            let n = $re.len().min($im.len()).min($out.len() / 2);
            let ((re, _), (im, _), (out, _)) =
                ($re.split_at(n), $im.split_at(n), $out.split_at_mut(2 * n));
            if n < $v {
                split_scalar::int(re, im, out);
            } else {
                let step = $step;
                let g = n / (8 * $v) * (8 * $v);
                let ((rg, rr), (ig, ir), (og, or)) =
                    (re.split_at(g), im.split_at(g), out.split_at_mut(2 * g));
                for ((r, i), o) in rg
                    .chunks_exact(8 * $v)
                    .zip(ig.chunks_exact(8 * $v))
                    .zip(og.chunks_exact_mut(16 * $v))
                {
                    // SAFETY: `r` and `i` are the eight vectors (128 bytes) that the block reads
                    // and `o` the 256 bytes that it writes; the block touches no other memory,
                    // no stack and no flags, and clobbers only the registers it names.
                    unsafe { group_st2!($t, r.as_ptr(), i.as_ptr(), o.as_mut_ptr()) }
                }
                let (rs, _) = rr.as_chunks::<$v>();
                let (is, _) = ir.as_chunks::<$v>();
                let (os, _) = or.as_chunks_mut::<{ 2 * $v }>();
                for ((r, i), o) in rs.iter().zip(is).zip(os) {
                    step(r, i, o);
                }
                if n % $v != 0 {
                    if let (Some(r), Some(i), Some(o)) = (
                        re.last_chunk::<$v>(),
                        im.last_chunk::<$v>(),
                        out.last_chunk_mut::<{ 2 * $v }>(),
                    ) {
                        step(r, i, o);
                    }
                }
            }
        }};
    }

    fearless_simd::kernel!(
        #[inline]
        pub(super) fn int_f32(_t: Neon, re: &[f32], im: &[f32], out: &mut [f32]) {
            whole_int_st2!(
                re,
                im,
                out,
                4,
                "4s",
                |r: &[f32; 4], i: &[f32; 4], o: &mut [f32; 8]| {
                    // SAFETY: `r` and `i` are the 4 values that `vld1q_f32` reads and `o` the 8 that
                    // `vst2q_f32` writes.
                    unsafe {
                        vst2q_f32(
                            o.as_mut_ptr(),
                            float32x4x2_t(vld1q_f32(r.as_ptr()), vld1q_f32(i.as_ptr())),
                        );
                    }
                }
            );
        }
    );

    fearless_simd::kernel!(
        pub(super) fn deint_f32(_t: Neon, input: &[f32], re: &mut [f32], im: &mut [f32]) {
            whole_deint!(
                input,
                re,
                im,
                4,
                8,
                |c: &[f32; 8], r: &mut [f32; 4], i: &mut [f32; 4]| {
                    // SAFETY: `c` is the 8 values that `vld2q_f32` reads, `r` and `i` the 4 that
                    // `vst1q_f32` writes.
                    unsafe {
                        let v = vld2q_f32(c.as_ptr());
                        vst1q_f32(r.as_mut_ptr(), v.0);
                        vst1q_f32(i.as_mut_ptr(), v.1);
                    }
                }
            );
        }
    );

    fearless_simd::kernel!(
        #[inline]
        pub(super) fn int_i16(_t: Neon, re: &[i16], im: &[i16], out: &mut [i16]) {
            whole_int_st2!(
                re,
                im,
                out,
                8,
                "8h",
                |r: &[i16; 8], i: &[i16; 8], o: &mut [i16; 16]| {
                    // SAFETY: `r` and `i` are the 8 values that `vld1q_s16` reads and `o` the 16 that
                    // `vst2q_s16` writes.
                    unsafe {
                        vst2q_s16(
                            o.as_mut_ptr(),
                            int16x8x2_t(vld1q_s16(r.as_ptr()), vld1q_s16(i.as_ptr())),
                        );
                    }
                }
            );
        }
    );

    fearless_simd::kernel!(
        pub(super) fn deint_i16(_t: Neon, input: &[i16], re: &mut [i16], im: &mut [i16]) {
            whole_deint!(
                input,
                re,
                im,
                8,
                16,
                |c: &[i16; 16], r: &mut [i16; 8], i: &mut [i16; 8]| {
                    // SAFETY: `c` is the 16 values that `vld2q_s16` reads, `r` and `i` the 8 that
                    // `vst1q_s16` writes.
                    unsafe {
                        let v = vld2q_s16(c.as_ptr());
                        vst1q_s16(r.as_mut_ptr(), v.0);
                        vst1q_s16(i.as_mut_ptr(), v.1);
                    }
                }
            );
        }
    );
}

/// The `simd128` paths of `interleave` and `deinterleave`: the zips and unzips of the token, in
/// the unrolled loop with the overlapping last step of [`whole_int`] and [`whole_deint`] (the
/// plain loop of the token keeps four pointers and a counter per vector; the engine counts every
/// instruction).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod split_wasm {
    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4, i16x8};

    use super::split_scalar;

    fearless_simd::kernel!(
        pub(super) fn int_f32(t: WasmSimd128, re: &[f32], im: &[f32], out: &mut [f32]) {
            whole_int!(
                re,
                im,
                out,
                4,
                8,
                |r: &[f32; 4], i: &[f32; 4], o: &mut [f32; 8]| {
                    let (r, i) = (f32x4::from_slice(t, r), f32x4::from_slice(t, i));
                    let (o0, o1) = o.split_at_mut(4);
                    r.zip_low(i).store_slice(o0);
                    r.zip_high(i).store_slice(o1);
                }
            );
        }
    );

    fearless_simd::kernel!(
        pub(super) fn deint_f32(t: WasmSimd128, input: &[f32], re: &mut [f32], im: &mut [f32]) {
            whole_deint!(
                input,
                re,
                im,
                4,
                8,
                |c: &[f32; 8], r: &mut [f32; 4], i: &mut [f32; 4]| {
                    let (a, b) = c.split_at(4);
                    let (a, b) = (f32x4::from_slice(t, a), f32x4::from_slice(t, b));
                    a.unzip_low(b).store_slice(r);
                    a.unzip_high(b).store_slice(i);
                }
            );
        }
    );

    fearless_simd::kernel!(
        pub(super) fn int_i16(t: WasmSimd128, re: &[i16], im: &[i16], out: &mut [i16]) {
            whole_int!(
                re,
                im,
                out,
                8,
                16,
                |r: &[i16; 8], i: &[i16; 8], o: &mut [i16; 16]| {
                    let (r, i) = (i16x8::from_slice(t, r), i16x8::from_slice(t, i));
                    let (o0, o1) = o.split_at_mut(8);
                    r.zip_low(i).store_slice(o0);
                    r.zip_high(i).store_slice(o1);
                }
            );
        }
    );

    fearless_simd::kernel!(
        pub(super) fn deint_i16(t: WasmSimd128, input: &[i16], re: &mut [i16], im: &mut [i16]) {
            whole_deint!(
                input,
                re,
                im,
                8,
                16,
                |c: &[i16; 16], r: &mut [i16; 8], i: &mut [i16; 8]| {
                    let (a, b) = c.split_at(8);
                    let (a, b) = (i16x8::from_slice(t, a), i16x8::from_slice(t, b));
                    a.unzip_low(b).store_slice(r);
                    a.unzip_high(b).store_slice(i);
                }
            );
        }
    );
}

/// The level-specific paths of `interleave` with both inputs and of `deinterleave` with both
/// outputs, for the current level: `true` where the level has a path, which then did the whole
/// call; `false` where the generic code of the token runs. The kernel returns right after a
/// path, so the dispatch of the run-time detection calls the path without keeping the
/// arguments for a rest.
macro_rules! level_paths {
    (
        $int:ident, $deint:ident, $elem:ty,
        x86_int: [$($xi:ident => $xip:path),*],
        x86_deint: [$($xd:ident => $xdp:path),*],
        neon: $ni:path, $nd:path,
        wasm: $wi:path, $wd:path
    ) => {
        #[inline(always)]
        fn $int<S: SimdToken>(simd: S, re: &[$elem], im: &[$elem], out: &mut [$elem]) -> bool {
            let level = simd.level();
            #[cfg(target_arch = "x86_64")]
            {
                $(if let Some(t) = level.$xi() {
                    $xip(t, re, im, out);
                    return true;
                })*
            }
            #[cfg(target_arch = "aarch64")]
            if let Some(t) = level.as_neon() {
                $ni(t, re, im, out);
                return true;
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            if let Some(t) = level.as_wasm_simd128() {
                $wi(t, re, im, out);
                return true;
            }
            let _ = (level, re, im, out);
            false
        }

        #[inline(always)]
        fn $deint<S: SimdToken>(
            simd: S,
            input: &[$elem],
            re: &mut [$elem],
            im: &mut [$elem],
        ) -> bool {
            let level = simd.level();
            #[cfg(target_arch = "x86_64")]
            {
                $(if let Some(t) = level.$xd() {
                    $xdp(t, input, re, im);
                    return true;
                })*
            }
            #[cfg(target_arch = "aarch64")]
            if let Some(t) = level.as_neon() {
                $nd(t, input, re, im);
                return true;
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            if let Some(t) = level.as_wasm_simd128() {
                $wd(t, input, re, im);
                return true;
            }
            let _ = (level, input, re, im);
            false
        }
    };
}

level_paths!(
    int_f32_level, deint_f32_level, f32,
    x86_int: [
        as_avx512 => split_x86::int_f32_512::avx512,
        as_avx2 => split_x86::int_f32::avx2,
        as_sse2 => split_x86::int_f32::sse2
    ],
    x86_deint: [
        as_avx512 => split_x86::deint_f32_512::avx512,
        as_avx2 => split_x86::deint_f32::avx2,
        as_sse2 => split_x86::deint_f32::sse2
    ],
    neon: split_neon::int_f32, split_neon::deint_f32,
    wasm: split_wasm::int_f32, split_wasm::deint_f32
);
level_paths!(
    int_i16_level, deint_i16_level, i16,
    x86_int: [
        as_avx512 => split_x86::int_i16_512::avx512,
        as_avx2 => split_x86::int_i16::avx2,
        as_sse2 => split_x86::int_i16::sse2
    ],
    x86_deint: [
        as_avx512 => split_x86::deint_i16_512::avx512,
        as_avx2 => split_x86::deint_i16::avx2,
        as_sse4_2 => split_x86::deint_i16::sse4_2,
        as_sse2 => split_x86::deint_i16::sse2
    ],
    neon: split_neon::int_i16, split_neon::deint_i16,
    wasm: split_wasm::int_i16, split_wasm::deint_i16
);

/// `out = (re[n], im[n])`, `im = None` giving zeros: whole vectors of samples by the path of the
/// level or the zips of the token, the rest (fewer samples than one step) by the plain loop of
/// the fallback. The lengths are checked by the caller (`out` holds two values per sample of
/// `re`, `im` as many as `re`). The whole call runs in the dispatched function, so that the
/// caller keeps no arguments across it (the fixed cost per call of the run-time detection).
macro_rules! interleave_kernel {
    ($name:ident, $vec:ident, $elem:ty, $zero:expr, $level:ident) => {
        #[inline(always)]
        fn $name<S: SimdToken>(simd: S, re: &[$elem], im: Option<&[$elem]>, out: &mut [$elem]) {
            let l = S::$vec::LEN;
            let done = match im {
                Some(im) => {
                    if $level(simd, re, im, out) {
                        return;
                    }
                    let mut n = 0;
                    for ((r, i), o) in re
                        .chunks_exact(l)
                        .zip(im.chunks_exact(l))
                        .zip(out.chunks_exact_mut(2 * l))
                    {
                        let (r, i) = (S::$vec::from_slice(simd, r), S::$vec::from_slice(simd, i));
                        let (o0, o1) = o.split_at_mut(l);
                        r.zip_low(i).store_slice(o0);
                        r.zip_high(i).store_slice(o1);
                        n += l;
                    }
                    n
                }
                None => {
                    let z = S::$vec::splat(simd, $zero);
                    let mut n = 0;
                    for (r, o) in re.chunks_exact(l).zip(out.chunks_exact_mut(2 * l)) {
                        let r = S::$vec::from_slice(simd, r);
                        let (o0, o1) = o.split_at_mut(l);
                        r.zip_low(z).store_slice(o0);
                        r.zip_high(z).store_slice(o1);
                        n += l;
                    }
                    n
                }
            };
            let re = re.get(done..).unwrap_or_default();
            let (out, _) = out
                .get_mut(2 * done..)
                .unwrap_or_default()
                .as_chunks_mut::<2>();
            match im.map(|im| im.get(done..).unwrap_or_default()) {
                Some(im) => {
                    for ((o, &r), &i) in out.iter_mut().zip(re).zip(im) {
                        *o = [r, i];
                    }
                }
                None => {
                    for (o, &r) in out.iter_mut().zip(re) {
                        *o = [r, $zero];
                    }
                }
            }
        }
    };
}

interleave_kernel!(interleave_f32, f32s, f32, 0.0, int_f32_level);
interleave_kernel!(interleave_i16, i16s, i16, 0, int_i16_level);

dispatch_fn!(#[cfg_attr(not(target_arch = "wasm32"), inline)] fn run_interleave_f32(re: &[f32], im: Option<&[f32]>, out: &mut [f32]) => interleave_f32);
dispatch_fn!(#[cfg_attr(not(target_arch = "wasm32"), inline)] fn run_interleave_i16(re: &[i16], im: Option<&[i16]>, out: &mut [i16]) => interleave_i16);

/// The real and imaginary parts of `input` (two values per sample) into the outputs given:
/// whole vectors of samples by the path of the level (both outputs) or the unzips of the token,
/// the rest by the plain loops of the fallback. The lengths are checked by the caller (every
/// output holds one value per sample of `input`); as [`interleave_kernel`], the whole call runs
/// in the dispatched function.
macro_rules! deinterleave_kernel {
    ($name:ident, $vec:ident, $elem:ty, $level:ident) => {
        #[inline(always)]
        fn $name<S: SimdToken>(
            simd: S,
            input: &[$elem],
            re: Option<&mut [$elem]>,
            im: Option<&mut [$elem]>,
        ) {
            let l = S::$vec::LEN;
            let load = |c: &[$elem]| {
                let (c0, c1) = c.split_at(l);
                (S::$vec::from_slice(simd, c0), S::$vec::from_slice(simd, c1))
            };
            let (mut re, mut im) = (re, im);
            let done = match (re.as_deref_mut(), im.as_deref_mut()) {
                (Some(re), Some(im)) => {
                    if $level(simd, input, re, im) {
                        return;
                    }
                    let mut k = 0;
                    for ((c, r), i) in input
                        .chunks_exact(2 * l)
                        .zip(re.chunks_exact_mut(l))
                        .zip(im.chunks_exact_mut(l))
                    {
                        let (a, b) = load(c);
                        a.unzip_low(b).store_slice(r);
                        a.unzip_high(b).store_slice(i);
                        k += l;
                    }
                    k
                }
                (Some(re), None) => {
                    let mut k = 0;
                    for (c, r) in input.chunks_exact(2 * l).zip(re.chunks_exact_mut(l)) {
                        let (a, b) = load(c);
                        a.unzip_low(b).store_slice(r);
                        k += l;
                    }
                    k
                }
                (None, Some(im)) => {
                    let mut k = 0;
                    for (c, i) in input.chunks_exact(2 * l).zip(im.chunks_exact_mut(l)) {
                        let (a, b) = load(c);
                        a.unzip_high(b).store_slice(i);
                        k += l;
                    }
                    k
                }
                (None, None) => return,
            };
            let (input, _) = input.get(2 * done..).unwrap_or_default().as_chunks::<2>();
            if let Some(re) = re.and_then(|r| r.get_mut(done..)) {
                for (o, x) in re.iter_mut().zip(input) {
                    *o = x[0];
                }
            }
            if let Some(im) = im.and_then(|i| i.get_mut(done..)) {
                for (o, x) in im.iter_mut().zip(input) {
                    *o = x[1];
                }
            }
        }
    };
}

deinterleave_kernel!(deinterleave_f32, f32s, f32, deint_f32_level);
deinterleave_kernel!(deinterleave_i16, i16s, i16, deint_i16_level);

dispatch_fn!(#[cfg_attr(not(target_arch = "wasm32"), inline)] fn run_deinterleave_f32(input: &[f32], re: Option<&mut [f32]>, im: Option<&mut [f32]>) => deinterleave_f32);
dispatch_fn!(#[cfg_attr(not(target_arch = "wasm32"), inline)] fn run_deinterleave_i16(input: &[i16], re: Option<&mut [i16]>, im: Option<&mut [i16]>) => deinterleave_i16);

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

// The entries whose whole call runs in the dispatched function are `#[inline]`: inlined into
// the caller, the call to the dispatched function takes its six arguments in registers and
// no frame of the entry is built around it (D322). Not on wasm32, where the engine counts the
// inlined dispatch as more instructions per call (`simd128` fuel).

impl Add<C32> for Simd {
    #[cfg_attr(not(target_arch = "wasm32"), inline)]
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
    #[cfg_attr(not(target_arch = "wasm32"), inline)]
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
    #[cfg_attr(not(target_arch = "wasm32"), inline)]
    fn add_real(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        check3(a.len(), b.len(), out.len())?;
        run_add_f32(a, b, out);
        Ok(())
    }
}

impl AddReal<Q15> for Simd {
    #[cfg_attr(not(target_arch = "wasm32"), inline)]
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
    #[cfg_attr(not(target_arch = "wasm32"), inline)]
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
    #[cfg_attr(not(target_arch = "wasm32"), inline)]
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
    #[cfg_attr(not(target_arch = "wasm32"), inline)]
    fn interleave(&self, re: &[f32], im: Option<&[f32]>, out: &mut [C32]) -> Result<()> {
        check_len(re.len(), out.len())?;
        if let Some(im) = im {
            check_len(re.len(), im.len())?;
        }
        run_interleave_f32(re, im, bytemuck::cast_slice_mut(out));
        Ok(())
    }

    #[cfg_attr(not(target_arch = "wasm32"), inline)]
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
        run_deinterleave_f32(bytemuck::cast_slice(input), re, im);
        Ok(())
    }
}

impl Interleave<CQ15> for Simd {
    #[cfg_attr(not(target_arch = "wasm32"), inline)]
    fn interleave(&self, re: &[Q15], im: Option<&[Q15]>, out: &mut [CQ15]) -> Result<()> {
        check_len(re.len(), out.len())?;
        if let Some(im) = im {
            check_len(re.len(), im.len())?;
        }
        run_interleave_i16(
            bytemuck::cast_slice(re),
            im.map(bytemuck::cast_slice),
            bytemuck::cast_slice_mut(out),
        );
        Ok(())
    }

    #[cfg_attr(not(target_arch = "wasm32"), inline)]
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
        run_deinterleave_i16(
            bytemuck::cast_slice(input),
            re.map(bytemuck::cast_slice_mut),
            im.map(bytemuck::cast_slice_mut),
        );
        Ok(())
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
    match osc_f32_level(simd, phase, step, out) {
        Some(n) => *done = n,
        None => osc_f32(simd, phase, step, out, done),
    }
}

/// The level-specific path of the `C32` oscillator for the current level: the number of
/// samples done, `None` where the level has no path (the generic code runs).
#[inline(always)]
fn osc_f32_level<S: SimdToken>(simd: S, phase: u32, step: u32, out: &mut [f32]) -> Option<usize> {
    let level = simd.level();
    #[cfg(all(feature = "nco-tables", target_arch = "x86_64"))]
    {
        if let Some(t) = level.as_avx512() {
            return Some(osc_x86::f32_avx512(t, phase, step, out));
        }
        if let Some(t) = level.as_avx2() {
            return Some(osc_x86::f32_avx2(t, phase, step, out));
        }
        if let Some(t) = level.as_sse4_2() {
            return Some(osc_x86::f32_sse4_2(t, phase, step, out));
        }
        if let Some(t) = level.as_sse2() {
            return Some(osc_x86::f32_sse2(t, phase, step, out));
        }
    }
    #[cfg(all(feature = "nco-tables", target_arch = "aarch64"))]
    if let Some(t) = level.as_neon() {
        return Some(osc_neon::f32(t, phase, step, out));
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return Some(osc_wasm::f32(t, phase, step, out));
    }
    let _ = (level, phase, step, out);
    None
}

#[inline(always)]
fn osc_q15_entry<S: SimdToken>(simd: S, phase: u32, step: u32, out: &mut [i16], done: &mut usize) {
    match osc_q15_level(simd, phase, step, out) {
        Some(n) => *done = n,
        None => osc_q15(simd, phase, step, out, done),
    }
}

/// The level-specific path of the `CQ15` oscillator for the current level (SIMD128 only): the
/// number of samples done, `None` where the level has no path (the generic code runs).
#[inline(always)]
fn osc_q15_level<S: SimdToken>(simd: S, phase: u32, step: u32, out: &mut [i16]) -> Option<usize> {
    let level = simd.level();
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return Some(osc_wasm::q15(t, phase, step, out));
    }
    let _ = (level, phase, step, out);
    None
}

/// The level-specific paths of the `C32` oscillator: the vector lookups of `table.rs` that
/// `fm_mod` uses (bit-exact with the fallback), on the phases `phase + n step` kept in a
/// register and advanced by one vector add per vector.
#[cfg(all(feature = "nco-tables", target_arch = "x86_64"))]
mod osc_x86 {
    use core::arch::x86_64::{
        __m128, __m128i, __m256, __m256i, _mm_add_epi32, _mm_set1_epi32, _mm_setr_epi32,
        _mm256_add_epi32, _mm256_add_ps, _mm256_and_si256, _mm256_castpd_ps, _mm256_castps_pd,
        _mm256_castps128_ps256, _mm256_cvtepi32_ps, _mm256_extract_epi32, _mm256_insertf128_ps,
        _mm256_mul_ps, _mm256_permutevar8x32_ps, _mm256_set1_epi32, _mm256_setr_epi32,
        _mm256_srli_epi32, _mm256_unpackhi_pd, _mm256_unpacklo_pd, _mm512_add_epi32,
        _mm512_set1_epi32, _mm512_setr_epi32,
    };

    use fearless_simd::{Simd as SimdToken, f32x4, f32x8, f32x16, prelude::*};

    use crate::table::entry_f32_at;
    use crate::table::x86::{cos_sin_f32_x4, cos_sin_f32_x16};

    /// The phases `phase + k step` of the lanes `k = 0..4` and the step of one vector.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn ramp_x4(phase: u32, step: u32) -> (__m128i, __m128i) {
        let s = |k: u32| phase.wrapping_add(k.wrapping_mul(step)) as i32;
        (
            _mm_setr_epi32(s(0), s(1), s(2), s(3)),
            _mm_set1_epi32(step.wrapping_mul(4) as i32),
        )
    }

    /// The oscillator over vectors of four samples (SSE2 and SSE4.2): the number of samples
    /// done.
    macro_rules! f32_x4 {
        ($t:ident, $phase:ident, $step:ident, $out:ident) => {{
            let (mut p, inc) = ramp_x4($phase, $step);
            let (vectors, _) = $out.as_chunks_mut::<8>();
            for y in vectors.iter_mut() {
                let (o0, o1) = cos_sin_f32_x4($t, p);
                let (d0, d1) = y.split_at_mut(4);
                f32x4::simd_from($t, o0).store_slice(d0);
                f32x4::simd_from($t, o1).store_slice(d1);
                p = _mm_add_epi32(p, inc);
            }
            vectors.len() * 4
        }};
    }

    fearless_simd::kernel!(
        pub(super) fn f32_sse2(t: Sse2, phase: u32, step: u32, out: &mut [f32]) -> usize {
            f32_x4!(t, phase, step, out)
        }
    );

    fearless_simd::kernel!(
        pub(super) fn f32_sse4_2(t: Sse4_2, phase: u32, step: u32, out: &mut [f32]) -> usize {
            f32_x4!(t, phase, step, out)
        }
    );

    /// The f32 table entry at the byte offset `o` as one vector.
    #[inline(always)]
    fn load_f32<S: SimdToken>(simd: S, o: i32) -> __m128 {
        f32x4::simd_from(simd, entry_f32_at(o as u32)).into()
    }

    /// `[lo | hi]` of two 128-bit vectors.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn pair(lo: __m128, hi: __m128) -> __m256 {
        _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(lo), hi)
    }

    /// Interleaved `(cos, sin)` of eight phases (samples 0 to 3, samples 4 to 7): the lookups
    /// of `cos_sin_f32_x4` with the interpolation on 256-bit vectors. The entries of samples
    /// `j` and `j + 2` share a vector, so that the in-lane unpacks of two such vectors give
    /// the values and the differences of four samples in output order.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn cos_sin_f32_x8<S: SimdToken>(simd: S, p: __m256i) -> (__m256, __m256) {
        let f = _mm256_cvtepi32_ps(_mm256_and_si256(p, _mm256_set1_epi32(0xF_FFFF)));
        let o = _mm256_and_si256(_mm256_srli_epi32::<16>(p), _mm256_set1_epi32(0xFFF0));
        let e = |k: i32| load_f32(simd, k);
        let (e0, e1) = (
            e(_mm256_extract_epi32::<0>(o)),
            e(_mm256_extract_epi32::<1>(o)),
        );
        let (e2, e3) = (
            e(_mm256_extract_epi32::<2>(o)),
            e(_mm256_extract_epi32::<3>(o)),
        );
        let (e4, e5) = (
            e(_mm256_extract_epi32::<4>(o)),
            e(_mm256_extract_epi32::<5>(o)),
        );
        let (e6, e7) = (
            e(_mm256_extract_epi32::<6>(o)),
            e(_mm256_extract_epi32::<7>(o)),
        );
        let interp = |x: __m256, y: __m256, idx: __m256i| {
            let (x, y) = (_mm256_castps_pd(x), _mm256_castps_pd(y));
            let a = _mm256_castpd_ps(_mm256_unpacklo_pd(x, y));
            let d = _mm256_castpd_ps(_mm256_unpackhi_pd(x, y));
            _mm256_add_ps(a, _mm256_mul_ps(d, _mm256_permutevar8x32_ps(f, idx)))
        };
        (
            interp(
                pair(e0, e2),
                pair(e1, e3),
                _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3),
            ),
            interp(
                pair(e4, e6),
                pair(e5, e7),
                _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7),
            ),
        )
    }

    fearless_simd::kernel!(
        pub(super) fn f32_avx2(t: Avx2, phase: u32, step: u32, out: &mut [f32]) -> usize {
            let s = |k: u32| phase.wrapping_add(k.wrapping_mul(step)) as i32;
            let mut p = _mm256_setr_epi32(s(0), s(1), s(2), s(3), s(4), s(5), s(6), s(7));
            let inc = _mm256_set1_epi32(step.wrapping_mul(8) as i32);
            let (vectors, _) = out.as_chunks_mut::<16>();
            for y in vectors.iter_mut() {
                let (o0, o1) = cos_sin_f32_x8(t, p);
                let (d0, d1) = y.split_at_mut(8);
                f32x8::simd_from(t, o0).store_slice(d0);
                f32x8::simd_from(t, o1).store_slice(d1);
                p = _mm256_add_epi32(p, inc);
            }
            vectors.len() * 8
        }
    );

    fearless_simd::kernel!(
        pub(super) fn f32_avx512(t: Avx512, phase: u32, step: u32, out: &mut [f32]) -> usize {
            let s = |k: u32| phase.wrapping_add(k.wrapping_mul(step)) as i32;
            let mut p = _mm512_setr_epi32(
                s(0),
                s(1),
                s(2),
                s(3),
                s(4),
                s(5),
                s(6),
                s(7),
                s(8),
                s(9),
                s(10),
                s(11),
                s(12),
                s(13),
                s(14),
                s(15),
            );
            let inc = _mm512_set1_epi32(step.wrapping_mul(16) as i32);
            let (vectors, _) = out.as_chunks_mut::<32>();
            for y in vectors.iter_mut() {
                let (o0, o1) = cos_sin_f32_x16(p);
                let (d0, d1) = y.split_at_mut(16);
                f32x16::simd_from(t, o0).store_slice(d0);
                f32x16::simd_from(t, o1).store_slice(d1);
                p = _mm512_add_epi32(p, inc);
            }
            vectors.len() * 16
        }
    );
}

/// The NEON path of the `C32` oscillator (see [`osc_x86`]).
#[cfg(all(feature = "nco-tables", target_arch = "aarch64"))]
mod osc_neon {
    use core::arch::aarch64::{uint32x4_t, vaddq_u32, vdupq_n_u32};

    use fearless_simd::{f32x4, prelude::*, u32x4};

    use crate::table::neon::cos_sin_f32_x4;

    fearless_simd::kernel!(
        pub(super) fn f32(t: Neon, phase: u32, step: u32, out: &mut [f32]) -> usize {
            let s = |k: u32| phase.wrapping_add(k.wrapping_mul(step));
            let mut p: uint32x4_t = u32x4::simd_from(t, [s(0), s(1), s(2), s(3)]).into();
            let inc = vdupq_n_u32(step.wrapping_mul(4));
            let (vectors, _) = out.as_chunks_mut::<8>();
            for y in vectors.iter_mut() {
                let (o0, o1) = cos_sin_f32_x4(t, p);
                let (d0, d1) = y.split_at_mut(4);
                f32x4::simd_from(t, o0).store_slice(d0);
                f32x4::simd_from(t, o1).store_slice(d1);
                p = vaddq_u32(p, inc);
            }
            vectors.len() * 4
        }
    );
}

/// The SIMD128 paths of the oscillator (see [`osc_x86`]): `C32` and `CQ15`, four samples per
/// step on the phases kept in a register.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod osc_wasm {
    use core::arch::wasm32::{i32x4_add, u32x4, u32x4_splat};

    use fearless_simd::{f32x4, i16x8, prelude::*};

    use crate::table::wasm::{cos_sin_f32_x4, cos_sin_q15_x4};

    fearless_simd::kernel!(
        pub(super) fn f32(t: WasmSimd128, phase: u32, step: u32, out: &mut [f32]) -> usize {
            let s = |k: u32| phase.wrapping_add(k.wrapping_mul(step));
            let mut p = u32x4(s(0), s(1), s(2), s(3));
            let inc = u32x4_splat(step.wrapping_mul(4));
            let (vectors, _) = out.as_chunks_mut::<8>();
            for y in vectors.iter_mut() {
                #[cfg(feature = "nco-tables")]
                let (o0, o1) = cos_sin_f32_x4(t, p);
                #[cfg(not(feature = "nco-tables"))]
                let (o0, o1) = cos_sin_f32_x4(p);
                let (d0, d1) = y.split_at_mut(4);
                f32x4::simd_from(t, o0).store_slice(d0);
                f32x4::simd_from(t, o1).store_slice(d1);
                p = i32x4_add(p, inc);
            }
            vectors.len() * 4
        }
    );

    fearless_simd::kernel!(
        pub(super) fn q15(t: WasmSimd128, phase: u32, step: u32, out: &mut [i16]) -> usize {
            let s = |k: u32| phase.wrapping_add(k.wrapping_mul(step));
            let mut p = u32x4(s(0), s(1), s(2), s(3));
            let inc = u32x4_splat(step.wrapping_mul(4));
            let (vectors, _) = out.as_chunks_mut::<8>();
            for y in vectors.iter_mut() {
                i16x8::simd_from(t, cos_sin_q15_x4(p)).store_slice(y);
                p = i32x4_add(p, inc);
            }
            vectors.len() * 4
        }
    );
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
