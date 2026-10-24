//! `quad_demod`: the FM discriminator `arg(x[n] * conj(x[n-1]))`.
//!
//! The argument is computed with a vectorised `atan2`: octant reduction to `[0, 1]`, a second
//! reduction to `[0, tan(pi/8)]`, then the Cephes `atanf` polynomial. The maximum error of the
//! approximation is 1e-7 rad (about 130 dB below a full-scale argument), well inside the
//! contract of 100 dB (`C32`) and 70 dB (`CQ15`). Each output depends only on the current and
//! the previous input sample, so the result is independent of how the input is split into
//! blocks.
//!
//! `C32` reads the previous samples from the input itself, one sample back, so that no vector
//! carries state from one iteration to the next. On SSE4.2 and AVX2 the products stay
//! interleaved and the horizontal add and subtract (`haddps`, `hsubps`) form the real and the
//! imaginary part; the other levels deinterleave both streams.

use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::demod::{QuadDemod, QuadDemodState};
use sdrk_core::{C32, CQ15, Q15, Result};

use crate::Simd;
use crate::level::{dispatch, is_sse2_baseline};
use crate::util::check_len;

/// `tan(pi / 8)`.
const TAN_PI_8: f32 = 0.414_213_57;
/// `3 pi / 4`.
const THREE_PI_4: f32 = 2.356_194_5;
/// `pi / 4` with the low 3 bits of the significand cleared, so that `m * PI4_HI` is exact for
/// every `m` in `0..=4`.
const PI4_HI: f32 = f32::from_bits(FRAC_PI_4.to_bits() & !7);
/// `pi / 4 - PI4_HI`.
const PI4_LO: f32 = (core::f64::consts::FRAC_PI_4 - PI4_HI as f64) as f32;
/// `PI4_LO / PI4_HI`: `m PI4_LO` from `m PI4_HI`.
const LO_PER_HI: f32 = (PI4_LO as f64 / PI4_HI as f64) as f32;
/// The most lanes of an `f32` vector (AVX-512).
const MAX_LANES: usize = 16;

/// `atan2(y, x)` in radians for `C32`; 0 where both are zero.
///
/// One division: `t = lo / hi` with `lo = min(|x|, |y|)` and `hi = max(|x|, |y|)`, or
/// `t = (lo - hi) / (lo + hi)` where `lo / hi` is above `tan(pi/8)` (then
/// `atan(lo / hi) = pi/4 + atan(t)`). The result is `m pi/4 +- atan(t)` with `m` in `0..=4`
/// chosen by the octant; the offset is added in two parts (`PI4_HI`, `PI4_LO`, Cody and Waite),
/// so that it carries no rounding of its own. Against the f64 reference this is within 1 dB of
/// a correctly rounded `atan2f` of the same `f32` products.
#[inline(always)]
fn atan2_c32<S: SimdToken>(simd: S, y: S::f32s, x: S::f32s) -> S::f32s {
    let zero = S::f32s::splat(simd, 0.0);
    let sign = S::u32s::splat(simd, 0x8000_0000);
    // `v` where the mask is set, else 0, and `-v` where it is set, else `v`: bitwise with the
    // mask, except with the mask registers of AVX-512 (16 lanes), where a masked move or `xor`
    // is one instruction.
    let and = |m: S::mask32s, v: S::f32s| {
        if S::f32s::LEN == 16 {
            m.select(v, zero)
        } else {
            (m.to_vector().bitcast::<S::u32s>() & v.bitcast::<S::u32s>()).bitcast::<S::f32s>()
        }
    };
    let negate_where = |m: S::mask32s, v: S::f32s| {
        if S::f32s::LEN == 16 {
            m.select(-v, v)
        } else {
            let flip = m.to_vector().bitcast::<S::u32s>() & sign;
            (v.bitcast::<S::u32s>() ^ flip).bitcast::<S::f32s>()
        }
    };
    let ax = x.abs();
    let ay = y.abs();
    // The smallest subnormal as the least `hi` turns 0 / 0 into 0 / 2^-149 = 0 (then `m` is 0,
    // and the result 0), and changes nothing for any other input.
    let hi = ax.max(ay).max(S::f32s::splat(simd, f32::from_bits(1)));
    let lo = ax.min(ay);
    let big = lo.simd_gt(hi * TAN_PI_8);
    // `lo - hi` and `lo + hi` where `big`, else `lo` and `hi`.
    let t = (lo - and(big, hi)) / (hi + and(big, lo));
    let z = t * t;
    let p = ((z * 8.053_744_5e-2 - 1.387_768_6e-1) * z + 1.997_771_1e-1) * z - 3.333_295e-1;
    // `swap` (|y| > |x|) reflects at pi/4, `neg` (x < 0, not -0, so that -0 / -0 gives 0) at
    // pi/2. `k = m PI4_HI`; in the first quadrant `m` is 0 or 1 below pi/4 (`big`) and 1 or 2
    // above it, and `m PI4_LO = k LO_PER_HI` (rounded, on a term below 1e-7).
    let swap = ay.simd_gt(ax);
    let neg = x.simd_lt(zero);
    let k = big.select(
        S::f32s::splat(simd, PI4_HI),
        and(swap, S::f32s::splat(simd, 2.0 * PI4_HI)),
    );
    let k = neg.select(S::f32s::splat(simd, 4.0 * PI4_HI) - k, k);
    // `atan(t) = t + t z p(z)`, with the sign of the reflections; the small terms are summed
    // first and `t` added to them, which rounds less than adding `PI4_LO` to `atan(t)`.
    let st = negate_where(swap ^ neg, t);
    let r = k + (st + (p * z * st + k * LO_PER_HI));
    // `r` is at least 0, so the sign of `y` is or-ed in.
    (r.bitcast::<S::u32s>() | (y.bitcast::<S::u32s>() & sign)).bitcast::<S::f32s>()
}

/// `atan2(y, x)` in radians for `CQ15`; 0 where both are zero. Maximum error about 1e-7.
///
/// The octant of the result is folded into a single addition `K + S * v` (`v` is the arc tangent
/// of the reduced argument, `K` a multiple of pi/4 chosen by the octant, `S` = +-1), so that the
/// result carries one rounding for the octant instead of one per reflection.
#[inline(always)]
fn atan2_q15<S: SimdToken>(simd: S, y: S::f32s, x: S::f32s) -> S::f32s {
    let zero = S::f32s::splat(simd, 0.0);
    let one = S::f32s::splat(simd, 1.0);
    let ax = x.abs();
    let ay = y.abs();
    let hi = ax.max(ay);
    let lo = ax.min(ay);
    let both_zero = hi.simd_eq(zero);
    // `a` in [0, 1]; the denominator is made 1 where both inputs are zero.
    let a = lo / both_zero.select(one, hi);
    // Second reduction: atan(a) = pi/4 + atan((a - 1) / (a + 1)) for a above tan(pi/8).
    let big = a.simd_gt(TAN_PI_8);
    let num = big.select(a - 1.0, a);
    let den = big.select(a + 1.0, one);
    let t = num / den;
    let z = t * t;
    let p = ((z * 8.053_744_5e-2 - 1.387_768_6e-1) * z + 1.997_771_1e-1) * z - 3.333_295e-1;
    let v = p * z * t + t;
    // Octant: `swap` (|y| > |x|) reflects at pi/4, `neg` (x < 0) at pi/2.
    let swap = ay.simd_gt(ax);
    let neg = x.is_sign_negative();
    let half_pi = S::f32s::splat(simd, FRAC_PI_2);
    let k_small = swap.select(half_pi, neg.select(S::f32s::splat(simd, PI), zero));
    let k_big = neg.select(
        S::f32s::splat(simd, THREE_PI_4),
        S::f32s::splat(simd, FRAC_PI_4),
    );
    let k = big.select(k_big, k_small);
    let sv = (swap ^ neg).select(-v, v);
    let r = (k + sv).copysign(y);
    both_zero.select(zero, r)
}

/// `gain * arg(re + j im)` for one vector.
#[inline(always)]
fn discriminator<S: SimdToken>(simd: S, re: S::f32s, im: S::f32s, gain: f32) -> S::f32s {
    atan2_c32(simd, im, re) * gain
}

/// `out[k] = gain * arg(x[k] * conj(p[k]))` for the interleaved samples `x` and their
/// predecessors `p`, over the whole vectors of `out` (`x` and `p` hold two floats per output).
#[inline(always)]
fn discriminate_vectors<S: SimdToken>(simd: S, x: &[f32], p: &[f32], out: &mut [f32], gain: f32) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let level = simd.level();
        if level.as_avx512().is_none() {
            if let Some(t) = level.as_avx2() {
                x86::discriminate_avx2(t, x, p, out, gain);
                return;
            }
            if let Some(t) = level.as_sse4_2() {
                x86::discriminate_sse4_2(t, x, p, out, gain);
                return;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = simd.level().as_neon() {
        neon::discriminate_neon(t, x, p, out, gain);
        return;
    }
    discriminate_generic(simd, x, p, out, gain);
}

/// The portable code of [`discriminate_vectors`], and its only path on SSE2, AVX-512 and
/// SIMD128.
#[inline(always)]
fn discriminate_generic<S: SimdToken>(simd: S, x: &[f32], p: &[f32], out: &mut [f32], gain: f32) {
    let h = S::f32s::LEN;
    for ((x, p), o) in x
        .chunks_exact(2 * h)
        .zip(p.chunks_exact(2 * h))
        .zip(out.chunks_exact_mut(h))
    {
        let ((x0, x1), (p0, p1)) = (x.split_at(h), p.split_at(h));
        let (xr, xi) = S::f32s::from_slice(simd, x0).deinterleave(S::f32s::from_slice(simd, x1));
        let (pr, pi) = S::f32s::from_slice(simd, p0).deinterleave(S::f32s::from_slice(simd, p1));
        let (re, im) = (xr * pr + xi * pi, xi * pr - xr * pi);
        discriminator(simd, re, im, gain).store_slice(o);
    }
}

/// The paths of SSE4.2 and AVX2: the products stay interleaved, `x * p` gives
/// `(xr pr, xi pi)` and `swap(x) * p` gives `(xi pr, xr pi)` per sample, and the horizontal sum
/// and difference of the pairs are `re` and `im`, in the order of operations of the fallback.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m128, __m256, _mm_hadd_ps, _mm_hsub_ps, _mm_mul_ps, _mm_shuffle_ps, _mm256_castpd_ps,
        _mm256_castps_pd, _mm256_hadd_ps, _mm256_hsub_ps, _mm256_mul_ps, _mm256_permute_ps,
        _mm256_permute4x64_pd,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128, __m256, _mm_hadd_ps, _mm_hsub_ps, _mm_mul_ps, _mm_shuffle_ps, _mm256_castpd_ps,
        _mm256_castps_pd, _mm256_hadd_ps, _mm256_hsub_ps, _mm256_mul_ps, _mm256_permute_ps,
        _mm256_permute4x64_pd,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4, f32x8};

    use super::discriminator;

    /// Lane order `1, 0, 3, 2`: the two floats of every sample swapped.
    const SWAP: i32 = 0b10_11_00_01;

    fearless_simd::kernel!(
        pub(super) fn discriminate_avx2(t: Avx2, x: &[f32], p: &[f32], out: &mut [f32], gain: f32) {
            for ((x, p), o) in x
                .chunks_exact(16)
                .zip(p.chunks_exact(16))
                .zip(out.chunks_exact_mut(8))
            {
                let ((x0, x1), (p0, p1)) = (x.split_at(8), p.split_at(8));
                let x0: __m256 = f32x8::from_slice(t, x0).into();
                let x1: __m256 = f32x8::from_slice(t, x1).into();
                let p0: __m256 = f32x8::from_slice(t, p0).into();
                let p1: __m256 = f32x8::from_slice(t, p1).into();
                // Within each 128-bit half: samples 0, 1 of `x0`, then 0, 1 of `x1` (low
                // half), 2, 3 of `x0`, then 2, 3 of `x1` (high half).
                let re = _mm256_hadd_ps(_mm256_mul_ps(x0, p0), _mm256_mul_ps(x1, p1));
                let im = _mm256_hsub_ps(
                    _mm256_mul_ps(_mm256_permute_ps::<SWAP>(x0), p0),
                    _mm256_mul_ps(_mm256_permute_ps::<SWAP>(x1), p1),
                );
                let r: __m256 = discriminator(t, re.simd_into(t), im.simd_into(t), gain).into();
                // Back to sample order: swap the middle two 64-bit quarters.
                let r =
                    _mm256_castpd_ps(_mm256_permute4x64_pd::<0b11_01_10_00>(_mm256_castps_pd(r)));
                f32x8::simd_from(t, r).store_slice(o);
            }
        }
    );

    fearless_simd::kernel!(
        pub(super) fn discriminate_sse4_2(
            t: Sse4_2,
            x: &[f32],
            p: &[f32],
            out: &mut [f32],
            gain: f32,
        ) {
            for ((x, p), o) in x
                .chunks_exact(8)
                .zip(p.chunks_exact(8))
                .zip(out.chunks_exact_mut(4))
            {
                let ((x0, x1), (p0, p1)) = (x.split_at(4), p.split_at(4));
                let x0: __m128 = f32x4::from_slice(t, x0).into();
                let x1: __m128 = f32x4::from_slice(t, x1).into();
                let p0: __m128 = f32x4::from_slice(t, p0).into();
                let p1: __m128 = f32x4::from_slice(t, p1).into();
                let re = _mm_hadd_ps(_mm_mul_ps(x0, p0), _mm_mul_ps(x1, p1));
                let im = _mm_hsub_ps(
                    _mm_mul_ps(_mm_shuffle_ps::<SWAP>(x0, x0), p0),
                    _mm_mul_ps(_mm_shuffle_ps::<SWAP>(x1, x1), p1),
                );
                discriminator(t, re.simd_into(t), im.simd_into(t), gain).store_slice(o);
            }
        }
    );
}

/// The path of NEON: `ld2` loads and deinterleaves two vectors in one instruction.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon {
    use core::arch::aarch64::{float32x4x2_t, vld2q_f32};
    use fearless_simd::f32x4;
    use fearless_simd::prelude::*;

    use super::discriminator;

    fearless_simd::kernel!(
        pub(super) fn discriminate_neon(t: Neon, x: &[f32], p: &[f32], out: &mut [f32], gain: f32) {
            for ((x, p), o) in x
                .chunks_exact(8)
                .zip(p.chunks_exact(8))
                .zip(out.chunks_exact_mut(4))
            {
                // SAFETY: `vld2q_f32` reads 8 floats from the pointer, and `x` and `p` are
                // chunks of exactly 8 floats.
                let (float32x4x2_t(xr, xi), float32x4x2_t(pr, pi)) =
                    unsafe { (vld2q_f32(x.as_ptr()), vld2q_f32(p.as_ptr())) };
                let (xr, xi) = (f32x4::simd_from(t, xr), f32x4::simd_from(t, xi));
                let (pr, pi) = (f32x4::simd_from(t, pr), f32x4::simd_from(t, pi));
                let (re, im) = (xr * pr + xi * pi, xi * pr - xr * pi);
                discriminator(t, re, im, gain).store_slice(o);
            }
        }
    );
}

/// The argument of `x[n] * conj(x[n-1])`, scaled by `gain`, for all samples of `input`
/// (interleaved, two floats per sample). The first vector, whose predecessors include the state
/// `prev`, and the last partial vector run zero-padded from the stack through the same vector
/// code, so every sample goes through the same arithmetic whatever the block size.
#[inline(always)]
fn quad_demod_f32<S: SimdToken>(simd: S, prev: C32, gain: f32, input: &[f32], out: &mut [f32]) {
    let h = S::f32s::LEN;
    let n = out.len().min(input.len() / 2);
    let mut x = [0f32; 2 * MAX_LANES];
    let mut p = [0f32; 2 * MAX_LANES];
    let mut o = [0f32; MAX_LANES];
    // The first vector: `p` is the state followed by the first `h - 1` samples.
    let first = n.min(h);
    if let (Some(dx), Some(sx), Some((p_head, dp)), Some(sp)) = (
        x.get_mut(..2 * first),
        input.get(..2 * first),
        p.get_mut(..2 * first)
            .and_then(|d| d.split_first_chunk_mut::<2>()),
        input.get(..(2 * first).saturating_sub(2)),
    ) {
        dx.copy_from_slice(sx);
        *p_head = [prev.re, prev.im];
        dp.copy_from_slice(sp);
    }
    if first == 0 {
        return;
    }
    if let (Some(vx), Some(vp), Some(vo)) = (x.get(..2 * h), p.get(..2 * h), o.get_mut(..h)) {
        discriminate_vectors(simd, vx, vp, vo, gain);
    }
    if let (Some(d), Some(s)) = (out.get_mut(..first), o.get(..first)) {
        d.copy_from_slice(s);
    }
    // The whole vectors after the first, with `p` one sample behind `x` in the input.
    let whole = n - n % h;
    if whole > h
        && let (Some(vx), Some(vp), Some(vo)) = (
            input.get(2 * h..2 * whole),
            input.get(2 * h - 2..2 * whole - 2),
            out.get_mut(h..whole),
        )
    {
        discriminate_vectors(simd, vx, vp, vo, gain);
    }
    // The last partial vector.
    let rest = n - whole;
    if whole >= h && rest > 0 {
        x = [0f32; 2 * MAX_LANES];
        p = [0f32; 2 * MAX_LANES];
        if let (Some(dx), Some(sx), Some(dp), Some(sp)) = (
            x.get_mut(..2 * rest),
            input.get(2 * whole..2 * n),
            p.get_mut(..2 * rest),
            input.get(2 * whole - 2..2 * n - 2),
        ) {
            dx.copy_from_slice(sx);
            dp.copy_from_slice(sp);
        }
        if let (Some(vx), Some(vp), Some(vo)) = (x.get(..2 * h), p.get(..2 * h), o.get_mut(..h)) {
            discriminate_vectors(simd, vx, vp, vo, gain);
        }
        if let (Some(d), Some(s)) = (out.get_mut(whole..n), o.get(..rest)) {
            d.copy_from_slice(s);
        }
    }
}

/// `[a[LEN-1], b[0], ..., b[LEN-2]]`: `b` shifted by one lane, with the last lane of `a`
/// entering at the front.
#[inline(always)]
fn slide_last<V: SimdBase<S>, S: SimdToken>(a: V, b: V) -> V {
    match V::LEN {
        2 => a.slide::<1>(b),
        4 => a.slide::<3>(b),
        8 => a.slide::<7>(b),
        16 => a.slide::<15>(b),
        _ => b,
    }
}

/// Constants of the Q15 discriminator.
struct Q15Consts<S: SimdToken> {
    gain: S::i32s,
    /// `32768 / pi`: the argument in units of `2^-15` turns of half a circle.
    k: S::f32s,
    lo: S::f32s,
    hi: S::f32s,
}

/// One vector of Q15 discriminator outputs (as `i32` lanes in `[-32768, 32767]`) for the
/// `h` samples packed in `a` (the low half is I, the high half is Q).
#[inline(always)]
fn discriminate<S: SimdToken>(
    simd: S,
    a: &[i16],
    prev_x: &mut S::i32s,
    c: &Q15Consts<S>,
) -> S::i32s {
    let x = S::i16s::from_slice(simd, a).bitcast::<S::i32s>();
    let p = slide_last(*prev_x, x);
    *prev_x = x;
    let (xr, xi) = ((x << 16u32) >> 16u32, x >> 16u32);
    let (pr, pi) = ((p << 16u32) >> 16u32, p >> 16u32);
    let re = (xr * pr).to_float::<S::f32s>() + (xi * pi).to_float::<S::f32s>();
    let im = (xi * pr).to_float::<S::f32s>() - (xr * pi).to_float::<S::f32s>();
    let arg = atan2_q15(simd, im, re);
    // `phi = atan2 / pi` in units of 2^-15, rounded half up, saturated.
    let v = arg * c.k + 0.5;
    let phi = if is_sse2_baseline(simd) {
        // SSE2 has no `roundps`, and `fearless_simd` floors there lane by lane through libm.
        // The bounds are integers, so clamping first gives the same floor; truncating the
        // clamped value and stepping down where that rounded up is exact (`|v| <= 2^23`).
        let v = v.max(c.lo).min(c.hi);
        let t = v.to_int::<S::i32s>();
        t + t.to_float::<S::f32s>().simd_gt(v).to_vector()
    } else {
        v.floor().max(c.lo).min(c.hi).to_int::<S::i32s>()
    };
    // `q15_mul(phi, gain)`: `(phi * gain + 2^14) >> 15`, saturated.
    let m = (phi * c.gain + (1 << 14)) >> 15u32;
    m.max(S::i32s::splat(simd, -32768))
        .min(S::i32s::splat(simd, 32767))
}

/// `CQ15` variant: products in `i32` (exact), the argument in `f32`. Two vectors of `h` samples
/// make one full output vector of `2 h` values; the rest is processed zero-padded.
#[inline(always)]
fn quad_demod_q15<S: SimdToken>(simd: S, prev: CQ15, gain: i16, input: &[i16], out: &mut [i16]) {
    let h = S::i32s::LEN;
    let packed_prev = i32::from(prev.re.0 as u16) | (i32::from(prev.im.0) << 16);
    let mut prev_x = S::i32s::splat(simd, packed_prev);
    let consts = Q15Consts::<S> {
        gain: S::i32s::splat(simd, i32::from(gain)),
        k: S::f32s::splat(simd, 32768.0 / PI),
        lo: S::f32s::splat(simd, -32768.0),
        hi: S::f32s::splat(simd, 32767.0),
    };
    let mut in_chunks = input.chunks_exact(4 * h);
    let mut out_chunks = out.chunks_exact_mut(2 * h);
    for (i, o) in (&mut in_chunks).zip(&mut out_chunks) {
        let (a, b) = i.split_at(2 * h);
        let first = discriminate::<S>(simd, a, &mut prev_x, &consts);
        let second = discriminate::<S>(simd, b, &mut prev_x, &consts);
        first.narrow(second).bitcast::<S::i16s>().store_slice(o);
    }
    let (tail_in, tail_out) = (in_chunks.remainder(), out_chunks.into_remainder());
    if !tail_in.is_empty() {
        let mut tmp_in = [0i16; 64];
        let mut tmp_out = [0i16; 32];
        if let Some(dst) = tmp_in.get_mut(..tail_in.len()) {
            dst.copy_from_slice(tail_in);
        }
        if let (Some(a), Some(b), Some(o)) = (
            tmp_in.get(..2 * h),
            tmp_in.get(2 * h..4 * h),
            tmp_out.get_mut(..2 * h),
        ) {
            let first = discriminate::<S>(simd, a, &mut prev_x, &consts);
            let second = discriminate::<S>(simd, b, &mut prev_x, &consts);
            first.narrow(second).bitcast::<S::i16s>().store_slice(o);
        }
        for (d, s) in tail_out.iter_mut().zip(&tmp_out) {
            *d = *s;
        }
    }
}

impl QuadDemod<C32> for Simd {
    fn quad_demod(
        &self,
        state: &mut QuadDemodState<C32>,
        input: &[C32],
        output: &mut [f32],
        gain: f32,
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let prev = state.prev();
        let x: &[f32] = bytemuck::cast_slice(input);
        dispatch!(simd => quad_demod_f32(simd, prev, gain, x, output));
        if let Some(last) = input.last() {
            state.set_prev(*last);
        }
        Ok(())
    }
}

impl QuadDemod<CQ15> for Simd {
    fn quad_demod(
        &self,
        state: &mut QuadDemodState<CQ15>,
        input: &[CQ15],
        output: &mut [Q15],
        gain: Q15,
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let prev = state.prev();
        let x: &[i16] = bytemuck::cast_slice(input);
        let y: &mut [i16] = bytemuck::cast_slice_mut(output);
        dispatch!(simd => quad_demod_q15(simd, prev, gain.0, x, y));
        if let Some(last) = input.last() {
            state.set_prev(*last);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use fearless_simd::Simd as SimdToken;

    use super::{discriminate_generic, discriminate_vectors};
    use crate::level::dispatch;

    /// The outputs of the path of the level and of the portable code for `n` outputs.
    #[inline(always)]
    fn both<S: SimdToken>(simd: S, x: &[f32], p: &[f32], n: usize) -> ([f32; 64], [f32; 64]) {
        let (mut level, mut generic) = ([0f32; 64], [0f32; 64]);
        discriminate_vectors(simd, x, p, &mut level[..n], -1.75);
        discriminate_generic(simd, x, p, &mut generic[..n], -1.75);
        (level, generic)
    }

    #[test]
    fn the_path_of_the_level_matches_the_portable_code_bit_for_bit() {
        // Noise with zeros and signed zeros in it, at every alignment of the slices.
        let mut state = 7u32;
        let data: [f32; 140] = core::array::from_fn(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            match i % 23 {
                5 => 0.0,
                11 => -0.0,
                _ => (state >> 8) as f32 / 8_388_608.0 - 1.0,
            }
        });
        for offset in 0..6 {
            for n in 0..=64 {
                let x = &data[offset + 2..offset + 2 + 2 * n];
                let p = &data[offset..offset + 2 * n];
                let (level, generic) = dispatch!(simd => both(simd, x, p, n));
                assert_eq!(
                    level.map(f32::to_bits),
                    generic.map(f32::to_bits),
                    "{offset} {n}"
                );
            }
        }
    }
}
