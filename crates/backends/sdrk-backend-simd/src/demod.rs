//! `quad_demod`: the FM discriminator `arg(x[n] * conj(x[n-1]))`.
//!
//! The argument is computed with a vectorised `atan2`: octant reduction to `[0, 1]`, a second
//! reduction to `[0, tan(pi/8)]`, then the Cephes `atanf` polynomial. The maximum error of the
//! approximation is 1e-7 rad (about 130 dB below a full-scale argument), well inside the
//! contract of 100 dB (`C32`) and 70 dB (`CQ15`). Each output depends only on the current and
//! the previous input sample, so the result is independent of how the input is split into
//! blocks.

use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::demod::{QuadDemod, QuadDemodState};
use sdrk_core::{C32, CQ15, Q15, Result};

use crate::Simd;
use crate::level::dispatch;
use crate::util::check_len;

/// `tan(pi / 8)`.
const TAN_PI_8: f32 = 0.414_213_57;
/// `3 pi / 4`.
const THREE_PI_4: f32 = 2.356_194_5;

/// `atan2(y, x)` in radians; 0 where both are zero. Maximum error about 1e-7.
///
/// The octant of the result is folded into a single addition `K + S * v` (`v` is the arc tangent
/// of the reduced argument, `K` a multiple of pi/4 chosen by the octant, `S` = +-1), so that the
/// result carries one rounding for the octant instead of one per reflection.
#[inline(always)]
fn atan2<S: SimdToken>(simd: S, y: S::f32s, x: S::f32s) -> S::f32s {
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

/// Coefficients `c0 .. c6` of `atan(t) = t * (c0 + c1 z + ... + c6 z^6)`, `z = t^2`, on
/// `[0, tan(pi/8)]`: the interpolating polynomial at the Chebyshev nodes, maximum error 8e-12.
const ATAN_F64: [f64; 7] = [
    0.999_999_999_978_399,
    -0.333_333_320_976_102_3,
    0.199_998_838_567_622_25,
    -0.142_815_887_760_021_27,
    0.110_404_892_674_962_96,
    -0.084_561_930_770_772_16,
    0.047_073_484_922_127_116,
];

/// `atan2(y, x)` in `f64`, correctly rounded to better than 1e-11; 0 where both are zero.
#[inline(always)]
fn atan2_f64<S: SimdToken>(simd: S, y: S::f64s, x: S::f64s) -> S::f64s {
    use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    let zero = S::f64s::splat(simd, 0.0);
    let one = S::f64s::splat(simd, 1.0);
    let ax = x.abs();
    let ay = y.abs();
    let hi = ax.max(ay);
    let lo = ax.min(ay);
    let both_zero = hi.simd_eq(zero);
    let a = lo / both_zero.select(one, hi);
    let big = a.simd_gt(0.414_213_562_373_095_03f64);
    let t = big.select(a - 1.0, a) / big.select(a + 1.0, one);
    let z = t * t;
    let mut p = S::f64s::splat(simd, ATAN_F64[6]);
    for c in [
        ATAN_F64[5],
        ATAN_F64[4],
        ATAN_F64[3],
        ATAN_F64[2],
        ATAN_F64[1],
        ATAN_F64[0],
    ] {
        p = p * z + c;
    }
    let v = p * t;
    let swap = ay.simd_gt(ax);
    let neg = x.is_sign_negative();
    let k_small = swap.select(
        S::f64s::splat(simd, FRAC_PI_2),
        neg.select(S::f64s::splat(simd, PI), zero),
    );
    let k_big = neg.select(
        S::f64s::splat(simd, 3.0 * FRAC_PI_4),
        S::f64s::splat(simd, FRAC_PI_4),
    );
    let k = big.select(k_big, k_small);
    let r = (k + (swap ^ neg).select(-v, v)).copysign(y);
    both_zero.select(zero, r)
}

/// `atan2` of a vector of `f32` pairs through `f64`, for the levels with 4 lanes (SSE2, NEON,
/// SIMD128), which have no spare instructions to lose against the libm-quality references.
#[inline(always)]
fn atan2_via_f64<S: SimdToken>(simd: S, y: S::f32s, x: S::f32s) -> S::f32s {
    let (y0, y1) = y.widen();
    let (x0, x1) = x.widen();
    let r0 = atan2_f64::<S>(simd, y0, x0);
    let r1 = atan2_f64::<S>(simd, y1, x1);
    r0.narrow(r1)
}

/// The argument of `(re, im)`: through `f64` on the 4-lane levels, in `f32` on the wider ones.
#[inline(always)]
fn argument<S: SimdToken>(simd: S, im: S::f32s, re: S::f32s) -> S::f32s {
    if S::f32s::LEN <= 4 {
        atan2_via_f64::<S>(simd, im, re)
    } else {
        atan2(simd, im, re)
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

/// One vector of discriminator outputs for the `2 h` floats `a ++ b` (`h` interleaved samples);
/// `prev_r` and `prev_i` carry the last samples between calls.
#[inline(always)]
fn demod_vector<S: SimdToken>(
    simd: S,
    a: &[f32],
    b: &[f32],
    prev_r: &mut S::f32s,
    prev_i: &mut S::f32s,
    gain: f32,
) -> S::f32s {
    let (xr, xi) = S::f32s::from_slice(simd, a).deinterleave(S::f32s::from_slice(simd, b));
    let pr = slide_last(*prev_r, xr);
    let pi = slide_last(*prev_i, xi);
    let re = xr * pr + xi * pi;
    let im = xi * pr - xr * pi;
    *prev_r = xr;
    *prev_i = xi;
    argument::<S>(simd, im, re) * gain
}

/// The argument of `x[n] * conj(x[n-1])`, scaled by `gain`, for all samples of `input`. The last
/// partial vector is processed zero-padded, so every sample goes through the same arithmetic
/// whatever the block size.
#[inline(always)]
fn quad_demod_f32<S: SimdToken>(simd: S, prev: C32, gain: f32, input: &[f32], out: &mut [f32]) {
    let h = S::f32s::LEN;
    let mut prev_r = S::f32s::splat(simd, prev.re);
    let mut prev_i = S::f32s::splat(simd, prev.im);
    let mut in_chunks = input.chunks_exact(2 * h);
    let mut out_chunks = out.chunks_exact_mut(h);
    for (i, o) in (&mut in_chunks).zip(&mut out_chunks) {
        let (a, b) = i.split_at(h);
        demod_vector::<S>(simd, a, b, &mut prev_r, &mut prev_i, gain).store_slice(o);
    }
    let (tail_in, tail_out) = (in_chunks.remainder(), out_chunks.into_remainder());
    if !tail_in.is_empty() {
        let mut tmp_in = [0f32; 32];
        let mut tmp_out = [0f32; 16];
        if let Some(dst) = tmp_in.get_mut(..tail_in.len()) {
            dst.copy_from_slice(tail_in);
        }
        if let (Some(i), Some(o)) = (tmp_in.get(..2 * h), tmp_out.get_mut(..h)) {
            let (a, b) = i.split_at(h);
            demod_vector::<S>(simd, a, b, &mut prev_r, &mut prev_i, gain).store_slice(o);
        }
        for (d, s) in tail_out.iter_mut().zip(&tmp_out) {
            *d = *s;
        }
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
    let arg = atan2(simd, im, re);
    // `phi = atan2 / pi` in units of 2^-15, rounded half up, saturated.
    let phi = (arg * c.k + 0.5)
        .floor()
        .max(c.lo)
        .min(c.hi)
        .to_int::<S::i32s>();
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
