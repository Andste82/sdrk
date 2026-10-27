//! `biquad` `C32` and `CQ15`: the real and the imaginary part are two independent recursions
//! with the same coefficients, run in two lanes of one vector with the operations of the
//! fallback in the same order per lane.
//!
//! `biquad_real` is a single recursion per section (every output depends on the previous output
//! of the same section) and stays with the fallback in the facade.
//!
//! - **`C32`:** one `f32x4` holds the state of a section, `[s1.re, s1.im, s2.re, s2.im]`, so the
//!   history is loaded and stored as it is. The step of the transposed direct form II is five
//!   vector operations on the recursion (the sum for `y`, the duplicate of `y`, the product
//!   with `[a1, a1, a2, a2]`, the difference and the sum with the shifted `s2`) plus the two
//!   products with the input and the flush of the subnormal range. There is no fused
//!   multiply-add, and the lanes carry the same values as the scalar code, bit for bit. The
//!   sum with `s2` adds `-0.0` in the lanes of `n2`, which leaves every value (also `-0.0`)
//!   unchanged.
//! - **`CQ15`:** the integer arithmetic of the fallback is exact in `f64`: the products of two
//!   16-bit values have at most 30 bits and five of them sum to less than 2^34, far below the
//!   53 bits of the mantissa. The coefficients are scaled by 2^-14 (exact), so that the
//!   accumulator is `acc / 2^14`. With the half `2^-15` added to the feed-forward part,
//!   `RNE(acc / 2^14 + 2^-15)` equals `floor((acc + 2^13) / 2^14)`, which is
//!   `round_shift(acc, 14)`: the value is `k + (r - 8191.5) / 2^14` for `acc + 2^13 = 2^14 k +
//!   r`, never on a tie. The round-to-nearest-even is the addition and subtraction of 1.5 *
//!   2^52, the saturation a minimum and a maximum. Four samples (one vector of `i16`) are
//!   converted in and out together; the state stays in `f64` for the whole buffer.
//!
//! Both sample types run section by section over the whole buffer, like the fallback.

use fearless_simd::{Simd as SimdToken, f32x4, f64x2, i16x8, i32x4, prelude::*};
use sdrk_core::kernels::biquad::{Biquad, BiquadSample, BiquadState};
use sdrk_core::{C32, CQ15, Fallback, Result};

use crate::Simd;
use crate::level::dispatch;
use crate::util::check_len;

/// `[a, b, c, d]` -> `[a, b, a, b]`.
#[inline(always)]
fn dup_low<S: SimdToken>(v: f32x4<S>) -> f32x4<S> {
    let d: f64x2<S> = v.bitcast();
    d.zip_low(d).bitcast()
}

/// `[a, b, c, d]` -> `[c, d, c, d]`.
#[inline(always)]
fn dup_high<S: SimdToken>(v: f32x4<S>) -> f32x4<S> {
    let d: f64x2<S> = v.bitcast();
    d.zip_high(d).bitcast()
}

/// The lanes 0 and 1 of `y0` and of `y1` as `[y0, y1]`.
#[inline(always)]
fn join<S: SimdToken>(y0: f32x4<S>, y1: f32x4<S>) -> f32x4<S> {
    let (a, b): (f64x2<S>, f64x2<S>) = (y0.bitcast(), y1.bitcast());
    a.zip_low(b).bitcast()
}

/// The recursion of one `C32` section.
struct Section32<S: SimdToken> {
    /// `[b0, b0, b0, b0]`.
    b0: f32x4<S>,
    /// `[b1, b1, b2, b2]`.
    b12: f32x4<S>,
    /// `[a1, a1, a2, a2]`.
    a: f32x4<S>,
    /// `-0.0` in every lane: the neutral element of the addition.
    neg_zero: f32x4<S>,
    /// `1e-30`, the limit of `flush_denormal`.
    limit: f32x4<S>,
    /// `[s1.re, s1.im, s2.re, s2.im]`.
    s: f32x4<S>,
}

impl<S: SimdToken> Section32<S> {
    #[inline(always)]
    fn new(simd: S, c: &[f32; 5], s: &[f32]) -> Self {
        let [b0, b1, b2, a1, a2] = *c;
        Section32 {
            b0: f32x4::splat(simd, b0),
            b12: f32x4::from_slice(simd, &[b1, b1, b2, b2]),
            a: f32x4::from_slice(simd, &[a1, a1, a2, a2]),
            neg_zero: f32x4::splat(simd, -0.0),
            limit: f32x4::splat(simd, 1e-30),
            s: f32x4::from_slice(simd, s),
        }
    }

    /// One sample; `d` is `[x.re, x.im, x.re, x.im]`. The output is in the lanes 0 and 1.
    #[inline(always)]
    fn step(&mut self, d: f32x4<S>) -> f32x4<S> {
        // y = b0 x + s1
        let y = d * self.b0 + self.s;
        // [b1 x - a1 y, ., b2 x - a2 y, .]
        let m = d * self.b12 - self.a * dup_low(y);
        // n1 = (b1 x - a1 y) + s2 and n2 = b2 x - a2 y (+ -0.0)
        let s2: f64x2<S> = self.s.bitcast();
        let nz: f64x2<S> = self.neg_zero.bitcast();
        let n = m + s2.zip_high(nz).bitcast::<f32x4<S>>();
        // flush_denormal: the lanes below the limit become +0.0, the others (NaN included) stay.
        // The mask is used as the bits it is: `select` would add a compare of the mask with
        // zero (`pcmpgtd`) to turn it into the representation of a boolean vector.
        let tiny = n.abs().simd_lt(self.limit).to_vector();
        self.s = (n.bitcast::<i32x4<S>>() & !tiny).bitcast();
        y
    }

    /// The last sample, `x = [re, im]`, written to `o`.
    #[inline(always)]
    fn tail(&mut self, simd: S, x: &[f32], o: &mut [f32]) {
        let mut t = [0.0_f32; 4];
        if let ([re, im], [t0, t1, ..]) = (x, &mut t) {
            (*t0, *t1) = (*re, *im);
        }
        let y = self.step(dup_low(f32x4::from_slice(simd, &t)));
        y.store_slice(&mut t);
        if let ([yr, yi], [t0, t1, ..]) = (o, t) {
            (*yr, *yi) = (t0, t1);
        }
    }

    /// Two samples (the four floats `v`) through the section; returns the two outputs.
    #[inline(always)]
    fn pair(&mut self, v: f32x4<S>) -> f32x4<S> {
        let y0 = self.step(dup_low(v));
        let y1 = self.step(dup_high(v));
        join(y0, y1)
    }

    /// The one to three samples that remain after the steps of four (`x` holds `[re, im]`
    /// pairs, at most six floats).
    #[inline(always)]
    fn rest(&mut self, simd: S, x: &[f32], o: &mut [f32]) {
        let whole = if x.len() >= 4 { 4 } else { 0 };
        let ((x, xt), (o, ot)) = (x.split_at(whole), o.split_at_mut(whole));
        if !x.is_empty() {
            self.pair(f32x4::from_slice(simd, x)).store_slice(o);
        }
        if !xt.is_empty() {
            self.tail(simd, xt, ot);
        }
    }
}

/// One `C32` section from `input` to `out` (two buffers), `h` its four history floats.
#[inline(always)]
fn section_c32<S: SimdToken>(simd: S, c: &[f32; 5], h: &mut [f32], input: &[f32], out: &mut [f32]) {
    let mut sec = Section32::new(simd, c, h);
    // Two pairs per step of the loop: one counter and one branch for four samples.
    let mut xs = input.chunks_exact(8);
    let mut ys = out.chunks_exact_mut(8);
    for (x, y) in (&mut xs).zip(&mut ys) {
        let ((x0, x1), (y0, y1)) = (x.split_at(4), y.split_at_mut(4));
        sec.pair(f32x4::from_slice(simd, x0)).store_slice(y0);
        sec.pair(f32x4::from_slice(simd, x1)).store_slice(y1);
    }
    sec.rest(simd, xs.remainder(), ys.into_remainder());
    sec.s.store_slice(h);
}

/// One `C32` section in place.
#[inline(always)]
fn section_c32_in_place<S: SimdToken>(simd: S, c: &[f32; 5], h: &mut [f32], io: &mut [f32]) {
    let mut sec = Section32::new(simd, c, h);
    let mut ys = io.chunks_exact_mut(8);
    for y in &mut ys {
        let (y0, y1) = y.split_at_mut(4);
        sec.pair(f32x4::from_slice(simd, y0)).store_slice(y0);
        sec.pair(f32x4::from_slice(simd, y1)).store_slice(y1);
    }
    let rest = ys.into_remainder();
    let mut copy = [0.0_f32; 6];
    if let Some(d) = copy.get_mut(..rest.len()) {
        d.copy_from_slice(rest);
    }
    if let Some(x) = copy.get(..rest.len()) {
        sec.rest(simd, x, rest);
    }
    sec.s.store_slice(h);
}

#[inline(always)]
fn cascade_c32<S: SimdToken>(
    simd: S,
    coefs: &[[f32; 5]],
    hist: &mut [C32],
    input: &[f32],
    out: &mut [f32],
) {
    for (k, (c, h)) in coefs.iter().zip(hist.chunks_exact_mut(4)).enumerate() {
        let h: &mut [f32] = bytemuck::cast_slice_mut(h);
        if let Some((h, _)) = h.split_at_mut_checked(4) {
            if k == 0 {
                section_c32(simd, c, h, input, out);
            } else {
                section_c32_in_place(simd, c, h, out);
            }
        }
    }
}

/// Whether the level runs the fallback instead of the two-lane path. AVX-512 does: the
/// fallback compiled for it (EVEX encoding, 32 registers, so fewer moves) takes 37.0
/// instructions per sample against 31.6 for the two-lane path, which saves 15 %, less than
/// the 25 % that M22 asks for; the other levels save 30 to 37 % (`docs/performance.md`). The
/// fallback is run inside the function of the level, so that it is compiled with its target
/// features and not for the baseline.
#[inline(always)]
fn fallback_level<S: SimdToken>(simd: S) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        simd.level().as_avx512().is_some()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = simd;
        false
    }
}

#[inline(always)]
fn biquad_c32<S: SimdToken, C, H>(
    simd: S,
    s: &mut BiquadState<C32, C, H>,
    input: &[C32],
    output: &mut [C32],
) -> Result<()>
where
    C: AsRef<[[f32; 5]]>,
    H: AsRef<[C32]> + AsMut<[C32]>,
{
    if fallback_level(simd) {
        return Biquad::<C32>::biquad(&Fallback, s, input, output);
    }
    check_len(input.len(), output.len())?;
    let (coefs, hist) = s.parts_mut();
    let x: &[f32] = bytemuck::cast_slice(input);
    let y: &mut [f32] = bytemuck::cast_slice_mut(output);
    cascade_c32(simd, coefs, hist, x, y);
    Ok(())
}

impl Biquad<C32> for Simd {
    fn biquad<C, H>(
        &self,
        s: &mut BiquadState<C32, C, H>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()>
    where
        C: AsRef<[[<C32 as BiquadSample>::Coef; 5]]>,
        H: AsRef<[C32]> + AsMut<[C32]>,
    {
        dispatch!(simd => biquad_c32(simd, s, input, output))
    }
}

/// `1.5 * 2^52`: adding and subtracting it rounds an `f64` of smaller magnitude to the nearest
/// integer, ties to even.
const MAGIC: f64 = 6_755_399_441_055_744.0;
/// `2^-14`: the scale of the coefficients.
const SCALE: f64 = 1.0 / 16_384.0;
/// `2^-15`: the half that turns the round-to-nearest-even of `acc / 2^14` into the
/// `round_shift` of the fallback (see the module description).
const HALF: f64 = 1.0 / 32_768.0;

/// The recursion of one `CQ15` section on `[re, im]`, the samples as integers in `f64`.
struct Section15<S: SimdToken> {
    b0: f64x2<S>,
    b1: f64x2<S>,
    b2: f64x2<S>,
    a1: f64x2<S>,
    a2: f64x2<S>,
    half: f64x2<S>,
    magic: f64x2<S>,
    hi: f64x2<S>,
    lo: f64x2<S>,
    x1: f64x2<S>,
    x2: f64x2<S>,
    y1: f64x2<S>,
    y2: f64x2<S>,
}

impl<S: SimdToken> Section15<S> {
    /// `h` holds `[x1, x2, y1, y2]` as interleaved `i16`.
    #[inline(always)]
    fn new(simd: S, c: &[i16; 5], h: &[i16]) -> Self {
        let [b0, b1, b2, a1, a2] = c.map(|v| f64::from(v) * SCALE);
        let pair = |i: usize| match h.get(2 * i..2 * i + 2) {
            Some(&[re, im]) => f64x2::from_slice(simd, &[f64::from(re), f64::from(im)]),
            _ => f64x2::splat(simd, 0.0),
        };
        Section15 {
            b0: f64x2::splat(simd, b0),
            b1: f64x2::splat(simd, b1),
            b2: f64x2::splat(simd, b2),
            a1: f64x2::splat(simd, a1),
            a2: f64x2::splat(simd, a2),
            half: f64x2::splat(simd, HALF),
            magic: f64x2::splat(simd, MAGIC),
            hi: f64x2::splat(simd, 32767.0),
            lo: f64x2::splat(simd, -32768.0),
            x1: pair(0),
            x2: pair(1),
            y1: pair(2),
            y2: pair(3),
        }
    }

    /// Writes the state back as `[x1, x2, y1, y2]` interleaved `i16`.
    #[inline(always)]
    fn store(&self, h: &mut [i16]) {
        for (v, d) in [self.x1, self.x2, self.y1, self.y2]
            .into_iter()
            .zip(h.chunks_exact_mut(2))
        {
            let mut t = [0.0_f64; 2];
            v.store_slice(&mut t);
            if let ([re, im], [t0, t1]) = (d, t) {
                (*re, *im) = (to_i16(t0), to_i16(t1));
            }
        }
    }

    /// One sample `x = [re, im]`; returns the output in the same form.
    #[inline(always)]
    fn step(&mut self, x: f64x2<S>) -> f64x2<S> {
        let ff = x * self.b0 + self.x1 * self.b1 + self.x2 * self.b2 + self.half;
        let q = ff - self.y1 * self.a1 - self.y2 * self.a2;
        let r = (q + self.magic) - self.magic;
        let y = r.min(self.hi).max(self.lo);
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    /// Four samples (one vector of `i16`).
    #[inline(always)]
    fn four(&mut self, simd: S, v: i16x8<S>) -> i16x8<S> {
        let (lo, hi) = v.widen();
        let (x0, x1) = f32x4::float_from(lo).widen();
        let (x2, x3) = f32x4::float_from(hi).widen();
        let y0 = self.step(x0);
        let y1 = self.step(x1);
        let y2 = self.step(x2);
        let y3 = self.step(x3);
        let a = i32x4::truncate_from(y0.narrow(y1));
        let b = i32x4::truncate_from(y2.narrow(y3));
        let _ = simd;
        a.saturating_narrow(b)
    }

    /// The one to three samples that remain: `x` and `o` hold `[re, im]` pairs.
    #[inline(always)]
    fn tail(&mut self, simd: S, x: &[i16], o: &mut [i16]) {
        for (x, o) in x.chunks_exact(2).zip(o.chunks_exact_mut(2)) {
            if let (&[re, im], [yr, yi]) = (x, o) {
                let y = self.step(f64x2::from_slice(simd, &[f64::from(re), f64::from(im)]));
                let mut t = [0.0_f64; 2];
                y.store_slice(&mut t);
                let [t0, t1] = t;
                (*yr, *yi) = (to_i16(t0), to_i16(t1));
            }
        }
    }
}

/// An integer-valued `f64` within the `i16` range as `i16`.
#[inline(always)]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the value is an integer in range"
)]
fn to_i16(v: f64) -> i16 {
    v as i16
}

/// One `CQ15` section from `input` to `out` (two buffers), `h` its eight history values.
#[inline(always)]
fn section_cq15<S: SimdToken>(
    simd: S,
    c: &[i16; 5],
    h: &mut [i16],
    input: &[i16],
    out: &mut [i16],
) {
    let mut sec = Section15::new(simd, c, h);
    let mut xs = input.chunks_exact(8);
    let mut ys = out.chunks_exact_mut(8);
    for (x, y) in (&mut xs).zip(&mut ys) {
        sec.four(simd, i16x8::from_slice(simd, x)).store_slice(y);
    }
    sec.tail(simd, xs.remainder(), ys.into_remainder());
    sec.store(h);
}

/// One `CQ15` section in place.
#[inline(always)]
fn section_cq15_in_place<S: SimdToken>(simd: S, c: &[i16; 5], h: &mut [i16], io: &mut [i16]) {
    let mut sec = Section15::new(simd, c, h);
    let mut ys = io.chunks_exact_mut(8);
    for y in &mut ys {
        let v = i16x8::from_slice(simd, y);
        sec.four(simd, v).store_slice(y);
    }
    let rest = ys.into_remainder();
    let mut copy = [0_i16; 6];
    if let Some(d) = copy.get_mut(..rest.len()) {
        d.copy_from_slice(rest);
    }
    if let Some(x) = copy.get(..rest.len()) {
        sec.tail(simd, x, rest);
    }
    sec.store(h);
}

#[inline(always)]
fn cascade_cq15<S: SimdToken>(
    simd: S,
    coefs: &[[i16; 5]],
    hist: &mut [CQ15],
    input: &[i16],
    out: &mut [i16],
) {
    for (k, (c, h)) in coefs.iter().zip(hist.chunks_exact_mut(4)).enumerate() {
        let h: &mut [i16] = bytemuck::cast_slice_mut(h);
        if k == 0 {
            section_cq15(simd, c, h, input, out);
        } else {
            section_cq15_in_place(simd, c, h, out);
        }
    }
}

impl Biquad<CQ15> for Simd {
    fn biquad<C, H>(
        &self,
        s: &mut BiquadState<CQ15, C, H>,
        input: &[CQ15],
        output: &mut [CQ15],
    ) -> Result<()>
    where
        C: AsRef<[[<CQ15 as BiquadSample>::Coef; 5]]>,
        H: AsRef<[CQ15]> + AsMut<[CQ15]>,
    {
        check_len(input.len(), output.len())?;
        let (coefs, hist) = s.parts_mut();
        let x: &[i16] = bytemuck::cast_slice(input);
        let y: &mut [i16] = bytemuck::cast_slice_mut(output);
        dispatch!(simd => cascade_cq15(simd, coefs, hist, x, y));
        Ok(())
    }
}
