//! Fallback implementations of `add`, `add_real`, `mul`, `mul_real`, `interleave` and
//! `deinterleave`.
//!
//! The loops are plain zips over the three slices (no indexing, no branches in the body), the
//! form the compilers vectorize on every target and that is fast without SIMD on 64-bit ARM.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::math::{Add, AddReal, Interleave, Mul, MulReal};
use crate::types::{C32, CQ15, ComplexSample, Q15, Sample};

fn check3<T>(a: &[T], b: &[T], out: &[T]) -> Result<()> {
    check_len(a.len(), b.len())?;
    check_len(a.len(), out.len())
}

impl Add<C32> for Fallback {
    fn add(&self, a: &[C32], b: &[C32], out: &mut [C32]) -> Result<()> {
        check3(a, b, out)?;
        for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
            *o = C32::new(x.re + y.re, x.im + y.im);
        }
        Ok(())
    }
}

impl Add<CQ15> for Fallback {
    fn add(&self, a: &[CQ15], b: &[CQ15], out: &mut [CQ15]) -> Result<()> {
        check3(a, b, out)?;
        for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
            *o = CQ15 {
                re: Q15(x.re.0.saturating_add(y.re.0)),
                im: Q15(x.im.0.saturating_add(y.im.0)),
            };
        }
        Ok(())
    }
}

impl AddReal<f32> for Fallback {
    fn add_real(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        check3(a, b, out)?;
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = x + y;
        }
        Ok(())
    }
}

impl AddReal<Q15> for Fallback {
    fn add_real(&self, a: &[Q15], b: &[Q15], out: &mut [Q15]) -> Result<()> {
        check3(a, b, out)?;
        zip_q15(a, b, out, i16::saturating_add);
        Ok(())
    }
}

/// `out[i] = f(a[i], b[i])` for the real `Q15` kernels, in the order of the samples.
///
/// On wasm32 the samples first run in groups of eight (D294): LLVM vectorizes the loop over the
/// groups four groups wide, 32 samples per iteration with constant load and store offsets, and
/// takes the 32-bit products of `mul_real` down to 16 bits with one `i8x16.shuffle` per pair of
/// vectors; the plain loop runs eight samples per iteration with three pointer increments, and
/// `and`/`and`/`narrow` per pair. The rest, and every sample on the other targets, goes through
/// the plain loop. The same `f` on the same samples, so the same bits.
#[inline]
fn zip_q15(a: &[Q15], b: &[Q15], out: &mut [Q15], f: impl Fn(i16, i16) -> i16) {
    #[cfg(target_arch = "wasm32")]
    let (a, b, out) = {
        let (a_groups, a_rest) = a.as_chunks::<8>();
        let (b_groups, b_rest) = b.as_chunks::<8>();
        let (out_groups, out_rest) = out.as_chunks_mut::<8>();
        for ((o, x), y) in out_groups.iter_mut().zip(a_groups).zip(b_groups) {
            for ((o, x), y) in o.iter_mut().zip(x).zip(y) {
                *o = Q15(f(x.0, y.0));
            }
        }
        (a_rest, b_rest, out_rest)
    };
    for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
        *o = Q15(f(x.0, y.0));
    }
}

impl Mul<C32> for Fallback {
    fn mul(&self, a: &[C32], b: &[C32], out: &mut [C32]) -> Result<()> {
        check3(a, b, out)?;
        mul_c32(a, b, out);
        Ok(())
    }
}

/// The plain loop of the `C32` product, the form LLVM vectorizes on every target (`ld2`/`st2`
/// on NEON, deinterleaving shuffles on x86).
#[cfg(not(all(target_feature = "sse3", not(target_feature = "avx512f"))))]
fn mul_c32(a: &[C32], b: &[C32], out: &mut [C32]) {
    for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
        *o = C32::new(x.re * y.re - x.im * y.im, x.re * y.im + x.im * y.re);
    }
}

/// x86 with SSE3 below AVX-512 (`sse4.2`, `avx2`): groups of eight samples, the four products
/// of every sample before its two sums. The loop over the groups is not vectorized, and the
/// SLP vectorizer turns each group into `movsldup`/`movshdup`, the swap of `b`, two products
/// and `addsubps` on the interleaved samples (2 resp. 4 vectors per group); the plain loop
/// deinterleaves both inputs and interleaves the output instead (D291). The same products and
/// sums in the same order as the plain loop, so the same bits.
#[cfg(all(target_feature = "sse3", not(target_feature = "avx512f")))]
fn mul_c32(a: &[C32], b: &[C32], out: &mut [C32]) {
    const GROUP: usize = 8;
    // `check3` has made the lengths equal; the common length lets LLVM drop the checks.
    let len = out.len().min(a.len()).min(b.len());
    let (a, b) = (a.get(..len).unwrap_or(&[]), b.get(..len).unwrap_or(&[]));
    let out = out.get_mut(..len).unwrap_or(&mut []);
    let mut oc = out.chunks_exact_mut(GROUP);
    let mut ac = a.chunks_exact(GROUP);
    let mut bc = b.chunks_exact(GROUP);
    for ((out_group, a_group), b_group) in (&mut oc).zip(&mut ac).zip(&mut bc) {
        let mut first = [C32::default(); GROUP];
        let mut second = [C32::default(); GROUP];
        let products = first.iter_mut().zip(&mut second).zip(a_group).zip(b_group);
        for (((p, q), x), y) in products {
            *p = C32::new(x.re * y.re, x.re * y.im);
            *q = C32::new(x.im * y.im, x.im * y.re);
        }
        for ((o, p), q) in out_group.iter_mut().zip(&first).zip(&second) {
            *o = C32::new(p.re - q.re, p.im + q.im);
        }
    }
    let rest = oc
        .into_remainder()
        .iter_mut()
        .zip(ac.remainder())
        .zip(bc.remainder());
    for ((o, x), y) in rest {
        *o = C32::new(x.re * y.re - x.im * y.im, x.re * y.im + x.im * y.re);
    }
}

/// The `Q15` product of the numeric conventions in `i32`: `|a·b| <= 2^30`, so the rounding term
/// cannot overflow, and the result is at least `-32768`, so only `32768` (`-1 · -1`) needs the
/// saturation.
#[inline]
fn mul_q15(a: i16, b: i16) -> i16 {
    ((i32::from(a) * i32::from(b) + (1 << 14)) >> 15).min(32767) as i16
}

impl Mul<CQ15> for Fallback {
    fn mul(&self, a: &[CQ15], b: &[CQ15], out: &mut [CQ15]) -> Result<()> {
        check3(a, b, out)?;
        mul_cq15(a, b, out);
        Ok(())
    }
}

/// The complex `Q15` product of R7 with the exact sums in `i32` (D292): `ar br - ai bi` lies in
/// `[-2^31 + 2^15, 2^31 - 2^15]`, so it and its rounding term fit. `ar bi + ai br` reaches `2^31`
/// (all four parts `-1`), so the imaginary part is computed negated, `t = (-ar) bi - ai br` in
/// `[-2^31, 2^31 - 2^16]`, and `floor((s + 2^14) / 2^15) = -floor((t + 2^14 - 1) / 2^15)` for
/// `s = -t`. Bit for bit the `i64` form of the reference; returns `(re, im)` before the
/// saturation.
#[inline]
fn mul_cq15_parts(ar: i32, ai: i32, br: i32, bi: i32) -> (i32, i32) {
    let re = (ar * br - ai * bi + (1 << 14)) >> 15;
    let im = -((-ar * bi - ai * br + ((1 << 14) - 1)) >> 15);
    (re.clamp(-32768, 32767), im.clamp(-32768, 32767))
}

/// The plain loop over the samples: products of 16-bit values in 32-bit lanes, which LLVM
/// vectorizes on every target (the `i64` sums before D292 stayed scalar at `sse2`, `sse4.2`,
/// `neon` and `simd128`).
#[cfg(not(all(target_feature = "avx2", not(target_feature = "avx512f"))))]
fn mul_cq15(a: &[CQ15], b: &[CQ15], out: &mut [CQ15]) {
    for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
        let (ar, ai) = (i32::from(x.re.0), i32::from(x.im.0));
        let (br, bi) = (i32::from(y.re.0), i32::from(y.im.0));
        let (re, im) = mul_cq15_parts(ar, ai, br, bi);
        *o = CQ15 {
            re: Q15(re as i16),
            im: Q15(im as i16),
        };
    }
}

/// AVX2 below AVX-512: every sample as one 32-bit word (little-endian, the real part in the low
/// half), the parts taken out by shifts. LLVM keeps the samples in their 32-bit lanes (two
/// shifts per input and one blend for the output) instead of deinterleaving both inputs into
/// 16-bit planes and widening them (D292). At AVX-512 the plain loop is shorter.
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
fn mul_cq15(a: &[CQ15], b: &[CQ15], out: &mut [CQ15]) {
    let a: &[[u8; 4]] = bytemuck::cast_slice(a);
    let b: &[[u8; 4]] = bytemuck::cast_slice(b);
    let out: &mut [[u8; 4]] = bytemuck::cast_slice_mut(out);
    for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
        let (x, y) = (u32::from_le_bytes(*x), u32::from_le_bytes(*y));
        let (ar, ai) = (i32::from(x as u16 as i16), (x as i32) >> 16);
        let (br, bi) = (i32::from(y as u16 as i16), (y as i32) >> 16);
        let (re, im) = mul_cq15_parts(ar, ai, br, bi);
        *o = (u32::from(re as u16) | (u32::from(im as u16) << 16)).to_le_bytes();
    }
}

impl MulReal<f32> for Fallback {
    fn mul_real(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        check3(a, b, out)?;
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = x * y;
        }
        Ok(())
    }
}

impl MulReal<Q15> for Fallback {
    fn mul_real(&self, a: &[Q15], b: &[Q15], out: &mut [Q15]) -> Result<()> {
        check3(a, b, out)?;
        zip_q15(a, b, out, mul_q15);
        Ok(())
    }
}

fn interleave_impl<T: ComplexSample>(
    re: &[T::Real],
    im: Option<&[T::Real]>,
    out: &mut [T],
) -> Result<()> {
    check_len(re.len(), out.len())?;
    match im {
        Some(im) => {
            check_len(re.len(), im.len())?;
            for ((o, &r), &i) in out.iter_mut().zip(re).zip(im) {
                *o = T::new(r, i);
            }
        }
        None => {
            for (o, &r) in out.iter_mut().zip(re) {
                *o = T::new(r, <T::Real as Sample>::ZERO);
            }
        }
    }
    Ok(())
}

fn deinterleave_impl<T: ComplexSample>(
    input: &[T],
    re: Option<&mut [T::Real]>,
    im: Option<&mut [T::Real]>,
) -> Result<()> {
    if let Some(re) = &re {
        check_len(input.len(), re.len())?;
    }
    if let Some(im) = &im {
        check_len(input.len(), im.len())?;
    }
    if let Some(re) = re {
        for (o, x) in re.iter_mut().zip(input) {
            *o = x.re();
        }
    }
    if let Some(im) = im {
        for (o, x) in im.iter_mut().zip(input) {
            *o = x.im();
        }
    }
    Ok(())
}

impl Interleave<C32> for Fallback {
    fn interleave(&self, re: &[f32], im: Option<&[f32]>, out: &mut [C32]) -> Result<()> {
        interleave_impl(re, im, out)
    }

    fn deinterleave(
        &self,
        input: &[C32],
        re: Option<&mut [f32]>,
        im: Option<&mut [f32]>,
    ) -> Result<()> {
        deinterleave_impl(input, re, im)
    }
}

impl Interleave<CQ15> for Fallback {
    fn interleave(&self, re: &[Q15], im: Option<&[Q15]>, out: &mut [CQ15]) -> Result<()> {
        interleave_impl(re, im, out)
    }

    fn deinterleave(
        &self,
        input: &[CQ15],
        re: Option<&mut [Q15]>,
        im: Option<&mut [Q15]>,
    ) -> Result<()> {
        deinterleave_impl(input, re, im)
    }
}
