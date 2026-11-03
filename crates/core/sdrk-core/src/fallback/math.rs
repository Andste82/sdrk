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
        for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
            *o = Q15(x.0.saturating_add(y.0));
        }
        Ok(())
    }
}

impl Mul<C32> for Fallback {
    fn mul(&self, a: &[C32], b: &[C32], out: &mut [C32]) -> Result<()> {
        check3(a, b, out)?;
        for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
            *o = C32::new(x.re * y.re - x.im * y.im, x.re * y.im + x.im * y.re);
        }
        Ok(())
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
        // The sum of two products reaches 2^31, so the accumulation is in `i64` (R7).
        let narrow = |v: i64| ((v + (1 << 14)) >> 15).clamp(-32768, 32767) as i16;
        for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
            let (ar, ai) = (i64::from(x.re.0), i64::from(x.im.0));
            let (br, bi) = (i64::from(y.re.0), i64::from(y.im.0));
            *o = CQ15 {
                re: Q15(narrow(ar * br - ai * bi)),
                im: Q15(narrow(ar * bi + ai * br)),
            };
        }
        Ok(())
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
        for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
            *o = Q15(mul_q15(x.0, y.0));
        }
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
