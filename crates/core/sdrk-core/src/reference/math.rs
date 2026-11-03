//! Reference implementation of `add`, `add_real`, `mul`, `mul_real`, `interleave` and
//! `deinterleave`: the formulas of the contract, one sample at a time (`f32` arithmetic in
//! the written order for `f32`, `i64` for `Q15`).

use crate::Reference;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::math::{Add, AddReal, Interleave, Mul, MulReal};
use crate::num::{round_shift, sat16};
use crate::types::{C32, CQ15, ComplexSample, Q15, Sample};

fn check3<T>(a: &[T], b: &[T], out: &[T]) -> Result<()> {
    check_len(a.len(), b.len())?;
    check_len(a.len(), out.len())
}

fn sat_add(a: Q15, b: Q15) -> Q15 {
    Q15(sat16(i64::from(a.0) + i64::from(b.0)))
}

fn q15_product(a: Q15, b: Q15) -> Q15 {
    Q15(sat16(round_shift(i64::from(a.0) * i64::from(b.0), 15)))
}

impl Add<C32> for Reference {
    fn add(&self, a: &[C32], b: &[C32], out: &mut [C32]) -> Result<()> {
        check3(a, b, out)?;
        for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
            *o = C32::new(x.re + y.re, x.im + y.im);
        }
        Ok(())
    }
}

impl Add<CQ15> for Reference {
    fn add(&self, a: &[CQ15], b: &[CQ15], out: &mut [CQ15]) -> Result<()> {
        check3(a, b, out)?;
        for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
            *o = CQ15 {
                re: sat_add(x.re, y.re),
                im: sat_add(x.im, y.im),
            };
        }
        Ok(())
    }
}

impl AddReal<f32> for Reference {
    fn add_real(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        check3(a, b, out)?;
        for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
            *o = x + y;
        }
        Ok(())
    }
}

impl AddReal<Q15> for Reference {
    fn add_real(&self, a: &[Q15], b: &[Q15], out: &mut [Q15]) -> Result<()> {
        check3(a, b, out)?;
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = sat_add(x, y);
        }
        Ok(())
    }
}

impl Mul<C32> for Reference {
    fn mul(&self, a: &[C32], b: &[C32], out: &mut [C32]) -> Result<()> {
        check3(a, b, out)?;
        for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
            let (rr, ii) = (x.re * y.re, x.im * y.im);
            let (ri, ir) = (x.re * y.im, x.im * y.re);
            *o = C32::new(rr - ii, ri + ir);
        }
        Ok(())
    }
}

impl Mul<CQ15> for Reference {
    fn mul(&self, a: &[CQ15], b: &[CQ15], out: &mut [CQ15]) -> Result<()> {
        check3(a, b, out)?;
        for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
            let (ar, ai) = (i64::from(x.re.0), i64::from(x.im.0));
            let (br, bi) = (i64::from(y.re.0), i64::from(y.im.0));
            *o = CQ15 {
                re: Q15(sat16(round_shift(ar * br - ai * bi, 15))),
                im: Q15(sat16(round_shift(ar * bi + ai * br, 15))),
            };
        }
        Ok(())
    }
}

impl MulReal<f32> for Reference {
    fn mul_real(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        check3(a, b, out)?;
        for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
            *o = x * y;
        }
        Ok(())
    }
}

impl MulReal<Q15> for Reference {
    fn mul_real(&self, a: &[Q15], b: &[Q15], out: &mut [Q15]) -> Result<()> {
        check3(a, b, out)?;
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = q15_product(x, y);
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
    if let Some(im) = im {
        check_len(re.len(), im.len())?;
    }
    for (n, (o, &r)) in out.iter_mut().zip(re).enumerate() {
        let i = im
            .and_then(|im| im.get(n))
            .copied()
            .unwrap_or(<T::Real as Sample>::ZERO);
        *o = T::new(r, i);
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

impl Interleave<C32> for Reference {
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

impl Interleave<CQ15> for Reference {
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
