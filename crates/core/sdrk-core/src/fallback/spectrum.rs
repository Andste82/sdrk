//! Fallback implementations of `window`, `mag_sq` and `power_db`.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::spectrum::{MagSq, PowerDb, Window};
use crate::num::{q15_mul, round_shift, sat16};
use crate::types::{C32, CQ15, Q15};

/// Smallest linear power that `power_db` evaluates; smaller, negative and NaN input use it.
const FLOOR: f32 = 1e-20;

impl Window<C32> for Fallback {
    fn window(&self, input: &[C32], output: &mut [C32], w: &[f32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        check_len(input.len(), w.len())?;
        for ((o, x), &w) in output.iter_mut().zip(input).zip(w) {
            *o = C32::new(x.re * w, x.im * w);
        }
        Ok(())
    }
}

impl Window<CQ15> for Fallback {
    fn window(&self, input: &[CQ15], output: &mut [CQ15], w: &[Q15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        check_len(input.len(), w.len())?;
        for ((o, x), &w) in output.iter_mut().zip(input).zip(w) {
            *o = CQ15 {
                re: q15_mul(x.re, w),
                im: q15_mul(x.im, w),
            };
        }
        Ok(())
    }
}

impl MagSq<C32> for Fallback {
    fn mag_sq(&self, input: &[C32], output: &mut [f32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (o, x) in output.iter_mut().zip(input) {
            *o = x.re * x.re + x.im * x.im;
        }
        Ok(())
    }
}

impl MagSq<CQ15> for Fallback {
    fn mag_sq(&self, input: &[CQ15], output: &mut [Q15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (o, x) in output.iter_mut().zip(input) {
            let (re, im) = (i64::from(x.re.0), i64::from(x.im.0));
            *o = Q15(sat16(round_shift(re * re + im * im, 15)));
        }
        Ok(())
    }
}

#[inline]
fn to_db(p: f32, offset_db: f32) -> f32 {
    let p = if p > FLOOR { p } else { FLOOR };
    10.0 * libm::log10f(p) + offset_db
}

impl PowerDb<f32> for Fallback {
    fn power_db(&self, input: &[f32], output: &mut [f32], offset_db: f32) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (o, &p) in output.iter_mut().zip(input) {
            *o = to_db(p, offset_db);
        }
        Ok(())
    }
}

impl PowerDb<Q15> for Fallback {
    fn power_db(&self, input: &[Q15], output: &mut [f32], offset_db: f32) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (o, &p) in output.iter_mut().zip(input) {
            *o = to_db(p.to_f32(), offset_db);
        }
        Ok(())
    }
}
