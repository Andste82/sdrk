//! Reference implementations of `window`, `mag_sq` and `power_db`.

use crate::Reference;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::spectrum::{MagSq, PowerDb, Window};
use crate::num::{round_shift, sat16};
use crate::types::{C32, CQ15, Q15};

impl Window<C32> for Reference {
    fn window(&self, input: &[C32], output: &mut [C32], w: &[f32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        check_len(input.len(), w.len())?;
        for ((o, x), &w) in output.iter_mut().zip(input).zip(w) {
            let w = f64::from(w);
            *o = C32::new((f64::from(x.re) * w) as f32, (f64::from(x.im) * w) as f32);
        }
        Ok(())
    }
}

impl Window<CQ15> for Reference {
    fn window(&self, input: &[CQ15], output: &mut [CQ15], w: &[Q15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        check_len(input.len(), w.len())?;
        let mul = |a: Q15, b: Q15| Q15(sat16(round_shift(i64::from(a.0) * i64::from(b.0), 15)));
        for ((o, x), &w) in output.iter_mut().zip(input).zip(w) {
            *o = CQ15 {
                re: mul(x.re, w),
                im: mul(x.im, w),
            };
        }
        Ok(())
    }
}

impl MagSq<C32> for Reference {
    fn mag_sq(&self, input: &[C32], output: &mut [f32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (o, x) in output.iter_mut().zip(input) {
            let (re, im) = (f64::from(x.re), f64::from(x.im));
            *o = (re * re + im * im) as f32;
        }
        Ok(())
    }
}

impl MagSq<CQ15> for Reference {
    fn mag_sq(&self, input: &[CQ15], output: &mut [Q15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (o, x) in output.iter_mut().zip(input) {
            let (re, im) = (i64::from(x.re.0), i64::from(x.im.0));
            *o = Q15(sat16(round_shift(re * re + im * im, 15)));
        }
        Ok(())
    }
}

fn to_db(p: f64, offset_db: f32) -> f32 {
    let p = if p > 1e-20 { p } else { 1e-20 };
    (10.0 * libm::log10(p) + f64::from(offset_db)) as f32
}

impl PowerDb<f32> for Reference {
    fn power_db(&self, input: &[f32], output: &mut [f32], offset_db: f32) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (o, &p) in output.iter_mut().zip(input) {
            *o = to_db(f64::from(p), offset_db);
        }
        Ok(())
    }
}

impl PowerDb<Q15> for Reference {
    fn power_db(&self, input: &[Q15], output: &mut [f32], offset_db: f32) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (o, &p) in output.iter_mut().zip(input) {
            *o = to_db(f64::from(p.0) / 32768.0, offset_db);
        }
        Ok(())
    }
}
