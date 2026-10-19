//! Reference implementation of `dot` and `dot_real`: f64 accumulation for `f32` data, exact
//! `i64` Q30 accumulation for Q15 data.

use crate::Reference;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::dot::{Dot, DotReal};
use crate::num::{round_shift, sat16};
use crate::types::{C32, CQ15, Q15};

impl Dot<C32> for Reference {
    fn dot(&self, x: &[C32], taps: &[f32]) -> Result<C32> {
        check_len(x.len(), taps.len())?;
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (s, &h) in x.iter().zip(taps) {
            re += f64::from(s.re) * f64::from(h);
            im += f64::from(s.im) * f64::from(h);
        }
        Ok(C32::new(re as f32, im as f32))
    }
}

impl Dot<CQ15> for Reference {
    fn dot(&self, x: &[CQ15], taps: &[Q15]) -> Result<CQ15> {
        check_len(x.len(), taps.len())?;
        let re: i64 = x
            .iter()
            .zip(taps)
            .map(|(s, h)| i64::from(s.re.0) * i64::from(h.0))
            .sum();
        let im: i64 = x
            .iter()
            .zip(taps)
            .map(|(s, h)| i64::from(s.im.0) * i64::from(h.0))
            .sum();
        Ok(CQ15 {
            re: Q15(sat16(round_shift(re, 15))),
            im: Q15(sat16(round_shift(im, 15))),
        })
    }
}

impl DotReal<f32> for Reference {
    fn dot_real(&self, x: &[f32], taps: &[f32]) -> Result<f32> {
        check_len(x.len(), taps.len())?;
        let mut acc = 0.0f64;
        for (&s, &h) in x.iter().zip(taps) {
            acc += f64::from(s) * f64::from(h);
        }
        Ok(acc as f32)
    }
}

impl DotReal<Q15> for Reference {
    fn dot_real(&self, x: &[Q15], taps: &[Q15]) -> Result<Q15> {
        check_len(x.len(), taps.len())?;
        let mut acc = 0i64;
        for (&s, h) in x.iter().zip(taps) {
            acc += i64::from(s.0) * i64::from(h.0);
        }
        Ok(Q15(sat16(round_shift(acc, 15))))
    }
}
