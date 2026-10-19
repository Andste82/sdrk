//! Fallback implementations of `dot` and `dot_real`.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::dot::{Dot, DotReal};
use crate::num::Acc64;
use crate::types::{C32, CQ15, Q15};

impl Dot<C32> for Fallback {
    fn dot(&self, x: &[C32], taps: &[f32]) -> Result<C32> {
        check_len(x.len(), taps.len())?;
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (s, &h) in x.iter().zip(taps) {
            re += s.re * h;
            im += s.im * h;
        }
        Ok(C32::new(re, im))
    }
}

impl Dot<CQ15> for Fallback {
    fn dot(&self, x: &[CQ15], taps: &[Q15]) -> Result<CQ15> {
        check_len(x.len(), taps.len())?;
        let (mut re, mut im) = (Acc64::default(), Acc64::default());
        for (s, &h) in x.iter().zip(taps) {
            re.mac(s.re, h);
            im.mac(s.im, h);
        }
        Ok(CQ15 {
            re: re.finish(),
            im: im.finish(),
        })
    }
}

impl DotReal<f32> for Fallback {
    fn dot_real(&self, x: &[f32], taps: &[f32]) -> Result<f32> {
        check_len(x.len(), taps.len())?;
        let mut acc = 0.0f32;
        for (&s, &h) in x.iter().zip(taps) {
            acc += s * h;
        }
        Ok(acc)
    }
}

impl DotReal<Q15> for Fallback {
    fn dot_real(&self, x: &[Q15], taps: &[Q15]) -> Result<Q15> {
        check_len(x.len(), taps.len())?;
        let mut acc = Acc64::default();
        for (&s, &h) in x.iter().zip(taps) {
            acc.mac(s, h);
        }
        Ok(acc.finish())
    }
}
