//! Reference implementation of `agc`.
//!
//! `C32` runs the algorithm in f64 with the f64 values of the f32 constants; the f64 gain is
//! kept in the state between calls. `CQ15` follows the integer model step by step.

use super::spectrum::nearest_sqrt;
use crate::Reference;
use crate::error::Result;
use crate::kernels::agc::{Agc, AgcState};
use crate::kernels::check_len;
use crate::num::{round_shift, sat16};
use crate::types::{C32, CQ15, Q15};

impl Agc<C32> for Reference {
    fn agc(&self, state: &mut AgcState<C32>, input: &[C32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let par = state.parts_mut();
        let reference = f64::from(par.ref_f);
        let (rate_a, rate_r) = (f64::from(par.ka_f), f64::from(par.kr_f));
        let (lo, hi) = (f64::from(par.gmin_f), f64::from(par.gmax_f));
        let mut g_now = *par.g_d;
        for (o, x) in output.iter_mut().zip(input) {
            let (yr, yi) = (g_now * f64::from(x.re), g_now * f64::from(x.im));
            *o = C32::new(yr as f32, yi as f32);
            let level = libm::sqrt(yr * yr + yi * yi);
            if !level.is_nan() {
                let err = reference - level;
                let coef = if err < 0.0 { rate_a } else { rate_r };
                g_now = (g_now * (1.0 + coef * err)).max(lo).min(hi);
            }
        }
        *par.g_d = g_now;
        *par.g_f = g_now as f32;
        Ok(())
    }
}

impl Agc<CQ15> for Reference {
    fn agc(&self, state: &mut AgcState<CQ15>, input: &[CQ15], output: &mut [CQ15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let par = state.parts_mut();
        for (o, x) in output.iter_mut().zip(input) {
            let gain = *par.g_q;
            let scale = |v: Q15| Q15(sat16(round_shift(i64::from(v.0) * gain, 32)));
            let y = CQ15 {
                re: scale(x.re),
                im: scale(x.im),
            };
            *o = y;
            let sum = i64::from(y.re.0).pow(2) + i64::from(y.im.0).pow(2);
            let err = par.ref_q - nearest_sqrt(sum);
            let coef = if err < 0 { par.ka_q } else { par.kr_q };
            let product = i128::from(gain) * i128::from(err) * i128::from(coef);
            let delta = (product + (1i128 << 47)) >> 48;
            let next = i128::from(gain) + delta;
            *par.g_q = next.clamp(i128::from(par.gmin_q), i128::from(par.gmax_q)) as i64;
        }
        Ok(())
    }
}
