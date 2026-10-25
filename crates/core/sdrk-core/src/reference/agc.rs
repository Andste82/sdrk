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
        let (lo_ref, hi_ref) = (f64::from(par.lo_f), f64::from(par.hi_f));
        let (rate_a, rate_r) = (f64::from(par.ra_f), f64::from(par.rr_f));
        let (lo, hi) = (f64::from(par.gmin_f), f64::from(par.gmax_f));
        let mut g_now = *par.g_d;
        for (o, x) in output.iter_mut().zip(input) {
            let (yr, yi) = (g_now * f64::from(x.re), g_now * f64::from(x.im));
            *o = C32::new(yr as f32, yi as f32);
            let level = libm::sqrt(yr * yr + yi * yi);
            if level > 0.0 {
                let err = reference - level;
                g_now = if err < 0.0 {
                    let div = level.min(hi_ref);
                    g_now * div / (div + rate_a * (level - reference))
                } else {
                    let div = level.max(lo_ref);
                    g_now * (div + rate_r * err) / div
                };
                g_now = g_now.max(lo).min(hi);
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
            if x.re.0 == 0 && x.im.0 == 0 {
                continue;
            }
            let sum = i64::from(y.re.0).pow(2) + i64::from(y.im.0).pow(2);
            let level = nearest_sqrt(sum);
            let err = i128::from(par.ref_q - level);
            let g = i128::from(gain);
            let next = if err < 0 {
                let div = i128::from(level.min(par.hi_q));
                let den = (div << 48) - i128::from(par.ra_q) * err;
                // floor(g d 2^48 / den + 1/2)
                (2 * ((g * div) << 48) + den).div_euclid(2 * den)
            } else {
                let den = i128::from(level.max(par.lo_q)) << 48;
                // g + floor(g r e / den + 1/2)
                g + (2 * g * i128::from(par.rr_q) * err + den).div_euclid(2 * den)
            };
            *par.g_q = next.clamp(i128::from(par.gmin_q), i128::from(par.gmax_q)) as i64;
        }
        Ok(())
    }
}
