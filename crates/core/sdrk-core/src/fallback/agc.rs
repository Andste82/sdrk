//! Fallback implementation of `agc`.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::agc::{Agc, AgcState};
use crate::kernels::check_len;
use crate::num::{round_shift, round_sqrt, sat16};
use crate::types::{C32, CQ15, Q15};

impl Agc<C32> for Fallback {
    fn agc(&self, state: &mut AgcState<C32>, input: &[C32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let par = state.parts_mut();
        let mut gain = *par.g_f;
        for (o, &x) in output.iter_mut().zip(input) {
            let y = C32::new(gain * x.re, gain * x.im);
            *o = y;
            let level = libm::sqrtf(y.re * y.re + y.im * y.im);
            if !level.is_nan() {
                let err = par.ref_f - level;
                let coef = if err < 0.0 { par.ka_f } else { par.kr_f };
                gain *= 1.0 + coef * err;
                gain = gain.max(par.gmin_f).min(par.gmax_f);
            }
        }
        *par.g_f = gain;
        *par.g_d = f64::from(gain);
        Ok(())
    }
}

impl Agc<CQ15> for Fallback {
    fn agc(&self, state: &mut AgcState<CQ15>, input: &[CQ15], output: &mut [CQ15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let par = state.parts_mut();
        let mut gain = *par.g_q;
        for (o, &x) in output.iter_mut().zip(input) {
            let y = CQ15 {
                re: Q15(sat16(round_shift(i64::from(x.re.0) * gain, 32))),
                im: Q15(sat16(round_shift(i64::from(x.im.0) * gain, 32))),
            };
            *o = y;
            let (re, im) = (i64::from(y.re.0), i64::from(y.im.0));
            let level = round_sqrt(re * re + im * im);
            let err = par.ref_q - level;
            let coef = if err < 0 { par.ka_q } else { par.kr_q };
            let prod = i128::from(gain) * i128::from(err) * i128::from(coef);
            let delta = (prod + (1 << 47)) >> 48;
            // `delta` is below 2^52 in magnitude, so the narrowing cannot lose bits.
            gain = gain
                .saturating_add(delta as i64)
                .clamp(par.gmin_q, par.gmax_q);
        }
        *par.g_q = gain;
        Ok(())
    }
}
