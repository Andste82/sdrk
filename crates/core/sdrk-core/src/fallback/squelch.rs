//! Fallback implementation of `squelch`.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::squelch::{Squelch, SquelchState};
use crate::num::{flush_denormal, round_shift};
use crate::types::{C32, CQ15, Sample};

impl Squelch<C32> for Fallback {
    fn squelch(
        &self,
        state: &mut SquelchState<C32>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let g = state.parts_mut();
        let mut p = *g.p_f;
        let mut open = *g.open;
        for (o, &x) in output.iter_mut().zip(input) {
            let power = x.re * x.re + x.im * x.im;
            if power.is_finite() {
                let diff = power - p;
                let step = g.alpha_f * diff;
                p = flush_denormal(p + step);
                if p >= g.open_f {
                    open = true;
                } else if p < g.close_f {
                    open = false;
                }
            }
            *o = if open { x } else { C32::ZERO };
        }
        *g.p_f = p;
        *g.open = open;
        Ok(())
    }
}

impl Squelch<CQ15> for Fallback {
    fn squelch(
        &self,
        state: &mut SquelchState<CQ15>,
        input: &[CQ15],
        output: &mut [CQ15],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let g = state.parts_mut();
        let mut p = *g.p_q;
        let mut open = *g.open;
        for (o, &x) in output.iter_mut().zip(input) {
            let (re, im) = (i64::from(x.re.0), i64::from(x.im.0));
            let power = re * re + im * im;
            p += round_shift(g.alpha_q * (power - p), 15);
            if p >= g.open_q {
                open = true;
            } else if p < g.close_q {
                open = false;
            }
            *o = if open { x } else { CQ15::ZERO };
        }
        *g.p_q = p;
        *g.open = open;
        Ok(())
    }
}
