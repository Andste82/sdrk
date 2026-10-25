//! Reference implementation of `squelch`.
//!
//! The gate decision is discrete, so an f64 model could switch one sample earlier or
//! later than the fallback. The specification therefore fixes the arithmetic (f32 for
//! `C32`, exact `i64` for `CQ15`) and this reference follows it step by step, written
//! as separate small functions.

use crate::Reference;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::squelch::{Squelch, SquelchState};
use crate::num::{flush_denormal, round_shift};
use crate::types::{C32, CQ15, Sample};

/// Gate decision with hysteresis: open at or above `open`, closed below `close`.
fn decide<P: PartialOrd + Copy>(was_open: bool, p: P, open: P, close: P) -> bool {
    if p >= open {
        true
    } else if p < close {
        false
    } else {
        was_open
    }
}

/// One f32 step; `None` for a non-finite power (nothing changes).
fn next_power_f32(p: f32, alpha: f32, re: f32, im: f32) -> Option<f32> {
    let re2 = re * re;
    let im2 = im * im;
    let q = re2 + im2;
    if !q.is_finite() {
        return None;
    }
    let delta = alpha * (q - p);
    Some(flush_denormal(p + delta))
}

impl Squelch<C32> for Reference {
    fn squelch(
        &self,
        state: &mut SquelchState<C32>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let g = state.parts_mut();
        for (o, x) in output.iter_mut().zip(input) {
            if let Some(p) = next_power_f32(*g.p_f, g.alpha_f, x.re, x.im) {
                *g.p_f = p;
                *g.open = decide(*g.open, p, g.open_f, g.close_f);
            }
            *o = if *g.open { *x } else { C32::ZERO };
        }
        Ok(())
    }
}

impl Squelch<CQ15> for Reference {
    fn squelch(
        &self,
        state: &mut SquelchState<CQ15>,
        input: &[CQ15],
        output: &mut [CQ15],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let g = state.parts_mut();
        for (o, x) in output.iter_mut().zip(input) {
            let q = i64::from(x.re.0).pow(2) + i64::from(x.im.0).pow(2);
            let step = round_shift(g.alpha_q * (q - *g.p_q), 15);
            *g.p_q += step;
            *g.open = decide(*g.open, *g.p_q, g.open_q, g.close_q);
            *o = if *g.open { *x } else { CQ15::ZERO };
        }
        Ok(())
    }
}
