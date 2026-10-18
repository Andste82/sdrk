//! Reference implementations of `quad_demod`, `deemphasis` and `dc_block` (f64 and
//! exact integer model).

use core::f64::consts::PI;

use crate::Reference;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::demod::{
    DcBlock, DcBlockState, Deemphasis, DeemphasisState, QuadDemod, QuadDemodState,
};
use crate::num::{flush_denormal, flush_denormal_f64, q15_from_f64, round_shift, sat16};
use crate::types::{C32, CQ15, Q15};

/// `arg(a * conj(b))` with `arg(0) = 0`, from the (exact or f64) product components.
fn arg_of(re: f64, im: f64) -> f64 {
    if re == 0.0 && im == 0.0 {
        0.0
    } else {
        libm::atan2(im, re)
    }
}

impl QuadDemod<C32> for Reference {
    fn quad_demod(
        &self,
        state: &mut QuadDemodState<C32>,
        input: &[C32],
        output: &mut [f32],
        gain: f32,
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let mut prev = state.prev();
        for (o, &x) in output.iter_mut().zip(input) {
            let (ar, ai) = (f64::from(x.re), f64::from(x.im));
            let (br, bi) = (f64::from(prev.re), f64::from(prev.im));
            let phase = arg_of(ar * br + ai * bi, ai * br - ar * bi);
            *o = (phase * f64::from(gain)) as f32;
            prev = x;
        }
        state.set_prev(prev);
        Ok(())
    }
}

impl QuadDemod<CQ15> for Reference {
    fn quad_demod(
        &self,
        state: &mut QuadDemodState<CQ15>,
        input: &[CQ15],
        output: &mut [Q15],
        gain: Q15,
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let mut prev = state.prev();
        let g = f64::from(gain.0) / 32768.0;
        for (o, &x) in output.iter_mut().zip(input) {
            let (ar, ai) = (i64::from(x.re.0), i64::from(x.im.0));
            let (br, bi) = (i64::from(prev.re.0), i64::from(prev.im.0));
            let phase = arg_of((ar * br + ai * bi) as f64, (ai * br - ar * bi) as f64);
            *o = q15_from_f64(phase / PI * g);
            prev = x;
        }
        state.set_prev(prev);
        Ok(())
    }
}

impl Deemphasis<f32> for Reference {
    fn deemphasis(
        &self,
        state: &mut DeemphasisState<f32>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (alpha, last) = state.parts_mut();
        let a = f64::from(alpha);
        let mut y = *last;
        for (o, &x) in output.iter_mut().zip(input) {
            let prev = f64::from(y);
            y = flush_denormal_f64(prev + a * (f64::from(x) - prev)) as f32;
            *o = y;
        }
        *last = y;
        Ok(())
    }
}

impl Deemphasis<Q15> for Reference {
    fn deemphasis(
        &self,
        state: &mut DeemphasisState<Q15>,
        input: &[Q15],
        output: &mut [Q15],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (alpha, last) = state.parts_mut();
        let mut y = i64::from(last.0);
        for (o, &x) in output.iter_mut().zip(input) {
            let d = i64::from(x.0) - y;
            y = i64::from(sat16(y + round_shift(i64::from(alpha.0) * d, 15)));
            *o = Q15(y as i16);
        }
        *last = Q15(y as i16);
        Ok(())
    }
}

fn dc_f64(r: f32, x: f32, x_prev: f32, y_prev: f32) -> f32 {
    ((f64::from(x) - f64::from(x_prev)) + f64::from(r) * f64::from(y_prev)) as f32
}

fn dc_q15(r: Q15, x: Q15, x_prev: Q15, y_prev: Q15) -> Q15 {
    let diff = i64::from(x.0) - i64::from(x_prev.0);
    let feedback = round_shift(i64::from(r.0) * i64::from(y_prev.0), 15);
    Q15(sat16(diff + feedback))
}

impl DcBlock<C32> for Reference {
    fn dc_block(
        &self,
        state: &mut DcBlockState<C32>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (r, x_prev, y_prev) = state.parts_mut();
        for (o, &x) in output.iter_mut().zip(input) {
            let y = C32::new(
                dc_f64(r, x.re, x_prev.re, y_prev.re),
                dc_f64(r, x.im, x_prev.im, y_prev.im),
            );
            *o = y;
            *x_prev = x;
            *y_prev = C32::new(flush_denormal(y.re), flush_denormal(y.im));
        }
        Ok(())
    }
}

impl DcBlock<CQ15> for Reference {
    fn dc_block(
        &self,
        state: &mut DcBlockState<CQ15>,
        input: &[CQ15],
        output: &mut [CQ15],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (r, x_prev, y_prev) = state.parts_mut();
        for (o, &x) in output.iter_mut().zip(input) {
            let y = CQ15 {
                re: dc_q15(r, x.re, x_prev.re, y_prev.re),
                im: dc_q15(r, x.im, x_prev.im, y_prev.im),
            };
            *o = y;
            *x_prev = x;
            *y_prev = y;
        }
        Ok(())
    }
}
