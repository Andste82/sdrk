//! Fallback implementations of `quad_demod`, `deemphasis` and `dc_block`.

use core::f64::consts::PI;

use crate::Fallback;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::demod::{
    DcBlock, DcBlockReal, DcBlockRealState, DcBlockState, Deemphasis, DeemphasisState, QuadDemod,
    QuadDemodState,
};
use crate::num::{flush_denormal, q15_from_f64, q15_mul, round_shift, sat16};
use crate::types::{C32, CQ15, Q15};

impl QuadDemod<C32> for Fallback {
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
            let re = x.re * prev.re + x.im * prev.im;
            let im = x.im * prev.re - x.re * prev.im;
            let arg = if re == 0.0 && im == 0.0 {
                0.0
            } else {
                libm::atan2f(im, re)
            };
            *o = arg * gain;
            prev = x;
        }
        state.set_prev(prev);
        Ok(())
    }
}

impl QuadDemod<CQ15> for Fallback {
    fn quad_demod(
        &self,
        state: &mut QuadDemodState<CQ15>,
        input: &[CQ15],
        output: &mut [Q15],
        gain: Q15,
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let mut prev = state.prev();
        for (o, &x) in output.iter_mut().zip(input) {
            let (ar, ai) = (i64::from(x.re.0), i64::from(x.im.0));
            let (br, bi) = (i64::from(prev.re.0), i64::from(prev.im.0));
            let re = ar * br + ai * bi;
            let im = ai * br - ar * bi;
            let phi = if re == 0 && im == 0 {
                0.0
            } else {
                libm::atan2(im as f64, re as f64) / PI
            };
            *o = q15_mul(q15_from_f64(phi), gain);
            prev = x;
        }
        state.set_prev(prev);
        Ok(())
    }
}

impl Deemphasis<f32> for Fallback {
    fn deemphasis(
        &self,
        state: &mut DeemphasisState<f32>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (alpha, last) = state.parts_mut();
        let mut y = *last;
        for (o, &x) in output.iter_mut().zip(input) {
            y += alpha * (x - y);
            y = flush_denormal(y);
            *o = y;
        }
        *last = y;
        Ok(())
    }
}

impl Deemphasis<Q15> for Fallback {
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

#[inline]
fn dc_f32(r: f32, x: f32, x_prev: f32, y_prev: f32) -> f32 {
    (x - x_prev) + r * y_prev
}

#[inline]
fn dc_q15(r: Q15, x: Q15, x_prev: Q15, y_prev: Q15) -> Q15 {
    let v = i64::from(x.0) - i64::from(x_prev.0)
        + round_shift(i64::from(r.0) * i64::from(y_prev.0), 15);
    Q15(sat16(v))
}

impl DcBlock<C32> for Fallback {
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
                dc_f32(r, x.re, x_prev.re, y_prev.re),
                dc_f32(r, x.im, x_prev.im, y_prev.im),
            );
            *o = y;
            *x_prev = x;
            *y_prev = C32::new(flush_denormal(y.re), flush_denormal(y.im));
        }
        Ok(())
    }
}

impl DcBlock<CQ15> for Fallback {
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

impl DcBlockReal<f32> for Fallback {
    fn dc_block_real(
        &self,
        state: &mut DcBlockRealState<f32>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (r, x_prev, y_prev) = state.parts_mut();
        for (o, &x) in output.iter_mut().zip(input) {
            let y = dc_f32(r, x, *x_prev, *y_prev);
            *o = y;
            *x_prev = x;
            *y_prev = flush_denormal(y);
        }
        Ok(())
    }
}

impl DcBlockReal<Q15> for Fallback {
    fn dc_block_real(
        &self,
        state: &mut DcBlockRealState<Q15>,
        input: &[Q15],
        output: &mut [Q15],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (r, x_prev, y_prev) = state.parts_mut();
        for (o, &x) in output.iter_mut().zip(input) {
            let y = dc_q15(r, x, *x_prev, *y_prev);
            *o = y;
            *x_prev = x;
            *y_prev = y;
        }
        Ok(())
    }
}
