//! Reference implementation of `noise_blank`.
//!
//! The impulse decision is discrete, so an f64 model could blank one sample earlier or later
//! than the fallback. The specification therefore fixes the arithmetic (f32 for `C32`, exact
//! integers with a Q61 average for `CQ15`) and this reference follows it step by step, one
//! sample at a time, with the delay line shifted by one sample per input.

use crate::Reference;
use crate::error::{Error, Result};
use crate::kernels::check_len;
use crate::kernels::noise_blank::{NoiseBlank, NoiseBlankParts, NoiseBlankState};
use crate::num::{flush_denormal, round_shift};
use crate::types::{C32, CQ15, Sample};

/// One `C32` step of the average; true for an impulse.
fn power_step_f32(p: &mut f32, alpha: f32, k: f32, x: C32) -> bool {
    let re2 = x.re * x.re;
    let im2 = x.im * x.im;
    let q = re2 + im2;
    let impulse = !q.is_finite() || (*p > 0.0 && q > k * *p);
    if impulse {
        let target = k * *p;
        *p = flush_denormal(*p + alpha * (target - *p));
    } else if *p > 0.0 {
        *p = flush_denormal(*p + alpha * (q - *p));
    } else {
        *p = flush_denormal(q);
    }
    impulse
}

/// One `CQ15` step of the average (`q` Q30, `p` Q61, `alpha` Q31, `k` Q16); true for an
/// impulse.
fn power_step_q61(p: &mut i64, alpha_q31: i64, k_q16: i64, x: CQ15) -> bool {
    let q = i64::from(x.re.0).pow(2) + i64::from(x.im.0).pow(2);
    let p30 = *p >> 31;
    let impulse = p30 > 0 && (i128::from(q) << 16) > i128::from(k_q16) * i128::from(p30);
    if impulse {
        let target = (k_q16 * p30) >> 16;
        *p += alpha_q31 * (target - round_shift(*p, 31));
    } else if p30 > 0 {
        *p += alpha_q31 * (q - round_shift(*p, 31));
    } else {
        *p = q << 31;
    }
    impulse
}

/// Steps (3) and (4) for one sample: the delay line and the counter.
fn delay_and_blank<T: Sample>(g: &mut NoiseBlankParts<'_, T>, impulse: bool, x: T) -> Result<T> {
    if impulse {
        *g.count = g.span;
    }
    let y = if g.hist.is_empty() {
        x
    } else {
        let oldest = *g
            .hist
            .first()
            .ok_or(Error::Internal("noise blank delay line"))?;
        g.hist.rotate_left(1);
        *g.hist
            .last_mut()
            .ok_or(Error::Internal("noise blank delay line"))? = x;
        oldest
    };
    if *g.count > 0 {
        *g.count -= 1;
        *g.blanked = g.blanked.saturating_add(1);
        Ok(T::ZERO)
    } else {
        Ok(y)
    }
}

impl NoiseBlank<C32> for Reference {
    fn noise_blank<H>(
        &self,
        s: &mut NoiseBlankState<C32, H>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()>
    where
        H: AsRef<[C32]> + AsMut<[C32]>,
    {
        check_len(input.len(), output.len())?;
        let mut g = s.parts_mut();
        for (o, &x) in output.iter_mut().zip(input) {
            let impulse = power_step_f32(g.p_f, g.alpha_f, g.k_f, x);
            *o = delay_and_blank(&mut g, impulse, x)?;
        }
        Ok(())
    }
}

impl NoiseBlank<CQ15> for Reference {
    fn noise_blank<H>(
        &self,
        s: &mut NoiseBlankState<CQ15, H>,
        input: &[CQ15],
        output: &mut [CQ15],
    ) -> Result<()>
    where
        H: AsRef<[CQ15]> + AsMut<[CQ15]>,
    {
        check_len(input.len(), output.len())?;
        let mut g = s.parts_mut();
        for (o, &x) in output.iter_mut().zip(input) {
            let impulse = power_step_q61(g.p_q, g.alpha_q, g.k_q, x);
            *o = delay_and_blank(&mut g, impulse, x)?;
        }
        Ok(())
    }
}
