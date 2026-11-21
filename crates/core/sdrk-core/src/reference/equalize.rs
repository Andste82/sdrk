//! Reference implementation of `equalize`.
//!
//! The loop feeds back on itself, so an f64 model would drift away from the fallback. The
//! specification therefore fixes the f32 arithmetic and this reference follows it step by
//! step: the joint history moves sample by sample, the filter part is the reference
//! `dot_complex`, the tracker rotation uses the reference cosine of `costas`, and the weights
//! are read and written by index.

#![allow(clippy::many_single_char_names)]

use crate::Reference;
use crate::error::{Error, Result};
use crate::kernels::check_len;
use crate::kernels::dot::DotComplex;
use crate::kernels::equalize::{
    CMA_CLIP, EqMode, Equalize, EqualizeParts, EqualizeState, MSE_ALPHA, NLMS_EPS, PHASE_MAX_FREQ,
};
use crate::num::flush_denormal;
use crate::reference::costas::{QUARTER, correction, cos_f32, rotate_back};
use crate::types::C32;

/// `v` limited to `[-limit, limit]` (NaN stays NaN).
fn limit(v: f32, l: f32) -> f32 {
    if v > l {
        l
    } else if v < -l {
        -l
    } else {
        v
    }
}

/// `v e^(j theta)` from `c = cos theta`, `s = sin theta`.
fn turn(v: C32, c: f32, s: f32) -> C32 {
    let rc = v.re * c;
    let is = v.im * s;
    let rs = v.re * s;
    let ic = v.im * c;
    C32::new(rc - is, rs + ic)
}

/// Steps 2 to 6 for the finite filter output `y`: returns the output `z` and the value to feed
/// back.
#[allow(clippy::many_single_char_names)]
fn adapt(g: &mut EqualizeParts<'_>, y: C32) -> Result<(C32, C32)> {
    let internal = || Error::Internal("equalizer index");
    let n = g.taps;
    let len = g.weights.len();
    let mut fed = C32::new(0.0, 0.0);
    // Step 2: the tracker rotation, the decision and its error power.
    let (c, sn) = if *g.phase == 0 {
        (1.0, 0.0)
    } else {
        (cos_f32(*g.phase), cos_f32(g.phase.wrapping_sub(QUARTER)))
    };
    let z = rotate_back(y, c, sn);
    let d = g.constellation.decide(z);
    let ed = C32::new(z.re - d.re, z.im - d.im);
    let p = ed.re * ed.re + ed.im * ed.im;
    if p.is_finite() {
        *g.mse = flush_denormal(*g.mse + MSE_ALPHA * (p - *g.mse));
    }
    // Step 3: the error of the mode and the value to feed back.
    let (e, mu) = if *g.mode == EqMode::Cma {
        let m = y.re * y.re + y.im * y.im;
        let f = m - g.constellation.cma_r2();
        let er = limit(y.re * f, CMA_CLIP);
        let ei = limit(y.im * f, CMA_CLIP);
        (C32::new(er, ei), g.mu_cma)
    } else {
        fed = turn(d, c, sn);
        (turn(ed, c, sn), g.mu_dd)
    };
    // Step 4: the normalized step.
    let mut power = 0.0f32;
    let mut forward = 0.0f32;
    for i in 0..len {
        let h = *g.hist.get(i).ok_or_else(internal)?;
        let q = h.re * h.re + h.im * h.im;
        power += q;
        if i + 1 == n {
            forward = power;
        }
    }
    let m = mu / (power + NLMS_EPS);
    let step = C32::new(m * e.re, m * e.im);
    // A forward window of silence adapts nothing and feeds nothing back.
    if forward == 0.0 {
        fed = C32::new(0.0, 0.0);
    }
    // Step 5: the update, all weights or none.
    if forward != 0.0 && step.re.is_finite() && step.im.is_finite() {
        let mut overflow = false;
        for i in 0..len {
            let h = *g.hist.get(i).ok_or_else(internal)?;
            let w = g.weights.get_mut(i).ok_or_else(internal)?;
            // g conj(h) = (g.re h.re + g.im h.im, g.im h.re - g.re h.im)
            let cr = step.re * h.re + step.im * h.im;
            let ci = step.im * h.re - step.re * h.im;
            *w = C32::new(flush_denormal(w.re - cr), flush_denormal(w.im - ci));
            overflow |= !(w.re.is_finite() && w.im.is_finite());
        }
        if overflow {
            for (i, w) in g.weights.iter_mut().enumerate() {
                *w = if i == n / 2 {
                    C32::new(1.0, 0.0)
                } else {
                    C32::new(0.0, 0.0)
                };
            }
        }
    }
    // Step 6: the phase tracker.
    let a = z.im * d.re;
    let b = z.re * d.im;
    let pd = a - b;
    let pd = if pd.is_finite() { pd } else { 0.0 };
    *g.freq = limit(flush_denormal(*g.freq + g.ki * pd), PHASE_MAX_FREQ);
    let v = *g.freq + g.kp * pd;
    *g.phase = g.phase.wrapping_add(correction(v));
    Ok((z, fed))
}

impl Equalize for Reference {
    #[allow(clippy::too_many_lines)]
    fn equalize<W, H>(
        &self,
        s: &mut EqualizeState<W, H>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()>
    where
        W: AsRef<[C32]> + AsMut<[C32]>,
        H: AsRef<[C32]> + AsMut<[C32]>,
    {
        check_len(input.len(), output.len())?;
        let internal = || Error::Internal("equalizer index");
        let mut g = s.parts_mut();
        let n = g.taps;
        let len = g.weights.len();
        for (o, &x) in output.iter_mut().zip(input) {
            // Step 1: the forward history, newest first, and the filter.
            let finite = x.re.is_finite() && x.im.is_finite();
            for i in (1..n).rev() {
                let older = *g.hist.get(i - 1).ok_or_else(internal)?;
                *g.hist.get_mut(i).ok_or_else(internal)? = older;
            }
            *g.hist.get_mut(0).ok_or_else(internal)? = if finite { x } else { C32::new(0.0, 0.0) };
            let y = Reference.dot_complex(g.hist, g.weights)?;
            let mut fed = C32::new(0.0, 0.0);
            if finite && y.re.is_finite() && y.im.is_finite() {
                let (z, f) = adapt(&mut g, y)?;
                fed = f;
                *o = z;
            } else {
                // Step 8: output 0, weights, mse and tracker unchanged.
                *o = C32::new(0.0, 0.0);
            }
            // Step 7: the fed-back decisions, newest first.
            if len > n {
                for i in (n + 1..len).rev() {
                    let older = *g.hist.get(i - 1).ok_or_else(internal)?;
                    *g.hist.get_mut(i).ok_or_else(internal)? = older;
                }
                *g.hist.get_mut(n).ok_or_else(internal)? = fed;
            }
            // Step 9: the mode switch.
            if *g.mode == EqMode::Cma && g.cma_symbols != u32::MAX {
                *g.count = g.count.saturating_add(1);
                if *g.count >= g.cma_symbols {
                    *g.mode = EqMode::Decision;
                }
            }
        }
        Ok(())
    }
}
