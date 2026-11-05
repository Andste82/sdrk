//! Fallback implementation of `preemph_limit` (the implementation on every target: the kernel
//! is not accelerated).
//!
//! Both variants are bit-exact with `Reference`; only the form differs.
//!
//! The pre-emphasis is a one-pole recursion on `v1` and the gain of a sample depends on the
//! envelope that this very sample updates, so the work cannot be vectorized without changing the
//! result. What the fallback saves is work per sample:
//!
//! * `f32`: a sample within the input limit passes `x` unchanged; the check is one compare of `|x|`
//!   with `INPUT_LIMIT`, the clamp and the replacement of NaN and infinities are on the rare
//!   path. The clamp of the output is two selects.
//! * `Q15`: the quotient `|v| T_q30 / env` is a `u64` division when the product is below `2^63`
//!   (`|v|` below 8.0 in value terms, the usual case), and a `u128` division only above that. A
//!   64-bit division is much cheaper than a 128-bit one on x86-64 and the soft division of
//!   `thumbv7em`, RISC-V32 and ESP32. The coefficient products stay in `i128` (they exceed `i64`
//!   for the largest coefficients).
//! * Stereo (`C32`): the `f32` loop with both channels, the factor `T / env` once per sample
//!   for both and the larger magnitude as one select.

#![allow(clippy::many_single_char_names)]

use crate::Fallback;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::preemph_limit::{
    INPUT_LIMIT, PreemphLimit, PreemphLimitState, PreemphLimitStereo, PreemphLimitStereoState,
    V_LIMIT,
};
use crate::num::flush_denormal;
use crate::types::{C32, Q15};

/// The input rule: `0.0` for NaN and infinities, a finite value clamped to `+-INPUT_LIMIT`.
/// A value within the limit passes after one compare; the rest is the rare path.
#[inline]
fn sanitize(sample: f32) -> f32 {
    let a = sample.abs();
    if a <= INPUT_LIMIT {
        sample
    } else if a <= f32::MAX {
        if sample < 0.0 {
            -INPUT_LIMIT
        } else {
            INPUT_LIMIT
        }
    } else {
        0.0
    }
}

/// Below this `|v|` (Q30) the product with `T_q30 <= 2^30` fits in 63 bits.
const NARROW: u64 = 1 << 33;

impl PreemphLimit<f32> for Fallback {
    fn preemph_limit(
        &self,
        state: &mut PreemphLimitState<f32>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let g = state.parts_mut();
        let (b0, b1, a1, r, t) = (g.b0_f, g.b1_f, g.a1_f, g.r_f, g.t_f);
        let (mut x1, mut v1, mut env, mut hold) = (*g.x1_f, *g.v1_f, *g.env_f, *g.hold);
        for (o, &sample) in output.iter_mut().zip(input) {
            let x = sanitize(sample);
            let v = (b0 * x + b1 * x1) - a1 * v1;
            x1 = x;
            v1 = flush_denormal(v);
            let m = v.abs();
            if m >= env {
                env = m;
                hold = g.hold_len;
            } else if hold > 0 {
                hold -= 1;
            } else {
                env = flush_denormal(env * r);
            }
            let y = if env > t { v * (t / env) } else { v };
            let y = if y > t { t } else { y };
            *o = if y < -t { -t } else { y };
        }
        *g.x1_f = x1;
        *g.v1_f = v1;
        *g.env_f = env;
        *g.hold = hold;
        Ok(())
    }
}

/// `sign(v) * floor((|v| t + floor(env / 2)) / env)` for `env > t > 0`.
#[inline]
fn scaled(v: i64, t: i64, env: i64) -> i64 {
    let m = v.unsigned_abs();
    let q = if m < NARROW {
        // `m t < 2^63` and `env / 2 < 2^62`, so the sum fits in 64 bits.
        ((m * t as u64 + (env as u64 >> 1)) / env as u64) as i64
    } else {
        ((u128::from(m) * t as u128 + (env as u128 >> 1)) / env as u128) as i64
    };
    if v < 0 { -q } else { q }
}

impl PreemphLimit<Q15> for Fallback {
    fn preemph_limit(
        &self,
        state: &mut PreemphLimitState<Q15>,
        input: &[Q15],
        output: &mut [Q15],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let g = state.parts_mut();
        let (b0, b1, a1) = (i128::from(g.b0_q), i128::from(g.b1_q), i128::from(g.a1_q));
        let (r, t30, t15) = (i128::from(g.r_q), g.t_q30, g.t_q15);
        let (mut x1, mut v1, mut env, mut hold) = (*g.x1_q, *g.v1_q, *g.env_q, *g.hold);
        for (o, &sample) in output.iter_mut().zip(input) {
            let x = i64::from(sample.0);
            let forward = b0 * i128::from(x) + b1 * i128::from(x1);
            let sum = (forward << 15) - a1 * i128::from(v1);
            let v = ((sum + (1 << 23)) >> 24).clamp(-i128::from(V_LIMIT), i128::from(V_LIMIT));
            let v = v as i64;
            x1 = x;
            v1 = v;
            let m = v.abs();
            if m >= env {
                env = m;
                hold = g.hold_len;
            } else if hold > 0 {
                hold -= 1;
            } else {
                env = ((i128::from(env) * r + (1 << 30)) >> 31) as i64;
            }
            let y = if env > t30 { scaled(v, t30, env) } else { v };
            let out = ((y + (1 << 14)) >> 15).clamp(-32768, 32767);
            *o = Q15(out.clamp(-t15, t15) as i16);
        }
        *g.x1_q = x1;
        *g.v1_q = v1;
        *g.env_q = env;
        *g.hold = hold;
        Ok(())
    }
}

impl PreemphLimitStereo for Fallback {
    fn preemph_limit_stereo(
        &self,
        state: &mut PreemphLimitStereoState,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (b0, b1, a1, r, t, hold_len) = (
            state.b0,
            state.b1,
            state.a1,
            state.r,
            state.t,
            state.hold_len,
        );
        let (mut xl1, mut xr1) = (state.x1.re, state.x1.im);
        let (mut vl1, mut vr1) = (state.v1.re, state.v1.im);
        let (mut env, mut hold) = (state.env, state.hold);
        for (o, s) in output.iter_mut().zip(input) {
            let xl = sanitize(s.re);
            let xr = sanitize(s.im);
            let vl = (b0 * xl + b1 * xl1) - a1 * vl1;
            let vr = (b0 * xr + b1 * xr1) - a1 * vr1;
            xl1 = xl;
            xr1 = xr;
            vl1 = flush_denormal(vl);
            vr1 = flush_denormal(vr);
            let (al, ar) = (vl.abs(), vr.abs());
            let m = if al > ar { al } else { ar };
            if m >= env {
                env = m;
                hold = hold_len;
            } else if hold > 0 {
                hold -= 1;
            } else {
                env = flush_denormal(env * r);
            }
            let (yl, yr) = if env > t {
                let g = t / env;
                (vl * g, vr * g)
            } else {
                (vl, vr)
            };
            let yl = if yl > t { t } else { yl };
            let yr = if yr > t { t } else { yr };
            *o = C32::new(if yl < -t { -t } else { yl }, if yr < -t { -t } else { yr });
        }
        state.x1 = C32::new(xl1, xr1);
        state.v1 = C32::new(vl1, vr1);
        state.env = env;
        state.hold = hold;
        Ok(())
    }
}
