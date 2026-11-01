//! Reference implementation of `preemph_limit`.
//!
//! The limiter compares and switches, so an f64 model could limit one sample earlier or later
//! than the fallback. The specification therefore fixes the arithmetic (float32 without FMA for
//! `f32`, `i64` with `i128` products for `Q15`) and this reference follows it step by step, in
//! separate small functions and without any shortcut.

use crate::Reference;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::preemph_limit::{PreemphLimit, PreemphLimitParts, PreemphLimitState, V_LIMIT};
use crate::num::{flush_denormal, round_shift, sat16};
use crate::types::Q15;

/// Step 1: a non-finite input counts as zero.
fn sanitize(x: f32) -> f32 {
    if x.is_nan() || x.is_infinite() {
        0.0
    } else {
        x
    }
}

/// Step 3 for `f32`: instant attack, hold, release. Returns `(env, hold)`.
fn envelope_f32(m: f32, env: f32, hold: u32, hold_len: u32, r: f32) -> (f32, u32) {
    if m >= env {
        (m, hold_len)
    } else if hold > 0 {
        (env, hold - 1)
    } else {
        (flush_denormal(env * r), hold)
    }
}

/// Step 4 for `f32`: the gain applied to `v`, then the clamp to `[-T, T]`.
fn limit_f32(v: f32, env: f32, t: f32) -> f32 {
    let y = if env > t { v * (t / env) } else { v };
    if y > t {
        t
    } else if y < -t {
        -t
    } else {
        y
    }
}

/// `round_shift` of an `i128` value: add half, arithmetic shift.
fn round_shift_wide(x: i128, s: u32) -> i128 {
    (x + (1_i128 << (s - 1))) >> s
}

/// Step 2 for `Q15`: the pre-emphasized value in Q30, saturated to `+-2^62`.
fn pre_emphasis_q15(g: &PreemphLimitParts<'_>, x: i64) -> i64 {
    let forward = i128::from(g.b0_q) * i128::from(x) + i128::from(g.b1_q) * i128::from(*g.x1_q);
    let sum = forward * (1_i128 << 15) - i128::from(g.a1_q) * i128::from(*g.v1_q);
    let v = round_shift_wide(sum, 24);
    v.clamp(-i128::from(V_LIMIT), i128::from(V_LIMIT)) as i64
}

/// Step 3 for `Q15`. Returns `(env, hold)`.
fn envelope_q15(m: i64, env: i64, hold: u32, hold_len: u32, r_q31: i64) -> (i64, u32) {
    if m >= env {
        (m, hold_len)
    } else if hold > 0 {
        (env, hold - 1)
    } else {
        let released = round_shift_wide(i128::from(env) * i128::from(r_q31), 31);
        (released as i64, hold)
    }
}

/// `sign(n) * floor((|n| + floor(d / 2)) / d)` for `d > 0`.
fn div_round(n: i128, d: i128) -> i128 {
    let q = (n.abs() + d / 2) / d;
    if n < 0 { -q } else { q }
}

/// Step 4 for `Q15`: the gain, the rounding to Q15 and the clamp to `+-T_q15`.
fn limit_q15(g: &PreemphLimitParts<'_>, v: i64, env: i64) -> i16 {
    let y = if env > g.t_q30 {
        div_round(i128::from(v) * i128::from(g.t_q30), i128::from(env)) as i64
    } else {
        v
    };
    let out = i64::from(sat16(round_shift(y, 15)));
    out.clamp(-g.t_q15, g.t_q15) as i16
}

impl PreemphLimit<f32> for Reference {
    fn preemph_limit(
        &self,
        state: &mut PreemphLimitState<f32>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let g = state.parts_mut();
        for (o, &sample) in output.iter_mut().zip(input) {
            let x = sanitize(sample);
            let forward = g.b0_f * x + g.b1_f * *g.x1_f;
            let v = forward - g.a1_f * *g.v1_f;
            *g.x1_f = x;
            *g.v1_f = flush_denormal(v);
            let (env, hold) = envelope_f32(v.abs(), *g.env_f, *g.hold, g.hold_len, g.r_f);
            *g.env_f = env;
            *g.hold = hold;
            *o = limit_f32(v, env, g.t_f);
        }
        Ok(())
    }
}

impl PreemphLimit<Q15> for Reference {
    fn preemph_limit(
        &self,
        state: &mut PreemphLimitState<Q15>,
        input: &[Q15],
        output: &mut [Q15],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let g = state.parts_mut();
        for (o, x) in output.iter_mut().zip(input) {
            let x = i64::from(x.0);
            let v = pre_emphasis_q15(&g, x);
            *g.x1_q = x;
            *g.v1_q = v;
            let (env, hold) = envelope_q15(v.abs(), *g.env_q, *g.hold, g.hold_len, g.r_q);
            *g.env_q = env;
            *g.hold = hold;
            *o = Q15(limit_q15(&g, v, env));
        }
        Ok(())
    }
}
