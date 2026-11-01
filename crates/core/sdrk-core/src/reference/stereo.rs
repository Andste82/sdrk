//! Reference implementation of `stereo_pll` and `stereo_matrix`.
//!
//! The loop is a discrete feedback system, so an f64 model would drift away from the
//! fallback. The specification fixes the arithmetic (f32 for `f32`, exact `i64` for `Q15`)
//! and this reference follows it step by step, written as separate small functions. The
//! interpolation of the cosine tables is an independent copy of the rule of `rotate`.

#![allow(clippy::many_single_char_names)]

use crate::Reference;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::stereo::{StereoMatrix, StereoPll, StereoPllState};
use crate::num::{flush_denormal, round_shift, sat16};
use crate::tables::{COS_F32, COS_Q15};
use crate::types::{C32, CQ15, Q15};

/// The two table entries of a slice of length 2 (the index is below 4096, so the slice exists).
fn pair<X: Copy>(entries: Option<&[X]>, none: X) -> (X, X) {
    match entries {
        Some(&[a, b]) => (a, b),
        _ => (none, none),
    }
}

/// Cosine of the phase from the f32 table, linear interpolation in f32 without FMA.
fn cos_f32(phase: u32) -> f32 {
    let idx = (phase >> 20) as usize;
    let frac = (phase & 0xF_FFFF) as f32 * (1.0 / 1_048_576.0);
    let (a, b) = pair(COS_F32.get(idx..idx + 2), 0.0);
    let diff = b - a;
    a + diff * frac
}

/// Cosine of the phase from the Q15 table: `sat16(a + round_shift((b - a) f15, 15))`.
fn cos_q15(phase: u32) -> i64 {
    let idx = (phase >> 20) as usize;
    let f15 = i64::from((phase >> 5) & 0x7FFF);
    let (a, b) = pair(COS_Q15.get(idx..idx + 2), 0);
    let (a, b) = (i64::from(a), i64::from(b));
    i64::from(sat16(a + round_shift((b - a) * f15, 15)))
}

const QUARTER: u32 = 1 << 30;

/// `(sin theta, cos theta, sin 2 theta)` as three table lookups.
fn trig_f32(phase: u32) -> (f32, f32, f32) {
    let c = cos_f32(phase);
    let s = cos_f32(phase.wrapping_sub(QUARTER));
    let s2 = cos_f32(phase.wrapping_mul(2).wrapping_sub(QUARTER));
    (s, c, s2)
}

fn trig_q15(phase: u32) -> (i64, i64, i64) {
    let c = cos_q15(phase);
    let s = cos_q15(phase.wrapping_sub(QUARTER));
    let s2 = cos_q15(phase.wrapping_mul(2).wrapping_sub(QUARTER));
    (s, c, s2)
}

/// `floor(v + 0.5)` of the phase correction as `u32` through `i64`, as the contract writes
/// it.
fn correction_f32(f: f32) -> u32 {
    let v = f64::from(f) * 4_294_967_296.0 + 0.5;
    libm::floor(v) as i64 as u32
}

/// Lock decision with hysteresis: on at or above `on`, off below `off`.
fn decide<P: PartialOrd + Copy>(was: bool, level: P, on: P, off: P) -> bool {
    if level >= on {
        true
    } else if level < off {
        false
    } else {
        was
    }
}

impl StereoPll<f32> for Reference {
    fn stereo_pll(
        &self,
        state: &mut StereoPllState<f32>,
        mpx: &[f32],
        pilot: &[f32],
        output: &mut [C32],
    ) -> Result<()> {
        check_len(mpx.len(), pilot.len())?;
        check_len(mpx.len(), output.len())?;
        let g = state.parts_mut();
        for ((o, &x), &p) in output.iter_mut().zip(mpx).zip(pilot) {
            let (s, c, s2) = trig_f32(*g.phase);
            let (e, q) = if p.is_finite() {
                (p * c, p * s)
            } else {
                (0.0, 0.0)
            };
            if p.is_finite() {
                let step = g.beta * (q - *g.level);
                *g.level = flush_denormal(*g.level + step);
                *g.stereo = decide(*g.stereo, *g.level, g.on, g.off);
            }
            let diff = if *g.stereo && g.enabled { x * s2 } else { 0.0 };
            *o = C32::new(x, diff);
            let accumulated = flush_denormal(*g.integ + g.ki * e);
            *g.integ = accumulated.clamp(-g.imax, g.imax);
            let f = g.kp * e + *g.integ;
            *g.phase = g
                .phase
                .wrapping_add(g.step0)
                .wrapping_add(correction_f32(f));
        }
        Ok(())
    }
}

impl StereoPll<Q15> for Reference {
    fn stereo_pll(
        &self,
        state: &mut StereoPllState<Q15>,
        mpx: &[Q15],
        pilot: &[Q15],
        output: &mut [CQ15],
    ) -> Result<()> {
        check_len(mpx.len(), pilot.len())?;
        check_len(mpx.len(), output.len())?;
        let g = state.parts_mut();
        for ((o, &x), &p) in output.iter_mut().zip(mpx).zip(pilot) {
            let (s, c, s2) = trig_q15(*g.phase);
            let p = i64::from(p.0);
            let (e, q) = (p * c, p * s);
            let smoothed = round_shift(*g.level_q61, 31);
            *g.level_q61 += g.beta_q31 * (q - smoothed);
            *g.stereo = decide(*g.stereo, *g.level_q61, g.on_q61, g.off_q61);
            let diff = if *g.stereo && g.enabled {
                sat16(round_shift(i64::from(x.0) * s2, 15))
            } else {
                0
            };
            *o = CQ15 {
                re: x,
                im: Q15(diff),
            };
            *g.integ_q = (*g.integ_q + g.ki_q * e).clamp(-g.imax_q, g.imax_q);
            let f = g.kp_q * e + *g.integ_q;
            *g.phase = g
                .phase
                .wrapping_add(g.step0)
                .wrapping_add(round_shift(f, 32) as u32);
        }
        Ok(())
    }
}

impl StereoMatrix<C32> for Reference {
    fn stereo_matrix(&self, input: &[C32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (o, i) in output.iter_mut().zip(input) {
            let d2 = i.im + i.im;
            let left = i.re + d2;
            let right = i.re - d2;
            *o = C32::new(left, right);
        }
        Ok(())
    }
}

impl StereoMatrix<CQ15> for Reference {
    fn stereo_matrix(&self, input: &[CQ15], output: &mut [CQ15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (o, i) in output.iter_mut().zip(input) {
            let (s, d2) = (i64::from(i.re.0), 2 * i64::from(i.im.0));
            *o = CQ15 {
                re: Q15(sat16(s + d2)),
                im: Q15(sat16(s - d2)),
            };
        }
        Ok(())
    }
}
