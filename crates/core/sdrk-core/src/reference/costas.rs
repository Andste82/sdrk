//! Reference implementation of `costas`.
//!
//! The loop is a discrete feedback system, so an f64 model would drift away from the
//! fallback. The specification fixes the arithmetic (the float32 model of `docs/kernels.md`) and
//! this reference follows it step by step, written as separate small functions. The
//! interpolation of the cosine table is an independent copy of the rule of `rotate`.

#![allow(clippy::many_single_char_names)]

use crate::Reference;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::costas::{Costas, CostasState};
use crate::num::flush_denormal;
use crate::tables::COS_F32;
use crate::types::C32;

const QUARTER: u32 = 1 << 30;

/// Cosine of the phase from the f32 table, linear interpolation in f32 without FMA.
fn cos_f32(phase: u32) -> f32 {
    let idx = (phase >> 20) as usize;
    let frac = (phase & 0xF_FFFF) as f32 * (1.0 / 1_048_576.0);
    let (a, b) = match COS_F32.get(idx..idx + 2) {
        Some(&[a, b]) => (a, b),
        _ => (0.0, 0.0),
    };
    let diff = b - a;
    a + diff * frac
}

fn sgn(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `x exp(-j phi)` with `c = cos phi`, `s = sin phi`.
fn rotate_back(x: C32, c: f32, s: f32) -> C32 {
    let rr = x.re * c;
    let ii = x.im * s;
    let re = rr + ii;
    let ic = x.im * c;
    let rs = x.re * s;
    let im = ic - rs;
    C32::new(re, im)
}

fn phase_error(order: u32, y: C32) -> f32 {
    let e = if order == 2 {
        y.re * y.im
    } else {
        let a = sgn(y.re) * y.im;
        let b = sgn(y.im) * y.re;
        a - b
    };
    if e.is_finite() { e } else { 0.0 }
}

fn metric(order: u32, y: C32) -> f32 {
    let re2 = y.re * y.re;
    let im2 = y.im * y.im;
    let power = re2 + im2;
    let diff = re2 - im2;
    if order == 2 {
        diff / power
    } else {
        let four_re2 = 4.0 * re2;
        let cross = four_re2 * im2;
        let diff2 = diff * diff;
        let num = cross - diff2;
        let den = power * power;
        num / den
    }
}

/// `floor(v 2^32 + 0.5)` of the phase correction as `u32` through `i64`, as the contract
/// writes it.
fn correction(v: f32) -> u32 {
    let w = f64::from(v) * 4_294_967_296.0 + 0.5;
    libm::floor(w) as i64 as u32
}

impl Costas for Reference {
    fn costas(&self, s: &mut CostasState, input: &[C32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (o, &x) in output.iter_mut().zip(input) {
            let c = cos_f32(s.phase);
            let sn = cos_f32(s.phase.wrapping_sub(QUARTER));
            let y = rotate_back(x, c, sn);
            *o = y;
            let e = phase_error(s.order, y);
            let integrated = flush_denormal(s.freq + s.ki * e);
            s.freq = integrated.clamp(-s.max, s.max);
            let step = s.kp * e;
            let v = s.freq + step;
            s.phase = s.phase.wrapping_add(correction(v));
            let m = metric(s.order, y);
            if m.is_finite() {
                let delta = m - s.level;
                s.level = flush_denormal(s.level + 0.01 * delta);
                if s.level > 0.5 {
                    s.locked = true;
                } else if s.level < 0.3 {
                    s.locked = false;
                }
            }
        }
        Ok(())
    }
}
