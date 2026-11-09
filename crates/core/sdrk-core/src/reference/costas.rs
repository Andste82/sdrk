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
use crate::kernels::constellation::Qam;
use crate::kernels::costas::{Costas, CostasConstellation, CostasState};
use crate::num::flush_denormal;
use crate::tables::COS_F32;
use crate::types::C32;

pub(super) const QUARTER: u32 = 1 << 30;

/// Cosine of the phase from the f32 table, linear interpolation in f32 without FMA.
pub(super) fn cos_f32(phase: u32) -> f32 {
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
pub(super) fn rotate_back(x: C32, c: f32, s: f32) -> C32 {
    let rr = x.re * c;
    let ii = x.im * s;
    let re = rr + ii;
    let ic = x.im * c;
    let rs = x.re * s;
    let im = ic - rs;
    C32::new(re, im)
}

/// The 8PSK points `exp(j (2k + 1) pi / 8)`, each part rounded once to `f32`.
#[allow(
    clippy::excessive_precision,
    reason = "the literals are the shortest decimals of the f32 values"
)]
const POINTS8: [(f32, f32); 8] = [
    (0.923_879_5, 0.382_683_43),
    (0.382_683_43, 0.923_879_5),
    (-0.382_683_43, 0.923_879_5),
    (-0.923_879_5, 0.382_683_43),
    (-0.923_879_5, -0.382_683_43),
    (-0.382_683_43, -0.923_879_5),
    (0.382_683_43, -0.923_879_5),
    (0.923_879_5, -0.382_683_43),
];

/// The index `k` of the half-open octant `[k pi / 4, (k + 1) pi / 4)` of `y`, written as the
/// contract states it: the quadrant from the signs, then the comparison of the magnitudes;
/// zero and NaN fall to 0.
fn octant(y: C32) -> usize {
    let (ar, ai) = (y.re.abs(), y.im.abs());
    let q = if y.re > 0.0 && y.im >= 0.0 {
        0
    } else if y.re <= 0.0 && y.im > 0.0 {
        1
    } else if y.re < 0.0 && y.im <= 0.0 {
        2
    } else if y.re >= 0.0 && y.im < 0.0 {
        3
    } else {
        return 0;
    };
    // In quadrants 0 and 2 the first octant is the one nearer the real axis, in 1 and 3 the
    // one nearer the imaginary axis.
    let first = if q % 2 == 0 { ar > ai } else { ai > ar };
    2 * q + usize::from(!first)
}

/// The point of the octant of `y`.
fn point8(y: C32) -> (f32, f32) {
    POINTS8.get(octant(y)).copied().unwrap_or((0.0, 0.0))
}

/// The grid level of one component, written as the contract states it: the floor of
/// `v (sqrt(E) / 2) + m / 2` clamped to `0 .. m - 1` (NaN to 0), as the level value of the
/// unit-power grid.
fn grid_level(q: Qam, v: f32) -> f32 {
    let levels = q.levels();
    let m = levels.len() as f32;
    let half_sqrt_e = (libm::sqrt(q.energy()) / 2.0) as f32;
    let t = v * half_sqrt_e + m / 2.0;
    let t = if t.is_nan() {
        0.0
    } else {
        t.clamp(0.0, m - 1.0)
    };
    levels.get(libm::floorf(t) as usize).copied().unwrap_or(0.0)
}

/// The QAM decision `d` of `y` and the parts `(re, im)` of `y conj d`.
fn qam_conj(q: Qam, y: C32) -> ((f32, f32), (f32, f32)) {
    let dr = grid_level(q, y.re);
    let di = grid_level(q, y.im);
    let rr = y.re * dr;
    let ii = y.im * di;
    let ir = y.im * dr;
    let ri = y.re * di;
    ((dr, di), (rr + ii, ir - ri))
}

fn phase_error(c: CostasConstellation, y: C32) -> f32 {
    if let Some(q) = c.qam() {
        let (_, (r, i)) = qam_conj(q, y);
        let e = libm::atan2f(i, r);
        return if e.is_finite() { e } else { 0.0 };
    }
    let e = match c.psk_order().unwrap_or(2) {
        2 => y.re * y.im,
        4 => {
            let a = sgn(y.re) * y.im;
            let b = sgn(y.im) * y.re;
            a - b
        }
        _ => {
            let (dr, di) = point8(y);
            let a = y.im * dr;
            let b = y.re * di;
            a - b
        }
    };
    if e.is_finite() { e } else { 0.0 }
}

fn metric(c: CostasConstellation, y: C32) -> f32 {
    let re2 = y.re * y.re;
    let im2 = y.im * y.im;
    let power = re2 + im2;
    let diff = re2 - im2;
    if let Some(q) = c.qam() {
        let ((dr, di), (r, _)) = qam_conj(q, y);
        let d2 = dr * dr + di * di;
        let r2 = r * r;
        let den = power * d2;
        let ratio = r2 / den;
        let twice = ratio * 2.0;
        let u = twice - 1.0;
        let w = u * u;
        let eight_w = 8.0 * w;
        let w_less_one = w - 1.0;
        let prod = eight_w * w_less_one;
        return prod + 1.0;
    }
    match c.psk_order().unwrap_or(2) {
        2 => diff / power,
        4 => {
            let four_re2 = 4.0 * re2;
            let cross = four_re2 * im2;
            let diff2 = diff * diff;
            let num = cross - diff2;
            let den = power * power;
            num / den
        }
        _ => {
            let (dr, di) = point8(y);
            let rr = y.re * dr;
            let ii = y.im * di;
            let r = rr + ii;
            let r2 = r * r;
            let ratio = r2 / power;
            let twice = ratio * 2.0;
            // u = cos(2 err); the metric is T4(u) = 8 u^4 - 8 u^2 + 1 = cos(8 err), written
            // as ((8 w) (w - 1)) + 1 with w = u u.
            let u = twice - 1.0;
            let w = u * u;
            let eight_w = 8.0 * w;
            let w_less_one = w - 1.0;
            let prod = eight_w * w_less_one;
            prod + 1.0
        }
    }
}

/// `floor(v 2^32 + 0.5)` of the phase correction as `u32` through `i64`, as the contract
/// writes it.
pub(super) fn correction(v: f32) -> u32 {
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
            let e = phase_error(s.constellation, y);
            let integrated = flush_denormal(s.freq + s.ki * e);
            s.freq = integrated.clamp(-s.max, s.max);
            let step = s.kp * e;
            let v = s.freq + step;
            s.phase = s.phase.wrapping_add(correction(v));
            let m = metric(s.constellation, y);
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
