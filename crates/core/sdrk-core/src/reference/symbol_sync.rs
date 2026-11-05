//! Reference implementation of `symbol_sync`.
//!
//! The loop feeds back on itself (each strobe position depends on the error of the previous
//! strobe), so an f64 model would drift away from the fallback: the specification fixes the
//! arithmetic (the float32 model of `docs/kernels.md`) and this reference follows it step by
//! step. Unlike the fallback it streams the samples one at a time through a window of the last
//! 4 samples and handles a position when the sample that completes its window arrives.

#![allow(clippy::many_single_char_names)]

use crate::Reference;
use crate::error::{Error, Result};
use crate::kernels::symbol_sync::{HALF, HIST, SIXTH, SymbolSync, SymbolSyncState, THIRD, Ted};
use crate::num::flush_denormal_f64;
use crate::timing::Progress;
use crate::types::{C32, Sample};

fn internal() -> Error {
    Error::Internal("symbol sync reference")
}

/// The cubic Lagrange interpolation of one component.
fn lagrange(xm1: f32, x0: f32, x1: f32, x2: f32, mu: f32) -> f32 {
    let c0 = x0;
    let c1 = -(xm1 * THIRD);
    let c1 = c1 - x0 * HALF;
    let c1 = c1 + x1;
    let c1 = c1 - x2 * SIXTH;
    let c2 = xm1 * HALF;
    let c2 = c2 - x0;
    let c2 = c2 + x1 * HALF;
    let c3 = -(xm1 * SIXTH);
    let c3 = c3 + x0 * HALF;
    let c3 = c3 - x1 * HALF;
    let c3 = c3 + x2 * SIXTH;
    let y = c3 * mu;
    let y = y + c2;
    let y = y * mu;
    let y = y + c1;
    let y = y * mu;
    y + c0
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

/// The model arithmetic per sample type.
trait Model: Sample {
    fn interp(w: [Self; 4], mu: f32) -> Self;
    fn ted_gardner(y_prev: Self, y: Self, m: Self) -> f32;
    fn decision(y: Self) -> Self;
    fn ted_mm(d_prev: Self, y: Self, d: Self, y_prev: Self) -> f32;
}

impl Model for f32 {
    fn interp(w: [f32; 4], mu: f32) -> f32 {
        lagrange(w[0], w[1], w[2], w[3], mu)
    }

    fn ted_gardner(y_prev: f32, y: f32, m: f32) -> f32 {
        let diff = y_prev - y;
        diff * m
    }

    fn decision(y: f32) -> f32 {
        sgn(y)
    }

    fn ted_mm(d_prev: f32, y: f32, d: f32, y_prev: f32) -> f32 {
        let a = d_prev * y;
        let b = d * y_prev;
        a - b
    }
}

impl Model for C32 {
    fn interp(w: [C32; 4], mu: f32) -> C32 {
        let re = lagrange(w[0].re, w[1].re, w[2].re, w[3].re, mu);
        let im = lagrange(w[0].im, w[1].im, w[2].im, w[3].im, mu);
        C32::new(re, im)
    }

    /// `re((y_prev - y) conj(m))`.
    fn ted_gardner(y_prev: C32, y: C32, m: C32) -> f32 {
        let dr = y_prev.re - y.re;
        let di = y_prev.im - y.im;
        let a = dr * m.re;
        let b = di * m.im;
        a + b
    }

    fn decision(y: C32) -> C32 {
        C32::new(sgn(y.re), sgn(y.im))
    }

    /// `re(conj(d_prev) y) - re(conj(d) y_prev)`.
    fn ted_mm(d_prev: C32, y: C32, d: C32, y_prev: C32) -> f32 {
        let a = d_prev.re * y.re;
        let a = a + d_prev.im * y.im;
        let b = d.re * y_prev.re;
        let b = b + d.im * y_prev.im;
        a - b
    }
}

/// A position `i + mu` split at the next lower integer.
fn split(base: usize, v: f64) -> Result<(usize, f64)> {
    let f = libm::floor(v);
    let i = base.checked_add_signed(f as isize).ok_or_else(internal)?;
    Ok((i, v - f))
}

/// The position the loop needs next: the mid sample of the next strobe (Gardner, once a strobe
/// has run, until it is computed), else the strobe.
fn next_needed<T: Model>(s: &SymbolSyncState<T>) -> Result<(bool, usize, f64)> {
    if s.ted == Ted::Gardner && s.started && !s.mid_ready {
        let (i, mu) = split(s.i, s.mu - s.omega * 0.5)?;
        Ok((true, i, mu))
    } else {
        Ok((false, s.i, s.mu))
    }
}

fn run<T: Model>(s: &mut SymbolSyncState<T>, input: &[T], output: &mut [T]) -> Result<Progress> {
    // `b[j]`: the history for `j < 3`, then the input.
    let b = |hist: &[T; HIST], j: usize| -> Result<T> {
        if j < HIST {
            hist.get(j).copied().ok_or_else(internal)
        } else {
            input.get(j - HIST).copied().ok_or_else(internal)
        }
    };
    let [h0, h1, h2] = s.hist;
    let mut win = [T::ZERO, h0, h1, h2];
    let mut produced = 0usize;
    'samples: for (j, &x) in input.iter().enumerate() {
        // The window ends at `b[k]`.
        let k = j + HIST;
        win = [win[1], win[2], win[3], x];
        loop {
            if produced == output.len() {
                break 'samples;
            }
            let (is_mid, i, mu) = next_needed(s)?;
            if i + 2 > k {
                break;
            }
            if i + 2 != k {
                return Err(internal());
            }
            let v = T::interp(win, mu as f32);
            if is_mid {
                s.mid = v;
                s.mid_ready = true;
                continue;
            }
            let y = v;
            let mut e = 0.0f32;
            match s.ted {
                Ted::Gardner => {
                    if s.started {
                        e = T::ted_gardner(s.y_prev, y, s.mid);
                    }
                    s.mid_ready = false;
                }
                Ted::MuellerMuller => {
                    let d = T::decision(y);
                    if s.started {
                        e = T::ted_mm(s.d_prev, y, d, s.y_prev);
                    }
                    s.d_prev = d;
                }
            }
            if !e.is_finite() {
                e = 0.0;
            }
            s.y_prev = y;
            s.started = true;
            *output.get_mut(produced).ok_or_else(internal)? = y;
            produced += 1;
            let e = f64::from(e);
            let integ = flush_denormal_f64(s.integ + s.ki * e);
            s.integ = integ.max(-s.lim).min(s.lim);
            s.omega = s.sps + s.integ;
            let step = (s.kp * e).max(-(s.omega * 0.25)).min(s.omega * 0.25);
            let (i, mu) = split(s.i, (s.mu + s.omega) + step)?;
            s.i = i;
            s.mu = mu;
        }
    }
    let (_, pending, _) = next_needed(s)?;
    let k = pending
        .checked_sub(1)
        .ok_or_else(internal)?
        .min(input.len());
    let old = s.hist;
    s.hist = [b(&old, k)?, b(&old, k + 1)?, b(&old, k + 2)?];
    s.i -= k;
    Ok(Progress {
        consumed: k,
        produced,
    })
}

impl SymbolSync<C32> for Reference {
    fn symbol_sync(
        &self,
        s: &mut SymbolSyncState<C32>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<Progress> {
        run(s, input, output)
    }
}

impl SymbolSync<f32> for Reference {
    fn symbol_sync(
        &self,
        s: &mut SymbolSyncState<f32>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<Progress> {
        run(s, input, output)
    }
}
