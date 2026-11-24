//! Fallback implementation of `symbol_sync`.
//!
//! Bit-exact with `Reference`. The strobes read their interpolation windows straight from the
//! input slice (no copy of the input); only a window that reaches into the 3 history samples is
//! put together sample by sample. The loop is a recursion over the strobes (each position
//! depends on the error of the previous strobe), so the kernel is not vectorized; per symbol it
//! costs one interpolation (two for Gardner) and a few f64 operations of the loop filter.

#![allow(clippy::many_single_char_names)]

use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::symbol_sync::{HIST, SymbolSync, SymbolSyncState, SyncSample, Ted};
use crate::num::{floor_f64, flush_denormal_f64};
use crate::timing::Progress;
use crate::types::C32;

fn internal() -> Error {
    Error::Internal("symbol sync window")
}

/// `b[j]` of the samples `b` = history followed by input.
#[inline]
fn at<T: SyncSample>(hist: &[T; HIST], input: &[T], j: usize) -> Result<T> {
    if j < HIST {
        hist.get(j).copied().ok_or_else(internal)
    } else {
        input.get(j - HIST).copied().ok_or_else(internal)
    }
}

/// The window `b[i-1] .. b[i+2]` of the position with integer part `i` (`i >= 1`).
#[inline]
fn window<T: SyncSample>(hist: &[T; HIST], input: &[T], i: usize) -> Result<[T; 4]> {
    let start = i.checked_sub(1).ok_or_else(internal)?;
    if start >= HIST {
        input
            .get(start - HIST..)
            .and_then(<[T]>::first_chunk::<4>)
            .copied()
            .ok_or_else(internal)
    } else {
        Ok([
            at(hist, input, start)?,
            at(hist, input, start + 1)?,
            at(hist, input, start + 2)?,
            at(hist, input, start + 3)?,
        ])
    }
}

/// The position `t - omega/2` of the mid sample of the strobe at `i + mu` as
/// `(integer part, fraction)`.
#[inline]
fn mid_position(i: usize, mu: f64, omega: f64) -> Result<(usize, f64)> {
    let v = mu - omega * 0.5;
    // `-omega / 2 < v < 1` and `v` is never `-0.0` (`mu >= +0.0`), so the floor is the
    // truncation, one less when it is above `v` (the usual case, `v` is negative).
    #[allow(clippy::cast_possible_truncation, reason = "|v| < 2^52")]
    #[allow(clippy::cast_precision_loss, reason = "an integer below 2^52")]
    let (fi, f) = if v.abs() < 4_503_599_627_370_496.0 {
        let ti = v as i64;
        let t = ti as f64;
        if t > v { (ti - 1, t - 1.0) } else { (ti, t) }
    } else {
        let f = floor_f64(v);
        (f as i64, f)
    };
    let i = i
        .checked_add_signed(isize::try_from(fi).map_err(|_| internal())?)
        .ok_or_else(internal)?;
    Ok((i, v - f))
}

fn run<T: SyncSample>(
    s: &mut SymbolSyncState<T>,
    input: &[T],
    output: &mut [T],
) -> Result<Progress> {
    let n = input.len();
    let len = n + HIST;
    let gardner = s.ted == Ted::Gardner;
    // The state of the recursion in locals, written back once: the strobe loop is one chain
    // from symbol to symbol, and the state in registers keeps stores and loads off it.
    let hist = s.hist;
    let (sps, kp, ki, lim) = (s.sps, s.kp, s.ki, s.lim);
    let q_min = (sps + (-lim)) * 0.25;
    let (mut i, mut mu, mut omega, mut integ) = (s.i, s.mu, s.omega, s.integ);
    let (mut y_prev, mut d_prev, mut mid) = (s.y_prev, s.d_prev, s.mid);
    let (mut mid_ready, mut started) = (s.mid_ready, s.started);
    let mut produced = 0usize;
    for o in output.iter_mut() {
        if gardner && started && !mid_ready {
            let (im, f) = mid_position(i, mu, omega)?;
            if im + 2 >= len {
                break;
            }
            mid = T::interp(&window(&hist, input, im)?, f as f32);
            mid_ready = true;
        }
        if i + 2 >= len {
            break;
        }
        let y = T::interp(&window(&hist, input, i)?, mu as f32);
        let e = if gardner {
            let e = if started {
                T::gardner(y_prev, y, mid)
            } else {
                0.0
            };
            mid_ready = false;
            e
        } else {
            let d = T::decide(y);
            let e = if started {
                T::mm(d_prev, y, d, y_prev)
            } else {
                0.0
            };
            d_prev = d;
            e
        };
        y_prev = y;
        started = true;
        *o = y;
        produced += 1;
        let e = if e.is_finite() { f64::from(e) } else { 0.0 };
        // The clamps as written, `x.max(-c).min(c)`, give `x` itself whenever `|x| <= c` (the
        // values are finite); the test is a branch that is almost always taken the same way,
        // so the clamps leave the chain from strobe to strobe. `q >= q_min` because
        // `omega = sps + integ >= sps + (-lim)` (rounding is monotone; `* 0.25` is exact).
        // `flush_denormal_f64` gives `x` itself when `|x| >= 1e-30`: one test for both.
        let x = integ + ki * e;
        let a = x.abs();
        integ = if a >= 1e-30 && a <= lim {
            x
        } else {
            flush_denormal_f64(x).max(-lim).min(lim)
        };
        omega = sps + integ;
        let p = kp * e;
        let p = if p.abs() <= q_min {
            p
        } else {
            let q = omega * 0.25;
            p.max(-q).min(q)
        };
        let v = (mu + omega) + p;
        // `v >= mu + 0.75 omega > 0`, so its floor is the truncation.
        #[allow(clippy::cast_possible_truncation, reason = "0 < v < 2^52")]
        #[allow(clippy::cast_precision_loss, reason = "an integer below 2^52")]
        let (fi, f) = if v < 4_503_599_627_370_496.0 {
            let ti = v as u64;
            (ti, ti as f64)
        } else {
            let f = floor_f64(v);
            (f as u64, f)
        };
        i = i
            .checked_add(usize::try_from(fi).map_err(|_| internal())?)
            .ok_or_else(internal)?;
        mu = v - f;
    }
    // The next position the loop needs decides what stays.
    let pending = if gardner && started && !mid_ready {
        mid_position(i, mu, omega)?.0
    } else {
        i
    };
    let k = pending.checked_sub(1).ok_or_else(internal)?.min(n);
    s.hist = [
        at(&hist, input, k)?,
        at(&hist, input, k + 1)?,
        at(&hist, input, k + 2)?,
    ];
    s.i = i - k;
    s.mu = mu;
    s.omega = omega;
    s.integ = integ;
    s.y_prev = y_prev;
    s.d_prev = d_prev;
    s.mid = mid;
    s.mid_ready = mid_ready;
    s.started = started;
    Ok(Progress {
        consumed: k,
        produced,
    })
}

impl SymbolSync<C32> for Fallback {
    fn symbol_sync(
        &self,
        s: &mut SymbolSyncState<C32>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<Progress> {
        run(s, input, output)
    }
}

impl SymbolSync<f32> for Fallback {
    fn symbol_sync(
        &self,
        s: &mut SymbolSyncState<f32>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<Progress> {
        run(s, input, output)
    }
}
