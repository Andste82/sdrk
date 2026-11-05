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
use crate::num::flush_denormal_f64;
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

/// The position `t - omega/2` of the mid sample as `(integer part, fraction)`.
#[inline]
fn mid_position<T: SyncSample>(s: &SymbolSyncState<T>) -> Result<(usize, f64)> {
    let v = s.mu - s.omega * 0.5;
    let f = libm::floor(v);
    let i = s.i.checked_add_signed(f as isize).ok_or_else(internal)?;
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
    let mut produced = 0usize;
    for o in output.iter_mut() {
        if gardner && s.started && !s.mid_ready {
            let (im, mu) = mid_position(s)?;
            if im + 2 >= len {
                break;
            }
            s.mid = T::interp(&window(&s.hist, input, im)?, mu as f32);
            s.mid_ready = true;
        }
        if s.i + 2 >= len {
            break;
        }
        let y = T::interp(&window(&s.hist, input, s.i)?, s.mu as f32);
        let e = if gardner {
            let e = if s.started {
                T::gardner(s.y_prev, y, s.mid)
            } else {
                0.0
            };
            s.mid_ready = false;
            e
        } else {
            let d = T::decide(y);
            let e = if s.started {
                T::mm(s.d_prev, y, d, s.y_prev)
            } else {
                0.0
            };
            s.d_prev = d;
            e
        };
        s.y_prev = y;
        s.started = true;
        *o = y;
        produced += 1;
        let e = if e.is_finite() { f64::from(e) } else { 0.0 };
        s.integ = flush_denormal_f64(s.integ + s.ki * e)
            .max(-s.lim)
            .min(s.lim);
        s.omega = s.sps + s.integ;
        let q = s.omega * 0.25;
        let v = (s.mu + s.omega) + (s.kp * e).max(-q).min(q);
        let f = libm::floor(v);
        s.i = s.i.checked_add(f as usize).ok_or_else(internal)?;
        s.mu = v - f;
    }
    // The next position the loop needs decides what stays.
    let pending = if gardner && s.started && !s.mid_ready {
        mid_position(s)?.0
    } else {
        s.i
    };
    let k = pending.checked_sub(1).ok_or_else(internal)?.min(n);
    s.hist = [
        at(&s.hist, input, k)?,
        at(&s.hist, input, k + 1)?,
        at(&s.hist, input, k + 2)?,
    ];
    s.i -= k;
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
