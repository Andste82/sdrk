//! Fallback implementation of `auto_notch`.
//!
//! Bit-exact with `Reference`: the sums run in the same ascending order from `+0.0` and the
//! update uses the same factor. The reference vector `u[i] = x[n - delay - i]` occupies the
//! ring positions `pos + 1` to `pos + L` (mod `L + delay`) in reverse order, so it is read as
//! at most two contiguous slices, each backwards, instead of with one modulo per tap.

#![allow(clippy::many_single_char_names)]

use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::auto_notch::{AutoNotch, AutoNotchState};
use crate::kernels::check_len;
use crate::num::flush_denormal;

/// The prediction `y = Σ w[i] u[i]` and the power `P = Σ u[i]^2` continued over one segment,
/// where `u` runs backwards through `seg`.
#[inline]
fn sums(w: &[f32], seg: &[f32], mut y: f32, mut power: f32) -> (f32, f32) {
    for (&wi, &ui) in w.iter().zip(seg.iter().rev()) {
        y += wi * ui;
        power += ui * ui;
    }
    (y, power)
}

/// `w[i] = flush_denormal(w[i] + f u[i])` over one segment, `u` backwards through `seg`.
#[inline]
fn update(w: &mut [f32], seg: &[f32], f: f32) {
    for (wi, &ui) in w.iter_mut().zip(seg.iter().rev()) {
        *wi = flush_denormal(*wi + f * ui);
    }
}

impl AutoNotch for Fallback {
    fn auto_notch<W, X>(
        &self,
        s: &mut AutoNotchState<W, X>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()>
    where
        W: AsRef<[f32]> + AsMut<[f32]>,
        X: AsRef<[f32]> + AsMut<[f32]>,
    {
        check_len(input.len(), output.len())?;
        let internal = || Error::Internal("auto notch ring");
        let g = s.parts_mut();
        let (weights, hist, mu) = (g.weights, g.hist, g.mu);
        let taps = weights.len();
        let m = hist.len();
        let mut pos = *g.pos;
        for (o, &raw) in output.iter_mut().zip(input) {
            let finite = raw.is_finite();
            let x = if finite { raw } else { 0.0 };
            // The window in memory: `hist[start..start + taps]` (mod m). `u` is the window
            // backwards: first the wrapped part `hist[..wrap]` backwards, then
            // `hist[start..]` backwards.
            let start = if pos + 1 == m { 0 } else { pos + 1 };
            let wrap = (start + taps).saturating_sub(m);
            let (low, high) = if wrap == 0 {
                let high = hist.get(start..start + taps).ok_or_else(internal)?;
                (&[][..], high)
            } else {
                (
                    hist.get(..wrap).ok_or_else(internal)?,
                    hist.get(start..).ok_or_else(internal)?,
                )
            };
            let (w_low, w_high) = weights
                .split_at_mut_checked(low.len())
                .ok_or_else(internal)?;
            let (y, power) = sums(w_low, low, 0.0, 0.0);
            let (y, power) = sums(w_high, high, y, power);
            if finite {
                let e = x - y;
                let f = (mu * e) / (power + 1e-6);
                // A non-finite factor (a huge finite input) leaves the weights unchanged and a
                // non-finite error gives +0.0, so the state stays finite (D422).
                if f.is_finite() {
                    update(w_low, low, f);
                    update(w_high, high, f);
                }
                *o = if e.is_finite() { e } else { 0.0 };
            } else {
                *o = 0.0;
            }
            *hist.get_mut(pos).ok_or_else(internal)? = x;
            pos = if pos + 1 == m { 0 } else { pos + 1 };
        }
        *g.pos = pos;
        Ok(())
    }
}
