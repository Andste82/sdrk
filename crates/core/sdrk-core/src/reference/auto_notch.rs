//! Reference implementation of `auto_notch`.
//!
//! The loop feeds back on itself, so an f64 model would drift away from the fallback. The
//! specification therefore fixes the f32 arithmetic and this reference follows it step by
//! step, reading the reference vector from the ring sample by sample.

#![allow(clippy::many_single_char_names)]

use crate::Reference;
use crate::error::{Error, Result};
use crate::kernels::auto_notch::{AutoNotch, AutoNotchState};
use crate::kernels::check_len;
use crate::num::flush_denormal;

impl AutoNotch for Reference {
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
        let taps = g.weights.len();
        let m = g.hist.len();
        for (o, &raw) in output.iter_mut().zip(input) {
            // The ring holds x[n - m] (at `pos`) to x[n - 1]; u[i] = x[n - delay - i] lies at
            // `pos + taps - i` (mod m), because m - delay = taps.
            let u = |i: usize| -> Result<f32> {
                let at = (*g.pos + taps - i) % m;
                g.hist.get(at).copied().ok_or_else(internal)
            };
            let finite = raw.is_finite();
            let x = if finite { raw } else { 0.0 };
            let mut y = 0.0f32;
            let mut power = 0.0f32;
            for (i, w) in g.weights.iter().enumerate() {
                let ui = u(i)?;
                y += w * ui;
                power += ui * ui;
            }
            if finite {
                let e = x - y;
                let f = (g.mu * e) / (power + 1e-6);
                // A non-finite factor leaves the weights unchanged; a non-finite error gives
                // +0.0 (D422).
                if f.is_finite() {
                    for i in 0..taps {
                        let ui = u(i)?;
                        let w = g.weights.get_mut(i).ok_or_else(internal)?;
                        *w = flush_denormal(*w + f * ui);
                    }
                }
                *o = if e.is_finite() { e } else { 0.0 };
            } else {
                *o = 0.0;
            }
            *g.hist.get_mut(*g.pos).ok_or_else(internal)? = x;
            *g.pos = (*g.pos + 1) % m;
        }
        Ok(())
    }
}
