//! Reference implementation of `channelize`: the f64 model of the filter bank.
//!
//! Per hop, with the window `w` of the newest `M L` samples (oldest first), the time-reversed
//! taps `g` (stored twice each, `g[i]` at `2 i`) and the absolute index `n_i` of `w[i]`
//! (`n_i mod M = (i + s) mod M`, `s = M / 2` for an even output index and 0 for an odd one),
//! channel `k` is
//! `sum_q u[q] e^(-j 2 pi k q / M)` with `u[q] = sum_{i: n_i mod M = q} g[i] w[i]`, all in f64
//! (the twiddle of `k q` reduced modulo `M` in integers first, `libm::cos` and `libm::sin`),
//! rounded once to `C32`. The same input buffering as the fallback (`channelize_with`), so the
//! hop boundaries are those of the definition.

#![allow(clippy::many_single_char_names)]

use num_complex::Complex64;

use crate::Reference;
use crate::error::{Error, Result};
use crate::kernels::channelize::{Channelize, ChannelizeState, channelize_with};
use crate::timing::Progress;
use crate::types::{C32, ComplexSample};

/// `e^(-j 2 pi ((k q) mod m) / m)` in f64.
fn twiddle(k: usize, q: usize, m: usize) -> Complex64 {
    let idx = (k as u64 * q as u64) % m as u64;
    let theta = 2.0 * core::f64::consts::PI * idx as f64 / m as f64;
    Complex64::new(libm::cos(theta), -libm::sin(theta))
}

impl Channelize<C32> for Reference {
    fn channelize<Taps, Hist, Tw>(
        &self,
        s: &mut ChannelizeState<C32, Taps, Hist, Tw>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<Progress>
    where
        Taps: AsRef<[f32]>,
        Hist: AsRef<[C32]> + AsMut<[C32]>,
        Tw: AsRef<[C32]>,
    {
        channelize_with(s, input, output, false, |taps, window, swap, _plan, group, _| {
            let m = group.len();
            let shift = if swap { m / 2 } else { 0 };
            for (k, out) in group.iter_mut().enumerate() {
                let mut acc = Complex64::new(0.0, 0.0);
                for q in 0..m {
                    // The column `c` with `(c + shift) mod m == q`.
                    let c = (q + m - shift) % m;
                    let mut u = Complex64::new(0.0, 0.0);
                    // The stored taps hold every tap twice: a row of `M` samples has `2 M`.
                    for (t, w) in taps.chunks_exact(2 * m).zip(window.chunks_exact(m)) {
                        let g =
                            f64::from(*t.get(2 * c).ok_or(Error::Internal("reference tap"))?);
                        u += w
                            .get(c)
                            .ok_or(Error::Internal("reference window"))?
                            .to_c64()
                            * g;
                    }
                    acc += u * twiddle(k, q, m);
                }
                *out = C32::from_c64_round(acc);
            }
            Ok(())
        })
    }
}
