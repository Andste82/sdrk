//! Reference implementation of the `scramble` kernel: the decomposition of the contract in
//! exact arithmetic. The input as `f64` (`CQ15`: `raw / 32768`); for a residue `r ≠ 0` the
//! product with `(cos θ_r, sin θ_r)`, `θ_r = 2π r / 2^b` from `libm` in `f64`; then the
//! quadrant map in `f64` (negation and swap, exact); rounded once (`C32`: `as f32`; `CQ15`:
//! `q15_from_f64`, so `−(−1.0) = 1.0` becomes 32767 like `q15_neg_sat`).

use core::f64::consts::PI;

use crate::Reference;
use crate::error::Result;
use crate::kernels::scramble::{Scramble, check_args, phase_index, split_index};
use crate::num::q15_from_f64;
use crate::types::{C32, CQ15};

/// `(re, im)` times `exp(j 2π k / 2^bits)` by the decomposition of the contract, in `f64`.
// The names of the contract (milestone M49 §1).
#[allow(clippy::many_single_char_names)]
fn phasor_f64(re: f64, im: f64, k: u32, bits: u32) -> (f64, f64) {
    let (q, r) = split_index(k, bits);
    let (re, im) = if r == 0 {
        (re, im)
    } else {
        let theta = 2.0 * PI * f64::from(r) / f64::from(1u32 << bits);
        let (c, s) = (libm::cos(theta), libm::sin(theta));
        (re * c - im * s, re * s + im * c)
    };
    match q & 3 {
        0 => (re, im),
        1 => (-im, re),
        2 => (-re, -im),
        _ => (im, -re),
    }
}

impl Scramble<C32> for Reference {
    fn scramble(
        &self,
        input: &[C32],
        phase: &[u8],
        bits: u32,
        inverse: bool,
        output: &mut [C32],
    ) -> Result<()> {
        check_args(bits, input.len(), phase.len(), output.len())?;
        for ((o, x), &p) in output.iter_mut().zip(input).zip(phase) {
            let k = phase_index(p, bits, inverse);
            let (re, im) = phasor_f64(f64::from(x.re), f64::from(x.im), k, bits);
            *o = C32::new(re as f32, im as f32);
        }
        Ok(())
    }
}

impl Scramble<CQ15> for Reference {
    fn scramble(
        &self,
        input: &[CQ15],
        phase: &[u8],
        bits: u32,
        inverse: bool,
        output: &mut [CQ15],
    ) -> Result<()> {
        check_args(bits, input.len(), phase.len(), output.len())?;
        for ((o, x), &p) in output.iter_mut().zip(input).zip(phase) {
            let k = phase_index(p, bits, inverse);
            let re = f64::from(x.re.0) / 32768.0;
            let im = f64::from(x.im.0) / 32768.0;
            let (re, im) = phasor_f64(re, im, k, bits);
            *o = CQ15 {
                re: q15_from_f64(re),
                im: q15_from_f64(im),
            };
        }
        Ok(())
    }
}
