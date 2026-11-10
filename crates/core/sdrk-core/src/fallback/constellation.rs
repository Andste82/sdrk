//! `qam_soft_bits` `C32` (M44 §2): the piecewise-linear max-log soft bits of 16- and 64-QAM.
//!
//! The loops run over the components in memory order (`re`, `im` of every symbol, as `f32`
//! through `bytemuck`), two (16-QAM) or three (64-QAM) soft bits per component; the in-phase
//! first bit order of M44 §1 is exactly that order. The compiler vectorizes them where the
//! target has vectors (D1204); every product is a plain `f32` multiplication, so the bits are the
//! same on every target.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::constellation::{Qam, QamSoftBits, check_soft_bits_len};
use crate::types::C32;

/// The two soft bits of every component of `x` (16-QAM): `s v` and `s (2 - |v|)` with
/// `v = x √10`. `out` holds two values per component.
#[inline]
fn qam16(s: f32, c: f32, x: &[f32], out: &mut [f32]) {
    for (o, &x) in out.chunks_exact_mut(2).zip(x) {
        let v = x * c;
        if let [b0, b1] = o {
            *b0 = s * v;
            *b1 = s * (2.0 - libm::fabsf(v));
        }
    }
}

/// The three soft bits of every component of `x` (64-QAM): `s v`, `s (4 - a)` and
/// `s (2 - |a - 4|)` with `v = x √42` and `a = |v|`. `out` holds three values per component.
#[inline]
fn qam64(s: f32, c: f32, x: &[f32], out: &mut [f32]) {
    for (o, &x) in out.chunks_exact_mut(3).zip(x) {
        let v = x * c;
        let mag = libm::fabsf(v);
        if let [b0, b1, b2] = o {
            *b0 = s * v;
            *b1 = s * (4.0 - mag);
            *b2 = s * (2.0 - libm::fabsf(mag - 4.0));
        }
    }
}

impl QamSoftBits<C32> for Fallback {
    fn qam_soft_bits(&self, qam: Qam, input: &[C32], output: &mut [f32], scale: f32) -> Result<()> {
        check_soft_bits_len(qam, input.len(), output.len())?;
        let x: &[f32] = bytemuck::cast_slice(input);
        let c = qam.sqrt_energy();
        // The scale 1 has a loop of its own without the products (`1 v = v` exactly for every
        // value but NaN, whose bits are not part of the contract): the soft bits of `qam_demod`
        // and of the default `slice`.
        match (qam, scale == 1.0) {
            (Qam::Qam16, true) => qam16(1.0, c, x, output),
            (Qam::Qam16, false) => qam16(scale, c, x, output),
            (Qam::Qam64, true) => qam64(1.0, c, x, output),
            (Qam::Qam64, false) => qam64(scale, c, x, output),
        }
        Ok(())
    }
}
