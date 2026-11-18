//! `qam_soft_bits` `C32` (M44 §2, M44.2 §2): the piecewise-linear max-log soft bits of every
//! square QAM, one generic implementation over the bits per axis `B`.
//!
//! The loop runs over the components in memory order (`re`, `im` of every symbol, as `f32`
//! through `bytemuck`), `B` soft bits per component; the in-phase first bit order of M44 §1 is
//! exactly that order. `B` is a `const` parameter, so the compiler unrolls the recursion and
//! writes the soft bits of a component in their final order, and it vectorizes the loop where
//! the target has vectors (D1204); every operation is a plain `f32` operation, so the bits are
//! the same on every target. 16-QAM and 64-QAM keep the loops of M44 (`qam16`, `qam64`): they
//! have the bits of the instances of `B` = 2 and 3, and the instance of `B` = 3 simulated
//! slower (D1503).

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

/// The knees `2^(B - i)` of the steps `i = 1 .. B - 1` of the recursion (the first `B - 1`
/// entries are used).
#[inline]
const fn knees<const B: usize>() -> [f32; 4] {
    let mut k = [0.0f32; 4];
    let mut i = 1;
    while i < B && i <= 4 {
        #[allow(
            clippy::indexing_slicing,
            reason = "const evaluation only: i - 1 < 4 by the loop condition"
        )]
        {
            k[i - 1] = (1u32 << (B - i)) as f32;
        }
        i += 1;
    }
    k
}

/// The `B` soft bits of every component of `x`: `t_0 = x c`, `t_i = 2^(B - i) - |t_(i - 1)|`,
/// each times `s`. `out` holds `B` values per component.
#[inline]
fn soft_bits<const B: usize>(s: f32, c: f32, x: &[f32], out: &mut [f32]) {
    let knees = const { knees::<B>() };
    let (out, _) = out.as_chunks_mut::<B>();
    for (o, &x) in out.iter_mut().zip(x) {
        if let [first, rest @ ..] = o.as_mut_slice() {
            let mut t = x * c;
            *first = s * t;
            for (o, &k) in rest.iter_mut().zip(&knees) {
                t = k - libm::fabsf(t);
                *o = s * t;
            }
        }
    }
}

/// [`soft_bits`] with the scale 1 in a loop of its own, without the products (`1 v = v`
/// exactly for every value but NaN, whose bits are not part of the contract): the soft bits of
/// `qam_demod` and of the default `symbol_demap`.
#[inline]
fn instance<const B: usize>(scale: f32, c: f32, x: &[f32], out: &mut [f32]) {
    if scale == 1.0 {
        soft_bits::<B>(1.0, c, x, out);
    } else {
        soft_bits::<B>(scale, c, x, out);
    }
}

impl QamSoftBits<C32> for Fallback {
    fn qam_soft_bits(&self, qam: Qam, input: &[C32], output: &mut [f32], scale: f32) -> Result<()> {
        check_soft_bits_len(qam, input.len(), output.len())?;
        let x: &[f32] = bytemuck::cast_slice(input);
        let c = qam.sqrt_energy();
        // The scale 1 has a loop of its own without the products (`1 v = v` exactly for every
        // value but NaN, whose bits are not part of the contract): the soft bits of `qam_demod`
        // and of the default `symbol_demap`.
        match qam {
            Qam::Qpsk => instance::<1>(scale, c, x, output),
            Qam::Qam16 if scale == 1.0 => qam16(1.0, c, x, output),
            Qam::Qam16 => qam16(scale, c, x, output),
            Qam::Qam64 if scale == 1.0 => qam64(1.0, c, x, output),
            Qam::Qam64 => qam64(scale, c, x, output),
            Qam::Qam256 => instance::<4>(scale, c, x, output),
            Qam::Qam1024 => instance::<5>(scale, c, x, output),
        }
        Ok(())
    }
}
