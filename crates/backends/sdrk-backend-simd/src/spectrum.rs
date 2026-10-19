//! `mag_sq`.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::spectrum::MagSq;
use sdrk_core::{C32, CQ15, Q15, Result};

use crate::Simd;
use crate::level::dispatch;
use crate::util::{check_len, mac};

/// `re*re + im*im` of `input` (interleaved floats) into `out`, bit-exact with the fallback.
#[inline(always)]
fn mag_sq_f32<S: SimdToken>(simd: S, input: &[f32], out: &mut [f32]) {
    let l = S::f32s::LEN;
    let mut in_chunks = input.chunks_exact(2 * l);
    let mut out_chunks = out.chunks_exact_mut(l);
    for (i, o) in (&mut in_chunks).zip(&mut out_chunks) {
        let (a, b) = i.split_at(l);
        let a = S::f32s::from_slice(simd, a);
        let b = S::f32s::from_slice(simd, b);
        let (re, im) = (a.unzip_low(b), a.unzip_high(b));
        mac::<S>(re, re, im * im).store_slice(o);
    }
    for (o, x) in out_chunks
        .into_remainder()
        .iter_mut()
        .zip(in_chunks.remainder().chunks_exact(2))
    {
        if let [re, im] = *x {
            *o = re * re + im * im;
        }
    }
}

impl MagSq<C32> for Simd {
    fn mag_sq(&self, input: &[C32], output: &mut [f32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let input: &[f32] = bytemuck::cast_slice(input);
        dispatch!(simd => mag_sq_f32(simd, input, output));
        Ok(())
    }
}

/// `(re*re + im*im + 2^14) >> 15` of one vector of Q15 samples, at most 32767. The sum of two
/// squares is at most `2^31`, so it fits an unsigned 32-bit lane.
#[inline(always)]
fn squares_q15<S: SimdToken>(simd: S, s: &[i16]) -> S::u32s {
    let x: S::i32s = S::i16s::from_slice(simd, s).bitcast();
    let re = (x << 16u32) >> 16u32;
    let im = x >> 16u32;
    let sum: S::u32s = (re * re + im * im).bitcast();
    ((sum + (1u32 << 14)) >> 15u32).min(S::u32s::splat(simd, 32767))
}

/// `sat16((re*re + im*im + 2^14) >> 15)` for Q15 samples, bit-exact with the fallback.
#[inline(always)]
fn mag_sq_q15<S: SimdToken>(simd: S, input: &[i16], out: &mut [i16]) {
    let n = S::i16s::LEN;
    let mut in_chunks = input.chunks_exact(2 * n);
    let mut out_chunks = out.chunks_exact_mut(n);
    for (i, o) in (&mut in_chunks).zip(&mut out_chunks) {
        let (a, b) = i.split_at(n);
        let lo = squares_q15::<S>(simd, a);
        let hi = squares_q15::<S>(simd, b);
        let packed: S::u16s = lo.narrow(hi);
        packed.bitcast::<S::i16s>().store_slice(o);
    }
    for (o, x) in out_chunks
        .into_remainder()
        .iter_mut()
        .zip(in_chunks.remainder().chunks_exact(2))
    {
        if let [re, im] = *x {
            let sum =
                (i64::from(re) * i64::from(re) + i64::from(im) * i64::from(im) + (1 << 14)) >> 15;
            *o = sum.min(32767) as i16;
        }
    }
}

impl MagSq<CQ15> for Simd {
    fn mag_sq(&self, input: &[CQ15], output: &mut [Q15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let x: &[i16] = bytemuck::cast_slice(input);
        let y: &mut [i16] = bytemuck::cast_slice_mut(output);
        dispatch!(simd => mag_sq_q15(simd, x, y));
        Ok(())
    }
}
