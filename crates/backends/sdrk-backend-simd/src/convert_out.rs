//! `convert_out`: `C32` / `CQ15` to raw device bytes.
//!
//! The same formats as in `convert` run vectorised (`U8`, `I8` and little-endian `I16`); the
//! rest goes to the fallback. Bit-exact with the fallback, including `floor(v + 0.5)`
//! rounding, saturation and NaN as zero:
//!
//! - `I16` and `I8` have a power-of-two scale and no offset, so `x * scale + 0.5` is exact in
//!   `f32` wherever it can influence a non-saturated result.
//! - `U8` has the offset 127.5, whose sum with `x * 128` rounds differently in `f32` and `f64`,
//!   so the vector path computes it in `f64` like the specification.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::convert::ConvertOut;
use sdrk_core::{C32, CQ15, Fallback, Result};
use sdrk_device::SampleFormat;

use crate::Simd;
use crate::convert::{Kind, fast_kind};
use crate::level::dispatch;

/// Number of output bytes for the fast path, else `None` (the fallback reports the error).
fn fast_bytes(kind: Kind, components: usize, output_len: usize) -> Option<usize> {
    let bytes = components.checked_mul(kind.width())?;
    (output_len >= bytes).then_some(bytes)
}

/// 1.5 * 2^23: adding and subtracting it rounds a float of magnitude below 2^22 to the nearest
/// integer, ties to even, using only additions.
const MAGIC: f32 = 12_582_912.0;

/// `floor(x * scale + 0.5)` as `i32` lanes, NaN as 0, saturated to `[lo, hi]` by the caller's
/// narrowing. Exact: with a power-of-two `scale` the product is exact, the sum with 0.5 is never
/// formed (it would round), and the result is the nearest integer, with the ties rounded up
/// instead of to even.
///
/// `lo` and `hi` are the first values beyond the saturation range (`-32769` and `32768` for
/// 16 bits): any product outside is clamped to them first, which keeps the rounding trick
/// within its range and does not change the saturated result.
#[inline(always)]
fn quantize<S: SimdToken>(simd: S, x: S::f32s, scale: f32, lo: f32, hi: f32) -> S::i32s {
    let v = x * scale;
    let not_nan = v.simd_eq(v);
    let c = v.max(lo).min(hi);
    let magic = S::f32s::splat(simd, MAGIC);
    let nearest = (c + magic) - magic;
    let up = (c - nearest).simd_eq(0.5f32);
    let one = S::f32s::splat(simd, 1.0);
    let zero = S::f32s::splat(simd, 0.0);
    let rounded = nearest + up.select(one, zero);
    rounded.to_int::<S::i32s>() & not_nan.to_vector()
}

/// Scalar twin of [`quantize`] for the tail, saturated to `[lo, hi]` (the saturation range).
#[inline(always)]
fn quantize_scalar(x: f32, scale: f32, lo: f32, hi: f32) -> i32 {
    let v = x * scale;
    let t = libm::floorf(v);
    let t = if v - t >= 0.5 { t + 1.0 } else { t };
    let t = if t.is_nan() { 0.0 } else { t };
    let t = if t > hi { hi } else { t };
    let t = if t < lo { lo } else { t };
    t as i32
}

/// `C32` to `CS16`.
#[inline(always)]
fn f32_to_i16<S: SimdToken>(simd: S, input: &[f32], out: &mut [u8]) {
    let h = S::f32s::LEN;
    let mut ic = input.chunks_exact(2 * h);
    let mut oc = out.chunks_exact_mut(4 * h);
    for (i, o) in (&mut ic).zip(&mut oc) {
        let (a, b) = i.split_at(h);
        let a = quantize::<S>(
            simd,
            S::f32s::from_slice(simd, a),
            32768.0,
            -32769.0,
            32768.0,
        );
        let b = quantize::<S>(
            simd,
            S::f32s::from_slice(simd, b),
            32768.0,
            -32769.0,
            32768.0,
        );
        let q: S::i16s = a.saturating_narrow(b);
        q.bitcast::<S::u8s>().store_slice(o);
    }
    for (&x, o) in ic
        .remainder()
        .iter()
        .zip(oc.into_remainder().chunks_exact_mut(2))
    {
        let q = quantize_scalar(x, 32768.0, -32768.0, 32767.0) as i16;
        o.copy_from_slice(&q.to_le_bytes());
    }
}

/// `C32` to `CS8`.
#[inline(always)]
fn f32_to_i8<S: SimdToken>(simd: S, input: &[f32], out: &mut [u8]) {
    let h = S::f32s::LEN;
    let mut ic = input.chunks_exact(4 * h);
    let mut oc = out.chunks_exact_mut(4 * h);
    for (i, o) in (&mut ic).zip(&mut oc) {
        let (ab, cd) = i.split_at(2 * h);
        let (a, b) = ab.split_at(h);
        let (c, d) = cd.split_at(h);
        let qa = quantize::<S>(simd, S::f32s::from_slice(simd, a), 128.0, -129.0, 128.0);
        let qb = quantize::<S>(simd, S::f32s::from_slice(simd, b), 128.0, -129.0, 128.0);
        let qc = quantize::<S>(simd, S::f32s::from_slice(simd, c), 128.0, -129.0, 128.0);
        let qd = quantize::<S>(simd, S::f32s::from_slice(simd, d), 128.0, -129.0, 128.0);
        let lo: S::i16s = qa.saturating_narrow(qb);
        let hi: S::i16s = qc.saturating_narrow(qd);
        let v: S::i8s = lo.saturating_narrow(hi);
        v.bitcast::<S::u8s>().store_slice(o);
    }
    for (&x, o) in ic.remainder().iter().zip(oc.into_remainder()) {
        *o = (quantize_scalar(x, 128.0, -128.0, 127.0) as i8) as u8;
    }
}

/// `floor((x * 128 + 127.5) + 0.5)` clamped to `[0, 255]` in `f64` lanes, as `i64` lanes.
#[inline(always)]
fn quantize_u8<S: SimdToken>(simd: S, x: S::f64s) -> S::i64s {
    let zero = S::f64s::splat(simd, 0.0);
    let x = x.is_nan().select(zero, x);
    let v = (x * 128.0 + 127.5 + 0.5).floor();
    let v = v.min(S::f64s::splat(simd, 255.0)).max(zero);
    v.to_int::<S::i64s>()
}

/// One vector of `f32` quantised for `CU8`, as `i32` lanes.
#[inline(always)]
fn quantize_u8_vector<S: SimdToken>(simd: S, s: &[f32]) -> S::i32s {
    let (lo, hi) = S::f32s::from_slice(simd, s).widen();
    quantize_u8(simd, lo).narrow(quantize_u8(simd, hi))
}

/// `C32` to `CU8`.
#[inline(always)]
fn f32_to_u8<S: SimdToken>(simd: S, input: &[f32], out: &mut [u8]) {
    let h = S::f32s::LEN;
    let mut ic = input.chunks_exact(4 * h);
    let mut oc = out.chunks_exact_mut(4 * h);
    for (i, o) in (&mut ic).zip(&mut oc) {
        let (ab, cd) = i.split_at(2 * h);
        let (a, b) = ab.split_at(h);
        let (c, d) = cd.split_at(h);
        // Four f32 vectors become eight f64 vectors, which narrow back through i32 and i16 to
        // bytes; the values are in `0..=255`, so the wrapping narrows are exact.
        let lo: S::i16s = quantize_u8_vector::<S>(simd, a).narrow(quantize_u8_vector::<S>(simd, b));
        let hi: S::i16s = quantize_u8_vector::<S>(simd, c).narrow(quantize_u8_vector::<S>(simd, d));
        let v: S::i8s = lo.narrow(hi);
        v.bitcast::<S::u8s>().store_slice(o);
    }
    for (&x, o) in ic.remainder().iter().zip(oc.into_remainder()) {
        let a = if x.is_nan() { 0.0 } else { f64::from(x) };
        let v = libm::floor(a * 128.0 + 127.5 + 0.5);
        *o = v.clamp(0.0, 255.0) as u8;
    }
}

/// `(q + 128) >> 8` of one vector of Q15 values (saturating add).
#[inline(always)]
fn q15_i8_half<S: SimdToken>(simd: S, s: &[i16]) -> S::i16s {
    S::i16s::from_slice(simd, s).saturating_add(128i16) >> 8u32
}

/// Q15 to `CS8`: `(q + 128) >> 8` with a saturating add.
#[inline(always)]
fn q15_to_i8<S: SimdToken>(simd: S, input: &[i16], out: &mut [u8]) {
    let n = S::i16s::LEN;
    let mut ic = input.chunks_exact(2 * n);
    let mut oc = out.chunks_exact_mut(2 * n);
    for (i, o) in (&mut ic).zip(&mut oc) {
        let (a, b) = i.split_at(n);
        let v: S::i8s = q15_i8_half::<S>(simd, a).narrow(q15_i8_half::<S>(simd, b));
        v.bitcast::<S::u8s>().store_slice(o);
    }
    for (&q, o) in ic.remainder().iter().zip(oc.into_remainder()) {
        *o = ((q.saturating_add(128)) >> 8) as u8;
    }
}

/// `(q + 32768) >> 8` of one vector of Q15 values, as the logical shift of the offset binary value.
#[inline(always)]
fn q15_u8_half<S: SimdToken>(simd: S, s: &[i16]) -> S::u16s {
    let v: S::u16s = (S::i16s::from_slice(simd, s) ^ i16::MIN).bitcast();
    v >> 8u32
}

/// Q15 to `CU8`: `(q + 32768) >> 8` (logical, on the offset binary value).
#[inline(always)]
fn q15_to_u8<S: SimdToken>(simd: S, input: &[i16], out: &mut [u8]) {
    let n = S::i16s::LEN;
    let mut ic = input.chunks_exact(2 * n);
    let mut oc = out.chunks_exact_mut(2 * n);
    for (i, o) in (&mut ic).zip(&mut oc) {
        let (a, b) = i.split_at(n);
        let v: S::u8s = q15_u8_half::<S>(simd, a).narrow(q15_u8_half::<S>(simd, b));
        v.store_slice(o);
    }
    for (&q, o) in ic.remainder().iter().zip(oc.into_remainder()) {
        *o = (((q as u16) ^ 0x8000) >> 8) as u8;
    }
}

impl ConvertOut<C32> for Simd {
    fn convert_out(
        &self,
        format: &SampleFormat,
        input: &[C32],
        output: &mut [u8],
    ) -> Result<usize> {
        let Some(kind) = fast_kind(format) else {
            return Fallback.convert_out(format, input, output);
        };
        let Some(bytes) = fast_bytes(kind, input.len().saturating_mul(2), output.len()) else {
            return Fallback.convert_out(format, input, output);
        };
        let Some(out) = output.get_mut(..bytes) else {
            return Fallback.convert_out(format, input, output);
        };
        let input: &[f32] = bytemuck::cast_slice(input);
        match kind {
            Kind::I16 => dispatch!(simd => f32_to_i16(simd, input, out)),
            Kind::I8 => dispatch!(simd => f32_to_i8(simd, input, out)),
            Kind::U8 => dispatch!(simd => f32_to_u8(simd, input, out)),
        }
        Ok(bytes)
    }
}

impl ConvertOut<CQ15> for Simd {
    fn convert_out(
        &self,
        format: &SampleFormat,
        input: &[CQ15],
        output: &mut [u8],
    ) -> Result<usize> {
        let Some(kind) = fast_kind(format) else {
            return Fallback.convert_out(format, input, output);
        };
        let Some(bytes) = fast_bytes(kind, input.len().saturating_mul(2), output.len()) else {
            return Fallback.convert_out(format, input, output);
        };
        let Some(out) = output.get_mut(..bytes) else {
            return Fallback.convert_out(format, input, output);
        };
        let bytes_in: &[u8] = bytemuck::cast_slice(input);
        let input: &[i16] = bytemuck::cast_slice(input);
        match kind {
            // Little-endian `cs16` is byte-identical to `CQ15`.
            Kind::I16 => out.copy_from_slice(bytes_in),
            Kind::I8 => dispatch!(simd => q15_to_i8(simd, input, out)),
            Kind::U8 => dispatch!(simd => q15_to_u8(simd, input, out)),
        }
        Ok(bytes)
    }
}
