//! `convert`: raw device bytes to `C32` / `CQ15`.
//!
//! The common interleaved formats (`U8`, `I8` and little-endian `I16` with the full container
//! as valid bits, power-of-two full scale, no I/Q swap) run vectorised; every other format and
//! every invalid call is handed to the fallback, which also produces the error values. All
//! paths are bit-exact with the fallback: the scale factors are powers of two, so the
//! multiplication by the reciprocal equals the division of the specification.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::convert::Convert;
use sdrk_core::{C32, CQ15, Fallback, Result};
use sdrk_device::{Align, Container, Endian, FullScale, Layout, SampleFormat};

use crate::Simd;
use crate::level::{dispatch, is_sse2_baseline};

/// The formats with a vector path.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    U8,
    I8,
    I16,
}

impl Kind {
    /// Bytes per component.
    pub(crate) const fn width(self) -> usize {
        match self {
            Kind::U8 | Kind::I8 => 1,
            Kind::I16 => 2,
        }
    }
}

/// The vector-path kind of a format, if it has one.
pub(crate) fn fast_kind(format: &SampleFormat) -> Option<Kind> {
    if format.layout() != Layout::Interleaved
        || format.swap_iq()
        || format.full_scale() != FullScale::PowerOfTwo
        || format.align() != Align::Lsb
    {
        return None;
    }
    match (format.container(), format.bits(), format.signed()) {
        (Container::U8, 8, false) => Some(Kind::U8),
        (Container::I8, 8, true) => Some(Kind::I8),
        (Container::I16, 16, true) if format.endian() == Endian::Little => Some(Kind::I16),
        _ => None,
    }
}

/// The number of samples if `input_len` bytes and `output_len` samples are valid for the fast
/// path (whole samples and enough room), else `None` so that the fallback reports the error.
fn fast_samples(kind: Kind, input_len: usize, output_len: usize) -> Option<usize> {
    let per_sample = 2 * kind.width();
    if !input_len.is_multiple_of(per_sample) {
        return None;
    }
    let samples = input_len / per_sample;
    (output_len >= samples).then_some(samples)
}

/// Reciprocal of the full scale of a power-of-two format: `2^-7` or `2^-15`.
const INV_128: f32 = 1.0 / 128.0;
const INV_32768: f32 = 1.0 / 32768.0;

/// `CS16` to `f32`: `x * 2^-15`.
#[inline(always)]
fn i16_to_f32<S: SimdToken>(simd: S, bytes: &[u8], out: &mut [f32]) {
    let n = S::i16s::LEN;
    let h = S::f32s::LEN;
    let scale = S::f32s::splat(simd, INV_32768);
    let mut ic = bytes.chunks_exact(2 * n);
    let mut oc = out.chunks_exact_mut(n);
    for (b, o) in (&mut ic).zip(&mut oc) {
        let v: S::i16s = S::u8s::from_slice(simd, b).bitcast();
        let (lo, hi) = v.widen();
        let (o_lo, o_hi) = o.split_at_mut(h);
        (lo.to_float::<S::f32s>() * scale).store_slice(o_lo);
        (hi.to_float::<S::f32s>() * scale).store_slice(o_hi);
    }
    for (o, b) in oc
        .into_remainder()
        .iter_mut()
        .zip(ic.remainder().chunks_exact(2))
    {
        if let [a, c] = *b {
            *o = f32::from(i16::from_le_bytes([a, c])) * INV_32768;
        }
    }
}

/// `CS8` to `f32` without a sign-extending widening: interleaving with zero puts every byte in
/// the top of a 16-bit and then of a 32-bit lane (`x << 8`, then `x << 24`), the integer to float
/// conversion makes `x * 2^24` (exact: eight significant bits) and `2^-31` brings it to `x * 2^-7`.
/// Four interleaves and four conversions per 16 bytes replace the compare-and-unpack widening of
/// the baseline SSE2 level.
#[inline(always)]
fn i8_to_f32_zip<S: SimdToken>(simd: S, bytes: &[u8], out: &mut [f32]) {
    let n = S::i8s::LEN;
    let h = S::f32s::LEN;
    let scale = S::f32s::splat(simd, 1.0 / 2_147_483_648.0);
    let zero8 = S::u8s::splat(simd, 0);
    let zero16 = S::u16s::splat(simd, 0);
    let mut ic = bytes.chunks_exact(n);
    let mut oc = out.chunks_exact_mut(n);
    for (b, o) in (&mut ic).zip(&mut oc) {
        let v = S::u8s::from_slice(simd, b);
        let w_lo: S::u16s = zero8.zip_low(v).bitcast();
        let w_hi: S::u16s = zero8.zip_high(v).bitcast();
        let d0: S::i32s = zero16.zip_low(w_lo).bitcast();
        let d1: S::i32s = zero16.zip_high(w_lo).bitcast();
        let d2: S::i32s = zero16.zip_low(w_hi).bitcast();
        let d3: S::i32s = zero16.zip_high(w_hi).bitcast();
        let (o01, o23) = o.split_at_mut(2 * h);
        let (o0, o1) = o01.split_at_mut(h);
        let (o2, o3) = o23.split_at_mut(h);
        (d0.to_float::<S::f32s>() * scale).store_slice(o0);
        (d1.to_float::<S::f32s>() * scale).store_slice(o1);
        (d2.to_float::<S::f32s>() * scale).store_slice(o2);
        (d3.to_float::<S::f32s>() * scale).store_slice(o3);
    }
    for (o, &b) in oc.into_remainder().iter_mut().zip(ic.remainder()) {
        *o = f32::from(b as i8) * INV_128;
    }
}

/// `CS8` to `f32`: `x * 2^-7`.
#[inline(always)]
fn i8_to_f32<S: SimdToken>(simd: S, bytes: &[u8], out: &mut [f32]) {
    if is_sse2_baseline(simd) {
        return i8_to_f32_zip(simd, bytes, out);
    }
    let n = S::i8s::LEN;
    let h = S::f32s::LEN;
    let scale = S::f32s::splat(simd, INV_128);
    let mut ic = bytes.chunks_exact(n);
    let mut oc = out.chunks_exact_mut(n);
    for (b, o) in (&mut ic).zip(&mut oc) {
        let v: S::i8s = S::u8s::from_slice(simd, b).bitcast();
        let (a, b2) = v.widen();
        let (a0, a1) = a.widen();
        let (b0, b1) = b2.widen();
        let (o01, o23) = o.split_at_mut(2 * h);
        let (o0, o1) = o01.split_at_mut(h);
        let (o2, o3) = o23.split_at_mut(h);
        (a0.to_float::<S::f32s>() * scale).store_slice(o0);
        (a1.to_float::<S::f32s>() * scale).store_slice(o1);
        (b0.to_float::<S::f32s>() * scale).store_slice(o2);
        (b1.to_float::<S::f32s>() * scale).store_slice(o3);
    }
    for (o, &b) in oc.into_remainder().iter_mut().zip(ic.remainder()) {
        *o = f32::from(b as i8) * INV_128;
    }
}

/// `CU8` to `f32`: `(x - 127.5) * 2^-7`.
#[inline(always)]
fn u8_to_f32<S: SimdToken>(simd: S, bytes: &[u8], out: &mut [f32]) {
    let n = S::u8s::LEN;
    let h = S::f32s::LEN;
    let scale = S::f32s::splat(simd, INV_128);
    let offset = S::f32s::splat(simd, 127.5);
    let mut ic = bytes.chunks_exact(n);
    let mut oc = out.chunks_exact_mut(n);
    for (b, o) in (&mut ic).zip(&mut oc) {
        let v = S::u8s::from_slice(simd, b);
        let (a, b2) = v.widen();
        let (a0, a1) = a.widen();
        let (b0, b1) = b2.widen();
        let (o01, o23) = o.split_at_mut(2 * h);
        let (o0, o1) = o01.split_at_mut(h);
        let (o2, o3) = o23.split_at_mut(h);
        ((a0.to_float::<S::f32s>() - offset) * scale).store_slice(o0);
        ((a1.to_float::<S::f32s>() - offset) * scale).store_slice(o1);
        ((b0.to_float::<S::f32s>() - offset) * scale).store_slice(o2);
        ((b1.to_float::<S::f32s>() - offset) * scale).store_slice(o3);
    }
    for (o, &b) in oc.into_remainder().iter_mut().zip(ic.remainder()) {
        *o = (f32::from(b) - 127.5) * INV_128;
    }
}

/// `CS8` to Q15: `x << 8`.
#[inline(always)]
fn i8_to_q15<S: SimdToken>(simd: S, bytes: &[u8], out: &mut [i16]) {
    let n = S::i8s::LEN;
    let h = S::i16s::LEN;
    let mut ic = bytes.chunks_exact(n);
    let mut oc = out.chunks_exact_mut(n);
    for (b, o) in (&mut ic).zip(&mut oc) {
        let v: S::i8s = S::u8s::from_slice(simd, b).bitcast();
        let (lo, hi) = v.widen();
        let (o_lo, o_hi) = o.split_at_mut(h);
        (lo << 8u32).store_slice(o_lo);
        (hi << 8u32).store_slice(o_hi);
    }
    for (o, &b) in oc.into_remainder().iter_mut().zip(ic.remainder()) {
        *o = i16::from(b as i8) << 8;
    }
}

/// `CU8` to Q15: `(2x - 255) << 7`.
#[inline(always)]
fn u8_to_q15<S: SimdToken>(simd: S, bytes: &[u8], out: &mut [i16]) {
    let n = S::u8s::LEN;
    let h = S::i16s::LEN;
    let c255 = S::i16s::splat(simd, 255);
    let mut ic = bytes.chunks_exact(n);
    let mut oc = out.chunks_exact_mut(n);
    for (b, o) in (&mut ic).zip(&mut oc) {
        let v = S::u8s::from_slice(simd, b);
        let (lo, hi) = v.widen();
        let lo: S::i16s = lo.bitcast();
        let hi: S::i16s = hi.bitcast();
        let (o_lo, o_hi) = o.split_at_mut(h);
        (((lo + lo) - c255) << 7u32).store_slice(o_lo);
        (((hi + hi) - c255) << 7u32).store_slice(o_hi);
    }
    for (o, &b) in oc.into_remainder().iter_mut().zip(ic.remainder()) {
        *o = (2 * i16::from(b) - 255) << 7;
    }
}

/// The bytes of a `CQ15` slice as `i16` values.
fn q15_view(out: &mut [CQ15]) -> &mut [i16] {
    bytemuck::cast_slice_mut(out)
}

impl Convert<C32> for Simd {
    fn convert(&self, format: &SampleFormat, input: &[u8], output: &mut [C32]) -> Result<usize> {
        let Some(kind) = fast_kind(format) else {
            return Fallback.convert(format, input, output);
        };
        let Some(samples) = fast_samples(kind, input.len(), output.len()) else {
            return Fallback.convert(format, input, output);
        };
        let Some(out) = output.get_mut(..samples) else {
            return Fallback.convert(format, input, output);
        };
        let out: &mut [f32] = bytemuck::cast_slice_mut(out);
        match kind {
            Kind::I16 => dispatch!(simd => i16_to_f32(simd, input, out)),
            Kind::I8 => dispatch!(simd => i8_to_f32(simd, input, out)),
            Kind::U8 => dispatch!(simd => u8_to_f32(simd, input, out)),
        }
        Ok(samples)
    }
}

impl Convert<CQ15> for Simd {
    fn convert(&self, format: &SampleFormat, input: &[u8], output: &mut [CQ15]) -> Result<usize> {
        let Some(kind) = fast_kind(format) else {
            return Fallback.convert(format, input, output);
        };
        let Some(samples) = fast_samples(kind, input.len(), output.len()) else {
            return Fallback.convert(format, input, output);
        };
        let Some(out) = output.get_mut(..samples) else {
            return Fallback.convert(format, input, output);
        };
        let out = q15_view(out);
        match kind {
            // Little-endian `cs16` is byte-identical to `CQ15`.
            Kind::I16 => {
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(out);
                bytes.copy_from_slice(input);
            }
            Kind::I8 => dispatch!(simd => i8_to_q15(simd, input, out)),
            Kind::U8 => dispatch!(simd => u8_to_q15(simd, input, out)),
        }
        Ok(samples)
    }
}
