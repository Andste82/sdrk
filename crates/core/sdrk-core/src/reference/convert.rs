//! Reference implementation of the `convert` kernels: bit-by-bit decoding and
//! f64 / exact integer mapping, written from `docs/device.md`.

use sdrk_device::{Align, Container, Endian, FullScale, Layout, SampleFormat};

use crate::Reference;
use crate::error::{Error, Result};
use crate::kernels::convert::{Convert, ConvertReal, plan};
use crate::num::{q15_from_f32, round_shift, sat16};
use crate::types::{C32, CQ15, ComplexSample, Q15};

/// One decoded component.
#[derive(Clone, Copy)]
enum Raw {
    /// Effective integer value.
    Int(i64),
    /// Value of an `F32` container.
    Float(f32),
}

/// Reads `count` bits starting at bit `start` of a little-endian bit stream.
fn bit_field(input: &[u8], start: usize, count: usize) -> Option<u64> {
    let mut v = 0u64;
    for i in 0..count {
        let bit = start + i;
        let byte = *input.get(bit / 8)?;
        v |= u64::from((byte >> (bit % 8)) & 1) << i;
    }
    Some(v)
}

/// Reads the container word number `k` (not packed) as an unsigned integer.
fn container_word(input: &[u8], k: usize, nbytes: usize, endian: Endian) -> Option<u64> {
    let chunk = input.get(k * nbytes..(k + 1) * nbytes)?;
    let mut v = 0u64;
    for (i, b) in chunk.iter().enumerate() {
        let pos = if endian == Endian::Little {
            i
        } else {
            nbytes - 1 - i
        };
        v |= u64::from(*b) << (8 * pos);
    }
    Some(v)
}

fn sign_extend(v: u64, bits: u32) -> i64 {
    let half = 1i64 << (bits - 1);
    ((v & ((1u64 << bits) - 1)) as i64 ^ half) - half
}

fn decode(format: &SampleFormat, input: &[u8], k: usize) -> Option<Raw> {
    let bits = u32::from(format.bits());
    match format.container() {
        Container::Packed12 | Container::Packed10 => {
            let v = bit_field(input, k * bits as usize, bits as usize)?;
            Some(Raw::Int(sign_extend(v, bits)))
        }
        Container::F32 => {
            let w = container_word(input, k, 4, format.endian())?;
            Some(Raw::Float(f32::from_bits(w as u32)))
        }
        container => {
            let cbits: u32 = match container {
                Container::U8 | Container::I8 => 8,
                Container::I16 => 16,
                Container::I32 => 32,
                _ => return None,
            };
            let w = container_word(input, k, cbits as usize / 8, format.endian())?;
            let e = match (format.signed(), format.align()) {
                (true, Align::Lsb) => sign_extend(w, bits),
                (true, Align::Msb) => sign_extend(w, cbits) >> (cbits - bits),
                (false, Align::Lsb) => (w & ((1u64 << bits) - 1)) as i64,
                (false, Align::Msb) => (w >> (cbits - bits)) as i64,
            };
            Some(Raw::Int(e))
        }
    }
}

fn to_f32(format: &SampleFormat, raw: Raw) -> f32 {
    match raw {
        Raw::Float(f) => f,
        Raw::Int(e) => {
            let offset = if format.container() == Container::U8 {
                127.5
            } else {
                0.0
            };
            ((e as f64 - offset) / format.full_scale_value()) as f32
        }
    }
}

fn to_q15(format: &SampleFormat, raw: Raw) -> Q15 {
    let bits = u32::from(format.bits());
    let e = match raw {
        Raw::Float(f) => return q15_from_f32(f),
        Raw::Int(e) => e,
    };
    let q = match (format.container(), format.full_scale()) {
        (Container::U8, FullScale::PowerOfTwo) => (2 * e - 255) << (15 - bits.min(15)),
        (Container::U8, FullScale::Explicit(n)) => {
            let n = i64::from(n);
            ((2 * e - 255) * 32768 + n).div_euclid(2 * n)
        }
        (_, FullScale::Explicit(n)) => {
            let n = i64::from(n);
            (e * 65536 + n).div_euclid(2 * n)
        }
        (_, FullScale::PowerOfTwo) if bits <= 16 => e << (16 - bits),
        (_, FullScale::PowerOfTwo) => round_shift(e, bits - 16),
    };
    Q15(sat16(q))
}

fn components(format: &SampleFormat, input: &[u8], samples: usize, s: usize) -> Option<(Raw, Raw)> {
    let (a, b) = if format.layout() == Layout::Planar {
        (
            decode(format, input, s)?,
            decode(format, input, samples + s)?,
        )
    } else {
        (
            decode(format, input, 2 * s)?,
            decode(format, input, 2 * s + 1)?,
        )
    };
    Some(if format.swap_iq() { (b, a) } else { (a, b) })
}

fn complex<T: ComplexSample>(
    format: &SampleFormat,
    input: &[u8],
    output: &mut [T],
    samples: usize,
    map: impl Fn(&SampleFormat, Raw) -> T::Real,
) -> Result<usize> {
    let internal = || Error::Internal("conversion index");
    let out = output.get_mut(..samples).ok_or_else(internal)?;
    for (s, o) in out.iter_mut().enumerate() {
        let (i, q) = components(format, input, samples, s).ok_or_else(internal)?;
        *o = T::new(map(format, i), map(format, q));
    }
    Ok(samples)
}

fn real<T: Copy>(
    format: &SampleFormat,
    input: &[u8],
    output: &mut [T],
    samples: usize,
    map: impl Fn(&SampleFormat, Raw) -> T,
) -> Result<usize> {
    let internal = || Error::Internal("conversion index");
    let out = output.get_mut(..samples).ok_or_else(internal)?;
    for (s, o) in out.iter_mut().enumerate() {
        *o = map(format, decode(format, input, s).ok_or_else(internal)?);
    }
    Ok(samples)
}

impl Convert<C32> for Reference {
    fn convert(&self, format: &SampleFormat, input: &[u8], output: &mut [C32]) -> Result<usize> {
        let n = plan(format, input.len(), output.len(), false)?;
        complex(format, input, output, n, to_f32)
    }
}

impl Convert<CQ15> for Reference {
    fn convert(&self, format: &SampleFormat, input: &[u8], output: &mut [CQ15]) -> Result<usize> {
        let n = plan(format, input.len(), output.len(), false)?;
        complex(format, input, output, n, to_q15)
    }
}

impl ConvertReal<f32> for Reference {
    fn convert_real(
        &self,
        format: &SampleFormat,
        input: &[u8],
        output: &mut [f32],
    ) -> Result<usize> {
        let n = plan(format, input.len(), output.len(), true)?;
        real(format, input, output, n, to_f32)
    }
}

impl ConvertReal<Q15> for Reference {
    fn convert_real(
        &self,
        format: &SampleFormat,
        input: &[u8],
        output: &mut [Q15],
    ) -> Result<usize> {
        let n = plan(format, input.len(), output.len(), true)?;
        real(format, input, output, n, to_q15)
    }
}
