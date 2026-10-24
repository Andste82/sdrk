//! Reference implementation of the `convert_out` kernels: the f64 value mapping and a
//! bit-by-bit writer, written from `docs/device.md`.
//!
//! Q15 samples are encoded as the amplitude `q / 32768` through the same f64 formula as
//! `f32` samples; this equals the integer rules of the specification for every format
//! (all intermediate values are exact in f64).

use sdrk_device::{Align, Container, Endian, Layout, SampleFormat};

use crate::Reference;
use crate::error::{Error, Result};
use crate::kernels::convert::{ConvertOut, ConvertOutReal, plan_out};
use crate::types::{C32, CQ15, Q15};

/// One component to encode.
#[derive(Clone, Copy)]
enum Value {
    /// An amplitude as f64 (NaN counts as 0).
    Amplitude(f64),
    /// An `f32` whose bits are stored unchanged in `F32` containers.
    Float(f32),
}

fn amplitude_raw(format: &SampleFormat, x: f64) -> i64 {
    let x = if x.is_nan() { 0.0 } else { x };
    let offset = if format.container() == Container::U8 {
        127.5
    } else {
        0.0
    };
    let bits = u32::from(format.bits());
    let (lo, hi) = if format.signed() {
        (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1)
    } else {
        (0, (1i64 << bits) - 1)
    };
    let v = libm::roundeven(x * format.full_scale_value() + offset);
    v.clamp(lo as f64, hi as f64) as i64
}

/// The bits to store for one value and the number of bits of the storage word.
fn storage(format: &SampleFormat, v: Value) -> (u64, u32) {
    let (word_bits, packed) = match format.container() {
        Container::U8 | Container::I8 => (8, false),
        Container::I16 => (16, false),
        Container::I32 | Container::F32 => (32, false),
        Container::Packed12 => (12, true),
        _ => (10, true),
    };
    let bits = match v {
        Value::Float(f) if format.container() == Container::F32 => u64::from(f.to_bits()),
        Value::Float(f) => {
            let raw = amplitude_raw(format, f64::from(f));
            align(format, raw, word_bits, packed)
        }
        Value::Amplitude(a) => {
            if format.container() == Container::F32 {
                u64::from((a as f32).to_bits())
            } else {
                align(format, amplitude_raw(format, a), word_bits, packed)
            }
        }
    };
    (bits, word_bits)
}

fn align(format: &SampleFormat, raw: i64, word_bits: u32, packed: bool) -> u64 {
    let bits = u32::from(format.bits());
    let stored = if format.align() == Align::Msb && !packed {
        raw << (word_bits - bits)
    } else {
        raw
    };
    (stored as u64) & ((1u64 << word_bits) - 1)
}

/// Sets `count` bits starting at bit `start` of a little-endian bit stream.
fn set_bits(out: &mut [u8], start: usize, count: u32, value: u64) -> Option<()> {
    for i in 0..count as usize {
        let bit = start + i;
        let byte = out.get_mut(bit / 8)?;
        let on = (value >> i) & 1 == 1;
        *byte = (*byte & !(1 << (bit % 8))) | (u8::from(on) << (bit % 8));
    }
    Some(())
}

/// Writes value number `k` of the stream.
fn put(format: &SampleFormat, out: &mut [u8], k: usize, v: Value) -> Option<()> {
    let (bits, word_bits) = storage(format, v);
    let packed = matches!(
        format.container(),
        Container::Packed12 | Container::Packed10
    );
    if packed {
        return set_bits(out, k * word_bits as usize, word_bits, bits);
    }
    let nbytes = word_bits as usize / 8;
    for i in 0..nbytes {
        let pos = if format.endian() == Endian::Big {
            nbytes - 1 - i
        } else {
            i
        };
        *out.get_mut(k * nbytes + i)? = (bits >> (8 * pos)) as u8;
    }
    Some(())
}

fn complex<T: Copy>(
    format: &SampleFormat,
    input: &[T],
    output: &mut [u8],
    parts: impl Fn(T) -> (Value, Value),
) -> Result<usize> {
    let bytes = plan_out(format, input.len(), output.len(), false)?;
    let internal = || Error::Internal("conversion index");
    let out = output.get_mut(..bytes).ok_or_else(internal)?;
    out.fill(0);
    let n = input.len();
    for (s, x) in input.iter().enumerate() {
        let (re, im) = parts(*x);
        let (a, b) = if format.swap_iq() { (im, re) } else { (re, im) };
        let (ka, kb) = if format.layout() == Layout::Planar {
            (s, n + s)
        } else {
            (2 * s, 2 * s + 1)
        };
        put(format, out, ka, a).ok_or_else(internal)?;
        put(format, out, kb, b).ok_or_else(internal)?;
    }
    Ok(bytes)
}

fn real<T: Copy>(
    format: &SampleFormat,
    input: &[T],
    output: &mut [u8],
    value: impl Fn(T) -> Value,
) -> Result<usize> {
    let bytes = plan_out(format, input.len(), output.len(), true)?;
    let internal = || Error::Internal("conversion index");
    let out = output.get_mut(..bytes).ok_or_else(internal)?;
    out.fill(0);
    for (k, x) in input.iter().enumerate() {
        put(format, out, k, value(*x)).ok_or_else(internal)?;
    }
    Ok(bytes)
}

fn from_q15(q: Q15) -> Value {
    Value::Amplitude(f64::from(q.0) / 32768.0)
}

impl ConvertOut<C32> for Reference {
    fn convert_out(
        &self,
        format: &SampleFormat,
        input: &[C32],
        output: &mut [u8],
    ) -> Result<usize> {
        complex(format, input, output, |x| {
            (Value::Float(x.re), Value::Float(x.im))
        })
    }
}

impl ConvertOut<CQ15> for Reference {
    fn convert_out(
        &self,
        format: &SampleFormat,
        input: &[CQ15],
        output: &mut [u8],
    ) -> Result<usize> {
        complex(format, input, output, |x| (from_q15(x.re), from_q15(x.im)))
    }
}

impl ConvertOutReal<f32> for Reference {
    fn convert_out_real(
        &self,
        format: &SampleFormat,
        input: &[f32],
        output: &mut [u8],
    ) -> Result<usize> {
        real(format, input, output, Value::Float)
    }
}

impl ConvertOutReal<Q15> for Reference {
    fn convert_out_real(
        &self,
        format: &SampleFormat,
        input: &[Q15],
        output: &mut [u8],
    ) -> Result<usize> {
        real(format, input, output, from_q15)
    }
}
