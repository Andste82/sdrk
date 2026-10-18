//! Raw-format encoder and reference decoder: the independent oracle for the
//! conversion kernels of a later milestone.
//!
//! Both functions are written from the value mapping documented in
//! `docs/device.md` and call no `sdrk` conversion code; only the plain getters
//! of [`SampleFormat`] are used. Keep it that way: an oracle that shares code
//! with the implementation under test proves nothing.
//!
//! Value mapping (per component): `amplitude = (raw - offset) / full_scale`
//! with `offset = 127.5` for `U8` and 0 otherwise. Encoding is the inverse:
//! `raw = floor(amplitude * full_scale + offset + 0.5)` (round half up),
//! saturated to the range of `bits`, shifted left for MSB alignment.
//! `swap_iq` puts Q first in every pair (for planar data: the first half holds
//! Q). `Endian::Big` reverses the bytes of multi-byte containers. Packed
//! containers pack their values LSB-first. `F32` is stored unscaled.

use num_complex::Complex64;
use sdrk_device::{Align, Container, Endian, FullScale, Layout, SampleFormat};

use crate::suite::Failure;

const CASE: &str = "encode";

/// Small internal error. `Failure` is large (a 160-byte detail string), and
/// every `Result<_, Failure>` temporary counts against the 4 KiB stack-frame
/// budget; the public functions convert once at the boundary.
#[derive(Clone, Copy)]
struct Oops {
    msg: &'static str,
    need: usize,
    have: usize,
}

fn err(msg: &'static str) -> Oops {
    Oops {
        msg,
        need: 0,
        have: 0,
    }
}

impl Oops {
    fn into_failure(self) -> Failure {
        let f = Failure::new(CASE, self.msg);
        if self.need == 0 && self.have == 0 {
            f
        } else {
            f.detail(format_args!("need {}, have {}", self.need, self.have))
        }
    }
}

/// Everything derived from a format that encoder and decoder need.
struct Spec {
    container: Container,
    bits: u32,
    signed: bool,
    layout: Layout,
    align: Align,
    endian: Endian,
    swap: bool,
    full_scale: f64,
    /// Bits of the storage word of unpacked containers.
    word_bits: u32,
    /// Values per packing group and bytes per group (unpacked: 1 value).
    group_values: usize,
    group_bytes: usize,
}

impl Spec {
    fn of(f: &SampleFormat) -> Result<Self, Oops> {
        let container = f.container();
        let bits = u32::from(f.bits());
        let (word_bits, group_values, group_bytes) = match container {
            Container::U8 | Container::I8 => (8, 1, 1),
            Container::I16 => (16, 1, 2),
            Container::I32 | Container::F32 => (32, 1, 4),
            Container::Packed12 => (12, 2, 3),
            Container::Packed10 => (10, 4, 5),
            _ => return Err(err("container is not supported by the oracle")),
        };
        if bits == 0 || bits > word_bits {
            return Err(err("bits do not fit the container"));
        }
        let full_scale = match f.full_scale() {
            FullScale::Explicit(v) => f64::from(v),
            FullScale::PowerOfTwo => (1u64 << (bits - 1)) as f64,
        };
        Ok(Self {
            container,
            bits,
            signed: f.signed(),
            layout: f.layout(),
            align: f.align(),
            endian: f.endian(),
            swap: f.swap_iq(),
            full_scale,
            word_bits,
            group_values,
            group_bytes,
        })
    }

    fn values_per_sample(&self) -> usize {
        if self.layout == Layout::Real { 1 } else { 2 }
    }

    fn offset(&self) -> f64 {
        if self.container == Container::U8 {
            127.5
        } else {
            0.0
        }
    }

    fn is_float(&self) -> bool {
        self.container == Container::F32
    }

    fn is_packed(&self) -> bool {
        matches!(self.container, Container::Packed12 | Container::Packed10)
    }

    /// Position of value `k` of the stream as (sample, `is_second_component`).
    fn locate(&self, k: usize, samples: usize) -> (usize, bool) {
        match self.layout {
            Layout::Real => (k, false),
            Layout::Planar => (k % samples.max(1), k >= samples),
            Layout::Interleaved => (k / 2, k % 2 == 1),
        }
    }
}

fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn sign_extend(v: u64, bits: u32) -> i64 {
    let s = 64 - bits;
    ((v << s) as i64) >> s
}

/// Amplitude to raw integer, saturated to the range of `bits`.
fn quantize(spec: &Spec, x: f64) -> i64 {
    let x = if x.is_nan() { 0.0 } else { x };
    let v = libm::floor(x * spec.full_scale + spec.offset() + 0.5);
    let (lo, hi) = if spec.signed {
        let half = 1i64 << (spec.bits - 1);
        (-half, half - 1)
    } else {
        (0, (1i64 << spec.bits) - 1)
    };
    v.clamp(lo as f64, hi as f64) as i64
}

/// Encodes `samples` into the raw bytes of `format` and returns the number of
/// bytes written. `Real` layouts use only the real part. Amplitudes outside
/// the representable range saturate; NaN encodes as 0.
///
/// # Errors
/// A [`Failure`] for `Bits1`, if the sample count does not fill whole packing
/// groups, or if `out` is too small.
///
/// ```
/// use num_complex::Complex64;
/// use sdrk_device::formats;
/// let mut out = [0u8; 2];
/// let n = sdrk_testkit::encode::encode(
///     &formats::CU8,
///     &[Complex64::new(0.0, -1.0)],
///     &mut out,
/// )?;
/// assert_eq!((n, out), (2, [128, 0]));
/// # Ok::<(), sdrk_testkit::Failure>(())
/// ```
pub fn encode(
    format: &SampleFormat,
    samples: &[Complex64],
    out: &mut [u8],
) -> Result<usize, Failure> {
    encode_inner(format, samples, out).map_err(Oops::into_failure)
}

fn encode_inner(
    format: &SampleFormat,
    samples: &[Complex64],
    out: &mut [u8],
) -> Result<usize, Oops> {
    let spec = Spec::of(format)?;
    let total = samples
        .len()
        .checked_mul(spec.values_per_sample())
        .ok_or_else(|| err("sample count overflows"))?;
    if total % spec.group_values != 0 {
        return Err(err("sample count does not fill whole packing groups"));
    }
    let nbytes = (total / spec.group_values)
        .checked_mul(spec.group_bytes)
        .ok_or_else(|| err("byte count overflows"))?;
    let have = out.len();
    let out = out.get_mut(..nbytes).ok_or(Oops {
        msg: "output buffer too small",
        need: nbytes,
        have,
    })?;

    let value = |k: usize| -> f64 {
        let (s, second) = spec.locate(k, samples.len());
        let c = samples.get(s).copied().unwrap_or_default();
        let (first, other) = if spec.swap {
            (c.im, c.re)
        } else {
            (c.re, c.im)
        };
        if spec.layout == Layout::Real {
            c.re
        } else if second {
            other
        } else {
            first
        }
    };

    for (g, chunk) in out.chunks_exact_mut(spec.group_bytes).enumerate() {
        let mut word: u64 = 0;
        for j in 0..spec.group_values {
            let x = value(g * spec.group_values + j);
            let part = if spec.is_float() {
                u64::from((x as f32).to_bits())
            } else {
                let raw = quantize(&spec, x);
                let stored = if spec.align == Align::Msb && !spec.is_packed() {
                    raw << (spec.word_bits - spec.bits)
                } else {
                    raw
                };
                stored as u64 & mask(spec.word_bits)
            };
            word |= part << (spec.word_bits * j as u32);
        }
        for (b, byte) in chunk.iter_mut().enumerate() {
            *byte = (word >> (8 * b)) as u8;
        }
        if spec.endian == Endian::Big && !spec.is_packed() {
            chunk.reverse();
        }
    }
    Ok(nbytes)
}

/// Decodes raw bytes with the forward value mapping in f64 and returns the
/// number of samples written. `Real` layouts produce a zero imaginary part.
/// Trailing bytes that do not make a whole sample (or packing group) are
/// ignored.
///
/// # Errors
/// A [`Failure`] for `Bits1` or if `out` cannot hold all samples.
///
/// ```
/// use num_complex::Complex64;
/// use sdrk_device::formats;
/// let mut out = [Complex64::new(0.0, 0.0); 1];
/// sdrk_testkit::encode::decode_reference(&formats::CS16, &[0, 64, 0, 192], &mut out)?;
/// assert_eq!(out[0], Complex64::new(0.5, -0.5));
/// # Ok::<(), sdrk_testkit::Failure>(())
/// ```
pub fn decode_reference(
    format: &SampleFormat,
    bytes: &[u8],
    out: &mut [Complex64],
) -> Result<usize, Failure> {
    decode_inner(format, bytes, out).map_err(Oops::into_failure)
}

fn decode_inner(format: &SampleFormat, bytes: &[u8], out: &mut [Complex64]) -> Result<usize, Oops> {
    let spec = Spec::of(format)?;
    let groups = bytes.len() / spec.group_bytes;
    let n = groups * spec.group_values / spec.values_per_sample();
    let have = out.len();
    let out = out.get_mut(..n).ok_or(Oops {
        msg: "output buffer too small",
        need: n,
        have,
    })?;

    let read = |k: usize| -> Result<f64, Oops> {
        let g = k / spec.group_values;
        let j = (k % spec.group_values) as u32;
        let chunk = bytes
            .get(g * spec.group_bytes..(g + 1) * spec.group_bytes)
            .ok_or_else(|| err("internal read out of range"))?;
        let mut word: u64 = 0;
        for (b, byte) in chunk.iter().enumerate() {
            let pos = if spec.endian == Endian::Big && !spec.is_packed() {
                chunk.len() - 1 - b
            } else {
                b
            };
            word |= u64::from(*byte) << (8 * pos);
        }
        let part = (word >> (spec.word_bits * j)) & mask(spec.word_bits);
        if spec.is_float() {
            return Ok(f64::from(f32::from_bits(part as u32)));
        }
        let shift = spec.word_bits - spec.bits;
        let raw = match (spec.signed, spec.align) {
            (true, Align::Msb) if !spec.is_packed() => sign_extend(part, spec.word_bits) >> shift,
            (true, _) => sign_extend(part & mask(spec.bits), spec.bits),
            (false, Align::Msb) => (part >> shift) as i64,
            (false, _) => (part & mask(spec.bits)) as i64,
        };
        Ok((raw as f64 - spec.offset()) / spec.full_scale)
    };

    for (s, o) in out.iter_mut().enumerate() {
        let first_k = match spec.layout {
            Layout::Real | Layout::Planar => s,
            Layout::Interleaved => 2 * s,
        };
        let second_k = match spec.layout {
            Layout::Planar => n + s,
            Layout::Real | Layout::Interleaved => first_k + 1,
        };
        let first = read(first_k)?;
        *o = if spec.layout == Layout::Real {
            Complex64::new(first, 0.0)
        } else {
            let second = read(second_k)?;
            if spec.swap {
                Complex64::new(second, first)
            } else {
                Complex64::new(first, second)
            }
        };
    }
    Ok(n)
}
