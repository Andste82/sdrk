//! Fallback implementation of the `convert_out` kernels.

use sdrk_device::{Align, Container, Endian, FullScale, Layout, SampleFormat};

use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::convert::{ConvertOut, ConvertOutReal, plan_out};
use crate::num::round_shift;
use crate::types::{C32, CQ15, Q15};

/// Everything the encoder needs to know about a format.
#[derive(Clone, Copy)]
struct Enc {
    /// `F32` container: the bits are copied.
    float: bool,
    /// `U8` container (offset 127.5).
    u8_container: bool,
    /// Bits of the storage word (12 and 10 for the packed containers).
    cbits: u32,
    /// Valid bits.
    bits: u32,
    /// Left shift of `Msb` alignment (0 otherwise and for packed containers).
    shift: u32,
    big: bool,
    explicit: Option<i64>,
    offset: f64,
    full_scale: f64,
    lo: i64,
    hi: i64,
}

impl Enc {
    fn new(format: &SampleFormat) -> Self {
        let bits = u32::from(format.bits());
        let (cbits, packed) = match format.container() {
            Container::U8 | Container::I8 => (8, false),
            Container::I16 => (16, false),
            Container::I32 | Container::F32 => (32, false),
            Container::Packed12 => (12, true),
            Container::Packed10 => (10, true),
            _ => (1, true),
        };
        let (lo, hi) = if format.signed() {
            let half = 1i64 << (bits - 1);
            (-half, half - 1)
        } else {
            (0, (1i64 << bits) - 1)
        };
        let u8_container = format.container() == Container::U8;
        Self {
            float: format.container() == Container::F32,
            u8_container,
            cbits,
            bits,
            shift: if format.align() == Align::Msb && !packed {
                cbits.saturating_sub(bits)
            } else {
                0
            },
            big: format.endian() == Endian::Big,
            explicit: match format.full_scale() {
                FullScale::Explicit(n) => Some(i64::from(n)),
                FullScale::PowerOfTwo => None,
            },
            offset: if u8_container { 127.5 } else { 0.0 },
            full_scale: format.full_scale_value(),
            lo,
            hi,
        }
    }

    fn clamp(&self, raw: i64) -> i64 {
        raw.clamp(self.lo, self.hi)
    }

    /// The effective raw value of an f32 amplitude: `floor(x * full_scale + offset + 0.5)`,
    /// computed in f64 (NaN counts as 0), saturated.
    fn f32_raw(&self, x: f32) -> i64 {
        let a = if x.is_nan() { 0.0 } else { f64::from(x) };
        let v = libm::floor(a * self.full_scale + self.offset + 0.5);
        if v >= self.hi as f64 {
            self.hi
        } else if v <= self.lo as f64 {
            self.lo
        } else {
            v as i64
        }
    }

    /// The effective raw value of a Q15 sample by the integer rules.
    fn q15_raw(&self, q: Q15) -> i64 {
        let q = i64::from(q.0);
        let raw = match (self.u8_container, self.explicit) {
            (true, None) => {
                let s = 15 - self.bits.min(15);
                round_shift(q + (255 << s), s + 1)
            }
            (true, Some(n)) => (q * n + 255 * 16384 + 16384).div_euclid(32768),
            (false, Some(n)) => (q * 2 * n + 32768).div_euclid(65536),
            (false, None) if self.bits <= 16 => round_shift(q, 16 - self.bits),
            (false, None) => q << (self.bits - 16),
        };
        self.clamp(raw)
    }

    /// The storage word of an effective raw value (alignment, truncation to the container).
    fn word(&self, raw: i64) -> u32 {
        let mask = if self.cbits >= 32 {
            u64::from(u32::MAX)
        } else {
            (1u64 << self.cbits) - 1
        };
        ((raw << self.shift) as u64 & mask) as u32
    }

    fn f32_word(&self, x: f32) -> u32 {
        if self.float {
            x.to_bits()
        } else {
            self.word(self.f32_raw(x))
        }
    }

    fn q15_word(&self, q: Q15) -> u32 {
        if self.float {
            q.to_f32().to_bits()
        } else {
            self.word(self.q15_raw(q))
        }
    }
}

/// The first `N` bytes of a byte array as a slice.
fn head<const N: usize>(bytes: &[u8; N], n: usize) -> Result<&[u8]> {
    bytes.get(..n).ok_or(Error::Internal("conversion index"))
}

/// Writes the `values` storage words of the stream (`word(k)` is value `k` in stream order).
fn write_words(
    format: &SampleFormat,
    enc: &Enc,
    values: usize,
    word: impl Fn(usize) -> Option<u32>,
    output: &mut [u8],
) -> Result<()> {
    let internal = || Error::Internal("conversion index");
    let w = |k: usize| word(k).ok_or_else(internal);
    match format.container() {
        Container::U8 | Container::I8 => {
            for (k, b) in output.iter_mut().take(values).enumerate() {
                *b = w(k)? as u8;
            }
        }
        Container::I16 => {
            for (k, chunk) in output.chunks_exact_mut(2).take(values).enumerate() {
                let v = w(k)? as u16;
                chunk.copy_from_slice(&if enc.big {
                    v.to_be_bytes()
                } else {
                    v.to_le_bytes()
                });
            }
        }
        Container::I32 | Container::F32 => {
            for (k, chunk) in output.chunks_exact_mut(4).take(values).enumerate() {
                let v = w(k)?;
                chunk.copy_from_slice(&if enc.big {
                    v.to_be_bytes()
                } else {
                    v.to_le_bytes()
                });
            }
        }
        Container::Packed12 => {
            for (g, chunk) in output.chunks_exact_mut(3).take(values / 2).enumerate() {
                let acc = (w(2 * g)? & 0xFFF) | (w(2 * g + 1)? & 0xFFF) << 12;
                chunk.copy_from_slice(head(&acc.to_le_bytes(), 3)?);
            }
        }
        Container::Packed10 => {
            for (g, chunk) in output.chunks_exact_mut(5).take(values / 4).enumerate() {
                let mut acc = 0u64;
                for j in 0..4 {
                    acc |= u64::from(w(4 * g + j)? & 0x3FF) << (10 * j);
                }
                chunk.copy_from_slice(head(&acc.to_le_bytes(), 5)?);
            }
        }
        _ => return Err(Error::Unsupported("1-bit containers")),
    }
    Ok(())
}

/// Encodes complex samples; `conv(sample, imag)` gives the storage word of one component.
fn encode_complex<T: Copy>(
    format: &SampleFormat,
    enc: &Enc,
    input: &[T],
    output: &mut [u8],
    conv: impl Fn(T, bool) -> u32,
) -> Result<usize> {
    let bytes = plan_out(format, input.len(), output.len(), false)?;
    let n = input.len();
    let planar = format.layout() == Layout::Planar;
    let swap = format.swap_iq();
    let word = |k: usize| -> Option<u32> {
        let (s, second) = if planar {
            if k >= n { (k - n, true) } else { (k, false) }
        } else {
            (k / 2, k % 2 == 1)
        };
        input.get(s).map(|x| conv(*x, second != swap))
    };
    let out = output
        .get_mut(..bytes)
        .ok_or(Error::Internal("conversion index"))?;
    write_words(format, enc, n * 2, word, out)?;
    Ok(bytes)
}

/// Encodes real samples; `conv(sample)` gives the storage word.
fn encode_real<T: Copy>(
    format: &SampleFormat,
    enc: &Enc,
    input: &[T],
    output: &mut [u8],
    conv: impl Fn(T) -> u32,
) -> Result<usize> {
    let bytes = plan_out(format, input.len(), output.len(), true)?;
    let word = |k: usize| input.get(k).map(|x| conv(*x));
    let out = output
        .get_mut(..bytes)
        .ok_or(Error::Internal("conversion index"))?;
    write_words(format, enc, input.len(), word, out)?;
    Ok(bytes)
}

impl ConvertOut<C32> for Fallback {
    fn convert_out(
        &self,
        format: &SampleFormat,
        input: &[C32],
        output: &mut [u8],
    ) -> Result<usize> {
        let enc = Enc::new(format);
        encode_complex(format, &enc, input, output, |x, im| {
            enc.f32_word(if im { x.im } else { x.re })
        })
    }
}

impl ConvertOut<CQ15> for Fallback {
    fn convert_out(
        &self,
        format: &SampleFormat,
        input: &[CQ15],
        output: &mut [u8],
    ) -> Result<usize> {
        let enc = Enc::new(format);
        encode_complex(format, &enc, input, output, |x, im| {
            enc.q15_word(if im { x.im } else { x.re })
        })
    }
}

impl ConvertOutReal<f32> for Fallback {
    fn convert_out_real(
        &self,
        format: &SampleFormat,
        input: &[f32],
        output: &mut [u8],
    ) -> Result<usize> {
        let enc = Enc::new(format);
        encode_real(format, &enc, input, output, |x| enc.f32_word(x))
    }
}

impl ConvertOutReal<Q15> for Fallback {
    fn convert_out_real(
        &self,
        format: &SampleFormat,
        input: &[Q15],
        output: &mut [u8],
    ) -> Result<usize> {
        let enc = Enc::new(format);
        encode_real(format, &enc, input, output, |x| enc.q15_word(x))
    }
}
