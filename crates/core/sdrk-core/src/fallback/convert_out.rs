//! Fallback implementation of the `convert_out` kernels.

use sdrk_device::{Align, Container, Endian, FullScale, Layout, SampleFormat};

use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::convert::{ConvertOut, ConvertOutReal, plan_out};
use crate::types::{C32, CQ15, Q15};

/// `x / 2^s` rounded to the nearest integer, ties to even (arithmetic shift, `s < 63`).
fn shift_even(x: i64, s: u32) -> i64 {
    if s == 0 {
        return x;
    }
    let q = x >> s;
    let r = x - (q << s);
    let half = 1i64 << (s - 1);
    if r > half || (r == half && q & 1 == 1) {
        q + 1
    } else {
        q
    }
}

/// `a / d` for `d > 0` rounded to the nearest integer, ties to even.
fn div_even(a: i64, d: i64) -> i64 {
    let q = a.div_euclid(d);
    let r2 = 2 * a.rem_euclid(d);
    if r2 > d || (r2 == d && q & 1 == 1) {
        q + 1
    } else {
        q
    }
}

/// Whether [`Enc::f32_raw`] takes the integer path where the format allows it. Only on bare-metal
/// 32-bit Arm and RISC-V: there `f64` arithmetic is emulated in software and the integer path
/// costs about an eighth of the instructions (measured on Cortex-M4F, Cortex-M55 and RV32IMAC);
/// on x86, AArch64 and wasm32 the `f64` formula with hardware rounding is as fast or faster.
const INTEGER_PATH: bool = cfg!(all(
    target_os = "none",
    any(target_arch = "arm", target_arch = "riscv32")
));

/// `x * 2^k` of the f32 `bits` (not NaN) rounded to the nearest integer with ties to even, exact
/// within `±2^23`; larger magnitudes (infinity included) give `±2^24`.
fn int_raw(bits: u32, k: i32) -> i64 {
    let biased = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7F_FFFF;
    let (mant, exp) = if biased == 0 {
        (frac, 1)
    } else {
        (frac | 0x80_0000, biased)
    };
    // The value is mant * 2^(exp - 150 + k): a right shift of mant by `shift`.
    let shift = 150 - exp - k;
    let mag: i64 = if exp == 255 || shift <= 0 {
        1 << 24
    } else if shift > 25 {
        // mant < 2^24, so the value is below 1/4.
        0
    } else {
        let q = mant >> shift;
        let r = mant - (q << shift);
        let half = 1u32 << (shift - 1);
        i64::from(if r > half || (r == half && q & 1 == 1) {
            q + 1
        } else {
            q
        })
    };
    if bits >> 31 != 0 { -mag } else { mag }
}

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
    /// `log2(full_scale)` where [`int_raw`] gives the `f64` result: no offset, a power-of-two full
    /// scale up to `2^21` and a range within `±2^22`.
    int_scale: Option<i32>,
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
        let full_scale = format.full_scale_value();
        let int_scale = (!u8_container
            && format.container() != Container::F32
            && full_scale.to_bits() & ((1u64 << 52) - 1) == 0
            && full_scale <= f64::from(1u32 << 21)
            && hi < 1 << 22)
            .then(|| ((full_scale.to_bits() >> 52) as i32) - 1023);
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
            full_scale,
            int_scale,
            lo,
            hi,
        }
    }

    fn clamp(&self, raw: i64) -> i64 {
        raw.clamp(self.lo, self.hi)
    }

    /// The effective raw value of an f32 amplitude: `x * full_scale + offset` computed in f64
    /// (NaN counts as 0), rounded to the nearest integer with ties to even, saturated.
    fn f32_raw(&self, x: f32) -> i64 {
        match self.int_scale {
            Some(k) if INTEGER_PATH => self.f32_raw_int(x, k),
            _ => self.f32_raw_f64(x),
        }
    }

    /// [`Enc::f32_raw`] in integer arithmetic for `int_scale == Some(k)`: `x * 2^k` is exact, and
    /// the clamp to the integer bounds commutes with the rounding.
    fn f32_raw_int(&self, x: f32, k: i32) -> i64 {
        if x.is_nan() {
            0
        } else {
            self.clamp(int_raw(x.to_bits(), k))
        }
    }

    /// [`Enc::f32_raw`] by the `f64` formula.
    fn f32_raw_f64(&self, x: f32) -> i64 {
        let a = if x.is_nan() { 0.0 } else { f64::from(x) };
        let v = libm::roundeven(a * self.full_scale + self.offset);
        if v >= self.hi as f64 {
            self.hi
        } else if v <= self.lo as f64 {
            self.lo
        } else {
            v as i64
        }
    }

    /// The effective raw value of a Q15 sample by the integer rules (ties to even, like the f64
    /// formula on `q / 32768`).
    fn q15_raw(&self, q: Q15) -> i64 {
        let q = i64::from(q.0);
        let raw = match (self.u8_container, self.explicit) {
            (true, None) => {
                let s = 15 - self.bits.min(15);
                shift_even(q + (255 << s), s + 1)
            }
            (true, Some(n)) => div_even(q * n + 255 * 16384, 32768),
            (false, Some(n)) => div_even(q * n, 32768),
            (false, None) if self.bits <= 16 => shift_even(q, 16 - self.bits),
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use sdrk_device::formats;

    /// `x * 2^k` rounded to the nearest integer with ties to even, saturated to `±2^24`.
    fn f64_raw(x: f32, k: i32) -> i64 {
        let v = libm::roundeven(f64::from(x) * libm::ldexp(1.0, k));
        v.clamp(-f64::from(1u32 << 24), f64::from(1u32 << 24)) as i64
    }

    /// `int_raw` is exact within `±2^23` and saturated beyond, so after the clamp to a range
    /// within `±2^22` it equals the `f64` formula.
    fn agrees(bits: u32, k: i32) -> bool {
        let x = f32::from_bits(bits);
        if x.is_nan() {
            return true;
        }
        let (a, b) = (int_raw(bits, k), f64_raw(x, k));
        a == b || (a.abs() >= 1 << 23 && b.abs() >= 1 << 23 && a.signum() == b.signum())
    }

    #[test]
    fn integer_path_at_ties_bounds_and_extremes() {
        let mut values = std::vec![0.0f32, -0.0, f32::MIN_POSITIVE, f32::MAX, f32::INFINITY];
        values.extend([f32::from_bits(1), 0.25, 0.5, 1.5, 2.5, 0.999_999_94, 1.0]);
        for k in 0..=21 {
            let unit = libm::ldexpf(1.0, -k);
            for n in [1i32, 2, 3, 127, 128, 2047, 2048, 32767, 32768] {
                let t = n as f32 * unit;
                for d in [-0.5f32, 0.5, -0.25, 0.75] {
                    values.push(t + d * unit);
                }
                values.extend([
                    t,
                    libm::nextafterf(t, 0.0),
                    libm::nextafterf(t, f32::INFINITY),
                ]);
            }
        }
        for k in 0..=21 {
            for &v in &values {
                for x in [v, -v] {
                    assert!(agrees(x.to_bits(), k), "x = {x:e}, k = {k}");
                }
            }
        }
        for f in [
            formats::CS8,
            formats::CS16,
            formats::PACKED12,
            formats::PACKED10,
            formats::SC16_Q11,
        ] {
            let enc = Enc::new(&f);
            let k = enc.int_scale.unwrap();
            for &v in &values {
                for x in [v, -v, f32::NAN] {
                    assert_eq!(enc.f32_raw_int(x, k), enc.f32_raw_f64(x), "x = {x:e}");
                }
            }
        }
        for f in [formats::CU8, formats::CF32, formats::I2S32_MSB] {
            assert!(Enc::new(&f).int_scale.is_none());
        }
    }

    /// Every f32 bit pattern at every scale `2^0` to `2^21` (nightly, `--ignored`).
    #[test]
    #[ignore = "exhaustive, about half a minute per scale on one core"]
    fn integer_path_for_every_f32() {
        let threads =
            std::thread::available_parallelism().map_or(4, core::num::NonZero::get) as u64;
        for k in 0..=21 {
            let bad: u64 = std::thread::scope(|scope| {
                let workers: std::vec::Vec<_> = (0..threads)
                    .map(|t| {
                        scope.spawn(move || {
                            let (from, to) = ((t << 32) / threads, ((t + 1) << 32) / threads);
                            (from..to).filter(|&b| !agrees(b as u32, k)).count() as u64
                        })
                    })
                    .collect();
                workers.into_iter().map(|w| w.join().unwrap()).sum()
            });
            assert_eq!(bad, 0, "k = {k}");
        }
    }
}
