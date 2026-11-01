//! Fallback implementation of the `convert` kernels.

use sdrk_device::{Align, Container, Endian, FullScale, Layout, SampleFormat};

use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::convert::{Convert, ConvertReal, plan};
use crate::num::{q15_from_f32, round_shift, sat16};
use crate::types::{C32, CQ15, ComplexSample, Q15};

/// The formats with a direct loop over the values: one value per byte or 16-bit word, all bits
/// of the container valid, power-of-two full scale, interleaved without I/Q swap or real (the
/// values are then in stream order). The general decoder below costs tens of instructions per
/// value; these loops are a few, and the compiler vectorizes them where the target can.
#[derive(Clone, Copy)]
enum Direct {
    U8,
    I8,
    I16Le,
    I16Be,
}

impl Direct {
    fn of(format: &SampleFormat) -> Option<Self> {
        if format.layout() == Layout::Planar
            || format.swap_iq()
            || format.full_scale() != FullScale::PowerOfTwo
            || format.align() != Align::Lsb
        {
            return None;
        }
        match (format.container(), format.bits(), format.signed()) {
            (Container::U8, 8, false) => Some(Self::U8),
            (Container::I8, 8, true) => Some(Self::I8),
            (Container::I16, 16, true) if format.endian() == Endian::Big => Some(Self::I16Be),
            (Container::I16, 16, true) => Some(Self::I16Le),
            _ => None,
        }
    }

    /// Converts all values of `input` to `out` (as many as both hold) with the formulas of
    /// [`ToF32`]: every result is exact in `f32` (`x * 2^-15`, `x * 2^-7`, `u * 2^-7 - 255/256`),
    /// so these give the same bits.
    fn to_f32(self, input: &[u8], out: &mut [f32]) {
        const INV_128: f32 = 1.0 / 128.0;
        const INV_32768: f32 = 1.0 / 32768.0;
        match self {
            Self::U8 => {
                for (o, &b) in out.iter_mut().zip(input) {
                    *o = f32::from(b) * INV_128 - 255.0 / 256.0;
                }
            }
            Self::I8 => {
                for (o, &b) in out.iter_mut().zip(input) {
                    *o = f32::from(b as i8) * INV_128;
                }
            }
            Self::I16Le => {
                for (o, b) in out.iter_mut().zip(input.as_chunks::<2>().0) {
                    *o = f32::from(i16::from_le_bytes(*b)) * INV_32768;
                }
            }
            Self::I16Be => {
                for (o, b) in out.iter_mut().zip(input.as_chunks::<2>().0) {
                    *o = f32::from(i16::from_be_bytes(*b)) * INV_32768;
                }
            }
        }
    }

    /// Converts all values of `input` to `out` with the formulas of [`ToQ15`] (`U8Pow(7)`,
    /// `Shl(8)`, `Shl(0)`), none of which saturates.
    fn to_q15(self, input: &[u8], out: &mut [i16]) {
        match self {
            Self::U8 => {
                for (o, &b) in out.iter_mut().zip(input) {
                    *o = (2 * i16::from(b) - 255) << 7;
                }
            }
            Self::I8 => {
                for (o, &b) in out.iter_mut().zip(input) {
                    *o = i16::from(b as i8) << 8;
                }
            }
            Self::I16Le => {
                for (o, b) in out.iter_mut().zip(input.as_chunks::<2>().0) {
                    *o = i16::from_le_bytes(*b);
                }
            }
            Self::I16Be => {
                for (o, b) in out.iter_mut().zip(input.as_chunks::<2>().0) {
                    *o = i16::from_be_bytes(*b);
                }
            }
        }
    }
}

/// Reads `N` bytes at `off`.
fn bytes<const N: usize>(input: &[u8], off: usize) -> Option<[u8; N]> {
    input.get(off..off.checked_add(N)?)?.try_into().ok()
}

/// The effective value of a container word (`word` holds `cbits` valid bits).
fn extend(word: u32, cbits: u32, bits: u32, signed: bool, align: Align) -> i32 {
    if signed {
        let left = if align == Align::Lsb {
            32 - bits
        } else {
            32 - cbits
        };
        ((word << left) as i32) >> (32 - bits)
    } else if align == Align::Lsb {
        (word & (u32::MAX >> (32 - bits))) as i32
    } else {
        (word >> (cbits - bits)) as i32
    }
}

/// Maps the effective integer value (or the raw float bits) to f32.
#[derive(Clone, Copy)]
struct ToF32 {
    float: bool,
    wide: bool,
    offset: f32,
    full_scale: f32,
    offset64: f64,
    full_scale64: f64,
}

impl ToF32 {
    fn new(format: &SampleFormat) -> Self {
        let offset64 = if format.container() == Container::U8 {
            127.5
        } else {
            0.0
        };
        let full_scale64 = format.full_scale_value();
        let big_scale = matches!(format.full_scale(), FullScale::Explicit(n) if n > 1 << 24);
        Self {
            float: format.container() == Container::F32,
            wide: format.bits() > 24 || big_scale,
            offset: offset64 as f32,
            full_scale: full_scale64 as f32,
            offset64,
            full_scale64,
        }
    }

    #[inline]
    fn map(self, e: i32) -> f32 {
        if self.float {
            f32::from_bits(e as u32)
        } else if self.wide {
            ((f64::from(e) - self.offset64) / self.full_scale64) as f32
        } else {
            (e as f32 - self.offset) / self.full_scale
        }
    }
}

/// Maps the effective integer value (or the raw float bits) to Q15.
#[derive(Clone, Copy)]
enum ToQ15 {
    Float,
    /// `U8`, power-of-two full scale: `(2u - 255) << shift`.
    U8Pow(u32),
    /// `U8`, explicit full scale.
    U8Explicit(i64),
    /// Signed, at most 16 bits: `e << shift`.
    Shl(u32),
    /// Signed, more than 16 bits: rounding right shift.
    Round(u32),
    /// Signed, explicit full scale.
    Explicit(i64),
}

impl ToQ15 {
    fn new(format: &SampleFormat) -> Self {
        let bits = u32::from(format.bits());
        match (format.container(), format.full_scale()) {
            (Container::F32, _) => Self::Float,
            (Container::U8, FullScale::PowerOfTwo) => Self::U8Pow(15u32.saturating_sub(bits)),
            (Container::U8, FullScale::Explicit(n)) => Self::U8Explicit(i64::from(n)),
            (_, FullScale::Explicit(n)) => Self::Explicit(i64::from(n)),
            (_, FullScale::PowerOfTwo) if bits <= 16 => Self::Shl(16 - bits),
            (_, FullScale::PowerOfTwo) => Self::Round(bits - 16),
        }
    }

    #[inline]
    fn map(self, e: i32) -> Q15 {
        let e = i64::from(e);
        match self {
            Self::Float => q15_from_f32(f32::from_bits(e as u32)),
            Self::U8Pow(shift) => Q15(sat16((2 * e - 255) << shift)),
            Self::U8Explicit(n) => Q15(sat16(((2 * e - 255) * 32768 + n).div_euclid(2 * n))),
            Self::Shl(shift) => Q15(sat16(e << shift)),
            Self::Round(shift) => Q15(sat16(round_shift(e, shift))),
            Self::Explicit(n) => Q15(sat16((e * 65536 + n).div_euclid(2 * n))),
        }
    }
}

/// Decodes value number `k` (in stream order) of `input`.
fn value(format: &SampleFormat, input: &[u8], k: usize) -> Option<i32> {
    let bits = u32::from(format.bits());
    let (signed, align) = (format.signed(), format.align());
    let big = format.endian() == Endian::Big;
    match format.container() {
        Container::U8 | Container::I8 => {
            let b = *input.get(k)?;
            Some(extend(u32::from(b), 8, bits, signed, align))
        }
        Container::I16 => {
            let b = bytes::<2>(input, k.checked_mul(2)?)?;
            let w = if big {
                u16::from_be_bytes(b)
            } else {
                u16::from_le_bytes(b)
            };
            Some(extend(u32::from(w), 16, bits, signed, align))
        }
        Container::I32 => {
            let b = bytes::<4>(input, k.checked_mul(4)?)?;
            let w = if big {
                u32::from_be_bytes(b)
            } else {
                u32::from_le_bytes(b)
            };
            Some(extend(w, 32, bits, signed, align))
        }
        Container::F32 => {
            let b = bytes::<4>(input, k.checked_mul(4)?)?;
            let w = if big {
                u32::from_be_bytes(b)
            } else {
                u32::from_le_bytes(b)
            };
            Some(w as i32)
        }
        Container::Packed12 => {
            let group = bytes::<3>(input, (k / 2).checked_mul(3)?)?;
            let all = u32::from(group[0]) | u32::from(group[1]) << 8 | u32::from(group[2]) << 16;
            Some(extend(
                (all >> ((k % 2) * 12)) & 0xFFF,
                12,
                12,
                true,
                Align::Lsb,
            ))
        }
        Container::Packed10 => {
            let group = bytes::<5>(input, (k / 4).checked_mul(5)?)?;
            let mut all = 0u64;
            for (i, b) in group.iter().enumerate() {
                all |= u64::from(*b) << (8 * i);
            }
            let w = ((all >> ((k % 4) * 10)) & 0x3FF) as u32;
            Some(extend(w, 10, 10, true, Align::Lsb))
        }
        _ => None,
    }
}

/// Decodes `samples` complex samples. `map` converts one component.
fn run_complex<T: ComplexSample>(
    format: &SampleFormat,
    input: &[u8],
    output: &mut [T],
    samples: usize,
    map: impl Fn(i32) -> T::Real,
) -> Result<usize> {
    let internal = || Error::Internal("conversion index");
    let planar = format.layout() == Layout::Planar;
    let swap = format.swap_iq();
    let out = output.get_mut(..samples).ok_or_else(internal)?;
    for (s, o) in out.iter_mut().enumerate() {
        let (a, b) = if planar {
            (value(format, input, s), value(format, input, samples + s))
        } else {
            (value(format, input, 2 * s), value(format, input, 2 * s + 1))
        };
        let (a, b) = (a.ok_or_else(internal)?, b.ok_or_else(internal)?);
        let (i, q) = if swap { (b, a) } else { (a, b) };
        *o = T::new(map(i), map(q));
    }
    Ok(samples)
}

impl Convert<C32> for Fallback {
    fn convert(&self, format: &SampleFormat, input: &[u8], output: &mut [C32]) -> Result<usize> {
        let samples = plan(format, input.len(), output.len(), false)?;
        if let Some(direct) = Direct::of(format) {
            let out = output
                .get_mut(..samples)
                .ok_or(Error::Internal("conversion index"))?;
            direct.to_f32(input, bytemuck::cast_slice_mut(out));
            return Ok(samples);
        }
        let map = ToF32::new(format);
        run_complex(format, input, output, samples, |e| map.map(e))
    }
}

impl Convert<CQ15> for Fallback {
    fn convert(&self, format: &SampleFormat, input: &[u8], output: &mut [CQ15]) -> Result<usize> {
        let samples = plan(format, input.len(), output.len(), false)?;
        if let Some(direct) = Direct::of(format) {
            let out = output
                .get_mut(..samples)
                .ok_or(Error::Internal("conversion index"))?;
            direct.to_q15(input, bytemuck::cast_slice_mut(out));
            return Ok(samples);
        }
        let map = ToQ15::new(format);
        run_complex(format, input, output, samples, |e| map.map(e))
    }
}

fn run_real<T: Copy>(
    format: &SampleFormat,
    input: &[u8],
    output: &mut [T],
    samples: usize,
    map: impl Fn(i32) -> T,
) -> Result<usize> {
    let internal = || Error::Internal("conversion index");
    let out = output.get_mut(..samples).ok_or_else(internal)?;
    for (s, o) in out.iter_mut().enumerate() {
        *o = map(value(format, input, s).ok_or_else(internal)?);
    }
    Ok(samples)
}

impl ConvertReal<f32> for Fallback {
    fn convert_real(
        &self,
        format: &SampleFormat,
        input: &[u8],
        output: &mut [f32],
    ) -> Result<usize> {
        let samples = plan(format, input.len(), output.len(), true)?;
        if let Some(direct) = Direct::of(format) {
            let out = output
                .get_mut(..samples)
                .ok_or(Error::Internal("conversion index"))?;
            direct.to_f32(input, out);
            return Ok(samples);
        }
        let map = ToF32::new(format);
        run_real(format, input, output, samples, |e| map.map(e))
    }
}

impl ConvertReal<Q15> for Fallback {
    fn convert_real(
        &self,
        format: &SampleFormat,
        input: &[u8],
        output: &mut [Q15],
    ) -> Result<usize> {
        let samples = plan(format, input.len(), output.len(), true)?;
        if let Some(direct) = Direct::of(format) {
            let out = output
                .get_mut(..samples)
                .ok_or(Error::Internal("conversion index"))?;
            direct.to_q15(input, bytemuck::cast_slice_mut(out));
            return Ok(samples);
        }
        let map = ToQ15::new(format);
        run_real(format, input, output, samples, |e| map.map(e))
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
mod tests {
    use super::*;
    use sdrk_device::formats;

    /// The direct loops give the values of the general mappings for every byte and for 16-bit
    /// words around zero and the extremes.
    #[test]
    fn direct_loops_match_the_general_mappings() {
        let bytes: std::vec::Vec<u8> = (0..=255).collect();
        for (format, direct) in [(formats::CU8, Direct::U8), (formats::CS8, Direct::I8)] {
            let (q, f) = (ToQ15::new(&format), ToF32::new(&format));
            let mut out_q = std::vec![0i16; 256];
            let mut out_f = std::vec![0f32; 256];
            direct.to_q15(&bytes, &mut out_q);
            direct.to_f32(&bytes, &mut out_f);
            for (k, &b) in bytes.iter().enumerate() {
                let e = value(&format, &bytes, k).unwrap();
                assert_eq!(out_q[k], q.map(e).0, "{format:?}, byte {b}");
                assert_eq!(
                    out_f[k].to_bits(),
                    f.map(e).to_bits(),
                    "{format:?}, byte {b}"
                );
            }
        }
        // Every length up to three vector blocks plus a tail, at several byte offsets of the
        // pattern: the vector body (where the compiler vectorizes the loop), its remainder and
        // the scalar tail give the same bits, and nothing past the input is written.
        let format = formats::CU8;
        let f = ToF32::new(&format);
        for len in 0..=67 {
            for start in [0usize, 1, 7, 128, 200, 255] {
                let input: std::vec::Vec<u8> =
                    (0..len).map(|k| ((start + 37 * k) % 256) as u8).collect();
                let mut out = std::vec![f32::NAN; len + 1];
                Direct::U8.to_f32(&input, &mut out);
                for (k, &b) in input.iter().enumerate() {
                    let e = value(&format, &input, k).unwrap();
                    let (got, want) = (out[k].to_bits(), f.map(e).to_bits());
                    assert_eq!(got, want, "len {len}, byte {b}");
                }
                assert!(out[len].is_nan(), "len {len}: wrote past the input");
            }
        }
        let words: std::vec::Vec<i16> = (-300..300)
            .chain([i16::MIN, i16::MIN + 1, i16::MAX - 1, i16::MAX])
            .collect();
        for (format, direct, big) in [
            (formats::CS16, Direct::I16Le, false),
            (
                SampleFormat::new(
                    Container::I16,
                    16,
                    true,
                    Layout::Interleaved,
                    Align::Lsb,
                    Endian::Big,
                )
                .unwrap(),
                Direct::I16Be,
                true,
            ),
        ] {
            let bytes: std::vec::Vec<u8> = words
                .iter()
                .flat_map(|w| {
                    if big {
                        w.to_be_bytes()
                    } else {
                        w.to_le_bytes()
                    }
                })
                .collect();
            let (q, f) = (ToQ15::new(&format), ToF32::new(&format));
            let mut out_q = std::vec![0i16; words.len()];
            let mut out_f = std::vec![0f32; words.len()];
            direct.to_q15(&bytes, &mut out_q);
            direct.to_f32(&bytes, &mut out_f);
            for k in 0..words.len() {
                let e = value(&format, &bytes, k).unwrap();
                assert_eq!(out_q[k], q.map(e).0, "{format:?}, word {}", words[k]);
                assert_eq!(out_f[k].to_bits(), f.map(e).to_bits(), "{format:?}");
            }
        }
    }
}
