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

/// `x` clamped to `[lo, hi]` (`x` is not NaN).
#[inline]
fn f64_clamp(x: f64, lo: f64, hi: f64) -> f64 {
    let x = if x < lo { lo } else { x };
    if x > hi { hi } else { x }
}

/// `1.5 * 2^52`: adding it to an `f64` of magnitude below `2^51` rounds that to an integer with
/// ties to even, which then sits in the low bits of the sum.
const ROUND_MAGIC: f64 = 6_755_399_441_055_744.0;

/// `x` (`|x| < 2^51`) rounded to the nearest integer with ties to even. With hardware `f64` one
/// addition of [`ROUND_MAGIC`] (in the default rounding mode, which Rust assumes); where `f64` is
/// emulated in software ([`INTEGER_PATH`]) `libm::roundeven`, which costs less than an emulated
/// addition.
#[inline]
fn round_even(x: f64) -> i64 {
    if INTEGER_PATH {
        libm::roundeven(x) as i64
    } else {
        ((x + ROUND_MAGIC).to_bits() as i64).wrapping_sub(ROUND_MAGIC.to_bits() as i64)
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

/// The formats with a direct loop from `f32` values: one value per byte or 16-bit word, all bits
/// of the container valid, power-of-two full scale, interleaved without I/Q swap or real (the
/// values are then in stream order, the same as the `f32` values of the input). The loops work
/// in `f32` only and have no branches, so the compiler vectorizes them where the target can; the
/// general encoder below forms every value in `f64`. Not where `f64` and `f32` are emulated in
/// software ([`INTEGER_PATH`]): there the integer path of the general encoder stays.
#[derive(Clone, Copy)]
enum DirectOut {
    U8,
    I8,
    I16Le,
    I16Be,
}

/// `1.5 * 2^23`: adding it to an `f32` of magnitude at most `2^22` rounds that to an integer with
/// ties to even; the integer `k` then sits in the low 22 bits of the sum (`bits(M + k) =
/// bits(M) + k`, and the low 22 bits of `bits(M)` are zero).
const ROUND_MAGIC_F32: f32 = 12_582_912.0;

/// `2^-47`, added to `x * 128` for `U8` (see [`u8_direct`]).
const TINY: f32 = 1.0 / (1u64 << 47) as f32;

/// Whether [`i16_direct`] and [`u8_direct`] clear NaN first and clamp with `f32::max` and
/// `f32::min`: on x86 and AArch64, where that compiles to one vector instruction each (for
/// `I16` 0.38 instead of 0.46 cycles per value at `avx2`, 2.0 instead of 3.0 on Cortex-A72).
/// Elsewhere [`clamp_nan0`]: the minimum and maximum of wasm32 propagate NaN, so `f32::max` and
/// `f32::min` there take about four times the instructions.
const MAX_MIN_CLAMP: bool = cfg!(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64"
));

/// The storage word of `x` for `I16`: `x * 32768` is exact in `f32`, so its rounding to even
/// there is that of the `f64` formula of [`Enc::f32_raw`]. NaN is cleared and the value clamped
/// to `[-32768, 32767]` before the rounding (with integer bounds the clamp commutes with it); the
/// low 16 bits of the sum with [`ROUND_MAGIC_F32`] are then the two's complement word.
#[inline]
fn i16_direct(x: f32) -> u16 {
    i16_word::<MAX_MIN_CLAMP>(x)
}

/// [`i16_direct`] with the clamp of `MAX_MIN` (see [`MAX_MIN_CLAMP`]; both give the same
/// word).
#[inline]
fn i16_word<const MAX_MIN: bool>(x: f32) -> u16 {
    let v = x * 32768.0;
    let v = if MAX_MIN {
        let v = if v.is_nan() { 0.0 } else { v };
        #[allow(
            clippy::manual_clamp,
            reason = "`clamp` compiles to compares and selects here"
        )]
        v.max(-32768.0).min(32767.0)
    } else {
        clamp_nan0(v, -32768.0, 32767.0)
    };
    (v + ROUND_MAGIC_F32).to_bits() as u16
}

/// `x` clamped to `[lo, hi]` (`lo <= 0 <= hi`) with NaN as 0: the sum of the positive part capped
/// at `hi` and the negative part capped at `lo`, one of which is zero (each compare is false for
/// NaN). It needs no NaN test and no select of a mask, which the compiler would otherwise move
/// behind the narrowing of the bytes.
#[inline]
fn clamp_nan0(v: f32, lo: f32, hi: f32) -> f32 {
    let p = if v > 0.0 { v } else { 0.0 };
    let p = if p < hi { p } else { hi };
    let n = if v < 0.0 { v } else { 0.0 };
    let n = if n > lo { n } else { lo };
    p + n
}

/// The storage byte of `x` for `I8`, as [`i16_direct`] with the scale 128.
#[inline]
fn i8_direct(x: f32) -> u8 {
    let v = clamp_nan0(x * 128.0, -128.0, 127.0);
    (v + ROUND_MAGIC_F32).to_bits() as u8
}

/// The `U8` byte of `x`, `rint(x * 128 + 127.5)` of the `f64` formula of [`Enc::f32_raw`] (NaN
/// as 0, saturated), in `f32`. With `y = x * 128` (exact) the byte is `128 + rint(y - 0.5)`,
/// and `rint(y - 0.5)` is `floor(y)` unless `y` is an integer, where it is the even one of
/// `y - 1` and `y`. The `f64` sum of the formula rounds away the bits of `y` below `2^-46`, so a
/// negative `y` at or above `-2^-47` gives 128 like zero; adding `2^-47` to `y` in `f32` first
/// reproduces that (it leaves every `|y| >= 2^-22` unchanged, where no integer is that close,
/// and moves exactly those tiny negative values to zero or above). `y` is clamped to
/// `[-128, 127.5]` first, which gives the saturated bytes 0 and 255.
///
/// With `r = rint(y)` (ties to even) and the exact `d = y - r` in `[-1/2, 1/2]`, the result is
/// `r - 1` where `d < 0`, and where `d = 0` and `r` is odd; else `r`. A nonzero `d` with an odd
/// `r` (so `|y| >= 1/2`) is at least `2^-24` in magnitude, so both cases are the one comparison
/// `d < t` with `t = 0` for an even `r` and `t = 2^-126` (the bits `1 << 23`) for an odd one.
/// Checked against the `f64` formula for every `f32` (the test `direct_for_every_f32`).
#[inline]
fn u8_direct(x: f32) -> u8 {
    u8_byte::<MAX_MIN_CLAMP>(x)
}

/// [`u8_direct`] with the clamp of `MAX_MIN` (see [`MAX_MIN_CLAMP`]; both give the same byte).
#[inline]
fn u8_byte<const MAX_MIN: bool>(x: f32) -> u8 {
    let y = if MAX_MIN {
        let x = if x.is_nan() { 0.0 } else { x };
        #[allow(
            clippy::manual_clamp,
            reason = "`clamp` compiles to compares and selects here"
        )]
        (x * 128.0 + TINY).max(-128.0).min(127.5)
    } else {
        // NaN gives 0, as `2^-47` would.
        clamp_nan0(x * 128.0 + TINY, -128.0, 127.5)
    };
    let sum = y + ROUND_MAGIC_F32;
    let r = sum - ROUND_MAGIC_F32;
    // `r + 128` as an integer.
    let byte = (sum.to_bits() as i32).wrapping_sub(ROUND_MAGIC_F32.to_bits() as i32 - 128);
    let t = f32::from_bits(((byte & 1) as u32) << 23);
    (byte - i32::from(y - r < t)) as u8
}

impl DirectOut {
    fn of(format: &SampleFormat) -> Option<Self> {
        if INTEGER_PATH
            || format.layout() == Layout::Planar
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

    /// Encodes the values of `input` in stream order into `out` (as many as both hold): the
    /// bytes [`Enc::word`] of [`Enc::f32_raw`] gives.
    fn run(self, input: &[f32], out: &mut [u8]) {
        match self {
            Self::U8 => {
                for (o, &x) in out.iter_mut().zip(input) {
                    *o = u8_direct(x);
                }
            }
            Self::I8 => {
                for (o, &x) in out.iter_mut().zip(input) {
                    *o = i8_direct(x);
                }
            }
            Self::I16Le => {
                for (o, &x) in out.as_chunks_mut::<2>().0.iter_mut().zip(input) {
                    *o = i16_direct(x).to_le_bytes();
                }
            }
            Self::I16Be => {
                for (o, &x) in out.as_chunks_mut::<2>().0.iter_mut().zip(input) {
                    *o = i16_direct(x).to_be_bytes();
                }
            }
        }
    }
}

/// Everything the encoder needs to know about a format.
#[derive(Clone, Copy)]
struct Enc {
    /// `F32` container: the bits are copied.
    float: bool,
    /// `U8` container (offset 127.5).
    u8_container: bool,
    /// Valid bits.
    bits: u32,
    /// Left shift of `Msb` alignment (0 otherwise and for packed containers).
    shift: u32,
    explicit: Option<i64>,
    offset: f64,
    full_scale: f64,
    /// `log2(full_scale)` where [`int_raw`] gives the `f64` result: no offset, a power-of-two full
    /// scale up to `2^21` and a range within `±2^22`.
    int_scale: Option<i32>,
    lo: i64,
    hi: i64,
    /// `lo` and `hi` as `f64`.
    lo_f: f64,
    hi_f: f64,
    /// Mask of the storage word (12 and 10 bits for the packed containers).
    mask: u64,
}

impl Enc {
    fn new(format: &SampleFormat) -> Self {
        let bits = u32::from(format.bits());
        let (cbits, packed): (u32, bool) = match format.container() {
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
            bits,
            shift: if format.align() == Align::Msb && !packed {
                cbits.saturating_sub(bits)
            } else {
                0
            },
            explicit: match format.full_scale() {
                FullScale::Explicit(n) => Some(i64::from(n)),
                FullScale::PowerOfTwo => None,
            },
            offset: if u8_container { 127.5 } else { 0.0 },
            full_scale,
            int_scale,
            lo,
            hi,
            lo_f: lo as f64,
            hi_f: hi as f64,
            mask: if cbits >= 32 {
                u64::from(u32::MAX)
            } else {
                (1u64 << cbits) - 1
            },
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

    /// [`Enc::f32_raw`] by the `f64` formula. The sum is clamped to the integer bounds before
    /// the rounding, which commutes with it (the bounds are integers).
    fn f32_raw_f64(&self, x: f32) -> i64 {
        let a = if x.is_nan() { 0.0 } else { f64::from(x) };
        round_even(f64_clamp(
            a * self.full_scale + self.offset,
            self.lo_f,
            self.hi_f,
        ))
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
        ((raw << self.shift) as u64 & self.mask) as u32
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

/// Writes the `values` storage words of a packed stream (`Packed12`, `Packed10`; `word(k)` is
/// value `k` in stream order).
fn write_packed(
    format: &SampleFormat,
    values: usize,
    word: impl Fn(usize) -> Option<u32>,
    output: &mut [u8],
) -> Result<()> {
    let internal = || Error::Internal("conversion index");
    let w = |k: usize| word(k).ok_or_else(internal);
    match format.container() {
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

/// The order of the components in the stream.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// One value per sample.
    Real,
    /// Both components of a sample next to each other.
    Interleaved,
    /// All first components, then all second ones.
    Planar,
}

/// Writes one storage word of `W` bytes (1, 2 or 4) into `o` (`W` bytes long).
#[inline]
fn put<const W: usize>(o: &mut [u8], v: u32, big: bool) {
    let bytes = if big {
        v.to_be_bytes()
    } else {
        v.to_le_bytes()
    };
    let at = if big { 4 - W } else { 0 };
    if let Some(src) = bytes.get(at..at + W) {
        o.copy_from_slice(src);
    }
}

/// Writes the storage words of `input` in a byte-aligned container of `W` bytes: `word(x, c)`
/// is component `c` of the stream (0 or 1; always 0 for [`Shape::Real`]) of sample `x`.
#[inline]
fn write_aligned<T: Copy, const W: usize>(
    shape: Shape,
    big: bool,
    input: &[T],
    out: &mut [u8],
    word: impl Fn(T, usize) -> u32,
) {
    match shape {
        Shape::Real => {
            for (&x, o) in input.iter().zip(out.chunks_exact_mut(W)) {
                put::<W>(o, word(x, 0), big);
            }
        }
        Shape::Interleaved => {
            for (&x, o) in input.iter().zip(out.chunks_exact_mut(2 * W)) {
                let (a, b) = o.split_at_mut(W);
                put::<W>(a, word(x, 0), big);
                put::<W>(b, word(x, 1), big);
            }
        }
        Shape::Planar => {
            let (first, second) = out.split_at_mut(input.len().saturating_mul(W).min(out.len()));
            for (c, half) in [first, second].into_iter().enumerate() {
                for (&x, o) in input.iter().zip(half.chunks_exact_mut(W)) {
                    put::<W>(o, word(x, c), big);
                }
            }
        }
    }
}

/// Encodes `input` (complex samples for `real == false`, else real ones) into the first bytes of
/// `output`; `word(x, c)` is the storage word of component `c` of sample `x` (0 the real part, 1
/// the imaginary one; 0 for a real sample). Returns the number of bytes written.
fn encode<T: Copy>(
    format: &SampleFormat,
    input: &[T],
    output: &mut [u8],
    real: bool,
    word: impl Fn(T, usize) -> u32,
) -> Result<usize> {
    let bytes = plan_out(format, input.len(), output.len(), real)?;
    let out = output
        .get_mut(..bytes)
        .ok_or(Error::Internal("conversion index"))?;
    let shape = match format.layout() {
        Layout::Real => Shape::Real,
        Layout::Planar => Shape::Planar,
        Layout::Interleaved => Shape::Interleaved,
    };
    let swap = usize::from(format.swap_iq() && shape != Shape::Real);
    // Component `c` of the stream is component `c ^ swap` of the sample.
    let word = |x: T, c: usize| word(x, c ^ swap);
    let big = format.endian() == Endian::Big;
    match format.container() {
        Container::U8 | Container::I8 => write_aligned::<T, 1>(shape, big, input, out, word),
        Container::I16 => write_aligned::<T, 2>(shape, big, input, out, word),
        Container::I32 | Container::F32 => write_aligned::<T, 4>(shape, big, input, out, word),
        _ => {
            let n = input.len();
            let values = if real { n } else { 2 * n };
            let at = |k: usize| -> Option<u32> {
                let (s, c) = match shape {
                    Shape::Interleaved => (k / 2, k % 2),
                    Shape::Planar if k >= n => (k - n, 1),
                    Shape::Real | Shape::Planar => (k, 0),
                };
                input.get(s).map(|&x| word(x, c))
            };
            write_packed(format, values, at, out)?;
        }
    }
    Ok(bytes)
}

/// [`encode`] for a [`DirectOut`] format: `values` are the `f32` values of the samples in
/// memory order (two per complex sample for `real == false`), which is their stream order.
fn encode_direct(
    direct: DirectOut,
    format: &SampleFormat,
    values: &[f32],
    output: &mut [u8],
    real: bool,
) -> Result<usize> {
    let samples = if real { values.len() } else { values.len() / 2 };
    let bytes = plan_out(format, samples, output.len(), real)?;
    let out = output
        .get_mut(..bytes)
        .ok_or(Error::Internal("conversion index"))?;
    direct.run(values, out);
    Ok(bytes)
}

impl ConvertOut<C32> for Fallback {
    fn convert_out(
        &self,
        format: &SampleFormat,
        input: &[C32],
        output: &mut [u8],
    ) -> Result<usize> {
        if let Some(direct) = DirectOut::of(format) {
            return encode_direct(direct, format, bytemuck::cast_slice(input), output, false);
        }
        let enc = Enc::new(format);
        let part = |x: C32, c: usize| if c == 0 { x.re } else { x.im };
        if enc.float {
            encode(format, input, output, false, |x, c| part(x, c).to_bits())
        } else {
            encode(format, input, output, false, |x, c| {
                enc.word(enc.f32_raw(part(x, c)))
            })
        }
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
        encode(format, input, output, false, |x, c| {
            enc.q15_word(if c == 0 { x.re } else { x.im })
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
        if let Some(direct) = DirectOut::of(format) {
            return encode_direct(direct, format, input, output, true);
        }
        let enc = Enc::new(format);
        if enc.float {
            encode(format, input, output, true, |x, _| x.to_bits())
        } else {
            encode(format, input, output, true, |x, _| enc.word(enc.f32_raw(x)))
        }
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
        encode(format, input, output, true, |x, _| enc.q15_word(x))
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

    /// `x` rounded with ties to even by `libm`, the formula before the clamp was moved in front
    /// of the rounding.
    fn libm_raw(enc: &Enc, x: f32) -> i64 {
        let a = if x.is_nan() { 0.0 } else { f64::from(x) };
        let v = libm::roundeven(a * enc.full_scale + enc.offset);
        if v >= enc.hi as f64 {
            enc.hi
        } else if v <= enc.lo as f64 {
            enc.lo
        } else {
            v as i64
        }
    }

    #[test]
    fn round_even_at_ties_and_bounds() {
        let mut values = std::vec![0.0f64, -0.0, 0.25, -0.25, 0.75, -0.75];
        for k in -70i32..70 {
            let t = f64::from(k) + 0.5;
            values.extend([
                t,
                libm::nextafter(t, f64::INFINITY),
                libm::nextafter(t, f64::NEG_INFINITY),
                f64::from(k),
            ]);
        }
        for b in [
            127.5f64,
            255.0,
            2047.5,
            32767.5,
            8_388_607.5,
            2_147_483_647.0,
        ] {
            for t in [b, -b - 1.0, b - 1.0, -b] {
                values.extend([
                    t,
                    libm::nextafter(t, f64::INFINITY),
                    libm::nextafter(t, f64::NEG_INFINITY),
                ]);
            }
        }
        for &x in &values {
            assert_eq!(round_even(x), libm::roundeven(x) as i64, "x = {x:e}");
        }
    }

    #[test]
    fn f64_path_matches_libm_rounding_for_every_format() {
        let mut values = std::vec![
            0.0f32,
            -0.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            f32::MIN,
            f32::from_bits(1),
            1.0,
            -1.0,
            1.0e-30,
            -1.0e-30,
            -1.0e-15,
            -1.0e-17,
            65536.0,
            -65536.0,
        ];
        // Ties and their neighbours at the LSB of every scale, with and without the offset of
        // `U8` (`(k + 1/2) / 2^b` and `k / 2^b`).
        for b in [7, 11, 15, 23, 31] {
            let unit = libm::ldexpf(1.0, -b);
            for k in -300i32..300 {
                for t in [(k as f32 + 0.5) * unit, k as f32 * unit] {
                    values.extend([
                        t,
                        libm::nextafterf(t, f32::INFINITY),
                        libm::nextafterf(t, f32::NEG_INFINITY),
                    ]);
                }
            }
        }
        for f in [
            formats::CU8,
            formats::CS8,
            formats::CS16,
            formats::CS16_PLANAR,
            formats::PACKED12,
            formats::PACKED10,
            formats::SC16_Q11,
            formats::I2S32_MSB,
            formats::RS16,
            SampleFormat::new(
                Container::I32,
                32,
                true,
                Layout::Interleaved,
                Align::Lsb,
                Endian::Big,
            )
            .unwrap(),
        ] {
            let enc = Enc::new(&f);
            for &x in &values {
                assert_eq!(enc.f32_raw_f64(x), libm_raw(&enc, x), "{f:?}, x = {x:e}");
            }
        }
    }

    /// The direct formats of the general encoder and the byte order of their storage words.
    fn direct_formats() -> std::vec::Vec<(SampleFormat, DirectOut)> {
        let real = |container, bits, signed, endian| {
            SampleFormat::new(container, bits, signed, Layout::Real, Align::Lsb, endian).unwrap()
        };
        std::vec![
            (formats::CU8, DirectOut::U8),
            (formats::CS8, DirectOut::I8),
            (formats::CS16, DirectOut::I16Le),
            (formats::RS16, DirectOut::I16Le),
            (real(Container::U8, 8, false, Endian::Little), DirectOut::U8),
            (real(Container::I8, 8, true, Endian::Little), DirectOut::I8),
            (
                real(Container::I16, 16, true, Endian::Big),
                DirectOut::I16Be
            ),
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
                DirectOut::I16Be,
            ),
        ]
    }

    /// The bytes of the general encoder for `x` in a direct format.
    fn general_bytes(enc: &Enc, direct: DirectOut, x: f32) -> std::vec::Vec<u8> {
        let w = enc.word(enc.f32_raw_f64(x));
        match direct {
            DirectOut::U8 | DirectOut::I8 => std::vec![w as u8],
            DirectOut::I16Le => (w as u16).to_le_bytes().to_vec(),
            DirectOut::I16Be => (w as u16).to_be_bytes().to_vec(),
        }
    }

    /// Ties, their neighbours and the bounds at the LSB of the 8-bit and 16-bit scales, with and
    /// without the offset of `U8`, tiny values around zero and the special values.
    fn direct_values() -> std::vec::Vec<f32> {
        let mut values = std::vec![
            0.0f32,
            -0.0,
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            f32::MIN,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MIN_POSITIVE,
            1.0,
            -1.0,
            65536.0,
            -65536.0,
        ];
        for e in 15..60 {
            let t = libm::ldexpf(1.0, -e);
            for v in [t, -t, libm::nextafterf(t, 0.0), -libm::nextafterf(t, 0.0)] {
                values.extend([
                    v,
                    libm::nextafterf(v, f32::INFINITY),
                    libm::nextafterf(v, f32::NEG_INFINITY),
                ]);
            }
        }
        for b in [7, 8, 15, 16] {
            let unit = libm::ldexpf(1.0, -b);
            for k in -300i32..300 {
                for t in [(k as f32 + 0.5) * unit, k as f32 * unit] {
                    values.extend([
                        t,
                        libm::nextafterf(t, f32::INFINITY),
                        libm::nextafterf(t, f32::NEG_INFINITY),
                    ]);
                }
            }
        }
        values
    }

    #[test]
    fn direct_loops_match_the_general_encoder() {
        let values = direct_values();
        for (format, direct) in direct_formats() {
            assert!(
                DirectOut::of(&format).is_some() != INTEGER_PATH,
                "{format:?}"
            );
            let enc = Enc::new(&format);
            let width = enc.mask.count_ones() as usize / 8;
            let mut out = std::vec![0u8; values.len() * width];
            direct.run(&values, &mut out);
            for (k, &x) in values.iter().enumerate() {
                assert_eq!(
                    out[k * width..(k + 1) * width],
                    general_bytes(&enc, direct, x)[..],
                    "{format:?}, x = {x:e}"
                );
                // Both clamp forms (see `MAX_MIN_CLAMP`) give the same.
                assert_eq!(i16_word::<true>(x), i16_word::<false>(x), "x = {x:e}");
                assert_eq!(u8_byte::<true>(x), u8_byte::<false>(x), "x = {x:e}");
            }
        }
        // Planar, swapped, MSB-aligned, explicit full scale and narrower formats take the
        // general encoder.
        for f in [
            formats::CS16_PLANAR,
            formats::SC16_Q11,
            formats::I2S32_MSB,
            formats::PACKED12,
            formats::CF32,
        ] {
            assert!(DirectOut::of(&f).is_none(), "{f:?}");
        }
    }

    #[test]
    fn direct_paths_through_the_kernels_at_every_length() {
        let values = direct_values();
        for (format, direct) in direct_formats() {
            let enc = Enc::new(&format);
            let real = format.layout() == Layout::Real;
            for n in [0usize, 1, 2, 3, 7, 8, 15, 16, 17, 31, 33, 64, 100] {
                let input = &values[..2 * n];
                let want: std::vec::Vec<u8> = input
                    .iter()
                    .flat_map(|&x| general_bytes(&enc, direct, x))
                    .collect();
                // One spare byte: the kernels write only the bytes of the samples.
                let mut out = std::vec![0xA5u8; want.len() + 1];
                let bytes = if real {
                    Fallback.convert_out_real(&format, input, &mut out).unwrap()
                } else {
                    let c: std::vec::Vec<C32> = input
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|p| C32::new(p[0], p[1]))
                        .collect();
                    Fallback.convert_out(&format, &c, &mut out).unwrap()
                };
                assert_eq!(bytes, want.len());
                assert_eq!(out[..bytes], want[..], "{format:?}, n = {n}");
                assert_eq!(out[bytes], 0xA5);
            }
        }
    }

    /// The direct loops against the general encoder for every f32 bit pattern (nightly,
    /// `--ignored`).
    #[test]
    #[ignore = "exhaustive, about a minute on one core"]
    fn direct_for_every_f32() {
        let threads =
            std::thread::available_parallelism().map_or(4, core::num::NonZero::get) as u64;
        for (format, direct) in [
            (formats::CU8, DirectOut::U8),
            (formats::CS8, DirectOut::I8),
            (formats::CS16, DirectOut::I16Le),
        ] {
            let enc = Enc::new(&format);
            let width = enc.mask.count_ones() as usize / 8;
            let bad: u64 = std::thread::scope(|scope| {
                let workers: std::vec::Vec<_> = (0..threads)
                    .map(|t| {
                        scope.spawn(move || {
                            let (from, to) = ((t << 32) / threads, ((t + 1) << 32) / threads);
                            let mut bad = 0u64;
                            let mut xs = std::vec::Vec::with_capacity(4096);
                            let mut out = std::vec![0u8; 4096 * width];
                            let mut b = from;
                            while b < to {
                                let end = (b + 4096).min(to);
                                xs.clear();
                                xs.extend((b..end).map(|v| f32::from_bits(v as u32)));
                                direct.run(&xs, &mut out);
                                for (k, &x) in xs.iter().enumerate() {
                                    if out[k * width..(k + 1) * width]
                                        != general_bytes(&enc, direct, x)[..]
                                    {
                                        bad += 1;
                                    }
                                    // The clamp form of the other targets gives the same.
                                    let other = match direct {
                                        DirectOut::U8 => u8_byte::<true>(x) != u8_byte::<false>(x),
                                        DirectOut::I8 => false,
                                        DirectOut::I16Le | DirectOut::I16Be => {
                                            i16_word::<true>(x) != i16_word::<false>(x)
                                        }
                                    };
                                    bad += u64::from(other);
                                }
                                b = end;
                            }
                            bad
                        })
                    })
                    .collect();
                workers.into_iter().map(|w| w.join().unwrap()).sum()
            });
            assert_eq!(bad, 0, "{format:?}");
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
