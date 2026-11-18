//! Keyed phase scrambling (`scramble`, M49): element-wise multiplication of a complex stream by
//! unit phasors `exp(j 2π p[n] / 2^b)` from a caller-supplied index sequence, and the
//! descrambler (the conjugate phasors).
//!
//! The index `k = p[n] & (2^b − 1)` (for `inverse` the index `(−k) & (2^b − 1)` of the conjugate
//! phasor) is split into a quadrant `q` and a residue `r` (shared conventions §A.4 of M49 to
//! M56): `q = k >> (b − 2)`, `r = k & (2^(b−2) − 1)` for `b ≥ 2`, and `q = 2k`, `r = 0` for
//! `b = 1`. The output is the exact quadrant map ([`c32_quadrant`], [`cq15_quadrant`]: sign and
//! swap, never a product) of `y`, where `y = x` for `r = 0` and otherwise the `rotate` product of
//! `x` with the table phasor at the residual phase `r << (32 − b)`, read directly from
//! [`COS_F32`] or [`COS_Q15`] at `i = r << (12 − b)` (cosine) and `i + 3072` (sine). For `b ≤ 8`
//! the residual phase has zero low 20 bits, so the direct lookup gives the bits of the `rotate`
//! interpolation. The kernel is stateless; the class is tolerance (`docs/kernels.md`).
//!
//! [`c32_quadrant`]: crate::num::c32_quadrant
//! [`cq15_quadrant`]: crate::num::cq15_quadrant

use crate::error::{Error, Result};
use crate::num::{c32_quadrant, cq15_quadrant};
use crate::tables::{COS_F32, COS_Q15};
use crate::types::{C32, CQ15, ComplexSample, Q15};

/// The smallest number of phase bits.
pub const SCRAMBLE_BITS_MIN: u32 = 1;
/// The largest number of phase bits.
pub const SCRAMBLE_BITS_MAX: u32 = 8;

/// The table offset of the sine: the cosine at the phase `− 2^30`, as `rotate` reads its sine.
const SINE_OFFSET: usize = 3072;

/// Multiplication of a complex stream by unit phasors `exp(j 2π p[n] / 2^b)` (`b` phase bits).
///
/// - `bits` (`b`) is 1 to 8, else `InvalidParameter`; `input`, `phase` and `output` must have
///   the same length, else `LengthMismatch`. Nothing is written on an error; the parameter is
///   checked first.
/// - Each index is taken modulo `2^b` (`k = p[n] & (2^b − 1)`), so a large byte never indexes
///   out of the table and is not an error. `inverse = true` uses `(−k) & (2^b − 1)`, the index
///   of the conjugate phasor, with the same rule as the forward direction.
/// - `b = 1` gives `±x`, `b = 2` the four quadrant maps (the DVB-S2 physical-layer scrambling
///   pattern `j^{R_n}`), both without a multiplication; `k = 0` is the identity at every `b`.
///
/// ```
/// use sdrk_core::{kernels::scramble::Scramble, Fallback, C32};
/// let x = [C32::new(1.0, 2.0); 4];
/// let mut y = [C32::default(); 4];
/// Fallback.scramble(&x, &[0, 1, 2, 3], 2, false, &mut y)?;
/// assert_eq!(y, [C32::new(1.0, 2.0), C32::new(-2.0, 1.0), C32::new(-1.0, -2.0), C32::new(2.0, -1.0)]);
/// let mut back = [C32::default(); 4];
/// Fallback.scramble(&y, &[0, 1, 2, 3], 2, true, &mut back)?;
/// assert_eq!(back, x);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Scramble<T: ComplexSample> {
    /// `output[n] = input[n] · exp(± j 2π p[n] / 2^bits)` (`−` for `inverse`), with the exact
    /// quadrant part and the residual `rotate` product described in the module documentation.
    fn scramble(
        &self,
        input: &[T],
        phase: &[u8],
        bits: u32,
        inverse: bool,
        output: &mut [T],
    ) -> Result<()>;
}

/// `Ok` when `bits` is 1 to 8, else `InvalidParameter`.
///
/// ```
/// use sdrk_core::kernels::scramble::check_bits;
/// assert!(check_bits(8).is_ok());
/// assert!(check_bits(0).is_err() && check_bits(9).is_err());
/// ```
pub fn check_bits(bits: u32) -> Result<()> {
    if (SCRAMBLE_BITS_MIN..=SCRAMBLE_BITS_MAX).contains(&bits) {
        Ok(())
    } else {
        Err(Error::InvalidParameter("scramble bits must be 1 to 8"))
    }
}

/// Checks the arguments of a call: `bits` first (`InvalidParameter`), then the lengths of
/// `phase` and `output` against `input` (`LengthMismatch`). For backend crates.
///
/// ```
/// use sdrk_core::kernels::scramble::check_args;
/// assert!(check_args(2, 4, 4, 4).is_ok());
/// assert!(check_args(2, 4, 3, 4).is_err());
/// assert!(check_args(9, 4, 4, 4).is_err());
/// ```
pub fn check_args(bits: u32, input: usize, phase: usize, output: usize) -> Result<()> {
    check_bits(bits)?;
    super::check_len(input, phase)?;
    super::check_len(input, output)
}

/// The index of the phasor of the phase byte `p`: `p & (2^bits − 1)`, or for `inverse` the
/// index `(−p) & (2^bits − 1)` of the conjugate phasor. `bits` must be 1 to 8 (larger values
/// keep more bits of `p`); for any other `bits` the result is unspecified, but the function
/// never panics (the shift wraps as in [`u32::wrapping_shl`]). For backend crates and tests.
///
/// ```
/// use sdrk_core::kernels::scramble::phase_index;
/// assert_eq!(phase_index(255, 2, false), 3);
/// assert_eq!(phase_index(3, 2, true), 1);
/// assert_eq!(phase_index(0, 8, true), 0);
/// ```
#[inline]
pub const fn phase_index(p: u8, bits: u32, inverse: bool) -> u32 {
    let mask = low_mask(bits);
    let k = p as u32;
    if inverse {
        k.wrapping_neg() & mask
    } else {
        k & mask
    }
}

/// The quadrant `q` (0 to 3) and the residue `r` (0 to `2^(bits − 2) − 1`) of the index `k`
/// (masked to `bits` bits first): `q = k >> (bits − 2)`, `r = k & (2^(bits − 2) − 1)` for
/// `bits ≥ 2`, `q = 2k`, `r = 0` for `bits = 1`. `bits` must be 1 to 8; for any other `bits`
/// the result is unspecified, but the function never panics (the shifts wrap as in
/// [`u32::wrapping_shl`]). For backend crates and tests.
///
/// ```
/// use sdrk_core::kernels::scramble::split_index;
/// assert_eq!(split_index(1, 1), (2, 0));
/// assert_eq!(split_index(3, 2), (3, 0));
/// assert_eq!(split_index(5, 3), (2, 1));
/// assert_eq!(split_index(0xff, 8), (3, 63));
/// ```
#[inline]
pub const fn split_index(k: u32, bits: u32) -> (u32, u32) {
    let k = k & low_mask(bits);
    if bits < 2 {
        (k << 1, 0)
    } else {
        let s = bits - 2;
        (k.wrapping_shr(s), k & low_mask(s))
    }
}

/// `2^bits − 1` for `bits` below 32. The shift wraps (`bits & 31`), so a `bits` outside the
/// contract gives a wrong mask but no overflow panic (R5.4); for `bits` 0 to 8 it is the plain
/// `(1 << bits) − 1`, with the same machine code as the unchecked shift in a release build.
#[inline]
const fn low_mask(bits: u32) -> u32 {
    1u32.wrapping_shl(bits).wrapping_sub(1)
}

/// The table index `i = r << (12 − bits)` of the cosine of the residue `r` (the sine is at
/// `i + 3072`). `bits` must be 2 to 8 and `r` below `2^(bits − 2)`, so `i ≤ 1023`.
#[inline]
const fn table_index(r: u32, bits: u32) -> usize {
    (r << (12 - bits)) as usize
}

pub(crate) fn table_error() -> Error {
    Error::Internal("scramble table index")
}

/// The `C32` residual phasor `(c, s)` of the residue `r ≠ 0` at `bits`: `COS_F32[i]`,
/// `COS_F32[i + 3072]` with `i = r << (12 − bits)`. For backend crates and tests.
///
/// ```
/// use sdrk_core::{kernels::scramble::residual_c32, tables::COS_F32};
/// assert_eq!(residual_c32(1, 3), Some((COS_F32[512], COS_F32[3584])));
/// assert_eq!(residual_c32(4, 3), None);
/// ```
#[inline]
pub fn residual_c32(r: u32, bits: u32) -> Option<(f32, f32)> {
    if !(3..=SCRAMBLE_BITS_MAX).contains(&bits) || r == 0 || r >> (bits - 2) != 0 {
        return None;
    }
    let i = table_index(r, bits);
    Some((*COS_F32.get(i)?, *COS_F32.get(i + SINE_OFFSET)?))
}

/// The `CQ15` residual phasor `(c, s)` of the residue `r ≠ 0` at `bits`: `COS_Q15[i]`,
/// `COS_Q15[i + 3072]` with `i = r << (12 − bits)`, both in `(0, 32767]`. For backend crates
/// and tests.
///
/// ```
/// use sdrk_core::{kernels::scramble::residual_q15, tables::COS_Q15};
/// assert_eq!(residual_q15(1, 3), Some((COS_Q15[512], COS_Q15[3584])));
/// ```
#[inline]
pub fn residual_q15(r: u32, bits: u32) -> Option<(i16, i16)> {
    if !(3..=SCRAMBLE_BITS_MAX).contains(&bits) || r == 0 || r >> (bits - 2) != 0 {
        return None;
    }
    let i = table_index(r, bits);
    Some((*COS_Q15.get(i)?, *COS_Q15.get(i + SINE_OFFSET)?))
}

/// The `rotate` `C32` product `(re c − im s, re s + im c)` in `f32` without FMA.
#[inline]
pub(crate) fn product_c32(x: C32, c: f32, s: f32) -> C32 {
    C32::new(x.re * c - x.im * s, x.re * s + x.im * c)
}

/// The `rotate` `CQ15` product `cq15_mul(x, (c, s))` in `i32`: `|xr c − xi s| ≤ 2 · 32768 ·
/// 32767 < 2^31 − 2^14`, so it is exact.
#[inline]
pub(crate) fn product_cq15(x: CQ15, c: i16, s: i16) -> CQ15 {
    let (xr, xi) = (i32::from(x.re.0), i32::from(x.im.0));
    let (c, s) = (i32::from(c), i32::from(s));
    let narrow = |v: i32| ((v + (1 << 14)) >> 15).clamp(-32768, 32767) as i16;
    CQ15 {
        re: Q15(narrow(xr * c - xi * s)),
        im: Q15(narrow(xr * s + xi * c)),
    }
}

/// The largest number of residues `2^(b − 2)` (`b = 8`).
pub(crate) const RESIDUES: usize = 64;

/// The `C32` residual phasors of a call at `bits` (3 to 8), indexed by the residue; entry 0
/// (`r = 0`, where the input is taken unchanged) and the entries above `2^(bits − 2) − 1` are
/// `(0, 0)` and never used.
pub(crate) fn residual_table_c32(bits: u32) -> Result<[(f32, f32); RESIDUES]> {
    let mut t = [(0.0, 0.0); RESIDUES];
    let n = 1usize << (bits - 2);
    for (r, e) in t.iter_mut().enumerate().take(n).skip(1) {
        *e = residual_c32(r as u32, bits).ok_or_else(table_error)?;
    }
    Ok(t)
}

/// The `CQ15` residual phasors of a call at `bits` (3 to 8), as [`residual_table_c32`].
pub(crate) fn residual_table_q15(bits: u32) -> Result<[(i16, i16); RESIDUES]> {
    let mut t = [(0, 0); RESIDUES];
    let n = 1usize << (bits - 2);
    for (r, e) in t.iter_mut().enumerate().take(n).skip(1) {
        *e = residual_q15(r as u32, bits).ok_or_else(table_error)?;
    }
    Ok(t)
}

/// [`scramble_one_c32`] without the check of `bits` (1 to 8).
// The names of the contract (milestone M49 §1).
#[allow(clippy::many_single_char_names)]
#[inline]
pub(crate) fn one_c32(x: C32, k: u32, bits: u32) -> Result<C32> {
    let (q, r) = split_index(k, bits);
    let y = if r == 0 {
        x
    } else {
        let (c, s) = residual_c32(r, bits).ok_or_else(table_error)?;
        product_c32(x, c, s)
    };
    Ok(c32_quadrant(y, q))
}

/// [`scramble_one_cq15`] without the check of `bits` (1 to 8).
// The names of the contract (milestone M49 §1).
#[allow(clippy::many_single_char_names)]
#[inline]
pub(crate) fn one_cq15(x: CQ15, k: u32, bits: u32) -> Result<CQ15> {
    let (q, r) = split_index(k, bits);
    let y = if r == 0 {
        x
    } else {
        let (c, s) = residual_q15(r, bits).ok_or_else(table_error)?;
        product_cq15(x, c, s)
    };
    Ok(cq15_quadrant(y, q))
}

/// One `C32` sample of the kernel at the index `k` (already inverted for the descrambler, masked
/// to `bits` bits inside): `c32_quadrant(y, q)` with `y = x` for `r = 0` and the `rotate`
/// product with `(COS_F32[i], COS_F32[i + 3072])` otherwise. `bits` outside 1 to 8 is
/// `InvalidParameter`. For backend crates (vector tails) and tests; the fallback computes every
/// sample with the same formula (the phasor from a table of the call with the same entries).
///
/// ```
/// use sdrk_core::{kernels::scramble::scramble_one_c32, C32};
/// assert_eq!(scramble_one_c32(C32::new(1.0, 2.0), 1, 2)?, C32::new(-2.0, 1.0));
/// assert_eq!(scramble_one_c32(C32::new(1.0, 2.0), 0, 8)?, C32::new(1.0, 2.0));
/// assert!(scramble_one_c32(C32::new(1.0, 2.0), 0, 9).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[inline]
pub fn scramble_one_c32(x: C32, k: u32, bits: u32) -> Result<C32> {
    check_bits(bits)?;
    one_c32(x, k, bits)
}

/// One `CQ15` sample of the kernel at the index `k` (already inverted for the descrambler,
/// masked to `bits` bits inside): `cq15_quadrant(y, q)` with `y = x` for `r = 0` and
/// `cq15_mul(x, (COS_Q15[i], COS_Q15[i + 3072]))` in `i32` otherwise. `bits` outside 1 to 8 is
/// `InvalidParameter`. For backend crates (vector tails) and tests; the fallback computes every
/// sample with the same formula (the phasor from a table of the call with the same entries).
///
/// ```
/// use sdrk_core::{kernels::scramble::scramble_one_cq15, CQ15, Q15};
/// let x = CQ15 { re: Q15(-32768), im: Q15(5) };
/// assert_eq!(scramble_one_cq15(x, 1, 1)?, CQ15 { re: Q15(32767), im: Q15(-5) });
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[inline]
pub fn scramble_one_cq15(x: CQ15, k: u32, bits: u32) -> Result<CQ15> {
    check_bits(bits)?;
    one_cq15(x, k, bits)
}
