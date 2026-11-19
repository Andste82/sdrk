//! Keyed phase scrambling (`scramble`, M49, revised by M49.1): element-wise multiplication of a
//! complex stream by unit phasors `exp(j 2π p[n] / 2^b)` from a caller-supplied index sequence,
//! and the descrambler (the conjugate phasors).
//!
//! The index is `k = p[n] & (2^b − 1)` (for `inverse` the index `(−k) & (2^b − 1)` of the
//! conjugate phasor).
//!
//! - `b ≤ 2`: the exact quadrant map ([`c32_quadrant`], [`cq15_quadrant`]: sign and swap, never a
//!   product) with the quadrant `q = 2k` (`b = 1`) or `q = k` (`b = 2`); lossless (shared
//!   conventions §A of M49 to M56).
//! - `b ≥ 3`: one table phasor and one product per sample (M49.1): the `rotate` product of `x`
//!   with the phasor at the full phase `k << (32 − b)`, read directly from [`COS_F32`] or
//!   [`COS_Q15`] at `i = k << (12 − b)` (cosine) and `(i + 3072) & 4095` (sine). For `b ≤ 8` the
//!   phase has zero low 20 bits, so the direct lookup gives the bits of the `rotate`
//!   interpolation: the output equals one `rotate` step at that phase bit for bit. There is no
//!   quadrant map and no exception for `k = 0` (which multiplies by `(32767, 0)` in `CQ15` and by
//!   `(1, s₀)` with the tiny table sine in `C32`).
//!
//! The kernel is stateless; the class is tolerance (`docs/kernels.md`).
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
///   pattern `j^{R_n}`), both without a multiplication; `k = 0` is the identity there.
/// - `b ≥ 3` multiplies every sample by the table phasor of `k` (the `rotate` product, also for
///   `k = 0`).
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
    /// `output[n] = input[n] · exp(± j 2π p[n] / 2^bits)` (`−` for `inverse`): the exact
    /// quadrant map for `bits ≤ 2`, the `rotate` product with the table phasor for `bits ≥ 3`, as
    /// described in the module documentation.
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
/// [`u32::wrapping_shl`]). The kernel uses the quadrant for `bits ≤ 2` only (where `r = 0`);
/// since M49.1 `bits ≥ 3` takes the phasor of the whole index. For backend crates and tests.
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

/// The table index `i = k << (12 − bits)` of the cosine of the index `k` (masked to `bits`
/// bits); the sine is at `(i + 3072) & 4095`. `bits` must be 1 to 8 (larger `bits` give an
/// unspecified index below 4096, never a panic).
#[inline]
const fn table_index(k: u32, bits: u32) -> usize {
    ((k & low_mask(bits)).wrapping_shl(12u32.wrapping_sub(bits)) & 4095) as usize
}

/// The table index of the sine of the cosine index `i`: the cosine at the phase `− 2^30`, as
/// `rotate` reads its sine.
#[inline]
const fn sine_index(i: usize) -> usize {
    (i + SINE_OFFSET) & 4095
}

/// The `C32` table phasor `(COS_F32[i], COS_F32[(i + 3072) & 4095])`, `i = k << (12 − bits)`,
/// of the index `k` (masked to `bits` bits): the phasor of the kernel for `bits ≥ 3`.
#[inline]
pub(crate) fn phasor_c32(k: u32, bits: u32) -> (f32, f32) {
    let i = table_index(k, bits);
    (
        COS_F32.get(i).copied().unwrap_or(0.0),
        COS_F32.get(sine_index(i)).copied().unwrap_or(0.0),
    )
}

/// The `CQ15` table phasor of the index `k`, as [`phasor_c32`] from `COS_Q15`.
#[inline]
pub(crate) fn phasor_q15(k: u32, bits: u32) -> (i16, i16) {
    let i = table_index(k, bits);
    (
        COS_Q15.get(i).copied().unwrap_or(0),
        COS_Q15.get(sine_index(i)).copied().unwrap_or(0),
    )
}

/// The `C32` table phasor `(c, s)` of the residue `r ≠ 0` at `bits` (3 to 8): `COS_F32[i]`,
/// `COS_F32[i + 3072]` with `i = r << (12 − bits)`, `r` below `2^(bits − 2)` (an angle in
/// `(0, π/2)`); `None` otherwise. Since M49.1 the kernel uses the phasor of the whole index for
/// `bits ≥ 3`, of which this is the first quadrant (the same entries for `k = r`). For backend
/// crates and tests.
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

/// The `CQ15` table phasor `(c, s)` of the residue `r ≠ 0` at `bits`: `COS_Q15[i]`,
/// `COS_Q15[i + 3072]` with `i = r << (12 − bits)`, both in `(0, 32767]`, as
/// [`residual_c32`]. For backend crates and tests.
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

/// The `rotate` `CQ15` product `cq15_mul(x, (c, s))` in `i32`, exact: a table phasor has
/// `|c| + |s| ≤ 46341` (at most `32768 · √2`), so `|xr c − xi s| ≤ 32768 · 46341 < 2^31 − 2^14`.
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

/// [`scramble_one_c32`] without the check of `bits` (1 to 8).
#[inline]
fn one_c32(x: C32, k: u32, bits: u32) -> C32 {
    if bits <= 2 {
        let (q, _) = split_index(k, bits);
        c32_quadrant(x, q)
    } else {
        let (c, s) = phasor_c32(k, bits);
        product_c32(x, c, s)
    }
}

/// [`scramble_one_cq15`] without the check of `bits` (1 to 8).
#[inline]
fn one_cq15(x: CQ15, k: u32, bits: u32) -> CQ15 {
    if bits <= 2 {
        let (q, _) = split_index(k, bits);
        cq15_quadrant(x, q)
    } else {
        let (c, s) = phasor_q15(k, bits);
        product_cq15(x, c, s)
    }
}

/// One `C32` sample of the kernel at the index `k` (already inverted for the descrambler, masked
/// to `bits` bits inside): for `bits ≤ 2` the exact quadrant map `c32_quadrant(x, q)` (`q` of
/// [`split_index`]); for `bits ≥ 3` the `rotate` product `(re c − im s, re s + im c)` in `f32`
/// without FMA with `c = COS_F32[i]`, `s = COS_F32[(i + 3072) & 4095]`, `i = k << (12 − bits)`.
/// `bits` outside 1 to 8 is `InvalidParameter`. For backend crates (vector tails) and tests;
/// the fallback computes every sample with the same formula.
///
/// ```
/// use sdrk_core::{kernels::scramble::scramble_one_c32, tables::COS_F32, C32};
/// assert_eq!(scramble_one_c32(C32::new(1.0, 2.0), 1, 2)?, C32::new(-2.0, 1.0));
/// // bits 8, k = 64 (a quarter turn): the product with the table phasor, not an exact swap.
/// let (c, s) = (COS_F32[1024], COS_F32[0]);
/// assert_eq!(
///     scramble_one_c32(C32::new(1.0, 2.0), 64, 8)?,
///     C32::new(1.0 * c - 2.0 * s, 1.0 * s + 2.0 * c)
/// );
/// assert!(scramble_one_c32(C32::new(1.0, 2.0), 0, 9).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[inline]
pub fn scramble_one_c32(x: C32, k: u32, bits: u32) -> Result<C32> {
    check_bits(bits)?;
    Ok(one_c32(x, k, bits))
}

/// One `CQ15` sample of the kernel at the index `k` (already inverted for the descrambler,
/// masked to `bits` bits inside): for `bits ≤ 2` the exact quadrant map `cq15_quadrant(x, q)`
/// (saturating negation); for `bits ≥ 3` `cq15_mul(x, (COS_Q15[i], COS_Q15[(i + 3072) & 4095]))`
/// in `i32`, `i = k << (12 − bits)` (the `rotate` product). `bits` outside 1 to 8 is
/// `InvalidParameter`. For backend crates (vector tails) and tests; the fallback computes every
/// sample with the same formula.
///
/// ```
/// use sdrk_core::{kernels::scramble::scramble_one_cq15, CQ15, Q15};
/// let x = CQ15 { re: Q15(-32768), im: Q15(5) };
/// assert_eq!(scramble_one_cq15(x, 1, 1)?, CQ15 { re: Q15(32767), im: Q15(-5) });
/// // bits 3, k = 0: the product with (32767, 0) scales by 1 − 2^-15.
/// let y = CQ15 { re: Q15(-32768), im: Q15(16384) };
/// assert_eq!(scramble_one_cq15(y, 0, 3)?, CQ15 { re: Q15(-32767), im: Q15(16384) });
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[inline]
pub fn scramble_one_cq15(x: CQ15, k: u32, bits: u32) -> Result<CQ15> {
    check_bits(bits)?;
    Ok(one_cq15(x, k, bits))
}
