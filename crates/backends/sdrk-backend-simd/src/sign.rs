//! The exact `±1` and `±j` operations of the shared conventions §A (M49 §1.1) as vector helpers,
//! written once against `fearless_simd` (the portable lane-mask form).
//!
//! `±1` and `±j` are applied by the exact sign and swap operations of the shared conventions §A
//! (`q15_neg_sat`, `cq15_quadrant`, `c32_quadrant`, introduced in M49), never by a product. In
//! `CQ15` a stored negation saturates (`−(−32768) = 32767`); inside a wide accumulation it is
//! plain integer negation.
//!
//! - **Negation (`Q15`):** `0 −sat x` (`psubsw`, `vqsubq_s16`, `i16x8.sub_sat_s`); conditional
//!   negation by a lane mask `m ∈ {0, −1}` as `(x ^ m) −sat m`, which gives 32767 for −32768.
//!   Never `psignw`/`vpsignw`, which wraps −32768 to −32768.
//! - **Negation (`C32`):** XOR with `0x8000_0000`, masked per lane for a conditional sign (only
//!   the sign bit flips: exact for `±0`, `±∞` and NaN payloads).
//! - **Swap of `re` and `im`:** a rotation of every 32-bit lane by 16 (`CQ15`, one sample per
//!   lane), of every 64-bit lane by 32 (`C32`, one sample per lane).
//! - **Quadrant by lane masks:** swap where `q & 1`, then negate `re` where `(q ^ (q >> 1)) & 1`
//!   and `im` where `(q >> 1) & 1`, which equals the table of `sdrk_core::num::cq15_quadrant`.
//!
//! The helpers work on vectors of any width (the native width of the level, or a fixed one).
//! Kernels with a faster form at one level (a byte-shuffle table, a lane permute, a masked
//! rotate) may use it where it gives the same bits; `scramble` does so (see `scramble.rs`).
//!
//! The public functions of this module apply the helpers to slices at the level of the running
//! build, with the scalar helpers of `sdrk_core::num` for the tail. They exist for the tests of
//! the helpers (`tests/sign.rs`) and for callers outside the kernels; the kernels use the vector
//! forms directly.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::num;
use sdrk_core::{C32, CQ15, Q15, Result};

use crate::level::dispatch_fn;
use crate::util::check_len;

/// The sign bit of an `f32`.
const SIGN: u32 = 0x8000_0000;

/// `−x` of every `i16` lane, saturating: `0 −sat x` (−32768 gives 32767).
#[inline(always)]
pub(crate) fn neg_sat<S: SimdToken, V: SimdInt<S, Element = i16>>(x: V) -> V {
    V::splat(x.token(), 0).saturating_sub(x)
}

/// `x` negated (saturating) in the `i16` lanes where `m = −1`, unchanged where `m = 0`:
/// `(x ^ m) −sat m`.
#[inline(always)]
pub(crate) fn neg_sat_where<S: SimdToken, V: SimdInt<S, Element = i16>>(x: V, m: V) -> V {
    (x ^ m).saturating_sub(m)
}

/// The bits of `a` where `m` is set, of `b` elsewhere.
#[inline(always)]
pub(crate) fn select_bits<S: SimdToken, V: SimdInt<S>>(m: V, a: V, b: V) -> V {
    b ^ ((a ^ b) & m)
}

/// `re` and `im` swapped in every 32-bit lane that holds one `CQ15` sample.
#[inline(always)]
pub(crate) fn swap_cq15<S: SimdToken, V: SimdInt<S, Element = u32>>(x: V) -> V {
    (x << 16) | (x >> 16)
}

/// `re` and `im` swapped in every 64-bit lane that holds the bits of one `C32` sample.
#[inline(always)]
pub(crate) fn swap_c32<S: SimdToken, V: SimdInt<S, Element = u64>>(x: V) -> V {
    (x << 32) | (x >> 32)
}

/// The lane masks of the quadrant `q` (low two bits of every 32-bit lane) for a `CQ15` sample
/// per lane: `(swap, neg)`, `swap` all ones where `q & 1`, `neg` with the low 16 bits set where
/// `re` is negated (`(q ^ (q >> 1)) & 1`) and the high 16 bits where `im` is (`(q >> 1) & 1`).
#[inline(always)]
pub(crate) fn cq15_masks<S: SimdToken, V: SimdInt<S, Element = u32>>(q: V) -> (V, V) {
    let zero = V::splat(q.token(), 0);
    let swap = zero - (q & 1);
    let neg_re = zero - ((q ^ (q >> 1)) & 1);
    let neg_im = zero - ((q >> 1) & 1);
    (swap, (neg_re & 0xFFFF) | (neg_im << 16))
}

/// The lane masks of the quadrant `q` (low two bits of every 64-bit lane) for a `C32` sample
/// per lane: `(swap, sign)`, `swap` all ones where `q & 1`, `sign` with bit 31 set where `re`
/// is negated and bit 63 where `im` is.
#[inline(always)]
pub(crate) fn c32_masks<S: SimdToken, V: SimdInt<S, Element = u64>>(q: V) -> (V, V) {
    let zero = V::splat(q.token(), 0);
    let swap = zero - (q & 1);
    let sign = (((q ^ (q >> 1)) & 1) << 31) | (((q >> 1) & 1) << 63);
    (swap, sign)
}

/// Multiplication of the `CQ15` sample in every 32-bit lane of `x` by `j^(q & 3)` (`q` per
/// lane), by swap and saturating negation: the vector form of
/// [`sdrk_core::num::cq15_quadrant`].
#[inline(always)]
pub(crate) fn cq15_quadrant_lanes<S: SimdToken>(x: S::u32s, q: S::u32s) -> S::u32s {
    let (swap, neg) = cq15_masks(q);
    let y = select_bits(swap, swap_cq15(x), x);
    neg_sat_where(y.bitcast::<S::i16s>(), neg.bitcast::<S::i16s>()).bitcast()
}

/// Multiplication of the `C32` sample in every 64-bit lane of `x` (its bits) by `j^(q & 3)`
/// (`q` per lane), by swap and sign flips: the vector form of [`sdrk_core::num::c32_quadrant`].
#[inline(always)]
pub(crate) fn c32_quadrant_lanes<S: SimdToken>(x: S::u64s, q: S::u64s) -> S::u64s {
    let (swap, sign) = c32_masks(q);
    select_bits(swap, swap_c32(x), x) ^ sign
}

/// The binary chip `1 − 2 (b & 1)` applied to the `CQ15` sample in every 32-bit lane of `x`
/// (`b` per lane): both components negated (saturating) where `b & 1`. The vector form of
/// [`sdrk_core::num::cq15_chip`].
#[inline(always)]
pub(crate) fn cq15_chip_lanes<S: SimdToken>(x: S::u32s, b: S::u32s) -> S::u32s {
    let m = S::u32s::splat(b.token(), 0) - (b & 1);
    neg_sat_where(x.bitcast::<S::i16s>(), m.bitcast::<S::i16s>()).bitcast()
}

/// The binary chip `1 − 2 (b & 1)` applied to the `C32` sample in every 64-bit lane of `x`
/// (its bits; `b` per lane): both sign bits flipped where `b & 1`. The vector form of
/// [`sdrk_core::num::c32_chip`].
#[inline(always)]
pub(crate) fn c32_chip_lanes<S: SimdToken>(x: S::u64s, b: S::u64s) -> S::u64s {
    let m = S::u64s::splat(b.token(), 0) - (b & 1);
    x ^ (m & (u64::from(SIGN) | (u64::from(SIGN) << 32)))
}

// ---------------------------------------------------------------------------------------------
// Slices at the level of the running build.

/// `0 −sat x` of every value; returns the number of values done.
#[inline(always)]
fn neg_sat_run<S: SimdToken>(simd: S, x: &[i16], y: &mut [i16], done: &mut usize) {
    let l = S::i16s::LEN;
    let n = x.len().min(y.len()) / l * l;
    let (x, _) = x.split_at(n);
    let (y, _) = y.split_at_mut(n);
    for (i, o) in x.chunks_exact(l).zip(y.chunks_exact_mut(l)) {
        neg_sat(S::i16s::from_slice(simd, i)).store_slice(o);
    }
    *done = n;
}

/// The `CQ15` quadrant map of every sample (its two `i16`, one sample per 32-bit lane) with
/// the index `q[n]`. The slices are `i16` because a `CQ15` is only 2-byte aligned.
#[inline(always)]
fn cq15_quadrant_run<S: SimdToken>(
    simd: S,
    x: &[i16],
    q: &[u8],
    y: &mut [i16],
    chip: bool,
    done: &mut usize,
) {
    let l = S::u32s::LEN;
    let n = (x.len().min(y.len()) / 2).min(q.len()) / l * l;
    let (x, _) = x.split_at(2 * n);
    let (y, _) = y.split_at_mut(2 * n);
    for ((i, o), q) in x
        .chunks_exact(2 * l)
        .zip(y.chunks_exact_mut(2 * l))
        .zip(q.chunks_exact(l))
    {
        let v = S::i16s::from_slice(simd, i).bitcast::<S::u32s>();
        let k = S::u32s::from_fn(simd, |j| u32::from(q.get(j).copied().unwrap_or(0)));
        let r = if chip {
            cq15_chip_lanes::<S>(v, k)
        } else {
            cq15_quadrant_lanes::<S>(v, k)
        };
        r.bitcast::<S::i16s>().store_slice(o);
    }
    *done = n;
}

/// The `C32` quadrant map of every sample (its two `f32` bits as `u32`, one sample per 64-bit
/// lane) with the index `q[n]`. The slices are `u32` because a `C32` is only 4-byte aligned.
#[inline(always)]
fn c32_quadrant_run<S: SimdToken>(
    simd: S,
    x: &[u32],
    q: &[u8],
    y: &mut [u32],
    chip: bool,
    done: &mut usize,
) {
    let l = S::u64s::LEN;
    let n = (x.len().min(y.len()) / 2).min(q.len()) / l * l;
    let (x, _) = x.split_at(2 * n);
    let (y, _) = y.split_at_mut(2 * n);
    for ((i, o), q) in x
        .chunks_exact(2 * l)
        .zip(y.chunks_exact_mut(2 * l))
        .zip(q.chunks_exact(l))
    {
        let v = S::u32s::from_slice(simd, i).bitcast::<S::u64s>();
        let k = S::u64s::from_fn(simd, |j| u64::from(q.get(j).copied().unwrap_or(0)));
        let r = if chip {
            c32_chip_lanes::<S>(v, k)
        } else {
            c32_quadrant_lanes::<S>(v, k)
        };
        r.bitcast::<S::u32s>().store_slice(o);
    }
    *done = n;
}

// The entries, one function per level (see `dispatch_fn!`).
dispatch_fn!(fn run_neg(x: &[i16], y: &mut [i16], done: &mut usize) => neg_sat_run);
dispatch_fn!(
    fn run_cq15(x: &[i16], q: &[u8], y: &mut [i16], chip: bool, done: &mut usize) => cq15_quadrant_run
);
dispatch_fn!(
    fn run_c32(x: &[u32], q: &[u8], y: &mut [u32], chip: bool, done: &mut usize) => c32_quadrant_run
);

/// `output[n] = q15_neg_sat(input[n])` with the vector helper of the level (saturating
/// negation, `−(−32768) = 32767`). `LengthMismatch` if the lengths differ.
///
/// ```
/// use sdrk_backend_simd::sign;
/// use sdrk_core::Q15;
/// let mut y = [Q15(0); 3];
/// sign::q15_neg_sat(&[Q15(-32768), Q15(5), Q15(0)], &mut y)?;
/// assert_eq!(y, [Q15(32767), Q15(-5), Q15(0)]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub fn q15_neg_sat(input: &[Q15], output: &mut [Q15]) -> Result<()> {
    check_len(input.len(), output.len())?;
    let mut done = 0;
    run_neg(
        bytemuck::cast_slice(input),
        bytemuck::cast_slice_mut(output),
        &mut done,
    );
    for (o, &x) in output.iter_mut().zip(input).skip(done) {
        *o = num::q15_neg_sat(x);
    }
    Ok(())
}

/// `output[n] = cq15_quadrant(input[n], q[n])` (multiplication by `j^(q[n] & 3)`) with the
/// vector helper of the level. `LengthMismatch` if the lengths differ.
///
/// ```
/// use sdrk_backend_simd::sign;
/// use sdrk_core::{CQ15, Q15};
/// let x = [CQ15 { re: Q15(100), im: Q15(-32768) }; 2];
/// let mut y = [CQ15::default(); 2];
/// sign::cq15_quadrant(&x, &[1, 2], &mut y)?;
/// assert_eq!(y[0], CQ15 { re: Q15(32767), im: Q15(100) });
/// assert_eq!(y[1], CQ15 { re: Q15(-100), im: Q15(32767) });
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub fn cq15_quadrant(input: &[CQ15], q: &[u8], output: &mut [CQ15]) -> Result<()> {
    cq15_slice(input, q, output, false)
}

/// `output[n] = c32_quadrant(input[n], q[n])` (multiplication by `j^(q[n] & 3)`, IEEE sign
/// flips) with the vector helper of the level. `LengthMismatch` if the lengths differ.
///
/// ```
/// use sdrk_backend_simd::sign;
/// use sdrk_core::C32;
/// let mut y = [C32::default(); 1];
/// sign::c32_quadrant(&[C32::new(1.0, f32::INFINITY)], &[1], &mut y)?;
/// assert_eq!((y[0].re, y[0].im), (f32::NEG_INFINITY, 1.0));
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub fn c32_quadrant(input: &[C32], q: &[u8], output: &mut [C32]) -> Result<()> {
    c32_slice(input, q, output, false)
}

/// `output[n] = cq15_chip(input[n], b[n])` (the chip `1 − 2 (b[n] & 1)`, saturating) with the
/// vector helper of the level. `LengthMismatch` if the lengths differ.
///
/// ```
/// use sdrk_backend_simd::sign;
/// use sdrk_core::{CQ15, Q15};
/// let x = [CQ15 { re: Q15(-32768), im: Q15(7) }; 2];
/// let mut y = [CQ15::default(); 2];
/// sign::cq15_chip(&x, &[0, 1], &mut y)?;
/// assert_eq!(y, [x[0], CQ15 { re: Q15(32767), im: Q15(-7) }]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub fn cq15_chip(input: &[CQ15], b: &[u8], output: &mut [CQ15]) -> Result<()> {
    cq15_slice(input, b, output, true)
}

/// `output[n] = c32_chip(input[n], b[n])` (the chip `1 − 2 (b[n] & 1)`, IEEE sign flips) with
/// the vector helper of the level. `LengthMismatch` if the lengths differ.
///
/// ```
/// use sdrk_backend_simd::sign;
/// use sdrk_core::C32;
/// let mut y = [C32::default(); 1];
/// sign::c32_chip(&[C32::new(0.0, f32::NEG_INFINITY)], &[1], &mut y)?;
/// assert_eq!(y[0].re.to_bits(), (-0.0f32).to_bits());
/// assert_eq!(y[0].im, f32::INFINITY);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub fn c32_chip(input: &[C32], b: &[u8], output: &mut [C32]) -> Result<()> {
    c32_slice(input, b, output, true)
}

fn cq15_slice(input: &[CQ15], q: &[u8], output: &mut [CQ15], chip: bool) -> Result<()> {
    check_len(input.len(), q.len())?;
    check_len(input.len(), output.len())?;
    let mut done = 0;
    run_cq15(
        bytemuck::cast_slice(input),
        q,
        bytemuck::cast_slice_mut(output),
        chip,
        &mut done,
    );
    for ((o, &x), &k) in output.iter_mut().zip(input).zip(q).skip(done) {
        let k = u32::from(k);
        *o = if chip {
            num::cq15_chip(x, k)
        } else {
            num::cq15_quadrant(x, k)
        };
    }
    Ok(())
}

fn c32_slice(input: &[C32], q: &[u8], output: &mut [C32], chip: bool) -> Result<()> {
    check_len(input.len(), q.len())?;
    check_len(input.len(), output.len())?;
    let mut done = 0;
    run_c32(
        bytemuck::cast_slice(input),
        q,
        bytemuck::cast_slice_mut(output),
        chip,
        &mut done,
    );
    for ((o, &x), &k) in output.iter_mut().zip(input).zip(q).skip(done) {
        let k = u32::from(k);
        *o = if chip {
            num::c32_chip(x, k)
        } else {
            num::c32_quadrant(x, k)
        };
    }
    Ok(())
}
