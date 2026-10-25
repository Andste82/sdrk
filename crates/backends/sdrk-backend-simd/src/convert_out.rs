//! `convert_out`: `C32` / `CQ15` to raw device bytes.
//!
//! The same formats as in `convert` run vectorised (`U8`, `I8` and little-endian `I16`); the
//! rest goes to the fallback. Bit-exact with the fallback: `x * full_scale + offset` rounded to
//! the nearest integer with ties to even, saturation and NaN as amplitude 0.
//!
//! - `I16` and `I8` have a power-of-two scale and no offset, so `v = x * scale` is exact in
//!   `f32` and its rounding to even is the specification. NaN is cleared and the upper side
//!   capped at `scale` (the first value beyond the range) where the conversion would not do it;
//!   the saturating narrowing does the rest.
//! - `U8` adds the offset 127.5 in `f64`, which this path does not form. With `y = x * 128`
//!   (exact) the raw value is `128 + rint(y - 0.5)`. The `f64` sum of the specification rounds
//!   away the bits of `y` below `2^-46`, so a negative `y` at or above `-2^-47` gives 128 like
//!   zero; adding `2^-47` to `y` in `f32` first reproduces that (it leaves every `|y| >= 2^-22`
//!   unchanged and moves exactly those tiny negative values to zero or above). The generic code
//!   and AVX-512 take `g = floor(y)`, which is the value unless `y` is an integer, where it is the
//!   even one of `y - 1` and `y`: `rint(g - 0.5)`. The level paths of SSE2, SSE4.2, AVX2 and NEON
//!   convert `y - C` with `C = 0.5 - 2^-25` instead (x86 `(y + 2^-47) - C`, which gives the same
//!   integers), which `f32` rounds to the right integer for every `y` but the negative ones above
//!   `-2^-25` (they give 0 instead of -1; with 0.5 instead of `C`, `y = -1 + 2^-24` would also
//!   round to the tie -1.5), and bound the saturated bytes:
//!   by -1 where `y < -2^-47`. That bound `b` is a byte per value, -1 (`0xFF`) there and 127
//!   elsewhere, with the low 7 bits set on a sign byte: on x86 the signed pack of the bits of
//!   `y + 2^-47`, on NEON the narrowed lane masks of the compare. On x86 `cvtps2dq` also gives
//!   `i32::MIN` (the byte -128) where `y - C >= 2^31`, so `b` is applied twice there: as unsigned
//!   minimum, which turns the negative bytes of the lanes with `b = 127` (where the value cannot
//!   be negative otherwise) into 127, and as signed minimum (SSE2, which has no `pminsb`: the
//!   unsigned minimum after flipping the sign bits of both). Each form is checked against the
//!   `f64` formula for all 2^32 inputs. `rint(...) - 128` is saturated to `i8` and turned into
//!   the offset binary byte by flipping the sign bit.
//!
//! Each level has its own path (`core::arch`, decision D1) for whole vectors; the generic
//! `fearless_simd` code runs on a level without one, and the few values after the last whole
//! vector go through the scalar formulas.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::convert::ConvertOut;
use sdrk_core::{C32, CQ15, Fallback, Result};
use sdrk_device::SampleFormat;

use crate::Simd;
use crate::convert::{Kind, fast_kind};
use crate::level::dispatch;
use crate::util;

/// Number of output bytes for the fast path, else `None` (the fallback reports the error).
fn fast_bytes(kind: Kind, components: usize, output_len: usize) -> Option<usize> {
    let bytes = components.checked_mul(kind.width())?;
    (output_len >= bytes).then_some(bytes)
}

/// Full scale of `CS16`.
const S16: f32 = 32768.0;
/// Full scale of `CS8` and `CU8`.
const S8: f32 = 128.0;
/// `2^-47`, added to `x * 128` for `CU8` (see the module documentation).
const TINY: f32 = 1.0 / (1u64 << 47) as f32;
/// `0.5 - 2^-25`, subtracted from `x * 128 + 2^-47` for `CU8` on the level paths (see the module
/// documentation).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const C: f32 = 0.5 - 1.0 / (1u32 << 25) as f32;

/// Runs `body` on every whole block of `step` floats and `bytes` output bytes (see
/// [`util::blocks`]); returns the number of floats done.
#[inline(always)]
fn blocks(
    input: &[f32],
    out: &mut [u8],
    step: usize,
    bytes: usize,
    body: impl FnMut(&[f32], &mut [u8]),
) -> usize {
    util::blocks(input, out, step, bytes, body) * step
}

/// Runs `unit` (one block of `step` floats to `bytes` output bytes) over all whole blocks,
/// `unroll` blocks per iteration of the main loop; returns the number of floats done. Only the
/// level paths use it.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn run(
    input: &[f32],
    out: &mut [u8],
    step: usize,
    bytes: usize,
    unroll: usize,
    mut unit: impl FnMut(&[f32], &mut [u8]),
) -> usize {
    let done = blocks(input, out, unroll * step, unroll * bytes, |i, o| {
        for (i, o) in i.chunks_exact(step).zip(o.chunks_exact_mut(bytes)) {
            unit(i, o);
        }
    });
    let input = input.get(done..).unwrap_or_default();
    let out = out.get_mut(done / step * bytes..).unwrap_or_default();
    done + blocks(input, out, step, bytes, unit)
}

/// The four vectors of 4 floats of a block of 16. Only the 128-bit level paths use it.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn quarters(i: &[f32]) -> [&[f32]; 4] {
    let (ab, cd) = i.split_at(8);
    let (a, b) = ab.split_at(4);
    let (c, d) = cd.split_at(4);
    [a, b, c, d]
}

/// `rint(x * scale)` for `CS16` and `CS8`, NaN as 0, saturated to `[-scale, scale - 1]`.
#[inline(always)]
fn signed_scalar(x: f32, scale: f32) -> i32 {
    let v = x * scale;
    if v.is_nan() {
        0
    } else {
        libm::roundevenf(v).clamp(-scale, scale - 1.0) as i32
    }
}

/// The `CU8` byte of the specification, `rint(x * 128 + 127.5)` in `f64` (NaN as 0), saturated.
#[inline(always)]
fn u8_scalar(x: f32) -> u8 {
    let a = if x.is_nan() { 0.0 } else { f64::from(x) };
    libm::roundeven(a * 128.0 + 127.5).clamp(0.0, 255.0) as u8
}

/// The scalar formulas for the values after the last whole vector.
#[inline(always)]
fn tail(kind: Kind, input: &[f32], out: &mut [u8]) {
    match kind {
        Kind::I16 => {
            for (&x, o) in input.iter().zip(out.chunks_exact_mut(2)) {
                o.copy_from_slice(&(signed_scalar(x, S16) as i16).to_le_bytes());
            }
        }
        Kind::I8 => {
            for (&x, o) in input.iter().zip(out) {
                *o = signed_scalar(x, S8) as i8 as u8;
            }
        }
        Kind::U8 => {
            for (&x, o) in input.iter().zip(out) {
                *o = u8_scalar(x);
            }
        }
    }
}

/// Generic `rint(x * scale)` as `i32` lanes, NaN as 0, capped at `scale`; the caller's
/// saturating narrowing saturates.
#[inline(always)]
fn quantize<S: SimdToken>(x: S::f32s, scale: f32) -> S::i32s {
    let v = x * scale;
    let ordered: S::i32s = v.simd_eq(v).to_vector();
    let r: S::i32s = v.min(scale).round_ties_even().to_int();
    r & ordered
}

/// Generic `CU8` value minus 128 as `i32` lanes (see the module documentation), NaN as 0.
#[inline(always)]
fn quantize_u8<S: SimdToken>(simd: S, x: S::f32s) -> S::i32s {
    let y = x * S8 + TINY;
    let ordered: S::i32s = y.simd_eq(y).to_vector();
    let c = y.min(S8);
    let g = c.floor();
    let half: S::i32s = S::f32s::splat(simd, 0.5).bitcast();
    let tie: S::i32s = g.simd_eq(c).to_vector();
    let d = g - (tie & half).bitcast::<S::f32s>();
    let r: S::i32s = d.round_ties_even().to_int();
    r & ordered
}

/// The generic vector code over whole vectors; returns the number of floats done.
#[inline(always)]
fn generic<S: SimdToken>(simd: S, kind: Kind, input: &[f32], out: &mut [u8]) -> usize {
    let h = S::f32s::LEN;
    let load = |s: &[f32]| S::f32s::from_slice(simd, s);
    match kind {
        Kind::I16 => blocks(input, out, 2 * h, 4 * h, |i, o| {
            let (a, b) = i.split_at(h);
            let q: S::i16s =
                quantize::<S>(load(a), S16).saturating_narrow(quantize::<S>(load(b), S16));
            q.bitcast::<S::u8s>().store_slice(o);
        }),
        Kind::I8 | Kind::U8 => {
            let sign = S::u8s::splat(simd, 0x80);
            let q = |s: &[f32]| {
                if kind == Kind::U8 {
                    quantize_u8(simd, load(s))
                } else {
                    quantize::<S>(load(s), S8)
                }
            };
            blocks(input, out, 4 * h, 4 * h, |i, o| {
                let (ab, cd) = i.split_at(2 * h);
                let (a, b) = ab.split_at(h);
                let (c, d) = cd.split_at(h);
                let lo: S::i16s = q(a).saturating_narrow(q(b));
                let hi: S::i16s = q(c).saturating_narrow(q(d));
                let v = lo.saturating_narrow(hi).bitcast::<S::u8s>();
                if kind == Kind::U8 { v ^ sign } else { v }.store_slice(o);
            })
        }
    }
}

/// The path of the current level over whole vectors, if it has one; returns the number of
/// floats done.
#[inline(always)]
fn level_path<S: SimdToken>(simd: S, kind: Kind, input: &[f32], out: &mut [u8]) -> Option<usize> {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx512() {
            return Some(x86::avx512(t, kind, input, out));
        }
        if let Some(t) = level.as_avx2() {
            return Some(x86::avx2(t, kind, input, out));
        }
        if let Some(t) = level.as_sse4_2() {
            return Some(x86::sse4_2(t, kind, input, out));
        }
        if let Some(t) = level.as_sse2() {
            return Some(x86::sse2(t, kind, input, out));
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return Some(arm::neon(t, kind, input, out));
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return Some(wasm::simd128(t, kind, input, out));
    }
    let _ = (level, kind, input, out);
    None
}

/// `C32` (as interleaved `f32`) to `CS16`, `CS8` or `CU8`.
#[inline(always)]
fn f32_out<S: SimdToken>(simd: S, kind: Kind, input: &[f32], out: &mut [u8]) {
    let done = match level_path(simd, kind, input, out) {
        Some(done) => done,
        None => generic(simd, kind, input, out),
    };
    let input = input.get(done..).unwrap_or_default();
    let out = out.get_mut(done * kind.width()..).unwrap_or_default();
    tail(kind, input, out);
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod x86 {
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64 as arch;

    use arch::{
        __m128, __m128i, __m256, __m256i, __m512, __m512i, _CMP_EQ_OQ, _CMP_ORD_Q,
        _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEG_INF, _mm_add_ps, _mm_and_ps, _mm_castps_si128,
        _mm_cmpord_ps, _mm_cvtps_epi32, _mm_load_ps, _mm_min_epi8, _mm_min_epu8, _mm_min_ps,
        _mm_mul_ps, _mm_or_si128, _mm_packs_epi16, _mm_packs_epi32, _mm_set1_epi8, _mm_set1_ps,
        _mm_sub_ps, _mm_xor_si128, _mm256_and_ps, _mm256_castps_si256, _mm256_cmp_ps,
        _mm256_cvtps_epi32, _mm256_fmadd_ps, _mm256_min_epi8, _mm256_min_epu8, _mm256_min_ps,
        _mm256_mul_ps, _mm256_or_si256, _mm256_packs_epi16, _mm256_packs_epi32,
        _mm256_permute4x64_epi64, _mm256_permutevar8x32_epi32, _mm256_set1_epi8, _mm256_set1_ps,
        _mm256_setr_epi32, _mm256_sub_ps, _mm256_xor_si256, _mm512_cmp_ps_mask,
        _mm512_cvtsepi32_epi8, _mm512_cvtsepi32_epi16, _mm512_fmadd_ps,
        _mm512_mask_cvtsepi32_storeu_epi8, _mm512_mask_cvtsepi32_storeu_epi16, _mm512_mask_sub_ps,
        _mm512_maskz_cvtps_epi32, _mm512_maskz_loadu_ps, _mm512_min_ps, _mm512_mul_ps,
        _mm512_roundscale_ps, _mm512_set1_ps,
    };
    use core::hint::black_box;
    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4, f32x8, f32x16, u8x16, u8x32};

    use super::{C, Kind, S8, S16, TINY, quarters, run, tail};

    fearless_simd::kernel!(
        /// AVX-512 over all values (the last partial vector with masked loads and stores);
        /// returns the number of floats done. `vcvtps2dq` rounds to even, its zero-masking
        /// clears NaN, `vpmovsdw`/`vpmovsdb` saturate.
        #[inline(always)]
        pub(super) fn avx512(t: Avx512, kind: Kind, input: &[f32], out: &mut [u8]) -> usize {
            let s16 = _mm512_set1_ps(S16);
            let s8 = _mm512_set1_ps(S8);
            let tiny = _mm512_set1_ps(TINY);
            let half = _mm512_set1_ps(0.5);
            let sign = _mm_set1_epi8(i8::MIN);
            // `rint(v)`, NaN as 0 (mask `k`), capped at `s`.
            let signed = |v: __m512, s: __m512| -> __m512i {
                let k = _mm512_cmp_ps_mask::<_CMP_ORD_Q>(v, v);
                _mm512_maskz_cvtps_epi32(k, _mm512_min_ps(v, s))
            };
            // The `CU8` value minus 128 of `x` (see the module documentation).
            let offset = |x: __m512| -> __m512i {
                let y = _mm512_fmadd_ps(x, s8, tiny);
                let k = _mm512_cmp_ps_mask::<_CMP_ORD_Q>(y, y);
                let c = _mm512_min_ps(y, s8);
                let g = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC }>(c);
                let tie = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(g, c);
                _mm512_maskz_cvtps_epi32(k, _mm512_mask_sub_ps(g, tie, g, half))
            };
            let load = |x: &[f32]| -> __m512 { f32x16::from_slice(t, x).into() };
            let i8_bytes = |x: __m512, u8_out: bool| -> __m128i {
                if u8_out {
                    _mm_xor_si128(_mm512_cvtsepi32_epi8(offset(x)), sign)
                } else {
                    _mm512_cvtsepi32_epi8(signed(_mm512_mul_ps(x, s8), s8))
                }
            };
            let done = match kind {
                Kind::I16 => run(input, out, 16, 32, 8, |i, o| {
                    let q = _mm512_cvtsepi32_epi16(signed(_mm512_mul_ps(load(i), s16), s16));
                    let q: u8x32<_> = q.simd_into(t);
                    q.store_slice(o);
                }),
                Kind::I8 | Kind::U8 => run(input, out, 16, 16, 4, |i, o| {
                    let q: u8x16<_> = i8_bytes(load(i), kind == Kind::U8).simd_into(t);
                    q.store_slice(o);
                }),
            };
            let rest = input.get(done..).unwrap_or_default();
            let n = rest.len().min(15);
            let Some(o) = out.get_mut(done * kind.width()..) else {
                return done;
            };
            if n == 0 || o.len() < n * kind.width() {
                return done;
            }
            let k = (1u16 << n) - 1;
            // SAFETY: the lanes of the mask `k` are the first `n` floats of `rest`, which are in
            // bounds; masked-off lanes are neither read nor faulted on.
            let x = unsafe { _mm512_maskz_loadu_ps(k, rest.as_ptr()) };
            match kind {
                Kind::I16 => {
                    let q = signed(_mm512_mul_ps(x, s16), s16);
                    // SAFETY: `o` holds at least `2 n` bytes, the `n` masked `i16` lanes; the
                    // store is unaligned and writes nothing for masked-off lanes.
                    unsafe { _mm512_mask_cvtsepi32_storeu_epi16(o.as_mut_ptr().cast(), k, q) };
                }
                Kind::I8 => {
                    let q = signed(_mm512_mul_ps(x, s8), s8);
                    // SAFETY: `o` holds at least `n` bytes, the `n` masked `i8` lanes; the store
                    // is unaligned and writes nothing for masked-off lanes.
                    unsafe { _mm512_mask_cvtsepi32_storeu_epi8(o.as_mut_ptr().cast(), k, q) };
                }
                Kind::U8 => {
                    let q: [u8; 16] = u8x16::simd_from(t, i8_bytes(x, true)).into();
                    if let (Some(d), Some(s)) = (o.get_mut(..n), q.get(..n)) {
                        d.copy_from_slice(s);
                    }
                }
            }
            done + n
        }
    );

    fearless_simd::kernel!(
        /// AVX2 over whole vectors; returns the number of floats done. `vcvtps2dq` rounds to
        /// even; NaN is cleared and the upper side of `CS16` and `CS8` capped before it (it
        /// gives `i32::MIN` for both), the packs saturate, the byte bound of the module
        /// documentation finishes `CU8` and `vpermq`/`vpermd` undo the in-lane order of the
        /// packs.
        #[inline(always)]
        pub(super) fn avx2(t: Avx2, kind: Kind, input: &[f32], out: &mut [u8]) -> usize {
            let s16 = _mm256_set1_ps(S16);
            let s8 = _mm256_set1_ps(S8);
            let tiny = _mm256_set1_ps(TINY);
            let half_down = _mm256_set1_ps(C);
            let low = _mm256_set1_epi8(i8::MAX);
            let sign = _mm256_set1_epi8(i8::MIN);
            let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            let load = |x: &[f32]| -> __m256 { f32x8::from_slice(t, x).into() };
            let signed = |x: &[f32], s: __m256| -> __m256i {
                let v = _mm256_mul_ps(load(x), s);
                let v = _mm256_and_ps(v, _mm256_cmp_ps::<_CMP_ORD_Q>(v, v));
                _mm256_cvtps_epi32(_mm256_min_ps(v, s))
            };
            // `w = x * 128 + 2^-47` with NaN as 0, and `rint(w - C)`.
            let offset = |x: &[f32]| -> __m256 {
                let w = _mm256_fmadd_ps(load(x), s8, tiny);
                _mm256_and_ps(w, _mm256_cmp_ps::<_CMP_ORD_Q>(w, w))
            };
            let rounded = |w: __m256| _mm256_cvtps_epi32(_mm256_sub_ps(w, half_down));
            let bits = _mm256_castps_si256;
            match kind {
                Kind::I16 => run(input, out, 16, 32, 8, |i, o| {
                    let (a, b) = i.split_at(8);
                    let q = _mm256_packs_epi32(signed(a, s16), signed(b, s16));
                    let q: u8x32<_> = _mm256_permute4x64_epi64::<0xd8>(q).simd_into(t);
                    q.store_slice(o);
                }),
                Kind::I8 | Kind::U8 => run(input, out, 32, 32, 4, |i, o| {
                    let (ab, cd) = i.split_at(16);
                    let (a, b) = ab.split_at(8);
                    let (c, d) = cd.split_at(8);
                    let q = if kind == Kind::U8 {
                        let [a, b, c, d] = [a, b, c, d].map(offset);
                        let lo = _mm256_packs_epi32(rounded(a), rounded(b));
                        let hi = _mm256_packs_epi32(rounded(c), rounded(d));
                        let q = _mm256_packs_epi16(lo, hi);
                        let lo = _mm256_packs_epi32(bits(a), bits(b));
                        let hi = _mm256_packs_epi32(bits(c), bits(d));
                        let limit = _mm256_or_si256(_mm256_packs_epi16(lo, hi), low);
                        let q = _mm256_min_epi8(_mm256_min_epu8(q, limit), limit);
                        _mm256_xor_si256(q, sign)
                    } else {
                        let lo = _mm256_packs_epi32(signed(a, s8), signed(b, s8));
                        let hi = _mm256_packs_epi32(signed(c, s8), signed(d, s8));
                        _mm256_packs_epi16(lo, hi)
                    };
                    let q: u8x32<_> = _mm256_permutevar8x32_epi32(q, order).simd_into(t);
                    q.store_slice(o);
                }),
            }
        }
    );

    /// The 128-bit path, for a token of SSE2 or of SSE4.2, over whole vectors; returns the
    /// number of floats done. `$bound` applies the signed minimum with the byte bound `b` of the
    /// module documentation to the `CU8` bytes (the value minus 128, saturated) and flips their
    /// sign bits: SSE4.1 has `pminsb`, SSE2 takes the unsigned minimum of the flipped
    /// bytes.
    ///
    /// The first values up to a 16-byte boundary of the input go through the scalar formulas.
    /// NaN is cleared from `x` with `cmpordps` and `andps`, whose second read of `x` is an
    /// aligned load from an address the compiler cannot prove equal to the first (offset by an
    /// opaque zero): it folds into `andps` instead of costing a register copy, as `x` is needed
    /// twice. `cvtps2dq` rounds to even, the upper side of `CS16` and `CS8` is capped before it
    /// (it gives `i32::MIN` there) and the packs saturate.
    macro_rules! sse {
        ($name:ident, $token:ident, $bound:expr) => {
            fearless_simd::kernel!(
                #[inline(always)]
                pub(super) fn $name(t: $token, kind: Kind, input: &[f32], out: &mut [u8]) -> usize {
                    let width = kind.width();
                    let head = (input.as_ptr().addr().wrapping_neg() % 16 / 4)
                        .min(input.len())
                        .min(out.len() / width);
                    let (first, input) = input.split_at(head);
                    let (first_out, out) = out.split_at_mut(head * width);
                    tail(kind, first, first_out);
                    let zero = black_box(0);
                    let s16 = _mm_set1_ps(S16);
                    let s8 = _mm_set1_ps(S8);
                    let tiny = _mm_set1_ps(TINY);
                    let half_down = _mm_set1_ps(C);
                    let low = _mm_set1_epi8(i8::MAX);
                    // `x` of 4 floats at a 16-byte boundary with NaN as 0.
                    let clean = |x: &[f32]| -> __m128 {
                        let v: __m128 = f32x4::from_slice(t, x).into();
                        // SAFETY: `zero` is 0 (`black_box` returns its argument), so this reads
                        // the 4 floats of `x` (every block of `run` has a multiple of 4 floats);
                        // they start at a 16-byte boundary, as the input after `head` does.
                        let again = unsafe { _mm_load_ps(x.as_ptr().wrapping_add(zero)) };
                        _mm_and_ps(_mm_cmpord_ps(v, v), again)
                    };
                    let signed = |x: &[f32], s: __m128| -> __m128i {
                        _mm_cvtps_epi32(_mm_min_ps(_mm_mul_ps(clean(x), s), s))
                    };
                    // `w = x * 128 + 2^-47` with NaN as 0, and `rint(w - C)`.
                    let offset = |x: &[f32]| _mm_add_ps(_mm_mul_ps(clean(x), s8), tiny);
                    let rounded = |w: __m128| _mm_cvtps_epi32(_mm_sub_ps(w, half_down));
                    let bits = _mm_castps_si128;
                    let store = |o: &mut [u8], q: __m128i| {
                        let q: u8x16<_> = q.simd_into(t);
                        q.store_slice(o);
                    };
                    head + match kind {
                        Kind::I16 => run(input, out, 8, 16, 8, |i, o| {
                            let (a, b) = i.split_at(4);
                            store(o, _mm_packs_epi32(signed(a, s16), signed(b, s16)));
                        }),
                        Kind::I8 => run(input, out, 16, 16, 8, |i, o| {
                            let [a, b, c, d] = quarters(i);
                            let lo = _mm_packs_epi32(signed(a, s8), signed(b, s8));
                            let hi = _mm_packs_epi32(signed(c, s8), signed(d, s8));
                            store(o, _mm_packs_epi16(lo, hi));
                        }),
                        Kind::U8 => run(input, out, 16, 16, 4, |i, o| {
                            let [a, b, c, d] = quarters(i).map(offset);
                            let lo = _mm_packs_epi32(rounded(a), rounded(b));
                            let hi = _mm_packs_epi32(rounded(c), rounded(d));
                            let q = _mm_packs_epi16(lo, hi);
                            let lo = _mm_packs_epi32(bits(a), bits(b));
                            let hi = _mm_packs_epi32(bits(c), bits(d));
                            let limit = _mm_or_si128(_mm_packs_epi16(lo, hi), low);
                            store(o, $bound(_mm_min_epu8(q, limit), limit));
                        }),
                    }
                }
            );
        };
    }
    sse!(sse2, Sse2, |q: __m128i, b: __m128i| {
        let sign = _mm_set1_epi8(i8::MIN);
        _mm_min_epu8(_mm_xor_si128(q, sign), _mm_xor_si128(b, sign))
    });
    sse!(sse4_2, Sse4_2, |q: __m128i, b: __m128i| {
        let sign = _mm_set1_epi8(i8::MIN);
        _mm_xor_si128(_mm_min_epi8(q, b), sign)
    });
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use core::arch::aarch64::{
        float32x4_t, int8x16_t, int32x4_t, uint8x16_t, uint32x4_t, vcltq_f32, vcvtnq_s32_f32,
        vdupq_n_f32, vdupq_n_s8, vdupq_n_u8, veorq_u8, vminq_s8, vmulq_n_f32, vorrq_s8,
        vqmovn_high_s16, vqmovn_high_s32, vqmovn_s16, vqmovn_s32, vreinterpretq_s8_u8,
        vreinterpretq_u8_s8, vreinterpretq_u8_s16, vreinterpretq_u8_u16, vreinterpretq_u16_u32,
        vsubq_f32, vuzp1q_u8, vuzp1q_u16,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4, u8x16};

    use super::{C, Kind, S8, S16, TINY, quarters, run};

    fearless_simd::kernel!(
        /// NEON over whole vectors; returns the number of floats done. `fcvtns` rounds to even,
        /// saturates and gives 0 for NaN, `sqxtn` saturates; `CU8` is bounded by -1 per byte
        /// where `x * 128 < -2^-47` (see the module documentation).
        #[inline(always)]
        pub(super) fn neon(t: Neon, kind: Kind, input: &[f32], out: &mut [u8]) -> usize {
            let below = vdupq_n_f32(-TINY);
            let half_down = vdupq_n_f32(C);
            let low = vdupq_n_s8(i8::MAX);
            let sign = vdupq_n_u8(0x80);
            let load = |x: &[f32]| -> float32x4_t { f32x4::from_slice(t, x).into() };
            let signed =
                |x: &[f32], s: f32| -> int32x4_t { vcvtnq_s32_f32(vmulq_n_f32(load(x), s)) };
            let store = |o: &mut [u8], q: uint8x16_t| {
                let q: u8x16<_> = q.simd_into(t);
                q.store_slice(o);
            };
            // 4 vectors to 16 saturated bytes.
            let bytes = |[a, b, c, d]: [int32x4_t; 4]| -> int8x16_t {
                let lo = vqmovn_high_s32(vqmovn_s32(a), b);
                let hi = vqmovn_high_s32(vqmovn_s32(c), d);
                vqmovn_high_s16(vqmovn_s16(lo), hi)
            };
            // 4 lane masks to 16 byte masks.
            let masks = |[a, b, c, d]: [uint32x4_t; 4]| -> int8x16_t {
                let lo = vuzp1q_u16(vreinterpretq_u16_u32(a), vreinterpretq_u16_u32(b));
                let hi = vuzp1q_u16(vreinterpretq_u16_u32(c), vreinterpretq_u16_u32(d));
                vreinterpretq_s8_u8(vuzp1q_u8(
                    vreinterpretq_u8_u16(lo),
                    vreinterpretq_u8_u16(hi),
                ))
            };
            match kind {
                Kind::I16 => run(input, out, 8, 16, 8, |i, o| {
                    let (a, b) = i.split_at(4);
                    let q = vqmovn_high_s32(vqmovn_s32(signed(a, S16)), signed(b, S16));
                    store(o, vreinterpretq_u8_s16(q));
                }),
                Kind::I8 => run(input, out, 16, 16, 8, |i, o| {
                    store(
                        o,
                        vreinterpretq_u8_s8(bytes(quarters(i).map(|x| signed(x, S8)))),
                    );
                }),
                // `rint(y - C)` with `y = x * 128`, bounded by -1 where `y < -2^-47` (the
                // compare is false for NaN, whose conversion gives 0).
                Kind::U8 => run(input, out, 16, 16, 4, |i, o| {
                    let y = quarters(i).map(|x| vmulq_n_f32(load(x), S8));
                    let q = bytes(y.map(|y| vcvtnq_s32_f32(vsubq_f32(y, half_down))));
                    let limit = vorrq_s8(masks(y.map(|y| vcltq_f32(y, below))), low);
                    store(o, veorq_u8(vreinterpretq_u8_s8(vminq_s8(q, limit)), sign));
                }),
            }
        }
    );
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::{
        f32x4_add, f32x4_eq, f32x4_floor, f32x4_mul, f32x4_nearest, f32x4_splat, f32x4_sub,
        i8x16_narrow_i16x8, i16x8_narrow_i32x4, i32x4_trunc_sat_f32x4, u8x16_splat, v128, v128_and,
        v128_xor,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4, u8x16};

    use super::{Kind, S8, S16, TINY, run};

    fearless_simd::kernel!(
        /// SIMD128 over whole vectors; returns the number of floats done. `f32x4.nearest`
        /// rounds to even, `i32x4.trunc_sat_f32x4_s` saturates and gives 0 for NaN, the
        /// narrowings saturate.
        #[inline(always)]
        pub(super) fn simd128(t: WasmSimd128, kind: Kind, input: &[f32], out: &mut [u8]) -> usize {
            let s16 = f32x4_splat(S16);
            let s8 = f32x4_splat(S8);
            let tiny = f32x4_splat(TINY);
            let half = f32x4_splat(0.5);
            let sign = u8x16_splat(0x80);
            let load = |x: &[f32]| -> v128 { f32x4::from_slice(t, x).into() };
            let signed = |x: &[f32], s: v128| -> v128 {
                i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_mul(load(x), s)))
            };
            // NaN stays NaN up to the conversion, which gives 0; no cap is needed.
            let offset = |x: &[f32]| -> v128 {
                let y = f32x4_add(f32x4_mul(load(x), s8), tiny);
                let g = f32x4_floor(y);
                let d = f32x4_sub(g, v128_and(f32x4_eq(g, y), half));
                i32x4_trunc_sat_f32x4(f32x4_nearest(d))
            };
            let store = |o: &mut [u8], q: v128| {
                let q: u8x16<_> = q.simd_into(t);
                q.store_slice(o);
            };
            // 16 floats to 16 saturated bytes; a function of its own for each quantiser, so
            // that the choice of the format is not made per vector.
            #[inline(always)]
            fn bytes(i: &[f32], q: impl Fn(&[f32]) -> v128) -> v128 {
                let (ab, cd) = i.split_at(8);
                let (a, b) = ab.split_at(4);
                let (c, d) = cd.split_at(4);
                let lo = i16x8_narrow_i32x4(q(a), q(b));
                let hi = i16x8_narrow_i32x4(q(c), q(d));
                i8x16_narrow_i16x8(lo, hi)
            }
            match kind {
                Kind::I16 => run(input, out, 8, 16, 8, |i, o| {
                    let (a, b) = i.split_at(4);
                    store(o, i16x8_narrow_i32x4(signed(a, s16), signed(b, s16)));
                }),
                Kind::I8 => run(input, out, 16, 16, 4, |i, o| {
                    store(o, bytes(i, |x| signed(x, s8)));
                }),
                Kind::U8 => run(input, out, 16, 16, 4, |i, o| {
                    store(o, v128_xor(bytes(i, offset), sign));
                }),
            }
        }
    );
}

/// `q / 256` of one vector of Q15 values rounded to the nearest integer, ties to even:
/// `(q + 127 + ((q >> 8) & 1)) >> 8` with a saturating add (only values that round to 128 or
/// more saturate, and they saturate to 127 anyway).
#[inline(always)]
fn q15_i8_half<S: SimdToken>(simd: S, s: &[i16]) -> S::i16s {
    let q = S::i16s::from_slice(simd, s);
    q.saturating_add(((q >> 8u32) & 1i16) + 127i16) >> 8u32
}

/// Q15 to `CS8`: `q / 256` rounded to even, saturated.
#[inline(always)]
fn q15_to_i8<S: SimdToken>(simd: S, input: &[i16], out: &mut [u8]) {
    let n = S::i16s::LEN;
    let mut ic = input.chunks_exact(2 * n);
    let mut oc = out.chunks_exact_mut(2 * n);
    for (i, o) in (&mut ic).zip(&mut oc) {
        let (a, b) = i.split_at(n);
        let v: S::i8s = q15_i8_half::<S>(simd, a).saturating_narrow(q15_i8_half::<S>(simd, b));
        v.bitcast::<S::u8s>().store_slice(o);
    }
    for (&q, o) in ic.remainder().iter().zip(oc.into_remainder()) {
        *o = (q.saturating_add(((q >> 8) & 1) + 127) >> 8) as u8;
    }
}

/// `(q + 32640) / 256` of one vector of Q15 values rounded to the nearest integer, ties to
/// even. With the offset binary value `u = q + 32768 = 256 a + r` that is `a`, except for a tie
/// (`r == 0`) with an odd `a`, which gives `a - 1`: `(u >> 8) - ((u & 0x1FF) == 0x100)`.
#[inline(always)]
fn q15_u8_half<S: SimdToken>(simd: S, s: &[i16]) -> S::u16s {
    let u: S::u16s = (S::i16s::from_slice(simd, s) ^ i16::MIN).bitcast();
    let tie: S::u16s = (u & 0x1FFu16).simd_eq(0x100u16).to_vector().bitcast();
    (u >> 8u32) + tie
}

/// Q15 to `CU8`: `(q + 32640) / 256` rounded to even.
#[inline(always)]
fn q15_to_u8<S: SimdToken>(simd: S, input: &[i16], out: &mut [u8]) {
    let n = S::i16s::LEN;
    let mut ic = input.chunks_exact(2 * n);
    let mut oc = out.chunks_exact_mut(2 * n);
    for (i, o) in (&mut ic).zip(&mut oc) {
        let (a, b) = i.split_at(n);
        let v: S::u8s = q15_u8_half::<S>(simd, a).narrow(q15_u8_half::<S>(simd, b));
        v.store_slice(o);
    }
    for (&q, o) in ic.remainder().iter().zip(oc.into_remainder()) {
        let u = (q as u16) ^ 0x8000;
        *o = ((u >> 8) - u16::from(u & 0x1FF == 0x100)) as u8;
    }
}

impl ConvertOut<C32> for Simd {
    fn convert_out(
        &self,
        format: &SampleFormat,
        input: &[C32],
        output: &mut [u8],
    ) -> Result<usize> {
        let Some(kind) = fast_kind(format) else {
            return Fallback.convert_out(format, input, output);
        };
        let Some(bytes) = fast_bytes(kind, input.len().saturating_mul(2), output.len()) else {
            return Fallback.convert_out(format, input, output);
        };
        let Some(out) = output.get_mut(..bytes) else {
            return Fallback.convert_out(format, input, output);
        };
        let input: &[f32] = bytemuck::cast_slice(input);
        dispatch!(simd => f32_out(simd, kind, input, out));
        Ok(bytes)
    }
}

impl ConvertOut<CQ15> for Simd {
    fn convert_out(
        &self,
        format: &SampleFormat,
        input: &[CQ15],
        output: &mut [u8],
    ) -> Result<usize> {
        let Some(kind) = fast_kind(format) else {
            return Fallback.convert_out(format, input, output);
        };
        let Some(bytes) = fast_bytes(kind, input.len().saturating_mul(2), output.len()) else {
            return Fallback.convert_out(format, input, output);
        };
        let Some(out) = output.get_mut(..bytes) else {
            return Fallback.convert_out(format, input, output);
        };
        let bytes_in: &[u8] = bytemuck::cast_slice(input);
        let input: &[i16] = bytemuck::cast_slice(input);
        match kind {
            // Little-endian `cs16` is byte-identical to `CQ15`.
            Kind::I16 => out.copy_from_slice(bytes_in),
            Kind::I8 => dispatch!(simd => q15_to_i8(simd, input, out)),
            Kind::U8 => dispatch!(simd => q15_to_u8(simd, input, out)),
        }
        Ok(bytes)
    }
}
