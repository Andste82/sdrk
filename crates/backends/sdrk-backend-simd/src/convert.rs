//! `convert`: raw device bytes to `C32` / `CQ15`.
//!
//! The common interleaved formats (`U8`, `I8` and little-endian `I16` with the full container
//! as valid bits, power-of-two full scale, no I/Q swap) run vectorised; every other format and
//! every invalid call is handed to the fallback, which also produces the error values. All
//! paths are bit-exact with the fallback: the scale factors are powers of two and every result
//! (`x * 2^-15`, `x * 2^-7`, `(u - 127.5) * 2^-7`) is representable in `f32`, so any sequence of
//! operations whose intermediate values are exact gives the bits of the specification. No path
//! produces `-0.0` (`x = 0` gives `+0.0`, and `U8` has no zero).
//!
//! To `C32`, each level has its own path (`core::arch`, decision D1) over whole vectors,
//! unrolled explicitly so that the run-time dispatched build (an out-of-line call) runs the
//! same loop as the static one:
//!
//! - **SSE4.2, AVX2, AVX-512:** `pmovsx`/`pmovzx` with the load folded in, the conversion and
//!   the scale (`I8`, `I16`); `U8` as `u * 2^-7 - 255/256` with one FMA where there is FMA.
//!   AVX-512 converts the last partial vector with masked loads and stores.
//! - **SSE2** (no `pmovsx`), and `U8` at SSE4.2: the float bits are built directly. A value
//!   `u` of 8 or 16 bits placed below a constant exponent half-word is the float `c + u * 2^-k`,
//!   and one subtraction of a constant in the same binade gives the result exactly (Sterbenz).
//! - **NEON:** the fixed-point conversion `scvtf #n`, which is exactly `x * 2^-n`; `I8` places
//!   each byte at the top of a 32-bit lane with one table lookup (`x << 24`, then `#31`), `U8`
//!   builds the float bits with a two-register table lookup.
//! - **SIMD128:** `I8` and `U8` build the float bits with one shuffle per vector, `I16` widens
//!   and converts; the blocks are written out in a function of their own, where the addresses
//!   fold into the offsets of the loads and stores.
//!
//! The generic `fearless_simd` code runs on a level without a path, and the values after the
//! last whole vector go through the scalar formulas.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::convert::Convert;
use sdrk_core::{C32, CQ15, Fallback, Result};
use sdrk_device::{Align, Container, Endian, FullScale, Layout, SampleFormat, formats};

use crate::Simd;
use crate::level::{dispatch, is_sse2_baseline};
use crate::util;

/// The formats with a vector path.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    U8,
    I8,
    I16,
}

impl Kind {
    /// Bytes per component.
    pub(crate) const fn width(self) -> usize {
        match self {
            Kind::U8 | Kind::I8 => 1,
            Kind::I16 => 2,
        }
    }
}

/// The vector-path kind of a format, if it has one.
///
/// The three well-known formats are recognised by one comparison each before the field-by-field
/// test, which also accepts the 8-bit formats with an `Endian::Big` tag (meaningless for one
/// byte).
#[inline(always)]
pub(crate) fn fast_kind(format: &SampleFormat) -> Option<Kind> {
    if *format == formats::CS16 {
        return Some(Kind::I16);
    }
    if *format == formats::CS8 {
        return Some(Kind::I8);
    }
    if *format == formats::CU8 {
        return Some(Kind::U8);
    }
    if format.layout() != Layout::Interleaved
        || format.swap_iq()
        || format.full_scale() != FullScale::PowerOfTwo
        || format.align() != Align::Lsb
    {
        return None;
    }
    match (format.container(), format.bits(), format.signed()) {
        (Container::U8, 8, false) => Some(Kind::U8),
        (Container::I8, 8, true) => Some(Kind::I8),
        (Container::I16, 16, true) if format.endian() == Endian::Little => Some(Kind::I16),
        _ => None,
    }
}

/// The number of samples if `input_len` bytes and `output_len` samples are valid for the fast
/// path (whole samples and enough room), else `None` so that the fallback reports the error.
fn fast_samples(kind: Kind, input_len: usize, output_len: usize) -> Option<usize> {
    let per_sample = 2 * kind.width();
    if !input_len.is_multiple_of(per_sample) {
        return None;
    }
    let samples = input_len / per_sample;
    (output_len >= samples).then_some(samples)
}

/// Reciprocal of the full scale of a power-of-two format: `2^-7` or `2^-15`.
const INV_128: f32 = 1.0 / 128.0;
const INV_32768: f32 = 1.0 / 32768.0;
/// `127.5 * 2^-7`: `CU8` is `u * 2^-7 - U8_OFFSET`, both steps exact.
const U8_OFFSET: f32 = 255.0 / 256.0;

// The float bits built by the SSE2, NEON (`U8` only) and SIMD128 paths: a value `u` below the
// upper half-word `0x4380` or `0x4780` of a 32-bit lane.

/// The bits `0x4380_0000 | u << 8` (`u` an unsigned byte) are the float `256 + u * 2^-7`;
/// subtracting this constant gives the `CU8` value `(u - 127.5) * 2^-7`.
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
const U8_MAGIC: f32 = 256.0 + U8_OFFSET;
/// The upper half-word of the `CU8` and `CS16` magic floats.
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
const HIGH_U8_I16: u16 = 0x4380;
/// The bits `0x4780_0000 | u` (`u = x + 128` for a signed byte `x`) are the float
/// `65536 + u * 2^-7 = 65537 + x * 2^-7`; subtracting this constant gives the `CS8` value.
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
const I8_MAGIC: f32 = 65537.0;
/// The upper half-word of the `CS8` magic float.
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
const HIGH_I8: u16 = 0x4780;
/// The bits `0x4380_0000 | u` (`u = x + 2^15` for a signed 16-bit `x`) are the float
/// `256 + u * 2^-15 = 257 + x * 2^-15`; subtracting this constant gives the `CS16` value.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const I16_MAGIC: f32 = 257.0;

/// Runs `unit` (`IN` input bytes to `OUT` values) over all whole units that fit both slices,
/// `U` units per iteration of the main loop, then four and then one per iteration for the rest;
/// returns the number of values done.
///
/// The zipped slice iterators of whole blocks give one counted loop (zipping `&mut ChunksExact`
/// would test both ends in every iteration), and the inner loops over the units of a block are
/// unrolled completely, also in a kernel that is not inlined into its caller.
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn run<const IN: usize, const OUT: usize, const U: usize>(
    bytes: &[u8],
    out: &mut [f32],
    mut unit: impl FnMut(&[u8; IN], &mut [f32; OUT]),
) -> usize {
    let (ib, _) = bytes.as_chunks::<IN>();
    let (ob, _) = out.as_chunks_mut::<OUT>();
    let n = ib.len().min(ob.len());
    let (ib, _) = ib.split_at(n);
    let (ob, _) = ob.split_at_mut(n);
    let (ib, ir) = ib.as_chunks::<U>();
    let (ob, or) = ob.as_chunks_mut::<U>();
    for (b, o) in ib.iter().zip(ob) {
        for (b, o) in b.iter().zip(o) {
            unit(b, o);
        }
    }
    if ir.is_empty() {
        return n * OUT;
    }
    let (ib, ir) = ir.as_chunks::<4>();
    let (ob, or) = or.as_chunks_mut::<4>();
    for (b, o) in ib.iter().zip(ob) {
        for (b, o) in b.iter().zip(o) {
            unit(b, o);
        }
    }
    for (b, o) in ir.iter().zip(or) {
        unit(b, o);
    }
    n * OUT
}

/// The scalar formulas for the values after the last whole vector.
#[inline(always)]
fn tail(kind: Kind, bytes: &[u8], out: &mut [f32]) {
    match kind {
        Kind::I16 => {
            for (o, b) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
                *o = f32::from(i16::from_le_bytes(*b)) * INV_32768;
            }
        }
        Kind::I8 => {
            for (o, &b) in out.iter_mut().zip(bytes) {
                *o = f32::from(b as i8) * INV_128;
            }
        }
        Kind::U8 => {
            for (o, &b) in out.iter_mut().zip(bytes) {
                *o = f32::from(b) * INV_128 - U8_OFFSET;
            }
        }
    }
}

/// Runs `body` on every whole block of `step` bytes and `step / width` values; returns the
/// number of values done.
#[inline(always)]
fn blocks(
    bytes: &[u8],
    out: &mut [f32],
    step: usize,
    width: usize,
    body: impl FnMut(&[u8], &mut [f32]),
) -> usize {
    let values = step / width;
    util::blocks(bytes, out, step, values, body) * values
}

/// The generic vector code over whole vectors; returns the number of values done.
#[inline(always)]
fn generic<S: SimdToken>(simd: S, kind: Kind, bytes: &[u8], out: &mut [f32]) -> usize {
    let h = S::f32s::LEN;
    let n = S::u8s::LEN;
    let load = |b: &[u8]| S::u8s::from_slice(simd, b);
    match kind {
        Kind::I16 => {
            let scale = S::f32s::splat(simd, INV_32768);
            blocks(bytes, out, n, 2, |b, o| {
                let v: S::i16s = load(b).bitcast();
                let (lo, hi) = v.widen();
                let (o_lo, o_hi) = o.split_at_mut(h);
                (lo.to_float::<S::f32s>() * scale).store_slice(o_lo);
                (hi.to_float::<S::f32s>() * scale).store_slice(o_hi);
            })
        }
        Kind::I8 | Kind::U8 => {
            let scale = S::f32s::splat(simd, INV_128);
            let offset = S::f32s::splat(simd, if kind == Kind::U8 { 127.5 } else { 0.0 });
            blocks(bytes, out, n, 1, |b, o| {
                let (a, c) = if kind == Kind::U8 {
                    let (a, c) = load(b).widen();
                    (a.bitcast::<S::i16s>(), c.bitcast::<S::i16s>())
                } else {
                    load(b).bitcast::<S::i8s>().widen()
                };
                let (a0, a1) = a.widen();
                let (c0, c1) = c.widen();
                let (o01, o23) = o.split_at_mut(2 * h);
                let (o0, o1) = o01.split_at_mut(h);
                let (o2, o3) = o23.split_at_mut(h);
                ((a0.to_float::<S::f32s>() - offset) * scale).store_slice(o0);
                ((a1.to_float::<S::f32s>() - offset) * scale).store_slice(o1);
                ((c0.to_float::<S::f32s>() - offset) * scale).store_slice(o2);
                ((c1.to_float::<S::f32s>() - offset) * scale).store_slice(o3);
            })
        }
    }
}

/// The path of the current level over whole vectors, if it has one; returns the number of
/// values done.
#[inline(always)]
fn level_path<S: SimdToken>(simd: S, kind: Kind, bytes: &[u8], out: &mut [f32]) -> Option<usize> {
    let level = simd.level();
    // SSE2 has no `pmovsx`, and for `U8` the float bits are as short as the conversion.
    let float_bits = kind == Kind::U8 || is_sse2_baseline(simd);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if let Some(t) = level.as_avx512() {
            return Some(x86::avx512(t, kind, bytes, out));
        }
        if let Some(t) = level.as_avx2() {
            return Some(x86::avx2(t, kind, bytes, out));
        }
        if float_bits && let Some(t) = level.as_sse2() {
            return Some(x86::sse2(t, kind, bytes, out));
        }
        if let Some(t) = level.as_sse4_2() {
            return Some(x86::sse4_2(t, kind == Kind::I16, bytes, out));
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return Some(arm::neon(t, kind, bytes, out));
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return Some(wasm::simd128(t, kind, bytes, out));
    }
    let _ = (level, float_bits, kind, bytes, out);
    None
}

/// `CU8`, `CS8` or `CS16` to `C32` (as interleaved `f32`).
#[inline(always)]
fn f32_in<S: SimdToken>(simd: S, kind: Kind, bytes: &[u8], out: &mut [f32]) {
    let done = match level_path(simd, kind, bytes, out) {
        Some(done) => done,
        None => generic(simd, kind, bytes, out),
    };
    // Usually nothing is left (the lengths of SDR buffers are multiples of the vectors).
    if done == out.len() {
        return;
    }
    let bytes = bytes.get(done * kind.width()..).unwrap_or_default();
    let out = out.get_mut(done..).unwrap_or_default();
    tail(kind, bytes, out);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86 as arch;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64 as arch;

    use arch::{
        __m128i, _mm_castsi128_ps, _mm_cvtepi8_epi32, _mm_cvtepi16_epi32, _mm_cvtepi32_ps,
        _mm_cvtsi32_si128, _mm_maskz_loadu_epi8, _mm_mul_ps, _mm_set_epi64x, _mm_set1_epi8,
        _mm_set1_epi16, _mm_set1_ps, _mm_setzero_si128, _mm_sub_ps, _mm_unpackhi_epi8,
        _mm_unpackhi_epi16, _mm_unpacklo_epi8, _mm_unpacklo_epi16, _mm_xor_si128,
        _mm256_cvtepi8_epi32, _mm256_cvtepi16_epi32, _mm256_cvtepi32_ps, _mm256_cvtepu8_epi32,
        _mm256_fmadd_ps, _mm256_maskz_loadu_epi16, _mm256_mul_ps, _mm256_set1_ps,
        _mm512_cvtepi8_epi32, _mm512_cvtepi16_epi32, _mm512_cvtepi32_ps, _mm512_cvtepu8_epi32,
        _mm512_fmadd_ps, _mm512_mask_storeu_ps, _mm512_mul_ps, _mm512_set1_ps,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4, f32x8, f32x16, u8x16, u8x32};

    use super::{
        HIGH_I8, HIGH_U8_I16, I8_MAGIC, I16_MAGIC, INV_128, INV_32768, Kind, U8_MAGIC, U8_OFFSET,
        run,
    };

    /// Output vectors per iteration of the main loops.
    const UNROLL: usize = 16;

    fearless_simd::kernel!(
        /// AVX-512 over all values (the last partial vector with masked loads and stores);
        /// returns the number of values done.
        #[inline(always)]
        pub(super) fn avx512(t: Avx512, kind: Kind, bytes: &[u8], out: &mut [f32]) -> usize {
            let store = |p: &mut [f32; 16], v| f32x16::simd_from(t, v).store_array(p);
            let s16 = _mm512_set1_ps(INV_32768);
            let s8 = _mm512_set1_ps(INV_128);
            let offset = _mm512_set1_ps(-U8_OFFSET);
            let i16_f32 = |v| _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(v)), s16);
            let i8_f32 = |v| _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v)), s8);
            let u8_f32 =
                |v| _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v)), s8, offset);
            let done = match kind {
                Kind::I16 => run::<_, _, UNROLL>(bytes, out, |q: &[u8; 32], p| {
                    store(p, i16_f32(u8x32::load_array_ref(t, q).into()));
                }),
                Kind::I8 => run::<_, _, UNROLL>(bytes, out, |q: &[u8; 16], p| {
                    store(p, i8_f32(u8x16::load_array_ref(t, q).into()));
                }),
                Kind::U8 => run::<_, _, UNROLL>(bytes, out, |q: &[u8; 16], p| {
                    store(p, u8_f32(u8x16::load_array_ref(t, q).into()));
                }),
            };
            if done == out.len() {
                return done;
            }
            let (Some(b), Some(o)) = (bytes.get(done * kind.width()..), out.get_mut(done..)) else {
                return done;
            };
            let n = o.len().min(b.len() / kind.width()).min(15);
            if n == 0 {
                return done;
            }
            let k = (1u16 << n) - 1;
            // The lanes of the mask `k` are the first `n` values of `b`, which are in bounds.
            let v = match kind {
                Kind::I16 => {
                    // SAFETY: the `n` masked `i16` lanes are in `b` (see above); masked-off
                    // lanes are neither read nor faulted on.
                    i16_f32(unsafe { _mm256_maskz_loadu_epi16(k, b.as_ptr().cast()) })
                }
                Kind::I8 => {
                    // SAFETY: the `n` masked bytes are in `b`; masked-off lanes are neither read
                    // nor faulted on.
                    i8_f32(unsafe { _mm_maskz_loadu_epi8(k, b.as_ptr().cast()) })
                }
                Kind::U8 => {
                    // SAFETY: as for `I8`.
                    u8_f32(unsafe { _mm_maskz_loadu_epi8(k, b.as_ptr().cast()) })
                }
            };
            // SAFETY: `o` holds at least `n` floats, the lanes of the mask `k`; the store is
            // unaligned and writes nothing for masked-off lanes.
            unsafe { _mm512_mask_storeu_ps(o.as_mut_ptr(), k, v) };
            done + n
        }
    );

    fearless_simd::kernel!(
        /// AVX2 over whole vectors; returns the number of values done.
        #[inline(always)]
        pub(super) fn avx2(t: Avx2, kind: Kind, bytes: &[u8], out: &mut [f32]) -> usize {
            let store = |p: &mut [f32; 8], v| f32x8::simd_from(t, v).store_array(p);
            // The 8 bytes as the low half of a vector (the load folds into `vpmovsx`/`vpmovzx`).
            let low64 = |q: &[u8; 8]| _mm_set_epi64x(0, i64::from_le_bytes(*q));
            match kind {
                Kind::I16 => {
                    let scale = _mm256_set1_ps(INV_32768);
                    run::<_, _, UNROLL>(bytes, out, |q: &[u8; 16], p| {
                        let v = _mm256_cvtepi16_epi32(u8x16::load_array_ref(t, q).into());
                        store(p, _mm256_mul_ps(_mm256_cvtepi32_ps(v), scale));
                    })
                }
                Kind::I8 => {
                    let scale = _mm256_set1_ps(INV_128);
                    run::<_, _, UNROLL>(bytes, out, |q: &[u8; 8], p| {
                        let v = _mm256_cvtepi8_epi32(low64(q));
                        store(p, _mm256_mul_ps(_mm256_cvtepi32_ps(v), scale));
                    })
                }
                Kind::U8 => {
                    let scale = _mm256_set1_ps(INV_128);
                    let offset = _mm256_set1_ps(-U8_OFFSET);
                    run::<_, _, UNROLL>(bytes, out, |q: &[u8; 8], p| {
                        let v = _mm256_cvtepu8_epi32(low64(q));
                        store(p, _mm256_fmadd_ps(_mm256_cvtepi32_ps(v), scale, offset));
                    })
                }
            }
        }
    );

    fearless_simd::kernel!(
        /// SSE4.2 over whole vectors of `I16` (`wide`) or `I8`; returns the number of values
        /// done. `U8` takes the SSE2 path, which is one instruction per 16 values shorter.
        #[inline(always)]
        pub(super) fn sse4_2(t: Sse4_2, wide: bool, bytes: &[u8], out: &mut [f32]) -> usize {
            let store = |p: &mut [f32; 4], v| f32x4::simd_from(t, v).store_array(p);
            if wide {
                let scale = _mm_set1_ps(INV_32768);
                run::<_, _, UNROLL>(bytes, out, |q: &[u8; 8], p| {
                    // The 8 bytes as the low half of a vector: the load folds into `pmovsxwd`.
                    let v = _mm_cvtepi16_epi32(_mm_set_epi64x(0, i64::from_le_bytes(*q)));
                    store(p, _mm_mul_ps(_mm_cvtepi32_ps(v), scale));
                })
            } else {
                let scale = _mm_set1_ps(INV_128);
                run::<_, _, UNROLL>(bytes, out, |q: &[u8; 4], p| {
                    let v = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(i32::from_le_bytes(*q)));
                    store(p, _mm_mul_ps(_mm_cvtepi32_ps(v), scale));
                })
            }
        }
    );

    fearless_simd::kernel!(
        /// SSE2 over whole vectors, building the float bits (see the module documentation);
        /// returns the number of values done. Also the path of `U8` at SSE4.2.
        #[inline(always)]
        pub(super) fn sse2(t: Sse2, kind: Kind, bytes: &[u8], out: &mut [f32]) -> usize {
            let zero = _mm_setzero_si128();
            let store = |p: &mut [f32], v| f32x4::simd_from(t, v).store_slice(p);
            match kind {
                Kind::I16 => {
                    let flip = _mm_set1_epi16(i16::MIN);
                    let high = _mm_set1_epi16(HIGH_U8_I16 as i16);
                    let magic = _mm_set1_ps(I16_MAGIC);
                    run::<_, _, { UNROLL / 2 }>(bytes, out, |q: &[u8; 16], p: &mut [f32; 8]| {
                        let u = _mm_xor_si128(u8x16::load_array_ref(t, q).into(), flip);
                        let lo = _mm_castsi128_ps(_mm_unpacklo_epi16(u, high));
                        let hi = _mm_castsi128_ps(_mm_unpackhi_epi16(u, high));
                        let (pl, ph) = p.split_at_mut(4);
                        store(pl, _mm_sub_ps(lo, magic));
                        store(ph, _mm_sub_ps(hi, magic));
                    })
                }
                Kind::I8 | Kind::U8 => {
                    let signed = kind == Kind::I8;
                    let flip = _mm_set1_epi8(i8::MIN);
                    let high = _mm_set1_epi16(if signed { HIGH_I8 } else { HIGH_U8_I16 } as i16);
                    let magic = _mm_set1_ps(if signed { I8_MAGIC } else { U8_MAGIC });
                    run::<_, _, { UNROLL / 4 }>(bytes, out, |q: &[u8; 16], p: &mut [f32; 16]| {
                        let v: __m128i = u8x16::load_array_ref(t, q).into();
                        // `I8`: the byte `x + 128` in byte 0 of the 32-bit lane; `U8`: the
                        // byte in byte 1.
                        let (w_lo, w_hi) = if signed {
                            let u = _mm_xor_si128(v, flip);
                            (_mm_unpacklo_epi8(u, zero), _mm_unpackhi_epi8(u, zero))
                        } else {
                            (_mm_unpacklo_epi8(zero, v), _mm_unpackhi_epi8(zero, v))
                        };
                        let (p01, p23) = p.split_at_mut(8);
                        let (p0, p1) = p01.split_at_mut(4);
                        let (p2, p3) = p23.split_at_mut(4);
                        let f = |w| _mm_sub_ps(_mm_castsi128_ps(w), magic);
                        store(p0, f(_mm_unpacklo_epi16(w_lo, high)));
                        store(p1, f(_mm_unpackhi_epi16(w_lo, high)));
                        store(p2, f(_mm_unpacklo_epi16(w_hi, high)));
                        store(p3, f(_mm_unpackhi_epi16(w_hi, high)));
                    })
                }
            }
        }
    );
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use core::arch::aarch64::{
        float32x4x2_t, float32x4x4_t, uint8x16_t, uint8x16x2_t, vcvtq_n_f32_s32, vdupq_n_f32,
        vget_low_s16, vmovl_high_s16, vmovl_s16, vqtbl1q_u8, vqtbl2q_u8, vreinterpretq_f32_u8,
        vreinterpretq_s16_u8, vreinterpretq_s32_u8, vsubq_f32,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{f32x8, f32x16, u8x16};

    use super::{HIGH_U8_I16, Kind, U8_MAGIC, run};

    /// Output vectors per iteration of the main loops (`unrolled8` writes out eight units of
    /// four).
    const UNROLL: usize = 32;

    /// A table index beyond the table: the lookup gives 0.
    const Z: u8 = 0xff;
    /// `I8`: the table indices that put byte `4g + k` at the top of 32-bit lane `k` (`x << 24`)
    /// for the group `g`, with zeros below.
    const I8_INDEX: [[u8; 16]; 4] = [
        [Z, Z, Z, 0, Z, Z, Z, 1, Z, Z, Z, 2, Z, Z, Z, 3],
        [Z, Z, Z, 4, Z, Z, Z, 5, Z, Z, Z, 6, Z, Z, Z, 7],
        [Z, Z, Z, 8, Z, Z, Z, 9, Z, Z, Z, 10, Z, Z, Z, 11],
        [Z, Z, Z, 12, Z, Z, Z, 13, Z, Z, Z, 14, Z, Z, Z, 15],
    ];
    /// The second table register of `U8`: zero and the bytes of the half-word `0x4380`.
    const MAGIC_BYTES: [u8; 16] = {
        let [lo, hi] = HIGH_U8_I16.to_le_bytes();
        [0, lo, hi, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    };
    /// The indices of `MAGIC_BYTES` in the two-register table: zero, the low and the high
    /// byte of `0x4380`.
    const O: u8 = 16;
    const L: u8 = 17;
    const H: u8 = 18;
    /// `U8`: the table indices that put byte `4g + k` into byte 1 of 32-bit lane `k` for the
    /// group `g`, with zero below and `0x4380` above (`0x4380_uu00`).
    const U8_INDEX: [[u8; 16]; 4] = [
        [O, 0, L, H, O, 1, L, H, O, 2, L, H, O, 3, L, H],
        [O, 4, L, H, O, 5, L, H, O, 6, L, H, O, 7, L, H],
        [O, 8, L, H, O, 9, L, H, O, 10, L, H, O, 11, L, H],
        [O, 12, L, H, O, 13, L, H, O, 14, L, H, O, 15, L, H],
    ];

    /// [`run`] for `unit` (16 bytes to 16 values) with eight units per iteration written out:
    /// LLVM does not unroll a loop over table lookups (`tbl`) by itself. Returns the number of
    /// values done.
    #[inline(always)]
    fn unrolled8(
        bytes: &[u8],
        out: &mut [f32],
        mut unit: impl FnMut(&[u8; 16], &mut [f32; 16]),
    ) -> usize {
        let done = run::<_, _, 1>(bytes, out, |q: &[u8; 128], p: &mut [f32; 128]| {
            if let ([q0, q1, q2, q3, q4, q5, q6, q7], [p0, p1, p2, p3, p4, p5, p6, p7]) =
                (q.as_chunks().0, p.as_chunks_mut().0)
            {
                unit(q0, p0);
                unit(q1, p1);
                unit(q2, p2);
                unit(q3, p3);
                unit(q4, p4);
                unit(q5, p5);
                unit(q6, p6);
                unit(q7, p7);
            }
        });
        let bytes = bytes.get(done..).unwrap_or_default();
        let out = out.get_mut(done..).unwrap_or_default();
        done + run::<_, _, 1>(bytes, out, unit)
    }

    fearless_simd::kernel!(
        /// NEON over whole vectors; returns the number of values done.
        #[inline(always)]
        pub(super) fn neon(t: Neon, kind: Kind, bytes: &[u8], out: &mut [f32]) -> usize {
            let load = |q: &[u8; 16]| u8x16::load_array_ref(t, q).into();
            let store2 = |p: &mut [f32; 8], a, b| {
                f32x8::simd_from(t, float32x4x2_t(a, b)).store_array(p);
            };
            let store4 = |p: &mut [f32; 16], a, b, c, d| {
                f32x16::simd_from(t, float32x4x4_t(a, b, c, d)).store_array(p);
            };
            match kind {
                Kind::I16 => run::<_, _, { UNROLL / 2 }>(bytes, out, |q: &[u8; 16], p| {
                    let v = vreinterpretq_s16_u8(load(q));
                    let lo = vcvtq_n_f32_s32::<15>(vmovl_s16(vget_low_s16(v)));
                    let hi = vcvtq_n_f32_s32::<15>(vmovl_high_s16(v));
                    store2(p, lo, hi);
                }),
                Kind::I8 => {
                    let [i0, i1, i2, i3]: [uint8x16_t; 4] =
                        I8_INDEX.map(|i| u8x16::simd_from(t, i).into());
                    unrolled8(bytes, out, |q, p| {
                        let v = load(q);
                        let f = |i| vcvtq_n_f32_s32::<31>(vreinterpretq_s32_u8(vqtbl1q_u8(v, i)));
                        store4(p, f(i0), f(i1), f(i2), f(i3));
                    })
                }
                Kind::U8 => {
                    let [i0, i1, i2, i3]: [uint8x16_t; 4] =
                        U8_INDEX.map(|i| u8x16::simd_from(t, i).into());
                    let c = u8x16::simd_from(t, MAGIC_BYTES).into();
                    let magic = vdupq_n_f32(U8_MAGIC);
                    unrolled8(bytes, out, |q, p| {
                        let table = uint8x16x2_t(load(q), c);
                        let f = |i| vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(table, i)), magic);
                        store4(p, f(i0), f(i1), f(i2), f(i3));
                    })
                }
            }
        }
    );
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::{
        f32x4_convert_i32x4, f32x4_mul, f32x4_splat, f32x4_sub, i8x16_shuffle,
        i32x4_extend_high_i16x8, i32x4_extend_low_i16x8, u8x16, u8x16_splat, v128, v128_xor,
    };
    use fearless_simd::f32x4;
    use fearless_simd::prelude::*;

    use super::{HIGH_I8, HIGH_U8_I16, I8_MAGIC, INV_32768, Kind, U8_MAGIC};

    /// `i8x16.shuffle` of `a` and `b` with the 16 byte indices as a list.
    macro_rules! shuffle {
        ($a:expr, $b:expr; $($i:literal)*) => {
            i8x16_shuffle::<$($i),*>($a, $b)
        };
    }

    /// The second operand of the shuffles: zero, then the upper half-word `high` (indices 16,
    /// 17 and 18).
    #[inline(always)]
    fn second(high: u16) -> v128 {
        let [lo, hi] = high.to_le_bytes();
        u8x16(0, lo, hi, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    }

    /// Defines a function that runs `unit` on the units of one block, written out and in a
    /// function of its own: LLVM unrolls no loop for wasm32, and in a loop it computes every
    /// address with an `i32.add`, whereas here the addresses are constant offsets from the
    /// parameters and fold into the `offset` immediates of the loads and stores.
    macro_rules! block {
        ($name:ident, $n:literal: $($i:literal)*) => {
            #[doc = concat!("A block of ", $n, " units.")]
            #[inline(never)]
            fn $name<const IN: usize, const OUT: usize>(
                q: &[[u8; IN]; $n],
                p: &mut [[f32; OUT]; $n],
                unit: impl Fn(&[u8; IN], &mut [f32; OUT]),
            ) {
                $(unit(&q[$i], &mut p[$i]);)*
            }
        };
    }
    block!(block16, 16: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);
    block!(block4, 4: 0 1 2 3);

    /// Runs `unit` (`IN` input bytes to `OUT` values) over all whole units that fit both
    /// slices, 16 and then 4 per block, and the rest one by one; returns the number of values
    /// done.
    #[inline(always)]
    fn run<const IN: usize, const OUT: usize>(
        bytes: &[u8],
        out: &mut [f32],
        unit: impl Fn(&[u8; IN], &mut [f32; OUT]) + Copy,
    ) -> usize {
        let (ib, _) = bytes.as_chunks::<IN>();
        let (ob, _) = out.as_chunks_mut::<OUT>();
        let n = ib.len().min(ob.len());
        let (ib, _) = ib.split_at(n);
        let (ob, _) = ob.split_at_mut(n);
        let (ib, ir) = ib.as_chunks::<16>();
        let (ob, or) = ob.as_chunks_mut::<16>();
        for (b, o) in ib.iter().zip(ob) {
            block16(b, o, unit);
        }
        let (ib, ir) = ir.as_chunks::<4>();
        let (ob, or) = or.as_chunks_mut::<4>();
        for (b, o) in ib.iter().zip(ob) {
            block4(b, o, unit);
        }
        for (b, o) in ir.iter().zip(or) {
            unit(b, o);
        }
        n * OUT
    }

    fearless_simd::kernel!(
        /// SIMD128 over whole vectors; returns the number of values done. Counted in wasm
        /// operators, `I16` is shortest with the conversion (two vectors per load), `I8` and
        /// `U8` with the float bits built by one shuffle per vector (see the module
        /// documentation).
        #[inline(always)]
        pub(super) fn simd128(t: WasmSimd128, kind: Kind, bytes: &[u8], out: &mut [f32]) -> usize {
            let load =
                move |q: &[u8; 16]| -> v128 { fearless_simd::u8x16::load_array_ref(t, q).into() };
            let store = move |p: &mut [f32], v: v128| f32x4::simd_from(t, v).store_slice(p);
            match kind {
                Kind::I16 => run(bytes, out, move |q: &[u8; 16], p: &mut [f32; 8]| {
                    let v = load(q);
                    let scale = f32x4_splat(INV_32768);
                    let f = |w| f32x4_mul(f32x4_convert_i32x4(w), scale);
                    let (pl, ph) = p.split_at_mut(4);
                    store(pl, f(i32x4_extend_low_i16x8(v)));
                    store(ph, f(i32x4_extend_high_i16x8(v)));
                }),
                Kind::I8 => run(bytes, out, move |q: &[u8; 16], p: &mut [f32; 16]| {
                    // `x + 128` in byte 0, zero in byte 1.
                    let u = v128_xor(load(q), u8x16_splat(0x80));
                    let c = second(HIGH_I8);
                    let d0 = shuffle!(u, c; 0 16 17 18 1 16 17 18 2 16 17 18 3 16 17 18);
                    let d1 = shuffle!(u, c; 4 16 17 18 5 16 17 18 6 16 17 18 7 16 17 18);
                    let d2 = shuffle!(u, c; 8 16 17 18 9 16 17 18 10 16 17 18 11 16 17 18);
                    let d3 = shuffle!(u, c; 12 16 17 18 13 16 17 18 14 16 17 18 15 16 17 18);
                    four(p, [d0, d1, d2, d3], I8_MAGIC, store);
                }),
                Kind::U8 => run(bytes, out, move |q: &[u8; 16], p: &mut [f32; 16]| {
                    // The byte in byte 1, zero in byte 0.
                    let v = load(q);
                    let c = second(HIGH_U8_I16);
                    let d0 = shuffle!(v, c; 16 0 17 18 16 1 17 18 16 2 17 18 16 3 17 18);
                    let d1 = shuffle!(v, c; 16 4 17 18 16 5 17 18 16 6 17 18 16 7 17 18);
                    let d2 = shuffle!(v, c; 16 8 17 18 16 9 17 18 16 10 17 18 16 11 17 18);
                    let d3 = shuffle!(v, c; 16 12 17 18 16 13 17 18 16 14 17 18 16 15 17 18);
                    four(p, [d0, d1, d2, d3], U8_MAGIC, store);
                }),
            }
        }
    );

    /// Stores the four magic floats `d` minus `magic` to `p`.
    #[inline(always)]
    fn four(p: &mut [f32; 16], d: [v128; 4], magic: f32, store: impl Fn(&mut [f32], v128)) {
        let m = f32x4_splat(magic);
        let [d0, d1, d2, d3] = d;
        if let [p0, p1, p2, p3] = p.as_chunks_mut::<4>().0 {
            store(p0, f32x4_sub(d0, m));
            store(p1, f32x4_sub(d1, m));
            store(p2, f32x4_sub(d2, m));
            store(p3, f32x4_sub(d3, m));
        }
    }
}

/// `CS8` to Q15: `x << 8`.
#[inline(always)]
fn i8_to_q15<S: SimdToken>(simd: S, bytes: &[u8], out: &mut [i16]) {
    let n = S::i8s::LEN;
    let h = S::i16s::LEN;
    let mut ic = bytes.chunks_exact(n);
    let mut oc = out.chunks_exact_mut(n);
    for (b, o) in (&mut ic).zip(&mut oc) {
        let v: S::i8s = S::u8s::from_slice(simd, b).bitcast();
        let (lo, hi) = v.widen();
        let (o_lo, o_hi) = o.split_at_mut(h);
        (lo << 8u32).store_slice(o_lo);
        (hi << 8u32).store_slice(o_hi);
    }
    for (o, &b) in oc.into_remainder().iter_mut().zip(ic.remainder()) {
        *o = i16::from(b as i8) << 8;
    }
}

/// `CU8` to Q15: `(2x - 255) << 7`.
#[inline(always)]
fn u8_to_q15<S: SimdToken>(simd: S, bytes: &[u8], out: &mut [i16]) {
    let n = S::u8s::LEN;
    let h = S::i16s::LEN;
    let c255 = S::i16s::splat(simd, 255);
    let mut ic = bytes.chunks_exact(n);
    let mut oc = out.chunks_exact_mut(n);
    for (b, o) in (&mut ic).zip(&mut oc) {
        let v = S::u8s::from_slice(simd, b);
        let (lo, hi) = v.widen();
        let lo: S::i16s = lo.bitcast();
        let hi: S::i16s = hi.bitcast();
        let (o_lo, o_hi) = o.split_at_mut(h);
        (((lo + lo) - c255) << 7u32).store_slice(o_lo);
        (((hi + hi) - c255) << 7u32).store_slice(o_hi);
    }
    for (o, &b) in oc.into_remainder().iter_mut().zip(ic.remainder()) {
        *o = (2 * i16::from(b) - 255) << 7;
    }
}

/// The bytes of a `CQ15` slice as `i16` values.
fn q15_view(out: &mut [CQ15]) -> &mut [i16] {
    bytemuck::cast_slice_mut(out)
}

impl Convert<C32> for Simd {
    fn convert(&self, format: &SampleFormat, input: &[u8], output: &mut [C32]) -> Result<usize> {
        let Some(kind) = fast_kind(format) else {
            return Fallback.convert(format, input, output);
        };
        let Some(samples) = fast_samples(kind, input.len(), output.len()) else {
            return Fallback.convert(format, input, output);
        };
        let Some(out) = output.get_mut(..samples) else {
            return Fallback.convert(format, input, output);
        };
        let out: &mut [f32] = bytemuck::cast_slice_mut(out);
        dispatch!(simd => f32_in(simd, kind, input, out));
        Ok(samples)
    }
}

impl Convert<CQ15> for Simd {
    fn convert(&self, format: &SampleFormat, input: &[u8], output: &mut [CQ15]) -> Result<usize> {
        let Some(kind) = fast_kind(format) else {
            return Fallback.convert(format, input, output);
        };
        let Some(samples) = fast_samples(kind, input.len(), output.len()) else {
            return Fallback.convert(format, input, output);
        };
        let Some(out) = output.get_mut(..samples) else {
            return Fallback.convert(format, input, output);
        };
        let out = q15_view(out);
        match kind {
            // Little-endian `cs16` is byte-identical to `CQ15`.
            Kind::I16 => {
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(out);
                bytes.copy_from_slice(input);
            }
            Kind::I8 => dispatch!(simd => i8_to_q15(simd, input, out)),
            Kind::U8 => dispatch!(simd => u8_to_q15(simd, input, out)),
        }
        Ok(samples)
    }
}
