//! `am_mod`, bit-exact with the fallback (the contract of the kernel is bit-exact,
//! `docs/kernels.md`).
//!
//! `C32`: the loop of the fallback (`carrier * (1 + depth * x')` with `x'` the clamped input and
//! NaN masked to 0, the pair `(re, 0)` per sample) compiled at the level of the dispatch, which
//! LLVM vectorizes at every level as it does the fallback at the level of the build (`maxps`,
//! `minps`, `cmpordps`, `andps`, two products and a sum without fusion, then `unpcklps` /
//! `unpckhps` with zero or, from AVX2 on, `vpmovzxdq`). The backend has it so that the build with
//! run-time detection runs it at the level of the CPU; with a level fixed at compile time it is
//! the code of the fallback.
//!
//! `CQ15`: the `i32` form of the fallback, `re = sat16(c + round_shift(c t, 15))` with
//! `t = round_shift(m x, 15)` (`c`, `m` the `Q15` gains of carrier and depth, both 0 to 32767;
//! the proof that this equals the `i64` formula of the contract is at the fallback), on whole
//! vectors of 16-bit lanes: the two rounding products are the rounding high multiplication of
//! the level (`pmulhrsw`, SSSE3, so SSE4.2, AVX2 and AVX-512 on the AVX2 path; `sqrdmulh` on
//! NEON, which computes `(2 a b + 2^15) >> 16`, the same value; `i16x8.q15mulr_sat_s` on
//! SIMD128). Each computes `(a b + 2^14) >> 15` exactly; the only product outside the 16-bit
//! range, `-32768 * -32768`, cannot occur because `m` and `c` are at most 32767. `t` lies in
//! `[-32767, 32766]`, the sum `c + round_shift(c t, 15)` in `[0, 65532]`, so the saturating
//! 16-bit addition (`paddsw`, `sqadd`, `i16x8.add_sat_s`) is `sat16` of the sum. The samples
//! `(re, 0)` are the zero extension of `re` to 32 bits: `punpcklwd` / `punpckhwd` with zero
//! (SSE4.2), `vpmovzxwd` of each half (AVX2), `zip1` / `zip2` with zero (NEON),
//! `i32x4.extend_low/high_i16x8_u` (SIMD128). The rest of the call, shorter than one vector, and
//! SSE2 (no rounding high multiplication) run the `i32` loop of the fallback at the level of the
//! dispatch.

use fearless_simd::Simd as SimdToken;
use sdrk_core::kernels::modulate::AmMod;
use sdrk_core::{C32, CQ15, Fallback, Q15, Result};

use crate::Simd;
use crate::level::dispatch_fn;
use crate::util::check_len;

/// Whether `carrier` and `depth` are valid (`0 ..= 1`); invalid calls go to the fallback for its
/// error.
#[inline(always)]
fn valid(carrier: f32, depth: f32) -> bool {
    (0.0..=1.0).contains(&carrier) && (0.0..=1.0).contains(&depth)
}

/// The `Q15` gain of the fallback: `floor(v * 32768 + 0.5)` limited to 32767 for `v` in
/// `0 ..= 1` (the truncation of a positive value is its floor).
#[inline(always)]
fn gain(v: f32) -> i16 {
    ((f64::from(v) * 32768.0 + 0.5) as i32).min(32767) as i16
}

/// The `C32` value of the fallback, the same operations in the same order.
#[inline(always)]
fn am_f32(carrier: f32, depth: f32, x: f32) -> f32 {
    let mask = 0u32.wrapping_sub(u32::from(!x.is_nan()));
    let x = f32::from_bits(x.clamp(-1.0, 1.0).to_bits() & mask);
    let t = depth * x;
    let u = 1.0 + t;
    carrier * u
}

/// The `CQ15` value of the fallback in `i32`.
#[inline(always)]
fn am_q15(c: i32, m: i32, x: i16) -> i16 {
    let t = (m * i32::from(x) + (1 << 14)) >> 15;
    let r = (c * t + (1 << 14)) >> 15;
    (c + r).min(32767) as i16
}

/// The whole `C32` call: the loop of the fallback at the level of `_simd`.
#[inline(always)]
fn am_c32<S: SimdToken>(_simd: S, carrier: f32, depth: f32, input: &[f32], out: &mut [[f32; 2]]) {
    for (o, &x) in out.iter_mut().zip(input) {
        *o = [am_f32(carrier, depth, x), 0.0];
    }
}

/// The level-specific `CQ15` path for the current level, if there is one: returns the number of
/// leading samples done.
#[inline(always)]
fn am_cq15_level<S: SimdToken>(
    simd: S,
    c: i16,
    m: i16,
    input: &[i16],
    out: &mut [[i16; 2]],
) -> usize {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx2() {
            return x86::avx2(t, c, m, input, out);
        }
        if let Some(t) = level.as_sse4_2() {
            return x86::sse4_2(t, c, m, input, out);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return neon::am(t, c, m, input, out);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return wasm::am(t, c, m, input, out);
    }
    let _ = (level, c, m, input, out);
    0
}

/// The whole `CQ15` call: the path of the level, then the `i32` loop for the rest.
#[inline(always)]
fn am_cq15<S: SimdToken>(simd: S, c: i16, m: i16, input: &[i16], out: &mut [[i16; 2]]) {
    let done = am_cq15_level(simd, c, m, input, out);
    let (c, m) = (i32::from(c), i32::from(m));
    let rest = out
        .get_mut(done..)
        .unwrap_or_default()
        .iter_mut()
        .zip(input.get(done..).unwrap_or_default());
    for (o, &x) in rest {
        *o = [am_q15(c, m, x), 0];
    }
}

// The entries, one function per level (see `dispatch_fn!`).
dispatch_fn!(fn run_c32(carrier: f32, depth: f32, input: &[f32], out: &mut [[f32; 2]]) => am_c32);
dispatch_fn!(fn run_cq15(c: i16, m: i16, input: &[i16], out: &mut [[i16; 2]]) => am_cq15);

impl AmMod<C32> for Simd {
    fn am_mod(&self, carrier: f32, depth: f32, input: &[f32], output: &mut [C32]) -> Result<()> {
        if !valid(carrier, depth) {
            return Fallback.am_mod(carrier, depth, input, output);
        }
        check_len(input.len(), output.len())?;
        run_c32(carrier, depth, input, bytemuck::cast_slice_mut(output));
        Ok(())
    }
}

impl AmMod<CQ15> for Simd {
    fn am_mod(&self, carrier: f32, depth: f32, input: &[Q15], output: &mut [CQ15]) -> Result<()> {
        if !valid(carrier, depth) {
            return Fallback.am_mod(carrier, depth, input, output);
        }
        check_len(input.len(), output.len())?;
        run_cq15(
            gain(carrier),
            gain(depth),
            bytemuck::cast_slice(input),
            bytemuck::cast_slice_mut(output),
        );
        Ok(())
    }
}

/// `$body` with `$i` the blocks of `$v` input values and `$o` the blocks of `$v` output samples;
/// evaluates to the number of samples done (a multiple of `$v`).
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! blocks {
    ($input:expr, $out:expr, $v:literal, |$i:ident, $o:ident| $body:block) => {{
        let n = $input.len().min($out.len()) / $v * $v;
        let (input, _) = $input.split_at(n);
        let (out, _) = $out.split_at_mut(n);
        let body = |$i: &[i16; $v], $o: &mut [[i16; 2]; $v]| $body;
        let (ib, _) = input.as_chunks::<$v>();
        let (ob, _) = out.as_chunks_mut::<$v>();
        for (i, o) in ib.iter().zip(ob) {
            body(i, o);
        }
        n
    }};
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::{
        _mm_adds_epi16, _mm_mulhrs_epi16, _mm_set1_epi16, _mm_setzero_si128, _mm_unpackhi_epi16,
        _mm_unpacklo_epi16, _mm256_adds_epi16, _mm256_castsi256_si128, _mm256_cvtepu16_epi32,
        _mm256_extracti128_si256, _mm256_mulhrs_epi16, _mm256_set1_epi16,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{i16x8, i16x16};

    fearless_simd::kernel!(
        pub(super) fn sse4_2(
            t: Sse4_2,
            c: i16,
            m: i16,
            input: &[i16],
            out: &mut [[i16; 2]],
        ) -> usize {
            let (cv, mv, zero) = (_mm_set1_epi16(c), _mm_set1_epi16(m), _mm_setzero_si128());
            blocks!(input, out, 8, |i, o| {
                let x = i16x8::from_slice(t, i).into();
                let r = _mm_mulhrs_epi16(_mm_mulhrs_epi16(x, mv), cv);
                let re = _mm_adds_epi16(r, cv);
                let (o0, o1) = o.as_flattened_mut().split_at_mut(8);
                i16x8::simd_from(t, _mm_unpacklo_epi16(re, zero)).store_slice(o0);
                i16x8::simd_from(t, _mm_unpackhi_epi16(re, zero)).store_slice(o1);
            })
        }
    );

    fearless_simd::kernel!(
        pub(super) fn avx2(t: Avx2, c: i16, m: i16, input: &[i16], out: &mut [[i16; 2]]) -> usize {
            let (cv, mv) = (_mm256_set1_epi16(c), _mm256_set1_epi16(m));
            blocks!(input, out, 16, |i, o| {
                let x = i16x16::from_slice(t, i).into();
                let r = _mm256_mulhrs_epi16(_mm256_mulhrs_epi16(x, mv), cv);
                let re = _mm256_adds_epi16(r, cv);
                let (o0, o1) = o.as_flattened_mut().split_at_mut(16);
                let lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(re));
                let hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(re));
                i16x16::simd_from(t, lo).store_slice(o0);
                i16x16::simd_from(t, hi).store_slice(o1);
            })
        }
    );
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::{vdupq_n_s16, vqaddq_s16, vqrdmulhq_s16, vzip1q_s16, vzip2q_s16};
    use fearless_simd::i16x8;
    use fearless_simd::prelude::*;

    fearless_simd::kernel!(
        pub(super) fn am(t: Neon, c: i16, m: i16, input: &[i16], out: &mut [[i16; 2]]) -> usize {
            let (cv, mv, zero) = (vdupq_n_s16(c), vdupq_n_s16(m), vdupq_n_s16(0));
            blocks!(input, out, 8, |i, o| {
                let x = i16x8::from_slice(t, i).into();
                let r = vqrdmulhq_s16(vqrdmulhq_s16(x, mv), cv);
                let re = vqaddq_s16(r, cv);
                let (o0, o1) = o.as_flattened_mut().split_at_mut(8);
                i16x8::simd_from(t, vzip1q_s16(re, zero)).store_slice(o0);
                i16x8::simd_from(t, vzip2q_s16(re, zero)).store_slice(o1);
            })
        }
    );
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::{
        i16x8_add_sat, i16x8_q15mulr_sat, i16x8_splat, u32x4_extend_high_u16x8,
        u32x4_extend_low_u16x8,
    };
    use fearless_simd::i16x8;
    use fearless_simd::prelude::*;

    fearless_simd::kernel!(
        pub(super) fn am(
            t: WasmSimd128,
            c: i16,
            m: i16,
            input: &[i16],
            out: &mut [[i16; 2]],
        ) -> usize {
            let (cv, mv) = (i16x8_splat(c), i16x8_splat(m));
            blocks!(input, out, 8, |i, o| {
                let x = i16x8::from_slice(t, i).into();
                let r = i16x8_q15mulr_sat(i16x8_q15mulr_sat(x, mv), cv);
                let re = i16x8_add_sat(r, cv);
                let (o0, o1) = o.as_flattened_mut().split_at_mut(8);
                i16x8::simd_from(t, u32x4_extend_low_u16x8(re)).store_slice(o0);
                i16x8::simd_from(t, u32x4_extend_high_u16x8(re)).store_slice(o1);
            })
        }
    );
}
