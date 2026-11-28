//! `am_mod`, bit-exact with the fallback (the contract of the kernel is bit-exact,
//! `docs/kernels.md`).
//!
//! `C32`: `carrier * (1 + depth * x')` with `x'` the clamped input and NaN masked to 0, the pair
//! `(re, 0)` per sample, the operations of the fallback in its order (two products and a sum
//! without fusion).
//!
//! - NEON (O2, D2970): `fcmeq` of `x` with itself for the mask, `fmax` / `fmin` with -1 and 1
//!   instead of the compare-and-select pairs that LLVM makes of the clamp of the fallback, and
//!   `st2` of the values with a zero vector for the pairs, two vectors per iteration.
//! - SIMD128 (O2, D2970): the instructions of the compiled fallback (`f32x4.eq`, `f32x4.pmax` /
//!   `pmin`, `v128.and`, the products and the sum, two shuffles with zero), in blocks of four
//!   vectors written out in a function of its own, so that the addresses fold into the offsets
//!   of the loads and stores (the compiled loop adds every address).
//! - x86: the loop of the fallback compiled at the level of the dispatch, which LLVM
//!   vectorizes as it does the fallback at the level of the build (`maxps`, `minps`,
//!   `cmpordps`, `andps`, the products and the sum, then `unpcklps` / `unpckhps` with zero or,
//!   from AVX2 on, `vpmovzxdq`). A hand-written loop of the same instructions, unrolled, gains
//!   less than 10 % in `llvm-mca` and loses on `skylake` (D2970), so x86 keeps the compiled
//!   loop: the build with run-time detection runs it at the level of the CPU; with a level fixed
//!   at compile time it is the code of the fallback.
//!
//! `CQ15`: the `i32` form of the fallback, `re = sat16(c + round_shift(c t, 15))` with
//! `t = round_shift(m x, 15)` (`c`, `m` the `Q15` gains of carrier and depth, both 0 to 32767;
//! the proof that this equals the `i64` formula of the contract is at the fallback), on whole
//! vectors of 16-bit lanes: the two rounding products are the rounding high multiplication of
//! the level (`pmulhrsw`, SSSE3, so SSE4.2, AVX2 and AVX-512 on the AVX2 path; at SSE2 its six
//! SSE2 instructions of [`crate::util::mulhrs_sse2!`], the same bits, D3008; `sqrdmulh` on
//! NEON, which computes `(2 a b + 2^15) >> 16`, the same value; `i16x8.q15mulr_sat_s` on
//! SIMD128). Each computes `(a b + 2^14) >> 15` exactly; the only product outside the 16-bit
//! range, `-32768 * -32768`, cannot occur because `m` and `c` are at most 32767. `t` lies in
//! `[-32767, 32766]`, the sum `c + round_shift(c t, 15)` in `[0, 65532]`, so the saturating
//! 16-bit addition (`paddsw`, `sqadd`, `i16x8.add_sat_s`) is `sat16` of the sum. The samples
//! `(re, 0)` are the zero extension of `re` to 32 bits: `punpcklwd` / `punpckhwd` with zero
//! (SSE2, SSE4.2), `vpmovzxwd` of each half (AVX2), `zip1` / `zip2` with zero (NEON),
//! `i32x4.extend_low/high_i16x8_u` (SIMD128). The rest of the call, shorter than one vector, runs
//! the `i32` loop of the fallback at the level of the dispatch.

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

/// The level-specific `C32` path for the current level, if there is one: returns the number of
/// leading samples done.
#[inline(always)]
fn am_c32_level<S: SimdToken>(
    simd: S,
    carrier: f32,
    depth: f32,
    input: &[f32],
    out: &mut [[f32; 2]],
) -> usize {
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = simd.level().as_neon() {
        return neon::am_c32(t, carrier, depth, input, out);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = simd.level().as_wasm_simd128() {
        return wasm::am_c32(t, carrier, depth, input, out);
    }
    let _ = (simd, carrier, depth, input, out);
    0
}

/// The whole `C32` call: the path of the level, then the loop of the fallback at the level of
/// `simd` for the rest (the whole call where the level has no path).
#[inline(always)]
fn am_c32<S: SimdToken>(simd: S, carrier: f32, depth: f32, input: &[f32], out: &mut [[f32; 2]]) {
    let done = am_c32_level(simd, carrier, depth, input, out);
    let rest = out
        .get_mut(done..)
        .unwrap_or_default()
        .iter_mut()
        .zip(input.get(done..).unwrap_or_default());
    for (o, &x) in rest {
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
        if let Some(t) = level.as_sse2() {
            return x86::sse2(t, c, m, input, out);
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

    use crate::util::mulhrs_sse2;

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
        /// [`sse4_2`] with `pmulhrsw` in SSE2 instructions ([`mulhrs_sse2!`], decision D3008).
        pub(super) fn sse2(t: Sse2, c: i16, m: i16, input: &[i16], out: &mut [[i16; 2]]) -> usize {
            let (cv, mv, zero) = (_mm_set1_epi16(c), _mm_set1_epi16(m), _mm_setzero_si128());
            blocks!(input, out, 8, |i, o| {
                let x = i16x8::from_slice(t, i).into();
                let r = mulhrs_sse2!(mulhrs_sse2!(x, mv), cv);
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
#[allow(unsafe_code)]
mod neon {
    use core::arch::aarch64::{
        float32x4_t, float32x4x2_t, vaddq_f32, vandq_u32, vceqq_f32, vdupq_n_f32, vdupq_n_s16,
        vmaxq_f32, vminq_f32, vmulq_n_f32, vqaddq_s16, vqrdmulhq_s16, vreinterpretq_f32_u32,
        vreinterpretq_u32_f32, vst2q_f32, vzip1q_s16, vzip2q_s16,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4, i16x8};

    fearless_simd::kernel!(
        /// `C32` on whole blocks of four samples, two per iteration: `fcmeq` of `x` with itself
        /// (the mask of the fallback), `fmax` / `fmin` with -1 and 1 (they equal the clamp on
        /// every value that is not NaN, and the mask takes NaN to `+0` whatever they give), the
        /// `and`, then `depth x'`, `+ 1` and `carrier u` without fusion, and `st2` of the values
        /// with a zero vector writes the pairs `(re, 0)`. The operations and their order are the
        /// fallback's on every lane, so the bits are the same. Returns the number of samples
        /// done.
        pub(super) fn am_c32(
            t: Neon,
            carrier: f32,
            depth: f32,
            input: &[f32],
            out: &mut [[f32; 2]],
        ) -> usize {
            let (lo, hi, zero) = (vdupq_n_f32(-1.0), vdupq_n_f32(1.0), vdupq_n_f32(0.0));
            let am = |x: float32x4_t| {
                let keep = vceqq_f32(x, x);
                let c = vminq_f32(vmaxq_f32(x, lo), hi);
                let c = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(c), keep));
                vmulq_n_f32(vaddq_f32(vmulq_n_f32(c, depth), hi), carrier)
            };
            let step = |i: &[f32; 4], o: &mut [[f32; 2]; 4]| {
                let re = am(f32x4::from_slice(t, i).into());
                // SAFETY: `o` is exactly 4 pairs (8 floats), which `vst2q_f32` writes.
                unsafe { vst2q_f32(o.as_flattened_mut().as_mut_ptr(), float32x4x2_t(re, zero)) };
            };
            let n = input.len().min(out.len()) / 4 * 4;
            let (ib, _) = input.split_at(n);
            let (ob, _) = out.split_at_mut(n);
            let (ib, _) = ib.as_chunks::<4>();
            let (ob, _) = ob.as_chunks_mut::<4>();
            let (i2, ir) = ib.as_chunks::<2>();
            let (o2, or) = ob.as_chunks_mut::<2>();
            for ([i0, i1], [o0, o1]) in i2.iter().zip(o2) {
                step(i0, o0);
                step(i1, o1);
            }
            for (i, o) in ir.iter().zip(or) {
                step(i, o);
            }
            n
        }
    );

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
        f32x4_add, f32x4_eq, f32x4_mul, f32x4_pmax, f32x4_pmin, f32x4_splat, i16x8_add_sat,
        i16x8_q15mulr_sat, i16x8_splat, i32x4_shuffle, u32x4_extend_high_u16x8,
        u32x4_extend_low_u16x8, v128, v128_and,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{WasmSimd128, f32x4, i16x8};

    /// `C32` of one vector of four samples into four pairs (see [`am_c32`]).
    #[inline(always)]
    fn unit(t: WasmSimd128, dv: v128, cv: v128, i: &[f32; 4], o: &mut [f32; 8]) {
        let (lo, hi, zero) = (f32x4_splat(-1.0), f32x4_splat(1.0), f32x4_splat(0.0));
        let x: v128 = f32x4::from_slice(t, i).into();
        let c = v128_and(f32x4_pmin(f32x4_pmax(x, lo), hi), f32x4_eq(x, x));
        let re = f32x4_mul(cv, f32x4_add(hi, f32x4_mul(dv, c)));
        let (o0, o1) = o.split_at_mut(4);
        f32x4::simd_from(t, i32x4_shuffle::<0, 4, 1, 5>(re, zero)).store_slice(o0);
        f32x4::simd_from(t, i32x4_shuffle::<2, 6, 3, 7>(re, zero)).store_slice(o1);
    }

    /// Four units written out in a function of its own: in a loop LLVM computes every address
    /// with an `i32.add`, here the addresses are constant offsets from the parameters and fold
    /// into the `offset` immediates of the loads and stores (as the blocks of `convert`).
    #[inline(never)]
    fn block(t: WasmSimd128, dv: v128, cv: v128, i: &[[f32; 4]; 4], o: &mut [[f32; 8]; 4]) {
        unit(t, dv, cv, &i[0], &mut o[0]);
        unit(t, dv, cv, &i[1], &mut o[1]);
        unit(t, dv, cv, &i[2], &mut o[2]);
        unit(t, dv, cv, &i[3], &mut o[3]);
    }

    fearless_simd::kernel!(
        /// `C32` on whole blocks of sixteen samples ([`block`]), then whole vectors: `f32x4.eq`
        /// of `x` with itself (the mask of the fallback), `f32x4.pmax` / `f32x4.pmin` with -1 and
        /// 1 (the clamp on every value that is not NaN; the mask takes NaN to `+0`),
        /// `v128.and`, `depth x'`, `1 + t`, `carrier u`, and two `i32x4.shuffle` with zero for
        /// the pairs `(re, 0)`. The operations of the fallback in its order, so the same bits.
        /// Returns the number of samples done.
        pub(super) fn am_c32(
            t: WasmSimd128,
            carrier: f32,
            depth: f32,
            input: &[f32],
            out: &mut [[f32; 2]],
        ) -> usize {
            let (dv, cv) = (f32x4_splat(depth), f32x4_splat(carrier));
            let (ib, _) = input.as_chunks::<4>();
            let (ob, _) = out.as_flattened_mut().as_chunks_mut::<8>();
            let n = ib.len().min(ob.len());
            let (ib, _) = ib.split_at(n);
            let (ob, _) = ob.split_at_mut(n);
            let (ib, ir) = ib.as_chunks::<4>();
            let (ob, or) = ob.as_chunks_mut::<4>();
            for (i, o) in ib.iter().zip(ob) {
                block(t, dv, cv, i, o);
            }
            for (i, o) in ir.iter().zip(or) {
                unit(t, dv, cv, i, o);
            }
            4 * n
        }
    );

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
