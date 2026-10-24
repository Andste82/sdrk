//! `mag_sq`.
//!
//! Both variants are bit-exact with the fallback. The loops split the buffers once into whole
//! blocks of [`UNROLL`] vectors, whole vectors and the scalar tail, and iterate over the chunks
//! of each part by value: a `Zip` of two owned chunk iterators runs on one counter, while a zip
//! of `&mut` iterators tests both ends in every iteration.
//!
//! Level-specific paths (see `level.rs`):
//! - `C32` on NEON adds the squares with `faddp` and stores two registers per `stp`; on wasm32
//!   it runs the generic arithmetic in the loop form of [`steps!`], which folds the load and
//!   store offsets. The x86 levels use the generic code (`haddps` from SSE3 on).
//! - `CQ15` uses the multiply-add of adjacent 16-bit pairs (`pmaddwd`, `i32x4.dot_i16x8_s`) on
//!   x86 and wasm32, and `ld4`/`st2` with the saturating doubling multiply-accumulate
//!   (`sqdmull`/`sqdmlal`, `sqrshrn`) on NEON.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::spectrum::MagSq;
use sdrk_core::{C32, CQ15, Q15, Result};

use crate::Simd;
use crate::level::dispatch;
use crate::util::check_len;

/// Vectors per iteration of the main loops. On wasm32 LLVM no longer unrolls the inner loop of
/// a block of 16 vectors, so the blocks there have 8 (the level-specific paths write out
/// blocks of 32 steps instead, see [`steps!`]).
const UNROLL: usize = if cfg!(target_arch = "wasm32") { 8 } else { 16 };

/// The input (two values per output) and the output of a kernel, cut to the same number of
/// samples and split into the leading whole blocks of `block` samples and the rest.
struct Split<'a, T, U> {
    blocks: (&'a [T], &'a mut [U]),
    rest: (&'a [T], &'a mut [U]),
}

#[inline(always)]
fn split<'a, T, U>(input: &'a [T], out: &'a mut [U], block: usize) -> Split<'a, T, U> {
    let n = out.len().min(input.len() / 2);
    let whole = n - n % block;
    // In bounds: `2 whole <= 2 n <= input.len()` and `whole <= n <= out.len()`.
    let (in_blocks, in_rest) = input.split_at(2 * whole);
    let (out_blocks, out_rest) = out.split_at_mut(whole);
    Split {
        blocks: (in_blocks, out_blocks),
        rest: (in_rest, out_rest),
    }
}

/// The loops of the generic code: `$body` with `$i: &[_]` (`2 l` values) and `$o: &mut [_]`
/// (`l` values) over the whole blocks of [`UNROLL`] vectors of `l = $l` samples, then over the
/// remaining whole vectors. Evaluates to the rest for the scalar tail.
macro_rules! vectors {
    ($input:expr, $out:expr, $l:expr, |$i:ident, $o:ident| $body:block) => {{
        let l = $l;
        let Split { blocks: (bi, bo), rest } = split($input, $out, UNROLL * l);
        for (i, o) in bi.chunks_exact(2 * UNROLL * l).zip(bo.chunks_exact_mut(UNROLL * l)) {
            for ($i, $o) in i.chunks_exact(2 * l).zip(o.chunks_exact_mut(l)) $body
        }
        let Split { blocks: (vi, vo), rest } = split(rest.0, rest.1, l);
        for ($i, $o) in vi.chunks_exact(2 * l).zip(vo.chunks_exact_mut(l)) $body
        rest
    }};
}

/// The loops of a level-specific path: `$body` with `$i: &[_; 2 V]` and `$o: &mut [_; V]` over
/// the whole blocks of `$unroll` steps of `V = $v` samples, then over the remaining whole steps.
/// Evaluates to the number of samples done. A macro, not a function with a closure: the body
/// must be compiled inside the `#[target_feature]` function of the path.
///
/// On wasm32 the block loop counts the input blocks with a third iterator that touches no
/// memory, and takes both the input and the output block from their own iterators. LLVM
/// (also with LTO) turns the pointer that controls a loop into its induction variable, after
/// which the offsets from that pointer are no longer folded; with a separate counter, the
/// constant offsets within both blocks go into the `offset=` immediates of the loads and the
/// stores, while with a `zip` every address costs an `i32.add` and its constant. Elsewhere the
/// `zip` is the shorter loop. LLVM does not unroll an inner loop of more than 8 steps there, so
/// the block of 32 steps is written out, as 32 calls of the body as a closure (which keeps the
/// stack frame of the unoptimized function small). Only the level paths use it.
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! steps {
    ($input:expr, $out:expr, $v:literal, $unroll:expr, |$i:ident, $o:ident| $body:block) => {{
        let Split { blocks: (bi, bo), rest } = split($input, $out, $unroll * $v);
        let mut done = bo.len();
        let (bi, _) = bi.as_chunks::<{ 2 * $unroll * $v }>();
        let (bo, _) = bo.as_chunks_mut::<{ $unroll * $v }>();
        #[cfg(not(target_arch = "wasm32"))]
        for (i, o) in bi.iter().zip(bo) {
            let (i, _) = i.as_chunks::<{ 2 * $v }>();
            let (o, _) = o.as_chunks_mut::<$v>();
            for ($i, $o) in i.iter().zip(o) $body
        }
        #[cfg(target_arch = "wasm32")]
        {
            let step = |$i: &[_; 2 * $v], $o: &mut [_; $v]| $body;
            // `step` for the steps `$k..$k + $n` of the block `($bi, $bo)`, written out.
            macro_rules! unrolled {
                (1, $k:expr, $bi:ident, $bo:ident) => {
                    step(&$bi[$k], &mut $bo[$k])
                };
                (2, $k:expr, $bi:ident, $bo:ident) => {{
                    unrolled!(1, $k, $bi, $bo);
                    unrolled!(1, $k + 1, $bi, $bo);
                }};
                (4, $k:expr, $bi:ident, $bo:ident) => {{
                    unrolled!(2, $k, $bi, $bo);
                    unrolled!(2, $k + 2, $bi, $bo);
                }};
                (8, $k:expr, $bi:ident, $bo:ident) => {{
                    unrolled!(4, $k, $bi, $bo);
                    unrolled!(4, $k + 4, $bi, $bo);
                }};
                (16, $k:expr, $bi:ident, $bo:ident) => {{
                    unrolled!(8, $k, $bi, $bo);
                    unrolled!(8, $k + 8, $bi, $bo);
                }};
                (32, $k:expr, $bi:ident, $bo:ident) => {{
                    unrolled!(16, $k, $bi, $bo);
                    unrolled!(16, $k + 16, $bi, $bo);
                }};
            }
            const { assert!($unroll == 32, "the wasm32 block is unrolled 32 times") };
            let mut ii = bi.iter();
            let mut oo = bo.iter_mut();
            for _ in bi {
                let (Some(i), Some(o)) = (ii.next(), oo.next()) else {
                    break;
                };
                let (i, _) = i.as_chunks::<{ 2 * $v }>();
                let (o, _) = o.as_chunks_mut::<$v>();
                let (Ok(i), Ok(o)) = (<&[_; $unroll]>::try_from(i), <&mut [_; $unroll]>::try_from(o))
                else {
                    break;
                };
                unrolled!(32, 0, i, o);
            }
        }
        let Split { blocks: (vi, vo), rest: _ } = split(rest.0, rest.1, $v);
        done += vo.len();
        let (vi, _) = vi.as_chunks::<{ 2 * $v }>();
        let (vo, _) = vo.as_chunks_mut::<$v>();
        for ($i, $o) in vi.iter().zip(vo) $body
        done
    }};
}

/// `re*re + im*im` of one vector of `2 l` interleaved floats into `l` outputs.
#[inline(always)]
fn mag_sq_vector<S: SimdToken>(simd: S, i: &[f32], o: &mut [f32]) {
    let l = S::f32s::LEN;
    let (a, b) = i.split_at(l);
    let a = S::f32s::from_slice(simd, a);
    let b = S::f32s::from_slice(simd, b);
    // Square first and unzip the squares: the lane shuffle of the two unzips is then the same,
    // and the compiler applies it once to the sum instead of to every operand (SSE3 `haddps`,
    // NEON `faddp`).
    let (sa, sb) = (a * a, b * b);
    (sa.unzip_low(sb) + sa.unzip_high(sb)).store_slice(o);
}

/// `re*re + im*im` of `input` (interleaved floats) into `out`, bit-exact with the fallback.
#[inline(always)]
fn mag_sq_f32<S: SimdToken>(simd: S, input: &[f32], out: &mut [f32]) {
    let done = mag_sq_f32_level(simd, input, out);
    let (Some(input), Some(out)) = (input.get(2 * done..), out.get_mut(done..)) else {
        return;
    };
    let (input, out) = vectors!(input, out, S::f32s::LEN, |i, o| {
        mag_sq_vector(simd, i, o);
    });
    for (o, x) in out.iter_mut().zip(input.chunks_exact(2)) {
        if let [re, im] = *x {
            *o = re * re + im * im;
        }
    }
}

/// The level-specific path of `C32` for the current level, if there is one: returns the number
/// of leading samples done.
#[inline(always)]
fn mag_sq_f32_level<S: SimdToken>(simd: S, input: &[f32], out: &mut [f32]) -> usize {
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = simd.level().as_neon() {
        return neon::mag_sq_f32(t, input, out);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = simd.level().as_wasm_simd128() {
        return wasm::mag_sq_f32(t, input, out);
    }
    let _ = (simd, input, out);
    0
}

impl MagSq<C32> for Simd {
    fn mag_sq(&self, input: &[C32], output: &mut [f32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let input: &[f32] = bytemuck::cast_slice(input);
        dispatch!(simd => mag_sq_f32(simd, input, output));
        Ok(())
    }
}

/// `(re*re + im*im + 2^14) >> 15` of one vector of Q15 samples, at most 32767. The sum of two
/// squares is at most `2^31`, so it fits an unsigned 32-bit lane.
#[inline(always)]
fn squares_q15<S: SimdToken>(simd: S, s: &[i16]) -> S::u32s {
    let x: S::i32s = S::i16s::from_slice(simd, s).bitcast();
    let re = (x << 16u32) >> 16u32;
    let im = x >> 16u32;
    let sum: S::u32s = (re * re + im * im).bitcast();
    ((sum + (1u32 << 14)) >> 15u32).min(S::u32s::splat(simd, 32767))
}

/// `mag_sq` of one vector of `n` Q15 samples (`2 n` values) into `n` outputs.
#[inline(always)]
fn mag_sq_q15_vector<S: SimdToken>(simd: S, i: &[i16], o: &mut [i16]) {
    let (a, b) = i.split_at(S::i16s::LEN);
    let packed: S::u16s = squares_q15::<S>(simd, a).narrow(squares_q15::<S>(simd, b));
    packed.bitcast::<S::i16s>().store_slice(o);
}

/// `sat16((re*re + im*im + 2^14) >> 15)` for Q15 samples, bit-exact with the fallback.
#[inline(always)]
fn mag_sq_q15<S: SimdToken>(simd: S, input: &[i16], out: &mut [i16]) {
    let done = mag_sq_q15_level(simd, input, out);
    let (Some(input), Some(out)) = (input.get(2 * done..), out.get_mut(done..)) else {
        return;
    };
    let (input, out) = vectors!(input, out, S::i16s::LEN, |i, o| {
        mag_sq_q15_vector(simd, i, o);
    });
    for (o, x) in out.iter_mut().zip(input.chunks_exact(2)) {
        if let [re, im] = *x {
            let sum =
                (i64::from(re) * i64::from(re) + i64::from(im) * i64::from(im) + (1 << 14)) >> 15;
            *o = sum.min(32767) as i16;
        }
    }
}

/// The level-specific path of `CQ15` for the current level, if there is one: returns the number
/// of leading samples done.
#[inline(always)]
fn mag_sq_q15_level<S: SimdToken>(simd: S, input: &[i16], out: &mut [i16]) -> usize {
    let level = simd.level();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if let Some(t) = level.as_avx512() {
            return x86::mag_sq_q15_avx512(t, input, out);
        }
        if let Some(t) = level.as_avx2() {
            return x86::mag_sq_q15_avx2(t, input, out);
        }
        if let Some(t) = level.as_sse4_2() {
            return x86::mag_sq_q15_sse4_2(t, input, out);
        }
        if let Some(t) = crate::level::sse2_only(level) {
            return x86::mag_sq_q15_sse2(t, input, out);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return neon::mag_sq_q15(t, input, out);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return wasm::mag_sq_q15(t, input, out);
    }
    let _ = (level, input, out);
    0
}

impl MagSq<CQ15> for Simd {
    fn mag_sq(&self, input: &[CQ15], output: &mut [Q15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let x: &[i16] = bytemuck::cast_slice(input);
        let y: &mut [i16] = bytemuck::cast_slice_mut(output);
        dispatch!(simd => mag_sq_q15(simd, x, y));
        Ok(())
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    //! `CQ15` with `pmaddwd`: `re*re + im*im` of a sample is one lane of the multiply-add of
    //! adjacent pairs of `x * x`. The only sum that does not fit `i32` is `2^31` (both parts
    //! `-32768`), which `pmaddwd` returns as `0x8000_0000`, the bits of the unsigned sum; so the
    //! rounding and the shift run on unsigned lanes, and the signed saturating narrowing
    //! (`packssdw`, `vpmovsdw`) applies the limit 32767.

    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m128i, __m256i, __m512i, _mm_add_epi32, _mm_madd_epi16, _mm_packs_epi32, _mm_set1_epi32,
        _mm_srli_epi32, _mm256_add_epi32, _mm256_madd_epi16, _mm256_packs_epi32,
        _mm256_permute4x64_epi64, _mm256_set1_epi32, _mm256_srli_epi32, _mm512_add_epi32,
        _mm512_cvtsepi32_epi16, _mm512_madd_epi16, _mm512_set1_epi32, _mm512_srli_epi32,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128i, __m256i, __m512i, _mm_add_epi32, _mm_madd_epi16, _mm_packs_epi32, _mm_set1_epi32,
        _mm_srli_epi32, _mm256_add_epi32, _mm256_madd_epi16, _mm256_packs_epi32,
        _mm256_permute4x64_epi64, _mm256_set1_epi32, _mm256_srli_epi32, _mm512_add_epi32,
        _mm512_cvtsepi32_epi16, _mm512_madd_epi16, _mm512_set1_epi32, _mm512_srli_epi32,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{i16x8, i16x16, i16x32};

    use super::{Split, UNROLL, split};

    /// The rounding term `2^14`.
    const HALF: i32 = 1 << 14;

    /// The 128-bit path, for a token of SSE2 or of SSE4.2 (which adds nothing it uses).
    macro_rules! sse {
        ($name:ident, $token:ident) => {
            fearless_simd::kernel!(
                pub(super) fn $name(t: $token, input: &[i16], out: &mut [i16]) -> usize {
                    macro_rules! sums {
                        ($x:expr) => {{
                            let x: __m128i = i16x8::from_slice(t, $x).into();
                            let sum = _mm_madd_epi16(x, x);
                            _mm_srli_epi32::<15>(_mm_add_epi32(sum, _mm_set1_epi32(HALF)))
                        }};
                    }
                    steps!(input, out, 8, UNROLL, |i, o| {
                        let (a, b) = i.split_at(8);
                        i16x8::simd_from(t, _mm_packs_epi32(sums!(a), sums!(b))).store_array(o);
                    })
                }
            );
        };
    }
    sse!(mag_sq_q15_sse2, Sse2);
    sse!(mag_sq_q15_sse4_2, Sse4_2);

    fearless_simd::kernel!(
        pub(super) fn mag_sq_q15_avx2(t: Avx2, input: &[i16], out: &mut [i16]) -> usize {
            macro_rules! sums {
                ($x:expr) => {{
                    let x: __m256i = i16x16::from_slice(t, $x).into();
                    let sum = _mm256_madd_epi16(x, x);
                    _mm256_srli_epi32::<15>(_mm256_add_epi32(sum, _mm256_set1_epi32(HALF)))
                }};
            }
            steps!(input, out, 16, UNROLL, |i, o| {
                let (a, b) = i.split_at(16);
                // `vpackssdw` packs within the 128-bit halves; `vpermq` restores the order.
                let packed = _mm256_packs_epi32(sums!(a), sums!(b));
                let packed = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);
                i16x16::simd_from(t, packed).store_array(o);
            })
        }
    );

    fearless_simd::kernel!(
        pub(super) fn mag_sq_q15_avx512(t: Avx512, input: &[i16], out: &mut [i16]) -> usize {
            // One vector per step: `vpmovsdw` narrows a single vector, so no pairs are needed.
            steps!(input, out, 16, UNROLL, |i, o| {
                let x: __m512i = i16x32::load_array_ref(t, i).into();
                let sum = _mm512_madd_epi16(x, x);
                let sum = _mm512_srli_epi32::<15>(_mm512_add_epi32(sum, _mm512_set1_epi32(HALF)));
                i16x16::simd_from(t, _mm512_cvtsepi32_epi16(sum)).store_array(o);
            })
        }
    );
}

#[cfg(target_arch = "aarch64")]
mod neon {
    //! `C32`: the squares of a register of interleaved samples are added in adjacent pairs
    //! with `faddp` (`re*re + im*im`, the operations of the fallback, no fused multiply-add),
    //! and the results of two registers are stored as one pair (`stp`).
    //!
    //! `CQ15`: `ld4` loads 16 samples split into the real and the imaginary parts of the even
    //! and of the odd samples, and `st2` stores the even and the odd results interleaved again,
    //! so a step of 16 samples takes two memory instructions. `sqdmull`/`sqdmlal` form
    //! `2 (re*re + im*im)` with saturation at `2^31 - 1`, and the rounding saturating narrowing
    //! `sqrshrn #16` gives `(re*re + im*im + 2^14) >> 15` limited to 32767: the doubled sum is
    //! exact below `2^31`, and where it saturates the result is above 32767 as well.
    #![allow(unsafe_code)]

    use core::arch::aarch64::{
        float32x4x2_t, float32x4x4_t, int16x8_t, int16x8x2_t, int16x8x4_t, vget_low_s16, vld4q_s16,
        vmulq_f32, vpaddq_f32, vqdmlal_high_s16, vqdmlal_s16, vqdmull_high_s16, vqdmull_s16,
        vqrshrn_high_n_s32, vqrshrn_n_s32, vst2q_s16,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{f32x8, f32x16};

    use super::{Split, UNROLL, split};

    /// The 32 values of `x` (16 samples) split four ways: the real parts of the even samples,
    /// their imaginary parts, then the same of the odd samples.
    #[inline(always)]
    fn load4(x: &[i16; 32]) -> int16x8x4_t {
        // SAFETY: `vld4q_s16` reads 32 consecutive `i16` with no alignment requirement beyond
        // that of `i16`, and `x` holds exactly 32. NEON is part of the AArch64 baseline.
        unsafe { vld4q_s16(x.as_ptr()) }
    }

    /// Stores the results of the even and of the odd samples interleaved into `o`.
    #[inline(always)]
    fn store2(o: &mut [i16; 16], even: int16x8_t, odd: int16x8_t) {
        // SAFETY: `vst2q_s16` writes 16 consecutive `i16` with no alignment requirement beyond
        // that of `i16`, and `o` holds exactly 16 and is borrowed exclusively.
        unsafe { vst2q_s16(o.as_mut_ptr(), int16x8x2_t(even, odd)) }
    }

    fearless_simd::kernel!(
        pub(super) fn mag_sq_f32(t: Neon, input: &[f32], out: &mut [f32]) -> usize {
            steps!(input, out, 8, UNROLL, |i, o| {
                let x: float32x4x4_t = f32x16::load_array_ref(t, i).into();
                let (a, b) = (vmulq_f32(x.0, x.0), vmulq_f32(x.1, x.1));
                let (c, d) = (vmulq_f32(x.2, x.2), vmulq_f32(x.3, x.3));
                let sums = float32x4x2_t(vpaddq_f32(a, b), vpaddq_f32(c, d));
                f32x8::simd_from(t, sums).store_array(o);
            })
        }
    );

    fearless_simd::kernel!(
        pub(super) fn mag_sq_q15(t: Neon, input: &[i16], out: &mut [i16]) -> usize {
            macro_rules! mag {
                ($re:expr, $im:expr) => {{
                    let (re, im) = ($re, $im);
                    let lo = vqdmull_s16(vget_low_s16(re), vget_low_s16(re));
                    let lo = vqdmlal_s16(lo, vget_low_s16(im), vget_low_s16(im));
                    let hi = vqdmlal_high_s16(vqdmull_high_s16(re, re), im, im);
                    vqrshrn_high_n_s32::<16>(vqrshrn_n_s32::<16>(lo), hi)
                }};
            }
            // One step per iteration: the loads and stores then advance their pointers
            // themselves (post-index), which `ld4`/`st2` need for want of an offset form.
            steps!(input, out, 16, 1, |i, o| {
                let x = load4(i);
                store2(o, mag!(x.0, x.1), mag!(x.2, x.3));
            })
        }
    );
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    //! `C32`: the generic arithmetic on array chunks (see [`steps!`]). `CQ15` with
    //! `i32x4.dot_i16x8_s`, the multiply-add of adjacent 16-bit pairs. Below `2^31` the sum of
    //! two squares is exact, the rounded shift `(sum + 2^14) >> 15` is at most 65534, and the
    //! signed saturating narrowing applies the limit 32767. The one sum of `2^31` (both parts
    //! `-32768`) wraps to `i32::MIN` and becomes -32768; as the narrowed lanes are otherwise in
    //! `0..=32767`, an unsigned minimum with 32767 corrects it. (With a logical shift, LLVM
    //! knows the lanes are not negative and replaces the saturating narrowing by a minimum and
    //! an unsigned narrowing for each input vector, two more instructions per vector.)

    use core::arch::wasm32::{
        i16x8_narrow_i32x4, i32x4_add, i32x4_dot_i16x8, i32x4_shr, i32x4_splat, u16x8_min,
        u16x8_splat, v128,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4, i16x8};

    use super::{Split, split};

    /// Steps per block: [`steps!`] writes the blocks out on wasm32, which pays off up to 32
    /// steps (64 gain about 1% more for twice the code and the longer tails).
    pub(super) const UNROLL: usize = 32;

    fearless_simd::kernel!(
        pub(super) fn mag_sq_f32(t: WasmSimd128, input: &[f32], out: &mut [f32]) -> usize {
            steps!(input, out, 4, UNROLL, |i, o| {
                let (a, b) = i.split_at(4);
                let a = f32x4::from_slice(t, a);
                let b = f32x4::from_slice(t, b);
                let (sa, sb) = (a * a, b * b);
                (sa.unzip_low(sb) + sa.unzip_high(sb)).store_array(o);
            })
        }
    );

    fearless_simd::kernel!(
        pub(super) fn mag_sq_q15(t: WasmSimd128, input: &[i16], out: &mut [i16]) -> usize {
            macro_rules! sums {
                ($x:expr) => {{
                    let x: v128 = i16x8::from_slice(t, $x).into();
                    i32x4_shr(i32x4_add(i32x4_dot_i16x8(x, x), i32x4_splat(1 << 14)), 15)
                }};
            }
            let limit = u16x8_splat(32767);
            steps!(input, out, 8, UNROLL, |i, o| {
                let (a, b) = i.split_at(8);
                let packed = u16x8_min(i16x8_narrow_i32x4(sums!(a), sums!(b)), limit);
                i16x8::simd_from(t, packed).store_array(o);
            })
        }
    );
}

#[cfg(test)]
mod tests {
    use super::{mag_sq_f32_level, mag_sq_q15_level};
    use crate::level::{dispatch, name};

    /// The samples per step of the level-specific paths of `C32` and `CQ15` at `level`, `0` where
    /// the level has none.
    fn steps(level: &str) -> (usize, usize) {
        match level {
            "avx512" | "avx2" => (0, 16),
            "sse4.2" | "sse2" => (0, 8),
            "neon" => (8, 16),
            "simd128" => (4, 8),
            _ => (0, 0),
        }
    }

    #[test]
    fn the_path_of_the_current_level_runs_on_all_whole_steps() {
        let (f32_step, q15_step) = steps(name());
        let input_f32 = [0.5f32; 2 * 300];
        let input_q15 = [1000i16; 2 * 300];
        for n in 0..=300 {
            let (mut out_f32, mut out_q15) = ([0f32; 300], [0i16; 300]);
            let ((xf, _), (yf, _)) = (input_f32.split_at(2 * n), out_f32.split_at_mut(n));
            let ((xq, _), (yq, _)) = (input_q15.split_at(2 * n), out_q15.split_at_mut(n));
            let done = dispatch!(simd => mag_sq_f32_level(simd, xf, yf));
            assert_eq!(
                done,
                n.checked_rem(f32_step).map_or(0, |r| n - r),
                "C32, n {n}"
            );
            assert!(yf.iter().take(done).all(|&v| v == 0.5), "C32, n {n}");
            let done = dispatch!(simd => mag_sq_q15_level(simd, xq, yq));
            assert_eq!(
                done,
                n.checked_rem(q15_step).map_or(0, |r| n - r),
                "CQ15, n {n}"
            );
            assert!(yq.iter().take(done).all(|&v| v == 61), "CQ15, n {n}");
        }
    }
}
