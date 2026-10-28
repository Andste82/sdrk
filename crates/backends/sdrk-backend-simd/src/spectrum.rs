//! `mag_sq`, `mag`, `window` and `power_db`.
//!
//! All variants are bit-exact with the fallback. The loops split the buffers once into whole
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
//!
//! `mag` is the same loop with a vector square root. `C32` runs the generic code on every level
//! (`sqrt` is correctly rounded, like `libm::sqrtf` of the fallback). `CQ15` rounds the integer
//! square root of the sum of squares: a single-precision estimate within one of the result and a
//! correction step in each direction (`root_q15` for the idea and the bound); every level but
//! the generic one has a path of its own that takes the products from `pmaddwd`, `smlal` or
//! `i32x4.dot_i16x8_s` (module `x86`, `neon`, `wasm`).
//!
//! `window` multiplies each component by the weight of its sample, the weights duplicated for
//! both parts by a shuffle. `C32` runs the generic code on every level but NEON, where `ld2`/`st2`
//! split and interleave the parts instead. `CQ15` is `q15_mul`:
//! `pmulhrsw` with a fix of the one overflow at SSE4.2, AVX2 and AVX-512, `pmulhw`/`pmullw` at
//! SSE2, `sqrdmulh` on NEON and `i16x8.q15mulr_sat_s` on wasm32 (loop form [`window_steps!`]).
//!
//! `power_db` is `libm::log10f` step by step on whole vectors ([`power_db_vector`]), so it is
//! bit-exact with the fallback as well; the generic code on every level.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::spectrum::{Mag, MagSq, PowerDb, Window};
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

/// The loop of a level-specific `window` path: `$body` with `$i: &[_; 2 V]`, `$o: &mut [_; 2 V]`
/// (the interleaved values of `V = $v` samples) and `$w: &[_; V]` (their weights) over the whole
/// steps, in blocks of [`WINDOW_UNROLL`] steps (an inner loop of constant length, which the
/// compiler writes out in the run-time-dispatched build as in the static ones) and then one
/// step at a time. Evaluates to the number of samples done.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! window_steps {
    ($input:expr, $out:expr, $w:expr, $v:literal, |$i:ident, $o:ident, $wv:ident| $body:block) => {{
        // A closure, not inline code: two expansions of the body would exceed the stack frame
        // the lint allows. It inherits the target features of the function and is inlined.
        let step = |$i: &[_; 2 * $v], $o: &mut [_; 2 * $v], $wv: &[_; $v]| $body;
        let n = $out.len().min($input.len()).min(2 * $w.len()) / (2 * $v);
        let (input, _) = $input.split_at(2 * $v * n);
        let (out, _) = $out.split_at_mut(2 * $v * n);
        let (w, _) = $w.split_at($v * n);
        let (bi, input) = input.as_chunks::<{ 2 * $v * $crate::spectrum::WINDOW_UNROLL }>();
        let (bo, out) = out.as_chunks_mut::<{ 2 * $v * $crate::spectrum::WINDOW_UNROLL }>();
        let (bw, w) = w.as_chunks::<{ $v * $crate::spectrum::WINDOW_UNROLL }>();
        for ((i, o), w) in bi.iter().zip(bo).zip(bw) {
            let (i, _) = i.as_chunks::<{ 2 * $v }>();
            let (o, _) = o.as_chunks_mut::<{ 2 * $v }>();
            let (w, _) = w.as_chunks::<$v>();
            for ((i, o), w) in i.iter().zip(o).zip(w) {
                step(i, o, w);
            }
        }
        let (input, _) = input.as_chunks::<{ 2 * $v }>();
        let (out, _) = out.as_chunks_mut::<{ 2 * $v }>();
        let (w, _) = w.as_chunks::<$v>();
        for ((i, o), w) in input.iter().zip(out).zip(w) {
            step(i, o, w);
        }
        $v * n
    }};
}

/// Steps per block of the `window` loops.
pub(crate) const WINDOW_UNROLL: usize = 4;

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
    #[cfg(target_arch = "x86_64")]
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

/// `sqrt(re*re + im*im)` of one vector of `2 l` interleaved floats into `l` outputs. The square
/// root of IEEE 754 is correctly rounded on every level, like `libm::sqrtf` of the fallback, so
/// the result is bit-exact; the sum is the one of [`mag_sq_vector`] (no fused multiply-add).
#[inline(always)]
fn mag_vector<S: SimdToken>(simd: S, i: &[f32], o: &mut [f32]) {
    let l = S::f32s::LEN;
    let (a, b) = i.split_at(l);
    let a = S::f32s::from_slice(simd, a);
    let b = S::f32s::from_slice(simd, b);
    let (sa, sb) = (a * a, b * b);
    (sa.unzip_low(sb) + sa.unzip_high(sb)).sqrt().store_slice(o);
}

/// `sqrt(re*re + im*im)` of `input` (interleaved floats) into `out`, bit-exact with the
/// fallback.
#[inline(always)]
fn mag_f32<S: SimdToken>(simd: S, input: &[f32], out: &mut [f32]) {
    let (input, out) = vectors!(input, out, S::f32s::LEN, |i, o| {
        mag_vector(simd, i, o);
    });
    for (o, x) in out.iter_mut().zip(input.chunks_exact(2)) {
        if let [re, im] = *x {
            *o = sdrk_core::num::sqrt_f32(re * re + im * im);
        }
    }
}

impl Mag<C32> for Simd {
    fn mag(&self, input: &[C32], output: &mut [f32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let input: &[f32] = bytemuck::cast_slice(input);
        dispatch!(simd => mag_f32(simd, input, output));
        Ok(())
    }
}

/// The rounded integer square root of the sum of squares of one vector of Q15 samples, at most
/// 32767: the smallest `y` with `y (y + 1) >= re*re + im*im`, which is `r + 1` exactly where
/// `s - r*r > r` for the floor root `r` (the rule of the fallback), because `y (y + 1) >= s`
/// holds from `r + 1` on exactly when `s > r (r + 1)`.
///
/// The sum is limited to `2^30` first (the root of anything above it saturates anyway), which
/// makes it a non-negative `i32` and `y (y + 1)` fit `i32`. The single-precision root of the
/// (rounded) sum plus one half, truncated, is within one of `y`: the error of the two roundings
/// and of the addition is below 0.01 where the root is below 32768, and the estimate differs
/// from the exact root plus one half by less than that, so it can only miss when that value is
/// within 0.01 of an integer. The two comparisons step the estimate to `y`. The exhaustive test
/// over all 2^32 raw pairs (`tests/spectrum_mag.rs`) confirms the bound.
#[inline(always)]
fn root_q15<S: SimdToken>(simd: S, s: &[i16]) -> S::i32s {
    let x: S::i32s = S::i16s::from_slice(simd, s).bitcast();
    let re = (x << 16u32) >> 16u32;
    let im = x >> 16u32;
    // `2^31` (both parts `-32768`) is `i32::MIN` as a signed lane: the unsigned minimum fixes it.
    let sum: S::u32s = (re * re + im * im).bitcast();
    let sum: S::i32s = sum.min(S::u32s::splat(simd, 1 << 30)).bitcast();
    let y = (sum.to_float::<S::f32s>().sqrt() + 0.5).to_int::<S::i32s>();
    // Too small where `y (y + 1) < s`, too large where `(y - 1) y >= s`.
    let up = (y * (y + 1)).simd_lt(sum).to_vector();
    let down = ((y - 1) * y).simd_ge(sum).to_vector();
    // The masks are -1 where true. `y - 1` for `s = 0` is -1 and `down` fires: the maximum
    // restores 0.
    (y - up + down)
        .max(S::i32s::splat(simd, 0))
        .min(S::i32s::splat(simd, 32767))
}

/// `mag` of one vector of `n` Q15 samples (`2 n` values) into `n` outputs.
#[inline(always)]
fn mag_q15_vector<S: SimdToken>(simd: S, i: &[i16], o: &mut [i16]) {
    let (a, b) = i.split_at(S::i16s::LEN);
    let packed: S::i16s = root_q15::<S>(simd, a).narrow(root_q15::<S>(simd, b));
    packed.store_slice(o);
}

/// The rounded root of the sum of squares for Q15 samples, bit-exact with the fallback.
#[inline(always)]
fn mag_q15<S: SimdToken>(simd: S, input: &[i16], out: &mut [i16]) {
    let done = mag_q15_level(simd, input, out);
    let (Some(input), Some(out)) = (input.get(2 * done..), out.get_mut(done..)) else {
        return;
    };
    let (input, out) = vectors!(input, out, S::i16s::LEN, |i, o| {
        mag_q15_vector(simd, i, o);
    });
    for (o, x) in out.iter_mut().zip(input.chunks_exact(2)) {
        if let [re, im] = *x {
            let s = (i64::from(re) * i64::from(re) + i64::from(im) * i64::from(im)) as u64;
            let r = sdrk_core::num::isqrt_u64(s);
            *o = (r + u64::from(s - r * r > r)).min(32767) as i16;
        }
    }
}

/// The level-specific path of `mag` `CQ15` for the current level, if there is one: returns the
/// number of leading samples done.
#[inline(always)]
fn mag_q15_level<S: SimdToken>(simd: S, input: &[i16], out: &mut [i16]) -> usize {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx512() {
            return x86::mag_q15_avx512(t, input, out);
        }
        if let Some(t) = level.as_avx2() {
            return x86::mag_q15_avx2(t, input, out);
        }
        if let Some(t) = level.as_sse4_2() {
            return x86::mag_q15_sse4_2(t, input, out);
        }
        if let Some(t) = crate::level::sse2_only(level) {
            return x86::mag_q15_sse2(t, input, out);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return neon::mag_q15(t, input, out);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return wasm::mag_q15(t, input, out);
    }
    let _ = (level, input, out);
    0
}

impl Mag<CQ15> for Simd {
    fn mag(&self, input: &[CQ15], output: &mut [Q15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let x: &[i16] = bytemuck::cast_slice(input);
        let y: &mut [i16] = bytemuck::cast_slice_mut(output);
        dispatch!(simd => mag_q15(simd, x, y));
        Ok(())
    }
}

/// `x * w` per component of `input` (interleaved floats) with one weight per sample.
#[inline(always)]
fn window_f32<S: SimdToken>(simd: S, input: &[f32], out: &mut [f32], w: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    let (input, out, w) = {
        let done = match simd.level().as_neon() {
            Some(t) => neon::window_f32(t, input, out, w),
            None => 0,
        };
        let (Some(input), Some(out), Some(w)) = (
            input.get(2 * done..),
            out.get_mut(2 * done..),
            w.get(done..),
        ) else {
            return;
        };
        (input, out, w)
    };
    // Blocks of `WINDOW_UNROLL` vectors of weights (an inner loop of constant length, which the
    // compiler writes out on every level), then the scalar tail, which the compiler vectorizes.
    // A second vector loop for the rest would double the body and exceed the stack frame the
    // lint allows.
    let l = S::f32s::LEN;
    let block = WINDOW_UNROLL * l;
    let n = out.len().min(input.len()).min(2 * w.len()) / (2 * block);
    let (vi, ri) = input.split_at(2 * block * n);
    let (vo, ro) = out.split_at_mut(2 * block * n);
    let (vw, rw) = w.split_at(block * n);
    for ((i, o), w) in vi
        .chunks_exact(2 * block)
        .zip(vo.chunks_exact_mut(2 * block))
        .zip(vw.chunks_exact(block))
    {
        for ((i, o), w) in i
            .chunks_exact(2 * l)
            .zip(o.chunks_exact_mut(2 * l))
            .zip(w.chunks_exact(l))
        {
            let w = S::f32s::from_slice(simd, w);
            // `[w0, w0, w1, w1, ...]` and the upper half of the same.
            let (wa, wb) = w.interleave(w);
            let (i0, i1) = i.split_at(l);
            let (o0, o1) = o.split_at_mut(l);
            (S::f32s::from_slice(simd, i0) * wa).store_slice(o0);
            (S::f32s::from_slice(simd, i1) * wb).store_slice(o1);
        }
    }
    for ((o, x), &w) in ro
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(ri.as_chunks::<2>().0)
        .zip(rw)
    {
        let ([o0, o1], [re, im]) = (o, *x);
        (*o0, *o1) = (re * w, im * w);
    }
}

impl Window<C32> for Simd {
    fn window(&self, input: &[C32], output: &mut [C32], w: &[f32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        check_len(input.len(), w.len())?;
        let input: &[f32] = bytemuck::cast_slice(input);
        let output: &mut [f32] = bytemuck::cast_slice_mut(output);
        dispatch!(simd => window_f32(simd, input, output, w));
        Ok(())
    }
}

/// `q15_mul(x, w) = sat16((x w + 2^14) >> 15)` of one vector of values and one of weights.
#[inline(always)]
fn q15_mul_vector<S: SimdToken>(simd: S, x: S::i16s, w: S::i16s) -> S::i16s {
    let (x0, x1) = x.widen();
    let (w0, w1) = w.widen();
    let half = S::i32s::splat(simd, 1 << 14);
    let p0 = (x0 * w0 + half) >> 15u32;
    let p1 = (x1 * w1 + half) >> 15u32;
    // Only `(-32768)^2` leaves the range: 32768 saturates to 32767.
    p0.saturating_narrow(p1)
}

/// `q15_mul` per component of `input` (interleaved Q15 values) with one weight per sample,
/// bit-exact with the fallback.
#[inline(always)]
fn window_q15<S: SimdToken>(simd: S, input: &[i16], out: &mut [i16], w: &[i16]) {
    let done = window_q15_level(simd, input, out, w);
    let (Some(input), Some(out), Some(w)) = (
        input.get(2 * done..),
        out.get_mut(2 * done..),
        w.get(done..),
    ) else {
        return;
    };
    // The generic code serves the tails of the level paths and the levels without one, so it
    // runs one step at a time.
    let l = S::i16s::LEN;
    let n = out.len().min(input.len()).min(2 * w.len()) / (2 * l);
    let (vi, ri) = input.split_at(2 * l * n);
    let (vo, ro) = out.split_at_mut(2 * l * n);
    let (vw, rw) = w.split_at(l * n);
    for ((i, o), w) in vi
        .chunks_exact(2 * l)
        .zip(vo.chunks_exact_mut(2 * l))
        .zip(vw.chunks_exact(l))
    {
        let w = S::i16s::from_slice(simd, w);
        let (wa, wb) = w.interleave(w);
        let (i0, i1) = i.split_at(l);
        let (o0, o1) = o.split_at_mut(l);
        q15_mul_vector(simd, S::i16s::from_slice(simd, i0), wa).store_slice(o0);
        q15_mul_vector(simd, S::i16s::from_slice(simd, i1), wb).store_slice(o1);
    }
    for ((o, x), &w) in ro
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(ri.as_chunks::<2>().0)
        .zip(rw)
    {
        let ([o0, o1], [re, im]) = (o, *x);
        let mul = |x: i16| ((i32::from(x) * i32::from(w) + (1 << 14)) >> 15).min(32767) as i16;
        (*o0, *o1) = (mul(re), mul(im));
    }
}

/// The level-specific path of `window` `CQ15` for the current level, if there is one: returns
/// the number of leading samples done.
#[inline(always)]
fn window_q15_level<S: SimdToken>(simd: S, input: &[i16], out: &mut [i16], w: &[i16]) -> usize {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx512() {
            return x86::window_q15_avx512(t, input, out, w);
        }
        if let Some(t) = level.as_avx2() {
            return x86::window_q15_avx2(t, input, out, w);
        }
        if let Some(t) = level.as_sse4_2() {
            return x86::window_q15_sse4_2(t, input, out, w);
        }
        if let Some(t) = crate::level::sse2_only(level) {
            return x86::window_q15_sse2(t, input, out, w);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return neon::window_q15(t, input, out, w);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return wasm::window_q15(t, input, out, w);
    }
    let _ = (level, input, out, w);
    0
}

impl Window<CQ15> for Simd {
    fn window(&self, input: &[CQ15], output: &mut [CQ15], w: &[Q15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        check_len(input.len(), w.len())?;
        let input: &[i16] = bytemuck::cast_slice(input);
        let output: &mut [i16] = bytemuck::cast_slice_mut(output);
        let w: &[i16] = bytemuck::cast_slice(w);
        dispatch!(simd => window_q15(simd, input, output, w));
        Ok(())
    }
}

/// Smallest linear power that `power_db` evaluates (the `FLOOR` of the fallback).
const FLOOR: f32 = 1e-20;

/// The scalar `power_db` of the fallback: `10 log10f(max(p, 1e-20)) + offset_db`.
#[inline(always)]
fn to_db(p: f32, offset_db: f32) -> f32 {
    let p = if p > FLOOR { p } else { FLOOR };
    10.0 * libm::log10f(p) + offset_db
}

/// `10 log10(max(p, 1e-20)) + offset` of one vector, bit-exact with the fallback: the steps of
/// `libm::log10f` (FreeBSD `e_log10f.c`) in the same order and without fused multiply-add. After
/// the floor every lane is a normal positive number or `+inf`, so of the special cases of
/// `log10f` only `+inf` remains (`log10f(1) = 0` is also what the general steps give).
#[inline(always)]
fn power_db_vector<S: SimdToken>(simd: S, p: S::f32s, offset: S::f32s) -> S::f32s {
    // The constants of `libm::log10f` (FreeBSD `e_log10f.c`), by their bits.
    const IVLN10HI: f32 = f32::from_bits(0x3ede_6000);
    const IVLN10LO: f32 = f32::from_bits(0xb804_ead9);
    const LOG10_2HI: f32 = f32::from_bits(0x3e9a_2080);
    const LOG10_2LO: f32 = f32::from_bits(0x3554_27db);
    const LG1: f32 = f32::from_bits(0x3f2a_aaaa);
    const LG2: f32 = f32::from_bits(0x3ecc_ce13);
    const LG3: f32 = f32::from_bits(0x3e91_e9ee);
    const LG4: f32 = f32::from_bits(0x3e78_9e26);
    let floor = S::f32s::splat(simd, FLOOR);
    // `p > FLOOR ? p : FLOOR`: NaN takes the floor too. On x86 that is `maxps(p, FLOOR)`, which
    // returns its second operand where either is NaN (the scalar level uses `f32::max`, which
    // returns the operand that is not NaN); elsewhere the maximum may return NaN.
    #[cfg(target_arch = "x86_64")]
    let x = p.max(floor);
    #[cfg(not(target_arch = "x86_64"))]
    let x = p.simd_gt(floor).select(p, floor);
    // Reduce `x` to `2^k m` with `m` in `[sqrt(2)/2, sqrt(2))`.
    let ix = x.bitcast::<S::u32s>() + (0x3f80_0000 - 0x3f35_04f3);
    let k: S::i32s = (ix >> 23u32).bitcast::<S::i32s>() - 0x7f;
    let m: S::f32s = ((ix & 0x007f_ffff) + 0x3f35_04f3).bitcast();
    let f = m - 1.0;
    let s = f / (f + 2.0);
    let z = s * s;
    let w = z * z;
    let t1 = w * (w * LG4 + LG2);
    let t2 = z * (w * LG3 + LG1);
    let r = t2 + t1;
    let hfsq = f * 0.5 * f;
    let hi: S::f32s = ((f - hfsq).bitcast::<S::u32s>() & 0xffff_f000).bitcast();
    let lo = f - hi - hfsq + s * (hfsq + r);
    let dk: S::f32s = k.to_float();
    let y = dk * LOG10_2LO + (lo + hi) * IVLN10LO + lo * IVLN10HI + hi * IVLN10HI + dk * LOG10_2HI;
    let y = x.simd_eq(S::f32s::splat(simd, f32::INFINITY)).select(x, y);
    y * 10.0 + offset
}

/// `power_db` of linear powers in `f32`.
#[inline(always)]
fn power_db_f32<S: SimdToken>(simd: S, input: &[f32], out: &mut [f32], offset_db: f32) {
    let l = S::f32s::LEN;
    let offset = S::f32s::splat(simd, offset_db);
    let n = out.len().min(input.len()) / l;
    let (vi, ri) = input.split_at(l * n);
    let (vo, ro) = out.split_at_mut(l * n);
    for (i, o) in vi.chunks_exact(l).zip(vo.chunks_exact_mut(l)) {
        power_db_vector(simd, S::f32s::from_slice(simd, i), offset).store_slice(o);
    }
    for (o, &p) in ro.iter_mut().zip(ri) {
        *o = to_db(p, offset_db);
    }
}

impl PowerDb<f32> for Simd {
    fn power_db(&self, input: &[f32], output: &mut [f32], offset_db: f32) -> Result<()> {
        check_len(input.len(), output.len())?;
        dispatch!(simd => power_db_f32(simd, input, output, offset_db));
        Ok(())
    }
}

/// `power_db` of Q15 powers (`raw / 32768`, exact in `f32`).
#[inline(always)]
fn power_db_q15<S: SimdToken>(simd: S, input: &[i16], out: &mut [f32], offset_db: f32) {
    let l = S::i16s::LEN;
    let offset = S::f32s::splat(simd, offset_db);
    let n = out.len().min(input.len()) / l;
    let (vi, ri) = input.split_at(l * n);
    let (vo, ro) = out.split_at_mut(l * n);
    for (i, o) in vi.chunks_exact(l).zip(vo.chunks_exact_mut(l)) {
        let (a, b) = S::i16s::from_slice(simd, i).widen();
        let scale = 1.0 / 32768.0;
        let a = a.to_float::<S::f32s>() * scale;
        let b = b.to_float::<S::f32s>() * scale;
        let (o0, o1) = o.split_at_mut(l / 2);
        power_db_vector(simd, a, offset).store_slice(o0);
        power_db_vector(simd, b, offset).store_slice(o1);
    }
    for (o, &p) in ro.iter_mut().zip(ri) {
        *o = to_db(f32::from(p) / 32768.0, offset_db);
    }
}

impl PowerDb<Q15> for Simd {
    fn power_db(&self, input: &[Q15], output: &mut [f32], offset_db: f32) -> Result<()> {
        check_len(input.len(), output.len())?;
        let input: &[i16] = bytemuck::cast_slice(input);
        dispatch!(simd => power_db_q15(simd, input, output, offset_db));
        Ok(())
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    //! `CQ15` with `pmaddwd`: `re*re + im*im` of a sample is one lane of the multiply-add of
    //! adjacent pairs of `x * x`. The only sum that does not fit `i32` is `2^31` (both parts
    //! `-32768`), which `pmaddwd` returns as `0x8000_0000`, the bits of the unsigned sum; so the
    //! rounding and the shift run on unsigned lanes, and the signed saturating narrowing
    //! (`packssdw`, `vpmovsdw`) applies the limit 32767.

    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128i, __m256i, __m512i, _mm_add_epi16, _mm_add_epi32, _mm_adds_epi16, _mm_avg_epu16,
        _mm_cmpeq_epi16, _mm_cmpgt_epi32, _mm_cvtepi32_ps, _mm_cvtps_epi32, _mm_madd_epi16,
        _mm_max_epi16, _mm_min_ps, _mm_mulhi_epi16, _mm_mulhrs_epi16, _mm_mullo_epi16,
        _mm_packs_epi32, _mm_set1_epi16, _mm_set1_epi32, _mm_set1_ps, _mm_setzero_si128,
        _mm_sqrt_ps, _mm_srli_epi16, _mm_srli_epi32, _mm_sub_epi32, _mm_unpackhi_epi16,
        _mm_unpacklo_epi16, _mm_xor_si128, _mm256_add_epi32, _mm256_cmpeq_epi16,
        _mm256_cmpgt_epi32, _mm256_cvtepi32_ps, _mm256_cvtps_epi32, _mm256_madd_epi16,
        _mm256_max_epi16, _mm256_min_ps, _mm256_mulhrs_epi16, _mm256_packs_epi32,
        _mm256_permute4x64_epi64, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_ps,
        _mm256_sqrt_ps, _mm256_srli_epi32, _mm256_sub_epi32, _mm256_unpackhi_epi16,
        _mm256_unpacklo_epi16, _mm256_xor_si256, _mm512_add_epi16, _mm512_add_epi32,
        _mm512_cmpeq_epi16_mask, _mm512_cmpge_epi32_mask, _mm512_cmpgt_epi32_mask,
        _mm512_cvtepi32_ps, _mm512_cvtps_epi32, _mm512_cvtsepi32_epi16, _mm512_madd_epi16,
        _mm512_mask_add_epi32, _mm512_mask_mov_epi16, _mm512_mask_sub_epi32, _mm512_max_epi16,
        _mm512_max_epi32, _mm512_min_ps, _mm512_mulhrs_epi16, _mm512_permutexvar_epi16,
        _mm512_set_epi16, _mm512_set1_epi16, _mm512_set1_epi32, _mm512_set1_ps, _mm512_sqrt_ps,
        _mm512_srli_epi32, _mm512_sub_epi32,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{i16x8, i16x16, i16x32};

    use super::{Split, UNROLL, split};

    // `window` of `CQ15`: `pmulhrsw` (SSSE3) is `(x w + 2^14) >> 15` in 16 bits, the rounding of
    // `q15_mul`. The one product outside the range is `(-32768)^2`, whose 32768 wraps to
    // -32768; any other result is at least -32767 (`-32768 * 32767` gives -32767), so the
    // lanes equal to -32768 are exactly those to saturate, and the XOR with the mask of the
    // comparison turns them into 32767. Each weight is duplicated for the real and the
    // imaginary part with the unpack instructions.

    fearless_simd::kernel!(
        pub(super) fn window_q15_sse4_2(
            t: Sse4_2,
            input: &[i16],
            out: &mut [i16],
            w: &[i16],
        ) -> usize {
            let min = _mm_set1_epi16(i16::MIN);
            let mul = |x: &[i16], w: __m128i| -> __m128i {
                let x: __m128i = i16x8::from_slice(t, x).into();
                let r = _mm_mulhrs_epi16(x, w);
                _mm_xor_si128(r, _mm_cmpeq_epi16(r, min))
            };
            window_steps!(input, out, w, 8, |i, o, wv| {
                let wv: __m128i = i16x8::load_array_ref(t, wv).into();
                let (i0, i1) = i.split_at(8);
                let (o0, o1) = o.split_at_mut(8);
                i16x8::simd_from(t, mul(i0, _mm_unpacklo_epi16(wv, wv))).store_slice(o0);
                i16x8::simd_from(t, mul(i1, _mm_unpackhi_epi16(wv, wv))).store_slice(o1);
            })
        }
    );

    // SSE2 has no `pmulhrsw`: the product `P = 2^16 h + l` comes in its signed high half `h`
    // (`pmulhw`) and its unsigned low half `l` (`pmullw`), and `(P + 2^14) >> 15` is
    // `2 h + ((l + 2^14) >> 15)`, where the second term is `((l >> 14) + 1) >> 1`, the rounding
    // average (`pavgw`) of `l >> 14` and 0. The doubling saturates (`paddsw`): `2 h` leaves the
    // range only for `P = 2^30`, the product `(-32768)^2`, whose result saturates to 32767 (its
    // low half is 0); every other sum stays in range.
    fearless_simd::kernel!(
        pub(super) fn window_q15_sse2(t: Sse2, input: &[i16], out: &mut [i16], w: &[i16]) -> usize {
            let zero = _mm_setzero_si128();
            let mul = |x: &[i16], w: __m128i| -> __m128i {
                let x: __m128i = i16x8::from_slice(t, x).into();
                let h = _mm_mulhi_epi16(x, w);
                let l = _mm_mullo_epi16(x, w);
                let round = _mm_avg_epu16(_mm_srli_epi16::<14>(l), zero);
                _mm_add_epi16(_mm_adds_epi16(h, h), round)
            };
            window_steps!(input, out, w, 8, |i, o, wv| {
                let wv: __m128i = i16x8::load_array_ref(t, wv).into();
                let (i0, i1) = i.split_at(8);
                let (o0, o1) = o.split_at_mut(8);
                i16x8::simd_from(t, mul(i0, _mm_unpacklo_epi16(wv, wv))).store_slice(o0);
                i16x8::simd_from(t, mul(i1, _mm_unpackhi_epi16(wv, wv))).store_slice(o1);
            })
        }
    );

    fearless_simd::kernel!(
        pub(super) fn window_q15_avx2(t: Avx2, input: &[i16], out: &mut [i16], w: &[i16]) -> usize {
            let min = _mm256_set1_epi16(i16::MIN);
            let mul = |x: &[i16], w: __m256i| -> __m256i {
                let x: __m256i = i16x16::from_slice(t, x).into();
                let r = _mm256_mulhrs_epi16(x, w);
                _mm256_xor_si256(r, _mm256_cmpeq_epi16(r, min))
            };
            window_steps!(input, out, w, 16, |i, o, wv| {
                // `vpunpck*wd` work within the 128-bit halves: `vpermq` first puts the weights of
                // samples 0-3 and 8-11 into the low half, those of 4-7 and 12-15 into the high.
                let wv: __m256i = i16x16::load_array_ref(t, wv).into();
                let wv = _mm256_permute4x64_epi64::<0b11_01_10_00>(wv);
                let (i0, i1) = i.split_at(16);
                let (o0, o1) = o.split_at_mut(16);
                i16x16::simd_from(t, mul(i0, _mm256_unpacklo_epi16(wv, wv))).store_slice(o0);
                i16x16::simd_from(t, mul(i1, _mm256_unpackhi_epi16(wv, wv))).store_slice(o1);
            })
        }
    );

    fearless_simd::kernel!(
        pub(super) fn window_q15_avx512(
            t: Avx512,
            input: &[i16],
            out: &mut [i16],
            w: &[i16],
        ) -> usize {
            let min = _mm512_set1_epi16(i16::MIN);
            let max = _mm512_set1_epi16(i16::MAX);
            // The word indices `[0, 0, 1, 1, ..., 15, 15]` and `[16, 16, ..., 31, 31]`.
            let lo = _mm512_set_epi16(
                15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4,
                3, 3, 2, 2, 1, 1, 0, 0,
            );
            let hi = _mm512_add_epi16(lo, _mm512_set1_epi16(16));
            let mul = |x: &[i16], w: __m512i| -> __m512i {
                let x: __m512i = i16x32::from_slice(t, x).into();
                let r = _mm512_mulhrs_epi16(x, w);
                _mm512_mask_mov_epi16(r, _mm512_cmpeq_epi16_mask(r, min), max)
            };
            window_steps!(input, out, w, 32, |i, o, wv| {
                let wv: __m512i = i16x32::load_array_ref(t, wv).into();
                let (i0, i1) = i.split_at(32);
                let (o0, o1) = o.split_at_mut(32);
                let w0 = _mm512_permutexvar_epi16(lo, wv);
                let w1 = _mm512_permutexvar_epi16(hi, wv);
                i16x32::simd_from(t, mul(i0, w0)).store_slice(o0);
                i16x32::simd_from(t, mul(i1, w1)).store_slice(o1);
            })
        }
    );

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

    // `mag` of `CQ15`: the rounded root `y` of `s = re*re + im*im`, the smallest `y` with
    // `y (y + 1) >= s`, from a single-precision root and a correction of at most one step in
    // each direction (see `root_q15` for the bound of the error of the estimate). The products
    // come from `pmaddwd` as well: `s` from the samples, and `y * y` from the lanes of `y`,
    // whose high 16 bits are zero (`y <= 32768`, and `(-32768)^2` is exact).
    //
    // The input is limited to `-32767` first: a sample with a part of `-32768` has a magnitude
    // of at least 32768 and so saturates, as it does with `-32767`, and the largest sum
    // `2 * 32767^2` then fits a signed lane (SSE2 has no unsigned minimum). The root is limited
    // to 32768 (`minps`) so that `y * y` fits too; above it the output saturates anyway: `y`
    // ends at 32768 at the least, and the saturating narrowing gives 32767. The one result
    // below zero is `-1` for `s = 0` (the correction steps down from 0); a maximum with 0 after
    // the narrowing restores it.

    /// The path for the 128-bit vectors, for a token of SSE2 or of SSE4.2 (which adds nothing it
    /// uses).
    macro_rules! sse_mag {
        ($name:ident, $token:ident) => {
            fearless_simd::kernel!(
                pub(super) fn $name(t: $token, input: &[i16], out: &mut [i16]) -> usize {
                    // A closure, not a macro: the four expansions of the arithmetic (two vectors
                    // per step, two loops) would exceed the stack frame the lint allows. It
                    // inherits the target features of the function and is inlined.
                    let roots = |x: &[i16]| -> __m128i {
                        {
                            let x: __m128i = i16x8::from_slice(t, x).into();
                            let x = _mm_max_epi16(x, _mm_set1_epi16(-32767));
                            let s = _mm_madd_epi16(x, x);
                            let r = _mm_sqrt_ps(_mm_cvtepi32_ps(s));
                            let y = _mm_cvtps_epi32(_mm_min_ps(r, _mm_set1_ps(32768.0)));
                            let yy = _mm_madd_epi16(y, y);
                            // Too small where `y (y + 1) < s`, too large where
                            // `(y - 1) y >= s`, that is `y (y - 1) + 1 > s`; -1 where true.
                            let up = _mm_cmpgt_epi32(s, _mm_add_epi32(yy, y));
                            let one = _mm_set1_epi32(1);
                            let down = _mm_cmpgt_epi32(_mm_add_epi32(yy, _mm_sub_epi32(one, y)), s);
                            _mm_add_epi32(_mm_sub_epi32(y, up), down)
                        }
                    };
                    steps!(input, out, 8, UNROLL, |i, o| {
                        let (a, b) = i.split_at(8);
                        let packed = _mm_packs_epi32(roots(a), roots(b));
                        i16x8::simd_from(t, _mm_max_epi16(packed, _mm_set1_epi16(0)))
                            .store_array(o);
                    })
                }
            );
        };
    }
    sse_mag!(mag_q15_sse2, Sse2);
    sse_mag!(mag_q15_sse4_2, Sse4_2);

    fearless_simd::kernel!(
        pub(super) fn mag_q15_avx2(t: Avx2, input: &[i16], out: &mut [i16]) -> usize {
            // A closure, not a macro (see the 128-bit path).
            let roots = |x: &[i16]| -> __m256i {
                {
                    let x: __m256i = i16x16::from_slice(t, x).into();
                    let x = _mm256_max_epi16(x, _mm256_set1_epi16(-32767));
                    let s = _mm256_madd_epi16(x, x);
                    let r = _mm256_sqrt_ps(_mm256_cvtepi32_ps(s));
                    let y = _mm256_cvtps_epi32(_mm256_min_ps(r, _mm256_set1_ps(32768.0)));
                    let yy = _mm256_madd_epi16(y, y);
                    let up = _mm256_cmpgt_epi32(s, _mm256_add_epi32(yy, y));
                    let one = _mm256_set1_epi32(1);
                    let down =
                        _mm256_cmpgt_epi32(_mm256_add_epi32(yy, _mm256_sub_epi32(one, y)), s);
                    _mm256_add_epi32(_mm256_sub_epi32(y, up), down)
                }
            };
            steps!(input, out, 16, UNROLL, |i, o| {
                let (a, b) = i.split_at(16);
                // `vpackssdw` packs within the 128-bit halves; `vpermq` restores the order.
                let packed = _mm256_packs_epi32(roots(a), roots(b));
                let packed = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);
                i16x16::simd_from(t, _mm256_max_epi16(packed, _mm256_set1_epi16(0))).store_array(o);
            })
        }
    );

    fearless_simd::kernel!(
        pub(super) fn mag_q15_avx512(t: Avx512, input: &[i16], out: &mut [i16]) -> usize {
            // A closure, not inline code (see the 128-bit path).
            let roots = |i: &[i16; 32]| -> __m512i {
                let x: __m512i = i16x32::load_array_ref(t, i).into();
                let x = _mm512_max_epi16(x, _mm512_set1_epi16(-32767));
                let s = _mm512_madd_epi16(x, x);
                let r = _mm512_sqrt_ps(_mm512_cvtepi32_ps(s));
                let y = _mm512_cvtps_epi32(_mm512_min_ps(r, _mm512_set1_ps(32768.0)));
                let yy = _mm512_madd_epi16(y, y);
                // The masks of the corrections: one up where `y (y + 1) < s`, one down where
                // `(y - 1) y >= s`.
                let one = _mm512_set1_epi32(1);
                let up = _mm512_cmpgt_epi32_mask(s, _mm512_add_epi32(yy, y));
                let down = _mm512_cmpge_epi32_mask(_mm512_sub_epi32(yy, y), s);
                let y = _mm512_mask_add_epi32(y, up, y, one);
                let y = _mm512_mask_sub_epi32(y, down, y, one);
                // `-1` for `s = 0`; the signed saturating narrowing gives 32767 above.
                _mm512_max_epi32(y, _mm512_set1_epi32(0))
            };
            // One vector per step: `vpmovsdw` narrows a single vector, so no pairs are needed.
            steps!(input, out, 16, UNROLL, |i, o| {
                let y = roots(i);
                i16x16::simd_from(t, _mm512_cvtsepi32_epi16(y)).store_array(o);
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
        float32x4_t, float32x4x2_t, float32x4x4_t, int16x8_t, int16x8x2_t, int16x8x4_t, uint32x4_t,
        vcgtq_u32, vcvtaq_u32_f32, vcvtq_f32_u32, vdupq_n_f32, vget_low_s16, vld2q_f32, vld2q_s16,
        vld4q_s16, vmlal_high_s16, vmlal_s16, vmlaq_u32, vmull_high_s16, vmull_s16, vmulq_f32,
        vpaddq_f32, vqdmlal_high_s16, vqdmlal_s16, vqdmull_high_s16, vqdmull_s16, vqmovn_high_s32,
        vqmovn_s32, vqrdmulhq_s16, vqrshrn_high_n_s32, vqrshrn_n_s32, vreinterpretq_s32_u32,
        vreinterpretq_u32_s32, vsqrtq_f32, vst1q_s16, vst2q_f32, vst2q_s16, vsubq_f32, vsubq_u32,
        vzip1q_s16, vzip2q_s16,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4, f32x8, f32x16, i16x8};

    use super::{Split, UNROLL, split};

    /// The 8 values of `x` (4 samples) split into the real and the imaginary parts.
    #[inline(always)]
    fn load2_f32(x: &[f32; 8]) -> float32x4x2_t {
        // SAFETY: `vld2q_f32` reads 8 consecutive `f32` with no alignment requirement beyond
        // that of `f32`, and `x` holds exactly 8. NEON is part of the AArch64 baseline.
        unsafe { vld2q_f32(x.as_ptr()) }
    }

    /// Stores the real and the imaginary parts of 4 samples interleaved into `o`.
    #[inline(always)]
    fn store2_f32(o: &mut [f32; 8], re: float32x4_t, im: float32x4_t) {
        // SAFETY: `vst2q_f32` writes 8 consecutive `f32` with no alignment requirement beyond
        // that of `f32`, and `o` holds exactly 8 and is borrowed exclusively.
        unsafe { vst2q_f32(o.as_mut_ptr(), float32x4x2_t(re, im)) }
    }

    fearless_simd::kernel!(
        pub(super) fn window_f32(t: Neon, input: &[f32], out: &mut [f32], w: &[f32]) -> usize {
            // `ld2` splits the samples into their real and imaginary parts, both are multiplied
            // by the weights as they are, and `st2` interleaves them again: no shuffle of the
            // weights. Four steps per iteration, which advance the pointers by post-indexing.
            window_steps!(input, out, w, 4, |i, o, wv| {
                let wv: float32x4_t = f32x4::load_array_ref(t, wv).into();
                let x = load2_f32(i);
                store2_f32(o, vmulq_f32(x.0, wv), vmulq_f32(x.1, wv));
            })
        }
    );

    fearless_simd::kernel!(
        pub(super) fn window_q15(t: Neon, input: &[i16], out: &mut [i16], w: &[i16]) -> usize {
            // `sqrdmulh` is `sat((2 x w + 2^15) >> 16)`, which is `q15_mul` exactly, saturation
            // of `(-32768)^2` included; `zip1`/`zip2` duplicate each weight for both parts.
            window_steps!(input, out, w, 8, |i, o, wv| {
                let wv: int16x8_t = i16x8::load_array_ref(t, wv).into();
                let (i0, i1) = i.split_at(8);
                let (o0, o1) = o.split_at_mut(8);
                let x0: int16x8_t = i16x8::from_slice(t, i0).into();
                let x1: int16x8_t = i16x8::from_slice(t, i1).into();
                i16x8::simd_from(t, vqrdmulhq_s16(x0, vzip1q_s16(wv, wv))).store_slice(o0);
                i16x8::simd_from(t, vqrdmulhq_s16(x1, vzip2q_s16(wv, wv))).store_slice(o1);
            })
        }
    );

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

    /// The 16 values of `x` (8 samples) split into the real and the imaginary parts.
    #[inline(always)]
    fn load2(x: &[i16; 16]) -> int16x8x2_t {
        // SAFETY: `vld2q_s16` reads 16 consecutive `i16` with no alignment requirement beyond
        // that of `i16`, and `x` holds exactly 16. NEON is part of the AArch64 baseline.
        unsafe { vld2q_s16(x.as_ptr()) }
    }

    /// Stores the 8 results.
    #[inline(always)]
    fn store1(o: &mut [i16; 8], v: int16x8_t) {
        // SAFETY: `vst1q_s16` writes 8 consecutive `i16` with no alignment requirement beyond
        // that of `i16`, and `o` holds exactly 8 and is borrowed exclusively.
        unsafe { vst1q_s16(o.as_mut_ptr(), v) }
    }

    fearless_simd::kernel!(
        pub(super) fn mag_q15(t: Neon, input: &[i16], out: &mut [i16]) -> usize {
            // The rounded roots of four sums of squares (`s`, at most `2^31`, as `u32`): the
            // smallest `y` with `y (y + 1) >= s`, which is the root rounded to the nearest
            // integer. The single-precision root is off by less than 0.0063 (the conversion of
            // `s`, the root and the subtraction below, each half an ulp at most, at 46341), so
            // after the subtraction of 1/64 it lies in `(sqrt(s) - 0.022, sqrt(s))`, and its
            // nearest integer (`fcvtau`, which turns a negative value into 0) is `y` or `y - 1`
            // and never above `y`. One test (`s > y_est (y_est + 1)`, which fits `u32`: the
            // largest root is 46341) then adds the one that may be missing. Proven by the sweep
            // over all 2^32 raw pairs (`mag_cq15_exhaustive_every_raw_pair`).
            // A closure, not a macro: the four expansions would exceed the stack frame the lint
            // allows. It inherits the target features of the function and is inlined.
            let roots = |s: uint32x4_t| -> uint32x4_t {
                let r = vsubq_f32(vsqrtq_f32(vcvtq_f32_u32(s)), vdupq_n_f32(1.0 / 64.0));
                let y = vcvtaq_u32_f32(r);
                // The mask is all ones (-1) where true.
                vsubq_u32(y, vcgtq_u32(s, vmlaq_u32(y, y, y)))
            };
            // One step per iteration: `ld2` has no offset form, only the post-index form of the
            // loop.
            steps!(input, out, 8, 1, |i, o| {
                let x = load2(i);
                let (re, im) = (x.0, x.1);
                // Products of at most `2^30`; the sum of two is at most `2^31`, which the
                // unsigned lanes hold (the signed `smlal` wraps to the same bits).
                let lo = vmlal_s16(
                    vmull_s16(vget_low_s16(re), vget_low_s16(re)),
                    vget_low_s16(im),
                    vget_low_s16(im),
                );
                let hi = vmlal_high_s16(vmull_high_s16(re, re), im, im);
                let lo = vreinterpretq_s32_u32(roots(vreinterpretq_u32_s32(lo)));
                let hi = vreinterpretq_s32_u32(roots(vreinterpretq_u32_s32(hi)));
                store1(o, vqmovn_high_s32(vqmovn_s32(lo), hi));
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
        f32x4_add, f32x4_convert_i32x4, f32x4_min, f32x4_splat, f32x4_sqrt, i16x8_max,
        i16x8_narrow_i32x4, i16x8_q15mulr_sat, i16x8_shuffle, i16x8_splat, i32x4_add,
        i32x4_dot_i16x8, i32x4_gt, i32x4_shr, i32x4_splat, i32x4_sub, i32x4_trunc_sat_f32x4,
        u16x8_min, u16x8_splat, v128,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4, i16x8};

    use super::{Split, split};

    fearless_simd::kernel!(
        pub(super) fn window_q15(
            t: WasmSimd128,
            input: &[i16],
            out: &mut [i16],
            w: &[i16],
        ) -> usize {
            // `i16x8.q15mulr_sat_s` is `q15_mul` exactly, saturation of `(-32768)^2` included.
            window_steps!(input, out, w, 8, |i, o, wv| {
                let wv: v128 = i16x8::load_array_ref(t, wv).into();
                let (i0, i1) = i.split_at(8);
                let (o0, o1) = o.split_at_mut(8);
                let x0: v128 = i16x8::from_slice(t, i0).into();
                let x1: v128 = i16x8::from_slice(t, i1).into();
                let w0 = i16x8_shuffle::<0, 0, 1, 1, 2, 2, 3, 3>(wv, wv);
                let w1 = i16x8_shuffle::<4, 4, 5, 5, 6, 6, 7, 7>(wv, wv);
                i16x8::simd_from(t, i16x8_q15mulr_sat(x0, w0)).store_slice(o0);
                i16x8::simd_from(t, i16x8_q15mulr_sat(x1, w1)).store_slice(o1);
            })
        }
    );

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

    fearless_simd::kernel!(
        pub(super) fn mag_q15(t: WasmSimd128, input: &[i16], out: &mut [i16]) -> usize {
            // The rounded roots as on x86 (see `x86`): the input limited to -32767 so that the
            // sums fit a signed lane, the root limited to 32768 so that `y * y` does too.
            // A closure, not a macro (see `neon`).
            let roots = |x: &[i16]| -> v128 {
                let x: v128 = i16x8::from_slice(t, x).into();
                let x = i16x8_max(x, i16x8_splat(-32767));
                let s = i32x4_dot_i16x8(x, x);
                let r = f32x4_sqrt(f32x4_convert_i32x4(s));
                let r = f32x4_add(f32x4_min(r, f32x4_splat(32768.0)), f32x4_splat(0.5));
                let y = i32x4_trunc_sat_f32x4(r);
                let yy = i32x4_dot_i16x8(y, y);
                let up = i32x4_gt(s, i32x4_add(yy, y));
                let down = i32x4_gt(i32x4_add(yy, i32x4_sub(i32x4_splat(1), y)), s);
                i32x4_add(i32x4_sub(y, up), down)
            };
            steps!(input, out, 8, UNROLL, |i, o| {
                let (a, b) = i.split_at(8);
                let packed = i16x8_narrow_i32x4(roots(a), roots(b));
                i16x8::simd_from(t, i16x8_max(packed, i16x8_splat(0))).store_array(o);
            })
        }
    );
}

#[cfg(test)]
mod tests {
    use super::{mag_q15_level, mag_sq_f32_level, mag_sq_q15_level, window_q15_level};
    use crate::level::{dispatch, name};

    #[test]
    fn the_window_path_of_the_current_level_runs_on_all_whole_steps() {
        let step = match name() {
            "avx512" => 32,
            "avx2" => 16,
            "sse4.2" | "sse2" | "neon" | "simd128" => 8,
            _ => 0,
        };
        // `q15_mul(1000, 16384)` is 500.
        let input = [1000i16; 2 * 300];
        let w = [16384i16; 300];
        for n in 0..=300 {
            let mut out = [0i16; 2 * 300];
            let ((x, _), (y, _)) = (input.split_at(2 * n), out.split_at_mut(2 * n));
            let done = dispatch!(simd => window_q15_level(simd, x, y, &w[..n]));
            assert_eq!(done, n.checked_rem(step).map_or(0, |r| n - r), "n {n}");
            assert!(y.iter().take(2 * done).all(|&v| v == 500), "n {n}");
            assert!(y.iter().skip(2 * done).all(|&v| v == 0), "n {n}");
        }
    }

    /// The samples per step of the level-specific path of `mag` `CQ15` at `level`, `0` where the
    /// level has none (`mag` `C32` has none on any level).
    fn mag_steps(level: &str) -> usize {
        match level {
            "avx512" | "avx2" => 16,
            "sse4.2" | "sse2" | "neon" | "simd128" => 8,
            _ => 0,
        }
    }

    #[test]
    fn the_mag_path_of_the_current_level_runs_on_all_whole_steps() {
        let step = mag_steps(name());
        // `(1000, 1000)` has the magnitude 1414.2, rounded 1414.
        let input = [1000i16; 2 * 300];
        for n in 0..=300 {
            let mut out = [0i16; 300];
            let ((x, _), (y, _)) = (input.split_at(2 * n), out.split_at_mut(n));
            let done = dispatch!(simd => mag_q15_level(simd, x, y));
            assert_eq!(done, n.checked_rem(step).map_or(0, |r| n - r), "n {n}");
            assert!(y.iter().take(done).all(|&v| v == 1414), "n {n}");
        }
    }

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
