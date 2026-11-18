//! `gauss_noise` (M48 §5), bit-exact with the fallback (the contract of the kernel is bit-exact,
//! `docs/kernels.md`).
//!
//! Every lane runs the integer formula of the contract, so the bits follow from the arithmetic:
//!
//! 1. **Lane-parallel PCG32.** Lane `i` of a block of `L` lanes starts at the scalar state
//!    stepped `i` times, and every lane moves with the jump of `L` draws
//!    ([`Pcg32::jump`]), so lane `i` of block `b` holds draw `b · L + i`. The draw is the
//!    output function of the old state ([`sdrk_core::kernels::noise::pcg32_output`]): the
//!    xorshift on the 64-bit lanes and a rotation of 32-bit lanes (variable shifts, `vprorvd`,
//!    or a product with `2^(31 - rot)` where the level has no variable shift). At the end of
//!    the call the scalar generator advances by the draws of the vector part
//!    ([`Pcg32::advance`]) and the remainder shorter than one block runs the scalar
//!    [`normal_q17`].
//! 2. **The inverse CDF.** The octave `L`, the segment `sub` and the fraction `frac` of the
//!    odd `x = 2r + 1` come from `clz` and a variable shift where the level has them (AVX-512,
//!    NEON). Elsewhere they come from the bits of the exact `f64` value `x / 2 = v + 0.5`
//!    (`v = r & 0x7FFF_FFFF`, converted with `cvtdq2pd`): its high word is
//!    `sign | (1022 + L) << 20 | sub << 15 | frac`, so `(hi >> 15) & 0xFFFF` is the table index
//!    plus 32704 and `hi & 0x7FFF` the fraction. The table entry is gathered (AVX2, AVX-512) or
//!    loaded per lane, the interpolation is the rounding Q15 product of the contract
//!    (`pmulhrsw`, `vqrdmulh`, `q15mulr`; `pmaddwd` with the rounding constant on SSE2) and the
//!    sign is applied branch-free.
//! 3. **The output.** `f32`: `(zq as f32) · s17`, one conversion and one multiplication, no
//!    fused multiply-add. `Q15`: `floor((zq · ms + 2^(sh-1)) / 2^sh)` exactly, with a signed
//!    64-bit product and an arithmetic shift where the level has them (AVX-512, NEON, SIMD128),
//!    else as `((zq + 2^20) · ms + k1) >> sh - k2` with unsigned products (see [`Q15Shift`]);
//!    the saturating pack to 16 bits is the clamp of the contract.
//!
//! A complex buffer is the real kernel on its `2n` values in draw order, as in the fallback.
//! `sigma` counting as 0 fills `+0.0` (or 0) and jumps the generator, as the fallback does.
// The helpers of the level paths are unused on a target without one (wasm32 without
// `simd128`, where the scalar loop of `fill_f32` and `fill_q15` runs).
#![cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code)
)]

use fearless_simd::Simd as SimdToken;
use sdrk_core::kernels::noise::{
    GaussNoise, GaussNoiseState, GaussScale, PCG32_MULT, Pcg32, normal_q17,
};
use sdrk_core::{C32, CQ15, Q15, Result, Sample};

use crate::Simd;
use crate::level::dispatch_fn;

/// The states of `L` consecutive draws and the jump `s -> a · s + c` of `L` draws.
#[derive(Clone, Copy)]
struct Lanes<const L: usize> {
    s: [u64; L],
    a: u64,
    c: u64,
}

impl<const L: usize> Lanes<L> {
    #[inline(always)]
    fn new(rng: &Pcg32) -> Self {
        let inc = rng.increment();
        let mut s = [0; L];
        let mut x = rng.state();
        for v in &mut s {
            *v = x;
            x = x.wrapping_mul(PCG32_MULT).wrapping_add(inc);
        }
        let (a, c) = rng.jump(L as u64);
        Self { s, a, c }
    }
}

/// The exact `Q15` output with unsigned 64-bit products, for the levels without a signed
/// 64-bit product or an arithmetic 64-bit shift.
///
/// `|zq| < 2^20`, `ms < 2^24` and `21 ≤ sh ≤ 44`, so `zq · ms + half + 2^44` lies in
/// `(0, 2^45 + 2^43]` and `2^44` is a multiple of `2^sh`:
/// `floor((zq · ms + half) / 2^sh) = ((zq + 2^20) · ms + k1) >> sh − k2` with
/// `k1 = half + 2^44 − 2^20 · ms` and `k2 = 2^(44 − sh)`. The result lies within
/// `[-2^23 - 1, 2^23]`, so the signed saturating pack to 16 bits is the clamp of the contract.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
struct Q15Shift {
    ms: u32,
    sh: u32,
    half: u64,
    k1: u64,
    k2: i32,
}

impl Q15Shift {
    #[inline(always)]
    fn new(scale: GaussScale) -> Self {
        let (ms, sh) = (scale.ms(), scale.sh());
        let half = 1_u64 << (sh - 1);
        Self {
            ms,
            sh,
            half,
            k1: half + (1_u64 << 44) - (u64::from(ms) << 20),
            k2: 1_i32 << (44 - sh),
        }
    }
}

/// Fills `out` from the generator: the whole blocks of `W` values (`W` a multiple of `L`) with
/// `body`, which receives the lane states of the first block, then the scalar generator jumps
/// over the draws of the blocks and the remainder runs `scalar` on the next draws.
#[inline(always)]
fn drive<V, const L: usize, const W: usize>(
    rng: &mut Pcg32,
    out: &mut [V],
    scalar: impl Fn(u32) -> V,
    body: impl FnOnce(&Lanes<L>, &mut [V]),
) {
    let whole = out.len() - out.len() % W;
    let (head, tail) = out.split_at_mut(whole);
    if !head.is_empty() {
        body(&Lanes::<L>::new(rng), head);
        rng.advance(whole as u64);
    }
    for o in tail {
        *o = scalar(rng.next_u32());
    }
}

/// Runs `emit` on every chunk of `C` values of `body` with the results of its `K` blocks, in
/// block order, software-pipelined two blocks deep: `prepare` (the draws of the next block, the
/// generator step and the index arithmetic) runs two blocks ahead of `finish` (the table
/// lookups, the interpolation and the sign), so the long dependency chain of one block
/// overlaps with the next two blocks instead of filling the reorder window. The two extra
/// `prepare` at the end only compute. `C` is `K` times the values of a block.
#[inline(always)]
#[cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(dead_code)
)]
fn pipeline<P: Copy, Z, V, const C: usize, const K: usize>(
    body: &mut [V],
    mut prepare: impl FnMut() -> P,
    mut finish: impl FnMut(P) -> Z,
    mut emit: impl FnMut([Z; K], &mut [V]),
) {
    let mut cur = prepare();
    let mut next = prepare();
    for chunk in body.chunks_exact_mut(C) {
        let z = core::array::from_fn(|_| {
            let after = prepare();
            let z = finish(cur);
            cur = next;
            next = after;
            z
        });
        emit(z, chunk);
    }
}

/// [`pipeline`] without the pipelining: every block is prepared and finished in order. The
/// `Q15` paths of AVX2 and AVX-512 use it, where the pipelined form was slower on the
/// counter-model (`alderlake` 4.04 against 3.09, `sapphirerapids` 3.67 against 1.86 cycles per
/// sample) and not faster on the reference model.
#[inline(always)]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
fn in_order<P, Z, V, const C: usize, const K: usize>(
    body: &mut [V],
    mut prepare: impl FnMut() -> P,
    mut finish: impl FnMut(P) -> Z,
    mut emit: impl FnMut([Z; K], &mut [V]),
) {
    for chunk in body.chunks_exact_mut(C) {
        emit(core::array::from_fn(|_| finish(prepare())), chunk);
    }
}

/// The `f32` values of the draws, at the level of `simd`.
#[inline(always)]
fn fill_f32<S: SimdToken>(simd: S, rng: &mut Pcg32, scale: GaussScale, out: &mut [f32]) {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx512() {
            return x86::f32_avx512(t, rng, scale, out);
        }
        if let Some(t) = level.as_avx2() {
            return x86::f32_avx2(t, rng, scale, out);
        }
        if let Some(t) = level.as_sse4_2() {
            return x86::f32_sse4_2(t, rng, scale, out);
        }
        if let Some(t) = level.as_sse2() {
            return x86::f32_sse2(t, rng, scale, out);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return neon::f32_neon(t, rng, scale, out);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return wasm::f32_simd128(t, rng, scale, out);
    }
    let _ = level;
    for o in out {
        *o = scale.f32_value(normal_q17(rng.next_u32()));
    }
}

/// The `Q15` values of the draws, at the level of `simd`.
#[inline(always)]
fn fill_q15<S: SimdToken>(simd: S, rng: &mut Pcg32, scale: GaussScale, out: &mut [i16]) {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx512() {
            return x86::q15_avx512(t, rng, scale, out);
        }
        if let Some(t) = level.as_avx2() {
            return x86::q15_avx2(t, rng, scale, out);
        }
        if let Some(t) = level.as_sse4_2() {
            return x86::q15_sse4_2(t, rng, scale, out);
        }
        if let Some(t) = level.as_sse2() {
            return x86::q15_sse2(t, rng, scale, out);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return neon::q15_neon(t, rng, scale, out);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return wasm::q15_simd128(t, rng, scale, out);
    }
    let _ = level;
    for o in out {
        *o = scale.q15_value(normal_q17(rng.next_u32())).0;
    }
}

// The entries, one function per level (see `dispatch_fn!`).
dispatch_fn!(fn run_f32(rng: &mut Pcg32, scale: GaussScale, out: &mut [f32]) => fill_f32);
dispatch_fn!(fn run_q15(rng: &mut Pcg32, scale: GaussScale, out: &mut [i16]) => fill_q15);

/// One call over the real values `values` of an output of `len` samples of `T`: the rules of
/// the state, the zeros of a zero scale (the generator jumps over the draws), else `run`.
#[inline(always)]
fn call<T: Sample, V: Copy + Default>(
    state: &mut GaussNoiseState,
    len: usize,
    values: &mut [V],
    run: fn(&mut Pcg32, GaussScale, &mut [V]),
) -> Result<()> {
    let Some((rng, scale)) = state.begin(T::TYPE, len)? else {
        return Ok(());
    };
    if scale.is_zero() {
        // `V::default()` is `+0.0` or 0.
        values.fill(V::default());
        rng.advance(values.len() as u64);
        return Ok(());
    }
    run(rng, scale, values);
    Ok(())
}

impl GaussNoise<f32> for Simd {
    fn gauss_noise(&self, state: &mut GaussNoiseState, output: &mut [f32]) -> Result<()> {
        call::<f32, f32>(state, output.len(), output, run_f32)
    }
}

impl GaussNoise<Q15> for Simd {
    fn gauss_noise(&self, state: &mut GaussNoiseState, output: &mut [Q15]) -> Result<()> {
        let len = output.len();
        call::<Q15, i16>(state, len, bytemuck::cast_slice_mut(output), run_q15)
    }
}

impl GaussNoise<C32> for Simd {
    fn gauss_noise(&self, state: &mut GaussNoiseState, output: &mut [C32]) -> Result<()> {
        let len = output.len();
        call::<C32, f32>(state, len, bytemuck::cast_slice_mut(output), run_f32)
    }
}

impl GaussNoise<CQ15> for Simd {
    fn gauss_noise(&self, state: &mut GaussNoiseState, output: &mut [CQ15]) -> Result<()> {
        let len = output.len();
        call::<CQ15, i16>(state, len, bytemuck::cast_slice_mut(output), run_q15)
    }
}

/// The table entry of the index `i` (masked into the table, so the lookup never fails).
#[inline(always)]
fn entry(i: u32) -> u32 {
    let t = sdrk_core::kernels::noise::gauss_icdf_table();
    t.get((i & 1023) as usize).copied().unwrap_or(0)
}

/// The paths of x86-64: SSE2 and SSE4.2 (4 lanes, the `f64` form of the octave and per-lane
/// table loads), AVX2 (8 lanes, the `f64` form and `vpgatherdd`) and AVX-512 (16 lanes,
/// `vpmullq`, `vprorvd`, `vplzcntd` and `vpgatherdd`).
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod x86 {
    use core::arch::x86_64::{
        __m128i, __m256i, __m512i, _mm_add_epi32, _mm_add_epi64, _mm_add_pd, _mm_and_si128,
        _mm_castpd_ps, _mm_castps_si128, _mm_castsi128_ps, _mm_cvtepi32_pd, _mm_cvtepi32_ps,
        _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_cvttps_epi32, _mm_extract_epi32, _mm_madd_epi16,
        _mm_mul_epu32, _mm_mul_ps, _mm_mulhrs_epi16, _mm_or_si128, _mm_packs_epi32, _mm_set_epi64x,
        _mm_set1_epi32, _mm_set1_epi64x, _mm_set1_pd, _mm_set1_ps, _mm_setr_epi32,
        _mm_shuffle_epi32, _mm_shuffle_ps, _mm_sign_epi32, _mm_slli_epi32, _mm_slli_epi64,
        _mm_srai_epi32, _mm_srl_epi64, _mm_srli_epi32, _mm_srli_epi64, _mm_sub_epi32,
        _mm_unpackhi_epi64, _mm_xor_si128, _mm256_add_epi32, _mm256_add_epi64, _mm256_add_pd,
        _mm256_and_si256, _mm256_blend_epi32, _mm256_castpd_ps, _mm256_castps_si256,
        _mm256_castsi256_ps, _mm256_castsi256_si128, _mm256_cvtepi32_pd, _mm256_cvtepi32_ps,
        _mm256_extracti128_si256, _mm256_i32gather_epi32, _mm256_mul_epu32, _mm256_mul_ps,
        _mm256_mulhrs_epi16, _mm256_or_si256, _mm256_packs_epi32, _mm256_permute4x64_epi64,
        _mm256_set1_epi32, _mm256_set1_epi64x, _mm256_set1_pd, _mm256_set1_ps, _mm256_setr_epi64x,
        _mm256_shuffle_ps, _mm256_sign_epi32, _mm256_slli_epi64, _mm256_sllv_epi32,
        _mm256_srai_epi32, _mm256_srl_epi64, _mm256_srli_epi32, _mm256_srli_epi64,
        _mm256_srlv_epi32, _mm256_sub_epi32, _mm256_xor_si256, _mm512_add_epi32, _mm512_add_epi64,
        _mm512_and_si512, _mm512_cvtepi32_ps, _mm512_cvtsepi32_epi16, _mm512_i32gather_epi32,
        _mm512_lzcnt_epi32, _mm512_mask_blend_epi32, _mm512_mask_sub_epi32, _mm512_movepi32_mask,
        _mm512_mul_epi32, _mm512_mul_ps, _mm512_mulhrs_epi16, _mm512_mullo_epi64, _mm512_or_si512,
        _mm512_permutex2var_epi32, _mm512_rorv_epi32, _mm512_set1_epi32, _mm512_set1_epi64,
        _mm512_set1_ps, _mm512_setr_epi32, _mm512_setr_epi64, _mm512_setzero_si512,
        _mm512_slli_epi32, _mm512_slli_epi64, _mm512_sllv_epi32, _mm512_sra_epi64,
        _mm512_srli_epi32, _mm512_srli_epi64, _mm512_sub_epi32, _mm512_xor_si512,
    };

    use fearless_simd::{f32x4, f32x8, f32x16, i16x8, i16x16, prelude::*};
    use sdrk_core::kernels::noise::{GaussScale, Pcg32, gauss_icdf_table, normal_q17};

    use super::{Lanes, Q15Shift, drive, entry, in_order, pipeline};

    /// `(hi >> 15) & 0xFFFF` of the `f64` form minus the table index (`1022 << 5`).
    const F64_INDEX_BIAS: usize = 32_704;

    // -----------------------------------------------------------------------------------------
    // SSE2 and SSE4.2: two `u64` lanes per register, four draws per block.

    /// The LCG constants of the lanes: the low and the high word of `a`, and `c`.
    #[derive(Clone, Copy)]
    struct Jump128 {
        a_lo: __m128i,
        a_hi: __m128i,
        c: __m128i,
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    fn jump128(a: u64, c: u64) -> Jump128 {
        Jump128 {
            a_lo: _mm_set1_epi64x(a as i64),
            a_hi: _mm_set1_epi64x((a >> 32) as i64),
            c: _mm_set1_epi64x(c as i64),
        }
    }

    /// `a · s + c` per `u64` lane (three `pmuludq`).
    #[target_feature(enable = "sse2")]
    #[inline]
    fn step128(s: __m128i, j: Jump128) -> __m128i {
        let lo = _mm_mul_epu32(s, j.a_lo);
        let cross = _mm_add_epi64(
            _mm_mul_epu32(_mm_srli_epi64::<32>(s), j.a_lo),
            _mm_mul_epu32(s, j.a_hi),
        );
        _mm_add_epi64(_mm_add_epi64(lo, j.c), _mm_slli_epi64::<32>(cross))
    }

    /// The draws of the two states of `s` in the low words of the `u64` lanes: the xorshift,
    /// and the rotation by `rot = s >> 59` as the product with `2^(31 - rot)` (built from the
    /// `f32` exponent field; `cvttps2dq` of `2^31` gives `0x8000_0000`, which is the right
    /// factor): with `y = xs · 2^(31 - rot)`, `rotr(xs, rot)` is the low word of
    /// `(y << 1) | (y >> 31)`.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn draws128(s: __m128i) -> __m128i {
        let xs = _mm_srli_epi64::<27>(_mm_xor_si128(_mm_srli_epi64::<18>(s), s));
        let rot = _mm_srli_epi64::<59>(s);
        let exp = _mm_slli_epi32::<23>(_mm_sub_epi32(_mm_set1_epi32(158), rot));
        let y = _mm_mul_epu32(xs, _mm_cvttps_epi32(_mm_castsi128_ps(exp)));
        _mm_or_si128(_mm_slli_epi64::<1>(y), _mm_srli_epi64::<31>(y))
    }

    /// The four draws of the states `e` (draws 0, 1) and `o` (draws 2, 3), in draw order.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn draws4(e: __m128i, o: __m128i) -> __m128i {
        let (e, o) = (_mm_castsi128_ps(draws128(e)), _mm_castsi128_ps(draws128(o)));
        _mm_castps_si128(_mm_shuffle_ps::<0x88>(e, o))
    }

    /// The high words of the `f64` values `(r & 0x7FFF_FFFF) + 0.5` of four draws.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn high_words4(r: __m128i) -> __m128i {
        let v = _mm_and_si128(r, _mm_set1_epi32(0x7FFF_FFFF));
        let half = _mm_set1_pd(0.5);
        let a = _mm_add_pd(_mm_cvtepi32_pd(v), half);
        let b = _mm_add_pd(_mm_cvtepi32_pd(_mm_unpackhi_epi64(v, v)), half);
        _mm_castps_si128(_mm_shuffle_ps::<0xDD>(_mm_castpd_ps(a), _mm_castpd_ps(b)))
    }

    /// The table indices of the high words plus 32768, a multiple of 1024 that [`entry`] masks
    /// away (`(hi >> 15) + 64`; the sign bit lands in bit 16 and is masked away as well).
    #[target_feature(enable = "sse2")]
    #[inline]
    fn indices4(hi: __m128i) -> __m128i {
        _mm_add_epi32(_mm_srli_epi32::<15>(hi), _mm_set1_epi32(64))
    }

    /// The table entries of four indices, extracted with `movd` and shuffles.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn lookup4_sse2(i: __m128i) -> __m128i {
        let lane = |k: __m128i| entry(_mm_cvtsi128_si32(k) as u32) as i32;
        _mm_setr_epi32(
            lane(i),
            lane(_mm_shuffle_epi32::<0x55>(i)),
            lane(_mm_shuffle_epi32::<0xAA>(i)),
            lane(_mm_shuffle_epi32::<0xFF>(i)),
        )
    }

    /// The table entries of four indices, extracted with `pextrd`.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn lookup4_sse41(i: __m128i) -> __m128i {
        let lane = |v: i32| entry(v as u32) as i32;
        _mm_setr_epi32(
            lane(_mm_cvtsi128_si32(i)),
            lane(_mm_extract_epi32::<1>(i)),
            lane(_mm_extract_epi32::<2>(i)),
            lane(_mm_extract_epi32::<3>(i)),
        )
    }

    /// `base - ((slope · frac + 2^14) >> 15)` with `pmaddwd`: the pairs `(slope, 1)` and
    /// `(frac, 2^14)` (the high halves of both lanes are 0 and 1 << 14 after the `or`).
    #[target_feature(enable = "sse2")]
    #[inline]
    fn interp4_sse2(e: __m128i, hi: __m128i) -> __m128i {
        let slope = _mm_or_si128(
            _mm_and_si128(e, _mm_set1_epi32(0xFFF)),
            _mm_set1_epi32(1 << 16),
        );
        let frac = _mm_or_si128(
            _mm_and_si128(hi, _mm_set1_epi32(0x7FFF)),
            _mm_set1_epi32(1 << 30),
        );
        let prod = _mm_srli_epi32::<15>(_mm_madd_epi16(slope, frac));
        _mm_sub_epi32(_mm_srli_epi32::<12>(e), prod)
    }

    /// `base - ((slope · frac + 2^14) >> 15)` with `pmulhrsw` (the high halves are 0).
    #[target_feature(enable = "ssse3")]
    #[inline]
    fn interp4_ssse3(e: __m128i, hi: __m128i) -> __m128i {
        let slope = _mm_and_si128(e, _mm_set1_epi32(0xFFF));
        let frac = _mm_and_si128(hi, _mm_set1_epi32(0x7FFF));
        _mm_sub_epi32(_mm_srli_epi32::<12>(e), _mm_mulhrs_epi16(slope, frac))
    }

    /// `mag` with the sign of the draw `r`: `(mag ^ s) - s`, `s = r >> 31` (arithmetic).
    #[target_feature(enable = "sse2")]
    #[inline]
    fn sign4_sse2(mag: __m128i, r: __m128i) -> __m128i {
        let s = _mm_srai_epi32::<31>(r);
        _mm_sub_epi32(_mm_xor_si128(mag, s), s)
    }

    /// `mag` with the sign of the draw `r` (`psignd` of `r | 1`, never 0).
    #[target_feature(enable = "ssse3")]
    #[inline]
    fn sign4_ssse3(mag: __m128i, r: __m128i) -> __m128i {
        _mm_sign_epi32(mag, _mm_or_si128(r, _mm_set1_epi32(1)))
    }

    /// The exact `Q15` values of four `zq` as `i32` (see [`Q15Shift`]), with `k` its constants
    /// (`ms`, `k1`, the shift count, `k2`).
    #[target_feature(enable = "sse2")]
    #[inline]
    fn q15_4(zq: __m128i, k: Q15Consts128) -> __m128i {
        let zb = _mm_add_epi32(zq, _mm_set1_epi32(1 << 20));
        let e = _mm_mul_epu32(zb, k.ms);
        let o = _mm_mul_epu32(_mm_srli_epi64::<32>(zb), k.ms);
        let e = _mm_srl_epi64(_mm_add_epi64(e, k.k1), k.sh);
        let o = _mm_srl_epi64(_mm_add_epi64(o, k.k1), k.sh);
        _mm_sub_epi32(_mm_or_si128(e, _mm_slli_epi64::<32>(o)), k.k2)
    }

    /// The vector constants of [`Q15Shift`].
    #[derive(Clone, Copy)]
    struct Q15Consts128 {
        ms: __m128i,
        k1: __m128i,
        sh: __m128i,
        k2: __m128i,
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    fn q15_consts128(q: Q15Shift) -> Q15Consts128 {
        Q15Consts128 {
            ms: _mm_set1_epi64x(i64::from(q.ms)),
            k1: _mm_set1_epi64x(q.k1 as i64),
            sh: _mm_cvtsi32_si128(q.sh as i32),
            k2: _mm_set1_epi32(q.k2),
        }
    }

    /// The lane states of draws 0, 1 and 2, 3.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn states128(l: &Lanes<4>) -> (__m128i, __m128i) {
        let [s0, s1, s2, s3] = l.s;
        let v = |a: u64, b: u64| _mm_set_epi64x(b as i64, a as i64);
        (v(s0, s1), v(s2, s3))
    }

    /// The draws of the next block, their high words and their table indices; the states
    /// step.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn prepare4(e: &mut __m128i, o: &mut __m128i, j: Jump128) -> [__m128i; 3] {
        let r = draws4(*e, *o);
        *e = step128(*e, j);
        *o = step128(*o, j);
        let hi = high_words4(r);
        [r, hi, indices4(hi)]
    }

    /// The SSE2 and SSE4.2 paths: the same blocks, with the lookup, the interpolation and the
    /// sign of the level.
    macro_rules! sse_paths {
        ($token:ident, $f32:ident, $q15:ident, $lookup:ident, $interp:ident, $sign:ident) => {
            fearless_simd::kernel!(
                #[inline(always)]
                pub(crate) fn $f32(t: $token, rng: &mut Pcg32, scale: GaussScale, out: &mut [f32]) {
                    let scalar = |r| scale.f32_value(normal_q17(r));
                    drive::<f32, 4, 4>(rng, out, scalar, |l: &Lanes<4>, body: &mut [f32]| {
                        let j = jump128(l.a, l.c);
                        let (mut e, mut o) = states128(l);
                        let s17 = _mm_set1_ps(scale.s17());
                        pipeline::<_, _, _, 4, 1>(
                            body,
                            || prepare4(&mut e, &mut o, j),
                            |[r, hi, idx]: [__m128i; 3]| $sign($interp($lookup(idx), hi), r),
                            |[zq]: [__m128i; 1], chunk: &mut [f32]| {
                                f32x4::simd_from(t, _mm_mul_ps(_mm_cvtepi32_ps(zq), s17))
                                    .store_slice(chunk);
                            },
                        );
                    });
                }
            );

            fearless_simd::kernel!(
                #[inline(always)]
                pub(crate) fn $q15(t: $token, rng: &mut Pcg32, scale: GaussScale, out: &mut [i16]) {
                    let scalar = |r| scale.q15_value(normal_q17(r)).0;
                    let k = q15_consts128(Q15Shift::new(scale));
                    drive::<i16, 4, 8>(rng, out, scalar, |l: &Lanes<4>, body: &mut [i16]| {
                        let j = jump128(l.a, l.c);
                        let (mut e, mut o) = states128(l);
                        pipeline::<_, _, _, 8, 2>(
                            body,
                            || prepare4(&mut e, &mut o, j),
                            |[r, hi, idx]: [__m128i; 3]| {
                                q15_4($sign($interp($lookup(idx), hi), r), k)
                            },
                            |[m0, m1]: [__m128i; 2], chunk: &mut [i16]| {
                                i16x8::simd_from(t, _mm_packs_epi32(m0, m1)).store_slice(chunk);
                            },
                        );
                    });
                }
            );
        };
    }

    sse_paths!(
        Sse2,
        f32_sse2,
        q15_sse2,
        lookup4_sse2,
        interp4_sse2,
        sign4_sse2
    );
    sse_paths!(
        Sse4_2,
        f32_sse4_2,
        q15_sse4_2,
        lookup4_sse41,
        interp4_ssse3,
        sign4_ssse3
    );

    // -----------------------------------------------------------------------------------------
    // AVX2: four `u64` lanes per register, eight draws per block.

    #[derive(Clone, Copy)]
    struct Jump256 {
        a_lo: __m256i,
        a_hi: __m256i,
        c: __m256i,
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn jump256(a: u64, c: u64) -> Jump256 {
        Jump256 {
            a_lo: _mm256_set1_epi64x(a as i64),
            a_hi: _mm256_set1_epi64x((a >> 32) as i64),
            c: _mm256_set1_epi64x(c as i64),
        }
    }

    /// `a · s + c` per `u64` lane (three `vpmuludq`).
    #[target_feature(enable = "avx2")]
    #[inline]
    fn step256(s: __m256i, j: Jump256) -> __m256i {
        let lo = _mm256_mul_epu32(s, j.a_lo);
        let cross = _mm256_add_epi64(
            _mm256_mul_epu32(_mm256_srli_epi64::<32>(s), j.a_lo),
            _mm256_mul_epu32(s, j.a_hi),
        );
        _mm256_add_epi64(_mm256_add_epi64(lo, j.c), _mm256_slli_epi64::<32>(cross))
    }

    /// The lane states of draws 0 to 3 and 4 to 7.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn states256(l: &Lanes<8>) -> (__m256i, __m256i) {
        let [s0, s1, s2, s3, s4, s5, s6, s7] = l.s;
        let v = |a: u64, b: u64, c: u64, d: u64| {
            _mm256_setr_epi64x(a as i64, b as i64, c as i64, d as i64)
        };
        (v(s0, s1, s2, s3), v(s4, s5, s6, s7))
    }

    /// The high words of the `f64` form and the table indices (plus 32704) of the eight draws
    /// of the states `e` (draws 0 to 3) and `o` (draws 4 to 7), in draw order; the states step.
    ///
    /// The in-lane `vshufps` of the two state registers gives the draws in the order
    /// `0 1 4 5 | 2 3 6 7`; the conversions of the two halves to `f64` and the `vshufps` of
    /// their high words put them back in draw order. The sign travels in the `f64`: a negative
    /// draw is converted as `-(v + 1)` (`r ^ 0x7FFF_FFFF`), so the value is `-(v + 0.5)`.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn prepare8(e: &mut __m256i, o: &mut __m256i, j: Jump256) -> [__m256i; 2] {
        let (se, so) = (*e, *o);
        *e = step256(se, j);
        *o = step256(so, j);
        let xorshift =
            |s: __m256i| _mm256_srli_epi64::<27>(_mm256_xor_si256(_mm256_srli_epi64::<18>(s), s));
        let (xe, xo) = (
            _mm256_castsi256_ps(xorshift(se)),
            _mm256_castsi256_ps(xorshift(so)),
        );
        let xs = _mm256_castps_si256(_mm256_shuffle_ps::<0x88>(xe, xo));
        let (he, ho) = (_mm256_castsi256_ps(se), _mm256_castsi256_ps(so));
        let rot = _mm256_srli_epi32::<27>(_mm256_castps_si256(_mm256_shuffle_ps::<0xDD>(he, ho)));
        // A left shift by 32 gives 0, so `rot = 0` is right.
        let r = _mm256_or_si256(
            _mm256_srlv_epi32(xs, rot),
            _mm256_sllv_epi32(xs, _mm256_sub_epi32(_mm256_set1_epi32(32), rot)),
        );
        let sv = _mm256_xor_si256(r, _mm256_srli_epi32::<1>(_mm256_srai_epi32::<31>(r)));
        let half = _mm256_set1_pd(0.5);
        let a = _mm256_add_pd(_mm256_cvtepi32_pd(_mm256_castsi256_si128(sv)), half);
        let b = _mm256_add_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256::<1>(sv)), half);
        let hi = _mm256_castps_si256(_mm256_shuffle_ps::<0xDD>(
            _mm256_castpd_ps(a),
            _mm256_castpd_ps(b),
        ));
        let idx = _mm256_and_si256(_mm256_srli_epi32::<15>(hi), _mm256_set1_epi32(0xFFFF));
        [hi, idx]
    }

    /// `zq` of eight draws from their high words and indices: `vpgatherdd`, the interpolation
    /// and the sign.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn finish8([hi, idx]: [__m256i; 2]) -> __m256i {
        let base = gauss_icdf_table()
            .as_ptr()
            .cast::<i32>()
            .wrapping_sub(F64_INDEX_BIAS);
        // SAFETY: every index lies in `32704 ..= 33727` (the exponent field 1022 to 1053 and
        // the segment), so every address `base + 4 · idx` is one of the 1024 entries of the
        // table.
        let ent = unsafe { _mm256_i32gather_epi32::<4>(base, idx) };
        let slope = _mm256_and_si256(ent, _mm256_set1_epi32(0xFFF));
        let frac = _mm256_and_si256(hi, _mm256_set1_epi32(0x7FFF));
        let mag = _mm256_sub_epi32(
            _mm256_srli_epi32::<12>(ent),
            _mm256_mulhrs_epi16(slope, frac),
        );
        // The high word is never 0 (its exponent field is at least 1022).
        _mm256_sign_epi32(mag, hi)
    }

    /// The exact `Q15` values of eight `zq` as `i32` (see [`Q15Shift`]).
    #[target_feature(enable = "avx2")]
    #[inline]
    fn q15_8(zq: __m256i, k: Q15Consts256) -> __m256i {
        let zb = _mm256_add_epi32(zq, _mm256_set1_epi32(1 << 20));
        let e = _mm256_mul_epu32(zb, k.ms);
        let o = _mm256_mul_epu32(_mm256_srli_epi64::<32>(zb), k.ms);
        let e = _mm256_srl_epi64(_mm256_add_epi64(e, k.k1), k.sh);
        let o = _mm256_srl_epi64(_mm256_add_epi64(o, k.k1), k.sh);
        let m = _mm256_blend_epi32::<0xAA>(e, _mm256_slli_epi64::<32>(o));
        _mm256_sub_epi32(m, k.k2)
    }

    #[derive(Clone, Copy)]
    struct Q15Consts256 {
        ms: __m256i,
        k1: __m256i,
        sh: __m128i,
        k2: __m256i,
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn q15_consts256(q: Q15Shift) -> Q15Consts256 {
        Q15Consts256 {
            ms: _mm256_set1_epi64x(i64::from(q.ms)),
            k1: _mm256_set1_epi64x(q.k1 as i64),
            sh: _mm_cvtsi32_si128(q.sh as i32),
            k2: _mm256_set1_epi32(q.k2),
        }
    }

    fearless_simd::kernel!(
        #[inline(always)]
        pub(crate) fn f32_avx2(t: Avx2, rng: &mut Pcg32, scale: GaussScale, out: &mut [f32]) {
            let scalar = |r| scale.f32_value(normal_q17(r));
            drive::<f32, 8, 8>(rng, out, scalar, |l: &Lanes<8>, body: &mut [f32]| {
                let j = jump256(l.a, l.c);
                let (mut e, mut o) = states256(l);
                let s17 = _mm256_set1_ps(scale.s17());
                pipeline::<_, _, _, 8, 1>(
                    body,
                    || prepare8(&mut e, &mut o, j),
                    |p: [__m256i; 2]| finish8(p),
                    |[zq]: [__m256i; 1], chunk: &mut [f32]| {
                        f32x8::simd_from(t, _mm256_mul_ps(_mm256_cvtepi32_ps(zq), s17))
                            .store_slice(chunk);
                    },
                );
            });
        }
    );

    fearless_simd::kernel!(
        #[inline(always)]
        pub(crate) fn q15_avx2(t: Avx2, rng: &mut Pcg32, scale: GaussScale, out: &mut [i16]) {
            let scalar = |r| scale.q15_value(normal_q17(r)).0;
            let k = q15_consts256(Q15Shift::new(scale));
            drive::<i16, 8, 16>(rng, out, scalar, |l: &Lanes<8>, body: &mut [i16]| {
                let j = jump256(l.a, l.c);
                let (mut e, mut o) = states256(l);
                in_order::<_, _, _, 16, 2>(
                    body,
                    || prepare8(&mut e, &mut o, j),
                    |p: [__m256i; 2]| q15_8(finish8(p), k),
                    |[m0, m1]: [__m256i; 2], chunk: &mut [i16]| {
                        // The in-lane pack gives values 0 to 3, 8 to 11 | 4 to 7, 12 to 15.
                        let p = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(m0, m1));
                        i16x16::simd_from(t, p).store_slice(chunk);
                    },
                );
            });
        }
    );

    // -----------------------------------------------------------------------------------------
    // AVX-512: eight `u64` lanes per register, sixteen draws per block.

    /// The lane states of draws 0 to 7 and 8 to 15.
    #[target_feature(enable = "avx512f")]
    #[inline]
    fn states512(l: &Lanes<16>) -> (__m512i, __m512i) {
        let [
            s0,
            s1,
            s2,
            s3,
            s4,
            s5,
            s6,
            s7,
            s8,
            s9,
            s10,
            s11,
            s12,
            s13,
            s14,
            s15,
        ] = l.s;
        let v = |s: [u64; 8]| {
            let [a, b, c, d, e, f, g, h] = s.map(|x| x as i64);
            _mm512_setr_epi64(a, b, c, d, e, f, g, h)
        };
        (
            v([s0, s1, s2, s3, s4, s5, s6, s7]),
            v([s8, s9, s10, s11, s12, s13, s14, s15]),
        )
    }

    /// The jump of sixteen draws.
    #[derive(Clone, Copy)]
    struct Jump512 {
        a: __m512i,
        c: __m512i,
    }

    /// The draws of the states `e` (draws 0 to 7) and `o` (draws 8 to 15) in draw order, the
    /// table indices (minus 960) and the fractions of `x = 2r + 1`; the states step.
    /// `vpermt2d` collects the low and the high words, `vprorvd` rotates, and `vplzcntd` with
    /// `vpsllvd` gives the octave and the position of `x`.
    #[target_feature(enable = "avx512f,avx512bw,avx512cd,avx512dq")]
    #[inline]
    fn prepare16(e: &mut __m512i, o: &mut __m512i, j: Jump512) -> [__m512i; 3] {
        let (se, so) = (*e, *o);
        *e = _mm512_add_epi64(_mm512_mullo_epi64(se, j.a), j.c);
        *o = _mm512_add_epi64(_mm512_mullo_epi64(so, j.a), j.c);
        let xorshift =
            |s: __m512i| _mm512_srli_epi64::<27>(_mm512_xor_si512(_mm512_srli_epi64::<18>(s), s));
        let low = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let high = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        let xs = _mm512_permutex2var_epi32(xorshift(se), low, xorshift(so));
        let rot = _mm512_srli_epi32::<27>(_mm512_permutex2var_epi32(se, high, so));
        let r = _mm512_rorv_epi32(xs, rot);
        let x = _mm512_or_si512(_mm512_add_epi32(r, r), _mm512_set1_epi32(1));
        let c = _mm512_lzcnt_epi32(x);
        let n = _mm512_sllv_epi32(x, c);
        // `32 L + sub = (n >> 26) - 32 c + 960`; the 960 is in the base of the gather.
        let idx = _mm512_sub_epi32(_mm512_srli_epi32::<26>(n), _mm512_slli_epi32::<5>(c));
        let frac = _mm512_and_si512(_mm512_srli_epi32::<11>(n), _mm512_set1_epi32(0x7FFF));
        [r, idx, frac]
    }

    /// `zq` of sixteen draws from their indices and fractions: `vpgatherdd`, the interpolation
    /// and the sign (a masked subtraction).
    #[target_feature(enable = "avx512f,avx512bw,avx512cd,avx512dq")]
    #[inline]
    fn finish16([r, idx, frac]: [__m512i; 3]) -> __m512i {
        let base = gauss_icdf_table().as_ptr().cast::<i32>().wrapping_add(960);
        // SAFETY: every index lies in `-960 ..= 63` (`c` in 0 to 31, `n >> 26` in 32 to 63),
        // so every address `base + 4 · idx` is one of the 1024 entries of the table.
        let ent = unsafe { _mm512_i32gather_epi32::<4>(idx, base) };
        let slope = _mm512_and_si512(ent, _mm512_set1_epi32(0xFFF));
        let mag = _mm512_sub_epi32(
            _mm512_srli_epi32::<12>(ent),
            _mm512_mulhrs_epi16(slope, frac),
        );
        _mm512_mask_sub_epi32(mag, _mm512_movepi32_mask(r), _mm512_setzero_si512(), mag)
    }

    fearless_simd::kernel!(
        #[inline(always)]
        pub(crate) fn f32_avx512(t: Avx512, rng: &mut Pcg32, scale: GaussScale, out: &mut [f32]) {
            let scalar = |r| scale.f32_value(normal_q17(r));
            drive::<f32, 16, 16>(rng, out, scalar, |l: &Lanes<16>, body: &mut [f32]| {
                let j = Jump512 {
                    a: _mm512_set1_epi64(l.a as i64),
                    c: _mm512_set1_epi64(l.c as i64),
                };
                let (mut e, mut o) = states512(l);
                let s17 = _mm512_set1_ps(scale.s17());
                pipeline::<_, _, _, 16, 1>(
                    body,
                    || prepare16(&mut e, &mut o, j),
                    |p: [__m512i; 3]| finish16(p),
                    |[zq]: [__m512i; 1], chunk: &mut [f32]| {
                        f32x16::simd_from(t, _mm512_mul_ps(_mm512_cvtepi32_ps(zq), s17))
                            .store_slice(chunk);
                    },
                );
            });
        }
    );

    fearless_simd::kernel!(
        #[inline(always)]
        pub(crate) fn q15_avx512(t: Avx512, rng: &mut Pcg32, scale: GaussScale, out: &mut [i16]) {
            let scalar = |r| scale.q15_value(normal_q17(r)).0;
            let q = Q15Shift::new(scale);
            let ms = _mm512_set1_epi64(i64::from(q.ms));
            let half = _mm512_set1_epi64(q.half as i64);
            let sh = _mm_cvtsi32_si128(q.sh as i32);
            drive::<i16, 16, 16>(rng, out, scalar, |l: &Lanes<16>, body: &mut [i16]| {
                let j = Jump512 {
                    a: _mm512_set1_epi64(l.a as i64),
                    c: _mm512_set1_epi64(l.c as i64),
                };
                let (mut e, mut o) = states512(l);
                in_order::<_, _, _, 16, 1>(
                    body,
                    || prepare16(&mut e, &mut o, j),
                    |p: [__m512i; 3]| finish16(p),
                    |[zq]: [__m512i; 1], chunk: &mut [i16]| {
                        // `floor((zq · ms + half) / 2^sh)` with the signed product and shift.
                        let pe =
                            _mm512_sra_epi64(_mm512_add_epi64(_mm512_mul_epi32(zq, ms), half), sh);
                        let po = _mm512_sra_epi64(
                            _mm512_add_epi64(
                                _mm512_mul_epi32(_mm512_srli_epi64::<32>(zq), ms),
                                half,
                            ),
                            sh,
                        );
                        let m = _mm512_mask_blend_epi32(0xAAAA, pe, _mm512_slli_epi64::<32>(po));
                        i16x16::simd_from(t, _mm512_cvtsepi32_epi16(m)).store_slice(chunk);
                    },
                );
            });
        }
    );
}

/// The path of NEON: four lanes per block, `clz` and variable shifts for the octave, four lane
/// loads of the table and `sqrdmulh`. NEON has no 64-bit lane product, so the LCG has two forms
/// with the same bits, chosen per output type by `cargo measure` (cycles per value, reference
/// `cortex-a72` / counter `cortex-a76`):
///
/// - `f32` (and `C32`): the LCG on the split low and high words of the states (`umlal` and
///   `mla`) and the output function in NEON (`ushl` for the rotation): 8.04 / 7.07, against
///   8.62 / 6.83 in the hybrid form (7 % slower on the reference model, 3 % faster on the
///   counter-model; the sum decides).
/// - `Q15` (and `CQ15`): the hybrid form, four LCG states in general registers (`madd`) and the
///   output function in scalar code, then the inverse CDF in NEON: 7.88 / 6.80, against 7.94 /
///   7.05 with the vector LCG (faster on both models).
#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::{
        int32x4_t, uint32x4_t, uint64x2_t, vaddq_u32, vandq_u32, vclzq_u32, vcvtq_f32_s32,
        vdupq_n_f32, vdupq_n_s32, vdupq_n_s64, vdupq_n_u32, vdupq_n_u64, veorq_s32, veorq_u32,
        vget_low_s32, vget_low_u32, vgetq_lane_u32, vmlal_high_s32, vmlal_high_u32, vmlal_s32,
        vmlal_u32, vmlaq_u32, vmulq_f32, vmulq_u32, vnegq_s32, vorrq_u32, vqmovn_high_s32,
        vqmovn_s32, vqrdmulhq_s16, vreinterpretq_s16_u32, vreinterpretq_s32_s64,
        vreinterpretq_s32_u32, vreinterpretq_u32_s16, vreinterpretq_u32_u64, vshlq_s64, vshlq_u32,
        vshrq_n_s32, vshrq_n_u32, vsliq_n_u32, vsubq_s32, vsubq_u32, vuzp1q_s32, vuzp1q_u32,
        vuzp2q_u32,
    };

    use fearless_simd::{Neon, f32x4, i16x8, prelude::*, u32x4};
    use sdrk_core::kernels::noise::{GaussScale, Pcg32, normal_q17};

    use super::{Lanes, Q15Shift, drive, entry, pipeline};

    /// Four LCG states as their low and high words.
    #[derive(Clone, Copy)]
    struct Split {
        lo: uint32x4_t,
        hi: uint32x4_t,
    }

    /// The jump of four draws: the words of `a`, and `c`.
    #[derive(Clone, Copy)]
    struct Jump {
        a_lo: uint32x4_t,
        a_hi: uint32x4_t,
        c: uint64x2_t,
    }

    /// The lane states and the jump of a block.
    #[target_feature(enable = "neon")]
    #[inline]
    fn start(t: Neon, l: &Lanes<4>) -> (Split, Jump) {
        let lo = l.s.map(|s| s as u32);
        let hi = l.s.map(|s| (s >> 32) as u32);
        (
            Split {
                lo: u32x4::from_slice(t, &lo).into(),
                hi: u32x4::from_slice(t, &hi).into(),
            },
            Jump {
                a_lo: vdupq_n_u32(l.a as u32),
                a_hi: vdupq_n_u32((l.a >> 32) as u32),
                c: vdupq_n_u64(l.c),
            },
        )
    }

    /// `a · s + c` per state: `c + lo · a_lo` widened (`umlal`, `umlal2`), the cross terms
    /// `hi · a_lo + lo · a_hi` modulo 2^32 added to the high words.
    #[target_feature(enable = "neon")]
    #[inline]
    fn step(s: Split, j: Jump) -> Split {
        let pe = vreinterpretq_u32_u64(vmlal_u32(j.c, vget_low_u32(s.lo), vget_low_u32(j.a_lo)));
        let po = vreinterpretq_u32_u64(vmlal_high_u32(j.c, s.lo, j.a_lo));
        let cross = vmlaq_u32(vmulq_u32(s.hi, j.a_lo), s.lo, j.a_hi);
        Split {
            lo: vuzp1q_u32(pe, po),
            hi: vaddq_u32(vuzp2q_u32(pe, po), cross),
        }
    }

    /// The four draws of the states: `xorshifted = (s >> 27) ^ (s >> 45)` in 32 bits from the
    /// words, rotated right by `s >> 59` (a left shift by 32 gives 0, so `rot = 0` is right).
    #[target_feature(enable = "neon")]
    #[inline]
    fn draws(s: Split) -> uint32x4_t {
        let xs = veorq_u32(
            vsliq_n_u32::<5>(vshrq_n_u32::<27>(s.lo), s.hi),
            vshrq_n_u32::<13>(s.hi),
        );
        let rot = vreinterpretq_s32_u32(vshrq_n_u32::<27>(s.hi));
        vorrq_u32(
            vshlq_u32(xs, vnegq_s32(rot)),
            vshlq_u32(xs, vsubq_s32(vdupq_n_s32(32), rot)),
        )
    }

    /// The first half of the inverse CDF of four draws: the table indices, the fractions and
    /// the signs.
    #[derive(Clone, Copy)]
    struct Prepared {
        idx: uint32x4_t,
        frac: uint32x4_t,
        sign: int32x4_t,
    }

    /// The draws of the next block and its first half; the generator steps.
    #[target_feature(enable = "neon")]
    #[inline]
    fn prepare(s: &mut Split, j: Jump) -> Prepared {
        let r = draws(*s);
        *s = step(*s, j);
        transform(r)
    }

    /// The hybrid form: the four LCG states in general registers (`madd`) and the output
    /// function in scalar code, the inverse CDF in NEON.
    #[target_feature(enable = "neon")]
    #[inline]
    fn prepare_scalar(t: Neon, s: &mut [u64; 4], a: u64, c: u64) -> Prepared {
        let r = s.map(sdrk_core::kernels::noise::pcg32_output);
        *s = s.map(|x| x.wrapping_mul(a).wrapping_add(c));
        transform(u32x4::from_slice(t, &r).into())
    }

    /// The first half of the inverse CDF of the draws `r`.
    #[target_feature(enable = "neon")]
    #[inline]
    fn transform(r: uint32x4_t) -> Prepared {
        let x = vsliq_n_u32::<1>(vdupq_n_u32(1), r);
        let c = vclzq_u32(x);
        let n = vshlq_u32(x, vreinterpretq_s32_u32(c));
        Prepared {
            // `(L << 5) | sub` with `L = 31 - c`.
            idx: vsliq_n_u32::<5>(vshrq_n_u32::<26>(n), vsubq_u32(vdupq_n_u32(31), c)),
            frac: vandq_u32(vshrq_n_u32::<11>(n), vdupq_n_u32(0x7FFF)),
            sign: vshrq_n_s32::<31>(vreinterpretq_s32_u32(r)),
        }
    }

    /// The second half: the table entries, the interpolation and the sign, `zq` of the block.
    #[target_feature(enable = "neon")]
    #[inline]
    fn finish(t: Neon, p: Prepared) -> int32x4_t {
        let e = [
            entry(vgetq_lane_u32::<0>(p.idx)),
            entry(vgetq_lane_u32::<1>(p.idx)),
            entry(vgetq_lane_u32::<2>(p.idx)),
            entry(vgetq_lane_u32::<3>(p.idx)),
        ];
        let ent: uint32x4_t = u32x4::from_slice(t, &e).into();
        let slope = vreinterpretq_s16_u32(vandq_u32(ent, vdupq_n_u32(0xFFF)));
        let prod = vreinterpretq_u32_s16(vqrdmulhq_s16(slope, vreinterpretq_s16_u32(p.frac)));
        let mag = vreinterpretq_s32_u32(vsubq_u32(vshrq_n_u32::<12>(ent), prod));
        vsubq_s32(veorq_s32(mag, p.sign), p.sign)
    }

    /// Runs `emit` on every chunk of `C = 4 K` values of `body` with `zq` of its `K` blocks
    /// ([`pipeline`]), with the LCG in general registers (`SCALAR_LCG`, the hybrid form) or in
    /// NEON.
    #[target_feature(enable = "neon")]
    #[inline]
    fn blocks<V, const C: usize, const K: usize, const SCALAR_LCG: bool>(
        t: Neon,
        l: &Lanes<4>,
        body: &mut [V],
        emit: impl FnMut([int32x4_t; K], &mut [V]),
    ) {
        if SCALAR_LCG {
            let mut s = l.s;
            let (a, c) = (l.a, l.c);
            pipeline::<_, _, _, C, K>(
                body,
                || prepare_scalar(t, &mut s, a, c),
                |p| finish(t, p),
                emit,
            );
            return;
        }
        let (mut s, j) = start(t, l);
        pipeline::<_, _, _, C, K>(body, || prepare(&mut s, j), |p| finish(t, p), emit);
    }

    fearless_simd::kernel!(
        #[inline(always)]
        pub(crate) fn f32_neon(t: Neon, rng: &mut Pcg32, scale: GaussScale, out: &mut [f32]) {
            let scalar = |r| scale.f32_value(normal_q17(r));
            drive::<f32, 4, 4>(rng, out, scalar, |l: &Lanes<4>, body: &mut [f32]| {
                let s17 = vdupq_n_f32(scale.s17());
                blocks::<_, 4, 1, false>(t, l, body, |[zq], chunk: &mut [f32]| {
                    let v = vmulq_f32(vcvtq_f32_s32(zq), s17);
                    f32x4::simd_from(t, v).store_slice(chunk);
                });
            });
        }
    );

    fearless_simd::kernel!(
        #[inline(always)]
        pub(crate) fn q15_neon(t: Neon, rng: &mut Pcg32, scale: GaussScale, out: &mut [i16]) {
            let scalar = |r| scale.q15_value(normal_q17(r)).0;
            let q = Q15Shift::new(scale);
            let ms = vdupq_n_s32(q.ms as i32);
            let half = vdupq_n_s64(q.half as i64);
            let shift = vdupq_n_s64(-i64::from(q.sh));
            // `floor((zq · ms + half) / 2^sh)` with the signed product and shift.
            let exact = |zq: int32x4_t| {
                let pe = vshlq_s64(vmlal_s32(half, vget_low_s32(zq), vget_low_s32(ms)), shift);
                let po = vshlq_s64(vmlal_high_s32(half, zq, ms), shift);
                vuzp1q_s32(vreinterpretq_s32_s64(pe), vreinterpretq_s32_s64(po))
            };
            drive::<i16, 4, 8>(rng, out, scalar, |l: &Lanes<4>, body: &mut [i16]| {
                blocks::<_, 8, 2, true>(t, l, body, |[z0, z1], chunk: &mut [i16]| {
                    let p = vqmovn_high_s32(vqmovn_s32(exact(z0)), exact(z1));
                    i16x8::simd_from(t, p).store_slice(chunk);
                });
            });
        }
    );
}

/// The path of SIMD128: four lanes per block, `i64x2.mul` for the LCG, the rotation as the
/// product with `2^(31 - rot)` (`u64x2.extmul`), the `f64` form of the octave (SIMD128 has no
/// `clz` and no variable shift per lane), four lane loads of the table and `q15mulr`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::{
        f32x4_convert_i32x4, f32x4_mul, f32x4_splat, f64x2_add, f64x2_convert_low_i32x4,
        f64x2_splat, i16x8_narrow_i32x4, i16x8_q15mulr_sat, i32x4_add, i32x4_shl, i32x4_shr,
        i32x4_shuffle, i32x4_splat, i32x4_sub, i64x2, i64x2_add, i64x2_extmul_high_i32x4,
        i64x2_extmul_low_i32x4, i64x2_mul, i64x2_shr, i64x2_splat, u32x4, u32x4_extract_lane,
        u32x4_shr, u32x4_splat, u32x4_trunc_sat_f32x4, u64x2_extmul_high_u32x4,
        u64x2_extmul_low_u32x4, u64x2_shl, u64x2_shr, v128, v128_and, v128_or, v128_xor,
    };

    use fearless_simd::{f32x4, i16x8, prelude::*};
    use sdrk_core::kernels::noise::{GaussScale, Pcg32, normal_q17};

    use super::{Lanes, Q15Shift, drive, entry};

    /// The states of draws 0, 1 and 2, 3.
    #[inline(always)]
    fn states(l: &Lanes<4>) -> (v128, v128) {
        let [s0, s1, s2, s3] = l.s.map(|s| s as i64);
        (i64x2(s0, s1), i64x2(s2, s3))
    }

    /// `a · s + c` per `u64` lane.
    #[inline(always)]
    fn step(s: v128, a: v128, c: v128) -> v128 {
        i64x2_add(i64x2_mul(s, a), c)
    }

    /// The four draws of the states `e` (draws 0, 1) and `o` (draws 2, 3), in draw order: the
    /// rotation by `rot` is the product `y = xs · 2^(31 - rot)` (the factor from the `f32`
    /// exponent field) and the low word of `(y << 1) | (y >> 31)`.
    #[inline(always)]
    fn draws(e: v128, o: v128) -> v128 {
        let xorshift = |s: v128| u64x2_shr(v128_xor(u64x2_shr(s, 18), s), 27);
        let xs = i32x4_shuffle::<0, 2, 4, 6>(xorshift(e), xorshift(o));
        let rot = u32x4_shr(i32x4_shuffle::<1, 3, 5, 7>(e, o), 27);
        let m = u32x4_trunc_sat_f32x4(i32x4_shl(i32x4_sub(i32x4_splat(158), rot), 23));
        let rotate = |y: v128| v128_or(u64x2_shl(y, 1), u64x2_shr(y, 31));
        let lo = rotate(u64x2_extmul_low_u32x4(xs, m));
        let hi = rotate(u64x2_extmul_high_u32x4(xs, m));
        i32x4_shuffle::<0, 2, 4, 6>(lo, hi)
    }

    /// `zq` of four draws, through the high words of the `f64` values
    /// `(r & 0x7FFF_FFFF) + 0.5`.
    #[inline(always)]
    fn zq4(r: v128) -> v128 {
        let v = v128_and(r, u32x4_splat(0x7FFF_FFFF));
        let half = f64x2_splat(0.5);
        let a = f64x2_add(f64x2_convert_low_i32x4(v), half);
        let b = f64x2_add(
            f64x2_convert_low_i32x4(i32x4_shuffle::<2, 3, 2, 3>(v, v)),
            half,
        );
        let hi = i32x4_shuffle::<1, 3, 5, 7>(a, b);
        // The index plus 32768, a multiple of 1024 that `entry` masks away.
        let idx = i32x4_add(u32x4_shr(hi, 15), u32x4_splat(64));
        let ent = u32x4(
            entry(u32x4_extract_lane::<0>(idx)),
            entry(u32x4_extract_lane::<1>(idx)),
            entry(u32x4_extract_lane::<2>(idx)),
            entry(u32x4_extract_lane::<3>(idx)),
        );
        let slope = v128_and(ent, u32x4_splat(0xFFF));
        let frac = v128_and(hi, u32x4_splat(0x7FFF));
        let mag = i32x4_sub(u32x4_shr(ent, 12), i16x8_q15mulr_sat(slope, frac));
        let s = i32x4_shr(r, 31);
        i32x4_sub(v128_xor(mag, s), s)
    }

    fearless_simd::kernel!(
        #[inline(always)]
        pub(crate) fn f32_simd128(
            t: WasmSimd128,
            rng: &mut Pcg32,
            scale: GaussScale,
            out: &mut [f32],
        ) {
            let scalar = |r| scale.f32_value(normal_q17(r));
            drive::<f32, 4, 4>(rng, out, scalar, |l: &Lanes<4>, body: &mut [f32]| {
                let (a, c) = (i64x2_splat(l.a as i64), i64x2_splat(l.c as i64));
                let (mut e, mut o) = states(l);
                let s17 = f32x4_splat(scale.s17());
                for chunk in body.chunks_exact_mut(4) {
                    let r = draws(e, o);
                    e = step(e, a, c);
                    o = step(o, a, c);
                    let v = f32x4_mul(f32x4_convert_i32x4(zq4(r)), s17);
                    f32x4::simd_from(t, v).store_slice(chunk);
                }
            });
        }
    );

    fearless_simd::kernel!(
        #[inline(always)]
        pub(crate) fn q15_simd128(
            t: WasmSimd128,
            rng: &mut Pcg32,
            scale: GaussScale,
            out: &mut [i16],
        ) {
            let scalar = |r| scale.q15_value(normal_q17(r)).0;
            let q = Q15Shift::new(scale);
            let ms = i32x4_splat(q.ms as i32);
            let half = i64x2_splat(q.half as i64);
            drive::<i16, 4, 8>(rng, out, scalar, |l: &Lanes<4>, body: &mut [i16]| {
                let (a, c) = (i64x2_splat(l.a as i64), i64x2_splat(l.c as i64));
                let (mut e, mut o) = states(l);
                // `floor((zq · ms + half) / 2^sh)` with the signed product and shift.
                let mut block = || {
                    let r = draws(e, o);
                    e = step(e, a, c);
                    o = step(o, a, c);
                    let zq = zq4(r);
                    let pe = i64x2_shr(i64x2_add(i64x2_extmul_low_i32x4(zq, ms), half), q.sh);
                    let po = i64x2_shr(i64x2_add(i64x2_extmul_high_i32x4(zq, ms), half), q.sh);
                    i32x4_shuffle::<0, 2, 4, 6>(pe, po)
                };
                for chunk in body.chunks_exact_mut(8) {
                    let (m0, m1) = (block(), block());
                    i16x8::simd_from(t, i16x8_narrow_i32x4(m0, m1)).store_slice(chunk);
                }
            });
        }
    );
}
