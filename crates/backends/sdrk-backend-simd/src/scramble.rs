//! `scramble` (M49 §5, revised by M49.1 §3): keyed phase scrambling, bit-identical to the
//! fallback on every level.
//!
//! Every lane runs the element formula of the fallback, so the bits follow from the arithmetic.
//!
//! **`b ≤ 2`: the exact quadrant map** (unchanged since M49).
//!
//! 1. *Indices.* The phase bytes are widened to one lane per sample (`pmovzxbd`/`pmovzxbq`,
//!    unpacks on SSE2, `u32x4.extend` on SIMD128) or kept as bytes (NEON); `k = ±p` (`psignd`,
//!    a masked subtraction on AVX-512, `(p ^ m) − m` with `m = −1` for `inverse` elsewhere) and
//!    the quadrant `q` in the low two bits of `(k << 1) >> (b − 1)`. The bits of `p` above `b`
//!    never reach `q`, so `k` needs no mask by `2^b − 1`.
//! 2. *Quadrant.* Swap where `q & 1` (a shuffle and a blend, a masked rotate on AVX-512 `CQ15`, a
//!    blend of the planar `re` and `im` for `C32`), then negate `re` where `(q ^ (q >> 1)) & 1`
//!    and `im` where `(q >> 1) & 1`: `(x ^ m) −sat m` in `CQ15` (32767 for −32768), XOR with the
//!    sign bit in `C32`. The masks come from a table of the four quadrants (`vpermd`, `pshufb`,
//!    `tbl`, `i8x16.swizzle`) or from shifts of `q`.
//!
//! **`b ≥ 3`: one table phasor and one product per sample** (M49.1; no quadrant map, no
//! selection).
//!
//! 1. *Index.* `k = (±p) & (2^b − 1)`.
//! 2. *Phasor.* The `2^b` phasors of `b` phase bits sit at the offset `2^b` of static tables
//!    built at compile time from `COS_F32`/`COS_Q15` of `sdrk-core` (`c = COS[k << (12 − b)]`,
//!    `s = COS[((k << (12 − b)) + 3072) & 4095]`, the entries of the fallback; 512 entries per
//!    table, entries 0 to 7 unused). The per-sample `(c, s)` come, by level and `b` (the forms
//!    chosen by simulated cycles, M49.1 §3, `docs/performance.md`): from an in-register permute
//!    of the `2^b` entries (`vpermd`/`vpermps` up to 8 entries, two and a blend up to 16 on AVX2;
//!    `vpermd`, `vpermi2d`, `vpermq`, `vpermi2q` and blends up to 64 on AVX-512; `tbl`/`tbx` on
//!    NEON up to 16 `C32` and 64 `CQ15` entries), from gathers (AVX-512 above 64 entries; `C32`
//!    then sixteen samples in the planar form), or from one load of the pair per sample (SSE2,
//!    SSE4.2, AVX2 above 16 entries, NEON lane loads above the `tbl` tables, SIMD128). The AVX2
//!    gather form was measured and is slower than the fallback on `alderlake`.
//! 3. *Product.* `CQ15`: `pmaddwd` of the `(re, im)` pairs with `(c, !s)` and `(s, c)`
//!    (`i32x4.dot_i16x8_s`), where `!s = −s − 1` cannot overflow (`s` is −32768 at the
//!    three-quarter turn) and `xi` (an arithmetic shift of the pair by 16) is added back with the
//!    rounding constant `2^14`, then an arithmetic shift by 15 and the saturating pack; on NEON
//!    `smull`/`smlsl`/`smlal` on the planar form and `sqrshrn #15`: the `cq15_mul` of the
//!    fallback, exact in `i32`. `C32`: `(re c − im s, re s + im c)` by two products and one sum
//!    per component in the order of the fallback (planar on every level but AVX-512, which forms
//!    the components as `re c + im (−s)` and `im c + re s`, the same values for every non-NaN
//!    result; NaN by class, D296), no fused multiply-add.
//!
//! A remainder shorter than one vector step runs `scramble_one_c32`/`scramble_one_cq15`.
// The helpers of the level paths are unused on a target without one.
#![cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code, unused_imports, unused_variables)
)]

use fearless_simd::Simd as SimdToken;
use sdrk_core::kernels::scramble::{
    Scramble, check_args, phase_index, scramble_one_c32, scramble_one_cq15,
};
use sdrk_core::tables::{COS_F32, COS_Q15};
use sdrk_core::{C32, CQ15, Result};

use crate::Simd;
use crate::level::dispatch_fn;

/// The entries of the static phasor tables: the `2^b` phasors of `b` phase bits (3 to 8) at
/// the offset `2^b`.
const ENTRIES: usize = 512;

/// The cosine-table indices `(i, (i + 3072) & 4095)` of the entry `e = 2^b + k` (`e ≥ 8`):
/// `i = k << (12 − b)`, the cosine and the sine of the phasor of `k` (M49.1 §1).
const fn table_indices(e: usize) -> (usize, usize) {
    let bits = (usize::BITS - 1 - e.leading_zeros()) as usize;
    let i = (e - (1 << bits)) << (12 - bits);
    (i, (i + 3072) & 4095)
}

/// The cosines `c` of the phasors of every `b` (3 to 8) at the offset `2^b`.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
// Evaluated at compile time: an index out of range is a build error, never a panic at run time.
#[allow(clippy::indexing_slicing)]
static C_F32: [f32; ENTRIES] = {
    let mut t = [0.0; ENTRIES];
    let mut e = 8;
    while e < ENTRIES {
        t[e] = COS_F32[table_indices(e).0];
        e += 1;
    }
    t
};

/// The sines `s` of the phasors, as [`C_F32`].
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[allow(clippy::indexing_slicing)]
static S_F32: [f32; ENTRIES] = {
    let mut t = [0.0; ENTRIES];
    let mut e = 8;
    while e < ENTRIES {
        t[e] = COS_F32[table_indices(e).1];
        e += 1;
    }
    t
};

/// The `C32` phasors as one word `bits(c) | bits(s) << 32` per entry (the per-sample loads and
/// the AVX-512 tables).
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
#[allow(clippy::indexing_slicing)]
static CS_C32: [u64; ENTRIES] = {
    let mut t = [0; ENTRIES];
    let mut e = 8;
    while e < ENTRIES {
        let (i, j) = table_indices(e);
        t[e] = COS_F32[i].to_bits() as u64 | ((COS_F32[j].to_bits() as u64) << 32);
        e += 1;
    }
    t
};

/// The `CQ15` `pmaddwd` operand `(c, !s)` (low half first; `!s = −s − 1`) per entry.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[allow(clippy::indexing_slicing)]
static A_Q15: [u32; ENTRIES] = {
    let mut t = [0; ENTRIES];
    let mut e = 8;
    while e < ENTRIES {
        let (i, j) = table_indices(e);
        t[e] = (COS_Q15[i] as u16) as u32 | (((!COS_Q15[j]) as u16 as u32) << 16);
        e += 1;
    }
    t
};

/// The `CQ15` `pmaddwd` operand `(s, c)` per entry.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[allow(clippy::indexing_slicing)]
static B_Q15: [u32; ENTRIES] = {
    let mut t = [0; ENTRIES];
    let mut e = 8;
    while e < ENTRIES {
        let (i, j) = table_indices(e);
        t[e] = (COS_Q15[j] as u16) as u32 | ((COS_Q15[i] as u16 as u32) << 16);
        e += 1;
    }
    t
};

/// Both `CQ15` operands as one word `a | b << 32` per entry (the per-sample loads of x86).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[allow(clippy::indexing_slicing)]
static AB_Q15: [u64; ENTRIES] = {
    let mut t = [0; ENTRIES];
    let mut e = 8;
    while e < ENTRIES {
        t[e] = A_Q15[e] as u64 | ((B_Q15[e] as u64) << 32);
        e += 1;
    }
    t
};

/// The `CQ15` phasor as `c | s << 16` per entry (the planar per-sample loads of NEON).
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
#[allow(clippy::indexing_slicing)]
static CS_Q15: [u32; ENTRIES] = {
    let mut t = [0; ENTRIES];
    let mut e = 8;
    while e < ENTRIES {
        let (i, j) = table_indices(e);
        t[e] = (COS_Q15[i] as u16) as u32 | ((COS_Q15[j] as u16 as u32) << 16);
        e += 1;
    }
    t
};

/// The entry `e` of a static table (0 outside the table, which a valid entry never is).
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
fn at<T: Copy + Default>(t: &[T; ENTRIES], e: usize) -> T {
    t.get(e).copied().unwrap_or_default()
}

/// The constants of one call.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Plan {
    /// All ones for `inverse` (`k = (p ^ m) − m`), else 0.
    neg: u32,
    /// `b − 1`: `q = ((k << 1) >> (b − 1)) & 3` (`b ≤ 2`).
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    qshift: u32,
    /// `b`.
    bits: u32,
    /// `2^b − 1`: `k = (±p) & mask` (`b ≥ 3`).
    mask: u32,
    /// `2^b`: the offset of the phasors of `b` in the static tables.
    base: u32,
}

impl Plan {
    fn new(bits: u32, inverse: bool) -> Self {
        Self {
            neg: if inverse { u32::MAX } else { 0 },
            qshift: bits - 1,
            bits,
            mask: (1 << bits) - 1,
            base: 1 << bits,
        }
    }

    /// Whether the call takes the phasor and the product (`b ≥ 3`), not the quadrant map.
    #[inline(always)]
    fn phasor(&self) -> bool {
        self.bits >= 3
    }

    /// The number of phasors `2^b`.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    #[inline(always)]
    fn entries(&self) -> usize {
        self.base as usize
    }

    /// The table entry `2^b + ((±p) & (2^b − 1))` of the phase byte `p` (scalar, for the
    /// per-sample loads).
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    #[inline(always)]
    fn entry(&self, p: u8) -> usize {
        ((((u32::from(p) ^ self.neg).wrapping_sub(self.neg)) & self.mask) | self.base) as usize
            & (ENTRIES - 1)
    }

    /// The `2^b` entries of a static table for this call (the in-register tables).
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    #[inline(always)]
    fn slice<T>(&self, t: &'static [T; ENTRIES]) -> &'static [T] {
        let b = self.base as usize;
        t.get(b..2 * b).unwrap_or(&[])
    }
}

/// The level path for `CQ15` (samples as `i16` pairs, `re` first; `i16` because a `CQ15` is
/// only 2-byte aligned): returns the number of leading samples done.
#[inline(always)]
fn cq15_level<S: SimdToken>(simd: S, x: &[i16], p: &[u8], y: &mut [i16], plan: &Plan) -> usize {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx512() {
            return x86::cq15_avx512(t, x, p, y, plan);
        }
        if let Some(t) = level.as_avx2() {
            return x86::cq15_avx2(t, x, p, y, plan);
        }
        if let Some(t) = level.as_sse4_2() {
            return x86::cq15_sse4_2(t, x, p, y, plan);
        }
        if let Some(t) = level.as_sse2() {
            return x86::cq15_sse2(t, x, p, y, plan);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return neon::cq15(t, x, p, y, plan);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return wasm::cq15(t, x, p, y, plan);
    }
    let _ = (level, x, p, y, plan);
    0
}

/// The level path for `C32` (samples as two `f32`): returns the number of leading samples done.
#[inline(always)]
fn c32_level<S: SimdToken>(simd: S, x: &[f32], p: &[u8], y: &mut [f32], plan: &Plan) -> usize {
    let level = simd.level();
    // `b = 1`: `k = p & 1` in both directions, the binary chip of the shared conventions §A.1
    // (both sign bits flipped where `p & 1`, the bits of the quadrant map with `q = 2 k`; M49.1
    // §5, `scramble_c32_b1`). SSE2, SSE4.2, AVX2, NEON and SIMD128 have a path of their own;
    // AVX-512 takes the lane-mask helper of `sign.rs` at its native width.
    #[cfg(target_arch = "x86_64")]
    if plan.bits == 1 && level.as_avx512().is_none() {
        if let Some(t) = level.as_avx2() {
            return x86::c32_avx2(t, x, p, y, plan);
        }
        if let Some(t) = level.as_sse4_2() {
            return x86::c32_sse4_2(t, x, p, y, plan);
        }
        if let Some(t) = level.as_sse2() {
            return x86::c32_sse2(t, x, p, y, plan);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if plan.bits == 1
        && let Some(t) = level.as_neon()
    {
        return neon::c32(t, x, p, y, plan);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if plan.bits == 1
        && let Some(t) = level.as_wasm_simd128()
    {
        return wasm::c32(t, x, p, y, plan);
    }
    if plan.bits == 1 {
        return crate::sign::c32_chip_level(
            simd,
            bytemuck::cast_slice(x),
            p,
            bytemuck::cast_slice_mut(y),
        );
    }
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx512() {
            return x86::c32_avx512(t, x, p, y, plan);
        }
        if let Some(t) = level.as_avx2() {
            return x86::c32_avx2(t, x, p, y, plan);
        }
        if let Some(t) = level.as_sse4_2() {
            return x86::c32_sse4_2(t, x, p, y, plan);
        }
        if let Some(t) = level.as_sse2() {
            return x86::c32_sse2(t, x, p, y, plan);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return neon::c32(t, x, p, y, plan);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return wasm::c32(t, x, p, y, plan);
    }
    let _ = (level, x, p, y, plan);
    0
}

#[inline(always)]
fn cq15_run<S: SimdToken>(
    simd: S,
    x: &[i16],
    p: &[u8],
    y: &mut [i16],
    plan: &Plan,
    done: &mut usize,
) {
    *done = cq15_level(simd, x, p, y, plan);
}

#[inline(always)]
fn c32_run<S: SimdToken>(
    simd: S,
    x: &[f32],
    p: &[u8],
    y: &mut [f32],
    plan: &Plan,
    done: &mut usize,
) {
    *done = c32_level(simd, x, p, y, plan);
}

// The entries, one function per level (see `dispatch_fn!`).
dispatch_fn!(
    fn run_cq15(x: &[i16], p: &[u8], y: &mut [i16], plan: &Plan, done: &mut usize) => cq15_run
);
dispatch_fn!(
    fn run_c32(x: &[f32], p: &[u8], y: &mut [f32], plan: &Plan, done: &mut usize) => c32_run
);

impl Scramble<CQ15> for Simd {
    fn scramble(
        &self,
        input: &[CQ15],
        phase: &[u8],
        bits: u32,
        inverse: bool,
        output: &mut [CQ15],
    ) -> Result<()> {
        check_args(bits, input.len(), phase.len(), output.len())?;
        let plan = Plan::new(bits, inverse);
        let mut done = 0;
        run_cq15(
            bytemuck::cast_slice(input),
            phase,
            bytemuck::cast_slice_mut(output),
            &plan,
            &mut done,
        );
        // The rest (and everything on a level without a path).
        for ((o, &x), &p) in output.iter_mut().zip(input).zip(phase).skip(done) {
            *o = scramble_one_cq15(x, phase_index(p, bits, inverse), bits)?;
        }
        Ok(())
    }
}

impl Scramble<C32> for Simd {
    fn scramble(
        &self,
        input: &[C32],
        phase: &[u8],
        bits: u32,
        inverse: bool,
        output: &mut [C32],
    ) -> Result<()> {
        check_args(bits, input.len(), phase.len(), output.len())?;
        let plan = Plan::new(bits, inverse);
        let mut done = 0;
        run_c32(
            bytemuck::cast_slice(input),
            phase,
            bytemuck::cast_slice_mut(output),
            &plan,
            &mut done,
        );
        for ((o, &x), &p) in output.iter_mut().zip(input).zip(phase).skip(done) {
            *o = scramble_one_c32(x, phase_index(p, bits, inverse), bits)?;
        }
        Ok(())
    }
}

/// The quadrant masks of the four quadrants, one entry per `q` (used as permute and shuffle
/// tables): swap (`q & 1`), the `CQ15` negation masks (`re` in the low half where
/// `(q ^ (q >> 1)) & 1`, `im` in the high half where `(q >> 1) & 1`) and the `C32` sign bits of
/// `re` and `im`.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const Q_SWAP: [u32; 4] = [0, u32::MAX, 0, u32::MAX];
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const Q_NEG: [u32; 4] = [0, 0x0000_FFFF, u32::MAX, 0xFFFF_0000];
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const Q_SIGN_RE: [u32; 4] = [0, 0x8000_0000, 0x8000_0000, 0];
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const Q_SIGN_IM: [u32; 4] = [0, 0, 0x8000_0000, 0x8000_0000];

/// The paths of x86-64: SSE2 and SSE4.2 (four samples per step), AVX2 (eight) and AVX-512
/// (sixteen `CQ15`, eight `C32`).
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod x86 {
    use core::arch::x86_64::{
        __m128, __m128i, __m256, __m256i, __m512, __m512i, _mm_add_epi32, _mm_add_ps, _mm_and_ps,
        _mm_and_si128, _mm_andnot_ps, _mm_andnot_si128, _mm_blendv_epi8, _mm_blendv_ps,
        _mm_castps_si128, _mm_castsi128_ps, _mm_cvtepu8_epi32, _mm_cvtsi32_si128,
        _mm_cvtsi64_si128, _mm_madd_epi16, _mm_mul_ps, _mm_or_ps, _mm_or_si128, _mm_packs_epi32,
        _mm_set_epi64x, _mm_set1_epi8, _mm_set1_epi32, _mm_setr_epi8, _mm_setzero_si128,
        _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_shuffle_ps, _mm_shufflehi_epi16,
        _mm_shufflelo_epi16, _mm_sign_epi32, _mm_slli_epi32, _mm_srai_epi16, _mm_srai_epi32,
        _mm_srl_epi32, _mm_srli_epi32, _mm_sub_epi8, _mm_sub_epi32, _mm_sub_ps, _mm_subs_epi16,
        _mm_unpackhi_epi16, _mm_unpackhi_epi64, _mm_unpackhi_ps, _mm_unpacklo_epi8,
        _mm_unpacklo_epi16, _mm_unpacklo_ps, _mm_xor_ps, _mm_xor_si128, _mm256_add_epi32,
        _mm256_add_ps, _mm256_and_si256, _mm256_blendv_epi8, _mm256_blendv_ps,
        _mm256_broadcastsi128_si256, _mm256_castps_si256, _mm256_castsi256_ps,
        _mm256_cvtepu8_epi32, _mm256_madd_epi16, _mm256_mul_ps, _mm256_packs_epi32,
        _mm256_permute4x64_epi64, _mm256_permutevar8x32_epi32, _mm256_set_epi64x, _mm256_set_m128i,
        _mm256_set1_epi32, _mm256_shuffle_epi8, _mm256_shuffle_ps, _mm256_sign_epi32,
        _mm256_slli_epi32, _mm256_srai_epi32, _mm256_srl_epi32, _mm256_sub_ps, _mm256_subs_epi16,
        _mm256_unpackhi_ps, _mm256_unpacklo_ps, _mm256_xor_ps, _mm256_xor_si256, _mm512_add_epi32,
        _mm512_add_ps, _mm512_and_si512, _mm512_broadcast_i32x4, _mm512_castpd_ps,
        _mm512_castps_pd, _mm512_castps_si512, _mm512_castsi512_ps, _mm512_cvtepu8_epi32,
        _mm512_cvtepu8_epi64, _mm512_i32gather_epi32, _mm512_loadu_si512, _mm512_madd_epi16,
        _mm512_mask_blend_epi32, _mm512_mask_blend_epi64, _mm512_mask_blend_pd,
        _mm512_mask_ror_epi32, _mm512_mask_sub_epi32, _mm512_mask_sub_epi64, _mm512_mul_ps,
        _mm512_or_si512, _mm512_packs_epi32, _mm512_permute_ps, _mm512_permutex2var_epi32,
        _mm512_permutex2var_epi64, _mm512_permutex2var_ps, _mm512_permutexvar_epi32,
        _mm512_permutexvar_epi64, _mm512_set1_epi32, _mm512_set1_epi64, _mm512_setzero_ps,
        _mm512_setzero_si512, _mm512_shuffle_epi8, _mm512_slli_epi32, _mm512_slli_epi64,
        _mm512_srai_epi32, _mm512_srl_epi32, _mm512_srl_epi64, _mm512_sub_ps, _mm512_subs_epi16,
        _mm512_test_epi32_mask, _mm512_test_epi64_mask, _mm512_xor_si512,
    };

    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4, f32x8, f32x16, i16x8, i16x16, i16x32, u8x16, u32x8, u32x16};

    use super::{
        A_Q15, AB_Q15, B_Q15, C_F32, CS_C32, ENTRIES, Plan, Q_NEG, Q_SIGN_IM, Q_SIGN_RE, Q_SWAP,
        S_F32, at,
    };

    /// `2^14`, the rounding constant of the `Q15` product.
    const ROUND: i32 = 1 << 14;

    /// `[t[0], t[1], t[2], t[3]]` twice, for `vpermd` with an index whose low two bits are the
    /// quadrant (bit 2 may be set).
    #[inline(always)]
    fn table8(t: [u32; 4]) -> [u32; 8] {
        [t[0], t[1], t[2], t[3], t[0], t[1], t[2], t[3]]
    }

    /// The table entries of the phase bytes `p`, for the per-sample loads.
    #[inline(always)]
    fn entries<const N: usize>(p: &[u8; N], plan: &Plan) -> [usize; N] {
        p.map(|v| plan.entry(v))
    }

    // -----------------------------------------------------------------------------------------
    // SSE2 and SSE4.2: four samples per step, one 32-bit lane per sample for the indices.

    /// The index constants of a call (128-bit).
    #[derive(Clone, Copy)]
    struct Idx128 {
        /// All ones for `inverse` (SSE2: `k = (p ^ m) − m`).
        neg: __m128i,
        /// −1 for `inverse`, else 1 (`psignd`).
        sgn: __m128i,
        qshift: __m128i,
        three: __m128i,
        /// The sign bit in every lane.
        sign: __m128i,
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    fn idx128(plan: &Plan) -> Idx128 {
        Idx128 {
            neg: _mm_set1_epi32(plan.neg as i32),
            sgn: _mm_set1_epi32(if plan.neg == 0 { 1 } else { -1 }),
            qshift: _mm_cvtsi32_si128(plan.qshift as i32),
            three: _mm_set1_epi32(3),
            sign: _mm_set1_epi32(i32::MIN),
        }
    }

    /// `k = ±p` of four phase bytes in the 32-bit lanes (SSE2: two unpacks with zero and
    /// `(p ^ m) − m`).
    #[target_feature(enable = "sse2")]
    #[inline]
    fn k4_sse2(p: [u8; 4], c: Idx128) -> __m128i {
        let z = _mm_setzero_si128();
        let v = _mm_cvtsi32_si128(i32::from_le_bytes(p));
        let v = _mm_unpacklo_epi16(_mm_unpacklo_epi8(v, z), z);
        _mm_sub_epi32(_mm_xor_si128(v, c.neg), c.neg)
    }

    /// `k = ±p` of four phase bytes (`pmovzxbd`, `psignd`).
    #[target_feature(enable = "sse4.2")]
    #[inline]
    fn k4_sse41(p: [u8; 4], c: Idx128) -> __m128i {
        _mm_sign_epi32(
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(i32::from_le_bytes(p))),
            c.sgn,
        )
    }

    /// The quadrant `q` of `k` (0..3).
    #[target_feature(enable = "sse2")]
    #[inline]
    fn q128(k: __m128i, c: Idx128) -> __m128i {
        _mm_and_si128(_mm_srl_epi32(_mm_slli_epi32::<1>(k), c.qshift), c.three)
    }

    /// The bits of `a` where `m`, of `b` elsewhere (SSE2: and, andnot, or).
    #[target_feature(enable = "sse2")]
    #[inline]
    fn select_sse2(m: __m128i, a: __m128i, b: __m128i) -> __m128i {
        _mm_or_si128(_mm_and_si128(m, a), _mm_andnot_si128(m, b))
    }

    /// The bytes of `a` where `m` (whole bytes set or clear), of `b` elsewhere (`pblendvb`).
    #[target_feature(enable = "sse4.2")]
    #[inline]
    fn select_sse41(m: __m128i, a: __m128i, b: __m128i) -> __m128i {
        _mm_blendv_epi8(b, a, m)
    }

    /// `a` where the lane mask `m` (all ones or zero) is set, `b` elsewhere.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn select_ps_sse2(m: __m128i, a: __m128, b: __m128) -> __m128 {
        let m = _mm_castsi128_ps(m);
        _mm_or_ps(_mm_and_ps(m, a), _mm_andnot_ps(m, b))
    }

    #[target_feature(enable = "sse4.2")]
    #[inline]
    fn select_ps_sse41(m: __m128i, a: __m128, b: __m128) -> __m128 {
        _mm_blendv_ps(b, a, _mm_castsi128_ps(m))
    }

    /// The lane mask of a sign-bit mask for [`select_ps_sse2`] (`psrad 31`).
    #[target_feature(enable = "sse2")]
    #[inline]
    fn sign_mask_sse2(m: __m128i) -> __m128i {
        _mm_srai_epi32::<31>(m)
    }

    /// `blendvps` reads the sign bit only: the sign-bit mask as it is.
    #[target_feature(enable = "sse4.2")]
    #[inline]
    fn sign_mask_sse41(m: __m128i) -> __m128i {
        m
    }

    /// The `CQ15` quadrant masks `(swap, neg)` of `q` from shifts (SSE2): `swap` from bit 0,
    /// `neg` from the sign bits of the 16-bit halves of
    /// `(q << 30) | ((q ^ (q >> 1)) << 15)` (bit 15: `re`, bit 31: `im`).
    #[target_feature(enable = "sse2")]
    #[inline]
    fn masks_cq15_sse2(q: __m128i) -> (__m128i, __m128i) {
        let swap = _mm_srai_epi32::<31>(_mm_slli_epi32::<31>(q));
        let t = _mm_or_si128(
            _mm_slli_epi32::<30>(q),
            _mm_slli_epi32::<15>(_mm_xor_si128(q, _mm_srli_epi32::<1>(q))),
        );
        (swap, _mm_srai_epi16::<15>(t))
    }

    /// The `CQ15` quadrant masks `(swap, neg)` of `q` from byte tables (`pshufb`): `q`
    /// replicated into the four bytes of its lane, then looked up.
    #[target_feature(enable = "sse4.2")]
    #[inline]
    fn masks_cq15_ssse3(q: __m128i) -> (__m128i, __m128i) {
        let rep = _mm_shuffle_epi8(
            q,
            _mm_setr_epi8(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12),
        );
        // Bytes 0, 1 of a lane look up `re` (entries 0..3), bytes 2, 3 `im` (entries 4..7).
        let half = _mm_setr_epi8(0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4);
        let swap_t = _mm_setr_epi8(0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let neg_t = _mm_setr_epi8(0, -1, -1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0);
        (
            _mm_shuffle_epi8(swap_t, rep),
            _mm_shuffle_epi8(neg_t, _mm_or_si128(rep, half)),
        )
    }

    /// `re` and `im` swapped in every lane (SSE2: `pshuflw` and `pshufhw`).
    #[target_feature(enable = "sse2")]
    #[inline]
    fn swap_cq15_sse2(y: __m128i) -> __m128i {
        _mm_shufflehi_epi16::<0xB1>(_mm_shufflelo_epi16::<0xB1>(y))
    }

    /// `re` and `im` swapped in every lane (`pshufb`).
    #[target_feature(enable = "sse4.2")]
    #[inline]
    fn swap_cq15_ssse3(y: __m128i) -> __m128i {
        _mm_shuffle_epi8(
            y,
            _mm_setr_epi8(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13),
        )
    }

    /// The packed products `[re0 re1 re2 re3 im0 im1 im2 im3]` interleaved (SSE2: an unpack).
    #[target_feature(enable = "sse2")]
    #[inline]
    fn zip_sse2(v: __m128i) -> __m128i {
        _mm_unpacklo_epi16(v, _mm_unpackhi_epi64(v, v))
    }

    /// The packed products interleaved (`pshufb`).
    #[target_feature(enable = "sse4.2")]
    #[inline]
    fn zip_ssse3(v: __m128i) -> __m128i {
        _mm_shuffle_epi8(
            v,
            _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15),
        )
    }

    /// The `cq15_mul` of four samples `x` with the phasors `a = (c, !s)`, `b = (s, c)`, packed
    /// as `[re0..re3 im0..im3]`: `pmaddwd`, `xi + 2^14` added to the real part (`!s = −s − 1`),
    /// `>> 15`, saturating pack.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn product_cq15_sse(x: __m128i, a: __m128i, b: __m128i) -> __m128i {
        let round = _mm_set1_epi32(ROUND);
        let re = _mm_add_epi32(
            _mm_madd_epi16(x, a),
            _mm_add_epi32(_mm_srai_epi32::<16>(x), round),
        );
        let im = _mm_add_epi32(_mm_madd_epi16(x, b), round);
        _mm_packs_epi32(_mm_srai_epi32::<15>(re), _mm_srai_epi32::<15>(im))
    }

    /// The first and the second halves of the table words `t[e0], t[e1], t[e2], t[e3]` (two
    /// 64-bit loads per register and `shufps`).
    #[target_feature(enable = "sse2")]
    #[inline]
    fn pairs4(w: [u64; 4]) -> (__m128, __m128) {
        let v01 = _mm_castsi128_ps(_mm_set_epi64x(w[1] as i64, w[0] as i64));
        let v23 = _mm_castsi128_ps(_mm_set_epi64x(w[3] as i64, w[2] as i64));
        (
            _mm_shuffle_ps::<0x88>(v01, v23),
            _mm_shuffle_ps::<0xDD>(v01, v23),
        )
    }

    /// The SSE2 and SSE4.2 paths: the same steps with the helpers of the level.
    macro_rules! sse_paths {
        (
            $token:ident, $feature:literal, $cq15:ident, $c32:ident,
            k: $k:ident, select: $select:ident, select_ps: $select_ps:ident,
            masks: $masks:ident, swap: $swap:ident, zip: $zip:ident,
            sign_mask: $sign_mask:ident
        ) => {
            /// Four `CQ15` samples, `b ≤ 2`: the quadrant map.
            #[target_feature(enable = $feature)]
            #[inline]
            fn quadrant_cq15(x: __m128i, p: &[u8; 4], c: Idx128) -> __m128i {
                let q = q128($k(*p, c), c);
                let (swap, neg) = $masks(q);
                let y = $select(swap, $swap(x), x);
                _mm_subs_epi16(_mm_xor_si128(y, neg), neg)
            }

            /// Four `CQ15` samples, `b ≥ 3`: the table phasor by per-sample pair loads and
            /// the product.
            #[target_feature(enable = $feature)]
            #[inline]
            fn phasor_cq15(x: __m128i, p: &[u8; 4], plan: &Plan) -> __m128i {
                let (a, b) = pairs4(entries(p, plan).map(|e| at(&AB_Q15, e)));
                $zip(product_cq15_sse(
                    x,
                    _mm_castps_si128(a),
                    _mm_castps_si128(b),
                ))
            }

            /// Four `C32` samples (two registers), `b ≤ 2`, in the planar form: `re` and `im`
            /// by `shufps`, the masks by shifts. The quadrant bits stay unmasked in `qr` (bits
            /// 0 and 1 are the quadrant): the swap mask is bit 0 moved to the sign bit, the sign
            /// of `im` bit 1, the sign of `re` their XOR.
            #[target_feature(enable = $feature)]
            #[inline]
            fn quadrant_c32(x: [__m128; 2], p: &[u8; 4], c: Idx128) -> [__m128; 2] {
                let k = $k(*p, c);
                let qr = _mm_srl_epi32(_mm_slli_epi32::<1>(k), c.qshift);
                let re = _mm_shuffle_ps::<0x88>(x[0], x[1]);
                let im = _mm_shuffle_ps::<0xDD>(x[0], x[1]);
                let swap = _mm_slli_epi32::<31>(qr);
                let sim = _mm_and_si128(_mm_slli_epi32::<30>(qr), c.sign);
                let sre = _mm_castsi128_ps(_mm_xor_si128(swap, sim));
                let sim = _mm_castsi128_ps(sim);
                let swap = $sign_mask(swap);
                let ore = _mm_xor_ps($select_ps(swap, im, re), sre);
                let oim = _mm_xor_ps($select_ps(swap, re, im), sim);
                [_mm_unpacklo_ps(ore, oim), _mm_unpackhi_ps(ore, oim)]
            }

            /// Eight `C32` samples (four registers), `b = 1`: the sign bit of every component
            /// flipped where `p & 1`. The eight phase bytes are spread to one dword per `f32`
            /// (`punpcklbw`, `punpck*wd`, `pshufd`), and `pslld 31` keeps bit 0 of each.
            #[target_feature(enable = $feature)]
            #[inline]
            fn chip_c32(x: [__m128; 4], p: &[u8; 8]) -> [__m128; 4] {
                let b = _mm_cvtsi64_si128(i64::from_le_bytes(*p));
                let t = _mm_unpacklo_epi8(b, b);
                let lo = _mm_unpacklo_epi16(t, t);
                let hi = _mm_unpackhi_epi16(t, t);
                let flip = |v: __m128, m: __m128i| {
                    _mm_xor_ps(v, _mm_castsi128_ps(_mm_slli_epi32::<31>(m)))
                };
                let [x0, x1, x2, x3] = x;
                [
                    flip(x0, _mm_shuffle_epi32::<0x50>(lo)),
                    flip(x1, _mm_shuffle_epi32::<0xFA>(lo)),
                    flip(x2, _mm_shuffle_epi32::<0x50>(hi)),
                    flip(x3, _mm_shuffle_epi32::<0xFA>(hi)),
                ]
            }

            /// Four `C32` samples, `b ≥ 3`, planar: the phasor by per-sample pair loads, the
            /// product in the order of the fallback.
            #[target_feature(enable = $feature)]
            #[inline]
            fn phasor_c32(x: [__m128; 2], p: &[u8; 4], plan: &Plan) -> [__m128; 2] {
                let re = _mm_shuffle_ps::<0x88>(x[0], x[1]);
                let im = _mm_shuffle_ps::<0xDD>(x[0], x[1]);
                let (cv, sv) = pairs4(entries(p, plan).map(|e| at(&CS_C32, e)));
                let pre = _mm_sub_ps(_mm_mul_ps(re, cv), _mm_mul_ps(im, sv));
                let pim = _mm_add_ps(_mm_mul_ps(re, sv), _mm_mul_ps(im, cv));
                [_mm_unpacklo_ps(pre, pim), _mm_unpackhi_ps(pre, pim)]
            }

            fearless_simd::kernel!(
                pub(crate) fn $cq15(
                    t: $token,
                    x: &[i16],
                    p: &[u8],
                    y: &mut [i16],
                    plan: &Plan,
                ) -> usize {
                    let c = idx128(plan);
                    let (xs, _) = x.as_chunks::<8>();
                    let (ps, _) = p.as_chunks::<4>();
                    let (ys, _) = y.as_chunks_mut::<8>();
                    let mut n = 0;
                    let it = xs.iter().zip(ps).zip(ys);
                    if plan.phasor() {
                        for ((xv, pv), yv) in it {
                            let v = phasor_cq15(i16x8::from_slice(t, xv).into(), pv, plan);
                            i16x8::simd_from(t, v).store_array(yv);
                            n += 4;
                        }
                    } else {
                        for ((xv, pv), yv) in it {
                            let v = quadrant_cq15(i16x8::from_slice(t, xv).into(), pv, c);
                            i16x8::simd_from(t, v).store_array(yv);
                            n += 4;
                        }
                    }
                    n
                }
            );

            fearless_simd::kernel!(
                pub(crate) fn $c32(
                    t: $token,
                    x: &[f32],
                    p: &[u8],
                    y: &mut [f32],
                    plan: &Plan,
                ) -> usize {
                    let c = idx128(plan);
                    // Eight samples per iteration: two steps of four.
                    let (xs, _) = x.as_chunks::<16>();
                    let (ps, _) = p.as_chunks::<8>();
                    let (ys, _) = y.as_chunks_mut::<16>();
                    let mut n = 0;
                    if plan.bits == 1 {
                        for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
                            let x4: &[[f32; 4]; 4] = bytemuck::cast_ref(xv);
                            let v = x4.map(|c| __m128::from(f32x4::from_slice(t, &c)));
                            let r = chip_c32(v, pv);
                            let y4: &mut [[f32; 4]; 4] = bytemuck::cast_mut(yv);
                            for (o, r) in y4.iter_mut().zip(r) {
                                f32x4::simd_from(t, r).store_array(o);
                            }
                            n += 8;
                        }
                        return n;
                    }
                    let mut body = |xv: &[f32; 16], pv: &[u8; 8], yv: &mut [f32; 16], ph: bool| {
                        let (xa, xb) = xv.split_at(8);
                        let (pa, pb) = pv.split_at(4);
                        let (ya, yb) = yv.split_at_mut(8);
                        for ((xh, ph4), yh) in
                            [(xa, pa, ya), (xb, pb, yb)].map(|(xh, ph4, yh)| ((xh, ph4), yh))
                        {
                            let (lo, hi) = xh.split_at(4);
                            let v = [
                                f32x4::from_slice(t, lo).into(),
                                f32x4::from_slice(t, hi).into(),
                            ];
                            let ph4: &[u8; 4] = ph4.try_into().unwrap_or(&[0; 4]);
                            let [a, b] = if ph {
                                phasor_c32(v, ph4, plan)
                            } else {
                                quadrant_c32(v, ph4, c)
                            };
                            let (lo, hi) = yh.split_at_mut(4);
                            f32x4::simd_from(t, a).store_slice(lo);
                            f32x4::simd_from(t, b).store_slice(hi);
                        }
                        n += 8;
                    };
                    if plan.phasor() {
                        for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
                            body(xv, pv, yv, true);
                        }
                    } else {
                        for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
                            body(xv, pv, yv, false);
                        }
                    }
                    n
                }
            );
        };
    }

    // The macro body names the helpers of this module.
    #[allow(clippy::wildcard_imports)]
    mod sse2 {
        use super::*;
        sse_paths!(
            Sse2, "sse2", cq15_sse2, c32_sse2,
            k: k4_sse2, select: select_sse2, select_ps: select_ps_sse2,
            masks: masks_cq15_sse2, swap: swap_cq15_sse2, zip: zip_sse2,
            sign_mask: sign_mask_sse2
        );
    }
    #[allow(clippy::wildcard_imports)]
    mod sse4_2 {
        use super::*;
        sse_paths!(
            Sse4_2, "sse4.2", cq15_sse4_2, c32_sse4_2,
            k: k4_sse41, select: select_sse41, select_ps: select_ps_sse41,
            masks: masks_cq15_ssse3, swap: swap_cq15_ssse3, zip: zip_ssse3,
            sign_mask: sign_mask_sse41
        );
    }
    pub(super) use sse2::{c32_sse2, cq15_sse2};
    pub(super) use sse4_2::{c32_sse4_2, cq15_sse4_2};

    // -----------------------------------------------------------------------------------------
    // AVX2: eight samples per step, one 32-bit lane per sample for the indices.

    /// The constants of a call (256-bit).
    #[derive(Clone, Copy)]
    struct Idx256 {
        /// −1 for `inverse`, else 1 (`vpsignd`).
        sgn: __m256i,
        qshift: __m128i,
        /// `2^b − 1` (`b ≥ 3`).
        mask: __m256i,
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn idx256(plan: &Plan) -> Idx256 {
        Idx256 {
            sgn: _mm256_set1_epi32(if plan.neg == 0 { 1 } else { -1 }),
            qshift: _mm_cvtsi32_si128(plan.qshift as i32),
            mask: _mm256_set1_epi32(plan.mask as i32),
        }
    }

    /// `k = ±p` of eight phase bytes, one 32-bit lane each (`vpmovzxbd`, `vpsignd`).
    #[target_feature(enable = "avx2")]
    #[inline]
    fn k8(p: [u8; 8], c: Idx256) -> __m256i {
        let v = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(i64::from_le_bytes(p)));
        _mm256_sign_epi32(v, c.sgn)
    }

    /// The quadrant index `(k << 1) >> (b − 1)` (the quadrant in the low two bits; the bits
    /// above are not cleared: the tables are repeated and the masks use bits 0 and 1 only).
    #[target_feature(enable = "avx2")]
    #[inline]
    fn q8(k: __m256i, c: Idx256) -> __m256i {
        _mm256_srl_epi32(_mm256_slli_epi32::<1>(k), c.qshift)
    }

    /// The forms of the `(c, s)` source of AVX2 at `b ≥ 3` (M49.1 §3).
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Avx2Form {
        /// `vpermd`/`vpermps` of the `2^b` entries in one or two registers (`b` 3 and 4).
        Permute,
        /// One 64-bit load of the pair per sample.
        Loads,
    }

    /// The form of AVX2 at `b ≥ 3`, chosen by the simulated cycles (M49.1 §3,
    /// `docs/performance.md`): the gather form (`vpgatherdd`/`vgatherdps` from the static
    /// tables) simulated 0.80 / 3.93 (`C32`) and 0.66 / 4.81 (`CQ15`) cycles per sample at
    /// `b = 8` (`znver3` / `alderlake`), slower than the fallback on `alderlake`; the
    /// per-sample loads 1.84 / 1.73 and 1.78 / 1.67, since [`words8`] 1.31 / 1.17 and
    /// 1.21 / 1.13 (I13.1-3).
    #[inline(always)]
    fn avx2_form(plan: &Plan) -> Avx2Form {
        if plan.entries() <= 16 {
            Avx2Form::Permute
        } else {
            Avx2Form::Loads
        }
    }

    /// Up to 16 dword entries in two registers: one `vpermd`, or two and a blend on bit 3.
    #[derive(Clone, Copy)]
    struct Lookup256 {
        lo: __m256i,
        hi: __m256i,
        entries: usize,
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn lookup256(t: &[u32]) -> Lookup256 {
        let ld = |i: usize| -> __m256i {
            let a: [i32; 8] = core::array::from_fn(|j| t.get(8 * i + j).map_or(0, |&v| v as i32));
            // SAFETY: `a` holds eight `i32`, 32 bytes.
            unsafe { core::ptr::read_unaligned(a.as_ptr().cast::<__m256i>()) }
        };
        Lookup256 {
            lo: ld(0),
            hi: ld(1),
            entries: t.len(),
        }
    }

    /// The table entries of the indices `k` (each below `l.entries` ≤ 16).
    #[target_feature(enable = "avx2")]
    #[inline]
    fn get256(l: &Lookup256, k: __m256i) -> __m256i {
        if l.entries <= 8 {
            _mm256_permutevar8x32_epi32(l.lo, k)
        } else {
            let m = _mm256_castsi256_ps(_mm256_slli_epi32::<28>(k));
            _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(_mm256_permutevar8x32_epi32(l.lo, k)),
                _mm256_castsi256_ps(_mm256_permutevar8x32_epi32(l.hi, k)),
                m,
            ))
        }
    }

    /// The first and the second halves of eight table words, in the lane order of the
    /// registers `[w0 w1 | w2 w3]`, `[w4 w5 | w6 w7]` after `shufps` (`[0 1 4 5 | 2 3 6 7]`),
    /// from 64-bit loads.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn pairs8(w: [u64; 8]) -> (__m256, __m256) {
        let a = _mm256_castsi256_ps(_mm256_set_epi64x(
            w[3] as i64,
            w[2] as i64,
            w[1] as i64,
            w[0] as i64,
        ));
        let b = _mm256_castsi256_ps(_mm256_set_epi64x(
            w[7] as i64,
            w[6] as i64,
            w[5] as i64,
            w[4] as i64,
        ));
        (
            _mm256_shuffle_ps::<0x88>(a, b),
            _mm256_shuffle_ps::<0xDD>(a, b),
        )
    }

    /// The constants of the per-sample loads of AVX2 ([`Avx2Form::Loads`]): `k = (±p) & mask`
    /// in byte lanes (`2^b − 1` fits a byte for `b ≤ 8`), and the `2^8` table words from the
    /// offset `2^b`, which `k` indexes without a bound check.
    #[derive(Clone, Copy)]
    struct Loads8 {
        /// All ones for `inverse`, else 0 (`k = (p ^ m) − m`).
        neg: __m128i,
        mask: __m128i,
        words: &'static [u64; 256],
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn loads8(plan: &Plan, t: &'static [u64; ENTRIES]) -> Loads8 {
        static NONE: [u64; 256] = [0; 256];
        let base = plan.base as usize;
        Loads8 {
            neg: _mm_set1_epi8(plan.neg as i8),
            mask: _mm_set1_epi8(plan.mask as i8),
            // `2^b ≤ 256`: the words `2^b .. 2^b + 256` are in the table.
            words: t
                .get(base..base + 256)
                .and_then(|w| w.try_into().ok())
                .unwrap_or(&NONE),
        }
    }

    /// The table words of eight phase bytes, the entries of [`Plan::entry`]: `k` of the eight
    /// bytes in one vector (`vpxor`, `vpsubb`, `vpand`), moved to a general register with one
    /// `vmovq`, each byte of it the index of one 64-bit load. Written as the compiler chose,
    /// the indices took a different form in each build: the `avx2` build computed four of them
    /// in a vector and four in general registers, with a test of `inverse` per sample, the
    /// build with the level detected at run time computed all eight in vectors and moved each
    /// out with a `vpextrw` (two micro-ops, one of them on the shuffle port that the loads'
    /// `vmovlhps`/`vinsertf128` already fill), 1.34 / 1.45 (`C32`) and 1.22 / 1.53 (`CQ15`)
    /// times the cycles of the `avx2` build at `b = 8` (I13.1-3). The `vmovq` is an `asm!`
    /// block, so that neither build turns the byte extraction back into vector extractions.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn words8(p: [u8; 8], c: Loads8) -> [u64; 8] {
        let v = _mm_cvtsi64_si128(i64::from_le_bytes(p));
        let k = _mm_and_si128(_mm_sub_epi8(_mm_xor_si128(v, c.neg), c.neg), c.mask);
        let w: u64;
        // SAFETY: `vmovq` reads one `xmm` register and writes one general register; it touches
        // no memory, no stack and no flags, and AVX is a target feature of this function.
        unsafe {
            core::arch::asm!(
                "vmovq {w}, {k}",
                w = out(reg) w,
                k = in(xmm_reg) k,
                options(pure, nomem, nostack, preserves_flags)
            );
        }
        core::array::from_fn(|i| {
            // A byte indexes the 256 words: no bound check.
            let k = usize::from((w >> (8 * i)) as u8);
            c.words.get(k).copied().unwrap_or_default()
        })
    }

    fearless_simd::kernel!(
        pub(super) fn cq15_avx2(t: Avx2, x: &[i16], p: &[u8], y: &mut [i16], plan: &Plan) -> usize {
            let c = idx256(plan);
            let (xs, _) = x.as_chunks::<16>();
            let (ps, _) = p.as_chunks::<8>();
            let (ys, _) = y.as_chunks_mut::<16>();
            let mut n = 0;
            let zip_b = _mm256_broadcastsi128_si256(_mm_setr_epi8(
                0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
            ));
            if plan.phasor() {
                let round = _mm256_set1_epi32(ROUND);
                let la = lookup256(plan.slice(&A_Q15));
                let lb = lookup256(plan.slice(&B_Q15));
                let lw = loads8(plan, &AB_Q15);
                let mut body = |xv: &[i16; 16], pv: &[u8; 8], yv: &mut [i16; 16], form| {
                    let xr: __m256i = i16x16::from_slice(t, xv).into();
                    let (a, b) = match form {
                        Avx2Form::Permute => {
                            let k = _mm256_and_si256(k8(*pv, c), c.mask);
                            (get256(&la, k), get256(&lb, k))
                        }
                        Avx2Form::Loads => {
                            // Lanes `[0 1 2 3 | 4 5 6 7]`: the words in the order 0 1 4 5, 2 3 6 7.
                            let w = words8(*pv, lw);
                            let (a, b) = pairs8([w[0], w[1], w[4], w[5], w[2], w[3], w[6], w[7]]);
                            (_mm256_castps_si256(a), _mm256_castps_si256(b))
                        }
                    };
                    // The real part `xr c + xi (−s − 1) + xi`, see `product_cq15_sse`.
                    let re = _mm256_add_epi32(
                        _mm256_madd_epi16(xr, a),
                        _mm256_add_epi32(_mm256_srai_epi32::<16>(xr), round),
                    );
                    let im = _mm256_add_epi32(_mm256_madd_epi16(xr, b), round);
                    let v = _mm256_packs_epi32(
                        _mm256_srai_epi32::<15>(re),
                        _mm256_srai_epi32::<15>(im),
                    );
                    i16x16::simd_from(t, _mm256_shuffle_epi8(v, zip_b)).store_array(yv);
                    n += 8;
                };
                // One loop per form, so that no build keeps the choice inside the loop.
                let it = xs.iter().zip(ps).zip(ys);
                match avx2_form(plan) {
                    Avx2Form::Permute => {
                        it.for_each(|((xv, pv), yv)| body(xv, pv, yv, Avx2Form::Permute));
                    }
                    Avx2Form::Loads => {
                        it.for_each(|((xv, pv), yv)| body(xv, pv, yv, Avx2Form::Loads));
                    }
                }
                return n;
            }
            let swap_t: __m256i = u32x8::simd_from(t, table8(Q_SWAP)).into();
            let neg_t: __m256i = u32x8::simd_from(t, table8(Q_NEG)).into();
            // The swap mask is made opaque to the compiler (XOR with `qshift >> 16`, which is 0
            // because `qshift` is at most 7): with a known constant, LLVM's generic x86-64 tuning
            // (the run-time-detection build) turns the one `vpshufb` into `vpshuflw` and
            // `vpshufhw`, one more shuffle per eight samples.
            let opaque = _mm256_set1_epi32((plan.qshift >> 16) as i32);
            let swap_b = _mm256_xor_si256(
                _mm256_broadcastsi128_si256(_mm_setr_epi8(
                    2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13,
                )),
                opaque,
            );
            let mut body = |xv: &[i16; 16], pv: &[u8; 8], yv: &mut [i16; 16]| {
                let v: __m256i = i16x16::from_slice(t, xv).into();
                let q = q8(k8(*pv, c), c);
                let swap = _mm256_permutevar8x32_epi32(swap_t, q);
                let neg = _mm256_permutevar8x32_epi32(neg_t, q);
                let v = _mm256_blendv_epi8(v, _mm256_shuffle_epi8(v, swap_b), swap);
                let v = _mm256_subs_epi16(_mm256_xor_si256(v, neg), neg);
                i16x16::simd_from(t, v).store_array(yv);
                n += 8;
            };
            // The quadrant map (13 instructions per eight samples) is unrolled by two by hand:
            // the generic x86-64 tuning of the run-time-detection build does not unroll it, and
            // its loop overhead was 1.07 times the instructions of the `avx2` build.
            let (x2, _) = xs.as_chunks::<2>();
            let (p2, _) = ps.as_chunks::<2>();
            let (y2, _) = ys.as_chunks_mut::<2>();
            for ((xv, pv), yv) in x2.iter().zip(p2).zip(y2) {
                let [ya, yb] = yv;
                body(&xv[0], &pv[0], ya);
                body(&xv[1], &pv[1], yb);
            }
            // An odd last step of eight.
            let done = 2 * x2.len().min(p2.len());
            if let (Some(xv), Some(pv), Some(yv)) = (xs.get(done), ps.get(done), ys.get_mut(done)) {
                body(xv, pv, yv);
            }
            n
        }
    );

    // `b = 1` in `c32_avx2`: sixteen interleaved `C32` samples (four registers) with the sign
    // bit of both components flipped where `p & 1`. The sixteen phase bytes are broadcast to
    // both 128-bit lanes; one `vpshufb` per register puts the byte of each sample into the low
    // byte of its two dwords (the other bytes zero), and `vpslld 31` keeps bit 0 of it.

    fearless_simd::kernel!(
        pub(super) fn c32_avx2(t: Avx2, x: &[f32], p: &[u8], y: &mut [f32], plan: &Plan) -> usize {
            if plan.bits == 1 {
                // `b = 1`: sixteen samples per step (see `chip_c32_avx2`).
                const Z: i8 = -1;
                let idx: [__m256i; 4] = core::array::from_fn(|r| {
                    // Register `r` holds the samples `4r .. 4r + 3`, two per 128-bit lane.
                    let s = |j: usize| (4 * r + j) as i8;
                    let lane = |a: i8, b: i8| {
                        _mm_setr_epi8(a, Z, Z, Z, a, Z, Z, Z, b, Z, Z, Z, b, Z, Z, Z)
                    };
                    let lo = lane(s(0), s(1));
                    let hi = lane(s(2), s(3));
                    _mm256_set_m128i(hi, lo)
                });
                let (xs, _) = x.as_chunks::<32>();
                let (ps, _) = p.as_chunks::<16>();
                let (ys, _) = y.as_chunks_mut::<32>();
                let mut n = 0;
                for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
                    let b = _mm256_broadcastsi128_si256(u8x16::from_slice(t, pv).into());
                    let x8: &[[f32; 8]; 4] = bytemuck::cast_ref(xv);
                    let y8: &mut [[f32; 8]; 4] = bytemuck::cast_mut(yv);
                    for ((xr, yr), i) in x8.iter().zip(y8.iter_mut()).zip(idx) {
                        let m = _mm256_slli_epi32::<31>(_mm256_shuffle_epi8(b, i));
                        let v: __m256 = f32x8::from_slice(t, xr).into();
                        let r = _mm256_xor_ps(v, _mm256_castsi256_ps(m));
                        f32x8::simd_from(t, r).store_array(yr);
                    }
                    n += 16;
                }
                return n;
            }
            let c = idx256(plan);
            let sign = _mm256_set1_epi32(i32::MIN);
            let lc = lookup256(bytemuck::cast_slice(plan.slice(&C_F32)));
            let ls = lookup256(bytemuck::cast_slice(plan.slice(&S_F32)));
            let lw = loads8(plan, &CS_C32);
            let (xs, _) = x.as_chunks::<16>();
            let (ps, _) = p.as_chunks::<8>();
            let (ys, _) = y.as_chunks_mut::<16>();
            let mut n = 0;
            // The planar form: `re` and `im` of the eight samples by `shufps` in the lane order
            // `[0 1 4 5 | 2 3 6 7]`, the indices permuted to the same order. `form`: `None` the
            // quadrant map (`b ≤ 2`), else the source of the phasor.
            let mut body =
                |xv: &[f32; 16], pv: &[u8; 8], yv: &mut [f32; 16], form: Option<Avx2Form>| {
                    let (lo, hi) = xv.split_at(8);
                    let x0: __m256 = f32x8::from_slice(t, lo).into();
                    let x1: __m256 = f32x8::from_slice(t, hi).into();
                    let re = _mm256_shuffle_ps::<0x88>(x0, x1);
                    let im = _mm256_shuffle_ps::<0xDD>(x0, x1);
                    let k = _mm256_permute4x64_epi64::<0xD8>(k8(*pv, c));
                    let (ore, oim) = if let Some(form) = form {
                        let (cv, sv) = match form {
                            Avx2Form::Permute => {
                                let k = _mm256_and_si256(k, c.mask);
                                (
                                    _mm256_castsi256_ps(get256(&lc, k)),
                                    _mm256_castsi256_ps(get256(&ls, k)),
                                )
                            }
                            Avx2Form::Loads => pairs8(words8(*pv, lw)),
                        };
                        (
                            _mm256_sub_ps(_mm256_mul_ps(re, cv), _mm256_mul_ps(im, sv)),
                            _mm256_add_ps(_mm256_mul_ps(re, sv), _mm256_mul_ps(im, cv)),
                        )
                    } else {
                        let q = q8(k, c);
                        // `vblendvps` reads the sign bit: bit 0 of `q` shifted there; the sign of
                        // `im` is bit 1, that of `re` their XOR.
                        let swap = _mm256_slli_epi32::<31>(q);
                        let sim = _mm256_and_si256(_mm256_slli_epi32::<30>(q), sign);
                        let sre = _mm256_castsi256_ps(_mm256_xor_si256(swap, sim));
                        let sim = _mm256_castsi256_ps(sim);
                        let swap = _mm256_castsi256_ps(swap);
                        (
                            _mm256_xor_ps(_mm256_blendv_ps(re, im, swap), sre),
                            _mm256_xor_ps(_mm256_blendv_ps(im, re, swap), sim),
                        )
                    };
                    let (lo, hi) = yv.split_at_mut(8);
                    f32x8::simd_from(t, _mm256_unpacklo_ps(ore, oim)).store_slice(lo);
                    f32x8::simd_from(t, _mm256_unpackhi_ps(ore, oim)).store_slice(hi);
                    n += 8;
                };
            // One loop per form, so that no build keeps the choice inside the loop.
            let it = xs.iter().zip(ps).zip(ys);
            let form = plan.phasor().then(|| avx2_form(plan));
            match form {
                None => it.for_each(|((xv, pv), yv)| body(xv, pv, yv, None)),
                Some(Avx2Form::Permute) => {
                    it.for_each(|((xv, pv), yv)| body(xv, pv, yv, Some(Avx2Form::Permute)));
                }
                Some(Avx2Form::Loads) => {
                    it.for_each(|((xv, pv), yv)| body(xv, pv, yv, Some(Avx2Form::Loads)));
                }
            }
            n
        }
    );

    // -----------------------------------------------------------------------------------------
    // AVX-512: sixteen `CQ15` samples (32-bit lanes) or eight `C32` samples (64-bit lanes) per
    // step.

    /// Up to 64 dword entries in four registers, looked up by `vpermd`, `vpermi2d` and a blend
    /// on bit 5.
    #[derive(Clone, Copy)]
    struct Lookup512d {
        t: [__m512i; 4],
        entries: usize,
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn lookup512d(t: &[u32]) -> Lookup512d {
        let mut regs = [_mm512_setzero_si512(); 4];
        for (i, d) in regs.iter_mut().enumerate() {
            let a: [u32; 16] = core::array::from_fn(|j| t.get(16 * i + j).copied().unwrap_or(0));
            // SAFETY: `a` is sixteen `u32`, 64 bytes.
            *d = unsafe { _mm512_loadu_si512(a.as_ptr().cast()) };
        }
        Lookup512d {
            t: regs,
            entries: t.len(),
        }
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn get512d(l: &Lookup512d, k: __m512i) -> __m512i {
        let [t0, t1, t2, t3] = l.t;
        if l.entries <= 16 {
            _mm512_permutexvar_epi32(k, t0)
        } else if l.entries <= 32 {
            _mm512_permutex2var_epi32(t0, k, t1)
        } else {
            let lo = _mm512_permutex2var_epi32(t0, k, t1);
            let hi = _mm512_permutex2var_epi32(t2, k, t3);
            _mm512_mask_blend_epi32(_mm512_test_epi32_mask(k, _mm512_set1_epi32(32)), lo, hi)
        }
    }

    /// Up to 64 qword entries in eight registers, looked up by `vpermq`, `vpermi2q` and blends
    /// on bits 4 and 5.
    #[derive(Clone, Copy)]
    struct Lookup512q {
        t: [__m512i; 8],
        entries: usize,
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn lookup512q(t: &[u64], map: fn(u64) -> u64) -> Lookup512q {
        let mut regs = [_mm512_setzero_si512(); 8];
        for (i, d) in regs.iter_mut().enumerate() {
            let a: [u64; 8] = core::array::from_fn(|j| t.get(8 * i + j).map_or(0, |&w| map(w)));
            // SAFETY: `a` is eight `u64`, 64 bytes.
            *d = unsafe { _mm512_loadu_si512(a.as_ptr().cast()) };
        }
        Lookup512q {
            t: regs,
            entries: t.len(),
        }
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn get512q(l: &Lookup512q, k: __m512i) -> __m512i {
        let [t0, t1, t2, t3, t4, t5, t6, t7] = l.t;
        if l.entries <= 8 {
            return _mm512_permutexvar_epi64(k, t0);
        }
        let a = _mm512_permutex2var_epi64(t0, k, t1);
        if l.entries <= 16 {
            return a;
        }
        let bit4 = _mm512_test_epi64_mask(k, _mm512_set1_epi64(16));
        let b = _mm512_permutex2var_epi64(t2, k, t3);
        let ab = _mm512_mask_blend_epi64(bit4, a, b);
        if l.entries <= 32 {
            return ab;
        }
        let c = _mm512_permutex2var_epi64(t4, k, t5);
        let d = _mm512_permutex2var_epi64(t6, k, t7);
        let cd = _mm512_mask_blend_epi64(bit4, c, d);
        _mm512_mask_blend_epi64(_mm512_test_epi64_mask(k, _mm512_set1_epi64(32)), ab, cd)
    }

    /// The largest number of phasors that AVX-512 permutes from registers; above it (`b` 7
    /// and 8) the phasors are gathered (M49.1 §3).
    const AVX512_PERMUTE_MAX: usize = 64;

    /// The constants of an AVX-512 `CQ15` call.
    struct Cq15x16 {
        inv: u16,
        qshift: __m128i,
        mask: __m512i,
        base: __m512i,
        neg_t: __m512i,
        zip_b: __m512i,
        la: Lookup512d,
        lb: Lookup512d,
    }

    /// Sixteen `CQ15` samples `x` with the phase bytes `p`. `FORM`: 0 the quadrant map
    /// (`b ≤ 2`), 1 the phasor from the register tables, 2 the gathered phasor.
    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    fn cq15_x16<const FORM: u8>(x: __m512i, p: __m128i, c: &Cq15x16) -> __m512i {
        let v = _mm512_cvtepu8_epi32(p);
        // `k = ±p`.
        let k = _mm512_mask_sub_epi32(v, c.inv, _mm512_setzero_si512(), v);
        if FORM == 0 {
            // The quadrant in the low two bits of `q` (the table repeats every four entries
            // and the swap mask tests bit 0).
            let q = _mm512_srl_epi32(_mm512_slli_epi32::<1>(k), c.qshift);
            let swap = _mm512_test_epi32_mask(q, _mm512_set1_epi32(1));
            let v = _mm512_mask_ror_epi32::<16>(x, swap, x);
            let neg = _mm512_permutexvar_epi32(q, c.neg_t);
            return _mm512_subs_epi16(_mm512_xor_si512(v, neg), neg);
        }
        let k = _mm512_and_si512(k, c.mask);
        let (a, b) = if FORM == 1 {
            (get512d(&c.la, k), get512d(&c.lb, k))
        } else {
            let e = _mm512_or_si512(k, c.base);
            // SAFETY: every lane of `e` is `2^b + k`, `k < 2^b`, `b` 7 or 8: below 512.
            unsafe {
                (
                    _mm512_i32gather_epi32::<4>(e, A_Q15.as_ptr().cast()),
                    _mm512_i32gather_epi32::<4>(e, B_Q15.as_ptr().cast()),
                )
            }
        };
        let round = _mm512_set1_epi32(ROUND);
        // The real part `xr c + xi (−s − 1) + xi`, see `product_cq15_sse`.
        let re = _mm512_add_epi32(
            _mm512_madd_epi16(x, a),
            _mm512_add_epi32(_mm512_srai_epi32::<16>(x), round),
        );
        let im = _mm512_add_epi32(_mm512_madd_epi16(x, b), round);
        let v = _mm512_packs_epi32(_mm512_srai_epi32::<15>(re), _mm512_srai_epi32::<15>(im));
        _mm512_shuffle_epi8(v, c.zip_b)
    }

    fearless_simd::kernel!(
        pub(super) fn cq15_avx512(
            t: Avx512,
            x: &[i16],
            p: &[u8],
            y: &mut [i16],
            plan: &Plan,
        ) -> usize {
            let permute = plan.entries() <= AVX512_PERMUTE_MAX;
            let c = Cq15x16 {
                inv: if plan.neg == 0 { 0 } else { u16::MAX },
                qshift: _mm_cvtsi32_si128(plan.qshift as i32),
                mask: _mm512_set1_epi32(plan.mask as i32),
                base: _mm512_set1_epi32(plan.base as i32),
                neg_t: u32x16::simd_from(t, table16(Q_NEG)).into(),
                zip_b: _mm512_broadcast_i32x4(_mm_setr_epi8(
                    0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
                )),
                la: lookup512d(if permute { plan.slice(&A_Q15) } else { &[] }),
                lb: lookup512d(if permute { plan.slice(&B_Q15) } else { &[] }),
            };
            let (xs, _) = x.as_chunks::<32>();
            let (ps, _) = p.as_chunks::<16>();
            let (ys, _) = y.as_chunks_mut::<32>();
            let mut n = 0;
            let form = if !plan.phasor() {
                0
            } else if permute {
                1
            } else {
                2
            };
            for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
                let xr: __m512i = i16x32::from_slice(t, xv).into();
                let pb: __m128i = u8x16::from_slice(t, pv).into();
                let v = match form {
                    0 => cq15_x16::<0>(xr, pb, &c),
                    1 => cq15_x16::<1>(xr, pb, &c),
                    _ => cq15_x16::<2>(xr, pb, &c),
                };
                i16x32::simd_from(t, v).store_array(yv);
                n += 16;
            }
            n
        }
    );

    /// `t` repeated four times (sixteen lanes, `vpermd` by an index whose low two bits are
    /// the quadrant).
    #[inline(always)]
    fn table16(t: [u32; 4]) -> [u32; 16] {
        core::array::from_fn(|i| t.get(i % 4).copied().unwrap_or(0))
    }

    /// `[lo | hi << 32]` per quadrant, repeated twice (eight lanes, `vpermq`).
    #[inline(always)]
    fn table8q(lo: [u32; 4], hi: [u32; 4]) -> [u64; 8] {
        core::array::from_fn(|i| {
            let (l, h) = (lo.get(i % 4).copied(), hi.get(i % 4).copied());
            u64::from(l.unwrap_or(0)) | (u64::from(h.unwrap_or(0)) << 32)
        })
    }

    /// The constants of an AVX-512 `C32` call (eight samples per step).
    struct C32x8 {
        inv: u8,
        qshift: __m128i,
        mask: __m512i,
        sign_t: __m512i,
        lc: Lookup512q,
        ls: Lookup512q,
    }

    /// Eight `C32` samples `x` (one per 64-bit lane) with the phase bytes `p`. `FORM`: 0 the
    /// quadrant map (`b ≤ 2`), 1 the phasor `[c c]`, `[−s s]` from the register tables
    /// (`x · [c c] + swap(x) · [−s s]`).
    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    fn c32_x8<const FORM: u8>(x: __m512, p: [u8; 8], c: &C32x8) -> __m512 {
        let v = _mm512_cvtepu8_epi64(_mm_cvtsi64_si128(i64::from_le_bytes(p)));
        let k = _mm512_mask_sub_epi64(v, c.inv, _mm512_setzero_si512(), v);
        if FORM == 0 {
            let q = _mm512_srl_epi64(_mm512_slli_epi64::<1>(k), c.qshift);
            let swap = _mm512_test_epi64_mask(q, _mm512_set1_epi64(1));
            let v = _mm512_castpd_ps(_mm512_mask_blend_pd(
                swap,
                _mm512_castps_pd(x),
                _mm512_castps_pd(_mm512_permute_ps::<0xB1>(x)),
            ));
            let sign = _mm512_permutexvar_epi64(q, c.sign_t);
            return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(v), sign));
        }
        let k = _mm512_and_si512(k, c.mask);
        let cv = _mm512_castsi512_ps(get512q(&c.lc, k));
        let sv = _mm512_castsi512_ps(get512q(&c.ls, k));
        _mm512_add_ps(
            _mm512_mul_ps(x, cv),
            _mm512_mul_ps(_mm512_permute_ps::<0xB1>(x), sv),
        )
    }

    /// The constants of [`c32_x8`]: the quadrant sign table and the phasor tables `[c c]` and
    /// `[−s s]` of the `2^b` entries (up to [`AVX512_PERMUTE_MAX`]).
    #[target_feature(enable = "avx512f")]
    #[inline]
    fn c32x8(plan: &Plan) -> C32x8 {
        let sign_t = table8q(Q_SIGN_RE, Q_SIGN_IM);
        let tables = if plan.phasor() {
            plan.slice(&CS_C32)
        } else {
            &[]
        };
        C32x8 {
            inv: if plan.neg == 0 { 0 } else { u8::MAX },
            qshift: _mm_cvtsi32_si128(plan.qshift as i32),
            mask: _mm512_set1_epi64(i64::from(plan.mask)),
            // SAFETY: `sign_t` is eight `u64`, 64 bytes.
            sign_t: unsafe { _mm512_loadu_si512(sign_t.as_ptr().cast()) },
            // `[c c]` and `[−s s]` of every entry, from the halves of `cs`.
            lc: lookup512q(tables, |w| (w & 0xFFFF_FFFF) | (w << 32)),
            ls: lookup512q(tables, |w| ((w >> 32) ^ 0x8000_0000) | (w & !0xFFFF_FFFF)),
        }
    }

    /// The constants of the planar `C32` form of sixteen samples (`b` 7 and 8).
    struct C32x16 {
        inv: u16,
        mask: __m512i,
        base: __m512i,
        /// `vpermt2ps` controls: the even and the odd elements, and the interleave.
        even: __m512i,
        odd: __m512i,
        zip_lo: __m512i,
        zip_hi: __m512i,
        /// `0.0` in every lane, the start value of the gathers ([`gather16`]).
        zero: __m512,
    }

    /// Sixteen dword indices `f(0) .. f(15)`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    fn idx16(f: impl Fn(u32) -> u32) -> __m512i {
        let a: [u32; 16] = core::array::from_fn(|i| f(i as u32));
        // SAFETY: `a` is sixteen `u32`, 64 bytes.
        unsafe { _mm512_loadu_si512(a.as_ptr().cast()) }
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn c32x16(plan: &Plan) -> C32x16 {
        C32x16 {
            inv: if plan.neg == 0 { 0 } else { u16::MAX },
            mask: _mm512_set1_epi32(plan.mask as i32),
            base: _mm512_set1_epi32(plan.base as i32),
            even: idx16(|i| 2 * i),
            odd: idx16(|i| 2 * i + 1),
            zip_lo: idx16(|i| i / 2 + 16 * (i % 2)),
            zip_hi: idx16(|i| 8 + i / 2 + 16 * (i % 2)),
            zero: _mm512_setzero_ps(),
        }
    }

    /// The sixteen entries `e` of the static table `t` (`vgatherdps`). The destination is
    /// written by a `vmovaps` of `zero` (a register the loop keeps) before the gather, in an
    /// `asm!` block: the zero idiom the compiler emits for it (`vxorps`) waits for the gather of
    /// the step before in the `sapphirerapids` model, which serialised the loop (2.66 cycles
    /// per sample, about 30 cycles per gather; with the copy 0.90, D2820). The wait exists only
    /// in the simulator: real cores break a zero idiom's dependency at rename, so on hardware
    /// the copy gains nothing (open item IO1-8).
    #[target_feature(enable = "avx512f")]
    #[inline]
    fn gather16(t: &'static [f32; ENTRIES], e: __m512i, zero: __m512) -> __m512 {
        let r: __m512;
        // SAFETY: every lane of `e` is an entry `2^b + k` below 512 (see the caller), the length
        // of `t`; the gather reads the sixteen floats, writes `r` and clears the mask register,
        // which is an output of the block. No stack, no flags; AVX-512F is a target feature.
        unsafe {
            core::arch::asm!(
                "vmovaps {r}, {z}",
                "kmovw {k}, {ones}",
                "vgatherdps {r}{{{k}}}, [{t} + {e} * 4]",
                r = out(zmm_reg) r,
                k = out(kreg) _,
                z = in(zmm_reg) zero,
                ones = in(kreg) u16::MAX,
                t = in(reg) t.as_ptr(),
                e = in(zmm_reg) e,
                options(pure, readonly, nostack, preserves_flags)
            );
        }
        r
    }

    /// Sixteen `C32` samples `x0`, `x1` with the phase bytes `p` in the planar form: `re` and
    /// `im` by `vpermt2ps`, `c` and `s` gathered from the static tables ([`gather16`]), the
    /// product in the order of the fallback, interleaved again by `vpermt2ps`.
    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    fn c32_x16_gather(x0: __m512, x1: __m512, p: __m128i, c: &C32x16) -> [__m512; 2] {
        let v = _mm512_cvtepu8_epi32(p);
        let k = _mm512_mask_sub_epi32(v, c.inv, _mm512_setzero_si512(), v);
        // Every lane is `2^b + k`, `k < 2^b`, `b` 7 or 8: below 512.
        let e = _mm512_or_si512(_mm512_and_si512(k, c.mask), c.base);
        let (cv, sv) = (gather16(&C_F32, e, c.zero), gather16(&S_F32, e, c.zero));
        let re = _mm512_permutex2var_ps(x0, c.even, x1);
        let im = _mm512_permutex2var_ps(x0, c.odd, x1);
        let pre = _mm512_sub_ps(_mm512_mul_ps(re, cv), _mm512_mul_ps(im, sv));
        let pim = _mm512_add_ps(_mm512_mul_ps(re, sv), _mm512_mul_ps(im, cv));
        [
            _mm512_permutex2var_ps(pre, c.zip_lo, pim),
            _mm512_permutex2var_ps(pre, c.zip_hi, pim),
        ]
    }

    fearless_simd::kernel!(
        /// `b` 7 and 8: sixteen samples per step in the planar form, the phasors gathered.
        fn c32_avx512_gather(t: Avx512, x: &[f32], p: &[u8], y: &mut [f32], plan: &Plan) -> usize {
            let c = c32x16(plan);
            let (xs, _) = x.as_chunks::<32>();
            let (ps, _) = p.as_chunks::<16>();
            let (ys, _) = y.as_chunks_mut::<32>();
            let mut n = 0;
            for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
                let (lo, hi) = xv.split_at(16);
                let x0: __m512 = f32x16::from_slice(t, lo).into();
                let x1: __m512 = f32x16::from_slice(t, hi).into();
                let pb: __m128i = u8x16::from_slice(t, pv).into();
                let [y0, y1] = c32_x16_gather(x0, x1, pb, &c);
                let (lo, hi) = yv.split_at_mut(16);
                f32x16::simd_from(t, y0).store_slice(lo);
                f32x16::simd_from(t, y1).store_slice(hi);
                n += 16;
            }
            n
        }
    );

    fearless_simd::kernel!(
        pub(super) fn c32_avx512(
            t: Avx512,
            x: &[f32],
            p: &[u8],
            y: &mut [f32],
            plan: &Plan,
        ) -> usize {
            let mut n = 0;
            if plan.phasor() && plan.entries() > AVX512_PERMUTE_MAX {
                return c32_avx512_gather(t, x, p, y, plan);
            }
            let c = c32x8(plan);
            let (xs, _) = x.as_chunks::<16>();
            let (ps, _) = p.as_chunks::<8>();
            let (ys, _) = y.as_chunks_mut::<16>();
            let phasor = plan.phasor();
            for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
                let xr: __m512 = f32x16::from_slice(t, xv).into();
                let v = if phasor {
                    c32_x8::<1>(xr, *pv, &c)
                } else {
                    c32_x8::<0>(xr, *pv, &c)
                };
                f32x16::simd_from(t, v).store_array(yv);
                n += 8;
            }
            n
        }
    );
}

/// The path of AArch64: eight samples per step in the planar form of `ld2`/`st2`, the indices
/// and the quadrant masks as bytes (`tbl`), widened to the lanes.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon {
    use core::arch::aarch64::{
        float32x4_t, float32x4x2_t, int8x8_t, int16x8_t, int16x8x2_t, uint8x8_t, uint8x16_t,
        uint8x16x4_t, uint16x8_t, uint32x4_t, vaddq_f32, vand_u8, vandq_u32, vbslq_f32, vbslq_s16,
        vcombine_s16, vcombine_u8, vcreate_u8, vdup_n_s8, vdup_n_u8, vdupq_n_u8, vdupq_n_u16,
        vdupq_n_u32, vdupq_n_u64, veor_u8, veorq_s16, veorq_u32, vget_high_s16, vget_low_s16,
        vld1_u8, vld1q_f32, vld1q_lane_u32, vld1q_lane_u64, vld1q_u8, vld2q_f32, vld2q_s16,
        vmlal_s16, vmlsl_s16, vmovl_s8, vmovl_s16, vmull_s16, vmulq_f32, vorrq_u8, vqrshrn_n_s32,
        vqsubq_s16, vqtbl1q_u8, vqtbl4q_u8, vqtbx4q_u8, vreinterpret_s8_u8, vreinterpretq_f32_u8,
        vreinterpretq_f32_u32, vreinterpretq_f32_u64, vreinterpretq_s16_u8, vreinterpretq_s16_u32,
        vreinterpretq_u8_u16, vreinterpretq_u16_s16, vreinterpretq_u32_f32, vreinterpretq_u32_s32,
        vreinterpretq_u32_u8, vshl_n_u8, vshl_u8, vshlq_n_u8, vst1q_f32, vst2q_f32, vst2q_s16,
        vsub_u8, vsubq_f32, vsubq_u8, vtbl1_u8, vuzp1q_f32, vuzp1q_s16, vuzp2q_f32, vuzp2q_s16,
        vzip1q_u8,
    };

    use super::{CS_C32, CS_Q15, Plan};

    /// The index constants of a call (bytes).
    #[derive(Clone, Copy)]
    struct Idx {
        neg: uint8x8_t,
        /// `+1` for `b = 1`, `−(b − 2)` otherwise: `q = (k << shift) & 3` (`ushl`, `b ≤ 2`).
        qshift: int8x8_t,
        /// `2^b − 1` (`b ≥ 3`).
        mask: uint8x8_t,
        three: uint8x8_t,
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn idx(plan: &Plan) -> Idx {
        let bits = plan.bits;
        let shift = if bits == 1 { 1 } else { -((bits - 2) as i8) };
        Idx {
            neg: vdup_n_u8(plan.neg as u8),
            qshift: vdup_n_s8(shift),
            mask: vdup_n_u8(plan.mask as u8),
            three: vdup_n_u8(3),
        }
    }

    /// `k = ±p` of eight phase bytes (the bits above `b` are not cleared).
    #[target_feature(enable = "neon")]
    #[inline]
    fn k8(p: &[u8; 8], c: Idx) -> uint8x8_t {
        // SAFETY: `p` is eight bytes.
        let v = unsafe { vld1_u8(p.as_ptr()) };
        vsub_u8(veor_u8(v, c.neg), c.neg)
    }

    /// The quadrant `q` of eight phase bytes (`b ≤ 2`; the bits of `p` above `b` never reach
    /// `q`).
    #[target_feature(enable = "neon")]
    #[inline]
    fn q8(p: [u8; 8], c: Idx) -> uint8x8_t {
        vand_u8(vshl_u8(k8(&p, c), c.qshift), c.three)
    }

    /// The masks `(swap, neg re, neg im)` of eight quadrants as bytes 0 or 0xFF.
    #[target_feature(enable = "neon")]
    #[inline]
    fn quadrant_bytes(q: uint8x8_t) -> (uint8x8_t, uint8x8_t, uint8x8_t) {
        (
            vtbl1_u8(vcreate_u8(0xFF00_FF00), q),
            vtbl1_u8(vcreate_u8(0x00FF_FF00), q),
            vtbl1_u8(vcreate_u8(0xFFFF_0000), q),
        )
    }

    /// Up to 256 bytes of a byte-indexed table in `vqtbl4q`/`vqtbx4q` registers.
    #[derive(Clone, Copy)]
    struct Table {
        t: [uint8x16x4_t; 4],
        /// The number of 64-byte parts in use.
        parts: usize,
    }

    /// The table of the values `value(e)` of `width` bytes each (2 or 4) of the `entries`
    /// entries, `value(e)` little-endian at byte `width · e`.
    #[target_feature(enable = "neon")]
    #[inline]
    fn table(entries: usize, width: usize, value: impl Fn(usize) -> u32) -> Table {
        let mut t = [uint8x16x4_t(vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0)); 4];
        let parts = (entries * width).div_ceil(64).clamp(1, 4);
        for (part, d) in t.iter_mut().enumerate().take(parts) {
            let reg = |i: usize| -> uint8x16_t {
                let a: [u8; 16] = core::array::from_fn(|j| {
                    let byte = 64 * part + 16 * i + j;
                    let e = byte / width;
                    if e < entries {
                        (value(e) >> (8 * (byte % width))) as u8
                    } else {
                        0
                    }
                });
                // SAFETY: `a` is sixteen bytes.
                unsafe { vld1q_u8(a.as_ptr()) }
            };
            *d = uint8x16x4_t(reg(0), reg(1), reg(2), reg(3));
        }
        Table { t, parts }
    }

    /// The bytes of the table at the byte indices `i` (each below `64 · parts`).
    #[target_feature(enable = "neon")]
    #[inline]
    fn get(t: &Table, i: uint8x16_t) -> uint8x16_t {
        let [t0, t1, t2, t3] = t.t;
        let mut v = vqtbl4q_u8(t0, i);
        if t.parts > 1 {
            let i = vsubq_u8(i, vdupq_n_u8(64));
            v = vqtbx4q_u8(v, t1, i);
            if t.parts > 2 {
                let i2 = vsubq_u8(i, vdupq_n_u8(64));
                v = vqtbx4q_u8(v, t2, i2);
                let i3 = vsubq_u8(i2, vdupq_n_u8(64));
                v = vqtbx4q_u8(v, t3, i3);
            }
        }
        v
    }

    /// The largest number of `CQ15` phasors that NEON looks up by `tbl`/`tbx` (128 bytes per
    /// table, two parts); above it (`b` 7 and 8) one 32-bit load of `(c, s)` per sample.
    const CQ15_TBL_MAX: usize = 64;

    /// The largest number of `C32` phasors looked up by `tbl` (one 64-byte part per table);
    /// above it one 64-bit load of `(c, s)` per sample.
    const C32_TBL_MAX: usize = 16;

    /// The constants of a `CQ15` call: the `i16` cosines and sines of the phasors.
    struct Cq15 {
        c: Idx,
        tc: Table,
        ts: Table,
    }

    /// Eight `CQ15` samples (planar `re`, `im`). `FORM`: 0 the quadrant map (`b ≤ 2`), 1 the
    /// phasor by `tbl`, 2 the phasor by per-sample loads.
    #[target_feature(enable = "neon")]
    #[inline]
    fn cq15_x8<const FORM: u8>(
        re: int16x8_t,
        im: int16x8_t,
        p: [u8; 8],
        k: &Cq15,
        plan: &Plan,
    ) -> int16x8x2_t {
        if FORM == 0 {
            let (sw, nre, nim) = quadrant_bytes(q8(p, k.c));
            let sw: uint16x8_t = vreinterpretq_u16_s16(vmovl_s8(vreinterpret_s8_u8(sw)));
            let nre: int16x8_t = vmovl_s8(vreinterpret_s8_u8(nre));
            let nim: int16x8_t = vmovl_s8(vreinterpret_s8_u8(nim));
            let ore = vbslq_s16(sw, im, re);
            let oim = vbslq_s16(sw, re, im);
            return int16x8x2_t(
                vqsubq_s16(veorq_s16(ore, nre), nre),
                vqsubq_s16(veorq_s16(oim, nim), nim),
            );
        }
        let (cv, sv) = if FORM == 1 {
            // Byte indices `[2k, 2k + 1]` per sample.
            let k2 = vshl_n_u8::<1>(vand_u8(k8(&p, k.c), k.c.mask));
            let k2 = vcombine_u8(k2, k2);
            let i = vorrq_u8(vzip1q_u8(k2, k2), vreinterpretq_u8_u16(vdupq_n_u16(0x0100)));
            (
                vreinterpretq_s16_u8(get(&k.tc, i)),
                vreinterpretq_s16_u8(get(&k.ts, i)),
            )
        } else {
            // `c | s << 16` per sample by lane loads (`ld1 {v.s}[i]`), `uzp1`/`uzp2` into the
            // planar `c` and `s`.
            let e = p.map(|v| plan.entry(v));
            let t = CS_Q15.as_ptr();
            // SAFETY: every entry is below 512 (`Plan::entry` masks it), inside `CS_Q15`.
            let (lo, hi) = unsafe {
                let z = vdupq_n_u32(0);
                let lo = vld1q_lane_u32::<0>(t.add(e[0]), z);
                let lo = vld1q_lane_u32::<1>(t.add(e[1]), lo);
                let lo = vld1q_lane_u32::<2>(t.add(e[2]), lo);
                let lo = vld1q_lane_u32::<3>(t.add(e[3]), lo);
                let hi = vld1q_lane_u32::<0>(t.add(e[4]), z);
                let hi = vld1q_lane_u32::<1>(t.add(e[5]), hi);
                let hi = vld1q_lane_u32::<2>(t.add(e[6]), hi);
                let hi = vld1q_lane_u32::<3>(t.add(e[7]), hi);
                (vreinterpretq_s16_u32(lo), vreinterpretq_s16_u32(hi))
            };
            (vuzp1q_s16(lo, hi), vuzp2q_s16(lo, hi))
        };
        let (rl, rh) = (vget_low_s16(re), vget_high_s16(re));
        let (il, ih) = (vget_low_s16(im), vget_high_s16(im));
        let (cl, ch) = (vget_low_s16(cv), vget_high_s16(cv));
        let (sl, sh) = (vget_low_s16(sv), vget_high_s16(sv));
        int16x8x2_t(
            vcombine_s16(
                vqrshrn_n_s32::<15>(vmlsl_s16(vmull_s16(rl, cl), il, sl)),
                vqrshrn_n_s32::<15>(vmlsl_s16(vmull_s16(rh, ch), ih, sh)),
            ),
            vcombine_s16(
                vqrshrn_n_s32::<15>(vmlal_s16(vmull_s16(rl, sl), il, cl)),
                vqrshrn_n_s32::<15>(vmlal_s16(vmull_s16(rh, sh), ih, ch)),
            ),
        )
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn cq15_consts(plan: &Plan) -> Cq15 {
        let tbl = plan.phasor() && plan.entries() <= CQ15_TBL_MAX;
        let cs = if tbl { plan.slice(&CS_Q15) } else { &[] };
        let n = cs.len();
        Cq15 {
            c: idx(plan),
            tc: table(n, 2, |e| cs.get(e).map_or(0, |&w| w & 0xFFFF)),
            ts: table(n, 2, |e| cs.get(e).map_or(0, |&w| w >> 16)),
        }
    }

    fearless_simd::kernel!(
        pub(super) fn cq15(t: Neon, x: &[i16], p: &[u8], y: &mut [i16], plan: &Plan) -> usize {
            let _ = t;
            let k = cq15_consts(plan);
            let form = if !plan.phasor() {
                0
            } else if plan.entries() <= CQ15_TBL_MAX {
                1
            } else {
                2
            };
            let (xs, _) = x.as_chunks::<16>();
            let (ps, _) = p.as_chunks::<8>();
            let (ys, _) = y.as_chunks_mut::<16>();
            let mut n = 0;
            for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
                // SAFETY: `xv` is sixteen `i16`.
                let v = unsafe { vld2q_s16(xv.as_ptr()) };
                let out = match form {
                    0 => cq15_x8::<0>(v.0, v.1, *pv, &k, plan),
                    1 => cq15_x8::<1>(v.0, v.1, *pv, &k, plan),
                    _ => cq15_x8::<2>(v.0, v.1, *pv, &k, plan),
                };
                // SAFETY: `yv` is sixteen `i16`.
                unsafe { vst2q_s16(yv.as_mut_ptr(), out) };
                n += 8;
            }
            n
        }
    );

    /// The 32-bit masks of the four samples `4h .. 4h + 3` of the byte masks `m` (0 or 0xFF).
    #[target_feature(enable = "neon")]
    #[inline]
    fn widen32(m: uint8x8_t, h: usize) -> uint32x4_t {
        let w = vmovl_s8(vreinterpret_s8_u8(m));
        let w = if h == 0 {
            vget_low_s16(w)
        } else {
            vget_high_s16(w)
        };
        vreinterpretq_u32_s32(vmovl_s16(w))
    }

    /// The constants of a `C32` call: the `f32` cosines and sines of the phasors and the byte
    /// controls of the lookups.
    struct C32k {
        c: Idx,
        tc: Table,
        ts: Table,
        /// Byte `j` of sample `i` reads index byte `4h + i` (`rep[h]`), plus `j % 4` (`lane`).
        rep: [uint8x16_t; 2],
        lane: uint8x16_t,
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn c32_consts(plan: &Plan) -> C32k {
        let tbl = plan.phasor() && plan.entries() <= C32_TBL_MAX;
        let cs = if tbl { plan.slice(&CS_C32) } else { &[] };
        let n = cs.len();
        let ld = |f: &dyn Fn(usize) -> u8| -> uint8x16_t {
            let a: [u8; 16] = core::array::from_fn(f);
            // SAFETY: `a` is sixteen bytes.
            unsafe { vld1q_u8(a.as_ptr()) }
        };
        C32k {
            c: idx(plan),
            tc: table(n, 4, |e| cs.get(e).map_or(0, |&w| w as u32)),
            ts: table(n, 4, |e| cs.get(e).map_or(0, |&w| (w >> 32) as u32)),
            rep: [ld(&|j| (j / 4) as u8), ld(&|j| 4 + (j / 4) as u8)],
            lane: ld(&|j| (j % 4) as u8),
        }
    }

    /// Eight `C32` samples (`xv`, sixteen `f32`) into `yv`.
    ///
    /// `MODE`: 0 the quadrant map (`b ≤ 2`), 1 the phasor by `tbl` from the register tables
    /// (up to 16 phasors, one 64-byte part per table), 2 one 64-bit load of the pair `(c, s)`
    /// per sample and `uzp1`/`uzp2` (more phasors: four `tbl`/`tbx` per table and 32 table
    /// registers were slower than the fallback, M49 §5).
    #[target_feature(enable = "neon")]
    #[inline]
    fn c32_x8<const MODE: u8>(
        xv: &[f32; 16],
        pv: &[u8; 8],
        yv: &mut [f32; 16],
        k: &C32k,
        plan: &Plan,
    ) {
        let sign = vdupq_n_u32(0x8000_0000);
        let (sw, nre, nim) = if MODE == 0 {
            quadrant_bytes(q8(*pv, k.c))
        } else {
            let z = vdup_n_u8(0);
            (z, z, z)
        };
        let k4 = if MODE == 1 {
            let k4 = vshl_n_u8::<2>(vand_u8(k8(pv, k.c), k.c.mask));
            vcombine_u8(k4, k4)
        } else {
            vdupq_n_u8(0)
        };
        let (xh, _) = xv.as_chunks::<8>();
        let (yh, _) = yv.as_chunks_mut::<8>();
        for (h, ((xq, yq), rep)) in xh.iter().zip(yh).zip(k.rep).enumerate() {
            // SAFETY: `xq` is eight `f32`.
            let v = unsafe { vld2q_f32(xq.as_ptr()) };
            let (re, im): (float32x4_t, float32x4_t) = (v.0, v.1);
            let out = if MODE != 0 {
                let (cv, sv) = if MODE == 1 {
                    let i = vorrq_u8(vqtbl1q_u8(k4, rep), k.lane);
                    (
                        vreinterpretq_f32_u8(get(&k.tc, i)),
                        vreinterpretq_f32_u8(get(&k.ts, i)),
                    )
                } else {
                    let e = |j: usize| plan.entry(pv.get(4 * h + j).copied().unwrap_or(0));
                    let t = CS_C32.as_ptr();
                    // SAFETY: every entry is below 512 (`Plan::entry` masks it), inside `CS_C32`.
                    let (lo, hi) = unsafe {
                        let z = vdupq_n_u64(0);
                        let lo =
                            vld1q_lane_u64::<1>(t.add(e(1)), vld1q_lane_u64::<0>(t.add(e(0)), z));
                        let hi =
                            vld1q_lane_u64::<1>(t.add(e(3)), vld1q_lane_u64::<0>(t.add(e(2)), z));
                        (vreinterpretq_f32_u64(lo), vreinterpretq_f32_u64(hi))
                    };
                    (vuzp1q_f32(lo, hi), vuzp2q_f32(lo, hi))
                };
                float32x4x2_t(
                    vsubq_f32(vmulq_f32(re, cv), vmulq_f32(im, sv)),
                    vaddq_f32(vmulq_f32(re, sv), vmulq_f32(im, cv)),
                )
            } else {
                let w = widen32(sw, h);
                let sre = vandq_u32(widen32(nre, h), sign);
                let sim = vandq_u32(widen32(nim, h), sign);
                let ore = vreinterpretq_u32_f32(vbslq_f32(w, im, re));
                let oim = vreinterpretq_u32_f32(vbslq_f32(w, re, im));
                float32x4x2_t(
                    vreinterpretq_f32_u32(veorq_u32(ore, sre)),
                    vreinterpretq_f32_u32(veorq_u32(oim, sim)),
                )
            };
            // SAFETY: `yq` is eight `f32`.
            unsafe { vst2q_f32(yq.as_mut_ptr(), out) };
        }
    }

    /// `b = 1`: the sign bit of both components flipped where `p & 1`, sixteen interleaved
    /// samples per step (two per register). The phase bytes shifted left by 7 put `p & 1` in
    /// bit 7; one `tbl` per register moves the byte of each sample into the top byte of its two
    /// lanes (the other bytes are zero), so the mask is bit 31 alone. Returns the number of
    /// samples done.
    #[target_feature(enable = "neon")]
    #[inline]
    fn chip_c32(x: &[f32], p: &[u8], y: &mut [f32]) -> usize {
        const Z: u8 = 0xFF;
        let idx: [uint8x16_t; 8] = core::array::from_fn(|j| {
            let (a, b) = (2 * j as u8, 2 * j as u8 + 1);
            let t = [Z, Z, Z, a, Z, Z, Z, a, Z, Z, Z, b, Z, Z, Z, b];
            // SAFETY: `t` is sixteen bytes.
            unsafe { vld1q_u8(t.as_ptr()) }
        });
        let (xs, _) = x.as_chunks::<32>();
        let (ps, _) = p.as_chunks::<16>();
        let (ys, _) = y.as_chunks_mut::<32>();
        let mut n = 0;
        for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
            // SAFETY: `pv` is sixteen bytes.
            let b = vshlq_n_u8::<7>(unsafe { vld1q_u8(pv.as_ptr()) });
            let (xq, _) = xv.as_chunks::<4>();
            let (yq, _) = yv.as_chunks_mut::<4>();
            for ((xq, yq), i) in xq.iter().zip(yq).zip(idx) {
                let m = vreinterpretq_u32_u8(vqtbl1q_u8(b, i));
                // SAFETY: `xq` and `yq` are four `f32`.
                unsafe {
                    let v = vreinterpretq_u32_f32(vld1q_f32(xq.as_ptr()));
                    vst1q_f32(yq.as_mut_ptr(), vreinterpretq_f32_u32(veorq_u32(v, m)));
                }
            }
            n += 16;
        }
        n
    }

    fearless_simd::kernel!(
        pub(super) fn c32(t: Neon, x: &[f32], p: &[u8], y: &mut [f32], plan: &Plan) -> usize {
            let _ = t;
            if plan.bits == 1 {
                return chip_c32(x, p, y);
            }
            let k = c32_consts(plan);
            let mode = if !plan.phasor() {
                0
            } else if plan.entries() <= C32_TBL_MAX {
                1
            } else {
                2
            };
            let (xs, _) = x.as_chunks::<16>();
            let (ps, _) = p.as_chunks::<8>();
            let (ys, _) = y.as_chunks_mut::<16>();
            let it = xs.iter().zip(ps).zip(ys);
            match mode {
                0 => it.for_each(|((xv, pv), yv)| c32_x8::<0>(xv, pv, yv, &k, plan)),
                1 => it.for_each(|((xv, pv), yv)| c32_x8::<1>(xv, pv, yv, &k, plan)),
                _ => it.for_each(|((xv, pv), yv)| c32_x8::<2>(xv, pv, yv, &k, plan)),
            }
            8 * xs.len().min(ps.len()).min(y.len() / 16)
        }
    );
}

/// The path of wasm32 with `simd128`: four samples per step, as SSE4.2 (`i8x16.swizzle` for the
/// quadrant tables, `i32x4.dot_i16x8_s` for the product, the phasors by per-sample loads).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::{
        f32x4_add, f32x4_mul, f32x4_sub, i8x16, i8x16_swizzle, i16x8_narrow_i32x4, i16x8_shuffle,
        i16x8_sub_sat, i32x4, i32x4_add, i32x4_dot_i16x8, i32x4_shl, i32x4_shr, i32x4_shuffle,
        i32x4_splat, i32x4_sub, u8x16, u16x8_extend_low_u8x16, u32x4_extend_low_u16x8, u32x4_shr,
        u64x2, v128, v128_and, v128_bitselect, v128_or, v128_xor,
    };

    use fearless_simd::prelude::*;
    use fearless_simd::{WasmSimd128, f32x4 as F32x4, i16x8 as I16x8, u8x16 as U8x16};

    use super::{A_Q15, B_Q15, CS_C32, Plan, at};

    /// The quadrant `q` of four phase bytes in the 32-bit lanes (`b ≤ 2`).
    #[inline(always)]
    fn q4(p: &[u8; 4], plan: &Plan) -> v128 {
        let v = i32x4(i32::from_le_bytes(*p), 0, 0, 0);
        let v = u32x4_extend_low_u16x8(u16x8_extend_low_u8x16(v));
        let neg = i32x4_splat(plan.neg as i32);
        // `k = ±p`; the bits of `p` above `b` never reach `q`.
        let k = i32x4_sub(v128_xor(v, neg), neg);
        v128_and(u32x4_shr(i32x4_shl(k, 1), plan.qshift), i32x4_splat(3))
    }

    fearless_simd::kernel!(
        pub(super) fn cq15(
            t: WasmSimd128,
            x: &[i16],
            p: &[u8],
            y: &mut [i16],
            plan: &Plan,
        ) -> usize {
            let rep = i8x16(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12);
            let half = i8x16(0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4);
            let swap_t = i8x16(0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            let neg_t = i8x16(0, -1, -1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0);
            let round = i32x4_splat(1 << 14);
            let phasor = plan.phasor();
            let (xs, _) = x.as_chunks::<8>();
            let (ps, _) = p.as_chunks::<4>();
            let (ys, _) = y.as_chunks_mut::<8>();
            let mut n = 0;
            for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
                let xr: v128 = I16x8::from_slice(t, xv).into();
                let v = if phasor {
                    let es = pv.map(|b| plan.entry(b));
                    let e = |i: usize| es.get(i).copied().unwrap_or(0);
                    let at_a = |i: usize| at(&A_Q15, e(i)) as i32;
                    let at_b = |i: usize| at(&B_Q15, e(i)) as i32;
                    let a = i32x4(at_a(0), at_a(1), at_a(2), at_a(3));
                    let b = i32x4(at_b(0), at_b(1), at_b(2), at_b(3));
                    // The real part `xr c + xi (−s − 1) + xi` (`a = (c, !s)`), as on x86.
                    let re = i32x4_add(i32x4_dot_i16x8(xr, a), i32x4_add(i32x4_shr(xr, 16), round));
                    let im = i32x4_add(i32x4_dot_i16x8(xr, b), round);
                    let v = i16x8_narrow_i32x4(i32x4_shr(re, 15), i32x4_shr(im, 15));
                    i16x8_shuffle::<0, 4, 1, 5, 2, 6, 3, 7>(v, v)
                } else {
                    let qrep = i8x16_swizzle(q4(pv, plan), rep);
                    let swap = i8x16_swizzle(swap_t, qrep);
                    let neg = i8x16_swizzle(neg_t, v128_or(qrep, half));
                    let v =
                        v128_bitselect(i16x8_shuffle::<1, 0, 3, 2, 5, 4, 7, 6>(xr, xr), xr, swap);
                    i16x8_sub_sat(v128_xor(v, neg), neg)
                };
                I16x8::simd_from(t, v).store_array(yv);
                n += 4;
            }
            n
        }
    );

    /// `b = 1`: sixteen interleaved `C32` samples per step (two per register) with the sign
    /// bit of both components flipped where `p & 1`: one `i8x16.swizzle` per register puts the
    /// byte of each sample into the low byte of its two lanes (index 255 gives 0), `i32x4.shl
    /// 31` keeps bit 0 of it. Returns the number of samples done.
    #[inline(always)]
    fn chip_c32(t: WasmSimd128, x: &[f32], p: &[u8], y: &mut [f32]) -> usize {
        const Z: u8 = 0xFF;
        let idx: [v128; 8] = core::array::from_fn(|j| {
            let (a, b) = (2 * j as u8, 2 * j as u8 + 1);
            u8x16(a, Z, Z, Z, a, Z, Z, Z, b, Z, Z, Z, b, Z, Z, Z)
        });
        let (xs, _) = x.as_chunks::<32>();
        let (ps, _) = p.as_chunks::<16>();
        let (ys, _) = y.as_chunks_mut::<32>();
        let mut n = 0;
        for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
            let b: v128 = U8x16::from_slice(t, pv).into();
            let x4: &[[f32; 4]; 8] = bytemuck::cast_ref(xv);
            let y4: &mut [[f32; 4]; 8] = bytemuck::cast_mut(yv);
            for ((xr, yr), i) in x4.iter().zip(y4.iter_mut()).zip(idx) {
                let m = i32x4_shl(i8x16_swizzle(b, i), 31);
                let v: v128 = F32x4::from_slice(t, xr).into();
                F32x4::simd_from(t, v128_xor(v, m)).store_array(yr);
            }
            n += 16;
        }
        n
    }

    fearless_simd::kernel!(
        pub(super) fn c32(
            t: WasmSimd128,
            x: &[f32],
            p: &[u8],
            y: &mut [f32],
            plan: &Plan,
        ) -> usize {
            if plan.bits == 1 {
                return chip_c32(t, x, p, y);
            }
            let phasor = plan.phasor();
            let (xs, _) = x.as_chunks::<8>();
            let (ps, _) = p.as_chunks::<4>();
            let (ys, _) = y.as_chunks_mut::<8>();
            let mut n = 0;
            for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
                let (lo, hi) = xv.split_at(4);
                let x0: v128 = F32x4::from_slice(t, lo).into();
                let x1: v128 = F32x4::from_slice(t, hi).into();
                // The planar form, as SSE.
                let re = i32x4_shuffle::<0, 2, 4, 6>(x0, x1);
                let im = i32x4_shuffle::<1, 3, 5, 7>(x0, x1);
                let (ore, oim) = if phasor {
                    // One 64-bit load of the pair `(c, s)` per sample, split by two shuffles.
                    let [w0, w1, w2, w3] = pv.map(|b| at(&CS_C32, plan.entry(b)));
                    let v01 = u64x2(w0, w1);
                    let v23 = u64x2(w2, w3);
                    let cv = i32x4_shuffle::<0, 2, 4, 6>(v01, v23);
                    let sv = i32x4_shuffle::<1, 3, 5, 7>(v01, v23);
                    (
                        f32x4_sub(f32x4_mul(re, cv), f32x4_mul(im, sv)),
                        f32x4_add(f32x4_mul(re, sv), f32x4_mul(im, cv)),
                    )
                } else {
                    let q = q4(pv, plan);
                    let swap = i32x4_shr(i32x4_shl(q, 31), 31);
                    let q1 = u32x4_shr(q, 1);
                    let sre = i32x4_shl(v128_xor(q, q1), 31);
                    let sim = i32x4_shl(q1, 31);
                    (
                        v128_xor(v128_bitselect(im, re, swap), sre),
                        v128_xor(v128_bitselect(re, im, swap), sim),
                    )
                };
                let y0 = i32x4_shuffle::<0, 4, 1, 5>(ore, oim);
                let y1 = i32x4_shuffle::<2, 6, 3, 7>(ore, oim);
                let (lo, hi) = yv.split_at_mut(4);
                F32x4::simd_from(t, y0).store_slice(lo);
                F32x4::simd_from(t, y1).store_slice(hi);
                n += 4;
            }
            n
        }
    );
}
