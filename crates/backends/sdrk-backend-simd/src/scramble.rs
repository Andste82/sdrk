//! `scramble` (M49 §5): keyed phase scrambling, bit-identical to the fallback on every level.
//!
//! Every lane runs the element formula of M49 §2, so the bits follow from the arithmetic:
//!
//! 1. **Indices.** The phase bytes are widened to one lane per sample (`pmovzxbd`/`pmovzxbq`,
//!    unpacks on SSE2, `u32x4.extend` on SIMD128) or kept as bytes (NEON); `k = ±p` (`psignd`,
//!    a masked subtraction on AVX-512, `(p ^ m) − m` with `m = −1` for `inverse` elsewhere),
//!    the quadrant `q` in the low two bits of `(k << 1) >> (b − 1)` and the residue
//!    `r = k & (2^(b−2) − 1)`. The bits of `p` above `b` never reach `q` or `r`, so `k` needs no
//!    mask by `2^b − 1`.
//! 2. **Residual product (`b ≥ 3`).** The phasor `(c, s)` of the residue comes from a table of the
//!    call (at most 64 entries, built from `COS_F32`/`COS_Q15` by the public helpers of
//!    `sdrk-core`): an in-register permute where the table fits (`vpermd` up to 8 entries, two
//!    and a blend up to 16 on AVX2; `vpermd`, `vpermi2d`/`vpermi2q` and blends up to 64 on
//!    AVX-512; `tbl`/`tbx` on NEON), otherwise one 64-bit load of the pair per sample (SSE2,
//!    SSE4.2, AVX2 above 16 entries, SIMD128). `CQ15`: `pmaddwd` of the `(re, im)` pairs with
//!    `(c, −s)` and `(s, c)` (`i32x4.dot_i16x8_s`; on NEON `smull`/`smlsl`/`smlal` on the planar
//!    form), plus `2^14`, an arithmetic shift by 15 and the saturating narrow (`sqrshrn #15` on
//!    NEON): the `cq15_mul` of the fallback, exact in `i32`. `C32`: `(re c − im s, re s + im c)`
//!    by two products and one sum per component in the order of the fallback (planar on every
//!    level but AVX-512, which forms the first component as `re c + im (−s)`, the same value for
//!    every non-NaN result; NaN by class, D296), no fused multiply-add. The samples with `r = 0`
//!    are **selected** from the input (a blend), never multiplied.
//! 3. **Quadrant (every `b`).** Swap where `q & 1` (a shuffle and a blend, a masked rotate on
//!    AVX-512 `CQ15`, a blend of the planar `re` and `im` for `C32`), then negate `re` where
//!    `(q ^ (q >> 1)) & 1` and `im` where `(q >> 1) & 1`: `(x ^ m) −sat m` in `CQ15` (32767 for
//!    −32768), XOR with the sign bit in `C32`. The masks come from a table of the four quadrants
//!    (`vpermd`, `pshufb`, `tbl`, `i8x16.swizzle`) or from shifts of `q`. For `b ≤ 2` this is
//!    the whole kernel.
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
    Scramble, check_args, phase_index, residual_c32, residual_q15, scramble_one_c32,
    scramble_one_cq15,
};
use sdrk_core::{C32, CQ15, Result};

use crate::Simd;
use crate::level::dispatch_fn;

/// The largest number of residues (`b = 8`).
const RESIDUES: usize = 64;

/// The constants of one call.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Plan {
    /// All ones for `inverse` (`k = (p ^ m) − m`), else 0.
    neg: u32,
    /// `b − 1`: `q = ((k << 1) >> (b − 1)) & 3`.
    qshift: u32,
    /// `2^(b−2) − 1` (0 for `b ≤ 2`): `r = k & rmask`.
    rmask: u32,
}

impl Plan {
    fn new(bits: u32, inverse: bool) -> Self {
        Self {
            neg: if inverse { u32::MAX } else { 0 },
            qshift: bits - 1,
            rmask: if bits >= 2 { (1 << (bits - 2)) - 1 } else { 0 },
        }
    }

    /// Whether the call has a residual product (`b ≥ 3`).
    #[inline(always)]
    fn res(&self) -> bool {
        self.rmask != 0
    }

    /// The number of residues `2^(b−2)` (1 for `b ≤ 2`).
    #[inline(always)]
    fn residues(&self) -> usize {
        self.rmask as usize + 1
    }

    /// The residue of the phase byte `p` (scalar, for the per-sample table loads).
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    #[inline(always)]
    fn residue(&self, p: u8) -> usize {
        ((u32::from(p) ^ self.neg).wrapping_sub(self.neg) & self.rmask) as usize & (RESIDUES - 1)
    }
}

/// The `CQ15` phasors of the residues as the `pmaddwd` operands: `a[r] = (c, −s)`,
/// `b[r] = (s, c)` (low half first), and both as one word `ab[r] = a[r] | b[r] << 32` for the
/// per-sample loads; entry 0 (`r = 0`, selected from the input) is 0.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub(crate) struct CoefQ15 {
    a: [u32; RESIDUES],
    b: [u32; RESIDUES],
    ab: [u64; RESIDUES],
}

impl CoefQ15 {
    fn new(bits: u32, plan: &Plan) -> Self {
        let mut t = Self {
            a: [0; RESIDUES],
            b: [0; RESIDUES],
            ab: [0; RESIDUES],
        };
        if plan.res() {
            let pack = |lo: i16, hi: i16| u32::from(lo as u16) | (u32::from(hi as u16) << 16);
            for (r, ((a, b), ab)) in
                t.a.iter_mut()
                    .zip(&mut t.b)
                    .zip(&mut t.ab)
                    .enumerate()
                    .take(plan.residues())
                    .skip(1)
            {
                if let Some((c, s)) = residual_q15(r as u32, bits) {
                    *a = pack(c, -s);
                    *b = pack(s, c);
                    *ab = u64::from(*a) | (u64::from(*b) << 32);
                }
            }
        }
        t
    }

    #[cfg_attr(target_arch = "x86_64", allow(dead_code))]
    #[inline(always)]
    fn a(&self, r: usize) -> u32 {
        self.a.get(r).copied().unwrap_or(0)
    }

    #[cfg_attr(target_arch = "x86_64", allow(dead_code))]
    #[inline(always)]
    fn b(&self, r: usize) -> u32 {
        self.b.get(r).copied().unwrap_or(0)
    }

    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    #[inline(always)]
    fn ab(&self, r: usize) -> u64 {
        self.ab.get(r).copied().unwrap_or(0)
    }
}

/// The `C32` phasors of the residues: `c[r]`, `s[r]`, and both as one word
/// `cs[r] = bits(c) | bits(s) << 32` for the per-sample loads; entry 0 is 0.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub(crate) struct CoefC32 {
    c: [f32; RESIDUES],
    s: [f32; RESIDUES],
    cs: [u64; RESIDUES],
}

impl CoefC32 {
    fn new(bits: u32, plan: &Plan) -> Self {
        let mut t = Self {
            c: [0.0; RESIDUES],
            s: [0.0; RESIDUES],
            cs: [0; RESIDUES],
        };
        if plan.res() {
            for (r, ((c, s), cs)) in
                t.c.iter_mut()
                    .zip(&mut t.s)
                    .zip(&mut t.cs)
                    .enumerate()
                    .take(plan.residues())
                    .skip(1)
            {
                if let Some((cv, sv)) = residual_c32(r as u32, bits) {
                    (*c, *s) = (cv, sv);
                    *cs = u64::from(cv.to_bits()) | (u64::from(sv.to_bits()) << 32);
                }
            }
        }
        t
    }

    #[cfg_attr(target_arch = "x86_64", allow(dead_code))]
    #[inline(always)]
    fn c(&self, r: usize) -> f32 {
        self.c.get(r).copied().unwrap_or(0.0)
    }

    #[cfg_attr(target_arch = "x86_64", allow(dead_code))]
    #[inline(always)]
    fn s(&self, r: usize) -> f32 {
        self.s.get(r).copied().unwrap_or(0.0)
    }

    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    #[inline(always)]
    fn cs(&self, r: usize) -> u64 {
        self.cs.get(r).copied().unwrap_or(0)
    }
}

/// The level path for `CQ15` (samples as `i16` pairs, `re` first; `i16` because a `CQ15` is
/// only 2-byte aligned): returns the number of leading samples done.
#[inline(always)]
fn cq15_level<S: SimdToken>(
    simd: S,
    x: &[i16],
    p: &[u8],
    y: &mut [i16],
    plan: &Plan,
    coef: &CoefQ15,
) -> usize {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx512() {
            return x86::cq15_avx512(t, x, p, y, plan, coef);
        }
        if let Some(t) = level.as_avx2() {
            return x86::cq15_avx2(t, x, p, y, plan, coef);
        }
        if let Some(t) = level.as_sse4_2() {
            return x86::cq15_sse4_2(t, x, p, y, plan, coef);
        }
        if let Some(t) = level.as_sse2() {
            return x86::cq15_sse2(t, x, p, y, plan, coef);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return neon::cq15(t, x, p, y, plan, coef);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return wasm::cq15(t, x, p, y, plan, coef);
    }
    let _ = (level, x, p, y, plan, coef);
    0
}

/// The level path for `C32` (samples as two `f32`): returns the number of leading samples done.
#[inline(always)]
fn c32_level<S: SimdToken>(
    simd: S,
    x: &[f32],
    p: &[u8],
    y: &mut [f32],
    plan: &Plan,
    coef: &CoefC32,
) -> usize {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx512() {
            return x86::c32_avx512(t, x, p, y, plan, coef);
        }
        if let Some(t) = level.as_avx2() {
            return x86::c32_avx2(t, x, p, y, plan, coef);
        }
        if let Some(t) = level.as_sse4_2() {
            return x86::c32_sse4_2(t, x, p, y, plan, coef);
        }
        if let Some(t) = level.as_sse2() {
            return x86::c32_sse2(t, x, p, y, plan, coef);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return neon::c32(t, x, p, y, plan, coef);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return wasm::c32(t, x, p, y, plan, coef);
    }
    let _ = (level, x, p, y, plan, coef);
    0
}

#[inline(always)]
fn cq15_run<S: SimdToken>(
    simd: S,
    x: &[i16],
    p: &[u8],
    y: &mut [i16],
    plan: &Plan,
    coef: &CoefQ15,
    done: &mut usize,
) {
    *done = cq15_level(simd, x, p, y, plan, coef);
}

#[inline(always)]
fn c32_run<S: SimdToken>(
    simd: S,
    x: &[f32],
    p: &[u8],
    y: &mut [f32],
    plan: &Plan,
    coef: &CoefC32,
    done: &mut usize,
) {
    *done = c32_level(simd, x, p, y, plan, coef);
}

// The entries, one function per level (see `dispatch_fn!`).
dispatch_fn!(
    fn run_cq15(
        x: &[i16],
        p: &[u8],
        y: &mut [i16],
        plan: &Plan,
        coef: &CoefQ15,
        done: &mut usize,
    ) => cq15_run
);
dispatch_fn!(
    fn run_c32(
        x: &[f32],
        p: &[u8],
        y: &mut [f32],
        plan: &Plan,
        coef: &CoefC32,
        done: &mut usize,
    ) => c32_run
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
        let coef = CoefQ15::new(bits, &plan);
        let mut done = 0;
        run_cq15(
            bytemuck::cast_slice(input),
            phase,
            bytemuck::cast_slice_mut(output),
            &plan,
            &coef,
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
        let coef = CoefC32::new(bits, &plan);
        let mut done = 0;
        run_c32(
            bytemuck::cast_slice(input),
            phase,
            bytemuck::cast_slice_mut(output),
            &plan,
            &coef,
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
        _mm_castps_si128, _mm_castsi128_ps, _mm_cmpeq_epi32, _mm_cvtepu8_epi32, _mm_cvtsi32_si128,
        _mm_cvtsi64_si128, _mm_madd_epi16, _mm_mul_ps, _mm_or_ps, _mm_or_si128, _mm_packs_epi32,
        _mm_set_epi64x, _mm_set1_epi32, _mm_setr_epi8, _mm_setzero_si128, _mm_shuffle_epi8,
        _mm_shuffle_ps, _mm_shufflehi_epi16, _mm_shufflelo_epi16, _mm_sign_epi32, _mm_slli_epi32,
        _mm_srai_epi16, _mm_srai_epi32, _mm_srl_epi32, _mm_srli_epi32, _mm_sub_epi32, _mm_sub_ps,
        _mm_subs_epi16, _mm_unpackhi_epi64, _mm_unpackhi_ps, _mm_unpacklo_epi8, _mm_unpacklo_epi16,
        _mm_unpacklo_ps, _mm_xor_ps, _mm_xor_si128, _mm256_add_epi32, _mm256_add_ps,
        _mm256_and_si256, _mm256_blendv_epi8, _mm256_blendv_ps, _mm256_broadcastsi128_si256,
        _mm256_castps_si256, _mm256_castsi256_ps, _mm256_cmpeq_epi32, _mm256_cvtepu8_epi32,
        _mm256_madd_epi16, _mm256_mul_ps, _mm256_packs_epi32, _mm256_permute4x64_epi64,
        _mm256_permutevar8x32_epi32, _mm256_set_epi64x, _mm256_set1_epi32, _mm256_setzero_si256,
        _mm256_shuffle_epi8, _mm256_shuffle_ps, _mm256_sign_epi32, _mm256_slli_epi32,
        _mm256_srai_epi32, _mm256_srl_epi32, _mm256_sub_ps, _mm256_subs_epi16, _mm256_unpackhi_ps,
        _mm256_unpacklo_ps, _mm256_xor_ps, _mm256_xor_si256, _mm512_add_epi32, _mm512_add_ps,
        _mm512_and_si512, _mm512_broadcast_i32x4, _mm512_castpd_ps, _mm512_castps_pd,
        _mm512_castps_si512, _mm512_castsi512_ps, _mm512_cmpeq_epi32_mask, _mm512_cmpeq_epi64_mask,
        _mm512_cvtepu8_epi32, _mm512_cvtepu8_epi64, _mm512_loadu_si512, _mm512_madd_epi16,
        _mm512_mask_blend_epi32, _mm512_mask_blend_epi64, _mm512_mask_blend_pd,
        _mm512_mask_ror_epi32, _mm512_mask_sub_epi32, _mm512_mask_sub_epi64, _mm512_mul_ps,
        _mm512_packs_epi32, _mm512_permute_ps, _mm512_permutex2var_epi32,
        _mm512_permutex2var_epi64, _mm512_permutexvar_epi32, _mm512_permutexvar_epi64,
        _mm512_set1_epi32, _mm512_set1_epi64, _mm512_setzero_si512, _mm512_shuffle_epi8,
        _mm512_slli_epi32, _mm512_slli_epi64, _mm512_srai_epi32, _mm512_srl_epi32,
        _mm512_srl_epi64, _mm512_subs_epi16, _mm512_test_epi32_mask, _mm512_test_epi64_mask,
        _mm512_xor_si512,
    };

    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4, f32x8, f32x16, i16x8, i16x16, i16x32, u8x16, u32x8, u32x16};

    use super::{CoefC32, CoefQ15, Plan, Q_NEG, Q_SIGN_IM, Q_SIGN_RE, Q_SWAP, RESIDUES};

    /// `2^14`, the rounding constant of the `Q15` product.
    const ROUND: i32 = 1 << 14;

    /// `[t[0], t[1], t[2], t[3]]` twice, for `vpermd` with an index whose low two bits are the
    /// quadrant (bit 2 may be set).
    #[inline(always)]
    fn table8(t: [u32; 4]) -> [u32; 8] {
        [t[0], t[1], t[2], t[3], t[0], t[1], t[2], t[3]]
    }

    /// The residues of the phase bytes `p`, for the per-sample loads.
    #[inline(always)]
    fn residues<const N: usize>(p: &[u8; N], plan: &Plan) -> [usize; N] {
        p.map(|v| plan.residue(v))
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
        rmask: __m128i,
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
            rmask: _mm_set1_epi32(plan.rmask as i32),
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

    /// `(q, r)` of `k` (`q` in 0..3, the residue bits of `k` in `r`).
    #[target_feature(enable = "sse2")]
    #[inline]
    fn qr128(k: __m128i, c: Idx128) -> (__m128i, __m128i) {
        let q = _mm_and_si128(_mm_srl_epi32(_mm_slli_epi32::<1>(k), c.qshift), c.three);
        (q, _mm_and_si128(k, c.rmask))
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

    /// The `cq15_mul` of four samples `x` with the phasors `a = (c, −s)`, `b = (s, c)`, packed
    /// as `[re0..re3 im0..im3]`: `pmaddwd`, `+ 2^14`, `>> 15`, saturating pack.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn product_cq15_sse(x: __m128i, a: __m128i, b: __m128i) -> __m128i {
        let round = _mm_set1_epi32(ROUND);
        let re = _mm_srai_epi32::<15>(_mm_add_epi32(_mm_madd_epi16(x, a), round));
        let im = _mm_srai_epi32::<15>(_mm_add_epi32(_mm_madd_epi16(x, b), round));
        _mm_packs_epi32(re, im)
    }

    /// The first and the second halves of the table words `t[r0], t[r1], t[r2], t[r3]` (two
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
            /// Four `CQ15` samples.
            #[target_feature(enable = $feature)]
            #[inline]
            fn step_cq15<const RES: bool>(
                x: __m128i,
                p: &[u8; 4],
                c: Idx128,
                plan: &Plan,
                coef: &CoefQ15,
            ) -> __m128i {
                let (q, r) = qr128($k(*p, c), c);
                let y = if RES {
                    let rs = residues(p, plan);
                    let (a, b) = pairs4(rs.map(|r| coef.ab(r)));
                    let v = $zip(product_cq15_sse(
                        x,
                        _mm_castps_si128(a),
                        _mm_castps_si128(b),
                    ));
                    $select(_mm_cmpeq_epi32(r, _mm_setzero_si128()), x, v)
                } else {
                    x
                };
                let (swap, neg) = $masks(q);
                let y = $select(swap, $swap(y), y);
                _mm_subs_epi16(_mm_xor_si128(y, neg), neg)
            }

            /// Four `C32` samples (two registers) in the planar form: `re` and `im` by
            /// `shufps`, the masks by shifts, the product in the order of the fallback. The
            /// quadrant bits stay unmasked in `qr` (bits 0 and 1 are the quadrant): the swap
            /// mask is bit 0 moved to the sign bit, the sign of `im` bit 1, the sign of `re`
            /// their XOR.
            #[target_feature(enable = $feature)]
            #[inline]
            fn step_c32<const RES: bool>(
                x: [__m128; 2],
                p: &[u8; 4],
                c: Idx128,
                plan: &Plan,
                coef: &CoefC32,
            ) -> [__m128; 2] {
                let k = $k(*p, c);
                let qr = _mm_srl_epi32(_mm_slli_epi32::<1>(k), c.qshift);
                let re = _mm_shuffle_ps::<0x88>(x[0], x[1]);
                let im = _mm_shuffle_ps::<0xDD>(x[0], x[1]);
                let (re, im) = if RES {
                    let r = _mm_and_si128(k, c.rmask);
                    let (cv, sv) = pairs4(residues(p, plan).map(|r| coef.cs(r)));
                    let pre = _mm_sub_ps(_mm_mul_ps(re, cv), _mm_mul_ps(im, sv));
                    let pim = _mm_add_ps(_mm_mul_ps(re, sv), _mm_mul_ps(im, cv));
                    let z = _mm_cmpeq_epi32(r, _mm_setzero_si128());
                    ($select_ps(z, re, pre), $select_ps(z, im, pim))
                } else {
                    (re, im)
                };
                let swap = _mm_slli_epi32::<31>(qr);
                let sim = _mm_and_si128(_mm_slli_epi32::<30>(qr), c.sign);
                let sre = _mm_castsi128_ps(_mm_xor_si128(swap, sim));
                let sim = _mm_castsi128_ps(sim);
                let swap = $sign_mask(swap);
                let ore = _mm_xor_ps($select_ps(swap, im, re), sre);
                let oim = _mm_xor_ps($select_ps(swap, re, im), sim);
                [_mm_unpacklo_ps(ore, oim), _mm_unpackhi_ps(ore, oim)]
            }

            fearless_simd::kernel!(
                pub(crate) fn $cq15(
                    t: $token,
                    x: &[i16],
                    p: &[u8],
                    y: &mut [i16],
                    plan: &Plan,
                    coef: &CoefQ15,
                ) -> usize {
                    let c = idx128(plan);
                    let (xs, _) = x.as_chunks::<8>();
                    let (ps, _) = p.as_chunks::<4>();
                    let (ys, _) = y.as_chunks_mut::<8>();
                    let mut n = 0;
                    if plan.res() {
                        for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
                            let v = step_cq15::<true>(
                                i16x8::from_slice(t, xv).into(),
                                pv,
                                c,
                                plan,
                                coef,
                            );
                            i16x8::simd_from(t, v).store_array(yv);
                            n += 4;
                        }
                    } else {
                        for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
                            let v = step_cq15::<false>(
                                i16x8::from_slice(t, xv).into(),
                                pv,
                                c,
                                plan,
                                coef,
                            );
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
                    coef: &CoefC32,
                ) -> usize {
                    let c = idx128(plan);
                    // Eight samples per iteration: two steps of four.
                    let (xs, _) = x.as_chunks::<16>();
                    let (ps, _) = p.as_chunks::<8>();
                    let (ys, _) = y.as_chunks_mut::<16>();
                    let mut n = 0;
                    let mut body = |xv: &[f32; 16], pv: &[u8; 8], yv: &mut [f32; 16], res: bool| {
                        let (xa, xb) = xv.split_at(8);
                        let (pa, pb) = pv.split_at(4);
                        let (ya, yb) = yv.split_at_mut(8);
                        for ((xh, ph), yh) in
                            [(xa, pa, ya), (xb, pb, yb)].map(|(xh, ph, yh)| ((xh, ph), yh))
                        {
                            let (lo, hi) = xh.split_at(4);
                            let v = [
                                f32x4::from_slice(t, lo).into(),
                                f32x4::from_slice(t, hi).into(),
                            ];
                            let ph: &[u8; 4] = ph.try_into().unwrap_or(&[0; 4]);
                            let [a, b] = if res {
                                step_c32::<true>(v, ph, c, plan, coef)
                            } else {
                                step_c32::<false>(v, ph, c, plan, coef)
                            };
                            let (lo, hi) = yh.split_at_mut(4);
                            f32x4::simd_from(t, a).store_slice(lo);
                            f32x4::simd_from(t, b).store_slice(hi);
                        }
                        n += 8;
                    };
                    if plan.res() {
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
        rmask: __m256i,
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn idx256(plan: &Plan) -> Idx256 {
        Idx256 {
            sgn: _mm256_set1_epi32(if plan.neg == 0 { 1 } else { -1 }),
            qshift: _mm_cvtsi32_si128(plan.qshift as i32),
            rmask: _mm256_set1_epi32(plan.rmask as i32),
        }
    }

    /// `k = ±p` of eight phase bytes, one 32-bit lane each (`vpmovzxbd`, `vpsignd`), and the
    /// quadrant index `(k << 1) >> (b − 1)` (the quadrant in the low two bits; the bits above
    /// are not cleared: the tables are repeated and the masks use bits 0 and 1 only).
    #[target_feature(enable = "avx2")]
    #[inline]
    fn k8(p: [u8; 8], c: Idx256) -> __m256i {
        let v = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(i64::from_le_bytes(p)));
        _mm256_sign_epi32(v, c.sgn)
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn q8(k: __m256i, c: Idx256) -> __m256i {
        _mm256_srl_epi32(_mm256_slli_epi32::<1>(k), c.qshift)
    }

    /// Up to 16 dword entries in two registers: one `vpermd`, or two and a blend on bit 3.
    #[derive(Clone, Copy)]
    struct Lookup256 {
        lo: __m256i,
        hi: __m256i,
        residues: usize,
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn lookup256(t: &[u32; RESIDUES], residues: usize) -> Lookup256 {
        let (ch, _) = t.as_chunks::<8>();
        let ld = |i: usize| -> __m256i {
            let a = ch.get(i).copied().unwrap_or([0; 8]).map(|v| v as i32);
            // SAFETY: `a` holds eight `i32`, 32 bytes.
            unsafe { core::ptr::read_unaligned(a.as_ptr().cast::<__m256i>()) }
        };
        Lookup256 {
            lo: ld(0),
            hi: ld(1),
            residues,
        }
    }

    /// The table entries of the residues `r` (each below `l.residues` ≤ 16).
    #[target_feature(enable = "avx2")]
    #[inline]
    fn get256(l: &Lookup256, r: __m256i) -> __m256i {
        if l.residues <= 8 {
            _mm256_permutevar8x32_epi32(l.lo, r)
        } else {
            let m = _mm256_castsi256_ps(_mm256_slli_epi32::<28>(r));
            _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(_mm256_permutevar8x32_epi32(l.lo, r)),
                _mm256_castsi256_ps(_mm256_permutevar8x32_epi32(l.hi, r)),
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

    fearless_simd::kernel!(
        pub(super) fn cq15_avx2(
            t: Avx2,
            x: &[i16],
            p: &[u8],
            y: &mut [i16],
            plan: &Plan,
            coef: &CoefQ15,
        ) -> usize {
            let c = idx256(plan);
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
            let zip_b = _mm256_broadcastsi128_si256(_mm_setr_epi8(
                0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
            ));
            let round = _mm256_set1_epi32(ROUND);
            let la = lookup256(&coef.a, plan.residues());
            let lb = lookup256(&coef.b, plan.residues());
            let (xs, _) = x.as_chunks::<16>();
            let (ps, _) = p.as_chunks::<8>();
            let (ys, _) = y.as_chunks_mut::<16>();
            let mut n = 0;
            // `RES`: 0 no product, 1 permuted tables, 2 per-sample loads.
            let mut body = |xv: &[i16; 16], pv: &[u8; 8], yv: &mut [i16; 16], res: u8| {
                let xr: __m256i = i16x16::from_slice(t, xv).into();
                let k = k8(*pv, c);
                let q = q8(k, c);
                let v = if res != 0 {
                    let r = _mm256_and_si256(k, c.rmask);
                    let (a, b) = if res == 1 {
                        (get256(&la, r), get256(&lb, r))
                    } else {
                        // Lanes `[0 1 2 3 | 4 5 6 7]`: the words in the order 0 1 4 5, 2 3 6 7.
                        let rs = residues(pv, plan).map(|r| coef.ab(r));
                        let (a, b) =
                            pairs8([rs[0], rs[1], rs[4], rs[5], rs[2], rs[3], rs[6], rs[7]]);
                        (_mm256_castps_si256(a), _mm256_castps_si256(b))
                    };
                    let re =
                        _mm256_srai_epi32::<15>(_mm256_add_epi32(_mm256_madd_epi16(xr, a), round));
                    let im =
                        _mm256_srai_epi32::<15>(_mm256_add_epi32(_mm256_madd_epi16(xr, b), round));
                    let v = _mm256_shuffle_epi8(_mm256_packs_epi32(re, im), zip_b);
                    _mm256_blendv_epi8(v, xr, _mm256_cmpeq_epi32(r, _mm256_setzero_si256()))
                } else {
                    xr
                };
                let swap = _mm256_permutevar8x32_epi32(swap_t, q);
                let neg = _mm256_permutevar8x32_epi32(neg_t, q);
                let v = _mm256_blendv_epi8(v, _mm256_shuffle_epi8(v, swap_b), swap);
                let v = _mm256_subs_epi16(_mm256_xor_si256(v, neg), neg);
                i16x16::simd_from(t, v).store_array(yv);
                n += 8;
            };
            let res = if !plan.res() {
                0
            } else if plan.residues() <= 16 {
                1
            } else {
                2
            };
            // One loop per form, so that no build keeps the choice inside the loop. The form
            // without a product (13 instructions per eight samples) is unrolled by two by hand:
            // the generic x86-64 tuning of the run-time-detection build does not unroll it, and
            // its loop overhead was 1.07 times the instructions of the `avx2` build.
            if res == 0 {
                let (x2, _) = xs.as_chunks::<2>();
                let (p2, _) = ps.as_chunks::<2>();
                let (y2, _) = ys.as_chunks_mut::<2>();
                for ((xv, pv), yv) in x2.iter().zip(p2).zip(y2) {
                    let [ya, yb] = yv;
                    body(&xv[0], &pv[0], ya, 0);
                    body(&xv[1], &pv[1], yb, 0);
                }
                // An odd last step of eight.
                let done = 2 * x2.len().min(p2.len());
                if let (Some(xv), Some(pv), Some(yv)) =
                    (xs.get(done), ps.get(done), ys.get_mut(done))
                {
                    body(xv, pv, yv, 0);
                }
                return n;
            }
            let it = xs.iter().zip(ps).zip(ys);
            match res {
                1 => it.for_each(|((xv, pv), yv)| body(xv, pv, yv, 1)),
                _ => it.for_each(|((xv, pv), yv)| body(xv, pv, yv, 2)),
            }
            n
        }
    );

    fearless_simd::kernel!(
        pub(super) fn c32_avx2(
            t: Avx2,
            x: &[f32],
            p: &[u8],
            y: &mut [f32],
            plan: &Plan,
            coef: &CoefC32,
        ) -> usize {
            let c = idx256(plan);
            let lc = lookup256(bytemuck::cast_ref(&coef.c), plan.residues());
            let ls = lookup256(bytemuck::cast_ref(&coef.s), plan.residues());
            let sign = _mm256_set1_epi32(i32::MIN);
            let (xs, _) = x.as_chunks::<16>();
            let (ps, _) = p.as_chunks::<8>();
            let (ys, _) = y.as_chunks_mut::<16>();
            let mut n = 0;
            // The planar form: `re` and `im` of the eight samples by `shufps` in the lane order
            // `[0 1 4 5 | 2 3 6 7]`, the indices permuted to the same order.
            let mut body = |xv: &[f32; 16], pv: &[u8; 8], yv: &mut [f32; 16], res: u8| {
                let (lo, hi) = xv.split_at(8);
                let x0: __m256 = f32x8::from_slice(t, lo).into();
                let x1: __m256 = f32x8::from_slice(t, hi).into();
                let re = _mm256_shuffle_ps::<0x88>(x0, x1);
                let im = _mm256_shuffle_ps::<0xDD>(x0, x1);
                let k = _mm256_permute4x64_epi64::<0xD8>(k8(*pv, c));
                let q = q8(k, c);
                let (re, im) = if res != 0 {
                    let r = _mm256_and_si256(k, c.rmask);
                    let (cv, sv) = if res == 1 {
                        (
                            _mm256_castsi256_ps(get256(&lc, r)),
                            _mm256_castsi256_ps(get256(&ls, r)),
                        )
                    } else {
                        pairs8(residues(pv, plan).map(|r| coef.cs(r)))
                    };
                    let pre = _mm256_sub_ps(_mm256_mul_ps(re, cv), _mm256_mul_ps(im, sv));
                    let pim = _mm256_add_ps(_mm256_mul_ps(re, sv), _mm256_mul_ps(im, cv));
                    let z = _mm256_castsi256_ps(_mm256_cmpeq_epi32(r, _mm256_setzero_si256()));
                    (_mm256_blendv_ps(pre, re, z), _mm256_blendv_ps(pim, im, z))
                } else {
                    (re, im)
                };
                // `vblendvps` reads the sign bit: bit 0 of `q` shifted there; the sign of `im` is
                // bit 1, that of `re` their XOR.
                let swap = _mm256_slli_epi32::<31>(q);
                let sim = _mm256_and_si256(_mm256_slli_epi32::<30>(q), sign);
                let sre = _mm256_castsi256_ps(_mm256_xor_si256(swap, sim));
                let sim = _mm256_castsi256_ps(sim);
                let swap = _mm256_castsi256_ps(swap);
                let ore = _mm256_xor_ps(_mm256_blendv_ps(re, im, swap), sre);
                let oim = _mm256_xor_ps(_mm256_blendv_ps(im, re, swap), sim);
                let (lo, hi) = yv.split_at_mut(8);
                f32x8::simd_from(t, _mm256_unpacklo_ps(ore, oim)).store_slice(lo);
                f32x8::simd_from(t, _mm256_unpackhi_ps(ore, oim)).store_slice(hi);
                n += 8;
            };
            let res = if !plan.res() {
                0
            } else if plan.residues() <= 16 {
                1
            } else {
                2
            };
            // One loop per form, so that no build keeps the choice inside the loop.
            let it = xs.iter().zip(ps).zip(ys);
            match res {
                0 => it.for_each(|((xv, pv), yv)| body(xv, pv, yv, 0)),
                1 => it.for_each(|((xv, pv), yv)| body(xv, pv, yv, 1)),
                _ => it.for_each(|((xv, pv), yv)| body(xv, pv, yv, 2)),
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
        residues: usize,
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn lookup512d(t: &[u32; RESIDUES], residues: usize) -> Lookup512d {
        let (ch, _) = t.as_chunks::<16>();
        let mut regs = [_mm512_setzero_si512(); 4];
        for (d, s) in regs.iter_mut().zip(ch) {
            // SAFETY: `s` is sixteen `u32`, 64 bytes.
            *d = unsafe { _mm512_loadu_si512(s.as_ptr().cast()) };
        }
        Lookup512d { t: regs, residues }
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn get512d(l: &Lookup512d, r: __m512i) -> __m512i {
        let [t0, t1, t2, t3] = l.t;
        if l.residues <= 16 {
            _mm512_permutexvar_epi32(r, t0)
        } else if l.residues <= 32 {
            _mm512_permutex2var_epi32(t0, r, t1)
        } else {
            let lo = _mm512_permutex2var_epi32(t0, r, t1);
            let hi = _mm512_permutex2var_epi32(t2, r, t3);
            _mm512_mask_blend_epi32(_mm512_test_epi32_mask(r, _mm512_set1_epi32(32)), lo, hi)
        }
    }

    /// Up to 64 qword entries in eight registers, looked up by `vpermq`, `vpermi2q` and blends
    /// on bits 4 and 5.
    #[derive(Clone, Copy)]
    struct Lookup512q {
        t: [__m512i; 8],
        residues: usize,
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn lookup512q(t: &[u64; RESIDUES], residues: usize, map: fn(u64) -> u64) -> Lookup512q {
        let (ch, _) = t.as_chunks::<8>();
        let mut regs = [_mm512_setzero_si512(); 8];
        for (d, s) in regs.iter_mut().zip(ch) {
            let s = s.map(map);
            // SAFETY: `s` is eight `u64`, 64 bytes.
            *d = unsafe { _mm512_loadu_si512(s.as_ptr().cast()) };
        }
        Lookup512q { t: regs, residues }
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn get512q(l: &Lookup512q, r: __m512i) -> __m512i {
        let [t0, t1, t2, t3, t4, t5, t6, t7] = l.t;
        if l.residues <= 8 {
            return _mm512_permutexvar_epi64(r, t0);
        }
        let a = _mm512_permutex2var_epi64(t0, r, t1);
        if l.residues <= 16 {
            return a;
        }
        let bit4 = _mm512_test_epi64_mask(r, _mm512_set1_epi64(16));
        let b = _mm512_permutex2var_epi64(t2, r, t3);
        let ab = _mm512_mask_blend_epi64(bit4, a, b);
        if l.residues <= 32 {
            return ab;
        }
        let c = _mm512_permutex2var_epi64(t4, r, t5);
        let d = _mm512_permutex2var_epi64(t6, r, t7);
        let cd = _mm512_mask_blend_epi64(bit4, c, d);
        _mm512_mask_blend_epi64(_mm512_test_epi64_mask(r, _mm512_set1_epi64(32)), ab, cd)
    }

    /// The constants of an AVX-512 `CQ15` call.
    struct Cq15x16 {
        inv: u16,
        rmask: __m512i,
        qshift: __m128i,
        neg_t: __m512i,
        zip_b: __m512i,
        la: Lookup512d,
        lb: Lookup512d,
    }

    /// Sixteen `CQ15` samples `x` with the phase bytes `p`.
    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    fn cq15_x16<const RES: bool>(x: __m512i, p: __m128i, c: &Cq15x16) -> __m512i {
        let v = _mm512_cvtepu8_epi32(p);
        // `k = ±p`; the quadrant in the low two bits of `q` (the table repeats every four
        // entries and the swap mask tests bit 0).
        let k = _mm512_mask_sub_epi32(v, c.inv, _mm512_setzero_si512(), v);
        let q = _mm512_srl_epi32(_mm512_slli_epi32::<1>(k), c.qshift);
        let v = if RES {
            let round = _mm512_set1_epi32(ROUND);
            let r = _mm512_and_si512(k, c.rmask);
            let (a, b) = (get512d(&c.la, r), get512d(&c.lb, r));
            let re = _mm512_srai_epi32::<15>(_mm512_add_epi32(_mm512_madd_epi16(x, a), round));
            let im = _mm512_srai_epi32::<15>(_mm512_add_epi32(_mm512_madd_epi16(x, b), round));
            let v = _mm512_shuffle_epi8(_mm512_packs_epi32(re, im), c.zip_b);
            let zero = _mm512_cmpeq_epi32_mask(r, _mm512_setzero_si512());
            _mm512_mask_blend_epi32(zero, v, x)
        } else {
            x
        };
        let swap = _mm512_test_epi32_mask(q, _mm512_set1_epi32(1));
        let v = _mm512_mask_ror_epi32::<16>(v, swap, v);
        let neg = _mm512_permutexvar_epi32(q, c.neg_t);
        _mm512_subs_epi16(_mm512_xor_si512(v, neg), neg)
    }

    fearless_simd::kernel!(
        pub(super) fn cq15_avx512(
            t: Avx512,
            x: &[i16],
            p: &[u8],
            y: &mut [i16],
            plan: &Plan,
            coef: &CoefQ15,
        ) -> usize {
            let c = Cq15x16 {
                inv: if plan.neg == 0 { 0 } else { u16::MAX },
                rmask: _mm512_set1_epi32(plan.rmask as i32),
                qshift: _mm_cvtsi32_si128(plan.qshift as i32),
                neg_t: u32x16::simd_from(t, table16(Q_NEG)).into(),
                zip_b: _mm512_broadcast_i32x4(_mm_setr_epi8(
                    0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
                )),
                la: lookup512d(&coef.a, plan.residues()),
                lb: lookup512d(&coef.b, plan.residues()),
            };
            let (xs, _) = x.as_chunks::<32>();
            let (ps, _) = p.as_chunks::<16>();
            let (ys, _) = y.as_chunks_mut::<32>();
            let mut n = 0;
            let res = plan.res();
            for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
                let xr: __m512i = i16x32::from_slice(t, xv).into();
                let pb: __m128i = u8x16::from_slice(t, pv).into();
                let v = if res {
                    cq15_x16::<true>(xr, pb, &c)
                } else {
                    cq15_x16::<false>(xr, pb, &c)
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

    /// The constants of an AVX-512 `C32` call.
    struct C32x8 {
        inv: u8,
        rmask: __m512i,
        qshift: __m128i,
        sign_t: __m512i,
        lc: Lookup512q,
        ls: Lookup512q,
    }

    /// Eight `C32` samples `x` (one per 64-bit lane) with the phase bytes `p`.
    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    fn c32_x8<const RES: bool>(x: __m512, p: [u8; 8], c: &C32x8) -> __m512 {
        let v = _mm512_cvtepu8_epi64(_mm_cvtsi64_si128(i64::from_le_bytes(p)));
        let k = _mm512_mask_sub_epi64(v, c.inv, _mm512_setzero_si512(), v);
        let q = _mm512_srl_epi64(_mm512_slli_epi64::<1>(k), c.qshift);
        let v = if RES {
            let r = _mm512_and_si512(k, c.rmask);
            let cv = _mm512_castsi512_ps(get512q(&c.lc, r));
            let sv = _mm512_castsi512_ps(get512q(&c.ls, r));
            let pr = _mm512_add_ps(
                _mm512_mul_ps(x, cv),
                _mm512_mul_ps(_mm512_permute_ps::<0xB1>(x), sv),
            );
            let zero = _mm512_cmpeq_epi64_mask(r, _mm512_setzero_si512());
            _mm512_castpd_ps(_mm512_mask_blend_pd(
                zero,
                _mm512_castps_pd(pr),
                _mm512_castps_pd(x),
            ))
        } else {
            x
        };
        let swap = _mm512_test_epi64_mask(q, _mm512_set1_epi64(1));
        let v = _mm512_castpd_ps(_mm512_mask_blend_pd(
            swap,
            _mm512_castps_pd(v),
            _mm512_castps_pd(_mm512_permute_ps::<0xB1>(v)),
        ));
        let sign = _mm512_permutexvar_epi64(q, c.sign_t);
        _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(v), sign))
    }

    /// The constants of [`c32_x8`]: the quadrant sign table and the phasor tables `[c c]` and
    /// `[−s s]` of the residues.
    #[target_feature(enable = "avx512f")]
    #[inline]
    fn c32x8(plan: &Plan, coef: &CoefC32) -> C32x8 {
        let sign_t = table8q(Q_SIGN_RE, Q_SIGN_IM);
        C32x8 {
            inv: if plan.neg == 0 { 0 } else { u8::MAX },
            rmask: _mm512_set1_epi64(i64::from(plan.rmask)),
            qshift: _mm_cvtsi32_si128(plan.qshift as i32),
            // SAFETY: `sign_t` is eight `u64`, 64 bytes.
            sign_t: unsafe { _mm512_loadu_si512(sign_t.as_ptr().cast()) },
            // `[c c]` and `[−s s]` of every residue, from the halves of `cs`.
            lc: lookup512q(&coef.cs, plan.residues(), |w| (w & 0xFFFF_FFFF) | (w << 32)),
            ls: lookup512q(&coef.cs, plan.residues(), |w| {
                ((w >> 32) ^ 0x8000_0000) | (w & !0xFFFF_FFFF)
            }),
        }
    }

    fearless_simd::kernel!(
        pub(super) fn c32_avx512(
            t: Avx512,
            x: &[f32],
            p: &[u8],
            y: &mut [f32],
            plan: &Plan,
            coef: &CoefC32,
        ) -> usize {
            let c = c32x8(plan, coef);
            let (xs, _) = x.as_chunks::<16>();
            let (ps, _) = p.as_chunks::<8>();
            let (ys, _) = y.as_chunks_mut::<16>();
            let mut n = 0;
            let res = plan.res();
            for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
                let xr: __m512 = f32x16::from_slice(t, xv).into();
                let v = if res {
                    c32_x8::<true>(xr, *pv, &c)
                } else {
                    c32_x8::<false>(xr, *pv, &c)
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
        vceq_u8, vceqq_u16, vcombine_s16, vcombine_u8, vcombine_u64, vcreate_u8, vcreate_u64,
        vdup_n_s8, vdup_n_u8, vdupq_n_u8, vdupq_n_u16, vdupq_n_u32, veor_u8, veorq_s16, veorq_u32,
        vget_high_s16, vget_low_s16, vld1_u8, vld1q_u8, vld2q_f32, vld2q_s16, vmlal_s16, vmlsl_s16,
        vmovl_s8, vmovl_s16, vmovl_u8, vmull_s16, vmulq_f32, vorrq_u8, vqrshrn_n_s32, vqsubq_s16,
        vqtbl1q_u8, vqtbl4q_u8, vqtbx4q_u8, vreinterpret_s8_u8, vreinterpretq_f32_u8,
        vreinterpretq_f32_u32, vreinterpretq_f32_u64, vreinterpretq_s16_u8, vreinterpretq_u8_u16,
        vreinterpretq_u16_s16, vreinterpretq_u32_f32, vreinterpretq_u32_s32, vshl_n_u8, vshl_u8,
        vst2q_f32, vst2q_s16, vsub_u8, vsubq_f32, vsubq_u8, vtbl1_u8, vuzp1q_f32, vuzp2q_f32,
        vzip1q_u8,
    };

    use super::{CoefC32, CoefQ15, Plan};

    /// The index constants of a call (bytes).
    #[derive(Clone, Copy)]
    struct Idx {
        neg: uint8x8_t,
        /// `+1` for `b = 1`, `−(b − 2)` otherwise: `q = (k << shift) & 3` (`ushl`).
        qshift: int8x8_t,
        rmask: uint8x8_t,
        three: uint8x8_t,
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn idx(plan: &Plan) -> Idx {
        let bits = plan.qshift + 1;
        let shift = if bits == 1 { 1 } else { -((bits - 2) as i8) };
        Idx {
            neg: vdup_n_u8(plan.neg as u8),
            qshift: vdup_n_s8(shift),
            rmask: vdup_n_u8(plan.rmask as u8),
            three: vdup_n_u8(3),
        }
    }

    /// `(q, r)` of eight phase bytes.
    #[target_feature(enable = "neon")]
    #[inline]
    fn qr(p: &[u8; 8], c: Idx) -> (uint8x8_t, uint8x8_t) {
        // SAFETY: `p` is eight bytes.
        let v = unsafe { vld1_u8(p.as_ptr()) };
        // `k = ±p`; the bits of `p` above `b` never reach `q` or `r`.
        let k = vsub_u8(veor_u8(v, c.neg), c.neg);
        (vand_u8(vshl_u8(k, c.qshift), c.three), vand_u8(k, c.rmask))
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

    /// The table of `entries` values of `width` bytes each (2 or 4), `value(r)` little-endian
    /// at byte `width · r`.
    #[target_feature(enable = "neon")]
    #[inline]
    fn table(entries: usize, width: usize, value: impl Fn(usize) -> u32) -> Table {
        let mut t = [uint8x16x4_t(vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0)); 4];
        for (part, d) in t.iter_mut().enumerate() {
            let reg = |i: usize| -> uint8x16_t {
                let a: [u8; 16] = core::array::from_fn(|j| {
                    let byte = 64 * part + 16 * i + j;
                    let r = byte / width;
                    if r < entries {
                        (value(r) >> (8 * (byte % width))) as u8
                    } else {
                        0
                    }
                });
                // SAFETY: `a` is sixteen bytes.
                unsafe { vld1q_u8(a.as_ptr()) }
            };
            *d = uint8x16x4_t(reg(0), reg(1), reg(2), reg(3));
        }
        Table {
            t,
            parts: (entries * width).div_ceil(64).max(1),
        }
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

    /// The constants of a `CQ15` call: the `i16` cosines and sines of the residues.
    struct Cq15 {
        c: Idx,
        tc: Table,
        ts: Table,
    }

    /// Eight `CQ15` samples (planar `re`, `im`).
    #[target_feature(enable = "neon")]
    #[inline]
    fn cq15_x8<const RES: bool>(re: int16x8_t, im: int16x8_t, p: [u8; 8], k: &Cq15) -> int16x8x2_t {
        let (q, r) = qr(&p, k.c);
        let (re, im) = if RES {
            // Byte indices `[2r, 2r + 1]` per sample.
            let r2 = vshl_n_u8::<1>(r);
            let r2 = vcombine_u8(r2, r2);
            let i = vorrq_u8(vzip1q_u8(r2, r2), vreinterpretq_u8_u16(vdupq_n_u16(0x0100)));
            let cv = vreinterpretq_s16_u8(get(&k.tc, i));
            let sv = vreinterpretq_s16_u8(get(&k.ts, i));
            let (rl, rh) = (vget_low_s16(re), vget_high_s16(re));
            let (il, ih) = (vget_low_s16(im), vget_high_s16(im));
            let (cl, ch) = (vget_low_s16(cv), vget_high_s16(cv));
            let (sl, sh) = (vget_low_s16(sv), vget_high_s16(sv));
            let pre = vcombine_s16(
                vqrshrn_n_s32::<15>(vmlsl_s16(vmull_s16(rl, cl), il, sl)),
                vqrshrn_n_s32::<15>(vmlsl_s16(vmull_s16(rh, ch), ih, sh)),
            );
            let pim = vcombine_s16(
                vqrshrn_n_s32::<15>(vmlal_s16(vmull_s16(rl, sl), il, cl)),
                vqrshrn_n_s32::<15>(vmlal_s16(vmull_s16(rh, sh), ih, ch)),
            );
            let z = vceqq_u16(vmovl_u8(r), vdupq_n_u16(0));
            (vbslq_s16(z, re, pre), vbslq_s16(z, im, pim))
        } else {
            (re, im)
        };
        let (sw, nre, nim) = quadrant_bytes(q);
        let sw: uint16x8_t = vreinterpretq_u16_s16(vmovl_s8(vreinterpret_s8_u8(sw)));
        let nre: int16x8_t = vmovl_s8(vreinterpret_s8_u8(nre));
        let nim: int16x8_t = vmovl_s8(vreinterpret_s8_u8(nim));
        let ore = vbslq_s16(sw, im, re);
        let oim = vbslq_s16(sw, re, im);
        int16x8x2_t(
            vqsubq_s16(veorq_s16(ore, nre), nre),
            vqsubq_s16(veorq_s16(oim, nim), nim),
        )
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn cq15_consts(plan: &Plan, coef: &CoefQ15) -> Cq15 {
        let n = plan.residues();
        Cq15 {
            c: idx(plan),
            // The low halves of `a = (c, −s)` and `b = (s, c)`.
            tc: table(n, 2, |r| coef.a(r) & 0xFFFF),
            ts: table(n, 2, |r| coef.b(r) & 0xFFFF),
        }
    }

    fearless_simd::kernel!(
        pub(super) fn cq15(
            t: Neon,
            x: &[i16],
            p: &[u8],
            y: &mut [i16],
            plan: &Plan,
            coef: &CoefQ15,
        ) -> usize {
            let _ = t;
            let k = cq15_consts(plan, coef);
            let res = plan.res();
            let (xs, _) = x.as_chunks::<16>();
            let (ps, _) = p.as_chunks::<8>();
            let (ys, _) = y.as_chunks_mut::<16>();
            let mut n = 0;
            for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
                // SAFETY: `xv` is sixteen `i16`.
                let v = unsafe { vld2q_s16(xv.as_ptr()) };
                let out = if res {
                    cq15_x8::<true>(v.0, v.1, *pv, &k)
                } else {
                    cq15_x8::<false>(v.0, v.1, *pv, &k)
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

    /// The constants of a `C32` call: the `f32` cosines and sines of the residues and the byte
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
    fn c32_consts(plan: &Plan, coef: &CoefC32) -> C32k {
        let n = plan.residues();
        let ld = |f: &dyn Fn(usize) -> u8| -> uint8x16_t {
            let a: [u8; 16] = core::array::from_fn(f);
            // SAFETY: `a` is sixteen bytes.
            unsafe { vld1q_u8(a.as_ptr()) }
        };
        C32k {
            c: idx(plan),
            tc: table(n, 4, |r| coef.c(r).to_bits()),
            ts: table(n, 4, |r| coef.s(r).to_bits()),
            rep: [ld(&|j| (j / 4) as u8), ld(&|j| 4 + (j / 4) as u8)],
            lane: ld(&|j| (j % 4) as u8),
        }
    }

    /// Eight `C32` samples (`xv`, sixteen `f32`) into `yv`.
    ///
    /// `MODE`: 0 no product (`b ≤ 2`), 1 the phasor by `tbl`/`tbx` from the register tables
    /// (up to 16 residues, one 64-byte part per table), 2 one 64-bit load of the pair `(c, s)`
    /// per sample and `uzp1`/`uzp2` (more residues: four `tbl`/`tbx` per table and 32 table
    /// registers were slower than the fallback, M49 §5).
    #[target_feature(enable = "neon")]
    #[inline]
    fn c32_x8<const MODE: u8>(
        xv: &[f32; 16],
        pv: &[u8; 8],
        yv: &mut [f32; 16],
        k: &C32k,
        plan: &Plan,
        coef: &CoefC32,
    ) {
        let (q, r) = qr(pv, k.c);
        let (sw, nre, nim) = quadrant_bytes(q);
        let r4 = vshl_n_u8::<2>(r);
        let r4 = vcombine_u8(r4, r4);
        let rz = vceq_u8(r, vdup_n_u8(0));
        let sign = vdupq_n_u32(0x8000_0000);
        let (xh, _) = xv.as_chunks::<8>();
        let (yh, _) = yv.as_chunks_mut::<8>();
        for (h, ((xq, yq), rep)) in xh.iter().zip(yh).zip(k.rep).enumerate() {
            // SAFETY: `xq` is eight `f32`.
            let v = unsafe { vld2q_f32(xq.as_ptr()) };
            let (re, im): (float32x4_t, float32x4_t) = (v.0, v.1);
            let (re, im) = if MODE != 0 {
                let (cv, sv) = if MODE == 1 {
                    let i = vorrq_u8(vqtbl1q_u8(r4, rep), k.lane);
                    (
                        vreinterpretq_f32_u8(get(&k.tc, i)),
                        vreinterpretq_f32_u8(get(&k.ts, i)),
                    )
                } else {
                    let w = |j: usize| {
                        let p = pv.get(4 * h + j).copied().unwrap_or(0);
                        vcreate_u64(coef.cs(plan.residue(p)))
                    };
                    let lo = vreinterpretq_f32_u64(vcombine_u64(w(0), w(1)));
                    let hi = vreinterpretq_f32_u64(vcombine_u64(w(2), w(3)));
                    (vuzp1q_f32(lo, hi), vuzp2q_f32(lo, hi))
                };
                let pre = vsubq_f32(vmulq_f32(re, cv), vmulq_f32(im, sv));
                let pim = vaddq_f32(vmulq_f32(re, sv), vmulq_f32(im, cv));
                let z = widen32(rz, h);
                (vbslq_f32(z, re, pre), vbslq_f32(z, im, pim))
            } else {
                (re, im)
            };
            let w = widen32(sw, h);
            let sre = vandq_u32(widen32(nre, h), sign);
            let sim = vandq_u32(widen32(nim, h), sign);
            let ore = vreinterpretq_u32_f32(vbslq_f32(w, im, re));
            let oim = vreinterpretq_u32_f32(vbslq_f32(w, re, im));
            let out = float32x4x2_t(
                vreinterpretq_f32_u32(veorq_u32(ore, sre)),
                vreinterpretq_f32_u32(veorq_u32(oim, sim)),
            );
            // SAFETY: `yq` is eight `f32`.
            unsafe { vst2q_f32(yq.as_mut_ptr(), out) };
        }
    }

    fearless_simd::kernel!(
        pub(super) fn c32(
            t: Neon,
            x: &[f32],
            p: &[u8],
            y: &mut [f32],
            plan: &Plan,
            coef: &CoefC32,
        ) -> usize {
            let _ = t;
            let k = c32_consts(plan, coef);
            let mode = if !plan.res() {
                0
            } else if plan.residues() <= 16 {
                1
            } else {
                2
            };
            let (xs, _) = x.as_chunks::<16>();
            let (ps, _) = p.as_chunks::<8>();
            let (ys, _) = y.as_chunks_mut::<16>();
            let it = xs.iter().zip(ps).zip(ys);
            match mode {
                0 => it.for_each(|((xv, pv), yv)| c32_x8::<0>(xv, pv, yv, &k, plan, coef)),
                1 => it.for_each(|((xv, pv), yv)| c32_x8::<1>(xv, pv, yv, &k, plan, coef)),
                _ => it.for_each(|((xv, pv), yv)| c32_x8::<2>(xv, pv, yv, &k, plan, coef)),
            }
            8 * xs.len().min(ps.len()).min(y.len() / 16)
        }
    );
}

/// The path of wasm32 with `simd128`: four samples per step, as SSE4.2 (`i8x16.swizzle` for the
/// tables, `i32x4.dot_i16x8_s` for the product).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::{
        f32x4, f32x4_add, f32x4_mul, f32x4_sub, i8x16, i8x16_swizzle, i16x8_narrow_i32x4,
        i16x8_shuffle, i16x8_sub_sat, i32x4, i32x4_add, i32x4_dot_i16x8, i32x4_eq, i32x4_shl,
        i32x4_shr, i32x4_shuffle, i32x4_splat, i32x4_sub, u16x8_extend_low_u8x16,
        u32x4_extend_low_u16x8, u32x4_shr, v128, v128_and, v128_bitselect, v128_or, v128_xor,
    };

    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4 as F32x4, i16x8 as I16x8};

    use super::{CoefC32, CoefQ15, Plan};

    /// `(q, r)` of four phase bytes in the 32-bit lanes.
    #[inline(always)]
    fn qr(p: &[u8; 4], plan: &Plan) -> (v128, v128) {
        let v = i32x4(i32::from_le_bytes(*p), 0, 0, 0);
        let v = u32x4_extend_low_u16x8(u16x8_extend_low_u8x16(v));
        let neg = i32x4_splat(plan.neg as i32);
        // `k = ±p`; the bits of `p` above `b` never reach `q` or `r`.
        let k = i32x4_sub(v128_xor(v, neg), neg);
        let q = v128_and(u32x4_shr(i32x4_shl(k, 1), plan.qshift), i32x4_splat(3));
        (q, v128_and(k, i32x4_splat(plan.rmask as i32)))
    }

    fearless_simd::kernel!(
        pub(super) fn cq15(
            t: WasmSimd128,
            x: &[i16],
            p: &[u8],
            y: &mut [i16],
            plan: &Plan,
            coef: &CoefQ15,
        ) -> usize {
            let rep = i8x16(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12);
            let half = i8x16(0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4);
            let swap_t = i8x16(0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            let neg_t = i8x16(0, -1, -1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0);
            let round = i32x4_splat(1 << 14);
            let zero = i32x4_splat(0);
            let res = plan.res();
            let (xs, _) = x.as_chunks::<8>();
            let (ps, _) = p.as_chunks::<4>();
            let (ys, _) = y.as_chunks_mut::<8>();
            let mut n = 0;
            for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
                let xr: v128 = I16x8::from_slice(t, xv).into();
                let (q, r) = qr(pv, plan);
                let v = if res {
                    let rs = pv.map(|b| plan.residue(b));
                    let at = |i: usize| coef.a(rs.get(i).copied().unwrap_or(0)) as i32;
                    let bt = |i: usize| coef.b(rs.get(i).copied().unwrap_or(0)) as i32;
                    let a = i32x4(at(0), at(1), at(2), at(3));
                    let b = i32x4(bt(0), bt(1), bt(2), bt(3));
                    let re = i32x4_shr(i32x4_add(i32x4_dot_i16x8(xr, a), round), 15);
                    let im = i32x4_shr(i32x4_add(i32x4_dot_i16x8(xr, b), round), 15);
                    let v = i16x8_narrow_i32x4(re, im);
                    let v = i16x8_shuffle::<0, 4, 1, 5, 2, 6, 3, 7>(v, v);
                    v128_bitselect(xr, v, i32x4_eq(r, zero))
                } else {
                    xr
                };
                let qrep = i8x16_swizzle(q, rep);
                let swap = i8x16_swizzle(swap_t, qrep);
                let neg = i8x16_swizzle(neg_t, v128_or(qrep, half));
                let v = v128_bitselect(i16x8_shuffle::<1, 0, 3, 2, 5, 4, 7, 6>(v, v), v, swap);
                let v = i16x8_sub_sat(v128_xor(v, neg), neg);
                I16x8::simd_from(t, v).store_array(yv);
                n += 4;
            }
            n
        }
    );

    fearless_simd::kernel!(
        pub(super) fn c32(
            t: WasmSimd128,
            x: &[f32],
            p: &[u8],
            y: &mut [f32],
            plan: &Plan,
            coef: &CoefC32,
        ) -> usize {
            let zero = i32x4_splat(0);
            let res = plan.res();
            let (xs, _) = x.as_chunks::<8>();
            let (ps, _) = p.as_chunks::<4>();
            let (ys, _) = y.as_chunks_mut::<8>();
            let mut n = 0;
            for ((xv, pv), yv) in xs.iter().zip(ps).zip(ys) {
                let (lo, hi) = xv.split_at(4);
                let x0: v128 = F32x4::from_slice(t, lo).into();
                let x1: v128 = F32x4::from_slice(t, hi).into();
                let (q, r) = qr(pv, plan);
                // The planar form, as SSE.
                let re = i32x4_shuffle::<0, 2, 4, 6>(x0, x1);
                let im = i32x4_shuffle::<1, 3, 5, 7>(x0, x1);
                let (re, im) = if res {
                    let rs = pv.map(|b| plan.residue(b));
                    let ct = |i: usize| coef.c(rs.get(i).copied().unwrap_or(0));
                    let st = |i: usize| coef.s(rs.get(i).copied().unwrap_or(0));
                    let cv = f32x4(ct(0), ct(1), ct(2), ct(3));
                    let sv = f32x4(st(0), st(1), st(2), st(3));
                    let pre = f32x4_sub(f32x4_mul(re, cv), f32x4_mul(im, sv));
                    let pim = f32x4_add(f32x4_mul(re, sv), f32x4_mul(im, cv));
                    let z = i32x4_eq(r, zero);
                    (v128_bitselect(re, pre, z), v128_bitselect(im, pim, z))
                } else {
                    (re, im)
                };
                let swap = i32x4_shr(i32x4_shl(q, 31), 31);
                let q1 = u32x4_shr(q, 1);
                let sre = i32x4_shl(v128_xor(q, q1), 31);
                let sim = i32x4_shl(q1, 31);
                let ore = v128_xor(v128_bitselect(im, re, swap), sre);
                let oim = v128_xor(v128_bitselect(re, im, swap), sim);
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
