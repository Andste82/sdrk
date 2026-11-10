//! `qam_soft_bits` `C32` (M44 §2, I44-1): the piecewise-linear max-log soft bits of 16- and
//! 64-QAM, bit-exact with the fallback on every level (a NaN soft bit may have another sign,
//! D296).
//!
//! The fallback forms, per component `x` (in memory order) with `v = x √E` and `a = |v|`, the soft
//! bits `s v`, `s (2 - a)` (16-QAM) or `s v`, `s (4 - a)`, `s (2 - |a - 4|)` (64-QAM), each
//! operation one rounded `f32` operation without FMA. The paths of the levels compute the same
//! operations on whole vectors and differ only in how the two or three soft bits of a component
//! reach their places in the output (D1414):
//!
//! - **Component layout, then interleave** (16-QAM at SSE2, SSE4.2, AVX-512, on NEON and wasm32,
//!   and at AVX2 for the scale 1; 64-QAM at AVX-512, on NEON and wasm32): the soft bits are
//!   computed per component lane and interleaved by shuffles (`unpcklps`/`unpckhps`; at AVX2 the
//!   in-lane unpacks put in order by `vinsertf128` and `vperm2f128`; `vpermt2ps` at AVX-512;
//!   `st2`/`st3` on NEON; `i32x4.shuffle` on wasm32).
//! - **Output layout** (64-QAM at SSE2, SSE4.2 and AVX2, 16-QAM at AVX2 for a scale other than
//!   1): `v` is shuffled into the output order first, one copy of each component per soft bit
//!   (`pshufd`/`shufps`, `vpermps`), and every lane then runs one uniform sequence with per-lane
//!   constants: `w = v | m1` (the sign bit for the second and third soft bits: `-a`),
//!   `y = w + k1` (`-0.0` for the first, which leaves `v` exactly as it is, also `-0.0` and NaN;
//!   4 for the others, 2 for the second of 16-QAM: `4 - a`, `2 - a`), and for 64-QAM
//!   `z = y | m2` (the sign bit for the third: `-|4 - a|`), `r = z + k2` (`-0.0`, `-0.0`, 2:
//!   `2 - |4 - a|`), then `r s`. `4 + (-a)` is `4 - a` and `|4 - a|` is `|a - 4|` exactly in
//!   IEEE arithmetic, so every soft bit has the bits of the fallback. Three shuffles per three
//!   output vectors instead of eight for the interleave of three streams, and the products with
//!   the scale on whole output vectors.
//!
//! Every path has a loop for the scale 1, which leaves out the products with the scale (`1 v`
//! is `v` exactly for every value but NaN), as the fallback has: the soft bits of `qam_demod`
//! and of the default `slice`. The loops run blocks of four steps, then single steps (on NEON
//! and wasm32 blocks of two steps only); the paths take whole vectors of components (a multiple
//! of two, whole symbols) and the fallback does the rest.

use fearless_simd::Simd as SimdToken;
use sdrk_core::kernels::constellation::{Qam, QamSoftBits, check_soft_bits_len};
use sdrk_core::{C32, Fallback, Result};

use crate::Simd;
use crate::level::dispatch_fn;

/// The loop of a level path: `$body` with `$i: &[f32; $vin]` (components) and
/// `$o: &mut [f32; $vout]` (their soft bits) over the whole steps of both slices, in blocks of
/// `$u` steps (one iteration of the loop, unrolled by the compiler) and then single steps.
/// Evaluates to the number of components done. A macro, so that the body is compiled inside
/// the `#[target_feature]` function of the path.
#[cfg(target_arch = "x86_64")]
macro_rules! steps {
    ($x:expr, $out:expr, $vin:literal, $vout:literal, $u:literal, |$i:ident, $o:ident| $body:block) => {{
        let (x, out): (&[f32], &mut [f32]) = ($x, $out);
        let (xb, _) = x.as_chunks::<{ $u * $vin }>();
        let (ob, _) = out.as_chunks_mut::<{ $u * $vout }>();
        let blocks = xb.len().min(ob.len());
        for (i, o) in xb.iter().zip(ob) {
            let (i, _) = i.as_chunks::<$vin>();
            let (o, _) = o.as_chunks_mut::<$vout>();
            for ($i, $o) in i.iter().zip(o) $body
        }
        let xr = x.get(blocks * $u * $vin..).unwrap_or(&[]);
        let or = out.get_mut(blocks * $u * $vout..).unwrap_or(&mut []);
        let (xs, _) = xr.as_chunks::<$vin>();
        let (os, _) = or.as_chunks_mut::<$vout>();
        let single = xs.len().min(os.len());
        for ($i, $o) in xs.iter().zip(os) $body
        (blocks * $u + single) * $vin
    }};
}

/// The loop of [`steps!`] without the single steps: `$body` over the whole blocks of `$u`
/// steps only, the rest (fewer than `$u` steps) left to the fallback. On NEON and wasm32, whose
/// fallback is compiled for the same target and vectorized the same way, the second loop would
/// only add set-up work to every call (D1414). Evaluates to the number of components done.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! blocks {
    ($x:expr, $out:expr, $vin:literal, $vout:literal, $u:literal, |$i:ident, $o:ident| $body:block) => {{
        let (x, out): (&[f32], &mut [f32]) = ($x, $out);
        let (xb, _) = x.as_chunks::<{ $u * $vin }>();
        let (ob, _) = out.as_chunks_mut::<{ $u * $vout }>();
        let blocks = xb.len().min(ob.len());
        for (i, o) in xb.iter().zip(ob) {
            let (i, _) = i.as_chunks::<$vin>();
            let (o, _) = o.as_chunks_mut::<$vout>();
            for ($i, $o) in i.iter().zip(o) $body
        }
        blocks * $u * $vin
    }};
}

/// The per-lane constants of the output layout of 64-QAM for three output vectors of `W` soft
/// bits each (`W` components, see the module documentation): entry `q` holds `[m1, k1, m2, k2]`
/// of the output vector `q`, its soft bit `l` being the soft bit `j % 3` of the component `j / 3`
/// with `j = W q + l`.
#[cfg(target_arch = "x86_64")]
#[allow(
    clippy::indexing_slicing,
    reason = "const evaluation only: an index out of range fails the build, never at run time"
)]
const fn lanes64<const W: usize>() -> [[[f32; W]; 4]; 3] {
    let mut c = [[[0.0f32; W]; 4]; 3];
    let mut j = 0;
    while j < 3 * W {
        let (m1, k1, m2, k2) = match j % 3 {
            0 => (0.0, -0.0, 0.0, -0.0),
            1 => (-0.0, 4.0, 0.0, -0.0),
            _ => (-0.0, 4.0, -0.0, 2.0),
        };
        let (q, l) = (j / W, j % W);
        c[q][0][l] = m1;
        c[q][1][l] = k1;
        c[q][2][l] = m2;
        c[q][3][l] = k2;
        j += 1;
    }
    c
}

/// The level path of 16-QAM (`qam16`) or 64-QAM for the current level, if there is one:
/// returns the number of leading components done.
#[inline(always)]
fn level_path<S: SimdToken>(
    simd: S,
    qam16: bool,
    x: &[f32],
    out: &mut [f32],
    s: f32,
    c: f32,
) -> usize {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx512() {
            return if qam16 {
                x86::qam16_avx512(t, x, out, s, c)
            } else {
                x86::qam64_avx512(t, x, out, s, c)
            };
        }
        if let Some(t) = level.as_avx2() {
            return if qam16 {
                x86::qam16_avx2(t, x, out, s, c)
            } else {
                x86::qam64_avx2(t, x, out, s, c)
            };
        }
        if let Some(t) = level.as_sse4_2() {
            return if qam16 {
                x86::qam16_sse4_2(t, x, out, s, c)
            } else {
                x86::qam64_sse4_2(t, x, out, s, c)
            };
        }
        if let Some(t) = crate::level::sse2_only(level) {
            return if qam16 {
                x86::qam16_sse2(t, x, out, s, c)
            } else {
                x86::qam64_sse2(t, x, out, s, c)
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return if qam16 {
            neon::qam16(t, x, out, s, c)
        } else {
            neon::qam64(t, x, out, s, c)
        };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return if qam16 {
            wasm::qam16(t, x, out, s, c)
        } else {
            wasm::qam64(t, x, out, s, c)
        };
    }
    let _ = (level, qam16, x, out, s, c);
    0
}

#[inline(always)]
fn qam16<S: SimdToken>(simd: S, x: &[f32], out: &mut [f32], s: f32, c: f32, done: &mut usize) {
    *done = level_path(simd, true, x, out, s, c);
}

#[inline(always)]
fn qam64<S: SimdToken>(simd: S, x: &[f32], out: &mut [f32], s: f32, c: f32, done: &mut usize) {
    *done = level_path(simd, false, x, out, s, c);
}

// The entries, one function per level (see `dispatch_fn!`).
dispatch_fn!(fn run16(x: &[f32], out: &mut [f32], s: f32, c: f32, done: &mut usize) => qam16);
dispatch_fn!(fn run64(x: &[f32], out: &mut [f32], s: f32, c: f32, done: &mut usize) => qam64);

impl QamSoftBits<C32> for Simd {
    fn qam_soft_bits(&self, qam: Qam, input: &[C32], output: &mut [f32], scale: f32) -> Result<()> {
        check_soft_bits_len(qam, input.len(), output.len())?;
        let x: &[f32] = bytemuck::cast_slice(input);
        let c = qam.sqrt_energy();
        let mut done = 0;
        match qam {
            Qam::Qam16 => run16(x, output, scale, c, &mut done),
            _ => run64(x, output, scale, c, &mut done),
        }
        // The rest (and everything on a level without a path): whole symbols.
        let n = done / 2;
        let bits = qam.bits_per_symbol();
        match (input.get(n..), output.get_mut(n * bits..)) {
            (Some(i), Some(o)) if !i.is_empty() => Fallback.qam_soft_bits(qam, i, o, scale),
            _ => Ok(()),
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    //! SSE2 and SSE4.2 (the same code), AVX2 and AVX-512. Every path has a loop for the scale 1
    //! (`s v = v` exactly for every value but NaN, whose bits are not part of the contract), which
    //! leaves out the products with the scale, and one for any other scale.
    #![allow(
        clippy::large_stack_frames,
        reason = "the lint counts the slices of the unrolled bodies before optimization; the loops \
                  keep them in registers"
    )]
    use core::arch::x86_64::{
        __m128, __m256, __m512, __m512i, _mm_add_ps, _mm_and_ps, _mm_castps_si128,
        _mm_castsi128_ps, _mm_mul_ps, _mm_or_ps, _mm_set1_epi32, _mm_set1_ps, _mm_shuffle_epi32,
        _mm_shuffle_ps, _mm_sub_ps, _mm_unpackhi_ps, _mm_unpacklo_ps, _mm256_add_ps, _mm256_and_ps,
        _mm256_castps256_ps128, _mm256_castsi256_ps, _mm256_insertf128_ps, _mm256_mul_ps,
        _mm256_or_ps, _mm256_permute2f128_ps, _mm256_permutevar8x32_ps, _mm256_set1_epi32,
        _mm256_set1_ps, _mm256_setr_epi32, _mm256_setr_ps, _mm256_sub_ps, _mm256_unpackhi_ps,
        _mm256_unpacklo_ps, _mm512_and_ps, _mm512_castsi512_ps, _mm512_mul_ps,
        _mm512_permutex2var_ps, _mm512_set1_epi32, _mm512_set1_ps, _mm512_setr_epi32,
        _mm512_sub_ps,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4, f32x8, f32x16, i32x16};

    use super::lanes64;

    /// The constants of the output layout of 64-QAM for four components (12 soft bits).
    static LANES64_SSE: [[[f32; 4]; 4]; 3] = lanes64::<4>();
    /// The constants of the output layout of 64-QAM for eight components (24 soft bits).
    static LANES64_AVX2: [[[f32; 8]; 4]; 3] = lanes64::<8>();

    /// `$v` times the scale `$sv` (`$mul`), or `$v` itself in the loop of the scale 1.
    macro_rules! scaled {
        (true, $mul:ident, $sv:expr, $v:expr) => {
            $v
        };
        (false, $mul:ident, $sv:expr, $v:expr) => {
            $mul($sv, $v)
        };
    }

    /// The 16-QAM path of SSE2 and SSE4.2: four components per step, the two soft bits per
    /// component lane, interleaved by `unpcklps` and `unpckhps`.
    macro_rules! qam16_sse {
        ($name:ident, $token:ident) => {
            fearless_simd::kernel!(
                pub(super) fn $name(
                    t: $token,
                    x: &[f32],
                    out: &mut [f32],
                    s: f32,
                    c: f32,
                ) -> usize {
                    let (sv, cv, two) = (_mm_set1_ps(s), _mm_set1_ps(c), _mm_set1_ps(2.0));
                    let abs = _mm_castsi128_ps(_mm_set1_epi32(0x7fff_ffff));
                    macro_rules! run {
                        ($unit:tt) => {
                            steps!(x, out, 4, 8, 4, |i, o| {
                                let xv: __m128 = f32x4::from_slice(t, i).into();
                                let v = _mm_mul_ps(xv, cv);
                                let b0 = scaled!($unit, _mm_mul_ps, sv, v);
                                let b1 = scaled!(
                                    $unit,
                                    _mm_mul_ps,
                                    sv,
                                    _mm_sub_ps(two, _mm_and_ps(v, abs))
                                );
                                let (o0, o1) = o.split_at_mut(4);
                                f32x4::simd_from(t, _mm_unpacklo_ps(b0, b1)).store_slice(o0);
                                f32x4::simd_from(t, _mm_unpackhi_ps(b0, b1)).store_slice(o1);
                            })
                        };
                    }
                    if s == 1.0 { run!(true) } else { run!(false) }
                }
            );
        };
    }
    qam16_sse!(qam16_sse2, Sse2);
    qam16_sse!(qam16_sse4_2, Sse4_2);

    /// The 64-QAM path of SSE2 and SSE4.2: four components per step in the output layout (the
    /// three copies of `v` by `pshufd`, which leaves `v` in its register, and the per-lane
    /// sequence of the module documentation).
    macro_rules! qam64_sse {
        ($name:ident, $token:ident) => {
            fearless_simd::kernel!(
                pub(super) fn $name(
                    t: $token,
                    x: &[f32],
                    out: &mut [f32],
                    s: f32,
                    c: f32,
                ) -> usize {
                    let (sv, cv) = (_mm_set1_ps(s), _mm_set1_ps(c));
                    let ld = |r: &[f32; 4]| -> __m128 { f32x4::from_slice(t, r).into() };
                    let lanes = |[m1, k1, m2, k2]: &[[f32; 4]; 4]| (ld(m1), ld(k1), ld(m2), ld(k2));
                    let [c0, c1, c2] = &LANES64_SSE;
                    let consts = [lanes(c0), lanes(c1), lanes(c2)];
                    macro_rules! run {
                        ($unit:tt) => {
                            steps!(x, out, 4, 12, 4, |i, o| {
                                let xv: __m128 = f32x4::from_slice(t, i).into();
                                let v = _mm_castps_si128(_mm_mul_ps(xv, cv));
                                // Components 0 0 0 1, 1 1 2 2, 2 3 3 3.
                                let d = [
                                    _mm_castsi128_ps(_mm_shuffle_epi32::<0x40>(v)),
                                    _mm_castsi128_ps(_mm_shuffle_epi32::<0xa5>(v)),
                                    _mm_shuffle_ps::<0xfe>(
                                        _mm_castsi128_ps(v),
                                        _mm_castsi128_ps(v),
                                    ),
                                ];
                                let (o0, rest) = o.split_at_mut(4);
                                let (o1, o2) = rest.split_at_mut(4);
                                for ((o, d), (m1, k1, m2, k2)) in
                                    [o0, o1, o2].into_iter().zip(d).zip(consts)
                                {
                                    let y = _mm_add_ps(_mm_or_ps(d, m1), k1);
                                    let r = _mm_add_ps(_mm_or_ps(y, m2), k2);
                                    f32x4::simd_from(t, scaled!($unit, _mm_mul_ps, sv, r))
                                        .store_slice(o);
                                }
                            })
                        };
                    }
                    if s == 1.0 { run!(true) } else { run!(false) }
                }
            );
        };
    }
    qam64_sse!(qam64_sse2, Sse2);
    qam64_sse!(qam64_sse4_2, Sse4_2);

    fearless_simd::kernel!(
        /// The 16-QAM path of AVX2: eight components per step. Scale 1: the two soft bits per
        /// component lane, interleaved by the in-lane unpacks and put in order by `vinsertf128`
        /// and `vperm2f128` (the least shuffle work at the store bound of `znver3`). Any other
        /// scale: the output layout, `v` copied per soft bit by two `vpermps` and the lanes of
        /// the second soft bit turned into `2 - |v|` by `w = v | m`, `w + k` (`m` the sign bit,
        /// `k` 2 there; 0 and -0.0 in the lanes of the first soft bit), so that the products
        /// with the scale run on whole output vectors.
        pub(super) fn qam16_avx2(t: Avx2, x: &[f32], out: &mut [f32], s: f32, c: f32) -> usize {
            let (sv, cv, two) = (_mm256_set1_ps(s), _mm256_set1_ps(c), _mm256_set1_ps(2.0));
            if s == 1.0 {
                let abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
                steps!(x, out, 8, 16, 4, |i, o| {
                    let xv: __m256 = f32x8::from_slice(t, i).into();
                    let v = _mm256_mul_ps(xv, cv);
                    let b1 = _mm256_sub_ps(two, _mm256_and_ps(v, abs));
                    let lo = _mm256_unpacklo_ps(v, b1);
                    let hi = _mm256_unpackhi_ps(v, b1);
                    let (o0, o1) = o.split_at_mut(8);
                    let first = _mm256_insertf128_ps::<1>(lo, _mm256_castps256_ps128(hi));
                    f32x8::simd_from(t, first).store_slice(o0);
                    f32x8::simd_from(t, _mm256_permute2f128_ps::<0x31>(lo, hi)).store_slice(o1);
                })
            } else {
                let sign = _mm256_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
                let k = _mm256_setr_ps(-0.0, 2.0, -0.0, 2.0, -0.0, 2.0, -0.0, 2.0);
                let lo = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
                let hi = _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7);
                steps!(x, out, 8, 16, 4, |i, o| {
                    let xv: __m256 = f32x8::from_slice(t, i).into();
                    let v = _mm256_mul_ps(xv, cv);
                    let (o0, o1) = o.split_at_mut(8);
                    for (idx, o) in [(lo, o0), (hi, o1)] {
                        let d = _mm256_permutevar8x32_ps(v, idx);
                        let r = _mm256_add_ps(_mm256_or_ps(d, sign), k);
                        f32x8::simd_from(t, _mm256_mul_ps(sv, r)).store_slice(o);
                    }
                })
            }
        }
    );

    fearless_simd::kernel!(
        /// The 64-QAM path of AVX2: eight components per step in the output layout (three
        /// `vpermps` of `v` and the per-lane sequence of the module documentation).
        pub(super) fn qam64_avx2(t: Avx2, x: &[f32], out: &mut [f32], s: f32, c: f32) -> usize {
            let (sv, cv) = (_mm256_set1_ps(s), _mm256_set1_ps(c));
            let ld = |r: &[f32; 8]| -> __m256 { f32x8::from_slice(t, r).into() };
            let lanes = |[m1, k1, m2, k2]: &[[f32; 8]; 4]| (ld(m1), ld(k1), ld(m2), ld(k2));
            let [c0, c1, c2] = &LANES64_AVX2;
            let consts = [lanes(c0), lanes(c1), lanes(c2)];
            let idx = [
                _mm256_setr_epi32(0, 0, 0, 1, 1, 1, 2, 2),
                _mm256_setr_epi32(2, 3, 3, 3, 4, 4, 4, 5),
                _mm256_setr_epi32(5, 5, 6, 6, 6, 7, 7, 7),
            ];
            macro_rules! run {
                ($unit:tt) => {
                    steps!(x, out, 8, 24, 4, |i, o| {
                        let xv: __m256 = f32x8::from_slice(t, i).into();
                        let v = _mm256_mul_ps(xv, cv);
                        let (o0, rest) = o.split_at_mut(8);
                        let (o1, o2) = rest.split_at_mut(8);
                        for ((o, idx), (m1, k1, m2, k2)) in
                            [o0, o1, o2].into_iter().zip(idx).zip(consts)
                        {
                            let d = _mm256_permutevar8x32_ps(v, idx);
                            let y = _mm256_add_ps(_mm256_or_ps(d, m1), k1);
                            let r = _mm256_add_ps(_mm256_or_ps(y, m2), k2);
                            f32x8::simd_from(t, scaled!($unit, _mm256_mul_ps, sv, r))
                                .store_slice(o);
                        }
                    })
                };
            }
            if s == 1.0 { run!(true) } else { run!(false) }
        }
    );

    fearless_simd::kernel!(
        /// The 16-QAM path of AVX-512: sixteen components per step, the two soft bits per
        /// component lane, interleaved by two `vpermt2ps`.
        pub(super) fn qam16_avx512(t: Avx512, x: &[f32], out: &mut [f32], s: f32, c: f32) -> usize {
            let (sv, cv, two) = (_mm512_set1_ps(s), _mm512_set1_ps(c), _mm512_set1_ps(2.0));
            let abs = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fff_ffff));
            let lo = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
            let hi =
                _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
            macro_rules! run {
                ($unit:tt) => {
                    steps!(x, out, 16, 32, 4, |i, o| {
                        let xv: __m512 = f32x16::from_slice(t, i).into();
                        let v = _mm512_mul_ps(xv, cv);
                        let b0 = scaled!($unit, _mm512_mul_ps, sv, v);
                        let b1 = scaled!(
                            $unit,
                            _mm512_mul_ps,
                            sv,
                            _mm512_sub_ps(two, _mm512_and_ps(v, abs))
                        );
                        let (o0, o1) = o.split_at_mut(16);
                        f32x16::simd_from(t, _mm512_permutex2var_ps(b0, lo, b1)).store_slice(o0);
                        f32x16::simd_from(t, _mm512_permutex2var_ps(b0, hi, b1)).store_slice(o1);
                    })
                };
            }
            if s == 1.0 { run!(true) } else { run!(false) }
        }
    );

    /// The `vpermt2ps` indices of the 64-QAM interleave at AVX-512 for the output vector `q`:
    /// the first selects the first (`a`, 0 to 15) and second (`b`, 16 to 31) soft bits of the
    /// components, the second keeps those lanes (0 to 15) and takes the third (`c`, 16 to 31).
    #[allow(
        clippy::indexing_slicing,
        reason = "const evaluation only: an index out of range fails the build, never at run time"
    )]
    const fn interleave64(q: usize) -> [[i32; 16]; 2] {
        let (mut ab, mut abc) = ([0i32; 16], [0i32; 16]);
        let mut l = 0;
        while l < 16 {
            let j = 16 * q + l;
            let comp = (j / 3) as i32;
            match j % 3 {
                0 => {
                    ab[l] = comp;
                    abc[l] = l as i32;
                }
                1 => {
                    ab[l] = 16 + comp;
                    abc[l] = l as i32;
                }
                _ => abc[l] = 16 + comp,
            }
            l += 1;
        }
        [ab, abc]
    }

    /// The indices of [`interleave64`] for the three output vectors.
    static INTERLEAVE64: [[[i32; 16]; 2]; 3] = [interleave64(0), interleave64(1), interleave64(2)];

    fearless_simd::kernel!(
        /// The 64-QAM path of AVX-512: sixteen components per step, the three soft bits per
        /// component lane, interleaved by two `vpermt2ps` per output vector.
        pub(super) fn qam64_avx512(t: Avx512, x: &[f32], out: &mut [f32], s: f32, c: f32) -> usize {
            let (sv, cv, two, four) = (
                _mm512_set1_ps(s),
                _mm512_set1_ps(c),
                _mm512_set1_ps(2.0),
                _mm512_set1_ps(4.0),
            );
            let abs = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fff_ffff));
            let ld = |r: &[i32; 16]| -> __m512i { i32x16::from_slice(t, r).into() };
            let [[a0, b0], [a1, b1], [a2, b2]] = &INTERLEAVE64;
            let idx = [(ld(a0), ld(b0)), (ld(a1), ld(b1)), (ld(a2), ld(b2))];
            macro_rules! run {
                ($unit:tt) => {
                    steps!(x, out, 16, 48, 4, |i, o| {
                        let xv: __m512 = f32x16::from_slice(t, i).into();
                        let v = _mm512_mul_ps(xv, cv);
                        let u = _mm512_sub_ps(four, _mm512_and_ps(v, abs));
                        let b0 = scaled!($unit, _mm512_mul_ps, sv, v);
                        let b1 = scaled!($unit, _mm512_mul_ps, sv, u);
                        let b2 = scaled!(
                            $unit,
                            _mm512_mul_ps,
                            sv,
                            _mm512_sub_ps(two, _mm512_and_ps(u, abs))
                        );
                        let (o0, rest) = o.split_at_mut(16);
                        let (o1, o2) = rest.split_at_mut(16);
                        for (o, (ab, abc)) in [o0, o1, o2].into_iter().zip(idx) {
                            let w =
                                _mm512_permutex2var_ps(_mm512_permutex2var_ps(b0, ab, b1), abc, b2);
                            f32x16::simd_from(t, w).store_slice(o);
                        }
                    })
                };
            }
            if s == 1.0 { run!(true) } else { run!(false) }
        }
    );
}

#[cfg(target_arch = "aarch64")]
mod neon {
    //! The soft bits per component lane, interleaved by `st2` (16-QAM) and `st3` (64-QAM); a loop
    //! for the scale 1 without the products, as on x86.
    #![allow(
        clippy::large_stack_frames,
        reason = "the lint counts the slices of the unrolled bodies before optimization; the loops \
                  keep them in registers"
    )]
    #![allow(unsafe_code)]

    use core::arch::aarch64::{
        float32x4_t, float32x4x2_t, float32x4x3_t, vabsq_f32, vdupq_n_f32, vmulq_f32, vst2q_f32,
        vst3q_f32, vsubq_f32,
    };
    use fearless_simd::f32x4;
    use fearless_simd::prelude::*;

    /// `$v` times the scale `$sv`, or `$v` itself in the loop of the scale 1.
    macro_rules! scaled {
        (true, $sv:expr, $v:expr) => {
            $v
        };
        (false, $sv:expr, $v:expr) => {
            vmulq_f32($sv, $v)
        };
    }

    /// Stores the two soft bits of four components interleaved into `o`.
    #[inline(always)]
    fn store2(o: &mut [f32; 8], a: float32x4_t, b: float32x4_t) {
        // SAFETY: `vst2q_f32` writes 8 consecutive `f32` with no alignment requirement beyond
        // that of `f32`, and `o` holds exactly 8 and is borrowed exclusively.
        unsafe { vst2q_f32(o.as_mut_ptr(), float32x4x2_t(a, b)) }
    }

    /// Stores the three soft bits of four components interleaved into `o`.
    #[inline(always)]
    fn store3(o: &mut [f32; 12], a: float32x4_t, b: float32x4_t, c: float32x4_t) {
        // SAFETY: `vst3q_f32` writes 12 consecutive `f32` with no alignment requirement beyond
        // that of `f32`, and `o` holds exactly 12 and is borrowed exclusively.
        unsafe { vst3q_f32(o.as_mut_ptr(), float32x4x3_t(a, b, c)) }
    }

    fearless_simd::kernel!(
        pub(super) fn qam16(t: Neon, x: &[f32], out: &mut [f32], s: f32, c: f32) -> usize {
            let (sv, cv, two) = (vdupq_n_f32(s), vdupq_n_f32(c), vdupq_n_f32(2.0));
            macro_rules! run {
                ($unit:tt) => {
                    blocks!(x, out, 4, 8, 2, |i, o| {
                        let xv: float32x4_t = f32x4::from_slice(t, i).into();
                        let v = vmulq_f32(xv, cv);
                        let b0 = scaled!($unit, sv, v);
                        let b1 = scaled!($unit, sv, vsubq_f32(two, vabsq_f32(v)));
                        store2(o, b0, b1);
                    })
                };
            }
            if s == 1.0 { run!(true) } else { run!(false) }
        }
    );

    fearless_simd::kernel!(
        pub(super) fn qam64(t: Neon, x: &[f32], out: &mut [f32], s: f32, c: f32) -> usize {
            let (sv, cv) = (vdupq_n_f32(s), vdupq_n_f32(c));
            let (two, four) = (vdupq_n_f32(2.0), vdupq_n_f32(4.0));
            macro_rules! run {
                ($unit:tt) => {
                    blocks!(x, out, 4, 12, 2, |i, o| {
                        let xv: float32x4_t = f32x4::from_slice(t, i).into();
                        let v = vmulq_f32(xv, cv);
                        let u = vsubq_f32(four, vabsq_f32(v));
                        let b0 = scaled!($unit, sv, v);
                        let b1 = scaled!($unit, sv, u);
                        let b2 = scaled!($unit, sv, vsubq_f32(two, vabsq_f32(u)));
                        store3(o, b0, b1, b2);
                    })
                };
            }
            if s == 1.0 { run!(true) } else { run!(false) }
        }
    );
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    //! The soft bits per component lane, interleaved by `i32x4.shuffle`; a loop for the scale 1
    //! without the products, as on x86.
    #![allow(
        clippy::large_stack_frames,
        reason = "the lint counts the slices of the unrolled bodies before optimization; the loops \
                  keep them in registers"
    )]
    use core::arch::wasm32::{f32x4_abs, f32x4_mul, f32x4_splat, f32x4_sub, i32x4_shuffle, v128};
    use fearless_simd::f32x4;
    use fearless_simd::prelude::*;

    /// `$v` times the scale `$sv`, or `$v` itself in the loop of the scale 1.
    macro_rules! scaled {
        (true, $sv:expr, $v:expr) => {
            $v
        };
        (false, $sv:expr, $v:expr) => {
            f32x4_mul($sv, $v)
        };
    }

    fearless_simd::kernel!(
        pub(super) fn qam16(t: WasmSimd128, x: &[f32], out: &mut [f32], s: f32, c: f32) -> usize {
            let (sv, cv, two) = (f32x4_splat(s), f32x4_splat(c), f32x4_splat(2.0));
            macro_rules! run {
                ($unit:tt) => {
                    blocks!(x, out, 4, 8, 2, |i, o| {
                        let xv: v128 = f32x4::from_slice(t, i).into();
                        let v = f32x4_mul(xv, cv);
                        let b0 = scaled!($unit, sv, v);
                        let b1 = scaled!($unit, sv, f32x4_sub(two, f32x4_abs(v)));
                        let (o0, o1) = o.split_at_mut(4);
                        f32x4::simd_from(t, i32x4_shuffle::<0, 4, 1, 5>(b0, b1)).store_slice(o0);
                        f32x4::simd_from(t, i32x4_shuffle::<2, 6, 3, 7>(b0, b1)).store_slice(o1);
                    })
                };
            }
            if s == 1.0 { run!(true) } else { run!(false) }
        }
    );

    fearless_simd::kernel!(
        pub(super) fn qam64(t: WasmSimd128, x: &[f32], out: &mut [f32], s: f32, c: f32) -> usize {
            let (sv, cv) = (f32x4_splat(s), f32x4_splat(c));
            let (two, four) = (f32x4_splat(2.0), f32x4_splat(4.0));
            macro_rules! run {
                ($unit:tt) => {
                    blocks!(x, out, 4, 12, 2, |i, o| {
                        let xv: v128 = f32x4::from_slice(t, i).into();
                        let v = f32x4_mul(xv, cv);
                        let u = f32x4_sub(four, f32x4_abs(v));
                        let b0 = scaled!($unit, sv, v);
                        let b1 = scaled!($unit, sv, u);
                        let b2 = scaled!($unit, sv, f32x4_sub(two, f32x4_abs(u)));
                        // a0 b0 c0 a1 | b1 c1 a2 b2 | c2 a3 b3 c3
                        let ab0 = i32x4_shuffle::<0, 4, 0, 1>(b0, b1);
                        let ab1 = i32x4_shuffle::<5, 2, 6, 0>(b0, b1);
                        let ab2 = i32x4_shuffle::<0, 3, 7, 0>(b0, b1);
                        let (o0, rest) = o.split_at_mut(4);
                        let (o1, o2) = rest.split_at_mut(4);
                        f32x4::simd_from(t, i32x4_shuffle::<0, 1, 4, 3>(ab0, b2)).store_slice(o0);
                        f32x4::simd_from(t, i32x4_shuffle::<0, 5, 1, 2>(ab1, b2)).store_slice(o1);
                        f32x4::simd_from(t, i32x4_shuffle::<6, 1, 2, 7>(ab2, b2)).store_slice(o2);
                    })
                };
            }
            if s == 1.0 { run!(true) } else { run!(false) }
        }
    );
}
