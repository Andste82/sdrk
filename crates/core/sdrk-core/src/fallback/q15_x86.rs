//! The one x86-64 module of `core::arch` for the `Q15` / `CQ15` fallbacks (the maintainer's
//! decision D3017 on IO2-23, extending D3005): `pmaddwd` (`_mm_madd_epi16`) for the exact
//! sums of the FIR fallbacks (IO1-5 `fir_decimate` `CQ15`, IO1-6 the `Q15` / `CQ15`
//! interpolators) and `vpmulhrsw` for `mul_real` `Q15` with AVX2 (IO1-7). The compiler forms
//! neither instruction from the portable code (O1, O2b), so the fallbacks name them here. The
//! 256-bit form of `mag_sq` `C32` with AVX-512 (IO1-7) was measured and not taken: CI run
//! 46126247106 gave 0.26 / 0.39 cycles per sample on `znver4` / `sapphirerapids` against 0.37 /
//! 0.26 of the compiled loop (0.840 / 1.405 of CMSIS-DSP against 1.210 / 0.942).
//!
//! Every function gives the bits of the portable form (the tests of each user compare them):
//!
//! - **The FIR sums.** A tap `t` is split into `256 h + l` (`h = t >> 8` in `-128..=127`,
//!   `l = t & 255` in `0..=255`), so that `pmaddwd` of a sample with `h` or `l` and the sum of two
//!   such products stay far below 2^31: `|x l| <= 2^15 * 255`, `|x h| <= 2^22`. The lanes sum at
//!   most 128 products each (the callers hold at most 256 taps per window, a quarter of them per
//!   lane and component), below 2^31 in magnitude, so every lane is exact in `i32`, and
//!   `256 H + L` in `i64` is the exact sum of the window, the value of the `i64` products of the
//!   portable form. `CQ15` windows are interleaved (`re, im` per sample): `pshuflw` / `pshufhw`
//!   put the two `re` and the two `im` of each half next to each other, so that one `pmaddwd`
//!   with the taps `[t0, t1, t0, t1, t2, t3, t2, t3]` gives the pair sums of both components.
//! - **`mul_real` `Q15`.** `vpmulhrsw` is `(a b + 2^14) >> 15` of the exact product, which is
//!   `round_shift(a b, 15)`; only `-32768 * -32768` gives `32768`, which the instruction wraps to
//!   `-32768` and `sat16` makes `32767`: no other product rounds to `-32768`
//!   (`-32768 * 32767` gives `-32767`), so a compare with `-32768` and an `xor` restore it.
//!
//! `cargo xtask check heap` allows `unsafe` in the core layer only in this module and in the
//! `matrix_x86` module of the `stereo_matrix` fallback (D3012, D3017).

use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_loadu_si128, _mm_madd_epi16, _mm_setzero_si128,
    _mm_shufflehi_epi16, _mm_shufflelo_epi16, _mm_storeu_si128,
};

/// Taps split for `pmaddwd`: the high parts `h` and the low parts `l` of `t = 256 h + l`.
#[inline(always)]
#[allow(clippy::inline_always, reason = "the split runs inside the preparation loops")]
pub(super) fn split(t: i16) -> (i16, i16) {
    (t >> 8, t & 0xFF)
}

/// The four `i32` lanes of `v`.
#[inline(always)]
#[allow(clippy::inline_always, reason = "a store of a register in the hot loops")]
fn lanes(v: __m128i) -> [i32; 4] {
    let mut out = [0i32; 4];
    // SAFETY: `out` is 16 bytes, which the unaligned store writes; SSE2 is part of x86-64.
    unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), v) };
    out
}

/// `256 H + L` of the lanes `idx` of `h` and `l`, in `i64`.
#[inline(always)]
#[allow(clippy::inline_always, reason = "the finish of every output")]
fn join(high: [i32; 4], low: [i32; 4], idx: [usize; 2]) -> i64 {
    let mut sum = 0i64;
    for lane in idx {
        let (Some(&hv), Some(&lv)) = (high.get(lane), low.get(lane)) else {
            continue;
        };
        sum += 256 * i64::from(hv) + i64::from(lv);
    }
    sum
}

/// The exact sums `sum_k t[k] x_j[k]` of `N` real windows `x_j = x[j step..]` of `8 c` values
/// each, `c = hi.len()`, with the split taps `hi` and `lo` (eight per block). `None` if `x` is
/// too short.
#[cfg(not(target_feature = "avx2"))]
#[inline(always)]
#[allow(clippy::inline_always, reason = "the group size must be a constant in the hot loop")]
pub(super) fn sums_real<const N: usize>(
    hi: &[[i16; 8]],
    lo: &[[i16; 8]],
    x: &[i16],
    step: usize,
) -> Option<[i64; N]> {
    let blocks = hi.len().min(lo.len());
    if x.len() < (N - 1) * step + 8 * blocks {
        return None;
    }
    // SAFETY: SSE2 is part of x86-64; every load reads 16 bytes inside `hi`, `lo` or `x`: the
    // block `b < blocks` of the taps, and `x[j step + 8 b..j step + 8 b + 8]` with
    // `j step + 8 b + 8 <= (N - 1) step + 8 blocks <= x.len()` (checked above).
    let (ah, al) = unsafe {
        let (mut ah, mut al) = ([_mm_setzero_si128(); N], [_mm_setzero_si128(); N]);
        for b in 0..blocks {
            let th = _mm_loadu_si128(hi.as_ptr().add(b).cast());
            let tl = _mm_loadu_si128(lo.as_ptr().add(b).cast());
            for (j, (h, l)) in ah.iter_mut().zip(al.iter_mut()).enumerate() {
                let v = _mm_loadu_si128(x.as_ptr().add(j * step + 8 * b).cast());
                *h = _mm_add_epi32(*h, _mm_madd_epi16(v, th));
                *l = _mm_add_epi32(*l, _mm_madd_epi16(v, tl));
            }
        }
        (ah, al)
    };
    let mut out = [0i64; N];
    for ((o, h), l) in out.iter_mut().zip(ah).zip(al) {
        let (h, l) = (lanes(h), lanes(l));
        *o = join(h, l, [0, 1]) + join(h, l, [2, 3]);
    }
    Some(out)
}

/// The exact sums `(re, im)` of `N` complex windows `x_j = x[j step..]` (interleaved `re, im`)
/// of `4 c` samples each, `c = hi.len()`, with the split taps `hi` and `lo` in the blocks
/// `[t0, t1, t0, t1, t2, t3, t2, t3]` of four consecutive taps. `None` if `x` is too short.
#[inline(always)]
#[allow(clippy::inline_always, reason = "the group size must be a constant in the hot loop")]
pub(super) fn sums_complex<const N: usize>(
    hi: &[[i16; 8]],
    lo: &[[i16; 8]],
    x: &[i16],
    step: usize,
) -> Option<[(i64, i64); N]> {
    let blocks = hi.len().min(lo.len());
    if x.len() < (N - 1) * step + 8 * blocks {
        return None;
    }
    // SAFETY: as in `sums_real`.
    let (ah, al) = unsafe {
        let (mut ah, mut al) = ([_mm_setzero_si128(); N], [_mm_setzero_si128(); N]);
        for b in 0..blocks {
            let th = _mm_loadu_si128(hi.as_ptr().add(b).cast());
            let tl = _mm_loadu_si128(lo.as_ptr().add(b).cast());
            for (j, (h, l)) in ah.iter_mut().zip(al.iter_mut()).enumerate() {
                let v = _mm_loadu_si128(x.as_ptr().add(j * step + 8 * b).cast());
                // `[r0, i0, r1, i1 | r2, i2, r3, i3]` to `[r0, r1, i0, i1 | r2, r3, i2, i3]`.
                let v = _mm_shufflehi_epi16::<0xD8>(_mm_shufflelo_epi16::<0xD8>(v));
                *h = _mm_add_epi32(*h, _mm_madd_epi16(v, th));
                *l = _mm_add_epi32(*l, _mm_madd_epi16(v, tl));
            }
        }
        (ah, al)
    };
    let mut out = [(0i64, 0i64); N];
    for ((o, h), l) in out.iter_mut().zip(ah).zip(al) {
        let (h, l) = (lanes(h), lanes(l));
        *o = (join(h, l, [0, 2]), join(h, l, [1, 3]));
    }
    Some(out)
}

/// `mul_real` `Q15` with AVX2: `q15_mul` of the pairs of `a` and `b` into `out`, sixteen values
/// per step; returns the number of values done (the rest runs the portable loop).
#[cfg(target_feature = "avx2")]
#[inline]
pub(super) fn mul_real_q15(lhs: &[i16], rhs: &[i16], out: &mut [i16]) -> usize {
    use core::arch::x86_64::{
        _mm256_cmpeq_epi16, _mm256_loadu_si256, _mm256_mulhrs_epi16, _mm256_set1_epi16,
        _mm256_storeu_si256, _mm256_xor_si256,
    };
    let done = lhs.len().min(rhs.len()).min(out.len()) / 16 * 16;
    // SAFETY: AVX2 is enabled for the build (`cfg(target_feature)`); every load and store
    // touches the 32 bytes of `lhs[at..at + 16]`, `rhs[at..at + 16]` and `out[at..at + 16]`
    // with `at + 16 <= done`, which is within all three slices.
    unsafe {
        let min = _mm256_set1_epi16(i16::MIN);
        let mut at = 0;
        while at < done {
            let va = _mm256_loadu_si256(lhs.as_ptr().add(at).cast());
            let vb = _mm256_loadu_si256(rhs.as_ptr().add(at).cast());
            let prod = _mm256_mulhrs_epi16(va, vb);
            let prod = _mm256_xor_si256(prod, _mm256_cmpeq_epi16(prod, min));
            _mm256_storeu_si256(out.as_mut_ptr().add(at).cast(), prod);
            at += 16;
        }
    }
    done
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::unwrap_used, clippy::cast_possible_truncation)]
mod tests {
    //! Every function against the portable form it replaces, bit for bit, on the extremes and
    //! on random values.

    use std::vec::Vec;

    /// A small linear congruential generator.
    fn next(s: &mut u32) -> i16 {
        *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*s >> 16) as i16
    }

    /// Values with the extremes of `i16` every few positions.
    fn values(seed: &mut u32, n: usize) -> Vec<i16> {
        (0..n)
            .map(|i| match i % 5 {
                0 => i16::MIN,
                1 => i16::MAX,
                _ => next(seed),
            })
            .collect()
    }

    #[cfg(not(target_feature = "avx2"))]
    #[test]
    fn the_split_sums_are_the_exact_sums() {
        use super::{split, sums_complex, sums_real};
        let mut seed = 7;
        for blocks in [1usize, 2, 7, 16, 32, 64] {
            for extreme in [false, true] {
                // The extremes: taps of the largest high and low parts, samples of -32768.
                let taps: Vec<i16> = if extreme {
                    (0..8 * blocks)
                        .map(|k| if k % 2 == 0 { i16::MAX } else { i16::MIN })
                        .collect()
                } else {
                    values(&mut seed, 8 * blocks)
                };
                let x: Vec<i16> = if extreme {
                    (0..16 * blocks + 24).map(|_| i16::MIN).collect()
                } else {
                    values(&mut seed, 16 * blocks + 24)
                };
                let block = |slots: [usize; 8], per: usize| {
                    let mut hi = Vec::new();
                    let mut lo = Vec::new();
                    for b in 0..taps.len() / per {
                        let (mut h, mut l) = ([0i16; 8], [0i16; 8]);
                        for (i, j) in slots.into_iter().enumerate() {
                            (h[i], l[i]) = split(taps[b * per + j]);
                        }
                        hi.push(h);
                        lo.push(l);
                    }
                    (hi, lo)
                };
                // Real: eight taps per block, windows one value apart.
                let (hi, lo) = block([0, 1, 2, 3, 4, 5, 6, 7], 8);
                let got = sums_real::<4>(&hi, &lo, &x, 3).unwrap();
                for (j, g) in got.iter().enumerate() {
                    let want: i64 = taps
                        .iter()
                        .zip(&x[3 * j..])
                        .map(|(&t, &v)| i64::from(t) * i64::from(v))
                        .sum();
                    assert_eq!(*g, want, "real, {blocks} blocks, window {j}");
                }
                // Complex: four taps (two blocks of the components) per block, the samples
                // interleaved.
                let (hi, lo) = block([0, 1, 0, 1, 2, 3, 2, 3], 4);
                let got = sums_complex::<3>(&hi, &lo, &x, 6).unwrap();
                for (j, g) in got.iter().enumerate() {
                    let w = &x[6 * j..];
                    let comp = |c: usize| -> i64 {
                        taps.iter()
                            .enumerate()
                            .map(|(k, &t)| i64::from(t) * i64::from(w[2 * k + c]))
                            .sum()
                    };
                    assert_eq!(*g, (comp(0), comp(1)), "complex, {blocks} blocks, window {j}");
                }
                assert!(sums_complex::<4>(&hi, &lo, &x[..16 * blocks], 6).is_none());
            }
        }
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn mul_real_q15_is_q15_mul() {
        let mut seed = 11;
        let mut a = values(&mut seed, 1000);
        let mut b = values(&mut seed, 1000);
        // Every pair of the extremes and their neighbours.
        let edge = [i16::MIN, i16::MIN + 1, -1, 0, 1, i16::MAX - 1, i16::MAX];
        for &x in &edge {
            for &y in &edge {
                a.push(x);
                b.push(y);
            }
        }
        let mut out = vec_of(a.len());
        let done = super::mul_real_q15(&a, &b, &mut out);
        assert_eq!(done, a.len() / 16 * 16);
        for i in 0..done {
            let want = crate::num::q15_mul(crate::Q15(a[i]), crate::Q15(b[i])).0;
            assert_eq!(out[i], want, "{} * {}", a[i], b[i]);
        }
    }

    #[cfg(target_feature = "avx2")]
    fn vec_of(n: usize) -> Vec<i16> {
        std::vec![0; n]
    }
}
