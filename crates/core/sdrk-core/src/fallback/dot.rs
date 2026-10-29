//! Fallback implementations of `dot` and `dot_real`.
//!
//! The `f32` and `C32` sums are formed in the order of the contract (one accumulator per
//! component, ascending, no fused multiply-add), so they are a dependency chain of additions.
//!
//! The `Q15` and `CQ15` sums are exact integers, so their order is free. `Acc64::mac` adds with
//! saturation, which keeps the compiler from reordering (and vectorizing) the sum. Here the
//! products are summed with plain `i64` additions in runs of [`RUN`] samples, and the runs are
//! added to the accumulator with the saturation of `Acc64`: a product is at most `2^30` in
//! magnitude, so a run sum stays below `2^61` and never overflows, and the result equals the
//! sample-by-sample `Acc64` sum for every input whose partial sums stay inside `i64` (fewer
//! than `2^33` samples, so every slice that fits into memory).
//!
//! The run sums use `wrapping_add`, which is exact because they never overflow. A checked
//! `i64` addition (`+` with `overflow-checks = true`) is miscompiled for `thumbv7em` when
//! LLVM fuses it into `smlalbb`: the overflow test only looks at the low 32 bits and panics
//! once they pass `2^31`.
//!
//! `dot_real` `Q15` on x86-64 at SSE2 and from AVX2 sums in a form the compiler maps to
//! `pmaddwd` (16-bit products summed into 32-bit lanes) instead of products widened to 64 bits:
//! every sample is split into its high byte (signed, `x >> 8`) and its low byte (unsigned,
//! `x & 0xff`), `x = 256 h + l`; a product of a byte with a tap is below `2^23` in magnitude, so
//! the products of a block of [`SPLIT`] samples sum exactly in `i32`, and the block adds
//! `256 sum_h + sum_l` to the `i64` run sum. From AVX2 the two sums take one pass each (one sum
//! per loop, which the compiler turns into `pmaddwd` of adjacent pairs), at SSE2 one pass for
//! both; slices shorter than one block keep the widening form, which has no fixed cost per
//! block. The values are the same. SSE4.2 (`pmovsxwq`, `pmuldq`), `dot` `CQ15` (where the split
//! simulated slower on one core model of every x86 level) and the other targets keep the
//! widening form.
//!
//! The `f32` sum of `dot_real` on x86-64 from AVX2, on AArch64 and on wasm32 forms the products
//! of eight samples first (independent, so the compiler computes them in one vector) and adds
//! them to the chain in ascending order: the same roundings in the same order, fewer
//! instructions.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::dot::{Dot, DotReal};
use crate::num::Acc64;
use crate::types::{C32, CQ15, Q15};

/// Samples per run of plain `i64` additions (see the module documentation): `2^31 * 2^30`
/// stays below `2^63`, and the run length fits into `usize` on 32-bit targets.
const RUN: usize = 1 << 31;

impl Dot<C32> for Fallback {
    fn dot(&self, x: &[C32], taps: &[f32]) -> Result<C32> {
        check_len(x.len(), taps.len())?;
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (s, &h) in x.iter().zip(taps) {
            re += s.re * h;
            im += s.im * h;
        }
        Ok(C32::new(re, im))
    }
}

/// Samples per block of the byte-split sums (see the module documentation): a product of a
/// byte (`-128..=255`) with a tap is at most `255 * 2^15` in magnitude, so a block sum stays
/// below `256 * 255 * 2^15 < 2^31`.
#[cfg(target_arch = "x86_64")]
const SPLIT: usize = 256;

/// The form of the run sums of `dot_real` `Q15` on x86-64 (see the module documentation).
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Q15Form {
    /// The products widened to `i64` (SSE4.2: `pmovsxwq`, `pmuldq`).
    Wide,
    /// The byte split, both sums in one pass (SSE2).
    SplitOnePass,
    /// The byte split, one pass per sum (from AVX2).
    SplitTwoPass,
}

/// The form of the build's level: per level the one that simulated fastest on both core models
/// (`cargo measure` at `avx2`, the cycle tools at `sse2`, `sse4.2` and `avx512`).
#[cfg(target_arch = "x86_64")]
const Q15_FORM: Q15Form = if cfg!(target_feature = "avx2") {
    Q15Form::SplitTwoPass
} else if cfg!(target_feature = "sse4.1") {
    Q15Form::Wide
} else {
    Q15Form::SplitOnePass
};

/// The exact sums `(sum re * h, sum im * h)` of one run.
#[inline]
fn run_cq15(x: &[CQ15], taps: &[Q15]) -> (i64, i64) {
    let (mut re, mut im) = (0i64, 0i64);
    for (s, h) in x.iter().zip(taps) {
        let h = i32::from(h.0);
        re = re.wrapping_add(i64::from(i32::from(s.re.0) * h));
        im = im.wrapping_add(i64::from(i32::from(s.im.0) * h));
    }
    (re, im)
}

/// The exact sum `sum x * h` of one run. Below one block the widening form is taken at every
/// level: the split pays a fixed cost per block (at `avx2` and n = 64, 4.88 against 4.22
/// instructions per sample).
#[inline]
fn run_q15(x: &[Q15], taps: &[Q15]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    if Q15_FORM != Q15Form::Wide && x.len() >= SPLIT {
        let block = |xs: &[Q15], ts: &[Q15]| {
            let (high, low) = if Q15_FORM == Q15Form::SplitTwoPass {
                (sum_high(xs, ts), sum_low(xs, ts))
            } else {
                sum_both(xs, ts)
            };
            256 * i64::from(high) + i64::from(low)
        };
        return x
            .chunks(SPLIT)
            .zip(taps.chunks(SPLIT))
            .fold(0i64, |acc, (xs, ts)| acc.wrapping_add(block(xs, ts)));
    }
    x.iter().zip(taps).fold(0i64, |acc, (s, h)| {
        acc.wrapping_add(i64::from(i32::from(s.0) * i32::from(h.0)))
    })
}

/// [`sum_high`] and [`sum_low`] in one pass.
#[cfg(target_arch = "x86_64")]
#[inline]
fn sum_both(x: &[Q15], taps: &[Q15]) -> (i32, i32) {
    let (mut high, mut low) = (0i32, 0i32);
    for (s, h) in x.iter().zip(taps) {
        let h = i32::from(h.0);
        high += i32::from(s.0 >> 8) * h;
        low += i32::from(s.0 & 0xff) * h;
    }
    (high, low)
}

/// `sum (x >> 8) * h` of a block of at most [`SPLIT`] samples (exact in `i32`).
#[cfg(target_arch = "x86_64")]
#[inline]
fn sum_high(x: &[Q15], taps: &[Q15]) -> i32 {
    let mut sum = 0i32;
    for (s, h) in x.iter().zip(taps) {
        sum += i32::from(s.0 >> 8) * i32::from(h.0);
    }
    sum
}

/// `sum (x & 0xff) * h` of a block of at most [`SPLIT`] samples (exact in `i32`).
#[cfg(target_arch = "x86_64")]
#[inline]
fn sum_low(x: &[Q15], taps: &[Q15]) -> i32 {
    let mut sum = 0i32;
    for (s, h) in x.iter().zip(taps) {
        sum += i32::from(s.0 & 0xff) * i32::from(h.0);
    }
    sum
}

impl Dot<CQ15> for Fallback {
    fn dot(&self, x: &[CQ15], taps: &[Q15]) -> Result<CQ15> {
        check_len(x.len(), taps.len())?;
        let (mut re, mut im) = (Acc64::default(), Acc64::default());
        for (xs, ts) in x.chunks(RUN).zip(taps.chunks(RUN)) {
            let (r, i) = run_cq15(xs, ts);
            re.0 = re.0.saturating_add(r);
            im.0 = im.0.saturating_add(i);
        }
        Ok(CQ15 {
            re: re.finish(),
            im: im.finish(),
        })
    }
}

/// Whether `dot_real` `f32` forms the products of eight samples before it adds them (see the
/// module documentation): where the vector of products saves instructions without slowing the
/// chain. At SSE2 and SSE4.2 the cycle tools gave 2 to 4 % more cycles, and the embedded targets
/// have no vectors.
const PRODUCTS_FIRST: bool = cfg!(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    target_arch = "aarch64",
    target_arch = "wasm32"
));

impl DotReal<f32> for Fallback {
    fn dot_real(&self, x: &[f32], taps: &[f32]) -> Result<f32> {
        check_len(x.len(), taps.len())?;
        let mut acc = 0.0f32;
        let (xc, xr, tc, tr) = if PRODUCTS_FIRST {
            let (xc, xr) = x.as_chunks::<8>();
            let (tc, tr) = taps.as_chunks::<8>();
            (xc, xr, tc, tr)
        } else {
            (&[][..], x, &[][..], taps)
        };
        for (xs, ts) in xc.iter().zip(tc) {
            let mut p = [0.0f32; 8];
            for ((p, &s), &h) in p.iter_mut().zip(xs).zip(ts) {
                *p = s * h;
            }
            for v in p {
                acc += v;
            }
        }
        for (&s, &h) in xr.iter().zip(tr) {
            acc += s * h;
        }
        Ok(acc)
    }
}

impl DotReal<Q15> for Fallback {
    fn dot_real(&self, x: &[Q15], taps: &[Q15]) -> Result<Q15> {
        check_len(x.len(), taps.len())?;
        let mut acc = Acc64::default();
        for (xs, ts) in x.chunks(RUN).zip(taps.chunks(RUN)) {
            acc.0 = acc.0.saturating_add(run_q15(xs, ts));
        }
        Ok(acc.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    /// The sample-by-sample `Acc64` sums of the contract.
    fn contract_cq15(x: &[CQ15], taps: &[Q15]) -> CQ15 {
        let (mut re, mut im) = (Acc64::default(), Acc64::default());
        for (s, &h) in x.iter().zip(taps) {
            re.mac(s.re, h);
            im.mac(s.im, h);
        }
        CQ15 {
            re: re.finish(),
            im: im.finish(),
        }
    }

    fn contract_q15(x: &[Q15], taps: &[Q15]) -> Q15 {
        let mut acc = Acc64::default();
        for (&s, &h) in x.iter().zip(taps) {
            acc.mac(s, h);
        }
        acc.finish()
    }

    #[test]
    fn the_integer_dots_equal_the_sample_by_sample_sums() {
        let mut state = 0x2545_f491_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as i16
        };
        let extremes = [i16::MIN, i16::MAX, -1, 0, 1, i16::MIN + 1];
        // Every tail of the vector loops, and the edges of the blocks of `SPLIT` samples.
        for n in [
            0usize, 1, 2, 3, 7, 8, 15, 16, 17, 31, 33, 64, 255, 256, 257, 511, 512, 513, 1000,
        ] {
            for mode in 0..4 {
                let pick = |i: usize, r: i16| match mode {
                    0 => r,
                    1 => i16::MIN,
                    2 => extremes[i % extremes.len()],
                    _ => {
                        if i.is_multiple_of(2) {
                            i16::MIN
                        } else {
                            i16::MAX
                        }
                    }
                };
                let x: Vec<CQ15> = (0..n)
                    .map(|i| CQ15 {
                        re: Q15(pick(i, next())),
                        im: Q15(pick(i + 1, next())),
                    })
                    .collect();
                let t: Vec<Q15> = (0..n).map(|i| Q15(pick(i + 2, next()))).collect();
                let r: Vec<Q15> = x.iter().map(|s| s.re).collect();
                assert_eq!(Fallback.dot(&x, &t).ok(), Some(contract_cq15(&x, &t)));
                assert_eq!(Fallback.dot_real(&r, &t).ok(), Some(contract_q15(&r, &t)));
            }
        }
    }

    #[test]
    fn the_run_sums_are_exact_at_full_scale() {
        // The largest block sums of the byte split: every product at its extreme, both signs.
        for (a, b) in [
            (i16::MIN, i16::MIN),
            (i16::MIN, i16::MAX),
            (i16::MAX, i16::MAX),
            (-1, i16::MIN),
            (255, i16::MIN),
            (255, i16::MAX),
            (-256, i16::MIN),
        ] {
            for n in [1usize, 255, 256, 257, 4099] {
                let x = std::vec![CQ15 { re: Q15(a), im: Q15(b) }; n];
                let t = std::vec![Q15(b); n];
                let r: Vec<Q15> = std::vec![Q15(a); n];
                let want = i64::from(a) * i64::from(b) * n as i64;
                assert_eq!(run_q15(&r, &t), want, "{a} {b} {n}");
                let want_im = i64::from(b) * i64::from(b) * n as i64;
                assert_eq!(run_cq15(&x, &t), (want, want_im), "{a} {b} {n}");
            }
        }
    }

    #[test]
    fn dot_real_f32_is_the_ascending_chain() {
        let mut state = 0x9e37_79b9_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            f32::from_bits(0x3f00_0000 | (state >> 9)) - 0.75
        };
        for n in [0usize, 1, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1023] {
            let x: Vec<f32> = (0..n).map(|_| next() * 1e3).collect();
            let t: Vec<f32> = (0..n).map(|_| next()).collect();
            let mut want = 0.0f32;
            for (&s, &h) in x.iter().zip(&t) {
                want += s * h;
            }
            let got = Fallback.dot_real(&x, &t).ok();
            assert_eq!(got.map(f32::to_bits), Some(want.to_bits()), "{n}");
        }
    }
}
