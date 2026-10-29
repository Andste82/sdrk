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

/// The exact sum `sum x * h` of one run.
#[inline]
fn run_q15(x: &[Q15], taps: &[Q15]) -> i64 {
    x.iter().zip(taps).fold(0i64, |acc, (s, h)| {
        acc.wrapping_add(i64::from(i32::from(s.0) * i32::from(h.0)))
    })
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

impl DotReal<f32> for Fallback {
    fn dot_real(&self, x: &[f32], taps: &[f32]) -> Result<f32> {
        check_len(x.len(), taps.len())?;
        let mut acc = 0.0f32;
        for (&s, &h) in x.iter().zip(taps) {
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
        for n in [0usize, 1, 2, 3, 7, 8, 15, 16, 17, 31, 33, 64, 255, 1000] {
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
}
