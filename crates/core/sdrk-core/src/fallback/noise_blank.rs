//! Fallback implementation of `noise_blank`.
//!
//! Both variants are bit-exact with `Reference`. The delay line is applied to the whole call
//! at once: the output is first the input delayed by `D` (the old delay line followed by the
//! input, two copies), and the delay line becomes the last `D` samples of that sequence. A
//! second pass runs the recursion of the average and the counter over the input and sets the
//! output samples to zero while the counter is positive; the counter at input `n` decides the
//! output at `n`, so the two passes give the samples of the per-sample rule.
//!
//! A call runs in sections of `SECTION` samples. One `memcpy` of a large call becomes
//! `rep movsb` in glibc, which the simulated cycles cost per byte (811 cycles per sample on
//! `znver3` at `avx2` for 4096 samples, D424); a section of 1 KiB stays below glibc's threshold
//! for it, and the delay line is copied once more per section. One pass that reads the delayed
//! sample in the loop needs no copy, but costs `neon` 3 cycles per sample more (`cortex-a72`
//! 20.12 against 17.14); copying in fixed chunks does not help, the compiler joins them into one
//! `memcpy`.
//!
//! `CQ15`: the impulse test `q 2^16 > k_q p30` is evaluated in `i64` instead of `i128`. Both
//! sides are exact there: `q <= 2^31`, so `q 2^16 <= 2^47`; `k_q <= 10^4 2^16 < 2^30`, and the
//! average stays at most `2^62 + 2^29` (the target is at most `2^31`, `alpha <= 2^30`), so
//! `p30 <= 2^31` and the product is below `2^61`.

#![allow(clippy::many_single_char_names)]

use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::check_len;
use crate::kernels::noise_blank::{NoiseBlank, NoiseBlankParts, NoiseBlankState};
use crate::num::{flush_denormal, round_shift};
use crate::types::{C32, CQ15, Sample};

/// The samples of a section (see the module docs).
const SECTION: usize = 128;

/// Writes the input delayed by `hist.len()` into `output` (the old delay line first) and
/// keeps the last `hist.len()` samples of (delay line ++ input) in `hist`. The lengths of
/// `input` and `output` are equal.
fn delay<T: Sample>(hist: &mut [T], input: &[T], output: &mut [T]) -> Result<()> {
    let internal = || Error::Internal("noise blank delay line");
    let (d, n) = (hist.len(), input.len());
    if d == 0 {
        output.copy_from_slice(input);
    } else if n >= d {
        let (head, tail) = output.split_at_mut_checked(d).ok_or_else(internal)?;
        let (passed, kept) = input.split_at_checked(n - d).ok_or_else(internal)?;
        head.copy_from_slice(hist);
        tail.copy_from_slice(passed);
        hist.copy_from_slice(kept);
    } else {
        output.copy_from_slice(hist.get(..n).ok_or_else(internal)?);
        hist.rotate_left(n);
        hist.get_mut(d - n..)
            .ok_or_else(internal)?
            .copy_from_slice(input);
    }
    Ok(())
}

/// `q.is_finite()` for a power `q = re^2 + im^2` (never negative): false for NaN and `+inf`,
/// with one compare (as in the `squelch` fallback).
#[inline]
fn finite(q: f32) -> bool {
    q <= f32::MAX
}

/// The counter of one sample: set on an impulse, then the output is zero while it is positive.
#[inline]
fn blank<T: Sample>(g: &mut NoiseBlankParts<'_, T>, count: &mut u64, impulse: bool, o: &mut T) {
    if impulse {
        *count = g.span;
    }
    if *count > 0 {
        *count -= 1;
        *g.blanked = g.blanked.saturating_add(1);
        *o = T::ZERO;
    }
}

impl NoiseBlank<C32> for Fallback {
    fn noise_blank<H>(
        &self,
        s: &mut NoiseBlankState<C32, H>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()>
    where
        H: AsRef<[C32]> + AsMut<[C32]>,
    {
        check_len(input.len(), output.len())?;
        let mut g = s.parts_mut();
        for (input, output) in input.chunks(SECTION).zip(output.chunks_mut(SECTION)) {
            delay(g.hist, input, output)?;
            let (alpha, k) = (g.alpha_f, g.k_f);
            let mut p = *g.p_f;
            let mut count = *g.count;
            for (o, &x) in output.iter_mut().zip(input) {
                let q = x.re * x.re + x.im * x.im;
                let kp = k * p;
                let impulse = !finite(q) || (p > 0.0 && q > kp);
                p = if impulse {
                    flush_denormal(p + alpha * (kp - p))
                } else if p > 0.0 {
                    flush_denormal(p + alpha * (q - p))
                } else {
                    flush_denormal(q)
                };
                blank(&mut g, &mut count, impulse, o);
            }
            *g.p_f = p;
            *g.count = count;
        }
        Ok(())
    }
}

impl NoiseBlank<CQ15> for Fallback {
    fn noise_blank<H>(
        &self,
        s: &mut NoiseBlankState<CQ15, H>,
        input: &[CQ15],
        output: &mut [CQ15],
    ) -> Result<()>
    where
        H: AsRef<[CQ15]> + AsMut<[CQ15]>,
    {
        check_len(input.len(), output.len())?;
        let mut g = s.parts_mut();
        for (input, output) in input.chunks(SECTION).zip(output.chunks_mut(SECTION)) {
            delay(g.hist, input, output)?;
            let (alpha, k) = (g.alpha_q, g.k_q);
            let mut p = *g.p_q;
            let mut count = *g.count;
            for (o, &x) in output.iter_mut().zip(input) {
                let (re, im) = (i64::from(x.re.0), i64::from(x.im.0));
                let q = re * re + im * im;
                let p30 = p >> 31;
                let kp = k * p30;
                let impulse = p30 > 0 && (q << 16) > kp;
                p = if impulse {
                    p + alpha * ((kp >> 16) - round_shift(p, 31))
                } else if p30 > 0 {
                    p + alpha * (q - round_shift(p, 31))
                } else {
                    q << 31
                };
                blank(&mut g, &mut count, impulse, o);
            }
            *g.p_q = p;
            *g.count = count;
        }
        Ok(())
    }
}
