//! Fallback implementation of `agc`.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::agc::{Agc, AgcParts, AgcState};
use crate::kernels::check_len;
use crate::num::{round_shift, round_sqrt, sat16, sqrt_f32};
use crate::types::{C32, CQ15, Q15};

/// Samples per iteration of the `C32` loop. The loop is a recursion on the gain, so nothing is
/// vectorized; four samples per iteration share the counter, the end test and the branch of the
/// loop (about 3 of 26 instructions per sample at the baseline level), and more than four make
/// the compiler spill registers at that level.
const UNROLL: usize = 4;

/// One sample of the `C32` loop: scales `x` by `gain`, stores it and returns the next gain.
#[allow(
    clippy::inline_always,
    reason = "the recursion on the gain must stay in the registers of the unrolled loop; the instruction counts of the benchmark depend on it"
)]
#[inline(always)]
fn step_c32(par: &AgcParts<'_>, gain: f32, x: C32, o: &mut C32) -> f32 {
    let y = C32::new(gain * x.re, gain * x.im);
    *o = y;
    let level = sqrt_f32(y.re * y.re + y.im * y.im);
    // Compare and select instead of `f32::max`/`min`: their NaN rules cost several instructions
    // per call (on wasm32 and at the baseline of x86-64 more than a compare and select, and on
    // NEON the sequence measured more than `fcmp` and `fcsel`), and the operands are never NaN
    // here (the level is compared first, and the gain is never NaN).
    if level > par.ref_f {
        // Attack. The reference is positive (the parameters are validated), so a level above
        // it is above zero; a NaN level fails both tests and leaves the gain alone.
        let div = if level < par.hi_f { level } else { par.hi_f };
        let next = gain * div / (div + par.ra_f * (level - par.ref_f));
        // Only the attack can fall below the minimum.
        if next > par.gmin_f { next } else { par.gmin_f }
    } else if level > 0.0 {
        // Release (a level of exactly the reference gives a factor of one). A zero level
        // leaves the gain alone.
        let div = if level > par.lo_f { level } else { par.lo_f };
        let next = gain * (div + par.rr_f * (par.ref_f - level)) / div;
        // Only the release can exceed the maximum.
        if next < par.gmax_f { next } else { par.gmax_f }
    } else {
        gain
    }
}

impl Agc<C32> for Fallback {
    fn agc(&self, state: &mut AgcState<C32>, input: &[C32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let par = state.parts_mut();
        let mut gain = *par.g_f;
        let mut xs = input.chunks_exact(UNROLL);
        let mut os = output.chunks_exact_mut(UNROLL);
        for (x, o) in (&mut xs).zip(&mut os) {
            // The chunks have exactly `UNROLL` samples, so the compiler unrolls this loop.
            for (&x, o) in x.iter().zip(o.iter_mut()) {
                gain = step_c32(&par, gain, x, o);
            }
        }
        for (&x, o) in xs.remainder().iter().zip(os.into_remainder()) {
            gain = step_c32(&par, gain, x, o);
        }
        *par.g_f = gain;
        *par.g_d = f64::from(gain);
        Ok(())
    }
}

impl Agc<CQ15> for Fallback {
    fn agc(&self, state: &mut AgcState<CQ15>, input: &[CQ15], output: &mut [CQ15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let par = state.parts_mut();
        let mut gain = *par.g_q;
        for (o, &x) in output.iter_mut().zip(input) {
            let y = CQ15 {
                re: Q15(sat16(round_shift(i64::from(x.re.0) * gain, 32))),
                im: Q15(sat16(round_shift(i64::from(x.im.0) * gain, 32))),
            };
            *o = y;
            // An input of exactly zero carries no information: hold the gain.
            if x.re.0 == 0 && x.im.0 == 0 {
                continue;
            }
            let (re, im) = (i64::from(y.re.0), i64::from(y.im.0));
            let level = round_sqrt(re * re + im * im);
            let err = par.ref_q - level;
            // The same values as the `i128` formulas of `Reference` (which divide by
            // `2 * den` after adding `den`), with divisors that fit 64 bits: `floor(floor(a /
            // 2^48) / d) = floor(a / (d 2^48))`, and for odd `den` no tie can occur.
            let next = if err < 0 {
                // `d` is at most 1.5 * 32768; `a` is below 2^63, so `den` is below 2^64.
                let div = level.min(par.hi_q) as u64;
                let den = (div << 48) + (par.ra_q as u64) * err.unsigned_abs();
                let num = ((gain as u128 * u128::from(div)) << 48) + u128::from(den >> 1);
                // The quotient is at most `gain`.
                (num / u128::from(den)) as u64
            } else {
                let div = level.max(par.lo_q) as u64;
                let prod = gain as u128 * par.rr_q as u128 * err as u128;
                // `prod` is below 2^109, so the shifted value is below 2^62.
                let delta = (((prod + (u128::from(div) << 47)) >> 48) as u64) / div;
                gain as u64 + delta
            };
            // The clamp bounds `next` by the Q.32 limits, so the narrowing is exact.
            gain = (next as i64).clamp(par.gmin_q, par.gmax_q);
        }
        *par.g_q = gain;
        Ok(())
    }
}
