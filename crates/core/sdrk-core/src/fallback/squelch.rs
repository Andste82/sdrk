//! Fallback implementation of `squelch`.
//!
//! Both variants are bit-exact with `Reference`; only the grouping of exact operations
//! differs, so that the chain from one average to the next is as short as the contract
//! allows.
//!
//! `C32`: the chain is `q - p`, the product with `alpha` and the sum, fixed by the contract.
//! The flush of `flush_denormal` and the gate would add a compare and a select to it, and the
//! output a select or a branch per sample. Instead there is one loop per state of the gate:
//! the open loop copies the input, the closed loop writes zeros, and each leaves through a
//! branch that the processor predicts (no wait for the compare) at the first sample that may
//! change the gate, needs the flush or has a non-finite power; that sample is finished with
//! the rule of the contract and the matching loop resumes. The average is never negative
//! (`q >= 0`, `0 < alpha <= 0.5`, so `p + alpha (q - p) >= p / 2` after rounding, and `+0`
//! stays `+0`), so the flush changes a value only in `(0, 1e-30)`. The open loop leaves below
//! `max(close level, 1e-30)`: the close level may be anywhere down to `0` (a large hysteresis
//! below a low threshold), and every average that needs the flush is below that bound.
//!
//! `CQ15`: the loop carries `u = p + 2^30` instead of `p`, so `round_shift(p, 31)` is
//! `h = u >> 31` (`p` stays far below `2^63 - 2^30`: it converges to at most
//! `2^31 * 2^31`). With `u = h 2^31 + l`, `l = u & (2^31 - 1)`, the update
//! `u + alpha (q - h)` is `(l + alpha q) + (2^31 - alpha) h`: the chain is the shift, one
//! multiplication and the additions of `l` and `alpha q` (6 cycles on x86-64, against 9 for
//! the saturating `round_shift`, the subtraction, the multiplication and the sum; the
//! compiler adds the two terms one after the other, so `alpha q` alone is computed beside
//! the chain). The gate is a branch-free compare. Every term is below `2^63` and the sum is
//! the exact `i64` value of the contract.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::squelch::{Squelch, SquelchState};
use crate::num::flush_denormal;
use crate::types::{C32, CQ15, Sample};

/// Bit pattern of `1e-30_f32`, the smallest magnitude that `flush_denormal` keeps.
const FLUSH_BITS: u32 = 1e-30_f32.to_bits();

/// `2^30`: `u = p + 2^30` turns `round_shift(p, 31)` into `u >> 31`.
const HALF_Q31: i64 = 1 << 30;

/// The low 31 bits of `u`.
const LOW_Q31: i64 = (1 << 31) - 1;

impl Squelch<C32> for Fallback {
    fn squelch(
        &self,
        state: &mut SquelchState<C32>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let g = state.parts_mut();
        let (alpha, open_f, close_f) = (g.alpha_f, g.open_f, g.close_f);
        // The open loop leaves below this bound: it may close the gate or need the flush.
        let leave_f = close_f.max(f32::from_bits(FLUSH_BITS));
        let mut p = *g.p_f;
        let mut open = *g.open;
        let mut pairs = output.iter_mut().zip(input);
        loop {
            // One loop per state of the gate; each leaves at the first sample that may
            // change the gate or needs the flush, which the rule of the contract finishes.
            let mut exit = None;
            if open {
                for (o, &x) in pairs.by_ref() {
                    let power = x.re * x.re + x.im * x.im;
                    if !finite(power) {
                        exit = Some((o, x, false));
                        break;
                    }
                    p += alpha * (power - p);
                    // Every average the flush changes is below `1e-30`, every one that closes
                    // the gate below the close level.
                    if p < leave_f {
                        exit = Some((o, x, true));
                        break;
                    }
                    *o = x;
                }
            } else {
                for (o, &x) in pairs.by_ref() {
                    let power = x.re * x.re + x.im * x.im;
                    if !finite(power) {
                        exit = Some((o, x, false));
                        break;
                    }
                    p += alpha * (power - p);
                    if p >= open_f || needs_flush(p) {
                        exit = Some((o, x, true));
                        break;
                    }
                    *o = C32::ZERO;
                }
            }
            let Some((o, x, updated)) = exit else { break };
            if updated {
                p = flush_denormal(p);
                if p >= open_f {
                    open = true;
                } else if p < close_f {
                    open = false;
                }
            }
            *o = if open { x } else { C32::ZERO };
        }
        *g.p_f = p;
        *g.open = open;
        Ok(())
    }
}

/// `q.is_finite()` for a power `q = re^2 + im^2` (never negative, so `-inf` cannot occur):
/// false for NaN and `+inf`. One compare and one branch on x86-64, AArch64 and wasm32, where
/// `is_finite` moves the bits to an integer register on the first two; `q < inf` would be
/// rewritten by the compiler into an unordered-equal test that wasm32 spells with three
/// compares.
#[inline]
fn finite(q: f32) -> bool {
    q <= f32::MAX
}

/// True where `flush_denormal` changes an average `p >= +0`: in `(0, 1e-30)`, where the bits
/// minus one are below those of `1e-30` minus one (`+0` wraps to the largest value).
#[inline]
fn needs_flush(p: f32) -> bool {
    p.to_bits().wrapping_sub(1) < FLUSH_BITS - 1
}

impl Squelch<CQ15> for Fallback {
    fn squelch(
        &self,
        state: &mut SquelchState<CQ15>,
        input: &[CQ15],
        output: &mut [CQ15],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let g = state.parts_mut();
        let alpha = g.alpha_q;
        // `0 < alpha <= 2^30`, so the factor of `h` is positive.
        let keep = (1 << 31) - alpha;
        let (open_u, close_u) = (g.open_q + HALF_Q31, g.close_q + HALF_Q31);
        let mut u = *g.p_q + HALF_Q31;
        let mut open = *g.open;
        for (o, &x) in output.iter_mut().zip(input) {
            let (re, im) = (i64::from(x.re.0), i64::from(x.im.0));
            let power = re * re + im * im;
            let low = (u & LOW_Q31) + alpha * power;
            u = low + keep * (u >> 31);
            #[allow(
                clippy::needless_bitwise_bool,
                reason = "the lazy form compiles to branches per sample on x86-64"
            )]
            {
                open = (u >= open_u) | (open & (u >= close_u));
            }
            *o = if open { x } else { CQ15::ZERO };
        }
        *g.p_q = u - HALF_Q31;
        *g.open = open;
        Ok(())
    }
}
