//! Fallback implementation of `costas`.
//!
//! Bit-exact with `Reference`. The loop is a recursion over the symbols (the phase of every
//! symbol depends on the detector output of the previous one), so the kernel is not
//! vectorized; per symbol it costs one interpolated table pair, a complex rotation, the
//! detector, the loop filter and the lock metric (one division; order 8 adds the octant
//! decision of the 8PSK point to the detector and the metric, and the three products of
//! `T4` to the metric, D659). The loop keeps the state in
//! locals and is specialized for the order, so the detector and the metric have no branch on
//! it. The time is the chain from one symbol's phase to the next, which the forms of D980 keep
//! short with the same bits: `phase_step_f32` takes `floor(v 2^32 + 0.5) as i64 as u32` from
//! the bits of the `f32` with one arithmetic shift (the `phase_step` of `fm_mod` in `f64` on
//! `wasm32` and for large values), `detector4` turns sign bits instead of multiplying,
//! `integrate` takes the integrator inside its limit after one well-predicted branch, and a
//! non-finite detector value is handled out of line.

#![allow(clippy::many_single_char_names)]

use crate::Fallback;
use crate::error::{Error, Result};
use crate::fallback::mix::interp_f32_pair;
use crate::fallback::modulate::phase_step;
use crate::kernels::check_len;
use crate::kernels::costas::{
    Costas, CostasState, LOCK_ALPHA, LOCK_OFF, LOCK_ON, derotate, detector, lock_metric,
};
use crate::num::flush_denormal;
use crate::types::C32;

const TURN: f64 = 4_294_967_296.0;

#[cold]
fn table_error() -> Error {
    Error::Internal("cosine table index")
}

/// `v.clamp(-limit, limit)` for a limit that is not negative, as two selects (a NaN cannot
/// occur: `freq` and `ki e` are finite).
#[inline]
fn clamp_f32(v: f32, limit: f32) -> f32 {
    let v = if v > limit { limit } else { v };
    if v < -limit { -limit } else { v }
}

/// `clamp_f32(flush_denormal(t), limit)` with the test of the flush beside the clamp instead
/// of before it: a flushed `t` is 0, and the clamp of 0 is 0 for every limit that is not
/// negative, so `flush ? 0 : clamp(t)` is the same value, one select after the clamp on the
/// chain from symbol to symbol instead of the whole flush (D980).
#[inline]
fn flush_clamp(t: f32, limit: f32) -> f32 {
    let flushed = flush_denormal(t) == 0.0;
    let c = clamp_f32(t, limit);
    if flushed { 0.0 } else { c }
}

/// The integrator `clamp_f32(flush_denormal(t), limit)`: inside the limit and above the flush
/// threshold (the case of a running loop) it is `t` itself, so the common case takes `t` after
/// one well-predicted branch and the selects of [`flush_clamp`] leave the chain from symbol to
/// symbol (D980).
#[inline]
fn integrate(t: f32, limit: f32) -> f32 {
    let a = libm::fabsf(t);
    if a <= limit && a >= 1e-30 {
        t
    } else {
        flush_clamp(t, limit)
    }
}

/// `sign(x) * y` of the order-4 detector without the multiplication on the chain: `1 y` is `y`
/// and `-1 y` is `-y`, the sign of `y` turned by the sign bit of `x`; `0 y` (zero and NaN `x`)
/// is computed as `0.0 * y` beside it; one comparison `|x| > 0` selects (D980).
#[inline]
fn sign_times(x: f32, y: f32) -> f32 {
    const SIGN: u32 = 0x8000_0000;
    let turned = f32::from_bits(y.to_bits() ^ (x.to_bits() & SIGN));
    let zero = 0.0 * y;
    if libm::fabsf(x) > 0.0 { turned } else { zero }
}

/// The order-4 detector `sign(y.re) y.im - sign(y.im) y.re` of `detector` with [`sign_times`];
/// a non-finite value counts as 0, as there. The 0 comes from a cold call, so the compiler
/// takes a well-predicted branch instead of a select whose test (on AArch64 a move of `e` to
/// a general register) would lie on the chain from symbol to symbol (D980).
#[inline]
fn detector4(y: C32) -> f32 {
    let e = sign_times(y.re, y.im) - sign_times(y.im, y.re);
    if e.is_finite() { e } else { not_finite() }
}

/// The detector value of a non-finite `e` (D603), out of line.
#[cold]
#[inline(never)]
fn not_finite() -> f32 {
    0.0
}

/// The phase step `(floor(f64(v) 2^32 + 0.5) as i64) as u32` of the contract, from the bits of
/// the `f32` `v` with integer operations (D980). `v = +-m 2^(exp - 150)` with the 24-bit
/// significand `m`, so `X = v 2^32 = S 2^-(157 - exp)` with `S = +-(m << 39)` (`|S| < 2^63`);
/// `X` and `X + 0.5` are exact in `f64` wherever the step can be other than 0 (`X + 0.5` loses
/// bits only for `X < 2^-30`). With the arithmetic shift, `q = S >> (156 - exp)` is
/// `floor(2 X)` for both signs, and `floor(X + 0.5) = floor((2 X + 1) / 2) = (q + 1) >> 1`. The
/// shift is capped at 63: for `exp < 94` (`|X| < 0.5`, zero and the subnormals with their
/// wrong leading bit included) `q` is 0 or -1 and the step 0, as the formula gives. Exponents
/// above 140 (`|v| >= 2^14`, where `X` reaches `2^46`), the infinities and NaN take
/// [`phase_step`].
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn phase_step_f32(v: f32) -> u32 {
    let bits = v.to_bits();
    let exp = (bits >> 23) & 0xFF;
    if exp > 140 {
        return phase_step(f64::from(v), TURN);
    }
    let m = (i64::from((bits & 0x7F_FFFF) | 0x80_0000)) << 39;
    let s = if bits >> 31 == 0 { m } else { -m };
    let q = s >> (156 - exp).min(63);
    ((q + 1) >> 1) as u32
}

/// The phase step of the contract on `wasm32`: there the cost is counted in instructions (no
/// core model, D68), and the `f64` form of [`phase_step`] takes fewer of them than the integer
/// form, whose shortened chain only a core with out-of-order execution would turn into time.
#[cfg(target_arch = "wasm32")]
#[inline]
fn phase_step_f32(v: f32) -> u32 {
    phase_step(f64::from(v), TURN)
}

#[allow(
    clippy::inline_always,
    reason = "each order must be compiled as a loop of its own, without a call"
)]
#[inline(always)]
fn run<const ORDER: u32>(s: &mut CostasState, input: &[C32], output: &mut [C32]) -> Result<()> {
    let (kp, ki, max) = (s.kp, s.ki, s.max);
    let (mut phase, mut freq, mut level, mut locked) = (s.phase, s.freq, s.level, s.locked);
    let mut ok = true;
    for (o, &x) in output.iter_mut().zip(input) {
        let Some((c, sn)) = interp_f32_pair(phase) else {
            ok = false;
            break;
        };
        let y = derotate(x, c, sn);
        *o = y;
        let e = if ORDER == 4 {
            detector4(y)
        } else {
            detector(ORDER, y)
        };
        freq = integrate(freq + ki * e, max);
        phase = phase.wrapping_add(phase_step_f32(freq + kp * e));
        if let Some(m) = lock_metric(ORDER, y) {
            level = flush_denormal(level + LOCK_ALPHA * (m - level));
            if level > LOCK_ON {
                locked = true;
            } else if level < LOCK_OFF {
                locked = false;
            }
        }
    }
    (s.phase, s.freq, s.level, s.locked) = (phase, freq, level, locked);
    if ok { Ok(()) } else { Err(table_error()) }
}

impl Costas for Fallback {
    fn costas(&self, s: &mut CostasState, input: &[C32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        match s.order {
            2 => run::<2>(s, input, output),
            4 => run::<4>(s, input, output),
            _ => run::<8>(s, input, output),
        }
    }
}

#[cfg(test)]
mod tests {
    //! The forms of D980 must equal the formulas of the contract bit for bit (NaN payloads
    //! aside, which are not part of it).

    use std::vec::Vec;

    use super::{TURN, clamp_f32, detector4, flush_clamp, integrate, phase_step_f32};
    use crate::fallback::modulate::phase_step;
    use crate::kernels::costas::detector;
    use crate::num::flush_denormal;
    use crate::types::C32;

    /// Special values and boundaries of `f32`.
    fn specials() -> Vec<f32> {
        let mut v = std::vec![
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x8000_0001),
            1e-30,
            -1e-30,
            0.999_999_9e-30,
            f32::EPSILON,
            0.5,
            -0.5,
            1.0,
            -1.0,
            0.05,
            -0.05,
            0.25,
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        // Ties of the phase step (X = v 2^32 = k + 0.5) and their neighbours.
        for k in [0u32, 1, 2, 3, 1000, 1 << 20, (1 << 23) - 1] {
            for s in [1.0f64, -1.0] {
                let f = ((s * (f64::from(k) + 0.5)) / TURN) as f32;
                v.extend([
                    f,
                    f32::from_bits(f.to_bits() + 1),
                    f32::from_bits(f.to_bits() - 1),
                ]);
            }
        }
        v
    }

    #[test]
    fn phase_step_from_the_bits_equals_the_f64_formula() {
        let mut values = specials();
        // Every exponent and both signs, with low, high, shifted and random significands.
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for exp in 0u32..=255 {
            for sign in [0u32, 1] {
                let top = (sign << 31) | (exp << 23);
                for low in 0u32..4096 {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    values.extend([
                        f32::from_bits(top | low),
                        f32::from_bits(top | (0x7F_FFFF - low)),
                        f32::from_bits(top | (low << 11)),
                        f32::from_bits(top | (seed as u32 & 0x7F_FFFF)),
                    ]);
                }
            }
        }
        for v in values {
            assert_eq!(
                phase_step_f32(v),
                phase_step(f64::from(v), TURN),
                "{v:e} ({:#010x})",
                v.to_bits()
            );
        }
    }

    #[test]
    fn flush_beside_the_clamp_equals_flush_then_clamp() {
        for t in specials().into_iter().filter(|t| t.is_finite()) {
            for limit in [0.0f32, 1e-31, 1e-30, 0.05, 0.25, t.abs(), f32::MAX] {
                let want = clamp_f32(flush_denormal(t), limit);
                assert_eq!(
                    flush_clamp(t, limit).to_bits(),
                    want.to_bits(),
                    "{t:e} {limit:e}"
                );
                assert_eq!(
                    integrate(t, limit).to_bits(),
                    want.to_bits(),
                    "{t:e} {limit:e}"
                );
            }
        }
    }

    #[test]
    fn order_4_detector_without_the_multiplication_equals_the_contract() {
        let mut s = specials();
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..200 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            s.push(f32::from_bits(seed as u32));
        }
        for &re in &s {
            for &im in &s {
                let y = C32::new(re, im);
                let (got, want) = (detector4(y), detector(4, y));
                assert!(
                    got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                    "{re:e} {im:e}: {got:e} against {want:e}"
                );
            }
        }
    }
}
