//! Fallback implementation of `costas`.
//!
//! Bit-exact with `Reference`. The loop is a recursion over the symbols (the phase of every
//! symbol depends on the detector output of the previous one), so the kernel is not
//! vectorized; per symbol it costs one interpolated table pair, a complex rotation, the
//! detector, the loop filter and the lock metric (one division; 8PSK adds the octant
//! decision of the 8PSK point to the detector and the metric, and the three products of
//! `T4` to the metric, D659; 16-QAM and 64-QAM the grid decision, `libm::atan2f` and the
//! decision error of the metric, D1400, without a division). The loop keeps the state in locals and is specialized for the constellation, so
//! the detector and the metric have no branch on it. The time is the chain from one symbol's
//! phase to the next, which the forms of D1100 keep short with the same bits: `phase_step_f32`
//! takes `floor(v 2^32 + 0.5) as i64 as u32` from the bits of the `f32` with one arithmetic
//! shift (the `phase_step` of `fm_mod` in `f64` on `wasm32` and for large values), `detector4`
//! (QPSK) turns sign bits instead of multiplying, `integrate` takes the integrator inside its
//! limit after one well-predicted branch, and a non-finite detector value is handled out of
//! line. The QAM detector takes `libm::atan2f` inline for the quotients of a loop near lock,
//! with the bits of the call (`atan2f_inline`, D1418).

#![allow(clippy::many_single_char_names)]

use crate::Fallback;
use crate::error::{Error, Result};
use crate::fallback::mix::interp_f32_pair;
use crate::fallback::modulate::phase_step;
use crate::kernels::check_len;
use crate::kernels::costas::{
    BPSK, Costas, CostasState, LOCK_ALPHA, LOCK_OFF, LOCK_ON, PSK8, QAM16, QAM64, QPSK, derotate,
    detector, lock_metric, qam_of, qam_product,
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
/// chain from symbol to symbol instead of the whole flush (D1100).
#[inline]
fn flush_clamp(t: f32, limit: f32) -> f32 {
    let flushed = flush_denormal(t) == 0.0;
    let c = clamp_f32(t, limit);
    if flushed { 0.0 } else { c }
}

/// The integrator `clamp_f32(flush_denormal(t), limit)`: inside the limit and above the flush
/// threshold (the case of a running loop) it is `t` itself, so the common case takes `t` after
/// one well-predicted branch and the selects of [`flush_clamp`] leave the chain from symbol to
/// symbol (D1100).
#[inline]
pub(crate) fn integrate(t: f32, limit: f32) -> f32 {
    let a = libm::fabsf(t);
    if a <= limit && a >= 1e-30 {
        t
    } else {
        flush_clamp(t, limit)
    }
}

/// `sign(x) * y` of the order-4 detector without the multiplication on the chain: `1 y` is `y`
/// and `-1 y` is `-y`, the sign of `y` turned by the sign bit of `x`; `0 y` (zero and NaN `x`)
/// is computed as `0.0 * y` beside it; one comparison `|x| > 0` selects (D1100).
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
/// a general register) would lie on the chain from symbol to symbol (D1100).
#[inline]
fn detector4(y: C32) -> f32 {
    let e = sign_times(y.re, y.im) - sign_times(y.im, y.re);
    if e.is_finite() { e } else { not_finite() }
}

/// The QAM detector `atan2f(im(y conj d), re(y conj d))` of `detector`, a non-finite value
/// counting as 0 as there, with the 0 from the cold call of [`detector4`], so that the test
/// leaves the chain from symbol to symbol (D1418).
#[inline]
fn detector_qam(code: u8, y: C32) -> f32 {
    let (r, i) = qam_product(qam_of(code), y);
    let e = atan2f_inline(i, r);
    if e.is_finite() { e } else { not_finite() }
}

/// `libm::atan2f(y, x)` bit for bit, with the case of a loop near lock inline (D1418): for
/// `x > 0` finite and `t = |y| / x` in `[2^-12, 7/16)` msun's `atan2f` is `±atanf(t)` (the sign
/// of `y`; `|y / x|` and `|y| / x` are the same quotient), and its `atanf` takes no argument
/// reduction there: `t - t (s1 + s2)` with the polynomial of `atanf` in the same order. Every
/// other input (zeros, infinities, NaN, `x <= 0`, small and large quotients) takes the call.
#[inline]
pub(crate) fn atan2f_inline(y: f32, x: f32) -> f32 {
    // `aT[0..5]` of msun's `s_atanf.c`, the literals of `libm::atanf`.
    #[allow(
        clippy::excessive_precision,
        clippy::unreadable_literal,
        reason = "the literals of msun's s_atanf.c as libm writes them"
    )]
    const A_T: [f32; 5] = [
        3.3333328366e-01,
        -1.9999158382e-01,
        1.4253635705e-01,
        -1.0648017377e-01,
        6.1687607318e-02,
    ];
    let [a0, a1, a2, a3, a4] = A_T;
    let t = libm::fabsf(y) / x;
    let (xb, tb) = (x.to_bits(), t.to_bits());
    // `0 < x < inf` and `2^-12 <= t < 0.4375` by the bits (`0x39800000`, `0x3ee00000` as in
    // `atanf`); a NaN `x` or `t` fails both.
    if xb.wrapping_sub(1) < 0x7f7f_ffff && tb.wrapping_sub(0x3980_0000) < 0x3ee0_0000 - 0x3980_0000
    {
        let z = t * t;
        let w = z * z;
        let s1 = z * (a0 + w * (a2 + w * a4));
        let s2 = w * (a1 + w * a3);
        let a = t - t * (s1 + s2);
        if y.is_sign_negative() { -a } else { a }
    } else {
        libm::atan2f(y, x)
    }
}

/// The detector value of a non-finite `e` (D603), out of line.
#[cold]
#[inline(never)]
pub(crate) fn not_finite() -> f32 {
    0.0
}

/// The phase step `(floor(f64(v) 2^32 + 0.5) as i64) as u32` of the contract, from the bits of
/// the `f32` `v` with integer operations (D1100). `v = +-m 2^(exp - 150)` with the 24-bit
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
pub(crate) fn phase_step_f32(v: f32) -> u32 {
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
pub(crate) fn phase_step_f32(v: f32) -> u32 {
    phase_step(f64::from(v), TURN)
}

#[allow(
    clippy::inline_always,
    reason = "each constellation must be compiled as a loop of its own, without a call"
)]
#[inline(always)]
fn run<const CODE: u8>(s: &mut CostasState, input: &[C32], output: &mut [C32]) -> Result<()> {
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
        let e = if CODE == QPSK {
            detector4(y)
        } else if CODE == QAM16 || CODE == QAM64 {
            detector_qam(CODE, y)
        } else {
            detector(CODE, y)
        };
        freq = integrate(freq + ki * e, max);
        phase = phase.wrapping_add(phase_step_f32(freq + kp * e));
        if let Some(m) = lock_metric(CODE, y) {
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
        match s.constellation.code() {
            BPSK => run::<BPSK>(s, input, output),
            QPSK => run::<QPSK>(s, input, output),
            PSK8 => run::<PSK8>(s, input, output),
            code => run_qam(code, s, input, output),
        }
    }
}

/// The QAM loops, a function of their own, so that the PSK loops keep the code of M40.1.
#[inline(never)]
fn run_qam(code: u8, s: &mut CostasState, input: &[C32], output: &mut [C32]) -> Result<()> {
    if code == QAM16 {
        run::<QAM16>(s, input, output)
    } else {
        run::<QAM64>(s, input, output)
    }
}

#[cfg(test)]
mod tests {
    //! The forms of D1100 must equal the formulas of the contract bit for bit (NaN payloads
    //! aside, which are not part of it).

    use std::vec::Vec;

    use super::{
        TURN, atan2f_inline, clamp_f32, detector_qam, detector4, flush_clamp, integrate,
        phase_step_f32,
    };
    use crate::fallback::modulate::phase_step;
    use crate::kernels::costas::{QAM16, QAM64, QPSK, detector};
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
    fn inline_atan2f_equals_libm() {
        let mut s = specials();
        // The bounds of the inline case and their neighbours, as quotients and as operands.
        for b in [2.0f32.powi(-12), 0.4375, 1.0, 1.1875, 2.4375] {
            for k in 0..8u32 {
                s.extend([
                    f32::from_bits(b.to_bits() + k),
                    f32::from_bits(b.to_bits() - k),
                ]);
            }
        }
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..300 {
            s.push(f32::from_bits(next() as u32));
        }
        let check = |y: f32, x: f32| {
            let (got, want) = (atan2f_inline(y, x), libm::atan2f(y, x));
            assert!(
                got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                "{y:e} {x:e}: {got:e} against {want:e}"
            );
        };
        for &y in &s {
            for &x in &s {
                check(y, x);
            }
        }
        // Angles of a loop near lock: random quotients of every size around the inline range.
        for _ in 0..2_000_000 {
            let r = next();
            let x = f32::from_bits(0x3c00_0000 + (r as u32 % 0x0400_0000));
            let q = f32::from_bits(0x3900_0000 + ((r >> 32) as u32 % 0x0680_0000));
            let y = if r >> 63 == 0 { q * x } else { -(q * x) };
            check(y, x);
            check(y, -x);
        }
    }

    #[test]
    fn qam_detector_with_the_cold_zero_equals_the_contract() {
        let mut s = specials();
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..200 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            s.push(f32::from_bits(seed as u32));
        }
        for code in [QAM16, QAM64] {
            for &re in &s {
                for &im in &s {
                    let y = C32::new(re, im);
                    let (got, want) = (detector_qam(code, y), detector(code, y));
                    assert_eq!(got.to_bits(), want.to_bits(), "{code} {re:e} {im:e}");
                }
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
                let (got, want) = (detector4(y), detector(QPSK, y));
                assert!(
                    got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                    "{re:e} {im:e}: {got:e} against {want:e}"
                );
            }
        }
    }
}
