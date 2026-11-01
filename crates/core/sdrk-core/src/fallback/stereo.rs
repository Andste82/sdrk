//! Fallback implementation of `stereo_pll` and `stereo_matrix`.
//!
//! The arithmetic is that of the contract, bit for bit (see `Reference`); only the form
//! differs. The cosine of the doubled phase pair is not computed (only `sin(2 theta)` is
//! needed), and `floor(f 2^32 + 0.5) as i64 as u32` is the `phase_step` of `fm_mod`, which equals
//! that formula bit for bit without a library call to `floor`.

#![allow(clippy::many_single_char_names)]

use crate::Fallback;
use crate::error::{Error, Result};
use crate::fallback::mix::{interp_f32, interp_f32_pair, interp_q15};
use crate::fallback::modulate::phase_step;
use crate::kernels::check_len;
use crate::kernels::stereo::{StereoMatrix, StereoPll, StereoPllState};
use crate::num::flush_denormal;
use crate::types::{C32, CQ15, Q15};

const QUARTER: u32 = 1 << 30;
const TURN: f64 = 4_294_967_296.0;

#[cold]
fn table_error() -> Error {
    Error::Internal("cosine table index")
}

fn check_lengths<T>(mpx: &[T], pilot: &[T], out: usize) -> Result<()> {
    check_len(mpx.len(), pilot.len())?;
    check_len(mpx.len(), out)
}

/// `v.clamp(-limit, limit)` for a limit that is not negative, without the check of the bounds
/// that `f32::clamp` repeats in every call (a NaN stays a NaN). Written as two selects that the
/// compiler turns into `min` and `max` instructions.
#[inline]
fn clamp_f32(v: f32, limit: f32) -> f32 {
    let v = if v > limit { limit } else { v };
    if v < -limit { -limit } else { v }
}

/// The scalars of the `f32` loop.
struct LoopF32 {
    kp: f32,
    ki: f32,
    imax: f32,
    beta: f32,
    on: f32,
    off: f32,
    step0: u32,
    enabled: bool,
    phase: u32,
    integ: f32,
    level: f32,
    stereo: bool,
}

/// How the loop learns whether the difference channel is on (`StereoPllState::set_enabled`):
/// `MODE_RUNTIME` tests the flag in every sample, the other two are loops that know it.
const MODE_RUNTIME: u8 = 0;
const MODE_ENABLED: u8 = 1;
const MODE_DISABLED: u8 = 2;

/// Whether the loop is specialized for the flag. It saves instructions per sample on every
/// target, but the simulated cycles of the x86-64 models below AVX-512 get worse by 2 to 10
/// percent with it (`sse2`, `sse4.2`, `avx2`; D144), so only the other targets (AVX-512,
/// NEON, `simd128`, the microcontrollers) use it.
const HOIST_ENABLED: bool = !cfg!(all(target_arch = "x86_64", not(target_feature = "avx512f")));

/// The loop over a run of samples; `CHECK` tells whether a pilot sample may be non-finite (the
/// run without a check is the same code minus the test, used when the whole block is finite),
/// `MODE` how the loop learns whether the difference channel is on (see `HOIST_ENABLED`).
/// Returns `false` when a table index is out of range.
#[allow(
    clippy::inline_always,
    reason = "both variants must be compiled as loops of their own, without a call"
)]
#[inline(always)]
fn run_f32<const CHECK: bool, const MODE: u8>(
    l: &mut LoopF32,
    mpx: &[f32],
    pilot: &[f32],
    output: &mut [C32],
) -> bool {
    let (kp, ki, imax, beta) = (l.kp, l.ki, l.imax, l.beta);
    let (on, off, step0) = (l.on, l.off, l.step0);
    let diff = match MODE {
        MODE_RUNTIME => l.enabled,
        MODE_ENABLED => true,
        _ => false,
    };
    let (mut phase, mut integ, mut level, mut stereo) = (l.phase, l.integ, l.level, l.stereo);
    let mut ok = true;
    for ((o, &x), &p) in output.iter_mut().zip(mpx).zip(pilot) {
        let (Some((c, s)), Some(s2)) = (
            interp_f32_pair(phase),
            interp_f32(phase.wrapping_mul(2).wrapping_sub(QUARTER)),
        ) else {
            ok = false;
            break;
        };
        let e = if !CHECK || p.is_finite() {
            level = flush_denormal(level + beta * (p * s - level));
            if level >= on {
                stereo = true;
            } else if level < off {
                stereo = false;
            }
            p * c
        } else {
            0.0
        };
        *o = C32::new(x, if stereo && diff { x * s2 } else { 0.0 });
        integ = clamp_f32(flush_denormal(integ + ki * e), imax);
        let f = kp * e + integ;
        let corr = phase_step(f64::from(f), TURN);
        phase = phase.wrapping_add(step0).wrapping_add(corr);
    }
    (l.phase, l.integ, l.level, l.stereo) = (phase, integ, level, stereo);
    ok
}

impl StereoPll<f32> for Fallback {
    fn stereo_pll(
        &self,
        state: &mut StereoPllState<f32>,
        mpx: &[f32],
        pilot: &[f32],
        output: &mut [C32],
    ) -> Result<()> {
        check_lengths(mpx, pilot, output.len())?;
        let g = state.parts_mut();
        let mut l = LoopF32 {
            kp: g.kp,
            ki: g.ki,
            imax: g.imax,
            beta: g.beta,
            on: g.on,
            off: g.off,
            step0: g.step0,
            enabled: g.enabled,
            phase: *g.phase,
            integ: *g.integ,
            level: *g.level,
            stereo: *g.stereo,
        };
        // One pass over the pilot decides whether the loop needs the test per sample: the
        // largest magnitude pattern is below the infinity pattern iff every sample is finite.
        let top = pilot
            .iter()
            .fold(0u32, |m, p| m.max(p.to_bits() & 0x7FFF_FFFF));
        let finite = top < 0x7F80_0000;
        let mode = match (HOIST_ENABLED, l.enabled) {
            (false, _) => MODE_RUNTIME,
            (true, true) => MODE_ENABLED,
            (true, false) => MODE_DISABLED,
        };
        let ok = match (finite, mode) {
            (true, MODE_RUNTIME) => run_f32::<false, MODE_RUNTIME>(&mut l, mpx, pilot, output),
            (true, MODE_ENABLED) => run_f32::<false, MODE_ENABLED>(&mut l, mpx, pilot, output),
            (true, _) => run_f32::<false, MODE_DISABLED>(&mut l, mpx, pilot, output),
            (false, MODE_RUNTIME) => run_f32::<true, MODE_RUNTIME>(&mut l, mpx, pilot, output),
            (false, MODE_ENABLED) => run_f32::<true, MODE_ENABLED>(&mut l, mpx, pilot, output),
            (false, _) => run_f32::<true, MODE_DISABLED>(&mut l, mpx, pilot, output),
        };
        *g.phase = l.phase;
        *g.integ = l.integ;
        *g.level = l.level;
        *g.stereo = l.stereo;
        if ok { Ok(()) } else { Err(table_error()) }
    }
}

impl StereoPll<Q15> for Fallback {
    fn stereo_pll(
        &self,
        state: &mut StereoPllState<Q15>,
        mpx: &[Q15],
        pilot: &[Q15],
        output: &mut [CQ15],
    ) -> Result<()> {
        check_lengths(mpx, pilot, output.len())?;
        let g = state.parts_mut();
        let (kp, ki, imax, beta) = (g.kp_q, g.ki_q, g.imax_q, g.beta_q31);
        let (on, off, step0, enabled) = (g.on_q61, g.off_q61, g.step0, g.enabled);
        let (mut phase, mut integ, mut level, mut stereo) =
            (*g.phase, *g.integ_q, *g.level_q61, *g.stereo);
        let mut failed = false;
        for ((o, &x), &p) in output.iter_mut().zip(mpx).zip(pilot) {
            let (Some(c), Some(s), Some(s2)) = (
                interp_q15(phase),
                interp_q15(phase.wrapping_sub(QUARTER)),
                interp_q15(phase.wrapping_mul(2).wrapping_sub(QUARTER)),
            ) else {
                failed = true;
                break;
            };
            let pv = i64::from(p.0);
            let e = pv * i64::from(c);
            let q = pv * i64::from(s);
            level += beta * (q - ((level + (1 << 30)) >> 31));
            if level >= on {
                stereo = true;
            } else if level < off {
                stereo = false;
            }
            // |x s2| <= 2^30, so the rounded product is exact in i32; the saturation only acts
            // for -32768 * -32768.
            let diff = ((i32::from(x.0) * s2 + (1 << 14)) >> 15).clamp(-32768, 32767) as i16;
            *o = CQ15 {
                re: x,
                im: Q15(if stereo && enabled { diff } else { 0 }),
            };
            integ = (integ + ki * e).max(-imax).min(imax);
            let f = kp * e + integ;
            phase = phase
                .wrapping_add(step0)
                .wrapping_add(((f + (1 << 31)) >> 32) as u32);
        }
        *g.phase = phase;
        *g.integ_q = integ;
        *g.level_q61 = level;
        *g.stereo = stereo;
        if failed { Err(table_error()) } else { Ok(()) }
    }
}

impl StereoMatrix<C32> for Fallback {
    fn stereo_matrix(&self, input: &[C32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (o, i) in output.iter_mut().zip(input) {
            let d2 = i.im + i.im;
            *o = C32::new(i.re + d2, i.re - d2);
        }
        Ok(())
    }
}

/// `sat16(s + 2 d)` and `sat16(s - 2 d)` of one pair.
///
/// The saturating chains `(s + d) + d` and `(s - d) - d` equal them: if the first step is in
/// range the second is the exact saturating result; if it clamps, `d` has the sign of the clamp
/// and the second step pushes the same way, beyond the limit that `s +- 2 d` is beyond. All 2^32
/// pairs agree (the sweep of the backend tests and
/// `the_saturating_chain_equals_the_clamp_of_the_exact_sum` below). The chain maps to one
/// saturating instruction per step where the targets have them (NEON, `simd128`, SSE4.1 and
/// above vectorize it better than the clamp of `i32` sums, D145); SSE2 without SSE4.1 keeps the
/// clamp, which it compiles to fewer instructions.
#[inline]
fn matrix_pair_q15(s: i16, d: i16) -> (i16, i16) {
    #[cfg(all(target_arch = "x86_64", not(target_feature = "sse4.1")))]
    {
        let (s, d2) = (i32::from(s), 2 * i32::from(d));
        // |s +- 2 d| < 2^17: exact in i32, the clamp is `sat16`.
        (
            (s + d2).clamp(-32768, 32767) as i16,
            (s - d2).clamp(-32768, 32767) as i16,
        )
    }
    #[cfg(not(all(target_arch = "x86_64", not(target_feature = "sse4.1"))))]
    {
        (
            s.saturating_add(d).saturating_add(d),
            s.saturating_sub(d).saturating_sub(d),
        )
    }
}

impl StereoMatrix<CQ15> for Fallback {
    fn stereo_matrix(&self, input: &[CQ15], output: &mut [CQ15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (o, i) in output.iter_mut().zip(input) {
            let (re, im) = matrix_pair_q15(i.re.0, i.im.0);
            *o = CQ15 {
                re: Q15(re),
                im: Q15(im),
            };
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Fallback;

    /// The saturating chains of the `CQ15` matrix equal the clamp of the exact `i32` sum for
    /// every `d` and every 97th `s` plus the corners (the exhaustive sweep of all 2^32 pairs is
    /// `crates/backends/sdrk-backend-simd/tests/stereo_matrix_sweep.rs`).
    #[test]
    fn the_saturating_chain_equals_the_clamp_of_the_exact_sum() {
        let corners = [i16::MIN, i16::MIN + 1, -1, 0, 1, i16::MAX - 1, i16::MAX];
        let ss = (i16::MIN..=i16::MAX).step_by(97).chain(corners);
        let mut input = [CQ15::default(); 1024];
        let mut out = [CQ15::default(); 1024];
        for s in ss {
            for block in 0..64_i32 {
                for (k, i) in input.iter_mut().enumerate() {
                    let d = block * 1024 + k as i32 - 32768;
                    *i = CQ15 {
                        re: Q15(s),
                        im: Q15(d as i16),
                    };
                }
                Fallback.stereo_matrix(&input, &mut out).unwrap();
                for (i, o) in input.iter().zip(&out) {
                    let (s, d) = (i32::from(i.re.0), 2 * i32::from(i.im.0));
                    assert_eq!(i32::from(o.re.0), (s + d).clamp(-32768, 32767));
                    assert_eq!(i32::from(o.im.0), (s - d).clamp(-32768, 32767));
                    // The chain itself, whichever form the target compiles.
                    let (a, b) = (i.re.0, i.im.0);
                    assert_eq!(
                        i32::from(a.saturating_add(b).saturating_add(b)),
                        i32::from(o.re.0)
                    );
                    assert_eq!(
                        i32::from(a.saturating_sub(b).saturating_sub(b)),
                        i32::from(o.im.0)
                    );
                }
            }
        }
    }
}
