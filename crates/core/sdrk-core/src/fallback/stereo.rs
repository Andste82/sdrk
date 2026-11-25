//! Fallback implementation of `stereo_pll` and `stereo_matrix`.
//!
//! The arithmetic is that of the contract, bit for bit (see `Reference`); only the form
//! differs. The cosine of the doubled phase pair is not computed (only `sin(2 theta)` is
//! needed), and `floor(f 2^32 + 0.5) as i64 as u32` is the `phase_step` of `fm_mod`, which equals
//! that formula bit for bit without a library call to `floor`.
//!
//! `stereo_encode` reads the two sines with two table lookups per sample (the cosine of the
//! pilot phase is not needed) and runs the phase as a local accumulator.

#![allow(clippy::many_single_char_names)]

use crate::Fallback;
use crate::error::{Error, Result};
use crate::fallback::mix::{interp_f32, interp_f32_pair, interp_q15};
use crate::fallback::modulate::phase_step;
use crate::kernels::check_len;
use crate::kernels::stereo::{
    StereoEncode, StereoEncodeRds, StereoEncodeRdsState, StereoEncodeState, StereoMatrix,
    StereoPll, StereoPllState,
};
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

/// `clamp_f32(flush_denormal(v), limit)` with the test of the flush beside the clamp instead of
/// before it: where `v` is flushed, the clamp of `+0` is `+0` (the limit is not negative), and
/// elsewhere the flush leaves `v` as it is, so the value is the same for every `v` (NaN stays a
/// NaN). The clamp no longer waits for the flush, which shortens the loop-carried chain of the
/// phase by the comparison and the select of the flush.
#[inline]
fn flush_clamp(v: f32, limit: f32) -> f32 {
    let c = clamp_f32(v, limit);
    if libm::fabsf(v) < 1e-30 { 0.0 } else { c }
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
        integ = flush_clamp(integ + ki * e, imax);
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

/// Pairs per block of the `C32` matrix in [`matrix_blocks`], 0 for the plain loop: 8 in an
/// AVX2 build without AVX-512 (`znver3` / `alderlake` 0.63 / 0.86 to 0.44 / 0.64 cycles per
/// sample in the loop), 4 from SSE3 below AVX2 (`skylake` / `znver2` at `sse4.2` 1.52 / 1.46 to
/// 1.19 / 1.34), the plain loop at SSE2 (2.02 / 1.61 with blocks of 4), with AVX-512
/// (`sapphirerapids` 0.54 to 0.64 with blocks of 8) and on the other targets (`llvm-mca` of
/// the compiled loops).
const MATRIX_BLOCK: usize = if cfg!(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
)) {
    8
} else if cfg!(all(
    target_arch = "x86_64",
    target_feature = "sse3",
    not(target_feature = "avx2")
)) {
    4
} else {
    0
};

/// One `C32` pair of the matrix: `d2 = d + d`, `l = s + d2`, `r = s - d2`.
#[inline]
fn matrix_c32(i: C32) -> C32 {
    let d2 = i.im + i.im;
    C32::new(i.re + d2, i.re - d2)
}

/// The `C32` matrix in blocks of `P` pairs. The block is one set of vector lanes: the compiler
/// takes `s` and `2 d` with the duplicating shuffles (`movsldup`, `movshdup`), forms `l` and
/// `r` on all lanes and blends them, without the deinterleave and the interleave of the plain
/// loop. [`core::hint::black_box`] keeps it from vectorizing the loop once more across blocks,
/// which brings the shuffles back; it emits no instruction. The values are those of
/// [`matrix_c32`], pair by pair.
#[inline]
fn matrix_blocks<const P: usize>(input: &[C32], output: &mut [C32]) {
    let (ic, ir) = input.as_chunks::<P>();
    let (oc, or) = output.as_chunks_mut::<P>();
    for (o, i) in oc.iter_mut().zip(ic) {
        for (o, i) in o.iter_mut().zip(i) {
            *o = matrix_c32(*i);
        }
        core::hint::black_box(());
    }
    for (o, i) in or.iter_mut().zip(ir) {
        *o = matrix_c32(*i);
    }
}

impl StereoMatrix<C32> for Fallback {
    fn stereo_matrix(&self, input: &[C32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        match MATRIX_BLOCK {
            8 => matrix_blocks::<8>(input, output),
            4 => matrix_blocks::<4>(input, output),
            _ => {
                for (o, i) in output.iter_mut().zip(input) {
                    *o = matrix_c32(*i);
                }
            }
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

impl StereoEncode for Fallback {
    fn stereo_encode(
        &self,
        state: &mut StereoEncodeState,
        input: &[C32],
        output: &mut [f32],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (main, pilot, inc) = (state.main, state.pilot, state.nco.increment());
        let mut phase = state.nco.phase();
        for (o, x) in output.iter_mut().zip(input) {
            // The index is below 4096 and the table has 4097 entries: never `None`.
            let (Some(s1), Some(s2)) = (
                interp_f32(phase.wrapping_sub(QUARTER)),
                interp_f32(phase.wrapping_mul(2).wrapping_sub(QUARTER)),
            ) else {
                return Err(table_error());
            };
            #[allow(
                clippy::manual_midpoint,
                reason = "the contract fixes the f32 sum, then the halving; `midpoint` rounds differently"
            )]
            let m = (x.re + x.im) * 0.5;
            let d = (x.re - x.im) * 0.5;
            *o = (main * m + main * d * s2) + pilot * s1;
            phase = phase.wrapping_add(inc);
        }
        state.nco.set_phase(phase);
        Ok(())
    }
}

impl StereoEncodeRds for Fallback {
    fn stereo_encode_rds(
        &self,
        state: &mut StereoEncodeRdsState,
        input: &[C32],
        rds: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        check_len(input.len(), rds.len())?;
        check_len(input.len(), output.len())?;
        let enc = &mut state.enc;
        let (main, pilot, level, inc) = (enc.main, enc.pilot, state.rds, enc.nco.increment());
        let mut phase = enc.nco.phase();
        for ((o, x), &b) in output.iter_mut().zip(input).zip(rds) {
            // The index is below 4096 and the table has 4097 entries: never `None`.
            let (Some(s1), Some(s2), Some(s3)) = (
                interp_f32(phase.wrapping_sub(QUARTER)),
                interp_f32(phase.wrapping_mul(2).wrapping_sub(QUARTER)),
                interp_f32(phase.wrapping_mul(3).wrapping_sub(QUARTER)),
            ) else {
                return Err(table_error());
            };
            #[allow(
                clippy::manual_midpoint,
                reason = "the contract fixes the f32 sum, then the halving; `midpoint` rounds differently"
            )]
            let m = (x.re + x.im) * 0.5;
            let d = (x.re - x.im) * 0.5;
            *o = ((main * m + main * d * s2) + pilot * s1) + (level * b) * s3;
            phase = phase.wrapping_add(inc);
        }
        enc.nco.set_phase(phase);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Fallback;

    /// The block forms of the `C32` matrix give the pairs of the plain form bit for bit, for
    /// every length (whole blocks and the rest), with signed zeros, infinities and a NaN in one
    /// component.
    #[test]
    fn c32_matrix_blocks_equal_the_pairs() {
        let mut seed = 7u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / 8_388_608.0 - 1.0
        };
        let specials = [
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            1e-45,
            f32::MAX,
        ];
        let mut input: std::vec::Vec<C32> = (0..300).map(|_| C32::new(next(), next())).collect();
        for (k, &v) in specials.iter().enumerate() {
            input[3 * k] = C32::new(v, next());
            input[3 * k + 1] = C32::new(next(), v);
        }
        let bits = |v: &[C32]| -> std::vec::Vec<(u32, u32)> {
            v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
        };
        for n in [0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 64, 299, 300] {
            let x = &input[..n];
            let want: std::vec::Vec<C32> = x.iter().map(|&i| matrix_c32(i)).collect();
            let mut got4 = std::vec![C32::default(); n];
            let mut got8 = std::vec![C32::default(); n];
            let mut got = std::vec![C32::default(); n];
            matrix_blocks::<4>(x, &mut got4);
            matrix_blocks::<8>(x, &mut got8);
            Fallback.stereo_matrix(x, &mut got).unwrap();
            assert_eq!(bits(&got4), bits(&want), "n {n}");
            assert_eq!(bits(&got8), bits(&want), "n {n}");
            assert_eq!(bits(&got), bits(&want), "n {n}");
        }
    }

    /// `flush_clamp` is the clamp of the flushed value bit for bit: around the flush threshold
    /// and the limits, the signed zeros, the infinities and NaN, for limits from 0 to infinity
    /// and NaN, and every 4099th bit pattern.
    #[test]
    fn flush_clamp_equals_the_clamp_of_the_flushed_value() {
        let t = 1e-30f32;
        let specials = [
            0.0,
            -0.0,
            t,
            -t,
            f32::from_bits(t.to_bits() - 1),
            -f32::from_bits(t.to_bits() - 1),
            f32::from_bits(t.to_bits() + 1),
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1e-45,
            -1e-45,
            0.25,
            -0.25,
            1.0,
            -1.0,
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let limits = [
            0.0,
            1e-31,
            1e-30,
            1e-29,
            0.25,
            1.0,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
        ];
        let values = specials
            .into_iter()
            .chain((0..=u32::MAX).step_by(4099).map(f32::from_bits));
        for v in values {
            for &limit in &limits {
                let want = clamp_f32(flush_denormal(v), limit);
                let got = flush_clamp(v, limit);
                assert_eq!(got.to_bits(), want.to_bits(), "v {v:e} limit {limit:e}");
            }
        }
    }

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
