//! Fallback implementation of `stereo_pll` and `stereo_matrix`.
//!
//! The arithmetic is that of the contract, bit for bit (see `Reference`); only the form
//! differs. The cosine of the doubled phase pair is not computed (only `sin(2 theta)` is
//! needed), and `floor(f 2^32 + 0.5) as i64 as u32` is the `phase_step` of `fm_mod`, which equals
//! that formula bit for bit without a library call to `floor`.
//!
//! `stereo_encode` computes only the sines of the oscillator of `oscillate` (the Taylor step of
//! `nco_sin_f32`, two table reads per harmonic; the cosines are not needed), in passes of
//! `ENC_GROUP` samples at the phases `phase + k inc`, and runs the phase as a local
//! accumulator.

#![allow(clippy::many_single_char_names)]

use crate::Fallback;
use crate::error::{Error, Result};
use crate::fallback::mix::{interp_f32, interp_f32_pair, interp_q15, nco_sin_f32};
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

/// One `C32` pair of the matrix: `d2 = d + d`, `l = s + d2`, `r = s - d2`.
#[inline]
fn matrix_c32(i: C32) -> C32 {
    let d2 = i.im + i.im;
    C32::new(i.re + d2, i.re - d2)
}

/// The `C32` matrix of x86-64 below AVX-512 in the one-shuffle form of the SIMD backend (the
/// maintainer's decisions D2952 and D3005, open items IO1-4 and IO2-16): per step two vectors of
/// pairs, `shufps` 0x88 / 0xDD take the `s` and `d` of the step into one vector each, then
/// `d + d`, `s + d2` and `s - d2` on all lanes, and `unpcklps` / `unpckhps` give the pairs back,
/// 128-bit (four pairs per step) below AVX2 and 256-bit (eight pairs per step, the shuffles and
/// unpacks within the 128-bit lanes invert each other) with AVX2. The values are those of
/// [`matrix_c32`], lane by lane in the same order of the operands, bit for bit (the tests
/// below). No portable form compiles to this shuffle count (IO1-4: LLVM canonicalizes every
/// one to the duplicating shuffles or its own deinterleave).
///
/// The only module of `sdrk-core` with `unsafe` code (D3005): on x86-64 every `core::arch`
/// intrinsic is an `unsafe` call outside a function with the same `#[target_feature]`, and a
/// trait method cannot carry one. Its soundness is given by `cfg(target_feature)`: the module
/// is compiled only where the build itself enables the instructions it uses (`sse2`, and `avx`
/// for the 256-bit form), so every CPU the binary may run on has them; the loads and stores
/// read and write whole arrays of the slices through their pointers.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(target_feature = "avx512f")
))]
#[allow(
    unsafe_code,
    reason = "x86 shuffle intrinsics under cfg(target_feature), the maintainer's decision D3005"
)]
mod matrix_x86 {
    #[cfg(not(target_feature = "avx2"))]
    use core::arch::x86_64::{
        __m128, _mm_add_ps, _mm_loadu_ps, _mm_shuffle_ps, _mm_storeu_ps, _mm_sub_ps,
        _mm_unpackhi_ps, _mm_unpacklo_ps,
    };
    #[cfg(target_feature = "avx2")]
    use core::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_loadu_ps, _mm256_shuffle_ps, _mm256_storeu_ps, _mm256_sub_ps,
        _mm256_unpackhi_ps, _mm256_unpacklo_ps,
    };

    use crate::types::C32;

    /// Pairs per vector: two in 128 bits, four in 256 bits.
    #[cfg(not(target_feature = "avx2"))]
    const LANES: usize = 2;
    #[cfg(target_feature = "avx2")]
    const LANES: usize = 4;

    /// Pairs per step: two vectors.
    pub(super) const STEP: usize = 2 * LANES;

    /// One step of [`STEP`] pairs, 128-bit.
    #[cfg(not(target_feature = "avx2"))]
    #[inline]
    fn step(i: &[[C32; LANES]; 2], o: &mut [[C32; LANES]; 2]) {
        let [i0, i1] = i;
        let [o0, o1] = o;
        // SAFETY: this module is compiled only with `target_feature = "sse2"` enabled for the
        // whole build (the `cfg` of the module), so the instructions exist on every CPU that the
        // binary runs on. `C32` is `#[repr(C)]` with two `f32`, so `i0` and `i1` are 4 floats
        // each, readable through their pointers; `_mm_loadu_ps` needs no alignment.
        let (a, b): (__m128, __m128) = unsafe {
            (
                _mm_loadu_ps(i0.as_ptr().cast::<f32>()),
                _mm_loadu_ps(i1.as_ptr().cast::<f32>()),
            )
        };
        // SAFETY: SSE2 is enabled for the whole build (the `cfg` of the module); the intrinsics
        // only compute on registers.
        let (lo, hi) = unsafe {
            let s = _mm_shuffle_ps::<0x88>(a, b);
            let d = _mm_shuffle_ps::<0xDD>(a, b);
            let d2 = _mm_add_ps(d, d);
            let l = _mm_add_ps(s, d2);
            let r = _mm_sub_ps(s, d2);
            (_mm_unpacklo_ps(l, r), _mm_unpackhi_ps(l, r))
        };
        // SAFETY: SSE2 is enabled for the whole build (the `cfg` of the module). `o0` and `o1`
        // are 4 floats each (`C32` is `#[repr(C)]` with two `f32`), writable through their
        // pointers as the exclusive borrows; `_mm_storeu_ps` needs no alignment.
        unsafe {
            _mm_storeu_ps(o0.as_mut_ptr().cast::<f32>(), lo);
            _mm_storeu_ps(o1.as_mut_ptr().cast::<f32>(), hi);
        }
    }

    /// One step of [`STEP`] pairs, 256-bit.
    #[cfg(target_feature = "avx2")]
    #[inline]
    fn step(i: &[[C32; LANES]; 2], o: &mut [[C32; LANES]; 2]) {
        let [i0, i1] = i;
        let [o0, o1] = o;
        // SAFETY: this module is compiled only with `target_feature = "avx2"` (which includes
        // AVX) enabled for the whole build here (the `cfg` of the function and the module), so
        // the instructions exist on every CPU that the binary runs on. `C32` is `#[repr(C)]` with
        // two `f32`, so `i0` and `i1` are 8 floats each, readable through their pointers;
        // `_mm256_loadu_ps` needs no alignment.
        let (a, b): (__m256, __m256) = unsafe {
            (
                _mm256_loadu_ps(i0.as_ptr().cast::<f32>()),
                _mm256_loadu_ps(i1.as_ptr().cast::<f32>()),
            )
        };
        // SAFETY: AVX is enabled for the whole build (the `cfg`); the intrinsics only compute on
        // registers. The shuffles and unpacks work within each 128-bit lane, so the unpacks put
        // every pair back where the shuffles took it from.
        let (lo, hi) = unsafe {
            let s = _mm256_shuffle_ps::<0x88>(a, b);
            let d = _mm256_shuffle_ps::<0xDD>(a, b);
            let d2 = _mm256_add_ps(d, d);
            let l = _mm256_add_ps(s, d2);
            let r = _mm256_sub_ps(s, d2);
            (_mm256_unpacklo_ps(l, r), _mm256_unpackhi_ps(l, r))
        };
        // SAFETY: AVX is enabled for the whole build (the `cfg`). `o0` and `o1` are 8 floats
        // each (`C32` is `#[repr(C)]` with two `f32`), writable through their pointers as the
        // exclusive borrows; `_mm256_storeu_ps` needs no alignment.
        unsafe {
            _mm256_storeu_ps(o0.as_mut_ptr().cast::<f32>(), lo);
            _mm256_storeu_ps(o1.as_mut_ptr().cast::<f32>(), hi);
        }
    }

    /// The whole steps of `input` into `output` (of the same length); the pairs done.
    #[inline]
    pub(super) fn matrix(input: &[C32], output: &mut [C32]) -> usize {
        let (ic, _) = input.as_chunks::<LANES>();
        let (oc, _) = output.as_chunks_mut::<LANES>();
        let (ic, _) = ic.as_chunks::<2>();
        let (oc, _) = oc.as_chunks_mut::<2>();
        let mut done = 0;
        for (o, i) in oc.iter_mut().zip(ic) {
            step(i, o);
            done += STEP;
        }
        done
    }
}

impl StereoMatrix<C32> for Fallback {
    fn stereo_matrix(&self, input: &[C32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "sse2",
            not(target_feature = "avx512f")
        ))]
        let (input, output) = {
            let done = matrix_x86::matrix(input, output);
            (
                input.get(done..).unwrap_or(&[]),
                output.get_mut(done..).unwrap_or(&mut []),
            )
        };
        for (o, i) in output.iter_mut().zip(input) {
            *o = matrix_c32(*i);
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

/// Samples per pass of `stereo_encode` and `stereo_encode_rds`: four samples of one pass share
/// the phase and the loop overhead, and the compiler interleaves their sines. CI run
/// 46068881482 (cycles per sample, reference / counter model): AArch64 14.80 / 10.79 against
/// 17.18 / 20.08 with one sample per pass (`cortex-a72` / `cortex-a76`), `sse2` 8.30 / 7.55
/// against 12.73 / 13.41 (`skylake` / `znver2`). With AVX2 the pass of four was slower (6.41 /
/// 5.92 against 4.48 / 5.88 on `znver3` / `alderlake`) and on wasm32 it took more instructions
/// (46.30 against 45.79 per sample), so x86-64 with AVX2 and wasm32 keep one sample per pass, as
/// do the embedded targets (32-bit Arm, RISC-V). Every sample has the bits of the contract at
/// its own phase.
const ENC_GROUP: usize = if cfg!(any(
    all(target_arch = "x86_64", not(target_feature = "avx2")),
    target_arch = "aarch64"
)) {
    4
} else {
    1
};

/// One multiplex sample of `stereo_encode` at the pilot phase `phase`; `None` never happens
/// (the indices are below 4096 and the table has 4097 entries).
#[allow(
    clippy::inline_always,
    reason = "the sample is the body of the loops of both encoders"
)]
#[inline(always)]
fn encode_sample(x: C32, main: f32, pilot: f32, phase: u32) -> Option<f32> {
    let s1 = nco_sin_f32(phase)?;
    let s2 = nco_sin_f32(phase.wrapping_mul(2))?;
    #[allow(
        clippy::manual_midpoint,
        reason = "the contract fixes the f32 sum, then the halving; `midpoint` rounds differently"
    )]
    let m = (x.re + x.im) * 0.5;
    let d = (x.re - x.im) * 0.5;
    Some((main * m + main * d * s2) + pilot * s1)
}

/// One output sample of `stereo_encode_rds`: the multiplex of [`encode_sample`] plus
/// `(level b) s3`, added last.
#[allow(
    clippy::inline_always,
    reason = "the sample is the body of the loop of the encoder"
)]
#[inline(always)]
fn encode_rds_sample(x: C32, b: f32, main: f32, pilot: f32, level: f32, phase: u32) -> Option<f32> {
    let mpx = encode_sample(x, main, pilot, phase)?;
    let s3 = nco_sin_f32(phase.wrapping_mul(3))?;
    Some(mpx + (level * b) * s3)
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
        let (groups, rest) = output.as_chunks_mut::<ENC_GROUP>();
        let (inputs, rest_in) = input.as_chunks::<ENC_GROUP>();
        for (g, xs) in groups.iter_mut().zip(inputs) {
            for (k, (o, &x)) in g.iter_mut().zip(xs).enumerate() {
                let p = phase.wrapping_add(inc.wrapping_mul(k as u32));
                *o = encode_sample(x, main, pilot, p).ok_or_else(table_error)?;
            }
            phase = phase.wrapping_add(inc.wrapping_mul(ENC_GROUP as u32));
        }
        for (o, &x) in rest.iter_mut().zip(rest_in) {
            *o = encode_sample(x, main, pilot, phase).ok_or_else(table_error)?;
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
        let (groups, rest) = output.as_chunks_mut::<ENC_GROUP>();
        let (inputs, rest_in) = input.as_chunks::<ENC_GROUP>();
        let (bits, rest_rds) = rds.as_chunks::<ENC_GROUP>();
        for ((g, xs), bs) in groups.iter_mut().zip(inputs).zip(bits) {
            for (k, ((o, &x), &b)) in g.iter_mut().zip(xs).zip(bs).enumerate() {
                let p = phase.wrapping_add(inc.wrapping_mul(k as u32));
                *o = encode_rds_sample(x, b, main, pilot, level, p).ok_or_else(table_error)?;
            }
            phase = phase.wrapping_add(inc.wrapping_mul(ENC_GROUP as u32));
        }
        for ((o, &x), &b) in rest.iter_mut().zip(rest_in).zip(rest_rds) {
            *o = encode_rds_sample(x, b, main, pilot, level, phase).ok_or_else(table_error)?;
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

    /// The `C32` matrix of the fallback gives the pairs of the plain form bit for bit (on x86-64
    /// below AVX-512 the intrinsic form of `matrix_x86`, D3005), for every length (whole steps
    /// and the rest) and every offset into the buffers, with signed zeros, infinities,
    /// denormals, the largest values and NaNs of both signs with payloads in either component.
    #[test]
    fn c32_matrix_equals_the_pairs() {
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
            -f32::NAN,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0xff80_0001),
            1e-45,
            -1e-45,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
        ];
        let mut input: std::vec::Vec<C32> = (0..300).map(|_| C32::new(next(), next())).collect();
        for (k, &v) in specials.iter().enumerate() {
            input[3 * k] = C32::new(v, next());
            input[3 * k + 1] = C32::new(next(), v);
            input[3 * k + 2] = C32::new(v, specials[(k + 5) % specials.len()]);
        }
        let bits = |v: &[C32]| -> std::vec::Vec<(u32, u32)> {
            v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
        };
        for offset in [0, 1, 3] {
            for n in [0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 64, 290, 297] {
                let x = &input[offset..offset + n];
                let want: std::vec::Vec<C32> = x.iter().map(|&i| matrix_c32(i)).collect();
                let mut got = std::vec![C32::default(); n + 1];
                Fallback.stereo_matrix(x, &mut got[1..]).unwrap();
                assert_eq!(bits(&got[1..]), bits(&want), "offset {offset} n {n}");
            }
        }
        // Every pair of a list of bit patterns, in steps and in the rest.
        let patterns: std::vec::Vec<f32> = specials
            .iter()
            .copied()
            .chain((0..=u32::MAX).step_by(65_537 * 251).map(f32::from_bits))
            .collect();
        let all: std::vec::Vec<C32> = patterns
            .iter()
            .flat_map(|&a| patterns.iter().map(move |&b| C32::new(a, b)))
            .collect();
        let want: std::vec::Vec<C32> = all.iter().map(|&i| matrix_c32(i)).collect();
        let mut got = std::vec![C32::default(); all.len()];
        Fallback.stereo_matrix(&all, &mut got).unwrap();
        assert_eq!(bits(&got), bits(&want));
    }

    /// The intrinsic form of x86-64 below AVX-512 runs whole steps only and leaves the rest of
    /// the call to the plain loop (D3005).
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "avx512f")
    ))]
    #[test]
    fn the_x86_matrix_runs_whole_steps() {
        let step = matrix_x86::STEP;
        assert_eq!(step, if cfg!(target_feature = "avx2") { 8 } else { 4 });
        for n in [0, 1, step - 1, step, step + 1, 3 * step + 2] {
            let x = std::vec![C32::new(1.0, 2.0); n];
            let mut out = std::vec![C32::new(7.0, 7.0); n];
            assert_eq!(matrix_x86::matrix(&x, &mut out), n / step * step, "n {n}");
            for (k, o) in out.iter().enumerate() {
                let want = if k < n / step * step {
                    C32::new(5.0, -3.0)
                } else {
                    C32::new(7.0, 7.0)
                };
                assert_eq!(*o, want, "n {n} k {k}");
            }
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
