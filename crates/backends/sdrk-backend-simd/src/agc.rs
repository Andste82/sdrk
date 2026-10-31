//! `agc` `C32`.
//!
//! The gain is a recursion over the samples, so only what does not depend on it is vectorized:
//! the magnitudes `|x|` before the recursion and the products `y = g x` after it, per block of
//! [`BLOCK`] samples. The recursion is the one of `sdrk_core` ([`AgcState::<C32>::gains`]),
//! which takes the output level as `g |x|`; the square root is correctly rounded on every
//! level and the sums have no fused multiply-add, so the values are those of the fallback bit
//! for bit.
//!
//! `CQ15` stays with the fallback in the facade.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::agc::{Agc, AgcState};
use sdrk_core::{C32, Result};

use crate::Simd;
use crate::level::dispatch;
use crate::util::check_len;

/// Samples per block: the magnitudes and the gains of a block are two buffers of `f32` on the
/// stack (2 KiB). The fixed cost of a block (the calls, the set-up of the three loops) is
/// about 60 instructions, a quarter of an instruction per sample (128 samples measured 0.23
/// more per sample than 256 at NEON).
const BLOCK: usize = 256;

/// `sqrt(re*re + im*im)` of one vector of `2 l` interleaved floats into `l` magnitudes, the
/// operations of the fallback (see `mag`); returns the sums of the squares.
#[inline(always)]
fn mag_vector<S: SimdToken>(simd: S, i: &[f32], m: &mut [f32]) -> S::f32s {
    let l = S::f32s::LEN;
    let (a, b) = i.split_at(l);
    let a = S::f32s::from_slice(simd, a);
    let b = S::f32s::from_slice(simd, b);
    let (sa, sb) = (a * a, b * b);
    let sum = sa.unzip_low(sb) + sa.unzip_high(sb);
    sum.sqrt().store_slice(m);
    sum
}

/// `y = g x` for one vector of `l` gains and `2 l` interleaved floats.
#[inline(always)]
fn scale_vector<S: SimdToken>(simd: S, g: &[f32], i: &[f32], o: &mut [f32]) {
    let l = S::f32s::LEN;
    let g = S::f32s::from_slice(simd, g);
    let (a, b) = i.split_at(l);
    let (oa, ob) = o.split_at_mut(l);
    (S::f32s::from_slice(simd, a) * g.zip_low(g)).store_slice(oa);
    (S::f32s::from_slice(simd, b) * g.zip_high(g)).store_slice(ob);
}

/// The number of samples of a block of `n` that the vector loops take: whole steps of two
/// vectors. The rest (fewer than two vectors, only in the last block) is scalar.
#[inline(always)]
fn whole<S: SimdToken>(n: usize) -> usize {
    n - n % (2 * S::f32s::LEN)
}

/// The magnitudes of the samples `x` (interleaved) into `mag`; returns whether every one is
/// above zero (the sum of the squares is zero or NaN where it is not), so that the recursion
/// can leave out the test of the hold: one compare and one `and` per vector.
#[inline(always)]
fn magnitudes<S: SimdToken>(simd: S, x: &[f32], mag: &mut [f32]) -> bool {
    let l = S::f32s::LEN;
    // Two vectors per iteration share the counter and the branch of the loop; the buffers are
    // cut once to the whole steps, and the chunk iterators are zipped by value, which runs on
    // one counter.
    let whole = whole::<S>(mag.len());
    let (xv, xr) = x.split_at(2 * whole);
    let (mv, mr) = mag.split_at_mut(whole);
    let zero = S::f32s::splat(simd, 0.0);
    let mut positive = zero.simd_eq(zero);
    for (i, m) in xv.chunks_exact(4 * l).zip(mv.chunks_exact_mut(2 * l)) {
        let (i0, i1) = i.split_at(2 * l);
        let (m0, m1) = m.split_at_mut(l);
        let s0 = mag_vector(simd, i0, m0);
        let s1 = mag_vector(simd, i1, m1);
        positive = positive & s0.simd_gt(zero) & s1.simd_gt(zero);
    }
    let mut positive = positive.all_true();
    for (m, x) in mr.iter_mut().zip(xr.chunks_exact(2)) {
        if let [re, im] = *x {
            let sum = re * re + im * im;
            positive &= sum > 0.0;
            *m = tail_sqrt(sum);
        }
    }
    positive
}

/// The root of the scalar tail of [`magnitudes`], the same bits as `sdrk_core::num::sqrt_f32`.
/// Outside wasm32 it is `libm::sqrtf` (inline assembly on x86-64 and AArch64), which the
/// compiler does not vectorize: with the vectorizable `f32::sqrt` of `sqrt_f32` (feature `std`
/// of `sdrk-core`) it turns the tail of fewer than two vectors into a vector loop of its own,
/// whose registers cost spills in the recursion inlined into the same function (`avx2`,
/// `znver3` / `alderlake`, whole-loop: 14.22 / 16.24 instead of 13.18 / 14.47 cycles per
/// sample). On wasm32 `libm` is software, and `sqrt_f32` was vectorizable there before.
#[inline(always)]
fn tail_sqrt(x: f32) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        sdrk_core::num::sqrt_f32(x)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        libm::sqrtf(x)
    }
}

/// `out = gains * x` per sample (`x` and `out` interleaved), in the loop form of
/// [`magnitudes`].
#[inline(always)]
fn products<S: SimdToken>(simd: S, gains: &[f32], x: &[f32], out: &mut [f32]) {
    let l = S::f32s::LEN;
    let whole = whole::<S>(gains.len());
    let (xv, xr) = x.split_at(2 * whole);
    let (ov, or) = out.split_at_mut(2 * whole);
    let (gv, gr) = gains.split_at(whole);
    for ((i, o), g) in xv
        .chunks_exact(4 * l)
        .zip(ov.chunks_exact_mut(4 * l))
        .zip(gv.chunks_exact(2 * l))
    {
        let (i0, i1) = i.split_at(2 * l);
        let (o0, o1) = o.split_at_mut(2 * l);
        let (g0, g1) = g.split_at(l);
        scale_vector(simd, g0, i0, o0);
        scale_vector(simd, g1, i1, o1);
    }
    for ((o, x), &g) in or.chunks_exact_mut(2).zip(xr.chunks_exact(2)).zip(gr) {
        if let ([o0, o1], &[re, im]) = (o, x) {
            *o0 = g * re;
            *o1 = g * im;
        }
    }
}

/// The recursion with the minimum and maximum of the current level: on NEON the bare
/// `fminnm`/`fmaxnm` (see `neon`), elsewhere a compare and a select (shorter than `f32::min`
/// and `f32::max` there, whose NaN rule the operands do not need).
#[inline(always)]
fn gains_level<S: SimdToken>(
    simd: S,
    state: &mut AgcState<C32>,
    mag: &[f32],
    gains: &mut [f32],
    positive: bool,
) -> Result<()> {
    #[cfg(target_arch = "aarch64")]
    if simd.level().as_neon().is_some() {
        return neon::gains(state, mag, gains, positive);
    }
    let _ = simd;
    state.gains_with(
        mag,
        gains,
        positive,
        |a, b| if a < b { a } else { b },
        |a, b| if a > b { a } else { b },
    )
}

#[inline(always)]
fn agc_f32<S: SimdToken>(
    simd: S,
    state: &mut AgcState<C32>,
    input: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let mut mag = [0.0_f32; BLOCK];
    let mut gains = [0.0_f32; BLOCK];
    for (x, o) in input.chunks(2 * BLOCK).zip(out.chunks_mut(2 * BLOCK)) {
        let n = x.len() / 2;
        let (mag, _) = mag.split_at_mut(n);
        let (gains, _) = gains.split_at_mut(n);
        let positive = magnitudes(simd, x, mag);
        gains_level(simd, state, mag, gains, positive)?;
        products(simd, gains, x, o);
    }
    Ok(())
}

impl Agc<C32> for Simd {
    fn agc(&self, state: &mut AgcState<C32>, input: &[C32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let x: &[f32] = bytemuck::cast_slice(input);
        let y: &mut [f32] = bytemuck::cast_slice_mut(output);
        dispatch!(simd => agc_f32(simd, state, x, y))
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    //! The minimum and the maximum of the recursion as the single instructions `fminnm` and
    //! `fmaxnm`, written as `asm!`. `f32::min` and `f32::max` compile to the same
    //! instructions, but the compiler adds an `fminnm` of each operand with itself that it
    //! cannot prove to be no signalling NaN (any value from another basic block: the limits,
    //! the result of a branch). With the intrinsics (`vminnm_f32`) that step is gone, but the
    //! compiler then evaluates every branch of the step and selects (`fcsel`), about twice
    //! the instructions of the branch that is taken; an `asm!` block it does not move out of
    //! its branch. The operands are never NaN (see `AgcState::<C32>::gains_with`), so the
    //! values are those of every other minimum and maximum.
    #![allow(unsafe_code)]

    use core::arch::asm;

    use sdrk_core::kernels::agc::AgcState;
    use sdrk_core::{C32, Result};

    /// `fminnm`: the smaller of `a` and `b`.
    #[inline(always)]
    fn min(mut a: f32, b: f32) -> f32 {
        // SAFETY: `fminnm` reads two FP registers and writes one; it touches no memory, no
        // stack and no flags, and is part of the AArch64 baseline. Not `pure`, so that the
        // compiler keeps it in its branch.
        unsafe {
            asm!(
                "fminnm {a:s}, {a:s}, {b:s}",
                a = inout(vreg) a,
                b = in(vreg) b,
                options(nomem, nostack, preserves_flags)
            );
        }
        a
    }

    /// `fmaxnm`: the larger of `a` and `b`.
    #[inline(always)]
    fn max(mut a: f32, b: f32) -> f32 {
        // SAFETY: as in `min`.
        unsafe {
            asm!(
                "fmaxnm {a:s}, {a:s}, {b:s}",
                a = inout(vreg) a,
                b = in(vreg) b,
                options(nomem, nostack, preserves_flags)
            );
        }
        a
    }

    /// The recursion of `sdrk_core` with [`min`] and [`max`].
    #[inline(always)]
    pub(super) fn gains(
        state: &mut AgcState<C32>,
        mag: &[f32],
        gains: &mut [f32],
        positive: bool,
    ) -> Result<()> {
        state.gains_with(mag, gains, positive, min, max)
    }
}
