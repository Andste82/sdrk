//! `deemphasis` `f32`, `dc_block` `C32` and `dc_block_real` `f32`: the first-order recursion
//! `y[n] = a y[n-1] + u[n]` as a block scan in `f64` (maintainer decision of 2026-10-29: these
//! three are tolerance kernels of the backend, see `docs/kernels.md`).
//!
//! - `dc_block`, `dc_block_real`: `a = r`, `u[n] = x[n] - x[n-1]` (exact in `f64`).
//! - `deemphasis`: `y += alpha (x - y)` is `y = (1 - alpha) y + alpha x`, so `a = 1 - alpha` and
//!   `u[n] = alpha x[n]` (exact in `f64`).
//!
//! A block holds [`K`] vectors of `f64`, that is `L = K W` lanes for `W` lanes per vector, and
//! `L / C` samples of `C` components (1 for the real kernels, 2 for `C32`, interleaved as in
//! memory). Within a block the outputs are a lower-triangular Toeplitz matrix times the inputs
//! `u` plus the powers of `a` times the carried output `c` of the block before:
//!
//! ```text
//! y[i] = sum_{j <= i} a^(i - j) u[j] + a^(i + 1) c
//! ```
//!
//! The matrix product runs column by column, one broadcast of `u[j]` and one multiply-add per
//! vector that holds rows at or below `j`; it does not depend on `c`. The only chain from one
//! block to the next is `c' = a^(L / C) c + d`, one multiply-add (and a select for the flush)
//! per block, with `d` the last row of the product. All of it is `f64`, so the result is the
//! `f64` recursion up to a few roundings of `f64` and the one rounding of the output to `f32`:
//! about 152 dB against the golden (the optimum) instead of the 119 to 145 dB of the `f32` loop
//! of the fallback, whose error the recursion amplifies. On wasm32 `dc_block` `C32` runs the
//! fallback (more instructions there, the measure of that target).
//!
//! The carried value is flushed to 0 below `1e-30`, as the fallback flushes its stored output,
//! but per block (the outputs are not flushed). The stored state is the last output rounded to
//! `f32` and flushed, as in the fallback; between calls the recursion therefore restarts from
//! that `f32` value, which is why the bits depend on how the input is split (within the
//! regression guards of the golden cases). The remainder of a call that fills no block runs the
//! same `f64` recursion sample by sample.
//!
//! Non-finite values: a matrix product spreads a NaN or infinity of `u[j]` to the earlier rows of
//! its block (`0 * NaN`). A call whose input or state holds one is therefore given to the
//! fallback as a whole, which keeps its rule (a NaN or infinity poisons the state until
//! `reset()`). The test costs one subtraction and one addition per vector of input
//! (`x - x` is 0 for every finite `x`).

use fearless_simd::{Simd as SimdToken, f64x2, prelude::*};
use sdrk_core::kernels::demod::{
    DcBlock, DcBlockReal, DcBlockRealState, DcBlockState, Deemphasis, DeemphasisState,
};
use sdrk_core::num::flush_denormal;
use sdrk_core::{C32, Fallback, Result};

use crate::Simd;
use crate::level::dispatch;
use crate::util::check_len;

/// Vectors of `f64` per block: the chain of one multiply-add then serves `2 W` lanes, and the
/// matrix takes `1.5` multiply-adds per lane (`K = 1`: one, but the chain bounds; `K = 4`: 2.5).
const K: usize = 2;
/// The most `f64` lanes per vector (AVX-512).
const MAX_W: usize = 8;
/// The most lanes per block.
const MAX_L: usize = K * MAX_W;
/// The carried output is set to 0 below this magnitude (the threshold of `flush_denormal`).
const FLUSH: f64 = 1e-30;
/// Below this pole the powers of `a` in a block (up to `a^16`) times a carry near the flush
/// threshold come close to the subnormal range of `f64`; such a recursion forgets its past
/// within one sample anyway and runs in the fallback.
const MIN_POLE: f64 = 1e-8;

/// The constants of one call: the padded powers of `a` for the columns of the matrix, the powers
/// for the carry and the step of the carry from one block to the next.
struct Plan {
    /// `t[L + d] = a^(d / C)` for `0 <= d < L`, 0 below `L`: the column `j` of vector `v` is
    /// `t[L + v W - C j ..][..W]`.
    t: [f64; 2 * MAX_L],
    /// `p[m] = a^(m / C + 1)`: the weight of the carry in lane `m`.
    p: [f64; MAX_L],
    /// `a^(L / C)`.
    step: f64,
}

impl Plan {
    #[inline(always)]
    fn new<const C: usize>(a: f64, lanes: usize) -> Self {
        let mut t = [0.0; 2 * MAX_L];
        let mut p = [0.0; MAX_L];
        let mut pw = 1.0;
        for s in 0..lanes / C {
            for c in 0..C {
                if let Some(v) = t.get_mut(lanes + C * s + c) {
                    *v = pw;
                }
                if let Some(v) = p.get_mut(C * s + c) {
                    *v = pw * a;
                }
            }
            pw *= a;
        }
        Plan { t, p, step: pw }
    }
}

/// `W` lanes of `src` from `at`.
#[inline(always)]
fn load<S: SimdToken>(simd: S, src: &[f64], at: usize) -> S::f64s {
    let w = S::f64s::LEN;
    S::f64s::from_slice(simd, src.get(at..at + w).unwrap_or(&[]))
}

/// One value per sample of `src` (`C` components from `C s`) broadcast to every sample of a
/// vector.
#[inline(always)]
fn broadcast<S: SimdToken, const C: usize>(simd: S, src: &[f64], s: usize) -> S::f64s {
    if C == 1 {
        S::f64s::splat(simd, src.get(s).copied().unwrap_or(0.0))
    } else {
        let pair = f64x2::from_slice(simd, src.get(2 * s..2 * s + 2).unwrap_or(&[]));
        S::f64s::block_splat(pair)
    }
}

/// The last sample of `v` broadcast to every sample.
#[inline(always)]
fn broadcast_last<S: SimdToken, const C: usize>(simd: S, v: S::f64s) -> S::f64s {
    let w = S::f64s::LEN;
    broadcast::<S, C>(simd, v.as_slice(), w / C - 1)
}

/// The matrix product of one block: the lanes `ub` (`L = 2 W` of them, read as broadcasts)
/// times the powers of `a`, without the carry; the two vectors of the block.
#[inline(always)]
fn products<S: SimdToken, const C: usize>(simd: S, plan: &Plan, ub: &[f64]) -> (S::f64s, S::f64s) {
    let w = S::f64s::LEN;
    let l = K * w;
    // The columns of the samples of the first vector add to both sums, the others to the
    // second only. For one component the second sum runs in two halves, so that no chain is
    // longer than `W` multiply-adds (`avx2` 1 to 4 % faster); for `C32` the halves have two
    // columns each and the one chain was faster on `znver3` (2.95 against 3.53 cycles).
    let b = broadcast::<S, C>(simd, ub, 0);
    let mut acc0 = load(simd, &plan.t, l) * b;
    let mut acc1 = load(simd, &plan.t, l + w) * b;
    for j in 1..w / C {
        let b = broadcast::<S, C>(simd, ub, j);
        acc0 = load(simd, &plan.t, l - C * j).mul_add(b, acc0);
        acc1 = load(simd, &plan.t, l + w - C * j).mul_add(b, acc1);
    }
    if C == 1 {
        let b = broadcast::<S, C>(simd, ub, w / C);
        let mut acc2 = load(simd, &plan.t, l) * b;
        for j in w / C + 1..l / C {
            let b = broadcast::<S, C>(simd, ub, j);
            acc2 = load(simd, &plan.t, l + w - C * j).mul_add(b, acc2);
        }
        (acc0, acc1 + acc2)
    } else {
        for j in w / C..l / C {
            let b = broadcast::<S, C>(simd, ub, j);
            acc1 = load(simd, &plan.t, l + w - C * j).mul_add(b, acc1);
        }
        (acc0, acc1)
    }
}

/// The outputs of one block: the products `acc` plus the powers of `a` times the carry `c`
/// (the last output of the block before, broadcast per component), into `out` (`L` floats).
#[inline(always)]
fn outputs<S: SimdToken>(
    simd: S,
    plan: &Plan,
    acc: (S::f64s, S::f64s),
    c: S::f64s,
    out: &mut [f32],
) {
    let w = S::f64s::LEN;
    let y0 = load(simd, &plan.p, 0).mul_add(c, acc.0);
    let y1 = load(simd, &plan.p, w).mul_add(c, acc.1);
    y0.narrow(y1).store_slice(out);
}

/// `step c + d`, or `d` where `c` is below the flush threshold (decided beside the
/// multiply-add, so the chain is the multiply-add and the select).
#[inline(always)]
fn advance<S: SimdToken>(simd: S, step: f64, c: S::f64s, d: S::f64s) -> S::f64s {
    let next = S::f64s::splat(simd, step).mul_add(c, d);
    c.abs().simd_lt(S::f64s::splat(simd, FLUSH)).select(d, next)
}

/// Two blocks (`ub` and `out` hold `2 L` lanes and floats) from the carry `c`; returns the
/// carry after them. The loop runs two blocks per iteration: measured faster at `avx2` than one
/// (`dc_block` `C32` `znver3` / `alderlake` 2.95 / 2.76 cycles per sample against 2.90 / 2.87,
/// the real kernels 1 to 4 % faster), and faster than a chain of one multiply-add per two blocks
/// (`a^(2 L / C) c + (a^(L / C) d0 + d1)`: 3.79 / 2.79), which only adds work.
#[inline(always)]
fn pair<S: SimdToken, const C: usize>(
    simd: S,
    plan: &Plan,
    ub: &[f64],
    c: S::f64s,
    out: &mut [f32],
) -> S::f64s {
    let l = K * S::f64s::LEN;
    let (u0, u1) = ub.split_at(l);
    let (o0, o1) = out.split_at_mut(l);
    let c1 = single::<S, C>(simd, plan, u0, c, o0);
    single::<S, C>(simd, plan, u1, c1, o1)
}

/// One block (`ub` and `out` hold `L` lanes and floats) from the carry `c`; returns the next
/// carry.
#[inline(always)]
fn single<S: SimdToken, const C: usize>(
    simd: S,
    plan: &Plan,
    ub: &[f64],
    c: S::f64s,
    out: &mut [f32],
) -> S::f64s {
    let a = products::<S, C>(simd, plan, ub);
    let d = broadcast_last::<S, C>(simd, a.1);
    outputs(simd, plan, a, c, out);
    advance(simd, plan.step, c, d)
}

/// The inputs `u` of one block (`L` floats `x`, the floats `p` one sample earlier for `DIFF`)
/// into `ub`; adds `x - x` to `bad` (0 for finite `x`).
#[inline(always)]
#[allow(
    clippy::eq_op,
    clippy::assign_op_pattern,
    reason = "`v - v` is the test for finite values (0, or NaN for a NaN or an infinity)"
)]
fn inputs<S: SimdToken, const DIFF: bool>(
    simd: S,
    gain: S::f64s,
    x: &[f32],
    p: &[f32],
    ub: &mut [f64],
    bad: &mut S::f32s,
) {
    let w = S::f64s::LEN;
    let v = S::f32s::from_slice(simd, x);
    *bad = *bad + (v - v);
    let (lo, hi) = v.widen();
    let (u0, u1) = if DIFF {
        let (plo, phi) = S::f32s::from_slice(simd, p).widen();
        (lo - plo, hi - phi)
    } else {
        (lo * gain, hi * gain)
    };
    let (ub0, ub1) = ub.split_at_mut(w);
    u0.store_slice(ub0);
    u1.store_slice(ub1);
}

/// The recursion over `x` (`C` components per sample, interleaved) into `out`, from the
/// previous input `xp` and the previous output `y`. `DIFF`: `u = x - x_prev` (`dc_block`),
/// else `u = g x` (`deemphasis`). Returns the last output in `f64`, or `None` where the input
/// holds a NaN or an infinity (then `out` holds garbage and the caller runs the fallback).
#[inline(always)]
fn scan<S: SimdToken, const C: usize, const DIFF: bool>(
    simd: S,
    a: f64,
    g: f64,
    xp: [f32; 2],
    y: [f64; 2],
    x: &[f32],
    out: &mut [f32],
) -> Option<[f64; 2]> {
    let l = K * S::f64s::LEN;
    let mut c = if C == 1 {
        S::f64s::splat(simd, y[0])
    } else {
        S::f64s::block_splat(f64x2::from_slice(simd, &y))
    };
    let whole = x.len() - x.len() % l;
    let (xv, xr) = x.split_at(whole);
    let (ov, or) = out.split_at_mut(whole);
    let zero = S::f32s::splat(simd, 0.0);
    let mut bad = zero;
    // The last `C` inputs before the remainder.
    let mut last = xp;
    if whole > 0 {
        let plan = Plan::new::<C>(a, l);
        c = blocks::<S, C, DIFF>(simd, &plan, g, c, xp, xv, ov, &mut bad);
        if DIFF {
            last.get_mut(..C)
                .unwrap_or(&mut [])
                .copy_from_slice(xv.get(whole - C..).unwrap_or(&[]));
        }
    }
    if !bad.simd_eq(zero).all_true() {
        return None;
    }
    let mut yl = [0.0f64; 2];
    yl.get_mut(..C)
        .unwrap_or(&mut [])
        .copy_from_slice(c.as_slice().get(..C).unwrap_or(&[]));
    remainder::<C, DIFF>(a, g, yl, last, xr, or)
}

/// The whole blocks: `xv` and `ov` hold a multiple of `L` floats, at least `L`. Returns the
/// carry after the last block.
#[inline(always)]
fn blocks<S: SimdToken, const C: usize, const DIFF: bool>(
    simd: S,
    plan: &Plan,
    g: f64,
    mut c: S::f64s,
    xp: [f32; 2],
    xv: &[f32],
    ov: &mut [f32],
    bad: &mut S::f32s,
) -> S::f64s {
    let l = K * S::f64s::LEN;
    let gain = S::f64s::splat(simd, g);
    let whole = xv.len();
    // The first block, with the inputs shifted by one sample from `xp`.
    let (x0, xrest) = xv.split_at(l);
    let (o0, orest) = ov.split_at_mut(l);
    let mut first = [0.0f32; MAX_L];
    if DIFF {
        let (head, rest) = first.split_at_mut(C);
        head.copy_from_slice(xp.get(..C).unwrap_or(&[]));
        rest.get_mut(..l - C)
            .unwrap_or(&mut [])
            .copy_from_slice(x0.get(..l - C).unwrap_or(&[]));
    }
    let mut ub = [0.0f64; 2 * MAX_L];
    let ub0 = ub.get_mut(..l).unwrap_or(&mut []);
    inputs::<S, DIFF>(simd, gain, x0, first.get(..l).unwrap_or(&[]), ub0, bad);
    c = single::<S, C>(simd, plan, ub.get(..l).unwrap_or(&[]), c, o0);
    // The other blocks in pairs; their shifted inputs start `C` floats before each block.
    let n = xrest.len();
    let pairs = n - n % (2 * l);
    let shifted = xv.get(l - C..whole - C).unwrap_or(&[]);
    let (xp2, xs1) = xrest.split_at(pairs);
    let (op2, os1) = orest.split_at_mut(pairs);
    let (sp2, ss1) = shifted.split_at(pairs);
    for ((x2, s2), o2) in xp2
        .chunks_exact(2 * l)
        .zip(sp2.chunks_exact(2 * l))
        .zip(op2.chunks_exact_mut(2 * l))
    {
        let (xa, xb) = x2.split_at(l);
        let (sa, sb) = s2.split_at(l);
        let (ua, ubb) = ub.split_at_mut(l);
        inputs::<S, DIFF>(simd, gain, xa, sa, ua, bad);
        inputs::<S, DIFF>(simd, gain, xb, sb, ubb.get_mut(..l).unwrap_or(&mut []), bad);
        c = pair::<S, C>(simd, plan, ub.get(..2 * l).unwrap_or(&[]), c, o2);
    }
    if !xs1.is_empty() {
        let ua = ub.get_mut(..l).unwrap_or(&mut []);
        inputs::<S, DIFF>(simd, gain, xs1, ss1, ua, bad);
        c = single::<S, C>(simd, plan, ub.get(..l).unwrap_or(&[]), c, os1);
    }
    c
}

/// The samples after the last whole block, one by one in the same arithmetic (a plain multiply
/// and add) from the output `y` and the input `prev` before them; `None` at a NaN or an
/// infinity.
#[inline(always)]
fn remainder<const C: usize, const DIFF: bool>(
    a: f64,
    g: f64,
    mut y: [f64; 2],
    mut prev: [f32; 2],
    x: &[f32],
    out: &mut [f32],
) -> Option<[f64; 2]> {
    for (xs, os) in x.chunks_exact(C).zip(out.chunks_exact_mut(C)) {
        for ((&xi, o), (yc, pc)) in xs
            .iter()
            .zip(os.iter_mut())
            .zip(y.iter_mut().zip(prev.iter_mut()))
        {
            if !xi.is_finite() {
                return None;
            }
            let u = if DIFF {
                f64::from(xi) - f64::from(*pc)
            } else {
                g * f64::from(xi)
            };
            *yc = if yc.abs() < FLUSH { u } else { a * *yc + u };
            *o = *yc as f32;
            *pc = xi;
        }
    }
    Some(y)
}

/// Whether the block scan takes the recursion with pole `a` and the state values `v`.
#[inline]
fn usable(a: f64, v: &[f32]) -> bool {
    a >= MIN_POLE && v.iter().all(|x| x.is_finite())
}

impl DcBlockReal<f32> for Simd {
    fn dc_block_real(
        &self,
        state: &mut DcBlockRealState<f32>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (r, xp, yp) = (f64::from(state.r()), state.x_prev(), state.y_prev());
        if usable(r, &[xp, yp]) {
            let y = [f64::from(yp), 0.0];
            if let Some([y, _]) =
                dispatch!(simd => scan::<_, 1, true>(simd, r, 1.0, [xp, 0.0], y, input, output))
            {
                let x = input.last().copied().unwrap_or(xp);
                state.set_prev(x, flush_denormal(y as f32));
                return Ok(());
            }
        }
        Fallback.dc_block_real(state, input, output)
    }
}

impl DcBlock<C32> for Simd {
    fn dc_block(
        &self,
        state: &mut DcBlockState<C32>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (r, xp, yp) = (f64::from(state.r()), state.x_prev(), state.y_prev());
        // On wasm32 the scan of two interleaved components takes more instructions than the
        // fallback (36.56 against 26.53 per sample at `simd128`), the measure of that target.
        if cfg!(target_family = "wasm") {
            return Fallback.dc_block(state, input, output);
        }
        if usable(r, &[xp.re, xp.im, yp.re, yp.im]) {
            let x: &[f32] = bytemuck::cast_slice(input);
            let o: &mut [f32] = bytemuck::cast_slice_mut(output);
            let y = [f64::from(yp.re), f64::from(yp.im)];
            if let Some([re, im]) =
                dispatch!(simd => scan::<_, 2, true>(simd, r, 1.0, [xp.re, xp.im], y, x, o))
            {
                let x = input.last().copied().unwrap_or(xp);
                let y = C32::new(flush_denormal(re as f32), flush_denormal(im as f32));
                state.set_prev(x, y);
                return Ok(());
            }
        }
        Fallback.dc_block(state, input, output)
    }
}

impl Deemphasis<f32> for Simd {
    fn deemphasis(
        &self,
        state: &mut DeemphasisState<f32>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let alpha = f64::from(state.alpha());
        let a = 1.0 - alpha;
        let last = state.last();
        // `alpha = 1` (no memory) has `a = 0`: the block scan would give `alpha x` exactly as
        // well, but the fallback is the shorter loop for it.
        if usable(a, &[last]) {
            let y = [f64::from(last), 0.0];
            if let Some([y, _]) =
                dispatch!(simd => scan::<_, 1, false>(simd, a, alpha, [0.0; 2], y, input, output))
            {
                state.set_last(flush_denormal(y as f32));
                return Ok(());
            }
        }
        Fallback.deemphasis(state, input, output)
    }
}

#[cfg(test)]
mod tests {
    //! The block scan against the `f64` recursion over every length up to several passes (all
    //! tails), split into calls of every size up to 9, with the flush, non-finite input and the
    //! poles that run the fallback.

    extern crate std;

    use std::vec;
    use std::vec::Vec;

    use sdrk_core::kernels::demod::{
        DcBlock, DcBlockReal, DcBlockRealState, DcBlockState, Deemphasis, DeemphasisState,
    };
    use sdrk_core::{C32, Fallback, Rate};

    use crate::Simd;

    /// A small linear congruential generator in `[-0.5, 0.5)` plus `dc`.
    fn noise(n: usize, seed: u32, dc: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / 16_777_216.0 - 0.5 + dc
            })
            .collect()
    }

    /// `y = a y + u` in `f64` from the state `(xp, y)`, `u = x - xp` or `g x`, per component.
    fn model(a: f64, g: Option<f64>, x: &[f32], stride: usize, xp: &[f32], y: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; x.len()];
        for c in 0..stride {
            let (mut p, mut v) = (f64::from(xp[c]), y[c]);
            for i in (c..x.len()).step_by(stride) {
                let xi = f64::from(x[i]);
                let u = g.map_or(xi - p, |g| g * xi);
                v = a * v + u;
                out[i] = v;
                p = xi;
            }
        }
        out
    }

    /// The SNR of `got` against `want` in dB.
    fn snr(want: &[f64], got: &[f32]) -> f64 {
        let (mut s, mut e) = (0.0, 0.0);
        for (w, g) in want.iter().zip(got) {
            s += w * w;
            e += (w - f64::from(*g)).powi(2);
        }
        if e == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (s / e).log10()
        }
    }

    /// Splits `n` samples into calls of 1 to 9 samples.
    fn calls(n: usize, seed: u32) -> Vec<(usize, usize)> {
        let mut s = seed;
        let (mut out, mut at) = (Vec::new(), 0);
        while at < n {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let len = (1 + (s >> 8) as usize % 9).min(n - at);
            out.push((at, at + len));
            at += len;
        }
        out
    }

    #[test]
    fn dc_block_real_is_the_f64_recursion_for_every_length() {
        for r in [0.999f32, 0.9, 0.5] {
            for n in 0..300 {
                let x = noise(n, n as u32 + 1, 0.3);
                let mut st = DcBlockRealState::<f32>::new(r).unwrap();
                let mut y = vec![0.0f32; n];
                Simd.dc_block_real(&mut st, &x, &mut y).unwrap();
                let want = model(f64::from(r), None, &x, 1, &[0.0], &[0.0]);
                for (w, g) in want.iter().zip(&y) {
                    assert!(
                        (w - f64::from(*g)).abs() <= 1e-6 * w.abs().max(1e-3),
                        "r {r} n {n}"
                    );
                }
                if let Some(&last) = x.last() {
                    assert_eq!(st.x_prev(), last);
                    assert_eq!(st.y_prev(), y[n - 1]);
                }
            }
        }
    }

    #[test]
    fn dc_block_c32_is_the_f64_recursion_for_every_length_and_split() {
        // wasm32 runs the fallback for `C32` (see `DcBlock<C32> for Simd`).
        if cfg!(target_family = "wasm") {
            return;
        }
        for n in [0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 33, 255, 256, 257, 700] {
            let flat = noise(2 * n, 7 + n as u32, 0.2);
            let x: Vec<C32> = flat.chunks_exact(2).map(|p| C32::new(p[0], p[1])).collect();
            let want = model(
                f64::from(0.999f32),
                None,
                &flat,
                2,
                &[0.0, 0.0],
                &[0.0, 0.0],
            );
            let mut st = DcBlockState::<C32>::new(0.999).unwrap();
            let mut y = vec![C32::default(); n];
            Simd.dc_block(&mut st, &x, &mut y).unwrap();
            let got: Vec<f32> = y.iter().flat_map(|c| [c.re, c.im]).collect();
            // The optimum is the expected values rounded once to `f32`; on long inputs the
            // scan is within 1 dB of it (a short input is a few roundings, which spread).
            let rounded: Vec<f32> = want.iter().map(|&v| v as f32).collect();
            let (optimum, got_db) = (snr(&want, &rounded), snr(&want, &got));
            let floor = if n >= 255 { optimum - 1.0 } else { 135.0 };
            assert!(n == 0 || got_db >= floor, "n {n}: {got_db} of {optimum}");
            // In calls of 1 to 9 samples the state is rounded to `f32` between calls.
            let mut st = DcBlockState::<C32>::new(0.999).unwrap();
            let mut y = vec![C32::default(); n];
            for (a, b) in calls(n, n as u32) {
                Simd.dc_block(&mut st, &x[a..b], &mut y[a..b]).unwrap();
            }
            let got: Vec<f32> = y.iter().flat_map(|c| [c.re, c.im]).collect();
            assert!(
                n == 0 || snr(&want, &got) > 120.0,
                "n {n} split: {}",
                snr(&want, &got)
            );
        }
    }

    #[test]
    fn deemphasis_is_the_f64_recursion_and_keeps_its_state() {
        let rate = Rate::hz(48_000).unwrap();
        for tau in [50e-6, 75e-6, 1e-3] {
            let x = noise(1000, 3, 0.0);
            let st0 = DeemphasisState::<f32>::new(tau, rate).unwrap();
            let alpha = f64::from(st0.alpha());
            let want = model(1.0 - alpha, Some(alpha), &x, 1, &[0.0], &[0.0]);
            let (mut st, mut y) = (st0, vec![0.0f32; x.len()]);
            Simd.deemphasis(&mut st, &x, &mut y).unwrap();
            assert!(snr(&want, &y) > 145.0, "tau {tau}: {}", snr(&want, &y));
            assert_eq!(st.last(), y[x.len() - 1]);
            // The same input in two calls continues from the stored state.
            let (mut st, mut z) = (st0, vec![0.0f32; x.len()]);
            Simd.deemphasis(&mut st, &x[..333], &mut z[..333]).unwrap();
            Simd.deemphasis(&mut st, &x[333..], &mut z[333..]).unwrap();
            assert!(
                snr(&want, &z) > 140.0,
                "tau {tau} split: {}",
                snr(&want, &z)
            );
        }
        // `alpha = 1` (no memory): the output is the input.
        let mut st = DeemphasisState::<f32>::new(1e-9, rate).unwrap();
        assert_eq!(st.alpha(), 1.0);
        let x = noise(100, 5, 0.0);
        let mut y = vec![0.0f32; 100];
        Simd.deemphasis(&mut st, &x, &mut y).unwrap();
        assert_eq!(x, y);
    }

    #[test]
    fn non_finite_input_and_tiny_poles_give_the_fallback() {
        let mut x = noise(100, 9, 0.3);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for at in [0, 5, 47, 99] {
                x[at] = bad;
                let (mut sa, mut sb) = (
                    DcBlockRealState::<f32>::new(0.99).unwrap(),
                    DcBlockRealState::<f32>::new(0.99).unwrap(),
                );
                let (mut a, mut b) = (vec![0.0f32; 100], vec![0.0f32; 100]);
                Simd.dc_block_real(&mut sa, &x, &mut a).unwrap();
                Fallback.dc_block_real(&mut sb, &x, &mut b).unwrap();
                let bits = |v: &[f32]| {
                    v.iter()
                        .map(|f| if f.is_nan() { 0 } else { f.to_bits() })
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(&a), bits(&b), "{bad} at {at}");
                // The poisoned state stays poisoned (NaN) in the next call.
                let mut c = vec![0.0f32; 100];
                Simd.dc_block_real(&mut sa, &noise(100, 1, 0.0), &mut c)
                    .unwrap();
                assert!(c.iter().all(|v| v.is_nan()));
                x[at] = 0.25;
            }
        }
        // Outputs before a NaN keep their values (the call runs the fallback).
        let mut x = noise(64, 2, 0.0);
        x[40] = f32::NAN;
        let mut st = DeemphasisState::<f32>::new(75e-6, Rate::hz(48_000).unwrap()).unwrap();
        let mut y = vec![0.0f32; 64];
        Simd.deemphasis(&mut st, &x, &mut y).unwrap();
        assert!(y[..40].iter().all(|v| v.is_finite()) && y[40..].iter().all(|v| v.is_nan()));
        // A pole below 1e-8 runs the fallback: the same bits.
        let x = noise(64, 4, 0.0);
        let r = 1e-20f32;
        let (mut sa, mut sb) = (
            DcBlockRealState::<f32>::new(r).unwrap(),
            DcBlockRealState::<f32>::new(r).unwrap(),
        );
        let (mut a, mut b) = (vec![0.0f32; 64], vec![0.0f32; 64]);
        Simd.dc_block_real(&mut sa, &x, &mut a).unwrap();
        Fallback.dc_block_real(&mut sb, &x, &mut b).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn silence_decays_to_an_exact_zero_without_subnormals() {
        // After an impulse the carry falls below the flush threshold and becomes exactly 0.
        let mut x = vec![0.0f32; 200_000];
        x[0] = 1.0;
        let mut st = DcBlockRealState::<f32>::new(0.999).unwrap();
        let mut y = vec![0.0f32; x.len()];
        Simd.dc_block_real(&mut st, &x, &mut y).unwrap();
        assert!(y[150_000..].iter().all(|v| *v == 0.0));
        assert_eq!(st.y_prev(), 0.0);
        assert!(y.iter().all(|v| !v.is_subnormal()));
    }

    #[test]
    fn large_finite_input_stays_finite_where_the_output_fits() {
        let x = [
            1e30f32, -1e30, 1e30, 3e38, -3e38, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            10.0, 11.0,
        ];
        let mut st = DcBlockRealState::<f32>::new(0.5).unwrap();
        let mut y = [0.0f32; 17];
        Simd.dc_block_real(&mut st, &x, &mut y).unwrap();
        let want = model(0.5, None, &x, 1, &[0.0], &[0.0]);
        for (w, g) in want.iter().zip(&y) {
            if w.abs() < 3.4e38 {
                assert!(
                    (w - f64::from(*g)).abs() <= 1e-6 * w.abs().max(1e-3),
                    "{w} {g}"
                );
            } else {
                assert!(g.is_infinite());
            }
        }
    }
}
