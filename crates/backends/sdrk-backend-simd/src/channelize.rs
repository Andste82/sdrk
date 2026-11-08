//! `channelize` `C32` (M45): the polyphase channelizer.
//!
//! The driver is the one of the fallback (`sdrk_core::kernels::channelize::channelize_with`,
//! the history and the hop boundaries), compiled at the dispatched level; per hop the branch
//! sums are formed with the vectors of the level and the `M`-point transform is the `C32` FFT of
//! this backend at the same level (`fft::forward_at`, no second dispatch per hop; the radix-2
//! transform of `channel_fft` for 4 and 8 channels, which the FFT kernel does not cover). The
//! class is *tolerance* (`docs/kernels.md`): the branch sums use fused multiply-adds where the
//! level has them, and the FFT is within its own contract, so the output agrees with the
//! fallback within the 100 dB of the contract, not bit for bit.
//!
//! **Branch sums.** Column `c` of the window is multiplied by the real tap `g[p M + c]` of every
//! row `p` and summed over the rows: a real tap times an interleaved complex sample. The state
//! stores every tap twice (`g0 g0 g1 g1 ...`, I45-1), so a vector of taps is loaded as it is and
//! matches the vector of interleaved samples; a row of taps has `2 M` floats like a row of the
//! window.
//!
//! - Generic code (SSE2, SSE4.2, NEON, SIMD128 and the scalar level): blocks of eight columns
//!   (four and two for 8 and 4 channels), four 128-bit accumulators of two samples.
//! - AVX2 and AVX-512 (`x86` below): blocks of sixteen columns, four 256-bit accumulators of four
//!   samples.
//! - NEON from 16 channels on: two consecutive hops at once ([`sums_pair_generic`]), so that one
//!   load of taps and most loads of the window serve both hops; the single-hop code of a level
//!   was bound by its loads. The pair form simulated slower at SSE2 on its reference model
//!   (`skylake`) and is not used on `x86`.
//!
//! The partial sums of a block stay in registers over all rows, as in the fallback; each block
//! of eight columns is stored at its place after the parity shift (column `c` to
//! `(c + s) mod M`), which never splits it because `M / 2` is a multiple of eight from 16
//! channels on.

use fearless_simd::{Simd as SimdToken, f32x4, prelude::*};
use sdrk_core::kernels::channelize::{Channelize, ChannelizeState, channel_fft, channelize_with};
use sdrk_core::kernels::fft::FftPlan;
use sdrk_core::timing::Progress;
use sdrk_core::{C32, Result};

use crate::Simd;
use crate::fft::forward_at;
use crate::level::dispatch;
use crate::util::internal;

/// The error of a violated length invariant of the branch sums (the state checks the lengths).
#[cold]
fn bad_len() -> sdrk_core::Error {
    internal("channelize branch sums")
}

/// The sums of the `W2 / 2` columns from `c` over all rows: `body(acc, taps, window)` adds the
/// products of one row (its `W2` stored taps and `W2` window floats) to the accumulators.
#[inline(always)]
fn column_sums<A, const W2: usize>(
    taps: &[f32],
    window: &[f32],
    m: usize,
    c: usize,
    mut acc: A,
    mut body: impl FnMut(&mut A, &[f32; W2], &[f32; W2]),
) -> Result<A> {
    if 2 * c + W2 > 2 * m {
        return Err(bad_len());
    }
    // Whole rows of `2 M` floats in both: the bounds of the columns are checked once above (the
    // compiler hoists the checks below out of the loop).
    for (t, w) in taps.chunks_exact(2 * m).zip(window.chunks_exact(2 * m)) {
        let t = t.get(2 * c..2 * c + W2).and_then(|t| t.first_chunk::<W2>());
        let w = w.get(2 * c..2 * c + W2).and_then(|w| w.first_chunk::<W2>());
        let (t, w) = t.zip(w).ok_or_else(bad_len)?;
        body(&mut acc, t, w);
    }
    Ok(acc)
}

/// Stores the `2 W` floats `sums` of the columns from `c` at column `(c + s) mod M` of `out`
/// (`c < M`, `s` is 0 or `M / 2`).
#[inline(always)]
fn store(out: &mut [f32], m: usize, c: usize, s: usize, sums: &[f32]) -> Result<()> {
    let col = c + s;
    let at = 2 * if col >= m { col - m } else { col };
    out.get_mut(at..at + sums.len())
        .ok_or_else(bad_len)?
        .copy_from_slice(sums);
    Ok(())
}

/// The branch sums with 128-bit vectors: blocks of eight columns (four accumulators of two
/// samples; four or two columns for 8 and 4 channels).
#[inline(always)]
fn sums_generic<S: SimdToken>(
    simd: S,
    taps: &[f32],
    window: &[f32],
    m: usize,
    s: usize,
    out: &mut [f32],
) -> Result<()> {
    let zero = f32x4::splat(simd, 0.0);
    let mut buf = [0.0f32; 16];
    if m >= 16 {
        for c in (0..m).step_by(8) {
            let acc = column_sums::<_, 16>(taps, window, m, c, [zero; 4], |acc, t, w| {
                for ((a, g), x) in acc.iter_mut().zip(t.chunks_exact(4)).zip(w.chunks_exact(4)) {
                    *a = f32x4::from_slice(simd, x).mul_add(f32x4::from_slice(simd, g), *a);
                }
            })?;
            for (d, a) in buf.chunks_exact_mut(4).zip(acc) {
                a.store_slice(d);
            }
            store(out, m, c, s, &buf)?;
        }
    } else if m == 8 {
        for c in [0, 4] {
            let acc = column_sums::<_, 8>(taps, window, m, c, [zero; 2], |acc, t, w| {
                for ((a, g), x) in acc.iter_mut().zip(t.chunks_exact(4)).zip(w.chunks_exact(4)) {
                    *a = f32x4::from_slice(simd, x).mul_add(f32x4::from_slice(simd, g), *a);
                }
            })?;
            for (d, a) in buf.chunks_exact_mut(4).zip(acc) {
                a.store_slice(d);
            }
            store(out, m, c, s, buf.get(..8).ok_or_else(bad_len)?)?;
        }
    } else {
        for c in [0, 2] {
            let acc = column_sums::<_, 4>(taps, window, m, c, zero, |acc, t, w| {
                let g = f32x4::from_slice(simd, t);
                *acc = f32x4::from_slice(simd, w).mul_add(g, *acc);
            })?;
            acc.store_slice(buf.get_mut(..4).ok_or_else(bad_len)?);
            store(out, m, c, s, buf.get(..4).ok_or_else(bad_len)?)?;
        }
    }
    Ok(())
}

/// Loads `V` vectors of `4 V` stored taps (two samples each).
#[inline(always)]
fn load<S: SimdToken, const V: usize>(simd: S, t: &[f32]) -> [f32x4<S>; V] {
    let mut g = [f32x4::splat(simd, 0.0); V];
    for (g, t) in g.iter_mut().zip(t.chunks_exact(4)) {
        *g = f32x4::from_slice(simd, t);
    }
    g
}

/// Stores the `V` accumulators of `2 V` columns from `c` at their place after the shift `s`.
#[inline(always)]
fn store_acc<S: SimdToken, const V: usize>(
    acc: [f32x4<S>; V],
    out: &mut [f32],
    m: usize,
    c: usize,
    s: usize,
) -> Result<()> {
    let mut buf = [0.0f32; 16];
    for (d, a) in buf.chunks_exact_mut(4).zip(acc) {
        a.store_slice(d);
    }
    store(out, m, c, s, buf.get(..4 * V).ok_or_else(bad_len)?)
}

/// The branch sums of two consecutive hops at once with 128-bit vectors, from 16 channels on:
/// the window holds the newest `M L + D` samples, hop A the first `M L`, hop B the last `M L`
/// (`D = M / 2` later). Read as half rows of `D` samples, column `c < D` of row `p` of hop A is
/// half row `2 p`, column `c + D` is half row `2 p + 1`, and hop B is shifted by one half row; so
/// per row and block of `2 V` columns `c .. c + 2 V` (with `c + D ..` beside it) one tap load
/// serves both hops and every window vector but one serves two products:
///
/// - `A[c]     += g[p M + c]     x[2 p]`
/// - `B[c]     += g[p M + c]     x[2 p + 1]`
/// - `A[c + D] += g[p M + c + D] x[2 p + 1]`
/// - `B[c + D] += g[p M + c + D] x[2 p + 2]`, and `x[2 p + 2]` is `x[2 p]` of the next row.
///
/// Each column is summed over the rows in ascending order with the same fused multiply-adds as
/// [`sums_generic`], so the values are those of two single hops. `out_a` and `out_b` are the
/// groups of the two steps; `s` is the shift of hop A, hop B has the other one.
#[inline(always)]
fn sums_pair_generic<S: SimdToken, const V: usize>(
    simd: S,
    taps: &[f32],
    window: &[f32],
    m: usize,
    s: usize,
    out_a: &mut [f32],
    out_b: &mut [f32],
) -> Result<()> {
    let w = 2 * V;
    if m < 16
        || !m.is_power_of_two()
        || (m / 2) % w != 0
        || window.len() != taps.len() + m
        || out_a.len() != 2 * m
        || out_b.len() != 2 * m
    {
        return Err(bad_len());
    }
    // `M` as a shift: the row counts of the chunks below need no division.
    let m = 1usize << m.trailing_zeros();
    let d = m / 2;
    let s_b = d - s;
    let zero = f32x4::splat(simd, 0.0);
    let (head, body) = window.split_at(m);
    for c in (0..d).step_by(w) {
        let mut a_lo = [zero; V];
        let mut a_hi = [zero; V];
        let mut b_lo = [zero; V];
        let mut b_hi = [zero; V];
        let mut xa = load::<S, V>(simd, head.get(2 * c..2 * c + 2 * w).ok_or_else(bad_len)?);
        for (t, x) in taps.chunks_exact(2 * m).zip(body.chunks_exact(2 * m)) {
            let t_lo = t.get(2 * c..2 * c + 2 * w).ok_or_else(bad_len)?;
            let t_hi = t.get(m + 2 * c..m + 2 * c + 2 * w).ok_or_else(bad_len)?;
            let x_mid = x.get(2 * c..2 * c + 2 * w).ok_or_else(bad_len)?;
            let x_next = x.get(m + 2 * c..m + 2 * c + 2 * w).ok_or_else(bad_len)?;
            let g = load::<S, V>(simd, t_lo);
            for ((a, g), x) in a_lo.iter_mut().zip(g).zip(xa) {
                *a = x.mul_add(g, *a);
            }
            for ((b, g), x) in b_lo.iter_mut().zip(g).zip(x_mid.chunks_exact(4)) {
                *b = f32x4::from_slice(simd, x).mul_add(g, *b);
            }
            let g = load::<S, V>(simd, t_hi);
            for ((a, g), x) in a_hi.iter_mut().zip(g).zip(x_mid.chunks_exact(4)) {
                *a = f32x4::from_slice(simd, x).mul_add(g, *a);
            }
            for (((b, g), x), xa) in b_hi
                .iter_mut()
                .zip(g)
                .zip(x_next.chunks_exact(4))
                .zip(xa.iter_mut())
            {
                *xa = f32x4::from_slice(simd, x);
                *b = xa.mul_add(g, *b);
            }
        }
        store_acc(a_lo, out_a, m, c, s)?;
        store_acc(a_hi, out_a, m, c + d, s)?;
        store_acc(b_lo, out_b, m, c, s_b)?;
        store_acc(b_hi, out_b, m, c + d, s_b)?;
    }
    Ok(())
}

/// The branch sums of one hop at the level of `simd` (the definition of
/// `sdrk_core::kernels::channelize::branch_sums`, with fused multiply-adds where the level has
/// them): the sum of column `c` goes to `group[(c + s) mod M]`, `s = M / 2` when `swap` is set.
#[inline(always)]
fn branch_sums<S: SimdToken>(
    simd: S,
    taps: &[f32],
    window: &[C32],
    swap: bool,
    group: &mut [C32],
) -> Result<()> {
    let m = group.len();
    if m < 4
        || !m.is_power_of_two()
        || taps.len() != 2 * window.len()
        || window.len() & (m - 1) != 0
    {
        return Err(bad_len());
    }
    let window: &[f32] = bytemuck::cast_slice(window);
    let out: &mut [f32] = bytemuck::cast_slice_mut(group);
    let s = if swap { m / 2 } else { 0 };
    #[cfg(target_arch = "x86_64")]
    if m >= 16
        && let Some(t) = simd.level().as_avx2()
    {
        return x86::sums_avx2(t, taps, window, m, s, out);
    }
    sums_generic(simd, taps, window, m, s, out)
}

/// Vectors per column block of the pair form: `2 PAIR_VECTORS` columns.
const PAIR_VECTORS: usize = 4;

/// Whether the level sums two hops at once ([`sums_pair_generic`]).
#[inline(always)]
fn pair_level<S: SimdToken>(simd: S) -> bool {
    #[cfg(target_arch = "aarch64")]
    if simd.level().as_neon().is_some() {
        return true;
    }
    let _ = simd;
    true // TEMP
}

/// The transform of one output group: the backend FFT from 16 channels on, the radix-2
/// transform of `channel_fft` for 4 and 8.
#[inline(always)]
fn forward<S: SimdToken, Tw: AsRef<[C32]>>(
    simd: S,
    plan: &FftPlan<C32, Tw>,
    group: &mut [C32],
) -> Result<()> {
    if plan.len() >= 16 {
        forward_at(simd, plan, group)
    } else {
        channel_fft(&Simd, plan, group)
    }
}

/// The channelizer at the level of `simd`.
#[inline(always)]
fn channelize_level<S, Taps, Hist, Tw>(
    simd: S,
    s: &mut ChannelizeState<C32, Taps, Hist, Tw>,
    input: &[C32],
    output: &mut [C32],
) -> Result<Progress>
where
    S: SimdToken,
    Taps: AsRef<[f32]>,
    Hist: AsRef<[C32]> + AsMut<[C32]>,
    Tw: AsRef<[C32]>,
{
    // Two hops at once where the pair form of the branch sums is the faster one (I45-1).
    let pairs = s.channels() >= 16 && pair_level(simd);
    channelize_with(
        s,
        input,
        output,
        pairs,
        #[inline(always)]
        |taps: &[f32],
         window: &[C32],
         swap: bool,
         plan: &FftPlan<C32, Tw>,
         group: &mut [C32],
         second: Option<&mut [C32]>| {
            let m = group.len();
            if let Some(second) = second {
                let s = if swap { m / 2 } else { 0 };
                sums_pair_generic::<S, PAIR_VECTORS>(
                    simd,
                    taps,
                    bytemuck::cast_slice(window),
                    m,
                    s,
                    bytemuck::cast_slice_mut(group),
                    bytemuck::cast_slice_mut(second),
                )?;
                forward(simd, plan, group)?;
                forward(simd, plan, second)?;
            } else {
                branch_sums(simd, taps, window, swap, group)?;
                forward(simd, plan, group)?;
            }
            Ok(())
        },
    )
}

impl Channelize<C32> for Simd {
    fn channelize<Taps, Hist, Tw>(
        &self,
        s: &mut ChannelizeState<C32, Taps, Hist, Tw>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<Progress>
    where
        Taps: AsRef<[f32]>,
        Hist: AsRef<[C32]> + AsMut<[C32]>,
        Tw: AsRef<[C32]>,
    {
        dispatch!(simd => channelize_level(simd, s, input, output))
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    //! The AVX2 branch sums (also run at AVX-512): blocks of sixteen columns with four 256-bit
    //! accumulators of four samples over all rows (one 256-bit load of eight stored taps per
    //! accumulator); each half of a block (eight columns) is stored at its place after the
    //! parity shift.

    use core::arch::x86_64::{__m256, _mm256_fmadd_ps, _mm256_setzero_ps};
    use fearless_simd::f32x8;
    use fearless_simd::prelude::*;
    use sdrk_core::Result;

    use super::{column_sums, store};

    fearless_simd::kernel!(
        #[inline(always)]
        pub(super) fn sums_avx2(
            t: Avx2,
            taps: &[f32],
            window: &[f32],
            m: usize,
            s: usize,
            out: &mut [f32],
        ) -> Result<()> {
            let zero = _mm256_setzero_ps();
            let mut buf = [0.0f32; 32];
            for c in (0..m).step_by(16) {
                let acc = column_sums::<_, 32>(taps, window, m, c, [zero; 4], |acc, g, w| {
                    for ((a, g), w) in acc.iter_mut().zip(g.chunks_exact(8)).zip(w.chunks_exact(8))
                    {
                        let g: __m256 = f32x8::from_slice(t, g).into();
                        let w: __m256 = f32x8::from_slice(t, w).into();
                        *a = _mm256_fmadd_ps(w, g, *a);
                    }
                })?;
                for (d, a) in buf.chunks_exact_mut(8).zip(acc) {
                    f32x8::simd_from(t, a).store_slice(d);
                }
                // The two blocks of eight columns go to their halves separately: the parity
                // shift moves each by `s`.
                let (b0, b1) = buf.split_at(16);
                store(out, m, c, s, b0)?;
                store(out, m, c + 8, s, b1)?;
            }
            Ok(())
        }
    );
}
