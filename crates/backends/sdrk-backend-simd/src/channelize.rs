//! `channelize` `C32` (M45): the polyphase channelizer.
//!
//! The driver is the one of the fallback (`sdrk_core::kernels::channelize::channelize_with`,
//! the history and the hop boundaries), compiled at the dispatched level; per hop the branch
//! sums are formed with the vectors of the level and the `M`-point transform is the `C32` FFT of
//! this backend at the same level (`fft::forward_at`, no second dispatch per hop; the radix-2
//! transform of `channel_fft` for 4 and 8 channels, which the FFT kernel does not cover). The class is *tolerance* (`docs/kernels.md`): the branch sums use
//! fused multiply-adds where the level has them, and the FFT is within its own contract, so the
//! output agrees with the fallback within the 100 dB of the contract, not bit for bit.
//!
//! **Branch sums.** Column `c` of the window is multiplied by the real tap `g[p M + c]` of every
//! row `p` and summed over the rows: a real tap times an interleaved complex sample, so a vector
//! of `k` interleaved samples needs the vector of their `k` taps with every tap duplicated
//! (`g0 g0 g1 g1 ...`). One shuffle per vector of products makes that vector:
//!
//! - Generic code (SSE2, SSE4.2, NEON, SIMD128 and the scalar level): four taps in one 128-bit
//!   vector, `zip_low` / `zip_high` of it with itself for the two vectors of two samples each.
//! - AVX2 and AVX-512 (`x86` below): eight taps in one 256-bit vector, `vpermps` with the
//!   indices `0 0 1 1 2 2 3 3` and `4 4 5 5 6 6 7 7` for the two vectors of four samples each.
//!
//! The columns are summed in blocks (sixteen columns at AVX2 and AVX-512, eight in the generic
//! code, four and two for 8 and 4 channels) whose partial sums stay in registers over all rows,
//! as in the fallback; each block of eight columns is stored at its place after the parity shift
//! (column `c` to `(c + s) mod M`), which never splits it because `M / 2` is a multiple of eight
//! from 16 channels on.

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

/// The sums of the `W` columns from `c` over all rows: `body(acc, taps, window)` adds the
/// products of one row (its `W` taps and `2 W` window floats) to the accumulators.
#[inline(always)]
fn column_sums<A, const W: usize, const W2: usize>(
    taps: &[f32],
    window: &[f32],
    m: usize,
    c: usize,
    mut acc: A,
    mut body: impl FnMut(&mut A, &[f32; W], &[f32; W2]),
) -> Result<A> {
    if c + W > m || W2 != 2 * W {
        return Err(bad_len());
    }
    // Whole rows: the bounds of the columns are checked once above (the compiler hoists the
    // checks below out of the loop).
    for (t, w) in taps.chunks_exact(m).zip(window.chunks_exact(2 * m)) {
        let t = t.get(c..c + W).and_then(|t| t.first_chunk::<W>());
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
            let acc = column_sums::<_, 8, 16>(taps, window, m, c, [zero; 4], |acc, t, w| {
                let (ta, tb) = t.split_at(4);
                let ta = f32x4::from_slice(simd, ta);
                let tb = f32x4::from_slice(simd, tb);
                let g = [
                    ta.zip_low(ta),
                    ta.zip_high(ta),
                    tb.zip_low(tb),
                    tb.zip_high(tb),
                ];
                for ((a, g), x) in acc.iter_mut().zip(g).zip(w.chunks_exact(4)) {
                    *a = f32x4::from_slice(simd, x).mul_add(g, *a);
                }
            })?;
            for (d, a) in buf.chunks_exact_mut(4).zip(acc) {
                a.store_slice(d);
            }
            store(out, m, c, s, &buf)?;
        }
    } else if m == 8 {
        for c in [0, 4] {
            let acc = column_sums::<_, 4, 8>(taps, window, m, c, [zero; 2], |acc, t, w| {
                let t = f32x4::from_slice(simd, t);
                let g = [t.zip_low(t), t.zip_high(t)];
                for ((a, g), x) in acc.iter_mut().zip(g).zip(w.chunks_exact(4)) {
                    *a = f32x4::from_slice(simd, x).mul_add(g, *a);
                }
            })?;
            for (d, a) in buf.chunks_exact_mut(4).zip(acc) {
                a.store_slice(d);
            }
            store(out, m, c, s, buf.get(..8).ok_or_else(bad_len)?)?;
        }
    } else {
        for c in [0, 2] {
            let acc = column_sums::<_, 2, 4>(taps, window, m, c, zero, |acc, &[t0, t1], w| {
                let g = f32x4::from_slice(simd, &[t0, t0, t1, t1]);
                *acc = f32x4::from_slice(simd, w).mul_add(g, *acc);
            })?;
            acc.store_slice(buf.get_mut(..4).ok_or_else(bad_len)?);
            store(out, m, c, s, buf.get(..4).ok_or_else(bad_len)?)?;
        }
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
    if m < 4 || !m.is_power_of_two() || taps.len() != window.len() || taps.len() & (m - 1) != 0 {
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
    channelize_with(
        s,
        input,
        output,
        #[inline(always)]
        |taps: &[f32], window: &[C32], swap: bool, plan: &FftPlan<C32, Tw>, group: &mut [C32]| {
            branch_sums(simd, taps, window, swap, group)?;
            if plan.len() >= 16 {
                forward_at(simd, plan, group)
            } else {
                channel_fft(&Simd, plan, group)
            }
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
    //! accumulators of four samples over all rows; each half of a block (eight columns) is
    //! stored at its place after the parity shift.

    use core::arch::x86_64::{
        __m128, __m256, __m256i, _mm256_castps128_ps256, _mm256_fmadd_ps, _mm256_insertf128_ps,
        _mm256_permutevar_ps, _mm256_setr_epi32, _mm256_setzero_ps,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4, f32x8};
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
            // In-lane `vpermilps` indices on four taps in both halves: `0 0 1 1` in the low half,
            // `2 2 3 3` in the high half.
            let dup: __m256i = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
            let zero = _mm256_setzero_ps();
            let mut buf = [0.0f32; 32];
            for c in (0..m).step_by(16) {
                let acc = column_sums::<_, 16, 32>(taps, window, m, c, [zero; 4], |acc, g, w| {
                    for ((a, g), w) in acc.iter_mut().zip(g.chunks_exact(4)).zip(w.chunks_exact(8))
                    {
                        let g4: __m128 = f32x4::from_slice(t, g).into();
                        let g = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(g4), g4);
                        let w: __m256 = f32x8::from_slice(t, w).into();
                        *a = _mm256_fmadd_ps(w, _mm256_permutevar_ps(g, dup), *a);
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
