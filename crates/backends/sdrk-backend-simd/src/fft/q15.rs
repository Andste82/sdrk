//! The complex FFT for `CQ15`; see the module documentation of `fft`.
//!
//! # Level-specific paths (x86, wasm, AArch64): radix 4 in `i16` lanes
//!
//! After the bit reversal, radix-4 stages (decimation in time; the first one has no twiddles)
//! and, for an odd number of factors 2, one radix-2 stage at the end. Every twiddle product
//! rounds each part (`(c w + 2^14) >> 15`, `pmulhrsw`, `sqrdmulh`, `i16x8.q15mulr_sat_s`; plain
//! SSE2 sums the two products of a part with `pmaddwd` and rounds once) and the twiddle `w^3k`
//! is the rounded product of `w^k` and `w^2k`. The scaling (see [`HALVING`]):
//!
//! - fixed scaling divides every stage by its radix, `N` in all, as the fallback: every level of
//!   adds halves, except on x86 and wasm from 64 to [`SCALED_FIXED_MAX`] samples, where the
//!   stages but the last one scale `x0` and the twiddles by `2^-2` instead; an input beyond full
//!   scale (magnitude up to `sqrt 2`) scales the first stage by `2^-3` and gives the extra
//!   halving back in a later stage (see [`fixed_fits`]);
//! - block floating point shifts a radix-4 (radix-2) stage by the smallest `s` with
//!   `4 M / 2^s <= 32770` (`2 M`), `M` the largest complex magnitude of its input, which the
//!   previous stage tracks (`re^2 + im^2`); the shift scales `x0` and the twiddles.
//!
//! The result is within the contract of `docs/kernels.md` (tolerance), not bit-identical with the
//! fallback, and its exponent may differ from the fallback's (the output times `2^exponent` is
//! what the contract compares). Different levels give different bits within the tolerance.
//!
//! # Generic code (the levels without such a path, and the sizes below it)
//!
//! The radix-2 stages of the fallback, bit for bit, the block-floating-point exponent included:
//! `a` and `c` are loaded as two vectors of interleaved samples each, split into real and
//! imaginary lanes, widened to `i32`, multiplied and summed exactly, rounded
//! (`(P + 2^14) >> 15`), added, shifted with rounding and narrowed with saturation. The twiddles
//! of a vector of butterflies are built once per position and reused for every block, the stages
//! with fewer butterflies per block than lanes regroup two blocks with lane permutations, and
//! block floating point takes the largest component of a stage output from the outputs
//! themselves. The products and sums fit `i32` because the twiddles are those of `FftPlan::new`
//! (or the static tables, which hold the same values): `|wr| + |wi| <= 46341`, so
//! `|wr cr - wi ci| <= 46341 * 32768 < 2^31`.

use fearless_simd::{Bytes, Simd as SimdToken, prelude::*};
use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, FftScale, Scaling};
use sdrk_core::num::{round_shift, sat16};
use sdrk_core::{CQ15, Q15, Result};

use crate::Simd;
use crate::level::dispatch;
use crate::util::check_len;

/// Largest component magnitude above which a stage needs a shift of 1 (block floating point).
const BFP_SHIFT1_ABOVE: i32 = 13_572;
/// Largest component magnitude above which a stage needs a shift of 2.
const BFP_SHIFT2_ABOVE: i32 = 27_145;

/// The block-floating-point shift of a stage whose input has the largest component magnitude
/// `m`.
#[inline(always)]
fn bfp_shift(m: i32) -> u32 {
    if m > BFP_SHIFT2_ABOVE {
        2
    } else {
        u32::from(m > BFP_SHIFT1_ABOVE)
    }
}

/// Two `i16` vectors: the interleaved samples of `L` butterfly inputs or outputs.
type Pair<S> = (<S as SimdToken>::i16s, <S as SimdToken>::i16s);

/// `sat16(round_shift(v, shift))`.
#[inline(always)]
fn store_q15(v: i64, shift: u32) -> Q15 {
    Q15(sat16(round_shift(v, shift)))
}

/// One radix-2 stage, scalar; the arithmetic of the fallback. Used for the sizes below two
/// vectors of butterflies.
fn stage_scalar(
    data: &mut [CQ15],
    tw: &[CQ15],
    half: usize,
    step: usize,
    inverse: bool,
    shift: u32,
) {
    for block in data.chunks_exact_mut(2 * half) {
        let (lo, hi) = block.split_at_mut(half);
        for (k, (a, c)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
            let w = tw.get(k * step).copied().unwrap_or_default();
            let (wr, wi) = (i64::from(w.re.0), i64::from(w.im.0));
            let wi = if inverse { -wi } else { wi };
            let (cr, ci) = (i64::from(c.re.0), i64::from(c.im.0));
            let tr = round_shift(wr * cr - wi * ci, 15);
            let ti = round_shift(wr * ci + wi * cr, 15);
            let (ar, ai) = (i64::from(a.re.0), i64::from(a.im.0));
            *a = CQ15 {
                re: store_q15(ar + tr, shift),
                im: store_q15(ai + ti, shift),
            };
            *c = CQ15 {
                re: store_q15(ar - tr, shift),
                im: store_q15(ai - ti, shift),
            };
        }
    }
}

/// The twiddles of `L` butterflies widened to `i32`: real parts of the lower and upper half of
/// the lanes, then the imaginary parts, negated for the inverse transform (in `i32`, so that
/// `-(-32768)` is exact).
struct Twiddles<S: SimdToken> {
    re: (S::i32s, S::i32s),
    im: (S::i32s, S::i32s),
}

impl<S: SimdToken> Twiddles<S> {
    #[inline(always)]
    fn new(re: S::i16s, im: S::i16s, inverse: bool) -> Self {
        let (im0, im1) = im.widen();
        Self {
            re: re.widen(),
            im: if inverse { (-im0, -im1) } else { (im0, im1) },
        }
    }
}

/// The twiddles `w[(k0 + i) * step]` for `i < L` as real and imaginary `i16` lanes. Steps 1, 2
/// and 4 load the `L * step` table entries from `k0 * step` and keep every `step`-th by 32-bit
/// unzips; larger steps (stages with few butterflies per block, so few `k0`) gather lane by lane.
#[inline(always)]
fn twiddle_lanes<S: SimdToken>(simd: S, tw: &[i16], k0: usize, step: usize) -> (S::i16s, S::i16s) {
    let l = S::i16s::LEN;
    let contiguous = || -> Option<(S::i16s, S::i16s)> {
        let t = tw.get(2 * k0 * step..2 * (k0 + l) * step)?;
        let mut v = [S::i32s::splat(simd, 0); 8];
        for (slot, chunk) in v.iter_mut().zip(t.chunks_exact(l)) {
            *slot = S::i16s::from_slice(simd, chunk).bitcast::<S::i32s>();
        }
        let mut count = 2 * step;
        while count > 2 {
            for i in 0..count / 2 {
                if let (Some(&x), Some(&y)) = (v.get(2 * i), v.get(2 * i + 1))
                    && let Some(slot) = v.get_mut(i)
                {
                    *slot = x.unzip_low(y);
                }
            }
            count /= 2;
        }
        let [x0, x1, ..] = v.map(Bytes::bitcast::<S::i16s>);
        Some((x0.unzip_low(x1), x0.unzip_high(x1)))
    };
    if matches!(step, 1 | 2 | 4)
        && let Some(w) = contiguous()
    {
        return w;
    }
    let at =
        |i: usize, part: usize| -> i16 { tw.get(2 * (k0 + i) * step + part).copied().unwrap_or(0) };
    (
        S::i16s::from_fn(simd, |i| at(i, 0)),
        S::i16s::from_fn(simd, |i| at(i, 1)),
    )
}

/// `L` butterflies: `a` and `c` hold `L` samples each as two vectors of interleaved samples (in
/// any order of the samples, as long as `a`, `c` and the twiddles agree). Returns the outputs in
/// the same layout and folds them into the running maximum and minimum `range`. Bit-identical
/// with [`stage_scalar`] with the shift `SH`.
#[inline(always)]
fn butterflies<S: SimdToken, const SH: u32>(
    simd: S,
    a: (S::i16s, S::i16s),
    c: (S::i16s, S::i16s),
    w: &Twiddles<S>,
    range: &mut (S::i16s, S::i16s),
) -> (Pair<S>, Pair<S>) {
    let round = S::i32s::splat(simd, 1 << 14);
    let (ar, ai) = (a.0.unzip_low(a.1), a.0.unzip_high(a.1));
    let (cr, ci) = (c.0.unzip_low(c.1), c.0.unzip_high(c.1));
    let (ar0, ar1) = ar.widen();
    let (ai0, ai1) = ai.widen();
    let (cr0, cr1) = cr.widen();
    let (ci0, ci1) = ci.widen();
    let (wr0, wr1) = w.re;
    let (wi0, wi1) = w.im;
    let tr0 = (wr0 * cr0 - wi0 * ci0 + round) >> 15u32;
    let tr1 = (wr1 * cr1 - wi1 * ci1 + round) >> 15u32;
    let ti0 = (wr0 * ci0 + wi0 * cr0 + round) >> 15u32;
    let ti1 = (wr1 * ci1 + wi1 * cr1 + round) >> 15u32;
    // `sat16(round_shift(v, SH))` of two vectors into one.
    let finish = |x: S::i32s, y: S::i32s| -> S::i16s {
        if SH == 0 {
            x.saturating_narrow(y)
        } else {
            let r = S::i32s::splat(simd, 1 << (SH - 1));
            ((x + r) >> SH).saturating_narrow((y + r) >> SH)
        }
    };
    let yar = finish(ar0 + tr0, ar1 + tr1);
    let yai = finish(ai0 + ti0, ai1 + ti1);
    let ycr = finish(ar0 - tr0, ar1 - tr1);
    let yci = finish(ai0 - ti0, ai1 - ti1);
    range.0 = range.0.max(yar).max(yai).max(ycr).max(yci);
    range.1 = range.1.min(yar).min(yai).min(ycr).min(yci);
    (
        (yar.zip_low(yai), yar.zip_high(yai)),
        (ycr.zip_low(yci), ycr.zip_high(yci)),
    )
}

/// One radix-2 stage with the shift `SH` on `data` (interleaved, `n >= 2 L` samples) for
/// `half >= L`, `half < 4` and, with 128-bit vectors, `half == 4`. Returns the largest component
/// magnitude of the output.
#[inline(always)]
fn stage<S: SimdToken, const SH: u32>(
    simd: S,
    data: &mut [i16],
    tw: &[i16],
    half: usize,
    step: usize,
    inverse: bool,
) -> i32 {
    let l = S::i16s::LEN;
    let mut range = (
        S::i16s::splat(simd, i16::MIN),
        S::i16s::splat(simd, i16::MAX),
    );
    let load = |s: &[i16]| S::i16s::from_slice(simd, s);
    if half >= l {
        for k0 in (0..half).step_by(l) {
            let (re, im) = twiddle_lanes(simd, tw, k0, step);
            let w = Twiddles::<S>::new(re, im, inverse);
            for block in data.chunks_exact_mut(4 * half) {
                let (lo, hi) = block.split_at_mut(2 * half);
                let (Some(a), Some(c)) = (
                    lo.get_mut(2 * k0..2 * (k0 + l)),
                    hi.get_mut(2 * k0..2 * (k0 + l)),
                ) else {
                    continue;
                };
                let (a0, a1) = a.split_at_mut(l);
                let (c0, c1) = c.split_at_mut(l);
                let (ya, yc) = butterflies::<S, SH>(
                    simd,
                    (load(a0), load(a1)),
                    (load(c0), load(c1)),
                    &w,
                    &mut range,
                );
                ya.0.store_slice(a0);
                ya.1.store_slice(a1);
                yc.0.store_slice(c0);
                yc.1.store_slice(c1);
            }
        }
    } else {
        // Chunks of `2 L` samples in four vectors; lane `i` of the regrouped `a` and `c` is
        // butterfly `i % half` of its block.
        let w_at = |i: usize, part: usize| -> i16 {
            tw.get(2 * (i % half) * step + part).copied().unwrap_or(0)
        };
        let w = Twiddles::<S>::new(
            S::i16s::from_fn(simd, |i| w_at(i, 0)),
            S::i16s::from_fn(simd, |i| w_at(i, 1)),
            inverse,
        );
        for chunk in data.chunks_exact_mut(4 * l) {
            let (p, q) = chunk.split_at_mut(2 * l);
            let (s0, s1) = p.split_at_mut(l);
            let (s2, s3) = q.split_at_mut(l);
            let v = [load(s0), load(s1), load(s2), load(s3)];
            let (a, c) = regroup::<S>(v, half);
            let (ya, yc) = butterflies::<S, SH>(simd, a, c, &w, &mut range);
            let [o0, o1, o2, o3] = ungroup::<S>(ya, yc, half);
            o0.store_slice(s0);
            o1.store_slice(s1);
            o2.store_slice(s2);
            o3.store_slice(s3);
        }
    }
    i32::from(range.0.reduce_max()).max(-i32::from(range.1.reduce_min()))
}

/// The `a` and `c` samples of four consecutive vectors for `half` = 1 (32-bit unzips), 2 (64-bit
/// unzips) or 4 with 128-bit vectors (`v0`, `v2` are `a`, `v1`, `v3` are `c`).
#[inline(always)]
fn regroup<S: SimdToken>(v: [S::i16s; 4], half: usize) -> (Pair<S>, Pair<S>) {
    let [v0, v1, v2, v3] = v;
    match half {
        1 => {
            let [x0, x1, x2, x3] = [v0, v1, v2, v3].map(Bytes::bitcast::<S::i32s>);
            (
                (x0.unzip_low(x1).bitcast(), x2.unzip_low(x3).bitcast()),
                (x0.unzip_high(x1).bitcast(), x2.unzip_high(x3).bitcast()),
            )
        }
        2 => {
            let [x0, x1, x2, x3] = [v0, v1, v2, v3].map(Bytes::bitcast::<S::f64s>);
            (
                (x0.unzip_low(x1).bitcast(), x2.unzip_low(x3).bitcast()),
                (x0.unzip_high(x1).bitcast(), x2.unzip_high(x3).bitcast()),
            )
        }
        _ => ((v0, v2), (v1, v3)),
    }
}

/// The inverse of [`regroup`].
#[inline(always)]
fn ungroup<S: SimdToken>(
    a: (S::i16s, S::i16s),
    c: (S::i16s, S::i16s),
    half: usize,
) -> [S::i16s; 4] {
    match half {
        1 => {
            let [a0, a1, c0, c1] = [a.0, a.1, c.0, c.1].map(Bytes::bitcast::<S::i32s>);
            [
                a0.zip_low(c0).bitcast(),
                a0.zip_high(c0).bitcast(),
                a1.zip_low(c1).bitcast(),
                a1.zip_high(c1).bitcast(),
            ]
        }
        2 => {
            let [a0, a1, c0, c1] = [a.0, a.1, c.0, c.1].map(Bytes::bitcast::<S::f64s>);
            [
                a0.zip_low(c0).bitcast(),
                a0.zip_high(c0).bitcast(),
                a1.zip_low(c1).bitcast(),
                a1.zip_high(c1).bitcast(),
            ]
        }
        _ => [a.0, c.0, a.1, c.1],
    }
}

/// Largest component magnitude of the data.
#[inline(always)]
fn max_component<S: SimdToken>(simd: S, data: &[i16]) -> i32 {
    let l = S::i16s::LEN;
    let mut hi = S::i16s::splat(simd, i16::MIN);
    let mut lo = S::i16s::splat(simd, i16::MAX);
    let mut chunks = data.chunks_exact(l);
    for c in &mut chunks {
        let v = S::i16s::from_slice(simd, c);
        hi = hi.max(v);
        lo = lo.min(v);
    }
    let mut top = i32::from(hi.reduce_max());
    let mut bottom = i32::from(lo.reduce_min());
    for &v in chunks.remainder() {
        top = top.max(i32::from(v));
        bottom = bottom.min(i32::from(v));
    }
    top.max(-bottom)
}

// ---------------------------------------------------------------------------------------------
// Bit reversal

/// Swaps `data[i]` and `data[rev(i)]` for `i < rev(i)`, where `rev` reverses the lowest `log2n`
/// bits (scalar; used for sizes too small for the tile method).
fn bit_reverse<T>(data: &mut [T], log2n: u32) {
    let n = data.len();
    if n != 1usize << log2n {
        return;
    }
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            data.swap(i, j);
        }
        // Increment `j` as a bit-reversed counter.
        let mut bit = n >> 1;
        while bit != 0 && j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
    }
}

/// Reverses `rows` of a `V x V` tile: with `rows[a][c]` as the element `(a, c)` on entry,
/// `rows[k][j]` is the element `(rev(j), rev(k))` on exit (`rev` reverses `log2 V` bits). The
/// network is `log2 V` stages of `zip_low`/`zip_high` between rows at distance 1, 2, 4.
macro_rules! tile_network {
    ($rows:expr, $v:expr) => {{
        let mut d = 1usize;
        while d < $v {
            for r in 0..$v {
                if r & d == 0 {
                    if let (Some(&lo), Some(&hi)) = ($rows.get(r), $rows.get(r + d)) {
                        let (z0, z1) = (lo.zip_low(hi), lo.zip_high(hi));
                        if let Some(x) = $rows.get_mut(r) {
                            *x = z0;
                        }
                        if let Some(x) = $rows.get_mut(r + d) {
                            *x = z1;
                        }
                    }
                }
            }
            d *= 2;
        }
    }};
}

/// In-place bit-reversal permutation of the `n = 2^log2n` samples of `data` (two `i16` each), by
/// tiles: with `V` samples per vector (`u32` lanes), an index is `(a, m, c)` (`a` and `c` of
/// `log2 V` bits) and its reversed index is `(rev c, rev m, rev a)`. The samples with a fixed
/// middle part `m` form a `V x V` tile of `V` vector loads (rows `a`, stride `n / V`); the tile
/// goes to the tile of `rev m`, transposed and reversed in both directions by the zip network of
/// [`tile_network`]. About 1.5 instructions per sample instead of 30 for the scalar loop.
/// `false` (nothing done) below one tile or for a wrong length.
#[inline(always)]
fn bit_reverse_cq15<S: SimdToken>(simd: S, data: &mut [i16], log2n: u32) -> bool {
    let v = S::u32s::LEN;
    let q = v.trailing_zeros();
    if log2n < 2 * q || data.len() != 2 << log2n {
        return false;
    }
    let mid_bits = log2n - 2 * q;
    let stride = 2 * ((1usize << log2n) / v); // lanes between rows
    let width = 2 * v; // lanes of one row
    let zero = S::i16s::splat(simd, Default::default());
    // Local macros, not closures: the vector operations must be inlined into the
    // function that carries the target features.
    macro_rules! load {
        ($tile:expr, $rows:expr) => {
            for (a, slot) in $rows.iter_mut().take(v).enumerate() {
                let off = a * stride + $tile * width;
                let row = data
                    .get(off..off + width)
                    .map_or(zero, |s| S::i16s::from_slice(simd, s));
                *slot = row.bitcast::<S::u32s>();
            }
        };
    }
    macro_rules! store {
        ($tile:expr, $rows:expr) => {
            for (a, row) in $rows.iter().take(v).enumerate() {
                let off = a * stride + $tile * width;
                if let Some(dst) = data.get_mut(off..off + width) {
                    row.bitcast::<S::i16s>().store_slice(dst);
                }
            }
        };
    }
    let blank = zero.bitcast::<S::u32s>();
    for m in 0..1usize << mid_bits {
        let mirror = if mid_bits == 0 {
            0
        } else {
            m.reverse_bits() >> (usize::BITS - mid_bits)
        };
        if mirror < m {
            continue;
        }
        let mut x = [blank; 16];
        load!(m, x);
        tile_network!(x, v);
        if mirror == m {
            store!(m, x);
        } else {
            let mut y = [blank; 16];
            load!(mirror, y);
            tile_network!(y, v);
            store!(mirror, x);
            store!(m, y);
        }
    }
    true
}

// ---------------------------------------------------------------------------------------------
// Driver

/// The transform of `data` at the level of `simd`: the level-specific path where the level has
/// one and `n` is large enough, else the bit reversal and the generic radix-2 stages. Returns
/// the exponent.
#[inline(always)]
fn fft_q15_stages<S: SimdToken>(
    simd: S,
    data: &mut [CQ15],
    tw: &[CQ15],
    n: usize,
    log2n: u32,
    stride: usize,
    inverse: bool,
    block_float: bool,
) -> i32 {
    let l = S::i16s::LEN;
    let tw_i16: &[i16] = bytemuck::cast_slice(tw);
    if let Some(exponent) = level_transform(simd, data, tw_i16, log2n, stride, inverse, block_float)
    {
        return exponent;
    }
    if !bit_reverse_cq15::<S>(simd, bytemuck::cast_slice_mut(&mut *data), log2n) {
        bit_reverse(data, log2n);
    }
    // Largest component magnitude of the stage input (block floating point only).
    let mut top = if block_float {
        max_component(simd, bytemuck::cast_slice(&*data))
    } else {
        0
    };
    let mut exponent = 0i32;
    for s in 1..=log2n {
        let shift = if block_float { bfp_shift(top) } else { 1 };
        let half = 1usize << (s - 1);
        let step = (n >> s) * stride;
        let regrouped = half >= l || half < 4 || (half == 4 && l == 8);
        if n >= 2 * l && regrouped {
            let d = bytemuck::cast_slice_mut(&mut *data);
            top = match shift {
                0 => stage::<S, 0>(simd, d, tw_i16, half, step, inverse),
                1 => stage::<S, 1>(simd, d, tw_i16, half, step, inverse),
                _ => stage::<S, 2>(simd, d, tw_i16, half, step, inverse),
            };
        } else {
            stage_scalar(data, tw, half, step, inverse, shift);
            if block_float {
                top = max_component(simd, bytemuck::cast_slice(&*data));
            }
        }
        exponent += shift as i32;
    }
    exponent
}

/// The transform with the level-specific path of the level of `simd`, if the level has one and
/// `n` is large enough for it; returns the exponent.
#[inline(always)]
fn level_transform<S: SimdToken>(
    simd: S,
    data: &mut [CQ15],
    tw: &[i16],
    log2n: u32,
    stride: usize,
    inverse: bool,
    block_float: bool,
) -> Option<i32> {
    let level = simd.level();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if let Some(t) = level.as_avx512() {
            return radix4::transform(
                t,
                bytemuck::cast_slice_mut(data),
                tw,
                log2n,
                stride,
                inverse,
                block_float,
            );
        }
        if let Some(t) = level.as_avx2() {
            return radix4::transform(
                t,
                bytemuck::cast_slice_mut(data),
                tw,
                log2n,
                stride,
                inverse,
                block_float,
            );
        }
        if let Some(t) = level.as_sse4_2() {
            return radix4::transform(
                t,
                bytemuck::cast_slice_mut(data),
                tw,
                log2n,
                stride,
                inverse,
                block_float,
            );
        }
        if let Some(t) = level.as_sse2() {
            return radix4::transform(
                t,
                bytemuck::cast_slice_mut(data),
                tw,
                log2n,
                stride,
                inverse,
                block_float,
            );
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        if log2n < 6 {
            return None;
        }
        return Some(neon::stages(
            t,
            bytemuck::cast_slice_mut(data),
            tw,
            log2n,
            stride,
            inverse,
            block_float,
        ));
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return radix4::transform(
            t,
            bytemuck::cast_slice_mut(data),
            tw,
            log2n,
            stride,
            inverse,
            block_float,
        );
    }
    let _ = (level, data, tw, log2n, stride, inverse, block_float);
    None
}

impl Fft<CQ15> for Simd {
    fn fft<Tw: AsRef<[CQ15]>>(
        &self,
        plan: &FftPlan<CQ15, Tw>,
        data: &mut [CQ15],
        dir: Direction,
        scaling: Scaling,
    ) -> Result<FftScale> {
        let n = plan.len();
        check_len(n, data.len())?;
        let inverse = dir == Direction::Inverse;
        let block_float = scaling == Scaling::BlockFloat;
        let (tw, stride, log2n) = (plan.twiddles(), plan.twiddle_stride(), plan.log2_len());
        let exponent = dispatch!(simd => fft_q15_stages(simd, data, tw, n, log2n, stride, inverse, block_float));
        Ok(FftScale { exponent })
    }
}

// ---------------------------------------------------------------------------------------------
// Level-specific stages: radix-4 in `i16` lanes

/// The largest stage output magnitude the block-floating-point shift of a level-specific stage
/// allows: a radix-`r` stage shifts by the smallest `s` with `r M / 2^s <= 32770`, `M` the largest
/// complex magnitude of its input. The bound sits just above full scale, so that a full-scale
/// tone keeps its precision (its magnitude rounds up to 32768); a twiddle product of such an
/// input saturates by at most 3.
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
const STAGE_LIMIT: u64 = 32_770;

/// The scaling of a level-specific stage. Fixed scaling halves both levels of adds of a radix-4
/// butterfly (the one of a radix-2 one): `(a + b + 1) >> 1` for sums, `(a - b) >> 1` for
/// differences. Block floating point with a shift `s > 0` scales instead: `x0` by `2^-s` with
/// rounding and the twiddles of the other inputs to `w 2^-s` (rounded), so that every output is
/// a saturating sum of rounded products; with `s = 0` it adds without scaling.
///
/// The halving adds round once per level and keep fixed scaling within 1 dB of the fallback,
/// whose output shrinks by `N`; the scaled twiddles take half the instructions of the adds on
/// x86 and wasm (where a halving add is three instructions) and lose nothing that matters with
/// block floating point, which keeps the data at full scale. With fixed scaling they round the
/// four terms of an output each, which costs up to 3 dB when every stage does it. The error of
/// the last stage dominates (the earlier ones shrink with the data), so x86 and wasm scale the
/// twiddles in all stages but the last one, which halves: within 1 dB of the fallback from 64
/// up to [`SCALED_FIXED_MAX`] samples. Above that size, and on NEON (whose halving adds are
/// single instructions, so that it also takes the sizes below 64), every stage halves. Below 64
/// samples x86 and wasm take the generic code.
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
const HALVING: u8 = 0;
/// See [`HALVING`].
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
const SCALED: u8 = 1;
/// See [`HALVING`].
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
const PLAIN: u8 = 2;

/// The largest size at which fixed scaling on x86 and wasm uses the scaled twiddles in the
/// stages before the last one (from 64 samples on; see [`HALVING`]).
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
const SCALED_FIXED_MAX: usize = 8192;

/// The scaling and the shift of a radix-`radix` stage (4 or 2) whose input has the largest
/// squared magnitude `norm` (block floating point), or of fixed scaling (`None`: 1 per factor of
/// 2, with the scaling `fixed`).
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn plan(norm: Option<u64>, radix: u64, fixed: u8) -> (u8, u32) {
    let levels = if radix == 4 { 2 } else { 1 };
    norm.map_or((fixed, levels), |norm| {
        // radix M / 2^s <= limit, squared; M^2 <= 2^31, so the shift never exceeds levels + 1.
        let mut s = 0;
        while s <= levels && radix * radix * norm > (STAGE_LIMIT * STAGE_LIMIT) << (2 * s) {
            s += 1;
        }
        (if s == 0 { PLAIN } else { SCALED }, s)
    })
}

/// Whether fixed scaling of an input whose largest squared magnitude is at most `norm` stays in
/// the `i16` lanes with the plain plan: the twiddle products of a halving stage are formed at the
/// scale of its input and the outputs of a scaled stage are sums of four terms, so a sample
/// beyond [`STAGE_LIMIT`] (a corner such as `(-1, 1)`, magnitude `sqrt 2`) would saturate them.
/// Such an input takes the over-range plan instead: the twiddle-free first stage scales its
/// inputs by `2^-3`, one halving more, so that its outputs stay below `sqrt 2 / 2` of full scale
/// and the twiddle products of the next stage fit. The halving is owed until a later stage gives
/// it back (see [`owed_stage`]), so the overall scale stays `1 / N`; the last stage saturates
/// where the fallback's output saturates.
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn fixed_fits(norm: u64) -> bool {
    norm <= STAGE_LIMIT * STAGE_LIMIT
}

/// The shift of a radix-4 stage of the over-range plan (see [`fixed_fits`]) while a halving is
/// owed, and whether it stays owed; such a stage always has the scaling `SCALED`. The last stage
/// gives the halving back (`2^-1`); an earlier stage does as soon as the largest squared
/// magnitude `bound` of its input lets a stage with the shift 1 stay in range (which keeps the
/// data at full scale for the stages after it), else it scales by `2^-2` and tracks its output
/// for the next one.
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn owed_stage(bound: u64, last: bool) -> (u32, bool) {
    if last || plan(Some(bound), 4, SCALED).1 <= 1 {
        (1, false)
    } else {
        (2, true)
    }
}

/// The twiddles `w[(k0 + i) * step]` of the samples `i` of one vector, interleaved. Steps 1, 2
/// and 4 load the `step` vectors from `k0 * step` and keep every `step`-th sample by unzips;
/// larger steps (the stages with few butterflies per block, so few `k0`) gather.
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn twiddle_vector<T: SimdToken>(t: T, tw: &[i16], k0: usize, step: usize) -> T::i16s {
    let n = T::i16s::LEN;
    let at = 2 * k0 * step;
    if step <= 4
        && step.is_power_of_two()
        && let Some(src) = tw.get(at..at + n * step)
    {
        let mut parts = src
            .chunks_exact(n)
            .map(|s| T::i16s::from_slice(t, s).bitcast());
        let mut next = || parts.next().unwrap_or(T::i32s::splat(t, 0));
        let v: T::i32s = match step {
            1 => next(),
            2 => next().unzip_low(next()),
            _ => next().unzip_low(next()).unzip_low(next().unzip_low(next())),
        };
        return v.bitcast();
    }
    T::i16s::from_fn(t, |i| {
        tw.get(2 * (k0 + i / 2) * step + i % 2)
            .copied()
            .unwrap_or(0)
    })
}

/// The twiddles of one vector from `chunk`, the `STEP` vectors of table entries from its first
/// position: every `STEP`-th sample, by unzips.
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn twiddle_chunk<T: SimdToken, const STEP: usize>(t: T, chunk: &[i16]) -> T::i16s {
    let n = T::i16s::LEN;
    let p0 = vector_at(t, chunk, 0);
    let v: T::i32s = match STEP {
        1 => p0,
        2 => p0.unzip_low(vector_at(t, chunk, n)),
        _ => p0
            .unzip_low(vector_at(t, chunk, n))
            .unzip_low(vector_at(t, chunk, 2 * n).unzip_low(vector_at(t, chunk, 3 * n))),
    };
    v.bitcast()
}

/// The vector of `chunk` from `at`, as `i32` lanes (zero past the end).
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn vector_at<T: SimdToken>(t: T, chunk: &[i16], at: usize) -> T::i32s {
    chunk
        .get(at..at + T::i16s::LEN)
        .map_or(T::i32s::splat(t, 0), |s| {
            T::i16s::from_slice(t, s).bitcast()
        })
}

/// The radix-4 stages on x86 and wasm.
///
/// The samples stay interleaved. A twiddle product rounds every part (`pmulhrsw`,
/// `i16x8.q15mulr_sat_s`; on plain SSE2 the two products of a part are summed exactly with
/// `pmaddwd` and rounded once): `(cr wr - ci wi, ci wr + cr wi)` is the saturating sum of `c` times
/// `(wr, wr)` and the swapped `c` times `(-wi, wi)`. The halving adds (see [`HALVING`]) are the
/// unsigned rounding average (`pavgw`, `avgr_u`) of the lanes offset by 2^15, `(a + b + 1) >> 1`,
/// and, for the differences, the average with the complement, `(a - b) >> 1`. The block-float
/// shift comes from the largest `re^2 + im^2` (`pmaddwd` of the outputs with themselves).
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod radix4 {
    use core::hint::black_box;

    use fearless_simd::{Simd as SimdToken, prelude::*};

    use super::{HALVING, PLAIN, SCALED, fixed_fits, owed_stage, plan};

    /// The operations of the radix-4 stages that `fearless_simd` does not offer. A "block" is a
    /// 128-bit part of a vector (4 samples); the operations work within each block.
    pub(super) trait Radix4: SimdToken {
        /// A twiddle vector in the form of [`Radix4::cmul`].
        type Twiddle: Copy;
        /// The twiddles `w` (interleaved, one per sample, no part at -32768) for the forward
        /// transform, or conjugated for the inverse one.
        fn twiddle(self, w: Self::i16s, inverse: bool, s: Self::Swap) -> Self::Twiddle;
        /// The product of every sample of `c` with its twiddle, each part rounded and saturated.
        fn cmul(self, c: Self::i16s, w: Self::Twiddle, s: Self::Swap) -> Self::i16s;
        /// [`Radix4::cmul`] of two vectors, which some levels finish together.
        #[inline(always)]
        fn cmul2(
            self,
            a: (Self::i16s, Self::Twiddle),
            b: (Self::i16s, Self::Twiddle),
            s: Self::Swap,
        ) -> (Self::i16s, Self::i16s) {
            (self.cmul(a.0, a.1, s), self.cmul(b.0, b.1, s))
        }
        /// The operands of [`Radix4::swap`] and [`Radix4::twiddle`] that stay in registers (the
        /// `pshufb` patterns: as constants, LLVM may lower them to two shuffles), or `()`.
        type Swap: Copy;
        /// The value of [`Radix4::Swap`].
        fn swap_pattern(self) -> Self::Swap;
        /// `v 2^-shift` with rounding, `1 <= shift <= 3`.
        fn scale(self, v: Self::i16s, shift: u32) -> Self::i16s;
        /// Both `i16` lanes of every sample exchanged.
        fn swap(self, v: Self::i16s, s: Self::Swap) -> Self::i16s;
        /// `(a + b + 1) >> 1` of the lanes taken as unsigned.
        fn avg_u(self, a: Self::i16s, b: Self::i16s) -> Self::i16s;
        /// `acc` with `re^2 + im^2` of every sample of `v` folded in, as [`Radix4::norm_bound`]
        /// reads it (starting from zero).
        fn max_norm(self, acc: Self::u32s, v: Self::i16s) -> Self::u32s;
        /// An upper bound of the largest `re^2 + im^2` folded into `acc` (`2^31` for the sample
        /// `(-32768, -32768)`).
        fn norm_bound(self, acc: Self::u32s) -> u64;
        /// A tile of `V` vectors of `V` samples each (`V` samples per vector).
        type Tile: Copy + AsRef<[Self::i16s]> + AsMut<[Self::i16s]>;
        /// A tile of vectors `v`.
        fn tile(self, v: Self::i16s) -> Self::Tile;
        /// The transposition of a tile in place: sample `j` of vector `i` becomes sample `i` of
        /// vector `j`.
        fn transpose(self, x: &mut Self::Tile);
        /// For vectors of more than one block: the quarters `q` (blocks) of the groups of 16
        /// samples in `x` as vector `q` of the result, the groups in their order.
        fn quarters(self, x: [Self::i16s; 4]) -> [Self::i16s; 4];
        /// The inverse of [`Radix4::quarters`].
        fn unquarters(self, x: [Self::i16s; 4]) -> [Self::i16s; 4];
    }

    /// The constants of a stage, kept in registers: LLVM would otherwise reload the splats from
    /// memory in every iteration of the loops.
    struct Consts<T: Radix4> {
        /// See [`Radix4::Swap`].
        swap: T::Swap,
        /// The offset between signed and unsigned lanes, 2^15.
        bias: T::i16s,
        /// All bits set.
        ones: T::i16s,
        /// `swap(d) * rot` is `-j d` (forward) or `j d` (inverse).
        rot: T::i16s,
        /// `swap(d) ^ minus_im` and `^ minus_re` are the offset lanes of `-j d` and `j d` for the
        /// halving differences (forward; exchanged for the inverse transform).
        minus_im: T::i16s,
        minus_re: T::i16s,
        /// The smallest twiddle part, -32767.
        floor: T::i16s,
        /// The shift of the stage.
        shift: u32,
    }

    impl<T: Radix4> Consts<T> {
        #[inline(always)]
        fn new(t: T, inverse: bool, shift: u32) -> Self {
            let alternate =
                |re: i16, im: i16| T::i16s::from_fn(t, |i| if i % 2 == 0 { re } else { im });
            let (minus_im, minus_re) = (alternate(0, -1), alternate(-1, 0));
            Self {
                swap: black_box(t.swap_pattern()),
                bias: black_box(T::i16s::splat(t, i16::MIN)),
                ones: black_box(T::i16s::splat(t, -1)),
                rot: black_box(if inverse {
                    alternate(-1, 1)
                } else {
                    alternate(1, -1)
                }),
                minus_im: black_box(if inverse { minus_re } else { minus_im }),
                minus_re: black_box(if inverse { minus_im } else { minus_re }),
                floor: black_box(T::i16s::splat(t, -i16::MAX)),
                shift,
            }
        }
    }

    /// The radix-4 butterflies of one vector of groups: `x0` and the products `b2 = x1 w^2k`,
    /// `b1 = x2 w^k`, `b3 = x3 w^3k` (the inputs are in bit-reversed order). With [`HALVING`]
    /// both levels of adds halve, else they saturate.
    #[inline(always)]
    fn butterfly<T: Radix4, const KIND: u8>(
        t: T,
        x0: T::i16s,
        b2: T::i16s,
        b1: T::i16s,
        b3: T::i16s,
        k: &Consts<T>,
    ) -> [T::i16s; 4] {
        if KIND == HALVING {
            // Offset lanes throughout: y1 = q -+ j d and y3 = q +- j d halve with the complement
            // of the part that is subtracted.
            let (x0, b2, b1, b3) = (x0 ^ k.bias, b2 ^ k.bias, b1 ^ k.bias, b3 ^ k.bias);
            let (p, q) = (t.avg_u(x0, b2), t.avg_u(x0, b2 ^ k.ones));
            let (r, d) = (t.avg_u(b1, b3), t.avg_u(b1, b3 ^ k.ones));
            let e = t.swap(d, k.swap);
            return [
                t.avg_u(p, r) ^ k.bias,
                t.avg_u(q, e ^ k.minus_im) ^ k.bias,
                t.avg_u(p, r ^ k.ones) ^ k.bias,
                t.avg_u(q, e ^ k.minus_re) ^ k.bias,
            ];
        }
        let (p, q) = (x0.saturating_add(b2), x0.saturating_sub(b2));
        let (r, d) = (b1.saturating_add(b3), b1.saturating_sub(b3));
        // |d| stays below 2^15, so its negation fits: block floating point keeps it at most
        // 16385; with scaled fixed scaling `b1` and `b3` are products with `w / 4`, at most
        // 46341 / 4 per part, so |d| <= 23171.
        let e = t.swap(d, k.swap) * k.rot;
        [
            p.saturating_add(r),
            q.saturating_add(e),
            p.saturating_sub(r),
            q.saturating_sub(e),
        ]
    }

    /// The twiddles of the butterflies of a stage from `w^k` (`w1`) and `w^2k` (`w2`; `None`:
    /// the rounded square of `w1`, cheaper than a gather), no part below -32767: for `x1`
    /// (`w^2k`), `x2` (`w^k`) and `x3` (`w^3k`, the rounded product of the other two), scaled by
    /// `2^-shift` for [`SCALED`].
    #[inline(always)]
    fn twiddles<T: Radix4, const KIND: u8>(
        t: T,
        w1: T::i16s,
        w2: Option<T::i16s>,
        inverse: bool,
        k: &Consts<T>,
    ) -> [T::Twiddle; 3] {
        let w1 = w1.max(k.floor);
        let w2 = w2
            .unwrap_or_else(|| t.cmul(w1, t.twiddle(w1, false, k.swap), k.swap))
            .max(k.floor);
        let w3 = t
            .cmul(w1, t.twiddle(w2, false, k.swap), k.swap)
            .max(k.floor);
        let (w1, w2, w3) = if KIND == SCALED {
            (
                t.scale(w1, k.shift),
                t.scale(w2, k.shift),
                t.scale(w3, k.shift),
            )
        } else {
            (w1, w2, w3)
        };
        [
            t.twiddle(w2, inverse, k.swap),
            t.twiddle(w1, inverse, k.swap),
            t.twiddle(w3, inverse, k.swap),
        ]
    }

    /// Scaling, products, butterflies and tracking of one group of four input vectors (`w`:
    /// `None` for the twiddle-free first stage).
    #[inline(always)]
    fn group<T: Radix4, const KIND: u8, const TRACK: bool>(
        t: T,
        x: [T::i16s; 4],
        w: Option<&[T::Twiddle; 3]>,
        k: &Consts<T>,
        norm: &mut T::u32s,
    ) -> [T::i16s; 4] {
        let [x0, x1, x2, x3] = x;
        let x0 = if KIND == SCALED {
            t.scale(x0, k.shift)
        } else {
            x0
        };
        let (b2, b1, b3) = if let Some(&[w2, w1, w3]) = w {
            let (b1, b3) = t.cmul2((x2, w1), (x3, w3), k.swap);
            (t.cmul(x1, w2, k.swap), b1, b3)
        } else if KIND == SCALED {
            (
                t.scale(x1, k.shift),
                t.scale(x2, k.shift),
                t.scale(x3, k.shift),
            )
        } else {
            (x1, x2, x3)
        };
        let y = butterfly::<T, KIND>(t, x0, b2, b1, b3, k);
        if TRACK {
            for v in y {
                *norm = t.max_norm(*norm, v);
            }
        }
        y
    }

    /// The groups `k0..k0 + V` of every block of a radix-4 stage with the quarter `h >= V`.
    #[inline(always)]
    fn blocks<T: Radix4, const KIND: u8, const TRACK: bool>(
        t: T,
        data: &mut [i16],
        h: usize,
        k0: usize,
        w: &[T::Twiddle; 3],
        k: &Consts<T>,
        norm: &mut T::u32s,
    ) {
        let len = T::i16s::LEN;
        let at = 2 * k0..2 * k0 + len;
        if at.end > 2 * h {
            return;
        }
        for block in data.chunks_exact_mut(8 * h) {
            let (lo, hi) = block.split_at_mut(4 * h);
            let (s0, s1) = lo.split_at_mut(2 * h);
            let (s2, s3) = hi.split_at_mut(2 * h);
            let (Some(s0), Some(s1), Some(s2), Some(s3)) = (
                s0.get_mut(at.clone()),
                s1.get_mut(at.clone()),
                s2.get_mut(at.clone()),
                s3.get_mut(at.clone()),
            ) else {
                continue;
            };
            let x = [
                T::i16s::from_slice(t, s0),
                T::i16s::from_slice(t, s1),
                T::i16s::from_slice(t, s2),
                T::i16s::from_slice(t, s3),
            ];
            let [y0, y1, y2, y3] = group::<T, KIND, TRACK>(t, x, Some(w), k, norm);
            y0.store_slice(s0);
            y1.store_slice(s1);
            y2.store_slice(s2);
            y3.store_slice(s3);
        }
    }

    /// A radix-4 stage of one block (the last one of an even number of factors 2) with the
    /// twiddle step `STEP` (1 or 2): the quarters and the twiddles walked together, without a
    /// bounds check per group.
    #[inline(always)]
    fn single_block<T: Radix4, const KIND: u8, const TRACK: bool, const STEP: usize>(
        t: T,
        data: &mut [i16],
        tw: &[i16],
        inverse: bool,
        k: &Consts<T>,
        norm: &mut T::u32s,
    ) {
        let len = T::i16s::LEN;
        let quarter = data.len() / 4;
        let (lo, hi) = data.split_at_mut(2 * quarter);
        let (q0, q1) = lo.split_at_mut(quarter);
        let (q2, q3) = hi.split_at_mut(quarter);
        let groups = q0
            .chunks_exact_mut(len)
            .zip(q1.chunks_exact_mut(len))
            .zip(q2.chunks_exact_mut(len).zip(q3.chunks_exact_mut(len)));
        let w1 = tw.chunks_exact(len * STEP);
        let w2 = tw.chunks_exact(2 * len * STEP);
        for (((s0, s1), (s2, s3)), (c1, c2)) in groups.zip(w1.zip(w2)) {
            let (w1, w2) = if STEP == 1 {
                (
                    super::twiddle_chunk::<T, 1>(t, c1),
                    super::twiddle_chunk::<T, 2>(t, c2),
                )
            } else {
                (
                    super::twiddle_chunk::<T, 2>(t, c1),
                    super::twiddle_chunk::<T, 4>(t, c2),
                )
            };
            let w = twiddles::<T, KIND>(t, w1, Some(w2), inverse, k);
            let x = [
                T::i16s::from_slice(t, s0),
                T::i16s::from_slice(t, s1),
                T::i16s::from_slice(t, s2),
                T::i16s::from_slice(t, s3),
            ];
            let [y0, y1, y2, y3] = group::<T, KIND, TRACK>(t, x, Some(&w), k, norm);
            y0.store_slice(s0);
            y1.store_slice(s1);
            y2.store_slice(s2);
            y3.store_slice(s3);
        }
    }

    /// One radix-4 stage with the quarter `h` (blocks of `4 h` samples); returns the largest
    /// squared magnitude of the output if `TRACK`.
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn stage<T: Radix4, const KIND: u8, const TRACK: bool>(
        t: T,
        data: &mut [i16],
        tw: &[i16],
        h: usize,
        step: usize,
        inverse: bool,
        shift: u32,
    ) -> u64 {
        let len = T::i16s::LEN;
        let samples = len / 2;
        let k = Consts::<T>::new(t, inverse, shift);
        let mut norm = T::u32s::splat(t, 0);
        if 8 * h == data.len() && step == 1 {
            single_block::<T, KIND, TRACK, 1>(t, data, tw, inverse, &k, &mut norm);
        } else if 8 * h == data.len() && step == 2 {
            single_block::<T, KIND, TRACK, 2>(t, data, tw, inverse, &k, &mut norm);
        } else if h >= samples {
            // The twiddle vectors of consecutive positions `k0` from consecutive chunks of the
            // table where the steps allow it (`w^k` up to step 4, `w^2k` up to step 2).
            let positions = (0..h / samples).map(|i| i * samples);
            match step {
                1 => {
                    let chunks = tw.chunks_exact(len).zip(tw.chunks_exact(2 * len));
                    for (k0, (c1, c2)) in positions.zip(chunks) {
                        let w1 = super::twiddle_chunk::<T, 1>(t, c1);
                        let w2 = super::twiddle_chunk::<T, 2>(t, c2);
                        let w = twiddles::<T, KIND>(t, w1, Some(w2), inverse, &k);
                        blocks::<T, KIND, TRACK>(t, data, h, k0, &w, &k, &mut norm);
                    }
                }
                2 => {
                    let chunks = tw.chunks_exact(2 * len).zip(tw.chunks_exact(4 * len));
                    for (k0, (c1, c2)) in positions.zip(chunks) {
                        let w1 = super::twiddle_chunk::<T, 2>(t, c1);
                        let w2 = super::twiddle_chunk::<T, 4>(t, c2);
                        let w = twiddles::<T, KIND>(t, w1, Some(w2), inverse, &k);
                        blocks::<T, KIND, TRACK>(t, data, h, k0, &w, &k, &mut norm);
                    }
                }
                4 => {
                    for (k0, c1) in positions.zip(tw.chunks_exact(4 * len)) {
                        let w1 = super::twiddle_chunk::<T, 4>(t, c1);
                        let w = twiddles::<T, KIND>(t, w1, None, inverse, &k);
                        blocks::<T, KIND, TRACK>(t, data, h, k0, &w, &k, &mut norm);
                    }
                }
                _ => {
                    for k0 in positions {
                        let w1 = super::twiddle_vector(t, tw, k0, step);
                        let w = twiddles::<T, KIND>(t, w1, None, inverse, &k);
                        blocks::<T, KIND, TRACK>(t, data, h, k0, &w, &k, &mut norm);
                    }
                }
            }
        } else {
            // `h` 4 with vectors of more than one block: four vectors at a time, regrouped so
            // that vector `q` holds the inputs `x_q` (whole blocks) of the groups; the twiddle of
            // a sample is that of its position in a block.
            let w1 = T::i16s::from_fn(t, |i| {
                tw.get(2 * (i / 2 % h) * step + i % 2).copied().unwrap_or(0)
            });
            let w = twiddles::<T, KIND>(t, w1, None, inverse, &k);
            for chunk in data.chunks_exact_mut(4 * len) {
                let (lo, hi) = chunk.split_at_mut(2 * len);
                let (s0, s1) = lo.split_at_mut(len);
                let (s2, s3) = hi.split_at_mut(len);
                let x = t.quarters([
                    T::i16s::from_slice(t, s0),
                    T::i16s::from_slice(t, s1),
                    T::i16s::from_slice(t, s2),
                    T::i16s::from_slice(t, s3),
                ]);
                let y = group::<T, KIND, TRACK>(t, x, Some(&w), &k, &mut norm);
                let [y0, y1, y2, y3] = t.unquarters(y);
                y0.store_slice(s0);
                y1.store_slice(s1);
                y2.store_slice(s2);
                y3.store_slice(s3);
            }
        }
        t.norm_bound(norm)
    }

    /// `x` with its lowest `bits` bits reversed.
    #[inline(always)]
    fn reverse(x: usize, bits: u32) -> usize {
        if bits == 0 {
            0
        } else {
            x.reverse_bits() >> (usize::BITS - bits)
        }
    }

    /// The bit reversal and the first radix-4 stage (`h` 1, no twiddles) in one pass, by tiles.
    ///
    /// With `V` samples per vector, the index of a sample is `(a, m, c)` (`a` and `c` of
    /// `log2 V` bits) and the reversed index `(rev c, rev m, rev a)`. The samples with a fixed
    /// middle part `m` form a tile of `V` vectors (rows `a`, `N / V` samples apart) that goes to
    /// the tile `rev m`, transposed with both indices reversed: output row `r` is the
    /// transposition of the rows taken in reversed order, at the reversed position. The four
    /// inputs of a group differ in the top two bits of `a`, so before the transposition the
    /// butterflies work on whole rows (`a`, `a + V/2`, `a + V/4`, `a + 3V/4` are `x0` to `x3`),
    /// and the outputs `y_q` replace them. Returns the largest squared magnitude of the output if
    /// `TRACK`.
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn first_pass<T: Radix4, const KIND: u8, const TRACK: bool>(
        t: T,
        data: &mut [i16],
        log2n: u32,
        inverse: bool,
        shift: u32,
    ) -> u64 {
        let v = T::i16s::LEN / 2;
        let mid = log2n - 2 * v.trailing_zeros();
        let k = Consts::<T>::new(t, inverse, shift);
        let mut norm = T::u32s::splat(t, 0);
        // The `V` rows of `N / V` samples; all have the same length, so that one bounds check
        // serves a whole tile. The splits cannot fail: `data` holds the `2^log2n` samples
        // (checked by the caller).
        let mut rows: [&mut [i16]; 16] = Default::default();
        let mut rest = data;
        for row in rows.iter_mut().take(v) {
            let (head, tail) = rest.split_at_mut((2 << log2n) / v);
            *row = head;
            rest = tail;
        }
        let last = (1usize << mid) - 1;
        let mut next = 0;
        for m in 0..=last {
            let mirror = next;
            // `m + 1` flips the trailing ones of `m` and the zero above them, so the reversed
            // index flips as many of its top bits (past the end: all, harmlessly).
            next ^= last ^ (last >> ((m + 1).trailing_zeros() + 1));
            if mirror < m {
                continue;
            }
            let Some(x) = tile_load(t, &rows, m) else {
                continue;
            };
            let x = tile_process::<T, KIND, TRACK>(t, x, &k, &mut norm);
            if mirror == m {
                tile_store::<T>(&mut rows, m, x);
            } else if let Some(y) = tile_load(t, &rows, mirror) {
                let y = tile_process::<T, KIND, TRACK>(t, y, &k, &mut norm);
                tile_store::<T>(&mut rows, mirror, x);
                tile_store::<T>(&mut rows, m, y);
            }
        }
        t.norm_bound(norm)
    }

    /// The tile `m` of [`first_pass`] from its rows (`None` past their end, which the caller
    /// never asks for).
    #[inline(always)]
    fn tile_load<T: Radix4>(t: T, rows: &[&mut [i16]; 16], m: usize) -> Option<T::Tile> {
        let len = T::i16s::LEN;
        let mut tile = t.tile(T::i16s::splat(t, 0));
        for (slot, row) in tile.as_mut().iter_mut().zip(rows) {
            *slot = T::i16s::from_slice(t, row.get(m * len..m * len + len)?);
        }
        Some(tile)
    }

    /// Stores the tile `m` of [`first_pass`] into its rows.
    #[inline(always)]
    fn tile_store<T: Radix4>(rows: &mut [&mut [i16]; 16], m: usize, tile: T::Tile) {
        let len = T::i16s::LEN;
        for (v, row) in tile.as_ref().iter().zip(rows) {
            if let Some(dst) = row.get_mut(m * len..m * len + len) {
                v.store_slice(dst);
            }
        }
    }

    /// The butterflies of a tile of [`first_pass`] and the transposition with both indices
    /// reversed.
    #[inline(always)]
    fn tile_process<T: Radix4, const KIND: u8, const TRACK: bool>(
        t: T,
        mut x: T::Tile,
        k: &Consts<T>,
        norm: &mut T::u32s,
    ) -> T::Tile {
        let v = T::i16s::LEN / 2;
        let q = v.trailing_zeros();
        let zero = T::i16s::splat(t, 0);
        let rows = x.as_mut();
        for a in 0..v / 4 {
            let at = [a, a + v / 2, a + v / 4, a + 3 * v / 4];
            let mut input = [zero; 4];
            for (slot, &i) in input.iter_mut().zip(&at) {
                *slot = rows.get(i).copied().unwrap_or(zero);
            }
            let y = group::<T, KIND, TRACK>(t, input, None, k, norm);
            for (i, y) in at.into_iter().zip(y) {
                if let Some(slot) = rows.get_mut(i) {
                    *slot = y;
                }
            }
        }
        let mut reversed = t.tile(zero);
        for (i, slot) in reversed.as_mut().iter_mut().enumerate() {
            *slot = rows.get(reverse(i, q)).copied().unwrap_or(zero);
        }
        let mut x = reversed;
        t.transpose(&mut x);
        let mut out = t.tile(zero);
        for (r, slot) in out.as_mut().iter_mut().enumerate() {
            *slot = x.as_ref().get(reverse(r, q)).copied().unwrap_or(zero);
        }
        out
    }

    /// The last stage of an odd number of radix-2 factors: radix 2 with `h = n / 2`.
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn last_radix2<T: Radix4, const KIND: u8>(
        t: T,
        data: &mut [i16],
        tw: &[i16],
        h: usize,
        step: usize,
        inverse: bool,
        shift: u32,
    ) {
        let len = T::i16s::LEN;
        let k = Consts::<T>::new(t, inverse, shift);
        let (lo, hi) = data.split_at_mut(2 * h);
        for (i, (a, c)) in lo
            .chunks_exact_mut(len)
            .zip(hi.chunks_exact_mut(len))
            .enumerate()
        {
            let w = super::twiddle_vector(t, tw, i * (len / 2), step).max(k.floor);
            let w = t.twiddle(
                if KIND == SCALED { t.scale(w, shift) } else { w },
                inverse,
                k.swap,
            );
            let av = T::i16s::from_slice(t, a);
            let av = if KIND == SCALED {
                t.scale(av, shift)
            } else {
                av
            };
            let b = t.cmul(T::i16s::from_slice(t, c), w, k.swap);
            let (x, y) = if KIND == HALVING {
                let (av, b) = (av ^ k.bias, b ^ k.bias);
                (t.avg_u(av, b) ^ k.bias, t.avg_u(av, b ^ k.ones) ^ k.bias)
            } else {
                (av.saturating_add(b), av.saturating_sub(b))
            };
            x.store_slice(a);
            y.store_slice(c);
        }
    }

    /// The largest squared magnitude of the samples of `data`.
    #[inline(always)]
    fn max_norm<T: Radix4>(t: T, data: &[i16]) -> u64 {
        let mut norm = T::u32s::splat(t, 0);
        for chunk in data.chunks_exact(T::i16s::LEN) {
            norm = t.max_norm(norm, T::i16s::from_slice(t, chunk));
        }
        t.norm_bound(norm)
    }

    /// The transform of the samples in natural order: the bit reversal with the first stage,
    /// then the other stages; `None` below a tile of samples (`V^2`, `V` samples per vector) or
    /// four vectors. Fixed scaling of an input beyond [`super::STAGE_LIMIT`] takes the
    /// over-range plan (see [`fixed_fits`]), which has a copy of the stages of its own, so that
    /// block floating point and fixed scaling in range run without its bookkeeping.
    #[inline(always)]
    pub(super) fn transform<T: Radix4>(
        t: T,
        data: &mut [i16],
        tw: &[i16],
        log2n: u32,
        stride: usize,
        inverse: bool,
        block_float: bool,
    ) -> Option<i32> {
        let n = 1usize << log2n;
        let v = T::i16s::LEN / 2;
        // Below 64 samples every stage halves, and the halving adds of x86 and wasm (three
        // instructions, a floor) lose 3 dB against the fallback there: those sizes take the
        // generic code, which is the fallback bit for bit. AVX2 has no path below 64 anyway.
        let halving_is_exact = cfg!(target_arch = "aarch64");
        if n < v * v || n < 4 * v || (n < 64 && !halving_is_exact) {
            return None;
        }
        let input = max_norm(t, data);
        if !block_float && !fixed_fits(input) {
            Some(stages::<T, true>(
                t, data, tw, log2n, stride, inverse, false, input,
            ))
        } else {
            Some(stages::<T, false>(
                t,
                data,
                tw,
                log2n,
                stride,
                inverse,
                block_float,
                input,
            ))
        }
    }

    /// The stages of [`transform`] for an input whose largest squared magnitude is `input`;
    /// returns the exponent. `OWED`: the over-range plan of fixed scaling, which owes a halving
    /// after the first stage (see `fixed_fits`). `#[inline(always)]` only without debug
    /// assertions: unoptimised, the two instances inlined into one frame would need more than
    /// the 2 MiB stack of a thread (each keeps a slot for every inlined value).
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn stages<T: Radix4, const OWED: bool>(
        t: T,
        data: &mut [i16],
        tw: &[i16],
        log2n: u32,
        stride: usize,
        inverse: bool,
        block_float: bool,
        input: u64,
    ) -> i32 {
        let n = 1usize << log2n;
        let mut owed = OWED;
        let mut norm = block_float.then_some(input);
        let fixed = if (64..=super::SCALED_FIXED_MAX).contains(&n) {
            SCALED
        } else {
            HALVING
        };
        let (kind, shift) = if owed {
            (SCALED, 3)
        } else {
            plan(norm, 4, fixed)
        };
        let track = block_float || owed;
        let m = match (kind, track) {
            (HALVING, _) => first_pass::<T, HALVING, false>(t, data, log2n, inverse, shift),
            (SCALED, false) => first_pass::<T, SCALED, false>(t, data, log2n, inverse, shift),
            (SCALED, true) => first_pass::<T, SCALED, true>(t, data, log2n, inverse, shift),
            (_, false) => first_pass::<T, PLAIN, false>(t, data, log2n, inverse, shift),
            (_, true) => first_pass::<T, PLAIN, true>(t, data, log2n, inverse, shift),
        };
        // The largest squared magnitude of the stage input: block floating point, or the
        // over-range plan while it owes the halving.
        let mut bound = m;
        if block_float {
            norm = Some(m);
        }
        let mut exponent = shift as i32;
        let mut h = 4;
        // The last stage of fixed scaling halves (see `HALVING`).
        while 4 * h <= n {
            let last = 4 * h == n;
            let (kind, shift) = if owed {
                let (shift, still) = owed_stage(bound, last);
                owed = still;
                (SCALED, shift)
            } else {
                plan(norm, 4, if last { HALVING } else { fixed })
            };
            let step = n / (4 * h) * stride;
            let track = (block_float || owed) && !last;
            let d = &mut *data;
            let m = match (kind, track) {
                (HALVING, _) => stage::<T, HALVING, false>(t, d, tw, h, step, inverse, shift),
                (SCALED, false) => stage::<T, SCALED, false>(t, d, tw, h, step, inverse, shift),
                (SCALED, true) => stage::<T, SCALED, true>(t, d, tw, h, step, inverse, shift),
                (_, false) => stage::<T, PLAIN, false>(t, d, tw, h, step, inverse, shift),
                (_, true) => stage::<T, PLAIN, true>(t, d, tw, h, step, inverse, shift),
            };
            if track {
                bound = m;
                if block_float {
                    norm = Some(m);
                }
            }
            exponent += shift as i32;
            h *= 4;
        }
        if h < n {
            // A halving still owed is given back here: the stage adds without scaling.
            let (kind, shift) = if owed {
                (PLAIN, 0)
            } else {
                plan(norm, 2, HALVING)
            };
            match kind {
                HALVING => last_radix2::<T, HALVING>(t, data, tw, h, stride, inverse, shift),
                SCALED => last_radix2::<T, SCALED>(t, data, tw, h, stride, inverse, shift),
                _ => last_radix2::<T, PLAIN>(t, data, tw, h, stride, inverse, shift),
            }
            exponent += shift as i32;
        }
        exponent
    }
}

/// A method of a level trait whose body runs as a `kernel!` of the level `$level` (`$t` is the
/// token).
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! method {
    ($level:ident, fn $name:ident($t:ident $(, $arg:ident: $ty:ty)*) -> $ret:ty { $($body:tt)* }) => {
        #[inline(always)]
        fn $name(self $(, $arg: $ty)*) -> $ret {
            fearless_simd::kernel!(
                #[inline(always)]
                fn kernel($t: $level $(, $arg: $ty)*) -> $ret { $($body)* }
            );
            kernel(self $(, $arg)*)
        }
    };
}

/// [`radix4::Radix4`] on x86.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m128i, __m256i, __m512i, _mm_add_epi32, _mm_adds_epi16, _mm_avg_epu16, _mm_cvtsi32_si128,
        _mm_madd_epi16, _mm_max_epi16, _mm_max_epu32, _mm_mulhrs_epi16, _mm_packs_epi32,
        _mm_set1_epi16, _mm_set1_epi32, _mm_setr_epi8, _mm_shuffle_epi8, _mm_shufflehi_epi16,
        _mm_shufflelo_epi16, _mm_sra_epi16, _mm_srai_epi32, _mm_srli_epi32, _mm_unpackhi_epi16,
        _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi16, _mm_unpacklo_epi32,
        _mm_unpacklo_epi64, _mm256_avg_epu16, _mm256_madd_epi16, _mm256_max_epu32,
        _mm256_mulhrs_epi16, _mm256_permute2x128_si256, _mm256_set1_epi16, _mm256_setr_epi8,
        _mm256_shuffle_epi8, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi32,
        _mm256_unpacklo_epi64, _mm512_avg_epu16, _mm512_broadcast_i32x4, _mm512_madd_epi16,
        _mm512_max_epu32, _mm512_mulhrs_epi16, _mm512_rol_epi32, _mm512_set1_epi16,
        _mm512_shuffle_epi8, _mm512_shuffle_i64x2, _mm512_unpackhi_epi32, _mm512_unpackhi_epi64,
        _mm512_unpacklo_epi32, _mm512_unpacklo_epi64,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128i, __m256i, __m512i, _mm_add_epi32, _mm_adds_epi16, _mm_avg_epu16, _mm_cvtsi32_si128,
        _mm_madd_epi16, _mm_max_epi16, _mm_max_epu32, _mm_mulhrs_epi16, _mm_packs_epi32,
        _mm_set1_epi16, _mm_set1_epi32, _mm_setr_epi8, _mm_shuffle_epi8, _mm_shufflehi_epi16,
        _mm_shufflelo_epi16, _mm_sra_epi16, _mm_srai_epi32, _mm_srli_epi32, _mm_unpackhi_epi16,
        _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi16, _mm_unpacklo_epi32,
        _mm_unpacklo_epi64, _mm256_avg_epu16, _mm256_madd_epi16, _mm256_max_epu32,
        _mm256_mulhrs_epi16, _mm256_permute2x128_si256, _mm256_set1_epi16, _mm256_setr_epi8,
        _mm256_shuffle_epi8, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi32,
        _mm256_unpacklo_epi64, _mm512_avg_epu16, _mm512_broadcast_i32x4, _mm512_madd_epi16,
        _mm512_max_epu32, _mm512_mulhrs_epi16, _mm512_rol_epi32, _mm512_set1_epi16,
        _mm512_shuffle_epi8, _mm512_shuffle_i64x2, _mm512_unpackhi_epi32, _mm512_unpackhi_epi64,
        _mm512_unpacklo_epi32, _mm512_unpacklo_epi64,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{
        Avx2, Avx512, SimdFrom, Sse2, Sse4_2, i16x8, i16x16, i16x32, u32x4, u32x8, u32x16,
    };

    use super::radix4::Radix4;

    /// `pshufb` patterns within a block: both lanes of every sample exchanged, the real part in
    /// both lanes, the imaginary part in both lanes.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn swap_pattern() -> __m128i {
        _mm_setr_epi8(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13)
    }
    /// See [`swap_pattern`].
    #[target_feature(enable = "sse2")]
    #[inline]
    fn re_pattern() -> __m128i {
        _mm_setr_epi8(0, 1, 0, 1, 4, 5, 4, 5, 8, 9, 8, 9, 12, 13, 12, 13)
    }
    /// See [`swap_pattern`].
    #[target_feature(enable = "sse2")]
    #[inline]
    fn im_pattern() -> __m128i {
        _mm_setr_epi8(2, 3, 2, 3, 6, 7, 6, 7, 10, 11, 10, 11, 14, 15, 14, 15)
    }

    /// The 4 x 4 transposition of the 32-bit elements of the blocks of four vectors, with the
    /// unpack instructions `$lo32`, `$hi32`, `$lo64`, `$hi64` of the vector width.
    macro_rules! transpose {
        ($x:expr, $lo32:ident, $hi32:ident, $lo64:ident, $hi64:ident) => {{
            let [x0, x1, x2, x3] = $x;
            let (t0, t1) = ($lo32(x0, x1), $lo32(x2, x3));
            let (t2, t3) = ($hi32(x0, x1), $hi32(x2, x3));
            [$lo64(t0, t1), $hi64(t0, t1), $lo64(t2, t3), $hi64(t2, t3)]
        }};
    }

    /// The 4 x 4 transposition of four vectors of one block.
    macro_rules! transpose4_128 {
        ($name:ident, $level:ident) => {
            fearless_simd::kernel!(
                #[inline(always)]
                fn $name(t: $level, x: [i16x8<$level>; 4]) -> [i16x8<$level>; 4] {
                    let [x0, x1, x2, x3] = x.map(Into::into);
                    transpose!(
                        [x0, x1, x2, x3],
                        _mm_unpacklo_epi32,
                        _mm_unpackhi_epi32,
                        _mm_unpacklo_epi64,
                        _mm_unpackhi_epi64
                    )
                    .map(|v| i16x8::simd_from(t, v))
                }
            );
        };
    }
    transpose4_128!(transpose4_sse2, Sse2);
    transpose4_128!(transpose4_sse4_2, Sse4_2);

    impl Radix4 for Sse2 {
        /// The coefficient pairs of `pmaddwd` for the real and the imaginary part.
        type Twiddle = (i16x8<Sse2>, i16x8<Sse2>);

        #[inline(always)]
        fn twiddle(self, w: i16x8<Sse2>, inverse: bool, _s: ()) -> Self::Twiddle {
            let alternate =
                |re: i16, im: i16| i16x8::from_fn(self, |i| if i % 2 == 0 { re } else { im });
            let swapped = self.swap(w, ());
            // Forward: re = (wr, -wi) . (cr, ci), im = (wi, wr) . (cr, ci); inverse: (wr, wi)
            // and (-wi, wr).
            if inverse {
                (w, swapped * alternate(-1, 1))
            } else {
                (w * alternate(1, -1), swapped)
            }
        }
        type Swap = ();
        #[inline(always)]
        fn swap_pattern(self) {}
        method!(Sse2, fn cmul(t, c: i16x8<Sse2>, w: (i16x8<Sse2>, i16x8<Sse2>), _s: ()) -> i16x8<Sse2> {
            // The two products of a part summed exactly and rounded once.
            let round = _mm_set1_epi32(1 << 14);
            let c = c.into();
            let re = _mm_srai_epi32::<15>(_mm_add_epi32(_mm_madd_epi16(c, w.0.into()), round));
            let im = _mm_srai_epi32::<15>(_mm_add_epi32(_mm_madd_epi16(c, w.1.into()), round));
            i16x8::simd_from(t, _mm_unpacklo_epi16(_mm_packs_epi32(re, re), _mm_packs_epi32(im, im)))
        });
        method!(Sse2, fn cmul2(t, a: (i16x8<Sse2>, (i16x8<Sse2>, i16x8<Sse2>)), b: (i16x8<Sse2>, (i16x8<Sse2>, i16x8<Sse2>)), _s: ()) -> (i16x8<Sse2>, i16x8<Sse2>) {
            // The real parts of both products narrowed into one vector, the imaginary parts into
            // another, then interleaved.
            let round = _mm_set1_epi32(1 << 14);
            let part = |c: __m128i, w: i16x8<Sse2>| {
                _mm_srai_epi32::<15>(_mm_add_epi32(_mm_madd_epi16(c, w.into()), round))
            };
            let (ca, cb) = (a.0.into(), b.0.into());
            let re = _mm_packs_epi32(part(ca, a.1.0), part(cb, b.1.0));
            let im = _mm_packs_epi32(part(ca, a.1.1), part(cb, b.1.1));
            (
                i16x8::simd_from(t, _mm_unpacklo_epi16(re, im)),
                i16x8::simd_from(t, _mm_unpackhi_epi16(re, im)),
            )
        });
        method!(Sse2, fn swap(t, v: i16x8<Sse2>, _s: ()) -> i16x8<Sse2> {
            i16x8::simd_from(t, _mm_shufflehi_epi16::<0xB1>(_mm_shufflelo_epi16::<0xB1>(v.into())))
        });
        method!(Sse2, fn avg_u(t, a: i16x8<Sse2>, b: i16x8<Sse2>) -> i16x8<Sse2> {
            i16x8::simd_from(t, _mm_avg_epu16(a.into(), b.into()))
        });
        method!(Sse2, fn scale(t, v: i16x8<Sse2>, shift: u32) -> i16x8<Sse2> {
            // (v + 2^(shift - 1)) >> shift; the add saturates, which rounds the few values
            // within 2^(shift - 1) of 32767 down by one.
            let round = _mm_set1_epi16(1 << (shift - 1));
            let v = _mm_adds_epi16(v.into(), round);
            i16x8::simd_from(t, _mm_sra_epi16(v, _mm_cvtsi32_si128(shift as i32)))
        });
        // SSE2 has no unsigned (or signed) 32-bit maximum: `(re^2 + im^2) >> 1 < 2^30`, so the
        // signed 16-bit maximum of its upper halves bounds it (the lower halves are ignored).
        method!(Sse2, fn max_norm(t, acc: u32x4<Sse2>, v: i16x8<Sse2>) -> u32x4<Sse2> {
            let v = v.into();
            let half = _mm_srli_epi32::<1>(_mm_madd_epi16(v, v));
            u32x4::simd_from(t, _mm_max_epi16(acc.into(), half))
        });
        #[inline(always)]
        fn norm_bound(self, acc: u32x4<Sse2>) -> u64 {
            (u64::from((acc >> 16u32).reduce_max()) + 1) << 17
        }
        type Tile = [i16x8<Sse2>; 4];
        #[inline(always)]
        fn tile(self, v: i16x8<Sse2>) -> Self::Tile {
            [v; 4]
        }
        #[inline(always)]
        fn transpose(self, x: &mut [i16x8<Sse2>; 4]) {
            *x = transpose4_sse2(self, *x);
        }
        // A vector is one block: no stage regroups whole blocks.
        #[inline(always)]
        fn quarters(self, x: [i16x8<Sse2>; 4]) -> [i16x8<Sse2>; 4] {
            x
        }
        #[inline(always)]
        fn unquarters(self, x: [i16x8<Sse2>; 4]) -> [i16x8<Sse2>; 4] {
            x
        }
    }

    /// The `pshufb` patterns of SSE4.2: swap, real parts, imaginary parts.
    type Patterns128 = (i16x8<Sse4_2>, i16x8<Sse4_2>, i16x8<Sse4_2>);

    impl Radix4 for Sse4_2 {
        /// `(wr, wr)` and `(-wi, wi)` (forward) or `(wi, -wi)` (inverse) per sample.
        type Twiddle = (i16x8<Sse4_2>, i16x8<Sse4_2>);

        method!(Sse4_2, fn twiddle(t, w: i16x8<Sse4_2>, inverse: bool, s: Patterns128) -> (i16x8<Sse4_2>, i16x8<Sse4_2>) {
            let w = w.into();
            let sign = i16x8::from_fn(t, |i| if (i % 2 == 0) == inverse { 1 } else { -1 });
            (
                i16x8::simd_from(t, _mm_shuffle_epi8(w, s.1.into())),
                i16x8::simd_from(t, _mm_shuffle_epi8(w, s.2.into())) * sign,
            )
        });
        /// The patterns of [`swap_pattern`], [`re_pattern`] and [`im_pattern`].
        type Swap = Patterns128;
        method!(Sse4_2, fn swap_pattern(t) -> Patterns128 {
            (
                i16x8::simd_from(t, swap_pattern()),
                i16x8::simd_from(t, re_pattern()),
                i16x8::simd_from(t, im_pattern()),
            )
        });
        method!(Sse4_2, fn cmul(t, c: i16x8<Sse4_2>, w: (i16x8<Sse4_2>, i16x8<Sse4_2>), s: Patterns128) -> i16x8<Sse4_2> {
            let c = c.into();
            let rr = _mm_mulhrs_epi16(c, w.0.into());
            let ii = _mm_mulhrs_epi16(_mm_shuffle_epi8(c, s.0.into()), w.1.into());
            i16x8::simd_from(t, rr).saturating_add(i16x8::simd_from(t, ii))
        });
        method!(Sse4_2, fn swap(t, v: i16x8<Sse4_2>, s: Patterns128) -> i16x8<Sse4_2> {
            i16x8::simd_from(t, _mm_shuffle_epi8(v.into(), s.0.into()))
        });
        method!(Sse4_2, fn avg_u(t, a: i16x8<Sse4_2>, b: i16x8<Sse4_2>) -> i16x8<Sse4_2> {
            i16x8::simd_from(t, _mm_avg_epu16(a.into(), b.into()))
        });
        method!(Sse4_2, fn scale(t, v: i16x8<Sse4_2>, shift: u32) -> i16x8<Sse4_2> {
            let factor = _mm_set1_epi16(1 << (15 - shift));
            i16x8::simd_from(t, _mm_mulhrs_epi16(v.into(), factor))
        });
        #[inline(always)]
        fn norm_bound(self, acc: u32x4<Sse4_2>) -> u64 {
            u64::from(acc.reduce_max())
        }
        method!(Sse4_2, fn max_norm(t, acc: u32x4<Sse4_2>, v: i16x8<Sse4_2>) -> u32x4<Sse4_2> {
            let v = v.into();
            u32x4::simd_from(t, _mm_max_epu32(acc.into(), _mm_madd_epi16(v, v)))
        });
        type Tile = [i16x8<Sse4_2>; 4];
        #[inline(always)]
        fn tile(self, v: i16x8<Sse4_2>) -> Self::Tile {
            [v; 4]
        }
        #[inline(always)]
        fn transpose(self, x: &mut [i16x8<Sse4_2>; 4]) {
            *x = transpose4_sse4_2(self, *x);
        }
        // A vector is one block: no stage regroups whole blocks.
        #[inline(always)]
        fn quarters(self, x: [i16x8<Sse4_2>; 4]) -> [i16x8<Sse4_2>; 4] {
            x
        }
        #[inline(always)]
        fn unquarters(self, x: [i16x8<Sse4_2>; 4]) -> [i16x8<Sse4_2>; 4] {
            x
        }
    }

    /// The `pshufb` pattern of [`swap_pattern`] (and its friends) in every block of a 256-bit
    /// vector.
    macro_rules! pattern256 {
        ($a:expr, $b:expr, $c:expr, $d:expr) => {
            _mm256_setr_epi8(
                $a,
                $a + 1,
                $b,
                $b + 1,
                $c,
                $c + 1,
                $d,
                $d + 1,
                $a + 8,
                $a + 9,
                $b + 8,
                $b + 9,
                $c + 8,
                $c + 9,
                $d + 8,
                $d + 9,
                $a,
                $a + 1,
                $b,
                $b + 1,
                $c,
                $c + 1,
                $d,
                $d + 1,
                $a + 8,
                $a + 9,
                $b + 8,
                $b + 9,
                $c + 8,
                $c + 9,
                $d + 8,
                $d + 9,
            )
        };
    }

    fearless_simd::kernel!(
        /// The 4 x 4 transpositions within the blocks of four vectors.
        #[inline(always)]
        fn transpose4_avx2(t: Avx2, x: [i16x16<Avx2>; 4]) -> [i16x16<Avx2>; 4] {
            let [x0, x1, x2, x3] = x.map(Into::into);
            transpose!(
                [x0, x1, x2, x3],
                _mm256_unpacklo_epi32,
                _mm256_unpackhi_epi32,
                _mm256_unpacklo_epi64,
                _mm256_unpackhi_epi64
            )
            .map(|v| i16x16::simd_from(t, v))
        }
    );

    fearless_simd::kernel!(
        /// The low blocks of `a` and `b`, and the high ones.
        #[inline(always)]
        fn exchange_avx2(
            t: Avx2,
            a: i16x16<Avx2>,
            b: i16x16<Avx2>,
        ) -> (i16x16<Avx2>, i16x16<Avx2>) {
            let (a, b) = (a.into(), b.into());
            (
                i16x16::simd_from(t, _mm256_permute2x128_si256::<0x20>(a, b)),
                i16x16::simd_from(t, _mm256_permute2x128_si256::<0x31>(a, b)),
            )
        }
    );

    /// The `pshufb` patterns of AVX2: swap, real parts, imaginary parts.
    type Patterns256 = (i16x16<Avx2>, i16x16<Avx2>, i16x16<Avx2>);

    impl Radix4 for Avx2 {
        /// `(wr, wr)` and `(-wi, wi)` (forward) or `(wi, -wi)` (inverse) per sample.
        type Twiddle = (i16x16<Avx2>, i16x16<Avx2>);

        method!(Avx2, fn twiddle(t, w: i16x16<Avx2>, inverse: bool, s: Patterns256) -> (i16x16<Avx2>, i16x16<Avx2>) {
            let w = w.into();
            let sign = i16x16::from_fn(t, |i| if (i % 2 == 0) == inverse { 1 } else { -1 });
            (
                i16x16::simd_from(t, _mm256_shuffle_epi8(w, s.1.into())),
                i16x16::simd_from(t, _mm256_shuffle_epi8(w, s.2.into())) * sign,
            )
        });
        /// The patterns of [`swap_pattern`], [`re_pattern`] and [`im_pattern`] in every block.
        type Swap = Patterns256;
        method!(Avx2, fn swap_pattern(t) -> Patterns256 {
            (
                i16x16::simd_from(t, pattern256!(2, 0, 6, 4)),
                i16x16::simd_from(t, pattern256!(0, 0, 4, 4)),
                i16x16::simd_from(t, pattern256!(2, 2, 6, 6)),
            )
        });
        method!(Avx2, fn cmul(t, c: i16x16<Avx2>, w: (i16x16<Avx2>, i16x16<Avx2>), s: Patterns256) -> i16x16<Avx2> {
            let c = c.into();
            let rr = _mm256_mulhrs_epi16(c, w.0.into());
            let swapped = _mm256_shuffle_epi8(c, s.0.into());
            let ii = _mm256_mulhrs_epi16(swapped, w.1.into());
            i16x16::simd_from(t, rr).saturating_add(i16x16::simd_from(t, ii))
        });
        method!(Avx2, fn swap(t, v: i16x16<Avx2>, s: Patterns256) -> i16x16<Avx2> {
            i16x16::simd_from(t, _mm256_shuffle_epi8(v.into(), s.0.into()))
        });
        method!(Avx2, fn avg_u(t, a: i16x16<Avx2>, b: i16x16<Avx2>) -> i16x16<Avx2> {
            i16x16::simd_from(t, _mm256_avg_epu16(a.into(), b.into()))
        });
        method!(Avx2, fn scale(t, v: i16x16<Avx2>, shift: u32) -> i16x16<Avx2> {
            let factor = _mm256_set1_epi16(1 << (15 - shift));
            i16x16::simd_from(t, _mm256_mulhrs_epi16(v.into(), factor))
        });
        #[inline(always)]
        fn norm_bound(self, acc: u32x8<Avx2>) -> u64 {
            u64::from(acc.reduce_max())
        }
        method!(Avx2, fn max_norm(t, acc: u32x8<Avx2>, v: i16x16<Avx2>) -> u32x8<Avx2> {
            let v = v.into();
            u32x8::simd_from(t, _mm256_max_epu32(acc.into(), _mm256_madd_epi16(v, v)))
        });
        type Tile = [i16x16<Avx2>; 8];
        #[inline(always)]
        fn tile(self, v: i16x16<Avx2>) -> Self::Tile {
            [v; 8]
        }
        // The 4 x 4 transpositions within the blocks of rows 0-3 and 4-7, then the blocks
        // exchanged.
        #[inline(always)]
        fn transpose(self, x: &mut [i16x16<Avx2>; 8]) {
            let [x0, x1, x2, x3, x4, x5, x6, x7] = *x;
            let [a0, a1, a2, a3] = transpose4_avx2(self, [x0, x1, x2, x3]);
            let [b0, b1, b2, b3] = transpose4_avx2(self, [x4, x5, x6, x7]);
            let (c0, d0) = exchange_avx2(self, a0, b0);
            let (c1, d1) = exchange_avx2(self, a1, b1);
            let (c2, d2) = exchange_avx2(self, a2, b2);
            let (c3, d3) = exchange_avx2(self, a3, b3);
            *x = [c0, c1, c2, c3, d0, d1, d2, d3];
        }
        // Groups of two vectors: the quarters 0, 1 are the blocks of the first one.
        method!(Avx2, fn quarters(t, x: [i16x16<Avx2>; 4]) -> [i16x16<Avx2>; 4] {
            let [x0, x1, x2, x3]: [__m256i; 4] = x.map(Into::into);
            [
                _mm256_permute2x128_si256::<0x20>(x0, x2),
                _mm256_permute2x128_si256::<0x31>(x0, x2),
                _mm256_permute2x128_si256::<0x20>(x1, x3),
                _mm256_permute2x128_si256::<0x31>(x1, x3),
            ]
            .map(|v| i16x16::simd_from(t, v))
        });
        method!(Avx2, fn unquarters(t, y: [i16x16<Avx2>; 4]) -> [i16x16<Avx2>; 4] {
            let [y0, y1, y2, y3]: [__m256i; 4] = y.map(Into::into);
            [
                _mm256_permute2x128_si256::<0x20>(y0, y1),
                _mm256_permute2x128_si256::<0x20>(y2, y3),
                _mm256_permute2x128_si256::<0x31>(y0, y1),
                _mm256_permute2x128_si256::<0x31>(y2, y3),
            ]
            .map(|v| i16x16::simd_from(t, v))
        });
    }

    /// A 128-bit `pshufb` pattern in every block of a 512-bit vector.
    #[target_feature(enable = "avx512f")]
    #[inline]
    fn pattern512(p: __m128i) -> __m512i {
        _mm512_broadcast_i32x4(p)
    }

    fearless_simd::kernel!(
        /// The 4 x 4 transpositions within the blocks of four vectors, in place.
        #[inline(always)]
        fn transpose4_avx512(
            t: Avx512,
            a: &mut i16x32<Avx512>,
            b: &mut i16x32<Avx512>,
            c: &mut i16x32<Avx512>,
            d: &mut i16x32<Avx512>,
        ) {
            let (x0, x1, x2, x3) = ((*a).into(), (*b).into(), (*c).into(), (*d).into());
            let (t0, t1) = (_mm512_unpacklo_epi32(x0, x1), _mm512_unpacklo_epi32(x2, x3));
            let (t2, t3) = (_mm512_unpackhi_epi32(x0, x1), _mm512_unpackhi_epi32(x2, x3));
            *a = i16x32::simd_from(t, _mm512_unpacklo_epi64(t0, t1));
            *b = i16x32::simd_from(t, _mm512_unpackhi_epi64(t0, t1));
            *c = i16x32::simd_from(t, _mm512_unpacklo_epi64(t2, t3));
            *d = i16x32::simd_from(t, _mm512_unpackhi_epi64(t2, t3));
        }
    );

    fearless_simd::kernel!(
        /// The 4 x 4 transposition of the blocks of four vectors, in place.
        #[inline(always)]
        fn blocks4_avx512(
            t: Avx512,
            a: &mut i16x32<Avx512>,
            b: &mut i16x32<Avx512>,
            c: &mut i16x32<Avx512>,
            d: &mut i16x32<Avx512>,
        ) {
            let (x0, x1, x2, x3) = ((*a).into(), (*b).into(), (*c).into(), (*d).into());
            let u0 = _mm512_shuffle_i64x2::<0b01_00_01_00>(x0, x1);
            let u1 = _mm512_shuffle_i64x2::<0b11_10_11_10>(x0, x1);
            let u2 = _mm512_shuffle_i64x2::<0b01_00_01_00>(x2, x3);
            let u3 = _mm512_shuffle_i64x2::<0b11_10_11_10>(x2, x3);
            *a = i16x32::simd_from(t, _mm512_shuffle_i64x2::<0b10_00_10_00>(u0, u2));
            *b = i16x32::simd_from(t, _mm512_shuffle_i64x2::<0b11_01_11_01>(u0, u2));
            *c = i16x32::simd_from(t, _mm512_shuffle_i64x2::<0b10_00_10_00>(u1, u3));
            *d = i16x32::simd_from(t, _mm512_shuffle_i64x2::<0b11_01_11_01>(u1, u3));
        }
    );

    impl Radix4 for Avx512 {
        /// `(wr, wr)` and `(-wi, wi)` (forward) or `(wi, -wi)` (inverse) per sample.
        type Twiddle = (i16x32<Avx512>, i16x32<Avx512>);

        method!(Avx512, fn twiddle(t, w: i16x32<Avx512>, inverse: bool, _s: ()) -> (i16x32<Avx512>, i16x32<Avx512>) {
            let w = w.into();
            let sign = i16x32::from_fn(t, |i| if (i % 2 == 0) == inverse { 1 } else { -1 });
            (
                i16x32::simd_from(t, _mm512_shuffle_epi8(w, pattern512(re_pattern()))),
                i16x32::simd_from(t, _mm512_shuffle_epi8(w, pattern512(im_pattern()))) * sign,
            )
        });
        type Swap = ();
        #[inline(always)]
        fn swap_pattern(self) {}
        method!(Avx512, fn cmul(t, c: i16x32<Avx512>, w: (i16x32<Avx512>, i16x32<Avx512>), _s: ()) -> i16x32<Avx512> {
            let c = c.into();
            let rr = _mm512_mulhrs_epi16(c, w.0.into());
            let ii = _mm512_mulhrs_epi16(_mm512_rol_epi32::<16>(c), w.1.into());
            i16x32::simd_from(t, rr).saturating_add(i16x32::simd_from(t, ii))
        });
        method!(Avx512, fn swap(t, v: i16x32<Avx512>, _s: ()) -> i16x32<Avx512> {
            i16x32::simd_from(t, _mm512_rol_epi32::<16>(v.into()))
        });
        method!(Avx512, fn avg_u(t, a: i16x32<Avx512>, b: i16x32<Avx512>) -> i16x32<Avx512> {
            i16x32::simd_from(t, _mm512_avg_epu16(a.into(), b.into()))
        });
        method!(Avx512, fn scale(t, v: i16x32<Avx512>, shift: u32) -> i16x32<Avx512> {
            let factor = _mm512_set1_epi16(1 << (15 - shift));
            i16x32::simd_from(t, _mm512_mulhrs_epi16(v.into(), factor))
        });
        #[inline(always)]
        fn norm_bound(self, acc: u32x16<Avx512>) -> u64 {
            u64::from(acc.reduce_max())
        }
        method!(Avx512, fn max_norm(t, acc: u32x16<Avx512>, v: i16x32<Avx512>) -> u32x16<Avx512> {
            let v = v.into();
            u32x16::simd_from(t, _mm512_max_epu32(acc.into(), _mm512_madd_epi16(v, v)))
        });
        type Tile = [i16x32<Avx512>; 16];
        #[inline(always)]
        fn tile(self, v: i16x32<Avx512>) -> Self::Tile {
            [v; 16]
        }
        // The 4 x 4 transpositions within the blocks of the rows 4g..4g + 4, then the 4 x 4
        // transpositions of the blocks of the rows j, 4 + j, 8 + j, 12 + j (see `quarters`).
        #[inline(always)]
        fn transpose(self, x: &mut [i16x32<Avx512>; 16]) {
            let [
                x0,
                x1,
                x2,
                x3,
                x4,
                x5,
                x6,
                x7,
                x8,
                x9,
                x10,
                x11,
                x12,
                x13,
                x14,
                x15,
            ] = x;
            transpose4_avx512(self, x0, x1, x2, x3);
            transpose4_avx512(self, x4, x5, x6, x7);
            transpose4_avx512(self, x8, x9, x10, x11);
            transpose4_avx512(self, x12, x13, x14, x15);
            blocks4_avx512(self, x0, x4, x8, x12);
            blocks4_avx512(self, x1, x5, x9, x13);
            blocks4_avx512(self, x2, x6, x10, x14);
            blocks4_avx512(self, x3, x7, x11, x15);
        }
        // Groups of one vector: the 4 x 4 transposition of the blocks.
        #[inline(always)]
        fn quarters(self, x: [i16x32<Avx512>; 4]) -> [i16x32<Avx512>; 4] {
            let [mut a, mut b, mut c, mut d] = x;
            blocks4_avx512(self, &mut a, &mut b, &mut c, &mut d);
            [a, b, c, d]
        }
        #[inline(always)]
        fn unquarters(self, y: [i16x32<Avx512>; 4]) -> [i16x32<Avx512>; 4] {
            self.quarters(y)
        }
    }
}

/// [`radix4::Radix4`] on wasm32 with `simd128`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::{
        i16x8_q15mulr_sat, i16x8_shuffle, i16x8_splat, i32x4_dot_i16x8, i32x4_shuffle,
        i64x2_shuffle, u16x8_avgr, u32x4_max,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{SimdFrom, WasmSimd128, i16x8, u32x4};

    use super::radix4::Radix4;

    fearless_simd::kernel!(
        /// The 4 x 4 transposition of four vectors.
        #[inline(always)]
        fn transpose4(t: WasmSimd128, x: [i16x8<WasmSimd128>; 4]) -> [i16x8<WasmSimd128>; 4] {
            let [x0, x1, x2, x3] = x.map(Into::into);
            let (t0, t1) = (
                i32x4_shuffle::<0, 4, 1, 5>(x0, x1),
                i32x4_shuffle::<0, 4, 1, 5>(x2, x3),
            );
            let (t2, t3) = (
                i32x4_shuffle::<2, 6, 3, 7>(x0, x1),
                i32x4_shuffle::<2, 6, 3, 7>(x2, x3),
            );
            [
                i64x2_shuffle::<0, 2>(t0, t1),
                i64x2_shuffle::<1, 3>(t0, t1),
                i64x2_shuffle::<0, 2>(t2, t3),
                i64x2_shuffle::<1, 3>(t2, t3),
            ]
            .map(|v| i16x8::simd_from(t, v))
        }
    );

    impl Radix4 for WasmSimd128 {
        /// `(wr, wr)` and `(-wi, wi)` (forward) or `(wi, -wi)` (inverse) per sample.
        type Twiddle = (i16x8<WasmSimd128>, i16x8<WasmSimd128>);

        method!(WasmSimd128, fn twiddle(t, w: i16x8<WasmSimd128>, inverse: bool, _s: ()) -> (i16x8<WasmSimd128>, i16x8<WasmSimd128>) {
            let w = w.into();
            let sign = i16x8::from_fn(t, |i| if (i % 2 == 0) == inverse { 1 } else { -1 });
            (
                i16x8::simd_from(t, i16x8_shuffle::<0, 0, 2, 2, 4, 4, 6, 6>(w, w)),
                i16x8::simd_from(t, i16x8_shuffle::<1, 1, 3, 3, 5, 5, 7, 7>(w, w)) * sign,
            )
        });
        type Swap = ();
        #[inline(always)]
        fn swap_pattern(self) {}
        method!(WasmSimd128, fn cmul(t, c: i16x8<WasmSimd128>, w: (i16x8<WasmSimd128>, i16x8<WasmSimd128>), _s: ()) -> i16x8<WasmSimd128> {
            let c = c.into();
            let rr = i16x8_q15mulr_sat(c, w.0.into());
            let swapped = i16x8_shuffle::<1, 0, 3, 2, 5, 4, 7, 6>(c, c);
            let ii = i16x8_q15mulr_sat(swapped, w.1.into());
            i16x8::simd_from(t, rr).saturating_add(i16x8::simd_from(t, ii))
        });
        method!(WasmSimd128, fn swap(t, v: i16x8<WasmSimd128>, _s: ()) -> i16x8<WasmSimd128> {
            let v = v.into();
            i16x8::simd_from(t, i16x8_shuffle::<1, 0, 3, 2, 5, 4, 7, 6>(v, v))
        });
        method!(WasmSimd128, fn avg_u(t, a: i16x8<WasmSimd128>, b: i16x8<WasmSimd128>) -> i16x8<WasmSimd128> {
            i16x8::simd_from(t, u16x8_avgr(a.into(), b.into()))
        });
        method!(WasmSimd128, fn scale(t, v: i16x8<WasmSimd128>, shift: u32) -> i16x8<WasmSimd128> {
            let factor = i16x8_splat(1 << (15 - shift));
            i16x8::simd_from(t, i16x8_q15mulr_sat(v.into(), factor))
        });
        #[inline(always)]
        fn norm_bound(self, acc: u32x4<WasmSimd128>) -> u64 {
            u64::from(acc.reduce_max())
        }
        method!(WasmSimd128, fn max_norm(t, acc: u32x4<WasmSimd128>, v: i16x8<WasmSimd128>) -> u32x4<WasmSimd128> {
            let v = v.into();
            u32x4::simd_from(t, u32x4_max(acc.into(), i32x4_dot_i16x8(v, v)))
        });
        type Tile = [i16x8<WasmSimd128>; 4];
        #[inline(always)]
        fn tile(self, v: i16x8<WasmSimd128>) -> Self::Tile {
            [v; 4]
        }
        #[inline(always)]
        fn transpose(self, x: &mut [i16x8<WasmSimd128>; 4]) {
            *x = transpose4(self, *x);
        }
        // A vector is one block: no stage regroups whole blocks.
        #[inline(always)]
        fn quarters(self, x: [i16x8<WasmSimd128>; 4]) -> [i16x8<WasmSimd128>; 4] {
            x
        }
        #[inline(always)]
        fn unquarters(self, x: [i16x8<WasmSimd128>; 4]) -> [i16x8<WasmSimd128>; 4] {
            x
        }
    }
}

/// The radix-4 stages on AArch64: the samples of 8 butterflies are loaded with `ld2` (real and
/// imaginary lanes); a twiddle product rounds every part (`sqrdmulh`) and sums with saturation;
/// the halving adds are `srhadd` (sums, rounding) and `shsub` (differences, truncating). The
/// first stage (no twiddles) is fused with the bit reversal: tiles of 8 x 8 samples loaded with
/// `ld2`, the butterflies on whole rows, then the transposition of the real and the imaginary
/// lanes (see [`first_pass`](neon::first_pass)). The block-float shift comes from an upper bound
/// of the largest `re^2 + im^2`: `(re^2 >> 15 + im^2 >> 15) >> 1` per sample (`sqdmulh`,
/// `shadd`), in units of 2^16.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon {
    use core::arch::aarch64::{
        int16x8_t, int16x8x2_t, vdupq_n_s16, vhaddq_s16, vhsubq_s16, vld1q_s16, vld2q_s16,
        vld4q_s16, vmaxq_s16, vmaxvq_s16, vnegq_s16, vqaddq_s16, vqdmulhq_s16, vqrdmulhq_s16,
        vqsubq_s16, vreinterpretq_s16_s32, vreinterpretq_s16_s64, vreinterpretq_s32_s16,
        vreinterpretq_s64_s16, vrhaddq_s16, vrshlq_s16, vst2q_s16, vtrn1q_s16, vtrn1q_s32,
        vtrn1q_s64, vtrn2q_s16, vtrn2q_s32, vtrn2q_s64, vzip1q_s64, vzip2q_s64,
    };

    use super::{HALVING, PLAIN, SCALED, fixed_fits, owed_stage, plan};

    /// The samples of one pair of `ld2` vectors.
    const L: usize = 8;

    /// `ld2`: real and imaginary parts of 8 interleaved samples.
    #[target_feature(enable = "neon")]
    #[inline]
    fn load2(src: &[i16; 16]) -> int16x8x2_t {
        // SAFETY: `src` is the 16 values that `vld2q_s16` reads.
        unsafe { vld2q_s16(src.as_ptr()) }
    }

    /// `st2`: the inverse of [`load2`].
    #[target_feature(enable = "neon")]
    #[inline]
    fn store2(dst: &mut [i16; 16], v: int16x8x2_t) {
        // SAFETY: `dst` is the 16 values that `vst2q_s16` writes.
        unsafe { vst2q_s16(dst.as_mut_ptr(), v) }
    }

    /// The 8 samples from `at` (in `i16`) of `s`.
    fn pair(s: &mut [i16], at: usize) -> Option<&mut [i16; 16]> {
        s.get_mut(at..).and_then(<[i16]>::first_chunk_mut::<16>)
    }

    /// `ld1` of 8 lanes.
    #[target_feature(enable = "neon")]
    #[inline]
    fn load1(src: &[i16; 8]) -> int16x8_t {
        // SAFETY: `src` is the 8 values that `vld1q_s16` reads.
        unsafe { vld1q_s16(src.as_ptr()) }
    }

    /// The 64-bit halves low, low' (`high == false`) or high, high' of `x` and `y`.
    #[target_feature(enable = "neon")]
    #[inline]
    fn zip64(x: int16x8_t, y: int16x8_t, high: bool) -> int16x8_t {
        let (x, y) = (vreinterpretq_s64_s16(x), vreinterpretq_s64_s16(y));
        vreinterpretq_s16_s64(if high {
            vzip2q_s64(x, y)
        } else {
            vzip1q_s64(x, y)
        })
    }

    /// The twiddles `w[(k0 + i % period) * step]`, `i < 8`, as real and imaginary lanes: one
    /// `ld2` for step 1, one `ld4` for step 2 (both for a period of 8), lane by lane otherwise.
    #[target_feature(enable = "neon")]
    #[inline]
    fn twiddles(tw: &[i16], k0: usize, step: usize, period: usize) -> int16x8x2_t {
        let at = 2 * k0 * step;
        if step == 1
            && period == L
            && let Some(src) = tw.get(at..).and_then(<[i16]>::first_chunk::<16>)
        {
            load2(src)
        } else if step == 2
            && period == L
            && let Some(src) = tw.get(at..).and_then(<[i16]>::first_chunk::<32>)
        {
            // SAFETY: `src` is the 32 values that `vld4q_s16` reads.
            let v = unsafe { vld4q_s16(src.as_ptr()) };
            int16x8x2_t(v.0, v.1)
        } else {
            let part = |p: usize| -> [i16; 8] {
                core::array::from_fn(|i| {
                    tw.get(2 * (k0 + i % period) * step + p)
                        .copied()
                        .unwrap_or(0)
                })
            };
            int16x8x2_t(load1(&part(0)), load1(&part(1)))
        }
    }

    /// `w` with no part below -32767.
    #[target_feature(enable = "neon")]
    #[inline]
    fn clamp(w: int16x8x2_t) -> int16x8x2_t {
        let floor = vdupq_n_s16(-i16::MAX);
        int16x8x2_t(vmaxq_s16(w.0, floor), vmaxq_s16(w.1, floor))
    }

    /// `w` conjugated for the inverse transform.
    #[target_feature(enable = "neon")]
    #[inline]
    fn direction(w: int16x8x2_t, inverse: bool) -> int16x8x2_t {
        if inverse {
            int16x8x2_t(w.0, vnegq_s16(w.1))
        } else {
            w
        }
    }

    /// The product of 8 samples with their twiddles, each part rounded, summed with saturation.
    #[target_feature(enable = "neon")]
    #[inline]
    fn cmul(c: int16x8x2_t, w: int16x8x2_t) -> int16x8x2_t {
        let (cr, ci, wr, wi) = (c.0, c.1, w.0, w.1);
        int16x8x2_t(
            vqsubq_s16(vqrdmulhq_s16(cr, wr), vqrdmulhq_s16(ci, wi)),
            vqaddq_s16(vqrdmulhq_s16(ci, wr), vqrdmulhq_s16(cr, wi)),
        )
    }

    /// The sum and the difference of one level: halving (`srhadd`, `shsub`) or saturating.
    #[target_feature(enable = "neon")]
    #[inline]
    fn sum<const HALF: bool>(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        if HALF {
            vrhaddq_s16(a, b)
        } else {
            vqaddq_s16(a, b)
        }
    }
    /// See [`sum`].
    #[target_feature(enable = "neon")]
    #[inline]
    fn dif<const HALF: bool>(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        if HALF {
            vhsubq_s16(a, b)
        } else {
            vqsubq_s16(a, b)
        }
    }

    /// `v 2^-shift` with rounding (`shift` given as the vector of `-shift`).
    #[target_feature(enable = "neon")]
    #[inline]
    fn scale(v: int16x8x2_t, minus_shift: int16x8_t) -> int16x8x2_t {
        int16x8x2_t(vrshlq_s16(v.0, minus_shift), vrshlq_s16(v.1, minus_shift))
    }

    /// An upper bound of `(re^2 + im^2) >> 16` per sample, minus 2.
    #[target_feature(enable = "neon")]
    #[inline]
    fn norm(re: int16x8_t, im: int16x8_t) -> int16x8_t {
        vhaddq_s16(vqdmulhq_s16(re, re), vqdmulhq_s16(im, im))
    }

    /// The radix-4 butterflies of 8 groups on real and imaginary lanes; `x0` and the products
    /// `b2 = x1 w^2k`, `b1 = x2 w^k`, `b3 = x3 w^3k`. `HALF`: the adds of both levels halve.
    #[target_feature(enable = "neon")]
    #[inline]
    fn butterfly<const HALF: bool>(
        x0: int16x8x2_t,
        b2: int16x8x2_t,
        b1: int16x8x2_t,
        b3: int16x8x2_t,
        inverse: bool,
    ) -> [int16x8x2_t; 4] {
        let p = int16x8x2_t(sum::<HALF>(x0.0, b2.0), sum::<HALF>(x0.1, b2.1));
        let q = int16x8x2_t(dif::<HALF>(x0.0, b2.0), dif::<HALF>(x0.1, b2.1));
        let r = int16x8x2_t(sum::<HALF>(b1.0, b3.0), sum::<HALF>(b1.1, b3.1));
        let d = int16x8x2_t(dif::<HALF>(b1.0, b3.0), dif::<HALF>(b1.1, b3.1));
        // q - j d = (q_re + d_im, q_im - d_re), q + j d = (q_re - d_im, q_im + d_re).
        let minus_j = int16x8x2_t(sum::<HALF>(q.0, d.1), dif::<HALF>(q.1, d.0));
        let plus_j = int16x8x2_t(dif::<HALF>(q.0, d.1), sum::<HALF>(q.1, d.0));
        let (y1, y3) = if inverse {
            (plus_j, minus_j)
        } else {
            (minus_j, plus_j)
        };
        [
            int16x8x2_t(sum::<HALF>(p.0, r.0), sum::<HALF>(p.1, r.1)),
            y1,
            int16x8x2_t(dif::<HALF>(p.0, r.0), dif::<HALF>(p.1, r.1)),
            y3,
        ]
    }

    /// Scaling, products, butterflies and tracking of one vector of groups.
    #[target_feature(enable = "neon")]
    #[inline]
    fn group<const KIND: u8, const TRACK: bool>(
        x: [int16x8x2_t; 4],
        w: &[int16x8x2_t; 3],
        inverse: bool,
        minus_shift: int16x8_t,
        norm_max: &mut int16x8_t,
    ) -> [int16x8x2_t; 4] {
        let [x0, x1, x2, x3] = x;
        let x0 = if KIND == SCALED {
            scale(x0, minus_shift)
        } else {
            x0
        };
        let [w2, w1, w3] = *w;
        let (b2, b1, b3) = (cmul(x1, w2), cmul(x2, w1), cmul(x3, w3));
        let y = if KIND == HALVING {
            butterfly::<true>(x0, b2, b1, b3, inverse)
        } else {
            butterfly::<false>(x0, b2, b1, b3, inverse)
        };
        if TRACK {
            for v in y {
                *norm_max = vmaxq_s16(*norm_max, norm(v.0, v.1));
            }
        }
        y
    }

    /// The twiddles of `x1`, `x2`, `x3` (`w^2k`, `w^k`, `w^3k`) of the butterflies
    /// `k0 + i % period` of the lanes `i` of a stage whose `w^k` has the table step `step`,
    /// scaled by `2^-shift` for [`SCALED`].
    #[target_feature(enable = "neon")]
    #[inline]
    fn stage_twiddles<const KIND: u8>(
        tw: &[i16],
        k0: usize,
        step: usize,
        period: usize,
        inverse: bool,
        minus_shift: int16x8_t,
    ) -> [int16x8x2_t; 3] {
        let w1 = twiddles(tw, k0, step, period);
        let w2 = (step == 1 && period == L).then(|| twiddles(tw, k0, 2, L));
        twiddle_set::<KIND>(w1, w2, inverse, minus_shift)
    }

    /// The twiddles of `x1`, `x2`, `x3` from `w^k` (`w1`) and `w^2k` (`w2`; `None`: the rounded
    /// square of `w1`, cheaper than a gather), no part below -32767, scaled by `2^-shift` for
    /// [`SCALED`].
    #[target_feature(enable = "neon")]
    #[inline]
    fn twiddle_set<const KIND: u8>(
        w1: int16x8x2_t,
        w2: Option<int16x8x2_t>,
        inverse: bool,
        minus_shift: int16x8_t,
    ) -> [int16x8x2_t; 3] {
        let w1 = clamp(w1);
        let w2 = clamp(w2.unwrap_or_else(|| cmul(w1, w1)));
        let w3 = clamp(cmul(w1, w2));
        [w2, w1, w3].map(|w| {
            let w = if KIND == SCALED {
                scale(w, minus_shift)
            } else {
                w
            };
            direction(w, inverse)
        })
    }

    /// The groups of a radix-4 stage with `h >= 8`, 8 per twiddle vector and block.
    #[target_feature(enable = "neon")]
    #[inline]
    fn blocks<const KIND: u8, const TRACK: bool>(
        data: &mut [i16],
        tw: &[i16],
        h: usize,
        step: usize,
        inverse: bool,
        minus_shift: int16x8_t,
        norm_max: &mut int16x8_t,
    ) {
        let mut k0 = 0;
        while k0 < h {
            let w = stage_twiddles::<KIND>(tw, k0, step, L, inverse, minus_shift);
            for block in data.chunks_exact_mut(8 * h) {
                let (lo, hi) = block.split_at_mut(4 * h);
                let (s0, s1) = lo.split_at_mut(2 * h);
                let (s2, s3) = hi.split_at_mut(2 * h);
                let at = 2 * k0;
                let (Some(s0), Some(s1), Some(s2), Some(s3)) =
                    (pair(s0, at), pair(s1, at), pair(s2, at), pair(s3, at))
                else {
                    continue;
                };
                group_in_place::<KIND, TRACK>([s0, s1, s2, s3], &w, inverse, minus_shift, norm_max);
            }
            k0 += L;
        }
    }

    /// The groups of a radix-4 stage with `h` 4: two groups of 16 samples at a time, regrouped
    /// with 64-bit zips; the twiddle of a lane is that of its position in a quarter.
    #[target_feature(enable = "neon")]
    #[inline]
    fn quarters<const KIND: u8, const TRACK: bool>(
        data: &mut [i16],
        tw: &[i16],
        step: usize,
        inverse: bool,
        minus_shift: int16x8_t,
        norm_max: &mut int16x8_t,
    ) {
        let w = stage_twiddles::<KIND>(tw, 0, step, 4, inverse, minus_shift);
        for chunk in data.chunks_exact_mut(64) {
            let (g0, g1) = chunk.split_at_mut(32);
            let (s0, s1) = g0.split_at_mut(16);
            let (s2, s3) = g1.split_at_mut(16);
            let (Some(s0), Some(s1), Some(s2), Some(s3)) =
                (pair(s0, 0), pair(s1, 0), pair(s2, 0), pair(s3, 0))
            else {
                continue;
            };
            // s0 holds the quarters 0, 1 of the first group, s1 the quarters 2, 3; s2, s3 those
            // of the second group.
            let (u0, u1, u2, u3) = (load2(s0), load2(s1), load2(s2), load2(s3));
            let pick = |a: int16x8x2_t, b: int16x8x2_t, high: bool| {
                int16x8x2_t(zip64(a.0, b.0, high), zip64(a.1, b.1, high))
            };
            let x = [
                pick(u0, u2, false),
                pick(u0, u2, true),
                pick(u1, u3, false),
                pick(u1, u3, true),
            ];
            let [y0, y1, y2, y3] = group::<KIND, TRACK>(x, &w, inverse, minus_shift, norm_max);
            store2(s0, pick(y0, y1, false));
            store2(s1, pick(y2, y3, false));
            store2(s2, pick(y0, y1, true));
            store2(s3, pick(y2, y3, true));
        }
    }

    /// A radix-4 stage of one block (the last one of an even number of factors 2) with the
    /// twiddle step 1: the quarters and the twiddles walked together (`ld2` of `w^k`, `ld4` of
    /// `w^2k`), without a bounds check per group.
    #[target_feature(enable = "neon")]
    #[inline]
    fn single_block<const KIND: u8, const TRACK: bool>(
        data: &mut [i16],
        tw: &[i16],
        inverse: bool,
        minus_shift: int16x8_t,
        norm_max: &mut int16x8_t,
    ) {
        let quarter = data.len() / 4;
        let (lo, hi) = data.split_at_mut(2 * quarter);
        let (q0, q1) = lo.split_at_mut(quarter);
        let (q2, q3) = hi.split_at_mut(quarter);
        let groups = q0
            .chunks_exact_mut(16)
            .zip(q1.chunks_exact_mut(16))
            .zip(q2.chunks_exact_mut(16).zip(q3.chunks_exact_mut(16)));
        let twiddle = tw.chunks_exact(16).zip(tw.chunks_exact(32));
        for (((s0, s1), (s2, s3)), (c1, c2)) in groups.zip(twiddle) {
            let (Some(s0), Some(s1), Some(s2), Some(s3), Some(c1), Some(c2)) = (
                s0.first_chunk_mut::<16>(),
                s1.first_chunk_mut::<16>(),
                s2.first_chunk_mut::<16>(),
                s3.first_chunk_mut::<16>(),
                c1.first_chunk::<16>(),
                c2.first_chunk::<32>(),
            ) else {
                continue;
            };
            // SAFETY: `c2` is the 32 values that `vld4q_s16` reads.
            let w2 = unsafe { vld4q_s16(c2.as_ptr()) };
            let w = twiddle_set::<KIND>(
                load2(c1),
                Some(int16x8x2_t(w2.0, w2.1)),
                inverse,
                minus_shift,
            );
            group_in_place::<KIND, TRACK>([s0, s1, s2, s3], &w, inverse, minus_shift, norm_max);
        }
    }

    /// [`group`] on the four rows of 8 samples `s`, in place.
    #[target_feature(enable = "neon")]
    #[inline]
    fn group_in_place<const KIND: u8, const TRACK: bool>(
        s: [&mut [i16; 16]; 4],
        w: &[int16x8x2_t; 3],
        inverse: bool,
        minus_shift: int16x8_t,
        norm_max: &mut int16x8_t,
    ) {
        let [s0, s1, s2, s3] = s;
        let x = [load2(s0), load2(s1), load2(s2), load2(s3)];
        let [y0, y1, y2, y3] = group::<KIND, TRACK>(x, w, inverse, minus_shift, norm_max);
        store2(s0, y0);
        store2(s1, y1);
        store2(s2, y2);
        store2(s3, y3);
    }

    /// One radix-4 stage with the quarter `h >= 4`; returns the largest norm of the output (see
    /// [`norm`]) if `TRACK`.
    #[target_feature(enable = "neon")]
    #[inline]
    fn stage<const KIND: u8, const TRACK: bool>(
        data: &mut [i16],
        tw: &[i16],
        h: usize,
        step: usize,
        inverse: bool,
        shift: u32,
    ) -> i16 {
        let minus_shift = vdupq_n_s16(-(shift as i16));
        let mut norm_max = vdupq_n_s16(0);
        if 8 * h == data.len() && step == 1 {
            single_block::<KIND, TRACK>(data, tw, inverse, minus_shift, &mut norm_max);
        } else if h >= L {
            blocks::<KIND, TRACK>(data, tw, h, step, inverse, minus_shift, &mut norm_max);
        } else {
            quarters::<KIND, TRACK>(data, tw, step, inverse, minus_shift, &mut norm_max);
        }
        vmaxvq_s16(norm_max)
    }

    /// The `trn1`/`trn2` pair of `a` and `b` at the element width `W` bits.
    #[target_feature(enable = "neon")]
    #[inline]
    fn trn<const W: u32>(a: int16x8_t, b: int16x8_t) -> (int16x8_t, int16x8_t) {
        match W {
            16 => (vtrn1q_s16(a, b), vtrn2q_s16(a, b)),
            32 => {
                let (a, b) = (vreinterpretq_s32_s16(a), vreinterpretq_s32_s16(b));
                (
                    vreinterpretq_s16_s32(vtrn1q_s32(a, b)),
                    vreinterpretq_s16_s32(vtrn2q_s32(a, b)),
                )
            }
            _ => {
                let (a, b) = (vreinterpretq_s64_s16(a), vreinterpretq_s64_s16(b));
                (
                    vreinterpretq_s16_s64(vtrn1q_s64(a, b)),
                    vreinterpretq_s16_s64(vtrn2q_s64(a, b)),
                )
            }
        }
    }

    /// The 8 x 8 transposition of the `i16` lanes of `x` with both indices reversed: lane `j` of
    /// vector `i` becomes lane `rev i` of vector `rev j` (`trn` at 16, 32 and 64 bits).
    #[target_feature(enable = "neon")]
    #[inline]
    fn transpose8(x: [int16x8_t; 8]) -> [int16x8_t; 8] {
        let [x0, x1, x2, x3, x4, x5, x6, x7] = x;
        // The rows in reversed order: 0, 4, 2, 6, 1, 5, 3, 7.
        let (a0, a1) = trn::<16>(x0, x4);
        let (a2, a3) = trn::<16>(x2, x6);
        let (a4, a5) = trn::<16>(x1, x5);
        let (a6, a7) = trn::<16>(x3, x7);
        let (b0, b2) = trn::<32>(a0, a2);
        let (b1, b3) = trn::<32>(a1, a3);
        let (b4, b6) = trn::<32>(a4, a6);
        let (b5, b7) = trn::<32>(a5, a7);
        let (c0, c4) = trn::<64>(b0, b4);
        let (c1, c5) = trn::<64>(b1, b5);
        let (c2, c6) = trn::<64>(b2, b6);
        let (c3, c7) = trn::<64>(b3, b7);
        // Column `j` of the reversed rows is `c_j`; it goes to row `rev j`.
        [c0, c4, c2, c6, c1, c5, c3, c7]
    }

    /// The bit reversal and the first radix-4 stage (`h` 1, no twiddles) in one pass, by tiles
    /// of 8 x 8 samples loaded with `ld2` (see `radix4::first_pass` of x86): the butterflies
    /// work on whole rows (`a`, `a + 4`, `a + 2`, `a + 6` are `x0` to `x3`), then the real and the
    /// imaginary lanes are transposed with both indices reversed. Returns the largest norm (see
    /// [`norm`]) of the output if `TRACK`.
    #[target_feature(enable = "neon")]
    #[inline]
    fn first_pass<const KIND: u8, const TRACK: bool>(
        data: &mut [i16],
        log2n: u32,
        inverse: bool,
        shift: u32,
    ) -> i16 {
        let minus_shift = vdupq_n_s16(-(shift as i16));
        let mut norm_max = vdupq_n_s16(0);
        let mid = log2n - 6;
        // The 8 rows of `N / 8` samples, all of one length. The splits cannot fail: `data` holds
        // the `2^log2n` samples (checked by the caller).
        let mut rows: [&mut [i16]; 8] = Default::default();
        let mut rest = data;
        for row in &mut rows {
            let (head, tail) = rest.split_at_mut(2 << (log2n - 3));
            *row = head;
            rest = tail;
        }
        let last = (1usize << mid) - 1;
        let mut next = 0;
        for m in 0..=last {
            // The reversed index, counted as in `radix4::first_pass`.
            let mirror = next;
            next ^= last ^ (last >> ((m + 1).trailing_zeros() + 1));
            if mirror < m {
                continue;
            }
            let Some(x) = tile_load(&rows, m) else {
                continue;
            };
            let x = tile_process::<KIND, TRACK>(x, inverse, minus_shift, &mut norm_max);
            if mirror == m {
                tile_store(&mut rows, m, x);
            } else if let Some(y) = tile_load(&rows, mirror) {
                let y = tile_process::<KIND, TRACK>(y, inverse, minus_shift, &mut norm_max);
                tile_store(&mut rows, mirror, x);
                tile_store(&mut rows, m, y);
            }
        }
        vmaxvq_s16(norm_max)
    }

    /// The tile `m` of [`first_pass`] (`None` past the end of the rows, which the caller never
    /// asks for).
    #[target_feature(enable = "neon")]
    #[inline]
    fn tile_load(rows: &[&mut [i16]; 8], m: usize) -> Option<[int16x8x2_t; 8]> {
        let zero = vdupq_n_s16(0);
        let mut tile = [int16x8x2_t(zero, zero); 8];
        for (slot, row) in tile.iter_mut().zip(rows) {
            *slot = load2(row.get(16 * m..)?.first_chunk::<16>()?);
        }
        Some(tile)
    }

    /// Stores the tile `m` of [`first_pass`].
    #[target_feature(enable = "neon")]
    #[inline]
    fn tile_store(rows: &mut [&mut [i16]; 8], m: usize, tile: [int16x8x2_t; 8]) {
        for (v, row) in tile.into_iter().zip(rows) {
            if let Some(dst) = row
                .get_mut(16 * m..)
                .and_then(<[i16]>::first_chunk_mut::<16>)
            {
                store2(dst, v);
            }
        }
    }

    /// The butterflies of a tile of [`first_pass`] and the transposition.
    #[target_feature(enable = "neon")]
    #[inline]
    fn tile_process<const KIND: u8, const TRACK: bool>(
        x: [int16x8x2_t; 8],
        inverse: bool,
        minus_shift: int16x8_t,
        norm_max: &mut int16x8_t,
    ) -> [int16x8x2_t; 8] {
        let [r0, r1, r2, r3, r4, r5, r6, r7] = x;
        let [y0, y4, y2, y6] = first_butterflies::<KIND>([r0, r4, r2, r6], inverse, minus_shift);
        let [y1, y5, y3, y7] = first_butterflies::<KIND>([r1, r5, r3, r7], inverse, minus_shift);
        let y = [y0, y1, y2, y3, y4, y5, y6, y7];
        if TRACK {
            for v in y {
                *norm_max = vmaxq_s16(*norm_max, norm(v.0, v.1));
            }
        }
        transpose_tile(y)
    }

    /// The transposition of the real and of the imaginary lanes of a tile (see
    /// [`transpose8`]).
    #[target_feature(enable = "neon")]
    #[inline]
    fn transpose_tile(y: [int16x8x2_t; 8]) -> [int16x8x2_t; 8] {
        let [y0, y1, y2, y3, y4, y5, y6, y7] = y;
        let [a0, a1, a2, a3, a4, a5, a6, a7] =
            transpose8([y0.0, y1.0, y2.0, y3.0, y4.0, y5.0, y6.0, y7.0]);
        let [b0, b1, b2, b3, b4, b5, b6, b7] =
            transpose8([y0.1, y1.1, y2.1, y3.1, y4.1, y5.1, y6.1, y7.1]);
        [
            int16x8x2_t(a0, b0),
            int16x8x2_t(a1, b1),
            int16x8x2_t(a2, b2),
            int16x8x2_t(a3, b3),
            int16x8x2_t(a4, b4),
            int16x8x2_t(a5, b5),
            int16x8x2_t(a6, b6),
            int16x8x2_t(a7, b7),
        ]
    }

    /// The twiddle-free radix-4 butterflies of [`first_pass`] (inputs, scaled for [`SCALED`],
    /// and outputs in the order `x0`, `x1`, `x2`, `x3`).
    #[target_feature(enable = "neon")]
    #[inline]
    fn first_butterflies<const KIND: u8>(
        x: [int16x8x2_t; 4],
        inverse: bool,
        minus_shift: int16x8_t,
    ) -> [int16x8x2_t; 4] {
        let [x0, x1, x2, x3] = x;
        if KIND == HALVING {
            return butterfly::<true>(x0, x1, x2, x3, inverse);
        }
        if KIND == SCALED {
            let s = |v| scale(v, minus_shift);
            return butterfly::<false>(s(x0), s(x1), s(x2), s(x3), inverse);
        }
        butterfly::<false>(x0, x1, x2, x3, inverse)
    }

    /// The last stage of an odd number of radix-2 factors: radix 2 with `h = n / 2`.
    #[target_feature(enable = "neon")]
    #[inline]
    fn last_radix2<const KIND: u8>(
        data: &mut [i16],
        tw: &[i16],
        h: usize,
        step: usize,
        inverse: bool,
        shift: u32,
    ) {
        let minus_shift = vdupq_n_s16(-(shift as i16));
        let (lo, hi) = data.split_at_mut(2 * h);
        for (i, (a, c)) in lo
            .chunks_exact_mut(16)
            .zip(hi.chunks_exact_mut(16))
            .enumerate()
        {
            let (Some(a), Some(c)) = (a.first_chunk_mut::<16>(), c.first_chunk_mut::<16>()) else {
                continue;
            };
            let w = clamp(twiddles(tw, i * L, step, L));
            let w = direction(
                if KIND == SCALED {
                    scale(w, minus_shift)
                } else {
                    w
                },
                inverse,
            );
            let av = load2(a);
            let av = if KIND == SCALED {
                scale(av, minus_shift)
            } else {
                av
            };
            let b = cmul(load2(c), w);
            if KIND == HALVING {
                store2(
                    a,
                    int16x8x2_t(sum::<true>(av.0, b.0), sum::<true>(av.1, b.1)),
                );
                store2(
                    c,
                    int16x8x2_t(dif::<true>(av.0, b.0), dif::<true>(av.1, b.1)),
                );
            } else {
                store2(
                    a,
                    int16x8x2_t(sum::<false>(av.0, b.0), sum::<false>(av.1, b.1)),
                );
                store2(
                    c,
                    int16x8x2_t(dif::<false>(av.0, b.0), dif::<false>(av.1, b.1)),
                );
            }
        }
    }

    /// The upper bound of the largest squared magnitude of the samples from a norm of [`norm`].
    fn norm_bound(n: i16) -> u64 {
        (u64::try_from(n).unwrap_or(0) + 2) << 16
    }

    /// The norm (see [`norm`]) of the largest sample of `data`.
    #[target_feature(enable = "neon")]
    #[inline]
    fn input_norm(data: &[i16]) -> i16 {
        // 32 samples per iteration (`n >= 64`).
        let mut m = vdupq_n_s16(0);
        for chunk in data.chunks_exact(64) {
            for part in chunk.chunks_exact(16) {
                if let Some(src) = part.first_chunk::<16>() {
                    let v = load2(src);
                    m = vmaxq_s16(m, norm(v.0, v.1));
                }
            }
        }
        vmaxvq_s16(m)
    }

    fearless_simd::kernel!(
        /// The transform of the samples in natural order, at least 64 (the tiles of the first
        /// pass); returns the exponent. Fixed scaling of an input beyond
        /// [`super::STAGE_LIMIT`] takes the over-range plan (see [`fixed_fits`]) in the same
        /// loop. Unlike on x86 and wasm (see `radix4::transform`), a copy of the stages of its
        /// own does not pay here: next to the stages in range it makes the compiler move the
        /// stage functions out of line (3 % more instructions for every plan), and out of line
        /// it costs the over-range plan 7 % more.
        #[inline(always)]
        pub(super) fn stages(
            t: Neon,
            data: &mut [i16],
            tw: &[i16],
            log2n: u32,
            stride: usize,
            inverse: bool,
            block_float: bool,
        ) -> i32 {
            let n = 1usize << log2n;
            let input = norm_bound(input_norm(data));
            // Fixed scaling over range owes a halving after the first stage (see
            // `fixed_fits`).
            let mut owed = !block_float && !fixed_fits(input);
            let mut norm = block_float.then_some(input);
            // Fixed scaling halves the adds (`srhadd`, `shsub` cost what the saturating adds
            // cost).
            let fixed = HALVING;
            // The largest squared magnitude of the stage input: block floating point, or the
            // over-range plan while it owes the halving.
            let mut bound = input;
            let mut exponent = 0i32;
            let mut h = 1;
            while 4 * h <= n {
                let last = 4 * h == n;
                let (kind, shift) = if owed && h == 1 {
                    (SCALED, 3)
                } else if owed {
                    let (shift, still) = owed_stage(bound, last);
                    owed = still;
                    (SCALED, shift)
                } else {
                    plan(norm, 4, fixed)
                };
                let step = n / (4 * h) * stride;
                let track = (block_float || owed) && !last;
                let d = &mut *data;
                macro_rules! run {
                    ($kind:ident, $track:literal) => {
                        if h == 1 {
                            first_pass::<$kind, $track>(d, log2n, inverse, shift)
                        } else {
                            stage::<$kind, $track>(d, tw, h, step, inverse, shift)
                        }
                    };
                }
                let m = match (kind, track) {
                    (HALVING, _) => run!(HALVING, false),
                    (SCALED, false) => run!(SCALED, false),
                    (SCALED, true) => run!(SCALED, true),
                    (_, false) => run!(PLAIN, false),
                    (_, true) => run!(PLAIN, true),
                };
                if track {
                    bound = norm_bound(m);
                    if block_float {
                        norm = Some(bound);
                    }
                }
                exponent += shift as i32;
                h *= 4;
            }
            if h < n {
                // A halving still owed is given back here: the stage adds without scaling.
                let (kind, shift) = if owed {
                    (PLAIN, 0)
                } else {
                    plan(norm, 2, fixed)
                };
                match kind {
                    HALVING => last_radix2::<HALVING>(data, tw, h, stride, inverse, shift),
                    SCALED => last_radix2::<SCALED>(data, tw, h, stride, inverse, shift),
                    _ => last_radix2::<PLAIN>(data, tw, h, stride, inverse, shift),
                }
                exponent += shift as i32;
            }
            exponent
        }
    );
}
