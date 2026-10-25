//! `rfft` and `irfft` for `f32` around the `C32` FFT of `N / 2` points of this backend. The
//! result is within the contract of `docs/kernels.md` (110 dB), as that of the complex FFT.
//!
//! `rfft` packs the `N` real samples into `N / 2` complex ones (a copy), transforms them to `Z`
//! and forms the bins from `Z[k]`, `Z[N/2 - k]` and the twiddle `w_k` of the plan. At SSE2,
//! SSE4.2, NEON and SIMD128:
//! `X[k] = (s - j q) / 2` with `s = Z[k] + conj(Z[N/2 - k])`, `t = Z[k] - conj(Z[N/2 - k])` and
//! `q = w_k t`. The bin `N/2 - k` has `conj(s)` and `conj(q)` in their place, as
//! `w_{N/2-k} = -conj(w_k)`, so one vector of products serves the bins `k .. k + 3` and their
//! mirrors `N/2 - k - 3 .. N/2 - k`: 4 lanes in split form, `Z[N/2 - k]` loaded in reversed order
//! and the mirrored bins stored in reversed order (the shuffles of [`Post`]). The mirrored bins
//! take that identity instead of the plan's `w_{N/2-k}`, and with the feature `fft-tables` the
//! twiddles of up to 4096 points come from the split static table of the span `N` of the complex
//! FFT, which holds the same values as the plan of `RealFftPlan::new` (the caveat of
//! `docs/kernels.md` for tables of `RealFftPlan::from_twiddles`). The halving is exact and the
//! products are not fused, so each bin is the fallback's formula of its inputs; the bins next to
//! `N / 4` that do not fill a vector take the fallback's loop. AVX2 and AVX-512 run that loop for
//! every bin: the compiler vectorizes it with their wider vectors, which needs fewer instructions
//! than the 4 lanes of the path.
//!
//! `irfft` is the fallback's `sdrk_core::kernels::fft::irfft_with`, compiled at the dispatched
//! level, around the same complex FFT.

use fearless_simd::Simd as SimdToken;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
use fearless_simd::{f32x4, prelude::*};
use sdrk_core::kernels::fft::{
    Direction, Fft, FftPlan, FftScale, RealFft, RealFftPlan, Scaling, irfft_with,
};
use sdrk_core::{C32, Result};

use super::c32;
use crate::Simd;
use crate::level::dispatch;
use crate::util::{check_len, internal};

/// The inner complex transform: that of [`Simd`], through the transform of any size (see
/// `c32::fft_c32`).
struct Inner;

impl Fft<C32> for Inner {
    fn fft<Tw: AsRef<[C32]>>(
        &self,
        plan: &FftPlan<C32, Tw>,
        data: &mut [C32],
        dir: Direction,
        scaling: Scaling,
    ) -> Result<FftScale> {
        c32::fft_c32(plan, data, dir, scaling, false)
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
/// The data movement of the post-processing of `rfft` on a level with a path: 4 interleaved
/// complex numbers to and from split form, in natural or in reversed order. The default methods
/// are the portable operations (NEON and SIMD128 have single-instruction shuffles for them); x86
/// replaces the reversals by `shufps`/`unpck*ps`, as SSE2 has no byte shuffle.
trait Post: SimdToken {
    /// The real and the imaginary parts of the 4 complex numbers `v`.
    #[inline(always)]
    fn split(self, v: &[f32; 8]) -> (f32x4<Self>, f32x4<Self>) {
        let (a, b) = v.split_at(4);
        f32x4::from_slice(self, a).deinterleave(f32x4::from_slice(self, b))
    }

    /// As [`Post::split`], with lane `j` holding complex number `3 - j`.
    #[inline(always)]
    fn split_rev(self, v: &[f32; 8]) -> (f32x4<Self>, f32x4<Self>) {
        let (re, im) = self.split(v);
        (re.reverse(), im.reverse())
    }

    /// Stores split parts as 4 interleaved complex numbers.
    #[inline(always)]
    fn store(self, re: f32x4<Self>, im: f32x4<Self>, out: &mut [f32; 8]) {
        let (a, b) = re.interleave(im);
        let (lo, hi) = out.split_at_mut(4);
        a.store_slice(lo);
        b.store_slice(hi);
    }

    /// As [`Post::store`], with complex number `j` from lane `3 - j`.
    #[inline(always)]
    fn store_rev(self, re: f32x4<Self>, im: f32x4<Self>, out: &mut [f32; 8]) {
        self.store(re.reverse(), im.reverse(), out);
    }
}

#[cfg(target_arch = "aarch64")]
impl Post for fearless_simd::aarch64::Neon {}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
impl Post for fearless_simd::wasm32::WasmSimd128 {}

/// The reversals of [`Post`] with `shufps` and `unpck*ps`, for every x86 level (the SSE2 token
/// also serves the levels above it, which then encode the same instructions with VEX).
#[cfg(target_arch = "x86_64")]
mod x86 {
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{__m128, _mm_shuffle_ps, _mm_unpackhi_ps, _mm_unpacklo_ps};
    use fearless_simd::prelude::*;
    use fearless_simd::{f32x4, x86::Sse2};

    use super::Post;

    /// `_MM_SHUFFLE(z, y, x, w)`: lanes `w`, `x` of the first operand, then `y`, `z` of the
    /// second.
    const fn shuffle(z: i32, y: i32, x: i32, w: i32) -> i32 {
        (z << 6) | (y << 4) | (x << 2) | w
    }
    /// The real parts of the 4 numbers of two vectors `b`, `a` (numbers 2, 3 and 0, 1), reversed.
    const RE_REV: i32 = shuffle(0, 2, 0, 2);
    /// The imaginary parts, reversed.
    const IM_REV: i32 = shuffle(1, 3, 1, 3);
    /// The two complex numbers of a vector swapped.
    const SWAP: i32 = shuffle(1, 0, 3, 2);

    fearless_simd::kernel!(
        #[inline(always)]
        fn split_rev(t: Sse2, v: &[f32; 8]) -> (f32x4<Sse2>, f32x4<Sse2>) {
            let (a, b) = v.split_at(4);
            let a: __m128 = f32x4::from_slice(t, a).into();
            let b: __m128 = f32x4::from_slice(t, b).into();
            let re = _mm_shuffle_ps::<RE_REV>(b, a);
            let im = _mm_shuffle_ps::<IM_REV>(b, a);
            (re.simd_into(t), im.simd_into(t))
        }
    );

    fearless_simd::kernel!(
        #[inline(always)]
        fn store_rev(t: Sse2, re: f32x4<Sse2>, im: f32x4<Sse2>, out: &mut [f32; 8]) {
            let (re, im): (__m128, __m128) = (re.into(), im.into());
            // Numbers 3, 2 from the upper lanes, then 1, 0 from the lower ones.
            let hi = _mm_unpackhi_ps(re, im);
            let lo = _mm_unpacklo_ps(re, im);
            let first: f32x4<Sse2> = _mm_shuffle_ps::<SWAP>(hi, hi).simd_into(t);
            let second: f32x4<Sse2> = _mm_shuffle_ps::<SWAP>(lo, lo).simd_into(t);
            let (a, b) = out.split_at_mut(4);
            first.store_slice(a);
            second.store_slice(b);
        }
    );

    impl Post for Sse2 {
        #[inline(always)]
        fn split_rev(self, v: &[f32; 8]) -> (f32x4<Self>, f32x4<Self>) {
            split_rev(self, v)
        }

        #[inline(always)]
        fn store_rev(self, re: f32x4<Self>, im: f32x4<Self>, out: &mut [f32; 8]) {
            store_rev(self, re, im, out);
        }
    }
}

/// The bin `k` from `zk = Z[k]`, `zm = Z[N/2 - k]` and its twiddle `w` (the fallback's formula;
/// bins 0 and `N/2` use `Z[0]` twice).
#[allow(
    clippy::manual_midpoint,
    reason = "`(a + b) * 0.5` is the specified, exact halving of the fallback"
)]
#[inline(always)]
fn bin(zk: C32, zm: C32, w: C32) -> C32 {
    // E = (Z[k] + conj(Z[N/2-k])) / 2, O = (Z[k] - conj(Z[N/2-k])) / 2, p = w O.
    let e = C32::new((zk.re + zm.re) * 0.5, (zk.im - zm.im) * 0.5);
    let o = C32::new((zk.re - zm.re) * 0.5, (zk.im + zm.im) * 0.5);
    let p = C32::new(w.re * o.re - w.im * o.im, w.re * o.im + w.im * o.re);
    // X[k] = E - j p.
    C32::new(e.re + p.im, e.im - p.re)
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
/// The number of vectors of [`vector_bins`] for the `h - 1` bins `1 .. h - 1`: vector `v` covers
/// the bins `4 v + 1 .. 4 v + 4` and their mirrors from `h - 4 v - 4` on, while the two do not
/// meet (`8 v + 8 < h`).
#[inline(always)]
fn vectors(bins: usize) -> usize {
    bins / 8
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
/// The bins `1 .. N/2 - 1` (`middle`) from `zs = Z[1 .. N/2]`, 4 bins and their mirrors per
/// vector (see the module documentation); `twiddles(v)` gives the parts of the twiddles of the
/// bins of vector `v`. Returns the number of bins at the front (and as many at the back) that the
/// vectors covered.
#[inline(always)]
fn vector_bins<P: Post>(
    p: P,
    zs: &[C32],
    middle: &mut [C32],
    twiddles: impl Fn(usize) -> Option<(f32x4<P>, f32x4<P>)>,
) -> usize {
    let vectors = vectors(middle.len());
    let span = 4 * vectors;
    let (half, neg_half) = (f32x4::splat(p, 0.5), f32x4::splat(p, -0.5));
    let (Some(back_at), Some(zb_at)) = (middle.len().checked_sub(span), zs.len().checked_sub(span))
    else {
        return 0;
    };
    let (front, rest) = middle.split_at_mut(back_at);
    let (Some(front), Some(back), Some(zf), Some(zb)) = (
        front.get_mut(..span),
        rest.get_mut(..span),
        zs.get(..span),
        zs.get(zb_at..),
    ) else {
        return 0;
    };
    // Slices of `vectors` blocks each; the mirrors run from the last block down.
    let (fronts, backs) = (front.as_chunks_mut::<4>().0, back.as_chunks_mut::<4>().0);
    let (zfs, zbs) = (zf.as_chunks::<4>().0, zb.as_chunks::<4>().0);
    for v in 0..vectors {
        let r = vectors - 1 - v;
        let (Some(xf), Some(xb), Some(a), Some(b), Some((wr, wi))) = (
            fronts.get_mut(v),
            backs.get_mut(r),
            zfs.get(v),
            zbs.get(r),
            twiddles(v),
        ) else {
            // Not reached (every slice holds `vectors` blocks): the bins left are scalar.
            return 4 * v;
        };
        let (ar, ai) = p.split(bytemuck::cast_ref(a));
        let (br, bi) = p.split_rev(bytemuck::cast_ref(b));
        // s = a + conj(b), t = a - conj(b), q = w t.
        let (sr, si) = (ar + br, ai - bi);
        let (tr, ti) = (ar - br, ai + bi);
        let (qr, qi) = (wr * tr - wi * ti, wr * ti + wi * tr);
        // X[k] = (s - j q) / 2; X[h - k] = (conj(s) - j conj(q)) / 2.
        p.store((sr + qi) * half, (si - qr) * half, bytemuck::cast_mut(xf));
        let mirror = ((sr - qi) * half, (si + qr) * neg_half);
        p.store_rev(mirror.0, mirror.1, bytemuck::cast_mut(xb));
    }
    span
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
/// [`vector_bins`] with the twiddles `w` of the plan (`w_k` at index `k`) or, with the feature
/// `fft-tables` and up to 4096 points, the split static table of the span `N = 2^log2n` (the same
/// values, without the deinterleaving).
#[inline(always)]
fn bins_with<P: Post>(p: P, zs: &[C32], w: &[C32], log2n: u32, middle: &mut [C32]) -> usize {
    // The twiddles of the bins `1 .. 4 V` of the `V` vectors, in blocks of 4.
    let span = 4 * vectors(middle.len());
    let table =
        c32::real_twiddles(log2n).and_then(|(re, im)| Some((re.get(1..=span)?, im.get(1..=span)?)));
    if let Some((re, im)) = table {
        let (re, im) = (re.as_chunks::<4>().0, im.as_chunks::<4>().0);
        vector_bins(p, zs, middle, |v| {
            let (re, im) = (re.get(v)?, im.get(v)?);
            Some((f32x4::from_slice(p, re), f32x4::from_slice(p, im)))
        })
    } else {
        let w = w.get(1..=span).unwrap_or_default().as_chunks::<4>().0;
        vector_bins(p, zs, middle, |v| {
            Some(p.split(bytemuck::cast_ref(w.get(v)?)))
        })
    }
}

/// [`bins_with`] with the path of the level of `simd`; 0 on a level without one. AVX2 and AVX-512
/// have none: the fallback's loop over the bins (see [`rfft_level`]) vectorizes there with their
/// wider vectors and needs fewer instructions than 4 lanes.
#[inline(always)]
fn level_bins<S: SimdToken>(
    simd: S,
    zs: &[C32],
    w: &[C32],
    log2n: u32,
    middle: &mut [C32],
) -> usize {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    if let Some(p) = level.as_sse2()
        && level.as_avx2().is_none()
    {
        return bins_with(p, zs, w, log2n, middle);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(p) = level.as_neon() {
        return bins_with(p, zs, w, log2n, middle);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(p) = level.as_wasm_simd128() {
        return bins_with(p, zs, w, log2n, middle);
    }
    let _ = (level, zs, w, log2n, middle);
    0
}

/// Copies the real samples `input` to `dst` (the scratch as floats). On x86 in the loop of the
/// fallback, which the compiler vectorizes: `copy_from_slice` would call `memcpy`, which takes
/// `rep movsb` on CPUs with fast string moves, one instruction that SDE counts per byte (4 per
/// sample). Elsewhere `memcpy` is a vector copy.
#[inline(always)]
fn pack(input: &[f32], dst: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    for (o, pair) in dst
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(input.chunks_exact(2))
    {
        if let [re, im] = *pair {
            *o = [re, im];
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    if let Some(dst) = dst.get_mut(..input.len()) {
        dst.copy_from_slice(input);
    }
}

/// `rfft` at the level of `simd` (see the module documentation).
#[inline(always)]
fn rfft_level<S: SimdToken, Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>>(
    simd: S,
    plan: &RealFftPlan<Tw, Tw2>,
    input: &[f32],
    output: &mut [C32],
    scratch: &mut [C32],
) -> Result<FftScale> {
    let n = plan.len();
    let h = n / 2;
    // The checks of the fallback, in its order.
    check_len(n, input.len())?;
    check_len(h + 1, output.len())?;
    check_len(h, scratch.len())?;
    pack(input, bytemuck::cast_slice_mut(scratch));
    c32::fft_c32(
        plan.inner(),
        scratch,
        Direction::Forward,
        Scaling::Default,
        false,
    )?;
    let err = || internal("rfft bins");
    let z: &[C32] = scratch;
    let w = plan.post_twiddles();
    let z0 = *z.first().ok_or_else(err)?;
    let (first, rest) = output.split_first_mut().ok_or_else(err)?;
    let (last, middle) = rest.split_last_mut().ok_or_else(err)?;
    *first = bin(z0, z0, *w.first().ok_or_else(err)?);
    *last = bin(z0, z0, C32::new(-1.0, 0.0));
    // Bins 1 .. N/2 - 1: `middle[i]` is bin `i + 1`, `zs[i]` is `Z[i + 1]` and `zs[h - 2 - i]` is
    // `Z[h - 1 - i]`.
    let zs = z.get(1..h).ok_or_else(err)?;
    let ws = w.get(1..h).ok_or_else(err)?;
    let done = level_bins(simd, zs, w, n.trailing_zeros(), middle);
    // The bins the vectors left (all of them without a level path) in the fallback's loop:
    // `Z[k]` forwards, `Z[N/2 - k]` backwards (`zs[h - 2 - i]` for `middle[i]`).
    let end = middle.len() - done;
    let (Some(xs), Some(zr), Some(wr)) = (
        middle.get_mut(done..end),
        zs.get(done..end),
        ws.get(done..end),
    ) else {
        return Err(err());
    };
    for (((x, &zk), &zm), &wk) in xs.iter_mut().zip(zr).zip(zr.iter().rev()).zip(wr) {
        *x = bin(zk, zm, wk);
    }
    Ok(FftScale { exponent: 0 })
}

impl RealFft for Simd {
    fn rfft<Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>>(
        &self,
        plan: &RealFftPlan<Tw, Tw2>,
        input: &[f32],
        output: &mut [C32],
        scratch: &mut [C32],
    ) -> Result<FftScale> {
        dispatch!(simd => rfft_level(simd, plan, input, output, scratch))
    }

    fn irfft<Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>>(
        &self,
        plan: &RealFftPlan<Tw, Tw2>,
        input: &[C32],
        output: &mut [f32],
        scratch: &mut [C32],
    ) -> Result<FftScale> {
        dispatch!(_simd => irfft_with(&Inner, plan, input, output, scratch))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::vec::Vec;

    use sdrk_core::C32;

    use super::{bin, level_bins};
    use crate::level::dispatch;

    /// `n` values of a fixed pseudo-random sequence in `[-1, 1)`.
    fn noise(seed: u32, n: usize) -> Vec<C32> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from((state >> 16) as u16) / 32768.0 - 1.0
        };
        (0..n).map(|_| C32::new(next(), next())).collect()
    }

    #[test]
    fn the_vector_bins_of_the_level_equal_the_scalar_formula() {
        // The bins of the level path against `bin`, bit for bit: the front bins with the plan's
        // twiddle, their mirrors with `w_(h-k) = -conj(w_k)`. Without `fft-tables` the twiddles
        // come from the given table, with it (up to 4096 points) from the static one, so `w`
        // holds the exact twiddles here.
        for log2n in 5..=13u32 {
            let h = 1usize << (log2n - 1);
            let n = 2 * h;
            let z = noise(log2n, h);
            let w: Vec<C32> = (0..h)
                .map(|k| {
                    let theta = 2.0 * core::f64::consts::PI * (k as f64 / n as f64);
                    C32::new(libm::cos(theta) as f32, -libm::sin(theta) as f32)
                })
                .collect();
            let mut got = std::vec![C32::default(); h - 1];
            let done = dispatch!(simd => level_bins(simd, &z[1..], &w, log2n, &mut got));
            for k in 1..=done {
                let want = bin(z[k], z[h - k], w[k]);
                assert_eq!(got[k - 1], want, "n={n} k={k}");
                let mirror = C32::new(-w[k].re, w[k].im);
                let want = bin(z[h - k], z[k], mirror);
                assert_eq!(got[h - k - 1], want, "n={n} mirror of k={k}");
            }
            // The levels with a path cover the bins up to about `N / 4`.
            if ["sse2", "sse4.2", "neon", "simd128"].contains(&crate::level::name()) {
                assert!(done > 0 || h < 9, "n={n}");
            }
        }
    }
}
