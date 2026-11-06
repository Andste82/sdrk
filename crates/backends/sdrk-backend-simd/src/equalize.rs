//! `dot_complex` `C32` and `equalize` (M43).
//!
//! `dot_complex` is bit-exact: the vector lanes are the eight partial sums of the contract
//! (`docs/kernels.md`), each fed in the order of the fallback, and the final tree is the one of
//! the contract. A block of eight samples (sixteen floats of `x` and of `w`) is split into its
//! real and imaginary parts, the products are formed per lane in the order of the contract
//! (`x.re w.re - x.im w.im`, `x.re w.im + x.im w.re`, no fused multiply-add) and added to the
//! lane of their partial sum. The samples after the last whole block are copied into a
//! zero-padded block: a zero sample with a zero weight gives the product `+0`, and adding `+0`
//! to a partial sum does not change it (a partial sum that starts at `+0` is never `-0` under
//! round-to-nearest), so the bits are those of the fallback.
//!
//! - Generic code (SSE2, SSE4.2, NEON, SIMD128 and the scalar level): two 128-bit vectors per
//!   component, partial sums 0 to 3 and 4 to 7; `deinterleave` of two vectors of interleaved
//!   samples gives the parts of four samples in order.
//! - AVX2 and AVX-512 (one path, `x86` below): one 256-bit vector per component.
//!   `_mm256_shuffle_ps` splits the parts within each 128-bit half (one shuffle per part
//!   instead of the two permutations of a full deinterleave), so the lanes hold the samples
//!   `0 1 4 5 2 3 6 7` of the block; the lanes are put back in the order of the partial sums
//!   before the tree. AVX-512 has no wider form: sixteen samples per vector would need sixteen
//!   partial sums, and the contract fixes eight.
//!
//! `equalize` runs the loop of `sdrk-core` (`equalize_with`) with this `dot_complex` for the
//! filter part; the weight update stays the scalar code of `sdrk-core`, which the compiler may
//! vectorize (element-wise, the same bits).

use fearless_simd::{Simd as SimdToken, f32x4, prelude::*};
use sdrk_core::kernels::dot::{DOT_COMPLEX_PARTIALS, DotComplex};
use sdrk_core::kernels::equalize::{Equalize, EqualizeState, equalize_with};
use sdrk_core::{C32, Result};

use crate::Simd;
use crate::level::dispatch_fn;
use crate::util::check_len;

/// Floats of `x` (and of `w`) per block: eight interleaved samples.
const BLOCK: usize = 2 * DOT_COMPLEX_PARTIALS;

/// The partial sums of one component, in the order of the contract.
type Partials = [f32; DOT_COMPLEX_PARTIALS];

/// The final tree of the contract: `s[k] = a[k] + a[k + 4]`, `t[k] = s[k] + s[k + 2]`,
/// `t[0] + t[1]`.
#[inline(always)]
fn tree(a: Partials) -> f32 {
    let [a0, a1, a2, a3, a4, a5, a6, a7] = a;
    let (s0, s1, s2, s3) = (a0 + a4, a1 + a5, a2 + a6, a3 + a7);
    (s0 + s2) + (s1 + s3)
}

/// The zero-padded block of the samples after the last whole block (`rest` has fewer than
/// [`BLOCK`] floats).
#[inline(always)]
fn padded(rest: &[f32]) -> [f32; BLOCK] {
    let mut b = [0.0f32; BLOCK];
    for (d, &s) in b.iter_mut().zip(rest) {
        *d = s;
    }
    b
}

/// Adds the products of one block to the partial sums: `acc[0]`/`acc[1]` the real/imaginary
/// parts of the partial sums 0 to 3, `acc[2]`/`acc[3]` those of 4 to 7.
#[inline(always)]
fn block_generic<S: SimdToken>(
    simd: S,
    acc: &mut [f32x4<S>; 4],
    x: &[f32; BLOCK],
    w: &[f32; BLOCK],
) {
    let xq: [[f32; 4]; 4] = bytemuck::cast(*x);
    let wq: [[f32; 4]; 4] = bytemuck::cast(*w);
    let [x0, x1, x2, x3] = xq.map(|v| f32x4::from_slice(simd, &v));
    let [w0, w1, w2, w3] = wq.map(|v| f32x4::from_slice(simd, &v));
    for (pair, (xa, xb, wa, wb)) in acc
        .chunks_exact_mut(2)
        .zip([(x0, x1, w0, w1), (x2, x3, w2, w3)])
    {
        let (xr, xi) = simd.deinterleave_f32x4(xa, xb);
        let (wr, wi) = simd.deinterleave_f32x4(wa, wb);
        if let [re, im] = pair {
            *re += xr * wr - xi * wi;
            *im += xr * wi + xi * wr;
        }
    }
}

/// `dot_complex` with the generic vectors (see the module documentation).
#[inline(always)]
fn dot_complex_generic<S: SimdToken>(simd: S, x: &[f32], w: &[f32]) -> (Partials, Partials) {
    let zero = f32x4::splat(simd, 0.0);
    let mut acc = [zero; 4];
    let (xb, xr) = x.as_chunks::<BLOCK>();
    let (wb, wr) = w.as_chunks::<BLOCK>();
    for (xs, ws) in xb.iter().zip(wb) {
        block_generic(simd, &mut acc, xs, ws);
    }
    if !xr.is_empty() {
        block_generic(simd, &mut acc, &padded(xr), &padded(wr));
    }
    let lanes = acc.map(|v| {
        let mut a = [0.0f32; 4];
        a.copy_from_slice(v.as_slice());
        a
    });
    let [r0, i0, r1, i1] = lanes;
    let join = |lo: [f32; 4], hi: [f32; 4]| -> Partials {
        let [a0, a1, a2, a3] = lo;
        let [a4, a5, a6, a7] = hi;
        [a0, a1, a2, a3, a4, a5, a6, a7]
    };
    (join(r0, r1), join(i0, i1))
}

/// The partial sums of `dot_complex` at the level of `simd` (`x` and `w`: `2 n` interleaved
/// floats each).
#[inline(always)]
fn dot_complex_partials<S: SimdToken>(simd: S, x: &[f32], w: &[f32]) -> (Partials, Partials) {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = simd.level().as_avx2() {
        return x86::dot_complex_avx2(t, x, w);
    }
    dot_complex_generic(simd, x, w)
}

#[inline(always)]
fn dot_complex_c32<S: SimdToken>(simd: S, x: &[f32], w: &[f32], out: &mut (f32, f32)) {
    let (re, im) = dot_complex_partials(simd, x, w);
    *out = (tree(re), tree(im));
}

dispatch_fn!(fn run_dot_complex(x: &[f32], w: &[f32], out: &mut (f32, f32)) => dot_complex_c32);

impl DotComplex<C32> for Simd {
    #[inline]
    fn dot_complex(&self, x: &[C32], w: &[C32]) -> Result<C32> {
        check_len(x.len(), w.len())?;
        let mut out = (0.0, 0.0);
        run_dot_complex(bytemuck::cast_slice(x), bytemuck::cast_slice(w), &mut out);
        Ok(C32::new(out.0, out.1))
    }
}

impl Equalize for Simd {
    fn equalize<W, H>(
        &self,
        s: &mut EqualizeState<W, H>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()>
    where
        W: AsRef<[C32]> + AsMut<[C32]>,
        H: AsRef<[C32]> + AsMut<[C32]>,
    {
        equalize_with(&Simd, s, input, output)
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    //! The AVX2 path of `dot_complex` (also run at AVX-512): one 256-bit vector per component.

    use core::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_shuffle_ps, _mm256_sub_ps,
    };
    use fearless_simd::f32x8;
    use fearless_simd::prelude::*;

    use super::{BLOCK, Partials, padded};

    // `dot_complex_avx2` returns the partial sums `(re, im)` in the order of the contract.

    fearless_simd::kernel!(
        pub(super) fn dot_complex_avx2(t: Avx2, x: &[f32], w: &[f32]) -> (Partials, Partials) {
            let mut re = _mm256_setzero_ps();
            let mut im = _mm256_setzero_ps();
            // Lanes `0 1 4 5 2 3 6 7` of the block: within each 128-bit half the even floats
            // (`0x88`) and the odd floats (`0xdd`) of the two halves of the block.
            let mut block = |xs: &[f32; BLOCK], ws: &[f32; BLOCK]| {
                let (xa, xb) = xs.split_at(BLOCK / 2);
                let (wa, wb) = ws.split_at(BLOCK / 2);
                let xa: __m256 = f32x8::from_slice(t, xa).into();
                let xb: __m256 = f32x8::from_slice(t, xb).into();
                let wa: __m256 = f32x8::from_slice(t, wa).into();
                let wb: __m256 = f32x8::from_slice(t, wb).into();
                let xr = _mm256_shuffle_ps::<0x88>(xa, xb);
                let xi = _mm256_shuffle_ps::<0xdd>(xa, xb);
                let wr = _mm256_shuffle_ps::<0x88>(wa, wb);
                let wi = _mm256_shuffle_ps::<0xdd>(wa, wb);
                let pr = _mm256_sub_ps(_mm256_mul_ps(xr, wr), _mm256_mul_ps(xi, wi));
                let pi = _mm256_add_ps(_mm256_mul_ps(xr, wi), _mm256_mul_ps(xi, wr));
                re = _mm256_add_ps(re, pr);
                im = _mm256_add_ps(im, pi);
            };
            let (xb, xr) = x.as_chunks::<BLOCK>();
            let (wb, wr) = w.as_chunks::<BLOCK>();
            for (xs, ws) in xb.iter().zip(wb) {
                block(xs, ws);
            }
            if !xr.is_empty() {
                block(&padded(xr), &padded(wr));
            }
            let order = |v: __m256| -> Partials {
                let v = f32x8::simd_from(t, v);
                let mut a = [0.0f32; 8];
                a.copy_from_slice(v.as_slice());
                let [l0, l1, l2, l3, l4, l5, l6, l7] = a;
                // Lane j holds partial sum `[0, 1, 4, 5, 2, 3, 6, 7][j]`.
                [l0, l1, l4, l5, l2, l3, l6, l7]
            };
            (order(re), order(im))
        }
    );
}
