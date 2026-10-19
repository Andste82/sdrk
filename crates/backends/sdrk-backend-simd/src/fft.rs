//! The complex FFT for `C32` and `CQ15`.
//!
//! # `C32`
//!
//! A decimation-in-time transform in place, on the same bit-reversed order as the fallback:
//!
//! 1. bit-reversal permutation;
//! 2. conversion to blocks of `L` complex numbers stored as `L` real parts followed by `L`
//!    imaginary parts (`L` is the number of `f32` lanes), and the first `log2 L` radix-2
//!    stages inside each block with lane permutations;
//! 3. the remaining stages as radix-2² passes (two radix-2 stages fused into a four-point
//!    butterfly with three twiddle sets, one extra radix-2 pass first when the number of stages is
//!    odd), vectorised over `L` consecutive butterflies; the twiddles of `L` consecutive
//!    butterflies are the product of one table entry and a per-pass vector of table entries;
//! 4. conversion back to interleaved samples and, for the inverse transform, the scale `1/N`.
//!
//! The result is within the contract (110 dB) of the exact transform, not bit-identical with the
//! fallback: the stages are fused, FMA is used where the level has it, and twiddles are products
//! of two table entries.
//!
//! # `CQ15`
//!
//! The same radix-2 stages and integer arithmetic as the fallback (the twiddle products and
//! sums of one butterfly fit `i32` for twiddles made by `FftPlan::new`), vectorised where the
//! butterfly span is at least the vector width. The result is bit-identical with the fallback,
//! including the block-floating-point exponent.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, FftScale, Scaling};
use sdrk_core::num::{round_shift, sat16};
use sdrk_core::{C32, CQ15, Error, Q15, Result};

use crate::Simd;
use crate::level::dispatch;
use crate::util::check_len;

// ---------------------------------------------------------------------------------------------
// C32

/// Swaps `data[i]` and `data[rev(i)]` for `i < rev(i)`, where `rev` reverses the lowest `log2n`
/// bits (scalar; used for sizes too small for the tile method and as its reference).
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

/// [`bit_reverse`] for complex samples stored as pairs of floats.
fn bit_reverse_pairs(data: &mut [f32], log2n: u32) {
    let n = data.len() / 2;
    if n != 1usize << log2n {
        return;
    }
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            for k in 0..2 {
                data.swap(2 * i + k, 2 * j + k);
            }
        }
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

/// In-place bit-reversal permutation of `n = 2^b` elements of `$esize` floats/lanes each, by
/// tiles: with `V` elements per vector, an index is `(a, m, c)` (`a` and `c` of `log2 V` bits)
/// and its reversed index is `(rev c, rev m, rev a)`. The elements with a fixed middle part `m`
/// form a `V x V` tile of `V` vector loads (rows `a`, stride `n / V`); the tile goes to the tile
/// of `rev m`, transposed and reversed in both directions by the zip network of
/// [`tile_network`]. About 1.5 instructions per element instead of 30 for the scalar loop.
macro_rules! bit_reverse_tiles {
    ($name:ident, $lane:ty, $vec:ty, $row:ty, $elem_lanes:expr) => {
        #[inline(always)]
        fn $name<S: SimdToken>(simd: S, data: &mut [$lane], log2n: u32) -> bool {
            let v = <$vec>::LEN;
            let q = v.trailing_zeros();
            if log2n < 2 * q || data.len() != $elem_lanes << log2n {
                return false;
            }
            let mid_bits = log2n - 2 * q;
            let stride = $elem_lanes * ((1usize << log2n) / v); // lanes between rows
            let width = $elem_lanes * v; // lanes of one row
            let zero = <$row>::splat(simd, Default::default());
            // Local macros, not closures: the vector operations must be inlined into the
            // function that carries the target features.
            macro_rules! load {
                ($tile:expr, $rows:expr) => {
                    for (a, slot) in $rows.iter_mut().take(v).enumerate() {
                        let off = a * stride + $tile * width;
                        let row = data
                            .get(off..off + width)
                            .map_or(zero, |s| <$row>::from_slice(simd, s));
                        *slot = row.bitcast::<$vec>();
                    }
                };
            }
            macro_rules! store {
                ($tile:expr, $rows:expr) => {
                    for (a, row) in $rows.iter().take(v).enumerate() {
                        let off = a * stride + $tile * width;
                        if let Some(dst) = data.get_mut(off..off + width) {
                            row.bitcast::<$row>().store_slice(dst);
                        }
                    }
                };
            }
            let blank = zero.bitcast::<$vec>();
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
    };
}

// `C32`: elements of two floats, `V = f64 lanes`; `CQ15`: elements of two `i16`, `V = u32 lanes`.
bit_reverse_tiles!(bit_reverse_c32, f32, S::f64s, S::f32s, 2);
bit_reverse_tiles!(bit_reverse_cq15, i16, S::u32s, S::i16s, 2);

/// `(re, im)` of table entry `idx` (zero outside the table).
#[inline(always)]
fn table(tw: &[f32], idx: usize) -> (f32, f32) {
    match tw.get(2 * idx..2 * idx + 2) {
        Some([re, im]) => (*re, *im),
        _ => (0.0, 0.0),
    }
}

type Cv<S> = (<S as SimdToken>::f32s, <S as SimdToken>::f32s);

/// `w * a` for complex vectors in split form.
#[inline(always)]
fn cmul<S: SimdToken>(w: Cv<S>, a: Cv<S>) -> Cv<S> {
    let re = w.0.mul_sub(a.0, w.1 * a.1);
    let im = w.0.mul_add(a.1, w.1 * a.0);
    (re, im)
}

#[inline(always)]
fn cadd<S: SimdToken>(a: Cv<S>, b: Cv<S>) -> Cv<S> {
    (a.0 + b.0, a.1 + b.1)
}

#[inline(always)]
fn csub<S: SimdToken>(a: Cv<S>, b: Cv<S>) -> Cv<S> {
    (a.0 - b.0, a.1 - b.1)
}

/// The vector pair of the block that begins at `slice[0]` (`L` real parts, then `L` imaginary
/// parts).
#[inline(always)]
fn load_block<S: SimdToken>(simd: S, slice: &[f32]) -> Cv<S> {
    let l = S::f32s::LEN;
    let (re, im) = slice.split_at(l.min(slice.len()));
    (
        S::f32s::from_slice(simd, re),
        S::f32s::from_slice(simd, im.get(..l).unwrap_or(&[])),
    )
}

#[inline(always)]
fn store_block<S: SimdToken>(v: Cv<S>, slice: &mut [f32]) {
    let l = S::f32s::LEN;
    let (re, im) = slice.split_at_mut(l.min(slice.len()));
    v.0.store_slice(re);
    if let Some(im) = im.get_mut(..l) {
        v.1.store_slice(im);
    }
}

/// The table entry `idx` as one 64-bit word (real part in the low half), so that a twiddle is
/// loaded with one scalar load.
#[inline(always)]
fn table_bits(tw: &[f32], idx: usize) -> u64 {
    let bytes: &[u8] = bytemuck::cast_slice(tw);
    bytes
        .get(8 * idx..8 * idx + 8)
        .map_or(0, bytemuck::pod_read_unaligned::<u64>)
}

/// The twiddles `tw[(k0 + i) * step]` for the lanes `i` as split vectors; the imaginary part is
/// negated for the inverse transform. Each half of the lanes is gathered as 64-bit words
/// (real and imaginary part together) and then separated by one deinterleave.
#[inline(always)]
fn gather_twiddles<S: SimdToken>(
    simd: S,
    tw: &[f32],
    k0: usize,
    step: usize,
    inverse: bool,
) -> Cv<S> {
    let v = S::f64s::LEN;
    let lo = S::f64s::from_fn(simd, |i| f64::from_bits(table_bits(tw, (k0 + i) * step)));
    let hi = S::f64s::from_fn(simd, |i| {
        f64::from_bits(table_bits(tw, (k0 + v + i) * step))
    });
    let (re, im) = lo
        .bitcast::<S::f32s>()
        .deinterleave(hi.bitcast::<S::f32s>());
    (re, if inverse { -im } else { im })
}

/// Permutation constants of one in-block stage with partner distance `D`.
struct InBlock<S: SimdToken> {
    lo: <S::f32s as SimdBase<S>>::Mask,
    /// Twiddle vectors with the sign of the butterfly folded in: `+w` on the lanes of the upper
    /// input, `-w` on the lanes of the lower input.
    tr: S::f32s,
    ti: S::f32s,
    /// `-ti`.
    nti: S::f32s,
}

#[inline(always)]
fn in_block_consts<S: SimdToken, const D: usize>(
    simd: S,
    tw: &[f32],
    n: usize,
    stride: usize,
    inverse: bool,
) -> InBlock<S> {
    let sign = if inverse { -1.0 } else { 1.0 };
    let step = n / (2 * D) * stride;
    let w = |lane: usize| table(tw, (lane & (D - 1)) * step);
    let side = |lane: usize| if lane & D == 0 { 1.0f32 } else { -1.0 };
    let tr = S::f32s::from_fn(simd, |l| side(l) * w(l).0);
    let ti = S::f32s::from_fn(simd, |l| side(l) * sign * w(l).1);
    let lo = S::f32s::from_fn(simd, side).simd_gt(0.0f32);
    InBlock {
        lo,
        tr,
        ti,
        nti: -ti,
    }
}

/// The partner lanes `l ^ D` of a vector.
#[inline(always)]
fn partner<S: SimdToken, const D: usize>(c: &InBlock<S>, v: S::f32s) -> S::f32s {
    c.lo.select(
        v.rotate_elements_left::<D>(),
        v.rotate_elements_right::<D>(),
    )
}

/// One radix-2 stage between lanes `l` and `l ^ D` of a block.
#[inline(always)]
fn in_block_stage<S: SimdToken, const D: usize>(c: &InBlock<S>, x: Cv<S>) -> Cv<S> {
    let (pr, pi) = (partner::<S, D>(c, x.0), partner::<S, D>(c, x.1));
    // `a` holds the lower input of the pair, `b` the upper input, in both lanes of the pair.
    let (ar, ai) = (c.lo.select(x.0, pr), c.lo.select(x.1, pi));
    let (br, bi) = (c.lo.select(pr, x.0), c.lo.select(pi, x.1));
    (
        c.nti.mul_add(bi, c.tr.mul_add(br, ar)),
        c.ti.mul_add(br, c.tr.mul_add(bi, ai)),
    )
}

/// The first `log2 L` stages and the conversion to the block layout.
#[inline(always)]
fn first_stages<S: SimdToken>(
    simd: S,
    data: &mut [f32],
    tw: &[f32],
    n: usize,
    stride: usize,
    inverse: bool,
) {
    let l = S::f32s::LEN;
    let c1 = in_block_consts::<S, 1>(simd, tw, n, stride, inverse);
    let c2 = in_block_consts::<S, 2>(simd, tw, n, stride, inverse);
    let c4 = in_block_consts::<S, 4>(simd, tw, n, stride, inverse);
    let c8 = in_block_consts::<S, 8>(simd, tw, n, stride, inverse);
    for chunk in data.chunks_exact_mut(2 * l) {
        let (a, b) = chunk.split_at_mut(l);
        let (re, im) = S::f32s::from_slice(simd, a).deinterleave(S::f32s::from_slice(simd, b));
        let mut x = (re, im);
        x = in_block_stage::<S, 1>(&c1, x);
        if l > 2 {
            x = in_block_stage::<S, 2>(&c2, x);
        }
        if l > 4 {
            x = in_block_stage::<S, 4>(&c4, x);
        }
        if l > 8 {
            x = in_block_stage::<S, 8>(&c8, x);
        }
        x.0.store_slice(a);
        x.1.store_slice(b);
    }
}

/// One radix-2 stage with half size `m` (a multiple of `L`) on block-layout data.
#[inline(always)]
fn pass2<S: SimdToken>(
    simd: S,
    data: &mut [f32],
    tw: &[f32],
    n: usize,
    stride: usize,
    m: usize,
    inverse: bool,
) {
    let l = S::f32s::LEN;
    let step = n / (2 * m) * stride;
    for k0 in (0..m).step_by(l) {
        let w = gather_twiddles::<S>(simd, tw, k0, step, inverse);
        for block in data.chunks_exact_mut(4 * m) {
            let (lo, hi) = block.split_at_mut(2 * m);
            let (Some(a), Some(b)) = (
                lo.get_mut(2 * k0..2 * k0 + 2 * l),
                hi.get_mut(2 * k0..2 * k0 + 2 * l),
            ) else {
                continue;
            };
            let x0 = load_block::<S>(simd, a);
            let t = cmul::<S>(w, load_block::<S>(simd, b));
            store_block::<S>(cadd::<S>(x0, t), a);
            store_block::<S>(csub::<S>(x0, t), b);
        }
    }
}

/// Two fused radix-2 stages with half sizes `m` and `2 m` (`m` a multiple of `L`).
#[inline(always)]
fn pass22<S: SimdToken>(
    simd: S,
    data: &mut [f32],
    tw: &[f32],
    n: usize,
    stride: usize,
    m: usize,
    inverse: bool,
) {
    let l = S::f32s::LEN;
    let step2 = n / (4 * m) * stride;
    let step1 = n / (2 * m) * stride;
    for k0 in (0..m).step_by(l) {
        let w2 = gather_twiddles::<S>(simd, tw, k0, step2, inverse);
        let w1 = gather_twiddles::<S>(simd, tw, k0, step1, inverse);
        for block in data.chunks_exact_mut(8 * m) {
            let (h01, h23) = block.split_at_mut(4 * m);
            let (q0, q1) = h01.split_at_mut(2 * m);
            let (q2, q3) = h23.split_at_mut(2 * m);
            let (Some(s0), Some(s1), Some(s2), Some(s3)) = (
                q0.get_mut(2 * k0..2 * k0 + 2 * l),
                q1.get_mut(2 * k0..2 * k0 + 2 * l),
                q2.get_mut(2 * k0..2 * k0 + 2 * l),
                q3.get_mut(2 * k0..2 * k0 + 2 * l),
            ) else {
                continue;
            };
            let a0 = load_block::<S>(simd, s0);
            let a1 = load_block::<S>(simd, s1);
            let a2 = load_block::<S>(simd, s2);
            let a3 = load_block::<S>(simd, s3);
            let t1 = cmul::<S>(w1, a1);
            let t3 = cmul::<S>(w1, a3);
            let (b0, b1) = (cadd::<S>(a0, t1), csub::<S>(a0, t1));
            let (b2, b3) = (cadd::<S>(a2, t3), csub::<S>(a2, t3));
            let t2 = cmul::<S>(w2, b2);
            let u = cmul::<S>(w2, b3);
            let y0 = cadd::<S>(b0, t2);
            let y2 = csub::<S>(b0, t2);
            // y1 = b1 -+ j u, y3 = b1 +- j u (forward: -j).
            let (p, q) = if inverse {
                ((b1.0 - u.1, b1.1 + u.0), (b1.0 + u.1, b1.1 - u.0))
            } else {
                ((b1.0 + u.1, b1.1 - u.0), (b1.0 - u.1, b1.1 + u.0))
            };
            store_block::<S>(y0, s0);
            store_block::<S>(p, s1);
            store_block::<S>(y2, s2);
            store_block::<S>(q, s3);
        }
    }
}

/// Block layout back to interleaved samples, with the scale `g`.
#[inline(always)]
fn last_pass<S: SimdToken>(simd: S, data: &mut [f32], scale: Option<f32>) {
    let l = S::f32s::LEN;
    for chunk in data.chunks_exact_mut(2 * l) {
        let (mut re, mut im) = load_block::<S>(simd, chunk);
        if let Some(g) = scale {
            re *= g;
            im *= g;
        }
        let (o0, o1) = re.interleave(im);
        let (a, b) = chunk.split_at_mut(l);
        o0.store_slice(a);
        o1.store_slice(b);
    }
}

#[inline(always)]
fn fft_stages<S: SimdToken>(
    simd: S,
    data: &mut [f32],
    tw: &[f32],
    n: usize,
    stride: usize,
    inverse: bool,
) {
    let l = S::f32s::LEN;
    if !bit_reverse_c32::<S>(simd, data, n.trailing_zeros()) {
        bit_reverse_pairs(data, n.trailing_zeros());
    }
    first_stages::<S>(simd, data, tw, n, stride, inverse);
    let remaining = (n / l).trailing_zeros();
    let mut m = l;
    if remaining % 2 == 1 {
        pass2::<S>(simd, data, tw, n, stride, m, inverse);
        m *= 2;
    }
    while m < n {
        pass22::<S>(simd, data, tw, n, stride, m, inverse);
        m *= 4;
    }
    let scale = inverse.then(|| 1.0 / n as f32);
    last_pass::<S>(simd, data, scale);
}

impl Fft<C32> for Simd {
    fn fft<Tw: AsRef<[C32]>>(
        &self,
        plan: &FftPlan<C32, Tw>,
        data: &mut [C32],
        dir: Direction,
        scaling: Scaling,
    ) -> Result<FftScale> {
        if scaling == Scaling::BlockFloat {
            return Err(Error::Unsupported(
                "block floating point is not available for c32",
            ));
        }
        let n = plan.len();
        check_len(n, data.len())?;
        let inverse = dir == Direction::Inverse;
        let tw: &[f32] = bytemuck::cast_slice(plan.twiddles());
        let stride = plan.twiddle_stride();
        let floats: &mut [f32] = bytemuck::cast_slice_mut(data);
        dispatch!(simd => fft_stages(simd, floats, tw, n, stride, inverse));
        Ok(FftScale {
            exponent: if inverse { plan.log2_len() as i32 } else { 0 },
        })
    }
}

// ---------------------------------------------------------------------------------------------
// CQ15

/// Largest component magnitude above which a stage needs a shift of 1 (block floating point).
const BFP_SHIFT1_ABOVE: i32 = 13_572;
/// Largest component magnitude above which a stage needs a shift of 2.
const BFP_SHIFT2_ABOVE: i32 = 27_145;

/// `sat16(round_shift(v, shift))`.
#[inline(always)]
fn store_q15(v: i64, shift: u32) -> Q15 {
    Q15(sat16(round_shift(v, shift)))
}

/// One radix-2 stage, scalar; the arithmetic of the fallback.
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

/// Real and imaginary part of packed `CQ15` lanes (I in the low half, Q in the high half).
#[inline(always)]
fn split_q15<S: SimdToken>(v: S::i32s) -> (S::i32s, S::i32s) {
    ((v << 16u32) >> 16u32, v >> 16u32)
}

/// `sat16(round_shift(v, shift))` of one vector.
#[inline(always)]
fn finish_q15<S: SimdToken>(
    v: S::i32s,
    round: S::i32s,
    shift: u32,
    lo: S::i32s,
    hi: S::i32s,
) -> S::i32s {
    ((v + round) >> shift).max(lo).min(hi)
}

/// Packs real and imaginary parts (each in `[-32768, 32767]`) into 32-bit lanes.
#[inline(always)]
fn pack_q15<S: SimdToken>(re: S::i32s, im: S::i32s) -> S::i32s {
    (re & 0xFFFFi32) | (im << 16u32)
}

/// One radix-2 stage with `half >= i32 lanes`, `i32` arithmetic (the twiddle products and the
/// sums of a butterfly stay below `2^31`).
#[inline(always)]
fn stage_vector<S: SimdToken>(
    simd: S,
    data: &mut [i16],
    tw: &[i16],
    half: usize,
    step: usize,
    inverse: bool,
    shift: u32,
) {
    let lanes = S::i32s::LEN;
    let lo_clamp = S::i32s::splat(simd, -32768);
    let hi_clamp = S::i32s::splat(simd, 32767);
    let round = S::i32s::splat(simd, 1 << 14);
    let out_round = S::i32s::splat(simd, if shift > 0 { 1 << (shift - 1) } else { 0 });
    for block in data.chunks_exact_mut(4 * half) {
        let (lo, hi) = block.split_at_mut(2 * half);
        for k0 in (0..half).step_by(lanes) {
            let (Some(a_s), Some(c_s)) = (
                lo.get_mut(2 * k0..2 * k0 + 2 * lanes),
                hi.get_mut(2 * k0..2 * k0 + 2 * lanes),
            ) else {
                continue;
            };
            let a = S::i16s::from_slice(simd, a_s).bitcast::<S::i32s>();
            let c = S::i16s::from_slice(simd, c_s).bitcast::<S::i32s>();
            let w_at = |lane: usize| -> (i32, i32) {
                match tw.get(2 * (k0 + lane) * step..2 * (k0 + lane) * step + 2) {
                    Some([re, im]) => (i32::from(*re), i32::from(*im)),
                    _ => (0, 0),
                }
            };
            let wr = S::i32s::from_fn(simd, |l| w_at(l).0);
            let wi = S::i32s::from_fn(simd, |l| if inverse { -w_at(l).1 } else { w_at(l).1 });
            let (ar, ai) = split_q15::<S>(a);
            let (cr, ci) = split_q15::<S>(c);
            let tr = (wr * cr - wi * ci + round) >> 15u32;
            let ti = (wr * ci + wi * cr + round) >> 15u32;
            let a_re = finish_q15::<S>(ar + tr, out_round, shift, lo_clamp, hi_clamp);
            let a_im = finish_q15::<S>(ai + ti, out_round, shift, lo_clamp, hi_clamp);
            let c_re = finish_q15::<S>(ar - tr, out_round, shift, lo_clamp, hi_clamp);
            let c_im = finish_q15::<S>(ai - ti, out_round, shift, lo_clamp, hi_clamp);
            pack_q15::<S>(a_re, a_im)
                .bitcast::<S::i16s>()
                .store_slice(a_s);
            pack_q15::<S>(c_re, c_im)
                .bitcast::<S::i16s>()
                .store_slice(c_s);
        }
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
    let lanes = S::i32s::LEN;
    let tw_i16: &[i16] = bytemuck::cast_slice(tw);
    if !bit_reverse_cq15::<S>(simd, bytemuck::cast_slice_mut(&mut *data), log2n) {
        bit_reverse(data, log2n);
    }
    let mut exponent = 0i32;
    for s in 1..=log2n {
        let shift = if block_float {
            let m = max_component(simd, bytemuck::cast_slice(&*data));
            if m > BFP_SHIFT2_ABOVE {
                2
            } else {
                u32::from(m > BFP_SHIFT1_ABOVE)
            }
        } else {
            1
        };
        let half = 1usize << (s - 1);
        let step = (n >> s) * stride;
        if half >= lanes {
            stage_vector::<S>(
                simd,
                bytemuck::cast_slice_mut(&mut *data),
                tw_i16,
                half,
                step,
                inverse,
                shift,
            );
        } else {
            stage_scalar(data, tw, half, step, inverse, shift);
        }
        exponent += shift as i32;
    }
    exponent
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
