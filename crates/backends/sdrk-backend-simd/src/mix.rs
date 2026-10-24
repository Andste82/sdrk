//! `rotate`: frequency shift with a numerically controlled oscillator.
//!
//! `C32`: no cosine table and no polynomial per sample. The stream is cut into blocks of `B`
//! samples (a power of two) at absolute positions. The oscillator vector of the first vector of
//! a block, the *anchor*, is computed in `f64` and rounded once; vector `j = CHAIN k + r` of the
//! block is the anchor times the table entry `W^k`, `W = e^{j 2 pi CHAIN h inc / 2^32}`
//! (`h` samples per vector; a short table that every call computes in `f64`, only the entries
//! it uses), times `r` steps of `E = e^{j 2 pi h inc / 2^32}`. So a vector of samples costs two
//! complex multiplications, and the error does not grow along the block (the chains are
//! [`CHAIN`] vectors long). The position of a sample within its block is not stored anywhere: it
//! follows from the phase and the increment (see [`Blocks`]), so the output is a function of
//! `(phase_n, inc)` and does not depend on how the input is split into calls. Every quantity a
//! call computes once (the lane offsets of the anchors, the table, `E`) is a function of `inc`
//! alone, and the polynomials run on whole vectors of lanes, so the cost per call is a few
//! hundred instructions. The result meets the contract of the kernel (120 dB against the exact
//! rotation; this reaches about 142 dB, 134 dB in the worst single call) but is not
//! bit-identical with the fallback. FMA is used where the level has it. The samples are
//! processed deinterleaved, one vector of oscillator values per vector of real parts.
//!
//! `CQ15`: the table lookup and the integer arithmetic of the fallback, vectorised, bit-exact
//! (also for the last partial vector, which is processed zero-padded).

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::{C32, CQ15, Result};

use crate::Simd;
use crate::level::dispatch;
use crate::table::cos_sin_q15;
use crate::util::check_len;

/// Coefficients of `sin(pi/2 x) = x (S0 + S1 z + ... + S5 z^5)`, `z = x^2`, on `[-1, 1]`
/// (interpolating polynomial at the Chebyshev nodes, maximum error 3e-11).
const SIN: [f64; 6] = [
    1.570_796_326_768_066,
    -0.645_964_095_574_130_4,
    0.079_692_603_685_608_2,
    -0.004_681_657_707_348_082,
    0.000_160_254_590_296_085_3,
    -3.431_789_006_999_355e-6,
];
/// Coefficients of `cos(pi/2 x) = C0 + C1 z + ... + C6 z^6` on `[-1, 1]` (maximum error 8e-13).
const COS: [f64; 7] = [
    0.999_999_999_999_249_4,
    -1.233_700_550_062_564,
    0.253_669_506_722_137_67,
    -0.020_863_473_679_989_624,
    0.000_919_239_993_801_908_6,
    -2.517_218_990_027_375e-5,
    4.492_180_080_112_624_4e-7,
];

/// The most table entries per block.
const ENTRIES: usize = 32;
/// Vectors per table entry: the oscillator of the first one is the anchor times the entry, the
/// next ones follow by one complex multiplication each.
const CHAIN: usize = 4;

/// The polynomial with the coefficients `k` (lowest first) at `z`, by Horner's rule.
///
/// The helpers of the oscillator are functions, not closures: a closure that is not inlined is
/// compiled without the target features of the level when the level is detected at run time.
#[inline(always)]
fn horner<S: SimdToken>(simd: S, k: &[f64], z: S::f64s) -> S::f64s {
    let Some((&last, rest)) = k.split_last() else {
        return S::f64s::splat(simd, 0.0);
    };
    let mut acc = S::f64s::splat(simd, last);
    for &k in rest.iter().rev() {
        acc = acc.mul_add(z, k);
    }
    acc
}

/// `cos(pi/2 x)` and `sin(pi/2 x)` for `x` in `[-1, 1)` in `f64` (fused multiply-adds where the
/// level has them), accurate to 1e-10.
#[inline(always)]
fn cos_sin_poly<S: SimdToken>(simd: S, x: S::f64s) -> (S::f64s, S::f64s) {
    let z = x * x;
    (horner(simd, &COS, z), horner(simd, &SIN, z) * x)
}

/// A vector of complex values in `f64`: the real and imaginary parts of the low and the high
/// half of the lanes.
#[derive(Clone, Copy)]
struct Wide<S: SimdToken> {
    lo: (S::f64s, S::f64s),
    hi: (S::f64s, S::f64s),
}

/// `e^{j 2 pi p / 2^32}` for the phases `p` (in the low 32 bits of the lanes), cosines and
/// sines: `p << 1` read as a signed integer is the angle reduced to `[-pi/2, pi/2)`, which
/// [`cos_sin_poly`] evaluates; where the reduction removed a half turn, both signs flip.
#[inline(always)]
fn cos_sin_lanes<S: SimdToken>(simd: S, p: S::u64s) -> (S::f64s, S::f64s) {
    /// The bits of `2^52` as an `f64`: or-ed with an integer below `2^32`, `2^52` plus it.
    const EXP52: u64 = 0x4330_0000_0000_0000;
    let p2 = (p << 1u32) & 0xFFFF_FFFFu64;
    // The sign bit of an `f64`, set where a half turn was removed.
    let flip = ((p ^ p2) & 0x8000_0000u64) << 32u32;
    // `p2` as a signed integer plus `2^31`, so that it can be read as unsigned.
    let biased = ((p2 ^ 0x8000_0000u64) | EXP52).bitcast::<S::f64s>();
    let x = (biased - (4_503_599_627_370_496.0 + 2_147_483_648.0)) * (1.0 / 2_147_483_648.0);
    let (c, s) = cos_sin_poly(simd, x);
    (
        (c.bitcast::<S::u64s>() ^ flip).bitcast::<S::f64s>(),
        (s.bitcast::<S::u64s>() ^ flip).bitcast::<S::f64s>(),
    )
}

impl<S: SimdToken> Wide<S> {
    /// `e^{j 2 pi p / 2^32}` for every lane of the phases `p`.
    #[inline(always)]
    fn oscillator(simd: S, p: S::u32s) -> Wide<S> {
        let (lo, hi) = p.widen();
        Wide {
            lo: cos_sin_lanes(simd, lo),
            hi: cos_sin_lanes(simd, hi),
        }
    }

    /// `e^{j 2 pi s(l) step / 2^32}` for every lane `l`, `s = sample` a lane order with
    /// `s(1) = 1` and `s(l + h/2) = s(l) + d` (`d` a power of two). Only the low half is
    /// evaluated: the high half is the low half times `e^{j 2 pi d step / 2^32}`, which is the
    /// second lane squared `log2(d)` times.
    #[inline(always)]
    fn powers(simd: S, step: u32, sample: impl Fn(usize) -> usize) -> Wide<S> {
        let h = S::f32s::LEN;
        let phases = S::u32s::from_fn(simd, |l| sample(l) as u32) * S::u32s::splat(simd, step);
        let lo = Wide {
            lo: cos_sin_lanes(simd, phases.widen().0),
            hi: (S::f64s::splat(simd, 1.0), S::f64s::splat(simd, 0.0)),
        };
        let mut c = lo.lane(1);
        let mut d = 1;
        while d < sample(h / 2) {
            c = (c.0 * c.0 - c.1 * c.1, 2.0 * c.0 * c.1);
            d *= 2;
        }
        Wide {
            lo: lo.lo,
            hi: lo.times(simd, c).lo,
        }
    }

    /// Every lane times `c + j s`.
    #[inline(always)]
    fn times(self, simd: S, (c, s): (f64, f64)) -> Wide<S> {
        let (c, s) = (S::f64s::splat(simd, c), S::f64s::splat(simd, s));
        let ((lr, li), (hr, hi)) = (self.lo, self.hi);
        Wide {
            lo: (lr.mul_sub(c, li * s), lr.mul_add(s, li * c)),
            hi: (hr.mul_sub(c, hi * s), hr.mul_add(s, hi * c)),
        }
    }

    /// The oscillator vector of the phase `a` with these lane offsets: `e^{j 2 pi a / 2^32}`
    /// times every lane, rounded to `f32`.
    #[inline(always)]
    fn at(self, simd: S, a: u32) -> (S::f32s, S::f32s) {
        self.times(simd, cos_sin(simd, a)).narrow()
    }

    /// Lane `l` (`1` if there is no such lane).
    #[inline(always)]
    fn lane(self, l: usize) -> (f64, f64) {
        let half = S::f64s::LEN;
        let (re, im) = if l < half { self.lo } else { self.hi };
        let c = re.as_slice().get(l % half).copied().unwrap_or(1.0);
        let s = im.as_slice().get(l % half).copied().unwrap_or(0.0);
        (c, s)
    }

    /// The square of lane `l`.
    #[inline(always)]
    fn lane_squared(self, l: usize) -> (f64, f64) {
        let (c, s) = self.lane(l);
        (c * c - s * s, 2.0 * c * s)
    }

    /// The values rounded to `f32`: real parts, imaginary parts.
    #[inline(always)]
    fn narrow(self) -> (S::f32s, S::f32s) {
        (self.lo.0.narrow(self.hi.0), self.lo.1.narrow(self.hi.1))
    }
}

/// `e^{j 2 pi p / 2^32}` as [`cos_sin_lanes`] computes it, for one phase: cosine, sine.
#[inline(always)]
fn cos_sin<S: SimdToken>(simd: S, p: u32) -> (f64, f64) {
    let p2 = p << 1;
    let x = f64::from(p2 as i32) * (1.0 / 2_147_483_648.0);
    let (c, s) = cos_sin_poly(simd, S::f64s::splat(simd, x));
    let (c, s) = (
        c.as_slice().first().copied().unwrap_or(1.0),
        s.as_slice().first().copied().unwrap_or(0.0),
    );
    if (p ^ p2) & 0x8000_0000 == 0 {
        (c, s)
    } else {
        (-c, -s)
    }
}

/// The sample that lane `l` holds with the [`Split`] `L`.
#[inline(always)]
fn sample_of<S: SimdToken, L: Split<S>>(l: usize) -> usize {
    if L::IN_LANE {
        lane_sample(S::f32s::LEN, l)
    } else {
        l
    }
}

/// The phase offsets `s(l) inc` of the lanes, `s(l)` the sample that lane `l` holds.
#[inline(always)]
fn lane_phases<S: SimdToken, L: Split<S>>(simd: S, inc: u32) -> S::u32s {
    let s = S::u32s::from_fn(simd, |l| sample_of::<S, L>(l) as u32);
    s * S::u32s::splat(simd, inc)
}

/// `(ar + j ai) (br + j bi)`: two multiplications and two fused (or separate) multiply-adds.
/// The real part is `ar br - ai bi` as a multiply-subtract (`vfmsub`; on levels without FMA a
/// multiplication and a subtraction), on AArch64 as `-ai bi + ar br`, which is `fmls` (its
/// multiply-subtract would be a negation and an `fmla`).
#[inline(always)]
fn cmul<S: SimdToken>(ar: S::f32s, ai: S::f32s, br: S::f32s, bi: S::f32s) -> (S::f32s, S::f32s) {
    #[cfg(target_arch = "aarch64")]
    let re = (-ai).mul_add(bi, ar * br);
    #[cfg(not(target_arch = "aarch64"))]
    let re = ar.mul_sub(br, ai * bi);
    (re, ar.mul_add(bi, ai * br))
}

/// How a level loads `h` interleaved samples as a vector of real parts and a vector of
/// imaginary parts, and stores them back.
trait Split<S: SimdToken>: Copy {
    /// Whether the lanes hold the samples in the order of [`lane_sample`] instead of
    /// `0, 1, ..., h - 1`.
    const IN_LANE: bool = false;

    /// The real and imaginary parts of the `h` samples in `x` (`2 h` floats).
    #[inline(always)]
    fn load(self, simd: S, x: &[f32]) -> (S::f32s, S::f32s) {
        let (a, b) = x.split_at(S::f32s::LEN);
        S::f32s::from_slice(simd, a).deinterleave(S::f32s::from_slice(simd, b))
    }

    /// Stores the samples `re + j im` to `y` (`2 h` floats), the inverse of [`Split::load`].
    #[inline(always)]
    fn store(self, simd: S, re: S::f32s, im: S::f32s, y: &mut [f32]) {
        let _ = simd;
        let (a, b) = re.interleave(im);
        let (ya, yb) = y.split_at_mut(S::f32s::LEN);
        a.store_slice(ya);
        b.store_slice(yb);
    }
}

/// [`Split`] with the deinterleave and interleave of `fearless_simd`, for every level.
#[derive(Clone, Copy)]
struct Portable;

impl<S: SimdToken> Split<S> for Portable {}

/// The sample that lane `l` of an in-lane deinterleave of `h` samples holds: every 128-bit
/// block `k` of the result takes the samples `2k, 2k + 1` of the first half and the same two
/// of the second half (`0, 1, 4, 5, 2, 3, 6, 7` for `h = 8`; the identity for `h = 4`).
fn lane_sample(h: usize, l: usize) -> usize {
    let (k, r) = (l / 4, l % 4);
    let half = if r < 2 { 0 } else { h / 2 - 2 };
    2 * k + r + half
}

/// The table entry `e` (a pair) as a vector of the real part and one of the imaginary part.
#[inline(always)]
fn splat_pair<S: SimdToken>(simd: S, e: &[f32]) -> (S::f32s, S::f32s) {
    let r = e.first().copied().unwrap_or(0.0);
    let i = e.get(1).copied().unwrap_or(0.0);
    (S::f32s::splat(simd, r), S::f32s::splat(simd, i))
}

/// Rotates one vector: `y = x v`.
#[inline(always)]
fn apply<S: SimdToken, L: Split<S>>(
    simd: S,
    split: L,
    x: &[f32],
    y: &mut [f32],
    vr: S::f32s,
    vi: S::f32s,
) {
    let (xr, xi) = split.load(simd, x);
    let (yr, yi) = cmul::<S>(xr, xi, vr, vi);
    split.store(simd, yr, yi, y);
}

/// Rotates the samples `x` (fewer than `h` of them, `x.len()` floats) as the lanes from `lead`
/// (in floats) on of one vector with the oscillator `v`, through a zero-padded buffer.
#[inline(always)]
fn apply_part<S: SimdToken, L: Split<S>>(
    simd: S,
    split: L,
    x: &[f32],
    y: &mut [f32],
    lead: usize,
    vr: S::f32s,
    vi: S::f32s,
) {
    let h = S::f32s::LEN;
    let mut buf = [0f32; 32];
    let span = lead..lead + x.len();
    if let Some(d) = buf.get_mut(span.clone()) {
        d.copy_from_slice(x);
    }
    if let Some(v) = buf.get_mut(..2 * h) {
        let (xr, xi) = split.load(simd, v);
        let (yr, yi) = cmul::<S>(xr, xi, vr, vi);
        split.store(simd, yr, yi, v);
    }
    if let Some(s) = buf.get(span) {
        y.copy_from_slice(s);
    }
}

/// The block structure of one call.
///
/// The phase of sample `n` is `phase_0 + n inc` modulo `2^32`. With `inc = 2^t o`, `o` odd,
/// the low `t` bits of the phase never change and `(phase >> t) o^-1` counts the samples
/// modulo `2^(32 - t)` (`o^-1` is the inverse of `o` modulo `2^32`). So the position of a
/// sample modulo any power of two up to `2^(32 - t)` is a function of its phase and the
/// increment alone, up to an offset that is the same for the whole stream. The blocks are
/// aligned to that position: `B = min(max, 2^(32 - t))` samples, which is more than `h`
/// because the caller handles `h inc = 0` (`2^(32 - t) <= h`, `inc = 0` included) separately.
struct Blocks {
    /// Samples per block (`B`).
    len: usize,
    /// Position of the first sample of the call within its block.
    first: usize,
    /// Phase of the first sample of that block.
    anchor: u32,
}

impl Blocks {
    fn new(phase: u32, inc: u32, max: usize) -> Blocks {
        let t = inc.trailing_zeros();
        let odd = inc.wrapping_shr(t);
        // Newton's iteration for the inverse modulo 2^32: `3 o ^ 2` is right in the low 5
        // bits, and every step doubles them (5, 10, 20, 40).
        let mut inv = odd.wrapping_mul(3) ^ 2;
        for _ in 0..3 {
            inv = inv.wrapping_mul(2u32.wrapping_sub(odd.wrapping_mul(inv)));
        }
        let len = max.min(1usize << (32 - t).min(31));
        let first = ((phase >> t).wrapping_mul(inv) as usize) & (len - 1);
        let anchor = phase.wrapping_sub((first as u32).wrapping_mul(inc));
        Blocks { len, first, anchor }
    }
}

/// The table of a block: the entries `W^k`, `k = 0, 1, ...` (see [`Osc`]) as pairs of the real
/// and the imaginary part.
struct Table([f32; 2 * ENTRIES]);

impl Table {
    /// The entries `from..to` (and the others of their groups of `h`) for the increment `inc`.
    /// Entry `g h + l` is `W^(g h) W^l` in `f64`, rounded once to `f32`: the lanes
    /// `W^l = e^{j 2 pi l CHAIN h inc / 2^32}` from [`Wide::powers`], `W^(g h)` by `g`
    /// multiplications with `W^h = (W^(h/2))^2` from `W^0 = 1` on. So an entry does not depend
    /// on which entries a call needs.
    #[inline(always)]
    fn fill<S: SimdToken>(&mut self, simd: S, inc: u32, from: usize, to: usize) {
        let h = S::f32s::LEN;
        let w = ((CHAIN * h) as u32).wrapping_mul(inc);
        let lanes = Wide::powers(simd, w, |l| l);
        let (sr, si) = lanes.lane_squared(h / 2);
        let (mut pr, mut pi) = (1.0f64, 0.0f64);
        for (g, group) in self
            .0
            .chunks_exact_mut(2 * h)
            .take(to.div_ceil(h))
            .enumerate()
        {
            if g >= from / h {
                let (re, im) = lanes.times(simd, (pr, pi)).narrow();
                let (a, b) = re.interleave(im);
                let (ga, gb) = group.split_at_mut(h);
                a.store_slice(ga);
                b.store_slice(gb);
            }
            (pr, pi) = (pr * sr - pi * si, pr * si + pi * sr);
        }
    }
}

/// The oscillator vectors of one block: vector `j = CHAIN k + r` of the block is
/// `A W^k E^r`, the anchor `A` times the table entry `W^k = E^(CHAIN k)` times `r` steps of
/// `E = e^{j 2 pi h inc / 2^32}`, always multiplied in this order.
#[derive(Clone, Copy)]
struct Osc<'a, S: SimdToken, L: Split<S>> {
    simd: S,
    split: L,
    /// The anchor `A`.
    a: (S::f32s, S::f32s),
    /// The step `E`.
    e: (S::f32s, S::f32s),
    /// The table of the block.
    table: &'a [f32],
}

impl<S: SimdToken, L: Split<S>> Osc<'_, S, L> {
    /// `A W^k`.
    #[inline(always)]
    fn entry(&self, k: usize) -> (S::f32s, S::f32s) {
        let (tr, ti) = splat_pair(self.simd, self.table.get(2 * k..).unwrap_or(&[]));
        cmul::<S>(self.a.0, self.a.1, tr, ti)
    }

    /// `E v`.
    #[inline(always)]
    fn step(&self, v: (S::f32s, S::f32s)) -> (S::f32s, S::f32s) {
        cmul::<S>(self.e.0, self.e.1, v.0, v.1)
    }

    /// The oscillator vector `j` of the block.
    #[inline(always)]
    fn vector(&self, j: usize) -> (S::f32s, S::f32s) {
        let mut v = self.entry(j / CHAIN);
        for _ in 0..j % CHAIN {
            v = self.step(v);
        }
        v
    }

    /// Rotates the samples `x` of the block from the position `first` on (at most to the end
    /// of the block).
    #[inline(always)]
    fn block(&self, x: &[f32], y: &mut [f32], first: usize) {
        let (simd, split) = (self.simd, self.split);
        let h = S::f32s::LEN;
        let (mut j, lead) = (first / h, first % h);
        let (mut x, mut y) = (x, y);
        if lead != 0 {
            // The call starts inside a vector: its first samples are the lanes from `lead` on.
            let len = (2 * (h - lead)).min(x.len());
            let (x0, x1) = x.split_at(len);
            let (y0, y1) = y.split_at_mut(len);
            let (vr, vi) = self.vector(j);
            apply_part(simd, split, x0, y0, 2 * lead, vr, vi);
            (x, y) = (x1, y1);
            j += 1;
        }
        // Single vectors up to the start of a chain, then whole chains, then single vectors.
        let single = (CHAIN - j % CHAIN) % CHAIN;
        let len = (2 * h * single).min(x.len() / (2 * h) * (2 * h));
        let (x0, x1) = x.split_at(len);
        let (y0, y1) = y.split_at_mut(len);
        for (xv, yv) in x0.chunks_exact(2 * h).zip(y0.chunks_exact_mut(2 * h)) {
            let (vr, vi) = self.vector(j);
            apply(simd, split, xv, yv, vr, vi);
            j += 1;
        }
        let chains = x1.len() / (2 * h * CHAIN) * (2 * h * CHAIN);
        let (x0, x1) = x1.split_at(chains);
        let (y0, y1) = y1.split_at_mut(chains);
        self.run(x0, y0, j / CHAIN);
        j += chains / (2 * h);
        let mut xs = x1.chunks_exact(2 * h);
        let mut ys = y1.chunks_exact_mut(2 * h);
        for (xv, yv) in (&mut xs).zip(&mut ys) {
            let (vr, vi) = self.vector(j);
            apply(simd, split, xv, yv, vr, vi);
            j += 1;
        }
        let (xt, yt) = (xs.remainder(), ys.into_remainder());
        if !xt.is_empty() {
            let (vr, vi) = self.vector(j);
            apply_part(simd, split, xt, yt, 0, vr, vi);
        }
    }

    /// Rotates one chain `x` (`CHAIN h` samples) with the table entry `e`.
    #[inline(always)]
    fn chain(&self, x: &[f32], y: &mut [f32], e: &[f32]) {
        let (simd, split) = (self.simd, self.split);
        let h = S::f32s::LEN;
        let (tr, ti) = splat_pair(simd, e);
        let mut v = cmul::<S>(self.a.0, self.a.1, tr, ti);
        for (r, (xv, yv)) in x
            .chunks_exact(2 * h)
            .zip(y.chunks_exact_mut(2 * h))
            .enumerate()
        {
            if r > 0 {
                v = self.step(v);
            }
            apply(simd, split, xv, yv, v.0, v.1);
        }
    }

    /// Rotates the whole chains `x` (a multiple of `CHAIN h` samples) from the table entry `k`
    /// on.
    #[inline(always)]
    fn run(&self, x: &[f32], y: &mut [f32], k: usize) {
        let h = S::f32s::LEN;
        let table = self.table.get(2 * k..).unwrap_or(&[]);
        // Two chains per iteration (half the loop overhead), then the odd one.
        let len = 2 * h * CHAIN;
        let mut xs = x.chunks_exact(2 * len);
        let mut ys = y.chunks_exact_mut(2 * len);
        let odd = table.get(4 * (x.len() / (2 * len))..).unwrap_or(&[]);
        for ((xp, yp), e) in (&mut xs).zip(&mut ys).zip(table.chunks_exact(4)) {
            let (x0, x1) = xp.split_at(len);
            let (y0, y1) = yp.split_at_mut(len);
            let (e0, e1) = e.split_at(2);
            self.chain(x0, y0, e0);
            self.chain(x1, y1, e1);
        }
        let (xt, yt) = (xs.remainder(), ys.into_remainder());
        if !xt.is_empty() {
            self.chain(xt, yt, odd);
        }
    }
}

/// Rotates all samples (`x` and `y` interleaved, `2 n` floats each).
#[inline(always)]
fn rotate_split<S: SimdToken, L: Split<S>>(
    simd: S,
    split: L,
    phase: u32,
    inc: u32,
    x: &[f32],
    y: &mut [f32],
) {
    let h = S::f32s::LEN;
    let lanes = lane_phases::<S, L>(simd, inc);
    if (h as u32).wrapping_mul(inc) == 0 {
        // Every vector has the same phases: one oscillator vector serves the call.
        let (vr, vi) = Wide::oscillator(simd, S::u32s::splat(simd, phase) + lanes).narrow();
        let mut xs = x.chunks_exact(2 * h);
        let mut ys = y.chunks_exact_mut(2 * h);
        for (xv, yv) in (&mut xs).zip(&mut ys) {
            apply(simd, split, xv, yv, vr, vi);
        }
        apply_part(simd, split, xs.remainder(), ys.into_remainder(), 0, vr, vi);
        return;
    }
    let blocks = Blocks::new(phase, inc, h * CHAIN * ENTRIES);
    let n = x.len() / 2;
    // The table entries that the call uses: those of the rest of the first block, and those of
    // the following blocks.
    let span = CHAIN * h;
    let head = (blocks.len - blocks.first).min(n);
    let (k0, k1) = (blocks.first / span, blocks.len.div_ceil(span));
    let mut table = Table([0.0; 2 * ENTRIES]);
    match (n - head).min(blocks.len).div_ceil(span) {
        0 => table.fill(simd, inc, k0, (blocks.first + head).div_ceil(span)),
        rest if rest >= k0 => table.fill(simd, inc, 0, k1),
        rest => {
            table.fill(simd, inc, 0, rest);
            table.fill(simd, inc, k0, k1);
        }
    }
    // The anchor of the phase `a` is `e^{j 2 pi a / 2^32}` times these lane offsets in `f64`.
    let offsets = Wide::powers(simd, inc, sample_of::<S, L>);
    // `E = (e^{j 2 pi (h/2) inc / 2^32})^2` from the lane that holds sample `h/2`.
    let half = (0..h).find(|&l| sample_of::<S, L>(l) == h / 2).unwrap_or(0);
    let (er, ei) = offsets.lane_squared(half);
    let mut osc = Osc {
        simd,
        split,
        a: offsets.at(simd, blocks.anchor),
        e: (
            S::f32s::splat(simd, er as f32),
            S::f32s::splat(simd, ei as f32),
        ),
        table: &table.0,
    };
    let step = (blocks.len as u32).wrapping_mul(inc);
    let mut a = blocks.anchor;
    // The rest of the first block, the whole blocks, the start of the last block.
    let (x0, x) = x.split_at(2 * head);
    let (y0, y) = y.split_at_mut(2 * head);
    osc.block(x0, y0, blocks.first);
    for (xb, yb) in x.chunks(2 * blocks.len).zip(y.chunks_mut(2 * blocks.len)) {
        a = a.wrapping_add(step);
        osc.a = offsets.at(simd, a);
        osc.block(xb, yb, 0);
    }
}

/// Rotates all samples with the [`Split`] of the level of `simd`.
#[inline(always)]
fn rotate_f32<S: SimdToken>(simd: S, phase: u32, inc: u32, x: &[f32], y: &mut [f32]) {
    let level = simd.level();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if let Some(t) = level.as_avx512() {
            return rotate_split(t, x86::InLane, phase, inc, x, y);
        }
        if let Some(t) = level.as_avx2() {
            return rotate_split(t, x86::InLane, phase, inc, x, y);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return rotate_split(t, neon::Ld2, phase, inc, x, y);
    }
    let _ = level;
    rotate_split(simd, Portable, phase, inc, x, y);
}

/// [`Split`] on x86: the deinterleave of `fearless_simd` is right for SSE; on AVX2 and AVX-512
/// it crosses the 128-bit blocks (two shuffles more each way), so those levels use `shufps`
/// and `unpcklps`/`unpckhps`, which stay within the blocks, and the lane order of
/// [`lane_sample`].
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        _mm256_shuffle_ps, _mm256_unpackhi_ps, _mm256_unpacklo_ps, _mm512_shuffle_ps,
        _mm512_unpackhi_ps, _mm512_unpacklo_ps,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        _mm256_shuffle_ps, _mm256_unpackhi_ps, _mm256_unpacklo_ps, _mm512_shuffle_ps,
        _mm512_unpackhi_ps, _mm512_unpacklo_ps,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{Avx2, Avx512, SimdFrom, f32x8, f32x16};

    use super::Split;

    /// [`Split`] of AVX2 and AVX-512 with the in-lane shuffles.
    #[derive(Clone, Copy)]
    pub(super) struct InLane;

    impl Split<Avx2> for InLane {
        const IN_LANE: bool = true;

        #[inline(always)]
        fn load(self, simd: Avx2, x: &[f32]) -> (f32x8<Avx2>, f32x8<Avx2>) {
            fearless_simd::kernel!(
                #[inline(always)]
                fn kernel(t: Avx2, a: f32x8<Avx2>, b: f32x8<Avx2>) -> (f32x8<Avx2>, f32x8<Avx2>) {
                    let (a, b) = (a.into(), b.into());
                    (
                        f32x8::simd_from(t, _mm256_shuffle_ps::<0x88>(a, b)),
                        f32x8::simd_from(t, _mm256_shuffle_ps::<0xDD>(a, b)),
                    )
                }
            );
            let (a, b) = x.split_at(8);
            kernel(simd, f32x8::from_slice(simd, a), f32x8::from_slice(simd, b))
        }

        #[inline(always)]
        fn store(self, simd: Avx2, re: f32x8<Avx2>, im: f32x8<Avx2>, y: &mut [f32]) {
            fearless_simd::kernel!(
                #[inline(always)]
                fn kernel(t: Avx2, re: f32x8<Avx2>, im: f32x8<Avx2>) -> (f32x8<Avx2>, f32x8<Avx2>) {
                    let (re, im) = (re.into(), im.into());
                    (
                        f32x8::simd_from(t, _mm256_unpacklo_ps(re, im)),
                        f32x8::simd_from(t, _mm256_unpackhi_ps(re, im)),
                    )
                }
            );
            let (a, b) = kernel(simd, re, im);
            let (ya, yb) = y.split_at_mut(8);
            a.store_slice(ya);
            b.store_slice(yb);
        }
    }

    impl Split<Avx512> for InLane {
        const IN_LANE: bool = true;

        #[inline(always)]
        fn load(self, simd: Avx512, x: &[f32]) -> (f32x16<Avx512>, f32x16<Avx512>) {
            fearless_simd::kernel!(
                #[inline(always)]
                fn kernel(
                    t: Avx512,
                    a: f32x16<Avx512>,
                    b: f32x16<Avx512>,
                ) -> (f32x16<Avx512>, f32x16<Avx512>) {
                    let (a, b) = (a.into(), b.into());
                    (
                        f32x16::simd_from(t, _mm512_shuffle_ps::<0x88>(a, b)),
                        f32x16::simd_from(t, _mm512_shuffle_ps::<0xDD>(a, b)),
                    )
                }
            );
            let (a, b) = x.split_at(16);
            kernel(
                simd,
                f32x16::from_slice(simd, a),
                f32x16::from_slice(simd, b),
            )
        }

        #[inline(always)]
        fn store(self, simd: Avx512, re: f32x16<Avx512>, im: f32x16<Avx512>, y: &mut [f32]) {
            fearless_simd::kernel!(
                #[inline(always)]
                fn kernel(
                    t: Avx512,
                    re: f32x16<Avx512>,
                    im: f32x16<Avx512>,
                ) -> (f32x16<Avx512>, f32x16<Avx512>) {
                    let (re, im) = (re.into(), im.into());
                    (
                        f32x16::simd_from(t, _mm512_unpacklo_ps(re, im)),
                        f32x16::simd_from(t, _mm512_unpackhi_ps(re, im)),
                    )
                }
            );
            let (a, b) = kernel(simd, re, im);
            let (ya, yb) = y.split_at_mut(16);
            a.store_slice(ya);
            b.store_slice(yb);
        }
    }
}

/// [`Split`] on AArch64: `ld2` and `st2` deinterleave and interleave while loading and storing
/// (`fearless_simd` loads and shuffles: three instructions each way instead of one).
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon {
    use core::arch::aarch64::{float32x4x2_t, vld2q_f32, vst2q_f32};
    use fearless_simd::prelude::*;
    use fearless_simd::{Neon, SimdFrom, f32x4};

    use super::Split;

    /// [`Split`] with `ld2` and `st2`.
    #[derive(Clone, Copy)]
    pub(super) struct Ld2;

    impl Split<Neon> for Ld2 {
        #[inline(always)]
        fn load(self, simd: Neon, x: &[f32]) -> (f32x4<Neon>, f32x4<Neon>) {
            fearless_simd::kernel!(
                #[inline(always)]
                fn kernel(t: Neon, x: &[f32; 8]) -> (f32x4<Neon>, f32x4<Neon>) {
                    // SAFETY: `x` is the 8 values that `vld2q_f32` reads.
                    let v = unsafe { vld2q_f32(x.as_ptr()) };
                    (f32x4::simd_from(t, v.0), f32x4::simd_from(t, v.1))
                }
            );
            match x.first_chunk::<8>() {
                Some(x) => kernel(simd, x),
                None => (f32x4::splat(simd, 0.0), f32x4::splat(simd, 0.0)),
            }
        }

        #[inline(always)]
        fn store(self, simd: Neon, re: f32x4<Neon>, im: f32x4<Neon>, y: &mut [f32]) {
            fearless_simd::kernel!(
                #[inline(always)]
                fn kernel(_t: Neon, re: f32x4<Neon>, im: f32x4<Neon>, y: &mut [f32; 8]) {
                    // SAFETY: `y` is the 8 values that `vst2q_f32` writes.
                    unsafe { vst2q_f32(y.as_mut_ptr(), float32x4x2_t(re.into(), im.into())) }
                }
            );
            if let Some(y) = y.first_chunk_mut::<8>() {
                kernel(simd, re, im, y);
            }
        }
    }
}

/// Rotates one vector of `CQ15` samples (`h` samples per `i16` vector) with the phases `p`.
#[inline(always)]
fn q15_vector<S: SimdToken>(simd: S, x: &[i16], y: &mut [i16], p: S::u32s) {
    // One 32-bit lane per sample: the low half is I, the high half is Q.
    let x: S::i32s = S::i16s::from_slice(simd, x).bitcast();
    let xr = (x << 16u32) >> 16u32;
    let xi = x >> 16u32;
    let (c, s) = cos_sin_q15(simd, p);
    let round = S::i32s::splat(simd, 1 << 14);
    let yr = (xr * c - xi * s + round) >> 15u32;
    let yi = (xr * s + xi * c + round) >> 15u32;
    let lo = S::i32s::splat(simd, -32768);
    let hi = S::i32s::splat(simd, 32767);
    let yr = yr.max(lo).min(hi);
    let yi = yi.max(lo).min(hi);
    let packed = (yr & 0xFFFFi32) | (yi << 16u32);
    packed.bitcast::<S::i16s>().store_slice(y);
}

/// Rotates all `CQ15` samples; the last partial vector is processed zero-padded (every lane is
/// computed on its own, so the result is the same as for a whole vector).
#[inline(always)]
fn rotate_q15<S: SimdToken>(simd: S, phase: u32, inc: u32, input: &[i16], out: &mut [i16]) {
    let h = S::i32s::LEN;
    let ramp = S::u32s::from_fn(simd, |l| (l as u32).wrapping_mul(inc));
    let step = S::u32s::splat(simd, (h as u32).wrapping_mul(inc));
    let mut p = S::u32s::splat(simd, phase) + ramp;
    // Both sides cut to the common length, so that the loop has a single exit condition (with
    // run-time detection the kernel is not inlined behind the length check of the caller).
    let n = input.len().min(out.len());
    let (input, out) = (
        input.get(..n).unwrap_or_default(),
        out.get_mut(..n).unwrap_or_default(),
    );
    let mut xs = input.chunks_exact(2 * h);
    let mut ys = out.chunks_exact_mut(2 * h);
    for (x, y) in (&mut xs).zip(&mut ys) {
        q15_vector(simd, x, y, p);
        p += step;
    }
    let (xt, yt) = (xs.remainder(), ys.into_remainder());
    if !xt.is_empty() {
        let mut tmp_in = [0i16; 32];
        let mut tmp_out = [0i16; 32];
        if let Some(d) = tmp_in.get_mut(..xt.len()) {
            d.copy_from_slice(xt);
        }
        if let (Some(x), Some(y)) = (tmp_in.get(..2 * h), tmp_out.get_mut(..2 * h)) {
            q15_vector(simd, x, y, p);
        }
        if let Some(s) = tmp_out.get(..yt.len()) {
            yt.copy_from_slice(s);
        }
    }
}

impl Rotate<C32> for Simd {
    fn rotate(&self, nco: &mut Nco, input: &[C32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (phase, inc) = (nco.phase(), nco.increment());
        let x: &[f32] = bytemuck::cast_slice(input);
        let y: &mut [f32] = bytemuck::cast_slice_mut(output);
        dispatch!(simd => rotate_f32(simd, phase, inc, x, y));
        let skip = (input.len() as u32).wrapping_mul(inc);
        nco.set_phase(phase.wrapping_add(skip));
        Ok(())
    }
}

impl Rotate<CQ15> for Simd {
    fn rotate(&self, nco: &mut Nco, input: &[CQ15], output: &mut [CQ15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (phase, inc) = (nco.phase(), nco.increment());
        let x: &[i16] = bytemuck::cast_slice(input);
        let y: &mut [i16] = bytemuck::cast_slice_mut(output);
        dispatch!(simd => rotate_q15(simd, phase, inc, x, y));
        let skip = (input.len() as u32).wrapping_mul(inc);
        nco.set_phase(phase.wrapping_add(skip));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::vec;

    use super::{Portable, rotate_f32, rotate_split};
    use crate::level::dispatch;

    #[test]
    fn the_split_of_the_level_gives_the_bits_of_the_portable_split() {
        // The level-specific loads, stores and lane orders change where the values sit, not
        // how they are computed: every sample is bit-identical with `Portable`. The longest
        // call covers more than two blocks at every level (2048 samples at AVX-512).
        let x: vec::Vec<f32> = (0..10_000)
            .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        for inc in [
            0x0123_4567u32,
            0xFEDC_BA98,
            0x0000_0300,
            0x4000_0000,
            0x2000_0000,
            0,
        ] {
            for (phase, len) in [
                (0u32, 500),
                (0x8765_4321, 1),
                (0x1357_9BDF, 37),
                (3, 299),
                (0x0246_8ACE, 5000),
            ] {
                let (mut want, mut got) = (vec![0f32; 2 * len], vec![0f32; 2 * len]);
                let x = &x[..2 * len];
                dispatch!(simd => rotate_split(simd, Portable, phase, inc, x, &mut want));
                dispatch!(simd => rotate_f32(simd, phase, inc, x, &mut got));
                assert!(
                    want.iter()
                        .zip(&got)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "inc={inc:#x} phase={phase:#x} len={len}"
                );
            }
        }
    }
}
