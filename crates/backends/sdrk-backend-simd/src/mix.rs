//! `rotate`: frequency shift with a numerically controlled oscillator.
//!
//! `C32`: no cosine table and no polynomial per sample. The stream is cut into blocks of `B`
//! samples (a power of two) at absolute positions. The oscillator vector of the first vector of
//! a block, the *anchor*, is computed in `f64` and rounded once; vector `j = CHAIN k + r` of the
//! block is the anchor times the table entry `W^k`, `W = e^{j 2 pi CHAIN h inc / 2^32}`
//! (`h` samples per vector; a short table that every call computes in `f64`, only the entries
//! it uses), times the power `E^r` of `E = e^{j 2 pi h inc / 2^32}` (in `f64`, rounded
//! once). So a vector of samples costs at most two complex multiplications, neither of them on a
//! chain from vector to vector, and the error does not grow along the block (the chains are
//! [`CHAIN`] vectors long). The position of a sample within its block is not stored anywhere: it
//! follows from the phase and the increment (see [`Blocks`]), so the output is a function of
//! `(phase_n, inc)` and does not depend on how the input is split into calls. Every quantity a
//! call computes once (the lane offsets of the anchors, the table, the powers of `E`) is a
//! function of `inc` alone, and the polynomials run on whole vectors of lanes, so the cost per
//! call is a few hundred instructions. The result meets the contract of the kernel (120 dB against the exact
//! rotation; this reaches about 142 dB, 134 dB in the worst single call) but is not
//! bit-identical with the fallback. FMA is used where the level has it. The samples are
//! processed deinterleaved, one vector of oscillator values per vector of real parts.
//!
//! `CQ15`: the table lookup and the integer arithmetic of the fallback, vectorised, bit-exact
//! (also for the last partial vector, which is processed zero-padded). With the static tables
//! (`nco-tables`) every level has its own path: the complex product with `pmaddwd` (x86),
//! `i32x4.dot_i16x8_s` (SIMD128) or the widening `smull`/`smlsl`/`smlal` with `sqrshrn` (NEON)
//! instead of four 32-bit multiplications and the clamp; see the module `x86_q15`.

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

/// Whether the vectors of a chain follow one another by steps of `E` (AArch64) instead of the
/// powers `E^r`: the powers shorten the dependency chain of the loop, which was the bound on
/// `znver3` (0.69 to 0.67 cycles per sample, `sse2` 2.30 / 2.27 to 2.09 / 1.97), but simulated
/// slower on the Cortex-A76 (1.97 to 2.12).
const STEPPED: bool = cfg!(target_arch = "aarch64");

/// The oscillator vectors of one block: vector `j = CHAIN k + r` of the block is
/// `(A W^k) E^r`, the anchor `A` times the table entry `W^k = E^(CHAIN k)`, times the power
/// `E^r` of `E = e^{j 2 pi h inc / 2^32}` (`f64`, rounded once; none for `r = 0`), always
/// multiplied in this order. Every vector of a chain is one multiplication away from `A W^k`,
/// so the chain is no dependency chain. On AArch64 ([`STEPPED`]) vector `r` is `E` times vector
/// `r - 1` instead.
#[derive(Clone, Copy)]
struct Osc<'a, S: SimdToken, L: Split<S>> {
    simd: S,
    split: L,
    /// The anchor `A`.
    a: (S::f32s, S::f32s),
    /// The powers `E^1, ..., E^(CHAIN - 1)`.
    e: [(S::f32s, S::f32s); CHAIN - 1],
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

    /// `E^r v` (`v` for `r = 0`).
    #[inline(always)]
    fn power(&self, r: usize, v: (S::f32s, S::f32s)) -> (S::f32s, S::f32s) {
        match r.checked_sub(1).and_then(|i| self.e.get(i)) {
            Some(e) => cmul::<S>(e.0, e.1, v.0, v.1),
            None => v,
        }
    }

    /// The oscillator vector `j` of the block.
    #[inline(always)]
    fn vector(&self, j: usize) -> (S::f32s, S::f32s) {
        let mut v = self.entry(j / CHAIN);
        if STEPPED {
            for _ in 0..j % CHAIN {
                v = self.power(1, v);
            }
            v
        } else {
            self.power(j % CHAIN, v)
        }
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
        let base = cmul::<S>(self.a.0, self.a.1, tr, ti);
        let mut v = base;
        for (r, (xv, yv)) in x
            .chunks_exact(2 * h)
            .zip(y.chunks_exact_mut(2 * h))
            .enumerate()
        {
            v = match (r, STEPPED) {
                (0, _) => base,
                (_, true) => self.power(1, v),
                (_, false) => self.power(r, base),
            };
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
    // `E = (e^{j 2 pi (h/2) inc / 2^32})^2` from the lane that holds sample `h/2`, and its
    // powers, in `f64`.
    let half = (0..h).find(|&l| sample_of::<S, L>(l) == h / 2).unwrap_or(0);
    let e1 = offsets.lane_squared(half);
    let mut e = [(S::f32s::splat(simd, 1.0), S::f32s::splat(simd, 0.0)); CHAIN - 1];
    let mut p = e1;
    for v in &mut e {
        *v = (
            S::f32s::splat(simd, p.0 as f32),
            S::f32s::splat(simd, p.1 as f32),
        );
        p = (p.0 * e1.0 - p.1 * e1.1, p.0 * e1.1 + p.1 * e1.0);
    }
    let mut osc = Osc {
        simd,
        split,
        a: offsets.at(simd, blocks.anchor),
        e,
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
    #[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
mod x86 {
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

/// Runs `vector` over the vectors of `N` values (`N / 2` samples) of `x` and `y`; the last
/// partial vector is processed zero-padded (every lane is computed on its own, so the result
/// is the same as for a whole vector).
#[inline(always)]
fn q15_vectors<const N: usize>(
    x: &[i16],
    y: &mut [i16],
    mut vector: impl FnMut(&[i16], &mut [i16]),
) {
    // Both sides cut to the common length, so that the loop has a single exit condition (with
    // run-time detection the kernel is not inlined behind the length check of the caller).
    let n = x.len().min(y.len());
    let (x, y) = (
        x.get(..n).unwrap_or_default(),
        y.get_mut(..n).unwrap_or_default(),
    );
    let mut xs = x.chunks_exact(N);
    let mut ys = y.chunks_exact_mut(N);
    for (xv, yv) in (&mut xs).zip(&mut ys) {
        vector(xv, yv);
    }
    let (xt, yt) = (xs.remainder(), ys.into_remainder());
    if !xt.is_empty() {
        let (mut tmp_in, mut tmp_out) = ([0i16; N], [0i16; N]);
        if let Some(d) = tmp_in.get_mut(..xt.len()) {
            d.copy_from_slice(xt);
        }
        vector(&tmp_in, &mut tmp_out);
        if let Some(s) = tmp_out.get(..yt.len()) {
            yt.copy_from_slice(s);
        }
    }
}

/// Rotates all `CQ15` samples with the generic code.
#[inline(always)]
fn rotate_q15_generic<S: SimdToken>(simd: S, phase: u32, inc: u32, input: &[i16], out: &mut [i16]) {
    let h = S::i32s::LEN;
    let ramp = S::u32s::from_fn(simd, |l| (l as u32).wrapping_mul(inc));
    let step = S::u32s::splat(simd, (h as u32).wrapping_mul(inc));
    let mut p = S::u32s::splat(simd, phase) + ramp;
    let vector = |x: &[i16], y: &mut [i16]| {
        q15_vector(simd, x, y, p);
        p += step;
    };
    match h {
        4 => q15_vectors::<8>(input, out, vector),
        8 => q15_vectors::<16>(input, out, vector),
        _ => q15_vectors::<32>(input, out, vector),
    }
}

/// Rotates all `CQ15` samples, with the path of the level where it has one.
#[inline(always)]
fn rotate_q15<S: SimdToken>(simd: S, phase: u32, inc: u32, input: &[i16], out: &mut [i16]) {
    let level = simd.level();
    #[cfg(all(feature = "nco-tables", target_arch = "x86_64"))]
    {
        if let Some(t) = level.as_avx512() {
            return x86_q15::rotate_avx512(t, phase, inc, input, out);
        }
        if let Some(t) = level.as_avx2() {
            return x86_q15::rotate_avx2(t, phase, inc, input, out);
        }
        if let Some(t) = level.as_sse4_2() {
            return x86_q15::rotate_sse4_2(t, phase, inc, input, out);
        }
        if let Some(t) = crate::level::sse2_only(level) {
            return x86_q15::rotate_sse2(t, phase, inc, input, out);
        }
    }
    #[cfg(all(feature = "nco-tables", target_arch = "aarch64"))]
    if let Some(t) = level.as_neon() {
        return neon_q15::rotate_neon(t, phase, inc, input, out);
    }
    // SIMD128 also without the static tables, which wasm32 builds leave out by default.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return wasm_q15::rotate_simd128(t, phase, inc, input, out);
    }
    let _ = level;
    rotate_q15_generic(simd, phase, inc, input, out);
}

/// The phases `phase + l inc` of the lanes `l < N`.
#[cfg(any(
    feature = "nco-tables",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn ramp<const N: usize>(phase: u32, inc: u32) -> [u32; N] {
    core::array::from_fn(|l| phase.wrapping_add((l as u32).wrapping_mul(inc)))
}

/// The `CQ15` paths of x86.
///
/// The oscillator is the table lookup of `table::x86` (bit-exact with the fallback), except
/// that the byte offsets of the entries go from the vector to the general registers through
/// one store and scalar loads instead of `movd`/`pextrd`: those are two micro-ops each on the
/// ports that also serve the vector loads on Zen (`FP45`), which bounded the loop there.
///
/// The complex multiplication uses `pmaddwd` on the `(I, Q)` pairs of the samples and the
/// oscillator. With `x = (xr, xi)` and the oscillator `(c, s)` in one 32-bit lane,
/// `pmaddwd(x, (s, c))` is `xr s + xi c` and `pmaddwd(x, (c, !s))` is `xr c + xi (-s - 1)`, so
/// that adding `xi` gives `xr c - xi s`: `!s` is the negation that cannot overflow (`s` can be
/// -32768). The sums are exact in `i32` (`|x| |(c, s)| < 2^31`), and the arithmetic shift by 15
/// after adding `2^14` with the saturating pack is the `round_shift` and `sat16` of
/// `cq15_mul`, bit for bit.
#[cfg(all(feature = "nco-tables", target_arch = "x86_64"))]
#[allow(unsafe_code)]
mod x86_q15 {
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128, __m128i, __m256i, __m512i, _mm_add_epi16, _mm_add_epi32, _mm_and_si128,
        _mm_castpd_si128, _mm_castps_si128, _mm_castsi128_pd, _mm_castsi128_ps, _mm_loadh_pd,
        _mm_loadl_epi64, _mm_madd_epi16, _mm_mulhrs_epi16, _mm_or_si128, _mm_packs_epi32,
        _mm_set1_epi16, _mm_set1_epi32, _mm_setr_epi8, _mm_shuffle_epi8, _mm_shuffle_ps,
        _mm_shufflehi_epi16, _mm_shufflelo_epi16, _mm_slli_epi32, _mm_srai_epi32, _mm_srli_epi32,
        _mm_storeu_si128, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64,
        _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_xor_si128, _mm256_add_epi16, _mm256_add_epi32,
        _mm256_and_si256, _mm256_castps_si256, _mm256_castsi128_si256, _mm256_castsi256_ps,
        _mm256_inserti128_si256, _mm256_madd_epi16, _mm256_mulhrs_epi16, _mm256_packs_epi32,
        _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi8, _mm256_shuffle_epi8,
        _mm256_shuffle_ps, _mm256_slli_epi32, _mm256_srai_epi32, _mm256_srli_epi32,
        _mm256_storeu_si256, _mm256_xor_si256, _mm512_add_epi32, _mm512_broadcast_i32x4,
        _mm512_madd_epi16, _mm512_packs_epi32, _mm512_rol_epi32, _mm512_set1_epi32,
        _mm512_shuffle_epi8, _mm512_srai_epi32, _mm512_xor_si512,
    };

    use fearless_simd::{SimdFrom, i16x8, i16x16, i16x32, prelude::*, u32x4, u32x8, u32x16};

    use super::{q15_vectors, ramp};
    use crate::table::COS_SIN_Q15;
    use crate::table::x86::cos_sin_q15_x16;

    /// `!0xFFFF`: the mask that turns `(c, s)` into `(c, !s)`.
    const NOT_SIN: i32 = !0xFFFF;
    /// The rounding term of the shift by 15.
    const ROUND: i32 = 1 << 14;

    /// The byte offsets `8 * (p >> 20)` of the table entries of four phases.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn offsets_x4(p: __m128i) -> [u32; 4] {
        let o = _mm_and_si128(_mm_srli_epi32::<17>(p), _mm_set1_epi32(0x7FF8));
        let mut buf = [0u32; 4];
        let ptr = buf.as_mut_ptr();
        // SAFETY: `buf` is the 16 bytes that the store writes and the four `u32` that the
        // loads read; the volatile loads keep the compiler from turning them into extracts.
        unsafe {
            _mm_storeu_si128(ptr.cast(), o);
            core::array::from_fn(|k| core::ptr::read_volatile(ptr.add(k)))
        }
    }

    /// The byte offsets `8 * (p >> 20)` of the table entries of eight phases.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn offsets_x8(p: __m256i) -> [u32; 8] {
        let o = _mm256_and_si256(_mm256_srli_epi32::<17>(p), _mm256_set1_epi32(0x7FF8));
        let mut buf = [0u32; 8];
        let ptr = buf.as_mut_ptr();
        // SAFETY: `buf` is the 32 bytes that the store writes and the eight `u32` that the
        // loads read; the volatile loads keep the compiler from turning them into extracts.
        unsafe {
            _mm256_storeu_si256(ptr.cast(), o);
            core::array::from_fn(|k| core::ptr::read_volatile(ptr.add(k)))
        }
    }

    /// The table entries at the byte offsets `a` and `b` (from [`offsets_x4`] or
    /// [`offsets_x8`]) as the low and the high half of a vector.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn pair(a: u32, b: u32) -> __m128 {
        let base = core::ptr::from_ref(&COS_SIN_Q15).cast::<u8>();
        // SAFETY: the offsets are lanes masked with `0x7FF8`, multiples of 8 below 32 KiB,
        // so each 8-byte load reads one entry of the 32 KiB table.
        unsafe {
            let lo = _mm_loadl_epi64(base.add(a as usize).cast());
            let v = _mm_loadh_pd(_mm_castsi128_pd(lo), base.add(b as usize).cast());
            _mm_castsi128_ps(_mm_castpd_si128(v))
        }
    }

    /// The table values `a` (the pairs `(cos, sin)`) and the differences `d` of four phases.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn entries_x4(p: __m128i) -> (__m128i, __m128i) {
        let [i0, i1, i2, i3] = offsets_x4(p);
        let (e01, e23) = (pair(i0, i1), pair(i2, i3));
        (
            _mm_castps_si128(_mm_shuffle_ps::<0x88>(e01, e23)),
            _mm_castps_si128(_mm_shuffle_ps::<0xDD>(e01, e23)),
        )
    }

    /// `(cos, sin)` as a `CQ15` pair per lane of four phases, with `pmulhrsw`, as
    /// `cos_sin_q15_x4` of the table.
    #[target_feature(enable = "sse4.2")]
    #[inline]
    fn lookup_x4(p: __m128i) -> __m128i {
        let (a, d) = entries_x4(p);
        // The fractions `(p >> 5) & 0x7FFF` in both halves: bytes 1 and 2 of `p << 3`.
        let dup = _mm_setr_epi8(1, 2, 1, 2, 5, 6, 5, 6, 9, 10, 9, 10, 13, 14, 13, 14);
        let f = _mm_shuffle_epi8(_mm_slli_epi32::<3>(p), dup);
        let f = _mm_and_si128(f, _mm_set1_epi16(0x7FFF));
        _mm_add_epi16(a, _mm_mulhrs_epi16(d, f))
    }

    /// [`lookup_x4`] on SSE2, as `cos_sin_q15_x4_sse2` of the table: the rounding product
    /// with `pmaddwd` on the pairs `(d, 1) * (f15, 2^14)`, a shift and a pack.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn lookup_x4_sse2(p: __m128i) -> __m128i {
        let (a, d) = entries_x4(p);
        let f = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi32::<5>(p), _mm_set1_epi32(0x7FFF)),
            _mm_set1_epi32(1 << 30),
        );
        let one = _mm_set1_epi16(1);
        let lo = _mm_madd_epi16(_mm_unpacklo_epi16(d, one), _mm_unpacklo_epi32(f, f));
        let hi = _mm_madd_epi16(_mm_unpackhi_epi16(d, one), _mm_unpackhi_epi32(f, f));
        let r = _mm_packs_epi32(_mm_srai_epi32::<15>(lo), _mm_srai_epi32::<15>(hi));
        _mm_add_epi16(a, r)
    }

    /// `(cos, sin)` as a `CQ15` pair per lane of eight phases, as `cos_sin_q15_x8` of the
    /// table.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn lookup_x8(p: __m256i) -> __m256i {
        let [i0, i1, i2, i3, i4, i5, i6, i7] = offsets_x8(p);
        let quad = |a: u32, b: u32, c: u32, d: u32| {
            let lo = _mm256_castsi128_si256(_mm_castps_si128(pair(a, b)));
            _mm256_castsi256_ps(_mm256_inserti128_si256::<1>(
                lo,
                _mm_castps_si128(pair(c, d)),
            ))
        };
        // [e0 e1 | e4 e5], [e2 e3 | e6 e7]: one in-lane shuffle gives samples 0 to 7 in order.
        let e0 = quad(i0, i1, i4, i5);
        let e1 = quad(i2, i3, i6, i7);
        let a = _mm256_castps_si256(_mm256_shuffle_ps::<0x88>(e0, e1));
        let d = _mm256_castps_si256(_mm256_shuffle_ps::<0xDD>(e0, e1));
        let dup = _mm256_setr_epi8(
            1, 2, 1, 2, 5, 6, 5, 6, 9, 10, 9, 10, 13, 14, 13, 14, 1, 2, 1, 2, 5, 6, 5, 6, 9, 10, 9,
            10, 13, 14, 13, 14,
        );
        let f = _mm256_shuffle_epi8(_mm256_slli_epi32::<3>(p), dup);
        let f = _mm256_and_si256(f, _mm256_set1_epi16(0x7FFF));
        _mm256_add_epi16(a, _mm256_mulhrs_epi16(d, f))
    }

    /// The byte shuffle that swaps the 16-bit halves of every 32-bit lane.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn swap_bytes() -> __m128i {
        _mm_setr_epi8(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13)
    }

    /// The byte shuffle that interleaves `[r0 r1 r2 r3 i0 i1 i2 i3]` to `[r0 i0 r1 i1 ...]`.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn zip_bytes() -> __m128i {
        _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15)
    }

    /// The rotated samples `x` (four) with the oscillator `cs` and its swapped form `sc`, as
    /// `[re0 re1 re2 re3 im0 im1 im2 im3]`.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn cmul_x4(x: __m128i, cs: __m128i, sc: __m128i) -> __m128i {
        let round = _mm_set1_epi32(ROUND);
        let cn = _mm_xor_si128(cs, _mm_set1_epi32(NOT_SIN));
        let re = _mm_add_epi32(
            _mm_madd_epi16(x, cn),
            _mm_add_epi32(_mm_srai_epi32::<16>(x), round),
        );
        let im = _mm_add_epi32(_mm_madd_epi16(x, sc), round);
        _mm_packs_epi32(_mm_srai_epi32::<15>(re), _mm_srai_epi32::<15>(im))
    }

    /// Four samples on SSSE3 and later: the halves swapped and interleaved with `pshufb`.
    #[target_feature(enable = "sse4.2")]
    #[inline]
    fn rotate_x4(x: __m128i, cs: __m128i) -> __m128i {
        let v = cmul_x4(x, cs, _mm_shuffle_epi8(cs, swap_bytes()));
        _mm_shuffle_epi8(v, zip_bytes())
    }

    /// Four samples on SSE2: the halves swapped with `pshuflw`/`pshufhw`, interleaved with an
    /// unpack.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn rotate_x4_sse2(x: __m128i, cs: __m128i) -> __m128i {
        let sc = _mm_shufflehi_epi16::<0xB1>(_mm_shufflelo_epi16::<0xB1>(cs));
        let v = cmul_x4(x, cs, sc);
        _mm_unpacklo_epi16(v, _mm_unpackhi_epi64(v, v))
    }

    /// The `vpshufb` control that swaps the 16-bit halves of every 32-bit lane, hidden from
    /// the optimiser. Without tuning for a CPU with a fast `pshufb` (the build with run-time
    /// detection) LLVM lowers the visible constant to `vpshuflw` and `vpshufhw`, one
    /// instruction more per vector; the opaque value costs one load per call.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn swap_x8() -> __m256i {
        core::hint::black_box(_mm256_setr_epi8(
            2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11,
            8, 9, 14, 15, 12, 13,
        ))
    }

    /// Eight samples; `swap` is [`swap_x8`].
    #[target_feature(enable = "avx2")]
    #[inline]
    fn rotate_x8(x: __m256i, cs: __m256i, swap: __m256i) -> __m256i {
        let zip = _mm256_setr_epi8(
            0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5,
            12, 13, 6, 7, 14, 15,
        );
        let round = _mm256_set1_epi32(ROUND);
        let cn = _mm256_xor_si256(cs, _mm256_set1_epi32(NOT_SIN));
        let sc = _mm256_shuffle_epi8(cs, swap);
        let re = _mm256_add_epi32(
            _mm256_madd_epi16(x, cn),
            _mm256_add_epi32(_mm256_srai_epi32::<16>(x), round),
        );
        let im = _mm256_add_epi32(_mm256_madd_epi16(x, sc), round);
        let v = _mm256_packs_epi32(_mm256_srai_epi32::<15>(re), _mm256_srai_epi32::<15>(im));
        _mm256_shuffle_epi8(v, zip)
    }

    /// Sixteen samples.
    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    fn rotate_x16(x: __m512i, cs: __m512i) -> __m512i {
        let zip = _mm512_broadcast_i32x4(zip_bytes());
        let round = _mm512_set1_epi32(ROUND);
        let cn = _mm512_xor_si512(cs, _mm512_set1_epi32(NOT_SIN));
        let sc = _mm512_rol_epi32::<16>(cs);
        let re = _mm512_add_epi32(
            _mm512_madd_epi16(x, cn),
            _mm512_add_epi32(_mm512_srai_epi32::<16>(x), round),
        );
        let im = _mm512_add_epi32(_mm512_madd_epi16(x, sc), round);
        let v = _mm512_packs_epi32(_mm512_srai_epi32::<15>(re), _mm512_srai_epi32::<15>(im));
        _mm512_shuffle_epi8(v, zip)
    }

    fearless_simd::kernel!(
        pub(super) fn rotate_sse2(t: Sse2, phase: u32, inc: u32, x: &[i16], y: &mut [i16]) {
            let mut p: __m128i = u32x4::from_slice(t, &ramp::<4>(phase, inc)).into();
            let step = _mm_set1_epi32(inc.wrapping_mul(4) as i32);
            q15_vectors::<8>(x, y, |xv, yv| {
                let v = rotate_x4_sse2(i16x8::from_slice(t, xv).into(), lookup_x4_sse2(p));
                i16x8::simd_from(t, v).store_slice(yv);
                p = _mm_add_epi32(p, step);
            });
        }
    );

    fearless_simd::kernel!(
        pub(super) fn rotate_sse4_2(t: Sse4_2, phase: u32, inc: u32, x: &[i16], y: &mut [i16]) {
            let mut p: __m128i = u32x4::from_slice(t, &ramp::<4>(phase, inc)).into();
            let step = _mm_set1_epi32(inc.wrapping_mul(4) as i32);
            q15_vectors::<8>(x, y, |xv, yv| {
                let v = rotate_x4(i16x8::from_slice(t, xv).into(), lookup_x4(p));
                i16x8::simd_from(t, v).store_slice(yv);
                p = _mm_add_epi32(p, step);
            });
        }
    );

    fearless_simd::kernel!(
        pub(super) fn rotate_avx2(t: Avx2, phase: u32, inc: u32, x: &[i16], y: &mut [i16]) {
            let mut p: __m256i = u32x8::from_slice(t, &ramp::<8>(phase, inc)).into();
            let step = _mm256_set1_epi32(inc.wrapping_mul(8) as i32);
            let swap = swap_x8();
            q15_vectors::<16>(x, y, |xv, yv| {
                let v = rotate_x8(i16x16::from_slice(t, xv).into(), lookup_x8(p), swap);
                i16x16::simd_from(t, v).store_slice(yv);
                p = _mm256_add_epi32(p, step);
            });
        }
    );

    fearless_simd::kernel!(
        pub(super) fn rotate_avx512(t: Avx512, phase: u32, inc: u32, x: &[i16], y: &mut [i16]) {
            let mut p: __m512i = u32x16::from_slice(t, &ramp::<16>(phase, inc)).into();
            let step = _mm512_set1_epi32(inc.wrapping_mul(16) as i32);
            q15_vectors::<32>(x, y, |xv, yv| {
                let v = rotate_x16(i16x32::from_slice(t, xv).into(), cos_sin_q15_x16(p));
                i16x32::simd_from(t, v).store_slice(yv);
                p = _mm512_add_epi32(p, step);
            });
        }
    );
}

/// The `CQ15` path of NEON: eight samples per step, deinterleaved by `ld2`, the products
/// widened with `smull`/`smlsl`/`smlal` and narrowed by `sqrshrn #15`, which is the
/// `round_shift` and `sat16` of `cq15_mul` (the sums are exact in `i32`).
#[cfg(all(feature = "nco-tables", target_arch = "aarch64"))]
#[allow(unsafe_code)]
mod neon_q15 {
    use core::arch::aarch64::{
        int16x8x2_t, vaddq_u32, vdupq_n_u32, vget_low_s16, vld1q_u32, vld2q_s16, vmlal_high_s16,
        vmlal_s16, vmlsl_high_s16, vmlsl_s16, vmull_high_s16, vmull_s16, vqrshrn_high_n_s32,
        vqrshrn_n_s32, vreinterpretq_s16_u32, vst2q_s16, vuzp1q_s16, vuzp2q_s16,
    };

    use super::{q15_vectors, ramp};
    use crate::table::neon::cos_sin_q15_x4;

    fearless_simd::kernel!(
        pub(super) fn rotate_neon(_t: Neon, phase: u32, inc: u32, x: &[i16], y: &mut [i16]) {
            let r = ramp::<4>(phase, inc);
            // SAFETY: `r` is the four `u32` that `vld1q_u32` reads.
            let mut p = unsafe { vld1q_u32(r.as_ptr()) };
            let half = vdupq_n_u32(inc.wrapping_mul(4));
            let step = vdupq_n_u32(inc.wrapping_mul(8));
            q15_vectors::<16>(x, y, |xv, yv| {
                let (Some(xv), Some(yv)) = (xv.first_chunk::<16>(), yv.first_chunk_mut::<16>())
                else {
                    return;
                };
                // SAFETY: `xv` is the 16 values that `vld2q_s16` reads.
                let v = unsafe { vld2q_s16(xv.as_ptr()) };
                let (xr, xi) = (v.0, v.1);
                let lo = vreinterpretq_s16_u32(cos_sin_q15_x4(p));
                let hi = vreinterpretq_s16_u32(cos_sin_q15_x4(vaddq_u32(p, half)));
                let (c, s) = (vuzp1q_s16(lo, hi), vuzp2q_s16(lo, hi));
                let (xr0, xi0) = (vget_low_s16(xr), vget_low_s16(xi));
                let (c0, s0) = (vget_low_s16(c), vget_low_s16(s));
                let re0 = vmlsl_s16(vmull_s16(xr0, c0), xi0, s0);
                let re1 = vmlsl_high_s16(vmull_high_s16(xr, c), xi, s);
                let im0 = vmlal_s16(vmull_s16(xr0, s0), xi0, c0);
                let im1 = vmlal_high_s16(vmull_high_s16(xr, s), xi, c);
                let re = vqrshrn_high_n_s32::<15>(vqrshrn_n_s32::<15>(re0), re1);
                let im = vqrshrn_high_n_s32::<15>(vqrshrn_n_s32::<15>(im0), im1);
                // SAFETY: `yv` is the 16 values that `vst2q_s16` writes.
                unsafe { vst2q_s16(yv.as_mut_ptr(), int16x8x2_t(re, im)) };
                p = vaddq_u32(p, step);
            });
        }
    );
}

/// The `CQ15` path of SIMD128: the `pmaddwd` form of the x86 paths with `i32x4.dot_i16x8_s`
/// and the saturating `i16x8.narrow_i32x4_s`. The oscillator comes from the entry table with
/// `nco-tables` and from `COS_Q15` itself without (the default of wasm32 builds).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[cfg_attr(not(feature = "nco-tables"), allow(unsafe_code))]
mod wasm_q15 {
    use core::arch::wasm32::{
        i8x16_shuffle, i16x8_narrow_i32x4, i32x4_add, i32x4_dot_i16x8, i32x4_shr, i32x4_splat,
        u32x4, v128, v128_xor,
    };
    #[cfg(not(feature = "nco-tables"))]
    use core::arch::wasm32::{
        i16x8_add, i16x8_q15mulr_sat, i16x8_sub, i32x4_shl, i32x4_sub, u32x4_extract_lane,
        u32x4_shr, u32x4_splat, v128_and, v128_or,
    };

    use fearless_simd::{i16x8, prelude::*};

    use super::{q15_vectors, ramp};
    #[cfg(feature = "nco-tables")]
    use crate::table::wasm::cos_sin_q15_x4;
    #[cfg(not(feature = "nco-tables"))]
    use sdrk_core::tables::COS_Q15;

    /// `(cos, sin)` as a `CQ15` pair per lane of four phases from `COS_Q15`, bit-exact with
    /// the fallback: each neighbour pair `COS_Q15[i], COS_Q15[i + 1]` is one 32-bit load, and
    /// `a + round_shift((b - a) f15, 15)` is `i16x8.q15mulr_sat_s` (see `table`).
    #[cfg(not(feature = "nco-tables"))]
    #[inline(always)]
    fn cos_sin_q15_x4(p: v128) -> v128 {
        let base = COS_Q15.as_ptr();
        // SAFETY: `i = p >> 20` is at most 4095, so the two values `i` and `i + 1` that the
        // unaligned 32-bit load reads are inside the 4097 entries of the table.
        let pair =
            |i: u32| unsafe { core::ptr::read_unaligned(base.add(i as usize).cast::<u32>()) };
        let ic = u32x4_shr(p, 20);
        let is = u32x4_shr(i32x4_sub(p, u32x4_splat(1 << 30)), 20);
        let lanes = |i: v128| {
            u32x4(
                pair(u32x4_extract_lane::<0>(i)),
                pair(u32x4_extract_lane::<1>(i)),
                pair(u32x4_extract_lane::<2>(i)),
                pair(u32x4_extract_lane::<3>(i)),
            )
        };
        let (c, s) = (lanes(ic), lanes(is));
        // [a_c a_s] and [b_c b_s] per sample.
        let a = i8x16_shuffle::<0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29>(c, s);
        let b = i8x16_shuffle::<2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31>(c, s);
        let f = v128_and(u32x4_shr(p, 5), u32x4_splat(0x7FFF));
        let f = v128_or(f, i32x4_shl(f, 16));
        i16x8_add(a, i16x8_q15mulr_sat(i16x8_sub(b, a), f))
    }

    fearless_simd::kernel!(
        pub(super) fn rotate_simd128(
            t: WasmSimd128,
            phase: u32,
            inc: u32,
            x: &[i16],
            y: &mut [i16],
        ) {
            let [p0, p1, p2, p3] = ramp::<4>(phase, inc);
            let mut p = u32x4(p0, p1, p2, p3);
            let step = i32x4_splat(inc.wrapping_mul(4) as i32);
            let round = i32x4_splat(1 << 14);
            let not_sin = i32x4_splat(!0xFFFF);
            q15_vectors::<8>(x, y, |xv, yv| {
                let x: v128 = i16x8::from_slice(t, xv).into();
                let cs = cos_sin_q15_x4(p);
                let sc =
                    i8x16_shuffle::<2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13>(cs, cs);
                let re = i32x4_add(
                    i32x4_dot_i16x8(x, v128_xor(cs, not_sin)),
                    i32x4_add(i32x4_shr(x, 16), round),
                );
                let im = i32x4_add(i32x4_dot_i16x8(x, sc), round);
                let v = i16x8_narrow_i32x4(i32x4_shr(re, 15), i32x4_shr(im, 15));
                let v = i8x16_shuffle::<0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15>(v, v);
                i16x8::simd_from(t, v).store_slice(yv);
                p = i32x4_add(p, step);
            });
        }
    );
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
