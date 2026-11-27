//! The FIR family: `fir`, `fir_decimate` and the polyphase interpolator `fir_interpolate`
//! (also exposed as `resample`), and the polyphase kernels of real signals `resample_real` and
//! `fir_interpolate_real`.
//!
//! Every output is one dot product of the taps with a window of the input. The window may reach
//! back into the filter history, so the input is seen as the virtual stream `history ++ input`.
//!
//! - `C32` `fir` and `fir_decimate` compute several outputs at a time with the taps of a block of
//!   [`BLOCK_TAPS`] (a longer filter adds the partial dot products of its blocks in the output).
//!   Consecutive outputs (`fir`, and `fir_decimate` by 1) use the transposed form: a vector holds
//!   consecutive outputs and every broadcast tap multiplies the samples at its offset, so there
//!   is no horizontal sum. Decimation uses the chunk form: the dot product of every window with
//!   the taps in chunks of half a vector of complex samples, several windows sharing every tap
//!   vector. Both sum the products in short chains per block of taps (see the forms for the
//!   exact order). Runs of outputs whose windows lie in the input read it in place; the others
//!   (the windows that reach into the history, and those that reach past the end of the input)
//!   are computed in the same way from a copy in a scratch window, so an output is bit-identical
//!   whichever way the input was split into blocks. The levels with fused multiply-add (AVX2,
//!   AVX-512, NEON) use it (decision D5); the others multiply and add separately.
//! - `C32` `resample` and `fir_interpolate` run every branch of the polyphase filter as such a
//!   contiguous filter: the outputs with the same phase have their windows `M` samples apart
//!   (see [`polyphase_f32`]), and the filter length is not limited. `f32` `resample_real` and
//!   `fir_interpolate_real` run the same code with one component per sample instead of two (the
//!   const parameter `C` of the forms).
//! - `CQ15` `fir`, `fir_decimate` and the polyphase kernels, and `Q15` `resample_real` and
//!   `fir_interpolate_real`, run their outputs in progressions of four outputs per pass of the
//!   taps (the module `prog`; decision D340 at SSE4.2, D3007 on the other x86-64 levels, NEON
//!   and SIMD128): `pmaddwd` (`i32x4.dot_i16x8_s`) with taps split into `256 hi + lo` on x86-64
//!   and SIMD128, `smull` and `sadalp` into 64-bit lanes on NEON. The taps are prepared once per
//!   call (reversed, duplicated for the real and the imaginary lane); the polyphase kernels
//!   prepare those of all branches at once, and a filter that does not fit, and every `CQ15`
//!   filter with more than [`PREP_TAPS`] taps, runs on the fallback. The portable level (no SIMD
//!   level) computes one output at a time in the chunk form, a window that reaches into the
//!   history assembled chunk by chunk with the same chunk boundaries. No kernel and level runs
//!   the fallback in place of its branch any more (the switch of D65 is removed, D3007).
//!
//! `C32` and `f32` results are within the contract of `docs/kernels.md` (110 dB) but not
//! bit-identical with the fallback, because the sums are formed in a different order (and fused on
//! the levels with FMA). `CQ15` and `Q15` results are exact integer sums and therefore
//! bit-identical.

use fearless_simd::{Simd as SimdToken, f32x4, f32x8, f32x16, f64x2, prelude::*};
use sdrk_core::kernels::decim::{
    FirDecimState, FirDecimate, Resample, ResampleReal, ResampleRealState, ResampleState,
};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::interp::{
    FirInterpRealState, FirInterpState, FirInterpolate, FirInterpolateReal, SHORT_MAX_BRANCH,
    short_branches,
};
use sdrk_core::num::Acc64;
use sdrk_core::{C32, CQ15, Fallback, Progress, Q15, Result};

use crate::Simd;
use crate::level::dispatch;
use crate::util::{check_len, internal};

/// The longest `CQ15` contiguous filter (`fir`, `fir_decimate`) of this backend.
const PREP_TAPS: usize = 256;
/// Complex samples (window positions, whole chunks) in the prepared storage of one call (twice
/// as many real samples); the prepared branches of a polyphase filter must fit.
const CAPACITY: usize = 384;

/// The virtual stream `history ++ input`, both as interleaved component slices of `C`
/// components per sample (2 for complex samples, 1 for real ones).
struct Win<'a, T, const C: usize> {
    hist: &'a [T],
    input: &'a [T],
    /// Number of history samples.
    h: usize,
}

impl<'a, T: Copy + Default, const C: usize> Win<'a, T, C> {
    fn new(hist: &'a [T], input: &'a [T]) -> Self {
        Self {
            hist,
            input,
            h: hist.len() / C,
        }
    }

    /// `lanes` contiguous components from sample `pos` on, if they lie inside one slice.
    #[inline(always)]
    fn direct(&self, pos: usize, lanes: usize) -> Option<&'a [T]> {
        if pos >= self.h {
            let o = C * (pos - self.h);
            self.input.get(o..o + lanes)
        } else {
            let o = C * pos;
            self.hist.get(o..o + lanes)
        }
    }
}

/// Copies the samples `s .. end` of the virtual stream `hist ++ input` (interleaved component
/// slices of `C` components per sample), as far as they exist, to the start of `dst` (the rest
/// of `dst` is untouched): the part in the history, then the part in the input.
#[inline(always)]
fn fill_window<T: Copy, const C: usize>(
    hist: &[T],
    input: &[T],
    s: usize,
    end: usize,
    dst: &mut [T],
) {
    let (h, len) = (hist.len() / C, input.len() / C);
    if s < h
        && let (Some(src), Some(dst)) = (
            hist.get(C * s..C * end.min(h)),
            dst.get_mut(..C * (end.min(h) - s)),
        )
    {
        dst.copy_from_slice(src);
    }
    let (from, to) = (s.max(h), end.min(h + len));
    if from < to
        && let (Some(src), Some(dst)) = (
            input.get(C * (from - h)..C * (to - h)),
            dst.get_mut(C * (from - s)..C * (to - s)),
        )
    {
        dst.copy_from_slice(src);
    }
}

/// Geometry of the prepared branches of a polyphase filter with `n` taps and factor `up` for
/// chunks of `g` samples of `c` components (each tap is stored `c` times).
#[derive(Clone, Copy)]
struct Branches {
    n: usize,
    up: usize,
    g: usize,
    c: usize,
}

impl Branches {
    /// Taps of the branch `phase`.
    fn len(self, phase: usize) -> usize {
        if phase < self.n {
            (self.n - phase).div_ceil(self.up)
        } else {
            0
        }
    }

    /// Samples reserved for a branch of `t` taps (whole chunks).
    fn padded(self, t: usize) -> usize {
        t.div_ceil(self.g) * self.g
    }

    /// Offset in components of the branch `phase` in the prepared storage.
    fn offset(self, phase: usize) -> usize {
        let hi = self.n.div_ceil(self.up);
        let r = self.n - (hi - 1) * self.up; // branches 0..r have `hi` taps, the others `hi - 1`
        let first = phase.min(r);
        self.c * (first * self.padded(hi) + (phase - first) * self.padded(hi - 1))
    }

    /// Whether all branches fit into the storage.
    fn fits(self) -> bool {
        if self.up == 0 || self.n == 0 {
            return false;
        }
        let hi = self.n.div_ceil(self.up);
        let r = self.n - (hi - 1) * self.up;
        let branches = self.up.min(self.n);
        let total =
            r.min(branches) * self.padded(hi) + branches.saturating_sub(r) * self.padded(hi - 1);
        self.c * total <= 2 * CAPACITY
    }
}

// ---------------------------------------------------------------------------------------------
// C32 contiguous filters (`fir`, `fir_decimate`, the polyphase branches)

/// Taps in one block of the `C32` and `f32` contiguous filters. A longer filter adds the partial dot
/// products of its blocks in the output.
const BLOCK_TAPS: usize = 128;
/// Samples of the scratch window that serves the outputs whose windows reach into the history
/// or past the end of the input.
const SCRATCH: usize = 192;
/// Lane indices, for the mask of the partial last chunk.
const IOTA: [f32; 16] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
];

/// The `f32` vectors the `C32` and `f32` filters run on: `f32x4`, `f32x8` or `f32x16`, the native width
/// of the level.
trait Lanes<S: SimdToken>: SimdFloat<S, Element = f32> {
    /// Adds the upper half of the complex lanes to the lower half until two are left.
    fn fold(self) -> f32x4<S>;
}

impl<S: SimdToken> Lanes<S> for f32x4<S> {
    #[inline(always)]
    fn fold(self) -> f32x4<S> {
        self
    }
}

impl<S: SimdToken> Lanes<S> for f32x8<S> {
    #[inline(always)]
    fn fold(self) -> f32x4<S> {
        let (lo, hi) = self.split();
        lo + hi
    }
}

impl<S: SimdToken> Lanes<S> for f32x16<S> {
    #[inline(always)]
    fn fold(self) -> f32x4<S> {
        let (lo, hi) = self.split();
        (lo + hi).fold()
    }
}

/// `[sum a, sum b]`: the sums of the two complex lanes of `a` and of `b`.
#[inline(always)]
fn pair_sums<S: SimdToken>(a: f32x4<S>, b: f32x4<S>) -> f32x4<S> {
    let (a, b): (f64x2<S>, f64x2<S>) = (a.bitcast(), b.bitcast());
    a.zip_low(b).bitcast::<f32x4<S>>() + a.zip_high(b).bitcast::<f32x4<S>>()
}

/// `[sum a, sum b, sum c, sum d]`: the sums of the four lanes of each (real samples). Every sum
/// is `(x0 + x2) + (x1 + x3)` whatever the other vectors are.
#[inline(always)]
fn quad_sums<S: SimdToken>(a: f32x4<S>, b: f32x4<S>, c: f32x4<S>, d: f32x4<S>) -> f32x4<S> {
    let (p, r) = (pair_sums(a, b), pair_sums(c, d));
    p.unzip_low(r) + p.unzip_high(r)
}

/// `acc + a * b`, fused on the levels with FMA (decision D5). Elsewhere it is the separate
/// multiplication and addition, except with the feature `relaxed-simd` on `wasm32` built with the
/// target feature `relaxed-simd`: there it is one relaxed multiply-add (fused or not, whichever
/// the host runs faster), within the contract of `docs/kernels.md`.
#[inline(always)]
fn madd<S: SimdToken, V: Lanes<S>, const FMA: bool>(a: V, b: V, acc: V) -> V {
    if FMA {
        a.mul_add(b, acc)
    } else {
        #[cfg(all(
            feature = "relaxed-simd",
            target_arch = "wasm32",
            target_feature = "relaxed-simd"
        ))]
        {
            a.mul_add(b, acc)
        }
        #[cfg(not(all(
            feature = "relaxed-simd",
            target_arch = "wasm32",
            target_feature = "relaxed-simd"
        )))]
        {
            acc + a * b
        }
    }
}

/// The sum of `A` accumulators as a balanced tree: `a0`, `a0 + a1`, `(a0 + a1) + (a2 + a3)`
/// (`zero` for none).
#[inline(always)]
fn tree<S: SimdToken, V: Lanes<S>, const A: usize>(acc: [V; A], zero: V) -> V {
    match acc.as_slice() {
        [a0, a1, a2, a3] => (*a0 + *a1) + (*a2 + *a3),
        [a0, a1] => *a0 + *a1,
        [a0, rest @ ..] => rest.iter().fold(*a0, |s, a| s + *a),
        [] => zero,
    }
}

/// Stores (or adds to `out`) the first `out.len()` floats of `v`.
#[inline(always)]
fn put<S: SimdToken, V: Lanes<S>>(v: V, out: &mut [f32], add: bool) {
    let simd = v.token();
    match out.get_mut(..V::LEN) {
        Some(full) => {
            if add {
                (V::from_slice(simd, full) + v).store_slice(full);
            } else {
                v.store_slice(full);
            }
        }
        None => {
            for (o, x) in out.iter_mut().zip(v.as_slice()) {
                *o = if add { *o + *x } else { *x };
            }
        }
    }
}

/// How the outputs of one tap block are computed from a region of samples. The window of the
/// `i`-th output of a run begins at sample `i * step` of the region.
trait Form {
    /// Samples the windows of a run of `k >= 1` outputs read.
    fn needed(&self, k: usize) -> usize;

    /// The longest run whose windows lie in `avail` samples (`needed` of it is at most `avail`).
    fn fits(&self, avail: usize) -> usize;

    /// Computes the outputs of a run whose windows lie in `region` (`C` floats each), or adds
    /// them to `out` (`add`).
    fn run(&self, region: &[f32], out: &mut [f32], add: bool);

    /// The outputs of one full pass: a run of a multiple of them has no shorter last pass.
    fn pass_outputs(&self) -> usize;
}

/// A unit of the prepared taps that loads as one vector: a tap to broadcast (`f32`), four taps
/// or a broadcast tap on a 16-byte boundary ([`Quad`]), or a chunk of `N` taps.
trait Unit: bytemuck::Pod {
    /// The unit as a vector of the level.
    fn load<S: SimdToken, V: Lanes<S>>(&self, simd: S) -> V;
}

impl Unit for f32 {
    #[inline(always)]
    fn load<S: SimdToken, V: Lanes<S>>(&self, simd: S) -> V {
        V::splat(simd, *self)
    }
}

impl Unit for Quad {
    #[inline(always)]
    fn load<S: SimdToken, V: Lanes<S>>(&self, simd: S) -> V {
        // The copy is a load of the aligned type.
        let q = *self;
        V::from_slice(simd, &q.0)
    }
}

macro_rules! chunk_unit {
    ($($n:literal),*) => {$(
        impl Unit for [f32; $n] {
            #[inline(always)]
            fn load<S: SimdToken, V: Lanes<S>>(&self, simd: S) -> V {
                V::from_slice(simd, self)
            }
        }
    )*};
}
chunk_unit!(4, 8, 16);

/// A unit of the taps of the transposed form: one tap, plain or broadcast.
trait TapUnit: Unit {
    /// The unit of the tap `t`.
    fn of(t: f32) -> Self;
}

impl TapUnit for f32 {
    #[inline(always)]
    fn of(t: f32) -> Self {
        t
    }
}

impl TapUnit for Quad {
    #[inline(always)]
    fn of(t: f32) -> Self {
        Quad([t; 4])
    }
}

/// The chunk form: the dot product of a window with the prepared taps in chunks of `L / C`
/// samples of `C` components (unit `T`), for `B` outputs at a time that share every tap vector;
/// the windows are `step` samples apart, so this serves every decimation factor. The chunks run
/// in groups of `U`; the products of a group are summed in `I` interleaved chains (chunk `u` to
/// chain `u % I`), which are added and then added to the output's total. The chunks after the
/// last group form one more such group. The lanes of the partial last chunk beyond the window
/// are cleared before the multiplication, so a NaN or an infinity there does not leak in. The
/// lanes of a total are added by halving down to four, then two outputs (complex samples) or
/// four (real samples) share the last steps.
struct ChunkForm<
    'a,
    S: SimdToken,
    V: Lanes<S>,
    T: Unit,
    const L: usize,
    const C: usize,
    const B: usize,
    const U: usize,
    const I: usize,
    const FMA: bool,
> {
    simd: S,
    /// The prepared taps in chunks, the last one partial if `mask` is set (zero beyond the
    /// window).
    taps: &'a [T],
    /// The lanes of the partial chunk inside the window, if there is a partial chunk.
    mask: Option<V::Mask>,
    /// Samples from the window of one output to the window of the next.
    step: usize,
}

impl<
    S: SimdToken,
    V: Lanes<S>,
    T: Unit,
    const L: usize,
    const C: usize,
    const B: usize,
    const U: usize,
    const I: usize,
    const FMA: bool,
> ChunkForm<'_, S, V, T, L, C, B, U, I, FMA>
{
    /// Samples one window reads (whole chunks).
    #[inline(always)]
    fn span(&self) -> usize {
        self.taps.len() * (L / C)
    }

    /// Adds tap chunk `t` times the samples `d[b]` to chain `u % I` of every output `b` (the
    /// first product of a chain starts it), clearing the lanes of the samples outside `mask`.
    #[inline(always)]
    fn chunk(
        &self,
        chains: &mut [[V; I]; B],
        u: usize,
        t: &T,
        d: [&[f32; L]; B],
        mask: Option<V::Mask>,
    ) {
        let simd = self.simd;
        let zero = V::splat(simd, 0.0);
        let t = t.load::<S, V>(simd);
        for (c, d) in chains.iter_mut().zip(d) {
            if let Some(x) = c.get_mut(u % I) {
                let mut d = V::from_slice(simd, d);
                if let Some(m) = mask {
                    d = m.select(d, zero);
                }
                *x = if u < I {
                    d * t
                } else {
                    madd::<S, V, FMA>(d, t, *x)
                };
            }
        }
    }

    /// The totals of the `B` outputs whose windows (the chunks of one [`span`](Self::span)
    /// each) are `wins`.
    #[inline(always)]
    fn pass(&self, wins: [&[[f32; L]]; B]) -> [V; B] {
        let simd = self.simd;
        let zero = V::splat(simd, 0.0);
        let nc = self.taps.len();
        // The partial chunk is the last chunk of the last group or of the chunks after the
        // groups.
        let (ng, nr) = (nc / U, nc % U);
        let in_group = self.mask.is_some() && nr == 0;
        let plain = ng - usize::from(in_group);
        let none = [0.0; L];
        // The chunks of every window: the groups and the chunks after them.
        let mut groups: [&[[[f32; L]; U]]; B] = [&[]; B];
        let mut rest: [&[[f32; L]]; B] = [&[]; B];
        for (b, c) in wins.into_iter().enumerate() {
            let (Some(g), Some(r), Some(dg), Some(dr)) = (
                c.as_chunks::<U>().0.get(..ng),
                c.get(ng * U..nc),
                groups.get_mut(b),
                rest.get_mut(b),
            ) else {
                return [zero; B];
            };
            *dg = g;
            *dr = r;
        }
        let (Some(tg), Some(tr)) = (
            self.taps.as_chunks::<U>().0.get(..ng),
            self.taps.get(ng * U..nc),
        ) else {
            return [zero; B];
        };
        let mut tot = [zero; B];
        // `tg` holds `ng >= plain` groups. Leaving the loop on a miss (instead of skipping the
        // group) keeps the dead start of the chains and the extra branch out of the loop.
        for k in 0..plain {
            let Some(t) = tg.get(k) else {
                break;
            };
            let mut chains = [[zero; I]; B];
            for (u, t) in t.iter().enumerate() {
                let d = groups.map(|g| g.get(k).and_then(|g| g.get(u)).unwrap_or(&none));
                self.chunk(&mut chains, u, t, d, None);
            }
            for (t, c) in tot.iter_mut().zip(chains) {
                *t += tree(c, zero);
            }
        }
        if let (true, Some(t)) = (in_group, tg.get(plain)) {
            let mut chains = [[zero; I]; B];
            for (u, t) in t.iter().enumerate() {
                let d = groups.map(|g| g.get(plain).and_then(|g| g.get(u)).unwrap_or(&none));
                let mask = if u + 1 == U { self.mask } else { None };
                self.chunk(&mut chains, u, t, d, mask);
            }
            for (t, c) in tot.iter_mut().zip(chains) {
                *t += tree(c, zero);
            }
        }
        if nr > 0 {
            let mut chains = [[zero; I]; B];
            for u in 0..nr {
                if let Some(t) = tr.get(u) {
                    let d = rest.map(|r| r.get(u).unwrap_or(&none));
                    let mask = if u + 1 == nr { self.mask } else { None };
                    self.chunk(&mut chains, u, t, d, mask);
                }
            }
            for (t, c) in tot.iter_mut().zip(chains) {
                *t += tree(c, zero);
            }
        }
        tot
    }

    /// Stores (or adds) the totals of `B` consecutive outputs to `out`.
    #[inline(always)]
    fn put_all(tot: [V; B], out: &mut [f32], add: bool) {
        // A loop, not `map`: the compiler may keep the closure of `map` out of line, and a
        // closure does not inherit the target features of the level (the run-time detection
        // build then passes every vector through memory).
        let Some(first) = tot.first() else {
            return;
        };
        let mut sums = [f32x4::splat(first.token(), 0.0); B];
        for (s, t) in sums.iter_mut().zip(tot) {
            *s = t.fold();
        }
        if C == 1 {
            // Real samples: four outputs share the last steps.
            let (quads, rest) = sums.as_chunks::<4>();
            let mut d = out.chunks_mut(4);
            for (&[a, b, c, e], d) in quads.iter().zip(&mut d) {
                put(quad_sums(a, b, c, e), d, add);
            }
            // The outputs after the last four are padded with the last of them; `d` holds
            // only theirs.
            if let (Some(&last), Some(d)) = (rest.last(), d.next()) {
                // No closure: a closure does not inherit the target features of the level.
                let mut v = [last; 3];
                for (v, r) in v.iter_mut().zip(rest) {
                    *v = *r;
                }
                put(quad_sums(v[0], v[1], v[2], last), d, add);
            }
            return;
        }
        let (pairs, odd) = sums.as_chunks::<2>();
        let mut d = out.chunks_mut(4);
        for (&[a, b], d) in pairs.iter().zip(&mut d) {
            put(pair_sums(a, b), d, add);
        }
        // An odd last output is paired with itself and stores its half.
        if let (Some(&a), Some(d)) = (odd.first(), d.next()) {
            let len = d.len().min(2);
            put(pair_sums(a, a), d.get_mut(..len).unwrap_or(&mut []), add);
        }
    }
}

impl<
    S: SimdToken,
    V: Lanes<S>,
    T: Unit,
    const L: usize,
    const C: usize,
    const B: usize,
    const U: usize,
    const I: usize,
    const FMA: bool,
> Form for ChunkForm<'_, S, V, T, L, C, B, U, I, FMA>
{
    #[inline(always)]
    fn needed(&self, k: usize) -> usize {
        (k - 1) * self.step + self.span()
    }

    #[inline(always)]
    fn pass_outputs(&self) -> usize {
        B
    }

    #[inline(always)]
    fn fits(&self, avail: usize) -> usize {
        let span = self.span();
        if avail >= span {
            (avail - span) / self.step + 1
        } else {
            0
        }
    }

    #[inline(always)]
    fn run(&self, region: &[f32], out: &mut [f32], add: bool) {
        let k = out.len() / C;
        let (step, len) = (C * self.step, self.taps.len() * L);
        let mut i = 0;
        for dst in out.chunks_exact_mut(C * B) {
            let o = i * step;
            let Some(pass) = region.get(o..o + (B - 1) * step + len) else {
                return;
            };
            // The windows of the pass. On x86 every window is checked against the pass slice of
            // known length, which lets the window pointers derive from one base; elsewhere
            // stepping through the windows of the slice measured fewer instructions.
            #[cfg(target_arch = "x86_64")]
            let wins: [&[[f32; L]]; B] = core::array::from_fn(|b| {
                pass.get(b * step..b * step + len)
                    .map_or(&[][..], |w| w.as_chunks::<L>().0)
            });
            #[cfg(not(target_arch = "x86_64"))]
            let wins = {
                let mut it = pass.windows(len).step_by(step);
                let mut wins: [&[[f32; L]]; B] = [&[]; B];
                for w in &mut wins {
                    let Some(x) = it.next() else {
                        return;
                    };
                    *w = x.as_chunks::<L>().0;
                }
                wins
            };
            Self::put_all(self.pass(wins), dst, add);
            i += B;
        }
        if i < k {
            // The last pass repeats the window of the last output.
            let mut wins: [&[[f32; L]]; B] = [&[]; B];
            for (b, w) in wins.iter_mut().enumerate() {
                let o = (i + b).min(k - 1) * step;
                let Some(x) = region.get(o..o + len) else {
                    return;
                };
                *w = x.as_chunks::<L>().0;
            }
            Self::put_all(
                self.pass(wins),
                out.get_mut(C * i..).unwrap_or(&mut []),
                add,
            );
        }
    }
}

/// Taps per block of the transposed form.
const TBLOCK: usize = 8;

/// The transposed form for consecutive outputs (`step = 1`): a vector holds `L / C`
/// consecutive outputs, and every reversed tap multiplies the samples at its offset for `B` such
/// vectors at a time, so there is no partial chunk and no horizontal sum. The vectors of a pass
/// are `D` samples apart (`L / C` or a multiple of it): on x86, where the samples are the memory
/// operand of the multiplication, `D >= TBLOCK` keeps the loads of different taps apart (the
/// compiler would otherwise keep the overlapping ones in registers and spill), and a run covers
/// the outputs in between with further passes. The products of a block of [`TBLOCK`] taps are
/// summed in `I` interleaved chains (tap `m` to chain `m % I`), which are added and then added to
/// the output's total (the last block may be shorter). A tap is a unit `T`, broadcast at use or
/// stored broadcast.
struct TransForm<
    'a,
    S: SimdToken,
    V: Lanes<S>,
    T: TapUnit,
    const L: usize,
    const C: usize,
    const B: usize,
    const D: usize,
    const I: usize,
    const FMA: bool,
> {
    simd: S,
    /// The taps of the block, reversed: `taps[m]` pairs with sample `m` of the window.
    taps: &'a [T],
    _lanes: core::marker::PhantomData<V>,
}

impl<
    S: SimdToken,
    V: Lanes<S>,
    T: TapUnit,
    const L: usize,
    const C: usize,
    const B: usize,
    const D: usize,
    const I: usize,
    const FMA: bool,
> TransForm<'_, S, V, T, L, C, B, D, I, FMA>
{
    /// Adds the block of taps `tb` to `tot` for the `P` output vectors `D` samples apart whose
    /// windows begin at `w` (floats; tap `m` reads from float `C m` on).
    #[inline(always)]
    fn block<const P: usize>(&self, tb: &[T], w: &[f32], tot: &mut [V; P]) {
        let simd = self.simd;
        let mut chains = [[V::splat(simd, 0.0); I]; P];
        for (m, t) in tb.iter().enumerate() {
            let t = t.load::<S, V>(simd);
            for (p, c) in chains.iter_mut().enumerate() {
                let o = C * (m + p * D);
                if let (Some(d), Some(x)) = (w.get(o..o + L), c.get_mut(m % I)) {
                    let d = V::from_slice(simd, d);
                    *x = if m < I {
                        d * t
                    } else {
                        madd::<S, V, FMA>(d, t, *x)
                    };
                }
            }
        }
        // A chain that got no tap (a last block shorter than `I`) is zero.
        for (t, c) in tot.iter_mut().zip(chains) {
            *t += tree(c, V::splat(simd, 0.0));
        }
    }

    /// The `P` output vectors, `D` samples apart, whose windows begin at `x`, if `x` holds them.
    #[inline(always)]
    fn pass<const P: usize>(&self, x: &[f32]) -> Option<[V; P]> {
        let n = self.taps.len();
        let x = x.get(..C * ((P - 1) * D + L / C + n - 1))?;
        let span = C * ((P - 1) * D + L / C + TBLOCK - 1);
        let mut tot = [V::splat(self.simd, 0.0); P];
        let (blocks, rest) = self.taps.as_chunks::<TBLOCK>();
        for (k, tb) in blocks.iter().enumerate() {
            let w = x.get(C * k * TBLOCK..C * k * TBLOCK + span)?;
            self.block(tb, w, &mut tot);
        }
        if !rest.is_empty() {
            let w = x.get(C * blocks.len() * TBLOCK..)?;
            self.block(rest, w, &mut tot);
        }
        Some(tot)
    }
}

impl<
    S: SimdToken,
    V: Lanes<S>,
    T: TapUnit,
    const L: usize,
    const C: usize,
    const B: usize,
    const D: usize,
    const I: usize,
    const FMA: bool,
> Form for TransForm<'_, S, V, T, L, C, B, D, I, FMA>
{
    #[inline(always)]
    fn needed(&self, k: usize) -> usize {
        k.div_ceil(L / C) * (L / C) + self.taps.len() - 1
    }

    #[inline(always)]
    fn pass_outputs(&self) -> usize {
        B * D
    }

    #[inline(always)]
    fn fits(&self, avail: usize) -> usize {
        let g = L / C;
        let n = self.taps.len();
        if avail + 1 >= n + g {
            (avail + 1 - n) / g * g
        } else {
            0
        }
    }

    #[inline(always)]
    fn run(&self, region: &[f32], out: &mut [f32], add: bool) {
        let g = L / C;
        let k = out.len() / C;
        let mut i = 0;
        // `D / g` passes of `B` vectors cover the `B D` outputs from `i` on.
        for dst in out.chunks_exact_mut(C * B * D) {
            for q in 0..D / g {
                let x = region.get(C * (i + q * g)..).unwrap_or(&[]);
                let Some(v) = self.pass::<B>(x) else {
                    return;
                };
                let vectors = dst.as_chunks_mut::<L>().0.iter_mut().skip(q).step_by(D / g);
                for (v, d) in v.into_iter().zip(vectors) {
                    put(v, d, add);
                }
            }
            i += B * D;
        }
        while i < k {
            let x = region.get(C * i..).unwrap_or(&[]);
            let Some([v]) = self.pass::<1>(x) else {
                return;
            };
            put(v, out.get_mut(C * i..).unwrap_or(&mut []), add);
            i += g;
        }
    }
}

/// Samples at the end of the scratch window that hold the outputs of a run through the scratch
/// when the outputs are strided.
const SCRATCH_OUT: usize = 48;

/// The outputs `j, j + 1, ...` of a strided output: every `stride`-th sample of `out` from sample
/// `j * stride` on.
#[inline(always)]
fn strided<const C: usize>(
    out: &mut [f32],
    j: usize,
    stride: usize,
) -> impl Iterator<Item = &mut [f32]> {
    out.chunks_exact_mut(C).skip(j * stride).step_by(stride)
}

/// Runs `f` on `region` for the `y.len() / C` outputs from output `j` of the strided output
/// `out`, through `y`: with `add` the outputs are first gathered into `y`, and afterwards `y` is
/// spread over them.
#[inline(always)]
fn run_strided<F: Form, const C: usize>(
    f: &F,
    region: &[f32],
    y: &mut [f32],
    out: &mut [f32],
    j: usize,
    stride: usize,
    add: bool,
) {
    // Output `j + i` is the sample `(j + i) stride` of `out`. `step_by` keeps the step to the next
    // output a pointer addition: splitting off the rest of `out` per output made a chain of bound
    // checks and conditional moves. (Chunks of `stride` samples and an index per output simulated
    // a little faster at `avx2`, but their run-time detection builds took 1.02 to 1.13 times the
    // instructions of the `avx2` build.)
    let samples = out.as_chunks_mut::<C>().0;
    let first = j * stride;
    if add {
        let from = samples
            .get(first..)
            .unwrap_or_default()
            .iter()
            .step_by(stride);
        for (d, s) in y.as_chunks_mut::<C>().0.iter_mut().zip(from) {
            *d = *s;
        }
    }
    f.run(region, y, add);
    let to = samples
        .get_mut(first..)
        .unwrap_or_default()
        .iter_mut()
        .step_by(stride);
    for (d, s) in to.zip(y.as_chunks::<C>().0) {
        *d = *s;
    }
}

/// The outputs of one tap block: output `j` (of `out.len() / C`) has its window at sample
/// `first + j * step` of the virtual stream `hist ++ input`. Runs whose windows lie inside the
/// input read it in place; the others are copied to `scratch` first, so every output is computed
/// by the same arithmetic wherever its window lies.
#[inline(always)]
fn drive<F: Form, const C: usize>(
    f: &F,
    hist: &[f32],
    input: &[f32],
    out: &mut [f32],
    first: usize,
    step: usize,
    add: bool,
    scratch: &mut [f32; 2 * SCRATCH],
) {
    let (h, len) = (hist.len() / C, input.len() / C);
    let count = out.len() / C;
    let mut j = 0;
    while j < count {
        let s = first + j * step;
        let rest = count - j;
        if s >= h {
            let o = s - h;
            let k = f.fits(len.saturating_sub(o)).min(rest);
            if k > 0 {
                if let (Some(x), Some(y)) = (input.get(C * o..), out.get_mut(C * j..C * (j + k))) {
                    f.run(x, y, add);
                }
                j += k;
                continue;
            }
        }
        // Through the scratch: the outputs up to the first one whose window lies in the input.
        let mut k = f.fits(SCRATCH).min(rest);
        if s < h {
            k = k.min((h - s).div_ceil(step));
        }
        if k == 0 {
            return;
        }
        fill_window::<_, C>(hist, input, s, s + f.needed(k), scratch);
        if let Some(y) = out.get_mut(C * j..C * (j + k)) {
            f.run(scratch, y, add);
        }
        j += k;
    }
}

/// [`drive`] for strided outputs (the polyphase filters): output `j` (of `out.len() / C` divided
/// by `ostride`, rounded up) is sample `j * ostride` of `out`. The outputs of a run are computed
/// into the scratch and then spread over `out`: a run that reads the input in place uses the
/// whole scratch for its outputs (whole passes of the form where possible), a run through the
/// scratch its last [`SCRATCH_OUT`] samples.
#[inline(always)]
fn drive_strided<F: Form, const C: usize>(
    f: &F,
    hist: &[f32],
    input: &[f32],
    out: &mut [f32],
    first: usize,
    step: usize,
    ostride: usize,
    add: bool,
    scratch: &mut [f32; 2 * SCRATCH],
) {
    let (h, len) = (hist.len() / C, input.len() / C);
    let count = (out.len() / C).div_ceil(ostride);
    let pass = f.pass_outputs();
    let mut j = 0;
    while j < count {
        let s = first + j * step;
        let rest = count - j;
        if s >= h {
            let o = s - h;
            let k = f.fits(len.saturating_sub(o)).min(rest);
            let k = k.min((SCRATCH / pass).max(1) * pass).min(SCRATCH);
            if k > 0 {
                if let (Some(x), Some(y)) = (input.get(C * o..), scratch.get_mut(..C * k)) {
                    run_strided::<_, C>(f, x, y, out, j, ostride, add);
                }
                j += k;
                continue;
            }
        }
        // Through the scratch: the outputs up to the first one whose window lies in the input.
        let room = SCRATCH - SCRATCH_OUT;
        let mut k = f.fits(room).min(rest).min(SCRATCH_OUT);
        if s < h {
            k = k.min((h - s).div_ceil(step));
        }
        if k == 0 {
            return;
        }
        let (region, tail) = scratch.split_at_mut(C * room);
        fill_window::<_, C>(hist, input, s, s + f.needed(k), region);
        if let Some(y) = tail.get_mut(..C * k) {
            run_strided::<_, C>(f, region, y, out, j, ostride, add);
        }
        j += k;
    }
}

/// Four floats on a 16-byte boundary: the unit of the prepared taps, so that a vector of four
/// taps is an aligned load (SSE takes only those as the memory operand of `mulps`).
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
struct Quad([f32; 4]);

/// Whether the level has fused multiply-add for the `C32` and `f32` filters (decision D5: AVX2, AVX-512
/// and NEON; never on the levels without it, and never in the fallback).
#[inline(always)]
fn has_fma<S: SimdToken>(simd: S) -> bool {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        level.as_avx2().is_some()
    }
    #[cfg(target_arch = "aarch64")]
    {
        level.as_neon().is_some()
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = level;
        false
    }
}

/// Writes `f(t)` for every `tstride`-th float `t` of `block`, from the last one back (the taps
/// of a polyphase branch reversed), to `dst`; the branch of `resample` with `L = 1` has the
/// stride 1. `fir` and `fir_decimate` write their taps in a plain loop instead, which the
/// compiler vectorizes (through the closure it does not).
#[inline(always)]
fn prepare_strided<D>(dst: &mut [D], block: &[f32], tstride: usize, f: impl Fn(f32) -> D) {
    if tstride == 1 {
        for (d, &t) in dst.iter_mut().zip(block.iter().rev()) {
            *d = f(t);
        }
    } else {
        for (d, &t) in dst.iter_mut().zip(block.iter().rev().step_by(tstride)) {
            *d = f(t);
        }
    }
}

/// One tap block of a `C32` or `f32` contiguous filter with consecutive outputs: the transposed form
/// (tap unit `T`, `B` vectors `D` samples apart, `I` chains) with the taps of `block`, every
/// `tstride`-th float of it (see [`contiguous`]).
#[inline(always)]
fn transposed_block<
    S: SimdToken,
    V: Lanes<S>,
    T: TapUnit,
    const L: usize,
    const C: usize,
    const B: usize,
    const D: usize,
    const I: usize,
    const FMA: bool,
    const STRIDED: bool,
>(
    simd: S,
    block: &[f32],
    tstride: usize,
    scratch: &mut [f32; 2 * SCRATCH],
    hist: &[f32],
    input: &[f32],
    out: &mut [f32],
    first: usize,
    ostride: usize,
    add: bool,
) {
    let mut storage = [T::of(0.0); BLOCK_TAPS];
    let prepared = &mut storage[..];
    let cnt = block.len().div_ceil(tstride);
    if STRIDED {
        prepare_strided(prepared, block, tstride, T::of);
    } else {
        for (d, &t) in prepared.iter_mut().zip(block.iter().rev()) {
            *d = T::of(t);
        }
    }
    let form = TransForm::<S, V, T, L, C, B, D, I, FMA> {
        simd,
        taps: prepared.get(..cnt).unwrap_or(&[]),
        _lanes: core::marker::PhantomData,
    };
    if STRIDED {
        drive_strided::<_, C>(&form, hist, input, out, first, 1, ostride, add, scratch);
    } else {
        drive::<_, C>(&form, hist, input, out, first, 1, add, scratch);
    }
}

/// One tap block of a `C32` or `f32` contiguous filter with any step: the chunk form (chunk unit `T`,
/// `B` outputs, groups of `U` chunks, `I` chains) with the taps of `block`, every `tstride`-th
/// float of it (see [`contiguous`]).
#[inline(always)]
fn chunked_block<
    S: SimdToken,
    V: Lanes<S>,
    T: Unit,
    const L: usize,
    const C: usize,
    const B: usize,
    const U: usize,
    const I: usize,
    const FMA: bool,
    const STRIDED: bool,
>(
    simd: S,
    block: &[f32],
    tstride: usize,
    scratch: &mut [f32; 2 * SCRATCH],
    hist: &[f32],
    input: &[f32],
    out: &mut [f32],
    first: usize,
    step: usize,
    ostride: usize,
    add: bool,
) {
    // The taps written `C` times (real and imaginary lane of a complex sample), zero beyond the
    // window. The chunk width divides `BLOCK_TAPS`, so the chunks end inside the storage.
    let mut storage = [Quad([0.0; 4]); BLOCK_TAPS / 2];
    let floats: &mut [f32] = bytemuck::cast_slice_mut(&mut storage);
    let cnt = block.len().div_ceil(tstride);
    if STRIDED {
        prepare_strided(floats.as_chunks_mut::<C>().0, block, tstride, |t| [t; C]);
    } else {
        for (d, &t) in floats
            .as_chunks_mut::<C>()
            .0
            .iter_mut()
            .zip(block.iter().rev())
        {
            *d = [t; C];
        }
    }
    let g = L / C;
    let nc = cnt.div_ceil(g);
    let rem = cnt % g;
    let mask = (rem > 0).then(|| {
        V::from_slice(simd, IOTA.get(..L).unwrap_or(&[])).simd_lt(V::splat(simd, (C * rem) as f32))
    });
    let chunks: &[T] = bytemuck::cast_slice(&storage);
    let form = ChunkForm::<S, V, T, L, C, B, U, I, FMA> {
        simd,
        taps: chunks.get(..nc).unwrap_or(&[]),
        mask,
        step,
    };
    if STRIDED {
        drive_strided::<_, C>(&form, hist, input, out, first, step, ostride, add, scratch);
    } else {
        drive::<_, C>(&form, hist, input, out, first, step, add, scratch);
    }
}

/// A contiguous filter of `C32` (`C = 2`) or `f32` (`C = 1`) samples on vectors `V` of `L` lanes: output `j` is the dot product of the taps
/// with the window at sample `first + j * step` of `hist ++ input`. With `STRIDED` (a branch of a
/// polyphase filter) the taps are every `tstride`-th float of `taps` from the first on and output
/// `j` is sample `j * ostride` of `out` (see [`drive_strided`]); without it (`fir`, `fir_decimate`)
/// both strides are 1, and the filter is compiled without the bookkeeping of the strides.
/// `step = 1` runs the transposed form (tap unit `TT`, `TB` vectors `TD` samples apart, `TI`
/// chains), every other step the chunk form (chunk unit `CT`, `CB` outputs, groups of `CU` chunks,
/// `CI` chains). The taps run in blocks of [`BLOCK_TAPS`].
#[inline(always)]
fn contiguous<
    S: SimdToken,
    V: Lanes<S>,
    TT: TapUnit,
    CT: Unit,
    const L: usize,
    const C: usize,
    const FMA: bool,
    const TB: usize,
    const TD: usize,
    const TI: usize,
    const CB: usize,
    const CU: usize,
    const CI: usize,
    const STRIDED: bool,
>(
    simd: S,
    taps: &[f32],
    tstride: usize,
    hist: &[f32],
    input: &[f32],
    out: &mut [f32],
    first: usize,
    step: usize,
    ostride: usize,
) {
    // The parameters of a level must agree; a mismatch is a build error.
    const {
        assert!(V::LEN == L && size_of::<CT>() == 4 * L && (C == 1 || C == 2));
        assert!(BLOCK_TAPS.is_multiple_of(L / C) && TD.is_multiple_of(L / C));
    }
    let (tstride, ostride) = if STRIDED { (tstride, ostride) } else { (1, 1) };
    let n = taps.len().div_ceil(tstride);
    let mut scratch = [0f32; 2 * SCRATCH];
    for skip in (0..n).step_by(BLOCK_TAPS) {
        // Window sample `skip + m` pairs with `h[n - 1 - skip - m]`: the block is the taps
        // `n - skip - cnt .. n - skip`, reversed (the floats from the first to the last of
        // them).
        let end = n - skip;
        let from = tstride * end.saturating_sub(BLOCK_TAPS);
        let block = taps.get(from..tstride * (end - 1) + 1).unwrap_or(&[]);
        let (start, add) = (first + skip, skip > 0);
        if step == 1 {
            transposed_block::<S, V, TT, L, C, TB, TD, TI, FMA, STRIDED>(
                simd,
                block,
                tstride,
                &mut scratch,
                hist,
                input,
                out,
                start,
                ostride,
                add,
            );
        } else {
            chunked_block::<S, V, CT, L, C, CB, CU, CI, FMA, STRIDED>(
                simd,
                block,
                tstride,
                &mut scratch,
                hist,
                input,
                out,
                start,
                step,
                ostride,
                add,
            );
        }
    }
}

/// The `C32` contiguous filter (see [`contiguous`]) with the parameters of the level: `fir`
/// (`step = 1`), `fir_decimate` and, `STRIDED`, the branches of `resample` and
/// `fir_interpolate`.
#[inline(always)]
fn contiguous_c32<S: SimdToken, const STRIDED: bool>(
    simd: S,
    taps: &[f32],
    tstride: usize,
    hist: &[f32],
    input: &[f32],
    out: &mut [f32],
    first: usize,
    step: usize,
    ostride: usize,
) {
    let fma = has_fma(simd);
    let x86 = cfg!(target_arch = "x86_64");
    match S::f32s::LEN {
        16 => contiguous::<S, f32x16<S>, f32, [f32; 16], 16, 2, true, 14, 8, 1, 7, 8, 1, STRIDED>(
            simd, taps, tstride, hist, input, out, first, step, ostride,
        ),
        8 => contiguous::<S, f32x8<S>, f32, [f32; 8], 8, 2, true, 7, 8, 1, 6, 8, 1, STRIDED>(
            simd, taps, tstride, hist, input, out, first, step, ostride,
        ),
        _ if fma => {
            contiguous::<S, f32x4<S>, f32, [f32; 4], 4, 2, true, 7, 2, 1, 7, 8, 1, STRIDED>(
                simd, taps, tstride, hist, input, out, first, step, ostride,
            );
        }
        _ if x86 => contiguous::<S, f32x4<S>, Quad, Quad, 4, 2, false, 4, 8, 2, 7, 8, 1, STRIDED>(
            simd, taps, tstride, hist, input, out, first, step, ostride,
        ),
        _ => contiguous::<S, f32x4<S>, f32, [f32; 4], 4, 2, false, 8, 2, 1, 8, 8, 1, STRIDED>(
            simd, taps, tstride, hist, input, out, first, step, ostride,
        ),
    }
}

/// The `f32` contiguous filter of real samples (see [`contiguous`]) with the parameters of the
/// level: the branches of `resample_real` and `fir_interpolate_real` (always strided). A vector
/// holds twice as many real samples as complex ones, so the transposed form steps its vectors
/// `L` samples apart, and the chunk form computes a multiple of four outputs per pass (four
/// outputs share the last steps of their sums). AVX-512 runs the chunk form on `f32x8` with the
/// parameters of AVX2: on `f32x16` the halving of every total down to four lanes cost more than
/// the wider products saved (`resample_real` 3 / 2 with 64 taps per branch: 14.9 / 13.2 against
/// 8.8 / 8.2 cycles per sample on `znver4` / `sapphirerapids`). The `C32` chunk form keeps
/// `f32x16` there (`f32x8` was slower: `resample` 14.5 / 13.4 against 11.7 / 10.2).
#[inline(always)]
fn contiguous_real<S: SimdToken>(
    simd: S,
    taps: &[f32],
    tstride: usize,
    hist: &[f32],
    input: &[f32],
    out: &mut [f32],
    first: usize,
    step: usize,
    ostride: usize,
) {
    let fma = has_fma(simd);
    let x86 = cfg!(target_arch = "x86_64");
    match S::f32s::LEN {
        16 if step == 1 => {
            contiguous::<S, f32x16<S>, f32, [f32; 16], 16, 1, true, 12, 16, 1, 8, 8, 1, true>(
                simd, taps, tstride, hist, input, out, first, step, ostride,
            );
        }
        8 | 16 => contiguous::<S, f32x8<S>, f32, [f32; 8], 8, 1, true, 7, 8, 1, 8, 8, 1, true>(
            simd, taps, tstride, hist, input, out, first, step, ostride,
        ),
        _ if fma => contiguous::<S, f32x4<S>, f32, [f32; 4], 4, 1, true, 7, 4, 1, 8, 8, 1, true>(
            simd, taps, tstride, hist, input, out, first, step, ostride,
        ),
        _ if x86 => contiguous::<S, f32x4<S>, Quad, Quad, 4, 1, false, 4, 8, 2, 8, 8, 1, true>(
            simd, taps, tstride, hist, input, out, first, step, ostride,
        ),
        _ => contiguous::<S, f32x4<S>, f32, [f32; 4], 4, 1, false, 8, 4, 1, 8, 8, 1, true>(
            simd, taps, tstride, hist, input, out, first, step, ostride,
        ),
    }
}

// ---------------------------------------------------------------------------------------------
// CQ15

/// Storage for the prepared taps of `CQ15` and `Q15` (the taps reversed, `C` times each).
type TapStorageQ15 = [i16; 2 * CAPACITY];

struct Q15Engine<'a, S: SimdToken, const C: usize> {
    simd: S,
    taps: &'a [i16],
    win: Win<'a, i16, C>,
    out: &'a mut [i16],
    td: &'a mut TapStorageQ15,
}

/// Four `i64` accumulators of the exact `i16 x i16` products.
struct Q15Acc<S: SimdToken> {
    a0: S::i64s,
    a1: S::i64s,
    a2: S::i64s,
    a3: S::i64s,
}

impl<S: SimdToken> Q15Acc<S> {
    /// Adds the products of `d` and `t` (both `i16` vectors) exactly.
    #[inline(always)]
    fn mac(&mut self, d: S::i16s, t: S::i16s) {
        let (dl, dh) = d.widen();
        let (tl, th) = t.widen();
        let (p0, p1) = (dl * tl).widen();
        let (p2, p3) = (dh * th).widen();
        self.a0 += p0;
        self.a1 += p1;
        self.a2 += p2;
        self.a3 += p3;
    }
}

/// The taps of a contiguous filter in `td` as [`Q15Engine::prepare`] stores them (sample `m` of
/// the window pairs with `h[n - 1 - m]`, stored `C` times), for [`split_in_place`].
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn prepare_taps<const C: usize>(taps: &[i16], td: &mut [i16]) {
    for (pair, &h) in td.chunks_exact_mut(C).zip(taps.iter().rev()) {
        pair.fill(h);
    }
}

/// The taps of all branches of a polyphase filter `h[phase + up * i]` in `td` as
/// [`Q15Engine::prepare_branches`] stores them (each branch at its [`Branches::offset`], padded
/// with the zeros already in `td` to whole chunks), for [`split_in_place`].
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn prepare_branch_taps(taps: &[i16], geom: Branches, td: &mut [i16]) {
    let (up, c) = (geom.up, geom.c);
    for phase in 0..up.min(geom.n) {
        let t = geom.len(phase);
        let off = geom.offset(phase);
        for m in 0..t {
            let h = taps.get(phase + up * (t - 1 - m)).copied().unwrap_or(0);
            if let Some(lanes) = td.get_mut(off + c * m..off + c * m + c) {
                lanes.fill(h);
            }
        }
    }
}

impl<S: SimdToken, const C: usize> Q15Engine<'_, S, C> {
    /// Prepares the taps of the contiguous filter: sample `m` of the window pairs with
    /// `h[n - 1 - m]`.
    #[inline(always)]
    fn prepare(&mut self) {
        for (pair, &h) in self.td.chunks_exact_mut(C).zip(self.taps.iter().rev()) {
            pair.fill(h);
        }
    }

    /// Prepares the taps of all branches of a polyphase filter `h[phase + up * i]`.
    #[inline(always)]
    fn prepare_branches(&mut self, up: usize) {
        let n = self.taps.len();
        let geom = Branches {
            n,
            up,
            g: S::i16s::LEN / C,
            c: C,
        };
        for phase in 0..up.min(n) {
            let t = geom.len(phase);
            let off = geom.offset(phase);
            for m in 0..t {
                let h = self
                    .taps
                    .get(phase + up * (t - 1 - m))
                    .copied()
                    .unwrap_or(0);
                if let Some(lanes) = self.td.get_mut(off + C * m..off + C * m + C) {
                    lanes.fill(h);
                }
            }
        }
    }

    /// Dot product of the prepared taps at `tap_offset` (elements) with the `n_taps` samples
    /// that begin at sample `start` of the virtual stream; the result is the output with index
    /// `j`.
    #[inline(always)]
    fn dot(&mut self, j: usize, start: usize, n_taps: usize, tap_offset: usize) {
        let simd = self.simd;
        let l = S::i16s::LEN;
        let g = l / C;
        let (full, rem) = (n_taps / g, n_taps % g);
        let zero = S::i64s::splat(simd, 0);
        let mut acc = Q15Acc::<S> {
            a0: zero,
            a1: zero,
            a2: zero,
            a3: zero,
        };
        let td = self.td.get(tap_offset..).unwrap_or(&[]);
        let mut c = 0;
        if start >= self.win.h {
            let o = C * (start - self.win.h);
            if let (Some(win), Some(taps)) =
                (self.win.input.get(o..o + l * full), td.get(..l * full))
            {
                for (d, t) in win.chunks_exact(l).zip(taps.chunks_exact(l)) {
                    acc.mac(S::i16s::from_slice(simd, d), S::i16s::from_slice(simd, t));
                }
                c = full;
            }
        }
        while c < full || (c == full && rem > 0) {
            let take = if c < full { g } else { rem };
            let pos = start + c * g;
            let d = if take == g
                && let Some(d) = self.win.direct(pos, l)
            {
                S::i16s::from_slice(simd, d)
            } else {
                let mut tmp = [0i16; 64];
                fill_window::<_, C>(self.win.hist, self.win.input, pos, pos + take, &mut tmp);
                S::i16s::from_slice(simd, tmp.get(..l).unwrap_or(&[0; 64]))
            };
            let t = td
                .get(c * l..c * l + l)
                .map_or_else(|| S::i16s::splat(simd, 0), |t| S::i16s::from_slice(simd, t));
            acc.mac(d, t);
            c += 1;
        }
        let sum = (acc.a0 + acc.a1) + (acc.a2 + acc.a3);
        if C == 1 {
            let total: i64 = sum.as_slice().iter().sum();
            if let Some(o) = self.out.get_mut(j) {
                *o = Acc64(total).finish().0;
            }
            return;
        }
        let (mut re, mut im) = (0i64, 0i64);
        for pair in sum.as_slice().chunks_exact(2) {
            if let [r, i] = *pair {
                re += r;
                im += i;
            }
        }
        if let Some([o_re, o_im]) = self.out.get_mut(2 * j..2 * j + 2) {
            *o_re = Acc64(re).finish().0;
            *o_im = Acc64(im).finish().0;
        }
    }
}

/// Outputs `j = 0..count` of a contiguous `CQ15` filter; the window of output `j` begins at
/// sample `first + j * step` (the history holds `n - 1` samples, so that is its oldest sample).
#[inline(always)]
fn drive_contiguous<S: SimdToken, const C: usize>(
    e: &mut Q15Engine<'_, S, C>,
    n_taps: usize,
    count: usize,
    first: usize,
    step: usize,
) {
    e.prepare();
    for j in 0..count {
        e.dot(j, first + j * step, n_taps, 0);
    }
}

/// The outputs of a polyphase `CQ15` filter for the bookkeeping of [`polyphase_plan`]: output `k`
/// (of `produced`) has the phase and the window end of [`polyphase_outputs`] from the phase state
/// `q`.
#[inline(always)]
fn drive_polyphase<S: SimdToken, const C: usize>(
    e: &mut Q15Engine<'_, S, C>,
    geom: Branches,
    hist_len: usize,
    down: usize,
    q: i64,
    produced: usize,
) -> Result<()> {
    e.prepare_branches(geom.up);
    let outputs =
        polyphase_outputs(q, geom.up, down).ok_or_else(|| internal("resampler window"))?;
    for (k, (c, phase)) in outputs.take(produced).enumerate() {
        let count = geom.len(phase);
        // The newest sample of the window is virtual index `hist_len + c - 1`.
        let start = (hist_len + c)
            .checked_sub(count)
            .ok_or_else(|| internal("resampler window"))?;
        e.dot(k, start, count, geom.offset(phase));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// CQ15 and Q15 progressions: split taps and `pmaddwd` (x86-64, SIMD128), `smull` and `sadalp`
// (NEON)

/// Storage for the taps of the progressions of the module `prog`: on x86-64 and SIMD128 per
/// chunk of `W` lanes of the prepared taps (`W` the `i16` lanes of the level's vector) the `W`
/// high parts and then the `W` low parts ([`split_in_place`]), on NEON the prepared taps in the
/// order of the paired samples ([`pair_in_place`]). The upper half holds the prepared taps
/// before the expansion.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
type SplitStorage = [i16; 4 * CAPACITY];

/// The level of `simd` for the progressions of the module `prog`: every x86-64 level from SSE2
/// on, NEON and SIMD128 (decisions D340 and D3007).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn split_level<S: SimdToken>(simd: S) -> Option<prog::Split> {
    let level = simd.level();
    if let Some(t) = level.as_avx512() {
        Some(prog::Split::Avx512(t))
    } else if let Some(t) = level.as_avx2() {
        Some(prog::Split::Avx2(t))
    } else if let Some(t) = level.as_sse4_2() {
        Some(prog::Split::Sse4_2(t))
    } else {
        level.as_sse2().map(prog::Split::Sse2)
    }
}

/// The level of `simd` for the progressions of the module `prog` (decision D3007).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn split_level<S: SimdToken>(simd: S) -> Option<prog::Split> {
    simd.level().as_neon().map(prog::Split::Neon)
}

/// The level of `simd` for the progressions of the module `prog` (decision D3007).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn split_level<S: SimdToken>(simd: S) -> Option<prog::Split> {
    simd.level().as_wasm_simd128().map(prog::Split::Wasm)
}

/// Expands the first `chunks` chunks of prepared taps (`W` lanes each, `C` lanes per sample) in
/// the upper half of `s` into split taps at the start of `s`. For `C = 2` the lanes of each
/// group of 8 are put in the order of the samples after the pairing of the module `prog`
/// (`h0 h1 h0 h1 h2 h3 h2 h3`, the effect of [`pair_in_place`]). A tap `t` becomes `256 hi + lo` with `lo` the low byte as a
/// signed value (`-128..=127`) and `hi = (t - lo) / 256` (`-128..=128`). Chunk `k` is read
/// before it is written, and its `2 W` lanes end at or before the chunks not read yet
/// (`2 W (k + 1) <= 2 CAPACITY + W (k + 1)` for `k < 2 CAPACITY / W`), so the expansion runs in
/// place.
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn split_in_place<const C: usize, const W: usize>(s: &mut SplitStorage, chunks: usize) {
    const HALF: usize = 2 * CAPACITY;
    for k in 0..chunks.min(HALF / W) {
        let mut v = [0i16; W];
        if let Some(src) = s.get(HALF + W * k..HALF + W * k + W) {
            v.copy_from_slice(src);
        }
        if C == 2 {
            for g in v.chunks_exact_mut(8) {
                if let [a0, a1, b0, b1, c0, c1, d0, d1] = *g {
                    g.copy_from_slice(&[a0, b0, a1, b1, c0, d0, c1, d1]);
                }
            }
        }
        if let Some(dst) = s.get_mut(2 * W * k..2 * W * k + 2 * W) {
            let (hi, lo) = dst.split_at_mut(W);
            for ((h, l), &t) in hi.iter_mut().zip(lo.iter_mut()).zip(&v) {
                let low = i16::from(t as i8);
                *l = low;
                *h = ((i32::from(t) - i32::from(low)) >> 8) as i16;
            }
        }
    }
}

/// Moves the first `chunks` chunks of prepared taps (8 lanes each, `C` lanes per sample) from
/// the upper half of `s` to its start; for `C = 2` each chunk is put in the order of the paired
/// samples (`h0 h1 h0 h1 h2 h3 h2 h3`). Chunk `k` goes to lanes `8 k .. 8 k + 8`, at or before
/// its source, so the move runs in place.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn pair_in_place<const C: usize>(s: &mut SplitStorage, chunks: usize) {
    const HALF: usize = 2 * CAPACITY;
    for k in 0..chunks.min(HALF / 8) {
        let mut v = [0i16; 8];
        if let Some(src) = s.get(HALF + 8 * k..HALF + 8 * k + 8) {
            v.copy_from_slice(src);
        }
        if C == 2
            && let [a0, a1, b0, b1, c0, c1, d0, d1] = v
        {
            v = [a0, b0, a1, b1, c0, d0, c1, d1];
        }
        if let Some(dst) = s.get_mut(8 * k..8 * k + 8) {
            dst.copy_from_slice(&v);
        }
    }
}

/// The expansion of the prepared taps for the progressions of `level`: [`split_in_place`] at
/// the width of the level, or [`pair_in_place`] on NEON.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn expand<const C: usize>(level: prog::Split, s: &mut SplitStorage, chunks: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        let _ = level;
        pair_in_place::<C>(s, chunks);
    }
    #[cfg(not(target_arch = "aarch64"))]
    match level.lanes() {
        32 => split_in_place::<C, 32>(s, chunks),
        16 => split_in_place::<C, 16>(s, chunks),
        _ => split_in_place::<C, 8>(s, chunks),
    }
}

/// `fir` `CQ15` (`step = 1`) and `fir_decimate` `CQ15` in progressions: outputs `j = 0..count`
/// with the window of output `j` at sample `first + j step` (decisions D340 and D3007).
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn contiguous_split(
    level: prog::Split,
    taps: &[i16],
    hist: &[i16],
    input: &[i16],
    out: &mut [i16],
    first: usize,
    step: usize,
    count: usize,
) {
    let mut s: SplitStorage = [0; 4 * CAPACITY];
    if let Some(td) = s.get_mut(2 * CAPACITY..) {
        prepare_taps::<2>(taps, td);
    }
    let chunks = (2 * taps.len()).div_ceil(level.lanes());
    expand::<2>(level, &mut s, chunks);
    prog::progression::<2>(
        level,
        prog::Progression {
            hist,
            input,
            split: &s,
            chunks,
            first,
            stride: step,
            n: count,
            out,
            o0: 0,
            ostride: 1,
        },
    );
}

/// `resample` and `fir_interpolate` `CQ15` (`C = 2`), `resample_real` and `fir_interpolate_real`
/// `Q15` (`C = 1`) in progressions (decisions D340 and D3007): the outputs `r, r + L, r + 2 L,
/// ...` of a call share the phase of output `r` and their windows are `M` samples apart, so each
/// branch runs as one progression of the module `prog`. `geom` has chunks of the width of
/// `level`.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(
    clippy::large_stack_frames,
    reason = "the lint counts the temporaries of the inlined progressions before optimization; \
              the one array is the split taps (3072 bytes)"
)]
#[inline(always)]
fn polyphase_split<const C: usize>(
    level: prog::Split,
    taps: &[i16],
    hist: &[i16],
    input: &[i16],
    out: &mut [i16],
    geom: Branches,
    down: usize,
    q: i64,
    produced: usize,
) -> Result<()> {
    let window_err = || internal("resampler window");
    if geom.g * C != level.lanes() {
        return Err(window_err());
    }
    let mut s: SplitStorage = [0; 4 * CAPACITY];
    if let Some(td) = s.get_mut(2 * CAPACITY..) {
        prepare_branch_taps(taps, geom, td);
    }
    let up = geom.up;
    expand::<C>(level, &mut s, geom.offset(up.min(geom.n)) / level.lanes());
    let l = i128::try_from(up).map_err(|_| window_err())?;
    let hist_len = hist.len() / C;
    for r in 0..up.min(produced) {
        // `usize` to `i128` is lossless.
        let (c, phase) = polyphase_output(i128::from(q) + r as i128 * down as i128, l)?;
        let count = geom.len(phase);
        let p = prog::Progression {
            hist,
            input,
            split: s.get(prog::PARTS * geom.offset(phase)..).unwrap_or(&[]),
            chunks: geom.padded(count) / geom.g,
            first: (hist_len + c).checked_sub(count).ok_or_else(window_err)?,
            stride: down,
            n: (produced - r).div_ceil(up),
            out: &mut *out,
            o0: r,
            ostride: up,
        };
        prog::progression::<C>(level, p);
    }
    Ok(())
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod prog {
    //! The `CQ15` and `Q15` filters in progressions of outputs (decision D340 at SSE4.2, D3007
    //! at SSE2, AVX2, AVX-512, NEON and SIMD128).
    //!
    //! x86-64 and SIMD128: `pmaddwd` (`i32x4.dot_i16x8_s`) multiplies 16-bit lanes and adds
    //! adjacent products into 32-bit lanes; a pair of full-range products does not fit
    //! (`2 * 2^30`), so every tap is split once per call into `256 hi + lo`
    //! ([`super::split_in_place`], `|hi|, |lo| <= 128`): a pair of products of a sample with `hi`
    //! or `lo` is at most `2^23` in magnitude, and a window of at most `2 CAPACITY` lanes of taps
    //! adds at most 96 such pairs into each lane of the four 32-bit lanes the sums are folded to,
    //! without overflow (`96 * 2^23 < 2^30`). The output is `256 sum hi + sum lo` in `i64`, the
    //! exact sum, so the bits of the fallback. The vector is the widest of the level: 8 lanes at
    //! SSE2, SSE4.2 and SIMD128, 16 at AVX2, 32 at AVX-512, where `vpdpwssd` (AVX-512 VNNI, part
    //! of the level) multiplies and adds in one instruction, without saturation, the same sums.
    //!
    //! NEON: `smull` and `smull2` form the exact 32-bit products of 8 lanes and `sadalp` adds
    //! adjacent products into 64-bit lanes, so the taps need no split; two accumulators per
    //! output (the low and the high half of the vector).
    //!
    //! `CQ15` pairs two consecutive samples of the same component in every group of 8 lanes
    //! (`re0 re1 im0 im1 re2 re3 im2 im3`: `pshufb` from SSSE3 on, `pshuflw` and `pshufhw` at
    //! SSE2, `i8x16.shuffle` at SIMD128, `tbl` on NEON), so adjacent products belong to the same
    //! component and the 32-bit (NEON: 64-bit) lanes alternate between the real and the
    //! imaginary sums. [`GROUP`] outputs whose windows lie in the input are computed together,
    //! so that each vector of taps is loaded once for them and their sums are independent
    //! chains; the outputs whose windows reach into the history or past the end of the input are
    //! computed one at a time, a window that does not lie in one slice assembled chunk by chunk.
    //! The sums of an output are folded to four 32-bit lanes (one horizontal step per halving of
    //! the vector) or two 64-bit lanes before the `i64` sum.

    #[cfg(target_arch = "aarch64")]
    use core::arch::aarch64::{
        int16x8_t, int64x2_t, uint8x16_t, vaddq_s64, vaddvq_s64, vdupq_n_s64, vget_low_s16,
        vgetq_lane_s64, vmull_high_s16, vmull_s16, vpadalq_s32, vqtbl1q_s8, vreinterpretq_s8_s16,
        vreinterpretq_s16_s8,
    };
    #[cfg(target_arch = "wasm32")]
    use core::arch::wasm32::{i16x8_shuffle, i32x4_add, i32x4_dot_i16x8, u64x2_splat, v128};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128i, __m256i, __m512i, _mm_add_epi32, _mm_madd_epi16, _mm_setr_epi8, _mm_setzero_si128,
        _mm_shuffle_epi8, _mm_shufflehi_epi16, _mm_shufflelo_epi16, _mm256_add_epi32,
        _mm256_broadcastsi128_si256, _mm256_castsi256_si128, _mm256_extracti128_si256,
        _mm256_madd_epi16, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm512_broadcast_i32x4,
        _mm512_castsi512_si256, _mm512_dpwssd_epi32, _mm512_extracti64x4_epi64,
        _mm512_setzero_si512, _mm512_shuffle_epi8,
    };
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    use core::hint::black_box;
    #[cfg(target_arch = "wasm32")]
    use fearless_simd::WasmSimd128;
    use fearless_simd::i16x8;
    #[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
    use fearless_simd::i32x4;
    use fearless_simd::prelude::*;
    #[cfg(target_arch = "x86_64")]
    use fearless_simd::{Avx2, Avx512, Sse2, Sse4_2, i16x16, i16x32};
    #[cfg(target_arch = "aarch64")]
    use fearless_simd::{Neon, u8x16};
    use sdrk_core::num::Acc64;

    use super::{Win, fill_window};

    /// The level of a progression.
    #[derive(Clone, Copy)]
    pub(super) enum Split {
        #[cfg(target_arch = "x86_64")]
        Sse2(Sse2),
        #[cfg(target_arch = "x86_64")]
        Sse4_2(Sse4_2),
        #[cfg(target_arch = "x86_64")]
        Avx2(Avx2),
        #[cfg(target_arch = "x86_64")]
        Avx512(Avx512),
        #[cfg(target_arch = "aarch64")]
        Neon(Neon),
        #[cfg(target_arch = "wasm32")]
        Wasm(WasmSimd128),
    }

    impl Split {
        /// The `i16` lanes of the vector of the level (the width of a chunk of taps).
        #[inline(always)]
        pub(super) fn lanes(self) -> usize {
            match self {
                #[cfg(target_arch = "x86_64")]
                Self::Avx2(_) => 16,
                #[cfg(target_arch = "x86_64")]
                Self::Avx512(_) => 32,
                #[allow(unreachable_patterns, reason = "the only arm on NEON and SIMD128")]
                _ => 8,
            }
        }
    }

    /// The vectors of taps per chunk of samples: 2 (the split taps), 1 on NEON.
    pub(super) const PARTS: usize = if cfg!(target_arch = "aarch64") { 1 } else { 2 };

    /// Outputs computed together.
    const GROUP: usize = 4;

    /// The byte order of `pshufb` (`tbl`) that reorders a vector of interleaved `CQ15` samples
    /// to `re0 re1 im0 im1 re2 re3 im2 im3` (in every 128-bit lane).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    const PAIR: [i8; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];

    /// The order of `pshuflw` and `pshufhw` with the effect of [`PAIR`]: words 0, 2, 1, 3 of
    /// each half.
    #[cfg(target_arch = "x86_64")]
    const PAIR_WORDS: i32 = 0b11_01_10_00;

    /// One progression of outputs: output `j < n` has the window that begins at sample
    /// `first + j stride` of `hist ++ input` and goes to the output with index `o0 + j ostride`;
    /// `split` holds `chunks` chunks of taps ([`Split::parts`] vectors each).
    pub(super) struct Progression<'a> {
        pub(super) hist: &'a [i16],
        pub(super) input: &'a [i16],
        pub(super) split: &'a [i16],
        pub(super) chunks: usize,
        pub(super) first: usize,
        pub(super) stride: usize,
        pub(super) n: usize,
        pub(super) out: &'a mut [i16],
        pub(super) o0: usize,
        pub(super) ostride: usize,
    }

    /// Runs the progression `p` of `C` components per sample at `level`.
    #[inline(always)]
    pub(super) fn progression<const C: usize>(level: Split, p: Progression<'_>) {
        match (level, C) {
            #[cfg(target_arch = "x86_64")]
            (Split::Sse2(t), 1) => sse2_c1(t, p),
            #[cfg(target_arch = "x86_64")]
            (Split::Sse2(t), _) => sse2_c2(t, p),
            #[cfg(target_arch = "x86_64")]
            (Split::Sse4_2(t), 1) => sse4_2_c1(t, p),
            #[cfg(target_arch = "x86_64")]
            (Split::Sse4_2(t), _) => sse4_2_c2(t, p),
            #[cfg(target_arch = "x86_64")]
            (Split::Avx2(t), 1) => avx2_c1(t, p),
            #[cfg(target_arch = "x86_64")]
            (Split::Avx2(t), _) => avx2_c2(t, p),
            #[cfg(target_arch = "x86_64")]
            (Split::Avx512(t), 1) => avx512_c1(t, p),
            #[cfg(target_arch = "x86_64")]
            (Split::Avx512(t), _) => avx512_c2(t, p),
            #[cfg(target_arch = "aarch64")]
            (Split::Neon(t), 1) => neon_c1(t, p),
            #[cfg(target_arch = "aarch64")]
            (Split::Neon(t), _) => neon_c2(t, p),
            #[cfg(target_arch = "wasm32")]
            (Split::Wasm(t), 1) => wasm_c1(t, p),
            #[cfg(target_arch = "wasm32")]
            (Split::Wasm(t), _) => wasm_c2(t, p),
        }
    }

    /// Writes the output with index `k` from the high-part sums `hi` and the low-part sums `lo`
    /// (for `C = 2` the even lanes real, the odd lanes imaginary).
    #[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
    #[inline(always)]
    fn store<const C: usize>(out: &mut [i16], k: usize, hi: &[i32], lo: &[i32]) {
        let (mut even, mut odd) = (0i64, 0i64);
        for (h, l) in hi.chunks_exact(2).zip(lo.chunks_exact(2)) {
            if let (&[h0, h1], &[l0, l1]) = (h, l) {
                even += 256 * i64::from(h0) + i64::from(l0);
                odd += 256 * i64::from(h1) + i64::from(l1);
            }
        }
        if C == 1 {
            if let Some(o) = out.get_mut(k) {
                *o = Acc64(even + odd).finish().0;
            }
        } else if let Some([re, im]) = out.get_mut(2 * k..2 * k + 2) {
            *re = Acc64(even).finish().0;
            *im = Acc64(odd).finish().0;
        }
    }

    /// A progression function `$name` for `$c` components at the level `$token` with vectors
    /// `$v` of `$w` lanes (`$vec` the `fearless_simd` type of the load). `$order` is computed
    /// once and passed to `$pair` (the pairing of `CQ15` samples), `$mac(acc, d, t)` adds the
    /// pair products of `d` and `t` to `acc`, `$fold` folds a vector of 32-bit sums to four
    /// lanes.
    #[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
    macro_rules! progression {
        (
            $name:ident, $c:literal, $token:ident, $w:literal, $v:ty, $vec:ident,
            zero: $zero:expr, order: $order:expr, pair: $pair:expr, mac: $mac:expr,
            fold: $fold:expr $(,)?
        ) => {
            fearless_simd::kernel!(
                #[allow(
                    clippy::large_stack_frames,
                    reason = "the lint counts the vector temporaries of the inlined steps \
                              before optimization; the loops keep them in registers"
                )]
                #[inline(always)]
                fn $name(t: $token, p: Progression<'_>) {
                    let Progression {
                        hist,
                        input,
                        split,
                        chunks,
                        first,
                        stride,
                        n,
                        out,
                        o0,
                        ostride,
                    } = p;
                    let win = Win::<i16, $c>::new(hist, input);
                    let h = win.h;
                    let order = $order;
                    let pair_of = $pair;
                    let mac = $mac;
                    let fold = $fold;
                    let pair = |v: $v| -> $v { if $c == 2 { pair_of(order, v) } else { v } };
                    let load = |s: &[i16; $w]| -> $v { $vec::from_slice(t, s).into() };
                    let (lanes, _) = split
                        .get(..2 * $w * chunks)
                        .unwrap_or(&[])
                        .as_chunks::<$w>();
                    let (taps, _) = lanes.as_chunks::<2>();
                    let zero: $v = $zero;
                    let lanes_of = |v: $v| i32x4::simd_from(t, fold(v));
                    let one = |j: usize, out: &mut [i16]| {
                        let start = first + j * stride;
                        let (mut hi, mut lo) = (zero, zero);
                        for (k, [th, tl]) in taps.iter().enumerate() {
                            let pos = start + k * ($w / $c);
                            let mut tmp = [0i16; $w];
                            match win.direct(pos, $w) {
                                Some(d) => tmp.copy_from_slice(d),
                                None => {
                                    fill_window::<_, $c>(hist, input, pos, pos + $w / $c, &mut tmp)
                                }
                            }
                            let d = pair(load(&tmp));
                            hi = mac(hi, d, load(th));
                            lo = mac(lo, d, load(tl));
                        }
                        store::<$c>(
                            out,
                            o0 + j * ostride,
                            lanes_of(hi).as_slice(),
                            lanes_of(lo).as_slice(),
                        );
                    };
                    let mut j = 0;
                    while j < n && first + j * stride < h {
                        one(j, out);
                        j += 1;
                    }
                    let (span, gap) = ($w * chunks, $c * stride);
                    while j + GROUP <= n {
                        let o = $c * (first + j * stride - h);
                        let Some(w) = input.get(o..o + gap * (GROUP - 1) + span) else {
                            break;
                        };
                        let window = |g: usize| {
                            w.get(gap * g..gap * g + span)
                                .unwrap_or(&[])
                                .as_chunks::<$w>()
                                .0
                        };
                        let mut acc = [[zero; 2]; GROUP];
                        let step = |acc: &mut [$v; 2], d: &[i16; $w], th: $v, tl: $v| {
                            let d = pair(load(d));
                            acc[0] = mac(acc[0], d, th);
                            acc[1] = mac(acc[1], d, tl);
                        };
                        let [a0, a1, a2, a3] = &mut acc;
                        for ((((tk, d0), d1), d2), d3) in taps
                            .iter()
                            .zip(window(0))
                            .zip(window(1))
                            .zip(window(2))
                            .zip(window(3))
                        {
                            let [th, tl] = tk;
                            let (th, tl) = (load(th), load(tl));
                            step(a0, d0, th, tl);
                            step(a1, d1, th, tl);
                            step(a2, d2, th, tl);
                            step(a3, d3, th, tl);
                        }
                        for (g, [hi, lo]) in acc.into_iter().enumerate() {
                            store::<$c>(
                                out,
                                o0 + (j + g) * ostride,
                                lanes_of(hi).as_slice(),
                                lanes_of(lo).as_slice(),
                            );
                        }
                        j += GROUP;
                    }
                    while j < n {
                        one(j, out);
                        j += 1;
                    }
                }
            );
        };
    }

    /// The bytes of [`PAIR`] as a vector.
    #[cfg(target_arch = "x86_64")]
    macro_rules! pair_order {
        () => {{
            let [
                b0,
                b1,
                b2,
                b3,
                b4,
                b5,
                b6,
                b7,
                b8,
                b9,
                b10,
                b11,
                b12,
                b13,
                b14,
                b15,
            ] = PAIR;
            _mm_setr_epi8(
                b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15,
            )
        }};
    }

    #[cfg(target_arch = "x86_64")]
    macro_rules! sse2 {
        ($name:ident, $c:literal) => {
            progression!(
                $name, $c, Sse2, 8, __m128i, i16x8,
                zero: _mm_setzero_si128(),
                order: (),
                pair: |(): (), v: __m128i| {
                    _mm_shufflehi_epi16::<PAIR_WORDS>(_mm_shufflelo_epi16::<PAIR_WORDS>(v))
                },
                mac: |acc: __m128i, d: __m128i, t: __m128i| _mm_add_epi32(acc, _mm_madd_epi16(d, t)),
                fold: |v: __m128i| v,
            );
        };
    }

    #[cfg(target_arch = "x86_64")]
    macro_rules! sse4_2 {
        ($name:ident, $c:literal) => {
            progression!(
                $name, $c, Sse4_2, 8, __m128i, i16x8,
                zero: _mm_setzero_si128(),
                order: black_box(pair_order!()),
                pair: |o: __m128i, v: __m128i| _mm_shuffle_epi8(v, o),
                mac: |acc: __m128i, d: __m128i, t: __m128i| _mm_add_epi32(acc, _mm_madd_epi16(d, t)),
                fold: |v: __m128i| v,
            );
        };
    }

    #[cfg(target_arch = "x86_64")]
    macro_rules! avx2 {
        ($name:ident, $c:literal) => {
            progression!(
                $name, $c, Avx2, 16, __m256i, i16x16,
                zero: _mm256_setzero_si256(),
                order: black_box(_mm256_broadcastsi128_si256(pair_order!())),
                pair: |o: __m256i, v: __m256i| _mm256_shuffle_epi8(v, o),
                mac: |acc: __m256i, d: __m256i, t: __m256i| {
                    _mm256_add_epi32(acc, _mm256_madd_epi16(d, t))
                },
                fold: |v: __m256i| {
                    _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v))
                },
            );
        };
    }

    #[cfg(target_arch = "x86_64")]
    macro_rules! avx512 {
        ($name:ident, $c:literal) => {
            progression!(
                $name, $c, Avx512, 32, __m512i, i16x32,
                zero: _mm512_setzero_si512(),
                order: black_box(_mm512_broadcast_i32x4(pair_order!())),
                pair: |o: __m512i, v: __m512i| _mm512_shuffle_epi8(v, o),
                mac: |acc: __m512i, d: __m512i, t: __m512i| _mm512_dpwssd_epi32(acc, d, t),
                fold: |v: __m512i| {
                    let a = _mm256_add_epi32(
                        _mm512_castsi512_si256(v),
                        _mm512_extracti64x4_epi64::<1>(v),
                    );
                    _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256::<1>(a))
                },
            );
        };
    }

    #[cfg(target_arch = "x86_64")]
    sse2!(sse2_c1, 1);
    #[cfg(target_arch = "x86_64")]
    sse2!(sse2_c2, 2);
    #[cfg(target_arch = "x86_64")]
    sse4_2!(sse4_2_c1, 1);
    #[cfg(target_arch = "x86_64")]
    sse4_2!(sse4_2_c2, 2);
    #[cfg(target_arch = "x86_64")]
    avx2!(avx2_c1, 1);
    #[cfg(target_arch = "x86_64")]
    avx2!(avx2_c2, 2);
    #[cfg(target_arch = "x86_64")]
    avx512!(avx512_c1, 1);
    #[cfg(target_arch = "x86_64")]
    avx512!(avx512_c2, 2);

    #[cfg(target_arch = "wasm32")]
    macro_rules! wasm {
        ($name:ident, $c:literal) => {
            progression!(
                $name, $c, WasmSimd128, 8, v128, i16x8,
                zero: u64x2_splat(0),
                order: (),
                pair: |(): (), v: v128| i16x8_shuffle::<0, 2, 1, 3, 4, 6, 5, 7>(v, v),
                mac: |acc: v128, d: v128, t: v128| i32x4_add(acc, i32x4_dot_i16x8(d, t)),
                fold: |v: v128| v,
            );
        };
    }
    #[cfg(target_arch = "wasm32")]
    wasm!(wasm_c1, 1);
    #[cfg(target_arch = "wasm32")]
    wasm!(wasm_c2, 2);

    /// The NEON progression `$name` for `$c` components: the taps are not split (one vector per
    /// chunk), `smull` and `smull2` form the exact products and `sadalp` adds adjacent ones into
    /// two 64-bit accumulators per output.
    #[cfg(target_arch = "aarch64")]
    macro_rules! neon {
        ($name:ident, $c:literal) => {
            fearless_simd::kernel!(
                #[allow(
                    clippy::large_stack_frames,
                    reason = "the lint counts the vector temporaries of the inlined steps \
                              before optimization; the loops keep them in registers"
                )]
                #[inline(always)]
                fn $name(t: Neon, p: Progression<'_>) {
                    let Progression {
                        hist,
                        input,
                        split,
                        chunks,
                        first,
                        stride,
                        n,
                        out,
                        o0,
                        ostride,
                    } = p;
                    let win = Win::<i16, $c>::new(hist, input);
                    let h = win.h;
                    let order: uint8x16_t =
                        black_box(u8x16::from_slice(t, &PAIR.map(|b| b as u8)).into());
                    let pair = |v: int16x8_t| -> int16x8_t {
                        if $c == 2 {
                            vreinterpretq_s16_s8(vqtbl1q_s8(vreinterpretq_s8_s16(v), order))
                        } else {
                            v
                        }
                    };
                    let load = |s: &[i16; 8]| -> int16x8_t { i16x8::from_slice(t, s).into() };
                    let (taps, _) = split.get(..8 * chunks).unwrap_or(&[]).as_chunks::<8>();
                    let zero: int64x2_t = vdupq_n_s64(0);
                    let mac = |acc: &mut [int64x2_t; 2], d: int16x8_t, tv: int16x8_t| {
                        acc[0] = vpadalq_s32(acc[0], vmull_s16(vget_low_s16(d), vget_low_s16(tv)));
                        acc[1] = vpadalq_s32(acc[1], vmull_high_s16(d, tv));
                    };
                    let put = |out: &mut [i16], k: usize, acc: [int64x2_t; 2]| {
                        let s = vaddq_s64(acc[0], acc[1]);
                        if $c == 1 {
                            if let Some(o) = out.get_mut(k) {
                                *o = Acc64(vaddvq_s64(s)).finish().0;
                            }
                        } else if let Some([re, im]) = out.get_mut(2 * k..2 * k + 2) {
                            *re = Acc64(vgetq_lane_s64::<0>(s)).finish().0;
                            *im = Acc64(vgetq_lane_s64::<1>(s)).finish().0;
                        }
                    };
                    let one = |j: usize, out: &mut [i16]| {
                        let start = first + j * stride;
                        let mut acc = [zero; 2];
                        for (k, tk) in taps.iter().enumerate() {
                            let pos = start + k * (8 / $c);
                            let mut tmp = [0i16; 8];
                            match win.direct(pos, 8) {
                                Some(d) => tmp.copy_from_slice(d),
                                None => {
                                    fill_window::<_, $c>(hist, input, pos, pos + 8 / $c, &mut tmp)
                                }
                            }
                            mac(&mut acc, pair(load(&tmp)), load(tk));
                        }
                        put(out, o0 + j * ostride, acc);
                    };
                    let mut j = 0;
                    while j < n && first + j * stride < h {
                        one(j, out);
                        j += 1;
                    }
                    let (span, gap) = (8 * chunks, $c * stride);
                    while j + GROUP <= n {
                        let o = $c * (first + j * stride - h);
                        let Some(w) = input.get(o..o + gap * (GROUP - 1) + span) else {
                            break;
                        };
                        let window = |g: usize| {
                            w.get(gap * g..gap * g + span)
                                .unwrap_or(&[])
                                .as_chunks::<8>()
                                .0
                        };
                        let mut acc = [[zero; 2]; GROUP];
                        let [a0, a1, a2, a3] = &mut acc;
                        for ((((tk, d0), d1), d2), d3) in taps
                            .iter()
                            .zip(window(0))
                            .zip(window(1))
                            .zip(window(2))
                            .zip(window(3))
                        {
                            let tv = load(tk);
                            mac(a0, pair(load(d0)), tv);
                            mac(a1, pair(load(d1)), tv);
                            mac(a2, pair(load(d2)), tv);
                            mac(a3, pair(load(d3)), tv);
                        }
                        for (g, a) in acc.into_iter().enumerate() {
                            put(out, o0 + (j + g) * ostride, a);
                        }
                        j += GROUP;
                    }
                    while j < n {
                        one(j, out);
                        j += 1;
                    }
                }
            );
        };
    }
    #[cfg(target_arch = "aarch64")]
    neon!(neon_c1, 1);
    #[cfg(target_arch = "aarch64")]
    neon!(neon_c2, 2);
}

// ---------------------------------------------------------------------------------------------
// Bookkeeping shared by both sample types

/// `hist` becomes the last `hist.len()` samples of `hist ++ input`.
pub(crate) fn update_history<T: Copy>(hist: &mut [T], input: &[T]) -> Result<()> {
    let keep = hist.len();
    if input.len() >= keep {
        let tail = input
            .get(input.len() - keep..)
            .ok_or_else(|| internal("history update"))?;
        hist.copy_from_slice(tail);
    } else {
        hist.rotate_left(input.len());
        let dst = hist
            .get_mut(keep - input.len()..)
            .ok_or_else(|| internal("history update"))?;
        dst.copy_from_slice(input);
    }
    Ok(())
}

/// Moves the `consumed` samples of `input` into the history and reports the progress of a
/// call; `what` names the window in the error of a violated invariant.
pub(crate) fn commit<T: Copy>(
    hist: &mut [T],
    input: &[T],
    consumed: usize,
    produced: usize,
    what: &'static str,
) -> Result<Progress> {
    let used = input.get(..consumed).ok_or_else(|| internal(what))?;
    update_history(hist, used)?;
    Ok(Progress { consumed, produced })
}

/// The decimation bookkeeping of the fallback in closed form: outputs at the input indices
/// `countdown, countdown + factor, ...`. Returns `(consumed, produced, new_countdown)` for an
/// input of `len` samples and room for `room` outputs.
fn decimation_plan(
    countdown: usize,
    factor: usize,
    len: usize,
    room: usize,
) -> (usize, usize, usize) {
    let possible = if countdown < len {
        (len - 1 - countdown) / factor + 1
    } else {
        0
    };
    if possible <= room {
        let new_countdown = if possible == 0 {
            countdown - len
        } else {
            let last = countdown + (possible - 1) * factor;
            factor - 1 - (len - 1 - last)
        };
        (len, possible, new_countdown)
    } else {
        (countdown + room * factor, room, 0)
    }
}

/// The polyphase loop in closed form. Output `k` of the call is ready once
/// `c_k = floor((q + k M) / L) + 1` input samples are consumed (`q >= -L`, so `c_0 >= 0`), and
/// its phase is `(q + k M) mod L`; the loop stops at the first output that needs more input
/// than there is, or when the output is full, after consuming up to that output's `c_k`.
/// Returns `(consumed, produced, new_q)`, or `None` if the counts do not fit.
fn polyphase_plan(
    q: i64,
    up: usize,
    down: usize,
    in_len: usize,
    out_len: usize,
) -> Option<(usize, usize, i64)> {
    let (q, l, m) = (
        i128::from(q),
        i128::try_from(up).ok()?,
        i128::try_from(down).ok()?,
    );
    let (n_in, n_out) = (i128::try_from(in_len).ok()?, i128::try_from(out_len).ok()?);
    // Output `k` is ready while `q + k M < n_in L`.
    let ready = (n_in * l - q).max(0);
    let produced = ((ready + m - 1) / m).min(n_out);
    let consumed = ((q + produced * m).div_euclid(l) + 1).min(n_in);
    let new_q = q + produced * m - consumed * l;
    Some((
        usize::try_from(consumed).ok()?,
        usize::try_from(produced).ok()?,
        i64::try_from(new_q).ok()?,
    ))
}

/// The output of a polyphase call at `at = q + k M` (output `k` from the phase state `q`,
/// `q >= -L`), as in [`polyphase_plan`]: the number `floor(at / L) + 1` of input samples after
/// which it is ready (its window ends at the newest of them) and its phase `at mod L`.
///
/// # Errors
/// `Internal` if the numbers do not fit (`L > 0` is the caller's).
#[inline(always)]
fn polyphase_output(at: i128, l: i128) -> Result<(usize, usize)> {
    let window_err = || internal("resampler window");
    let c = usize::try_from(at.div_euclid(l) + 1).map_err(|_| window_err())?;
    let phase = usize::try_from(at.rem_euclid(l)).map_err(|_| window_err())?;
    Ok((c, phase))
}

/// The outputs of a polyphase call from the phase state `q` on: [`polyphase_output`] of
/// output 0, then one step of `M` phases per output.
fn polyphase_outputs(
    q: i64,
    up: usize,
    down: usize,
) -> Option<impl Iterator<Item = (usize, usize)>> {
    let l = i128::try_from(up).ok().filter(|&l| l > 0)?;
    let (c, phase) = polyphase_output(i128::from(q), l).ok()?;
    let (whole, part) = (down / up, down % up);
    Some(core::iter::successors(
        Some((c, phase)),
        move |&(c, phase)| {
            let (c, phase) = (c + whole, phase + part);
            Some(if phase >= up {
                (c + 1, phase - up)
            } else {
                (c, phase)
            })
        },
    ))
}

// ---------------------------------------------------------------------------------------------
// Entry points

#[inline(always)]
fn fir_cq15<S: SimdToken>(simd: S, taps: &[i16], hist: &[i16], input: &[i16], out: &mut [i16]) {
    let len = input.len() / 2;
    #[cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    if let Some(level) = split_level(simd) {
        contiguous_split(level, taps, hist, input, out, 0, 1, len);
        return;
    }
    let mut storage: TapStorageQ15 = [0i16; 2 * CAPACITY];
    let n = taps.len();
    let mut e = Q15Engine {
        simd,
        taps,
        win: Win::<_, 2>::new(hist, input),
        out,
        td: &mut storage,
    };
    drive_contiguous(&mut e, n, len, 0, 1);
}

#[inline(always)]
fn decimate_c32<S: SimdToken>(
    simd: S,
    taps: &[f32],
    hist: &[f32],
    input: &[f32],
    out: &mut [f32],
    first: usize,
    factor: usize,
    count: usize,
) {
    if let Some(out) = out.get_mut(..2 * count) {
        contiguous_c32::<_, false>(simd, taps, 1, hist, input, out, first, factor, 1);
    }
}

#[inline(always)]
fn decimate_cq15<S: SimdToken>(
    simd: S,
    taps: &[i16],
    hist: &[i16],
    input: &[i16],
    out: &mut [i16],
    first: usize,
    factor: usize,
    count: usize,
) {
    #[cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    if let Some(level) = split_level(simd) {
        contiguous_split(level, taps, hist, input, out, first, factor, count);
        return;
    }
    let mut storage: TapStorageQ15 = [0i16; 2 * CAPACITY];
    let n = taps.len();
    let mut e = Q15Engine {
        simd,
        taps,
        win: Win::<_, 2>::new(hist, input),
        out,
        td: &mut storage,
    };
    drive_contiguous(&mut e, n, count, first, factor);
}

/// `resample` and `fir_interpolate` for `C32` (`C = 2`), `resample_real` and
/// `fir_interpolate_real` for `f32` (`C = 1`), with the bookkeeping of [`polyphase_plan`]. The
/// outputs `r, r + L, r + 2 L, ...` of a call share the phase of output `r`, and their windows are
/// `M` samples apart, so each of these progressions is one contiguous filter with the taps of its
/// branch, step `M` and output stride `L`: the transposed form for `fir_interpolate` (`M = 1`), the
/// chunk form for every other `M`. Each output is therefore computed as `fir` or `fir_decimate`
/// computes it, whichever way the input was split into blocks.
#[inline(always)]
fn polyphase_f32<S: SimdToken, const C: usize>(
    simd: S,
    taps: &[f32],
    hist: &[f32],
    input: &[f32],
    out: &mut [f32],
    up: usize,
    down: usize,
    q: &mut i64,
) -> Result<(usize, usize)> {
    let window_err = || internal("resampler window");
    if up == 0 {
        return Err(window_err());
    }
    let n = taps.len();
    let (consumed, produced, new_q) =
        polyphase_plan(*q, up, down, input.len() / C, out.len() / C).ok_or_else(window_err)?;
    // `usize` to `i128` is lossless.
    let (q0, l, m) = (i128::from(*q), up as i128, down as i128);
    let hist_len = hist.len() / C;
    let out = out.get_mut(..C * produced).ok_or_else(window_err)?;
    for r in 0..up.min(produced) {
        // Output `r` is ready after `c` input samples; its window ends at the newest of them.
        let (c, phase) = polyphase_output(q0 + r as i128 * m, l)?;
        let y = out.get_mut(C * r..).ok_or_else(window_err)?;
        let Some(branch) = taps.get(phase..).filter(|b| !b.is_empty()) else {
            // A phase beyond the taps (`L > N`) has no taps: its outputs are zero.
            strided::<C>(y, 0, up).for_each(|d| d.fill(0.0));
            continue;
        };
        let t = (n - phase).div_ceil(up);
        let first = (hist_len + c).checked_sub(t).ok_or_else(window_err)?;
        if C == 2 {
            contiguous_c32::<_, true>(simd, branch, up, hist, input, y, first, down, up);
        } else {
            contiguous_real(simd, branch, up, hist, input, y, first, down, up);
        }
    }
    *q = new_q;
    Ok((consumed, produced))
}

/// `resample` and `fir_interpolate` for `CQ15` (`C = 2`), `resample_real` and `fir_interpolate_real`
/// for `Q15` (`C = 1`), with the bookkeeping of [`polyphase_plan`], one
/// output at a time; `None` (with nothing touched) if the branches do not fit the prepared
/// storage.
#[inline(always)]
fn polyphase_q15<S: SimdToken, const C: usize>(
    simd: S,
    taps: &[i16],
    hist: &[i16],
    input: &[i16],
    out: &mut [i16],
    up: usize,
    down: usize,
    q: &mut i64,
) -> Result<Option<(usize, usize)>> {
    let geom = Branches {
        n: taps.len(),
        up,
        g: S::i16s::LEN / C,
        c: C,
    };
    if !geom.fits() {
        return Ok(None);
    }
    let (consumed, produced, new_q) = polyphase_plan(*q, up, down, input.len() / C, out.len() / C)
        .ok_or_else(|| internal("resampler window"))?;
    #[cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    if let Some(level) = split_level(simd) {
        polyphase_split::<C>(level, taps, hist, input, out, geom, down, *q, produced)?;
        *q = new_q;
        return Ok(Some((consumed, produced)));
    }
    let hist_len = hist.len() / C;
    let mut storage: TapStorageQ15 = [0i16; 2 * CAPACITY];
    let mut e = Q15Engine {
        simd,
        taps,
        win: Win::<_, C>::new(hist, input),
        out,
        td: &mut storage,
    };
    drive_polyphase(&mut e, geom, hist_len, down, *q, produced)?;
    *q = new_q;
    Ok(Some((consumed, produced)))
}

impl Fir<C32> for Simd {
    fn fir<Taps, Hist>(
        &self,
        state: &mut FirState<C32, Taps, Hist>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()>
    where
        Taps: AsRef<[f32]>,
        Hist: AsRef<[C32]> + AsMut<[C32]>,
    {
        check_len(input.len(), output.len())?;
        let (taps, hist) = state.parts_mut();
        {
            let h: &[f32] = bytemuck::cast_slice(&*hist);
            let x: &[f32] = bytemuck::cast_slice(input);
            let y: &mut [f32] = bytemuck::cast_slice_mut(output);
            dispatch!(simd => contiguous_c32::<_, false>(simd, taps, 1, h, x, y, 0, 1, 1));
        }
        update_history(hist, input)
    }
}

impl Fir<CQ15> for Simd {
    fn fir<Taps, Hist>(
        &self,
        state: &mut FirState<CQ15, Taps, Hist>,
        input: &[CQ15],
        output: &mut [CQ15],
    ) -> Result<()>
    where
        Taps: AsRef<[Q15]>,
        Hist: AsRef<[CQ15]> + AsMut<[CQ15]>,
    {
        if state.taps().len() > PREP_TAPS {
            return Fallback.fir(state, input, output);
        }
        check_len(input.len(), output.len())?;
        let (taps, hist) = state.parts_mut();
        {
            let t: &[i16] = bytemuck::cast_slice(taps);
            let h: &[i16] = bytemuck::cast_slice(&*hist);
            let x: &[i16] = bytemuck::cast_slice(input);
            let y: &mut [i16] = bytemuck::cast_slice_mut(output);
            dispatch!(simd => fir_cq15(simd, t, h, x, y));
        }
        update_history(hist, input)
    }
}

macro_rules! decimate_impl {
    ($t:ty, $tap:ty, $lane:ty, $run:ident, $limit:expr, $switched:expr) => {
        impl FirDecimate<$t> for Simd {
            fn fir_decimate<Tp, Hist>(
                &self,
                state: &mut FirDecimState<$t, Tp, Hist>,
                input: &[$t],
                output: &mut [$t],
            ) -> Result<Progress>
            where
                Tp: AsRef<[$tap]>,
                Hist: AsRef<[$t]> + AsMut<[$t]>,
            {
                if state.taps().len() > $limit || $switched {
                    return Fallback.fir_decimate(state, input, output);
                }
                let p = state.parts_mut();
                let (consumed, produced, countdown) =
                    decimation_plan(*p.countdown, p.factor, input.len(), output.len());
                {
                    let t: &[$lane] = bytemuck::cast_slice(p.taps);
                    let h: &[$lane] = bytemuck::cast_slice(&*p.hist);
                    let x: &[$lane] = bytemuck::cast_slice(input);
                    let y: &mut [$lane] = bytemuck::cast_slice_mut(output);
                    let (first, factor) = (*p.countdown, p.factor);
                    dispatch!(simd => $run(simd, t, h, x, y, first, factor, produced));
                }
                *p.countdown = countdown;
                commit(p.hist, input, consumed, produced, "decimation window")
            }
        }
    };
}

decimate_impl!(C32, f32, f32, decimate_c32, usize::MAX, false);
decimate_impl!(CQ15, Q15, i16, decimate_cq15, PREP_TAPS, false);

/// `resample` (complex) and `resample_real` of `f32` taps: `C` floats per sample.
macro_rules! resample_f32_impl {
    ($tr:ident, $method:ident, $state:ident, $t:ty, $c:literal) => {
        impl $tr<$t> for Simd {
            fn $method<Tp, Hist>(
                &self,
                state: &mut $state<$t, Tp, Hist>,
                input: &[$t],
                output: &mut [$t],
            ) -> Result<Progress>
            where
                Tp: AsRef<[f32]>,
                Hist: AsRef<[$t]> + AsMut<[$t]>,
            {
                let p = state.parts_mut();
                let (consumed, produced) = {
                    let h: &[f32] = bytemuck::cast_slice(&*p.hist);
                    let x: &[f32] = bytemuck::cast_slice(input);
                    let y: &mut [f32] = bytemuck::cast_slice_mut(output);
                    let (taps, up, down) = (p.taps, p.up, p.down);
                    dispatch!(simd => polyphase_f32::<_, $c>(
                        simd, taps, h, x, y, up, down, &mut *p.q
                    ))?
                };
                commit(p.hist, input, consumed, produced, "resampler window")
            }
        }
    };
}

resample_f32_impl!(Resample, resample, ResampleState, C32, 2);
resample_f32_impl!(ResampleReal, resample_real, ResampleRealState, f32, 1);

/// `resample` (complex) and `resample_real` of `Q15` taps: `C` `i16` per sample. `$branch` is
/// the SIMD branch alone, a function of its own, which `fir_interpolate` (`down = 1`) also runs.
macro_rules! resample_q15_impl {
    ($tr:ident, $method:ident, $branch:ident, $state:ident, $t:ty, $c:literal) => {
        impl $tr<$t> for Simd {
            fn $method<Tp, Hist>(
                &self,
                state: &mut $state<$t, Tp, Hist>,
                input: &[$t],
                output: &mut [$t],
            ) -> Result<Progress>
            where
                Tp: AsRef<[Q15]>,
                Hist: AsRef<[$t]> + AsMut<[$t]>,
            {
                $branch(state, input, output)
            }
        }

        #[inline(never)]
        fn $branch<Tp, Hist>(
            state: &mut $state<$t, Tp, Hist>,
            input: &[$t],
            output: &mut [$t],
        ) -> Result<Progress>
        where
            Tp: AsRef<[Q15]>,
            Hist: AsRef<[$t]> + AsMut<[$t]>,
        {
            let p = state.parts_mut();
            let ran = {
                let t: &[i16] = bytemuck::cast_slice(p.taps);
                let h: &[i16] = bytemuck::cast_slice(&*p.hist);
                let x: &[i16] = bytemuck::cast_slice(input);
                let y: &mut [i16] = bytemuck::cast_slice_mut(output);
                let (up, down) = (p.up, p.down);
                dispatch!(simd => polyphase_q15::<_, $c>(
                    simd, t, h, x, y, up, down, &mut *p.q
                ))?
            };
            // A filter whose branches do not fit the prepared storage runs on the fallback;
            // nothing of the state has been touched then.
            let Some((consumed, produced)) = ran else {
                return Fallback.$method(state, input, output);
            };
            commit(p.hist, input, consumed, produced, "resampler window")
        }
    };
}

resample_q15_impl!(
    Resample,
    resample,
    resample_q15_branch,
    ResampleState,
    CQ15,
    2
);
resample_q15_impl!(
    ResampleReal,
    resample_real,
    resample_real_q15_branch,
    ResampleRealState,
    Q15,
    1
);

/// `fir_interpolate` (complex) and `fir_interpolate_real` of `f32` taps: the resampler with
/// `down = 1`.
macro_rules! interpolate_impl {
    ($tr:ident, $method:ident, $state:ident, $resample:ident, $t:ty) => {
        impl $tr<$t> for Simd {
            fn $method<Tp, Hist>(
                &self,
                state: &mut $state<$t, Tp, Hist>,
                input: &[$t],
                output: &mut [$t],
            ) -> Result<Progress>
            where
                Tp: AsRef<[f32]>,
                Hist: AsRef<[$t]> + AsMut<[$t]>,
            {
                self.$resample(state.inner_mut(), input, output)
            }
        }
    };
}

impl FirInterpolate<C32> for Simd {
    fn fir_interpolate<Tp, Hist>(
        &self,
        state: &mut FirInterpState<C32, Tp, Hist>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<Progress>
    where
        Tp: AsRef<[f32]>,
        Hist: AsRef<[C32]> + AsMut<[C32]>,
    {
        let (n, l) = (state.taps().len(), state.factor());
        if !short_branches(n, l) || !short_fits(l) {
            return self.resample(state.inner_mut(), input, output);
        }
        // The short-branch form (D1420), also for the outputs still pending of a sample and a
        // last sample with room for only some of its outputs, so that every output has the
        // same bits whichever way the input was split into calls.
        let p = state.inner_mut().parts_mut();
        let (consumed, produced, q) = {
            let h: &[f32] = bytemuck::cast_slice(&*p.hist);
            let x: &[f32] = bytemuck::cast_slice(input);
            let y: &mut [f32] = bytemuck::cast_slice_mut(output);
            short_call(p.taps, l, h, *p.q, x, y)
        };
        *p.q = q;
        commit(p.hist, input, consumed, produced, "interpolator window")
    }
}

/// [`short_c32`] at the current level. Not generic and not inlined, so that it is compiled
/// with the optimisation of this crate even when the generic `fir_interpolate` is instantiated in
/// an unoptimised crate (the inlined forms of every width would give a frame beyond the 1 MiB
/// stack of `wasm32` there).
#[inline(never)]
fn short_call(
    taps: &[f32],
    l: usize,
    hist: &[f32],
    q: i64,
    xs: &[f32],
    out: &mut [f32],
) -> (usize, usize, i64) {
    dispatch!(simd => short_c32(simd, taps, l, hist, q, xs, out))
}

/// The most vectors of outputs per sample (columns) of the short-branch form.
const SHORT_COLS: usize = 4;
/// Floats of the outputs of one sample in the short-branch form (`2 L`, at most [`SHORT_COLS`]
/// vectors of 16 outputs).
const SHORT_ONE: usize = 128;

/// The columns of the short-branch form: vectors of `lanes` outputs for the `l` outputs of a
/// sample.
#[inline(always)]
fn short_cols(l: usize, lanes: usize) -> usize {
    l.div_ceil(lanes)
}

/// Whether the `l` outputs of a sample fit [`SHORT_COLS`] vectors of the current level (else
/// the polyphase form runs).
fn short_fits(l: usize) -> bool {
    let lanes = dispatch!(simd => short_lanes(simd));
    short_cols(l, short_width(lanes, l)) <= SHORT_COLS
}

/// The floats of the native vector of the level.
#[inline(always)]
fn short_lanes<S: SimdToken>(_simd: S) -> usize {
    S::f32s::LEN
}

/// The short-branch form of `fir_interpolate` `C32` (D1420) on the native vectors of the level:
/// for every input sample its `L` outputs together, the real and the imaginary parts in
/// vectors of their own (one lane per output). Tap row `r` (`taps[r L .. r L + L]`,
/// contiguous, loaded as it is) multiplies the sample `r` back, its real and imaginary part
/// broadcast, for all `L` outputs; the outputs of two samples run together (independent
/// chains), and the parts are interleaved when they are stored. Rows ascending, fused on the
/// levels with FMA (D5); the last row of `rem < L` taps adds only to the lanes of its taps (a
/// select), so no output takes a sample that is not in its window. `hist` (`ceil(N / L)`
/// samples) and `xs` are interleaved `C32`. The plan of the polyphase state with `down = 1`:
/// first the outputs still pending of the newest sample of the history (`q < 0`), then whole
/// samples, then the first outputs of one more sample if the output has room for some but not
/// all of its outputs. Returns `(consumed, produced, q)` in samples; the caller has checked
/// [`short_fits`] and updates the history.
#[inline(always)]
fn short_c32<S: SimdToken>(
    simd: S,
    taps: &[f32],
    l: usize,
    hist: &[f32],
    q: i64,
    xs: &[f32],
    out: &mut [f32],
) -> (usize, usize, i64) {
    let fma = has_fma(simd);
    match (short_width(S::f32s::LEN, l), fma) {
        (16, _) => short_lv::<S, f32x16<S>, true>(simd, taps, l, hist, q, xs, out),
        (8, _) => short_lv::<S, f32x8<S>, true>(simd, taps, l, hist, q, xs, out),
        (_, true) => short_lv::<S, f32x4<S>, true>(simd, taps, l, hist, q, xs, out),
        _ => short_lv::<S, f32x4<S>, false>(simd, taps, l, hist, q, xs, out),
    }
}

/// `count` samples of the short-branch form with vectors `V`: the window of sample `j` ends at
/// the complex sample `end + j` of `src`, its `L` outputs go to `out[2 L j ..]`. A factor of
/// whole vectors runs [`short_whole`], any other [`short_run`].
#[inline(always)]
fn short_any<S: SimdToken, V: Lanes<S>, const FMA: bool>(
    simd: S,
    taps: &[f32],
    l: usize,
    src: &[f32],
    end: usize,
    count: usize,
    out: &mut [f32],
) {
    if l.is_multiple_of(V::LEN) {
        match l / V::LEN {
            1 => short_whole::<S, V, 1, FMA>(simd, taps, l, src, end, count, out),
            2 => short_whole::<S, V, 2, FMA>(simd, taps, l, src, end, count, out),
            3 => short_whole::<S, V, 3, FMA>(simd, taps, l, src, end, count, out),
            _ => short_whole::<S, V, SHORT_COLS, FMA>(simd, taps, l, src, end, count, out),
        }
        return;
    }
    match short_cols(l, V::LEN) {
        1 => short_run::<S, V, 1, FMA>(simd, taps, l, src, end, count, out),
        2 => short_run::<S, V, 2, FMA>(simd, taps, l, src, end, count, out),
        3 => short_run::<S, V, 3, FMA>(simd, taps, l, src, end, count, out),
        _ => short_run::<S, V, SHORT_COLS, FMA>(simd, taps, l, src, end, count, out),
    }
}

/// The vectors of the short-branch form for the factor `l` on a level of `native` lanes: the
/// widest of 16, 8 and 4 lanes (at most `native`) that divides `l` into at most [`SHORT_COLS`]
/// vectors (whole rows, no partial loads), else `native`.
#[inline(always)]
fn short_width(native: usize, l: usize) -> usize {
    [16, 8, 4]
        .into_iter()
        .find(|&w| w <= native && l.is_multiple_of(w) && l / w <= SHORT_COLS)
        .unwrap_or(native)
}

/// [`short_run`] for `L = COLS V::LEN`: the full tap rows load whole vectors in place and the
/// outputs store whole vectors, without the checks of the general form; `4 / COLS` samples
/// per pass (independent chains that hide the latency of the multiply-add), each output with
/// two sums (the first half of the rows and the rest, added at the end: shorter chains of
/// rounded additions than one sum over all rows).
#[inline(always)]
fn short_whole<S: SimdToken, V: Lanes<S>, const COLS: usize, const FMA: bool>(
    simd: S,
    taps: &[f32],
    l: usize,
    src: &[f32],
    end: usize,
    count: usize,
    out: &mut [f32],
) {
    match COLS {
        1 => short_pass::<S, V, COLS, 4, FMA>(simd, taps, l, src, end, count, out),
        2 => short_pass::<S, V, COLS, 2, FMA>(simd, taps, l, src, end, count, out),
        _ => short_pass::<S, V, COLS, 1, FMA>(simd, taps, l, src, end, count, out),
    }
}

/// [`short_whole`] with `P` samples per pass.
#[allow(
    clippy::large_stack_frames,
    reason = "clippy counts the generic vectors of every instance as 64-byte values; the sums \
              of an instance are at most 16 vectors of the level, kept in registers"
)]
#[inline(always)]
fn short_pass<S: SimdToken, V: Lanes<S>, const COLS: usize, const P: usize, const FMA: bool>(
    simd: S,
    taps: &[f32],
    l: usize,
    src: &[f32],
    end: usize,
    count: usize,
    out: &mut [f32],
) {
    if count == 0 || l != COLS * V::LEN {
        return;
    }
    let n = taps.len();
    let k = n.div_ceil(l);
    let rows = taps.chunks_exact(l);
    let last = rows.remainder();
    let rem = last.len();
    let zero = V::splat(simd, 0.0);
    let iota = V::from_slice(simd, IOTA.get(..V::LEN).unwrap_or_default());
    // The last row's taps (zero beyond them) and the lanes they add to, per column. No
    // closures here: they would be compiled without the target features of the level in a
    // build with run-time detection.
    let mut tail = [(zero, zero.simd_lt(zero)); COLS];
    for (c, slot) in tail.iter_mut().enumerate() {
        *slot = short_tail(simd, last, c, rem, iota);
    }
    let mut outs = out.chunks_exact_mut(2 * l);
    let mut j = 0;
    while j < count {
        // Samples `j .. j + P` (the last pass repeats its first sample where fewer are left).
        let live = P.min(count - j);
        // (A closure without vector operations: the slices of the windows.)
        let mut pairs: [core::slice::RChunksExact<'_, f32>; P] = core::array::from_fn(|p| {
            let s = j + p.min(live - 1);
            src.get(2 * (end + s + 1 - k)..2 * (end + s + 1))
                .unwrap_or_default()
                .rchunks_exact(2)
        });
        // Two sums per output: the first half of the rows (`re0`, `im0`) and the rest (`re1`,
        // `im1`), each in the loop of one sum, added at the end.
        let (mut re0, mut im0) = ([[zero; COLS]; P], [[zero; COLS]; P]);
        let (mut re1, mut im1) = ([[zero; COLS]; P], [[zero; COLS]; P]);
        let mut it = rows.clone();
        for row in it.by_ref().take((n / l).div_ceil(2)) {
            short_row::<S, V, COLS, P, FMA>(simd, row, &mut pairs, &mut re0, &mut im0);
        }
        for row in it {
            short_row::<S, V, COLS, P, FMA>(simd, row, &mut pairs, &mut re1, &mut im1);
        }
        if rem > 0 {
            // The last row belongs to the second half.
            let xs = short_parts::<S, V, P>(simd, &mut pairs);
            for (c, &(t, m)) in tail.iter().enumerate() {
                for ((r, i), (xr, xi)) in re1.iter_mut().zip(im1.iter_mut()).zip(xs) {
                    if let (Some(r), Some(i)) = (r.get_mut(c), i.get_mut(c)) {
                        *r = m.select(madd::<S, V, FMA>(xr, t, *r), *r);
                        *i = m.select(madd::<S, V, FMA>(xi, t, *i), *i);
                    }
                }
            }
        }
        let (mut re, mut im) = (re0, im0);
        for ((r, i), (r1, i1)) in re
            .iter_mut()
            .zip(im.iter_mut())
            .zip(re1.iter().zip(im1.iter()))
        {
            for ((a, b), (c, d)) in r.iter_mut().zip(i.iter_mut()).zip(r1.iter().zip(i1.iter())) {
                *a += *c;
                *b += *d;
            }
        }
        for (r, i) in re.iter().zip(&im).take(live) {
            if let Some(o) = outs.next() {
                for ((o, r), i) in o.chunks_exact_mut(2 * V::LEN).zip(r).zip(i) {
                    let (lo, hi) = o.split_at_mut(V::LEN);
                    r.zip_low(*i).store_slice(lo);
                    r.zip_high(*i).store_slice(hi);
                }
            }
        }
        j += live;
    }
}

/// One full tap row for `P` samples: the next sample of each window times the row, added to
/// the sums `re`, `im` (one vector per column).
#[inline(always)]
fn short_row<S: SimdToken, V: Lanes<S>, const COLS: usize, const P: usize, const FMA: bool>(
    simd: S,
    row: &[f32],
    pairs: &mut [core::slice::RChunksExact<'_, f32>; P],
    re: &mut [[V; COLS]; P],
    im: &mut [[V; COLS]; P],
) {
    let xs = short_parts::<S, V, P>(simd, pairs);
    for (c, t) in row.chunks_exact(V::LEN).enumerate() {
        let t = V::from_slice(simd, t);
        for ((r, i), (xr, xi)) in re.iter_mut().zip(im.iter_mut()).zip(xs) {
            if let (Some(r), Some(i)) = (r.get_mut(c), i.get_mut(c)) {
                *r = madd::<S, V, FMA>(xr, t, *r);
                *i = madd::<S, V, FMA>(xi, t, *i);
            }
        }
    }
}

/// The parts of the next sample of each of the `P` windows, broadcast (zero past a window).
#[inline(always)]
fn short_parts<S: SimdToken, V: Lanes<S>, const P: usize>(
    simd: S,
    pairs: &mut [core::slice::RChunksExact<'_, f32>; P],
) -> [(V, V); P] {
    let zero = V::splat(simd, 0.0);
    let mut xs = [(zero, zero); P];
    for (x, w) in xs.iter_mut().zip(pairs.iter_mut()) {
        if let Some([xr, xi]) = w.next() {
            *x = (V::splat(simd, *xr), V::splat(simd, *xi));
        }
    }
    xs
}

/// The taps of the last row `last` in column `c` (zero beyond them) and the lanes of its `rem`
/// taps.
#[inline(always)]
fn short_tail<S: SimdToken, V: Lanes<S>>(
    simd: S,
    last: &[f32],
    c: usize,
    rem: usize,
    iota: V,
) -> (V, V::Mask) {
    let mut t = [0.0f32; 16];
    for (d, s) in t.iter_mut().zip(last.get(c * V::LEN..).unwrap_or_default()) {
        *d = *s;
    }
    let m = (iota + V::splat(simd, (c * V::LEN) as f32)).simd_lt(V::splat(simd, rem as f32));
    (V::from_slice(simd, t.get(..V::LEN).unwrap_or_default()), m)
}

/// [`short_c32`] on the vectors `V`.
#[allow(
    clippy::large_stack_frames,
    reason = "clippy adds the buffers of both branches and every temporary; the frame holds the \
              256-byte window and one 512-byte buffer of outputs"
)]
#[inline(always)]
fn short_lv<S: SimdToken, V: Lanes<S>, const FMA: bool>(
    simd: S,
    taps: &[f32],
    l: usize,
    hist: &[f32],
    q: i64,
    xs: &[f32],
    out: &mut [f32],
) -> (usize, usize, i64) {
    let k = taps.len().div_ceil(l);
    let samples = xs.len() / 2;
    let room = out.len() / 2;
    let mut produced = 0;
    // The outputs still pending of the newest sample of the history (its window is the whole
    // history).
    if q < 0 {
        let pending = usize::try_from(-q).unwrap_or(0).min(l);
        let first = l - pending;
        let take = pending.min(room);
        let mut one = [0.0f32; SHORT_ONE];
        short_any::<S, V, FMA>(
            simd,
            taps,
            l,
            hist,
            k - 1,
            1,
            one.get_mut(..2 * l).unwrap_or_default(),
        );
        if let (Some(d), Some(s)) = (
            out.get_mut(..2 * take),
            one.get(2 * first..2 * (first + take)),
        ) {
            d.copy_from_slice(s);
        }
        produced = take;
        if take < pending {
            return (0, produced, q + take as i64);
        }
    }
    // The history and the first samples of the call as one window (oldest first).
    let mut stitched = [0.0f32; 4 * SHORT_MAX_BRANCH];
    let lead = xs.len().min(2 * k);
    let (a, b) = stitched.split_at_mut(2 * k);
    a.copy_from_slice(hist.get(..2 * k).unwrap_or_default());
    if let Some(b) = b.get_mut(..lead) {
        b.copy_from_slice(xs.get(..lead).unwrap_or_default());
    }
    // Whole samples: those whose windows reach into the history from `stitched` (the window of
    // sample `j` ends at its sample `k + j`), the others from `xs` (ends at sample `j`).
    let whole = samples.min((room - produced) / l);
    let head = whole.min(k - 1);
    let dst = out
        .get_mut(2 * produced..2 * (produced + whole * l))
        .unwrap_or_default();
    let (o_head, o_rest) = dst.split_at_mut(2 * head * l);
    let src_head = stitched.get(..2 * (k + head)).unwrap_or_default();
    short_any::<S, V, FMA>(simd, taps, l, src_head, k, head, o_head);
    short_any::<S, V, FMA>(simd, taps, l, xs, head, whole - head, o_rest);
    produced += whole * l;
    // One more sample with room for only some of its outputs.
    let part = room - produced;
    if whole < samples && part > 0 {
        let src = if whole + 1 >= k {
            xs
        } else {
            stitched.as_slice()
        };
        let end = if whole + 1 >= k { whole } else { k + whole };
        let mut one = [0.0f32; SHORT_ONE];
        short_any::<S, V, FMA>(
            simd,
            taps,
            l,
            src,
            end,
            1,
            one.get_mut(..2 * l).unwrap_or_default(),
        );
        if let (Some(d), Some(s)) = (out.get_mut(2 * produced..2 * room), one.get(..2 * part)) {
            d.copy_from_slice(s);
        }
        return (whole + 1, room, part as i64 - l as i64);
    }
    (whole, produced, 0)
}

/// The `C32` short-branch form for `count` samples, `COLS` vectors of `V::LEN` outputs each
/// (see [`short_c32`]).
#[inline(always)]
fn short_run<S: SimdToken, V: Lanes<S>, const COLS: usize, const FMA: bool>(
    simd: S,
    taps: &[f32],
    l: usize,
    src: &[f32],
    end: usize,
    count: usize,
    out: &mut [f32],
) {
    if count == 0 {
        return;
    }
    let n = taps.len();
    let k = n.div_ceil(l);
    let full = n / l;
    let rem = n - full * l;
    let zero = V::splat(simd, 0.0);
    // The lanes of the last row's taps per column (no closures with vector operations, see
    // `short_pass`).
    let iota = V::from_slice(simd, IOTA.get(..V::LEN).unwrap_or_default());
    let mut masks = [zero.simd_lt(zero); COLS];
    for (c, m) in masks.iter_mut().enumerate() {
        *m = (iota + V::splat(simd, (c * V::LEN) as f32)).simd_lt(V::splat(simd, rem as f32));
    }
    let mut outs = out.chunks_exact_mut(2 * l);
    let mut j = 0;
    while j < count {
        let two = j + 1 < count;
        let (mut ar, mut ai, mut br, mut bi) =
            ([zero; COLS], [zero; COLS], [zero; COLS], [zero; COLS]);
        // The windows: `k` complex samples ending at `end + j` (the oldest first).
        let wa = src
            .get(2 * (end + j + 1 - k)..2 * (end + j + 1))
            .unwrap_or_default();
        let wb = if two {
            src.get(2 * (end + j + 2 - k)..2 * (end + j + 2))
                .unwrap_or_default()
        } else {
            wa
        };
        let mut pa = wa.chunks_exact(2).rev();
        let mut pb = wb.chunks_exact(2).rev();
        for (r, (x, y)) in (0..full).zip(pa.by_ref().zip(pb.by_ref())) {
            let (xr, xi) = short_pair::<S, V>(simd, x);
            let (yr, yi) = short_pair::<S, V>(simd, y);
            for c in 0..COLS {
                let t = short_load::<S, V>(simd, taps, r * l + c * V::LEN);
                if let (Some(a0), Some(a1), Some(b0), Some(b1)) =
                    (ar.get_mut(c), ai.get_mut(c), br.get_mut(c), bi.get_mut(c))
                {
                    *a0 = madd::<S, V, FMA>(xr, t, *a0);
                    *a1 = madd::<S, V, FMA>(xi, t, *a1);
                    *b0 = madd::<S, V, FMA>(yr, t, *b0);
                    *b1 = madd::<S, V, FMA>(yi, t, *b1);
                }
            }
        }
        if let (true, Some(x), Some(y)) = (rem > 0, pa.next(), pb.next()) {
            let (xr, xi) = short_pair::<S, V>(simd, x);
            let (yr, yi) = short_pair::<S, V>(simd, y);
            for (c, m) in masks.iter().enumerate() {
                let t = short_load::<S, V>(simd, taps, full * l + c * V::LEN);
                if let (Some(a0), Some(a1), Some(b0), Some(b1)) =
                    (ar.get_mut(c), ai.get_mut(c), br.get_mut(c), bi.get_mut(c))
                {
                    *a0 = m.select(madd::<S, V, FMA>(xr, t, *a0), *a0);
                    *a1 = m.select(madd::<S, V, FMA>(xi, t, *a1), *a1);
                    *b0 = m.select(madd::<S, V, FMA>(yr, t, *b0), *b0);
                    *b1 = m.select(madd::<S, V, FMA>(yi, t, *b1), *b1);
                }
            }
        }
        short_store(outs.next(), &ar, &ai);
        if two {
            short_store(outs.next(), &br, &bi);
        }
        j += 2;
    }
}

/// The parts of the complex sample `x` (two floats), broadcast.
#[inline(always)]
fn short_pair<S: SimdToken, V: Lanes<S>>(simd: S, x: &[f32]) -> (V, V) {
    (
        V::splat(simd, x.first().copied().unwrap_or(0.0)),
        V::splat(simd, x.get(1).copied().unwrap_or(0.0)),
    )
}

/// The taps `taps[at .. at + V::LEN]`, zero beyond the taps (the lanes beyond `L` are not
/// stored, those beyond the last row's taps not added).
#[inline(always)]
fn short_load<S: SimdToken, V: Lanes<S>>(simd: S, taps: &[f32], at: usize) -> V {
    if let Some(t) = taps.get(at..at + V::LEN) {
        V::from_slice(simd, t)
    } else {
        let mut t = [0.0f32; 16];
        let avail = taps.get(at..).unwrap_or_default();
        for (d, s) in t.iter_mut().zip(avail) {
            *d = *s;
        }
        V::from_slice(simd, t.get(..V::LEN).unwrap_or_default())
    }
}

/// Stores the `L` outputs of the parts `re`, `im` interleaved to `out` (`2 L` floats).
#[inline(always)]
fn short_store<S: SimdToken, V: Lanes<S>, const COLS: usize>(
    out: Option<&mut [f32]>,
    re: &[V; COLS],
    im: &[V; COLS],
) {
    if let Some(o) = out {
        for (c, (r, i)) in re.iter().zip(im).enumerate() {
            let o = o.get_mut(2 * c * V::LEN..).unwrap_or_default();
            let (lo, hi) = o.split_at_mut(o.len().min(V::LEN));
            put(r.zip_low(*i), lo, false);
            put(r.zip_high(*i), hi, false);
        }
    }
}

interpolate_impl!(
    FirInterpolateReal,
    fir_interpolate_real,
    FirInterpRealState,
    resample_real,
    f32
);

/// `fir_interpolate` (complex) and `fir_interpolate_real` of `Q15` taps: the SIMD branch of the
/// resampler with `down = 1`.
macro_rules! interpolate_q15_impl {
    ($tr:ident, $method:ident, $state:ident, $branch:ident, $t:ty) => {
        impl $tr<$t> for Simd {
            fn $method<Tp, Hist>(
                &self,
                state: &mut $state<$t, Tp, Hist>,
                input: &[$t],
                output: &mut [$t],
            ) -> Result<Progress>
            where
                Tp: AsRef<[Q15]>,
                Hist: AsRef<[$t]> + AsMut<[$t]>,
            {
                $branch(state.inner_mut(), input, output)
            }
        }
    };
}

interpolate_q15_impl!(
    FirInterpolate,
    fir_interpolate,
    FirInterpState,
    resample_q15_branch,
    CQ15
);
interpolate_q15_impl!(
    FirInterpolateReal,
    fir_interpolate_real,
    FirInterpRealState,
    resample_real_q15_branch,
    Q15
);

#[cfg(all(test, feature = "std", target_arch = "x86_64"))]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
mod tests {
    //! The progressions of the module `x86` at the level detected at run time (in a build for
    //! a lower level, such as the run-time build that detects AVX2, D3007) give the outputs of
    //! the fallback bit for bit (the kernel tests in `tests/` check the outputs against the
    //! definition at every level).

    extern crate std;

    use std::vec;
    use std::vec::Vec;

    use sdrk_core::kernels::filter::{Fir, FirState};
    use sdrk_core::kernels::interp::{
        FirInterpRealState, FirInterpState, FirInterpolate, FirInterpolateReal,
    };
    use sdrk_core::{CQ15, Fallback, Q15};

    use crate::Simd;

    fn noise(s: &mut u32, n: usize) -> Vec<Q15> {
        (0..n)
            .map(|_| {
                *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                Q15((*s >> 12) as i16)
            })
            .collect()
    }

    fn cq15(v: &[Q15]) -> Vec<CQ15> {
        v.chunks_exact(2)
            .map(|p| CQ15 { re: p[0], im: p[1] })
            .collect()
    }

    #[test]
    fn the_detected_level_gives_the_bits_of_the_fallback() {
        let fb = Simd;
        let mut s = 7;
        for (n, l) in [(1usize, 2usize), (31, 2), (64, 4), (127, 3), (300, 8)] {
            let taps = noise(&mut s, n);
            let mut a =
                FirState::<CQ15, _, _>::new(taps.clone(), vec![CQ15::default(); n - 1]).unwrap();
            let mut b =
                FirState::<CQ15, _, _>::new(taps.clone(), vec![CQ15::default(); n - 1]).unwrap();
            let h = n.div_ceil(l);
            let mut ia =
                FirInterpState::<CQ15, _, _>::new(taps.clone(), vec![CQ15::default(); h], l)
                    .unwrap();
            let mut ib =
                FirInterpState::<CQ15, _, _>::new(taps.clone(), vec![CQ15::default(); h], l)
                    .unwrap();
            let mut ra =
                FirInterpRealState::<Q15, _, _>::new(taps.clone(), vec![Q15(0); h], l).unwrap();
            let mut rb = FirInterpRealState::<Q15, _, _>::new(taps, vec![Q15(0); h], l).unwrap();
            for len in [0usize, 1, 9, 100, 333] {
                let x = cq15(&noise(&mut s, 2 * len));
                let (mut ya, mut yb) = (vec![CQ15::default(); len], vec![CQ15::default(); len]);
                Fallback.fir(&mut a, &x, &mut ya).unwrap();
                fb.fir(&mut b, &x, &mut yb).unwrap();
                assert_eq!(ya, yb, "fir {n} taps, {len} samples");
                let room = len * l - len % 2;
                let (mut ya, mut yb) = (vec![CQ15::default(); room], vec![CQ15::default(); room]);
                let pa = Fallback.fir_interpolate(&mut ia, &x, &mut ya).unwrap();
                let pb = fb.fir_interpolate(&mut ib, &x, &mut yb).unwrap();
                assert_eq!((pa.consumed, pa.produced), (pb.consumed, pb.produced));
                assert_eq!(ya, yb, "fir_interpolate {n} taps, L {l}, {len} samples");
                let x = noise(&mut s, len);
                let (mut ya, mut yb) = (vec![Q15(0); room], vec![Q15(0); room]);
                let pa = Fallback.fir_interpolate_real(&mut ra, &x, &mut ya).unwrap();
                let pb = fb.fir_interpolate_real(&mut rb, &x, &mut yb).unwrap();
                assert_eq!((pa.consumed, pa.produced), (pb.consumed, pb.produced));
                assert_eq!(
                    ya, yb,
                    "fir_interpolate_real {n} taps, L {l}, {len} samples"
                );
            }
        }
    }
}
