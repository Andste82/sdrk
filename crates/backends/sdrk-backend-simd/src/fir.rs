//! The FIR family: `fir`, `fir_decimate` and the polyphase interpolator `fir_interpolate`
//! (also exposed as `resample`).
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
//!   (see [`polyphase_c32`]), and the filter length is not limited.
//! - `CQ15` `fir`, `fir_decimate` and the polyphase kernels compute one output at a time in the
//!   chunk form with the taps prepared once per call, reversed and duplicated for the real
//!   and the imaginary lane. A window that reaches into the history is assembled chunk by chunk
//!   with the same chunk boundaries. The polyphase kernels (`resample`, `fir_interpolate`)
//!   prepare the taps of all branches at once; a filter that does not fit, and every `CQ15`
//!   filter with more than [`PREP_TAPS`] taps, runs on the fallback.
//!
//! `C32` results are within the contract of `docs/kernels.md` (110 dB) but not bit-identical
//! with the fallback, because the sums are formed in a different order (and fused on the levels
//! with FMA). `CQ15` results are exact integer sums and therefore bit-identical.

use fearless_simd::{Simd as SimdToken, f32x4, f32x8, f32x16, f64x2, prelude::*};
use sdrk_core::kernels::decim::{FirDecimState, FirDecimate, Resample, ResampleState};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::interp::{FirInterpState, FirInterpolate};
use sdrk_core::num::Acc64;
use sdrk_core::{C32, CQ15, Fallback, Progress, Q15, Result};

use crate::Simd;
use crate::level::dispatch;
use crate::util::{check_len, internal};

/// The longest `CQ15` contiguous filter (`fir`, `fir_decimate`) of this backend.
const PREP_TAPS: usize = 256;
/// Samples (window positions, whole chunks) in the prepared storage of one call; the prepared
/// branches of a polyphase filter must fit.
const CAPACITY: usize = 384;

/// The virtual stream `history ++ input`, both as interleaved component slices.
struct Win<'a, T> {
    hist: &'a [T],
    input: &'a [T],
    /// Number of history samples.
    h: usize,
}

impl<'a, T: Copy + Default> Win<'a, T> {
    fn new(hist: &'a [T], input: &'a [T]) -> Self {
        Self {
            hist,
            input,
            h: hist.len() / 2,
        }
    }

    /// `lanes` contiguous components from sample `pos` on, if they lie inside one slice.
    #[inline(always)]
    fn direct(&self, pos: usize, lanes: usize) -> Option<&'a [T]> {
        if pos >= self.h {
            let o = 2 * (pos - self.h);
            self.input.get(o..o + lanes)
        } else {
            let o = 2 * pos;
            self.hist.get(o..o + lanes)
        }
    }
}

/// Copies the samples `s .. end` of the virtual stream `hist ++ input` (interleaved component
/// slices), as far as they exist, to the start of `dst` (the rest of `dst` is untouched): the part
/// in the history, then the part in the input.
#[inline(always)]
fn fill_window<T: Copy>(hist: &[T], input: &[T], s: usize, end: usize, dst: &mut [T]) {
    let (h, len) = (hist.len() / 2, input.len() / 2);
    if s < h
        && let (Some(src), Some(dst)) = (
            hist.get(2 * s..2 * end.min(h)),
            dst.get_mut(..2 * (end.min(h) - s)),
        )
    {
        dst.copy_from_slice(src);
    }
    let (from, to) = (s.max(h), end.min(h + len));
    if from < to
        && let (Some(src), Some(dst)) = (
            input.get(2 * (from - h)..2 * (to - h)),
            dst.get_mut(2 * (from - s)..2 * (to - s)),
        )
    {
        dst.copy_from_slice(src);
    }
}

/// Geometry of the prepared branches of a polyphase filter with `n` taps and factor `up` for
/// chunks of `g` samples.
#[derive(Clone, Copy)]
struct Branches {
    n: usize,
    up: usize,
    g: usize,
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

    /// Offset in floats of the branch `phase` in the prepared storage.
    fn offset(self, phase: usize) -> usize {
        let hi = self.n.div_ceil(self.up);
        let r = self.n - (hi - 1) * self.up; // branches 0..r have `hi` taps, the others `hi - 1`
        let first = phase.min(r);
        2 * (first * self.padded(hi) + (phase - first) * self.padded(hi - 1))
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
        total <= CAPACITY
    }
}

// ---------------------------------------------------------------------------------------------
// C32 contiguous filters (`fir`, `fir_decimate`, the polyphase branches)

/// Taps in one block of the `C32` contiguous filters. A longer filter adds the partial dot
/// products of its blocks in the output.
const BLOCK_TAPS: usize = 128;
/// Samples of the scratch window that serves the outputs whose windows reach into the history
/// or past the end of the input.
const SCRATCH: usize = 192;
/// Lane indices, for the mask of the partial last chunk.
const IOTA: [f32; 16] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
];

/// The `f32` vectors the `C32` filters run on: `f32x4`, `f32x8` or `f32x16`, the native width
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

    /// Computes the `out.len() / 2` outputs of a run whose windows lie in `region`, or adds
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

/// The chunk form: the dot product of a window with the prepared taps in chunks of `L / 2`
/// samples (unit `T`), for `B` outputs at a time that share every tap vector; the windows are
/// `step` samples apart, so this serves every decimation factor. The chunks run in groups of
/// `U`; the products of a group are summed in `I` interleaved chains (chunk `u` to chain
/// `u % I`), which are added and then added to the output's total. The chunks after the last
/// group form one more such group. The lanes of the partial last chunk beyond the window are
/// cleared before the multiplication, so a NaN or an infinity there does not leak in. The complex
/// lanes of a total are added by halving, and two outputs share the last step.
struct ChunkForm<
    'a,
    S: SimdToken,
    V: Lanes<S>,
    T: Unit,
    const L: usize,
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
    const B: usize,
    const U: usize,
    const I: usize,
    const FMA: bool,
> ChunkForm<'_, S, V, T, L, B, U, I, FMA>
{
    /// Samples one window reads (whole chunks).
    #[inline(always)]
    fn span(&self) -> usize {
        self.taps.len() * (L / 2)
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
        let sums = tot.map(V::fold);
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
    const B: usize,
    const U: usize,
    const I: usize,
    const FMA: bool,
> Form for ChunkForm<'_, S, V, T, L, B, U, I, FMA>
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
        let k = out.len() / 2;
        let (step, len) = (2 * self.step, self.taps.len() * L);
        let mut i = 0;
        for dst in out.chunks_exact_mut(2 * B) {
            let o = i * step;
            let Some(pass) = region.get(o..o + (B - 1) * step + len) else {
                return;
            };
            // The windows of the pass. On x86 every window is checked against the pass slice of
            // known length, which lets the window pointers derive from one base; elsewhere
            // stepping through the windows of the slice measured fewer instructions.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            let wins: [&[[f32; L]]; B] = core::array::from_fn(|b| {
                pass.get(b * step..b * step + len)
                    .map_or(&[][..], |w| w.as_chunks::<L>().0)
            });
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
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
                out.get_mut(2 * i..).unwrap_or(&mut []),
                add,
            );
        }
    }
}

/// Taps per block of the transposed form.
const TBLOCK: usize = 8;

/// The transposed form for consecutive outputs (`step = 1`): a vector holds `L / 2`
/// consecutive outputs, and every reversed tap multiplies the samples at its offset for `B` such
/// vectors at a time, so there is no partial chunk and no horizontal sum. The vectors of a pass
/// are `D` samples apart (`L / 2` or a multiple of it): on x86, where the samples are the memory
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
    const B: usize,
    const D: usize,
    const I: usize,
    const FMA: bool,
> TransForm<'_, S, V, T, L, B, D, I, FMA>
{
    /// Adds the block of taps `tb` to `tot` for the `P` output vectors `D` samples apart whose
    /// windows begin at `w` (floats; tap `m` reads from float `2 m` on).
    #[inline(always)]
    fn block<const P: usize>(&self, tb: &[T], w: &[f32], tot: &mut [V; P]) {
        let simd = self.simd;
        let mut chains = [[V::splat(simd, 0.0); I]; P];
        for (m, t) in tb.iter().enumerate() {
            let t = t.load::<S, V>(simd);
            for (p, c) in chains.iter_mut().enumerate() {
                let o = 2 * (m + p * D);
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
        let x = x.get(..2 * ((P - 1) * D + L / 2 + n - 1))?;
        let span = 2 * ((P - 1) * D + L / 2 + TBLOCK - 1);
        let mut tot = [V::splat(self.simd, 0.0); P];
        let (blocks, rest) = self.taps.as_chunks::<TBLOCK>();
        for (k, tb) in blocks.iter().enumerate() {
            let w = x.get(2 * k * TBLOCK..2 * k * TBLOCK + span)?;
            self.block(tb, w, &mut tot);
        }
        if !rest.is_empty() {
            let w = x.get(2 * blocks.len() * TBLOCK..)?;
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
    const B: usize,
    const D: usize,
    const I: usize,
    const FMA: bool,
> Form for TransForm<'_, S, V, T, L, B, D, I, FMA>
{
    #[inline(always)]
    fn needed(&self, k: usize) -> usize {
        k.div_ceil(L / 2) * (L / 2) + self.taps.len() - 1
    }

    #[inline(always)]
    fn pass_outputs(&self) -> usize {
        B * D
    }

    #[inline(always)]
    fn fits(&self, avail: usize) -> usize {
        let g = L / 2;
        let n = self.taps.len();
        if avail + 1 >= n + g {
            (avail + 1 - n) / g * g
        } else {
            0
        }
    }

    #[inline(always)]
    fn run(&self, region: &[f32], out: &mut [f32], add: bool) {
        let g = L / 2;
        let k = out.len() / 2;
        let mut i = 0;
        // `D / g` passes of `B` vectors cover the `B D` outputs from `i` on.
        for dst in out.chunks_exact_mut(2 * B * D) {
            for q in 0..D / g {
                let x = region.get(2 * (i + q * g)..).unwrap_or(&[]);
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
            let x = region.get(2 * i..).unwrap_or(&[]);
            let Some([v]) = self.pass::<1>(x) else {
                return;
            };
            put(v, out.get_mut(2 * i..).unwrap_or(&mut []), add);
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
fn strided(out: &mut [f32], j: usize, stride: usize) -> impl Iterator<Item = &mut [f32]> {
    out.chunks_exact_mut(2).skip(j * stride).step_by(stride)
}

/// Runs `f` on `region` for the `y.len() / 2` outputs from output `j` of the strided output
/// `out`, through `y`: with `add` the outputs are first gathered into `y`, and afterwards `y` is
/// spread over them.
#[inline(always)]
fn run_strided<F: Form>(
    f: &F,
    region: &[f32],
    y: &mut [f32],
    out: &mut [f32],
    j: usize,
    stride: usize,
    add: bool,
) {
    // Output `j + i` is the sample `(j + i) stride` of `out`: the first sample of what is left of
    // `out` after the previous output and the `stride - 1` samples after it.
    let skip = 2 * stride.saturating_sub(1);
    if add {
        let mut rest = out.get(2 * j * stride..).unwrap_or_default();
        for d in y.as_chunks_mut::<2>().0 {
            let Some((s, tail)) = rest.split_first_chunk::<2>() else {
                break;
            };
            *d = *s;
            rest = tail.get(skip..).unwrap_or_default();
        }
    }
    f.run(region, y, add);
    let mut rest = out.get_mut(2 * j * stride..).unwrap_or_default();
    for s in y.as_chunks::<2>().0 {
        let Some((d, tail)) = rest.split_first_chunk_mut::<2>() else {
            break;
        };
        *d = *s;
        rest = tail.get_mut(skip..).unwrap_or_default();
    }
}

/// The outputs of one tap block: output `j` (of `out.len() / 2`) has its window at sample
/// `first + j * step` of the virtual stream `hist ++ input`. Runs whose windows lie inside the
/// input read it in place; the others are copied to `scratch` first, so every output is computed
/// by the same arithmetic wherever its window lies.
#[inline(always)]
fn drive<F: Form>(
    f: &F,
    hist: &[f32],
    input: &[f32],
    out: &mut [f32],
    first: usize,
    step: usize,
    add: bool,
    scratch: &mut [f32; 2 * SCRATCH],
) {
    let (h, len) = (hist.len() / 2, input.len() / 2);
    let count = out.len() / 2;
    let mut j = 0;
    while j < count {
        let s = first + j * step;
        let rest = count - j;
        if s >= h {
            let o = s - h;
            let k = f.fits(len.saturating_sub(o)).min(rest);
            if k > 0 {
                if let (Some(x), Some(y)) = (input.get(2 * o..), out.get_mut(2 * j..2 * (j + k))) {
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
        fill_window(hist, input, s, s + f.needed(k), scratch);
        if let Some(y) = out.get_mut(2 * j..2 * (j + k)) {
            f.run(scratch, y, add);
        }
        j += k;
    }
}

/// [`drive`] for strided outputs (the polyphase filters): output `j` (of `out.len() / 2` divided
/// by `ostride`, rounded up) is sample `j * ostride` of `out`. The outputs of a run are computed
/// into the scratch and then spread over `out`: a run that reads the input in place uses the
/// whole scratch for its outputs (whole passes of the form where possible), a run through the
/// scratch its last [`SCRATCH_OUT`] samples.
#[inline(always)]
fn drive_strided<F: Form>(
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
    let (h, len) = (hist.len() / 2, input.len() / 2);
    let count = (out.len() / 2).div_ceil(ostride);
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
                if let (Some(x), Some(y)) = (input.get(2 * o..), scratch.get_mut(..2 * k)) {
                    run_strided(f, x, y, out, j, ostride, add);
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
        let (region, tail) = scratch.split_at_mut(2 * room);
        fill_window(hist, input, s, s + f.needed(k), region);
        if let Some(y) = tail.get_mut(..2 * k) {
            run_strided(f, region, y, out, j, ostride, add);
        }
        j += k;
    }
}

/// Four floats on a 16-byte boundary: the unit of the prepared taps, so that a vector of four
/// taps is an aligned load (SSE takes only those as the memory operand of `mulps`).
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
struct Quad([f32; 4]);

/// Whether the level has fused multiply-add for the `C32` filters (decision D5: AVX2, AVX-512
/// and NEON; never on the levels without it, and never in the fallback).
#[inline(always)]
fn has_fma<S: SimdToken>(simd: S) -> bool {
    let level = simd.level();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        level.as_avx2().is_some()
    }
    #[cfg(target_arch = "aarch64")]
    {
        level.as_neon().is_some()
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
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

/// One tap block of a `C32` contiguous filter with consecutive outputs: the transposed form
/// (tap unit `T`, `B` vectors `D` samples apart, `I` chains) with the taps of `block`, every
/// `tstride`-th float of it (see [`contiguous`]).
#[inline(always)]
fn transposed_block<
    S: SimdToken,
    V: Lanes<S>,
    T: TapUnit,
    const L: usize,
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
    let form = TransForm::<S, V, T, L, B, D, I, FMA> {
        simd,
        taps: prepared.get(..cnt).unwrap_or(&[]),
        _lanes: core::marker::PhantomData,
    };
    if STRIDED {
        drive_strided(&form, hist, input, out, first, 1, ostride, add, scratch);
    } else {
        drive(&form, hist, input, out, first, 1, add, scratch);
    }
}

/// One tap block of a `C32` contiguous filter with any step: the chunk form (chunk unit `T`,
/// `B` outputs, groups of `U` chunks, `I` chains) with the taps of `block`, every `tstride`-th
/// float of it (see [`contiguous`]).
#[inline(always)]
fn chunked_block<
    S: SimdToken,
    V: Lanes<S>,
    T: Unit,
    const L: usize,
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
    // The taps written twice (real and imaginary lane), zero beyond the window; a quad holds
    // two taps. The chunk width divides `BLOCK_TAPS`, so the chunks end inside the storage.
    let mut storage = [Quad([0.0; 4]); BLOCK_TAPS / 2];
    let floats: &mut [f32] = bytemuck::cast_slice_mut(&mut storage);
    let cnt = block.len().div_ceil(tstride);
    if STRIDED {
        prepare_strided(floats.as_chunks_mut::<2>().0, block, tstride, |t| [t, t]);
    } else {
        for (d, &t) in floats
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(block.iter().rev())
        {
            *d = [t, t];
        }
    }
    let g = L / 2;
    let nc = cnt.div_ceil(g);
    let rem = cnt % g;
    let mask = (rem > 0).then(|| {
        V::from_slice(simd, IOTA.get(..L).unwrap_or(&[])).simd_lt(V::splat(simd, (2 * rem) as f32))
    });
    let chunks: &[T] = bytemuck::cast_slice(&storage);
    let form = ChunkForm::<S, V, T, L, B, U, I, FMA> {
        simd,
        taps: chunks.get(..nc).unwrap_or(&[]),
        mask,
        step,
    };
    if STRIDED {
        drive_strided(&form, hist, input, out, first, step, ostride, add, scratch);
    } else {
        drive(&form, hist, input, out, first, step, add, scratch);
    }
}

/// A `C32` contiguous filter on vectors `V` of `L` lanes: output `j` is the dot product of the taps
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
        assert!(V::LEN == L && size_of::<CT>() == 4 * L && BLOCK_TAPS.is_multiple_of(L / 2));
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
            transposed_block::<S, V, TT, L, TB, TD, TI, FMA, STRIDED>(
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
            chunked_block::<S, V, CT, L, CB, CU, CI, FMA, STRIDED>(
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
    let x86 = cfg!(any(target_arch = "x86", target_arch = "x86_64"));
    match S::f32s::LEN {
        16 => contiguous::<S, f32x16<S>, f32, [f32; 16], 16, true, 14, 8, 1, 7, 8, 1, STRIDED>(
            simd, taps, tstride, hist, input, out, first, step, ostride,
        ),
        8 => contiguous::<S, f32x8<S>, f32, [f32; 8], 8, true, 7, 8, 1, 6, 8, 1, STRIDED>(
            simd, taps, tstride, hist, input, out, first, step, ostride,
        ),
        _ if fma => contiguous::<S, f32x4<S>, f32, [f32; 4], 4, true, 7, 2, 1, 7, 8, 1, STRIDED>(
            simd, taps, tstride, hist, input, out, first, step, ostride,
        ),
        _ if x86 => contiguous::<S, f32x4<S>, Quad, Quad, 4, false, 4, 8, 2, 7, 8, 1, STRIDED>(
            simd, taps, tstride, hist, input, out, first, step, ostride,
        ),
        _ => contiguous::<S, f32x4<S>, f32, [f32; 4], 4, false, 8, 2, 1, 8, 8, 1, STRIDED>(
            simd, taps, tstride, hist, input, out, first, step, ostride,
        ),
    }
}

// ---------------------------------------------------------------------------------------------
// CQ15

/// Storage for the prepared taps of `CQ15` (twice the taps, reversed and duplicated).
type TapStorageQ15 = [i16; 2 * CAPACITY];

struct Q15Engine<'a, S: SimdToken> {
    simd: S,
    taps: &'a [i16],
    win: Win<'a, i16>,
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

impl<S: SimdToken> Q15Engine<'_, S> {
    /// Prepares the taps of the contiguous filter: sample `m` of the window pairs with
    /// `h[n - 1 - m]`.
    #[inline(always)]
    fn prepare(&mut self) {
        for (pair, &h) in self.td.chunks_exact_mut(2).zip(self.taps.iter().rev()) {
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
            g: S::i16s::LEN / 2,
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
                if let Some([re, im]) = self.td.get_mut(off + 2 * m..off + 2 * m + 2) {
                    *re = h;
                    *im = h;
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
        let g = l / 2;
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
            let o = 2 * (start - self.win.h);
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
                fill_window(self.win.hist, self.win.input, pos, pos + take, &mut tmp);
                S::i16s::from_slice(simd, tmp.get(..l).unwrap_or(&[0; 64]))
            };
            let t = td
                .get(c * l..c * l + l)
                .map_or_else(|| S::i16s::splat(simd, 0), |t| S::i16s::from_slice(simd, t));
            acc.mac(d, t);
            c += 1;
        }
        let sum = (acc.a0 + acc.a1) + (acc.a2 + acc.a3);
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
fn drive_contiguous<S: SimdToken>(
    e: &mut Q15Engine<'_, S>,
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
fn drive_polyphase<S: SimdToken>(
    e: &mut Q15Engine<'_, S>,
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
// Bookkeeping shared by both sample types

/// `hist` becomes the last `hist.len()` samples of `hist ++ input`.
fn update_history<T: Copy>(hist: &mut [T], input: &[T]) -> Result<()> {
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
fn commit<T: Copy>(
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
    let mut storage: TapStorageQ15 = [0i16; 2 * CAPACITY];
    let n = taps.len();
    let mut e = Q15Engine {
        simd,
        taps,
        win: Win::new(hist, input),
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
    let mut storage: TapStorageQ15 = [0i16; 2 * CAPACITY];
    let n = taps.len();
    let mut e = Q15Engine {
        simd,
        taps,
        win: Win::new(hist, input),
        out,
        td: &mut storage,
    };
    drive_contiguous(&mut e, n, count, first, factor);
}

/// `resample` and `fir_interpolate` for `C32` (the bookkeeping of [`polyphase_plan`]). The outputs
/// `r, r + L, r + 2 L, ...` of a call share the phase of output `r`, and their windows are `M`
/// samples apart, so each of these progressions is one contiguous filter with the taps of its
/// branch, step `M` and output stride `L`: the transposed form for `fir_interpolate` (`M = 1`), the
/// chunk form for every other `M`. Each output is therefore computed as `fir` or `fir_decimate`
/// computes it, whichever way the input was split into blocks.
#[inline(always)]
fn polyphase_c32<S: SimdToken>(
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
        polyphase_plan(*q, up, down, input.len() / 2, out.len() / 2).ok_or_else(window_err)?;
    // `usize` to `i128` is lossless.
    let (q0, l, m) = (i128::from(*q), up as i128, down as i128);
    let hist_len = hist.len() / 2;
    let out = out.get_mut(..2 * produced).ok_or_else(window_err)?;
    for r in 0..up.min(produced) {
        // Output `r` is ready after `c` input samples; its window ends at the newest of them.
        let (c, phase) = polyphase_output(q0 + r as i128 * m, l)?;
        let y = out.get_mut(2 * r..).ok_or_else(window_err)?;
        let Some(branch) = taps.get(phase..).filter(|b| !b.is_empty()) else {
            // A phase beyond the taps (`L > N`) has no taps: its outputs are zero.
            strided(y, 0, up).for_each(|d| d.fill(0.0));
            continue;
        };
        let t = (n - phase).div_ceil(up);
        let first = (hist_len + c).checked_sub(t).ok_or_else(window_err)?;
        contiguous_c32::<_, true>(simd, branch, up, hist, input, y, first, down, up);
    }
    *q = new_q;
    Ok((consumed, produced))
}

/// `resample` and `fir_interpolate` for `CQ15` (the bookkeeping of [`polyphase_plan`]), one
/// output at a time; `None` (with nothing touched) if the branches do not fit the prepared
/// storage.
#[inline(always)]
fn polyphase_cq15<S: SimdToken>(
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
        g: S::i16s::LEN / 2,
    };
    if !geom.fits() {
        return Ok(None);
    }
    let (consumed, produced, new_q) = polyphase_plan(*q, up, down, input.len() / 2, out.len() / 2)
        .ok_or_else(|| internal("resampler window"))?;
    let hist_len = hist.len() / 2;
    let mut storage: TapStorageQ15 = [0i16; 2 * CAPACITY];
    let mut e = Q15Engine {
        simd,
        taps,
        win: Win::new(hist, input),
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
    ($t:ty, $tap:ty, $lane:ty, $run:ident, $limit:expr) => {
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
                if state.taps().len() > $limit {
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

decimate_impl!(C32, f32, f32, decimate_c32, usize::MAX);
decimate_impl!(CQ15, Q15, i16, decimate_cq15, PREP_TAPS);

impl Resample<C32> for Simd {
    fn resample<Tp, Hist>(
        &self,
        state: &mut ResampleState<C32, Tp, Hist>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<Progress>
    where
        Tp: AsRef<[f32]>,
        Hist: AsRef<[C32]> + AsMut<[C32]>,
    {
        let p = state.parts_mut();
        let (consumed, produced) = {
            let h: &[f32] = bytemuck::cast_slice(&*p.hist);
            let x: &[f32] = bytemuck::cast_slice(input);
            let y: &mut [f32] = bytemuck::cast_slice_mut(output);
            let (taps, up, down) = (p.taps, p.up, p.down);
            dispatch!(simd => polyphase_c32(simd, taps, h, x, y, up, down, &mut *p.q))?
        };
        commit(p.hist, input, consumed, produced, "resampler window")
    }
}

impl Resample<CQ15> for Simd {
    fn resample<Tp, Hist>(
        &self,
        state: &mut ResampleState<CQ15, Tp, Hist>,
        input: &[CQ15],
        output: &mut [CQ15],
    ) -> Result<Progress>
    where
        Tp: AsRef<[Q15]>,
        Hist: AsRef<[CQ15]> + AsMut<[CQ15]>,
    {
        let p = state.parts_mut();
        let ran = {
            let t: &[i16] = bytemuck::cast_slice(p.taps);
            let h: &[i16] = bytemuck::cast_slice(&*p.hist);
            let x: &[i16] = bytemuck::cast_slice(input);
            let y: &mut [i16] = bytemuck::cast_slice_mut(output);
            let (up, down) = (p.up, p.down);
            dispatch!(simd => polyphase_cq15(simd, t, h, x, y, up, down, &mut *p.q))?
        };
        // A filter whose branches do not fit the prepared storage runs on the fallback; nothing
        // of the state has been touched then.
        let Some((consumed, produced)) = ran else {
            return Fallback.resample(state, input, output);
        };
        commit(p.hist, input, consumed, produced, "resampler window")
    }
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
        self.resample(state.inner_mut(), input, output)
    }
}

impl FirInterpolate<CQ15> for Simd {
    fn fir_interpolate<Tp, Hist>(
        &self,
        state: &mut FirInterpState<CQ15, Tp, Hist>,
        input: &[CQ15],
        output: &mut [CQ15],
    ) -> Result<Progress>
    where
        Tp: AsRef<[Q15]>,
        Hist: AsRef<[CQ15]> + AsMut<[CQ15]>,
    {
        self.resample(state.inner_mut(), input, output)
    }
}
