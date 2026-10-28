//! `biquad` (`C32`, `CQ15`) and `biquad_real` (`f32`, `Q15`): a wavefront over the sections of
//! the cascade (decision D18).
//!
//! The sections of a cascade are in series, and each is a recursion: every output depends on
//! the previous output of the same section. Two consecutive sections still run side by side in
//! the lanes of one vector when the second one is one sample behind the first: at step `t` the
//! lanes of section `k0` filter the sample `t` and those of section `k0 + 1` the sample `t - 1`,
//! whose input is the output of section `k0` from step `t - 1`. Every lane does exactly the
//! operations of the fallback in the same order, so the result is the fallback's, bit for bit.
//! A call runs one step more than it has samples: the first step (the fill) leaves the state of
//! the second section unchanged, the last one (the drain) that of the first, with a lane mask,
//! so that the state after the call is the fallback's and the next call starts from it (the
//! result does not depend on how the input is split into calls).
//!
//! The cascade runs pair by pair over the whole buffer (the first pair reads the input, the
//! others filter the output in place); an odd last section runs alone (the two-lane path of
//! `C32` and `CQ15`, the scalar code of the fallback for `f32` and `Q15`). Padding a missing
//! section with a pass-through section would not do: `x + 0.0` turns `-0.0` into `+0.0`.
//!
//! - **`f32`:** one `f32x4` holds both sections, `[s1, s2]` of the first and of the second.
//!   A step is three products, three sums and the shuffles that bring the input and the
//!   output into place; the sum with the shifted `s2` adds `-0.0` in the lanes of the new `s2`,
//!   which leaves every value (also `-0.0`) unchanged.
//! - **`C32`:** one `f32x4` per state, `[re, im]` of the first and of the second section: the
//!   five products and four sums of the fallback per step, on four lanes.
//! - **`f32` flush:** both kinds flush on the schedule of [`FLUSH_PERIOD`]: section `k` after
//!   the sample `n` with `(n + k) % 16 == 15`. In a pair the section `k0 + 1` is at sample
//!   `t - 1` when the section `k0` is at `t`, so both flush after the same step: the loop runs in
//!   blocks of 16 steps between two flushes.
//! - **`Q15`, `CQ15` on x86 from SSE4.1 and on NEON:** the pairs run the integer arithmetic of
//!   the fallback itself in 64-bit integer lanes (the module `int`). Plain SSE2 runs the
//!   fallback for `Q15`.
//! - **`Q15`, `CQ15` elsewhere** (`CQ15` at SSE2, SIMD128, the portable level, and an odd last
//!   `CQ15` section everywhere): the integer arithmetic of the fallback is exact in `f64`: the
//!   products of two 16-bit values have at most 30 bits and five of them sum to less than 2^33, far
//!   below the 53 bits of the mantissa. The coefficients are scaled by 2^-14 (exact), so that the
//!   accumulator is `acc / 2^14`, and `round_shift(acc, 14) = floor(acc / 2^14 + 1/2)`. Where the
//!   level has a vector floor (SSE4.1, NEON, SIMD128) that is one addition and one `floor`; at SSE2
//!   and the portable level it is the round-to-nearest-even of `acc / 2^14 + 2^-15` (the addition
//!   and subtraction of 1.5 * 2^52), which is never on a tie: the value is
//!   `k + (r - 8191.5) / 2^14` for `acc + 2^13 = 2^14 k + r`. A round-to-nearest instruction on
//!   `acc / 2^14 + 1/2` would round the ties to even and is not used. The saturation is a minimum
//!   and a maximum. `Q15` holds the two sections in one `f64x2`, `CQ15` the four components
//!   (`[re, im]` of both) in one `f64x4`; one conversion from and to `i16` per sample serves both
//!   sections. Inside a block of sixteen steps a pair runs the transposed form of the same sum
//!   (`acc = b0 x + s1`, `s1 = (b1 x + s2) - a1 y`, `s2 = b2 x - a2 y`), which is exact as well
//!   and so gives the same accumulator, with two state vectors instead of four; the direct form I
//!   state is rebuilt from the last two steps of the block.

use fearless_simd::{
    Simd as SimdToken, f32x4, f64x2, f64x4, i16x8, i32x4, mask32x4, mask64x2, mask64x4, prelude::*,
};
#[cfg(target_arch = "x86_64")]
use sdrk_core::Fallback;
use sdrk_core::kernels::biquad::{
    Biquad, BiquadReal, BiquadRealState, BiquadSample, BiquadState, FLUSH_PERIOD,
};
use sdrk_core::num::{flush_denormal, round_shift, sat16};
use sdrk_core::{C32, CQ15, Q15, Result};

use crate::Simd;
use crate::level::dispatch_fn;
use crate::util::check_len;

/// The steps between two flushes.
const BLOCK: usize = 16;
const _: () = assert!(FLUSH_PERIOD == BLOCK);

/// The coefficient `i` of a section (zero outside, which never happens).
#[inline(always)]
fn coef<T: Copy + Default>(c: &[T; 5], i: usize) -> T {
    c.get(i).copied().unwrap_or_default()
}

/// Whether the flush is due after the step `t` of a call; `phase` is the phase of the state
/// plus the index of the (first) section, modulo the period.
#[inline(always)]
fn due(phase: usize, t: usize) -> bool {
    (phase + t) % BLOCK == BLOCK - 1
}

// The entries of the four kernels, one function per level (see `dispatch_fn!`): with the level
// detected at run time, `dispatch!` hands the kernel a closure over references, and the state of
// the sections then stays in memory instead of in registers (about a third more instructions per
// sample for `Q15` and `CQ15` at AVX2).
dispatch_fn!(
    fn run_f32(coefs: &[[f32; 5]], hist: &mut [f32], phase: usize, input: &[f32], out: &mut [f32])
        => cascade_f32
);
dispatch_fn!(
    fn run_c32(coefs: &[[f32; 5]], hist: &mut [C32], phase: usize, input: &[C32], out: &mut [C32])
        => cascade_c32
);
dispatch_fn!(
    fn run_q15(coefs: &[[i16; 5]], hist: &mut [i16], input: &[i16], out: &mut [i16])
        => cascade_q15
);
dispatch_fn!(
    fn run_cq15(
        coefs: &[[i16; 5]],
        hist: &mut [[i16; 2]],
        input: &[[i16; 2]],
        out: &mut [[i16; 2]],
    ) => cascade_cq15
);

// ---------------------------------------------------------------------------------------------
// The two drivers

/// Two consecutive sections in the lanes of one vector, the second one sample behind.
trait Pair {
    /// One sample.
    type T: Copy + Default;
    /// The first step of a call: the first section filters `x`, the second one keeps its
    /// state; both are flushed if `flush` (the second one was flushed after the same sample
    /// already, and the flush does not change a flushed value).
    fn fill(&mut self, x: Self::T, flush: bool);
    /// One step: the first section filters `x`, the second one the output of the first from
    /// the step before, which is returned.
    fn step(&mut self, x: Self::T) -> Self::T;
    /// The last step of a call: only the second section filters (and is flushed if `flush`);
    /// returns its output.
    fn drain(&mut self, flush: bool) -> Self::T;
    /// Sixteen steps and the flush after the last one: `x` the inputs of the first section,
    /// `o` the outputs of the second section, one sample earlier.
    fn block(&mut self, x: &[Self::T; BLOCK], o: &mut [Self::T; BLOCK]);
    /// The flush of both sections.
    fn flush(&mut self);
}

/// One section.
trait One {
    /// One sample.
    type T: Copy + Default;
    fn step(&mut self, x: Self::T) -> Self::T;
    /// Sixteen samples and the flush after the last one.
    fn block(&mut self, x: &[Self::T; BLOCK], o: &mut [Self::T; BLOCK]);
    fn flush(&mut self);
}

/// The sample `t` of the input of a pass: of `input` for the first pass, else of `io`.
#[inline(always)]
fn sample<T: Copy + Default>(input: Option<&[T]>, io: &[T], t: usize) -> T {
    input.unwrap_or(io).get(t).copied().unwrap_or_default()
}

/// One step of a pair at the step `t` of a pass, the output to the sample `t - 1` (`t >= 1`).
/// A function, not a closure: with the level detected at run time a closure is not compiled with
/// the target features of the level.
#[inline(always)]
fn pair_single<P: Pair>(p: &mut P, input: Option<&[P::T]>, io: &mut [P::T], t: usize) {
    let y = p.step(sample(input, io, t));
    if let Some(o) = io.get_mut(t.wrapping_sub(1)) {
        *o = y;
    }
}

/// One step of a single section at the sample `t` of a pass.
#[inline(always)]
fn one_single<O: One>(s: &mut O, input: Option<&[O::T]>, io: &mut [O::T], t: usize) {
    let y = s.step(sample(input, io, t));
    if let Some(o) = io.get_mut(t) {
        *o = y;
    }
}

/// One pass of a pair over the buffer: from `input` to `io`, or in place if `input` is `None`.
/// `phase` is the phase of the state plus the index of the first section of the pair.
#[inline(always)]
fn run_pair<P: Pair>(p: &mut P, phase: usize, input: Option<&[P::T]>, io: &mut [P::T]) {
    let n = io.len();
    if n == 0 {
        return;
    }
    p.fill(sample(input, io, 0), due(phase, 0));
    // The steps 1 ..= n - 1 have both sections; the first flush is after the step `lead`.
    let last = n - 1;
    let lead = BLOCK - (phase + 1) % BLOCK;
    if lead > last {
        for t in 1..=last {
            pair_single(p, input, io, t);
        }
    } else {
        for t in 1..=lead {
            pair_single(p, input, io, t);
        }
        p.flush();
        let mut t = lead + 1;
        while t + BLOCK <= n {
            match input {
                Some(input) => {
                    if let (Some(x), Some(o)) =
                        (input.get(t..t + BLOCK), io.get_mut(t - 1..t + BLOCK - 1))
                        && let (Ok(x), Ok(o)) = (
                            <&[P::T; BLOCK]>::try_from(x),
                            <&mut [P::T; BLOCK]>::try_from(o),
                        )
                    {
                        p.block(x, o);
                    }
                }
                None => {
                    // The inputs `t ..= t + 15` are read before the outputs `t - 1 ..= t + 14`
                    // are written.
                    if let Some(w) = io.get_mut(t - 1..t + BLOCK) {
                        let mut x = [P::T::default(); BLOCK];
                        if let Some(src) = w.get(1..) {
                            x.copy_from_slice(src);
                        }
                        if let Some(o) = w.get_mut(..BLOCK)
                            && let Ok(o) = <&mut [P::T; BLOCK]>::try_from(o)
                        {
                            p.block(&x, o);
                        }
                    }
                }
            }
            t += BLOCK;
        }
        for t in t..=last {
            pair_single(p, input, io, t);
        }
    }
    let y = p.drain(due(phase, n));
    if let Some(o) = io.get_mut(last) {
        *o = y;
    }
}

/// One pass of a single section over the buffer, as [`run_pair`]; `phase` is the phase of the
/// state plus the index of the section.
#[inline(always)]
fn run_one<O: One>(s: &mut O, phase: usize, input: Option<&[O::T]>, io: &mut [O::T]) {
    let n = io.len();
    // The first flush is after the sample `lead - 1`.
    let lead = BLOCK - phase % BLOCK;
    if lead > n {
        for t in 0..n {
            one_single(s, input, io, t);
        }
        return;
    }
    for t in 0..lead {
        one_single(s, input, io, t);
    }
    s.flush();
    let mut t = lead;
    while t + BLOCK <= n {
        let mut x = [O::T::default(); BLOCK];
        if let Some(src) = input.unwrap_or(io).get(t..t + BLOCK) {
            x.copy_from_slice(src);
        }
        if let Some(o) = io.get_mut(t..t + BLOCK)
            && let Ok(o) = <&mut [O::T; BLOCK]>::try_from(o)
        {
            s.block(&x, o);
        }
        t += BLOCK;
    }
    for t in t..n {
        one_single(s, input, io, t);
    }
}

/// A mask of the lanes `0 .. lanes / 2` (the first section of a pair) or of the others.
#[inline(always)]
fn first_half_f32<S: SimdToken>(simd: S, first: bool) -> mask32x4<S> {
    let idx = f32x4::simd_from(simd, [0.0, 0.0, 1.0, 1.0]);
    if first {
        idx.simd_lt(f32x4::splat(simd, 0.5))
    } else {
        idx.simd_gt(f32x4::splat(simd, 0.5))
    }
}

/// `flush_denormal` on every lane: the lanes below `1e-30` become `+0.0`, the others (NaN
/// included) stay. The mask is used as the bits it is: `select` would add a compare of the
/// mask with zero (`pcmpgtd`) to turn it into the representation of a boolean vector.
#[inline(always)]
fn flushed<S: SimdToken>(s: f32x4<S>, limit: f32x4<S>) -> f32x4<S> {
    let tiny = s.abs().simd_lt(limit).to_vector();
    (s.bitcast::<i32x4<S>>() & !tiny).bitcast()
}

// ---------------------------------------------------------------------------------------------
// f32

/// Transposed direct form II of one `f32` component without the flush: the fallback's code.
#[inline(always)]
fn df2t(c: &[f32; 5], s: &mut [f32; 2], x: f32) -> f32 {
    let [b0, b1, b2, a1, a2] = *c;
    let y = b0 * x + s[0];
    s[0] = (b1 * x + s[1]) - a1 * y;
    s[1] = b2 * x - a2 * y;
    y
}

/// Two `f32` sections A and B in one vector: `s = [s1 A, s2 A, s1 B, s2 B]`.
struct PairF32<S: SimdToken> {
    simd: S,
    /// `[b0, b0]` of A and of B.
    b0: f32x4<S>,
    /// `[b1, b2]` of A and of B.
    b12: f32x4<S>,
    /// `[a1, a2]` of A and of B.
    a12: f32x4<S>,
    limit: f32x4<S>,
    s: f32x4<S>,
    /// The outputs of the last step in the lanes 0 (A) and 2 (B).
    y: f32x4<S>,
}

impl<S: SimdToken> PairF32<S> {
    /// `h` holds the eight history entries of the two sections.
    #[inline(always)]
    fn new(simd: S, a: &[f32; 5], b: &[f32; 5], h: &[f32]) -> Self {
        let at = |i: usize| h.get(i).copied().unwrap_or(0.0);
        PairF32 {
            simd,
            b0: f32x4::simd_from(simd, [a[0], a[0], b[0], b[0]]),
            b12: f32x4::simd_from(simd, [a[1], a[2], b[1], b[2]]),
            a12: f32x4::simd_from(simd, [a[3], a[4], b[3], b[4]]),
            limit: f32x4::splat(simd, 1e-30),
            s: f32x4::simd_from(simd, [at(0), at(1), at(4), at(5)]),
            y: f32x4::splat(simd, 0.0),
        }
    }

    #[inline(always)]
    fn store(&self, h: &mut [f32]) {
        let [s1a, s2a, s1b, s2b] = *self.s;
        for (i, v) in [(0, s1a), (1, s2a), (4, s1b), (5, s2b)] {
            if let Some(d) = h.get_mut(i) {
                *d = v;
            }
        }
    }

    /// The outputs and the new state for the inputs `xv = [x A, x A, x B, x B]`.
    #[inline(always)]
    fn next(&self, xv: f32x4<S>) -> (f32x4<S>, f32x4<S>) {
        // y = b0 x + s1
        let y = xv * self.b0 + self.s;
        let [ya, _, yb, _] = *y;
        let yd = f32x4::simd_from(self.simd, [ya, ya, yb, yb]);
        // s1 = (b1 x + s2) - a1 y; s2 = (b2 x + -0.0) - a2 y
        let [_, s2a, _, s2b] = *self.s;
        let sh = f32x4::simd_from(self.simd, [s2a, -0.0, s2b, -0.0]);
        let s = (xv * self.b12 + sh) - yd * self.a12;
        (y, s)
    }

    #[inline(always)]
    fn input(&self, x: f32) -> f32x4<S> {
        let [yp, ..] = *self.y;
        f32x4::simd_from(self.simd, [x, x, yp, yp])
    }
}

impl<S: SimdToken> Pair for PairF32<S> {
    type T = f32;

    #[inline(always)]
    fn fill(&mut self, x: f32, flush: bool) {
        let (y, mut s) = self.next(f32x4::simd_from(self.simd, [x, x, 0.0, 0.0]));
        if flush {
            s = flushed(s, self.limit);
        }
        self.s = first_half_f32(self.simd, true).select(s, self.s);
        self.y = y;
    }

    #[inline(always)]
    fn step(&mut self, x: f32) -> f32 {
        let (y, s) = self.next(self.input(x));
        self.s = s;
        self.y = y;
        let [_, _, yb, _] = *y;
        yb
    }

    #[inline(always)]
    fn drain(&mut self, flush: bool) -> f32 {
        let (y, mut s) = self.next(self.input(0.0));
        if flush {
            s = flushed(s, self.limit);
        }
        self.s = first_half_f32(self.simd, false).select(s, self.s);
        let [_, _, yb, _] = *y;
        yb
    }

    #[inline(always)]
    fn block(&mut self, x: &[f32; BLOCK], o: &mut [f32; BLOCK]) {
        let (xs, _) = x.as_chunks::<4>();
        let (os, _) = o.as_chunks_mut::<4>();
        for (x, o) in xs.iter().zip(os) {
            let [a, b, c, d] = *x;
            let out = [self.step(a), self.step(b), self.step(c), self.step(d)];
            *o = *f32x4::simd_from(self.simd, out);
        }
        self.flush();
    }

    #[inline(always)]
    fn flush(&mut self) {
        self.s = flushed(self.s, self.limit);
    }
}

/// One `f32` section: the fallback's scalar code.
struct OneF32 {
    c: [f32; 5],
    s: [f32; 2],
}

impl One for OneF32 {
    type T = f32;

    #[inline(always)]
    fn step(&mut self, x: f32) -> f32 {
        df2t(&self.c, &mut self.s, x)
    }

    #[inline(always)]
    fn block(&mut self, x: &[f32; BLOCK], o: &mut [f32; BLOCK]) {
        for (o, x) in o.iter_mut().zip(x) {
            *o = df2t(&self.c, &mut self.s, *x);
        }
        self.flush();
    }

    #[inline(always)]
    fn flush(&mut self) {
        self.s = self.s.map(flush_denormal);
    }
}

#[inline(always)]
fn cascade_f32<S: SimdToken>(
    simd: S,
    coefs: &[[f32; 5]],
    hist: &mut [f32],
    phase: usize,
    input: &[f32],
    out: &mut [f32],
) {
    for (j, (c, h)) in coefs.chunks(2).zip(hist.chunks_mut(8)).enumerate() {
        let ph = (phase + 2 * j) % BLOCK;
        let src = (j == 0).then_some(input);
        match c {
            [a, b] => {
                let mut p = PairF32::new(simd, a, b, h);
                run_pair(&mut p, ph, src, out);
                p.store(h);
            }
            [a] => {
                let s = [h.first(), h.get(1)].map(|v| v.copied().unwrap_or(0.0));
                let mut one = OneF32 { c: *a, s };
                run_one(&mut one, ph, src, out);
                for (d, v) in h.iter_mut().zip(one.s) {
                    *d = v;
                }
            }
            _ => {}
        }
    }
}

impl BiquadReal<f32> for Simd {
    fn biquad_real<C, H>(
        &self,
        s: &mut BiquadRealState<f32, C, H>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()>
    where
        C: AsRef<[[<f32 as BiquadSample>::Coef; 5]]>,
        H: AsRef<[f32]> + AsMut<[f32]>,
    {
        check_len(input.len(), output.len())?;
        let (coefs, hist, phase) = s.parts_phase_mut();
        run_f32(coefs, hist, phase, input, output);
        s.advance_phase(input.len());
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// C32

/// Two `C32` sections A and B: every vector is `[re A, im A, re B, im B]`.
struct PairC32<S: SimdToken> {
    simd: S,
    b0: f32x4<S>,
    b1: f32x4<S>,
    b2: f32x4<S>,
    a1: f32x4<S>,
    a2: f32x4<S>,
    limit: f32x4<S>,
    s1: f32x4<S>,
    s2: f32x4<S>,
    /// The outputs of the last step.
    y: f32x4<S>,
}

impl<S: SimdToken> PairC32<S> {
    /// `h` holds the eight history entries of the two sections.
    #[inline(always)]
    fn new(simd: S, a: &[f32; 5], b: &[f32; 5], h: &[C32]) -> Self {
        let at = |i: usize| h.get(i).copied().unwrap_or_default();
        let c = |i: usize| {
            let (u, v) = (coef(a, i), coef(b, i));
            f32x4::simd_from(simd, [u, u, v, v])
        };
        let pair = |i: usize, j: usize| {
            let (u, v) = (at(i), at(j));
            f32x4::simd_from(simd, [u.re, u.im, v.re, v.im])
        };
        PairC32 {
            simd,
            b0: c(0),
            b1: c(1),
            b2: c(2),
            a1: c(3),
            a2: c(4),
            limit: f32x4::splat(simd, 1e-30),
            s1: pair(0, 4),
            s2: pair(1, 5),
            y: f32x4::splat(simd, 0.0),
        }
    }

    #[inline(always)]
    fn store(&self, h: &mut [C32]) {
        let [r1a, i1a, r1b, i1b] = *self.s1;
        let [r2a, i2a, r2b, i2b] = *self.s2;
        for (i, re, im) in [(0, r1a, i1a), (1, r2a, i2a), (4, r1b, i1b), (5, r2b, i2b)] {
            if let Some(d) = h.get_mut(i) {
                *d = C32::new(re, im);
            }
        }
    }

    /// The outputs and the new states for the inputs `xv`.
    #[inline(always)]
    fn next(&self, xv: f32x4<S>) -> (f32x4<S>, f32x4<S>, f32x4<S>) {
        let y = xv * self.b0 + self.s1;
        let s1 = (xv * self.b1 + self.s2) - self.a1 * y;
        let s2 = xv * self.b2 - self.a2 * y;
        (y, s1, s2)
    }

    #[inline(always)]
    fn input(&self, re: f32, im: f32) -> f32x4<S> {
        let [ya, yb, _, _] = *self.y;
        f32x4::simd_from(self.simd, [re, im, ya, yb])
    }

    #[inline(always)]
    fn masked(&mut self, xv: f32x4<S>, flush: bool, first: bool) {
        let (y, mut s1, mut s2) = self.next(xv);
        if flush {
            s1 = flushed(s1, self.limit);
            s2 = flushed(s2, self.limit);
        }
        let m = first_half_f32(self.simd, first);
        self.s1 = m.select(s1, self.s1);
        self.s2 = m.select(s2, self.s2);
        self.y = y;
    }
}

impl<S: SimdToken> Pair for PairC32<S> {
    type T = C32;

    #[inline(always)]
    fn fill(&mut self, x: C32, flush: bool) {
        self.masked(
            f32x4::simd_from(self.simd, [x.re, x.im, 0.0, 0.0]),
            flush,
            true,
        );
    }

    #[inline(always)]
    fn step(&mut self, x: C32) -> C32 {
        let (y, s1, s2) = self.next(self.input(x.re, x.im));
        (self.s1, self.s2, self.y) = (s1, s2, y);
        let [_, _, re, im] = *y;
        C32::new(re, im)
    }

    #[inline(always)]
    fn drain(&mut self, flush: bool) -> C32 {
        self.masked(self.input(0.0, 0.0), flush, false);
        let [_, _, re, im] = *self.y;
        C32::new(re, im)
    }

    #[inline(always)]
    fn block(&mut self, x: &[C32; BLOCK], o: &mut [C32; BLOCK]) {
        let x: &[f32; 2 * BLOCK] = bytemuck::cast_ref(x);
        let o: &mut [f32; 2 * BLOCK] = bytemuck::cast_mut(o);
        let (xs, _) = x.as_chunks::<4>();
        let (os, _) = o.as_chunks_mut::<4>();
        for (x, o) in xs.iter().zip(os) {
            let [ar, ai, br, bi] = *x;
            let a = self.step(C32::new(ar, ai));
            let b = self.step(C32::new(br, bi));
            *o = *f32x4::simd_from(self.simd, [a.re, a.im, b.re, b.im]);
        }
        self.flush();
    }

    #[inline(always)]
    fn flush(&mut self) {
        self.s1 = flushed(self.s1, self.limit);
        self.s2 = flushed(self.s2, self.limit);
    }
}

/// One `C32` section in one vector, `s = [s1.re, s1.im, s2.re, s2.im]`: the step is the sum
/// for `y`, the duplicate of `y`, the products with `[b1, b1, b2, b2]` and `[a1, a1, a2, a2]`,
/// the sum with the shifted `s2` (and `-0.0` in the lanes of the new `s2`) and the difference.
struct OneC32<S: SimdToken> {
    simd: S,
    b0: f32x4<S>,
    /// `[b1, b1, b2, b2]`.
    b12: f32x4<S>,
    /// `[a1, a1, a2, a2]`.
    a: f32x4<S>,
    limit: f32x4<S>,
    s: f32x4<S>,
}

impl<S: SimdToken> OneC32<S> {
    #[inline(always)]
    fn new(simd: S, c: &[f32; 5], h: &[C32]) -> Self {
        let [b0, b1, b2, a1, a2] = *c;
        let at = |i: usize| h.get(i).copied().unwrap_or_default();
        let (s1, s2) = (at(0), at(1));
        OneC32 {
            simd,
            b0: f32x4::splat(simd, b0),
            b12: f32x4::simd_from(simd, [b1, b1, b2, b2]),
            a: f32x4::simd_from(simd, [a1, a1, a2, a2]),
            limit: f32x4::splat(simd, 1e-30),
            s: f32x4::simd_from(simd, [s1.re, s1.im, s2.re, s2.im]),
        }
    }

    #[inline(always)]
    fn store(&self, h: &mut [C32]) {
        let [r1, i1, r2, i2] = *self.s;
        if let [h1, h2, ..] = h {
            (*h1, *h2) = (C32::new(r1, i1), C32::new(r2, i2));
        }
    }

    /// One sample; returns the vector with the output in the lanes 0 and 1.
    #[inline(always)]
    fn next(&mut self, re: f32, im: f32) -> f32x4<S> {
        let d = f32x4::simd_from(self.simd, [re, im, re, im]);
        let y = d * self.b0 + self.s;
        let [yr, yi, _, _] = *y;
        let yd = f32x4::simd_from(self.simd, [yr, yi, yr, yi]);
        let [_, _, s2r, s2i] = *self.s;
        let sh = f32x4::simd_from(self.simd, [s2r, s2i, -0.0, -0.0]);
        self.s = (d * self.b12 + sh) - self.a * yd;
        y
    }
}

impl<S: SimdToken> One for OneC32<S> {
    type T = C32;

    #[inline(always)]
    fn step(&mut self, x: C32) -> C32 {
        let y = self.next(x.re, x.im);
        let [re, im, _, _] = *y;
        C32::new(re, im)
    }

    #[inline(always)]
    fn block(&mut self, x: &[C32; BLOCK], o: &mut [C32; BLOCK]) {
        let x: &[f32; 2 * BLOCK] = bytemuck::cast_ref(x);
        let o: &mut [f32; 2 * BLOCK] = bytemuck::cast_mut(o);
        let (xs, _) = x.as_chunks::<4>();
        let (os, _) = o.as_chunks_mut::<4>();
        for (x, o) in xs.iter().zip(os) {
            let [ar, ai, br, bi] = *x;
            let [a0, a1, _, _] = *self.next(ar, ai);
            let [b0, b1, _, _] = *self.next(br, bi);
            *o = [a0, a1, b0, b1];
        }
        self.flush();
    }

    #[inline(always)]
    fn flush(&mut self) {
        self.s = flushed(self.s, self.limit);
    }
}

#[inline(always)]
fn cascade_c32<S: SimdToken>(
    simd: S,
    coefs: &[[f32; 5]],
    hist: &mut [C32],
    phase: usize,
    input: &[C32],
    out: &mut [C32],
) {
    for (j, (c, h)) in coefs.chunks(2).zip(hist.chunks_mut(8)).enumerate() {
        let ph = (phase + 2 * j) % BLOCK;
        let src = (j == 0).then_some(input);
        match c {
            [a, b] => {
                let mut p = PairC32::new(simd, a, b, h);
                run_pair(&mut p, ph, src, out);
                p.store(h);
            }
            [a] => {
                let mut one = OneC32::new(simd, a, h);
                run_one(&mut one, ph, src, out);
                one.store(h);
            }
            _ => {}
        }
    }
}

impl Biquad<C32> for Simd {
    fn biquad<C, H>(
        &self,
        s: &mut BiquadState<C32, C, H>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()>
    where
        C: AsRef<[[<C32 as BiquadSample>::Coef; 5]]>,
        H: AsRef<[C32]> + AsMut<[C32]>,
    {
        check_len(input.len(), output.len())?;
        let (coefs, hist, phase) = s.parts_phase_mut();
        run_c32(coefs, hist, phase, input, output);
        s.advance_phase(input.len());
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Q15 and CQ15

/// `1.5 * 2^52`: adding and subtracting it rounds an `f64` of smaller magnitude to the nearest
/// integer, ties to even.
const MAGIC: f64 = 6_755_399_441_055_744.0;
/// `2^-14`: the scale of the coefficients.
const SCALE: f64 = 1.0 / 16_384.0;

/// Whether the level has a vector `floor` of `f64` (SSE4.1, NEON, SIMD128); SSE2 and the
/// portable level round with [`MAGIC`] instead.
#[inline(always)]
fn has_floor<S: SimdToken>(simd: S) -> bool {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        level.as_sse4_2().is_some()
    }
    #[cfg(target_arch = "aarch64")]
    {
        level.as_neon().is_some()
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        level.as_wasm_simd128().is_some()
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        let _ = level;
        false
    }
}

/// The rounding constants of [`has_floor`]: the half added before the rounding.
#[inline(always)]
fn half<S: SimdToken>(simd: S) -> f64 {
    if has_floor(simd) { 0.5 } else { 1.0 / 32_768.0 }
}

/// An integer-valued `f64` within the `i16` range as `i16`.
#[inline(always)]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the value is an integer in range"
)]
fn to_i16(v: f64) -> i16 {
    v as i16
}

/// The coefficients of a section scaled by 2^-14.
#[inline(always)]
fn scaled(c: &[i16; 5]) -> [f64; 5] {
    c.map(|v| f64::from(v) * SCALE)
}

macro_rules! q15_arith {
    ($name:ident, $v:ident) => {
        /// The arithmetic of the `Q15` sections on one vector type.
        struct $name<S: SimdToken> {
            simd: S,
            half: $v<S>,
            magic: $v<S>,
            hi: $v<S>,
            lo: $v<S>,
        }

        impl<S: SimdToken> $name<S> {
            #[inline(always)]
            fn new(simd: S) -> Self {
                $name {
                    simd,
                    half: $v::splat(simd, half(simd)),
                    magic: $v::splat(simd, MAGIC),
                    hi: $v::splat(simd, 32767.0),
                    lo: $v::splat(simd, -32768.0),
                }
            }

            /// `sat16(round_shift(acc, 14))` of `acc / 2^14`.
            #[inline(always)]
            fn round(&self, v: $v<S>) -> $v<S> {
                let r = if has_floor(self.simd) {
                    (v + self.half).floor()
                } else {
                    ((v + self.half) + self.magic) - self.magic
                };
                r.min(self.hi).max(self.lo)
            }
        }
    };
}

q15_arith!(Arith2, f64x2);
q15_arith!(Arith4, f64x4);

/// The direct form I state of the lanes of one vector type: the scaled coefficients and the
/// last two inputs and outputs.
macro_rules! df1_state {
    ($name:ident, $v:ident, $arith:ident) => {
        struct $name<S: SimdToken> {
            ar: $arith<S>,
            b0: $v<S>,
            b1: $v<S>,
            b2: $v<S>,
            a1: $v<S>,
            a2: $v<S>,
            x1: $v<S>,
            x2: $v<S>,
            y1: $v<S>,
            y2: $v<S>,
        }

        impl<S: SimdToken> $name<S> {
            /// The output for the input `x` (the state is not changed).
            #[inline(always)]
            fn out(&self, x: $v<S>) -> $v<S> {
                let acc = x * self.b0 + self.x1 * self.b1 + self.x2 * self.b2
                    - self.y1 * self.a1
                    - self.y2 * self.a2;
                self.ar.round(acc)
            }

            /// One step: returns the output.
            #[inline(always)]
            fn step(&mut self, x: $v<S>) -> $v<S> {
                let y = self.out(x);
                (self.x2, self.x1, self.y2, self.y1) = (self.x1, x, self.y1, y);
                y
            }

            /// The state in the transposed form `(s1, s2)` that [`step_t`](Self::step_t)
            /// runs: `s1 = b1 x1 + b2 x2 - a1 y1 - a2 y2` and `s2 = b2 x1 - a2 y1`. Every
            /// product and sum is exact in `f64` (see the module description), so both forms
            /// give the same accumulator and the same output.
            #[inline(always)]
            fn transposed(&self) -> ($v<S>, $v<S>) {
                let s1 =
                    self.x1 * self.b1 + self.x2 * self.b2 - self.y1 * self.a1 - self.y2 * self.a2;
                let s2 = self.x1 * self.b2 - self.y1 * self.a2;
                (s1, s2)
            }

            /// One step in the transposed form: two state vectors instead of four, which
            /// leaves registers for the coefficients and the constants.
            #[inline(always)]
            fn step_t(&self, s: &mut ($v<S>, $v<S>), x: $v<S>) -> $v<S> {
                let y = self.ar.round(x * self.b0 + s.0);
                s.0 = (x * self.b1 + s.1) - y * self.a1;
                s.1 = x * self.b2 - y * self.a2;
                y
            }

            /// One step that changes the state of the lanes of `m` only.
            #[inline(always)]
            fn masked(&mut self, x: $v<S>, m: <$v<S> as SimdBase<S>>::Mask) -> $v<S> {
                let y = self.out(x);
                self.x2 = m.select(self.x1, self.x2);
                self.x1 = m.select(x, self.x1);
                self.y2 = m.select(self.y1, self.y2);
                self.y1 = m.select(y, self.y1);
                y
            }
        }
    };
}

df1_state!(Df1x2, f64x2, Arith2);
df1_state!(Df1x4, f64x4, Arith4);

/// The `f64` of the four `i16` values `x` (a widened half of an `i16x8`).
#[inline(always)]
fn to_f64<S: SimdToken>(x: i32x4<S>) -> (f64x2<S>, f64x2<S>) {
    f32x4::float_from(x).widen()
}

/// The eight `i16` of four pairs of integer-valued `f64` within range.
#[inline(always)]
fn to_i16x8<S: SimdToken>(v: [f64x2<S>; 4]) -> i16x8<S> {
    let [a, b, c, d] = v;
    let lo = i32x4::truncate_from(a.narrow(b));
    let hi = i32x4::truncate_from(c.narrow(d));
    lo.saturating_narrow(hi)
}

/// The input and the output vector of one step of a `Q15` pair.
type Io2<S> = (f64x2<S>, f64x2<S>);
/// The input and the output vector of one step of a `CQ15` pair.
type Io4<S> = (f64x4<S>, f64x4<S>);

/// Two `Q15` sections A and B in the lanes 0 and 1 of an `f64x2`.
struct PairQ15<S: SimdToken> {
    simd: S,
    d: Df1x2<S>,
}

impl<S: SimdToken> PairQ15<S> {
    /// `h` holds the eight history entries `[x1, x2, y1, y2]` of A and of B.
    #[inline(always)]
    fn new(simd: S, a: &[i16; 5], b: &[i16; 5], h: &[i16]) -> Self {
        let (ca, cb) = (scaled(a), scaled(b));
        let c = |i: usize| f64x2::simd_from(simd, [coef(&ca, i), coef(&cb, i)]);
        let at = |i: usize| f64::from(h.get(i).copied().unwrap_or(0));
        let st = |i: usize| f64x2::simd_from(simd, [at(i), at(i + 4)]);
        PairQ15 {
            simd,
            d: Df1x2 {
                ar: Arith2::new(simd),
                b0: c(0),
                b1: c(1),
                b2: c(2),
                a1: c(3),
                a2: c(4),
                x1: st(0),
                x2: st(1),
                y1: st(2),
                y2: st(3),
            },
        }
    }

    #[inline(always)]
    fn store(&self, h: &mut [i16]) {
        let d = &self.d;
        for (i, v) in [d.x1, d.x2, d.y1, d.y2].into_iter().enumerate() {
            let [a, b] = *v;
            for (j, w) in [(i, a), (i + 4, b)] {
                if let Some(o) = h.get_mut(j) {
                    *o = to_i16(w);
                }
            }
        }
    }

    /// One step in the transposed form: `x` the input of the first section, `ya` the output of
    /// the first section from the step before (updated); returns the input vector and the
    /// output vector of the step.
    #[inline(always)]
    fn step_t(&self, s: &mut (f64x2<S>, f64x2<S>), ya: &mut f64, x: f64) -> (f64x2<S>, f64x2<S>) {
        let xv = f64x2::simd_from(self.simd, [x, *ya]);
        let y = self.d.step_t(s, xv);
        let [a, _] = *y;
        *ya = a;
        (xv, y)
    }

    /// Two steps on the inputs `p = [x0, x1]`: the two outputs of the second section, and the
    /// input and output vectors of both steps.
    #[inline(always)]
    fn two_t(
        &self,
        s: &mut (f64x2<S>, f64x2<S>),
        ya: &mut f64,
        p: f64x2<S>,
    ) -> (f64x2<S>, [Io2<S>; 2]) {
        let [x0, x1] = *p;
        let a = self.step_t(s, ya, x0);
        let b = self.step_t(s, ya, x1);
        let ([_, ya], [_, yb]) = (*a.1, *b.1);
        (f64x2::simd_from(self.simd, [ya, yb]), [a, b])
    }

    /// Eight steps in the transposed form; returns the input and output vectors of the last
    /// two steps.
    #[inline(always)]
    fn eight(
        &self,
        s: &mut (f64x2<S>, f64x2<S>),
        ya: &mut f64,
        x: &[i16; 8],
        o: &mut [i16; 8],
    ) -> (Io2<S>, Io2<S>) {
        let (lo, hi) = i16x8::simd_from(self.simd, *x).widen();
        let (p0, p1) = to_f64(lo);
        let (p2, p3) = to_f64(hi);
        let (r0, _) = self.two_t(s, ya, p0);
        let (r1, _) = self.two_t(s, ya, p1);
        let (r2, _) = self.two_t(s, ya, p2);
        let (r3, [prev, last]) = self.two_t(s, ya, p3);
        *o = *to_i16x8([r0, r1, r2, r3]);
        (prev, last)
    }

    #[inline(always)]
    fn input(&self, x: f64) -> f64x2<S> {
        let [ya, _] = *self.d.y1;
        f64x2::simd_from(self.simd, [x, ya])
    }

    #[inline(always)]
    fn mask(&self, first: bool) -> mask64x2<S> {
        let idx = f64x2::simd_from(self.simd, [0.0, 1.0]);
        let half = f64x2::splat(self.simd, 0.5);
        if first {
            idx.simd_lt(half)
        } else {
            idx.simd_gt(half)
        }
    }
}

impl<S: SimdToken> Pair for PairQ15<S> {
    type T = i16;

    #[inline(always)]
    fn fill(&mut self, x: i16, _flush: bool) {
        let xv = f64x2::simd_from(self.simd, [f64::from(x), 0.0]);
        let m = self.mask(true);
        self.d.masked(xv, m);
    }

    #[inline(always)]
    fn step(&mut self, x: i16) -> i16 {
        let [_, yb] = *self.d.step(self.input(f64::from(x)));
        to_i16(yb)
    }

    #[inline(always)]
    fn drain(&mut self, _flush: bool) -> i16 {
        let xv = self.input(0.0);
        let m = self.mask(false);
        let [_, yb] = *self.d.masked(xv, m);
        to_i16(yb)
    }

    #[inline(always)]
    fn block(&mut self, x: &[i16; BLOCK], o: &mut [i16; BLOCK]) {
        // The sixteen steps run in the transposed form; the direct form I state is rebuilt
        // from the inputs and outputs of the last two steps. Written out with methods, not
        // closures or loops over arrays: with the level detected at run time a closure is not
        // compiled with the target features of the level, and a loop would carry the last
        // values in registers through every iteration.
        let mut s = self.d.transposed();
        let [mut ya, _] = *self.d.y1;
        let ((x0, x1), (o0, o1)) = (x.split_at(8), o.split_at_mut(8));
        let (Ok(x0), Ok(x1), Ok(o0), Ok(o1)) = (
            <&[i16; 8]>::try_from(x0),
            <&[i16; 8]>::try_from(x1),
            <&mut [i16; 8]>::try_from(o0),
            <&mut [i16; 8]>::try_from(o1),
        ) else {
            return;
        };
        let _ = self.eight(&mut s, &mut ya, x0, o0);
        let ((xv2, y2), (xv1, y1)) = self.eight(&mut s, &mut ya, x1, o1);
        (self.d.x1, self.d.x2, self.d.y1, self.d.y2) = (xv1, xv2, y1, y2);
    }

    #[inline(always)]
    fn flush(&mut self) {}
}

/// One `Q15` section: the fallback's integer code.
struct OneQ15 {
    c: [i64; 5],
    /// `[x1, x2, y1, y2]`.
    h: [i64; 4],
}

impl OneQ15 {
    #[inline(always)]
    fn df1(&mut self, x: i16) -> i16 {
        let [b0, b1, b2, a1, a2] = self.c;
        let [x1, x2, y1, y2] = self.h;
        let x = i64::from(x);
        let acc = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        let y = sat16(round_shift(acc, 14));
        self.h = [x, x1, i64::from(y), y1];
        y
    }
}

impl One for OneQ15 {
    type T = i16;

    #[inline(always)]
    fn step(&mut self, x: i16) -> i16 {
        self.df1(x)
    }

    #[inline(always)]
    fn block(&mut self, x: &[i16; BLOCK], o: &mut [i16; BLOCK]) {
        for (o, x) in o.iter_mut().zip(x) {
            *o = self.df1(*x);
        }
    }

    #[inline(always)]
    fn flush(&mut self) {}
}

/// The state of a `Q15` section as `i64`.
#[inline(always)]
fn load_q15(h: &[i16]) -> [i64; 4] {
    let at = |i: usize| i64::from(h.get(i).copied().unwrap_or(0));
    [at(0), at(1), at(2), at(3)]
}

#[inline(always)]
#[allow(clippy::cast_possible_truncation, reason = "the values are within i16")]
fn store_q15(state: &[i64; 4], h: &mut [i16]) {
    for (d, v) in h.iter_mut().zip(state) {
        *d = *v as i16;
    }
}

#[inline(always)]
fn cascade_q15<S: SimdToken>(
    simd: S,
    coefs: &[[i16; 5]],
    hist: &mut [i16],
    input: &[i16],
    out: &mut [i16],
) {
    #[cfg(target_arch = "x86_64")]
    {
        // The 128-bit lanes at every level; the saturation differs with and without AVX.
        let level = simd.level();
        if let Some(t) = level.as_sse4_2() {
            return if level.as_avx2().is_some() {
                int::cascade_q15::<core::arch::x86_64::__m128i>(t, coefs, hist, input, out);
            } else {
                int::cascade_q15::<int::x86::Sse>(t, coefs, hist, input, out);
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = simd.level().as_neon() {
        return int::cascade_q15::<core::arch::aarch64::int32x2_t>(t, coefs, hist, input, out);
    }
    for (j, (c, h)) in coefs.chunks(2).zip(hist.chunks_mut(8)).enumerate() {
        let src = (j == 0).then_some(input);
        match c {
            [a, b] => {
                let mut p = PairQ15::new(simd, a, b, h);
                run_pair(&mut p, 0, src, out);
                p.store(h);
            }
            [a] => {
                let mut one = OneQ15 {
                    c: a.map(i64::from),
                    h: load_q15(h),
                };
                run_one(&mut one, 0, src, out);
                store_q15(&one.h, h);
            }
            _ => {}
        }
    }
}

impl BiquadReal<Q15> for Simd {
    fn biquad_real<C, H>(
        &self,
        s: &mut BiquadRealState<Q15, C, H>,
        input: &[Q15],
        output: &mut [Q15],
    ) -> Result<()>
    where
        C: AsRef<[[<Q15 as BiquadSample>::Coef; 5]]>,
        H: AsRef<[Q15]> + AsMut<[Q15]>,
    {
        // Plain SSE2 has neither `pmuldq` nor `pminsd` (SSE4.1) for the integer lanes, and its
        // `f64` path is slower than the fallback (`llvm-mca`, cycles per sample: 33.7 against
        // 15.1 on Skylake, 23.0 against 15.1 on Zen 2), so it runs the fallback.
        #[cfg(target_arch = "x86_64")]
        if crate::level::sse2_only(crate::level::current()).is_some() {
            return Fallback.biquad_real(s, input, output);
        }
        check_len(input.len(), output.len())?;
        let (coefs, hist) = s.parts_mut();
        let hist: &mut [i16] = bytemuck::cast_slice_mut(hist);
        let x: &[i16] = bytemuck::cast_slice(input);
        let y: &mut [i16] = bytemuck::cast_slice_mut(output);
        run_q15(coefs, hist, x, y);
        Ok(())
    }
}

/// Two `CQ15` sections A and B: `[re A, im A, re B, im B]` in an `f64x4`.
struct PairCq15<S: SimdToken> {
    simd: S,
    d: Df1x4<S>,
}

impl<S: SimdToken> PairCq15<S> {
    /// `h` holds the sixteen history values `[x1, x2, y1, y2]` (`[re, im]` each) of A and B.
    #[inline(always)]
    fn new(simd: S, a: &[i16; 5], b: &[i16; 5], h: &[[i16; 2]]) -> Self {
        let (ca, cb) = (scaled(a), scaled(b));
        let c = |i: usize| {
            let (u, v) = (coef(&ca, i), coef(&cb, i));
            f64x4::simd_from(simd, [u, u, v, v])
        };
        let at = |i: usize| h.get(i).copied().unwrap_or_default().map(f64::from);
        let st = |i: usize| {
            let ([ra, ia], [rb, ib]) = (at(i), at(i + 4));
            f64x4::simd_from(simd, [ra, ia, rb, ib])
        };
        PairCq15 {
            simd,
            d: Df1x4 {
                ar: Arith4::new(simd),
                b0: c(0),
                b1: c(1),
                b2: c(2),
                a1: c(3),
                a2: c(4),
                x1: st(0),
                x2: st(1),
                y1: st(2),
                y2: st(3),
            },
        }
    }

    #[inline(always)]
    fn store(&self, h: &mut [[i16; 2]]) {
        let d = &self.d;
        for (i, v) in [d.x1, d.x2, d.y1, d.y2].into_iter().enumerate() {
            let [ra, ia, rb, ib] = *v;
            for (j, re, im) in [(i, ra, ia), (i + 4, rb, ib)] {
                if let Some(o) = h.get_mut(j) {
                    *o = [to_i16(re), to_i16(im)];
                }
            }
        }
    }

    /// One step in the transposed form: `p = [re, im]` the input of the first section, `ya`
    /// the output of the first section from the step before (updated); returns the input and
    /// the output vector of the step.
    #[inline(always)]
    fn step_t(
        &self,
        s: &mut (f64x4<S>, f64x4<S>),
        ya: &mut f64x2<S>,
        p: f64x2<S>,
    ) -> (f64x4<S>, f64x4<S>) {
        let xv = p.combine(*ya);
        let y = self.d.step_t(s, xv);
        *ya = y.split().0;
        (xv, y)
    }

    /// Four steps (eight `i16`) in the transposed form; returns the input and output vectors
    /// of the last two steps.
    #[inline(always)]
    fn four(
        &self,
        s: &mut (f64x4<S>, f64x4<S>),
        ya: &mut f64x2<S>,
        x: &[i16; 8],
        o: &mut [i16; 8],
    ) -> (Io4<S>, Io4<S>) {
        let (lo, hi) = i16x8::simd_from(self.simd, *x).widen();
        let (p0, p1) = to_f64(lo);
        let (p2, p3) = to_f64(hi);
        let a = self.step_t(s, ya, p0);
        let b = self.step_t(s, ya, p1);
        let c = self.step_t(s, ya, p2);
        let d = self.step_t(s, ya, p3);
        *o = *to_i16x8([a.1.split().1, b.1.split().1, c.1.split().1, d.1.split().1]);
        (c, d)
    }

    /// The input vector: `x` for A, the last output of A for B.
    #[inline(always)]
    fn input(&self, x: f64x2<S>) -> f64x4<S> {
        let (ya, _) = self.d.y1.split();
        x.combine(ya)
    }

    #[inline(always)]
    fn mask(&self, first: bool) -> mask64x4<S> {
        let idx = f64x4::simd_from(self.simd, [0.0, 0.0, 1.0, 1.0]);
        let half = f64x4::splat(self.simd, 0.5);
        if first {
            idx.simd_lt(half)
        } else {
            idx.simd_gt(half)
        }
    }

    #[inline(always)]
    fn pair(&self, x: [i16; 2]) -> f64x2<S> {
        f64x2::simd_from(self.simd, x.map(f64::from))
    }
}

impl<S: SimdToken> Pair for PairCq15<S> {
    type T = [i16; 2];

    #[inline(always)]
    fn fill(&mut self, x: [i16; 2], _flush: bool) {
        let xv = self.pair(x).combine(f64x2::splat(self.simd, 0.0));
        let m = self.mask(true);
        self.d.masked(xv, m);
    }

    #[inline(always)]
    fn step(&mut self, x: [i16; 2]) -> [i16; 2] {
        let y = self.d.step(self.input(self.pair(x)));
        let [_, _, re, im] = *y;
        [to_i16(re), to_i16(im)]
    }

    #[inline(always)]
    fn drain(&mut self, _flush: bool) -> [i16; 2] {
        let xv = self.input(f64x2::splat(self.simd, 0.0));
        let m = self.mask(false);
        let y = self.d.masked(xv, m);
        let [_, _, re, im] = *y;
        [to_i16(re), to_i16(im)]
    }

    #[inline(always)]
    fn block(&mut self, x: &[[i16; 2]; BLOCK], o: &mut [[i16; 2]; BLOCK]) {
        // As for `PairQ15`: the transposed form, written out in four steps of four samples.
        let x: &[i16; 2 * BLOCK] = bytemuck::cast_ref(x);
        let o: &mut [i16; 2 * BLOCK] = bytemuck::cast_mut(o);
        let (xs, _) = x.as_chunks::<8>();
        let (os, _) = o.as_chunks_mut::<8>();
        let ([x0, x1, x2, x3], [o0, o1, o2, o3]) = (xs, os) else {
            return;
        };
        let mut s = self.d.transposed();
        let mut ya = self.d.y1.split().0;
        let _ = self.four(&mut s, &mut ya, x0, o0);
        let _ = self.four(&mut s, &mut ya, x1, o1);
        let _ = self.four(&mut s, &mut ya, x2, o2);
        let ((xv2, y2), (xv1, y1)) = self.four(&mut s, &mut ya, x3, o3);
        (self.d.x1, self.d.x2, self.d.y1, self.d.y2) = (xv1, xv2, y1, y2);
    }

    #[inline(always)]
    fn flush(&mut self) {}
}

/// One `CQ15` section on `[re, im]` in an `f64x2`.
struct OneCq15<S: SimdToken> {
    simd: S,
    d: Df1x2<S>,
}

impl<S: SimdToken> OneCq15<S> {
    #[inline(always)]
    fn new(simd: S, c: &[i16; 5], h: &[[i16; 2]]) -> Self {
        let c = scaled(c);
        let at =
            |i: usize| f64x2::simd_from(simd, h.get(i).copied().unwrap_or_default().map(f64::from));
        OneCq15 {
            simd,
            d: Df1x2 {
                ar: Arith2::new(simd),
                b0: f64x2::splat(simd, c[0]),
                b1: f64x2::splat(simd, c[1]),
                b2: f64x2::splat(simd, c[2]),
                a1: f64x2::splat(simd, c[3]),
                a2: f64x2::splat(simd, c[4]),
                x1: at(0),
                x2: at(1),
                y1: at(2),
                y2: at(3),
            },
        }
    }

    #[inline(always)]
    fn store(&self, h: &mut [[i16; 2]]) {
        let d = &self.d;
        for (o, v) in h.iter_mut().zip([d.x1, d.x2, d.y1, d.y2]) {
            *o = (*v).map(to_i16);
        }
    }
}

impl<S: SimdToken> One for OneCq15<S> {
    type T = [i16; 2];

    #[inline(always)]
    fn step(&mut self, x: [i16; 2]) -> [i16; 2] {
        (*self.d.step(f64x2::simd_from(self.simd, x.map(f64::from)))).map(to_i16)
    }

    #[inline(always)]
    fn block(&mut self, x: &[[i16; 2]; BLOCK], o: &mut [[i16; 2]; BLOCK]) {
        let x: &[i16; 2 * BLOCK] = bytemuck::cast_ref(x);
        let o: &mut [i16; 2 * BLOCK] = bytemuck::cast_mut(o);
        let (xs, _) = x.as_chunks::<8>();
        let (os, _) = o.as_chunks_mut::<8>();
        for (x, o) in xs.iter().zip(os) {
            let (lo, hi) = i16x8::simd_from(self.simd, *x).widen();
            let (p0, p1) = to_f64(lo);
            let (p2, p3) = to_f64(hi);
            // Written out, as for `PairQ15`.
            let out = [
                self.d.step(p0),
                self.d.step(p1),
                self.d.step(p2),
                self.d.step(p3),
            ];
            *o = *to_i16x8(out);
        }
    }

    #[inline(always)]
    fn flush(&mut self) {}
}

#[inline(always)]
fn cascade_cq15<S: SimdToken>(
    simd: S,
    coefs: &[[i16; 5]],
    hist: &mut [[i16; 2]],
    input: &[[i16; 2]],
    out: &mut [[i16; 2]],
) {
    #[cfg(target_arch = "x86_64")]
    {
        let level = simd.level();
        if let Some(t) = level.as_avx2() {
            return int::cascade_cq15::<core::arch::x86_64::__m256i>(t, coefs, hist, input, out);
        }
        if let Some(t) = level.as_sse4_2() {
            return int::cascade_cq15::<int::x86::Two>(t, coefs, hist, input, out);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = simd.level().as_neon() {
        return int::cascade_cq15::<int::neon::Two>(t, coefs, hist, input, out);
    }
    for (j, (c, h)) in coefs.chunks(2).zip(hist.chunks_mut(8)).enumerate() {
        let src = (j == 0).then_some(input);
        match c {
            [a, b] => {
                let mut p = PairCq15::new(simd, a, b, h);
                run_pair(&mut p, 0, src, out);
                p.store(h);
            }
            [a] => {
                let mut one = OneCq15::new(simd, a, h);
                run_one(&mut one, 0, src, out);
                one.store(h);
            }
            _ => {}
        }
    }
}

impl Biquad<CQ15> for Simd {
    fn biquad<C, H>(
        &self,
        s: &mut BiquadState<CQ15, C, H>,
        input: &[CQ15],
        output: &mut [CQ15],
    ) -> Result<()>
    where
        C: AsRef<[[<CQ15 as BiquadSample>::Coef; 5]]>,
        H: AsRef<[CQ15]> + AsMut<[CQ15]>,
    {
        check_len(input.len(), output.len())?;
        let (coefs, hist) = s.parts_mut();
        let hist: &mut [[i16; 2]] = bytemuck::cast_slice_mut(hist);
        let x: &[[i16; 2]] = bytemuck::cast_slice(input);
        let y: &mut [[i16; 2]] = bytemuck::cast_slice_mut(output);
        run_cq15(coefs, hist, x, y);
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Q15 and CQ15 in integer lanes (x86 from SSE4.1 on, NEON)

/// The `Q15` and `CQ15` pairs in 64-bit integer lanes, on x86 from SSE4.1 on and on NEON: the
/// arithmetic of the fallback (`q15_wide`) itself instead of its image in `f64`.
///
/// Every value that enters a product is an `i16` (an input, an output, a coefficient) or a
/// negated feedback coefficient (at most 2^15), held in 32 bits; the product of two of them is
/// formed in 64 bits (`pmuldq` on x86, `smlal` on NEON), exactly, and the sums of five products
/// stay below 2^34 in magnitude. The transposed form of the fallback,
/// `y = sat16((b0 x + s) >> 14)`, `s = (b1 x + q) - a1 y`, `q = b2 x - a2 y + 2^13`, is therefore
/// exact here as there, and so is the output, bit for bit. Only the shift and the saturation
/// differ by level:
///
/// - **x86:** the shift is the logical one (`psrlq`, SSE and AVX2 have no arithmetic shift of
///   64-bit lanes); its low 32 bits are those of the arithmetic shift, because `acc >> 14` fits
///   in 21 bits, and the saturation (`pminsd`, `pmaxsd`; for `Q15` at SSE4.2 `packssdw` and
///   `pmovsxwd`, see `x86::Sse`) works on the 32-bit elements. The high half of a lane is left
///   with garbage that no product reads (`pmuldq` reads the low half).
/// - **NEON:** the coefficients and the rounding constant are scaled by 4 (exact, at most 2^17),
///   so the accumulator is `4 acc`; its saturating narrowing to 32 bits (`sqxtn`) and the
///   arithmetic shift by 16 give `sat16(acc >> 14)`: in range `floor(4 acc / 2^16)`, else the
///   bound (`(2^31 - 1) >> 16 = 32767`, `-2^31 >> 16 = -32768`).
///
/// A 128-bit block (two 32-bit halves on NEON) holds `[A, B]`, one value of each section: `Q15`
/// one block, `CQ15` one for `re` and one for `im`. The step is five products, the shift and
/// saturation and one shuffle that brings the last output of the first section into the input
/// of the second (`punpcklqdq`, `zip1`): the chain from one output to the next is a product, a
/// sum, the shift and the saturation, instead of the `f64` sums, `vroundpd` and the bounds.
/// Simulated cycles per sample (`llvm-mca`, two sections, against the fallback): `Q15` 8.2 /
/// 10.2 at AVX2 (Zen 3 / Alder Lake; fallback 11.1 / 12.1), `CQ15` 9.2 / 10.6 (21.2 / 21.3).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod int {
    use fearless_simd::Simd as SimdToken;

    use super::{BLOCK, OneCq15, OneQ15, Pair, coef, load_q15, run_one, run_pair, store_q15};

    /// Functions of a level's token around intrinsics (see `fearless_simd::kernel!`).
    macro_rules! kernels {
        ($($(#[$m:meta])* fn $name:ident($tok:ident $(, $a:ident: $ty:ty)*) -> $ret:ty {
            $($body:tt)*
        })*) => {$(
            $(#[$m])*
            #[inline(always)]
            fn $name(t: $tok $(, $a: $ty)*) -> $ret {
                fearless_simd::kernel!(
                    #[inline(always)]
                    fn inner(_t: $tok $(, $a: $ty)*) -> $ret { $($body)* }
                );
                inner(t $(, $a)*)
            }
        )*};
    }

    /// Integer lanes, `[A, B]` (one value of each section of a pair) in every block: `Self`
    /// holds the inputs, outputs and coefficients (32 significant bits), `Acc` the sums (64).
    pub(super) trait Lanes: Copy {
        type Tok: SimdToken;
        type Acc: Copy;
        /// The scale of the coefficients and of the rounding constant, as a shift.
        const SHIFT: u32;
        /// `[a, b]` in every block.
        fn ab(t: Self::Tok, a: i32, b: i32) -> Self;
        /// `a` in every lane.
        fn acc(t: Self::Tok, a: i64) -> Self::Acc;
        /// `acc + x c`, the product exact in 64 bits.
        fn mla(t: Self::Tok, acc: Self::Acc, x: Self, c: Self) -> Self::Acc;
        /// `sat16(acc >> (14 + SHIFT))`; `lo` and `hi` hold the bounds where the level needs
        /// them.
        fn round(t: Self::Tok, acc: Self::Acc, lo: Self, hi: Self) -> Self;
        /// `[x A, y A]` in every block: the input of a step, `x` for the first section and the
        /// output of the first section from the step before for the second.
        fn feed(t: Self::Tok, x: Self, y: Self) -> Self;
        /// The lanes A of `a` and B of `b`.
        fn first(t: Self::Tok, a: Self, b: Self) -> Self;
        /// [`first`](Lanes::first) of the sums.
        fn first_acc(t: Self::Tok, a: Self::Acc, b: Self::Acc) -> Self::Acc;
        /// `v`, unchanged, as a value the compiler cannot look into (an empty `asm!` block):
        /// it keeps a partial sum from being regrouped with the terms of the chain.
        fn keep(t: Self::Tok, v: Self::Acc) -> Self::Acc;
    }

    /// The `Q15` lanes: one block.
    pub(super) trait Real: Lanes {
        /// `x` in the lane A.
        fn one(t: Self::Tok, x: i16) -> Self;
        /// The lanes A and B as `i16` (values within range).
        fn get(t: Self::Tok, v: Self) -> [i16; 2];
        /// Four of the eight inputs (the last four if `high`), each in the lane A of a vector.
        /// Half a block at a time: the inputs of the second half are not live (in registers)
        /// during the first.
        fn load(t: Self::Tok, x: &[i16; 8], high: bool) -> [Self; 4];
        /// The lanes B of eight outputs.
        fn store(t: Self::Tok, y: [Self; 8]) -> [i16; 8];
    }

    /// The `CQ15` lanes: a block of `re` and one of `im`.
    pub(super) trait Complex: Lanes {
        /// `[re, im]` in the lanes A.
        fn one(t: Self::Tok, x: [i16; 2]) -> Self;
        /// `[re, im]` of the lanes A and of the lanes B (values within range).
        fn get(t: Self::Tok, v: Self) -> [[i16; 2]; 2];
        /// `[re, im]` of A and of B in the lanes.
        fn set(t: Self::Tok, a: [i16; 2], b: [i16; 2]) -> Self;
        /// Four samples (eight `i16`), each in the lanes A of a vector.
        fn load(t: Self::Tok, x: &[i16; 8]) -> [Self; 4];
        /// The lanes B of four outputs.
        fn store(t: Self::Tok, y: [Self; 4]) -> [i16; 8];
    }

    /// An `i32` that holds an `i16` (an input, an output or a state value).
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation, reason = "the values are within i16")]
    fn narrow<const N: usize>(v: [i32; N]) -> [i16; N] {
        v.map(|v| v as i16)
    }

    /// The coefficients of a pair (`a1` and `a2` negated, which is exact in 32 bits also for
    /// `a = -2.0`, raw -32768) and the arithmetic of one step.
    struct Core<L: Lanes> {
        t: L::Tok,
        b0: L,
        b1: L,
        b2: L,
        na1: L,
        na2: L,
        lo: L,
        hi: L,
        /// The rounding constant `2^13`.
        r: L::Acc,
    }

    impl<L: Lanes> Core<L> {
        #[inline(always)]
        fn new(t: L::Tok, a: &[i16; 5], b: &[i16; 5]) -> Self {
            let c = |i: usize, sign: i32| {
                let v = |c: &[i16; 5]| (sign * i32::from(coef(c, i))) << L::SHIFT;
                L::ab(t, v(a), v(b))
            };
            Core {
                t,
                b0: c(0, 1),
                b1: c(1, 1),
                b2: c(2, 1),
                na1: c(3, -1),
                na2: c(4, -1),
                lo: L::ab(t, -32768, -32768),
                hi: L::ab(t, 32767, 32767),
                r: L::acc(t, 1 << (13 + L::SHIFT)),
            }
        }

        /// One step of the transposed form on the state `(s, q)`: `xv` the input vector;
        /// returns the outputs. The product with the output is the last term of each sum, so
        /// that the chain from one output to the next is a product and two sums (the second
        /// with the product of the next input vector, which holds the output of the first
        /// section). `q + b1 x` is kept apart from the products: the compiler would otherwise
        /// add `b1 x` after `-a1 y`, one product and sum more in the chain.
        #[inline(always)]
        fn step(&self, st: &mut State<L>, xv: L) -> L {
            let t = self.t;
            let y = L::round(t, L::mla(t, st.0, xv, self.b0), self.lo, self.hi);
            let s = L::keep(t, L::mla(t, st.1, xv, self.b1));
            let q = L::mla(t, self.r, xv, self.b2);
            st.0 = L::mla(t, s, y, self.na1);
            st.1 = L::mla(t, q, y, self.na2);
            y
        }

        /// The transposed state of the direct form I history `[x1, x2, y1, y2]`:
        /// `s = b1 x1 + b2 x2 - a1 y1 - a2 y2 + 2^13`, `q = b2 x1 - a2 y1 + 2^13`.
        #[inline(always)]
        fn transposed(&self, h: [L; 4]) -> State<L> {
            let t = self.t;
            let [x1, x2, y1, y2] = h;
            let q = L::mla(t, L::mla(t, self.r, x1, self.b2), y1, self.na2);
            let s = L::mla(t, L::mla(t, self.r, x1, self.b1), x2, self.b2);
            let s = L::mla(t, L::mla(t, s, y1, self.na1), y2, self.na2);
            (s, q)
        }
    }

    /// The transposed state of a pair.
    type State<L> = (<L as Lanes>::Acc, <L as Lanes>::Acc);
    /// The input vectors and the outputs of the last two steps of a block.
    type Io<L> = ((L, L), (L, L));

    /// Two sections in the lanes, the second one step behind: the transposed state and the
    /// direct form I history (the inputs and outputs of the last two steps), which the state
    /// after a call is stored as.
    struct IntPair<L: Lanes> {
        c: Core<L>,
        st: State<L>,
        x1: L,
        x2: L,
        y1: L,
        y2: L,
    }

    impl<L: Lanes> IntPair<L> {
        #[inline(always)]
        fn new(c: Core<L>, h: [L; 4]) -> Self {
            let st = c.transposed(h);
            let [x1, x2, y1, y2] = h;
            IntPair {
                c,
                st,
                x1,
                x2,
                y1,
                y2,
            }
        }

        /// One step that changes the lanes A only (`first`) or B only: returns the output.
        #[inline(always)]
        fn masked(&mut self, xv: L, first: bool) -> L {
            let t = self.c.t;
            let mut st = self.st;
            let y = self.c.step(&mut st, xv);
            if first {
                self.st = (
                    L::first_acc(t, st.0, self.st.0),
                    L::first_acc(t, st.1, self.st.1),
                );
                (self.x2, self.x1) = (L::first(t, self.x1, self.x2), L::first(t, xv, self.x1));
                (self.y2, self.y1) = (L::first(t, self.y1, self.y2), L::first(t, y, self.y1));
            } else {
                self.st = (
                    L::first_acc(t, self.st.0, st.0),
                    L::first_acc(t, self.st.1, st.1),
                );
                (self.x2, self.x1) = (L::first(t, self.x2, self.x1), L::first(t, self.x1, xv));
                (self.y2, self.y1) = (L::first(t, self.y2, self.y1), L::first(t, self.y1, y));
            }
            y
        }

        /// The first step of a call: only the first section filters (`x` in the lanes A).
        #[inline(always)]
        fn fill(&mut self, x: L) {
            let _ = self.masked(x, true);
        }

        /// One step: returns the outputs, those of the second section in the lanes B.
        #[inline(always)]
        fn step(&mut self, x: L) -> L {
            let xv = L::feed(self.c.t, x, self.y1);
            let y = self.c.step(&mut self.st, xv);
            (self.x2, self.x1, self.y2, self.y1) = (self.x1, xv, self.y1, y);
            y
        }

        /// The last step of a call: only the second section filters (the last output of the
        /// first); `zero` is any vector, its lanes A are not used.
        #[inline(always)]
        fn drain(&mut self, zero: L) -> L {
            let xv = L::feed(self.c.t, zero, self.y1);
            self.masked(xv, false)
        }

        /// One step inside a block (the history is set at its end): returns the input vector
        /// and the output.
        #[inline(always)]
        fn run(&self, st: &mut State<L>, y: &mut L, x: L) -> (L, L) {
            let xv = L::feed(self.c.t, x, *y);
            *y = self.c.step(st, xv);
            (xv, *y)
        }

        /// Four steps: the outputs and the input vectors of the last two steps.
        #[inline(always)]
        fn four(&self, st: &mut State<L>, y: &mut L, x: [L; 4]) -> ([L; 4], L, L) {
            let [x0, x1, x2, x3] = x;
            let (_, a) = self.run(st, y, x0);
            let (_, b) = self.run(st, y, x1);
            let (v2, c) = self.run(st, y, x2);
            let (v3, d) = self.run(st, y, x3);
            ([a, b, c, d], v2, v3)
        }

        /// Sets the state after the steps of a block.
        #[inline(always)]
        fn end(&mut self, st: State<L>, io: Io<L>) {
            self.st = st;
            ((self.x2, self.x1), (self.y2, self.y1)) = io;
        }

        /// The history `[x1, x2, y1, y2]`.
        #[inline(always)]
        fn history(&self) -> [L; 4] {
            [self.x1, self.x2, self.y1, self.y2]
        }
    }

    /// Two `Q15` sections.
    struct PairQ15<L: Real>(IntPair<L>);

    impl<L: Real> PairQ15<L> {
        /// `h` holds the eight history entries `[x1, x2, y1, y2]` of A and of B.
        #[inline(always)]
        fn new(t: L::Tok, a: &[i16; 5], b: &[i16; 5], h: &[i16]) -> Self {
            let at = |i: usize| i32::from(h.get(i).copied().unwrap_or(0));
            let st = |i: usize| L::ab(t, at(i), at(i + 4));
            PairQ15(IntPair::new(
                Core::new(t, a, b),
                [st(0), st(1), st(2), st(3)],
            ))
        }

        #[inline(always)]
        fn store(&self, h: &mut [i16]) {
            for (i, v) in self.0.history().into_iter().enumerate() {
                let [a, b] = L::get(self.0.c.t, v);
                for (j, w) in [(i, a), (i + 4, b)] {
                    if let Some(o) = h.get_mut(j) {
                        *o = w;
                    }
                }
            }
        }

        /// Eight steps: the input vectors and outputs of the last two steps.
        #[inline(always)]
        fn eight(&self, st: &mut State<L>, y: &mut L, x: &[i16; 8], o: &mut [i16; 8]) -> Io<L> {
            let t = self.0.c.t;
            let ([a, b, c, d], _, _) = self.0.four(st, y, L::load(t, x, false));
            let ([e, f, g, h], v6, v7) = self.0.four(st, y, L::load(t, x, true));
            *o = L::store(t, [a, b, c, d, e, f, g, h]);
            ((v6, v7), (g, h))
        }
    }

    impl<L: Real> Pair for PairQ15<L> {
        type T = i16;

        #[inline(always)]
        fn fill(&mut self, x: i16, _flush: bool) {
            self.0.fill(L::one(self.0.c.t, x));
        }

        #[inline(always)]
        fn step(&mut self, x: i16) -> i16 {
            let t = self.0.c.t;
            let [_, y] = L::get(t, self.0.step(L::one(t, x)));
            y
        }

        #[inline(always)]
        fn drain(&mut self, _flush: bool) -> i16 {
            let t = self.0.c.t;
            let [_, y] = L::get(t, self.0.drain(L::one(t, 0)));
            y
        }

        #[inline(always)]
        fn block(&mut self, x: &[i16; BLOCK], o: &mut [i16; BLOCK]) {
            // Written out with methods, not closures or loops over arrays, as the `f64` path.
            let ((x0, x1), (o0, o1)) = (x.split_at(8), o.split_at_mut(8));
            let (Ok(x0), Ok(x1), Ok(o0), Ok(o1)) = (
                <&[i16; 8]>::try_from(x0),
                <&[i16; 8]>::try_from(x1),
                <&mut [i16; 8]>::try_from(o0),
                <&mut [i16; 8]>::try_from(o1),
            ) else {
                return;
            };
            let (mut st, mut y) = (self.0.st, self.0.y1);
            let _ = self.eight(&mut st, &mut y, x0, o0);
            let io = self.eight(&mut st, &mut y, x1, o1);
            self.0.end(st, io);
        }

        #[inline(always)]
        fn flush(&mut self) {}
    }

    /// Two `CQ15` sections.
    struct PairCq15<L: Complex>(IntPair<L>);

    impl<L: Complex> PairCq15<L> {
        /// `h` holds the sixteen history values `[x1, x2, y1, y2]` (`[re, im]` each) of A and B.
        #[inline(always)]
        fn new(t: L::Tok, a: &[i16; 5], b: &[i16; 5], h: &[[i16; 2]]) -> Self {
            let at = |i: usize| h.get(i).copied().unwrap_or_default();
            let st = |i: usize| L::set(t, at(i), at(i + 4));
            PairCq15(IntPair::new(
                Core::new(t, a, b),
                [st(0), st(1), st(2), st(3)],
            ))
        }

        #[inline(always)]
        fn store(&self, h: &mut [[i16; 2]]) {
            for (i, v) in self.0.history().into_iter().enumerate() {
                let [a, b] = L::get(self.0.c.t, v);
                for (j, w) in [(i, a), (i + 4, b)] {
                    if let Some(o) = h.get_mut(j) {
                        *o = w;
                    }
                }
            }
        }

        /// Four steps (eight `i16`): the input vectors and outputs of the last two steps.
        #[inline(always)]
        fn four(&self, st: &mut State<L>, y: &mut L, x: &[i16; 8], o: &mut [i16; 8]) -> Io<L> {
            let t = self.0.c.t;
            let ([a, b, c, d], v2, v3) = self.0.four(st, y, L::load(t, x));
            *o = L::store(t, [a, b, c, d]);
            ((v2, v3), (c, d))
        }
    }

    impl<L: Complex> Pair for PairCq15<L> {
        type T = [i16; 2];

        #[inline(always)]
        fn fill(&mut self, x: [i16; 2], _flush: bool) {
            self.0.fill(L::one(self.0.c.t, x));
        }

        #[inline(always)]
        fn step(&mut self, x: [i16; 2]) -> [i16; 2] {
            let t = self.0.c.t;
            let [_, y] = L::get(t, self.0.step(L::one(t, x)));
            y
        }

        #[inline(always)]
        fn drain(&mut self, _flush: bool) -> [i16; 2] {
            let t = self.0.c.t;
            let [_, y] = L::get(t, self.0.drain(L::one(t, [0, 0])));
            y
        }

        #[inline(always)]
        fn block(&mut self, x: &[[i16; 2]; BLOCK], o: &mut [[i16; 2]; BLOCK]) {
            let x: &[i16; 2 * BLOCK] = bytemuck::cast_ref(x);
            let o: &mut [i16; 2 * BLOCK] = bytemuck::cast_mut(o);
            let (xs, _) = x.as_chunks::<8>();
            let (os, _) = o.as_chunks_mut::<8>();
            let ([x0, x1, x2, x3], [o0, o1, o2, o3]) = (xs, os) else {
                return;
            };
            let (mut st, mut y) = (self.0.st, self.0.y1);
            let _ = self.four(&mut st, &mut y, x0, o0);
            let _ = self.four(&mut st, &mut y, x1, o1);
            let _ = self.four(&mut st, &mut y, x2, o2);
            let io = self.four(&mut st, &mut y, x3, o3);
            self.0.end(st, io);
        }

        #[inline(always)]
        fn flush(&mut self) {}
    }

    /// The `Q15` cascade with the pairs in the lanes of `L` (an odd last section runs the
    /// integer code of the fallback).
    #[inline(always)]
    pub(super) fn cascade_q15<L: Real>(
        t: L::Tok,
        coefs: &[[i16; 5]],
        hist: &mut [i16],
        input: &[i16],
        out: &mut [i16],
    ) {
        for (j, (c, h)) in coefs.chunks(2).zip(hist.chunks_mut(8)).enumerate() {
            let src = (j == 0).then_some(input);
            match c {
                [a, b] => {
                    let mut p = PairQ15::<L>::new(t, a, b, h);
                    run_pair(&mut p, 0, src, out);
                    p.store(h);
                }
                [a] => {
                    let mut one = OneQ15 {
                        c: a.map(i64::from),
                        h: load_q15(h),
                    };
                    run_one(&mut one, 0, src, out);
                    store_q15(&one.h, h);
                }
                _ => {}
            }
        }
    }

    /// The `CQ15` cascade with the pairs in the lanes of `L` (an odd last section runs the
    /// two-lane `f64` path).
    #[inline(always)]
    pub(super) fn cascade_cq15<L: Complex>(
        t: L::Tok,
        coefs: &[[i16; 5]],
        hist: &mut [[i16; 2]],
        input: &[[i16; 2]],
        out: &mut [[i16; 2]],
    ) {
        for (j, (c, h)) in coefs.chunks(2).zip(hist.chunks_mut(8)).enumerate() {
            let src = (j == 0).then_some(input);
            match c {
                [a, b] => {
                    let mut p = PairCq15::<L>::new(t, a, b, h);
                    run_pair(&mut p, 0, src, out);
                    p.store(h);
                }
                [a] => {
                    let mut one = OneCq15::new(t, a, h);
                    run_one(&mut one, 0, src, out);
                    one.store(h);
                }
                _ => {}
            }
        }
    }

    /// The lanes of x86: `__m128i` for `Q15` (as `Sse` at SSE4.2) and for `CQ15` at SSE4.2
    /// (two of them, `Two`), `__m256i` for `CQ15` at AVX2 and AVX-512 (`re` in the low block,
    /// `im` in the high one).
    #[cfg(target_arch = "x86_64")]
    #[allow(unsafe_code)]
    pub(super) mod x86 {
        use core::arch::asm;
        use core::arch::x86_64::{
            __m128i, __m256i, _mm_add_epi64, _mm_blend_epi16, _mm_castps_si128, _mm_castsi128_ps,
            _mm_cvtepi16_epi32, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_extract_epi32,
            _mm_max_epi32, _mm_min_epi32, _mm_mul_epi32, _mm_packs_epi32, _mm_set_epi32,
            _mm_set1_epi32, _mm_set1_epi64x, _mm_setr_epi8, _mm_shuffle_epi8, _mm_shuffle_ps,
            _mm_srli_epi64, _mm_srli_si128, _mm_unpackhi_epi64, _mm_unpacklo_epi16,
            _mm_unpacklo_epi64, _mm256_add_epi64, _mm256_blend_epi32, _mm256_castps_si256,
            _mm256_castsi256_ps, _mm256_castsi256_si128, _mm256_cvtepi16_epi32,
            _mm256_extract_epi32, _mm256_extracti128_si256, _mm256_max_epi32, _mm256_min_epi32,
            _mm256_mul_epi32, _mm256_packs_epi32, _mm256_set_epi32, _mm256_set1_epi64x,
            _mm256_shuffle_ps, _mm256_srli_epi64, _mm256_srli_si256, _mm256_unpackhi_epi64,
            _mm256_unpacklo_epi64,
        };

        use fearless_simd::{Avx2, Simd as SimdToken, Sse4_2, i16x8, prelude::*};

        use super::{Complex, Lanes, Real, narrow};

        kernels! {
            fn ab1(Sse4_2, a: i32, b: i32) -> __m128i { _mm_set_epi32(0, b, 0, a) }
            fn acc1(Sse4_2, a: i64) -> __m128i { _mm_set1_epi64x(a) }
            fn mla1(Sse4_2, acc: __m128i, x: __m128i, c: __m128i) -> __m128i {
                _mm_add_epi64(acc, _mm_mul_epi32(x, c))
            }
            /// The bounds are constants here, not values of the pair: the compiler takes them
            /// from memory when the registers run out instead of moving a coefficient there.
            fn round1(Sse4_2, v: __m128i) -> __m128i {
                let v = _mm_srli_epi64::<14>(v);
                _mm_max_epi32(_mm_min_epi32(v, _mm_set1_epi32(32767)), _mm_set1_epi32(-32768))
            }
            /// The saturation as `packssdw` and its sign extension back (`pmovsxwd`): the low
            /// halves of the lanes A and B end up in the elements 0 and 2 again. As fast as
            /// `pminsd` and `pmaxsd` at SSE4.2 and without the bounds, whose registers the
            /// two-operand code needs for copies (`llvm-mca`: 12.1 against 18.0 cycles per sample
            /// on Skylake, 11.3 against 17.9 on Zen 2); with AVX `pmovsxwd` takes 3 cycles on
            /// Alder Lake (10.2 against 12.2 for the bounds).
            fn round1p(Sse4_2, v: __m128i) -> __m128i {
                let v = _mm_srli_epi64::<14>(v);
                _mm_cvtepi16_epi32(_mm_packs_epi32(v, v))
            }
            fn feed1(Sse4_2, x: __m128i, y: __m128i) -> __m128i { _mm_unpacklo_epi64(x, y) }
            fn first1(Sse4_2, a: __m128i, b: __m128i) -> __m128i {
                _mm_blend_epi16::<0xF0>(a, b)
            }
            fn keep1(Sse4_2, v: __m128i) -> __m128i {
                let mut v = v;
                // SAFETY: an empty block: no instruction, no memory, stack or flags.
                unsafe {
                    asm!(
                        "/* {v} */",
                        v = inout(xmm_reg) v,
                        options(pure, nomem, nostack, preserves_flags)
                    );
                };
                v
            }
            fn one1(Sse4_2, x: i32) -> __m128i { _mm_cvtsi32_si128(x) }
            fn get1(Sse4_2, v: __m128i) -> [i32; 2] {
                [_mm_cvtsi128_si32(v), _mm_extract_epi32::<2>(v)]
            }
            /// The `i16` 0 to 3 (4 to 7 if `high`) as `i32`, each in the low element of a vector.
            fn spread1(Sse4_2, v: __m128i, high: bool) -> [__m128i; 4] {
                let w = _mm_cvtepi16_epi32(if high { _mm_srli_si128::<8>(v) } else { v });
                [
                    w,
                    _mm_srli_si128::<4>(w),
                    _mm_srli_si128::<8>(w),
                    _mm_srli_si128::<12>(w),
                ]
            }
            /// The lanes B of four vectors, as `i32`.
            fn gather1(Sse4_2, a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> __m128i {
                let ab = _mm_castsi128_ps(_mm_unpackhi_epi64(a, b));
                let cd = _mm_castsi128_ps(_mm_unpackhi_epi64(c, d));
                _mm_castps_si128(_mm_shuffle_ps::<0x88>(ab, cd))
            }
            fn packs1(Sse4_2, a: __m128i, b: __m128i) -> __m128i { _mm_packs_epi32(a, b) }
            /// `[r0, i0, r1, i1, ..]` to `[r0, r1, .., i0, i1, ..]`.
            fn split1(Sse4_2, v: __m128i) -> __m128i {
                let m = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
                _mm_shuffle_epi8(v, m)
            }
            /// `[r0, r1, .., i0, i1, ..]` to `[r0, i0, r1, i1, ..]`.
            fn join1(Sse4_2, v: __m128i) -> __m128i {
                let m = _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
                _mm_shuffle_epi8(v, m)
            }

            fn ab4(Avx2, a: i32, b: i32) -> __m256i { _mm256_set_epi32(0, b, 0, a, 0, b, 0, a) }
            fn acc4(Avx2, a: i64) -> __m256i { _mm256_set1_epi64x(a) }
            fn mla4(Avx2, acc: __m256i, x: __m256i, c: __m256i) -> __m256i {
                _mm256_add_epi64(acc, _mm256_mul_epi32(x, c))
            }
            fn round4(Avx2, v: __m256i, lo: __m256i, hi: __m256i) -> __m256i {
                _mm256_max_epi32(_mm256_min_epi32(_mm256_srli_epi64::<14>(v), hi), lo)
            }
            fn feed4(Avx2, x: __m256i, y: __m256i) -> __m256i { _mm256_unpacklo_epi64(x, y) }
            fn first4(Avx2, a: __m256i, b: __m256i) -> __m256i {
                _mm256_blend_epi32::<0b1100_1100>(a, b)
            }
            fn keep4(Avx2, v: __m256i) -> __m256i {
                let mut v = v;
                // SAFETY: an empty block: no instruction, no memory, stack or flags.
                unsafe {
                    asm!(
                        "/* {v} */",
                        v = inout(ymm_reg) v,
                        options(pure, nomem, nostack, preserves_flags)
                    );
                };
                v
            }
            fn one4(Avx2, re: i32, im: i32) -> __m256i {
                _mm256_set_epi32(0, 0, 0, im, 0, 0, 0, re)
            }
            /// `[re A, im A, re B, im B]`.
            fn get4(Avx2, v: __m256i) -> [i32; 4] {
                [
                    _mm256_extract_epi32::<0>(v),
                    _mm256_extract_epi32::<4>(v),
                    _mm256_extract_epi32::<2>(v),
                    _mm256_extract_epi32::<6>(v),
                ]
            }
            fn set4(Avx2, ra: i32, ia: i32, rb: i32, ib: i32) -> __m256i {
                _mm256_set_epi32(0, ib, 0, ia, 0, rb, 0, ra)
            }
            /// Four samples `[r0, i0, r1, i1, ..]`: `[rk, .. | ik, ..]` in the low elements of
            /// the blocks of the vector `k`.
            fn spread4(Avx2, v: __m128i) -> [__m256i; 4] {
                let m = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
                let w = _mm256_cvtepi16_epi32(_mm_shuffle_epi8(v, m));
                [
                    w,
                    _mm256_srli_si256::<4>(w),
                    _mm256_srli_si256::<8>(w),
                    _mm256_srli_si256::<12>(w),
                ]
            }
            /// The lanes B of four vectors as the samples `[r0, i0, r1, i1, ..]`.
            fn store4(Avx2, a: __m256i, b: __m256i, c: __m256i, d: __m256i) -> __m128i {
                let ab = _mm256_castsi256_ps(_mm256_unpackhi_epi64(a, b));
                let cd = _mm256_castsi256_ps(_mm256_unpackhi_epi64(c, d));
                let v = _mm256_castps_si256(_mm256_shuffle_ps::<0x88>(ab, cd));
                // `[r0, r1, r2, r3, ..]` and `[i0, i1, i2, i3, ..]` in the two blocks.
                let p = _mm256_packs_epi32(v, v);
                _mm_unpacklo_epi16(_mm256_castsi256_si128(p), _mm256_extracti128_si256::<1>(p))
            }
        }

        #[inline(always)]
        fn to_m128<S: SimdToken>(t: S, x: &[i16; 8]) -> __m128i {
            i16x8::simd_from(t, *x).into()
        }

        #[inline(always)]
        fn from_m128<S: SimdToken>(t: S, v: __m128i) -> [i16; 8] {
            *i16x8::simd_from(t, v)
        }

        impl Lanes for __m128i {
            type Tok = Sse4_2;
            type Acc = __m128i;
            const SHIFT: u32 = 0;
            #[inline(always)]
            fn ab(t: Sse4_2, a: i32, b: i32) -> Self {
                ab1(t, a, b)
            }
            #[inline(always)]
            fn acc(t: Sse4_2, a: i64) -> Self {
                acc1(t, a)
            }
            #[inline(always)]
            fn mla(t: Sse4_2, acc: Self, x: Self, c: Self) -> Self {
                mla1(t, acc, x, c)
            }
            #[inline(always)]
            fn round(t: Sse4_2, v: Self, _lo: Self, _hi: Self) -> Self {
                round1(t, v)
            }
            #[inline(always)]
            fn feed(t: Sse4_2, x: Self, y: Self) -> Self {
                feed1(t, x, y)
            }
            #[inline(always)]
            fn first(t: Sse4_2, a: Self, b: Self) -> Self {
                first1(t, a, b)
            }
            #[inline(always)]
            fn first_acc(t: Sse4_2, a: Self, b: Self) -> Self {
                first1(t, a, b)
            }
            #[inline(always)]
            fn keep(t: Sse4_2, v: Self) -> Self {
                keep1(t, v)
            }
        }

        impl Real for __m128i {
            #[inline(always)]
            fn one(t: Sse4_2, x: i16) -> Self {
                one1(t, i32::from(x))
            }
            #[inline(always)]
            fn get(t: Sse4_2, v: Self) -> [i16; 2] {
                narrow(get1(t, v))
            }
            #[inline(always)]
            fn load(t: Sse4_2, x: &[i16; 8], high: bool) -> [Self; 4] {
                spread1(t, to_m128(t, x), high)
            }
            #[inline(always)]
            fn store(t: Sse4_2, y: [Self; 8]) -> [i16; 8] {
                let [a, b, c, d, e, f, g, h] = y;
                let (lo, hi) = (gather1(t, a, b, c, d), gather1(t, e, f, g, h));
                from_m128(t, packs1(t, lo, hi))
            }
        }

        /// `Q15` at SSE4.2 (two-operand code): the lanes of `__m128i` with the saturation of
        /// [`round1p`].
        #[derive(Clone, Copy)]
        pub(in super::super) struct Sse(__m128i);

        impl Lanes for Sse {
            type Tok = Sse4_2;
            type Acc = __m128i;
            const SHIFT: u32 = 0;
            #[inline(always)]
            fn ab(t: Sse4_2, a: i32, b: i32) -> Self {
                Sse(ab1(t, a, b))
            }
            #[inline(always)]
            fn acc(t: Sse4_2, a: i64) -> __m128i {
                acc1(t, a)
            }
            #[inline(always)]
            fn mla(t: Sse4_2, acc: __m128i, x: Self, c: Self) -> __m128i {
                mla1(t, acc, x.0, c.0)
            }
            #[inline(always)]
            fn round(t: Sse4_2, v: __m128i, _lo: Self, _hi: Self) -> Self {
                Sse(round1p(t, v))
            }
            #[inline(always)]
            fn feed(t: Sse4_2, x: Self, y: Self) -> Self {
                Sse(feed1(t, x.0, y.0))
            }
            #[inline(always)]
            fn first(t: Sse4_2, a: Self, b: Self) -> Self {
                Sse(first1(t, a.0, b.0))
            }
            #[inline(always)]
            fn first_acc(t: Sse4_2, a: __m128i, b: __m128i) -> __m128i {
                first1(t, a, b)
            }
            #[inline(always)]
            fn keep(t: Sse4_2, v: __m128i) -> __m128i {
                keep1(t, v)
            }
        }

        impl Real for Sse {
            #[inline(always)]
            fn one(t: Sse4_2, x: i16) -> Self {
                Sse(<__m128i as Real>::one(t, x))
            }
            #[inline(always)]
            fn get(t: Sse4_2, v: Self) -> [i16; 2] {
                <__m128i as Real>::get(t, v.0)
            }
            #[inline(always)]
            fn load(t: Sse4_2, x: &[i16; 8], high: bool) -> [Self; 4] {
                <__m128i as Real>::load(t, x, high).map(Sse)
            }
            #[inline(always)]
            fn store(t: Sse4_2, y: [Self; 8]) -> [i16; 8] {
                <__m128i as Real>::store(t, y.map(|v| v.0))
            }
        }

        /// `CQ15` at SSE4.2: the blocks of `re` and `im` in two registers.
        #[derive(Clone, Copy)]
        pub(in super::super) struct Two(__m128i, __m128i);

        impl Lanes for Two {
            type Tok = Sse4_2;
            type Acc = Two;
            const SHIFT: u32 = 0;
            #[inline(always)]
            fn ab(t: Sse4_2, a: i32, b: i32) -> Self {
                let v = ab1(t, a, b);
                Two(v, v)
            }
            #[inline(always)]
            fn acc(t: Sse4_2, a: i64) -> Self {
                let v = acc1(t, a);
                Two(v, v)
            }
            #[inline(always)]
            fn mla(t: Sse4_2, acc: Self, x: Self, c: Self) -> Self {
                Two(mla1(t, acc.0, x.0, c.0), mla1(t, acc.1, x.1, c.1))
            }
            #[inline(always)]
            fn round(t: Sse4_2, v: Self, _lo: Self, _hi: Self) -> Self {
                Two(round1(t, v.0), round1(t, v.1))
            }
            #[inline(always)]
            fn feed(t: Sse4_2, x: Self, y: Self) -> Self {
                Two(feed1(t, x.0, y.0), feed1(t, x.1, y.1))
            }
            #[inline(always)]
            fn first(t: Sse4_2, a: Self, b: Self) -> Self {
                Two(first1(t, a.0, b.0), first1(t, a.1, b.1))
            }
            #[inline(always)]
            fn first_acc(t: Sse4_2, a: Self, b: Self) -> Self {
                Two::first(t, a, b)
            }
            #[inline(always)]
            fn keep(t: Sse4_2, v: Self) -> Self {
                Two(keep1(t, v.0), keep1(t, v.1))
            }
        }

        impl Complex for Two {
            #[inline(always)]
            fn one(t: Sse4_2, x: [i16; 2]) -> Self {
                Two(one1(t, i32::from(x[0])), one1(t, i32::from(x[1])))
            }
            #[inline(always)]
            fn get(t: Sse4_2, v: Self) -> [[i16; 2]; 2] {
                let ([ra, rb], [ia, ib]) = (narrow(get1(t, v.0)), narrow(get1(t, v.1)));
                [[ra, ia], [rb, ib]]
            }
            #[inline(always)]
            fn set(t: Sse4_2, a: [i16; 2], b: [i16; 2]) -> Self {
                let w = |a: i16, b: i16| ab1(t, i32::from(a), i32::from(b));
                Two(w(a[0], b[0]), w(a[1], b[1]))
            }
            #[inline(always)]
            fn load(t: Sse4_2, x: &[i16; 8]) -> [Self; 4] {
                let v = split1(t, to_m128(t, x));
                let ([r0, r1, r2, r3], [i0, i1, i2, i3]) =
                    (spread1(t, v, false), spread1(t, v, true));
                [Two(r0, i0), Two(r1, i1), Two(r2, i2), Two(r3, i3)]
            }
            #[inline(always)]
            fn store(t: Sse4_2, y: [Self; 4]) -> [i16; 8] {
                let [a, b, c, d] = y;
                let re = gather1(t, a.0, b.0, c.0, d.0);
                let im = gather1(t, a.1, b.1, c.1, d.1);
                from_m128(t, join1(t, packs1(t, re, im)))
            }
        }

        impl Lanes for __m256i {
            type Tok = Avx2;
            type Acc = __m256i;
            const SHIFT: u32 = 0;
            #[inline(always)]
            fn ab(t: Avx2, a: i32, b: i32) -> Self {
                ab4(t, a, b)
            }
            #[inline(always)]
            fn acc(t: Avx2, a: i64) -> Self {
                acc4(t, a)
            }
            #[inline(always)]
            fn mla(t: Avx2, acc: Self, x: Self, c: Self) -> Self {
                mla4(t, acc, x, c)
            }
            #[inline(always)]
            fn round(t: Avx2, v: Self, lo: Self, hi: Self) -> Self {
                round4(t, v, lo, hi)
            }
            #[inline(always)]
            fn feed(t: Avx2, x: Self, y: Self) -> Self {
                feed4(t, x, y)
            }
            #[inline(always)]
            fn first(t: Avx2, a: Self, b: Self) -> Self {
                first4(t, a, b)
            }
            #[inline(always)]
            fn first_acc(t: Avx2, a: Self, b: Self) -> Self {
                first4(t, a, b)
            }
            #[inline(always)]
            fn keep(t: Avx2, v: Self) -> Self {
                keep4(t, v)
            }
        }

        impl Complex for __m256i {
            #[inline(always)]
            fn one(t: Avx2, x: [i16; 2]) -> Self {
                one4(t, i32::from(x[0]), i32::from(x[1]))
            }
            #[inline(always)]
            fn get(t: Avx2, v: Self) -> [[i16; 2]; 2] {
                let [ra, ia, rb, ib] = narrow(get4(t, v));
                [[ra, ia], [rb, ib]]
            }
            #[inline(always)]
            fn set(t: Avx2, a: [i16; 2], b: [i16; 2]) -> Self {
                let w = i32::from;
                set4(t, w(a[0]), w(a[1]), w(b[0]), w(b[1]))
            }
            #[inline(always)]
            fn load(t: Avx2, x: &[i16; 8]) -> [Self; 4] {
                spread4(t, to_m128(t, x))
            }
            #[inline(always)]
            fn store(t: Avx2, y: [Self; 4]) -> [i16; 8] {
                let [a, b, c, d] = y;
                from_m128(t, store4(t, a, b, c, d))
            }
        }
    }

    /// The lanes of NEON: `int32x2_t` `[A, B]` with `int64x2_t` sums for `Q15`, a pair of
    /// each (`Two`, `re` and `im`) for `CQ15`.
    #[cfg(target_arch = "aarch64")]
    #[allow(unsafe_code)]
    pub(super) mod neon {
        use core::arch::aarch64::{
            int16x8_t, int32x2_t, int64x2_t, vcombine_s16, vcombine_s32, vcopy_lane_s32,
            vcopyq_laneq_s64, vdup_laneq_s32, vdup_n_s32, vdupq_n_s64, vget_high_s32,
            vget_lane_s32, vget_low_s16, vget_low_s32, vmlal_s32, vmovl_high_s16, vmovl_s16,
            vmovn_s32, vqmovn_s64, vset_lane_s32, vshr_n_s32, vuzp1q_s16, vuzp2q_s16, vzip1_s32,
            vzip1q_s32, vzip2_s32, vzip2q_s32,
        };
        use core::arch::asm;

        use fearless_simd::{Neon, i16x8, prelude::*};

        use super::{Complex, Lanes, Real, narrow};

        kernels! {
            fn ab2(Neon, a: i32, b: i32) -> int32x2_t { vset_lane_s32::<1>(b, vdup_n_s32(a)) }
            fn acc2(Neon, a: i64) -> int64x2_t { vdupq_n_s64(a) }
            fn mla2(Neon, acc: int64x2_t, x: int32x2_t, c: int32x2_t) -> int64x2_t {
                vmlal_s32(acc, x, c)
            }
            /// `sat16(acc >> 16)` of the sum scaled by 4: the saturation to 32 bits and the
            /// arithmetic shift by 16.
            fn round2(Neon, v: int64x2_t) -> int32x2_t { vshr_n_s32::<16>(vqmovn_s64(v)) }
            fn feed2(Neon, x: int32x2_t, y: int32x2_t) -> int32x2_t { vzip1_s32(x, y) }
            fn first2(Neon, a: int32x2_t, b: int32x2_t) -> int32x2_t {
                vcopy_lane_s32::<1, 1>(a, b)
            }
            fn first_acc2(Neon, a: int64x2_t, b: int64x2_t) -> int64x2_t {
                vcopyq_laneq_s64::<1, 1>(a, b)
            }
            fn keep2(Neon, v: int64x2_t) -> int64x2_t {
                let mut v = v;
                // SAFETY: an empty block: no instruction, no memory, stack or flags.
                unsafe {
                    asm!(
                        "/* {v:v} */",
                        v = inout(vreg) v,
                        options(pure, nomem, nostack, preserves_flags)
                    );
                };
                v
            }
            fn one2(Neon, x: i32) -> int32x2_t { vdup_n_s32(x) }
            fn get2(Neon, v: int32x2_t) -> [i32; 2] {
                [vget_lane_s32::<0>(v), vget_lane_s32::<1>(v)]
            }
            /// The `i16` 0 to 3 (4 to 7 if `high`) as `i32`, each in the lane A of a vector.
            fn spread2(Neon, v: int16x8_t, high: bool) -> [int32x2_t; 4] {
                let w = if high { vmovl_high_s16(v) } else { vmovl_s16(vget_low_s16(v)) };
                [
                    vget_low_s32(w),
                    vdup_laneq_s32::<1>(w),
                    vget_high_s32(w),
                    vdup_laneq_s32::<3>(w),
                ]
            }
            /// The lanes B of eight vectors as `i16`.
            fn gather2(Neon, y: [int32x2_t; 8]) -> int16x8_t {
                let [a, b, c, d, e, f, g, h] = y;
                let lo = vcombine_s32(vzip2_s32(a, b), vzip2_s32(c, d));
                let hi = vcombine_s32(vzip2_s32(e, f), vzip2_s32(g, h));
                vcombine_s16(vmovn_s32(lo), vmovn_s32(hi))
            }
            /// `[r0, i0, r1, i1, ..]` to `[r0, r1, .., i0, i1, ..]`.
            fn split2(Neon, v: int16x8_t) -> int16x8_t {
                vcombine_s16(vget_low_s16(vuzp1q_s16(v, v)), vget_low_s16(vuzp2q_s16(v, v)))
            }
            /// The lanes B of four pairs of vectors (`re`, `im`) as `[r0, i0, r1, i1, ..]`.
            fn gather2c(Neon, re: [int32x2_t; 4], im: [int32x2_t; 4]) -> int16x8_t {
                let [a, b, c, d] = re;
                let r = vcombine_s32(vzip2_s32(a, b), vzip2_s32(c, d));
                let [a, b, c, d] = im;
                let i = vcombine_s32(vzip2_s32(a, b), vzip2_s32(c, d));
                vcombine_s16(vmovn_s32(vzip1q_s32(r, i)), vmovn_s32(vzip2q_s32(r, i)))
            }
        }

        #[inline(always)]
        fn to_neon(t: Neon, x: &[i16; 8]) -> int16x8_t {
            i16x8::simd_from(t, *x).into()
        }

        #[inline(always)]
        fn from_neon(t: Neon, v: int16x8_t) -> [i16; 8] {
            *i16x8::simd_from(t, v)
        }

        impl Lanes for int32x2_t {
            type Tok = Neon;
            type Acc = int64x2_t;
            const SHIFT: u32 = 2;
            #[inline(always)]
            fn ab(t: Neon, a: i32, b: i32) -> Self {
                ab2(t, a, b)
            }
            #[inline(always)]
            fn acc(t: Neon, a: i64) -> int64x2_t {
                acc2(t, a)
            }
            #[inline(always)]
            fn mla(t: Neon, acc: int64x2_t, x: Self, c: Self) -> int64x2_t {
                mla2(t, acc, x, c)
            }
            #[inline(always)]
            fn round(t: Neon, v: int64x2_t, _lo: Self, _hi: Self) -> Self {
                round2(t, v)
            }
            #[inline(always)]
            fn feed(t: Neon, x: Self, y: Self) -> Self {
                feed2(t, x, y)
            }
            #[inline(always)]
            fn first(t: Neon, a: Self, b: Self) -> Self {
                first2(t, a, b)
            }
            #[inline(always)]
            fn first_acc(t: Neon, a: int64x2_t, b: int64x2_t) -> int64x2_t {
                first_acc2(t, a, b)
            }
            #[inline(always)]
            fn keep(t: Neon, v: int64x2_t) -> int64x2_t {
                keep2(t, v)
            }
        }

        impl Real for int32x2_t {
            #[inline(always)]
            fn one(t: Neon, x: i16) -> Self {
                one2(t, i32::from(x))
            }
            #[inline(always)]
            fn get(t: Neon, v: Self) -> [i16; 2] {
                narrow(get2(t, v))
            }
            #[inline(always)]
            fn load(t: Neon, x: &[i16; 8], high: bool) -> [Self; 4] {
                spread2(t, to_neon(t, x), high)
            }
            #[inline(always)]
            fn store(t: Neon, y: [Self; 8]) -> [i16; 8] {
                from_neon(t, gather2(t, y))
            }
        }

        /// `CQ15`: the lanes of `re` and of `im`.
        #[derive(Clone, Copy)]
        pub(in super::super) struct Two(int32x2_t, int32x2_t);

        impl Lanes for Two {
            type Tok = Neon;
            type Acc = (int64x2_t, int64x2_t);
            const SHIFT: u32 = 2;
            #[inline(always)]
            fn ab(t: Neon, a: i32, b: i32) -> Self {
                let v = ab2(t, a, b);
                Two(v, v)
            }
            #[inline(always)]
            fn acc(t: Neon, a: i64) -> Self::Acc {
                let v = acc2(t, a);
                (v, v)
            }
            #[inline(always)]
            fn mla(t: Neon, acc: Self::Acc, x: Self, c: Self) -> Self::Acc {
                (mla2(t, acc.0, x.0, c.0), mla2(t, acc.1, x.1, c.1))
            }
            #[inline(always)]
            fn round(t: Neon, v: Self::Acc, _lo: Self, _hi: Self) -> Self {
                Two(round2(t, v.0), round2(t, v.1))
            }
            #[inline(always)]
            fn feed(t: Neon, x: Self, y: Self) -> Self {
                Two(feed2(t, x.0, y.0), feed2(t, x.1, y.1))
            }
            #[inline(always)]
            fn first(t: Neon, a: Self, b: Self) -> Self {
                Two(first2(t, a.0, b.0), first2(t, a.1, b.1))
            }
            #[inline(always)]
            fn first_acc(t: Neon, a: Self::Acc, b: Self::Acc) -> Self::Acc {
                (first_acc2(t, a.0, b.0), first_acc2(t, a.1, b.1))
            }
            #[inline(always)]
            fn keep(t: Neon, v: Self::Acc) -> Self::Acc {
                (keep2(t, v.0), keep2(t, v.1))
            }
        }

        impl Complex for Two {
            #[inline(always)]
            fn one(t: Neon, x: [i16; 2]) -> Self {
                Two(one2(t, i32::from(x[0])), one2(t, i32::from(x[1])))
            }
            #[inline(always)]
            fn get(t: Neon, v: Self) -> [[i16; 2]; 2] {
                let ([ra, rb], [ia, ib]) = (narrow(get2(t, v.0)), narrow(get2(t, v.1)));
                [[ra, ia], [rb, ib]]
            }
            #[inline(always)]
            fn set(t: Neon, a: [i16; 2], b: [i16; 2]) -> Self {
                let w = |a: i16, b: i16| ab2(t, i32::from(a), i32::from(b));
                Two(w(a[0], b[0]), w(a[1], b[1]))
            }
            #[inline(always)]
            fn load(t: Neon, x: &[i16; 8]) -> [Self; 4] {
                let v = split2(t, to_neon(t, x));
                let ([r0, r1, r2, r3], [i0, i1, i2, i3]) =
                    (spread2(t, v, false), spread2(t, v, true));
                [Two(r0, i0), Two(r1, i1), Two(r2, i2), Two(r3, i3)]
            }
            #[inline(always)]
            fn store(t: Neon, y: [Self; 4]) -> [i16; 8] {
                let [a, b, c, d] = y;
                from_neon(t, gather2c(t, [a.0, b.0, c.0, d.0], [a.1, b.1, c.1, d.1]))
            }
        }
    }
}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    //! The integer paths of every x86 level the host has, each against the fallback: the facade
    //! tests reach only the path of the host's own level (not `Two` of SSE4.2 on an AVX2 host).

    extern crate std;

    use std::vec;
    use std::vec::Vec;

    use core::arch::x86_64::{__m128i, __m256i};
    use sdrk_core::kernels::biquad::{Biquad, BiquadReal, BiquadRealState, BiquadState};
    use sdrk_core::{CQ15, Fallback, Q15};

    use super::int;

    /// A small linear congruential generator.
    fn next(s: &mut u32) -> i16 {
        *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*s >> 16) as i16
    }

    /// Cascades of one to five sections: the extremes of Q2.14 (`a = -2.0`, raw -32768, whose
    /// negation needs 17 bits), a stable low-pass and random (unstable, saturating) sections.
    fn cascades(seed: &mut u32) -> Vec<Vec<[i16; 5]>> {
        let edge = [i16::MIN, i16::MAX, 0, 1, -1];
        let mut out = vec![
            vec![[i16::MIN; 5]; 2],
            vec![[i16::MAX; 5]; 2],
            vec![[i16::MIN, i16::MAX, i16::MIN, i16::MIN, i16::MAX]; 3],
            vec![
                [1277, 2554, 1277, -21642, 10367],
                [1014, 2028, 1014, -17180, 4852],
            ],
        ];
        for n in 1..=5 {
            out.push((0..n).map(|_| [(); 5].map(|()| next(seed))).collect());
            out.push(
                (0..n)
                    .map(|_| [(); 5].map(|()| edge[usize::from(next(seed).unsigned_abs()) % 5]))
                    .collect(),
            );
        }
        out
    }

    /// The input: the extremes, silence and random values.
    fn input(seed: &mut u32, n: usize) -> Vec<i16> {
        (0..n)
            .map(|i| match i % 7 {
                0 => i16::MIN,
                1 => i16::MAX,
                2 => 0,
                _ => next(seed),
            })
            .collect()
    }

    /// The calls of a run: lengths around the block of sixteen steps and the vector tails.
    const CALLS: [usize; 9] = [0, 1, 2, 3, 15, 16, 17, 33, 300];

    fn check_q15<L: int::Real>(t: L::Tok, coefs: &[[i16; 5]], x: &[i16]) {
        let n = coefs.len();
        let mut sw =
            BiquadRealState::<Q15, _, _>::new(coefs.to_vec(), vec![Q15(0); 4 * n]).unwrap();
        let mut hist = vec![0i16; 4 * n];
        let mut at = 0;
        for len in CALLS {
            let x = &x[at..at + len];
            at += len;
            let xq: Vec<Q15> = x.iter().map(|&v| Q15(v)).collect();
            let mut want = vec![Q15(0); len];
            Fallback.biquad_real(&mut sw, &xq, &mut want).unwrap();
            let mut got = vec![0i16; len];
            int::cascade_q15::<L>(t, coefs, &mut hist, x, &mut got);
            let want: Vec<i16> = want.iter().map(|v| v.0).collect();
            assert_eq!(want, got, "{coefs:?} len {len}");
            let wh: Vec<i16> = sw.history().iter().map(|v| v.0).collect();
            assert_eq!(wh, hist, "{coefs:?} len {len}");
        }
    }

    fn check_cq15<L: int::Complex>(t: L::Tok, coefs: &[[i16; 5]], x: &[i16]) {
        let n = coefs.len();
        let mut sw =
            BiquadState::<CQ15, _, _>::new(coefs.to_vec(), vec![CQ15::default(); 4 * n]).unwrap();
        let mut hist = vec![[0i16; 2]; 4 * n];
        let mut at = 0;
        for len in CALLS {
            let x: Vec<[i16; 2]> = x[at..at + 2 * len]
                .chunks_exact(2)
                .map(|c| [c[0], c[1]])
                .collect();
            at += 2 * len;
            let xq: Vec<CQ15> = x
                .iter()
                .map(|&[re, im]| CQ15 {
                    re: Q15(re),
                    im: Q15(im),
                })
                .collect();
            let mut want = vec![CQ15::default(); len];
            Fallback.biquad(&mut sw, &xq, &mut want).unwrap();
            let mut got = vec![[0i16; 2]; len];
            int::cascade_cq15::<L>(t, coefs, &mut hist, &x, &mut got);
            let want: Vec<[i16; 2]> = want.iter().map(|v| [v.re.0, v.im.0]).collect();
            assert_eq!(want, got, "{coefs:?} len {len}");
            let wh: Vec<[i16; 2]> = sw.history().iter().map(|v| [v.re.0, v.im.0]).collect();
            assert_eq!(wh, hist, "{coefs:?} len {len}");
        }
    }

    #[test]
    fn the_integer_lanes_of_every_x86_level_give_the_bits_of_the_fallback() {
        let level = crate::level::current();
        let mut seed = 0x1357_9bdf;
        let total: usize = CALLS.iter().sum();
        for coefs in cascades(&mut seed) {
            let x = input(&mut seed, 2 * total);
            if let Some(t) = level.as_sse4_2() {
                check_q15::<__m128i>(t, &coefs, &x);
                check_q15::<int::x86::Sse>(t, &coefs, &x);
                check_cq15::<int::x86::Two>(t, &coefs, &x);
            }
            if let Some(t) = level.as_avx2() {
                check_cq15::<__m256i>(t, &coefs, &x);
            }
        }
    }
}
