//! Fallback implementation of `biquad` and `biquad_real`.
//!
//! The cascade runs section by section over the whole buffer (the first section reads the
//! input, the others filter the output in place), so the state of a section stays in registers
//! for the length of the buffer. The result is the same as running every sample through all
//! sections: a section only depends on its own input stream. The `f32` states are flushed on
//! the schedule of [`FLUSH_PERIOD`], so the samples between two flushes run in a loop of a fixed
//! length without a test.

// The per-sample steps and the loops between two flushes must be inlined into one loop body
// for the state to stay in registers; plain `#[inline]` leaves that to the heuristics.
#![allow(clippy::inline_always)]

use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::biquad::{
    Biquad, BiquadReal, BiquadRealState, BiquadSample, BiquadState, FLUSH_PERIOD,
};
use crate::kernels::check_len;
use crate::num::flush_denormal;
use crate::types::{C32, CQ15, Q15, Sample};

/// One section of an `f32` sample type (`f32`, `C32`): the transposed direct form II with the
/// flush schedule of [`FLUSH_PERIOD`].
trait F32Section: Sample + BiquadSample<Coef = f32> {
    /// `[s1, s2]` of every component.
    type State: Copy;

    fn load(hist: &[Self]) -> Option<Self::State>;
    fn store(state: &Self::State, hist: &mut [Self]);
    fn step(c: &[f32; 5], state: &mut Self::State, x: Self) -> Self;
    /// [`step`](Self::step) with the output written to `o` before the state update (the same
    /// operations in the same order, so the same bits). Stored first, the output register is
    /// free for the last product of the update, which saves a register copy per sample on
    /// two-operand instruction sets (SSE); the default stores after the step.
    #[inline(always)]
    fn step_store(c: &[f32; 5], state: &mut Self::State, x: Self, o: &mut Self) {
        *o = Self::step(c, state, x);
    }
    fn flush(state: &mut Self::State);
    /// Whether two sections run sample by sample (each sample through both, `f32`) or one
    /// after the other over the buffer (`C32`, whose two states per section and component
    /// leave too few registers for a pair on x86).
    const PAIRED: bool;
}

/// Transposed direct form II of one `f32` component without the flush.
#[inline(always)]
fn df2t(c: &[f32; 5], s: &mut [f32; 2], x: f32) -> f32 {
    let y = df2t_out(c, s, x);
    df2t_update(c, s, x, y);
    y
}

/// The output `y = b0 x + s1` of [`df2t`].
#[inline(always)]
fn df2t_out(c: &[f32; 5], s: &[f32; 2], x: f32) -> f32 {
    c[0] * x + s[0]
}

/// The state update of [`df2t`] for the input `x` and the output `y`.
#[inline(always)]
fn df2t_update(c: &[f32; 5], s: &mut [f32; 2], x: f32, y: f32) {
    let [_, b1, b2, a1, a2] = *c;
    s[0] = (b1 * x + s[1]) - a1 * y;
    s[1] = b2 * x - a2 * y;
}

impl F32Section for f32 {
    type State = [f32; 2];
    const PAIRED: bool = true;

    #[inline]
    fn load(hist: &[f32]) -> Option<[f32; 2]> {
        let [s1, s2, ..] = *hist else { return None };
        Some([s1, s2])
    }

    #[inline]
    fn store(state: &[f32; 2], hist: &mut [f32]) {
        if let [h1, h2, ..] = hist {
            (*h1, *h2) = (state[0], state[1]);
        }
    }

    #[inline(always)]
    fn step(c: &[f32; 5], state: &mut [f32; 2], x: f32) -> f32 {
        df2t(c, state, x)
    }

    #[inline(always)]
    fn step_store(c: &[f32; 5], state: &mut [f32; 2], x: f32, o: &mut f32) {
        let y = df2t_out(c, state, x);
        *o = y;
        df2t_update(c, state, x, y);
    }

    #[inline(always)]
    fn flush(state: &mut [f32; 2]) {
        *state = state.map(flush_denormal);
    }
}

impl F32Section for C32 {
    type State = [[f32; 2]; 2];
    const PAIRED: bool = false;

    #[inline]
    fn load(hist: &[C32]) -> Option<[[f32; 2]; 2]> {
        let [s1, s2, ..] = *hist else { return None };
        Some([[s1.re, s2.re], [s1.im, s2.im]])
    }

    #[inline]
    fn store(state: &[[f32; 2]; 2], hist: &mut [C32]) {
        let [[r1, r2], [i1, i2]] = *state;
        if let [h1, h2, ..] = hist {
            (*h1, *h2) = (C32::new(r1, i1), C32::new(r2, i2));
        }
    }

    #[inline(always)]
    fn step(c: &[f32; 5], state: &mut [[f32; 2]; 2], x: C32) -> C32 {
        let [re, im] = state;
        C32::new(df2t(c, re, x.re), df2t(c, im, x.im))
    }

    #[inline(always)]
    fn flush(state: &mut [[f32; 2]; 2]) {
        *state = state.map(|s| s.map(flush_denormal));
    }
}

/// `f(x)` of every sample: from `input` to `output`, or in place in `output` if `input` is
/// `None`.
#[inline(always)]
fn apply<T: Copy>(input: Option<&[T]>, output: &mut [T], mut f: impl FnMut(T) -> T) {
    match input {
        Some(input) => {
            for (o, x) in output.iter_mut().zip(input) {
                *o = f(*x);
            }
        }
        None => {
            for o in output.iter_mut() {
                *o = f(*o);
            }
        }
    }
}

/// `input` split at `n` (an empty tail if it is shorter).
#[inline(always)]
fn split<T>(input: Option<&[T]>, n: usize) -> (Option<&[T]>, Option<&[T]>) {
    match input {
        Some(x) => {
            let (a, b) = x.split_at(n.min(x.len()));
            (Some(a), Some(b))
        }
        None => (None, None),
    }
}

const _: () = assert!(FLUSH_PERIOD == 16);

/// One `f32` section over the buffer with its flush schedule; `phase` is the phase of the
/// state plus the index of the section. The section flushes after the sample `lead - 1` and
/// then after every [`FLUSH_PERIOD`] samples; the samples between two flushes run without a
/// test.
#[inline(always)]
fn single_f32<T: F32Section>(
    c: &[f32; 5],
    state: &mut T::State,
    phase: usize,
    input: Option<&[T]>,
    output: &mut [T],
) {
    let lead = FLUSH_PERIOD - phase % FLUSH_PERIOD;
    if lead > output.len() {
        apply(input, output, |x| T::step(c, state, x));
        return;
    }
    let (oh, or) = output.split_at_mut(lead);
    let (xh, xr) = split(input, lead);
    apply(xh, oh, |x| T::step(c, state, x));
    T::flush(state);
    let mut os = or.chunks_exact_mut(FLUSH_PERIOD);
    let mut xs = xr.map(|x| x.chunks_exact(FLUSH_PERIOD));
    for o in &mut os {
        let x = xs.as_mut().and_then(Iterator::next);
        apply(x, o, |x| T::step(c, state, x));
        T::flush(state);
    }
    apply(xs.map(|x| x.remainder()), os.into_remainder(), |x| {
        T::step(c, state, x)
    });
}

/// Two consecutive `f32` sections A and B sample by sample (each sample through both), with
/// their flush schedules: `phase` is the phase of the state plus the index of A, which flushes
/// after the sample `lead - 1` and then after every [`FLUSH_PERIOD`] samples; B (one index
/// higher) flushes one sample earlier. The intermediate signal stays in registers.
#[inline(always)]
fn pair_f32<T: F32Section>(
    c: [&[f32; 5]; 2],
    sa: &mut T::State,
    sb: &mut T::State,
    phase: usize,
    input: Option<&[T]>,
    output: &mut [T],
) {
    let [ca, cb] = c;
    let lead = FLUSH_PERIOD - phase % FLUSH_PERIOD;
    // A sample with the tests of both flushes: `n` counts the samples of a segment whose A
    // flush is after its sample `lead - 1`.
    let checked = |sa: &mut T::State, sb: &mut T::State, n: &mut usize, x: T, lead: usize| {
        let y = T::step(ca, sa, x);
        if *n + 1 == lead {
            T::flush(sa);
        }
        let z = T::step(cb, sb, y);
        if *n + 2 == lead {
            T::flush(sb);
        }
        *n += 1;
        z
    };
    let mut n = 0;
    if lead > output.len() {
        apply(input, output, |x| checked(sa, sb, &mut n, x, lead));
        return;
    }
    let (oh, or) = output.split_at_mut(lead);
    let (xh, xr) = split(input, lead);
    apply(xh, oh, |x| checked(sa, sb, &mut n, x, lead));
    let mut os = or.chunks_exact_mut(FLUSH_PERIOD);
    match xr {
        Some(xr) => {
            for (o, x) in (&mut os).zip(xr.chunks_exact(FLUSH_PERIOD)) {
                if let (Ok(o), Ok(x)) = (<&mut [T; 16]>::try_from(o), <&[T; 16]>::try_from(x)) {
                    pair_block_f32(ca, cb, sa, sb, x, o);
                }
            }
        }
        None => {
            for o in &mut os {
                if let Ok(o) = <&mut [T; 16]>::try_from(o) {
                    let x = *o;
                    pair_block_f32(ca, cb, sa, sb, &x, o);
                }
            }
        }
    }
    let xs = xr.map(|x| {
        let whole = x.len() / FLUSH_PERIOD * FLUSH_PERIOD;
        x.split_at(whole).1
    });
    // Less than a period remains: A does not flush any more, B possibly after the sample
    // `FLUSH_PERIOD - 2` of the rest.
    let mut n = 0;
    apply(xs, os.into_remainder(), |x| {
        checked(sa, sb, &mut n, x, FLUSH_PERIOD)
    });
}

/// Sixteen samples through two `f32` sections A and B whose flushes are after the last sample
/// (A) and after the one before (B).
#[inline(always)]
fn pair_block_f32<T: F32Section>(
    ca: &[f32; 5],
    cb: &[f32; 5],
    sa: &mut T::State,
    sb: &mut T::State,
    x: &[T; 16],
    o: &mut [T; 16],
) {
    let mut f = |x: T, o: &mut T| {
        let y = T::step(ca, sa, x);
        T::step_store(cb, sb, y, o);
    };
    let (x15, x1) = x.split_at(15);
    let (o15, o1) = o.split_at_mut(15);
    // Three steps of five samples: a loop per sample would cost a counter and a branch per
    // sample, a sixth of the work.
    for (o, x) in o15.chunks_exact_mut(5).zip(x15.chunks_exact(5)) {
        for (o, x) in o.iter_mut().zip(x) {
            f(*x, o);
        }
    }
    // B flushes after its sample 14, which is the same as before its step of sample 15 and so
    // after the step of A (the two states are independent): both flushes sit between the two
    // steps of the last sample, where the four state values are flushed together. This sample
    // stores after the update: with `step_store` here aarch64 needs 6 more instructions per
    // block and x86 saves only one.
    for (o, x) in o1.iter_mut().zip(x1) {
        let y = T::step(ca, sa, *x);
        T::flush(sa);
        T::flush(sb);
        *o = T::step(cb, sb, y);
    }
}

fn cascade_f32<T: F32Section>(
    coefs: &[[f32; 5]],
    hist: &mut [T],
    phase: usize,
    input: &[T],
    output: &mut [T],
) -> Result<()> {
    check_len(input.len(), output.len())?;
    let internal = || Error::Internal("biquad history");
    let per = if T::PAIRED { 2 } else { 1 };
    for (j, (c, h)) in coefs.chunks(per).zip(hist.chunks_mut(4 * per)).enumerate() {
        let ph = (phase + per * j) % FLUSH_PERIOD;
        let src = (j == 0).then_some(input);
        match c {
            [a, b] => {
                let (ha, hb) = h.split_at_mut(4.min(h.len()));
                let mut sa = T::load(ha).ok_or_else(internal)?;
                let mut sb = T::load(hb).ok_or_else(internal)?;
                pair_f32([a, b], &mut sa, &mut sb, ph, src, output);
                T::store(&sa, ha);
                T::store(&sb, hb);
            }
            [a] => {
                let mut sa = T::load(h).ok_or_else(internal)?;
                single_f32(a, &mut sa, ph, src, output);
                T::store(&sa, h);
            }
            _ => return Err(internal()),
        }
    }
    Ok(())
}

/// The `Q15` and `CQ15` cascade for targets with 64-bit integer registers (64-bit targets and
/// wasm32): the state and the coefficients as `i64`, both components of a `CQ15` sample in one
/// pass, and the sections of `Q15` in pairs (each sample through both).
#[cfg_attr(
    all(target_pointer_width = "32", not(target_family = "wasm")),
    allow(dead_code)
)]
mod q15_wide {
    use super::{BiquadSample, CQ15, Error, Q15, Result, Sample, apply, check_len, split};

    /// The rounding constant of `round_shift(acc, 14)`.
    const ROUND: i64 = 1 << 13;

    /// One Q15 component of one section: direct form I with the history `x1, x2, y1, y2` and two
    /// carried partial sums, `s = b1 x1 + b2 x2 - a1 y1 - a2 y2 + 2^13` (the part of the next
    /// accumulator that is known before the next input) and `q = b2 x1 - a2 y1 + 2^13` (the part
    /// of the `s` after it that is known before the next output).
    ///
    /// `y = sat16((b0 x + s) >> 14)` is `sat16(round_shift(b0 x + b1 x1 + b2 x2 - a1 y1 - a2 y2,
    /// 14))` of the specification: the five products of 16-bit values sum to less than 2^33 in
    /// magnitude, every sum is exact in `i64`, so the grouping does not change the value. It
    /// changes the dependency chain from one output to the next to one product, a subtraction,
    /// the addition of `b0 x` and the shift: the next `s` is `(b1 x + q) - a1 y`, with `y` in a
    /// single term. The compiler regroups the terms of one expression (written as one sum, it
    /// adds `a1 y1 + a2 y2` first and the rounding constant after the subtraction, three more
    /// steps in the chain), but not across the iterations of the loop, so the carried sums keep
    /// their grouping.
    ///
    /// The step needs only `s` and `q`. The history `x1, x2, y1, y2` is the caller's state; it
    /// is kept up to date only by the steps of the last two samples of a buffer
    /// ([`Df1::step_tracked`]), so that the loop does not move it every sample (on wasm32 every
    /// move is an instruction).
    #[derive(Clone, Copy)]
    pub(super) struct Df1 {
        x1: i64,
        x2: i64,
        y1: i64,
        y2: i64,
        s: i64,
        q: i64,
    }

    impl Df1 {
        #[inline(always)]
        fn new(c: &[i64; 5], [x1, x2, y1, y2]: [i64; 4]) -> Df1 {
            let [_, b1, b2, a1, a2] = *c;
            Df1 {
                x1,
                x2,
                y1,
                y2,
                s: b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2 + ROUND,
                q: b2 * x1 - a2 * y1 + ROUND,
            }
        }

        #[inline(always)]
        fn history(&self) -> [i64; 4] {
            [self.x1, self.x2, self.y1, self.y2]
        }

        /// One sample; the history is left behind (see [`Df1::step_tracked`]).
        #[inline(always)]
        fn step(&mut self, c: &[i64; 5], x: i64) -> i64 {
            let [b0, b1, b2, a1, a2] = *c;
            let y = ((b0 * x + self.s) >> 14).clamp(-32768, 32767);
            self.s = (b1 * x + self.q) - a1 * y;
            self.q = b2 * x - a2 * y + ROUND;
            y
        }

        /// [`Df1::step`] that also moves the history; the last two samples of a buffer take it,
        /// so that the history is that of the last two samples afterwards (or, after a shorter
        /// buffer, moved on by its samples).
        #[inline(always)]
        fn step_tracked(&mut self, c: &[i64; 5], x: i64) -> i64 {
            let y = self.step(c, x);
            self.x2 = self.x1;
            self.y2 = self.y1;
            self.x1 = x;
            self.y1 = y;
            y
        }
    }

    /// One section of a Q15 sample type (`Q15`, `CQ15`): direct form I, the state as `i64`.
    pub(super) trait Q15Section: Sample + BiquadSample<Coef = i16> {
        /// Whether the cascade runs two sections per pass: `Q15` has one dependency chain per
        /// section, so a pair keeps two in flight; `CQ15` already has two (its components).
        const PAIRED: bool;
        type State: Copy;

        fn load(c: &[i64; 5], hist: &[Self]) -> Option<Self::State>;
        fn store(state: &Self::State, hist: &mut [Self]);
        /// One sample; the history is complete only after [`Q15Section::step_tracked`] of the
        /// last two samples.
        fn step(c: &[i64; 5], state: &mut Self::State, x: Self) -> Self;
        /// One sample that moves the history (the last two samples of a buffer).
        fn step_tracked(c: &[i64; 5], state: &mut Self::State, x: Self) -> Self;
    }

    /// An `i64` that holds an `i16` (the output of `sat16`).
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation, reason = "the value is within i16")]
    fn to_i16(v: i64) -> i16 {
        v as i16
    }

    impl Q15Section for Q15 {
        const PAIRED: bool = true;
        type State = Df1;

        #[inline]
        fn load(c: &[i64; 5], hist: &[Q15]) -> Option<Df1> {
            let [x1, x2, y1, y2] = *hist else { return None };
            Some(Df1::new(c, [x1, x2, y1, y2].map(|v| i64::from(v.0))))
        }

        #[inline]
        fn store(state: &Df1, hist: &mut [Q15]) {
            for (h, v) in hist.iter_mut().zip(state.history()) {
                *h = Q15(to_i16(v));
            }
        }

        #[inline(always)]
        fn step(c: &[i64; 5], state: &mut Df1, x: Q15) -> Q15 {
            Q15(to_i16(state.step(c, i64::from(x.0))))
        }

        #[inline(always)]
        fn step_tracked(c: &[i64; 5], state: &mut Df1, x: Q15) -> Q15 {
            Q15(to_i16(state.step_tracked(c, i64::from(x.0))))
        }
    }

    impl Q15Section for CQ15 {
        const PAIRED: bool = false;
        type State = [Df1; 2];

        #[inline]
        fn load(c: &[i64; 5], hist: &[CQ15]) -> Option<[Df1; 2]> {
            let [x1, x2, y1, y2] = *hist else { return None };
            let h = [x1, x2, y1, y2];
            Some([
                Df1::new(c, h.map(|v| i64::from(v.re.0))),
                Df1::new(c, h.map(|v| i64::from(v.im.0))),
            ])
        }

        #[inline]
        fn store(state: &[Df1; 2], hist: &mut [CQ15]) {
            let [re, im] = state;
            for ((h, r), i) in hist.iter_mut().zip(re.history()).zip(im.history()) {
                *h = CQ15 {
                    re: Q15(to_i16(r)),
                    im: Q15(to_i16(i)),
                };
            }
        }

        #[inline(always)]
        fn step(c: &[i64; 5], state: &mut [Df1; 2], x: CQ15) -> CQ15 {
            let [re, im] = state;
            CQ15 {
                re: Q15(to_i16(re.step(c, i64::from(x.re.0)))),
                im: Q15(to_i16(im.step(c, i64::from(x.im.0)))),
            }
        }

        #[inline(always)]
        fn step_tracked(c: &[i64; 5], state: &mut [Df1; 2], x: CQ15) -> CQ15 {
            let [re, im] = state;
            CQ15 {
                re: Q15(to_i16(re.step_tracked(c, i64::from(x.re.0)))),
                im: Q15(to_i16(im.step_tracked(c, i64::from(x.im.0)))),
            }
        }
    }

    /// `f(x, tracked)` of every sample, `tracked` being `true` for the last two only: from
    /// `input` to `output`, or in place in `output` if `input` is `None`.
    #[inline(always)]
    fn apply_tracked<T: Copy>(
        input: Option<&[T]>,
        output: &mut [T],
        mut f: impl FnMut(T, bool) -> T,
    ) {
        let (head, tail) = output.split_at_mut(output.len().saturating_sub(2));
        let (xh, xt) = split(input, head.len());
        apply(xh, head, |x| f(x, false));
        apply(xt, tail, |x| f(x, true));
    }

    pub(super) fn cascade_q15<T: Q15Section>(
        coefs: &[[i16; 5]],
        hist: &mut [T],
        input: &[T],
        output: &mut [T],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let internal = || Error::Internal("biquad history");
        let per = if T::PAIRED { 2 } else { 1 };
        for (j, (c, h)) in coefs.chunks(per).zip(hist.chunks_mut(4 * per)).enumerate() {
            let src = (j == 0).then_some(input);
            match c {
                [a, b] => {
                    let (ca, cb) = (a.map(i64::from), b.map(i64::from));
                    let (ha, hb) = h.split_at_mut(4.min(h.len()));
                    let mut sa = T::load(&ca, ha).ok_or_else(internal)?;
                    let mut sb = T::load(&cb, hb).ok_or_else(internal)?;
                    apply_tracked(src, output, |x, tracked| {
                        if tracked {
                            let y = T::step_tracked(&ca, &mut sa, x);
                            T::step_tracked(&cb, &mut sb, y)
                        } else {
                            let y = T::step(&ca, &mut sa, x);
                            T::step(&cb, &mut sb, y)
                        }
                    });
                    T::store(&sa, ha);
                    T::store(&sb, hb);
                }
                [a] => {
                    let ca = a.map(i64::from);
                    let mut sa = T::load(&ca, h).ok_or_else(internal)?;
                    apply_tracked(src, output, |x, tracked| {
                        if tracked {
                            T::step_tracked(&ca, &mut sa, x)
                        } else {
                            T::step(&ca, &mut sa, x)
                        }
                    });
                    T::store(&sa, h);
                }
                _ => return Err(internal()),
            }
        }
        Ok(())
    }
}

/// The `Q15` and `CQ15` cascade for targets with 32-bit integer registers (Cortex-M, 32-bit
/// RISC-V): the state as `i32`, each product a 32 x 32 -> 64 bit multiply-accumulate, the
/// components of a `CQ15` sample in two passes (D28). It gives the same output as
/// [`q15_wide`] bit for bit (the arithmetic is exact in both); on 64-bit targets and wasm32 it
/// costs more instructions (the widening of every operand, two loops for `CQ15`), on the
/// Cortex-M4 a quarter to a third of [`q15_wide`].
#[cfg_attr(
    not(all(target_pointer_width = "32", not(target_family = "wasm"))),
    allow(dead_code)
)]
mod q15_narrow {
    use super::{BiquadSample, CQ15, Error, Q15, Result, Sample, apply, check_len};

    /// Direct form I of one Q15 component with Q2.14 coefficients: the output and the new
    /// `[x1, x2, y1, y2]`. `c` is `[b0, b1, b2, -a1, -a2]` (see [`q15_coefs`]), so that the sum is
    /// one chain of products added to the rounding constant; every value is an `i16` held in an
    /// `i32`, and every product of two of them is exact in `i64` (a 32 x 32 -> 64 bit
    /// multiply-accumulate on 32-bit cores, where `i64` operands cost a full 64-bit product).
    #[inline(always)]
    fn df1(c: &[i32; 5], h: [i32; 4], x: i32) -> (i32, [i32; 4]) {
        let [b0, b1, b2, na1, na2] = *c;
        let [x1, x2, y1, y2] = h;
        let mul = |u: i32, v: i32| i64::from(u) * i64::from(v);
        // `sat16(round_shift(acc, 14))` with `acc = b0 x + b1 x1 + b2 x2 - a1 y1 - a2 y2`: the five
        // products of 16-bit values sum to less than 2^33 in magnitude, so the rounding addition
        // cannot overflow, and the shifted sum is below 2^20 in magnitude, so it is exact in `i32`,
        // where the clamp is one saturation (`ssat` on Arm) or two conditional moves.
        let acc =
            (1i64 << 13) + mul(b0, x) + mul(b1, x1) + mul(b2, x2) + mul(na1, y1) + mul(na2, y2);
        let y = to_i32(acc >> 14).clamp(-32768, 32767);
        (y, [x, x1, y, y1])
    }

    /// The coefficients of [`df1`]: `[b0, b1, b2, -a1, -a2]` as `i32` (`-a` is exact there, also
    /// for `a = -2.0`, raw -32768).
    #[inline]
    fn q15_coefs(c: &[i16; 5]) -> [i32; 5] {
        let [b0, b1, b2, a1, a2] = c.map(i32::from);
        [b0, b1, b2, -a1, -a2]
    }

    /// An `i32` that holds an `i16` (the output of `sat16`).
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation, reason = "the value is within i16")]
    fn to_i16(v: i32) -> i16 {
        v as i16
    }

    /// An `i64` below 2^20 in magnitude (the shifted sum of [`df1`]), as `i32`.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation, reason = "the value is within i32")]
    fn to_i32(v: i64) -> i32 {
        v as i32
    }

    /// One section on one component over the buffer: `get` reads the component of a sample and
    /// `set` replaces it; the history `h` holds `x1, x2, y1, y2` in that component. From `input`
    /// to `output` (the other components are copied from `input`), or in place if `input` is
    /// `None`. `None` if `h` does not have four entries.
    #[inline(always)]
    fn df1_pass<T: Copy>(
        c: &[i32; 5],
        h: &mut [T],
        input: Option<&[T]>,
        output: &mut [T],
        get: impl Fn(T) -> i16,
        set: impl Fn(T, i16) -> T,
    ) -> Option<()> {
        let [x1, x2, y1, y2] = *h else { return None };
        let mut state = [x1, x2, y1, y2].map(|v| i32::from(get(v)));
        apply(input, output, |x| {
            let (y, next) = df1(c, state, i32::from(get(x)));
            state = next;
            set(x, to_i16(y))
        });
        for (e, v) in h.iter_mut().zip(state) {
            *e = set(*e, to_i16(v));
        }
        Some(())
    }

    /// A Q15 sample type (`Q15`, `CQ15`): one section of direct form I over the buffer, one
    /// component after the other (the state of one component, `x1, x2, y1, y2` as `i32`, stays in
    /// registers on 32-bit cores; both components at once would not).
    pub(super) trait Q15Section: Sample + BiquadSample<Coef = i16> {
        /// [`df1_pass`] for every component; `None` if `h` does not have four entries.
        fn section(
            c: &[i32; 5],
            h: &mut [Self],
            input: Option<&[Self]>,
            output: &mut [Self],
        ) -> Option<()>;
    }

    impl Q15Section for Q15 {
        #[inline(always)]
        fn section(
            c: &[i32; 5],
            h: &mut [Q15],
            input: Option<&[Q15]>,
            output: &mut [Q15],
        ) -> Option<()> {
            df1_pass(c, h, input, output, |v| v.0, |_, y| Q15(y))
        }
    }

    impl Q15Section for CQ15 {
        #[inline(always)]
        fn section(
            c: &[i32; 5],
            h: &mut [CQ15],
            input: Option<&[CQ15]>,
            output: &mut [CQ15],
        ) -> Option<()> {
            // The first pass copies `im` from the input; the second then runs in place.
            df1_pass(
                c,
                h,
                input,
                output,
                |v| v.re.0,
                |v, y| CQ15 { re: Q15(y), ..v },
            )?;
            df1_pass(
                c,
                h,
                None,
                output,
                |v| v.im.0,
                |v, y| CQ15 { im: Q15(y), ..v },
            )
        }
    }

    pub(super) fn cascade_q15<T: Q15Section>(
        coefs: &[[i16; 5]],
        hist: &mut [T],
        input: &[T],
        output: &mut [T],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (k, (c, h)) in coefs.iter().zip(hist.chunks_exact_mut(4)).enumerate() {
            T::section(&q15_coefs(c), h, (k == 0).then_some(input), output)
                .ok_or(Error::Internal("biquad history"))?;
        }
        Ok(())
    }
}

#[cfg(all(target_pointer_width = "32", not(target_family = "wasm")))]
use q15_narrow::cascade_q15;
#[cfg(not(all(target_pointer_width = "32", not(target_family = "wasm"))))]
use q15_wide::cascade_q15;

impl Biquad<C32> for Fallback {
    fn biquad<C, H>(
        &self,
        s: &mut BiquadState<C32, C, H>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()>
    where
        C: AsRef<[[f32; 5]]>,
        H: AsRef<[C32]> + AsMut<[C32]>,
    {
        let (coefs, hist, phase) = s.parts_phase_mut();
        cascade_f32(coefs, hist, phase, input, output)?;
        s.advance_phase(input.len());
        Ok(())
    }
}

impl Biquad<CQ15> for Fallback {
    fn biquad<C, H>(
        &self,
        s: &mut BiquadState<CQ15, C, H>,
        input: &[CQ15],
        output: &mut [CQ15],
    ) -> Result<()>
    where
        C: AsRef<[[i16; 5]]>,
        H: AsRef<[CQ15]> + AsMut<[CQ15]>,
    {
        let (coefs, hist) = s.parts_mut();
        cascade_q15(coefs, hist, input, output)
    }
}

impl BiquadReal<f32> for Fallback {
    fn biquad_real<C, H>(
        &self,
        s: &mut BiquadRealState<f32, C, H>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()>
    where
        C: AsRef<[[f32; 5]]>,
        H: AsRef<[f32]> + AsMut<[f32]>,
    {
        let (coefs, hist, phase) = s.parts_phase_mut();
        cascade_f32(coefs, hist, phase, input, output)?;
        s.advance_phase(input.len());
        Ok(())
    }
}

impl BiquadReal<Q15> for Fallback {
    fn biquad_real<C, H>(
        &self,
        s: &mut BiquadRealState<Q15, C, H>,
        input: &[Q15],
        output: &mut [Q15],
    ) -> Result<()>
    where
        C: AsRef<[[i16; 5]]>,
        H: AsRef<[Q15]> + AsMut<[Q15]>,
    {
        let (coefs, hist) = s.parts_mut();
        cascade_q15(coefs, hist, input, output)
    }
}

#[cfg(test)]
mod tests {
    //! The two Q15 cascades must agree bit for bit; only one of them runs in the kernel tests
    //! of a target.

    use std::vec;
    use std::vec::Vec;

    use super::{CQ15, Q15, q15_narrow, q15_wide};

    /// A small linear congruential generator.
    fn next(s: &mut u32) -> i16 {
        *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*s >> 16) as i16
    }

    /// Coefficient sets: the fourth-order Butterworth low-pass of the counts, the extremes of
    /// Q2.14 (including a = -2.0, raw -32768) and random values (unstable filters too: the
    /// output saturates, which both must do alike).
    fn coefs(seed: &mut u32) -> [Vec<[i16; 5]>; 4] {
        let random: Vec<[i16; 5]> = (0..3).map(|_| [(); 5].map(|()| next(seed))).collect();
        [
            vec![
                [1277, 2554, 1277, -21642, 10367],
                [1014, 2028, 1014, -17180, 4852],
            ],
            vec![[i16::MIN, i16::MAX, i16::MIN, i16::MIN, i16::MAX]],
            vec![[i16::MAX, i16::MIN, i16::MAX, i16::MAX, i16::MIN]; 2],
            random,
        ]
    }

    #[test]
    fn the_narrow_q15_cascade_equals_the_wide_one() {
        let mut seed = 0x2468_ace1;
        for c in coefs(&mut seed) {
            for full_scale in [false, true] {
                let sample = |s: &mut u32| {
                    let v = next(s);
                    if full_scale { v } else { v / 8 }
                };
                let input: Vec<Q15> = (0..301).map(|_| Q15(sample(&mut seed))).collect();
                let cinput: Vec<CQ15> = (0..301)
                    .map(|_| CQ15 {
                        re: Q15(sample(&mut seed)),
                        im: Q15(sample(&mut seed)),
                    })
                    .collect();
                let mut hist = [
                    vec![Q15::default(); 4 * c.len()],
                    vec![Q15::default(); 4 * c.len()],
                ];
                let mut chist = [
                    vec![CQ15::default(); 4 * c.len()],
                    vec![CQ15::default(); 4 * c.len()],
                ];
                // Three calls of uneven length: the history must carry over alike.
                for range in [0..1, 1..100, 100..301] {
                    let mut out = [
                        vec![Q15::default(); range.len()],
                        vec![Q15::default(); range.len()],
                    ];
                    let [hw, hn] = &mut hist;
                    let [ow, on] = &mut out;
                    q15_wide::cascade_q15(&c, hw, &input[range.clone()], ow).unwrap();
                    q15_narrow::cascade_q15(&c, hn, &input[range.clone()], on).unwrap();
                    assert_eq!(ow, on, "{c:?}");
                    let mut cout = [
                        vec![CQ15::default(); range.len()],
                        vec![CQ15::default(); range.len()],
                    ];
                    let [hw, hn] = &mut chist;
                    let [ow, on] = &mut cout;
                    q15_wide::cascade_q15(&c, hw, &cinput[range.clone()], ow).unwrap();
                    q15_narrow::cascade_q15(&c, hn, &cinput[range.clone()], on).unwrap();
                    assert_eq!(ow, on, "{c:?}");
                }
                assert_eq!(hist[0], hist[1]);
                assert_eq!(chist[0], chist[1]);
            }
        }
    }
}
