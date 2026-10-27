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
    fn flush(state: &mut Self::State);
    /// Whether two sections run sample by sample (each sample through both, `f32`) or one
    /// after the other over the buffer (`C32`, whose two states per section and component
    /// leave too few registers for a pair on x86).
    const PAIRED: bool;
}

/// Transposed direct form II of one `f32` component without the flush.
#[inline(always)]
fn df2t(c: &[f32; 5], s: &mut [f32; 2], x: f32) -> f32 {
    let [b0, b1, b2, a1, a2] = *c;
    let y = b0 * x + s[0];
    s[0] = (b1 * x + s[1]) - a1 * y;
    s[1] = b2 * x - a2 * y;
    y
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
    let mut f = |x: T| {
        let y = T::step(ca, sa, x);
        T::step(cb, sb, y)
    };
    let (x15, x1) = x.split_at(15);
    let (o15, o1) = o.split_at_mut(15);
    // Three steps of five samples: a loop per sample would cost a counter and a branch per
    // sample, a sixth of the work.
    for (o, x) in o15.chunks_exact_mut(5).zip(x15.chunks_exact(5)) {
        for (o, x) in o.iter_mut().zip(x) {
            *o = f(*x);
        }
    }
    T::flush(sb);
    for (o, x) in o1.iter_mut().zip(x1) {
        let y = T::step(ca, sa, *x);
        T::flush(sa);
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

/// Direct form I of one Q15 component with Q2.14 coefficients (as `i64`): the output and the
/// new `[x1, x2, y1, y2]`.
#[inline(always)]
fn df1(c: &[i64; 5], h: [i64; 4], x: i64) -> (i64, [i64; 4]) {
    let [b0, b1, b2, a1, a2] = *c;
    let [x1, x2, y1, y2] = h;
    let acc = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    // `sat16(round_shift(acc, 14))`: the five products of 16-bit values sum to less than
    // 2^33 in magnitude, so the rounding addition cannot overflow and `round_shift` never
    // saturates; the clamp compiles to two conditional moves instead of branches.
    let y = ((acc + (1 << 13)) >> 14).clamp(-32768, 32767);
    (y, [x, x1, y, y1])
}

/// One section of a Q15 sample type (`Q15`, `CQ15`): direct form I, the state as `i64`.
trait Q15Section: Sample + BiquadSample<Coef = i16> {
    type State: Copy;

    fn load(hist: &[Self]) -> Option<Self::State>;
    fn store(state: &Self::State, hist: &mut [Self]);
    fn step(c: &[i64; 5], state: &mut Self::State, x: Self) -> Self;
}

/// An `i64` that holds an `i16` (the output of `sat16`).
#[inline(always)]
#[allow(clippy::cast_possible_truncation, reason = "the value is within i16")]
fn to_i16(v: i64) -> i16 {
    v as i16
}

impl Q15Section for Q15 {
    type State = [i64; 4];

    #[inline]
    fn load(hist: &[Q15]) -> Option<[i64; 4]> {
        let [x1, x2, y1, y2] = *hist else { return None };
        Some([x1, x2, y1, y2].map(|v| i64::from(v.0)))
    }

    #[inline]
    fn store(state: &[i64; 4], hist: &mut [Q15]) {
        for (h, v) in hist.iter_mut().zip(state) {
            *h = Q15(to_i16(*v));
        }
    }

    #[inline(always)]
    fn step(c: &[i64; 5], state: &mut [i64; 4], x: Q15) -> Q15 {
        let (y, next) = df1(c, *state, i64::from(x.0));
        *state = next;
        Q15(to_i16(y))
    }
}

impl Q15Section for CQ15 {
    type State = [[i64; 4]; 2];

    #[inline]
    fn load(hist: &[CQ15]) -> Option<[[i64; 4]; 2]> {
        let [x1, x2, y1, y2] = *hist else { return None };
        let h = [x1, x2, y1, y2];
        Some([h.map(|v| i64::from(v.re.0)), h.map(|v| i64::from(v.im.0))])
    }

    #[inline]
    fn store(state: &[[i64; 4]; 2], hist: &mut [CQ15]) {
        let [re, im] = state;
        for ((h, r), i) in hist.iter_mut().zip(re).zip(im) {
            *h = CQ15 {
                re: Q15(to_i16(*r)),
                im: Q15(to_i16(*i)),
            };
        }
    }

    #[inline(always)]
    fn step(c: &[i64; 5], state: &mut [[i64; 4]; 2], x: CQ15) -> CQ15 {
        let [re, im] = state;
        let (yr, nr) = df1(c, *re, i64::from(x.re.0));
        let (yi, ni) = df1(c, *im, i64::from(x.im.0));
        (*re, *im) = (nr, ni);
        CQ15 {
            re: Q15(to_i16(yr)),
            im: Q15(to_i16(yi)),
        }
    }
}

fn cascade_q15<T: Q15Section>(
    coefs: &[[i16; 5]],
    hist: &mut [T],
    input: &[T],
    output: &mut [T],
) -> Result<()> {
    check_len(input.len(), output.len())?;
    let internal = || Error::Internal("biquad history");
    for (k, (c, h)) in coefs.iter().zip(hist.chunks_exact_mut(4)).enumerate() {
        let c = c.map(i64::from);
        let mut state = T::load(h).ok_or_else(internal)?;
        if k == 0 {
            for (x, o) in input.iter().zip(output.iter_mut()) {
                *o = T::step(&c, &mut state, *x);
            }
        } else {
            for o in output.iter_mut() {
                *o = T::step(&c, &mut state, *o);
            }
        }
        T::store(&state, h);
    }
    Ok(())
}

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
