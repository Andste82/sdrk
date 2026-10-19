//! The FIR family: `fir`, `fir_decimate` and the polyphase interpolator `fir_interpolate`
//! (also exposed as `resample`).
//!
//! Every output is one dot product of the taps with a window of the input. The window may reach
//! back into the filter history, so the input is seen as the virtual stream `history ++ input`.
//! The dot product runs in chunks of half a vector of complex samples. The taps are prepared once
//! per call, reversed and duplicated for the real and the imaginary lane, so the data is read in
//! memory order and every output reuses the same tap vectors. A window that reaches into the
//! history is assembled chunk by chunk with the same chunk boundaries, so an output is
//! bit-identical whichever way the input was split into blocks.
//!
//! - `fir` and `fir_decimate` prepare the taps in blocks of [`PREP_TAPS`] taps (one block for the
//!   usual filters). Longer filters add the partial dot products of the blocks in the output.
//! - The polyphase kernels (`resample`, `fir_interpolate`) prepare the taps of all branches at
//!   once; a filter that does not fit, and every `CQ15` filter with more than [`PREP_TAPS`] taps,
//!   runs on the fallback.
//!
//! `C32` results are within the contract of `docs/kernels.md` (110 dB) but not bit-identical
//! with the fallback, because the sums are formed per vector lane. `CQ15` results are exact
//! integer sums and therefore bit-identical.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::decim::{FirDecimState, FirDecimate, Resample, ResampleState};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::interp::{FirInterpState, FirInterpolate};
use sdrk_core::num::Acc64;
use sdrk_core::{C32, CQ15, Error, Fallback, Progress, Q15, Result};

use crate::Simd;
use crate::level::dispatch;
use crate::util::{check_len, mac};

/// Taps in one block of the contiguous filters (`fir`, `fir_decimate`).
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

    /// Copies `take` samples from sample `pos` on into `tmp` (the rest of `tmp` is untouched).
    #[inline(never)]
    fn assemble(&self, pos: usize, take: usize, tmp: &mut [T]) {
        for (k, dst) in tmp.chunks_exact_mut(2).take(take).enumerate() {
            let p = pos + k;
            let src = if p >= self.h {
                self.input.get(2 * (p - self.h)..)
            } else {
                self.hist.get(2 * p..)
            };
            if let (Some([re, im, ..]), [d0, d1]) = (src, dst) {
                *d0 = *re;
                *d1 = *im;
            }
        }
    }
}

/// What the three drivers need from the arithmetic of a sample type.
trait Engine {
    /// Number of tap blocks of the contiguous filter.
    fn blocks(&self) -> usize;

    /// Prepares the taps of block `b` of the contiguous filter: sample `m` of the window pairs
    /// with `h[n - 1 - (b * PREP_TAPS + m)]`.
    fn prepare_block(&mut self, b: usize);

    /// Prepares the taps of all branches of a polyphase filter `h[phase + up * i]`.
    fn prepare_branches(&mut self, up: usize);

    /// Dot product of the prepared taps at `tap_offset` (floats) with the `n_taps` samples that
    /// begin at sample `start` of the virtual stream; the result is the output with index `j`
    /// (or is added to it when `add` is set).
    fn dot(&mut self, j: usize, start: usize, n_taps: usize, tap_offset: usize, add: bool);
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
// C32

/// Sums the even and the odd lanes (the real and imaginary parts) of an accumulator.
#[inline(always)]
fn hsum_pairs<S: SimdToken>(acc: S::f32s) -> (f32, f32) {
    let mut v = acc;
    match S::f32s::LEN {
        4 => v = v + v.slide::<2>(v),
        8 => {
            v = v + v.slide::<4>(v);
            v = v + v.slide::<2>(v);
        }
        16 => {
            v = v + v.slide::<8>(v);
            v = v + v.slide::<4>(v);
            v = v + v.slide::<2>(v);
        }
        _ => {}
    }
    match v.as_slice() {
        [re, im, ..] => (*re, *im),
        _ => (0.0, 0.0),
    }
}

/// Storage for the prepared taps; lives in the entry function so that it is not moved.
type TapStorage = [f32; 2 * CAPACITY];

struct F32Engine<'a, S: SimdToken> {
    simd: S,
    taps: &'a [f32],
    win: Win<'a, f32>,
    out: &'a mut [f32],
    /// Reversed taps, each repeated for the real and the imaginary lane.
    td: &'a mut TapStorage,
}

impl<'a, S: SimdToken> F32Engine<'a, S> {
    #[inline(always)]
    fn new(
        simd: S,
        taps: &'a [f32],
        storage: &'a mut TapStorage,
        hist: &'a [f32],
        input: &'a [f32],
        out: &'a mut [f32],
    ) -> Self {
        Self {
            simd,
            taps,
            win: Win::new(hist, input),
            out,
            td: storage,
        }
    }

    /// One chunk of the window that begins at sample `pos`: `g` samples if `take == g`, else the
    /// first `take` samples and zeros.
    #[inline(always)]
    fn chunk(&self, pos: usize, take: usize) -> S::f32s {
        let simd = self.simd;
        let l = S::f32s::LEN;
        if take == l / 2
            && let Some(d) = self.win.direct(pos, l)
        {
            return S::f32s::from_slice(simd, d);
        }
        let mut tmp = [0f32; 32];
        self.win.assemble(pos, take, &mut tmp);
        S::f32s::from_slice(simd, tmp.get(..l).unwrap_or(&[0.0; 32]))
    }

    /// The four accumulators over the `full` whole chunks when the window lies inside the input
    /// (no calls in this path, so the accumulators stay in registers). Chunk `c` goes to
    /// accumulator `c % 4`; four short chains of additions are more accurate than one long one.
    #[inline(always)]
    fn full_chunks_direct(&self, start: usize, full: usize, td: &[f32]) -> Option<[S::f32s; 4]> {
        let simd = self.simd;
        let l = S::f32s::LEN;
        let zero = S::f32s::splat(simd, 0.0);
        let o = 2 * start.checked_sub(self.win.h)?;
        let win = self.win.input.get(o..o + l * full)?;
        let taps = td.get(..l * full)?;
        let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
        let mut data = win.chunks_exact(4 * l);
        let mut tap = taps.chunks_exact(4 * l);
        for (d, t) in (&mut data).zip(&mut tap) {
            let (d01, d23) = d.split_at(2 * l);
            let (t01, t23) = t.split_at(2 * l);
            let ((d0, d1), (d2, d3)) = (d01.split_at(l), d23.split_at(l));
            let ((t0, t1), (t2, t3)) = (t01.split_at(l), t23.split_at(l));
            a0 = mac::<S>(
                S::f32s::from_slice(simd, d0),
                S::f32s::from_slice(simd, t0),
                a0,
            );
            a1 = mac::<S>(
                S::f32s::from_slice(simd, d1),
                S::f32s::from_slice(simd, t1),
                a1,
            );
            a2 = mac::<S>(
                S::f32s::from_slice(simd, d2),
                S::f32s::from_slice(simd, t2),
                a2,
            );
            a3 = mac::<S>(
                S::f32s::from_slice(simd, d3),
                S::f32s::from_slice(simd, t3),
                a3,
            );
        }
        for (i, (d, t)) in data
            .remainder()
            .chunks_exact(l)
            .zip(tap.remainder().chunks_exact(l))
            .enumerate()
        {
            let (d, t) = (S::f32s::from_slice(simd, d), S::f32s::from_slice(simd, t));
            match i {
                0 => a0 = mac::<S>(d, t, a0),
                1 => a1 = mac::<S>(d, t, a1),
                _ => a2 = mac::<S>(d, t, a2),
            }
        }
        Some([a0, a1, a2, a3])
    }

    /// The same accumulators for a window that reaches into the history.
    #[inline(always)]
    fn full_chunks_assembled(&self, start: usize, full: usize, td: &[f32]) -> [S::f32s; 4] {
        let simd = self.simd;
        let l = S::f32s::LEN;
        let g = l / 2;
        let zero = S::f32s::splat(simd, 0.0);
        let mut acc = [zero; 4];
        for c in 0..full {
            let t = td
                .get(c * l..c * l + l)
                .map_or(zero, |t| S::f32s::from_slice(simd, t));
            let d = self.chunk(start + c * g, g);
            if let Some(a) = acc.get_mut(c % 4) {
                *a = mac::<S>(d, t, *a);
            }
        }
        acc
    }

    /// The dot product with the prepared taps at `td_off`. Whole chunk `c` goes to accumulator
    /// `c % 4`; the partial last chunk is added to the one it would belong to. Both ways of
    /// reading the window give the same bits.
    #[inline(always)]
    fn dot_prepared(&self, start: usize, n_taps: usize, td_off: usize) -> S::f32s {
        let simd = self.simd;
        let l = S::f32s::LEN;
        let g = l / 2;
        let (full, rem) = (n_taps / g, n_taps % g);
        let zero = S::f32s::splat(simd, 0.0);
        let td = self.td.get(td_off..).unwrap_or(&[]);
        let [a0, a1, a2, a3] = match self.full_chunks_direct(start, full, td) {
            Some(acc) => acc,
            None => self.full_chunks_assembled(start, full, td),
        };
        let mut part = [zero; 4];
        if rem > 0 {
            let t = td
                .get(full * l..full * l + l)
                .map_or(zero, |t| S::f32s::from_slice(simd, t));
            if let Some(p) = part.get_mut(full % 4) {
                *p = self.chunk(start + full * g, rem) * t;
            }
        }
        let [p0, p1, p2, p3] = part;
        ((a0 + p0) + (a1 + p1)) + ((a2 + p2) + (a3 + p3))
    }
}

impl<S: SimdToken> Engine for F32Engine<'_, S> {
    fn blocks(&self) -> usize {
        self.taps.len().div_ceil(PREP_TAPS)
    }

    #[inline(always)]
    fn prepare_block(&mut self, b: usize) {
        let n = self.taps.len();
        let first = b * PREP_TAPS;
        let count = n.saturating_sub(first).min(PREP_TAPS);
        self.td.fill(0.0);
        // Window sample `m` pairs with `h[n - 1 - m]`, so the block holds the taps in reverse.
        let src = self.taps.get(n - first - count..n - first).unwrap_or(&[]);
        for (pair, &h) in self.td.chunks_exact_mut(2).zip(src.iter().rev()) {
            pair.fill(h);
        }
    }

    #[inline(always)]
    fn prepare_branches(&mut self, up: usize) {
        let n = self.taps.len();
        let geom = Branches {
            n,
            up,
            g: S::f32s::LEN / 2,
        };
        self.td.fill(0.0);
        for phase in 0..up.min(n) {
            let t = geom.len(phase);
            let off = geom.offset(phase);
            // Window sample `m` pairs with `h[phase + up * (t - 1 - m)]`.
            for m in 0..t {
                let h = self
                    .taps
                    .get(phase + up * (t - 1 - m))
                    .copied()
                    .unwrap_or(0.0);
                if let Some([re, im]) = self.td.get_mut(off + 2 * m..off + 2 * m + 2) {
                    *re = h;
                    *im = h;
                }
            }
        }
    }

    #[inline(always)]
    fn dot(&mut self, j: usize, start: usize, n_taps: usize, tap_offset: usize, add: bool) {
        let acc = self.dot_prepared(start, n_taps, tap_offset);
        let (re, im) = hsum_pairs::<S>(acc);
        if let Some([o_re, o_im]) = self.out.get_mut(2 * j..2 * j + 2) {
            if add {
                *o_re += re;
                *o_im += im;
            } else {
                *o_re = re;
                *o_im = im;
            }
        }
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

impl<S: SimdToken> Engine for Q15Engine<'_, S> {
    fn blocks(&self) -> usize {
        1
    }

    #[inline(always)]
    fn prepare_block(&mut self, _b: usize) {
        self.td.fill(0);
        for (pair, &h) in self.td.chunks_exact_mut(2).zip(self.taps.iter().rev()) {
            pair.fill(h);
        }
    }

    #[inline(always)]
    fn prepare_branches(&mut self, up: usize) {
        let n = self.taps.len();
        let geom = Branches {
            n,
            up,
            g: S::i16s::LEN / 2,
        };
        self.td.fill(0);
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

    #[inline(always)]
    fn dot(&mut self, j: usize, start: usize, n_taps: usize, tap_offset: usize, _add: bool) {
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
                self.win.assemble(pos, take, &mut tmp);
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

// ---------------------------------------------------------------------------------------------
// Drivers (independent of the sample type)

/// `hist` becomes the last `hist.len()` samples of `hist ++ input`.
fn update_history<T: Copy>(hist: &mut [T], input: &[T]) -> Result<()> {
    let keep = hist.len();
    if input.len() >= keep {
        let tail = input
            .get(input.len() - keep..)
            .ok_or(Error::Internal("history update"))?;
        hist.copy_from_slice(tail);
    } else {
        hist.rotate_left(input.len());
        let dst = hist
            .get_mut(keep - input.len()..)
            .ok_or(Error::Internal("history update"))?;
        dst.copy_from_slice(input);
    }
    Ok(())
}

/// Outputs `j = 0..count` of a contiguous filter; the window of output `j` begins at sample
/// `first + j * step` (the history holds `n - 1` samples, so that is its oldest sample).
#[inline(always)]
fn drive_contiguous<E: Engine>(e: &mut E, n_taps: usize, count: usize, first: usize, step: usize) {
    for b in 0..e.blocks() {
        e.prepare_block(b);
        let skip = b * PREP_TAPS;
        let len = n_taps.saturating_sub(skip).min(PREP_TAPS);
        for j in 0..count {
            e.dot(j, first + j * step + skip, len, 0, b > 0);
        }
    }
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

/// The polyphase loop of the fallback, with the dot product delegated to the engine. Returns
/// `(consumed, produced)`.
#[inline(always)]
fn drive_polyphase<E: Engine>(
    e: &mut E,
    geom: Branches,
    hist_len: usize,
    down: usize,
    q: &mut i64,
    in_len: usize,
    out_len: usize,
) -> Result<(usize, usize)> {
    let internal = || Error::Internal("resampler window");
    e.prepare_branches(geom.up);
    let (l, m) = (geom.up as i64, down as i64);
    let (mut consumed, mut produced) = (0usize, 0usize);
    loop {
        if *q < 0 {
            if produced == out_len {
                break;
            }
            let phase = usize::try_from(*q + l).map_err(|_| internal())?;
            let count = geom.len(phase);
            // Newest sample of the window is virtual index `hist_len + consumed - 1`.
            let start = (hist_len + consumed)
                .checked_sub(count)
                .ok_or_else(internal)?;
            e.dot(produced, start, count, geom.offset(phase), false);
            produced += 1;
            *q += m;
        } else {
            if consumed == in_len {
                break;
            }
            consumed += 1;
            *q -= l;
        }
    }
    Ok((consumed, produced))
}

// ---------------------------------------------------------------------------------------------
// Entry points

#[inline(always)]
fn fir_c32<S: SimdToken>(simd: S, taps: &[f32], hist: &[f32], input: &[f32], out: &mut [f32]) {
    let len = input.len() / 2;
    let mut storage: TapStorage = [0f32; 2 * CAPACITY];
    let n = taps.len();
    let mut e = F32Engine::new(simd, taps, &mut storage, hist, input, out);
    drive_contiguous(&mut e, n, len, 0, 1);
}

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
    let mut storage: TapStorage = [0f32; 2 * CAPACITY];
    let n = taps.len();
    let mut e = F32Engine::new(simd, taps, &mut storage, hist, input, out);
    drive_contiguous(&mut e, n, count, first, factor);
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
) -> Result<Option<(usize, usize)>> {
    let (in_len, out_len) = (input.len() / 2, out.len() / 2);
    let geom = Branches {
        n: taps.len(),
        up,
        g: S::f32s::LEN / 2,
    };
    if !geom.fits() {
        return Ok(None);
    }
    let hist_len = hist.len() / 2;
    let mut storage: TapStorage = [0f32; 2 * CAPACITY];
    let mut e = F32Engine::new(simd, taps, &mut storage, hist, input, out);
    drive_polyphase(&mut e, geom, hist_len, down, q, in_len, out_len).map(Some)
}

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
    let (in_len, out_len) = (input.len() / 2, out.len() / 2);
    let geom = Branches {
        n: taps.len(),
        up,
        g: S::i16s::LEN / 2,
    };
    if !geom.fits() {
        return Ok(None);
    }
    let hist_len = hist.len() / 2;
    let mut storage: TapStorageQ15 = [0i16; 2 * CAPACITY];
    let mut e = Q15Engine {
        simd,
        taps,
        win: Win::new(hist, input),
        out,
        td: &mut storage,
    };
    drive_polyphase(&mut e, geom, hist_len, down, q, in_len, out_len).map(Some)
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
            dispatch!(simd => fir_c32(simd, taps, h, x, y));
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
                let used = input.get(..consumed).ok_or(Error::Internal("decimation window"))?;
                update_history(p.hist, used)?;
                Ok(Progress { consumed, produced })
            }
        }
    };
}

decimate_impl!(C32, f32, f32, decimate_c32, usize::MAX);
decimate_impl!(CQ15, Q15, i16, decimate_cq15, PREP_TAPS);

/// Runs the polyphase loop of a resampler state; shared by `resample` and `fir_interpolate`.
macro_rules! resample_body {
    ($self:ident, $state:expr, $input:expr, $output:expr, $lane:ty, $run:ident, $trait:ident, $method:ident) => {{
        let p = $state.parts_mut();
        let ran = {
            let t: &[$lane] = bytemuck::cast_slice(p.taps);
            let h: &[$lane] = bytemuck::cast_slice(&*p.hist);
            let x: &[$lane] = bytemuck::cast_slice($input);
            let y: &mut [$lane] = bytemuck::cast_slice_mut($output);
            let (up, down) = (p.up, p.down);
            dispatch!(simd => $run(simd, t, h, x, y, up, down, &mut *p.q))?
        };
        // A filter whose branches do not fit the prepared storage runs on the fallback; nothing
        // of the state has been touched then.
        let Some((consumed, produced)) = ran else {
            return Fallback.$method($state, $input, $output);
        };
        let used = $input
            .get(..consumed)
            .ok_or(Error::Internal("resampler window"))?;
        update_history(p.hist, used)?;
        Ok(Progress { consumed, produced })
    }};
}

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
        resample_body!(
            self,
            state,
            input,
            output,
            f32,
            polyphase_c32,
            Resample,
            resample
        )
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
        resample_body!(
            self,
            state,
            input,
            output,
            i16,
            polyphase_cq15,
            Resample,
            resample
        )
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
