//! The halfband kernels: `halfband_decimate` (`C32`, `CQ15`), `halfband_decimate_real` (`f32`,
//! `Q15`) and `real_to_complex` (`C32`, `CQ15`).
//!
//! # Layout
//!
//! A halfband filter of `N = 2 m + 1` taps (`m` odd) has non-zero taps only at the even indices
//! `k = 2 i` (`i = 0 ..= m`) and at the centre `k = m`. With the decimation by 2 and the history
//! of `N - 1 = 2 m` samples, an output reads the even-index taps from samples of one parity of
//! the virtual stream `history ++ input` (the "even stream") and the centre tap from the other
//! parity (the "odd stream"). So `m + 1` taps are a plain FIR over the even stream, where
//! consecutive outputs read consecutive samples, and the centre tap is one product with the
//! odd stream:
//!
//! ```text
//! y[t] = sum_{i = 0..=m} h[2 i] s[t + m - i] + h[m] o[t]
//!   s[j] = v(first + 2 j),   o[t] = v(first + m + 2 t)
//! ```
//!
//! Each block of [`BLK`] outputs gathers `s` and `o` into planar scratch arrays (one plane per
//! component: the complex samples are split into real and imaginary parts), which also takes
//! care of the history without a second code path, and computes the outputs with lanes of
//! consecutive outputs (the transposed form, no horizontal sums).
//!
//! `real_to_complex` is the same filter over the shifted stream `u[n] = x[n] (-j)^n`, which is
//! real at the even stream (`x` or `-x`, the sign by `n mod 4`) and imaginary at the odd stream:
//! the real part is the real halfband core, the imaginary part the centre tap alone. The signs
//! are applied to the samples (with the saturating negation of `Q15`), not folded into taps.
//!
//! # Numerics
//!
//! `f32` sums use the symmetry of the taps (`h[2 i] (s[a] + s[b])`, decision D70), FMA on the
//! levels that have it (D5), and several chains per vector; the result is within the 110 dB
//! contract, not bit-identical with the fallback. A filter with non-symmetric taps runs on the
//! fallback. `Q15` sums are exact integers in `i32` (the bound on the sum of the absolute taps
//! below makes overflow impossible, else the fallback runs) rounded as `Acc64::finish`:
//! bit-identical with the fallback. A filter longer than [`MAX_N`] taps runs on the fallback.

use fearless_simd::{Simd as SimdToken, f32x4, f32x8, f32x16, prelude::*};
use sdrk_core::kernels::halfband::{
    HalfbandDecimate, HalfbandDecimateReal, HalfbandRealState, HalfbandState, RealToComplex,
    RealToComplexState,
};
use sdrk_core::{C32, CQ15, Fallback, Progress, Q15, Result};

use crate::Simd;
use crate::fir::update_history;
use crate::level::dispatch;
use crate::util::internal;

/// The longest filter of this backend (`m <= 511`); longer ones run on the fallback.
const MAX_N: usize = 1023;
const MAX_M: usize = (MAX_N - 1) / 2;
/// The longest filter of the small class of scratch (`m <= 63`): its stack arrays are a quarter
/// of the size, which a short call pays for in clearing them (the cycles of a call of a few
/// hundred samples fall by a third at `CQ15`).
const SMALL_N: usize = 127;
const SMALL_M: usize = (SMALL_N - 1) / 2;
/// Outputs per block.
const BLK: usize = 64;
/// Length of the even-stream scratch (values, two per sample) for filters up to `m`.
const fn slen(m: usize) -> usize {
    2 * (m + BLK)
}
/// Largest `sum |h|` (raw `Q15` units) for which the `i32` sums cannot overflow:
/// `32768 * 65535 + 2^14 < 2^31`.
const MAX_ABS_SUM: i64 = 65535;

/// Taps prepared for a block kernel.
struct Prep<'a, P> {
    /// `f32`: the pair coefficients `h[2 p]` for `p < (m + 1) / 2`; `Q15`: `h[2 i]`, `i <= m`.
    c: &'a [P],
    /// The centre tap.
    hc: P,
    m: usize,
}

/// A plane of one component of the samples (`f32` or the raw `i16` of `Q15`).
trait Plane: Copy + Default {
    /// The negation of the shift of `real_to_complex` (saturating for `Q15`).
    fn neg(self) -> Self;
    /// The zero sample.
    const ZERO: Self;
    /// The prepared taps (in `buf`), or `None` if the filter runs on the fallback (also if `buf`
    /// is too short for it).
    fn prepare<'a>(taps: &[Self], buf: &'a mut [Self]) -> Option<Prep<'a, Self>>;
    /// `out[u]` for `u < nf` (more may be written, up to the next multiple of 16): the even
    /// stream part, plus the centre tap with `o` if `CENTRE`. The streams hold `C` components
    /// per sample (interleaved), so `u` runs over `nf = C nb` values and the sample `p` places
    /// on is `C p` values on.
    fn compute<S: SimdToken, const CENTRE: bool, const C: usize>(
        simd: S,
        prep: &Prep<'_, Self>,
        s: &[Self],
        o: &[Self],
        out: &mut [Self],
        nf: usize,
    );
    /// `out[t]` is the centre tap times `o[t]` alone.
    fn centre(prep: &Prep<'_, Self>, o: &[Self], out: &mut [Self], nb: usize);
}

/// Whether the level has fused multiply-add for the float sums (decision D5).
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

/// `acc + a * b`, fused where `FMA`.
#[inline(always)]
fn madd<S: SimdToken, V: SimdFloat<S, Element = f32>, const FMA: bool>(a: V, b: V, acc: V) -> V {
    if FMA { a.mul_add(b, acc) } else { acc + a * b }
}

/// `NV` vectors of `V::LEN` consecutive outputs from `t` on.
#[inline(always)]
fn f32_vectors<
    S: SimdToken,
    V: SimdFloat<S, Element = f32>,
    const FMA: bool,
    const CENTRE: bool,
    const C: usize,
    const NV: usize,
>(
    simd: S,
    prep: &Prep<'_, f32>,
    s: &[f32],
    o: &[f32],
    out: &mut [f32],
    t: usize,
) {
    let l = V::LEN;
    let m = prep.m;
    let cp = prep.c;
    let zero = V::splat(simd, 0.0);
    let hc = V::splat(simd, prep.hc);
    let mut acc = [zero; NV];
    if CENTRE {
        for (v, a) in acc.iter_mut().enumerate() {
            if let Some(ov) = o.get(t + v * l..t + (v + 1) * l) {
                *a = hc * V::from_slice(simd, ov);
            }
        }
    }
    // Pair `p` adds the samples `p` after and `p` before the end of the window of `m + NV l / C`
    // samples (`C` values each); each pair is two slices of `NV l` values, cut once.
    let span = NV * l;
    if let Some(w) = s.get(t..t + span + C * m) {
        // `p <= m` is what lets the compiler see that both slices are inside `w`.
        for (p, &c) in cp.iter().enumerate().take(m + 1) {
            let cv = V::splat(simd, c);
            let (Some(fwd), Some(bwd)) = (
                w.get(C * p..C * p + span),
                w.get(C * (m - p)..C * (m - p) + span),
            ) else {
                break;
            };
            for ((a, f), b) in acc
                .iter_mut()
                .zip(fwd.chunks_exact(l))
                .zip(bwd.chunks_exact(l))
            {
                *a = madd::<S, V, FMA>(cv, V::from_slice(simd, f) + V::from_slice(simd, b), *a);
            }
        }
    }
    for (v, a) in acc.iter().enumerate() {
        if let Some(dst) = out.get_mut(t + v * l..t + (v + 1) * l) {
            a.store_slice(dst);
        }
    }
}

/// All outputs of a block with vectors of type `V`: four vectors at a time, then one.
#[inline(always)]
fn f32_block<
    S: SimdToken,
    V: SimdFloat<S, Element = f32>,
    const FMA: bool,
    const CENTRE: bool,
    const C: usize,
>(
    simd: S,
    prep: &Prep<'_, f32>,
    s: &[f32],
    o: &[f32],
    out: &mut [f32],
    nf: usize,
) {
    let l = V::LEN;
    let mut t = 0;
    while t + 8 * l <= nf {
        f32_vectors::<S, V, FMA, CENTRE, C, 8>(simd, prep, s, o, out, t);
        t += 8 * l;
    }
    while t + 4 * l <= nf {
        f32_vectors::<S, V, FMA, CENTRE, C, 4>(simd, prep, s, o, out, t);
        t += 4 * l;
    }
    while t < nf {
        f32_vectors::<S, V, FMA, CENTRE, C, 1>(simd, prep, s, o, out, t);
        t += l;
    }
}

impl Plane for f32 {
    const ZERO: f32 = 0.0;

    #[inline(always)]
    fn neg(self) -> f32 {
        -self
    }

    fn prepare<'a>(taps: &[f32], buf: &'a mut [f32]) -> Option<Prep<'a, f32>> {
        let n = taps.len();
        if !(3..=MAX_N).contains(&n) {
            return None;
        }
        // The pre-add needs exactly symmetric taps (a NaN tap is not equal to itself).
        if !taps.iter().zip(taps.iter().rev()).all(|(a, b)| a == b) {
            return None;
        }
        let m = (n - 1) / 2;
        let nc = m.div_ceil(2);
        let c = buf.get_mut(..nc)?;
        for (d, h) in c.iter_mut().zip(taps.iter().step_by(2)) {
            *d = *h;
        }
        Some(Prep {
            c,
            hc: *taps.get(m)?,
            m,
        })
    }

    #[inline(always)]
    fn compute<S: SimdToken, const CENTRE: bool, const C: usize>(
        simd: S,
        prep: &Prep<'_, f32>,
        s: &[f32],
        o: &[f32],
        out: &mut [f32],
        nf: usize,
    ) {
        let fma = has_fma(simd);
        match S::f32s::LEN {
            16 => f32_block::<S, f32x16<S>, true, CENTRE, C>(simd, prep, s, o, out, nf),
            8 => f32_block::<S, f32x8<S>, true, CENTRE, C>(simd, prep, s, o, out, nf),
            _ if fma => f32_block::<S, f32x4<S>, true, CENTRE, C>(simd, prep, s, o, out, nf),
            _ => f32_block::<S, f32x4<S>, false, CENTRE, C>(simd, prep, s, o, out, nf),
        }
    }

    #[inline(always)]
    fn centre(prep: &Prep<'_, f32>, o: &[f32], out: &mut [f32], nb: usize) {
        for (d, x) in out.iter_mut().zip(o).take(nb) {
            *d = prep.hc * x;
        }
    }
}

/// Outputs per accumulator group of the `Q15` kernel.
const GROUP: usize = 16;

impl Plane for i16 {
    const ZERO: i16 = 0;

    #[inline(always)]
    fn neg(self) -> i16 {
        self.saturating_neg()
    }

    fn prepare<'a>(taps: &[i16], buf: &'a mut [i16]) -> Option<Prep<'a, i16>> {
        let n = taps.len();
        if !(3..=MAX_N).contains(&n) {
            return None;
        }
        let abs: i64 = taps.iter().map(|h| i64::from(*h).abs()).sum();
        if abs > MAX_ABS_SUM {
            return None;
        }
        let m = (n - 1) / 2;
        let c = buf.get_mut(..=m)?;
        for (d, h) in c.iter_mut().zip(taps.iter().step_by(2)) {
            *d = *h;
        }
        Some(Prep {
            c,
            hc: *taps.get(m)?,
            m,
        })
    }

    #[inline(always)]
    fn compute<S: SimdToken, const CENTRE: bool, const C: usize>(
        _simd: S,
        prep: &Prep<'_, i16>,
        s: &[i16],
        o: &[i16],
        out: &mut [i16],
        nf: usize,
    ) {
        let m = prep.m;
        let cs = prep.c;
        let mut t = 0;
        while t < nf {
            let mut acc = [0i32; GROUP];
            if CENTRE && let Some(ov) = o.get(t..t + GROUP) {
                for (a, &x) in acc.iter_mut().zip(ov) {
                    *a = i32::from(prep.hc) * i32::from(x);
                }
            }
            // The value `s[t + C (m - i) + g]` pairs with tap `h[2 i]`.
            if let Some(w) = s.get(t..t + C * m + GROUP) {
                // Taken from the last: the window of the tap `h[2 (m - k)]` starts `C k` on, and
                // `k <= m` lets the compiler see that it is inside `w`.
                for (k, &c) in cs.iter().rev().take(m + 1).enumerate() {
                    let at = C * k;
                    if let Some(Ok(win)) = w.get(at..at + GROUP).map(<&[i16; GROUP]>::try_from) {
                        for (a, &x) in acc.iter_mut().zip(win) {
                            *a += i32::from(c) * i32::from(x);
                        }
                    }
                }
            }
            if let Some(dst) = out.get_mut(t..t + GROUP) {
                for (d, a) in dst.iter_mut().zip(acc) {
                    // `Acc64::finish`: no overflow in `i32` (see `MAX_ABS_SUM`).
                    *d = ((a + (1 << 14)) >> 15).clamp(-32768, 32767) as i16;
                }
            }
            t += GROUP;
        }
    }

    #[inline(always)]
    fn centre(prep: &Prep<'_, i16>, o: &[i16], out: &mut [i16], nb: usize) {
        for (d, &x) in out.iter_mut().zip(o).take(nb) {
            let a = i32::from(prep.hc) * i32::from(x);
            *d = ((a + (1 << 14)) >> 15).clamp(-32768, 32767) as i16;
        }
    }
}

/// Where the outputs of a call are (the decimation by 2 of the fallback).
struct Plan {
    first: usize,
    produced: usize,
    consumed: usize,
    next: usize,
}

fn plan(countdown: usize, input_len: usize, output_len: usize) -> Plan {
    let first = countdown;
    let avail = if first < input_len {
        (input_len - 1 - first) / 2 + 1
    } else {
        0
    };
    let produced = avail.min(output_len);
    let (consumed, next) = if produced < avail {
        (first + produced * 2, 0)
    } else {
        (input_len, first + produced * 2 - input_len)
    };
    Plan {
        first,
        produced,
        consumed,
        next,
    }
}

/// The scratch of a block: the even and odd streams and the results, `C <= 2` values per
/// sample, interleaved; `SL` is the length of the even stream ([`slen`]).
struct Scratch<P, const SL: usize> {
    s: [P; SL],
    o: [P; 2 * BLK],
    res: [P; 2 * BLK],
}

impl<P: Plane, const SL: usize> Scratch<P, SL> {
    fn new() -> Self {
        Self {
            s: [P::ZERO; SL],
            o: [P::ZERO; 2 * BLK],
            res: [P::ZERO; 2 * BLK],
        }
    }
}

/// Copies `count` samples `start, start + 2, ...` of the virtual stream `hist ++ input` to `dst`
/// (`W` values per sample, mapped by `fh` for the history, `fi` for the input; `fi` also gets the
/// index of the sample in `dst`). `hist` has `CH` values per sample and `input` `CX`; `hk` and
/// `xk` are the offset of the first value taken from a sample.
#[inline(always)]
fn gather<P: Copy, X: Copy, Q: Copy, const W: usize, const CH: usize, const CX: usize>(
    hist: &[P],
    hk: usize,
    input: &[X],
    xk: usize,
    start: usize,
    count: usize,
    dst: &mut [Q],
    fh: impl Fn(P) -> Q,
    fi: impl Fn(usize, X) -> Q,
) {
    let h = hist.len() / CH;
    let nh = if start < h {
        (h - start).div_ceil(2).min(count)
    } else {
        0
    };
    let hsrc = hist.get(CH * start + hk..).unwrap_or(&[]);
    if let Some(d) = dst.get_mut(..W * nh) {
        for (j, d) in d.chunks_exact_mut(W).enumerate() {
            if let Some(src) = hsrc.get(2 * CH * j..2 * CH * j + W) {
                for (d, x) in d.iter_mut().zip(src) {
                    *d = fh(*x);
                }
            }
        }
    }
    let first = (start + 2 * nh).saturating_sub(h);
    let isrc = input.get(CX * first + xk..).unwrap_or(&[]);
    if let Some(d) = dst.get_mut(W * nh..W * count) {
        // Every second sample: whole pairs of samples first, then the last sample if the input
        // ends after it.
        let pairs = isrc.chunks_exact(2 * CX);
        let tail = pairs.remainder();
        let mut done = 0;
        for (j, (d, x)) in d.chunks_exact_mut(W).zip(pairs).enumerate() {
            if let (Some((x, _)), Some((d, _))) =
                (x.split_first_chunk::<W>(), d.split_first_chunk_mut::<W>())
            {
                for (d, x) in d.iter_mut().zip(x) {
                    *d = fi(nh + j, *x);
                }
            }
            done = j + 1;
        }
        if let (Some((x, _)), Some((d, _))) = (
            tail.split_first_chunk::<W>(),
            d.get_mut(W * done..)
                .and_then(|d| d.split_first_chunk_mut::<W>()),
        ) {
            for (d, x) in d.iter_mut().zip(x) {
                *d = fi(nh + done, *x);
            }
        }
    }
}

/// `halfband_decimate` and `halfband_decimate_real` for `C` values per sample. `None` if the
/// filter runs on the fallback (nothing has been touched then).
#[inline(always)]
fn decimate<S: SimdToken, P: Plane, const C: usize>(
    simd: S,
    taps: &[P],
    hist: &mut [P],
    countdown: &mut usize,
    input: &[P],
    output: &mut [P],
) -> Result<Option<Progress>> {
    if taps.len() <= SMALL_N {
        decimate_in::<S, P, C, { SMALL_M + 1 }, { slen(SMALL_M) }>(
            simd, taps, hist, countdown, input, output,
        )
    } else {
        decimate_in::<S, P, C, { MAX_M + 1 }, { slen(MAX_M) }>(
            simd, taps, hist, countdown, input, output,
        )
    }
}

/// [`decimate`] with the scratch of a class of filters (`NC` taps, `SL` values of even stream).
#[inline(always)]
fn decimate_in<S: SimdToken, P: Plane, const C: usize, const NC: usize, const SL: usize>(
    simd: S,
    taps: &[P],
    hist: &mut [P],
    countdown: &mut usize,
    input: &[P],
    output: &mut [P],
) -> Result<Option<Progress>> {
    let mut cbuf = [P::ZERO; NC];
    let Some(prep) = P::prepare(taps, &mut cbuf) else {
        return Ok(None);
    };
    let m = prep.m;
    let p = plan(*countdown, input.len() / C, output.len() / C);
    let mut sc = Scratch::<P, SL>::new();
    let mut t0 = 0;
    while t0 < p.produced {
        let nb = BLK.min(p.produced - t0);
        let nf = C * nb;
        gather::<P, P, P, C, C, C>(
            hist,
            0,
            input,
            0,
            p.first + 2 * t0,
            m + nb,
            &mut sc.s,
            |x| x,
            |_, x| x,
        );
        gather::<P, P, P, C, C, C>(
            hist,
            0,
            input,
            0,
            p.first + m + 2 * t0,
            nb,
            &mut sc.o,
            |x| x,
            |_, x| x,
        );
        P::compute::<S, true, C>(simd, &prep, &sc.s, &sc.o, &mut sc.res, nf);
        if let (Some(out), Some(res)) = (output.get_mut(C * t0..C * t0 + nf), sc.res.get(..nf)) {
            out.copy_from_slice(res);
        }
        t0 += nb;
    }
    *countdown = p.next;
    let used = input
        .get(..C * p.consumed)
        .ok_or_else(|| internal("halfband input"))?;
    update_history(hist, used)?;
    Ok(Some(Progress {
        consumed: p.consumed,
        produced: p.produced,
    }))
}

/// `u[n] = x (-j)^n` for `n mod 4`: `(re, im)`.
#[inline(always)]
fn shifted<P: Plane>(x: P, n4: usize) -> (P, P) {
    match n4 & 3 {
        0 => (x, P::ZERO),
        1 => (P::ZERO, x.neg()),
        2 => (x.neg(), P::ZERO),
        _ => (P::ZERO, x),
    }
}

/// `real_to_complex`: `hist` holds the shifted samples (two components each), `n4` is `n mod 4`
/// of the next input sample.
#[inline(always)]
fn real_to_complex<S: SimdToken, P: Plane>(
    simd: S,
    taps: &[P],
    hist: &mut [P],
    n4: &mut usize,
    input: &[P],
    output: &mut [P],
) -> Result<Option<Progress>> {
    if taps.len() <= SMALL_N {
        real_to_complex_in::<S, P, { SMALL_M + 1 }, { slen(SMALL_M) }>(
            simd, taps, hist, n4, input, output,
        )
    } else {
        real_to_complex_in::<S, P, { MAX_M + 1 }, { slen(MAX_M) }>(
            simd, taps, hist, n4, input, output,
        )
    }
}

/// [`real_to_complex`] with the scratch of a class of filters.
#[inline(always)]
fn real_to_complex_in<S: SimdToken, P: Plane, const NC: usize, const SL: usize>(
    simd: S,
    taps: &[P],
    hist: &mut [P],
    n4: &mut usize,
    input: &[P],
    output: &mut [P],
) -> Result<Option<Progress>> {
    let mut cbuf = [P::ZERO; NC];
    let Some(prep) = P::prepare(taps, &mut cbuf) else {
        return Ok(None);
    };
    let m = prep.m;
    let h = hist.len() / 2;
    let p = plan(*n4 & 1, input.len(), output.len() / 2);
    let mut sc = Scratch::<P, SL>::new();
    let mut t0 = 0;
    while t0 < p.produced {
        let nb = BLK.min(p.produced - t0);
        // `n mod 4` of the stream samples: even stream `b + 2 j`, odd stream `c0 + 2 t`.
        let base = (*n4 + p.first + 2 * t0) & 3;
        let b = (base + 4 - (h & 3)) & 3;
        let c0 = (base + 4 - (m & 3)) & 3;
        let (r0, r1) = sc.res.split_at_mut(BLK);
        gather::<P, P, P, 1, 2, 1>(
            hist,
            0,
            input,
            0,
            p.first + 2 * t0,
            m + nb,
            &mut sc.s,
            |x| x,
            |j, x| if (b + 2 * j) & 3 == 2 { x.neg() } else { x },
        );
        gather::<P, P, P, 1, 2, 1>(
            hist,
            1,
            input,
            0,
            p.first + m + 2 * t0,
            nb,
            &mut sc.o,
            |x| x,
            |t, x| if (c0 + 2 * t) & 3 == 1 { x.neg() } else { x },
        );
        P::compute::<S, false, 1>(simd, &prep, &sc.s, &sc.o, r0, nb);
        P::centre(&prep, &sc.o, r1, nb);
        for ((d, &re), &im) in output
            .chunks_exact_mut(2)
            .skip(t0)
            .zip(r0.iter())
            .zip(r1.iter())
            .take(nb)
        {
            if let [dr, di] = d {
                *dr = re;
                *di = im;
            }
        }
        t0 += nb;
    }
    // The history holds the shifted samples of the last `N - 1` consumed inputs.
    let keep = h;
    let count = keep.min(p.consumed);
    hist.rotate_left(2 * count);
    let from = p.consumed - count;
    let dst = hist
        .get_mut(2 * (keep - count)..)
        .ok_or_else(|| internal("real_to_complex history"))?;
    let src = input
        .get(from..p.consumed)
        .ok_or_else(|| internal("real_to_complex history"))?;
    for (i, (d, x)) in dst.chunks_exact_mut(2).zip(src).enumerate() {
        let (re, im) = shifted(*x, *n4 + from + i);
        if let [dr, di] = d {
            *dr = re;
            *di = im;
        }
    }
    *n4 = (*n4 + p.consumed) & 3;
    Ok(Some(Progress {
        consumed: p.consumed,
        produced: p.produced,
    }))
}

macro_rules! halfband_impl {
    ($tr:ident, $method:ident, $state:ident, $t:ty, $tap:ty, $lane:ty, $c:literal) => {
        impl $tr<$t> for Simd {
            fn $method<Taps, Hist>(
                &self,
                state: &mut $state<$t, Taps, Hist>,
                input: &[$t],
                output: &mut [$t],
            ) -> Result<Progress>
            where
                Taps: AsRef<[$tap]>,
                Hist: AsRef<[$t]> + AsMut<[$t]>,
            {
                let p = state.parts_mut();
                let ran = {
                    let t: &[$lane] = bytemuck::cast_slice(p.taps);
                    let h: &mut [$lane] = bytemuck::cast_slice_mut(&mut *p.hist);
                    let x: &[$lane] = bytemuck::cast_slice(input);
                    let y: &mut [$lane] = bytemuck::cast_slice_mut(output);
                    dispatch!(simd => decimate::<_, $lane, $c>(
                        simd, t, h, &mut *p.countdown, x, y
                    ))?
                };
                match ran {
                    Some(progress) => Ok(progress),
                    None => Fallback.$method(state, input, output),
                }
            }
        }
    };
}

halfband_impl!(
    HalfbandDecimate,
    halfband_decimate,
    HalfbandState,
    C32,
    f32,
    f32,
    2
);
halfband_impl!(
    HalfbandDecimate,
    halfband_decimate,
    HalfbandState,
    CQ15,
    Q15,
    i16,
    2
);
halfband_impl!(
    HalfbandDecimateReal,
    halfband_decimate_real,
    HalfbandRealState,
    f32,
    f32,
    f32,
    1
);
halfband_impl!(
    HalfbandDecimateReal,
    halfband_decimate_real,
    HalfbandRealState,
    Q15,
    Q15,
    i16,
    1
);

macro_rules! real_to_complex_impl {
    ($t:ty, $tap:ty, $lane:ty) => {
        impl RealToComplex<$t> for Simd {
            fn real_to_complex<Taps, Hist>(
                &self,
                state: &mut RealToComplexState<$t, Taps, Hist>,
                input: &[$tap],
                output: &mut [$t],
            ) -> Result<Progress>
            where
                Taps: AsRef<[$tap]>,
                Hist: AsRef<[$t]> + AsMut<[$t]>,
            {
                let p = state.parts_mut();
                let ran = {
                    let t: &[$lane] = bytemuck::cast_slice(p.taps);
                    let h: &mut [$lane] = bytemuck::cast_slice_mut(&mut *p.hist);
                    let x: &[$lane] = bytemuck::cast_slice(input);
                    let y: &mut [$lane] = bytemuck::cast_slice_mut(output);
                    dispatch!(simd => real_to_complex::<_, $lane>(
                        simd, t, h, &mut *p.n4, x, y
                    ))?
                };
                match ran {
                    Some(progress) => Ok(progress),
                    None => Fallback.real_to_complex(state, input, output),
                }
            }
        }
    };
}

real_to_complex_impl!(C32, f32, f32);
real_to_complex_impl!(CQ15, Q15, i16);
