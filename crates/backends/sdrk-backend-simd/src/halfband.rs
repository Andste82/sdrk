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
//! `f32` sums use the symmetry of the taps (`h[2 i] (s[a] + s[b])`, decision D65), FMA on the
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
/// Outputs per block.
const BLK: usize = 64;
/// Extra scratch elements after a block (vector loads past the last output).
const PAD: usize = 16;
/// Length of the even-stream scratch.
const SLEN: usize = MAX_M + BLK + PAD;
/// Largest `sum |h|` (raw `Q15` units) for which the `i32` sums cannot overflow:
/// `32768 * 65535 + 2^14 < 2^31`.
const MAX_ABS_SUM: i64 = 65535;

/// Taps prepared for a block kernel.
struct Prep<P> {
    /// `f32`: the pair coefficients `h[2 p]` for `p < (m + 1) / 2`; `Q15`: `h[2 i]`, `i <= m`.
    c: [P; MAX_M + 1],
    /// Number of valid entries of `c`.
    nc: usize,
    /// The centre tap.
    hc: P,
    m: usize,
}

impl<P: Plane> Prep<P> {
    fn new() -> Self {
        Self {
            c: [P::ZERO; MAX_M + 1],
            nc: 0,
            hc: P::ZERO,
            m: 0,
        }
    }
}

/// A plane of one component of the samples (`f32` or the raw `i16` of `Q15`).
trait Plane: Copy + Default {
    /// The negation of the shift of `real_to_complex` (saturating for `Q15`).
    fn neg(self) -> Self;
    /// The zero sample.
    const ZERO: Self;
    /// The prepared taps, or `None` if the filter runs on the fallback.
    fn prepare(taps: &[Self], prep: &mut Prep<Self>) -> Option<()>;
    /// `out[t]` for `t < nb` (more may be written, up to the next multiple of 16): the even
    /// stream part, plus the centre tap with `o` if `CENTRE`.
    fn compute<S: SimdToken, const CENTRE: bool>(
        simd: S,
        prep: &Prep<Self>,
        s: &[Self],
        o: &[Self],
        out: &mut [Self; BLK],
        nb: usize,
    );
    /// `out[t]` is the centre tap times `o[t]` alone.
    fn centre(prep: &Prep<Self>, o: &[Self], out: &mut [Self; BLK], nb: usize);
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
    const NV: usize,
>(
    simd: S,
    prep: &Prep<f32>,
    s: &[f32],
    o: &[f32],
    out: &mut [f32; BLK],
    t: usize,
) {
    let l = V::LEN;
    let m = prep.m;
    let cp = prep.c.get(..prep.nc).unwrap_or(&[]);
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
    let windows: [&[f32]; NV] =
        core::array::from_fn(|v| s.get(t + v * l..t + v * l + m + l).unwrap_or(&[]));
    let mut its = windows.map(|w| w.windows(l).zip(w.windows(l).rev()));
    for &c in cp {
        let cv = V::splat(simd, c);
        for (a, it) in acc.iter_mut().zip(its.iter_mut()) {
            if let Some((b, e)) = it.next() {
                *a = madd::<S, V, FMA>(cv, V::from_slice(simd, e) + V::from_slice(simd, b), *a);
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
fn f32_block<S: SimdToken, V: SimdFloat<S, Element = f32>, const FMA: bool, const CENTRE: bool>(
    simd: S,
    prep: &Prep<f32>,
    s: &[f32],
    o: &[f32],
    out: &mut [f32; BLK],
    nb: usize,
) {
    let l = V::LEN;
    let mut t = 0;
    while t + 4 * l <= nb {
        f32_vectors::<S, V, FMA, CENTRE, 4>(simd, prep, s, o, out, t);
        t += 4 * l;
    }
    while t < nb {
        f32_vectors::<S, V, FMA, CENTRE, 1>(simd, prep, s, o, out, t);
        t += l;
    }
}

impl Plane for f32 {
    const ZERO: f32 = 0.0;

    #[inline(always)]
    fn neg(self) -> f32 {
        -self
    }

    fn prepare(taps: &[f32], prep: &mut Prep<f32>) -> Option<()> {
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
        for (d, h) in prep.c.iter_mut().zip(taps.iter().step_by(2)).take(nc) {
            *d = *h;
        }
        prep.nc = nc;
        prep.hc = *taps.get(m)?;
        prep.m = m;
        Some(())
    }

    #[inline(always)]
    fn compute<S: SimdToken, const CENTRE: bool>(
        simd: S,
        prep: &Prep<f32>,
        s: &[f32],
        o: &[f32],
        out: &mut [f32; BLK],
        nb: usize,
    ) {
        let fma = has_fma(simd);
        match S::f32s::LEN {
            16 => f32_block::<S, f32x16<S>, true, CENTRE>(simd, prep, s, o, out, nb),
            8 => f32_block::<S, f32x8<S>, true, CENTRE>(simd, prep, s, o, out, nb),
            _ if fma => f32_block::<S, f32x4<S>, true, CENTRE>(simd, prep, s, o, out, nb),
            _ => f32_block::<S, f32x4<S>, false, CENTRE>(simd, prep, s, o, out, nb),
        }
    }

    #[inline(always)]
    fn centre(prep: &Prep<f32>, o: &[f32], out: &mut [f32; BLK], nb: usize) {
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

    fn prepare(taps: &[i16], prep: &mut Prep<i16>) -> Option<()> {
        let n = taps.len();
        if !(3..=MAX_N).contains(&n) {
            return None;
        }
        let abs: i64 = taps.iter().map(|h| i64::from(*h).abs()).sum();
        if abs > MAX_ABS_SUM {
            return None;
        }
        let m = (n - 1) / 2;
        for (d, h) in prep.c.iter_mut().zip(taps.iter().step_by(2)) {
            *d = *h;
        }
        prep.nc = m + 1;
        prep.hc = *taps.get(m)?;
        prep.m = m;
        Some(())
    }

    #[inline(always)]
    fn compute<S: SimdToken, const CENTRE: bool>(
        _simd: S,
        prep: &Prep<i16>,
        s: &[i16],
        o: &[i16],
        out: &mut [i16; BLK],
        nb: usize,
    ) {
        let m = prep.m;
        let cs = prep.c.get(..prep.nc).unwrap_or(&[]);
        let mut t = 0;
        while t < nb {
            let mut acc = [0i32; GROUP];
            if CENTRE && let Some(ov) = o.get(t..t + GROUP) {
                for (a, &x) in acc.iter_mut().zip(ov) {
                    *a = i32::from(prep.hc) * i32::from(x);
                }
            }
            // Sample `s[t + m - i + g]` pairs with tap `h[2 i]`.
            let w = s.get(t..t + m + GROUP).unwrap_or(&[]);
            for (win, &c) in w.windows(GROUP).rev().zip(cs) {
                for (a, &x) in acc.iter_mut().zip(win) {
                    *a += i32::from(c) * i32::from(x);
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
    fn centre(prep: &Prep<i16>, o: &[i16], out: &mut [i16; BLK], nb: usize) {
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

/// The scratch of a block: the even and odd streams of up to two planes.
struct Scratch<P> {
    s: [[P; SLEN]; 2],
    o: [[P; BLK + PAD]; 2],
    res: [[P; BLK]; 2],
}

impl<P: Plane> Scratch<P> {
    fn new() -> Self {
        Self {
            s: [[P::ZERO; SLEN]; 2],
            o: [[P::ZERO; BLK + PAD]; 2],
            res: [[P::ZERO; BLK]; 2],
        }
    }
}

/// Copies `count` samples `start, start + 2, ...` of the virtual stream `hist ++ input` to `dst`
/// (mapped by `fh` for the history, `fi` for the input; `fi` also gets the index in `dst`).
/// `hist` has `CH` components per sample and `input` `CX`; `hk` and `xk` select the component.
#[inline(always)]
fn gather<P: Copy, X: Copy, Q: Copy, const CH: usize, const CX: usize>(
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
    if let Some(d) = dst.get_mut(..nh) {
        for (d, x) in d.iter_mut().zip(hsrc.iter().step_by(2 * CH)) {
            *d = fh(*x);
        }
    }
    let first = (start + 2 * nh).saturating_sub(h);
    let isrc = input.get(CX * first + xk..).unwrap_or(&[]);
    if let Some(d) = dst.get_mut(nh..count) {
        for (j, (d, x)) in d.iter_mut().zip(isrc.iter().step_by(2 * CX)).enumerate() {
            *d = fi(nh + j, *x);
        }
    }
}

/// `halfband_decimate` and `halfband_decimate_real` for `C` components per sample. `None`
/// if the filter runs on the fallback (nothing has been touched then).
#[inline(always)]
fn decimate<S: SimdToken, P: Plane, const C: usize>(
    simd: S,
    taps: &[P],
    hist: &mut [P],
    countdown: &mut usize,
    input: &[P],
    output: &mut [P],
) -> Result<Option<Progress>> {
    let mut prep = Prep::<P>::new();
    if P::prepare(taps, &mut prep).is_none() {
        return Ok(None);
    }
    let m = prep.m;
    let p = plan(*countdown, input.len() / C, output.len() / C);
    let mut sc = Scratch::<P>::new();
    let mut t0 = 0;
    while t0 < p.produced {
        let nb = BLK.min(p.produced - t0);
        for (k, ((s, o), r)) in
            sc.s.iter_mut()
                .zip(sc.o.iter_mut())
                .zip(sc.res.iter_mut())
                .take(C)
                .enumerate()
        {
            gather::<P, P, P, C, C>(
                hist,
                k,
                input,
                k,
                p.first + 2 * t0,
                m + nb,
                s,
                |x| x,
                |_, x| x,
            );
            gather::<P, P, P, C, C>(
                hist,
                k,
                input,
                k,
                p.first + m + 2 * t0,
                nb,
                o,
                |x| x,
                |_, x| x,
            );
            P::compute::<S, true>(simd, &prep, s, o, r, nb);
            for (d, &x) in output
                .iter_mut()
                .skip(C * t0 + k)
                .step_by(C)
                .zip(r.iter())
                .take(nb)
            {
                *d = x;
            }
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
    let mut prep = Prep::<P>::new();
    if P::prepare(taps, &mut prep).is_none() {
        return Ok(None);
    }
    let m = prep.m;
    let h = hist.len() / 2;
    let p = plan(*n4 & 1, input.len(), output.len() / 2);
    let mut sc = Scratch::<P>::new();
    let mut t0 = 0;
    while t0 < p.produced {
        let nb = BLK.min(p.produced - t0);
        // `n mod 4` of the stream samples: even stream `b + 2 j`, odd stream `c0 + 2 t`.
        let base = (*n4 + p.first + 2 * t0) & 3;
        let b = (base + 4 - (h & 3)) & 3;
        let c0 = (base + 4 - (m & 3)) & 3;
        let [s, _] = &mut sc.s;
        let [o, _] = &mut sc.o;
        let [r0, r1] = &mut sc.res;
        gather::<P, P, P, 2, 1>(
            hist,
            0,
            input,
            0,
            p.first + 2 * t0,
            m + nb,
            s,
            |x| x,
            |j, x| if (b + 2 * j) & 3 == 2 { x.neg() } else { x },
        );
        gather::<P, P, P, 2, 1>(
            hist,
            1,
            input,
            0,
            p.first + m + 2 * t0,
            nb,
            o,
            |x| x,
            |t, x| if (c0 + 2 * t) & 3 == 1 { x.neg() } else { x },
        );
        P::compute::<S, false>(simd, &prep, s, o, r0, nb);
        P::centre(&prep, o, r1, nb);
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
