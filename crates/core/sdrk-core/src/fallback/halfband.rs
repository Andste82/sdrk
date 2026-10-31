//! Fallback implementations of `halfband_decimate`, `halfband_decimate_real` and
//! `real_to_complex`: only the non-zero taps (the even indices and the centre), in ascending
//! tap order, one accumulator per output, no FMA.

use super::decim::FbMac;
use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::filter::update_history;
use crate::kernels::halfband::{
    HalfbandDecimate, HalfbandDecimateReal, HalfbandRealState, HalfbandState, RealToComplex,
    RealToComplexState, ShiftNeg, shifted,
};
use crate::timing::Progress;
use crate::types::{C32, CQ15, ComplexSample, Q15};

/// Outputs that a group computes together: their windows lie two samples apart and share the
/// taps; each output keeps its own accumulator.
const GROUP: usize = 8;

/// The non-zero taps of a halfband filter with their index, ascending: the even indices below
/// the centre `m`, the centre, the even indices above it.
#[inline]
fn nonzero<P: Copy>(taps: &[P]) -> impl Iterator<Item = (usize, P)> + '_ {
    let m = taps.len().saturating_sub(1) / 2;
    let evens = taps.iter().copied().enumerate().step_by(2);
    evens
        .clone()
        .take_while(move |(k, _)| *k < m)
        .chain(taps.get(m).map(|h| (m, *h)))
        .chain(evens.skip_while(move |(k, _)| *k < m))
}

/// The even-index taps, ascending.
#[inline]
fn evens<P: Copy>(taps: &[P]) -> impl Iterator<Item = (usize, P)> + '_ {
    taps.iter().copied().enumerate().step_by(2)
}

/// Where the outputs of a call are: `first` is the input index of the first one, `produced` their
/// number, `consumed` the input samples used and `next` the countdown after the call.
struct Plan {
    first: usize,
    produced: usize,
    consumed: usize,
    next: usize,
}

/// The decimation by 2: the producing samples are `first, first + 2, ...`; if the output is full
/// the call stops right before the next producing sample.
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

/// The sample `i` places before the newest of the `seen` samples of `input`, reaching into the
/// history (oldest first) after the input.
#[inline]
fn back<T: Copy>(hist: &[T], input: &[T], seen: usize, i: usize) -> Result<T> {
    let found = if i < seen {
        input.get(seen - 1 - i)
    } else {
        hist.len()
            .checked_sub(i - seen + 1)
            .and_then(|at| hist.get(at))
    };
    found.copied().ok_or(Error::Internal("halfband window"))
}

/// One output of the halfband decimator, `seen` input samples up to its newest sample.
fn single<T: FbMac>(taps: &[T::Tap], hist: &[T], input: &[T], seen: usize) -> Result<T> {
    let mut acc = T::zero();
    for (k, h) in nonzero(taps) {
        T::mac(&mut acc, h, back(hist, input, seen, k)?);
    }
    Ok(T::finish(acc))
}

/// `GROUP` outputs whose windows lie in `w` (two samples apart): output `t` has the window
/// `w[2 t .. 2 t + N]`, tap `k` multiplies `w[2 t + N - 1 - k]`.
#[inline(never)]
fn group<T: FbMac>(taps: &[T::Tap], w: &[T]) -> Result<[T; GROUP]> {
    let mut acc = [T::zero(); GROUP];
    let last = taps.len().saturating_sub(1);
    for (k, h) in nonzero(taps) {
        let xs = w.get(last - k..).ok_or(Error::Internal("halfband group"))?;
        for (a, x) in acc.iter_mut().zip(xs.iter().step_by(2)) {
            T::mac_exact(a, h, *x);
        }
    }
    Ok(acc.map(T::finish))
}

fn halfband<T: FbMac>(
    taps: &[T::Tap],
    hist: &mut [T],
    countdown: &mut usize,
    input: &[T],
    output: &mut [T],
) -> Result<Progress> {
    let p = plan(*countdown, input.len(), output.len());
    let n = taps.len();
    // A group sums `Q15` products in an `i64` without saturation: exact for fewer than 2^32 taps.
    let grouped = u32::try_from(n).is_ok();
    let out = output
        .get_mut(..p.produced)
        .ok_or(Error::Internal("halfband output"))?;
    let mut t = 0;
    while t < p.produced {
        let seen = p.first + 1 + 2 * t;
        if grouped && seen >= n && t + GROUP <= p.produced {
            let w = input
                .get(seen - n..seen + 2 * (GROUP - 1))
                .ok_or(Error::Internal("halfband window"))?;
            let ys = group::<T>(taps, w)?;
            let dst = out
                .get_mut(t..t + GROUP)
                .ok_or(Error::Internal("halfband output"))?;
            dst.copy_from_slice(&ys);
            t += GROUP;
        } else {
            let slot = out.get_mut(t).ok_or(Error::Internal("halfband output"))?;
            *slot = single(taps, hist, input, seen)?;
            t += 1;
        }
    }
    *countdown = p.next;
    update_history(
        hist,
        input
            .get(..p.consumed)
            .ok_or(Error::Internal("halfband input"))?,
    )?;
    Ok(Progress {
        consumed: p.consumed,
        produced: p.produced,
    })
}

macro_rules! halfband_impl {
    ($tr:ident, $method:ident, $state:ident, $t:ty, $tap:ty) => {
        impl $tr<$t> for Fallback {
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
                halfband::<$t>(p.taps, p.hist, p.countdown, input, output)
            }
        }
    };
}

halfband_impl!(HalfbandDecimate, halfband_decimate, HalfbandState, C32, f32);
halfband_impl!(
    HalfbandDecimate,
    halfband_decimate,
    HalfbandState,
    CQ15,
    Q15
);
halfband_impl!(
    HalfbandDecimateReal,
    halfband_decimate_real,
    HalfbandRealState,
    f32,
    f32
);
halfband_impl!(
    HalfbandDecimateReal,
    halfband_decimate_real,
    HalfbandRealState,
    Q15,
    Q15
);

/// `x` or its shift negation.
#[inline]
fn signed<R: ShiftNeg>(x: R, neg: bool) -> R {
    if neg { x.shift_neg() } else { x }
}

/// One output of `real_to_complex`. `q0` is `n mod 4` of the newest sample of the window.
/// Every `u[n - k]` at an even `k` has only a real part (`n` is even at an output), the centre
/// tap sees an odd index and only an imaginary part.
fn r2c_single<C>(
    taps: &[C::Real],
    hist: &[C],
    input: &[C::Real],
    seen: usize,
    q0: usize,
) -> Result<C>
where
    C: ComplexSample,
    C::Real: FbMac<Tap = C::Real> + ShiftNeg,
{
    let m = taps.len().saturating_sub(1) / 2;
    let mut re = <C::Real as FbMac>::zero();
    for (k, h) in evens(taps) {
        let x = if k < seen {
            let v = input
                .get(seen - 1 - k)
                .copied()
                .ok_or(Error::Internal("real_to_complex window"))?;
            signed(v, (q0 + 4 - (k & 3)) & 3 == 2)
        } else {
            back(hist, &[], seen, k)?.re()
        };
        <C::Real as FbMac>::mac(&mut re, h, x);
    }
    let mut im = <C::Real as FbMac>::zero();
    let h = taps
        .get(m)
        .copied()
        .ok_or(Error::Internal("real_to_complex centre"))?;
    let x = if m < seen {
        let v = input
            .get(seen - 1 - m)
            .copied()
            .ok_or(Error::Internal("real_to_complex window"))?;
        signed(v, (q0 + 4 - (m & 3)) & 3 == 1)
    } else {
        back(hist, &[], seen, m)?.im()
    };
    <C::Real as FbMac>::mac(&mut im, h, x);
    Ok(C::new(
        <C::Real as FbMac>::finish(re),
        <C::Real as FbMac>::finish(im),
    ))
}

/// `GROUP` outputs of `real_to_complex` with the windows in `w` (real input, two samples apart);
/// `q0` is `n mod 4` of the newest sample of the first window.
#[inline(never)]
fn r2c_group<C>(taps: &[C::Real], w: &[C::Real], q0: usize) -> Result<[C; GROUP]>
where
    C: ComplexSample,
    C::Real: FbMac<Tap = C::Real> + ShiftNeg,
{
    let last = taps.len().saturating_sub(1);
    let m = last / 2;
    let mut re = [<C::Real as FbMac>::zero(); GROUP];
    for (k, h) in evens(taps) {
        let xs = w
            .get(last - k..)
            .ok_or(Error::Internal("real_to_complex group"))?;
        let flip = (q0 + 4 - (k & 3)) & 3 == 2;
        for (t, (a, x)) in re.iter_mut().zip(xs.iter().step_by(2)).enumerate() {
            <C::Real as FbMac>::mac_exact(a, h, signed(*x, flip ^ (t & 1 == 1)));
        }
    }
    let mut im = [<C::Real as FbMac>::zero(); GROUP];
    let h = taps
        .get(m)
        .copied()
        .ok_or(Error::Internal("real_to_complex centre"))?;
    let xs = w
        .get(last - m..)
        .ok_or(Error::Internal("real_to_complex group"))?;
    let flip = (q0 + 4 - (m & 3)) & 3 == 1;
    for (t, (a, x)) in im.iter_mut().zip(xs.iter().step_by(2)).enumerate() {
        <C::Real as FbMac>::mac_exact(a, h, signed(*x, flip ^ (t & 1 == 1)));
    }
    let mut out = [C::ZERO; GROUP];
    for ((o, r), i) in out.iter_mut().zip(re).zip(im) {
        *o = C::new(<C::Real as FbMac>::finish(r), <C::Real as FbMac>::finish(i));
    }
    Ok(out)
}

fn real_to_complex<C>(
    taps: &[C::Real],
    hist: &mut [C],
    n4: &mut usize,
    input: &[C::Real],
    output: &mut [C],
) -> Result<Progress>
where
    C: ComplexSample,
    C::Real: FbMac<Tap = C::Real> + ShiftNeg,
{
    let err = |what| Error::Internal(what);
    let p = plan(*n4 & 1, input.len(), output.len());
    let n = taps.len();
    let grouped = u32::try_from(n).is_ok();
    let out = output
        .get_mut(..p.produced)
        .ok_or_else(|| err("real_to_complex output"))?;
    let mut t = 0;
    while t < p.produced {
        let seen = p.first + 1 + 2 * t;
        // `n mod 4` of the newest sample of this output's window.
        let q0 = (*n4 + seen - 1) & 3;
        if grouped && seen >= n && t + GROUP <= p.produced {
            let w = input
                .get(seen - n..seen + 2 * (GROUP - 1))
                .ok_or_else(|| err("real_to_complex window"))?;
            let ys = r2c_group::<C>(taps, w, q0)?;
            out.get_mut(t..t + GROUP)
                .ok_or_else(|| err("real_to_complex output"))?
                .copy_from_slice(&ys);
            t += GROUP;
        } else {
            *out.get_mut(t)
                .ok_or_else(|| err("real_to_complex output"))? =
                r2c_single::<C>(taps, hist, input, seen, q0)?;
            t += 1;
        }
    }
    // The history holds the shifted samples of the last `N - 1` consumed inputs.
    let keep = hist.len();
    let count = keep.min(p.consumed);
    hist.rotate_left(count);
    let dst = hist
        .get_mut(keep - count..)
        .ok_or_else(|| err("real_to_complex history"))?;
    let src = input
        .get(p.consumed - count..p.consumed)
        .ok_or_else(|| err("real_to_complex history"))?;
    for (i, (d, x)) in dst.iter_mut().zip(src).enumerate() {
        *d = shifted::<C>(*x, *n4 + (p.consumed - count) + i);
    }
    *n4 = (*n4 + p.consumed) & 3;
    Ok(Progress {
        consumed: p.consumed,
        produced: p.produced,
    })
}

macro_rules! real_to_complex_impl {
    ($t:ty) => {
        impl RealToComplex<$t> for Fallback {
            fn real_to_complex<Taps, Hist>(
                &self,
                state: &mut RealToComplexState<$t, Taps, Hist>,
                input: &[<$t as ComplexSample>::Real],
                output: &mut [$t],
            ) -> Result<Progress>
            where
                Taps: AsRef<[<$t as ComplexSample>::Real]>,
                Hist: AsRef<[$t]> + AsMut<[$t]>,
            {
                let p = state.parts_mut();
                real_to_complex::<$t>(p.taps, p.hist, p.n4, input, output)
            }
        }
    };
}

real_to_complex_impl!(C32);
real_to_complex_impl!(CQ15);
