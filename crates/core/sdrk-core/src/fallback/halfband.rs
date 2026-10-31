//! Fallback implementations of `halfband_decimate`, `halfband_decimate_real` and
//! `real_to_complex`: only the non-zero taps (the even indices and the centre), in ascending
//! tap order (the centre tap last, D94), one accumulator per output, no FMA.

use super::decim::FbMac;
use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::filter::update_history;
use crate::kernels::halfband::{
    HalfbandDecimate, HalfbandDecimateReal, HalfbandRealState, HalfbandState, RealToComplex,
    RealToComplexState, ShiftNeg, shifted,
};
use crate::num::Acc64;
use crate::timing::Progress;
use crate::types::{C32, CQ15, ComplexSample, Q15, Sample};

/// Outputs that a group computes together: their windows lie two samples apart and share the
/// taps; each output keeps its own accumulator.
#[cfg(not(target_arch = "wasm32"))]
const GROUP: usize = 8;
/// On wasm32 sixteen: the per-tap work (the splat of the tap, the bounds checks) is counted for
/// every operation there and is then shared by twice the outputs.
#[cfg(target_arch = "wasm32")]
const GROUP: usize = 16;

/// The non-zero taps of a halfband filter with their index: the even indices ascending, then
/// the centre `m` (D94: the large centre tap enters the sum last, as in liquid-dsp's `resamp2`).
#[inline]
fn nonzero<P: Copy>(taps: &[P]) -> impl Iterator<Item = (usize, P)> + '_ {
    let m = taps.len().saturating_sub(1) / 2;
    evens(taps).chain(taps.get(m).map(|h| (m, *h)))
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

/// The longest filter that a group handles (`m <= 255`); longer ones are computed one output at
/// a time.
const GROUP_MAX_TAPS: usize = 511;
/// Length of the even-offset samples of a group window: `m + GROUP`.
const GROUP_EVEN: usize = (GROUP_MAX_TAPS - 1) / 2 + GROUP;

/// `GROUP` accumulators plus the products of the tap `h` with the consecutive samples `xs`.
#[allow(
    clippy::inline_always,
    reason = "the group size must be a constant in the hot loops"
)]
#[inline(always)]
fn add_taps<T: FbMac>(acc: &mut [T::Acc; GROUP], h: T::Tap, xs: &[T]) -> Result<()> {
    let xs: &[T; GROUP] = xs
        .try_into()
        .map_err(|_| Error::Internal("halfband group"))?;
    for (a, x) in acc.iter_mut().zip(xs) {
        T::mac_exact(a, h, *x);
    }
    Ok(())
}

/// The sums of a group, per sample type. `even[j]` is the sample at the offset `2 j` of the
/// window, `mid[t]` the one the centre tap multiplies (`None`: no centre tap in the sum).
trait GroupSum: FbMac {
    /// Whether [`GroupSum::sum`] may be called with `narrow` for these taps.
    fn narrow(_taps: &[Self::Tap]) -> bool {
        false
    }

    /// The sums: the even taps in ascending order, then the centre (if `mid`). Output `t` pairs
    /// the even tap `2 i` with `even[m - i + t]`.
    fn sum(
        taps: &[Self::Tap],
        even: &[Self; GROUP_EVEN],
        mid: Option<&[Self; GROUP]>,
        _narrow: bool,
    ) -> Result<[Self; GROUP]> {
        sum_wide::<Self>(taps, even, mid)
    }
}

/// [`GroupSum::sum`] with the accumulators of [`FbMac`].
#[inline(always)]
#[allow(
    clippy::inline_always,
    reason = "the group size must be a constant in the hot loops"
)]
fn sum_wide<T: FbMac>(
    taps: &[T::Tap],
    even: &[T; GROUP_EVEN],
    mid: Option<&[T; GROUP]>,
) -> Result<[T; GROUP]> {
    let m = taps.len().saturating_sub(1) / 2;
    let mut acc = [T::zero(); GROUP];
    let windows = even
        .get(..m + GROUP)
        .ok_or(Error::Internal("halfband group"))?
        .windows(GROUP)
        .rev();
    for (h, win) in taps.chunks(2).zip(windows) {
        let h = *h.first().ok_or(Error::Internal("halfband group"))?;
        add_taps::<T>(&mut acc, h, win)?;
    }
    if let Some(mid) = mid {
        let h = *taps.get(m).ok_or(Error::Internal("halfband centre"))?;
        add_taps::<T>(&mut acc, h, mid)?;
    }
    Ok(acc.map(T::finish))
}

impl GroupSum for f32 {}
impl GroupSum for C32 {}

/// The `Q15` taps whose absolute sum is at most 65535 (raw): every partial sum of products with
/// samples of at most 32768 and the rounding constant stay below `2^31`, so `i32` sums are
/// exact (`32768 * 65535 + 2^14 < 2^31`) and equal the `i64` sums.
fn narrow_taps(taps: &[Q15]) -> bool {
    taps.iter().map(|h| i64::from(h.0).abs()).sum::<i64>() <= 65535
}

/// `sat16(round_shift(a, 15))` of an exact `i32` sum.
#[inline(always)]
#[allow(
    clippy::inline_always,
    reason = "the group size must be a constant in the hot loops"
)]
fn finish_i32(a: i32) -> Q15 {
    Acc64(i64::from(a)).finish()
}

/// The `i32` sums of a group on wasm32 with `simd128` (the compiler leaves the scalar loop of
/// [`GroupSum::sum`] alone there): `C` values per sample, interleaved. `even` holds `C` values
/// per sample, `mid` the centre samples with their tap; the result is the `C * GROUP` sums.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn narrow_wasm<const C: usize>(
    taps: &[Q15],
    even: &[i16],
    mid: Option<(i16, &[i16])>,
) -> Option<[i32; 32]> {
    use core::arch::wasm32::{
        i16x8_splat, i32x4_add, i32x4_extmul_high_i16x8, i32x4_extmul_low_i16x8,
        i32x4_extract_lane, i32x4_splat, v128,
    };
    let load = |x: &[i16]| -> Option<v128> {
        Some(bytemuck::cast::<[i16; 8], v128>(
            *<&[i16; 8]>::try_from(x).ok()?,
        ))
    };
    let m = taps.len().saturating_sub(1) / 2;
    let mut acc = [i32x4_splat(0); 8];
    let add = |acc: &mut [v128; 8], w: &[i16], hv: v128| -> Option<()> {
        for (k, c) in w.chunks_exact(8).take(C * GROUP / 8).enumerate() {
            let x = load(c)?;
            let lo = acc.get_mut(2 * k)?;
            *lo = i32x4_add(*lo, i32x4_extmul_low_i16x8(x, hv));
            let hi = acc.get_mut(2 * k + 1)?;
            *hi = i32x4_add(*hi, i32x4_extmul_high_i16x8(x, hv));
        }
        Some(())
    };
    if let Some((hc, mid)) = mid {
        add(&mut acc, mid, i16x8_splat(hc))?;
    }
    for (i, h) in taps.chunks(2).enumerate() {
        let at = C * m.checked_sub(i)?;
        add(
            &mut acc,
            even.get(at..at + C * GROUP)?,
            i16x8_splat(h.first()?.0),
        )?;
    }
    let mut out = [0i32; 32];
    for (d, a) in out.chunks_exact_mut(4).zip(acc) {
        let [d0, d1, d2, d3] = d else { return None };
        *d0 = i32x4_extract_lane::<0>(a);
        *d1 = i32x4_extract_lane::<1>(a);
        *d2 = i32x4_extract_lane::<2>(a);
        *d3 = i32x4_extract_lane::<3>(a);
    }
    Some(out)
}

impl GroupSum for Q15 {
    fn narrow(taps: &[Q15]) -> bool {
        narrow_taps(taps)
    }

    fn sum(
        taps: &[Q15],
        even: &[Q15; GROUP_EVEN],
        mid: Option<&[Q15; GROUP]>,
        narrow: bool,
    ) -> Result<[Q15; GROUP]> {
        if !narrow {
            return sum_wide::<Q15>(taps, even, mid);
        }
        let m = taps.len().saturating_sub(1) / 2;
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            let hc = taps.get(m).map(|h| h.0);
            let mid = mid
                .zip(hc)
                .map(|(x, h)| (h, bytemuck::cast_slice::<Q15, i16>(x)));
            if let Some(r) = narrow_wasm::<1>(taps, bytemuck::cast_slice::<Q15, i16>(even), mid) {
                let mut out = [Q15(0); GROUP];
                for (o, a) in out.iter_mut().zip(r) {
                    *o = finish_i32(a);
                }
                return Ok(out);
            }
        }
        let mut acc = [0i32; GROUP];
        if let Some(mid) = mid {
            let h = i32::from(taps.get(m).ok_or(Error::Internal("halfband centre"))?.0);
            for (a, x) in acc.iter_mut().zip(mid) {
                *a += h * i32::from(x.0);
            }
        }
        let windows = even
            .get(..m + GROUP)
            .ok_or(Error::Internal("halfband group"))?
            .windows(GROUP)
            .rev();
        for (h, win) in taps.chunks(2).zip(windows) {
            let h = i32::from(h.first().ok_or(Error::Internal("halfband group"))?.0);
            let win: &[Q15; GROUP] = win
                .try_into()
                .map_err(|_| Error::Internal("halfband group"))?;
            for (a, x) in acc.iter_mut().zip(win) {
                *a += h * i32::from(x.0);
            }
        }
        Ok(acc.map(finish_i32))
    }
}

impl GroupSum for CQ15 {
    fn narrow(taps: &[Q15]) -> bool {
        narrow_taps(taps)
    }

    fn sum(
        taps: &[Q15],
        even: &[CQ15; GROUP_EVEN],
        mid: Option<&[CQ15; GROUP]>,
        narrow: bool,
    ) -> Result<[CQ15; GROUP]> {
        if !narrow {
            return sum_wide::<CQ15>(taps, even, mid);
        }
        let m = taps.len().saturating_sub(1) / 2;
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            let hc = taps.get(m).map(|h| h.0);
            let mid = mid
                .zip(hc)
                .map(|(x, h)| (h, bytemuck::cast_slice::<CQ15, i16>(x)));
            if let Some(r) = narrow_wasm::<2>(taps, bytemuck::cast_slice::<CQ15, i16>(even), mid) {
                let mut out = [<CQ15 as Sample>::ZERO; GROUP];
                for (o, a) in out.iter_mut().zip(r.chunks_exact(2)) {
                    if let [re, im] = a {
                        *o = CQ15 {
                            re: finish_i32(*re),
                            im: finish_i32(*im),
                        };
                    }
                }
                return Ok(out);
            }
        }
        let mut re = [0i32; GROUP];
        let mut im = [0i32; GROUP];
        if let Some(mid) = mid {
            let h = i32::from(taps.get(m).ok_or(Error::Internal("halfband centre"))?.0);
            for ((r, i), x) in re.iter_mut().zip(im.iter_mut()).zip(mid) {
                *r += h * i32::from(x.re.0);
                *i += h * i32::from(x.im.0);
            }
        }
        let windows = even
            .get(..m + GROUP)
            .ok_or(Error::Internal("halfband group"))?
            .windows(GROUP)
            .rev();
        for (h, win) in taps.chunks(2).zip(windows) {
            let h = i32::from(h.first().ok_or(Error::Internal("halfband group"))?.0);
            let win: &[CQ15; GROUP] = win
                .try_into()
                .map_err(|_| Error::Internal("halfband group"))?;
            for ((r, i), x) in re.iter_mut().zip(im.iter_mut()).zip(win) {
                *r += h * i32::from(x.re.0);
                *i += h * i32::from(x.im.0);
            }
        }
        let mut out = [<CQ15 as Sample>::ZERO; GROUP];
        for ((o, r), i) in out.iter_mut().zip(re).zip(im) {
            *o = CQ15 {
                re: finish_i32(r),
                im: finish_i32(i),
            };
        }
        Ok(out)
    }
}

/// The samples at the even offsets and the centre samples of a group window into `even` and
/// `mid` (`fe` and `fm` map a sample from its index).
#[inline(always)]
#[allow(
    clippy::inline_always,
    reason = "the group size must be a constant in the hot loops"
)]
fn split_window<T: Copy>(
    w: &[T],
    m: usize,
    even: &mut [T; GROUP_EVEN],
    mid: &mut [T; GROUP],
    fe: impl Fn(usize, T) -> T,
    fm: impl Fn(usize, T) -> T,
) -> Result<()> {
    // `w` has an odd length `N + 2 (GROUP - 1)`: pairs (even offset, odd offset), then one more.
    let pairs = w.chunks_exact(2);
    let last = pairs.remainder().first().copied();
    for (j, (d, x)) in even.iter_mut().zip(pairs).enumerate() {
        if let [a, _] = x {
            *d = fe(j, *a);
        }
    }
    if let (Some(x), Some(d)) = (last, even.get_mut(m + GROUP - 1)) {
        *d = fe(m + GROUP - 1, x);
    }
    for (t, (d, x)) in mid
        .iter_mut()
        .zip(w.get(m..).unwrap_or(&[]).chunks(2))
        .enumerate()
    {
        *d = fm(t, *x.first().ok_or(Error::Internal("halfband group"))?);
    }
    Ok(())
}

/// `GROUP` outputs whose windows lie in `w` (two samples apart, `N <= GROUP_MAX_TAPS`): output
/// `t` has the window `w[2 t .. 2 t + N]`, tap `k` multiplies `w[2 t + N - 1 - k]`. The even taps
/// (`k = 2 i`) all read samples at even offsets of `w` (`N - 1` is even), so those are copied
/// once (into `even`, whose first `m + GROUP` places are written) and the products of a tap are
/// then with `GROUP` consecutive samples; the centre tap (`k = m`, odd) reads the odd offsets.
#[inline(never)]
fn group<T: GroupSum>(
    taps: &[T::Tap],
    w: &[T],
    narrow: bool,
    even: &mut [T; GROUP_EVEN],
) -> Result<[T; GROUP]> {
    let m = taps.len().saturating_sub(1) / 2;
    let mut mid = [T::finish(T::zero()); GROUP];
    split_window(w, m, even, &mut mid, |_, x| x, |_, x| x)?;
    T::sum(taps, even, Some(&mid), narrow)
}

fn halfband<T: GroupSum>(
    taps: &[T::Tap],
    hist: &mut [T],
    countdown: &mut usize,
    input: &[T],
    output: &mut [T],
) -> Result<Progress> {
    let p = plan(*countdown, input.len(), output.len());
    let n = taps.len();
    // A group sums `Q15` products in an `i64` without saturation: exact for fewer than 2^32 taps.
    let grouped = n <= GROUP_MAX_TAPS;
    let narrow = grouped && T::narrow(taps);
    let mut even = [T::finish(T::zero()); GROUP_EVEN];
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
            let ys = group::<T>(taps, w, narrow, &mut even)?;
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

/// `GROUP` outputs of `real_to_complex` with the windows in `w` (real input, two samples apart,
/// `N <= GROUP_MAX_TAPS`); `q0` is `n mod 4` of the newest sample of the first window. The real
/// part sums the even taps over the sign-shifted samples at the even offsets of `w`, the
/// imaginary part is the centre tap alone.
#[inline(never)]
fn r2c_group<C>(
    taps: &[C::Real],
    w: &[C::Real],
    q0: usize,
    narrow: bool,
    even: &mut [C::Real; GROUP_EVEN],
) -> Result<[C; GROUP]>
where
    C: ComplexSample,
    C::Real: GroupSum<Tap = C::Real> + ShiftNeg,
{
    type R<C> = <C as ComplexSample>::Real;
    let last = taps.len().saturating_sub(1);
    let m = last / 2;
    let zero = <R<C> as FbMac>::finish(<R<C> as FbMac>::zero());
    // `w[2 j]` has `n mod 4 = q0 + 2 j - last`; the real part is negated when that is 2.
    let centre_flips = (q0 + 4 - (m & 3)) & 3 == 1;
    let mut mid = [zero; GROUP];
    split_window(
        w,
        m,
        even,
        &mut mid,
        |j, x| signed(x, (q0 + 2 * j + 3 * last) & 3 == 2),
        |t, x| signed(x, centre_flips ^ (t & 1 == 1)),
    )?;
    let re = <R<C> as GroupSum>::sum(taps, even, None, narrow)?;
    let mut im = [<R<C> as FbMac>::zero(); GROUP];
    let h = *taps
        .get(m)
        .ok_or(Error::Internal("real_to_complex centre"))?;
    add_taps::<R<C>>(&mut im, h, &mid)?;
    let mut out = [C::ZERO; GROUP];
    for ((o, r), i) in out.iter_mut().zip(re).zip(im) {
        *o = C::new(r, <R<C> as FbMac>::finish(i));
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
    C::Real: GroupSum<Tap = C::Real> + ShiftNeg,
{
    let err = |what| Error::Internal(what);
    let p = plan(*n4 & 1, input.len(), output.len());
    let n = taps.len();
    let grouped = n <= GROUP_MAX_TAPS;
    let narrow = grouped && <C::Real as GroupSum>::narrow(taps);
    let mut even = [<C::Real as FbMac>::finish(<C::Real as FbMac>::zero()); GROUP_EVEN];
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
            let ys = r2c_group::<C>(taps, w, q0, narrow, &mut even)?;
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
