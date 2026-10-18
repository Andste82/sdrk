//! Reference implementations of `fir_decimate` and `resample`: f64 accumulation for the
//! float types, exact `i64` Q30 accumulation for the Q15 types.

use crate::Reference;
use crate::error::{Error, Result};
use crate::kernels::decim::{
    FirDecimState, FirDecimate, Resample, ResampleReal, ResampleRealState, ResampleState,
};
use crate::kernels::filter::update_history;
use crate::num::{round_shift, sat16};
use crate::reference::filter::sample_at;
use crate::timing::Progress;
use crate::types::{C32, CQ15, Q15};

/// Exact (or f64) multiply-accumulate rules of the reference for one sample type.
pub(super) trait RefMac: Copy {
    type Tap: Copy;
    type Acc: Copy;
    fn zero() -> Self::Acc;
    fn mac(acc: &mut Self::Acc, h: Self::Tap, x: Self);
    fn finish(acc: Self::Acc) -> Self;
}

impl RefMac for C32 {
    type Tap = f32;
    type Acc = (f64, f64);
    fn zero() -> Self::Acc {
        (0.0, 0.0)
    }
    fn mac(acc: &mut Self::Acc, h: f32, x: C32) {
        acc.0 += f64::from(h) * f64::from(x.re);
        acc.1 += f64::from(h) * f64::from(x.im);
    }
    fn finish(acc: Self::Acc) -> C32 {
        C32::new(acc.0 as f32, acc.1 as f32)
    }
}

impl RefMac for CQ15 {
    type Tap = Q15;
    type Acc = (i64, i64);
    fn zero() -> Self::Acc {
        (0, 0)
    }
    fn mac(acc: &mut Self::Acc, h: Q15, x: CQ15) {
        acc.0 += i64::from(h.0) * i64::from(x.re.0);
        acc.1 += i64::from(h.0) * i64::from(x.im.0);
    }
    fn finish(acc: Self::Acc) -> CQ15 {
        CQ15 {
            re: Q15(sat16(round_shift(acc.0, 15))),
            im: Q15(sat16(round_shift(acc.1, 15))),
        }
    }
}

impl RefMac for f32 {
    type Tap = f32;
    type Acc = f64;
    fn zero() -> f64 {
        0.0
    }
    fn mac(acc: &mut f64, h: f32, x: f32) {
        *acc += f64::from(h) * f64::from(x);
    }
    fn finish(acc: f64) -> f32 {
        acc as f32
    }
}

impl RefMac for Q15 {
    type Tap = Q15;
    type Acc = i64;
    fn zero() -> i64 {
        0
    }
    fn mac(acc: &mut i64, h: Q15, x: Q15) {
        *acc += i64::from(h.0) * i64::from(x.0);
    }
    fn finish(acc: i64) -> Q15 {
        Q15(sat16(round_shift(acc, 15)))
    }
}

/// `y_full[n]` of the stream (history ++ input): the sum over all taps of `h[k] x[n-k]`.
fn full_output<T: RefMac>(taps: &[T::Tap], hist: &[T], input: &[T], n: usize) -> Result<T> {
    let mut acc = T::zero();
    for (k, h) in taps.iter().enumerate() {
        let x = sample_at(hist, input, n, k).ok_or(Error::Internal("decimation window"))?;
        T::mac(&mut acc, *h, x);
    }
    Ok(T::finish(acc))
}

fn decimate<T: RefMac>(
    taps: &[T::Tap],
    hist: &mut [T],
    factor: usize,
    countdown: &mut usize,
    input: &[T],
    output: &mut [T],
) -> Result<Progress> {
    let internal = || Error::Internal("decimation window");
    let (mut consumed, mut produced) = (0usize, 0usize);
    for n in 0..input.len() {
        if *countdown == 0 {
            let Some(slot) = output.get_mut(produced) else {
                break;
            };
            *slot = full_output(taps, hist, input, n)?;
            produced += 1;
            *countdown = factor - 1;
        } else {
            *countdown -= 1;
        }
        consumed = n + 1;
    }
    update_history(hist, input.get(..consumed).ok_or_else(internal)?)?;
    Ok(Progress { consumed, produced })
}

/// Evaluates `y[j] = sum_k h[k] u[j M - k]` literally on the upsampled signal `u`.
///
/// With `q = j M - L c` (`c` samples consumed, `-L <= q < 0`) the tap `k` meets an
/// input sample iff `(k - q)` is a multiple of `L`; that sample is
/// `(k - q) / L - 1` places before the newest consumed one.
fn upsampled_output<T: RefMac>(
    taps: &[T::Tap],
    hist: &[T],
    input: &[T],
    consumed: usize,
    up: usize,
    q: i64,
) -> Result<T> {
    let internal = || Error::Internal("resampler window");
    let l = up as i64;
    let mut acc = T::zero();
    for (k, h) in taps.iter().enumerate() {
        let offset = k as i64 - q;
        if offset % l != 0 {
            continue;
        }
        let back = usize::try_from(offset / l - 1).map_err(|_| internal())?;
        // `consumed` input samples follow the history; `back = 0` is the newest sample.
        let x = sample_at(hist, input, consumed, back + 1).ok_or_else(internal)?;
        T::mac(&mut acc, *h, x);
    }
    Ok(T::finish(acc))
}

#[allow(clippy::too_many_arguments)]
fn rational<T: RefMac>(
    taps: &[T::Tap],
    hist: &mut [T],
    up: usize,
    down: usize,
    q: &mut i64,
    input: &[T],
    output: &mut [T],
) -> Result<Progress> {
    let internal = || Error::Internal("resampler window");
    let (l, m) = (up as i64, down as i64);
    let (mut consumed, mut produced) = (0usize, 0usize);
    loop {
        if *q < 0 {
            let Some(slot) = output.get_mut(produced) else {
                break;
            };
            *slot = upsampled_output(taps, hist, input, consumed, up, *q)?;
            produced += 1;
            *q += m;
        } else {
            if consumed == input.len() {
                break;
            }
            consumed += 1;
            *q -= l;
        }
    }
    update_history(hist, input.get(..consumed).ok_or_else(internal)?)?;
    Ok(Progress { consumed, produced })
}

macro_rules! fir_decimate_impl {
    ($t:ty, $tap:ty) => {
        impl FirDecimate<$t> for Reference {
            fn fir_decimate<Taps, Hist>(
                &self,
                state: &mut FirDecimState<$t, Taps, Hist>,
                input: &[$t],
                output: &mut [$t],
            ) -> Result<Progress>
            where
                Taps: AsRef<[$tap]>,
                Hist: AsRef<[$t]> + AsMut<[$t]>,
            {
                let p = state.parts_mut();
                decimate::<$t>(p.taps, p.hist, p.factor, p.countdown, input, output)
            }
        }
    };
}

fir_decimate_impl!(C32, f32);
fir_decimate_impl!(CQ15, Q15);

macro_rules! resample_impl {
    ($tr:ident, $method:ident, $state:ident, $t:ty, $tap:ty) => {
        impl $tr<$t> for Reference {
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
                rational::<$t>(p.taps, p.hist, p.up, p.down, p.q, input, output)
            }
        }
    };
}

resample_impl!(Resample, resample, ResampleState, C32, f32);
resample_impl!(Resample, resample, ResampleState, CQ15, Q15);
resample_impl!(ResampleReal, resample_real, ResampleRealState, f32, f32);
resample_impl!(ResampleReal, resample_real, ResampleRealState, Q15, Q15);
