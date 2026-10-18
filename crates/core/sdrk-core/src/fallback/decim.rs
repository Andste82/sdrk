//! Fallback implementations of `fir_decimate` and `resample`.

use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::decim::{
    FirDecimState, FirDecimate, Resample, ResampleReal, ResampleRealState, ResampleState,
};
use crate::kernels::filter::update_history;
use crate::num::Acc64;
use crate::timing::Progress;
use crate::types::{C32, CQ15, Q15};

/// Multiply-accumulate rules of the fallback for one sample type.
pub(super) trait FbMac: Copy {
    /// Tap type.
    type Tap: Copy;
    /// Accumulator.
    type Acc: Copy;
    fn zero() -> Self::Acc;
    fn mac(acc: &mut Self::Acc, h: Self::Tap, x: Self);
    fn finish(acc: Self::Acc) -> Self;
}

impl FbMac for C32 {
    type Tap = f32;
    type Acc = (f32, f32);
    #[inline]
    fn zero() -> Self::Acc {
        (0.0, 0.0)
    }
    #[inline]
    fn mac(acc: &mut Self::Acc, h: f32, x: C32) {
        acc.0 += h * x.re;
        acc.1 += h * x.im;
    }
    #[inline]
    fn finish(acc: Self::Acc) -> C32 {
        C32::new(acc.0, acc.1)
    }
}

impl FbMac for CQ15 {
    type Tap = Q15;
    type Acc = (Acc64, Acc64);
    #[inline]
    fn zero() -> Self::Acc {
        (Acc64::default(), Acc64::default())
    }
    #[inline]
    fn mac(acc: &mut Self::Acc, h: Q15, x: CQ15) {
        acc.0.mac(h, x.re);
        acc.1.mac(h, x.im);
    }
    #[inline]
    fn finish(acc: Self::Acc) -> CQ15 {
        CQ15 {
            re: acc.0.finish(),
            im: acc.1.finish(),
        }
    }
}

impl FbMac for f32 {
    type Tap = f32;
    type Acc = f32;
    #[inline]
    fn zero() -> f32 {
        0.0
    }
    #[inline]
    fn mac(acc: &mut f32, h: f32, x: f32) {
        *acc += h * x;
    }
    #[inline]
    fn finish(acc: f32) -> f32 {
        acc
    }
}

impl FbMac for Q15 {
    type Tap = Q15;
    type Acc = Acc64;
    #[inline]
    fn zero() -> Acc64 {
        Acc64::default()
    }
    #[inline]
    fn mac(acc: &mut Acc64, h: Q15, x: Q15) {
        acc.mac(h, x);
    }
    #[inline]
    fn finish(acc: Acc64) -> Q15 {
        acc.finish()
    }
}

/// Dot product of the taps (in the given order) with the samples, newest first.
#[inline]
fn dot<'a, T: FbMac + 'a>(
    taps: impl Iterator<Item = &'a T::Tap>,
    newest_first: impl Iterator<Item = &'a T>,
) -> T {
    let mut acc = T::zero();
    for (h, x) in taps.zip(newest_first) {
        T::mac(&mut acc, *h, *x);
    }
    T::finish(acc)
}

fn decimate<T: FbMac>(
    taps: &[T::Tap],
    hist: &mut [T],
    factor: usize,
    countdown: &mut usize,
    input: &[T],
    output: &mut [T],
) -> Result<Progress> {
    let internal = || Error::Internal("decimation window");
    let n_taps = taps.len();
    let (mut consumed, mut produced) = (0usize, 0usize);
    for n in 0..input.len() {
        if *countdown == 0 {
            let Some(slot) = output.get_mut(produced) else {
                break;
            };
            let upto = input.get(..=n).ok_or_else(internal)?;
            *slot = if n + 1 >= n_taps {
                let window = input.get(n + 1 - n_taps..=n).ok_or_else(internal)?;
                dot::<T>(taps.iter(), window.iter().rev())
            } else {
                dot::<T>(taps.iter(), upto.iter().rev().chain(hist.iter().rev()))
            };
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

#[allow(clippy::too_many_arguments)]
fn polyphase<T: FbMac>(
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
            let phase = usize::try_from(*q + l).map_err(|_| internal())?;
            let seen = input.get(..consumed).ok_or_else(internal)?;
            *slot = dot::<T>(
                taps.iter().skip(phase).step_by(up),
                seen.iter().rev().chain(hist.iter().rev()),
            );
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
        impl FirDecimate<$t> for Fallback {
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
                polyphase::<$t>(p.taps, p.hist, p.up, p.down, p.q, input, output)
            }
        }
    };
}

resample_impl!(Resample, resample, ResampleState, C32, f32);
resample_impl!(Resample, resample, ResampleState, CQ15, Q15);
resample_impl!(ResampleReal, resample_real, ResampleRealState, f32, f32);
resample_impl!(ResampleReal, resample_real, ResampleRealState, Q15, Q15);
