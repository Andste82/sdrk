//! Fallback implementation of the `fir` kernel.

use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::filter::{Fir, FirState, update_history};
use crate::num::Acc64;
use crate::types::{C32, CQ15, ComplexSample, Q15};

/// Dot product of the taps with the newest samples first (`x[n]`, `x[n-1]`, ...).
trait Dot: ComplexSample {
    fn dot<'a>(taps: &[Self::Real], newest_first: impl Iterator<Item = &'a Self>) -> Self;
}

impl Dot for C32 {
    #[inline]
    fn dot<'a>(taps: &[f32], newest_first: impl Iterator<Item = &'a C32>) -> C32 {
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (h, x) in taps.iter().zip(newest_first) {
            re += h * x.re;
            im += h * x.im;
        }
        C32::new(re, im)
    }
}

impl Dot for CQ15 {
    #[inline]
    fn dot<'a>(taps: &[Q15], newest_first: impl Iterator<Item = &'a CQ15>) -> CQ15 {
        let (mut re, mut im) = (Acc64::default(), Acc64::default());
        for (h, x) in taps.iter().zip(newest_first) {
            re.mac(*h, x.re);
            im.mac(*h, x.im);
        }
        CQ15 {
            re: re.finish(),
            im: im.finish(),
        }
    }
}

fn fir_generic<T, Taps, Hist>(
    state: &mut FirState<T, Taps, Hist>,
    input: &[T],
    output: &mut [T],
) -> Result<()>
where
    T: Dot,
    Taps: AsRef<[T::Real]>,
    Hist: AsRef<[T]> + AsMut<[T]>,
{
    if input.len() != output.len() {
        return Err(Error::LengthMismatch {
            expected: input.len(),
            got: output.len(),
        });
    }
    let internal = || Error::Internal("fir window");
    let (taps, hist) = state.parts_mut();
    let n_taps = taps.len();
    for (n, o) in output.iter_mut().enumerate() {
        // x[n], x[n-1], ..., then the history from its newest sample backwards.
        let upto = input.get(..=n).ok_or_else(internal)?;
        *o = if n + 1 >= n_taps {
            let window = input.get(n + 1 - n_taps..=n).ok_or_else(internal)?;
            T::dot(taps, window.iter().rev())
        } else {
            T::dot(taps, upto.iter().rev().chain(hist.iter().rev()))
        };
    }
    update_history(hist, input)
}

impl Fir<C32> for Fallback {
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
        fir_generic(state, input, output)
    }
}

impl Fir<CQ15> for Fallback {
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
        fir_generic(state, input, output)
    }
}
