//! Reference implementation of the `fir` kernel: f64 accumulation for `C32`,
//! exact `i64` Q30 accumulation for `CQ15`.

use crate::Reference;
use crate::error::{Error, Result};
use crate::kernels::filter::{Fir, FirState, update_history};
use crate::num::{round_shift, sat16};
use crate::types::{C32, CQ15, ComplexSample, Q15};

/// `x[n-k]` of the stream (history ++ input), `None` if out of range.
fn sample_at<T: Copy>(hist: &[T], input: &[T], n: usize, k: usize) -> Option<T> {
    let past = hist.len();
    let idx = (n + past).checked_sub(k)?;
    if idx < past {
        hist.get(idx).copied()
    } else {
        input.get(idx - past).copied()
    }
}

trait Accumulate: ComplexSample {
    fn filter_one(taps: &[Self::Real], at: impl Fn(usize) -> Option<Self>) -> Option<Self>;
}

impl Accumulate for C32 {
    fn filter_one(taps: &[f32], at: impl Fn(usize) -> Option<C32>) -> Option<C32> {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (k, h) in taps.iter().enumerate() {
            let x = at(k)?;
            re += f64::from(*h) * f64::from(x.re);
            im += f64::from(*h) * f64::from(x.im);
        }
        Some(C32::new(re as f32, im as f32))
    }
}

impl Accumulate for CQ15 {
    fn filter_one(taps: &[Q15], at: impl Fn(usize) -> Option<CQ15>) -> Option<CQ15> {
        let (mut re, mut im) = (0i64, 0i64);
        for (k, h) in taps.iter().enumerate() {
            let x = at(k)?;
            re += i64::from(h.0) * i64::from(x.re.0);
            im += i64::from(h.0) * i64::from(x.im.0);
        }
        Some(CQ15 {
            re: Q15(sat16(round_shift(re, 15))),
            im: Q15(sat16(round_shift(im, 15))),
        })
    }
}

fn fir_impl<T, Taps, Hist>(
    state: &mut FirState<T, Taps, Hist>,
    input: &[T],
    output: &mut [T],
) -> Result<()>
where
    T: Accumulate,
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
    for (n, o) in output.iter_mut().enumerate() {
        *o = T::filter_one(taps, |k| sample_at(hist, input, n, k)).ok_or_else(internal)?;
    }
    update_history(hist, input)
}

impl Fir<C32> for Reference {
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
        fir_impl(state, input, output)
    }
}

impl Fir<CQ15> for Reference {
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
        fir_impl(state, input, output)
    }
}
