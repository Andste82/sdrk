//! Reference implementation of `biquad` and `biquad_real`: direct form I for every sample
//! type, in `f64` for `C32` and `f32` and in exact `i64` integers for `CQ15` and `Q15`.
//!
//! Every sample goes through all sections in turn. The `f32` variants compute each section in
//! `f64` from the `f64` values of the `f32` coefficients and round its output once to `f32`,
//! which is the sample type of the history: `x1, x2, y1, y2` of section `k` are the entries
//! `4k ..= 4k + 3`, and all of them are exact `f32` values, so the result does not depend on
//! how the input is split into calls.

use crate::Reference;
use crate::error::{Error, Result};
use crate::kernels::biquad::{Biquad, BiquadReal, BiquadRealState, BiquadSample, BiquadState};
use crate::kernels::check_len;
use crate::num::{round_shift, sat16};
use crate::types::{C32, CQ15, Q15, Sample};

/// One section of direct form I on one sample type: reads the four history entries, writes the
/// output and the new history.
trait Direct: Sample + BiquadSample {
    fn section(c: &[Self::Coef; 5], hist: &mut [Self], x: Self) -> Option<Self>;
}

/// Direct form I of one `f32` component: `y = b0 x + b1 x1 + b2 x2 - a1 y1 - a2 y2` in
/// `f64`, rounded once to `f32`.
fn df1_f32(c: &[f32; 5], h: [f32; 4], x: f32) -> f32 {
    let [b0, b1, b2, a1, a2] = c.map(f64::from);
    let [x1, x2, y1, y2] = h.map(f64::from);
    (b0 * f64::from(x) + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) as f32
}

/// Direct form I of one Q15 component with Q2.14 coefficients.
fn df1_q15(c: &[i16; 5], h: [i16; 4], x: i16) -> i16 {
    let [b0, b1, b2, a1, a2] = c.map(i64::from);
    let [x1, x2, y1, y2] = h.map(i64::from);
    let acc = b0 * i64::from(x) + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    sat16(round_shift(acc, 14))
}

impl Direct for f32 {
    fn section(c: &[f32; 5], hist: &mut [f32], x: f32) -> Option<f32> {
        let [x1, x2, y1, y2] = hist else { return None };
        let y = df1_f32(c, [*x1, *x2, *y1, *y2], x);
        (*x2, *x1, *y2, *y1) = (*x1, x, *y1, y);
        Some(y)
    }
}

impl Direct for C32 {
    fn section(c: &[f32; 5], hist: &mut [C32], x: C32) -> Option<C32> {
        let [x1, x2, y1, y2] = hist else { return None };
        let re = df1_f32(c, [x1.re, x2.re, y1.re, y2.re], x.re);
        let im = df1_f32(c, [x1.im, x2.im, y1.im, y2.im], x.im);
        let y = C32::new(re, im);
        (*x2, *x1, *y2, *y1) = (*x1, x, *y1, y);
        Some(y)
    }
}

impl Direct for Q15 {
    fn section(c: &[i16; 5], hist: &mut [Q15], x: Q15) -> Option<Q15> {
        let [x1, x2, y1, y2] = hist else { return None };
        let y = Q15(df1_q15(c, [x1.0, x2.0, y1.0, y2.0], x.0));
        (*x2, *x1, *y2, *y1) = (*x1, x, *y1, y);
        Some(y)
    }
}

impl Direct for CQ15 {
    fn section(c: &[i16; 5], hist: &mut [CQ15], x: CQ15) -> Option<CQ15> {
        let [x1, x2, y1, y2] = hist else { return None };
        let re = df1_q15(c, [x1.re.0, x2.re.0, y1.re.0, y2.re.0], x.re.0);
        let im = df1_q15(c, [x1.im.0, x2.im.0, y1.im.0, y2.im.0], x.im.0);
        let y = CQ15 {
            re: Q15(re),
            im: Q15(im),
        };
        (*x2, *x1, *y2, *y1) = (*x1, x, *y1, y);
        Some(y)
    }
}

fn cascade<T: Direct>(
    coefs: &[[T::Coef; 5]],
    hist: &mut [T],
    input: &[T],
    output: &mut [T],
) -> Result<()> {
    check_len(input.len(), output.len())?;
    let internal = || Error::Internal("biquad history");
    for (x, o) in input.iter().zip(output.iter_mut()) {
        let mut v = *x;
        for (c, h) in coefs.iter().zip(hist.chunks_exact_mut(4)) {
            v = T::section(c, h, v).ok_or_else(internal)?;
        }
        *o = v;
    }
    Ok(())
}

macro_rules! impl_reference {
    ($($t:ty),+) => {$(
        impl Biquad<$t> for Reference {
            fn biquad<C, H>(
                &self,
                s: &mut BiquadState<$t, C, H>,
                input: &[$t],
                output: &mut [$t],
            ) -> Result<()>
            where
                C: AsRef<[[<$t as BiquadSample>::Coef; 5]]>,
                H: AsRef<[$t]> + AsMut<[$t]>,
            {
                let (coefs, hist) = s.parts_mut();
                cascade(coefs, hist, input, output)
            }
        }
    )+};
}

macro_rules! impl_reference_real {
    ($($t:ty),+) => {$(
        impl BiquadReal<$t> for Reference {
            fn biquad_real<C, H>(
                &self,
                s: &mut BiquadRealState<$t, C, H>,
                input: &[$t],
                output: &mut [$t],
            ) -> Result<()>
            where
                C: AsRef<[[<$t as BiquadSample>::Coef; 5]]>,
                H: AsRef<[$t]> + AsMut<[$t]>,
            {
                let (coefs, hist) = s.parts_mut();
                cascade(coefs, hist, input, output)
            }
        }
    )+};
}

impl_reference!(C32, CQ15);
impl_reference_real!(f32, Q15);
