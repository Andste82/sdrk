//! Fallback implementation of `biquad` and `biquad_real`.
//!
//! The cascade runs section by section over the whole buffer (the first section reads the
//! input, the others filter the output in place), so the two state values of a section stay in
//! registers for the length of the buffer. The result is the same as running every sample
//! through all sections: a section only depends on its own input stream.

use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::biquad::{Biquad, BiquadReal, BiquadRealState, BiquadSample, BiquadState};
use crate::kernels::check_len;
use crate::num::{flush_denormal, round_shift, sat16};
use crate::types::{C32, CQ15, Q15, Sample};

/// One section of one sample type: its state, loaded from and stored to four history entries.
trait Section: Sample + BiquadSample {
    type State: Copy;

    fn load(hist: &[Self]) -> Option<Self::State>;
    fn store(state: &Self::State, hist: &mut [Self]);
    fn step(c: &[Self::Coef; 5], state: &mut Self::State, x: Self) -> Self;
}

/// Transposed direct form II of one `f32` component; returns `(y, s1, s2)`.
#[inline]
fn df2t(c: &[f32; 5], s1: f32, s2: f32, x: f32) -> (f32, f32, f32) {
    let [b0, b1, b2, a1, a2] = *c;
    let y = b0 * x + s1;
    let n1 = (b1 * x - a1 * y) + s2;
    let n2 = b2 * x - a2 * y;
    (y, flush_denormal(n1), flush_denormal(n2))
}

/// Direct form I of one Q15 component with Q2.14 coefficients: the output and the new
/// `[x1, x2, y1, y2]`.
#[inline]
fn df1(c: &[i16; 5], h: [i16; 4], x: i16) -> (i16, [i16; 4]) {
    let [b0, b1, b2, a1, a2] = c.map(i64::from);
    let [x1, x2, y1, y2] = h.map(i64::from);
    let acc = b0 * i64::from(x) + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    let y = sat16(round_shift(acc, 14));
    (y, [x, h[0], y, h[2]])
}

impl Section for f32 {
    type State = [f32; 2];

    #[inline]
    fn load(hist: &[f32]) -> Option<[f32; 2]> {
        let [s1, s2, ..] = *hist else { return None };
        Some([s1, s2])
    }

    #[inline]
    fn store(state: &[f32; 2], hist: &mut [f32]) {
        if let [h1, h2, ..] = hist {
            (*h1, *h2) = (state[0], state[1]);
        }
    }

    #[inline]
    fn step(c: &[f32; 5], state: &mut [f32; 2], x: f32) -> f32 {
        let (y, s1, s2) = df2t(c, state[0], state[1], x);
        *state = [s1, s2];
        y
    }
}

impl Section for C32 {
    type State = [C32; 2];

    #[inline]
    fn load(hist: &[C32]) -> Option<[C32; 2]> {
        let [s1, s2, ..] = *hist else { return None };
        Some([s1, s2])
    }

    #[inline]
    fn store(state: &[C32; 2], hist: &mut [C32]) {
        if let [h1, h2, ..] = hist {
            (*h1, *h2) = (state[0], state[1]);
        }
    }

    #[inline]
    fn step(c: &[f32; 5], state: &mut [C32; 2], x: C32) -> C32 {
        let (yr, r1, r2) = df2t(c, state[0].re, state[1].re, x.re);
        let (yi, i1, i2) = df2t(c, state[0].im, state[1].im, x.im);
        *state = [C32::new(r1, i1), C32::new(r2, i2)];
        C32::new(yr, yi)
    }
}

impl Section for Q15 {
    type State = [i16; 4];

    #[inline]
    fn load(hist: &[Q15]) -> Option<[i16; 4]> {
        let [x1, x2, y1, y2] = *hist else { return None };
        Some([x1.0, x2.0, y1.0, y2.0])
    }

    #[inline]
    fn store(state: &[i16; 4], hist: &mut [Q15]) {
        if let [x1, x2, y1, y2] = hist {
            (*x1, *x2, *y1, *y2) = (Q15(state[0]), Q15(state[1]), Q15(state[2]), Q15(state[3]));
        }
    }

    #[inline]
    fn step(c: &[i16; 5], state: &mut [i16; 4], x: Q15) -> Q15 {
        let (y, next) = df1(c, *state, x.0);
        *state = next;
        Q15(y)
    }
}

impl Section for CQ15 {
    type State = [[i16; 4]; 2];

    #[inline]
    fn load(hist: &[CQ15]) -> Option<[[i16; 4]; 2]> {
        let [x1, x2, y1, y2] = *hist else { return None };
        Some([
            [x1.re.0, x2.re.0, y1.re.0, y2.re.0],
            [x1.im.0, x2.im.0, y1.im.0, y2.im.0],
        ])
    }

    #[inline]
    fn store(state: &[[i16; 4]; 2], hist: &mut [CQ15]) {
        let [re, im] = *state;
        if let [x1, x2, y1, y2] = hist {
            let pack = |i: usize| CQ15 {
                re: Q15(re.get(i).copied().unwrap_or(0)),
                im: Q15(im.get(i).copied().unwrap_or(0)),
            };
            (*x1, *x2, *y1, *y2) = (pack(0), pack(1), pack(2), pack(3));
        }
    }

    #[inline]
    fn step(c: &[i16; 5], state: &mut [[i16; 4]; 2], x: CQ15) -> CQ15 {
        let (yr, re) = df1(c, state[0], x.re.0);
        let (yi, im) = df1(c, state[1], x.im.0);
        *state = [re, im];
        CQ15 {
            re: Q15(yr),
            im: Q15(yi),
        }
    }
}

fn cascade<T: Section>(
    coefs: &[[T::Coef; 5]],
    hist: &mut [T],
    input: &[T],
    output: &mut [T],
) -> Result<()> {
    check_len(input.len(), output.len())?;
    let internal = || Error::Internal("biquad history");
    for (k, (c, h)) in coefs.iter().zip(hist.chunks_exact_mut(4)).enumerate() {
        let mut state = T::load(h).ok_or_else(internal)?;
        if k == 0 {
            for (x, o) in input.iter().zip(output.iter_mut()) {
                *o = T::step(c, &mut state, *x);
            }
        } else {
            for o in output.iter_mut() {
                *o = T::step(c, &mut state, *o);
            }
        }
        T::store(&state, h);
    }
    Ok(())
}

macro_rules! impl_fallback {
    ($($t:ty),+) => {$(
        impl Biquad<$t> for Fallback {
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

impl_fallback!(C32, CQ15);

macro_rules! impl_fallback_real {
    ($($t:ty),+) => {$(
        impl BiquadReal<$t> for Fallback {
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

impl_fallback_real!(f32, Q15);
