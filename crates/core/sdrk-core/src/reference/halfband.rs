//! Reference implementations of `halfband_decimate`, `halfband_decimate_real` and
//! `real_to_complex`: the definition evaluated literally over all taps (`fir_decimate` with
//! factor 2; the shifted signal `u` with its zero components for `real_to_complex`), in `f64`
//! for the float types and in exact `i64` Q30 for the Q15 types.

use super::decim::{RefMac, decimate};
use crate::Reference;
use crate::error::{Error, Result};
use crate::kernels::halfband::{
    HalfbandDecimate, HalfbandDecimateReal, HalfbandRealState, HalfbandState, RealToComplex,
    RealToComplexState, ShiftNeg, shifted,
};
use crate::timing::Progress;
use crate::types::{C32, CQ15, ComplexSample, Q15};

macro_rules! halfband_impl {
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
                decimate::<$t>(p.taps, p.hist, 2, p.countdown, input, output)
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

/// Per input sample: the shifted sample `u[n] = x[n] (-j)^n`, then the output
/// `sum_k h[k] u[n - k]` over all taps when `n` is even.
fn real_to_complex<C>(
    taps: &[C::Real],
    hist: &mut [C],
    n4: &mut usize,
    input: &[C::Real],
    output: &mut [C],
) -> Result<Progress>
where
    C: ComplexSample + RefMac<Tap = C::Real>,
    C::Real: ShiftNeg,
{
    let (mut consumed, mut produced) = (0usize, 0usize);
    for x in input {
        let u = shifted::<C>(*x, *n4);
        if n4.is_multiple_of(2) {
            let Some(slot) = output.get_mut(produced) else {
                break;
            };
            let mut acc = C::zero();
            for (k, h) in taps.iter().enumerate() {
                let v = if k == 0 {
                    u
                } else {
                    hist.len()
                        .checked_sub(k)
                        .and_then(|at| hist.get(at))
                        .copied()
                        .ok_or(Error::Internal("real_to_complex window"))?
                };
                C::mac(&mut acc, *h, v);
            }
            *slot = C::finish(acc);
            produced += 1;
        }
        // Push `u` into the history.
        hist.rotate_left(1);
        if let Some(last) = hist.last_mut() {
            *last = u;
        }
        *n4 = (*n4 + 1) & 3;
        consumed += 1;
    }
    Ok(Progress { consumed, produced })
}

macro_rules! real_to_complex_impl {
    ($t:ty) => {
        impl RealToComplex<$t> for Reference {
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
