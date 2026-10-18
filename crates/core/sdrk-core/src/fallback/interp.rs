//! Fallback implementation of `fir_interpolate`: the polyphase resampler with `down = 1`.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::decim::{Resample, ResampleReal};
use crate::kernels::interp::{
    FirInterpRealState, FirInterpState, FirInterpolate, FirInterpolateReal,
};
use crate::timing::Progress;
use crate::types::{C32, CQ15, Q15};

macro_rules! interpolate_impl {
    ($tr:ident, $method:ident, $inner:ident, $resample:ident, $state:ident, $t:ty, $tap:ty) => {
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
                $inner::<$t>::$resample(self, state.inner_mut(), input, output)
            }
        }
    };
}

interpolate_impl!(
    FirInterpolate,
    fir_interpolate,
    Resample,
    resample,
    FirInterpState,
    C32,
    f32
);
interpolate_impl!(
    FirInterpolate,
    fir_interpolate,
    Resample,
    resample,
    FirInterpState,
    CQ15,
    Q15
);
interpolate_impl!(
    FirInterpolateReal,
    fir_interpolate_real,
    ResampleReal,
    resample_real,
    FirInterpRealState,
    f32,
    f32
);
interpolate_impl!(
    FirInterpolateReal,
    fir_interpolate_real,
    ResampleReal,
    resample_real,
    FirInterpRealState,
    Q15,
    Q15
);
