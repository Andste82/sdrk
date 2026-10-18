//! Fallback implementation of the fused kernels: the chaining functions with `self`.

use crate::Fallback;
use crate::error::Result;
use crate::fused::{ddc_chained, duc_chained, fm_demod_chained};
use crate::kernels::decim::FirDecimState;
use crate::kernels::demod::{DeemphasisState, QuadDemodState};
use crate::kernels::fused::{Ddc, Duc, FmDemod};
use crate::kernels::interp::FirInterpState;
use crate::kernels::mix::Nco;
use crate::timing::Progress;
use crate::types::{C32, CQ15, ComplexSample};

macro_rules! fused_impl {
    ($t:ty) => {
        impl Ddc<$t> for Fallback {
            fn ddc<Taps, Hist>(
                &self,
                nco: &mut Nco,
                fir: &mut FirDecimState<$t, Taps, Hist>,
                input: &[$t],
                output: &mut [$t],
                scratch: &mut [$t],
            ) -> Result<Progress>
            where
                Taps: AsRef<[<$t as ComplexSample>::Real]>,
                Hist: AsRef<[$t]> + AsMut<[$t]>,
            {
                ddc_chained(self, nco, fir, input, output, scratch)
            }
        }

        impl Duc<$t> for Fallback {
            fn duc<Taps, Hist>(
                &self,
                fir: &mut FirInterpState<$t, Taps, Hist>,
                nco: &mut Nco,
                input: &[$t],
                output: &mut [$t],
                scratch: &mut [$t],
            ) -> Result<Progress>
            where
                Taps: AsRef<[<$t as ComplexSample>::Real]>,
                Hist: AsRef<[$t]> + AsMut<[$t]>,
            {
                duc_chained(self, fir, nco, input, output, scratch)
            }
        }

        impl FmDemod<$t> for Fallback {
            fn fm_demod(
                &self,
                quad: &mut QuadDemodState<$t>,
                deemph: &mut DeemphasisState<<$t as ComplexSample>::Real>,
                input: &[$t],
                output_audio: &mut [<$t as ComplexSample>::Real],
                scratch_real: &mut [<$t as ComplexSample>::Real],
                gain: <$t as ComplexSample>::Real,
            ) -> Result<()> {
                fm_demod_chained(self, quad, deemph, input, output_audio, scratch_real, gain)
            }
        }
    };
}

fused_impl!(C32);
fused_impl!(CQ15);
