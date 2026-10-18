//! The single place that maps kernels to implementations.

use crate::{C32, CQ15, Chained, Fallback, Progress, Q15, Result};
use sdrk_core::fused::{ddc_chained, duc_chained, fm_demod_chained};
use sdrk_core::kernels::basic::Scale;
use sdrk_core::kernels::convert::{Convert, ConvertOut, ConvertOutReal, ConvertReal};
use sdrk_core::kernels::decim::{
    FirDecimState, FirDecimate, Resample, ResampleReal, ResampleRealState, ResampleState,
};
use sdrk_core::kernels::demod::{
    DcBlock, DcBlockState, Deemphasis, DeemphasisState, QuadDemod, QuadDemodState,
};
use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, FftScale, RealFft, RealFftPlan, Scaling};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::fused::{Ddc, Duc, FmDemod};
use sdrk_core::kernels::interp::{
    FirInterpRealState, FirInterpState, FirInterpolate, FirInterpolateReal,
};
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::kernels::modulate::{FmMod, FmModState};
use sdrk_core::kernels::spectrum::{MagSq, PowerDb, Window};
use sdrk_device::SampleFormat;

crate::delegate! {
    Scale<C32>::scale(&self, input: &[C32], output: &mut [C32], gain: f32) -> Result<()> => Fallback;
    Scale<CQ15>::scale(&self, input: &[CQ15], output: &mut [CQ15], gain: Q15) -> Result<()> => Fallback;
    Convert<C32>::convert(&self, format: &SampleFormat, input: &[u8], output: &mut [C32]) -> Result<usize> => Fallback;
    Convert<CQ15>::convert(&self, format: &SampleFormat, input: &[u8], output: &mut [CQ15]) -> Result<usize> => Fallback;
    ConvertReal<f32>::convert_real(&self, format: &SampleFormat, input: &[u8], output: &mut [f32]) -> Result<usize> => Fallback;
    ConvertReal<Q15>::convert_real(&self, format: &SampleFormat, input: &[u8], output: &mut [Q15]) -> Result<usize> => Fallback;
    Rotate<C32>::rotate(&self, nco: &mut Nco, input: &[C32], output: &mut [C32]) -> Result<()> => Fallback;
    Rotate<CQ15>::rotate(&self, nco: &mut Nco, input: &[CQ15], output: &mut [CQ15]) -> Result<()> => Fallback;
    Fir<C32>::fir[Taps: AsRef<[f32]>, Hist: AsRef<[C32]> + AsMut<[C32]>](&self, state: &mut FirState<C32, Taps, Hist>, input: &[C32], output: &mut [C32]) -> Result<()> => Fallback;
    Fir<CQ15>::fir[Taps: AsRef<[Q15]>, Hist: AsRef<[CQ15]> + AsMut<[CQ15]>](&self, state: &mut FirState<CQ15, Taps, Hist>, input: &[CQ15], output: &mut [CQ15]) -> Result<()> => Fallback;
    FirDecimate<C32>::fir_decimate[Taps: AsRef<[f32]>, Hist: AsRef<[C32]> + AsMut<[C32]>](&self, state: &mut FirDecimState<C32, Taps, Hist>, input: &[C32], output: &mut [C32]) -> Result<Progress> => Fallback;
    FirDecimate<CQ15>::fir_decimate[Taps: AsRef<[Q15]>, Hist: AsRef<[CQ15]> + AsMut<[CQ15]>](&self, state: &mut FirDecimState<CQ15, Taps, Hist>, input: &[CQ15], output: &mut [CQ15]) -> Result<Progress> => Fallback;
    Resample<C32>::resample[Taps: AsRef<[f32]>, Hist: AsRef<[C32]> + AsMut<[C32]>](&self, state: &mut ResampleState<C32, Taps, Hist>, input: &[C32], output: &mut [C32]) -> Result<Progress> => Fallback;
    Resample<CQ15>::resample[Taps: AsRef<[Q15]>, Hist: AsRef<[CQ15]> + AsMut<[CQ15]>](&self, state: &mut ResampleState<CQ15, Taps, Hist>, input: &[CQ15], output: &mut [CQ15]) -> Result<Progress> => Fallback;
    ResampleReal<f32>::resample_real[Taps: AsRef<[f32]>, Hist: AsRef<[f32]> + AsMut<[f32]>](&self, state: &mut ResampleRealState<f32, Taps, Hist>, input: &[f32], output: &mut [f32]) -> Result<Progress> => Fallback;
    ResampleReal<Q15>::resample_real[Taps: AsRef<[Q15]>, Hist: AsRef<[Q15]> + AsMut<[Q15]>](&self, state: &mut ResampleRealState<Q15, Taps, Hist>, input: &[Q15], output: &mut [Q15]) -> Result<Progress> => Fallback;
    QuadDemod<C32>::quad_demod(&self, state: &mut QuadDemodState<C32>, input: &[C32], output: &mut [f32], gain: f32) -> Result<()> => Fallback;
    QuadDemod<CQ15>::quad_demod(&self, state: &mut QuadDemodState<CQ15>, input: &[CQ15], output: &mut [Q15], gain: Q15) -> Result<()> => Fallback;
    Deemphasis<f32>::deemphasis(&self, state: &mut DeemphasisState<f32>, input: &[f32], output: &mut [f32]) -> Result<()> => Fallback;
    Deemphasis<Q15>::deemphasis(&self, state: &mut DeemphasisState<Q15>, input: &[Q15], output: &mut [Q15]) -> Result<()> => Fallback;
    DcBlock<C32>::dc_block(&self, state: &mut DcBlockState<C32>, input: &[C32], output: &mut [C32]) -> Result<()> => Fallback;
    DcBlock<CQ15>::dc_block(&self, state: &mut DcBlockState<CQ15>, input: &[CQ15], output: &mut [CQ15]) -> Result<()> => Fallback;
    Window<C32>::window(&self, input: &[C32], output: &mut [C32], w: &[f32]) -> Result<()> => Fallback;
    Window<CQ15>::window(&self, input: &[CQ15], output: &mut [CQ15], w: &[Q15]) -> Result<()> => Fallback;
    MagSq<C32>::mag_sq(&self, input: &[C32], output: &mut [f32]) -> Result<()> => Fallback;
    MagSq<CQ15>::mag_sq(&self, input: &[CQ15], output: &mut [Q15]) -> Result<()> => Fallback;
    PowerDb<f32>::power_db(&self, input: &[f32], output: &mut [f32], offset_db: f32) -> Result<()> => Fallback;
    PowerDb<Q15>::power_db(&self, input: &[Q15], output: &mut [f32], offset_db: f32) -> Result<()> => Fallback;
    Fft<C32>::fft[Tw: AsRef<[C32]>](&self, plan: &FftPlan<C32, Tw>, data: &mut [C32], dir: Direction, scaling: Scaling) -> Result<FftScale> => Fallback;
    Fft<CQ15>::fft[Tw: AsRef<[CQ15]>](&self, plan: &FftPlan<CQ15, Tw>, data: &mut [CQ15], dir: Direction, scaling: Scaling) -> Result<FftScale> => Fallback;
    ConvertOut<C32>::convert_out(&self, format: &SampleFormat, input: &[C32], output: &mut [u8]) -> Result<usize> => Fallback;
    ConvertOut<CQ15>::convert_out(&self, format: &SampleFormat, input: &[CQ15], output: &mut [u8]) -> Result<usize> => Fallback;
    ConvertOutReal<f32>::convert_out_real(&self, format: &SampleFormat, input: &[f32], output: &mut [u8]) -> Result<usize> => Fallback;
    ConvertOutReal<Q15>::convert_out_real(&self, format: &SampleFormat, input: &[Q15], output: &mut [u8]) -> Result<usize> => Fallback;
    FirInterpolate<C32>::fir_interpolate[Taps: AsRef<[f32]>, Hist: AsRef<[C32]> + AsMut<[C32]>](&self, state: &mut FirInterpState<C32, Taps, Hist>, input: &[C32], output: &mut [C32]) -> Result<Progress> => Fallback;
    FirInterpolate<CQ15>::fir_interpolate[Taps: AsRef<[Q15]>, Hist: AsRef<[CQ15]> + AsMut<[CQ15]>](&self, state: &mut FirInterpState<CQ15, Taps, Hist>, input: &[CQ15], output: &mut [CQ15]) -> Result<Progress> => Fallback;
    FirInterpolateReal<f32>::fir_interpolate_real[Taps: AsRef<[f32]>, Hist: AsRef<[f32]> + AsMut<[f32]>](&self, state: &mut FirInterpRealState<f32, Taps, Hist>, input: &[f32], output: &mut [f32]) -> Result<Progress> => Fallback;
    FirInterpolateReal<Q15>::fir_interpolate_real[Taps: AsRef<[Q15]>, Hist: AsRef<[Q15]> + AsMut<[Q15]>](&self, state: &mut FirInterpRealState<Q15, Taps, Hist>, input: &[Q15], output: &mut [Q15]) -> Result<Progress> => Fallback;
    FmMod<C32>::fm_mod(&self, s: &mut FmModState, input: &[f32], output: &mut [C32]) -> Result<()> => Fallback;
    FmMod<CQ15>::fm_mod(&self, s: &mut FmModState, input: &[Q15], output: &mut [CQ15]) -> Result<()> => Fallback;
    @extra [
        ("rfft", f32, Fallback),
        ("irfft", f32, Fallback),
        ("ddc", C32, Chained),
        ("ddc", CQ15, Chained),
        ("duc", C32, Chained),
        ("duc", CQ15, Chained),
        ("fm_demod", C32, Chained),
        ("fm_demod", CQ15, Chained)
    ]
}

// `RealFft` has no sample-type parameter, so it is delegated by hand; the two rows above
// register it in the `backend_for` table.
impl RealFft for crate::Sdrk {
    #[inline]
    fn rfft<Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>>(
        &self,
        plan: &RealFftPlan<Tw, Tw2>,
        input: &[f32],
        output: &mut [C32],
        scratch: &mut [C32],
    ) -> Result<FftScale> {
        <Fallback as RealFft>::rfft(&Fallback, plan, input, output, scratch)
    }

    #[inline]
    fn irfft<Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>>(
        &self,
        plan: &RealFftPlan<Tw, Tw2>,
        input: &[C32],
        output: &mut [f32],
        scratch: &mut [C32],
    ) -> Result<FftScale> {
        <Fallback as RealFft>::irfft(&Fallback, plan, input, output, scratch)
    }
}

// The fused kernels are the chaining functions of `sdrk_core::fused` called with the facade
// itself, so they use the best single kernel for every step. A backend that provides a truly
// fused version must be bit-identical to this chain (see `docs/kernels.md`). The rows above
// register them in the `backend_for` table as `chained`.
macro_rules! fused_for {
    ($t:ty) => {
        impl Ddc<$t> for crate::Sdrk {
            #[inline]
            fn ddc<Taps, Hist>(
                &self,
                nco: &mut Nco,
                fir: &mut FirDecimState<$t, Taps, Hist>,
                input: &[$t],
                output: &mut [$t],
                scratch: &mut [$t],
            ) -> Result<Progress>
            where
                Taps: AsRef<[<$t as crate::ComplexSample>::Real]>,
                Hist: AsRef<[$t]> + AsMut<[$t]>,
            {
                ddc_chained(self, nco, fir, input, output, scratch)
            }
        }

        impl Duc<$t> for crate::Sdrk {
            #[inline]
            fn duc<Taps, Hist>(
                &self,
                fir: &mut FirInterpState<$t, Taps, Hist>,
                nco: &mut Nco,
                input: &[$t],
                output: &mut [$t],
                scratch: &mut [$t],
            ) -> Result<Progress>
            where
                Taps: AsRef<[<$t as crate::ComplexSample>::Real]>,
                Hist: AsRef<[$t]> + AsMut<[$t]>,
            {
                duc_chained(self, fir, nco, input, output, scratch)
            }
        }

        impl FmDemod<$t> for crate::Sdrk {
            #[inline]
            fn fm_demod(
                &self,
                quad: &mut QuadDemodState<$t>,
                deemph: &mut DeemphasisState<<$t as crate::ComplexSample>::Real>,
                input: &[$t],
                output_audio: &mut [<$t as crate::ComplexSample>::Real],
                scratch_real: &mut [<$t as crate::ComplexSample>::Real],
                gain: <$t as crate::ComplexSample>::Real,
            ) -> Result<()> {
                fm_demod_chained(self, quad, deemph, input, output_audio, scratch_real, gain)
            }
        }
    };
}

fused_for!(C32);
fused_for!(CQ15);
