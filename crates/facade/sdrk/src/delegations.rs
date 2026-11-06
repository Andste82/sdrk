//! The single place that maps kernels to implementations.

use crate::{C32, CQ15, Chained, ComplexSample, Fallback, Progress, Q15, Result};
use sdrk_core::fused::{ddc_chained, duc_chained, fm_demod_chained};
use sdrk_core::kernels::agc::{Agc, AgcState};
use sdrk_core::kernels::auto_notch::{AutoNotch, AutoNotchState};
use sdrk_core::kernels::basic::Scale;
use sdrk_core::kernels::biquad::{Biquad, BiquadReal, BiquadRealState, BiquadState};
use sdrk_core::kernels::convert::{Convert, ConvertOut, ConvertOutReal, ConvertReal};
use sdrk_core::kernels::costas::{Costas, CostasState};
use sdrk_core::kernels::decim::{
    FirDecimState, FirDecimate, Resample, ResampleReal, ResampleRealState, ResampleState,
};
use sdrk_core::kernels::demod::{
    DcBlock, DcBlockReal, DcBlockRealState, DcBlockState, Deemphasis, DeemphasisState, QuadDemod,
    QuadDemodState,
};
use sdrk_core::kernels::dot::{Dot, DotComplex, DotReal};
use sdrk_core::kernels::equalize::{Equalize, EqualizeState};
use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, FftScale, RealFft, RealFftPlan, Scaling};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::fused::{Ddc, Duc, FmDemod};
use sdrk_core::kernels::halfband::{
    HalfbandDecimate, HalfbandDecimateReal, HalfbandRealState, HalfbandState, RealToComplex,
    RealToComplexState,
};
use sdrk_core::kernels::interp::{
    FirInterpRealState, FirInterpState, FirInterpolate, FirInterpolateReal,
};
use sdrk_core::kernels::math::{Add, AddReal, Interleave, Mul, MulReal};
use sdrk_core::kernels::mix::{Nco, Oscillate, Rotate};
use sdrk_core::kernels::modulate::{AmMod, FmMod, FmModState};
use sdrk_core::kernels::noise_blank::{NoiseBlank, NoiseBlankState};
use sdrk_core::kernels::preemph_limit::{
    PreemphLimit, PreemphLimitState, PreemphLimitStereo, PreemphLimitStereoState,
};
use sdrk_core::kernels::spectrum::{Mag, MagSq, PowerDb, Window};
use sdrk_core::kernels::squelch::{Squelch, SquelchState};
use sdrk_core::kernels::stereo::{
    StereoEncode, StereoEncodeState, StereoMatrix, StereoPll, StereoPllState,
};
use sdrk_core::kernels::symbol_sync::{SymbolSync, SymbolSyncState};
use sdrk_device::SampleFormat;

// The implementation of the accelerated kernels: the SIMD backend, or the fallback when the
// feature `simd` is off, the build sets `--cfg sdrk_fallback_only` (RUSTFLAGS), or the target has
// no SIMD instructions to offer (a Cortex-M, or wasm32 without `simd128`).
#[cfg(not(all(
    feature = "simd",
    not(sdrk_fallback_only),
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )
)))]
use crate::Fallback as Fast;
#[cfg(all(
    feature = "simd",
    not(sdrk_fallback_only),
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )
))]
use sdrk_backend_simd::Simd as Fast;

crate::delegate! {
    Scale<C32>::scale(&self, input: &[C32], output: &mut [C32], gain: f32) -> Result<()> => Fast;
    Scale<CQ15>::scale(&self, input: &[CQ15], output: &mut [CQ15], gain: Q15) -> Result<()> => Fast;
    Convert<C32>::convert(&self, format: &SampleFormat, input: &[u8], output: &mut [C32]) -> Result<usize> => Fast;
    Convert<CQ15>::convert(&self, format: &SampleFormat, input: &[u8], output: &mut [CQ15]) -> Result<usize> => Fast;
    ConvertReal<f32>::convert_real(&self, format: &SampleFormat, input: &[u8], output: &mut [f32]) -> Result<usize> => Fast;
    ConvertReal<Q15>::convert_real(&self, format: &SampleFormat, input: &[u8], output: &mut [Q15]) -> Result<usize> => Fast;
    Rotate<C32>::rotate(&self, nco: &mut Nco, input: &[C32], output: &mut [C32]) -> Result<()> => Fast;
    Rotate<CQ15>::rotate(&self, nco: &mut Nco, input: &[CQ15], output: &mut [CQ15]) -> Result<()> => Fast;
    Fir<C32>::fir[Taps: AsRef<[f32]>, Hist: AsRef<[C32]> + AsMut<[C32]>](&self, state: &mut FirState<C32, Taps, Hist>, input: &[C32], output: &mut [C32]) -> Result<()> => Fast;
    Fir<CQ15>::fir[Taps: AsRef<[Q15]>, Hist: AsRef<[CQ15]> + AsMut<[CQ15]>](&self, state: &mut FirState<CQ15, Taps, Hist>, input: &[CQ15], output: &mut [CQ15]) -> Result<()> => Fast;
    FirDecimate<C32>::fir_decimate[Taps: AsRef<[f32]>, Hist: AsRef<[C32]> + AsMut<[C32]>](&self, state: &mut FirDecimState<C32, Taps, Hist>, input: &[C32], output: &mut [C32]) -> Result<Progress> => Fast;
    FirDecimate<CQ15>::fir_decimate[Taps: AsRef<[Q15]>, Hist: AsRef<[CQ15]> + AsMut<[CQ15]>](&self, state: &mut FirDecimState<CQ15, Taps, Hist>, input: &[CQ15], output: &mut [CQ15]) -> Result<Progress> => Fast;
    HalfbandDecimate<C32>::halfband_decimate[Taps: AsRef<[f32]>, Hist: AsRef<[C32]> + AsMut<[C32]>](&self, state: &mut HalfbandState<C32, Taps, Hist>, input: &[C32], output: &mut [C32]) -> Result<Progress> => Fast;
    HalfbandDecimate<CQ15>::halfband_decimate[Taps: AsRef<[Q15]>, Hist: AsRef<[CQ15]> + AsMut<[CQ15]>](&self, state: &mut HalfbandState<CQ15, Taps, Hist>, input: &[CQ15], output: &mut [CQ15]) -> Result<Progress> => Fast;
    HalfbandDecimateReal<f32>::halfband_decimate_real[Taps: AsRef<[f32]>, Hist: AsRef<[f32]> + AsMut<[f32]>](&self, state: &mut HalfbandRealState<f32, Taps, Hist>, input: &[f32], output: &mut [f32]) -> Result<Progress> => Fast;
    HalfbandDecimateReal<Q15>::halfband_decimate_real[Taps: AsRef<[Q15]>, Hist: AsRef<[Q15]> + AsMut<[Q15]>](&self, state: &mut HalfbandRealState<Q15, Taps, Hist>, input: &[Q15], output: &mut [Q15]) -> Result<Progress> => Fast;
    RealToComplex<C32>::real_to_complex[Taps: AsRef<[f32]>, Hist: AsRef<[C32]> + AsMut<[C32]>](&self, state: &mut RealToComplexState<C32, Taps, Hist>, input: &[f32], output: &mut [C32]) -> Result<Progress> => Fast;
    RealToComplex<CQ15>::real_to_complex[Taps: AsRef<[Q15]>, Hist: AsRef<[CQ15]> + AsMut<[CQ15]>](&self, state: &mut RealToComplexState<CQ15, Taps, Hist>, input: &[Q15], output: &mut [CQ15]) -> Result<Progress> => Fast;
    Resample<C32>::resample[Taps: AsRef<[f32]>, Hist: AsRef<[C32]> + AsMut<[C32]>](&self, state: &mut ResampleState<C32, Taps, Hist>, input: &[C32], output: &mut [C32]) -> Result<Progress> => Fast;
    Resample<CQ15>::resample[Taps: AsRef<[Q15]>, Hist: AsRef<[CQ15]> + AsMut<[CQ15]>](&self, state: &mut ResampleState<CQ15, Taps, Hist>, input: &[CQ15], output: &mut [CQ15]) -> Result<Progress> => Fast;
    ResampleReal<f32>::resample_real[Taps: AsRef<[f32]>, Hist: AsRef<[f32]> + AsMut<[f32]>](&self, state: &mut ResampleRealState<f32, Taps, Hist>, input: &[f32], output: &mut [f32]) -> Result<Progress> => Fast;
    ResampleReal<Q15>::resample_real[Taps: AsRef<[Q15]>, Hist: AsRef<[Q15]> + AsMut<[Q15]>](&self, state: &mut ResampleRealState<Q15, Taps, Hist>, input: &[Q15], output: &mut [Q15]) -> Result<Progress> => Fast;
    QuadDemod<C32>::quad_demod(&self, state: &mut QuadDemodState<C32>, input: &[C32], output: &mut [f32], gain: f32) -> Result<()> => Fast;
    QuadDemod<CQ15>::quad_demod(&self, state: &mut QuadDemodState<CQ15>, input: &[CQ15], output: &mut [Q15], gain: Q15) -> Result<()> => Fast;
    Deemphasis<f32>::deemphasis(&self, state: &mut DeemphasisState<f32>, input: &[f32], output: &mut [f32]) -> Result<()> => Fast;
    Deemphasis<Q15>::deemphasis(&self, state: &mut DeemphasisState<Q15>, input: &[Q15], output: &mut [Q15]) -> Result<()> => Fallback;
    DcBlock<C32>::dc_block(&self, state: &mut DcBlockState<C32>, input: &[C32], output: &mut [C32]) -> Result<()> => Fast;
    DcBlock<CQ15>::dc_block(&self, state: &mut DcBlockState<CQ15>, input: &[CQ15], output: &mut [CQ15]) -> Result<()> => Fallback;
    DcBlockReal<f32>::dc_block_real(&self, state: &mut DcBlockRealState<f32>, input: &[f32], output: &mut [f32]) -> Result<()> => Fast;
    DcBlockReal<Q15>::dc_block_real(&self, state: &mut DcBlockRealState<Q15>, input: &[Q15], output: &mut [Q15]) -> Result<()> => Fallback;
    Agc<C32>::agc(&self, s: &mut AgcState<C32>, input: &[C32], output: &mut [C32]) -> Result<()> => Fast;
    Agc<CQ15>::agc(&self, s: &mut AgcState<CQ15>, input: &[CQ15], output: &mut [CQ15]) -> Result<()> => Fallback;
    Biquad<C32>::biquad[C: AsRef<[[f32; 5]]>, H: AsRef<[C32]> + AsMut<[C32]>](&self, s: &mut BiquadState<C32, C, H>, input: &[C32], output: &mut [C32]) -> Result<()> => Fast;
    Biquad<CQ15>::biquad[C: AsRef<[[i16; 5]]>, H: AsRef<[CQ15]> + AsMut<[CQ15]>](&self, s: &mut BiquadState<CQ15, C, H>, input: &[CQ15], output: &mut [CQ15]) -> Result<()> => Fast;
    BiquadReal<f32>::biquad_real[C: AsRef<[[f32; 5]]>, H: AsRef<[f32]> + AsMut<[f32]>](&self, s: &mut BiquadRealState<f32, C, H>, input: &[f32], output: &mut [f32]) -> Result<()> => Fast;
    BiquadReal<Q15>::biquad_real[C: AsRef<[[i16; 5]]>, H: AsRef<[Q15]> + AsMut<[Q15]>](&self, s: &mut BiquadRealState<Q15, C, H>, input: &[Q15], output: &mut [Q15]) -> Result<()> => Fast;
    Squelch<C32>::squelch(&self, state: &mut SquelchState<C32>, input: &[C32], output: &mut [C32]) -> Result<()> => Fallback;
    Squelch<CQ15>::squelch(&self, state: &mut SquelchState<CQ15>, input: &[CQ15], output: &mut [CQ15]) -> Result<()> => Fallback;
    NoiseBlank<C32>::noise_blank[H: AsRef<[C32]> + AsMut<[C32]>](&self, s: &mut NoiseBlankState<C32, H>, input: &[C32], output: &mut [C32]) -> Result<()> => Fallback;
    NoiseBlank<CQ15>::noise_blank[H: AsRef<[CQ15]> + AsMut<[CQ15]>](&self, s: &mut NoiseBlankState<CQ15, H>, input: &[CQ15], output: &mut [CQ15]) -> Result<()> => Fallback;
    PreemphLimit<f32>::preemph_limit(&self, state: &mut PreemphLimitState<f32>, input: &[f32], output: &mut [f32]) -> Result<()> => Fallback;
    PreemphLimit<Q15>::preemph_limit(&self, state: &mut PreemphLimitState<Q15>, input: &[Q15], output: &mut [Q15]) -> Result<()> => Fallback;
    StereoPll<f32>::stereo_pll(&self, state: &mut StereoPllState<f32>, mpx: &[f32], pilot: &[f32], output: &mut [C32]) -> Result<()> => Fallback;
    StereoPll<Q15>::stereo_pll(&self, state: &mut StereoPllState<Q15>, mpx: &[Q15], pilot: &[Q15], output: &mut [CQ15]) -> Result<()> => Fallback;
    SymbolSync<C32>::symbol_sync(&self, s: &mut SymbolSyncState<C32>, input: &[C32], output: &mut [C32]) -> Result<Progress> => Fallback;
    SymbolSync<f32>::symbol_sync(&self, s: &mut SymbolSyncState<f32>, input: &[f32], output: &mut [f32]) -> Result<Progress> => Fallback;
    Add<C32>::add(&self, a: &[C32], b: &[C32], out: &mut [C32]) -> Result<()> => Fast;
    Add<CQ15>::add(&self, a: &[CQ15], b: &[CQ15], out: &mut [CQ15]) -> Result<()> => Fast;
    AddReal<f32>::add_real(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> => Fast;
    AddReal<Q15>::add_real(&self, a: &[Q15], b: &[Q15], out: &mut [Q15]) -> Result<()> => Fast;
    Mul<C32>::mul(&self, a: &[C32], b: &[C32], out: &mut [C32]) -> Result<()> => Fast;
    Mul<CQ15>::mul(&self, a: &[CQ15], b: &[CQ15], out: &mut [CQ15]) -> Result<()> => Fast;
    MulReal<f32>::mul_real(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> => Fast;
    MulReal<Q15>::mul_real(&self, a: &[Q15], b: &[Q15], out: &mut [Q15]) -> Result<()> => Fast;
    Oscillate<C32>::oscillate(&self, nco: &mut Nco, harmonic: u32, out: &mut [C32]) -> Result<()> => Fast;
    Oscillate<CQ15>::oscillate(&self, nco: &mut Nco, harmonic: u32, out: &mut [CQ15]) -> Result<()> => Fast;
    StereoMatrix<C32>::stereo_matrix(&self, input: &[C32], output: &mut [C32]) -> Result<()> => Fast;
    StereoMatrix<CQ15>::stereo_matrix(&self, input: &[CQ15], output: &mut [CQ15]) -> Result<()> => Fast;
    Window<C32>::window(&self, input: &[C32], output: &mut [C32], w: &[f32]) -> Result<()> => Fast;
    Window<CQ15>::window(&self, input: &[CQ15], output: &mut [CQ15], w: &[Q15]) -> Result<()> => Fast;
    MagSq<C32>::mag_sq(&self, input: &[C32], output: &mut [f32]) -> Result<()> => Fast;
    MagSq<CQ15>::mag_sq(&self, input: &[CQ15], output: &mut [Q15]) -> Result<()> => Fast;
    Mag<C32>::mag(&self, input: &[C32], output: &mut [f32]) -> Result<()> => Fast;
    Mag<CQ15>::mag(&self, input: &[CQ15], output: &mut [Q15]) -> Result<()> => Fast;
    PowerDb<f32>::power_db(&self, input: &[f32], output: &mut [f32], offset_db: f32) -> Result<()> => Fast;
    PowerDb<Q15>::power_db(&self, input: &[Q15], output: &mut [f32], offset_db: f32) -> Result<()> => Fast;
    Fft<C32>::fft[Tw: AsRef<[C32]>](&self, plan: &FftPlan<C32, Tw>, data: &mut [C32], dir: Direction, scaling: Scaling) -> Result<FftScale> => Fast;
    Fft<CQ15>::fft[Tw: AsRef<[CQ15]>](&self, plan: &FftPlan<CQ15, Tw>, data: &mut [CQ15], dir: Direction, scaling: Scaling) -> Result<FftScale> => Fast;
    ConvertOut<C32>::convert_out(&self, format: &SampleFormat, input: &[C32], output: &mut [u8]) -> Result<usize> => Fast;
    ConvertOut<CQ15>::convert_out(&self, format: &SampleFormat, input: &[CQ15], output: &mut [u8]) -> Result<usize> => Fast;
    ConvertOutReal<f32>::convert_out_real(&self, format: &SampleFormat, input: &[f32], output: &mut [u8]) -> Result<usize> => Fast;
    ConvertOutReal<Q15>::convert_out_real(&self, format: &SampleFormat, input: &[Q15], output: &mut [u8]) -> Result<usize> => Fast;
    FirInterpolate<C32>::fir_interpolate[Taps: AsRef<[f32]>, Hist: AsRef<[C32]> + AsMut<[C32]>](&self, state: &mut FirInterpState<C32, Taps, Hist>, input: &[C32], output: &mut [C32]) -> Result<Progress> => Fast;
    FirInterpolate<CQ15>::fir_interpolate[Taps: AsRef<[Q15]>, Hist: AsRef<[CQ15]> + AsMut<[CQ15]>](&self, state: &mut FirInterpState<CQ15, Taps, Hist>, input: &[CQ15], output: &mut [CQ15]) -> Result<Progress> => Fast;
    FirInterpolateReal<f32>::fir_interpolate_real[Taps: AsRef<[f32]>, Hist: AsRef<[f32]> + AsMut<[f32]>](&self, state: &mut FirInterpRealState<f32, Taps, Hist>, input: &[f32], output: &mut [f32]) -> Result<Progress> => Fast;
    FirInterpolateReal<Q15>::fir_interpolate_real[Taps: AsRef<[Q15]>, Hist: AsRef<[Q15]> + AsMut<[Q15]>](&self, state: &mut FirInterpRealState<Q15, Taps, Hist>, input: &[Q15], output: &mut [Q15]) -> Result<Progress> => Fast;
    FmMod<C32>::fm_mod(&self, s: &mut FmModState, input: &[f32], output: &mut [C32]) -> Result<()> => Fast;
    FmMod<CQ15>::fm_mod(&self, s: &mut FmModState, input: &[Q15], output: &mut [CQ15]) -> Result<()> => Fast;
    AmMod<C32>::am_mod(&self, carrier: f32, depth: f32, input: &[f32], output: &mut [C32]) -> Result<()> => Fast;
    AmMod<CQ15>::am_mod(&self, carrier: f32, depth: f32, input: &[Q15], output: &mut [CQ15]) -> Result<()> => Fast;
    Dot<C32>::dot(&self, x: &[C32], taps: &[f32]) -> Result<C32> => Fast;
    Dot<CQ15>::dot(&self, x: &[CQ15], taps: &[Q15]) -> Result<CQ15> => Fast;
    DotReal<f32>::dot_real(&self, x: &[f32], taps: &[f32]) -> Result<f32> => Fast;
    DotReal<Q15>::dot_real(&self, x: &[Q15], taps: &[Q15]) -> Result<Q15> => Fast;
    DotComplex<C32>::dot_complex(&self, x: &[C32], w: &[C32]) -> Result<C32> => Fast;
    @extra [
        ("interleave", C32, Fast),
        ("deinterleave", C32, Fast),
        ("interleave", CQ15, Fast),
        ("deinterleave", CQ15, Fast),
        ("rfft", f32, Fast),
        ("irfft", f32, Fast),
        ("auto_notch", f32, Fallback),
        ("preemph_limit_stereo", C32, Fallback),
        ("stereo_encode", C32, Fallback),
        ("costas", C32, Fallback),
        ("equalize", C32, Fast),
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
        <Fast as RealFft>::rfft(&Fast, plan, input, output, scratch)
    }

    #[inline]
    fn irfft<Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>>(
        &self,
        plan: &RealFftPlan<Tw, Tw2>,
        input: &[C32],
        output: &mut [f32],
        scratch: &mut [C32],
    ) -> Result<FftScale> {
        <Fast as RealFft>::irfft(&Fast, plan, input, output, scratch)
    }
}

// `AutoNotch` has no sample-type parameter (it exists for `f32` only), so it is delegated by
// hand like `RealFft`; the row above registers it in the `backend_for` table. It is a
// per-sample recursion and stays with the fallback on every backend.
impl AutoNotch for crate::Sdrk {
    #[inline]
    fn auto_notch<W, X>(
        &self,
        s: &mut AutoNotchState<W, X>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()>
    where
        W: AsRef<[f32]> + AsMut<[f32]>,
        X: AsRef<[f32]> + AsMut<[f32]>,
    {
        <Fallback as AutoNotch>::auto_notch(&Fallback, s, input, output)
    }
}

// `PreemphLimitStereo` and `StereoEncode` have no sample-type parameter (they exist for `C32`
// only), so they are delegated by hand; the rows above register them in the `backend_for`
// table. The limiter is a per-sample recursion and the encoder a table lookup per sample with
// little arithmetic around it: both stay with the fallback on every backend.
impl PreemphLimitStereo for crate::Sdrk {
    #[inline]
    fn preemph_limit_stereo(
        &self,
        state: &mut PreemphLimitStereoState,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()> {
        <Fallback as PreemphLimitStereo>::preemph_limit_stereo(&Fallback, state, input, output)
    }
}

impl StereoEncode for crate::Sdrk {
    #[inline]
    fn stereo_encode(
        &self,
        state: &mut StereoEncodeState,
        input: &[C32],
        output: &mut [f32],
    ) -> Result<()> {
        <Fallback as StereoEncode>::stereo_encode(&Fallback, state, input, output)
    }
}

// `Costas` has no sample-type parameter (the signature of the spec, `C32` only), so it is
// delegated by hand like `StereoEncode`; the row above registers it in the `backend_for` table.
// The loop is a recursion over the symbols and stays with the fallback on every backend.
impl Costas for crate::Sdrk {
    #[inline]
    fn costas(&self, s: &mut CostasState, input: &[C32], output: &mut [C32]) -> Result<()> {
        <Fallback as Costas>::costas(&Fallback, s, input, output)
    }
}

// `Equalize` has no sample-type parameter (the signature of the spec, `C32` only), so it is
// delegated by hand like `Costas`; the row above registers it in the `backend_for` table. The
// loop is a recursion over the symbols; the SIMD backend accelerates its filter part with its
// bit-exact `dot_complex` and keeps the scalar weight update, so every backend gives the bits of
// the fallback.
impl Equalize for crate::Sdrk {
    #[inline]
    fn equalize<W, H>(
        &self,
        s: &mut EqualizeState<W, H>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()>
    where
        W: AsRef<[C32]> + AsMut<[C32]>,
        H: AsRef<[C32]> + AsMut<[C32]>,
    {
        <Fast as Equalize>::equalize(&Fast, s, input, output)
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

// `Interleave` has two methods and the macro delegates one method per entry, so the trait is
// delegated by hand; the rows in `@extra` above register both methods in the `backend_for`
// table.
macro_rules! delegate_interleave {
    ($($ty:ty),+) => {$(
        impl Interleave<$ty> for crate::Sdrk {
            #[inline]
            fn interleave(
                &self,
                re: &[<$ty as ComplexSample>::Real],
                im: Option<&[<$ty as ComplexSample>::Real]>,
                out: &mut [$ty],
            ) -> Result<()> {
                <Fast as Interleave<$ty>>::interleave(&Fast, re, im, out)
            }

            #[inline]
            fn deinterleave(
                &self,
                input: &[$ty],
                re: Option<&mut [<$ty as ComplexSample>::Real]>,
                im: Option<&mut [<$ty as ComplexSample>::Real]>,
            ) -> Result<()> {
                <Fast as Interleave<$ty>>::deinterleave(&Fast, input, re, im)
            }
        }
    )+};
}

delegate_interleave!(C32, CQ15);
