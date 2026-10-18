//! The single place that maps kernels to implementations.

use crate::{C32, CQ15, Fallback, Progress, Q15, Result};
use sdrk_core::kernels::basic::Scale;
use sdrk_core::kernels::convert::{Convert, ConvertReal};
use sdrk_core::kernels::decim::{
    FirDecimState, FirDecimate, Resample, ResampleReal, ResampleRealState, ResampleState,
};
use sdrk_core::kernels::demod::{
    DcBlock, DcBlockState, Deemphasis, DeemphasisState, QuadDemod, QuadDemodState,
};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::mix::{Nco, Rotate};
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
}
