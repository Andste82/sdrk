//! The single place that maps kernels to implementations.

use crate::{C32, CQ15, Fallback, Q15, Result};
use sdrk_core::kernels::basic::Scale;
use sdrk_core::kernels::convert::{Convert, ConvertReal};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::mix::{Nco, Rotate};
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
}
