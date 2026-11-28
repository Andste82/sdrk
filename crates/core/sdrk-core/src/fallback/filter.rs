//! Fallback implementation of the `fir` kernel: the engine of `fir_decimate` (in `decim.rs`)
//! with one output per input sample.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::filter::{Fir, FirState};
use crate::types::{C32, CQ15, Q15};

impl Fir<C32> for Fallback {
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
        let (taps, hist) = state.parts_mut();
        super::decim::fir(taps, hist, input, output)
    }
}

impl Fir<CQ15> for Fallback {
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
        let (taps, hist) = state.parts_mut();
        super::decim::fir(taps, hist, input, output)
    }
}
