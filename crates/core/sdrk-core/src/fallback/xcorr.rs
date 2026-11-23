//! Fallback implementation of `xcorr_fft`: [`xcorr_fft_with`] with the overlap-save block of
//! [`overlap_save_block`] over the fallback FFT (the forward FFT, the spectral product of the
//! fixed formula without FMA, the inverse FFT).

use crate::Fallback;
use crate::error::Result;
use crate::kernels::xcorr::{XcorrFft, XcorrFftState, overlap_save_block, xcorr_fft_with};
use crate::timing::Progress;
use crate::types::C32;

impl XcorrFft<C32> for Fallback {
    fn xcorr_fft<S, Sh, Sc, Tw>(
        &self,
        state: &mut XcorrFftState<C32, S, Sh, Sc, Tw>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<Progress>
    where
        S: AsRef<[C32]>,
        Sh: AsRef<[C32]> + AsMut<[C32]>,
        Sc: AsRef<[C32]> + AsMut<[C32]>,
        Tw: AsRef<[C32]>,
    {
        xcorr_fft_with(state, input, output, |args| {
            overlap_save_block(&Fallback, args)
        })
    }
}
