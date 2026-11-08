//! Fallback implementation of `channelize`: [`channelize_with`] with the branch sums of
//! [`branch_sums`] (column blocks whose partial sums stay in registers over all rows, in f32
//! without FMA) and the transform of [`channel_fft`] with the fallback FFT.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::channelize::{
    Channelize, ChannelizeState, branch_sums, channel_fft, channelize_with,
};
use crate::timing::Progress;
use crate::types::C32;

impl Channelize<C32> for Fallback {
    fn channelize<Taps, Hist, Tw>(
        &self,
        s: &mut ChannelizeState<C32, Taps, Hist, Tw>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<Progress>
    where
        Taps: AsRef<[f32]>,
        Hist: AsRef<[C32]> + AsMut<[C32]>,
        Tw: AsRef<[C32]>,
    {
        channelize_with(s, input, output, false, |taps, window, swap, plan, group, _| {
            branch_sums(taps, window, swap, group)?;
            channel_fft(&Fallback, plan, group)
        })
    }
}
