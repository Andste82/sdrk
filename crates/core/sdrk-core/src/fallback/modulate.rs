//! Fallback implementation of the `fm_mod` kernel.

use crate::Fallback;
use crate::error::Result;
use crate::fallback::mix::{cos_sin_f32, cos_sin_q15};
use crate::kernels::check_len;
use crate::kernels::modulate::{FmMod, FmModState};
use crate::types::{C32, CQ15, Q15, RealSample};

/// The phase step `floor(x * k + 0.5)` as `i64`, wrapped into `u32`.
#[inline]
pub(crate) fn phase_step(x: f64, k: f64) -> u32 {
    (libm::floor(x * k + 0.5) as i64) as u32
}

impl FmMod<C32> for Fallback {
    fn fm_mod(&self, s: &mut FmModState, input: &[f32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (k, phase) = s.parts_mut();
        let mut p = *phase;
        for (o, &x) in output.iter_mut().zip(input) {
            p = p.wrapping_add(phase_step(x.to_f64(), k));
            let (c, sn) = cos_sin_f32(p)?;
            *o = C32::new(c, sn);
        }
        *phase = p;
        Ok(())
    }
}

impl FmMod<CQ15> for Fallback {
    fn fm_mod(&self, s: &mut FmModState, input: &[Q15], output: &mut [CQ15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (k, phase) = s.parts_mut();
        let mut p = *phase;
        for (o, &x) in output.iter_mut().zip(input) {
            p = p.wrapping_add(phase_step(x.to_f64(), k));
            *o = cos_sin_q15(p)?;
        }
        *phase = p;
        Ok(())
    }
}
