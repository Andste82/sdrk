//! Reference of `qam_soft_bits` (M44 §2): the definition in `f64` with the exact `√E`, each
//! soft bit rounded once to `f32` (the achievable optimum of an `f32` output, D1413).

use crate::Reference;
use crate::error::Result;
use crate::kernels::constellation::{Qam, QamSoftBits, check_soft_bits_len};
use crate::types::C32;

impl QamSoftBits<C32> for Reference {
    fn qam_soft_bits(&self, qam: Qam, input: &[C32], output: &mut [f32], scale: f32) -> Result<()> {
        check_soft_bits_len(qam, input.len(), output.len())?;
        let sqrt_e = libm::sqrt(qam.energy());
        let s = f64::from(scale);
        let per_axis = qam.bits_per_axis();
        let components = input.iter().flat_map(|x| [x.re, x.im]);
        for (o, x) in output.chunks_exact_mut(per_axis).zip(components) {
            let v = f64::from(x) * sqrt_e;
            let a = libm::fabs(v);
            let bits = match qam {
                Qam::Qam16 => [v, 2.0 - a, 0.0],
                Qam::Qam64 => [v, 4.0 - a, 2.0 - libm::fabs(a - 4.0)],
            };
            for (o, b) in o.iter_mut().zip(bits) {
                *o = (s * b) as f32;
            }
        }
        Ok(())
    }
}
