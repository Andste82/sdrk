//! Reference of `qam_soft_bits` (M44 §2, M44.2 §2): the definition in `f64` with the exact `√E`,
//! the recursion `t_0 = v`, `t_i = 2^(b - i) - |t_(i - 1)|` per axis, each soft bit rounded once
//! to `f32` (the achievable optimum of an `f32` output, D1413).

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
            let mut t = f64::from(x) * sqrt_e;
            for (i, o) in o.iter_mut().enumerate() {
                if i > 0 {
                    t = f64::from(1u32 << (per_axis - i)) - libm::fabs(t);
                }
                *o = (s * t) as f32;
            }
        }
        Ok(())
    }
}
