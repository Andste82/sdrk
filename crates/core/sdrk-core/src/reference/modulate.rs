//! Reference implementation of the `fm_mod` kernel: exact `cos`/`sin` in f64.

use core::f64::consts::PI;

use num_complex::Complex64;

use crate::Reference;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::modulate::{FmMod, FmModState};
use crate::types::{C32, CQ15, ComplexSample, Q15, RealSample};

/// Runs the modulator in f64 and rounds each output once; `amplitude` is the length of the
/// unit vector (`1 - 2^-15` for Q15 outputs).
fn modulate<T: ComplexSample>(
    s: &mut FmModState,
    input: &[T::Real],
    output: &mut [T],
    amplitude: f64,
) -> Result<()> {
    check_len(input.len(), output.len())?;
    let (k, phase) = s.parts_mut();
    let mut acc = i128::from(*phase);
    for (o, &x) in output.iter_mut().zip(input) {
        // The same integer phase model as the specification: floor(x k + 1/2) as i64.
        let step = libm::floor(x.to_f64() * k + 0.5) as i64;
        acc = (acc + i128::from(step)).rem_euclid(1 << 32);
        let theta = 2.0 * PI * (acc as f64) / 4_294_967_296.0;
        *o = T::from_c64_round(Complex64::new(
            amplitude * libm::cos(theta),
            amplitude * libm::sin(theta),
        ));
    }
    *phase = acc as u32;
    Ok(())
}

impl FmMod<C32> for Reference {
    fn fm_mod(&self, s: &mut FmModState, input: &[f32], output: &mut [C32]) -> Result<()> {
        modulate::<C32>(s, input, output, 1.0)
    }
}

impl FmMod<CQ15> for Reference {
    fn fm_mod(&self, s: &mut FmModState, input: &[Q15], output: &mut [CQ15]) -> Result<()> {
        modulate::<CQ15>(s, input, output, 32767.0 / 32768.0)
    }
}
