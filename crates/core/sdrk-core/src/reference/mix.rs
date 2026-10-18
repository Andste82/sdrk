//! Reference implementation of the `rotate` kernel (f64 with `libm`).

use crate::Reference;
use crate::error::{Error, Result};
use crate::kernels::mix::{Nco, Rotate};
use crate::types::{C32, CQ15, ComplexSample};
use num_complex::Complex64;

fn rotate_impl<T: ComplexSample>(nco: &mut Nco, input: &[T], output: &mut [T]) -> Result<()> {
    if input.len() != output.len() {
        return Err(Error::LengthMismatch {
            expected: input.len(),
            got: output.len(),
        });
    }
    let (phase0, inc) = (nco.phase(), nco.increment());
    for (n, (o, x)) in output.iter_mut().zip(input).enumerate() {
        let phase = phase0.wrapping_add(inc.wrapping_mul(n as u32));
        let theta = 2.0 * core::f64::consts::PI * f64::from(phase) / 4_294_967_296.0;
        let r = Complex64::new(libm::cos(theta), libm::sin(theta));
        *o = T::from_c64_round(x.to_c64() * r);
    }
    nco.advance(input.len());
    Ok(())
}

impl Rotate<C32> for Reference {
    fn rotate(&self, nco: &mut Nco, input: &[C32], output: &mut [C32]) -> Result<()> {
        rotate_impl(nco, input, output)
    }
}

impl Rotate<CQ15> for Reference {
    fn rotate(&self, nco: &mut Nco, input: &[CQ15], output: &mut [CQ15]) -> Result<()> {
        rotate_impl(nco, input, output)
    }
}
