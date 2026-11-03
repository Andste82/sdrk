//! Reference implementation of the `rotate` kernel (f64 with `libm`).

use crate::Reference;
use crate::error::{Error, Result};
use crate::kernels::mix::{Nco, Oscillate, Rotate, check_harmonic};
use crate::num::{round_shift, sat16};
use crate::tables::COS_Q15;
use crate::types::{C32, CQ15, ComplexSample, Q15};
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

/// The scaled phase `k · (phase0 + inc · n)` modulo 2^32.
fn scaled_phase(nco: Nco, harmonic: u32, n: usize) -> u32 {
    nco.phase()
        .wrapping_add(nco.increment().wrapping_mul(n as u32))
        .wrapping_mul(harmonic)
}

impl Oscillate<C32> for Reference {
    fn oscillate(&self, nco: &mut Nco, harmonic: u32, out: &mut [C32]) -> Result<()> {
        check_harmonic(harmonic)?;
        for (n, o) in out.iter_mut().enumerate() {
            let theta = 2.0 * core::f64::consts::PI * f64::from(scaled_phase(*nco, harmonic, n))
                / 4_294_967_296.0;
            *o = C32::from_c64_round(Complex64::new(libm::cos(theta), libm::sin(theta)));
        }
        nco.advance(out.len());
        Ok(())
    }
}

/// The cosine of the phase from the `Q15` table: `sat16(a + round_shift((b - a) f15, 15))`.
fn table_cos_q15(phase: u32) -> i16 {
    let idx = (phase >> 20) as usize;
    let f15 = i64::from((phase >> 5) & 0x7FFF);
    let (a, b) = match COS_Q15.get(idx..idx + 2) {
        Some(&[a, b]) => (i64::from(a), i64::from(b)),
        _ => (0, 0),
    };
    sat16(a + round_shift((b - a) * f15, 15))
}

impl Oscillate<CQ15> for Reference {
    fn oscillate(&self, nco: &mut Nco, harmonic: u32, out: &mut [CQ15]) -> Result<()> {
        check_harmonic(harmonic)?;
        for (n, o) in out.iter_mut().enumerate() {
            let phase = scaled_phase(*nco, harmonic, n);
            *o = CQ15 {
                re: Q15(table_cos_q15(phase)),
                im: Q15(table_cos_q15(phase.wrapping_sub(1 << 30))),
            };
        }
        nco.advance(out.len());
        Ok(())
    }
}
