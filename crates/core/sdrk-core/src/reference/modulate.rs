//! Reference implementations of the `fm_mod` kernel (exact `cos`/`sin` in f64) and the `am_mod`
//! kernel (the formulas of the contract one sample at a time: `f32` in the written order, `i64`
//! for `Q15`).

use core::f64::consts::PI;

use num_complex::Complex64;

use crate::Reference;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::modulate::{AmMod, FmMod, FmModState, check_am_params};
use crate::num::{round_shift, sat16};
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

impl AmMod<C32> for Reference {
    fn am_mod(&self, carrier: f32, depth: f32, input: &[f32], output: &mut [C32]) -> Result<()> {
        check_am_params(carrier, depth)?;
        check_len(input.len(), output.len())?;
        for (o, &x) in output.iter_mut().zip(input) {
            let x = if x.is_nan() { 0.0 } else { x.clamp(-1.0, 1.0) };
            let t = depth * x;
            let u = 1.0 + t;
            *o = C32::new(carrier * u, 0.0);
        }
        Ok(())
    }
}

/// `round(v * 32768)` limited to 32767, computed in `f64` (`v` is `0 ..= 1`).
fn q15_gain(v: f32) -> i64 {
    (libm::floor(f64::from(v) * 32768.0 + 0.5) as i64).min(32767)
}

impl AmMod<CQ15> for Reference {
    fn am_mod(&self, carrier: f32, depth: f32, input: &[Q15], output: &mut [CQ15]) -> Result<()> {
        check_am_params(carrier, depth)?;
        check_len(input.len(), output.len())?;
        let (c_q, m_q) = (q15_gain(carrier), q15_gain(depth));
        for (o, &x) in output.iter_mut().zip(input) {
            let t = round_shift(m_q * i64::from(x.0), 15);
            let u = 32768 + t;
            *o = CQ15 {
                re: Q15(sat16(round_shift(c_q * u, 15))),
                im: Q15(0),
            };
        }
        Ok(())
    }
}
