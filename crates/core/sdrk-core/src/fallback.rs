//! Portable, optimised-scalar implementations of all kernels.

use crate::error::Result;
use crate::kernels::basic::Scale;
use crate::map_kernel;
use crate::num::q15_mul;
use crate::types::{C32, CQ15, Q15};

/// The portable implementation of every kernel.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fallback;

impl Scale<C32> for Fallback {
    fn scale(&self, input: &[C32], output: &mut [C32], gain: f32) -> Result<()> {
        map_kernel!(input, output, |x: C32| C32::new(x.re * gain, x.im * gain))
    }
}

impl Scale<CQ15> for Fallback {
    fn scale(&self, input: &[CQ15], output: &mut [CQ15], gain: Q15) -> Result<()> {
        map_kernel!(input, output, |x: CQ15| CQ15 {
            re: q15_mul(x.re, gain),
            im: q15_mul(x.im, gain),
        })
    }
}

mod agc;
mod convert;
mod convert_out;
mod decim;
mod demod;
mod dot;
mod fft;
mod filter;
mod fused;
mod interp;
mod mix;
mod modulate;
mod spectrum;
mod squelch;
