//! Portable, optimised-scalar implementations of all kernels.

use crate::error::Result;
use crate::kernels::basic::Scale;
use crate::map_kernel;
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
        // `q15_mul` in `i32`: the product is at most 2^30 in magnitude, so the rounding cannot
        // overflow, and the result is at least -32768, so only 32768 (-1 * -1) needs the
        // saturation. The same values as `q15_mul`, in a form the compiler vectorizes.
        let g = i32::from(gain.0);
        let mul = |x: Q15| Q15(((i32::from(x.0) * g + (1 << 14)) >> 15).min(32767) as i16);
        map_kernel!(input, output, |x: CQ15| CQ15 {
            re: mul(x.re),
            im: mul(x.im),
        })
    }
}

mod agc;
mod biquad;
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
