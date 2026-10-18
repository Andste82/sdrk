//! Executable specification of all kernels: f64 for f32 kernels, the exact
//! integer model for Q15 kernels. Clarity over speed.

use crate::error::{Error, Result};
use crate::kernels::basic::Scale;
use crate::num::{round_shift, sat16};
use crate::types::{C32, CQ15, Q15};

/// The reference implementation of every kernel (feature `reference`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Reference;

fn check_len<T>(input: &[T], output: &[T]) -> Result<()> {
    if input.len() == output.len() {
        Ok(())
    } else {
        Err(Error::LengthMismatch {
            expected: input.len(),
            got: output.len(),
        })
    }
}

impl Scale<C32> for Reference {
    fn scale(&self, input: &[C32], output: &mut [C32], gain: f32) -> Result<()> {
        check_len(input, output)?;
        let g = f64::from(gain);
        for (o, i) in output.iter_mut().zip(input) {
            *o = C32::new((f64::from(i.re) * g) as f32, (f64::from(i.im) * g) as f32);
        }
        Ok(())
    }
}

impl Scale<CQ15> for Reference {
    fn scale(&self, input: &[CQ15], output: &mut [CQ15], gain: Q15) -> Result<()> {
        check_len(input, output)?;
        let mul = |a: Q15, b: Q15| Q15(sat16(round_shift(i64::from(a.0) * i64::from(b.0), 15)));
        for (o, i) in output.iter_mut().zip(input) {
            *o = CQ15 {
                re: mul(i.re, gain),
                im: mul(i.im, gain),
            };
        }
        Ok(())
    }
}

mod convert;
mod filter;
mod mix;
