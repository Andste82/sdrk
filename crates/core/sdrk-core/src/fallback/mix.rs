//! Fallback implementation of the `rotate` kernel.

use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::mix::{Nco, Rotate};
use crate::num::{cq15_mul, round_shift, sat16};
use crate::tables::{COS_F32, COS_Q15};
use crate::types::{C32, CQ15, Q15};

const QUARTER: u32 = 1 << 30;

fn table_error() -> Error {
    Error::Internal("cosine table index")
}

/// `(cos, sin)` of the phase with linear interpolation, in f32 without FMA.
#[inline]
fn cos_sin_f32(phase: u32) -> Result<(f32, f32)> {
    let interp = |p: u32| -> Option<f32> {
        let idx = (p >> 20) as usize;
        let f = (p & 0xF_FFFF) as f32 * (1.0 / 1_048_576.0);
        let a = *COS_F32.get(idx)?;
        let b = *COS_F32.get(idx + 1)?;
        Some(a + (b - a) * f)
    };
    let c = interp(phase).ok_or_else(table_error)?;
    let s = interp(phase.wrapping_sub(QUARTER)).ok_or_else(table_error)?;
    Ok((c, s))
}

/// `(cos, sin)` of the phase as Q15 with linear interpolation in i64.
#[inline]
fn cos_sin_q15(phase: u32) -> Result<CQ15> {
    let interp = |p: u32| -> Option<i16> {
        let idx = (p >> 20) as usize;
        let f15 = i64::from((p >> 5) & 0x7FFF);
        let a = i64::from(*COS_Q15.get(idx)?);
        let b = i64::from(*COS_Q15.get(idx + 1)?);
        Some(sat16(a + round_shift((b - a) * f15, 15)))
    };
    let c = interp(phase).ok_or_else(table_error)?;
    let s = interp(phase.wrapping_sub(QUARTER)).ok_or_else(table_error)?;
    Ok(CQ15 {
        re: Q15(c),
        im: Q15(s),
    })
}

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

impl Rotate<C32> for Fallback {
    fn rotate(&self, nco: &mut Nco, input: &[C32], output: &mut [C32]) -> Result<()> {
        check_len(input, output)?;
        let inc = nco.increment();
        let mut phase = nco.phase();
        for (o, x) in output.iter_mut().zip(input) {
            let (c, s) = cos_sin_f32(phase)?;
            *o = C32::new(x.re * c - x.im * s, x.re * s + x.im * c);
            phase = phase.wrapping_add(inc);
        }
        nco.advance(input.len());
        Ok(())
    }
}

impl Rotate<CQ15> for Fallback {
    fn rotate(&self, nco: &mut Nco, input: &[CQ15], output: &mut [CQ15]) -> Result<()> {
        check_len(input, output)?;
        let inc = nco.increment();
        let mut phase = nco.phase();
        for (o, x) in output.iter_mut().zip(input) {
            *o = cq15_mul(*x, cos_sin_q15(phase)?);
            phase = phase.wrapping_add(inc);
        }
        nco.advance(input.len());
        Ok(())
    }
}
