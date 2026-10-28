//! Fallback implementation of the `rotate` kernel.

use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::mix::{Nco, Rotate};
use crate::num::{round_shift, sat16};
use crate::tables::{COS_F32, COS_Q15};
use crate::types::{C32, CQ15, Q15};

const QUARTER: u32 = 1 << 30;

fn table_error() -> Error {
    Error::Internal("cosine table index")
}

/// `(cos, sin)` of the phase with linear interpolation, in f32 without FMA.
#[inline]
pub(super) fn cos_sin_f32(phase: u32) -> Result<(f32, f32)> {
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
pub(super) fn cos_sin_q15(phase: u32) -> Result<CQ15> {
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

/// `x` rotated by the phase: `cq15_mul(x, cos_sin_q15(phase))` in `i32`, bit for bit.
///
/// The interpolation `a + round_shift((b - a) f15, 15)` cannot overflow `i32` (`|b - a| < 2^16`,
/// `f15 < 2^15`) and lies between `a` and `b`, so the saturation of `cos_sin_q15` never acts.
/// The products `xr c - xi s` and `xr s + xi c` are below `|x| |(c, s)| <= 2^15 sqrt(2) (2^15 + 1)`
/// in magnitude, about `1.52e9`, so they and the rounding term are exact in `i32` as well; the
/// arithmetic shift and the clamp are `round_shift` and `sat16`. On 32-bit targets this avoids
/// the 64-bit multiplications of the generic helpers, on 64-bit targets the sign extensions and
/// the saturating addition.
#[inline]
fn rotate_q15(x: CQ15, phase: u32) -> Result<CQ15> {
    let interp = |p: u32| -> Option<i32> {
        let idx = (p >> 20) as usize;
        let f15 = ((p >> 5) & 0x7FFF) as i32;
        match COS_Q15.get(idx..idx + 2) {
            Some(&[a, b]) => {
                let (a, b) = (i32::from(a), i32::from(b));
                Some(a + (((b - a) * f15 + (1 << 14)) >> 15))
            }
            _ => None,
        }
    };
    let c = interp(phase).ok_or_else(table_error)?;
    let s = interp(phase.wrapping_sub(QUARTER)).ok_or_else(table_error)?;
    let (xr, xi) = (i32::from(x.re.0), i32::from(x.im.0));
    let narrow = |v: i32| ((v + (1 << 14)) >> 15).clamp(-32768, 32767) as i16;
    Ok(CQ15 {
        re: Q15(narrow(xr * c - xi * s)),
        im: Q15(narrow(xr * s + xi * c)),
    })
}

impl Rotate<CQ15> for Fallback {
    fn rotate(&self, nco: &mut Nco, input: &[CQ15], output: &mut [CQ15]) -> Result<()> {
        check_len(input, output)?;
        let inc = nco.increment();
        let mut phase = nco.phase();
        for (o, x) in output.iter_mut().zip(input) {
            *o = rotate_q15(*x, phase)?;
            phase = phase.wrapping_add(inc);
        }
        nco.advance(input.len());
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    //! The `i32` rotation of the `CQ15` kernel against the `i64` helpers that define it.

    use super::{cos_sin_q15, rotate_q15};
    use crate::num::cq15_mul;
    use crate::types::{CQ15, Q15};

    fn check(x: CQ15, phase: u32) {
        let want = cq15_mul(x, cos_sin_q15(phase).unwrap());
        assert_eq!(
            rotate_q15(x, phase).unwrap(),
            want,
            "x={x:?} phase={phase:#x}"
        );
    }

    #[test]
    fn the_i32_rotation_equals_cq15_mul_of_the_table_value() {
        // Every table index at several fractions, with full-scale samples of every sign
        // (products and sums at their largest, saturation of both components).
        let edges = [-32768i16, -32767, -1, 0, 1, 32767];
        let mut samples = std::vec::Vec::new();
        for re in edges {
            for im in edges {
                samples.push(CQ15 {
                    re: Q15(re),
                    im: Q15(im),
                });
            }
        }
        for idx in 0..4096u32 {
            for frac in [0u32, 1, 0x1F, 0x20, 0x8_0000, 0xF_FFE0, 0xF_FFFF] {
                let phase = idx << 20 | frac;
                for &x in &samples {
                    check(x, phase);
                }
            }
        }
        // Random samples and phases (xorshift64).
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        for _ in 0..200_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let x = CQ15 {
                re: Q15(s as i16),
                im: Q15((s >> 16) as i16),
            };
            check(x, (s >> 32) as u32);
        }
    }
}
