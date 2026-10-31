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
        // wasm32: blocks of 16 floats, so that one update of the pointers and the counter serves
        // four `f32x4.mul` (the compiler's own loop updates them for every two vectors: 6.53
        // instead of 5.5 instructions per sample with `simd128`). The same products.
        #[cfg(target_arch = "wasm32")]
        {
            if output.len() != input.len() {
                return Err(crate::Error::LengthMismatch {
                    expected: input.len(),
                    got: output.len(),
                });
            }
            let x: &[f32] = bytemuck::cast_slice(input);
            let y: &mut [f32] = bytemuck::cast_slice_mut(output);
            let (xc, xr) = x.as_chunks::<16>();
            let (yc, yr) = y.as_chunks_mut::<16>();
            for (o, i) in yc.iter_mut().zip(xc) {
                for (o, &v) in o.iter_mut().zip(i) {
                    *o = v * gain;
                }
            }
            for (o, &v) in yr.iter_mut().zip(xr) {
                *o = v * gain;
            }
            Ok(())
        }
        #[cfg(not(target_arch = "wasm32"))]
        map_kernel!(input, output, |x: C32| C32::new(x.re * gain, x.im * gain))
    }
}

impl Scale<CQ15> for Fallback {
    fn scale(&self, input: &[CQ15], output: &mut [CQ15], gain: Q15) -> Result<()> {
        // `q15_mul` in `i32`: the product is at most 2^30 in magnitude, so the rounding cannot
        // overflow, and the result is at least -32768, so only 32768 (-1 * -1) needs the
        // saturation. The same values as `q15_mul`, in a form the compiler vectorizes.
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        {
            if output.len() != input.len() {
                return Err(crate::Error::LengthMismatch {
                    expected: input.len(),
                    got: output.len(),
                });
            }
            let x: &[[u8; 4]] = bytemuck::cast_slice(input);
            let y: &mut [[u8; 4]] = bytemuck::cast_slice_mut(output);
            if gain.0 == i16::MIN {
                scale_words::<true>(x, y, gain.0);
            } else {
                scale_words::<false>(x, y, gain.0);
            }
            Ok(())
        }
        #[cfg(not(all(target_arch = "x86_64", not(target_feature = "avx512f"))))]
        {
            let g = i32::from(gain.0);
            let mul = |x: Q15| Q15(((i32::from(x.0) * g + (1 << 14)) >> 15).min(32767) as i16);
            map_kernel!(input, output, |x: CQ15| CQ15 {
                re: mul(x.re),
                im: mul(x.im),
            })
        }
    }
}

/// `scale` `CQ15` on x86-64 below AVX-512: every sample is read as one 32-bit word and both
/// halves are multiplied in its lanes (sign-extended in place, the results put back into their
/// halves), so the compiler needs no widening of 16-bit lanes and no repacking (`avx2`: 0.28 /
/// 0.42 instead of 0.67 / 0.57 simulated cycles per sample on `znver3` / `alderlake`; `sse2`
/// and `sse4.2` faster on both core models too; at `avx512` the widening form with `vpmovdw`
/// stays, the word form was 5 % slower on `sapphirerapids`). `SAT`: the gain is -32768, whose
/// product with -32768 is the only one that needs the saturation. Both halves get the same
/// operation, so the byte order does not matter.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
#[inline]
fn scale_words<const SAT: bool>(input: &[[u8; 4]], output: &mut [[u8; 4]], gain: i16) {
    let g = i32::from(gain);
    let mul = |x: i32| {
        let y = (x * g + (1 << 14)) >> 15;
        if SAT { y.min(32767) } else { y }
    };
    for (o, x) in output.iter_mut().zip(input) {
        let w = u32::from_ne_bytes(*x);
        let lo = mul(w.cast_signed() << 16 >> 16);
        let hi = mul(w.cast_signed() >> 16);
        *o = ((lo.cast_unsigned() & 0xffff) | (hi.cast_unsigned() << 16)).to_ne_bytes();
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
mod halfband;
mod interp;
mod mix;
mod modulate;
mod spectrum;
mod squelch;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::num::q15_mul;
    use std::vec::Vec;

    #[test]
    fn scale_cq15_is_q15_mul_for_every_value() {
        let values: Vec<i16> = (i16::MIN..=i16::MAX).collect();
        let input: Vec<CQ15> = values
            .iter()
            .zip(values.iter().rev())
            .map(|(&re, &im)| CQ15 {
                re: Q15(re),
                im: Q15(im),
            })
            .collect();
        for gain in [
            i16::MIN,
            i16::MIN + 1,
            -12345,
            -1,
            0,
            1,
            16384,
            24064,
            i16::MAX,
        ] {
            // Odd lengths and offsets: the head and the tail of the vector loop.
            for (start, len) in [(0usize, input.len()), (1, 7), (3, 1000), (5, 0)] {
                let x = &input[start..start + len];
                let mut out = std::vec![CQ15::default(); len];
                Fallback
                    .scale(x, &mut out, Q15(gain))
                    .unwrap_or_else(|_| unreachable!());
                for (o, s) in out.iter().zip(x) {
                    assert_eq!(o.re, q15_mul(s.re, Q15(gain)), "{gain} {s:?}");
                    assert_eq!(o.im, q15_mul(s.im, Q15(gain)), "{gain} {s:?}");
                }
            }
        }
        let mut short = [CQ15::default(); 2];
        assert!(Fallback.scale(&input[..3], &mut short, Q15(1)).is_err());
    }
}
