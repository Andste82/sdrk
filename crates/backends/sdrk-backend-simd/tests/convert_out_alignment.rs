//! `convert_out` of `C32` to `CU8`, `CS8` and `CS16` against the fallback, bit for bit, with the
//! input at every 4-byte offset from a 16-byte boundary (a `C32` slice is only 4-byte aligned)
//! and the output at every byte offset: the SSE2 path converts the values up to the first
//! 16-byte boundary of the input with the scalar formulas and then reads aligned vectors.
#![allow(clippy::indexing_slicing, clippy::unwrap_used)]

mod common;

use common::Rng;
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::convert::ConvertOut;
use sdrk_core::{C32, Fallback};
use sdrk_device::formats;

#[test]
fn convert_out_at_every_input_and_output_offset() {
    let mut rng = Rng::new(11);
    // Values of every kind: noise beyond full scale, the ties of every format, raw bit patterns
    // (NaN, infinities, subnormals, the tiny negative values of `CU8`).
    let mut value = || match rng.below(4) {
        0 => rng.unit() * 1.2,
        1 => (rng.below(1024) as f32 - 512.0) / 256.0,
        2 => {
            let exponent = 127 - 28 - rng.below(33) as u32;
            -f32::from_bits(exponent << 23 | rng.next_u32() & 0x7F_FFFF)
        }
        _ => f32::from_bits(rng.next_u32()),
    };
    let buffer: Vec<f32> = (0..260).map(|_| value()).collect();
    // From the first 16-byte boundary on.
    let floats = &buffer[buffer.as_ptr().addr().wrapping_neg() % 16 / 4..];
    for format in [formats::CU8, formats::CS8, formats::CS16] {
        let width = format.bytes_for_samples(1).unwrap();
        for offset in 0..4 {
            for len in (0..=72).chain([100, 101, 120]) {
                let input: &[C32] = bytemuck::cast_slice(&floats[offset..offset + 2 * len]);
                for skip in 0..4 {
                    let n = len * width;
                    let mut want = vec![0x5Au8; skip + n + 1];
                    let mut got = vec![0x5Au8; skip + n + 1];
                    let rw = Fallback.convert_out(&format, input, &mut want[skip..]);
                    let rg = Simd.convert_out(&format, input, &mut got[skip..]);
                    assert_eq!(rw, Ok(n));
                    assert_eq!(rg, Ok(n));
                    assert!(
                        want == got,
                        "{format:?} offset {offset} len {len} skip {skip}"
                    );
                }
            }
        }
    }
}
