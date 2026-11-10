//! `qam_soft_bits` (M44 §2, I44-1) against the fallback, bit for bit at the level of the running
//! build: both constellations, every symbol count up to 100 (every tail of the vector steps of
//! every level), every start offset of the input and of the output within a vector, several
//! scales, values around the soft-bit boundaries and special values (signed zeros, infinities,
//! NaN, denormals, huge values), and the length check.
#![allow(clippy::indexing_slicing, clippy::unwrap_used)]

mod common;

use common::Rng;
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::constellation::{Qam, QamSoftBits};
use sdrk_core::{C32, Error, Fallback};

const BOTH: [Qam; 2] = [Qam::Qam16, Qam::Qam64];
/// Start offsets in values: the kernel must not depend on the alignment of the buffers.
const OFFSETS: [usize; 4] = [0, 1, 2, 3];
const SCALES: [f32; 5] = [1.0, 0.25, 5.3, 1e-9, 1e9];

/// The bits of a soft bit, with every NaN folded to one value (the sign and payload of a NaN
/// are not part of the contract).
fn bits(x: f32) -> u32 {
    if x.is_nan() { u32::MAX } else { x.to_bits() }
}

/// A component: a special value, one near a soft-bit boundary (`|v|` near 0, 2, 4 or 6), or
/// a level with noise.
fn component(rng: &mut Rng, qam: Qam) -> f32 {
    let special = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1e-40,
        -1e-45,
        f32::MAX,
        f32::MIN,
        3e19,
    ];
    let c = qam.sqrt_energy();
    match rng.below(8) {
        0 => special[rng.below(special.len())],
        1 => {
            let k = [0.0f32, 2.0, 4.0, 6.0][rng.below(4)];
            let v = (k + 1e-6 * rng.unit()) / c;
            if rng.below(2) == 0 { v } else { -v }
        }
        _ => {
            let l = qam.levels();
            l[rng.below(l.len())] + 0.3 * rng.unit()
        }
    }
}

fn soft<K: QamSoftBits<C32>>(k: &K, qam: Qam, x: &[C32], out: &mut [f32], s: f32) {
    k.qam_soft_bits(qam, x, out, s).unwrap();
}

#[test]
fn qam_soft_bits_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(778);
    for qam in BOTH {
        let b = qam.bits_per_symbol();
        for n in 0..=100 {
            let x: Vec<C32> = (0..n + 3)
                .map(|_| C32::new(component(&mut rng, qam), component(&mut rng, qam)))
                .collect();
            for (i, &s) in SCALES.iter().enumerate() {
                let off = OFFSETS[(n + i) % OFFSETS.len()];
                let xs = &x[off..off + n];
                let mut simd = vec![7.0f32; n * b + 3];
                let mut fall = vec![7.0f32; n * b + 3];
                let oo = OFFSETS[(n + 2 * i) % OFFSETS.len()].min(3);
                soft(&Simd, qam, xs, &mut simd[oo..oo + n * b], s);
                soft(&Fallback, qam, xs, &mut fall[oo..oo + n * b], s);
                let (a, f): (Vec<u32>, Vec<u32>) = (
                    simd.iter().map(|&v| bits(v)).collect(),
                    fall.iter().map(|&v| bits(v)).collect(),
                );
                assert_eq!(a, f, "{qam:?} n {n} scale {s} offsets {off} {oo}");
            }
        }
    }
}

#[test]
fn qam_soft_bits_checks_the_output_length() {
    let x = [C32::new(0.1, -0.2); 5];
    for qam in BOTH {
        let b = qam.bits_per_symbol();
        for got in [0, 5 * b - 1, 5 * b + 1] {
            let mut out = vec![0.0f32; got];
            assert_eq!(
                Simd.qam_soft_bits(qam, &x, &mut out, 1.0),
                Err(Error::LengthMismatch {
                    expected: 5 * b,
                    got
                }),
                "{qam:?} {got}"
            );
        }
        assert_eq!(Simd.qam_soft_bits(qam, &[], &mut [], 1.0), Ok(()));
    }
}
