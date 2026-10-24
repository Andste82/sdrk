//! `mag_sq` of `C32` and `CQ15` against the fallback: bit-exact at every length from 0 to 4096,
//! at every start offset within a vector, and for the extreme values. The tests hold in every
//! build, the relaxed-SIMD build of wasm32 included: `mag_sq` uses no multiply-add.
#![allow(clippy::indexing_slicing, clippy::unwrap_used)]

mod common;

use common::{Rng, c32_noise, cq15_noise};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::spectrum::MagSq;
use sdrk_core::{C32, CQ15, Fallback, Q15};

/// The longest length; every length up to it is tested.
const MAX_LEN: usize = 4096;

/// Start offsets in samples: the kernels must not depend on the alignment of the buffers.
const OFFSETS: [usize; 4] = [0, 1, 2, 3];

/// Equal bits, or both NaN (the payload of a NaN result is not specified).
fn same_f32(want: &[f32], got: &[f32]) -> bool {
    want.iter()
        .zip(got)
        .all(|(w, g)| w.to_bits() == g.to_bits() || (w.is_nan() && g.is_nan()))
}

fn check_c32(x: &[C32], what: &str) {
    let (mut want, mut got) = (vec![0f32; x.len()], vec![0f32; x.len()]);
    Fallback.mag_sq(x, &mut want).unwrap();
    Simd.mag_sq(x, &mut got).unwrap();
    assert!(same_f32(&want, &got), "{what}: {want:?} {got:?}");
}

fn check_cq15(x: &[CQ15], what: &str) {
    let (mut want, mut got) = (vec![Q15(0); x.len()], vec![Q15(0); x.len()]);
    Fallback.mag_sq(x, &mut want).unwrap();
    Simd.mag_sq(x, &mut got).unwrap();
    assert_eq!(want, got, "{what}");
}

#[test]
fn c32_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(140);
    let x = c32_noise(&mut rng, MAX_LEN + OFFSETS.len(), 4.0);
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        check_c32(&x[off..off + n], &format!("n={n} offset={off}"));
    }
}

#[test]
fn cq15_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(141);
    let x = cq15_noise(&mut rng, MAX_LEN + OFFSETS.len());
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        check_cq15(&x[off..off + n], &format!("n={n} offset={off}"));
    }
}

#[test]
fn c32_extremes_are_bit_exact() {
    let v = [
        0.0,
        -0.0,
        1e-45,
        -1e-40,
        1e-20,
        1.5,
        -3.0,
        1e19,
        1.9e19,
        -3e38,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    // Every pair, repeated so that it passes through the vector loops as well as the tail.
    let x: Vec<C32> = v
        .iter()
        .flat_map(|&re| v.iter().map(move |&im| C32::new(re, im)))
        .collect();
    let x: Vec<C32> = x.iter().chain(&x).chain(&x).copied().collect();
    for off in OFFSETS {
        check_c32(&x[off..], &format!("offset={off}"));
    }
}

#[test]
fn cq15_is_bit_exact_for_every_real_part() {
    // Every real part against the imaginary parts at the ends of the range and around the
    // saturation of the sum (`181^2 * 2 > 2^16`, `23170^2 * 2 ~ 2^30`).
    for im in [-32768, -32767, -23171, -1, 0, 1, 181, 23170, 32766, 32767] {
        let x: Vec<CQ15> = (i16::MIN..=i16::MAX)
            .map(|re| CQ15 {
                re: Q15(re),
                im: Q15(im),
            })
            .collect();
        check_cq15(&x, &format!("im={im}"));
        check_cq15(&x[1..], &format!("im={im} offset=1"));
    }
}

#[test]
fn cq15_saturates_at_the_corner() {
    let x = vec![
        CQ15 {
            re: Q15(-32768),
            im: Q15(-32768),
        };
        67
    ];
    let mut got = vec![Q15(0); x.len()];
    Simd.mag_sq(&x, &mut got).unwrap();
    assert!(got.iter().all(|&q| q == Q15(32767)));
    check_cq15(&x, "corner");
}
