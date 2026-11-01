//! `stereo_matrix` against the fallback: bit-exact at every length from 0 to 130, at several
//! start offsets, for random and extreme `CQ15` values and for every class of `C32` bit
//! pattern (NaN, infinities, `-0`, denormals); a length mismatch changes nothing. When `s` and
//! `d` are both NaN, which payload an addition of two NaNs returns depends on the operand order
//! that the compiler chose, in the fallback as well, and no contract fixes it. There the test
//! requires the sign and the payload (with the quiet bit set) of one of the two input NaNs, and
//! a quiet NaN, in both implementations; every other case is compared bit for bit.
#![allow(clippy::indexing_slicing, clippy::unwrap_used)]

mod common;

use common::{Rng, c32_noise, cq, cq15_noise};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::stereo::StereoMatrix;
use sdrk_core::{C32, CQ15, Error, Fallback};

const MAX_LEN: usize = 130;
const OFFSETS: [usize; 5] = [0, 1, 2, 3, 7];

fn check_cq15(x: &[CQ15], what: &str) {
    let mut want = vec![CQ15::default(); x.len()];
    let mut got = vec![CQ15::default(); x.len()];
    Fallback.stereo_matrix(x, &mut want).unwrap();
    Simd.stereo_matrix(x, &mut got).unwrap();
    assert_eq!(want, got, "{what}");
}

fn check_c32(x: &[C32], what: &str) {
    let mut want = vec![C32::ZERO; x.len()];
    let mut got = vec![C32::ZERO; x.len()];
    Fallback.stereo_matrix(x, &mut want).unwrap();
    Simd.stereo_matrix(x, &mut got).unwrap();
    for (k, ((w, g), i)) in want.iter().zip(&got).zip(x).enumerate() {
        let both_nan = i.re.is_nan() && i.im.is_nan();
        let quiet = |v: f32| v.to_bits() | 0x0040_0000;
        let from_input = |v: f32| v.is_nan() && [i.re, i.im].iter().any(|n| quiet(*n) == quiet(v));
        let same = |a: f32, b: f32| {
            a.to_bits() == b.to_bits() || (both_nan && from_input(a) && from_input(b))
        };
        assert!(
            same(w.re, g.re) && same(w.im, g.im),
            "{what}: sample {k}: {w:?} against {g:?}"
        );
    }
}

#[test]
fn cq15_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(2400);
    let x = cq15_noise(&mut rng, MAX_LEN + 8);
    for n in 0..=MAX_LEN {
        for off in OFFSETS {
            check_cq15(&x[off..off + n], &format!("n={n} offset={off}"));
        }
    }
}

#[test]
fn cq15_extreme_values_are_bit_exact() {
    let v = [
        i16::MIN,
        i16::MIN + 1,
        -16384,
        -1,
        0,
        1,
        16384,
        i16::MAX - 1,
        i16::MAX,
    ];
    let mut x = Vec::new();
    for &s in &v {
        for &d in &v {
            x.push(cq(s, d));
        }
    }
    for n in [0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 64, 65, x.len()] {
        check_cq15(&x[..n], &format!("n={n}"));
    }
}

#[test]
fn c32_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(2401);
    let x = c32_noise(&mut rng, MAX_LEN + 8, 1.0);
    for n in 0..=MAX_LEN {
        for off in OFFSETS {
            check_c32(&x[off..off + n], &format!("n={n} offset={off}"));
        }
    }
}

#[test]
fn c32_special_values_are_bit_exact() {
    let special = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        1.0e-40,
        -1.0e-40,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0x7FA0_0001),
        f32::from_bits(0xFFC1_2345),
    ];
    let mut x = Vec::new();
    for &s in &special {
        for &d in &special {
            x.push(C32::new(s, d));
        }
    }
    for n in [0, 1, 3, 4, 5, 8, 9, 16, 17, 33, x.len()] {
        check_c32(&x[..n], &format!("n={n}"));
    }
}

#[test]
fn length_mismatch_changes_nothing() {
    let x = vec![cq(1, 2); 20];
    let mut out = vec![cq(9, 9); 19];
    assert_eq!(
        Simd.stereo_matrix(&x, &mut out),
        Err(Error::LengthMismatch {
            expected: 20,
            got: 19
        })
    );
    assert!(out.iter().all(|o| *o == cq(9, 9)));
    let y = vec![C32::new(1.0, 2.0); 20];
    let mut o = vec![C32::new(9.0, 9.0); 21];
    assert!(Simd.stereo_matrix(&y, &mut o).is_err());
    assert!(o.iter().all(|c| *c == C32::new(9.0, 9.0)));
}

#[test]
fn matches_the_formula_on_plain_values() {
    let x = [C32::new(1.0, 0.25), C32::new(-3.0, 1.0)];
    let mut y = [C32::ZERO; 2];
    Simd.stereo_matrix(&x, &mut y).unwrap();
    assert_eq!(y, [C32::new(1.5, 0.5), C32::new(-1.0, -5.0)]);
}
