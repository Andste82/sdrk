//! `am_mod` against the fallback, bit for bit: every length from 0 to 300 at four start offsets,
//! the shared tail lengths, every `Q15` input value, the `f32` corners (beyond the clamp, the
//! infinities, NaN, zeros of both signs, denormals), the gains at the corners and random ones,
//! and invalid calls (the same errors as the fallback).
#![allow(clippy::indexing_slicing, clippy::unwrap_used)]

mod common;

use common::{LENGTHS, Rng};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::modulate::AmMod;
use sdrk_core::{C32, CQ15, Fallback, Q15};

const OFFSETS: [usize; 4] = [0, 1, 2, 3];

const GAINS: [f32; 10] = [
    0.0,
    -0.0,
    1.0e-40,
    1.0e-6,
    0.25,
    0.5,
    0.8,
    0.999_984_7,
    0.999_99,
    1.0,
];

const F32_CORNERS: [f32; 18] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    1.000_000_1,
    -1.000_000_1,
    0.999_999_94,
    -0.999_999_94,
    2.0,
    -3.0e30,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    -f32::NAN,
    f32::MIN_POSITIVE,
    -1.0e-45,
    f32::MAX,
    f32::MIN,
];

fn bits(x: &[C32]) -> Vec<(u32, u32)> {
    x.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}

fn check_c32(carrier: f32, depth: f32, x: &[f32], what: &str) {
    let mut want = vec![C32::new(9.0, 9.0); x.len()];
    let mut got = vec![C32::new(7.0, 7.0); x.len()];
    Fallback.am_mod(carrier, depth, x, &mut want).unwrap();
    Simd.am_mod(carrier, depth, x, &mut got).unwrap();
    assert_eq!(
        bits(&want),
        bits(&got),
        "{what} carrier {carrier} depth {depth}"
    );
}

fn check_cq15(carrier: f32, depth: f32, x: &[Q15], what: &str) {
    let mut want = vec![CQ15::default(); x.len()];
    let mut got = vec![
        CQ15 {
            re: Q15(7),
            im: Q15(7)
        };
        x.len()
    ];
    Fallback.am_mod(carrier, depth, x, &mut want).unwrap();
    Simd.am_mod(carrier, depth, x, &mut got).unwrap();
    assert_eq!(want, got, "{what} carrier {carrier} depth {depth}");
}

/// Random values in `[-1.5, 1.5)` with a corner value at every seventh position.
fn f32_input(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            if i % 7 == 3 {
                F32_CORNERS[rng.below(F32_CORNERS.len())]
            } else {
                rng.unit() * 1.5
            }
        })
        .collect()
}

#[test]
fn c32_is_bit_exact_at_every_length_offset_and_gain() {
    let mut rng = Rng::new(350);
    let x = f32_input(&mut rng, 4096 + OFFSETS.len());
    for n in 0..=300 {
        let off = OFFSETS[n % OFFSETS.len()];
        let (c, m) = (
            GAINS[n % GAINS.len()],
            GAINS[(n / GAINS.len()) % GAINS.len()],
        );
        check_c32(c, m, &x[off..off + n], &format!("n={n} offset={off}"));
    }
    for n in LENGTHS {
        check_c32(0.5, 0.8, &x[..n], &format!("n={n}"));
    }
    for _ in 0..100 {
        let n = rng.below(600);
        let (c, m) = (rng.unit().abs(), rng.unit().abs());
        check_c32(c, m, &x[..n], &format!("random n={n}"));
    }
}

#[test]
fn c32_corners_are_bit_exact_for_every_gain_pair() {
    // Each corner at every position of a vector of up to 16 lanes.
    let x: Vec<f32> = (0..16 * F32_CORNERS.len())
        .map(|i| F32_CORNERS[(i + i / F32_CORNERS.len()) % F32_CORNERS.len()])
        .collect();
    for &c in &GAINS {
        for &m in &GAINS {
            check_c32(c, m, &x, "corners");
        }
    }
}

#[test]
fn cq15_is_bit_exact_for_every_input_value_and_gain_pair() {
    let x: Vec<Q15> = (i16::MIN..=i16::MAX).map(Q15).collect();
    for &c in &GAINS {
        for &m in &GAINS {
            check_cq15(c, m, &x, "all values");
        }
    }
    let mut rng = Rng::new(351);
    for _ in 0..20 {
        let (c, m) = (rng.unit().abs(), rng.unit().abs());
        check_cq15(c, m, &x, "all values, random gains");
    }
}

#[test]
fn cq15_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(352);
    let x: Vec<Q15> = (0..4096 + OFFSETS.len())
        .map(|i| match i % 9 {
            0 => Q15::MIN,
            1 => Q15::MAX,
            _ => Q15(rng.next_u32() as i16),
        })
        .collect();
    for n in 0..=300 {
        let off = OFFSETS[n % OFFSETS.len()];
        let (c, m) = (
            GAINS[n % GAINS.len()],
            GAINS[(n / GAINS.len()) % GAINS.len()],
        );
        check_cq15(c, m, &x[off..off + n], &format!("n={n} offset={off}"));
    }
    for n in LENGTHS {
        check_cq15(1.0, 1.0, &x[..n], &format!("n={n}"));
    }
}

#[test]
fn invalid_calls_give_the_errors_of_the_fallback() {
    let bad = [-1.0e-6, 1.000_000_1, f32::NAN, f32::INFINITY, -2.0];
    let x = [0.25f32; 4];
    let q = [Q15(100); 4];
    for &v in &bad {
        for (c, m) in [(v, 0.5), (0.5, v)] {
            let mut a = [C32::default(); 4];
            let mut b = [C32::default(); 4];
            assert_eq!(
                Simd.am_mod(c, m, &x, &mut a),
                Fallback.am_mod(c, m, &x, &mut b)
            );
            assert!(Simd.am_mod(c, m, &x, &mut a).is_err());
            let mut a = [CQ15::default(); 4];
            let mut b = [CQ15::default(); 4];
            assert_eq!(
                Simd.am_mod(c, m, &q, &mut a),
                Fallback.am_mod(c, m, &q, &mut b)
            );
        }
    }
    let mut a = [C32::default(); 3];
    let mut b = [C32::default(); 3];
    assert_eq!(
        Simd.am_mod(0.5, 0.5, &x, &mut a),
        Fallback.am_mod(0.5, 0.5, &x, &mut b)
    );
    assert!(Simd.am_mod(0.5, 0.5, &x, &mut a).is_err());
    let mut a = [CQ15::default(); 5];
    let mut b = [CQ15::default(); 5];
    assert_eq!(
        Simd.am_mod(0.5, 0.5, &q, &mut a),
        Fallback.am_mod(0.5, 0.5, &q, &mut b)
    );
    assert!(Simd.am_mod(0.5, 0.5, &q, &mut a).is_err());
}
