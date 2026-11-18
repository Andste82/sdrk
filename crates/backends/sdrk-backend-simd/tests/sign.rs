//! The vector helpers of the module `sign` (M49 §1.1, test 7) against the scalar helpers of
//! `sdrk_core::num` at the level of the running build: every `Q15` value in each component,
//! every quadrant 0 to 7 and every chip bit, at lengths that cover every vector tail; `C32` by
//! `to_bits()` with `±0`, `±∞`, subnormals and NaN payloads.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_possible_truncation
)]

mod common;

use common::Rng;
use sdrk_backend_simd::sign;
use sdrk_core::num::{c32_chip, c32_quadrant, cq15_chip, cq15_quadrant, q15_neg_sat};
use sdrk_core::{C32, CQ15, Error, Q15};

/// Every `Q15` value.
fn all_q15() -> Vec<Q15> {
    (i16::MIN..=i16::MAX).map(Q15).collect()
}

fn c32_bits(v: &[C32]) -> Vec<(u32, u32)> {
    v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}

/// `C32` values: specials (signed zeros, infinities, subnormals, NaN payloads) and random ones.
fn c32_values(n: usize) -> Vec<C32> {
    let sp = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x007F_FFFF),
        f32::from_bits(0x7FC0_0000),
        f32::from_bits(0xFFC0_1234),
        f32::from_bits(0x7F80_0001),
        f32::MAX,
        f32::MIN,
        1.5,
    ];
    let mut rng = Rng::new(696);
    let mut v: Vec<C32> = sp
        .iter()
        .flat_map(|&a| sp.iter().map(move |&b| C32::new(a, b)))
        .collect();
    while v.len() < n {
        v.push(C32::new(f32::from_bits(rng.next_u32()), rng.unit()));
    }
    v
}

#[test]
fn q15_neg_sat_equals_the_scalar_helper_on_every_value() {
    let x = all_q15();
    let mut y = vec![Q15(0); x.len()];
    sign::q15_neg_sat(&x, &mut y).unwrap();
    for (&a, &b) in x.iter().zip(&y) {
        assert_eq!(b, q15_neg_sat(a), "{a:?}");
    }
    assert_eq!(y[0], Q15(32767));
    // Every tail length.
    for n in 0..=67 {
        let mut y = vec![Q15(0); n];
        sign::q15_neg_sat(&x[..n], &mut y).unwrap();
        assert!(
            y.iter().zip(&x).all(|(&b, &a)| b == q15_neg_sat(a)),
            "n {n}"
        );
    }
}

#[test]
fn cq15_quadrant_equals_the_scalar_helper_on_every_value_in_each_component() {
    let all = all_q15();
    let mut rng = Rng::new(694);
    for q in 0..8u8 {
        for comp in 0..2 {
            let x: Vec<CQ15> = all
                .iter()
                .map(|&v| {
                    let o = Q15(rng.next_u32() as i16);
                    if comp == 0 {
                        CQ15 { re: v, im: o }
                    } else {
                        CQ15 { re: o, im: v }
                    }
                })
                .collect();
            let qs = vec![q; x.len()];
            let mut y = vec![CQ15::default(); x.len()];
            sign::cq15_quadrant(&x, &qs, &mut y).unwrap();
            for (&a, &b) in x.iter().zip(&y) {
                assert_eq!(b, cq15_quadrant(a, u32::from(q)), "q {q} {a:?}");
            }
        }
    }
}

#[test]
fn cq15_quadrant_with_a_quadrant_per_sample_at_every_length() {
    let mut rng = Rng::new(690);
    let x: Vec<CQ15> = (0..4096 + 4)
        .map(|i| CQ15 {
            re: Q15(if i % 5 == 0 {
                -32768
            } else {
                rng.next_u32() as i16
            }),
            im: Q15(if i % 7 == 0 {
                -32768
            } else {
                rng.next_u32() as i16
            }),
        })
        .collect();
    let q: Vec<u8> = (0..x.len()).map(|_| rng.next_u32() as u8).collect();
    for n in (0..=67).chain([4096]) {
        for off in 0..3 {
            let mut y = vec![CQ15::default(); n];
            sign::cq15_quadrant(&x[off..off + n], &q[off..off + n], &mut y).unwrap();
            for ((&a, &k), &b) in x[off..].iter().zip(&q[off..]).zip(&y) {
                assert_eq!(b, cq15_quadrant(a, u32::from(k)), "n {n} q {k}");
            }
        }
    }
}

#[test]
fn cq15_chip_equals_the_scalar_helper_on_every_value() {
    let all = all_q15();
    for b in 0..4u8 {
        let x: Vec<CQ15> = all
            .iter()
            .map(|&v| CQ15 {
                re: v,
                im: Q15(v.0.wrapping_mul(7).wrapping_add(3)),
            })
            .collect();
        let bs = vec![b; x.len()];
        let mut y = vec![CQ15::default(); x.len()];
        sign::cq15_chip(&x, &bs, &mut y).unwrap();
        for (&a, &o) in x.iter().zip(&y) {
            assert_eq!(o, cq15_chip(a, u32::from(b)), "b {b} {a:?}");
        }
    }
}

#[test]
fn c32_quadrant_and_chip_equal_the_scalar_helpers_bit_for_bit() {
    let x = c32_values(4096 + 4);
    let mut rng = Rng::new(699);
    let q: Vec<u8> = (0..x.len()).map(|_| rng.next_u32() as u8).collect();
    for n in (0..=67).chain([169, 4096]) {
        for off in 0..3 {
            let xs = &x[off..off + n];
            let qs = &q[off..off + n];
            let mut y = vec![C32::default(); n];
            sign::c32_quadrant(xs, qs, &mut y).unwrap();
            let want: Vec<C32> = xs
                .iter()
                .zip(qs)
                .map(|(&a, &k)| c32_quadrant(a, u32::from(k)))
                .collect();
            assert_eq!(c32_bits(&y), c32_bits(&want), "quadrant n {n} off {off}");
            sign::c32_chip(xs, qs, &mut y).unwrap();
            let want: Vec<C32> = xs
                .iter()
                .zip(qs)
                .map(|(&a, &k)| c32_chip(a, u32::from(k)))
                .collect();
            assert_eq!(c32_bits(&y), c32_bits(&want), "chip n {n} off {off}");
        }
    }
}

#[test]
fn the_round_trip_of_the_quadrant_is_exact_except_at_minus_32768() {
    let all = all_q15();
    for q in 0..4u8 {
        let x: Vec<CQ15> = all
            .iter()
            .map(|&v| CQ15 {
                re: v,
                im: Q15(v.0.reverse_bits()),
            })
            .collect();
        let mut y = vec![CQ15::default(); x.len()];
        let mut z = vec![CQ15::default(); x.len()];
        sign::cq15_quadrant(&x, &vec![q; x.len()], &mut y).unwrap();
        sign::cq15_quadrant(&y, &vec![(4 - q) & 3; x.len()], &mut z).unwrap();
        for (&a, &b) in x.iter().zip(&z) {
            let fix = |v: Q15, negated: bool| {
                if negated && v.0 == -32768 {
                    Q15(-32767)
                } else {
                    v
                }
            };
            let want = CQ15 {
                re: fix(a.re, q >= 2),
                im: fix(a.im, q == 1 || q == 2),
            };
            assert_eq!(b, want, "q {q} {a:?}");
        }
    }
}

#[test]
fn length_mismatches_are_errors() {
    let x = [CQ15::default(); 4];
    let mut y = [CQ15::default(); 4];
    assert!(matches!(
        sign::cq15_quadrant(&x, &[0; 3], &mut y),
        Err(Error::LengthMismatch { .. })
    ));
    assert!(matches!(
        sign::cq15_chip(&x, &[0; 4], &mut y[..3]),
        Err(Error::LengthMismatch { .. })
    ));
    let xc = [C32::default(); 4];
    let mut yc = [C32::default(); 4];
    assert!(matches!(
        sign::c32_quadrant(&xc, &[0; 5], &mut yc),
        Err(Error::LengthMismatch { .. })
    ));
    assert!(matches!(
        sign::q15_neg_sat(&[Q15(0); 2], &mut [Q15(0); 3]),
        Err(Error::LengthMismatch { .. })
    ));
}
