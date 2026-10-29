//! `dot` and `dot_real` against the fallback: the `Q15` and `CQ15` sums bit-exact at every
//! length up to 1100 and at longer lengths across the 32-bit flushes of the level paths, at every
//! start offset within a vector, and for the extreme values; the `f32` and `C32` sums within
//! the error bound of a float sum (they are formed in another order than the fallback's).
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_precision_loss
)]

mod common;

use common::{Rng, c32_noise, cq15_noise};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::dot::{Dot, DotReal};
use sdrk_core::{C32, CQ15, Error, Fallback, Q15};

/// Every length up to this one is tested, then [`LONG`].
const MAX_LEN: usize = 1100;

/// Longer lengths: around the flushes of the 32-bit sums (512 and 1024 samples and their
/// multiples) and around the vector widths.
const LONG: [usize; 12] = [
    1535, 1536, 1537, 2047, 2048, 2049, 4095, 4096, 4097, 5000, 8191, 8203,
];

/// Start offsets in samples: the kernels must not depend on the alignment of the buffers.
const OFFSETS: [usize; 4] = [0, 1, 2, 3];

fn lengths() -> impl Iterator<Item = usize> {
    (0..=MAX_LEN).chain(LONG)
}

fn q15_taps(rng: &mut Rng, n: usize) -> Vec<Q15> {
    (0..n).map(|_| Q15(rng.next_u32() as i16)).collect()
}

#[test]
fn cq15_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(170);
    let n_max = LONG[LONG.len() - 1] + OFFSETS.len();
    let x = cq15_noise(&mut rng, n_max);
    let t = q15_taps(&mut rng, n_max);
    for (k, n) in lengths().enumerate() {
        let (a, b) = (OFFSETS[k % 4], OFFSETS[(k / 4) % 4]);
        let (x, t) = (&x[a..a + n], &t[b..b + n]);
        assert_eq!(
            Simd.dot(x, t).unwrap(),
            Fallback.dot(x, t).unwrap(),
            "n={n} offsets {a} {b}"
        );
    }
}

#[test]
fn q15_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(171);
    let n_max = LONG[LONG.len() - 1] + OFFSETS.len();
    let x = q15_taps(&mut rng, n_max);
    let t = q15_taps(&mut rng, n_max);
    for (k, n) in lengths().enumerate() {
        let (a, b) = (OFFSETS[k % 4], OFFSETS[(k / 4) % 4]);
        let (x, t) = (&x[a..a + n], &t[b..b + n]);
        assert_eq!(
            Simd.dot_real(x, t).unwrap(),
            Fallback.dot_real(x, t).unwrap(),
            "n={n} offsets {a} {b}"
        );
    }
}

/// The scaled-down noise of the taps keeps the sums inside the Q15 range, so that the
/// comparison sees the exact sums and not only the saturation.
#[test]
fn integer_sums_below_saturation_are_bit_exact() {
    let mut rng = Rng::new(172);
    for n in lengths().step_by(7) {
        let x = cq15_noise(&mut rng, n);
        let t: Vec<Q15> = (0..n).map(|_| Q15((rng.next_u32() as i16) >> 9)).collect();
        let r: Vec<Q15> = x.iter().map(|s| s.re).collect();
        assert_eq!(
            Simd.dot(&x, &t).unwrap(),
            Fallback.dot(&x, &t).unwrap(),
            "n={n}"
        );
        assert_eq!(
            Simd.dot_real(&r, &t).unwrap(),
            Fallback.dot_real(&r, &t).unwrap(),
            "n={n}"
        );
    }
}

/// Full-scale products of one sign for 4096 samples and of the other sign for the next 4096
/// sum to zero, so a sum that overflowed anywhere on the way (in a 32-bit lane between two
/// flushes, or a pair of `(-32768)^2`) would show in the result instead of being hidden by the
/// saturation.
#[test]
fn extreme_values_cancel_exactly() {
    let half = 4096;
    for (a, b, c) in [
        (i16::MIN, i16::MIN, i16::MAX),
        (i16::MAX, i16::MAX, -i16::MAX),
        (i16::MIN, i16::MAX, -i16::MAX),
        (i16::MAX, i16::MIN, i16::MAX),
    ] {
        // `a * b` for the first half; `a * c` for the second, of the opposite sign.
        for n in [2 * half, 2 * half + 5] {
            let x: Vec<Q15> = (0..n).map(|_| Q15(a)).collect();
            let t: Vec<Q15> = (0..n).map(|i| Q15(if i < half { b } else { c })).collect();
            let xc: Vec<CQ15> = x.iter().map(|&q| CQ15 { re: q, im: q }).collect();
            let want = Fallback.dot_real(&x, &t).unwrap();
            assert_eq!(Simd.dot_real(&x, &t).unwrap(), want, "{a} {b} {c} n={n}");
            assert_eq!(
                Simd.dot(&xc, &t).unwrap(),
                CQ15 { re: want, im: want },
                "{a} {b} {c} n={n}"
            );
        }
    }
    // Saturation: every sample and tap -32768.
    let x = vec![Q15(i16::MIN); 3000];
    let xc = vec![
        CQ15 {
            re: Q15(i16::MIN),
            im: Q15(i16::MIN)
        };
        3000
    ];
    assert_eq!(Simd.dot_real(&x, &x).unwrap(), Q15(i16::MAX));
    assert_eq!(
        Simd.dot(&xc, &x).unwrap(),
        CQ15 {
            re: Q15(i16::MAX),
            im: Q15(i16::MAX)
        }
    );
}

/// `|got - exact| <= 2 n eps sum |x h|` (the bound of any order of a float sum, with room
/// for the fused and the unfused products).
fn within_bound(got: f32, x: &[f32], t: &[f32]) -> bool {
    let exact: f64 = x
        .iter()
        .zip(t)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum();
    let abs: f64 = x
        .iter()
        .zip(t)
        .map(|(&a, &b)| (f64::from(a) * f64::from(b)).abs())
        .sum();
    let bound = 2.0 * (x.len() as f64 + 1.0) * f64::from(f32::EPSILON) * abs;
    (f64::from(got) - exact).abs() <= bound
}

#[test]
fn f32_is_within_the_bound_of_a_float_sum() {
    let mut rng = Rng::new(173);
    let n_max = LONG[LONG.len() - 1] + OFFSETS.len();
    let x: Vec<f32> = (0..n_max).map(|_| rng.unit()).collect();
    let t: Vec<f32> = (0..n_max).map(|_| rng.unit()).collect();
    for (k, n) in lengths().enumerate() {
        let (a, b) = (OFFSETS[k % 4], OFFSETS[(k / 4) % 4]);
        let (x, t) = (&x[a..a + n], &t[b..b + n]);
        let got = Simd.dot_real(x, t).unwrap();
        assert!(within_bound(got, x, t), "n={n}: {got}");
        if n <= 1 {
            assert_eq!(got.to_bits(), Fallback.dot_real(x, t).unwrap().to_bits());
        }
    }
}

#[test]
fn c32_is_within_the_bound_of_a_float_sum() {
    let mut rng = Rng::new(174);
    let n_max = LONG[LONG.len() - 1] + OFFSETS.len();
    let x = c32_noise(&mut rng, n_max, 1.0);
    let t: Vec<f32> = (0..n_max).map(|_| rng.unit()).collect();
    for (k, n) in lengths().enumerate() {
        let (a, b) = (OFFSETS[k % 4], OFFSETS[(k / 4) % 4]);
        let (x, t) = (&x[a..a + n], &t[b..b + n]);
        let got = Simd.dot(x, t).unwrap();
        let re: Vec<f32> = x.iter().map(|s| s.re).collect();
        let im: Vec<f32> = x.iter().map(|s| s.im).collect();
        assert!(within_bound(got.re, &re, t), "n={n}: {got}");
        assert!(within_bound(got.im, &im, t), "n={n}: {got}");
    }
}

/// An exactly representable sum (small integers) is exact in any order: the real and the
/// imaginary lanes must not be mixed up, at any length.
#[test]
fn exact_sums_are_bit_exact() {
    let mut rng = Rng::new(175);
    for n in lengths().step_by(3) {
        let x: Vec<C32> = (0..n)
            .map(|_| C32::new(rng.below(17) as f32 - 8.0, rng.below(9) as f32 * 2.0))
            .collect();
        let t: Vec<f32> = (0..n).map(|_| rng.below(7) as f32 - 3.0).collect();
        let r: Vec<f32> = x.iter().map(|s| s.re).collect();
        assert_eq!(
            Simd.dot(&x, &t).unwrap(),
            Fallback.dot(&x, &t).unwrap(),
            "n={n}"
        );
        assert_eq!(
            Simd.dot_real(&r, &t).unwrap(),
            Fallback.dot_real(&r, &t).unwrap(),
            "n={n}"
        );
    }
}

/// NaN and the infinities propagate as in the fallback (a NaN result in both, or the same
/// infinity).
#[test]
fn specials_propagate() {
    let same = |a: f32, b: f32| (a.is_nan() && b.is_nan()) || a == b;
    for n in [1usize, 5, 16, 33, 100] {
        for (pos, v) in [
            (0, f32::NAN),
            (n - 1, f32::INFINITY),
            (n / 2, f32::NEG_INFINITY),
        ] {
            let mut x = vec![0.25f32; n];
            x[pos] = v;
            let t = vec![0.5f32; n];
            let xc: Vec<C32> = x.iter().map(|&r| C32::new(r, -r)).collect();
            let (a, b) = (
                Simd.dot_real(&x, &t).unwrap(),
                Fallback.dot_real(&x, &t).unwrap(),
            );
            assert!(same(a, b), "n={n} {v}: {a} {b}");
            let (a, b) = (Simd.dot(&xc, &t).unwrap(), Fallback.dot(&xc, &t).unwrap());
            assert!(same(a.re, b.re) && same(a.im, b.im), "n={n} {v}: {a} {b}");
        }
    }
}

#[test]
fn unequal_lengths_are_errors_and_empty_slices_give_zero() {
    let mismatch = Error::LengthMismatch {
        expected: 3,
        got: 2,
    };
    assert_eq!(
        Simd.dot(&[C32::default(); 3], &[0.0; 2]),
        Err(mismatch.clone())
    );
    assert_eq!(
        Simd.dot(&[CQ15::default(); 3], &[Q15(0); 2]),
        Err(mismatch.clone())
    );
    assert_eq!(
        Simd.dot_real(&[0.0f32; 3], &[0.0; 2]),
        Err(mismatch.clone())
    );
    assert_eq!(Simd.dot_real(&[Q15(0); 3], &[Q15(0); 2]), Err(mismatch));
    assert_eq!(Simd.dot(&[] as &[C32], &[]), Ok(C32::default()));
    assert_eq!(Simd.dot(&[] as &[CQ15], &[]), Ok(CQ15::default()));
    assert_eq!(Simd.dot_real(&[] as &[f32], &[]), Ok(0.0));
    assert_eq!(Simd.dot_real(&[] as &[Q15], &[]), Ok(Q15(0)));
}
