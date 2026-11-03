//! `add`, `add_real`, `mul`, `mul_real`, `interleave` and `deinterleave` against the fallback:
//! bit-exact at every length from 0 to 300, at every start offset within a vector, for random
//! samples and for the extreme and special values, and the errors of unequal lengths.
#![allow(clippy::indexing_slicing, clippy::unwrap_used)]

mod common;

use common::{Rng, c32_noise, cq15_noise};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::math::{Add, AddReal, Interleave, Mul, MulReal};
use sdrk_core::{C32, CQ15, Error, Fallback, Q15};

const MAX_LEN: usize = 300;
const OFFSETS: [usize; 4] = [0, 1, 2, 3];

fn same_f32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn same_c32(a: &[C32], b: &[C32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| same_f32(x.re, y.re) && same_f32(x.im, y.im))
}

fn q15_noise(rng: &mut Rng, n: usize) -> Vec<Q15> {
    (0..n).map(|_| Q15(rng.next_u32() as i16)).collect()
}

/// Every pair of the extreme values, as a stream of `v.len()^2` samples.
const EXTREMES: [i16; 9] = [
    i16::MIN,
    i16::MIN + 1,
    -16384,
    -1,
    0,
    1,
    16383,
    16384,
    i16::MAX,
];

fn extreme_pairs() -> (Vec<CQ15>, Vec<CQ15>) {
    let v = EXTREMES;
    let m = v.len();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for i in 0..m * m * m * m {
        a.push(CQ15 {
            re: Q15(v[i % m]),
            im: Q15(v[(i / m) % m]),
        });
        b.push(CQ15 {
            re: Q15(v[(i / (m * m)) % m]),
            im: Q15(v[(i / (m * m * m)) % m]),
        });
    }
    (a, b)
}

#[test]
fn c32_sum_and_product_are_bit_exact() {
    let mut rng = Rng::new(470);
    let mut a = c32_noise(&mut rng, MAX_LEN + 4, 3.0);
    let mut b = c32_noise(&mut rng, MAX_LEN + 4, 3.0);
    for (i, v) in [
        f32::NAN,
        f32::INFINITY,
        -0.0,
        1e-40,
        f32::MAX,
        f32::NEG_INFINITY,
    ]
    .into_iter()
    .enumerate()
    {
        a[7 * i + 3].re = v;
        b[11 * i + 5].im = v;
        a[13 * i + 1].im = v;
        b[5 * i + 2].re = v;
    }
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        let (x, y) = (&a[off..off + n], &b[off..off + n]);
        let (mut want, mut got) = (vec![C32::default(); n], vec![C32::default(); n]);
        Fallback.add(x, y, &mut want).unwrap();
        Simd.add(x, y, &mut got).unwrap();
        assert!(same_c32(&want, &got), "add n={n} offset={off}");
        Fallback.mul(x, y, &mut want).unwrap();
        Simd.mul(x, y, &mut got).unwrap();
        assert!(same_c32(&want, &got), "mul n={n} offset={off}");
    }
}

#[test]
fn f32_sum_and_product_are_bit_exact() {
    let mut rng = Rng::new(471);
    let mut a: Vec<f32> = c32_noise(&mut rng, MAX_LEN + 4, 3.0)
        .iter()
        .map(|v| v.re)
        .collect();
    let b: Vec<f32> = c32_noise(&mut rng, MAX_LEN + 4, 3.0)
        .iter()
        .map(|v| v.im)
        .collect();
    a[5] = f32::NAN;
    a[17] = f32::INFINITY;
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        let (x, y) = (&a[off..off + n], &b[off..off + n]);
        let (mut want, mut got) = (vec![0.0f32; n], vec![0.0f32; n]);
        Fallback.add_real(x, y, &mut want).unwrap();
        Simd.add_real(x, y, &mut got).unwrap();
        assert!(
            want.iter().zip(&got).all(|(p, q)| same_f32(*p, *q)),
            "add_real n={n}"
        );
        Fallback.mul_real(x, y, &mut want).unwrap();
        Simd.mul_real(x, y, &mut got).unwrap();
        assert!(
            want.iter().zip(&got).all(|(p, q)| same_f32(*p, *q)),
            "mul_real n={n}"
        );
    }
}

#[test]
fn q15_sums_and_products_are_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(472);
    let a = cq15_noise(&mut rng, MAX_LEN + 4);
    let b = cq15_noise(&mut rng, MAX_LEN + 4);
    let ra = q15_noise(&mut rng, MAX_LEN + 4);
    let rb = q15_noise(&mut rng, MAX_LEN + 4);
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        let (x, y) = (&a[off..off + n], &b[off..off + n]);
        let (mut want, mut got) = (vec![CQ15::default(); n], vec![CQ15::default(); n]);
        Fallback.add(x, y, &mut want).unwrap();
        Simd.add(x, y, &mut got).unwrap();
        assert_eq!(want, got, "add n={n} offset={off}");
        Fallback.mul(x, y, &mut want).unwrap();
        Simd.mul(x, y, &mut got).unwrap();
        assert_eq!(want, got, "mul n={n} offset={off}");
        let (x, y) = (&ra[off..off + n], &rb[off..off + n]);
        let (mut want, mut got) = (vec![Q15(0); n], vec![Q15(0); n]);
        Fallback.add_real(x, y, &mut want).unwrap();
        Simd.add_real(x, y, &mut got).unwrap();
        assert_eq!(want, got, "add_real n={n} offset={off}");
        Fallback.mul_real(x, y, &mut want).unwrap();
        Simd.mul_real(x, y, &mut got).unwrap();
        assert_eq!(want, got, "mul_real n={n} offset={off}");
    }
}

#[test]
fn q15_extreme_values_are_bit_exact() {
    // All 9^4 combinations of the four components, including the sums that reach 2^31.
    let (a, b) = extreme_pairs();
    let (mut want, mut got) = (
        vec![CQ15::default(); a.len()],
        vec![CQ15::default(); a.len()],
    );
    Fallback.mul(&a, &b, &mut want).unwrap();
    Simd.mul(&a, &b, &mut got).unwrap();
    assert_eq!(want, got);
    Fallback.add(&a, &b, &mut want).unwrap();
    Simd.add(&a, &b, &mut got).unwrap();
    assert_eq!(want, got);
    let ra: Vec<Q15> = a.iter().map(|v| v.re).collect();
    let rb: Vec<Q15> = b.iter().map(|v| v.im).collect();
    let (mut want, mut got) = (vec![Q15(0); ra.len()], vec![Q15(0); ra.len()]);
    Fallback.mul_real(&ra, &rb, &mut want).unwrap();
    Simd.mul_real(&ra, &rb, &mut got).unwrap();
    assert_eq!(want, got);
    // (-1) * (-1) = 32767 in every path.
    let m = [CQ15 {
        re: Q15::MIN,
        im: Q15::MIN,
    }; 40];
    let mut out = [CQ15::default(); 40];
    Simd.mul(&m, &m, &mut out).unwrap();
    assert!(out.iter().all(|v| v.re == Q15(0) && v.im == Q15::MAX));
}

#[test]
fn interleave_and_deinterleave_are_exact() {
    let mut rng = Rng::new(473);
    let x = c32_noise(&mut rng, MAX_LEN + 4, 1.0);
    let q = cq15_noise(&mut rng, MAX_LEN + 4);
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        let s = &x[off..off + n];
        let (mut re, mut im) = (vec![0.0f32; n], vec![0.0f32; n]);
        Simd.deinterleave(s, Some(&mut re), Some(&mut im)).unwrap();
        assert!(
            s.iter()
                .zip(&re)
                .zip(&im)
                .all(|((c, r), i)| c.re == *r && c.im == *i),
            "n={n}"
        );
        let mut back = vec![C32::default(); n];
        Simd.interleave(&re, Some(&im), &mut back).unwrap();
        assert_eq!(back, s, "n={n}");
        Simd.interleave(&re, None, &mut back).unwrap();
        assert!(
            back.iter().zip(&re).all(|(c, r)| c.re == *r && c.im == 0.0),
            "zero imaginary n={n}"
        );
        let (mut only_re, mut only_im) = (vec![9.0f32; n], vec![9.0f32; n]);
        Simd.deinterleave(s, Some(&mut only_re), None).unwrap();
        Simd.deinterleave(s, None, Some(&mut only_im)).unwrap();
        assert_eq!(only_re, re);
        assert_eq!(only_im, im);

        let s = &q[off..off + n];
        let (mut re, mut im) = (vec![Q15(0); n], vec![Q15(0); n]);
        Simd.deinterleave(s, Some(&mut re), Some(&mut im)).unwrap();
        assert!(
            s.iter()
                .zip(&re)
                .zip(&im)
                .all(|((c, r), i)| c.re == *r && c.im == *i),
            "q15 n={n}"
        );
        let mut back = vec![CQ15::default(); n];
        Simd.interleave(&re, Some(&im), &mut back).unwrap();
        assert_eq!(back, s, "q15 n={n}");
        Simd.interleave(&re, None, &mut back).unwrap();
        assert!(
            back.iter()
                .zip(&re)
                .all(|(c, r)| c.re == *r && c.im == Q15(0)),
            "q15 zero n={n}"
        );
    }
}

#[test]
fn unequal_lengths_are_errors_and_change_nothing() {
    let (a, b) = ([C32::default(); 3], [C32::default(); 2]);
    let mut out = [C32::new(7.0, 7.0); 3];
    let err = Err(Error::LengthMismatch {
        expected: 3,
        got: 2,
    });
    assert_eq!(Simd.add(&a, &b, &mut out), err);
    assert_eq!(Simd.mul(&a, &b, &mut out), err);
    assert_eq!(out, [C32::new(7.0, 7.0); 3]);
    let mut short = [C32::new(7.0, 7.0); 2];
    assert_eq!(Simd.mul(&a, &a, &mut short), err);
    assert_eq!(short, [C32::new(7.0, 7.0); 2]);
    assert_eq!(Simd.interleave(&[0.0; 3], Some(&[0.0; 2]), &mut out), err);
    assert_eq!(Simd.deinterleave(&a, Some(&mut [0.0; 2]), None), err);
}

mod oscillate {
    use sdrk_backend_simd::Simd;
    use sdrk_core::kernels::mix::{Nco, Oscillate};
    use sdrk_core::{C32, CQ15, Error, Fallback};

    const INCS: [u32; 6] = [
        0x0123_4567,
        0xFEDC_BA98,
        0,
        0x4000_0000,
        0x0000_0300,
        0xFFFF_FFFF,
    ];
    const PHASES: [u32; 3] = [0, 0x8765_4321, 0xFFFF_FF00];

    #[test]
    fn c32_and_cq15_are_bit_exact_with_the_fallback_for_every_harmonic_length_and_phase() {
        for harmonic in 1..=8u32 {
            for (i, &inc) in INCS.iter().enumerate() {
                let phase = PHASES[i % PHASES.len()];
                for n in [0usize, 1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 33, 100, 257, 1000] {
                    let (mut a, mut b) = (Nco::from_increment(inc), Nco::from_increment(inc));
                    a.set_phase(phase);
                    b.set_phase(phase);
                    let (mut want, mut got) = (vec![C32::default(); n], vec![C32::default(); n]);
                    Fallback.oscillate(&mut a, harmonic, &mut want).unwrap();
                    Simd.oscillate(&mut b, harmonic, &mut got).unwrap();
                    assert_eq!(want, got, "c32 k={harmonic} inc={inc:#x} n={n}");
                    assert_eq!((a.phase(), a.increment()), (b.phase(), b.increment()));
                    let (mut a, mut b) = (Nco::from_increment(inc), Nco::from_increment(inc));
                    a.set_phase(phase);
                    b.set_phase(phase);
                    let (mut want, mut got) = (vec![CQ15::default(); n], vec![CQ15::default(); n]);
                    Fallback.oscillate(&mut a, harmonic, &mut want).unwrap();
                    Simd.oscillate(&mut b, harmonic, &mut got).unwrap();
                    assert_eq!(want, got, "cq15 k={harmonic} inc={inc:#x} n={n}");
                    assert_eq!(a.phase(), b.phase());
                }
            }
        }
    }

    /// Two turns of the phase in steps of `2^16 + 7`: every table index of the cosine and the
    /// sine with about 16 fractions each, so every lookup of a level path meets every pair of
    /// neighbouring table entries.
    #[test]
    fn cq15_is_bit_exact_over_the_whole_table() {
        let n = 2 * 65_536;
        let (mut a, mut b) = (
            Nco::from_increment(0x0001_0007),
            Nco::from_increment(0x0001_0007),
        );
        a.set_phase(0x0000_1234);
        b.set_phase(0x0000_1234);
        let (mut want, mut got) = (vec![CQ15::default(); n], vec![CQ15::default(); n]);
        Fallback.oscillate(&mut a, 1, &mut want).unwrap();
        Simd.oscillate(&mut b, 1, &mut got).unwrap();
        assert!(want == got, "cq15 over the whole table");
        assert_eq!(a.phase(), b.phase());
    }

    #[test]
    fn chunked_calls_equal_one_call() {
        let inc = 0x0123_4567;
        let mut whole = vec![C32::default(); 700];
        let mut n = Nco::from_increment(inc);
        Simd.oscillate(&mut n, 3, &mut whole).unwrap();
        let mut parts = vec![C32::default(); 700];
        let mut m = Nco::from_increment(inc);
        let mut at = 0;
        for len in [1, 7, 64, 129, 3, 250, 246] {
            Simd.oscillate(&mut m, 3, &mut parts[at..at + len]).unwrap();
            at += len;
        }
        assert_eq!((whole, n.phase()), (parts, m.phase()));
    }

    #[test]
    fn a_bad_harmonic_is_an_error_and_changes_nothing() {
        for k in [0u32, 9, u32::MAX] {
            let mut nco = Nco::from_increment(5);
            let mut out = [C32::new(3.0, 3.0); 4];
            assert!(matches!(
                Simd.oscillate(&mut nco, k, &mut out),
                Err(Error::InvalidParameter(_))
            ));
            assert_eq!((nco.phase(), out), (0, [C32::new(3.0, 3.0); 4]));
        }
    }
}
