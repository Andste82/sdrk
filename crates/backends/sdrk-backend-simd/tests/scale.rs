//! `scale` `CQ15` against the fallback: bit-exact at every length from 0 to 1100, at every start
//! offset within a vector, for every gain of interest (the extremes, `-32768` with its own path
//! on x86, and random ones) and for the extreme samples. `C32` runs the fallback.
#![allow(clippy::indexing_slicing, clippy::unwrap_used)]

mod common;

use common::{Rng, cq15_noise};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::basic::Scale;
use sdrk_core::{CQ15, Error, Fallback, Q15};

const MAX_LEN: usize = 1100;
const OFFSETS: [usize; 4] = [0, 1, 2, 3];

fn check(x: &[CQ15], gain: Q15, what: &str) {
    let (mut want, mut got) = (
        vec![CQ15::default(); x.len()],
        vec![CQ15::default(); x.len()],
    );
    Fallback.scale(x, &mut want, gain).unwrap();
    Simd.scale(x, &mut got, gain).unwrap();
    assert_eq!(want, got, "{what} gain {}", gain.0);
}

#[test]
fn cq15_is_bit_exact_at_every_length_offset_and_gain() {
    let mut rng = Rng::new(180);
    let x = cq15_noise(&mut rng, MAX_LEN + OFFSETS.len());
    let gains = [
        i16::MIN,
        i16::MIN + 1,
        -16384,
        -1,
        0,
        1,
        16384,
        24_065,
        i16::MAX,
    ];
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        let g = gains[n % gains.len()];
        check(&x[off..off + n], Q15(g), &format!("n={n} offset={off}"));
    }
    for _ in 0..200 {
        let n = rng.below(300);
        check(&x[..n], Q15(rng.next_u32() as i16), &format!("n={n}"));
    }
}

#[test]
fn extreme_samples_and_gains_are_bit_exact() {
    let v = [
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
    let x: Vec<CQ15> = (0..3 * v.len() * v.len())
        .map(|i| CQ15 {
            re: Q15(v[i % v.len()]),
            im: Q15(v[(i / v.len()) % v.len()]),
        })
        .collect();
    for &g in &v {
        check(&x, Q15(g), "extremes");
    }
    // -1 * -1 saturates to 32767 in every path.
    let mut out = vec![CQ15::default(); x.len()];
    Simd.scale(&x, &mut out, Q15(i16::MIN)).unwrap();
    assert!(
        out.iter()
            .zip(&x)
            .all(|(o, s)| s.re.0 != i16::MIN || o.re.0 == i16::MAX)
    );
}

#[test]
fn unequal_lengths_are_errors() {
    let mut out = [CQ15::default(); 2];
    assert_eq!(
        Simd.scale(&[CQ15::default(); 3], &mut out, Q15(1)),
        Err(Error::LengthMismatch {
            expected: 3,
            got: 2
        })
    );
}

/// `C32`: one multiplication per component, the same bits as the fallback, specials included.
#[test]
fn c32_is_bit_exact_at_every_length_and_offset() {
    use sdrk_core::C32;
    let mut rng = Rng::new(181);
    let mut x = common::c32_noise(&mut rng, MAX_LEN + OFFSETS.len(), 3.0);
    for (i, v) in [f32::NAN, f32::INFINITY, -0.0, 1e-40, f32::MAX]
        .into_iter()
        .enumerate()
    {
        x[7 * i + 3].re = v;
        x[11 * i + 5].im = v;
    }
    let gains = [0.734_375, -1.5, 0.0, 1e30, f32::NAN];
    let same = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        let g = gains[n % gains.len()];
        let x = &x[off..off + n];
        let (mut want, mut got) = (vec![C32::default(); n], vec![C32::default(); n]);
        Fallback.scale(x, &mut want, g).unwrap();
        Simd.scale(x, &mut got, g).unwrap();
        assert!(
            want.iter()
                .zip(&got)
                .all(|(w, o)| same(w.re, o.re) && same(w.im, o.im)),
            "n={n} offset={off} gain {g}"
        );
    }
}
