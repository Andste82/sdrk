//! Tests of the M35 kernel `am_mod` of `sdrk-core`: fallback against reference (bit for bit),
//! the clamp and NaN rule, depth 0 and 1, the `CQ15` corners, length and parameter errors.

#![cfg(feature = "reference")]
#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::indexing_slicing,
    clippy::manual_midpoint
)]

use sdrk_core::kernels::modulate::AmMod;
use sdrk_core::{C32, CQ15, Error, Fallback, Q15, Reference};
use sdrk_device::{ErrorKind, SdrkError};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[-1.5, 1.5)`, so a sixth of the values lies beyond the clamp.
    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / 16_777_216.0) * 3.0 - 1.5
    }
}

/// The interesting `f32` inputs: the clamp limits and beyond, zeros, the non-finite values,
/// the smallest normal and denormal values.
const F32_CORNERS: [f32; 16] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    1.000_000_1,
    -1.000_000_1,
    0.999_999_94,
    -0.999_999_94,
    2.0,
    -1.0e30,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    -f32::NAN,
    f32::MIN_POSITIVE,
    -1.0e-45,
];

/// Carrier and depth values of interest, all valid.
const GAINS: [f32; 9] = [
    0.0,
    -0.0,
    1.0e-6,
    0.25,
    0.5,
    0.8,
    0.999_984_7,
    0.999_99,
    1.0,
];

fn bits(x: &[C32]) -> Vec<(u32, u32)> {
    x.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}

fn c32_input() -> Vec<f32> {
    let mut rng = Rng(0x0035_a3d1_77c0_ffee);
    let mut x: Vec<f32> = (0..1000).map(|_| rng.f32()).collect();
    x.extend_from_slice(&F32_CORNERS);
    x
}

fn q15_input() -> Vec<Q15> {
    // Every raw value once.
    (i16::MIN..=i16::MAX).map(Q15).collect()
}

#[test]
fn c32_fallback_equals_reference_bit_for_bit() {
    let x = c32_input();
    for &c in &GAINS {
        for &m in &GAINS {
            let mut want = vec![C32::default(); x.len()];
            let mut got = vec![C32::new(7.0, 7.0); x.len()];
            Reference.am_mod(c, m, &x, &mut want).unwrap();
            Fallback.am_mod(c, m, &x, &mut got).unwrap();
            assert_eq!(bits(&want), bits(&got), "carrier {c} depth {m}");
        }
    }
}

#[test]
fn cq15_fallback_equals_reference_for_every_input() {
    let x = q15_input();
    for &c in &GAINS {
        for &m in &GAINS {
            let mut want = vec![CQ15::default(); x.len()];
            let mut got = vec![CQ15::default(); x.len()];
            Reference.am_mod(c, m, &x, &mut want).unwrap();
            Fallback.am_mod(c, m, &x, &mut got).unwrap();
            assert_eq!(want, got, "carrier {c} depth {m}");
        }
    }
}

#[test]
fn c32_clamps_and_maps_nan_to_zero() {
    let x = [2.0, -3.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.5];
    let mut out = [C32::default(); 6];
    Fallback.am_mod(0.5, 0.8, &x, &mut out).unwrap();
    let at = |v: f32| 0.5 * (1.0 + 0.8 * v);
    let want = [at(1.0), at(-1.0), at(1.0), at(-1.0), 0.5, at(0.5)];
    for (o, w) in out.iter().zip(want) {
        assert_eq!(o.re.to_bits(), w.to_bits());
        assert_eq!(o.im.to_bits(), 0);
    }
}

#[test]
fn depth_zero_gives_the_constant_carrier() {
    let x = c32_input();
    let mut out = vec![C32::default(); x.len()];
    Fallback.am_mod(0.5, 0.0, &x, &mut out).unwrap();
    assert!(out.iter().all(|o| o.re == 0.5 && o.im.to_bits() == 0));

    let x = q15_input();
    let mut out = vec![CQ15::default(); x.len()];
    Fallback.am_mod(0.5, 0.0, &x, &mut out).unwrap();
    assert!(out.iter().all(|o| *o
        == CQ15 {
            re: Q15(16384),
            im: Q15(0)
        }));
}

#[test]
fn depth_one_with_a_full_scale_input_touches_zero() {
    let x = [-1.0, 1.0, 0.0];
    let mut out = [C32::default(); 3];
    Fallback.am_mod(0.5, 1.0, &x, &mut out).unwrap();
    assert_eq!(out.map(|o| o.re), [0.0, 1.0, 0.5]);

    // Q15: m_q = 32767 < 32768, so -32768 leaves u = 1 and the envelope one LSB above 0.
    let mut out = [CQ15::default(); 3];
    Fallback
        .am_mod(0.5, 1.0, &[Q15::MIN, Q15::MAX, Q15(0)], &mut out)
        .unwrap();
    assert_eq!(out.map(|o| o.re.0), [1, 32767, 16384]);
}

#[test]
fn cq15_corners_with_carrier_and_depth_one() {
    let x = [Q15::MIN, Q15(-32767), Q15(-1), Q15(0), Q15(1), Q15::MAX];
    let mut out = [CQ15::default(); 6];
    Fallback.am_mod(1.0, 1.0, &x, &mut out).unwrap();
    // c_q = m_q = 32767: u = 1, 2, 32767, 32768, 32769, 65534; the upper half saturates.
    assert_eq!(out.map(|o| o.re.0), [1, 2, 32766, 32767, 32767, 32767]);
    assert!(out.iter().all(|o| o.im.0 == 0));
    let mut want = [CQ15::default(); 6];
    Reference.am_mod(1.0, 1.0, &x, &mut want).unwrap();
    assert_eq!(out, want);
}

#[test]
fn unequal_lengths_are_rejected_and_nothing_is_written() {
    let mut out = [C32::new(3.0, 3.0); 4];
    let err = Fallback.am_mod(0.5, 0.5, &[0.0; 3], &mut out).unwrap_err();
    assert_eq!(
        err,
        Error::LengthMismatch {
            expected: 3,
            got: 4
        }
    );
    assert_eq!(err.kind(), ErrorKind::LengthMismatch);
    assert!(out.iter().all(|o| *o == C32::new(3.0, 3.0)));
    let err = Reference.am_mod(0.5, 0.5, &[0.0; 3], &mut out).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::LengthMismatch);

    let mut out = [CQ15::default(); 2];
    let err = Fallback
        .am_mod(0.5, 0.5, &[Q15(0); 3], &mut out)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::LengthMismatch);
    let err = Reference
        .am_mod(0.5, 0.5, &[Q15(0); 3], &mut out)
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::LengthMismatch);

    // Empty calls are fine.
    Fallback
        .am_mod(0.5, 0.5, &[], &mut [] as &mut [C32])
        .unwrap();
    Fallback
        .am_mod(0.5, 0.5, &[], &mut [] as &mut [CQ15])
        .unwrap();
}

#[test]
fn parameters_outside_zero_to_one_are_rejected_first() {
    let bad = [
        -1.0e-6,
        1.000_000_1,
        -1.0,
        2.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    for &v in &bad {
        for (c, m) in [(v, 0.5), (0.5, v)] {
            let mut out = [C32::new(3.0, 3.0); 2];
            // The parameters are checked before the lengths.
            let err = Fallback.am_mod(c, m, &[0.0; 1], &mut out).unwrap_err();
            assert!(matches!(err, Error::InvalidParameter(_)), "{c} {m}");
            assert_eq!(err.kind(), ErrorKind::InvalidArgument);
            assert!(out.iter().all(|o| *o == C32::new(3.0, 3.0)));
            let err = Reference.am_mod(c, m, &[0.0; 2], &mut out).unwrap_err();
            assert!(matches!(err, Error::InvalidParameter(_)));

            let mut out = [CQ15 {
                re: Q15(5),
                im: Q15(5),
            }; 2];
            let err = Fallback.am_mod(c, m, &[Q15(0); 2], &mut out).unwrap_err();
            assert!(matches!(err, Error::InvalidParameter(_)));
            assert!(out.iter().all(|o| o.re.0 == 5 && o.im.0 == 5));
            let err = Reference.am_mod(c, m, &[Q15(0); 2], &mut out).unwrap_err();
            assert!(matches!(err, Error::InvalidParameter(_)));
        }
    }
    // The limits themselves (and -0) are valid.
    let mut out = [C32::default(); 1];
    for (c, m) in [(0.0, 0.0), (1.0, 1.0), (-0.0, -0.0)] {
        Fallback.am_mod(c, m, &[0.5], &mut out).unwrap();
    }
}
