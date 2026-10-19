//! Tests of the `dot` and `dot_real` kernels of `sdrk-core` (M09).

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::unreadable_literal
)]

use sdrk_core::kernels::dot::{Dot, DotReal};
use sdrk_core::num::{round_shift, sat16};
use sdrk_core::{C32, CQ15, Error, Fallback, Q15};
use sdrk_device::{ErrorKind, SdrkError};

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn unit(state: &mut u64) -> f32 {
    (lcg(state) >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
}

#[test]
fn c32_dot_is_the_sum_of_products_in_ascending_order() {
    let x = [C32::new(1.0, 2.0), C32::new(3.0, -1.0), C32::new(-2.0, 0.5)];
    let taps = [0.5f32, 2.0, -1.0];
    let y = Fallback.dot(&x, &taps).unwrap();
    assert_eq!(y, C32::new(0.5 + 6.0 + 2.0, 1.0 - 2.0 - 0.5));
    // The order is ascending without fused multiply-add: reproduce it by hand.
    let mut s = 7u64;
    let x: Vec<C32> = (0..1023)
        .map(|_| C32::new(unit(&mut s), unit(&mut s)))
        .collect();
    let h: Vec<f32> = (0..1023).map(|_| unit(&mut s)).collect();
    let (mut re, mut im) = (0.0f32, 0.0f32);
    for (v, t) in x.iter().zip(&h) {
        re += v.re * t;
        im += v.im * t;
    }
    let got = Fallback.dot(&x, &h).unwrap();
    assert_eq!(
        (got.re.to_bits(), got.im.to_bits()),
        (re.to_bits(), im.to_bits())
    );
}

#[test]
fn real_dot_matches_the_complex_dot_of_a_real_signal() {
    let mut s = 9u64;
    let x: Vec<f32> = (0..300).map(|_| unit(&mut s)).collect();
    let cx: Vec<C32> = x.iter().map(|&v| C32::new(v, 0.0)).collect();
    let h: Vec<f32> = (0..300).map(|_| unit(&mut s)).collect();
    let a = Fallback.dot_real(&x, &h).unwrap();
    let b = Fallback.dot(&cx, &h).unwrap();
    assert_eq!(a.to_bits(), b.re.to_bits());
    assert_eq!(b.im, 0.0);
}

#[test]
fn q15_dot_rounds_and_saturates_once_at_the_end() {
    // 0.5 * 0.5 + 0.5 * -0.25 = 0.125.
    let x = [Q15(16384), Q15(16384)];
    assert_eq!(
        Fallback.dot_real(&x, &[Q15(16384), Q15(-8192)]).unwrap(),
        Q15(4096)
    );
    // Single rounding: the sum of 1 LSB products below half an LSB each does not vanish.
    let tiny = [Q15(1); 64];
    // Each product is 256 (far below half an LSB of 2^14); the sum 16384 rounds half up to 1.
    assert_eq!(Fallback.dot_real(&tiny, &[Q15(256); 64]).unwrap(), Q15(1));
    // Saturation at the end only: +full scale and -full scale products cancel exactly in the
    // accumulator instead of clipping in between.
    let x = [Q15(-32768); 4];
    let h = [Q15(-32768), Q15(-32768), Q15(32767), Q15(32767)];
    // 2 * 2^30 - 2 * 32768 * 32767 = 65536 = 2.0 in Q30 / 2^15; clipping after two terms
    // would have given a large negative number.
    assert_eq!(Fallback.dot_real(&x, &h).unwrap(), Q15(2));
    // A sum beyond the range saturates.
    assert_eq!(
        Fallback
            .dot_real(&[Q15(-32768); 2], &[Q15(-32768); 2])
            .unwrap(),
        Q15::MAX
    );
    assert_eq!(
        Fallback
            .dot_real(&[Q15(32767); 2], &[Q15(-32768); 2])
            .unwrap(),
        Q15::MIN
    );
    // The complex form does the same per component.
    let cx = [CQ15 {
        re: Q15(-32768),
        im: Q15(32767),
    }; 2];
    let y = Fallback.dot(&cx, &[Q15(-32768); 2]).unwrap();
    assert_eq!((y.re, y.im), (Q15::MAX, Q15::MIN));
}

#[test]
fn cq15_dot_matches_the_integer_model() {
    let mut s = 11u64;
    for n in [1usize, 2, 7, 64, 1023] {
        let x: Vec<CQ15> = (0..n)
            .map(|_| CQ15 {
                re: Q15(lcg(&mut s) as i16),
                im: Q15(lcg(&mut s) as i16),
            })
            .collect();
        let h: Vec<Q15> = (0..n).map(|_| Q15(lcg(&mut s) as i16)).collect();
        let y = Fallback.dot(&x, &h).unwrap();
        let re: i64 = x
            .iter()
            .zip(&h)
            .map(|(a, b)| i64::from(a.re.0) * i64::from(b.0))
            .sum();
        let im: i64 = x
            .iter()
            .zip(&h)
            .map(|(a, b)| i64::from(a.im.0) * i64::from(b.0))
            .sum();
        assert_eq!(y.re.0, sat16(round_shift(re, 15)), "n = {n}");
        assert_eq!(y.im.0, sat16(round_shift(im, 15)), "n = {n}");
    }
}

#[test]
fn empty_inputs_give_zero_and_unequal_lengths_are_errors() {
    assert_eq!(
        Fallback.dot(&[] as &[C32], &[]).unwrap(),
        C32::new(0.0, 0.0)
    );
    assert_eq!(Fallback.dot(&[] as &[CQ15], &[]).unwrap(), CQ15::default());
    assert_eq!(Fallback.dot_real(&[] as &[f32], &[]).unwrap(), 0.0);
    assert_eq!(Fallback.dot_real(&[] as &[Q15], &[]).unwrap(), Q15(0));
    let e = Fallback
        .dot(&[C32::new(1.0, 0.0); 3], &[1.0f32; 2])
        .unwrap_err();
    assert_eq!(
        e,
        Error::LengthMismatch {
            expected: 3,
            got: 2
        }
    );
    assert_eq!(e.kind(), ErrorKind::LengthMismatch);
    assert!(Fallback.dot(&[CQ15::default(); 2], &[Q15(0); 3]).is_err());
    assert!(Fallback.dot_real(&[0.0f32; 4], &[0.0f32; 1]).is_err());
    assert!(Fallback.dot_real(&[Q15(0); 1], &[Q15(0); 4]).is_err());
}

#[test]
fn f32_extremes_propagate_like_ieee() {
    let nan = Fallback.dot_real(&[f32::NAN, 1.0], &[1.0, 1.0]).unwrap();
    assert!(nan.is_nan());
    assert_eq!(
        Fallback
            .dot_real(&[f32::INFINITY, 1.0], &[2.0, 1.0])
            .unwrap(),
        f32::INFINITY
    );
    assert!(
        Fallback
            .dot_real(&[f32::INFINITY, 1.0], &[0.0, 1.0])
            .unwrap()
            .is_nan()
    );
    assert_eq!(
        Fallback
            .dot_real(&[f32::INFINITY], &[f32::NEG_INFINITY])
            .unwrap(),
        f32::NEG_INFINITY
    );
    let y = Fallback
        .dot(
            &[C32::new(f32::NAN, 1.0), C32::new(2.0, f32::INFINITY)],
            &[1.0, 1.0],
        )
        .unwrap();
    assert!(y.re.is_nan() && y.im == f32::INFINITY);
    // Silence.
    assert_eq!(
        Fallback.dot(&[C32::new(0.0, 0.0); 9], &[0.7; 9]).unwrap(),
        C32::new(0.0, 0.0)
    );
}

#[test]
fn the_dot_kernels_are_exposed_by_the_facade_types() {
    fn assert_send<T: Send>() {}
    assert_send::<Fallback>();
}

#[cfg(feature = "reference")]
#[test]
fn the_fallback_agrees_with_the_reference() {
    use sdrk_core::Reference;
    let mut s = 3u64;
    for n in [1usize, 7, 64, 1023] {
        let x: Vec<C32> = (0..n)
            .map(|_| C32::new(unit(&mut s) * 0.3, unit(&mut s) * 0.3))
            .collect();
        let h: Vec<f32> = (0..n).map(|_| unit(&mut s) * 0.5).collect();
        let a = Fallback.dot(&x, &h).unwrap();
        let b = Reference.dot(&x, &h).unwrap();
        let err = f64::from((a.re - b.re).abs()).hypot(f64::from((a.im - b.im).abs()));
        let mag = f64::from(b.re).hypot(f64::from(b.im)).max(1e-6);
        assert!(err / mag < 1e-5, "n = {n}: {err} / {mag}");
        let xr: Vec<f32> = x.iter().map(|v| v.re).collect();
        let a = Fallback.dot_real(&xr, &h).unwrap();
        let b = Reference.dot_real(&xr, &h).unwrap();
        assert!(
            f64::from((a - b).abs()) < 1e-4 * f64::from(b.abs()).max(1e-3),
            "n = {n}"
        );

        let xq: Vec<CQ15> = (0..n)
            .map(|_| CQ15 {
                re: Q15(lcg(&mut s) as i16),
                im: Q15(lcg(&mut s) as i16),
            })
            .collect();
        let hq: Vec<Q15> = (0..n).map(|_| Q15(lcg(&mut s) as i16)).collect();
        assert_eq!(
            Fallback.dot(&xq, &hq).unwrap(),
            Reference.dot(&xq, &hq).unwrap(),
            "n = {n}"
        );
        let xr: Vec<Q15> = xq.iter().map(|v| v.re).collect();
        assert_eq!(
            Fallback.dot_real(&xr, &hq).unwrap(),
            Reference.dot_real(&xr, &hq).unwrap(),
            "n = {n}"
        );
    }
    // Full-scale data saturates identically.
    let full = [CQ15 {
        re: Q15(-32768),
        im: Q15(32767),
    }; 40];
    let taps = [Q15(-32768); 40];
    assert_eq!(
        Fallback.dot(&full, &taps).unwrap(),
        Reference.dot(&full, &taps).unwrap()
    );
    assert!(Reference.dot(&[C32::new(0.0, 0.0)], &[]).is_err());
    assert!(Reference.dot_real(&[Q15(0)], &[]).is_err());
}
