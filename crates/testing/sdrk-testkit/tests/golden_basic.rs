//! Cross-checks `sdrk_core::num` and `scale` against the committed golden vectors.
#![cfg(feature = "std")]
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic
)]

use sdrk::Sdrk;
use sdrk_core::kernels::basic::Scale;
use sdrk_core::num::{cq15_mul, q15_from_f32, q15_mul};
use sdrk_core::{C32, CQ15, Q15, Reference};
use sdrk_testkit::golden::{self, Array, GoldenCase, Tolerance};

fn q15s(case: &GoldenCase, name: &str) -> Vec<Q15> {
    case.array(name).unwrap().as_q15().expect("i16 array")
}

fn cq15s(case: &GoldenCase, name: &str) -> Vec<CQ15> {
    match case.array(name).unwrap() {
        Array::CI16(v) => v.clone(),
        other => panic!("expected ci16, got {}", other.dtype()),
    }
}

#[test]
fn every_case_is_listed() {
    assert_eq!(
        golden::list("q15_ops").unwrap(),
        ["cmul_pairs_4096", "from_float_4096", "mul_pairs_4096"]
    );
    assert_eq!(
        golden::list("scale").unwrap(),
        ["c32_random_4096", "cq15_random_4096"]
    );
}

#[test]
fn q15_from_float() {
    let case = golden::load("q15_ops", "from_float_4096").unwrap();
    assert_eq!(case.tolerance, Tolerance::BitExact);
    let Array::F32(input) = case.array("input").unwrap() else {
        panic!("input must be f32");
    };
    let out: Vec<Q15> = input.iter().map(|&x| q15_from_f32(x)).collect();
    let c = golden::assert_matches(&case, case.array("expected").unwrap(), &out).unwrap();
    assert_eq!(c.first_mismatch, None);
    assert_eq!(c.compared, 4096);
}

#[test]
fn q15_multiply() {
    let case = golden::load("q15_ops", "mul_pairs_4096").unwrap();
    let (a, b) = (q15s(&case, "a"), q15s(&case, "b"));
    let out: Vec<Q15> = a.iter().zip(&b).map(|(&a, &b)| q15_mul(a, b)).collect();
    golden::assert_matches(&case, case.array("expected").unwrap(), &out).unwrap();
}

#[test]
fn complex_q15_multiply() {
    let case = golden::load("q15_ops", "cmul_pairs_4096").unwrap();
    let (a, b) = (cq15s(&case, "a"), cq15s(&case, "b"));
    let out: Vec<CQ15> = a.iter().zip(&b).map(|(&a, &b)| cq15_mul(a, b)).collect();
    golden::assert_matches(&case, case.array("expected").unwrap(), &out).unwrap();
}

#[test]
fn scale_c32() {
    let case = golden::load("scale", "c32_random_4096").unwrap();
    assert!(matches!(case.tolerance, Tolerance::Snr { min_db, .. } if min_db >= 140.0));
    let Array::CF32(input) = case.array("input").unwrap() else {
        panic!("input must be cf32");
    };
    let gain = case.param_f64("gain").unwrap() as f32;
    let expected = case.array("expected").unwrap();

    let mut out = vec![C32::new(0.0, 0.0); input.len()];
    Sdrk.scale(input, &mut out, gain).unwrap();
    golden::assert_matches(&case, expected, &out).unwrap();

    let mut reference = vec![C32::new(0.0, 0.0); input.len()];
    Reference.scale(input, &mut reference, gain).unwrap();
    golden::assert_matches(&case, expected, &reference).unwrap();
}

#[test]
fn scale_cq15() {
    let case = golden::load("scale", "cq15_random_4096").unwrap();
    let input = cq15s(&case, "input");
    let gain = Q15(i16::try_from(case.param_u64("gain_raw").unwrap()).unwrap());
    let expected = case.array("expected").unwrap();

    let mut out = vec![CQ15::default(); input.len()];
    Sdrk.scale(&input, &mut out, gain).unwrap();
    golden::assert_matches(&case, expected, &out).unwrap();

    let mut reference = vec![CQ15::default(); input.len()];
    Reference.scale(&input, &mut reference, gain).unwrap();
    golden::assert_matches(&case, expected, &reference).unwrap();
}

fn assert_send<T: Send>() {}

#[test]
fn public_types_are_send() {
    assert_send::<GoldenCase>();
    assert_send::<Array>();
    assert_send::<Tolerance>();
    assert_send::<golden::GoldenError>();
}
