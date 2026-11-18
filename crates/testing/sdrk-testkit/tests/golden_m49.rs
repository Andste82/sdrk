//! Golden tests of the M49 kernel `scramble` (test 1) through the facade `Sdrk`, `Fallback` and
//! `Reference`.
//!
//! Every case of `tests/golden/vectors/scramble` is run in one call and in 20 chunk plans
//! (seed 695, the phase bytes sliced with the input); every run must match the golden file
//! under the tolerance of the case. The `bit_exact` cases (`bits` 1 and 2) compare raw bits, the
//! `tolerance` cases (`bits` 3 and 8) the SNR floor of the `rotate` contract (120 dB `C32`, 80 dB
//! `CQ15`).
#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use sdrk::Sdrk;
use sdrk_core::kernels::scramble::Scramble;
use sdrk_core::{C32, CQ15, ComplexSample, Fallback, Reference};
use sdrk_testkit::ChunkPlan;
use sdrk_testkit::golden::{self, Array, GoldenCase, Tolerance};

/// The seed of the chunk plans (M49 test 3); plan `k` uses `PLAN_SEED * 100 + k`.
const PLAN_SEED: u64 = 695;
/// Chunk plans per case and implementation.
const PLANS: u64 = 20;
/// Largest chunk of the plans.
const MAX_CHUNK: usize = 300;

const CASES: [&str; 7] = [
    "bpsk_c32_2048",
    "qpsk_c32_2048",
    "oct_c32_2048",
    "qpsk_inv_c32_2048",
    "qpsk_cq15_2048",
    "byte_cq15_2048",
    "byte_inv_c32_2048",
];

fn phase(case: &GoldenCase) -> Vec<u8> {
    match case.array("phase").unwrap() {
        Array::U8(v) => v.clone(),
        other => panic!("u8 expected, got {}", other.dtype()),
    }
}

fn params(case: &GoldenCase) -> (u32, bool) {
    let bits = u32::try_from(case.param_u64("bits").unwrap()).unwrap();
    let inverse = case.params["inverse"].as_bool().unwrap();
    (bits, inverse)
}

/// Runs `k` over the case whole and in the chunk plans and compares every run with `expected`.
fn run<T: ComplexSample, K: Scramble<T>>(k: &K, case: &GoldenCase, input: &[T]) {
    let p = phase(case);
    let (bits, inverse) = params(case);
    assert_eq!(p.len(), input.len());
    let want = case.array("expected").unwrap();
    let mut out = vec![T::ZERO; input.len()];
    k.scramble(input, &p, bits, inverse, &mut out).unwrap();
    golden::assert_matches(case, want, &out).unwrap();
    for plan in 0..PLANS {
        let mut sizes = ChunkPlan::new(PLAN_SEED * 100 + plan, MAX_CHUNK).sizes();
        let mut chunked = vec![T::ZERO; input.len()];
        let mut pos = 0;
        while pos < input.len() {
            let e = (pos + sizes.next().unwrap()).min(input.len());
            k.scramble(
                &input[pos..e],
                &p[pos..e],
                bits,
                inverse,
                &mut chunked[pos..e],
            )
            .unwrap();
            pos = e;
        }
        assert!(
            sdrk_testkit::bit_exact(&out, &chunked).is_none(),
            "{}: plan {plan} changes the output",
            case.case
        );
    }
}

fn all<T: ComplexSample>(case: &GoldenCase, input: &[T])
where
    Sdrk: Scramble<T>,
    Fallback: Scramble<T>,
    Reference: Scramble<T>,
{
    run(&Sdrk, case, input);
    run(&Fallback, case, input);
    run(&Reference, case, input);
}

#[test]
fn every_scramble_case_matches_through_the_facade_fallback_and_reference() {
    let mut listed = golden::list("scramble").unwrap();
    listed.sort();
    let mut want: Vec<String> = CASES.iter().map(|s| (*s).to_owned()).collect();
    want.sort();
    assert_eq!(listed, want);
    for name in CASES {
        let case = golden::load("scramble", name).unwrap();
        let (bits, _) = params(&case);
        // bits 1 and 2 are exact, bits 3 and 8 tolerance cases.
        assert_eq!(
            matches!(case.tolerance, Tolerance::BitExact),
            bits <= 2,
            "{name}: class"
        );
        match case.array("input").unwrap() {
            Array::CF32(x) => all::<C32>(&case, x),
            Array::CI16(x) => all::<CQ15>(&case, x),
            other => panic!("{name}: unexpected input {}", other.dtype()),
        }
    }
}

#[test]
fn the_cq15_case_holds_the_q15_corners_and_their_saturating_negation() {
    let case = golden::load("scramble", "qpsk_cq15_2048").unwrap();
    let Array::CI16(x) = case.array("input").unwrap() else {
        panic!("ci16 expected")
    };
    assert!(x.iter().any(|v| v.re.0 == i16::MIN) && x.iter().any(|v| v.im.0 == i16::MIN));
    assert!(x.iter().any(|v| v.re.0 == i16::MAX) && x.iter().any(|v| v.im.0 == i16::MAX));
    let Array::CI16(y) = case.array("expected").unwrap() else {
        panic!("ci16 expected")
    };
    // Some -32768 is negated to 32767 by the case.
    let p = phase(&case);
    assert!(
        x.iter()
            .zip(y)
            .zip(&p)
            .any(|((a, b), &k)| { k & 3 == 2 && a.re.0 == i16::MIN && b.re.0 == i16::MAX })
    );
}
