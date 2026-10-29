//! Golden tests of the M20 kernels `mag`, `dc_block_real` and `agc` through the facade
//! `Sdrk`.
//!
//! Every case is run in one piece and in chunks of random size (the plan seed is the case
//! seed); both runs must match the golden file and each other bit for bit. The exception is
//! `dc_block_real` `f32`, a tolerance kernel of the SIMD backend whose bits may depend on the
//! split (maintainer decision of 2026-10-29): there every split must reach the regression guard
//! of the case (the fallback-only builds of the facade check the fallback).
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::too_many_lines
)]

use sdrk::Sdrk;
use sdrk_core::kernels::agc::{Agc, AgcState};
use sdrk_core::kernels::demod::{DcBlockReal, DcBlockRealState};
use sdrk_core::kernels::spectrum::Mag;
use sdrk_core::{C32, CQ15, ComplexSample, Q15, Rate, Sample};
use sdrk_testkit::golden::{self, Array, GoldenCase};
use sdrk_testkit::{ChunkPlan, bit_exact};

/// Runs `step` over `input` in one piece and in random chunks, compares both with the
/// golden file and with each other. `new` makes a fresh state per run.
fn check<I: Sample, O: Sample, S>(
    case: &GoldenCase,
    input: &[I],
    want: &Array,
    new: impl Fn() -> S,
    step: impl Fn(&mut S, &[I], &mut [O]),
) {
    let mut whole = vec![O::ZERO; input.len()];
    let mut state = new();
    step(&mut state, input, &mut whole);

    let mut chunked = vec![O::ZERO; input.len()];
    let mut state = new();
    let mut sizes = ChunkPlan::new(case.seed, 700).sizes();
    let mut pos = 0;
    while pos < input.len() {
        let n = sizes.next().unwrap().min(input.len() - pos);
        step(&mut state, &input[pos..pos + n], &mut chunked[pos..pos + n]);
        pos += n;
    }

    golden::assert_matches(case, want, &whole).unwrap();
    assert_eq!(
        bit_exact(&whole, &chunked),
        None,
        "{}: block size changed the output",
        case.case
    );
    golden::assert_matches(case, want, &chunked).unwrap();
}

/// Runs `step` over `input` in one piece and split by chunk plans of several largest sizes
/// (down to single samples); every run must reach the tolerance of the golden case, its
/// regression guard included. The bits may differ between the runs.
fn check_guarded<I: Sample, O: Sample, S>(
    case: &GoldenCase,
    input: &[I],
    want: &Array,
    new: impl Fn() -> S,
    step: impl Fn(&mut S, &[I], &mut [O]),
) {
    let mut whole = vec![O::ZERO; input.len()];
    step(&mut new(), input, &mut whole);
    golden::assert_matches(case, want, &whole).unwrap();
    for max in [1, 7, 40, 700] {
        let mut chunked = vec![O::ZERO; input.len()];
        let mut state = new();
        let mut sizes = ChunkPlan::new(case.seed + max as u64, max).sizes();
        let mut pos = 0;
        while pos < input.len() {
            let n = sizes.next().unwrap().min(input.len() - pos);
            step(&mut state, &input[pos..pos + n], &mut chunked[pos..pos + n]);
            pos += n;
        }
        golden::assert_matches(case, want, &chunked)
            .unwrap_or_else(|e| panic!("{}: chunks of at most {max}: {e}", case.case));
    }
}

fn agc_state<T: ComplexSample>(case: &GoldenCase) -> AgcState<T> {
    let rate = Rate::hz(case.param_u64("rate_hz").unwrap()).unwrap();
    AgcState::new(
        case.param_f64("reference").unwrap(),
        case.param_f64("attack_s").unwrap(),
        case.param_f64("release_s").unwrap(),
        case.param_f64("min_gain_db").unwrap(),
        case.param_f64("max_gain_db").unwrap(),
        case.param_f64("initial_gain_db").unwrap(),
        rate,
    )
    .unwrap()
}

#[test]
fn all_cases_are_present() {
    assert_eq!(golden::list("mag").unwrap(), ["c32_4096", "cq15_4096"]);
    assert_eq!(
        golden::list("dc_block_real").unwrap(),
        ["r0p999_f32_8192", "r0p999_q15_8192"]
    );
    assert_eq!(
        golden::list("agc").unwrap(),
        ["steps_c32_24000", "steps_cq15_24000"]
    );
}

#[test]
fn mag_golden() {
    let case = golden::load("mag", "c32_4096").unwrap();
    let Array::CF32(x) = case.array("input").unwrap() else {
        panic!("expected cf32");
    };
    check(
        &case,
        x,
        case.array("expected").unwrap(),
        || (),
        |(), i, o: &mut [f32]| Sdrk.mag(i, o).unwrap(),
    );

    let case = golden::load("mag", "cq15_4096").unwrap();
    let Array::CI16(x) = case.array("input").unwrap() else {
        panic!("expected ci16");
    };
    check(
        &case,
        x,
        case.array("expected").unwrap(),
        || (),
        |(), i, o: &mut [Q15]| Sdrk.mag(i, o).unwrap(),
    );
}

#[test]
fn dc_block_real_golden() {
    let case = golden::load("dc_block_real", "r0p999_f32_8192").unwrap();
    let Array::F32(x) = case.array("input").unwrap() else {
        panic!("expected f32");
    };
    let r = case.param_f64("r").unwrap() as f32;
    check_guarded(
        &case,
        x,
        case.array("expected").unwrap(),
        || DcBlockRealState::<f32>::new(r).unwrap(),
        |s, i, o: &mut [f32]| Sdrk.dc_block_real(s, i, o).unwrap(),
    );

    let case = golden::load("dc_block_real", "r0p999_q15_8192").unwrap();
    let input = case.array("input").unwrap().as_q15().unwrap();
    let r = Q15(i16::try_from(case.param_u64("r_raw").unwrap()).unwrap());
    check(
        &case,
        &input,
        case.array("expected").unwrap(),
        || DcBlockRealState::<Q15>::new(r).unwrap(),
        |s, i, o: &mut [Q15]| Sdrk.dc_block_real(s, i, o).unwrap(),
    );
}

#[test]
fn agc_golden() {
    let case = golden::load("agc", "steps_c32_24000").unwrap();
    let Array::CF32(x) = case.array("input").unwrap() else {
        panic!("expected cf32");
    };
    check(
        &case,
        x,
        case.array("expected").unwrap(),
        || agc_state::<C32>(&case),
        |s, i, o: &mut [C32]| Sdrk.agc(s, i, o).unwrap(),
    );

    let case = golden::load("agc", "steps_cq15_24000").unwrap();
    let Array::CI16(x) = case.array("input").unwrap() else {
        panic!("expected ci16");
    };
    check(
        &case,
        x,
        case.array("expected").unwrap(),
        || agc_state::<CQ15>(&case),
        |s, i, o: &mut [CQ15]| Sdrk.agc(s, i, o).unwrap(),
    );
}
