//! Golden tests of the `preemph_limit` kernel (M25) through the facade `Sdrk`, `Fallback` and
//! `Reference`, and of `preemphasis_coefs` against the design cases.
//!
//! Every kernel case is run in one piece and in chunks of random size (the plan seed is the case
//! seed); both runs must match the golden file and each other bit for bit. `retune_f32_8192`
//! calls `set_limit` before sample 4096 in both modes. The state is built from the coefficients
//! stored in the case (as IEEE bits), so a last-bit difference of the design function cannot hide
//! a kernel error; a separate test checks `preemphasis_coefs` against the same values.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::cast_possible_truncation
)]

use sdrk::Sdrk;
use sdrk_blocks::design::preemphasis_coefs;
use sdrk_core::kernels::preemph_limit::{PreemphLimit, PreemphLimitState};
use sdrk_core::{Fallback, Q15, Rate, RealSample, Reference};
use sdrk_testkit::golden::{self, Array, GoldenCase};
use sdrk_testkit::{ChunkPlan, bit_exact};

fn coefs_of(case: &GoldenCase) -> [f64; 3] {
    let bits = |key: &str| f64::from_bits(case.param_u64(key).unwrap());
    [bits("b0_bits"), bits("b1_bits"), bits("a1_bits")]
}

fn state_of<T: RealSample>(case: &GoldenCase) -> PreemphLimitState<T> {
    let rate = Rate::hz(case.param_u64("rate_hz").unwrap()).unwrap();
    PreemphLimitState::new(
        coefs_of(case),
        case.param_f64("limit").unwrap(),
        case.param_f64("hold_s").unwrap(),
        case.param_f64("release_s").unwrap(),
        rate,
    )
    .unwrap()
}

fn run<T: RealSample, K: PreemphLimit<T>>(
    kernel: &K,
    case: &GoldenCase,
    input: &[T],
    plan: Option<ChunkPlan>,
) -> Vec<T> {
    let mut state = state_of::<T>(case);
    let retune = case.param_u64("retune_at").ok().map(|at| {
        (
            usize::try_from(at).unwrap(),
            case.param_f64("retune_limit").unwrap(),
        )
    });
    let mut cuts = vec![0];
    cuts.extend(retune.map(|r| r.0));
    cuts.push(input.len());
    let mut sizes = plan.map(|p| p.sizes());
    let mut out = vec![T::ZERO; input.len()];
    for seg in cuts.windows(2) {
        let (mut pos, end) = (seg[0], seg[1]);
        if let Some((_, limit)) = retune.filter(|r| r.0 == pos) {
            state.set_limit(limit).unwrap();
        }
        while pos < end {
            let n = sizes
                .as_mut()
                .map_or(end - pos, |s| s.next().unwrap().min(end - pos));
            kernel
                .preemph_limit(&mut state, &input[pos..pos + n], &mut out[pos..pos + n])
                .unwrap();
            pos += n;
        }
    }
    out
}

fn check<T: RealSample, K: PreemphLimit<T>>(kernel: &K, case: &GoldenCase, input: &[T]) -> Vec<T> {
    let want = case.array("expected").unwrap();
    let whole = run(kernel, case, input, None);
    let chunked = run(kernel, case, input, Some(ChunkPlan::new(case.seed, 700)));
    golden::assert_matches(case, want, &whole).unwrap();
    assert_eq!(
        bit_exact(&whole, &chunked),
        None,
        "{}: block size changed the output",
        case.case
    );
    whole
}

fn f32_case(name: &str) {
    let case = golden::load("preemph_limit", name).unwrap();
    let Array::F32(input) = case.array("input").unwrap() else {
        panic!("f32 input expected");
    };
    let facade = check::<f32, _>(&Sdrk, &case, input);
    let fallback = check::<f32, _>(&Fallback, &case, input);
    let reference = check::<f32, _>(&Reference, &case, input);
    assert_eq!(bit_exact(&facade, &fallback), None);
    assert_eq!(bit_exact(&facade, &reference), None);
}

fn q15_case(name: &str) {
    let case = golden::load("preemph_limit", name).unwrap();
    let input = case.array("input").unwrap().as_q15().unwrap();
    let facade = check::<Q15, _>(&Sdrk, &case, &input);
    let fallback = check::<Q15, _>(&Fallback, &case, &input);
    let reference = check::<Q15, _>(&Reference, &case, &input);
    assert_eq!(bit_exact(&facade, &fallback), None);
    assert_eq!(bit_exact(&facade, &reference), None);
}

#[test]
fn all_cases_are_present() {
    assert_eq!(
        golden::list("preemph_limit").unwrap(),
        [
            "limit_f32_8192",
            "limit_q15_8192",
            "nonfinite_f32_4096",
            "preemph_f32_8192",
            "preemph_q15_8192",
            "retune_f32_8192"
        ]
    );
    let designs: Vec<String> = golden::list("design")
        .unwrap()
        .into_iter()
        .filter(|n| n.starts_with("preemphasis_"))
        .collect();
    assert_eq!(
        designs,
        [
            "preemphasis_50us_240k",
            "preemphasis_50us_48k",
            "preemphasis_530us_48k",
            "preemphasis_75us_48k"
        ]
    );
}

#[test]
fn preemph_f32_golden() {
    f32_case("preemph_f32_8192");
}

#[test]
fn limit_f32_golden() {
    f32_case("limit_f32_8192");
}

#[test]
fn nonfinite_f32_golden() {
    f32_case("nonfinite_f32_4096");
}

#[test]
fn retune_f32_golden() {
    f32_case("retune_f32_8192");
}

#[test]
fn limit_q15_golden() {
    q15_case("limit_q15_8192");
}

#[test]
fn preemph_q15_golden() {
    q15_case("preemph_q15_8192");
}

#[test]
fn the_design_cases_match_the_function() {
    for name in [
        "preemphasis_50us_48k",
        "preemphasis_75us_48k",
        "preemphasis_530us_48k",
        "preemphasis_50us_240k",
    ] {
        let case = golden::load("design", name).unwrap();
        let got = preemphasis_coefs(
            case.param_f64("tau_s").unwrap(),
            case.param_f64("fs_hz").unwrap(),
        )
        .unwrap();
        let Array::F64(want) = case.array("expected").unwrap() else {
            panic!("f64 expected");
        };
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= 1e-14, "{name}: {g} against {w}");
        }
    }
}

#[test]
fn the_coefficients_of_the_kernel_cases_are_the_design_values() {
    for name in [
        "preemph_f32_8192",
        "limit_f32_8192",
        "limit_q15_8192",
        "preemph_q15_8192",
        "nonfinite_f32_4096",
        "retune_f32_8192",
    ] {
        let case = golden::load("preemph_limit", name).unwrap();
        let got = preemphasis_coefs(
            case.param_f64("tau_s").unwrap(),
            case.param_f64("rate_hz").unwrap(),
        )
        .unwrap();
        let stored = coefs_of(&case);
        for (g, w) in got.iter().zip(stored) {
            assert!((g - w).abs() <= 1e-14, "{name}: {g} against {w}");
        }
    }
}
