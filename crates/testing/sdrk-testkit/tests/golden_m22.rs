//! Golden tests of M22: the design functions `biquad_coefs` and `butterworth_sos` and the
//! kernels `biquad` and `biquad_real` through the facade `Sdrk`.
//!
//! Every kernel case is run in one piece and in chunks of random size (the plan seed is the
//! case seed); both runs must match the golden file and each other bit for bit. The Q15 cases
//! also run on the reference.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::type_complexity
)]

use sdrk::Sdrk;
use sdrk_blocks::design::{BiquadResponse, biquad_coefs, butterworth_sos, to_f32_sos, to_q14};
use sdrk_core::kernels::biquad::{Biquad, BiquadReal, BiquadRealState, BiquadState};
use sdrk_core::{C32, CQ15, Q15, Reference, Sample};
use sdrk_testkit::golden::{self, Array, GoldenCase, Tolerance};
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

fn f32_array(case: &GoldenCase, name: &str) -> Vec<f32> {
    match case.array(name).unwrap() {
        Array::F32(v) => v.clone(),
        other => panic!("{name} must be f32, got {}", other.dtype()),
    }
}

fn rows_f32(case: &GoldenCase) -> Vec<[f32; 5]> {
    f32_array(case, "coefs")
        .chunks_exact(5)
        .map(|c| c.try_into().unwrap())
        .collect()
}

fn rows_q14(case: &GoldenCase) -> Vec<[i16; 5]> {
    match case.array("coefs").unwrap() {
        Array::I16(v) => v.chunks_exact(5).map(|c| c.try_into().unwrap()).collect(),
        other => panic!("coefs must be i16, got {}", other.dtype()),
    }
}

fn zero_hist<T: Sample>(rows: &[[impl Copy; 5]]) -> Vec<T> {
    vec![T::ZERO; 4 * rows.len()]
}

#[test]
fn all_cases_are_present() {
    assert_eq!(
        golden::list("biquad").unwrap(),
        ["butter4_lp_c32_8192", "butter4_lp_cq15_8192"]
    );
    assert_eq!(
        golden::list("biquad_real").unwrap(),
        [
            "bandpass_q2_f32_8192",
            "butter2_hp_q15_8192",
            "notch_q10_f32_8192",
            "retune_f32_8192"
        ]
    );
    let design = golden::list("design").unwrap();
    for name in [
        "biquad_lowpass",
        "biquad_highpass",
        "biquad_bandpass",
        "biquad_notch",
        "butterworth_lp2",
        "butterworth_lp4",
        "butterworth_lp8",
        "butterworth_hp6",
    ] {
        assert!(design.iter().any(|n| n == name), "missing design/{name}");
    }
}

fn max_abs_err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn expected_f64(case: &GoldenCase) -> Vec<f64> {
    match case.array("expected").unwrap() {
        Array::F64(v) => v.clone(),
        other => panic!("expected must be f64, got {}", other.dtype()),
    }
}

fn limit(case: &GoldenCase) -> f64 {
    match case.tolerance {
        Tolerance::MaxAbsErr { max, .. } => max,
        ref other => panic!("expected a max_abs_err tolerance, got {other:?}"),
    }
}

fn response(name: &str) -> BiquadResponse {
    match name {
        "lowpass" => BiquadResponse::Lowpass,
        "highpass" => BiquadResponse::Highpass,
        "bandpass" => BiquadResponse::Bandpass,
        "notch" => BiquadResponse::Notch,
        other => panic!("unknown response {other}"),
    }
}

#[test]
fn biquad_coefs_match_the_cookbook_formulas() {
    for name in ["lowpass", "highpass", "bandpass", "notch"] {
        let case = golden::load("design", &format!("biquad_{name}")).unwrap();
        let got = biquad_coefs(
            response(case.param_str("response").unwrap()),
            case.param_f64("f0_hz").unwrap(),
            case.param_f64("q").unwrap(),
            case.param_f64("fs_hz").unwrap(),
        )
        .unwrap();
        let err = max_abs_err(&got, &expected_f64(&case));
        assert!(err <= limit(&case), "{name}: largest error {err}");
    }
}

#[test]
fn butterworth_sos_matches_the_formulas_and_scipy() {
    for name in ["lp2", "lp4", "lp8", "hp6"] {
        let case = golden::load("design", &format!("butterworth_{name}")).unwrap();
        let order = usize::try_from(case.param_u64("order").unwrap()).unwrap();
        let mut sos = vec![[0.0f64; 5]; order / 2];
        butterworth_sos(
            response(case.param_str("response").unwrap()),
            order,
            case.param_f64("cutoff_hz").unwrap(),
            case.param_f64("fs_hz").unwrap(),
            &mut sos,
        )
        .unwrap();
        let flat: Vec<f64> = sos.iter().flatten().copied().collect();
        let err = max_abs_err(&flat, &expected_f64(&case));
        assert!(err <= limit(&case), "{name}: largest error {err}");
    }
}

#[test]
fn the_golden_coefficients_are_the_rounded_design() {
    // The coefficient arrays of the kernel cases are the designs of the Rust functions
    // rounded by `to_f32_sos` and `to_q14`.
    let case = golden::load("biquad", "butter4_lp_c32_8192").unwrap();
    let mut sos = [[0.0f64; 5]; 2];
    butterworth_sos(BiquadResponse::Lowpass, 4, 0.1, 1.0, &mut sos).unwrap();
    let mut f = [[0.0f32; 5]; 2];
    to_f32_sos(&sos, &mut f).unwrap();
    assert_eq!(rows_f32(&case), f);

    let case = golden::load("biquad", "butter4_lp_cq15_8192").unwrap();
    let mut q = [[0i16; 5]; 2];
    to_q14(&sos, &mut q).unwrap();
    assert_eq!(rows_q14(&case), q);

    let case = golden::load("biquad_real", "butter2_hp_q15_8192").unwrap();
    let mut sos = [[0.0f64; 5]; 1];
    butterworth_sos(BiquadResponse::Highpass, 2, 0.01, 1.0, &mut sos).unwrap();
    let mut q = [[0i16; 5]; 1];
    to_q14(&sos, &mut q).unwrap();
    assert_eq!(rows_q14(&case), q);

    let case = golden::load("biquad_real", "notch_q10_f32_8192").unwrap();
    let c = biquad_coefs(BiquadResponse::Notch, 0.05, 10.0, 1.0).unwrap();
    let mut f = [[0.0f32; 5]; 1];
    to_f32_sos(&[c], &mut f).unwrap();
    assert_eq!(rows_f32(&case), f);
}

#[test]
fn biquad_c32_golden() {
    let case = golden::load("biquad", "butter4_lp_c32_8192").unwrap();
    let Array::CF32(x) = case.array("input").unwrap() else {
        panic!("expected cf32");
    };
    let rows = rows_f32(&case);
    check(
        &case,
        x,
        case.array("expected").unwrap(),
        || BiquadState::<C32, _, _>::new(rows.clone(), zero_hist(&rows)).unwrap(),
        |s, i, o: &mut [C32]| Sdrk.biquad(s, i, o).unwrap(),
    );
}

#[test]
fn biquad_cq15_golden() {
    let case = golden::load("biquad", "butter4_lp_cq15_8192").unwrap();
    let Array::CI16(x) = case.array("input").unwrap() else {
        panic!("expected ci16");
    };
    let rows = rows_q14(&case);
    check(
        &case,
        x,
        case.array("expected").unwrap(),
        || BiquadState::<CQ15, _, _>::new(rows.clone(), zero_hist(&rows)).unwrap(),
        |s, i, o: &mut [CQ15]| Sdrk.biquad(s, i, o).unwrap(),
    );
    check(
        &case,
        x,
        case.array("expected").unwrap(),
        || BiquadState::<CQ15, _, _>::new(rows.clone(), zero_hist(&rows)).unwrap(),
        |s, i, o: &mut [CQ15]| Reference.biquad(s, i, o).unwrap(),
    );
}

#[test]
fn biquad_real_f32_golden() {
    for name in ["notch_q10_f32_8192", "bandpass_q2_f32_8192"] {
        let case = golden::load("biquad_real", name).unwrap();
        let Array::F32(x) = case.array("input").unwrap() else {
            panic!("expected f32");
        };
        let rows = rows_f32(&case);
        check(
            &case,
            x,
            case.array("expected").unwrap(),
            || BiquadRealState::<f32, _, _>::new(rows.clone(), zero_hist(&rows)).unwrap(),
            |s, i, o: &mut [f32]| Sdrk.biquad_real(s, i, o).unwrap(),
        );
    }
}

#[test]
fn biquad_real_q15_golden() {
    let case = golden::load("biquad_real", "butter2_hp_q15_8192").unwrap();
    let input = case.array("input").unwrap().as_q15().unwrap();
    let rows = rows_q14(&case);
    check(
        &case,
        &input,
        case.array("expected").unwrap(),
        || BiquadRealState::<Q15, _, _>::new(rows.clone(), zero_hist(&rows)).unwrap(),
        |s, i, o: &mut [Q15]| Sdrk.biquad_real(s, i, o).unwrap(),
    );
    check(
        &case,
        &input,
        case.array("expected").unwrap(),
        || BiquadRealState::<Q15, _, _>::new(rows.clone(), zero_hist(&rows)).unwrap(),
        |s, i, o: &mut [Q15]| Reference.biquad_real(s, i, o).unwrap(),
    );
}

#[test]
fn biquad_real_retune_golden() {
    let case = golden::load("biquad_real", "retune_f32_8192").unwrap();
    let Array::F32(x) = case.array("input").unwrap() else {
        panic!("expected f32");
    };
    let rows = rows_f32(&case);
    assert_eq!(rows.len(), 2);
    let switch = usize::try_from(case.param_u64("switch_at").unwrap()).unwrap();
    // The state starts with the first row; `set_coefs` installs the second at `switch`. The
    // chunk plan is cut at `switch`, as a block would apply a runtime change between calls.
    let run = |sizes: &mut dyn Iterator<Item = usize>| {
        let mut state = BiquadRealState::<f32, _, _>::new(vec![rows[0]], vec![0.0f32; 4]).unwrap();
        let mut out = vec![0.0f32; x.len()];
        let mut pos = 0;
        while pos < x.len() {
            if pos == switch {
                state.set_coefs(vec![rows[1]]).unwrap();
            }
            let end = if pos < switch { switch } else { x.len() };
            let n = sizes.next().unwrap().min(end - pos);
            Sdrk.biquad_real(&mut state, &x[pos..pos + n], &mut out[pos..pos + n])
                .unwrap();
            pos += n;
        }
        out
    };
    let whole = run(&mut std::iter::repeat(usize::MAX));
    let chunked = run(&mut ChunkPlan::new(case.seed, 700).sizes());
    let want = case.array("expected").unwrap();
    golden::assert_matches(&case, want, &whole).unwrap();
    assert_eq!(bit_exact(&whole, &chunked), None);
    golden::assert_matches(&case, want, &chunked).unwrap();
}
