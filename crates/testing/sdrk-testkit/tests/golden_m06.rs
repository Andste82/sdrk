//! Golden tests of the receive kernels of M06 through the facade `Sdrk`.
//!
//! Every stateful case is run in one piece and in chunks of random size (with a small
//! output buffer, so the `Progress` handling is exercised too); both runs must match the
//! golden file and each other bit for bit. The exceptions are `deemphasis` `f32` and
//! `dc_block` `C32`, tolerance kernels of the SIMD backend whose bits may depend on the split
//! (maintainer decision of 2026-10-29): there every split must reach the regression guard of
//! the case (`check_guarded`; the fallback-only builds of the facade check the fallback).
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::type_complexity
)]

use sdrk::Sdrk;
use sdrk_core::kernels::decim::{
    FirDecimState, FirDecimate, Resample, ResampleReal, ResampleRealState, ResampleState,
};
use sdrk_core::kernels::demod::{
    DcBlock, DcBlockState, Deemphasis, DeemphasisState, QuadDemod, QuadDemodState,
};
use sdrk_core::kernels::spectrum::{MagSq, PowerDb, Window};
use sdrk_core::window::{Symmetry, blackman_harris, hann, kaiser};
use sdrk_core::{C32, CQ15, Progress, Q15, Rate, Sample};
use sdrk_testkit::golden::{self, Array, GoldenCase};
use sdrk_testkit::{ChunkPlan, bit_exact};

/// Output buffer size of the chunked runs; deliberately smaller than most chunks.
const SMALL_ROOM: usize = 13;

/// Runs a rate-changing kernel over `input` in one piece. `room` is the output capacity.
fn run_whole<T: Copy + Default, S>(
    state: &mut S,
    input: &[T],
    room: usize,
    step: &mut impl FnMut(&mut S, &[T], &mut [T]) -> Progress,
) -> Vec<T> {
    let mut out = vec![T::default(); room];
    let p = step(state, input, &mut out);
    assert_eq!(p.consumed, input.len(), "the whole input must be consumed");
    out.truncate(p.produced);
    out
}

/// Runs a rate-changing kernel in chunks from `plan`, with `SMALL_ROOM` outputs per call;
/// unconsumed input is resubmitted.
fn run_chunked<T: Copy + Default, S>(
    state: &mut S,
    input: &[T],
    plan: ChunkPlan,
    step: &mut impl FnMut(&mut S, &[T], &mut [T]) -> Progress,
) -> Vec<T> {
    let mut all = Vec::new();
    let mut pos = 0;
    for size in plan.sizes() {
        if pos >= input.len() {
            break;
        }
        let end = (pos + size).min(input.len());
        let mut rest = &input[pos..end];
        loop {
            let mut buf = [T::default(); SMALL_ROOM];
            let p = step(state, rest, &mut buf);
            all.extend_from_slice(&buf[..p.produced]);
            rest = &rest[p.consumed..];
            if rest.is_empty() && p.produced < SMALL_ROOM {
                break;
            }
            assert!(
                p.consumed + p.produced > 0,
                "the kernel made no progress with {} samples left",
                rest.len()
            );
        }
        pos = end;
    }
    all
}

/// Runs a same-rate kernel in chunks from `plan`.
fn run_same_rate_chunked<I: Copy, O: Copy + Default, S>(
    state: &mut S,
    input: &[I],
    plan: ChunkPlan,
    step: &mut impl FnMut(&mut S, &[I], &mut [O]),
) -> Vec<O> {
    let mut all = Vec::new();
    let mut pos = 0;
    for size in plan.sizes() {
        if pos >= input.len() {
            break;
        }
        let end = (pos + size).min(input.len());
        let mut out = vec![O::default(); end - pos];
        step(state, &input[pos..end], &mut out);
        all.extend(out);
        pos = end;
    }
    all
}

fn check_both<T: Sample>(case: &GoldenCase, want: &Array, whole: &[T], chunked: &[T]) {
    golden::assert_matches(case, want, whole).unwrap();
    assert_eq!(
        bit_exact(whole, chunked),
        None,
        "{}: block size changed the output",
        case.case
    );
    golden::assert_matches(case, want, chunked).unwrap();
}

/// Runs `step` over `input` in one piece and split by chunk plans of several largest sizes
/// (down to single samples); every run must reach the tolerance of the golden case, its
/// regression guard included. The bits may differ between the runs.
fn check_guarded<T: Sample, S>(
    case: &GoldenCase,
    want: &Array,
    input: &[T],
    new: impl Fn() -> S,
    mut step: impl FnMut(&mut S, &[T], &mut [T]),
) {
    let mut whole = vec![T::ZERO; input.len()];
    step(&mut new(), input, &mut whole);
    golden::assert_matches(case, want, &whole).unwrap();
    for max in [1, 7, 40, 500] {
        let plan = ChunkPlan::new(case.seed + max as u64, max);
        let chunked = run_same_rate_chunked(&mut new(), input, plan, &mut step);
        golden::assert_matches(case, want, &chunked)
            .unwrap_or_else(|e| panic!("{}: chunks of at most {max}: {e}", case.case));
    }
}

fn usize_param(case: &GoldenCase, name: &str) -> usize {
    usize::try_from(case.param_u64(name).unwrap()).unwrap()
}

#[test]
fn all_cases_are_present() {
    let expect: [(&str, &[&str]); 11] = [
        (
            "fir_decimate",
            &["m10_n63_c32_20000", "m10_n63_cq15_20000", "m3_n31_c32_9000"],
        ),
        (
            "resample",
            &[
                "l1m5_n121_c32_10000",
                "l3m2_n96_c32_6000",
                "l147m160_n1470_c32_16000",
                "l3m2_n96_cq15_6000",
            ],
        ),
        ("resample_real", &["l1m5_n121_f32_10000"]),
        ("quad_demod", &["fm_tone_c32_20000", "fm_tone_cq15_20000"]),
        (
            "deemphasis",
            &["tau50us_fs48k_f32_8192", "tau50us_fs48k_q15_8192"],
        ),
        ("dc_block", &["r0p999_c32_8192", "r0p999_cq15_8192"]),
        (
            "window",
            &[
                "hann_periodic_1024",
                "hann_symmetric_1023",
                "blackmanharris_periodic_4096",
                "kaiser8_symmetric_255",
                "apply_hann_c32_1024",
                "apply_hann_cq15_1024",
            ],
        ),
        ("mag_sq", &["c32_4096", "cq15_4096"]),
        ("power_db", &["f32_4096"]),
        ("fir", &[]),
        ("rotate", &[]),
    ];
    for (kernel, cases) in expect {
        let names = golden::list(kernel).unwrap();
        for c in cases {
            assert!(names.iter().any(|n| n == c), "missing {kernel}/{c}");
        }
        if !cases.is_empty() {
            assert_eq!(names.len(), cases.len(), "{kernel}: unexpected cases");
        }
    }
}

#[test]
fn fir_decimate_golden() {
    for name in golden::list("fir_decimate").unwrap() {
        let case = golden::load("fir_decimate", &name).unwrap();
        let want = case.array("expected").unwrap();
        let m = usize_param(&case, "factor");
        let plan = ChunkPlan::new(case.seed, 700);
        match (case.array("input").unwrap(), case.array("taps").unwrap()) {
            (Array::CF32(x), Array::F32(h)) => {
                let new = || {
                    FirDecimState::<C32, _, _>::new(h.clone(), vec![C32::ZERO; h.len() - 1], m)
                        .unwrap()
                };
                let mut step =
                    |s: &mut FirDecimState<C32, Vec<f32>, Vec<C32>>, i: &[C32], o: &mut [C32]| {
                        FirDecimate::<C32>::fir_decimate(&Sdrk, s, i, o).unwrap()
                    };
                let whole = run_whole(&mut new(), x, want.len() + 3, &mut step);
                let chunked = run_chunked(&mut new(), x, plan, &mut step);
                check_both(&case, want, &whole, &chunked);
            }
            (Array::CI16(x), Array::I16(_)) => {
                let h = case.array("taps").unwrap().as_q15().unwrap();
                let new = || {
                    FirDecimState::<CQ15, _, _>::new(h.clone(), vec![CQ15::ZERO; h.len() - 1], m)
                        .unwrap()
                };
                let mut step = |s: &mut FirDecimState<CQ15, Vec<Q15>, Vec<CQ15>>,
                                i: &[CQ15],
                                o: &mut [CQ15]| {
                    FirDecimate::<CQ15>::fir_decimate(&Sdrk, s, i, o).unwrap()
                };
                let whole = run_whole(&mut new(), x, want.len() + 3, &mut step);
                let chunked = run_chunked(&mut new(), x, plan, &mut step);
                check_both(&case, want, &whole, &chunked);
            }
            _ => panic!("unexpected array types in {name}"),
        }
    }
}

#[test]
fn resample_golden() {
    for name in golden::list("resample").unwrap() {
        let case = golden::load("resample", &name).unwrap();
        let want = case.array("expected").unwrap();
        let (up, down) = (usize_param(&case, "up"), usize_param(&case, "down"));
        assert_eq!(want.len(), usize_param(&case, "outputs"));
        let plan = ChunkPlan::new(case.seed, 900);
        match (case.array("input").unwrap(), case.array("taps").unwrap()) {
            (Array::CF32(x), Array::F32(h)) => {
                let new = || {
                    ResampleState::<C32, _, _>::new(
                        h.clone(),
                        vec![C32::ZERO; h.len().div_ceil(up)],
                        up,
                        down,
                    )
                    .unwrap()
                };
                let mut step =
                    |s: &mut ResampleState<C32, Vec<f32>, Vec<C32>>, i: &[C32], o: &mut [C32]| {
                        Resample::<C32>::resample(&Sdrk, s, i, o).unwrap()
                    };
                let whole = run_whole(&mut new(), x, want.len() + 5, &mut step);
                let chunked = run_chunked(&mut new(), x, plan, &mut step);
                check_both(&case, want, &whole, &chunked);
            }
            (Array::CI16(x), Array::I16(_)) => {
                let h = case.array("taps").unwrap().as_q15().unwrap();
                let new = || {
                    ResampleState::<CQ15, _, _>::new(
                        h.clone(),
                        vec![CQ15::ZERO; h.len().div_ceil(up)],
                        up,
                        down,
                    )
                    .unwrap()
                };
                let mut step = |s: &mut ResampleState<CQ15, Vec<Q15>, Vec<CQ15>>,
                                i: &[CQ15],
                                o: &mut [CQ15]| {
                    Resample::<CQ15>::resample(&Sdrk, s, i, o).unwrap()
                };
                let whole = run_whole(&mut new(), x, want.len() + 5, &mut step);
                let chunked = run_chunked(&mut new(), x, plan, &mut step);
                check_both(&case, want, &whole, &chunked);
            }
            _ => panic!("unexpected array types in {name}"),
        }
    }
}

#[test]
fn resample_real_golden() {
    for name in golden::list("resample_real").unwrap() {
        let case = golden::load("resample_real", &name).unwrap();
        let want = case.array("expected").unwrap();
        let (up, down) = (usize_param(&case, "up"), usize_param(&case, "down"));
        assert_eq!(want.len(), usize_param(&case, "outputs"));
        let (Array::F32(x), Array::F32(h)) =
            (case.array("input").unwrap(), case.array("taps").unwrap())
        else {
            panic!("unexpected array types in {name}");
        };
        let new = || {
            ResampleRealState::<f32, _, _>::new(
                h.clone(),
                vec![0.0f32; h.len().div_ceil(up)],
                up,
                down,
            )
            .unwrap()
        };
        let mut step =
            |s: &mut ResampleRealState<f32, Vec<f32>, Vec<f32>>, i: &[f32], o: &mut [f32]| {
                ResampleReal::<f32>::resample_real(&Sdrk, s, i, o).unwrap()
            };
        let whole = run_whole(&mut new(), x, want.len() + 5, &mut step);
        let chunked = run_chunked(&mut new(), x, ChunkPlan::new(case.seed, 900), &mut step);
        check_both(&case, want, &whole, &chunked);
    }
}

#[test]
fn quad_demod_golden() {
    for name in golden::list("quad_demod").unwrap() {
        let case = golden::load("quad_demod", &name).unwrap();
        let want = case.array("expected").unwrap();
        let plan = ChunkPlan::new(case.seed, 600);
        match case.array("input").unwrap() {
            Array::CF32(x) => {
                let gain = case.param_f64("gain").unwrap() as f32;
                let mut step = |s: &mut QuadDemodState<C32>, i: &[C32], o: &mut [f32]| {
                    QuadDemod::<C32>::quad_demod(&Sdrk, s, i, o, gain).unwrap();
                };
                let mut whole = vec![0f32; x.len()];
                step(&mut QuadDemodState::new(), x, &mut whole);
                let chunked = run_same_rate_chunked(&mut QuadDemodState::new(), x, plan, &mut step);
                check_both(&case, want, &whole, &chunked);
            }
            Array::CI16(x) => {
                let gain = Q15(i16::try_from(case.param_u64("gain_raw").unwrap()).unwrap());
                let mut step = |s: &mut QuadDemodState<CQ15>, i: &[CQ15], o: &mut [Q15]| {
                    QuadDemod::<CQ15>::quad_demod(&Sdrk, s, i, o, gain).unwrap();
                };
                let mut whole = vec![Q15(0); x.len()];
                step(&mut QuadDemodState::new(), x, &mut whole);
                let chunked = run_same_rate_chunked(&mut QuadDemodState::new(), x, plan, &mut step);
                check_both(&case, want, &whole, &chunked);
            }
            other => panic!("unexpected input type {}", other.dtype()),
        }
    }
}

#[test]
fn deemphasis_golden() {
    for name in golden::list("deemphasis").unwrap() {
        let case = golden::load("deemphasis", &name).unwrap();
        let want = case.array("expected").unwrap();
        let tau = case.param_f64("tau_s").unwrap();
        let rate = Rate::hz(case.param_u64("rate_hz").unwrap()).unwrap();
        let plan = ChunkPlan::new(case.seed, 500);
        match case.array("input").unwrap() {
            Array::F32(x) => {
                let new = || DeemphasisState::<f32>::new(tau, rate).unwrap();
                assert!(
                    (f64::from(new().alpha()) - case.param_f64("alpha").unwrap()).abs() < 1e-15
                );
                check_guarded(&case, want, x, new, |s, i, o| {
                    Deemphasis::<f32>::deemphasis(&Sdrk, s, i, o).unwrap();
                });
            }
            Array::I16(raw) => {
                let x: Vec<Q15> = raw.iter().map(|&r| Q15(r)).collect();
                let new = || DeemphasisState::<Q15>::new(tau, rate).unwrap();
                assert_eq!(
                    u64::try_from(new().alpha().0).unwrap(),
                    case.param_u64("alpha_raw").unwrap()
                );
                let mut step = |s: &mut DeemphasisState<Q15>, i: &[Q15], o: &mut [Q15]| {
                    Deemphasis::<Q15>::deemphasis(&Sdrk, s, i, o).unwrap();
                };
                let mut whole = vec![Q15(0); x.len()];
                step(&mut new(), &x, &mut whole);
                let chunked = run_same_rate_chunked(&mut new(), &x, plan, &mut step);
                check_both(&case, want, &whole, &chunked);
            }
            other => panic!("unexpected input type {}", other.dtype()),
        }
    }
}

#[test]
fn dc_block_golden() {
    for name in golden::list("dc_block").unwrap() {
        let case = golden::load("dc_block", &name).unwrap();
        let want = case.array("expected").unwrap();
        let plan = ChunkPlan::new(case.seed, 400);
        match case.array("input").unwrap() {
            Array::CF32(x) => {
                let r = case.param_f64("r").unwrap() as f32;
                assert!((f64::from(r) - case.param_f64("r").unwrap()).abs() < 1e-15);
                let new = || DcBlockState::<C32>::new(r).unwrap();
                check_guarded(&case, want, x, new, |s, i, o| {
                    DcBlock::<C32>::dc_block(&Sdrk, s, i, o).unwrap();
                });
            }
            Array::CI16(x) => {
                let r = Q15(i16::try_from(case.param_u64("r_raw").unwrap()).unwrap());
                let new = || DcBlockState::<CQ15>::new(r).unwrap();
                let mut step = |s: &mut DcBlockState<CQ15>, i: &[CQ15], o: &mut [CQ15]| {
                    DcBlock::<CQ15>::dc_block(&Sdrk, s, i, o).unwrap();
                };
                let mut whole = vec![CQ15::ZERO; x.len()];
                step(&mut new(), x, &mut whole);
                let chunked = run_same_rate_chunked(&mut new(), x, plan, &mut step);
                check_both(&case, want, &whole, &chunked);
            }
            other => panic!("unexpected input type {}", other.dtype()),
        }
    }
}

fn symmetry_of(case: &GoldenCase) -> Symmetry {
    match case.param_str("symmetry").unwrap() {
        "periodic" => Symmetry::Periodic,
        "symmetric" => Symmetry::Symmetric,
        other => panic!("unknown symmetry {other}"),
    }
}

#[test]
fn window_generator_golden() {
    for name in [
        "hann_periodic_1024",
        "hann_symmetric_1023",
        "blackmanharris_periodic_4096",
        "kaiser8_symmetric_255",
    ] {
        let case = golden::load("window", name).unwrap();
        let n = usize_param(&case, "n");
        let sym = symmetry_of(&case);
        let mut w = vec![0f32; n];
        match case.param_str("window").unwrap() {
            "hann" => hann(&mut w, sym).unwrap(),
            "blackman_harris" => blackman_harris(&mut w, sym).unwrap(),
            "kaiser" => kaiser(&mut w, case.param_f64("beta").unwrap(), sym).unwrap(),
            other => panic!("unknown window {other}"),
        }
        golden::assert_matches(&case, case.array("expected").unwrap(), &w).unwrap();
    }
}

#[test]
fn window_apply_golden() {
    let case = golden::load("window", "apply_hann_c32_1024").unwrap();
    let (Array::CF32(x), Array::F32(w)) =
        (case.array("input").unwrap(), case.array("window").unwrap())
    else {
        panic!("unexpected array types");
    };
    let mut out = vec![C32::ZERO; x.len()];
    Window::<C32>::window(&Sdrk, x, &mut out, w).unwrap();
    golden::assert_matches(&case, case.array("expected").unwrap(), &out).unwrap();

    let case = golden::load("window", "apply_hann_cq15_1024").unwrap();
    let Array::CI16(x) = case.array("input").unwrap() else {
        panic!("unexpected input type");
    };
    let w = case.array("window").unwrap().as_q15().unwrap();
    let mut out = vec![CQ15::ZERO; x.len()];
    Window::<CQ15>::window(&Sdrk, x, &mut out, &w).unwrap();
    golden::assert_matches(&case, case.array("expected").unwrap(), &out).unwrap();
}

#[test]
fn mag_sq_golden() {
    let case = golden::load("mag_sq", "c32_4096").unwrap();
    let Array::CF32(x) = case.array("input").unwrap() else {
        panic!("unexpected input type");
    };
    let mut out = vec![0f32; x.len()];
    MagSq::<C32>::mag_sq(&Sdrk, x, &mut out).unwrap();
    golden::assert_matches(&case, case.array("expected").unwrap(), &out).unwrap();

    let case = golden::load("mag_sq", "cq15_4096").unwrap();
    let Array::CI16(x) = case.array("input").unwrap() else {
        panic!("unexpected input type");
    };
    let mut out = vec![Q15(0); x.len()];
    MagSq::<CQ15>::mag_sq(&Sdrk, x, &mut out).unwrap();
    golden::assert_matches(&case, case.array("expected").unwrap(), &out).unwrap();
}

#[test]
fn power_db_golden() {
    let case = golden::load("power_db", "f32_4096").unwrap();
    let Array::F32(p) = case.array("input").unwrap() else {
        panic!("unexpected input type");
    };
    let offset = case.param_f64("offset_db").unwrap() as f32;
    let mut out = vec![0f32; p.len()];
    PowerDb::<f32>::power_db(&Sdrk, p, &mut out, offset).unwrap();
    golden::assert_matches(&case, case.array("expected").unwrap(), &out).unwrap();
}
