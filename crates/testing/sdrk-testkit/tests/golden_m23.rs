//! Golden tests of M23: the design functions `halfband_len` and `halfband_taps` and the kernels
//! `halfband_decimate`, `halfband_decimate_real` and `real_to_complex` through the facade `Sdrk`.
//!
//! Every kernel case is run in one piece and in chunks of random size from the case seed (with a
//! small output buffer, so the `Progress` handling is exercised too). The Q15 cases must be
//! bit-exact in both runs and equal to each other. The tolerance cases must reach the SNR and the
//! regression guard of the case in both runs; the fallback additionally gives the same bits in
//! both runs. The Q15 cases are also compared with the `f64` model (the SNR of the integer result
//! against the unquantized arithmetic on the quantized values).
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::print_stderr
)]

use sdrk::Sdrk;
use sdrk_blocks::design::{halfband_len, halfband_taps, kaiser_beta};
use sdrk_core::kernels::halfband::{
    HalfbandDecimate, HalfbandDecimateReal, HalfbandRealState, HalfbandState, RealToComplex,
    RealToComplexState,
};
use sdrk_core::{C32, CQ15, Fallback, Progress, Q15, Sample};
use sdrk_testkit::golden::{self, Array, GoldenCase, Tolerance};
use sdrk_testkit::{ChunkPlan, bit_exact};

/// Output buffer size of the chunked runs; deliberately smaller than most chunks.
const SMALL_ROOM: usize = 13;

fn run_whole<T: Copy + Default, I: Copy, S>(
    state: &mut S,
    input: &[I],
    room: usize,
    step: &mut impl FnMut(&mut S, &[I], &mut [T]) -> Progress,
) -> Vec<T> {
    let mut out = vec![T::default(); room];
    let p = step(state, input, &mut out);
    assert_eq!(p.consumed, input.len(), "the whole input must be consumed");
    out.truncate(p.produced);
    out
}

fn run_chunked<T: Copy + Default, I: Copy, S>(
    state: &mut S,
    input: &[I],
    plan: ChunkPlan,
    step: &mut impl FnMut(&mut S, &[I], &mut [T]) -> Progress,
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

/// Both runs must match the golden file; `same_bits` demands that they are equal as well.
fn check_both<T: Sample>(
    case: &GoldenCase,
    want: &Array,
    whole: &[T],
    chunked: &[T],
    same_bits: bool,
) {
    golden::assert_matches(case, want, whole).unwrap();
    if same_bits {
        assert_eq!(
            bit_exact(whole, chunked),
            None,
            "{}: block size changed the output",
            case.case
        );
    }
    golden::assert_matches(case, want, chunked).unwrap();
}

fn usize_param(case: &GoldenCase, name: &str) -> usize {
    usize::try_from(case.param_u64(name).unwrap()).unwrap()
}

fn f64_array(case: &GoldenCase, name: &str) -> Vec<f64> {
    match case.array(name).unwrap() {
        Array::F64(v) => v.clone(),
        other => panic!("{name} must be f64, got {}", other.dtype()),
    }
}

fn max_abs_err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn is_tolerance(case: &GoldenCase) -> bool {
    matches!(case.tolerance, Tolerance::Snr { .. })
}

#[test]
fn all_cases_are_present() {
    for (kernel, cases) in [
        (
            "halfband_decimate",
            ["n23_c32_8192", "n23_cq15_8192"].as_slice(),
        ),
        (
            "halfband_decimate_real",
            ["n23_f32_8192", "n23_q15_8192"].as_slice(),
        ),
        (
            "real_to_complex",
            ["n63_c32_16384", "n63_cq15_16384"].as_slice(),
        ),
    ] {
        assert_eq!(golden::list(kernel).unwrap(), cases, "{kernel}");
        for (name, seed) in cases.iter().zip(match kernel {
            "halfband_decimate" => [430, 431],
            "halfband_decimate_real" => [432, 433],
            _ => [434, 435],
        }) {
            assert_eq!(golden::load(kernel, name).unwrap().seed, seed);
        }
    }
    let design = golden::list("design").unwrap();
    for name in [
        "halfband_n11",
        "halfband_n23",
        "halfband_n63",
        "halfband_len_table",
    ] {
        assert!(design.iter().any(|n| n == name), "missing design/{name}");
    }
}

#[test]
fn halfband_taps_match_scipy() {
    for n in [11usize, 23, 63] {
        let case = golden::load("design", &format!("halfband_n{n}")).unwrap();
        assert_eq!(usize_param(&case, "num_taps"), n);
        let beta = case.param_f64("beta").unwrap();
        assert_eq!(beta, kaiser_beta(case.param_f64("attenuation_db").unwrap()));
        let mut taps = vec![0.0; n];
        halfband_taps(beta, &mut taps).unwrap();
        let want = f64_array(&case, "expected");
        let Tolerance::MaxAbsErr { max, .. } = case.tolerance else {
            panic!("design cases use max_abs_err");
        };
        let err = max_abs_err(&taps, &want);
        assert!(err <= max, "n {n}: largest error {err}");
        // The structure: exact zeros, symmetric, outermost taps non-zero, sum 1.
        let m = (n - 1) / 2;
        for (k, t) in taps.iter().enumerate() {
            if k != m && k.abs_diff(m) % 2 == 0 {
                assert_eq!(t.to_bits(), 0.0f64.to_bits(), "n {n} tap {k}");
            } else {
                assert!(*t != 0.0, "n {n} tap {k}");
            }
            assert_eq!(*t, taps[n - 1 - k], "n {n} tap {k}");
        }
        assert!((taps.iter().sum::<f64>() - 1.0).abs() < 1e-14);
    }
}

#[test]
fn halfband_len_matches_scipy() {
    let case = golden::load("design", "halfband_len_table").unwrap();
    let a = f64_array(&case, "attenuation_db");
    let t = f64_array(&case, "transition_hz");
    let Array::I32(want) = case.array("expected").unwrap() else {
        panic!("expected must be i32");
    };
    let fs = case.param_f64("fs_hz").unwrap();
    assert_eq!(a.len(), 10);
    for ((a, t), n) in a.iter().zip(&t).zip(want) {
        let got = halfband_len(*t, *a, fs).unwrap();
        assert_eq!(
            got,
            usize::try_from(*n).unwrap(),
            "attenuation {a}, transition {t}"
        );
        assert_eq!(got % 4, 3);
    }
}

#[test]
fn halfband_decimate_golden() {
    for name in golden::list("halfband_decimate").unwrap() {
        let case = golden::load("halfband_decimate", &name).unwrap();
        let want = case.array("expected").unwrap();
        let plan = ChunkPlan::new(case.seed, 700);
        match (case.array("input").unwrap(), case.array("taps").unwrap()) {
            (Array::CF32(x), Array::F32(h)) => {
                assert!(is_tolerance(&case));
                let new = || {
                    HalfbandState::<C32, _, _>::new(h.clone(), vec![C32::ZERO; h.len() - 1])
                        .unwrap()
                };
                let mut step =
                    |s: &mut HalfbandState<C32, Vec<f32>, Vec<C32>>, i: &[C32], o: &mut [C32]| {
                        HalfbandDecimate::<C32>::halfband_decimate(&Sdrk, s, i, o).unwrap()
                    };
                let whole = run_whole(&mut new(), x, want.len() + 3, &mut step);
                let chunked = run_chunked(&mut new(), x, plan, &mut step);
                check_both(&case, want, &whole, &chunked, false);
                // The fallback is block-size invariant to the bit.
                let mut fb =
                    |s: &mut HalfbandState<C32, Vec<f32>, Vec<C32>>, i: &[C32], o: &mut [C32]| {
                        Fallback.halfband_decimate(s, i, o).unwrap()
                    };
                let whole = run_whole(&mut new(), x, want.len() + 3, &mut fb);
                let chunked = run_chunked(&mut new(), x, plan, &mut fb);
                check_both(&case, want, &whole, &chunked, true);
            }
            (Array::CI16(x), Array::I16(_)) => {
                let h = case.array("taps").unwrap().as_q15().unwrap();
                let new = || {
                    HalfbandState::<CQ15, _, _>::new(h.clone(), vec![CQ15::ZERO; h.len() - 1])
                        .unwrap()
                };
                let mut step = |s: &mut HalfbandState<CQ15, Vec<Q15>, Vec<CQ15>>,
                                i: &[CQ15],
                                o: &mut [CQ15]| {
                    HalfbandDecimate::<CQ15>::halfband_decimate(&Sdrk, s, i, o).unwrap()
                };
                let whole = run_whole(&mut new(), x, want.len() + 3, &mut step);
                let chunked = run_chunked(&mut new(), x, plan, &mut step);
                check_both(&case, want, &whole, &chunked, true);
                // The f64 model: the integer result stays within the arithmetic of 15-bit taps.
                let snr = q15_snr_db(
                    &h,
                    x.iter().map(|s| f64::from(s.re.0)),
                    whole.iter().map(|s| f64::from(s.re.0)),
                    2,
                );
                eprintln!("{name}: SNR against the f64 model {snr:.1} dB");
                assert!(snr >= Q15_MODEL_SNR_DB, "{name}: {snr}");
            }
            _ => panic!("unexpected array types in {name}"),
        }
    }
}

/// Floor of the SNR of the Q15 results against the f64 arithmetic on the same quantized taps
/// and inputs (the output rounding alone limits it to roughly 80 to 90 dB for these signals).
const Q15_MODEL_SNR_DB: f64 = 70.0;

/// SNR in dB of the Q15 outputs `got` (raw values) against `sum h[k] x[2 m - k]` computed in f64
/// on the quantized taps `h` and inputs `x` (raw values, `step` = decimation).
fn q15_snr_db(
    h: &[Q15],
    x: impl Iterator<Item = f64>,
    got: impl Iterator<Item = f64>,
    step: usize,
) -> f64 {
    let x: Vec<f64> = x.collect();
    let (mut sig, mut err) = (0.0, 0.0);
    for (m, g) in got.enumerate() {
        let n = m * step;
        let mut acc = 0.0;
        for (k, t) in h.iter().enumerate() {
            if let Some(i) = n.checked_sub(k) {
                acc += f64::from(t.0) / 32768.0 * x[i];
            }
        }
        let want = acc.clamp(-32768.0, 32767.0);
        sig += want * want;
        err += (want - g) * (want - g);
    }
    10.0 * (sig / err).log10()
}

#[test]
fn halfband_decimate_real_golden() {
    for name in golden::list("halfband_decimate_real").unwrap() {
        let case = golden::load("halfband_decimate_real", &name).unwrap();
        let want = case.array("expected").unwrap();
        let plan = ChunkPlan::new(case.seed, 700);
        match (case.array("input").unwrap(), case.array("taps").unwrap()) {
            (Array::F32(x), Array::F32(h)) => {
                assert!(is_tolerance(&case));
                let new = || {
                    HalfbandRealState::<f32, _, _>::new(h.clone(), vec![0.0; h.len() - 1]).unwrap()
                };
                let mut step = |s: &mut HalfbandRealState<f32, Vec<f32>, Vec<f32>>,
                                i: &[f32],
                                o: &mut [f32]| {
                    HalfbandDecimateReal::<f32>::halfband_decimate_real(&Sdrk, s, i, o).unwrap()
                };
                let whole = run_whole(&mut new(), x, want.len() + 3, &mut step);
                let chunked = run_chunked(&mut new(), x, plan, &mut step);
                check_both(&case, want, &whole, &chunked, false);
                let mut fb = |s: &mut HalfbandRealState<f32, Vec<f32>, Vec<f32>>,
                              i: &[f32],
                              o: &mut [f32]| {
                    Fallback.halfband_decimate_real(s, i, o).unwrap()
                };
                let whole = run_whole(&mut new(), x, want.len() + 3, &mut fb);
                let chunked = run_chunked(&mut new(), x, plan, &mut fb);
                check_both(&case, want, &whole, &chunked, true);
            }
            (Array::I16(x), Array::I16(_)) => {
                let h = case.array("taps").unwrap().as_q15().unwrap();
                let xq: Vec<Q15> = x.iter().map(|&v| Q15(v)).collect();
                let new = || {
                    HalfbandRealState::<Q15, _, _>::new(h.clone(), vec![Q15(0); h.len() - 1])
                        .unwrap()
                };
                let mut step = |s: &mut HalfbandRealState<Q15, Vec<Q15>, Vec<Q15>>,
                                i: &[Q15],
                                o: &mut [Q15]| {
                    HalfbandDecimateReal::<Q15>::halfband_decimate_real(&Sdrk, s, i, o).unwrap()
                };
                let whole = run_whole(&mut new(), &xq, want.len() + 3, &mut step);
                let chunked = run_chunked(&mut new(), &xq, plan, &mut step);
                check_both(&case, want, &whole, &chunked, true);
                let snr = q15_snr_db(
                    &h,
                    x.iter().map(|&v| f64::from(v)),
                    whole.iter().map(|s| f64::from(s.0)),
                    2,
                );
                eprintln!("{name}: SNR against the f64 model {snr:.1} dB");
                assert!(snr >= Q15_MODEL_SNR_DB, "{name}: {snr}");
            }
            _ => panic!("unexpected array types in {name}"),
        }
    }
}

#[test]
fn real_to_complex_golden() {
    for name in golden::list("real_to_complex").unwrap() {
        let case = golden::load("real_to_complex", &name).unwrap();
        let want = case.array("expected").unwrap();
        let plan = ChunkPlan::new(case.seed, 700);
        match (case.array("input").unwrap(), case.array("taps").unwrap()) {
            (Array::F32(x), Array::F32(h)) => {
                assert!(is_tolerance(&case));
                let new = || {
                    RealToComplexState::<C32, _, _>::new(h.clone(), vec![C32::ZERO; h.len() - 1])
                        .unwrap()
                };
                let mut step = |s: &mut RealToComplexState<C32, Vec<f32>, Vec<C32>>,
                                i: &[f32],
                                o: &mut [C32]| {
                    RealToComplex::<C32>::real_to_complex(&Sdrk, s, i, o).unwrap()
                };
                let whole = run_whole(&mut new(), x, want.len() + 3, &mut step);
                let chunked = run_chunked(&mut new(), x, plan, &mut step);
                check_both(&case, want, &whole, &chunked, false);
                let mut fb = |s: &mut RealToComplexState<C32, Vec<f32>, Vec<C32>>,
                              i: &[f32],
                              o: &mut [C32]| {
                    Fallback.real_to_complex(s, i, o).unwrap()
                };
                let whole = run_whole(&mut new(), x, want.len() + 3, &mut fb);
                let chunked = run_chunked(&mut new(), x, plan, &mut fb);
                check_both(&case, want, &whole, &chunked, true);
            }
            (Array::I16(x), Array::I16(_)) => {
                assert!(x.contains(&-32768), "the input must hold -32768");
                let h = case.array("taps").unwrap().as_q15().unwrap();
                let xq: Vec<Q15> = x.iter().map(|&v| Q15(v)).collect();
                let new = || {
                    RealToComplexState::<CQ15, _, _>::new(h.clone(), vec![CQ15::ZERO; h.len() - 1])
                        .unwrap()
                };
                let mut step = |s: &mut RealToComplexState<CQ15, Vec<Q15>, Vec<CQ15>>,
                                i: &[Q15],
                                o: &mut [CQ15]| {
                    RealToComplex::<CQ15>::real_to_complex(&Sdrk, s, i, o).unwrap()
                };
                let whole = run_whole(&mut new(), &xq, want.len() + 3, &mut step);
                let chunked = run_chunked(&mut new(), &xq, plan, &mut step);
                check_both(&case, want, &whole, &chunked, true);
            }
            _ => panic!("unexpected array types in {name}"),
        }
    }
}
