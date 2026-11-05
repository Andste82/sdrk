//! Golden tests of the M33 kernels `noise_blank` and `auto_notch` through the facade `Sdrk`,
//! `Fallback` and `Reference`.
//!
//! Every case is run in one piece and in chunks of random size (the plan seed is the case seed);
//! both runs must match the golden file bit for bit. The `noise_blank` cases also check the
//! count of blanked samples and that the coefficients of the state equal the ones of the
//! generator. Every case with `snr_db_guard` is checked against its own float64 loop on the
//! stored inputs (the regression guard of decision D111); a guard of `GUARD_IDENTICAL_DB` means
//! the model output equals the float64 output exactly (D372).
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_core::kernels::auto_notch::{AutoNotch, AutoNotchState};
use sdrk_core::kernels::noise_blank::{NoiseBlank, NoiseBlankState};
use sdrk_core::{C32, CQ15, ComplexSample, Fallback, Rate, Reference};
use sdrk_testkit::golden::{self, Array, GoldenCase};
use sdrk_testkit::{ChunkPlan, bit_exact};

/// The stored guard of a case whose model output equals the float64 output (D372).
const GUARD_IDENTICAL_DB: f64 = 300.0;

/// The chunk bounds of a run: the whole signal at once, or the sizes of the case's plan.
fn chunks(len: usize, plan: Option<ChunkPlan>) -> Vec<(usize, usize)> {
    let mut sizes = plan.map(|p| p.sizes());
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < len {
        let n = sizes
            .as_mut()
            .map_or(len - pos, |s| s.next().unwrap())
            .min(len - pos);
        out.push((pos, pos + n));
        pos += n;
    }
    out
}

fn plan_of(case: &GoldenCase) -> ChunkPlan {
    ChunkPlan::new(case.seed, 700)
}

fn usize_param(case: &GoldenCase, key: &str) -> usize {
    usize::try_from(case.param_u64(key).unwrap()).unwrap()
}

fn snr_db(reference: &[Complex64], test: &[Complex64]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (r, t) in reference.iter().zip(test) {
        num += r.norm_sqr();
        den += (r - t).norm_sqr();
    }
    if den == 0.0 {
        f64::INFINITY
    } else {
        10.0 * libm::log10(num / den)
    }
}

/// The SNR of `got` against the float64 output `want` from `guard_skip` on must reach the
/// stored guard and stay within 3 dB of it; the identical-output guard needs identical outputs.
fn check_guard(case: &GoldenCase, want: &[Complex64], got: &[Complex64]) {
    let skip = usize_param(case, "guard_skip");
    let guard = case.param_f64("snr_db_guard").unwrap();
    let snr = snr_db(&want[skip..], &got[skip..]);
    if guard == GUARD_IDENTICAL_DB {
        assert!(
            snr.is_infinite(),
            "{}: the output differs from the f64 loop (SNR {snr:.2} dB), the generator found it identical",
            case.case
        );
    } else {
        assert!(
            snr >= guard,
            "{}: SNR {snr:.2} dB against the f64 loop is below the guard {guard} dB",
            case.case
        );
        // The guard is the measured value minus 1 dB, so it is meaningful, not vacuous.
        assert!(
            snr < guard + 3.0,
            "{}: SNR {snr:.2}, guard {guard}",
            case.case
        );
    }
}

#[test]
fn the_cases_of_m33_are_present() {
    assert_eq!(
        golden::list("noise_blank").unwrap(),
        ["nb_c32_16384", "nb_clean_c32_4096", "nb_cq15_16384"]
    );
    assert_eq!(
        golden::list("auto_notch").unwrap(),
        ["an_sweep_f32_16384", "an_tone_f32_16384"]
    );
}

// --------------------------------------------------------------------------------------------
// noise_blank
// --------------------------------------------------------------------------------------------

fn nb_state<T: ComplexSample>(case: &GoldenCase) -> NoiseBlankState<T, Vec<T>> {
    let rate = Rate::hz(case.param_u64("rate_hz").unwrap()).unwrap();
    let state = NoiseBlankState::new(
        case.param_f64("threshold_db").unwrap(),
        case.param_f64("tau_s").unwrap(),
        case.param_f64("hold_s").unwrap(),
        vec![T::ZERO; usize_param(case, "lookahead")],
        rate,
    )
    .unwrap();
    assert_eq!(state.hold(), case.param_u64("hold").unwrap());
    state
}

/// Runs `kernel` whole and chunked; both must match `expected` and blank the stored count.
fn nb_check<T: ComplexSample, K: NoiseBlank<T>>(
    kernel: &K,
    case: &GoldenCase,
    input: &[T],
) -> Vec<T> {
    let want = case.array("expected").unwrap();
    let blanked = case.param_u64("blanked").unwrap();
    let mut runs = Vec::new();
    for plan in [None, Some(plan_of(case))] {
        let mut state = nb_state::<T>(case);
        let mut out = vec![T::ZERO; input.len()];
        for (a, b) in chunks(input.len(), plan) {
            kernel
                .noise_blank(&mut state, &input[a..b], &mut out[a..b])
                .unwrap();
        }
        golden::assert_matches(case, want, &out).unwrap();
        assert_eq!(state.blanked(), blanked, "{}: blanked count", case.case);
        runs.push(out);
    }
    assert_eq!(
        bit_exact(&runs[0], &runs[1]),
        None,
        "{}: block size changed the output",
        case.case
    );
    runs.swap_remove(0)
}

fn c32_input(case: &GoldenCase) -> Vec<C32> {
    match case.array("input").unwrap() {
        Array::CF32(v) => v.clone(),
        other => panic!("cf32 expected, got {}", other.dtype()),
    }
}

fn cq15_input(case: &GoldenCase) -> Vec<CQ15> {
    match case.array("input").unwrap() {
        Array::CI16(v) => v.clone(),
        other => panic!("ci16 expected, got {}", other.dtype()),
    }
}

#[test]
fn noise_blank_c32_golden() {
    for name in ["nb_c32_16384", "nb_clean_c32_4096"] {
        let case = golden::load("noise_blank", name).unwrap();
        let x = c32_input(&case);
        let facade = nb_check(&Sdrk, &case, &x);
        let fallback = nb_check(&Fallback, &case, &x);
        let reference = nb_check(&Reference, &case, &x);
        assert_eq!(bit_exact(&facade, &fallback), None);
        assert_eq!(bit_exact(&facade, &reference), None);
    }
}

#[test]
fn noise_blank_cq15_golden() {
    let case = golden::load("noise_blank", "nb_cq15_16384").unwrap();
    let x = cq15_input(&case);
    let facade = nb_check(&Sdrk, &case, &x);
    let fallback = nb_check(&Fallback, &case, &x);
    let reference = nb_check(&Reference, &case, &x);
    assert_eq!(bit_exact(&facade, &fallback), None);
    assert_eq!(bit_exact(&facade, &reference), None);
}

#[test]
fn a_clean_signal_passes_delayed_by_the_lookahead() {
    let case = golden::load("noise_blank", "nb_clean_c32_4096").unwrap();
    let x = c32_input(&case);
    let d = usize_param(&case, "lookahead");
    let out = nb_check(&Sdrk, &case, &x);
    assert!(out[..d].iter().all(|s| *s == C32::default()));
    assert_eq!(bit_exact(&out[d..], &x[..x.len() - d]), None);
}

/// The f64 coefficients of the kernel, computed as `NoiseBlankState::new` documents them.
struct NbCoefficients {
    alpha: f64,
    k: f64,
    d: usize,
    span: usize,
}

fn nb_coefficients(case: &GoldenCase) -> NbCoefficients {
    let fs = case.param_u64("rate_hz").unwrap() as f64;
    let d = usize_param(case, "lookahead");
    NbCoefficients {
        alpha: 1.0 - libm::exp(-1.0 / (fs * case.param_f64("tau_s").unwrap())),
        k: libm::pow(10.0, case.param_f64("threshold_db").unwrap() / 10.0),
        d,
        span: d + usize_param(case, "hold") + 1,
    }
}

#[test]
fn the_noise_blank_coefficients_equal_the_ones_of_the_generator() {
    for name in golden::list("noise_blank").unwrap() {
        let case = golden::load("noise_blank", &name).unwrap();
        let c = nb_coefficients(&case);
        let fs = case.param_u64("rate_hz").unwrap() as f64;
        assert_eq!(
            c.alpha.to_bits(),
            case.param_u64("alpha_bits").unwrap(),
            "{name}: alpha"
        );
        assert_eq!(
            c.k.to_bits(),
            case.param_u64("k_bits").unwrap(),
            "{name}: k (libm pow?)"
        );
        assert_eq!(
            libm::floor(c.alpha * 2_147_483_648.0 + 0.5) as u64,
            case.param_u64("alpha_q31").unwrap()
        );
        assert_eq!(
            libm::floor(c.k * 65_536.0 + 0.5) as u64,
            case.param_u64("k_q16").unwrap()
        );
        let lookahead = libm::floor(case.param_f64("lookahead_s").unwrap() * fs + 0.5) as u64;
        assert_eq!(lookahead, case.param_u64("lookahead").unwrap());
        let state = nb_state::<C32>(&case);
        assert_eq!(state.lookahead() as u64, lookahead);
    }
}

/// The loop of the kernel in float64 (same rules, no flush), as the generator runs it.
fn noise_blank_f64(c: &NbCoefficients, input: &[Complex64]) -> Vec<Complex64> {
    let (mut p, mut count) = (0.0f64, 0usize);
    let mut ring = std::collections::VecDeque::from(vec![Complex64::new(0.0, 0.0); c.d]);
    let mut out = Vec::with_capacity(input.len());
    for &v in input {
        let q = v.re * v.re + v.im * v.im;
        let kp = c.k * p;
        let impulse = !q.is_finite() || (p > 0.0 && q > kp);
        p = if impulse {
            p + c.alpha * (kp - p)
        } else if p > 0.0 {
            p + c.alpha * (q - p)
        } else {
            q
        };
        let y = if c.d == 0 {
            v
        } else {
            ring.push_back(v);
            ring.pop_front().unwrap()
        };
        if impulse {
            count = c.span;
        }
        if count > 0 {
            count -= 1;
            out.push(Complex64::new(0.0, 0.0));
        } else {
            out.push(y);
        }
    }
    out
}

#[test]
fn the_noise_blank_output_stays_within_the_f64_guard() {
    for name in ["nb_c32_16384", "nb_cq15_16384"] {
        let case = golden::load("noise_blank", name).unwrap();
        let c = nb_coefficients(&case);
        let (x, got): (Vec<Complex64>, Vec<Complex64>) = if name.contains("cq15") {
            let x = cq15_input(&case);
            let mut out = vec![CQ15::default(); x.len()];
            Sdrk.noise_blank(&mut nb_state(&case), &x, &mut out)
                .unwrap();
            (
                x.iter().map(|s| s.to_c64()).collect(),
                out.iter().map(|s| s.to_c64()).collect(),
            )
        } else {
            let x = c32_input(&case);
            let mut out = vec![C32::default(); x.len()];
            Sdrk.noise_blank(&mut nb_state(&case), &x, &mut out)
                .unwrap();
            (
                x.iter().map(|s| s.to_c64()).collect(),
                out.iter().map(|s| s.to_c64()).collect(),
            )
        };
        check_guard(&case, &noise_blank_f64(&c, &x), &got);
    }
}

// --------------------------------------------------------------------------------------------
// auto_notch
// --------------------------------------------------------------------------------------------

fn an_state(case: &GoldenCase) -> AutoNotchState<Vec<f32>, Vec<f32>> {
    let (taps, delay) = (usize_param(case, "taps"), usize_param(case, "delay"));
    AutoNotchState::new(
        vec![0.0; taps],
        vec![0.0; taps + delay],
        delay,
        case.param_f64("mu").unwrap() as f32,
    )
    .unwrap()
}

fn an_input(case: &GoldenCase) -> Vec<f32> {
    match case.array("input").unwrap() {
        Array::F32(v) => v.clone(),
        other => panic!("f32 expected, got {}", other.dtype()),
    }
}

fn an_check<K: AutoNotch>(kernel: &K, case: &GoldenCase, input: &[f32]) -> Vec<f32> {
    let want = case.array("expected").unwrap();
    let mut runs = Vec::new();
    for plan in [None, Some(plan_of(case))] {
        let mut state = an_state(case);
        let mut out = vec![0.0f32; input.len()];
        for (a, b) in chunks(input.len(), plan) {
            kernel
                .auto_notch(&mut state, &input[a..b], &mut out[a..b])
                .unwrap();
        }
        golden::assert_matches(case, want, &out).unwrap();
        runs.push(out);
    }
    assert_eq!(
        bit_exact(&runs[0], &runs[1]),
        None,
        "{}: block size changed the output",
        case.case
    );
    runs.swap_remove(0)
}

#[test]
fn auto_notch_golden() {
    for name in golden::list("auto_notch").unwrap() {
        let case = golden::load("auto_notch", &name).unwrap();
        let x = an_input(&case);
        let facade = an_check(&Sdrk, &case, &x);
        let fallback = an_check(&Fallback, &case, &x);
        let reference = an_check(&Reference, &case, &x);
        assert_eq!(bit_exact(&facade, &fallback), None);
        assert_eq!(bit_exact(&facade, &reference), None);
    }
}

/// The normalized LMS loop in float64 (same order, no flush), as the generator runs it.
fn auto_notch_f64(input: &[f64], taps: usize, delay: usize, mu: f64) -> Vec<f64> {
    let mut w = vec![0.0f64; taps];
    let mut buf = vec![0.0f64; taps + delay];
    let mut out = Vec::with_capacity(input.len());
    for (n, &x) in input.iter().enumerate() {
        let base = taps + n;
        let u: Vec<f64> = (0..taps).map(|i| buf[base - i]).collect();
        let (mut y, mut power) = (0.0, 0.0);
        for (wi, ui) in w.iter().zip(&u) {
            y += wi * ui;
        }
        for ui in &u {
            power += ui * ui;
        }
        let e = x - y;
        let f = (mu * e) / (power + 1e-6);
        for (wi, ui) in w.iter_mut().zip(&u) {
            *wi += f * ui;
        }
        out.push(e);
        buf.push(x);
    }
    out
}

#[test]
fn the_auto_notch_output_stays_within_the_f64_guard() {
    for name in golden::list("auto_notch").unwrap() {
        let case = golden::load("auto_notch", &name).unwrap();
        let x = an_input(&case);
        let mut out = vec![0.0f32; x.len()];
        Sdrk.auto_notch(&mut an_state(&case), &x, &mut out).unwrap();
        let x64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let want = auto_notch_f64(
            &x64,
            usize_param(&case, "taps"),
            usize_param(&case, "delay"),
            case.param_f64("mu").unwrap(),
        );
        let as_c = |v: &[f64]| {
            v.iter()
                .map(|&r| Complex64::new(r, 0.0))
                .collect::<Vec<_>>()
        };
        let got: Vec<f64> = out.iter().map(|&v| f64::from(v)).collect();
        check_guard(&case, &as_c(&want), &as_c(&got));
    }
}
