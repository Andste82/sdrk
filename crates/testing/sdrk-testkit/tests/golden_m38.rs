//! Golden tests of the M38 kernel `symbol_sync` through the facade `Sdrk`, `Fallback` and
//! `Reference`.
//!
//! Every case is run in one piece (one call with room for every output) and in 20 chunk plans
//! (seed 493): each call gets an input chunk of 1 to 700 samples and an output chunk of 1 to 200
//! samples, so calls also end with a full output and the caller passes the unconsumed input
//! again. Every run must match the golden file bit for bit (the exact float32 model), produce the
//! stored count and end in the same state. The state is built from the parameters stored in the
//! case, and its loop constants must equal the IEEE bits the generator computed. Every case
//! checks its `snr_db_guard` against its own float64 loop on the stored inputs (the regression
//! guard of decision D111).
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names
)]

use sdrk::Sdrk;
use sdrk_core::kernels::symbol_sync::{SymbolSync, SymbolSyncState, Ted};
use sdrk_core::{C32, Fallback, Reference, Sample};
use sdrk_testkit::golden::{self, Array, GoldenCase};
use sdrk_testkit::{ChunkPlan, bit_exact};

/// The seed of the chunk plans (test 1 of M38); plan `k` uses `PLAN_SEED * 100 + k`.
const PLAN_SEED: u64 = 493;
/// Chunk plans per kernel and case.
const PLANS: u64 = 20;
/// The largest input and output chunks of the plans.
const MAX_IN: usize = 700;
const MAX_OUT: usize = 200;

const CASES: [&str; 3] = [
    "gardner_c32_sps4p1_16384",
    "gardner_f32_sps8_8192",
    "mm_c32_sps2_8192",
];

#[test]
fn the_cases_of_m38_are_present() {
    assert_eq!(golden::list("symbol_sync").unwrap(), CASES);
}

fn ted_of(case: &GoldenCase) -> Ted {
    match case.param_str("ted").unwrap() {
        "gardner" => Ted::Gardner,
        "mm" => Ted::MuellerMuller,
        other => panic!("{}: unknown detector {other}", case.case),
    }
}

/// The state of the case; its loop constants must be the generator's bits.
fn state_of<T: Sample>(case: &GoldenCase) -> SymbolSyncState<T> {
    let p = |key: &str| case.param_f64(key).unwrap();
    let st = SymbolSyncState::<T>::new(
        p("sps"),
        ted_of(case),
        p("loop_bw"),
        p("damping"),
        p("max_deviation"),
        p("ted_gain"),
    )
    .unwrap();
    let g = st.gains();
    let bits = |key: &str| case.param_u64(key).unwrap();
    assert_eq!(g.kp.to_bits(), bits("kp_bits"), "{}: kp", case.case);
    assert_eq!(g.ki.to_bits(), bits("ki_bits"), "{}: ki", case.case);
    assert_eq!(
        g.max_integ.to_bits(),
        bits("max_integ_bits"),
        "{}: integrator limit",
        case.case
    );
    st
}

/// One run: the whole input in one call, or the chunks of plan `k`. Returns the outputs and the
/// final state.
fn run<T: Sample, K: SymbolSync<T>>(
    kernel: &K,
    case: &GoldenCase,
    input: &[T],
    plan: Option<u64>,
) -> (Vec<T>, SymbolSyncState<T>) {
    let mut st = state_of::<T>(case);
    let mut out = Vec::new();
    match plan {
        None => {
            let mut buf = vec![T::ZERO; input.len()];
            let p = kernel.symbol_sync(&mut st, input, &mut buf).unwrap();
            assert_eq!(p.consumed, input.len(), "{}: whole run consumed", case.case);
            out.extend_from_slice(&buf[..p.produced]);
        }
        Some(k) => {
            let mut ins = ChunkPlan::new(PLAN_SEED * 100 + k, MAX_IN).sizes();
            let mut outs = ChunkPlan::new(PLAN_SEED * 100 + 50 + k, MAX_OUT).sizes();
            let mut buf = [T::ZERO; MAX_OUT];
            let mut pos = 0;
            while pos < input.len() {
                let a = ins.next().unwrap().min(input.len() - pos);
                let m = outs.next().unwrap();
                let p = kernel
                    .symbol_sync(&mut st, &input[pos..pos + a], &mut buf[..m])
                    .unwrap();
                assert!(p.consumed <= a && p.produced <= m);
                // A call that cannot produce takes its whole chunk.
                assert!(p.produced == m || p.consumed == a, "{}: stalled", case.case);
                out.extend_from_slice(&buf[..p.produced]);
                pos += p.consumed;
            }
        }
    }
    (out, st)
}

/// Runs `kernel` whole and in the 20 plans; every run must match `expected`, produce the stored
/// count and end in the state of the whole run.
fn check<T: Sample, K: SymbolSync<T>>(kernel: &K, case: &GoldenCase, input: &[T]) -> Vec<T> {
    let want = case.array("expected").unwrap();
    let produced = usize::try_from(case.param_u64("produced").unwrap()).unwrap();
    let (whole, end) = run(kernel, case, input, None);
    assert_eq!(whole.len(), produced, "{}: output count", case.case);
    golden::assert_matches(case, want, &whole).unwrap();
    for k in 0..PLANS {
        let (out, st) = run(kernel, case, input, Some(k));
        assert_eq!(
            bit_exact(&whole, &out),
            None,
            "{}: plan {k} changed the output",
            case.case
        );
        assert_eq!(
            out.len(),
            whole.len(),
            "{}: plan {k} output count",
            case.case
        );
        assert_eq!(st, end, "{}: plan {k} changed the final state", case.case);
    }
    whole
}

// --------------------------------------------------------------------------------------------
// The float64 regression guard (decision D111)
// --------------------------------------------------------------------------------------------

/// The loop of section 1 in f64, in the operation order of the generator's guard: the
/// interpolation and the detector in f64 with the f64 constants, `mu` not rounded. `x` holds one
/// (`f32`) or two (`C32`) components per sample.
fn sync_f64(x: &[Vec<f64>], ted: Ted, st: &SymbolSyncState<f32>) -> Vec<Vec<f64>> {
    let g = st.gains();
    let (kp, ki, lim, sps) = (g.kp, g.ki, g.max_integ, st.sps());
    let nc = x[0].len();
    let mut b = vec![vec![0.0; nc]; 3];
    b.extend(x.iter().cloned());
    let len = b.len();
    let interp = |i: usize, mu: f64| -> Vec<f64> {
        (0..nc)
            .map(|c| {
                let (xm1, x0, x1, x2) = (b[i - 1][c], b[i][c], b[i + 1][c], b[i + 2][c]);
                let c1 = ((-(xm1 * (1.0 / 3.0)) - x0 * 0.5) + x1) - x2 * (1.0 / 6.0);
                let c2 = (xm1 * 0.5 - x0) + x1 * 0.5;
                let c3 = ((-(xm1 * (1.0 / 6.0)) + x0 * 0.5) - x1 * 0.5) + x2 * (1.0 / 6.0);
                ((c3 * mu + c2) * mu + c1) * mu + x0
            })
            .collect()
    };
    let sign = |v: f64| {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    };
    let (mut i, mut mu, mut omega, mut integ) = (4usize, 0.0f64, sps, 0.0f64);
    let (mut started, mut mid_ready) = (false, false);
    let (mut y_prev, mut d_prev, mut mid) = (vec![0.0; nc], vec![0.0; nc], vec![0.0; nc]);
    let mut out = Vec::new();
    loop {
        if ted == Ted::Gardner && started && !mid_ready {
            let v = mu - omega * 0.5;
            let f = libm::floor(v);
            let im = i.checked_add_signed(f as isize).unwrap();
            if im + 2 >= len {
                break;
            }
            mid = interp(im, v - f);
            mid_ready = true;
        }
        if i + 2 >= len {
            break;
        }
        let y = interp(i, mu);
        let mut e = 0.0;
        if ted == Ted::Gardner {
            if started {
                e = (y_prev[0] - y[0]) * mid[0];
                for c in 1..nc {
                    e += (y_prev[c] - y[c]) * mid[c];
                }
            }
            mid_ready = false;
        } else {
            let d: Vec<f64> = y.iter().map(|v| sign(*v)).collect();
            if started {
                let mut a = d_prev[0] * y[0];
                let mut q = d[0] * y_prev[0];
                for c in 1..nc {
                    a += d_prev[c] * y[c];
                    q += d[c] * y_prev[c];
                }
                e = a - q;
            }
            d_prev = d;
        }
        y_prev.clone_from(&y);
        started = true;
        out.push(y);
        let e = if e.is_finite() { e } else { 0.0 };
        let s = integ + ki * e;
        integ = if s.abs() < 1e-30 { 0.0 } else { s }.max(-lim).min(lim);
        omega = sps + integ;
        let q = omega * 0.25;
        let v = (mu + omega) + (kp * e).max(-q).min(q);
        let f = libm::floor(v);
        i += f as usize;
        mu = v - f;
    }
    out
}

/// The SNR of the model output against the f64 loop over the outputs both produced, from
/// `guard_skip` on, must reach the stored guard and stay below the guard plus 3 dB (so the guard
/// is not vacuous).
fn check_guard(case: &GoldenCase, x: &[Vec<f64>], got: &[Vec<f64>]) {
    let probe = state_of::<f32>(case);
    let want = sync_f64(x, ted_of(case), &probe);
    assert!(
        want.len().abs_diff(got.len()) <= 1,
        "{}: {} outputs of the f64 loop, {} of the model",
        case.case,
        want.len(),
        got.len()
    );
    let skip = usize::try_from(case.param_u64("guard_skip").unwrap()).unwrap();
    let n = want.len().min(got.len());
    let (mut num, mut den) = (0.0, 0.0);
    for (w, g) in want[skip..n].iter().zip(&got[skip..n]) {
        for (a, b) in w.iter().zip(g) {
            num += a * a;
            den += (a - b) * (a - b);
        }
    }
    let snr = 10.0 * libm::log10(num / den);
    let guard = case.param_f64("snr_db_guard").unwrap();
    assert!(
        snr >= guard && snr < guard + 3.0,
        "{}: SNR {snr:.2} dB against the float64 loop, guard {guard} dB",
        case.case
    );
}

// --------------------------------------------------------------------------------------------
// Cases
// --------------------------------------------------------------------------------------------

#[test]
fn symbol_sync_c32_golden() {
    for name in ["gardner_c32_sps4p1_16384", "mm_c32_sps2_8192"] {
        let case = golden::load("symbol_sync", name).unwrap();
        let Array::CF32(input) = case.array("input").unwrap() else {
            panic!("{name}: cf32 input expected");
        };
        let facade = check(&Sdrk, &case, input);
        let fallback = check(&Fallback, &case, input);
        let reference = check(&Reference, &case, input);
        assert_eq!(bit_exact(&facade, &fallback), None);
        assert_eq!(bit_exact(&facade, &reference), None);
        let x: Vec<Vec<f64>> = input
            .iter()
            .map(|v| vec![f64::from(v.re), f64::from(v.im)])
            .collect();
        let got: Vec<Vec<f64>> = facade
            .iter()
            .map(|v: &C32| vec![f64::from(v.re), f64::from(v.im)])
            .collect();
        check_guard(&case, &x, &got);
    }
}

#[test]
fn symbol_sync_f32_golden() {
    let case = golden::load("symbol_sync", "gardner_f32_sps8_8192").unwrap();
    let Array::F32(input) = case.array("input").unwrap() else {
        panic!("f32 input expected");
    };
    let facade = check(&Sdrk, &case, input);
    let fallback = check(&Fallback, &case, input);
    let reference = check(&Reference, &case, input);
    assert_eq!(bit_exact(&facade, &fallback), None);
    assert_eq!(bit_exact(&facade, &reference), None);
    let x: Vec<Vec<f64>> = input.iter().map(|v| vec![f64::from(*v)]).collect();
    let got: Vec<Vec<f64>> = facade.iter().map(|v| vec![f64::from(*v)]).collect();
    check_guard(&case, &x, &got);
}

/// The strobes settle near the symbol centres of the generator (stored as information); the
/// estimate of the samples per symbol stays within the integrator limit.
#[test]
fn the_loops_of_the_cases_settle() {
    for name in CASES {
        let case = golden::load("symbol_sync", name).unwrap();
        let err = case.param_f64("timing_error_last_256").unwrap();
        assert!(err < 0.1, "{name}: timing error {err} symbols");
    }
    let case = golden::load("symbol_sync", "gardner_c32_sps4p1_16384").unwrap();
    let Array::CF32(input) = case.array("input").unwrap() else {
        panic!("cf32 input expected");
    };
    let (_, st) = run(&Sdrk, &case, input, None);
    let sps = case.param_f64("sps").unwrap();
    let lim = st.gains().max_integ;
    assert!((st.sps_estimate() - sps).abs() <= lim);
    // 40 ppm faster transmit clock: the receiver sees fewer samples per symbol.
    assert!(st.sps_estimate() < sps, "{}", st.sps_estimate());
}
