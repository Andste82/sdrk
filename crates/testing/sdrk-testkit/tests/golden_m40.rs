//! Golden tests of M40: the design `design::rrc` against the formula of the generator, the RRC
//! matched pair, and the kernel `costas` through the facade `Sdrk`, `Fallback` and `Reference`.
//!
//! Every `costas` case is run in one piece, one symbol per call (which also checks the lock flag
//! after every symbol) and in 20 chunk plans (seed 559, chunks of 1 to 700 symbols). Every run
//! must match the golden file bit for bit (the exact float32 model) and end in the stored state.
//! The state is built from the parameters stored in the case, and its loop constants must equal
//! the IEEE bits the generator computed. Every case checks its `snr_db_guard` against its own
//! float64 loop on the stored inputs (the regression guard of decision D111).
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

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_blocks::design::{rrc, rrc_len};
use sdrk_core::kernels::costas::{Costas, CostasState};
use sdrk_core::tables::COS_F32;
use sdrk_core::{C32, Fallback, Reference};
use sdrk_testkit::golden::{self, Array, GoldenCase, Tolerance};
use sdrk_testkit::{ChunkPlan, bit_exact};
use sha2::{Digest, Sha256};

/// The seed of the chunk plans (test 6 of M40); plan `k` uses `PLAN_SEED * 100 + k`.
const PLAN_SEED: u64 = 559;
/// Chunk plans per case.
const PLANS: u64 = 20;
/// The largest chunk of the plans.
const MAX_CHUNK: usize = 700;

const CASES: [&str; 3] = [
    "bpsk_lock_c32_8192",
    "qpsk_lock_c32_8192",
    "qpsk_offset_c32_8192",
];

const RRC_CASES: [(&str, f64, usize, usize); 3] = [
    ("rrc_0p35_sps4_span10", 0.35, 4, 10),
    ("rrc_0p5_sps8_span8", 0.5, 8, 8),
    ("rrc_1p0_sps10_span6", 1.0, 10, 6),
];

#[test]
fn the_cases_of_m40_are_present() {
    assert_eq!(golden::list("costas").unwrap(), CASES);
    let design = golden::list("design").unwrap();
    for (name, ..) in RRC_CASES {
        assert!(design.iter().any(|n| n == name), "missing design/{name}");
    }
}

fn f64_array(case: &GoldenCase, name: &str) -> Vec<f64> {
    match case.array(name).unwrap() {
        Array::F64(v) => v.clone(),
        other => panic!("{name} must be f64, is {}", other.dtype()),
    }
}

fn c32_array(case: &GoldenCase, name: &str) -> Vec<C32> {
    match case.array(name).unwrap() {
        Array::CF32(v) => v.clone(),
        other => panic!("{name} must be cf32, is {}", other.dtype()),
    }
}

#[test]
fn rrc_matches_the_formula_of_the_generator() {
    for (name, rolloff, sps, span) in RRC_CASES {
        let case = golden::load("design", name).unwrap();
        assert_eq!(case.param_f64("rolloff").unwrap(), rolloff);
        assert_eq!(case.param_u64("samples_per_symbol").unwrap(), sps as u64);
        assert_eq!(case.param_u64("span").unwrap(), span as u64);
        let n = rrc_len(sps, span).unwrap();
        assert_eq!(case.param_u64("num_taps").unwrap(), n as u64);
        let mut taps = vec![0.0; n];
        rrc(rolloff, sps, span, &mut taps).unwrap();
        let want = f64_array(&case, "expected");
        let Tolerance::MaxAbsErr { max, .. } = case.tolerance else {
            panic!("design cases use max_abs_err");
        };
        let err = taps
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(err <= max, "{name}: largest error {err}");
        // Symmetric, unit energy, the peak in the centre.
        for k in 0..n {
            assert_eq!(taps[k], taps[n - 1 - k], "{name} tap {k}");
        }
        assert!((taps.iter().map(|h| h * h).sum::<f64>() - 1.0).abs() < 1e-12);
        let c = (n - 1) / 2;
        assert!(taps.iter().all(|&h| h <= taps[c]));
    }
}

/// `rrc * rrc` sampled at the symbol centres: the centre and the largest other value.
fn matched_pair(rolloff: f64, sps: usize, span: usize) -> (f64, f64) {
    let n = rrc_len(sps, span).unwrap();
    let mut h = vec![0.0; n];
    rrc(rolloff, sps, span, &mut h).unwrap();
    // g[lag] = sum_k h[k] h[k + lag] (the taps are symmetric, so this is the convolution).
    let g = |lag: usize| -> f64 { h.iter().zip(&h[lag..]).map(|(a, b)| a * b).sum() };
    let isi = (1..=span).map(|m| g(m * sps).abs()).fold(0.0, f64::max);
    (g(0), isi)
}

/// Test 1 of M40, the RRC matched pair: 1 at the centre, and below 1e-2 at the other symbol
/// centres for every span of at least 10 (D626). The response truncated to `span` symbols leaves
/// 1.5e-3 to 6.1e-3 there at span 10 for the roll-offs 0.35 to 1 at 2 to 10 samples per symbol,
/// and the value falls with the span in an oscillating way, so the 1e-3 of the first
/// specification is checked only on the configurations whose truncation reaches it (D605).
#[test]
fn rrc_matched_pair_is_nyquist_at_the_symbol_centres() {
    let tight: &[(f64, &[usize])] = &[
        (1.0, &[10, 11, 12, 13, 16, 20, 24, 32]),
        (0.5, &[12, 16, 20, 24, 28, 32]),
        (0.35, &[22, 23, 28, 29, 32]),
    ];
    for &(rolloff, spans) in tight {
        for &span in spans {
            for sps in [4usize, 8, 10] {
                let (centre, isi) = matched_pair(rolloff, sps, span);
                assert!(
                    (centre - 1.0).abs() < 1e-12,
                    "{rolloff} {sps} {span}: {centre}"
                );
                assert!(isi < 1e-3, "{rolloff} sps {sps} span {span}: {isi}");
            }
        }
    }
    for rolloff in [0.2, 0.35, 0.5, 0.7, 1.0] {
        for span in 10..=32 {
            for sps in [2usize, 4, 8] {
                let (centre, isi) = matched_pair(rolloff, sps, span);
                assert!(
                    (centre - 1.0).abs() < 1e-12,
                    "{rolloff} {sps} {span}: {centre}"
                );
                assert!(isi < 1e-2, "{rolloff} sps {sps} span {span}: {isi}");
            }
        }
    }
}

#[test]
fn rrc_rejects_its_bounds() {
    let mut h = vec![0.0; rrc_len(4, 10).unwrap()];
    assert!(rrc(0.049, 4, 10, &mut h).is_err());
    assert!(rrc(1.01, 4, 10, &mut h).is_err());
    assert!(rrc(f64::NAN, 4, 10, &mut h).is_err());
    assert!(rrc(0.35, 4, 10, &mut h[1..]).is_err());
    assert!(rrc_len(1, 10).is_err() && rrc_len(1025, 10).is_err());
    assert!(rrc_len(4, 3).is_err() && rrc_len(4, 33).is_err());
    assert_eq!(rrc_len(1024, 32).unwrap(), 32 * 1024 + 1);
}

/// The state of the case; its loop constants must be the generator's bits.
fn state_of(case: &GoldenCase) -> CostasState {
    let p = |key: &str| case.param_f64(key).unwrap();
    let st = CostasState::new(
        case.param_u64("order").unwrap() as u32,
        p("loop_bw"),
        p("damping"),
        p("max_offset"),
    )
    .unwrap();
    let g = st.gains();
    let bits = |key: &str| case.param_u64(key).unwrap() as u32;
    assert_eq!(g.kp.to_bits(), bits("kp_bits"), "{}: kp", case.case);
    assert_eq!(g.ki.to_bits(), bits("ki_bits"), "{}: ki", case.case);
    assert_eq!(
        g.max_offset.to_bits(),
        bits("max_bits"),
        "{}: max",
        case.case
    );
    st
}

fn check_final(case: &GoldenCase, st: &CostasState, what: &str) {
    let bits = |key: &str| case.param_u64(key).unwrap();
    assert_eq!(
        u64::from(st.phase()),
        bits("final_phase"),
        "{}: {what}",
        case.case
    );
    assert_eq!(
        u64::from((st.frequency() as f32).to_bits()),
        bits("final_freq_bits"),
        "{}: {what}",
        case.case
    );
    assert_eq!(
        u64::from(st.lock_level().to_bits()),
        bits("final_level_bits"),
        "{}: {what}",
        case.case
    );
    assert_eq!(
        u64::from(st.locked()),
        bits("final_locked"),
        "{}: {what}",
        case.case
    );
}

fn run<K: Costas>(
    kernel: &K,
    case: &GoldenCase,
    input: &[C32],
    plan: Option<u64>,
) -> (Vec<C32>, CostasState) {
    let mut st = state_of(case);
    let mut out = vec![C32::default(); input.len()];
    match plan {
        None => kernel.costas(&mut st, input, &mut out).unwrap(),
        Some(k) => {
            let mut sizes = ChunkPlan::new(PLAN_SEED * 100 + k, MAX_CHUNK).sizes();
            let mut pos = 0;
            while pos < input.len() {
                let a = sizes.next().unwrap().min(input.len() - pos);
                kernel
                    .costas(&mut st, &input[pos..pos + a], &mut out[pos..pos + a])
                    .unwrap();
                pos += a;
            }
        }
    }
    (out, st)
}

#[test]
fn costas_matches_the_float32_model() {
    for name in CASES {
        let case = golden::load("costas", name).unwrap();
        let input = c32_array(&case, "input");
        let want = c32_array(&case, "expected");
        let (a, sa) = run(&Sdrk, &case, &input, None);
        assert_eq!(bit_exact(&a, &want), None, "{name} facade");
        check_final(&case, &sa, "facade");
        let (b, sb) = run(&Fallback, &case, &input, None);
        assert_eq!(bit_exact(&b, &want), None, "{name} fallback");
        assert_eq!(sa, sb);
        let (c, sc) = run(&Reference, &case, &input, None);
        assert_eq!(bit_exact(&c, &want), None, "{name} reference");
        assert_eq!(sa, sc);
        for k in 0..PLANS {
            let (d, sd) = run(&Sdrk, &case, &input, Some(k));
            assert_eq!(bit_exact(&d, &want), None, "{name} plan {k}");
            assert_eq!(sa, sd, "{name} plan {k}");
        }
    }
}

#[test]
fn the_lock_flags_follow_the_model_symbol_by_symbol() {
    for name in CASES {
        let case = golden::load("costas", name).unwrap();
        let input = c32_array(&case, "input");
        let Array::U8(flags) = case.array("locked").unwrap() else {
            panic!("locked must be u8");
        };
        let mut st = state_of(&case);
        let mut y = [C32::default(); 1];
        for (n, (x, &f)) in input.iter().zip(flags).enumerate() {
            Sdrk.costas(&mut st, core::slice::from_ref(x), &mut y)
                .unwrap();
            assert_eq!(u8::from(st.locked()), f, "{name}: symbol {n}");
        }
        let first = flags.iter().position(|&f| f == 1).unwrap();
        assert_eq!(first as u64, case.param_u64("first_locked").unwrap());
        check_final(&case, &st, "symbol by symbol");
    }
}

/// The loop of section 2 in f64 with exact sine and cosine of the `u32` phase.
fn costas_f64(case: &GoldenCase, input: &[C32]) -> Vec<Complex64> {
    let st = state_of(case);
    let g = st.gains();
    let (kp, ki, mx) = (f64::from(g.kp), f64::from(g.ki), f64::from(g.max_offset));
    let order = st.order();
    let sg = |u: f64| {
        if u > 0.0 {
            1.0
        } else if u < 0.0 {
            -1.0
        } else {
            0.0
        }
    };
    let (mut phase, mut freq) = (0u32, 0.0f64);
    let mut out = Vec::with_capacity(input.len());
    for x in input {
        let theta = core::f64::consts::TAU * (f64::from(phase) / 4_294_967_296.0);
        let (s, c) = theta.sin_cos();
        let (xr, xi) = (f64::from(x.re), f64::from(x.im));
        let (yr, yi) = (xr * c + xi * s, xi * c - xr * s);
        out.push(Complex64::new(yr, yi));
        let e = if order == 2 {
            yr * yi
        } else {
            sg(yr) * yi - sg(yi) * yr
        };
        freq = (freq + ki * e).clamp(-mx, mx);
        phase = phase.wrapping_add(((freq + kp * e) * 4_294_967_296.0 + 0.5).floor() as i64 as u32);
    }
    out
}

#[test]
fn costas_stays_above_its_f64_guard() {
    for name in CASES {
        let case = golden::load("costas", name).unwrap();
        let input = c32_array(&case, "input");
        let reference = costas_f64(&case, &input);
        let (got, _) = run(&Sdrk, &case, &input, None);
        let skip = case.param_u64("guard_skip").unwrap() as usize;
        let num: f64 = reference[skip..].iter().map(Complex64::norm_sqr).sum();
        let den: f64 = reference[skip..]
            .iter()
            .zip(&got[skip..])
            .map(|(r, y)| (r - Complex64::new(f64::from(y.re), f64::from(y.im))).norm_sqr())
            .sum();
        let snr = 10.0 * (num / den).log10();
        let guard = case.param_f64("snr_db_guard").unwrap();
        assert!(snr >= guard, "{name}: {snr} dB below the guard {guard} dB");
    }
}

#[test]
fn the_cosine_table_is_the_one_of_the_generator() {
    let hash = Sha256::digest(bytemuck::cast_slice::<f32, u8>(&COS_F32));
    let hex = hash.iter().fold(String::new(), |mut s, b| {
        use std::fmt::Write;
        write!(s, "{b:02x}").unwrap();
        s
    });
    for name in CASES {
        let case = golden::load("costas", name).unwrap();
        assert_eq!(case.param_str("cos_f32_sha256").unwrap(), hex);
    }
}
