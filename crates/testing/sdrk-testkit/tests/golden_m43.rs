//! Golden tests of the M43 kernels `equalize` (test 1, the chunk invariance of test 6) and
//! `dot_complex` (the reusable complex dot product of decisions D800 and D801) through the facade
//! `Sdrk`, `Fallback` and `Reference`.
//!
//! `equalize`: every case is run in one piece and in 20 chunk plans (seed 595, chunks of 1 to
//! 700 symbols); every run must match the golden file bit for bit (the exact float32 model) and
//! end in the same state: the final weights of the case, the stored `mse` bits, mode, tracker
//! phase and frequency, and the history of the whole run. Every case checks its `snr_db_guard` against its own float64 loop on
//! the stored inputs (the regression guard of decision D111).
//!
//! `dot_complex`: every window of every case must match the golden file bit for bit on every
//! implementation, and the float64 sum of the products checks the stored `snr_db_guard`.
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

use sdrk::Sdrk;
use sdrk_core::kernels::dot::{DOT_COMPLEX_PARTIALS, DotComplex};
use sdrk_core::kernels::equalize::{EqConstellation, EqMode, EqParams, Equalize, EqualizeState};
use sdrk_core::{C32, Fallback, Reference};
use sdrk_testkit::golden::{self, Array, GoldenCase};
use sdrk_testkit::{ChunkPlan, bit_exact};

/// The seed of the chunk plans (test 6 of M43); plan `k` uses `PLAN_SEED * 100 + k`.
const PLAN_SEED: u64 = 595;
/// Chunk plans per kernel and case.
const PLANS: u64 = 20;
/// The largest chunk of the plans.
const MAX_CHUNK: usize = 700;

const EQ_CASES: [&str; 2] = ["cma_qpsk_c32_16384", "dd_8psk_c32_16384"];
const DOT_CASES: [&str; 2] = ["c32_1021", "c32_13"];

type State = EqualizeState<Vec<C32>, Vec<C32>>;

#[test]
fn the_cases_of_m43_are_present() {
    assert_eq!(golden::list("equalize").unwrap(), EQ_CASES);
    assert_eq!(golden::list("dot_complex").unwrap(), DOT_CASES);
}

fn cf32<'a>(case: &'a GoldenCase, name: &str) -> &'a [C32] {
    let Array::CF32(v) = case.array(name).unwrap() else {
        panic!("{}: cf32 array {name} expected", case.case);
    };
    v
}

// --------------------------------------------------------------------------------------------
// equalize
// --------------------------------------------------------------------------------------------

fn constellation_of(case: &GoldenCase) -> EqConstellation {
    match case.param_str("constellation").unwrap() {
        "qpsk" => EqConstellation::Qpsk,
        "8psk" => EqConstellation::Psk8,
        other => panic!("{}: unknown constellation {other}", case.case),
    }
}

/// The state of the case (the step sizes are the float32 values of the stored numbers, as in
/// the generator).
fn state_of(case: &GoldenCase) -> State {
    let taps = usize::try_from(case.param_u64("taps").unwrap()).unwrap();
    let feedback = usize::try_from(case.param_u64("feedback_taps").unwrap()).unwrap();
    EqualizeState::new(
        vec![C32::default(); taps + feedback],
        vec![C32::default(); taps + feedback],
        EqParams {
            constellation: constellation_of(case),
            feedback_taps: feedback,
            mu_cma: case.param_f64("mu_cma").unwrap() as f32,
            mu_dd: case.param_f64("mu_dd").unwrap() as f32,
            cma_symbols: u32::try_from(case.param_u64("cma_symbols").unwrap()).unwrap(),
            phase_bw: case.param_f64("phase_bw").unwrap(),
        },
    )
    .unwrap()
}

/// One run: the whole input in one call, or the chunks of plan `k`.
fn run<K: Equalize>(
    kernel: &K,
    case: &GoldenCase,
    input: &[C32],
    plan: Option<u64>,
) -> (Vec<C32>, State) {
    let mut st = state_of(case);
    let mut out = vec![C32::default(); input.len()];
    match plan {
        None => kernel.equalize(&mut st, input, &mut out).unwrap(),
        Some(k) => {
            let mut sizes = ChunkPlan::new(PLAN_SEED * 100 + k, MAX_CHUNK).sizes();
            let mut pos = 0;
            while pos < input.len() {
                let a = sizes.next().unwrap().min(input.len() - pos);
                kernel
                    .equalize(&mut st, &input[pos..pos + a], &mut out[pos..pos + a])
                    .unwrap();
                pos += a;
            }
        }
    }
    (out, st)
}

/// The bits of complex samples.
type Bits = Vec<(u32, u32)>;

/// The state as comparable bits: weights, history, mode, `mse`, tracker phase and frequency.
fn state_bits(st: &State) -> (Bits, Bits, EqMode, u32, u32, u32) {
    let bits = |v: &[C32]| v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect();
    (
        bits(st.weights()),
        bits(st.history()),
        st.mode(),
        st.mse().to_bits(),
        st.phase(),
        st.frequency().to_bits(),
    )
}

/// Runs `kernel` whole and in the 20 plans; every run must match the expected output, the final
/// weights, `mse` and mode of the case and end in the state of the whole run.
fn check_eq<K: Equalize>(kernel: &K, case: &GoldenCase, input: &[C32]) -> Vec<C32> {
    let (whole, end) = run(kernel, case, input, None);
    golden::assert_matches(case, case.array("expected").unwrap(), &whole).unwrap();
    let weights = cf32(case, "weights");
    assert_eq!(
        bit_exact(end.weights(), weights),
        None,
        "{}: final weights",
        case.case
    );
    assert_eq!(end.weights().len(), weights.len());
    assert_eq!(
        u64::from(end.mse().to_bits()),
        case.param_u64("final_mse_bits").unwrap(),
        "{}: final mse",
        case.case
    );
    let mode = match case.param_str("final_mode").unwrap() {
        "cma" => EqMode::Cma,
        _ => EqMode::Decision,
    };
    assert_eq!(end.mode(), mode, "{}: final mode", case.case);
    assert_eq!(
        u64::from(end.phase()),
        case.param_u64("final_phase").unwrap(),
        "{}: final tracker phase",
        case.case
    );
    assert_eq!(
        u64::from(end.frequency().to_bits()),
        case.param_u64("final_frequency_bits").unwrap(),
        "{}: final tracker frequency",
        case.case
    );
    let want = state_bits(&end);
    for k in 0..PLANS {
        let (out, st) = run(kernel, case, input, Some(k));
        assert_eq!(
            bit_exact(&whole, &out),
            None,
            "{}: plan {k} changed the output",
            case.case
        );
        assert!(
            state_bits(&st) == want,
            "{}: plan {k} changed the final state",
            case.case
        );
    }
    whole
}

/// The 8PSK points and the QPSK component of the float64 guard (the literals of the generator).
const C8_F64: f64 = 0.923_879_532_511_286_7;
const S8_F64: f64 = 0.382_683_432_365_089_8;
const QPSK_A_F64: f64 = core::f64::consts::FRAC_1_SQRT_2;

/// The half-open octant of D641 (zero and NaN give 0).
fn octant(yr: f64, yi: f64) -> usize {
    let (ar, ai) = (yr.abs(), yi.abs());
    if yr > 0.0 && yi >= 0.0 {
        usize::from(ar <= ai)
    } else if yr <= 0.0 && yi > 0.0 {
        if ai > ar { 2 } else { 3 }
    } else if yr < 0.0 && yi <= 0.0 {
        if ar > ai { 4 } else { 5 }
    } else if yr >= 0.0 && yi < 0.0 {
        if ai > ar { 6 } else { 7 }
    } else {
        0
    }
}

fn decide_f64(c: EqConstellation, yr: f64, yi: f64) -> (f64, f64) {
    let side = |v: f64, a: f64| if v < 0.0 { -a } else { a };
    match c {
        EqConstellation::Qpsk => (side(yr, QPSK_A_F64), side(yi, QPSK_A_F64)),
        EqConstellation::Psk8 => [
            (C8_F64, S8_F64),
            (S8_F64, C8_F64),
            (-S8_F64, C8_F64),
            (-C8_F64, S8_F64),
            (-C8_F64, -S8_F64),
            (-S8_F64, -C8_F64),
            (S8_F64, -C8_F64),
            (C8_F64, -S8_F64),
        ][octant(yr, yi)],
        _ => panic!("no float64 guard for {c:?}"),
    }
}

fn flush(v: f64) -> f64 {
    if v.abs() < 1e-30 { 0.0 } else { v }
}

/// The loop of section 1 in float64, in the operation order of the generator's guard: the filter
/// and power sums in ascending `i`, the float64 points, the exact cosine and sine of the `u32`
/// tracker phase, the step sizes and tracker gains as the float64 values of their f32.
fn equalize_f64(x: &[C32], st: &State) -> Vec<(f64, f64)> {
    let taps = st.taps();
    let n = taps + st.feedback_taps();
    let (mu_c, mu_d) = (f64::from(st.mu_cma()), f64::from(st.mu_dd()));
    let (kp, ki) = st.phase_gains();
    let (kp, ki) = (f64::from(kp), f64::from(ki));
    let cma_symbols = st.cma_symbols();
    let mut w = vec![(0.0f64, 0.0f64); n];
    w[taps / 2].0 = 1.0;
    let mut h = vec![(0.0f64, 0.0f64); n];
    let mut cma = cma_symbols != 0;
    let mut count = 0u32;
    let (mut phase, mut freq) = (0u32, 0.0f64);
    let mut out = Vec::with_capacity(x.len());
    for v in x {
        h[..taps].rotate_right(1);
        h[0] = (f64::from(v.re), f64::from(v.im));
        let (mut yr, mut yi) = (0.0, 0.0);
        for (&(ur, ui), &(pr, pi)) in h.iter().zip(&w) {
            yr += ur * pr - ui * pi;
            yi += ur * pi + ui * pr;
        }
        let theta = 2.0 * core::f64::consts::PI * (f64::from(phase) / 4_294_967_296.0);
        let (c, s) = (libm::cos(theta), libm::sin(theta));
        let (zr, zi) = (yr * c + yi * s, yi * c - yr * s);
        let (dr, di) = decide_f64(st.constellation(), zr, zi);
        let (mut er, mut ei) = (zr - dr, zi - di);
        let mut fed = (0.0, 0.0);
        let mu = if cma {
            let f = (yr * yr + yi * yi) - 1.0;
            er = (yr * f).clamp(-1.0, 1.0);
            ei = (yi * f).clamp(-1.0, 1.0);
            mu_c
        } else {
            (er, ei) = (er * c - ei * s, er * s + ei * c);
            fed = (dr * c - di * s, dr * s + di * c);
            mu_d
        };
        let power: f64 = h.iter().map(|&(ur, ui)| ur * ur + ui * ui).sum();
        let live = h[..taps].iter().any(|&(ur, ui)| ur != 0.0 || ui != 0.0);
        let m = if live { mu / (power + 1e-6) } else { 0.0 };
        if !live {
            fed = (0.0, 0.0);
        }
        let (gr, gi) = (m * er, m * ei);
        for (&(ur, ui), wk) in h.iter().zip(w.iter_mut()) {
            wk.0 = flush(wk.0 - (gr * ur + gi * ui));
            wk.1 = flush(wk.1 - (gi * ur - gr * ui));
        }
        let pd = zi * dr - zr * di;
        freq = (freq + ki * pd).clamp(-0.125, 0.125);
        let step = libm::floor((freq + kp * pd) * 4_294_967_296.0 + 0.5) as i64;
        phase = phase.wrapping_add(step as u32);
        if n > taps {
            h[taps..].rotate_right(1);
            h[taps] = fed;
        }
        out.push((zr, zi));
        if cma && cma_symbols != u32::MAX {
            count += 1;
            if count >= cma_symbols {
                cma = false;
            }
        }
    }
    out
}

/// The SNR of the model output against the float64 loop from `guard_skip` on must reach the
/// stored guard and stay below the guard plus 3 dB (so the guard is not vacuous).
fn check_eq_guard(case: &GoldenCase, input: &[C32], got: &[C32]) {
    let want = equalize_f64(input, &state_of(case));
    let skip = usize::try_from(case.param_u64("guard_skip").unwrap()).unwrap();
    let (mut num, mut den) = (0.0, 0.0);
    for (&(wr, wi), g) in want[skip..].iter().zip(&got[skip..]) {
        num += wr * wr + wi * wi;
        let (a, b) = (wr - f64::from(g.re), wi - f64::from(g.im));
        den += a * a + b * b;
    }
    let snr = 10.0 * libm::log10(num / den);
    let guard = case.param_f64("snr_db_guard").unwrap();
    assert!(
        snr >= guard && snr < guard + 3.0,
        "{}: SNR {snr:.2} dB against the float64 loop, guard {guard} dB",
        case.case
    );
}

#[test]
fn equalize_golden() {
    for name in EQ_CASES {
        let case = golden::load("equalize", name).unwrap();
        let input = cf32(&case, "input");
        let facade = check_eq(&Sdrk, &case, input);
        let fallback = check_eq(&Fallback, &case, input);
        let reference = check_eq(&Reference, &case, input);
        assert_eq!(bit_exact(&facade, &fallback), None);
        assert_eq!(bit_exact(&facade, &reference), None);
        check_eq_guard(&case, input, &facade);
    }
}

/// The information the generator asserted: CMA restores the constant modulus, decision mode
/// decides every symbol of the second half right.
#[test]
fn the_equalizers_of_the_cases_converge() {
    let cma = golden::load("equalize", "cma_qpsk_c32_16384").unwrap();
    let d_in = cma.param_f64("dispersion_in_last_4096").unwrap();
    let d_out = cma.param_f64("dispersion_out_last_4096").unwrap();
    assert!(d_out < 0.5 * d_in, "{d_in} {d_out}");
    let dd = golden::load("equalize", "dd_8psk_c32_16384").unwrap();
    assert_eq!(dd.param_u64("symbol_errors_second_half").unwrap(), 0);
    let mse = f32::from_bits(u32::try_from(dd.param_u64("final_mse_bits").unwrap()).unwrap());
    assert!(mse < 0.01, "{mse}");
}

// --------------------------------------------------------------------------------------------
// dot_complex
// --------------------------------------------------------------------------------------------

fn windows<K: DotComplex<C32>>(kernel: &K, x: &[C32], w: &[C32], rows: usize) -> Vec<C32> {
    (0..rows)
        .map(|r| kernel.dot_complex(&x[r..r + w.len()], w).unwrap())
        .collect()
}

#[test]
fn dot_complex_golden() {
    for name in DOT_CASES {
        let case = golden::load("dot_complex", name).unwrap();
        let x = cf32(&case, "input");
        let w = cf32(&case, "weights");
        let taps = usize::try_from(case.param_u64("taps").unwrap()).unwrap();
        let rows = usize::try_from(case.param_u64("rows").unwrap()).unwrap();
        assert_eq!(w.len(), taps);
        assert_eq!(x.len(), taps + rows - 1);
        assert_ne!(
            taps % DOT_COMPLEX_PARTIALS,
            0,
            "{name}: the last block must be partial"
        );
        assert_eq!(
            usize::try_from(case.param_u64("partials").unwrap()).unwrap(),
            DOT_COMPLEX_PARTIALS
        );
        let expected = case.array("expected").unwrap();
        let facade = windows(&Sdrk, x, w, rows);
        for got in [
            &facade,
            &windows(&Fallback, x, w, rows),
            &windows(&Reference, x, w, rows),
        ] {
            golden::assert_matches(&case, expected, got).unwrap();
        }
        // The float64 guard: the plain float64 sum of the float64 products.
        let (mut num, mut den) = (0.0, 0.0);
        for (r, g) in facade.iter().enumerate() {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (a, b) in x[r..r + taps].iter().zip(w) {
                let (ar, ai, br, bi) = (
                    f64::from(a.re),
                    f64::from(a.im),
                    f64::from(b.re),
                    f64::from(b.im),
                );
                re += ar * br - ai * bi;
                im += ar * bi + ai * br;
            }
            num += re * re + im * im;
            let (e, f) = (re - f64::from(g.re), im - f64::from(g.im));
            den += e * e + f * f;
        }
        let snr = 10.0 * libm::log10(num / den);
        let guard = case.param_f64("snr_db_guard").unwrap();
        assert!(
            snr >= guard && snr < guard + 3.0,
            "{name}: SNR {snr:.2} dB against the float64 sum, guard {guard} dB"
        );
    }
}
