//! Golden tests of the M48 kernel `gauss_noise` (test 1) through the facade `Sdrk`, `Fallback`
//! and `Reference`.
//!
//! - Every sample case is run in one call and in 20 chunk plans (seed 687); each run must equal
//!   the golden file bit for bit, leave the index at `start + n` and give the bits of the run in
//!   one call.
//! - Fidelity against the ideal `sigma · Φ⁻¹(q)` of the case (accuracy policy step 1): the largest
//!   absolute error stays within `max_abs_err_ideal` (for `Q15` and `CQ15` only where the ideal
//!   value lies inside full scale), and the SNR against `ideal` over the same values reaches the
//!   regression guard `snr_db_guard` of the case (step 3). The SNR is printed.
//! - `gauss_noise/icdf_table`: `gauss_icdf_table()` equals the table that the golden generator
//!   builds from `scipy.special.ndtri`, bit for bit (the only check of the AS241 transcription
//!   of `build.rs`), and `zmax` equals the first knot of the static times 2^-17. The exact
//!   distribution numbers of the case must meet the contract (`docs/kernels.md`).
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stderr,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_core::kernels::noise::{GaussNoise, GaussNoiseState, gauss_icdf_table};
use sdrk_core::{C32, CQ15, Fallback, Q15, Reference, Sample};
use sdrk_testkit::golden::{self, Array, GoldenCase};
use sdrk_testkit::{ChunkPlan, bit_exact};

/// The seed of the chunk plans (M48 test 5); plan `k` uses `PLAN_SEED * 100 + k`.
const PLAN_SEED: u64 = 687;
/// Chunk plans per case.
const PLANS: u64 = 20;
/// Largest chunk of the plans.
const MAX_CHUNK: usize = 700;
/// `2^-17`, the scale of the Q17 knots.
const Q17: f64 = 1.0 / 131_072.0;

/// The chunk sizes of a run: the whole buffer at once, or the sizes of `plan`.
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

/// The state of a case, positioned at its start index.
fn state(case: &GoldenCase) -> GaussNoiseState {
    let seed = case.param_u64("generator_seed").unwrap();
    let stream = case.param_u64("stream").unwrap();
    let bits = u32::try_from(case.param_u64("sigma_bits").unwrap()).unwrap();
    let sigma = f32::from_bits(bits);
    assert_eq!(
        f64::from(sigma),
        case.param_f64("sigma").unwrap(),
        "{}: sigma and sigma_bits disagree",
        case.case
    );
    let mut st = GaussNoiseState::new(seed, stream, sigma).unwrap();
    st.seek(case.param_u64("start_index").unwrap());
    st
}

/// Runs `k` over the case whole and in every chunk plan; checks each run against the golden
/// file and the index; returns the output of the run in one call.
fn run<T: Sample, K: GaussNoise<T>>(k: &K, case: &GoldenCase) -> Vec<T> {
    let n = usize::try_from(case.param_u64("len").unwrap()).unwrap();
    let start = case.param_u64("start_index").unwrap();
    let want = case.array("expected").unwrap();
    assert_eq!(want.len(), n);
    let mut whole = Vec::new();
    let plans = (0..PLANS).map(|p| Some(ChunkPlan::new(PLAN_SEED * 100 + p, MAX_CHUNK)));
    for plan in core::iter::once(None).chain(plans) {
        let mut st = state(case);
        let mut out = vec![T::ZERO; n];
        for (s, e) in chunks(n, plan) {
            k.gauss_noise(&mut st, &mut out[s..e]).unwrap();
        }
        assert_eq!(st.index(), start + n as u64, "{}: index", case.case);
        golden::assert_matches(case, want, &out).unwrap();
        if plan.is_none() {
            whole = out;
        } else {
            assert_eq!(
                bit_exact(&whole, &out),
                None,
                "{}: a chunk plan changes the output",
                case.case
            );
        }
    }
    whole
}

/// The output of the facade, `Fallback` and `Reference` (equal bit for bit); returns one.
fn run_all<T: Sample>(case: &GoldenCase) -> Vec<T>
where
    Sdrk: GaussNoise<T>,
    Fallback: GaussNoise<T>,
    Reference: GaussNoise<T>,
{
    let s = run::<T, _>(&Sdrk, case);
    let f = run::<T, _>(&Fallback, case);
    let r = run::<T, _>(&Reference, case);
    assert_eq!(
        bit_exact(&f, &r),
        None,
        "{}: fallback vs reference",
        case.case
    );
    assert_eq!(bit_exact(&s, &f), None, "{}: facade vs fallback", case.case);
    f
}

/// The ideal values of a case, one real value per component.
fn ideal(case: &GoldenCase) -> Vec<f64> {
    match case.array("ideal").unwrap() {
        Array::F64(v) => v.clone(),
        Array::CF64(v) => v.iter().flat_map(|c: &Complex64| [c.re, c.im]).collect(),
        other => panic!("{}: ideal has dtype {}", case.case, other.dtype()),
    }
}

/// The fidelity check against `ideal`: the largest absolute error and the SNR over the values
/// whose ideal lies inside full scale (all values for the float types).
fn fidelity(case: &GoldenCase, got: &[f64], full_scale: bool) {
    let ideal = ideal(case);
    assert_eq!(ideal.len(), got.len(), "{}", case.case);
    let inside = |x: f64| !full_scale || (-1.0..=32767.0 / 32768.0).contains(&x);
    let (mut sig, mut err, mut worst, mut count) = (0.0_f64, 0.0_f64, 0.0_f64, 0_u64);
    for (&i, &g) in ideal.iter().zip(got) {
        if inside(i) {
            sig += i * i;
            err += (i - g) * (i - g);
            worst = worst.max((i - g).abs());
            count += 1;
        }
    }
    assert_eq!(
        count,
        case.param_u64("compared_ideal").unwrap(),
        "{}: compared values",
        case.case
    );
    let bound = case.param_f64("max_abs_err_ideal").unwrap();
    assert!(
        worst <= bound,
        "{}: largest error against the ideal {worst}, bound {bound}",
        case.case
    );
    let snr = 10.0 * (sig / err).log10();
    let guard = case.param_f64("snr_db_guard").unwrap();
    eprintln!(
        "gauss_noise/{}: SNR against the ideal {snr:.2} dB (guard {guard} dB), largest error {worst:.3e}",
        case.case
    );
    assert!(snr >= guard, "{}: SNR {snr} below {guard}", case.case);
}

fn check(name: &str) {
    let case = golden::load("gauss_noise", name).unwrap();
    assert_eq!(case.tolerance, golden::Tolerance::BitExact, "{name}");
    match case.param_str("sample_type").unwrap() {
        "c32" => {
            let out = run_all::<C32>(&case);
            let got: Vec<f64> = out
                .iter()
                .flat_map(|c| [f64::from(c.re), f64::from(c.im)])
                .collect();
            fidelity(&case, &got, false);
        }
        "f32" => {
            let out = run_all::<f32>(&case);
            let got: Vec<f64> = out.iter().map(|&x| f64::from(x)).collect();
            fidelity(&case, &got, false);
        }
        "cq15" => {
            let out = run_all::<CQ15>(&case);
            let got: Vec<f64> = out
                .iter()
                .flat_map(|c| [c.re.to_f32(), c.im.to_f32()])
                .map(f64::from)
                .collect();
            fidelity(&case, &got, true);
            clipped(&case, out.iter().flat_map(|c| [c.re, c.im]));
        }
        "q15" => {
            let out = run_all::<Q15>(&case);
            let got: Vec<f64> = out.iter().map(|q| f64::from(q.to_f32())).collect();
            fidelity(&case, &got, true);
            clipped(&case, out.iter().copied());
        }
        other => panic!("{name}: unknown sample type {other}"),
    }
}

/// The saturated values of a `Q15` case equal the model's count (the clipping is part of the
/// contract).
fn clipped(case: &GoldenCase, values: impl Iterator<Item = Q15>) {
    let n = values
        .filter(|q| q.0 == i16::MAX || q.0 == i16::MIN)
        .count();
    let want = usize::try_from(case.param_u64("clipped").unwrap()).unwrap();
    assert_eq!(n, want, "{}: clipped values", case.case);
}

#[test]
fn c32_unit_4096() {
    check("c32_unit_4096");
}

#[test]
fn c32_seek_4096() {
    check("c32_seek_4096");
}

#[test]
fn f32_unit_4096() {
    check("f32_unit_4096");
}

#[test]
fn cq15_quarter_4096() {
    check("cq15_quarter_4096");
}

#[test]
fn q15_clip_4096() {
    check("q15_clip_4096");
}

#[test]
fn the_table_of_build_rs_equals_the_ndtri_table_of_the_generator() {
    let case = golden::load("gauss_noise", "icdf_table").unwrap();
    let Array::I32(raw) = case.array("entries").unwrap() else {
        panic!("icdf_table: i32 expected");
    };
    // The entries are u32 stored with the bits of i32.
    let entries: Vec<u32> = raw.iter().map(|&e| e as u32).collect();
    let table = gauss_icdf_table();
    assert_eq!(entries.len(), table.len());
    for (i, (want, got)) in entries.iter().zip(table.iter()).enumerate() {
        assert_eq!(
            got,
            want,
            "entry {i} (octave {}, segment {}): build.rs {got:#010x}, ndtri {want:#010x}",
            i >> 5,
            i & 31
        );
    }
    let zmax = case.param_f64("zmax").unwrap();
    assert_eq!(zmax, f64::from(table[0] >> 12) * Q17);
}

#[test]
fn the_exact_distribution_of_the_table_meets_the_contract() {
    let case = golden::load("gauss_noise", "icdf_table").unwrap();
    let p = |name: &str| case.param_f64(name).unwrap();
    // The numbers of the contract (`docs/kernels.md`, M48 section 2.4).
    assert!(p("max_abs_err_vs_ndtri") <= 5.7e-5);
    assert!((p("zmax") - 6.337_959).abs() < 1e-6);
    let bias = p("variance") - 1.0;
    assert!(bias > 0.0 && bias < 3e-5, "variance bias {bias}");
    assert!((p("kurtosis") - 3.0).abs() < 1e-4);
    assert!(p("ks_vs_phi") <= 2.3e-5);
    let tails = case.params["tails"].as_array().unwrap();
    assert_eq!(tails.len(), 8);
    for t in tails {
        let x = t["t"].as_f64().unwrap();
        let p_gen = t["p_gen"].as_f64().unwrap();
        let phi = t["p_phi"].as_f64().unwrap();
        assert!((phi - 0.5 * libm::erfc(x / core::f64::consts::SQRT_2)).abs() <= 1e-12 * phi);
        // The one-sided tail of the generator differs from Φ(−t) by at most one cell (2^-33)
        // plus the table error times the density φ(t) (M48 section 2.5).
        let density = libm::exp(-0.5 * x * x) / (2.0 * core::f64::consts::PI).sqrt();
        assert!(
            (p_gen - phi).abs() <= 2f64.powi(-33) + 5.7e-5 * density,
            "tail at {}: {p_gen} vs {phi}",
            t["t"]
        );
    }
}

#[test]
fn the_cases_of_m48_are_present() {
    assert_eq!(
        golden::list("gauss_noise").unwrap(),
        [
            "c32_seek_4096",
            "c32_unit_4096",
            "cq15_quarter_4096",
            "f32_unit_4096",
            "icdf_table",
            "q15_clip_4096",
        ]
    );
}
