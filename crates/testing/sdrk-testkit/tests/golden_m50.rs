//! Golden tests of the M50 kernel `xcorr_fft` (test 1) and its use as a correlator (test 3).
//!
//! Test 1: every case of `tests/golden/vectors/xcorr_fft` (seeds 700 to 703) runs through the
//! facade `Sdrk`, `Fallback` and `Reference` in one piece and through the facade and `Fallback`
//! in 20 chunk plans (seed 705, chunks of 1 to 1000 samples; the samples a call does not consume
//! are resubmitted with the next chunk). Every whole run must reach the SNR floor of the case
//! (100 dB against the exact f64 convolution) and its regression guard, and every chunked run must
//! give the bits of the whole run of the same implementation, output and history. The stored
//! taps of the code cases must be the conj-reversed PRBS code of `sdrk_proto::prbs` (chip value
//! `1 - 2 b`), and the stored peak index must be the largest output of the kernel.
//!
//! Test 3 (seed 704): a 127-chip `prbs7` code starting at a known input index in unit-power noise
//! (testkit `Awgn`) peaks at that index plus `L - 1`, with a peak-to-RMS ratio near `sqrt(L)`
//! (statistical: within 0.8 to 1.25 times `sqrt(L)` over 10 trials).
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::print_stderr,
    clippy::float_cmp
)]

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_core::kernels::fft::FftPlan;
use sdrk_core::kernels::xcorr::{XcorrFft, XcorrFftState};
use sdrk_core::{C32, Fallback, Reference};
use sdrk_proto::prbs::{Poly, Prbs};
use sdrk_testkit::ChunkPlan;
use sdrk_testkit::channel::{Awgn, Impairment};
use sdrk_testkit::golden::{self, Array, GoldenCase};

/// The seed of the correlation test (M50 test 3).
const CORR_SEED: u64 = 704;
/// The seed of the chunk plans (M50 tests 1 and 4); plan `k` uses `PLAN_SEED * 100 + k`.
const PLAN_SEED: u64 = 705;
/// Chunk plans per case and implementation.
const PLANS: u64 = 20;
/// Largest chunk of the plans.
const MAX_CHUNK: usize = 1000;

const CASES: [&str; 4] = [
    "code_1023_n4096",
    "code_127_n512",
    "rand_300_n1024",
    "rrc_65_n256",
];

type State = XcorrFftState<C32, Vec<C32>, Vec<C32>, Vec<C32>, Vec<C32>>;

fn state(taps: &[C32], n: usize) -> State {
    let plan = FftPlan::new(n, vec![C32::default(); n / 2]).unwrap();
    XcorrFftState::new(
        taps,
        vec![C32::default(); n],
        vec![C32::default(); taps.len() - 1],
        vec![C32::default(); n],
        plan,
    )
    .unwrap()
}

fn usize_param(case: &GoldenCase, name: &str) -> usize {
    usize::try_from(case.param_u64(name).unwrap()).unwrap()
}

fn cf32(case: &GoldenCase, name: &str) -> Vec<C32> {
    match case.array(name).unwrap() {
        Array::CF32(v) => v.clone(),
        other => panic!("{name} must be cf32, is {}", other.dtype()),
    }
}

fn bits(x: &[C32]) -> Vec<(u32, u32)> {
    x.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}

/// Output and history of one run: the whole input in one call, or the chunks of plan `k`
/// (received samples that a call does not consume are resubmitted with the next chunk).
fn run<K: XcorrFft<C32>>(
    kernel: &K,
    taps: &[C32],
    n: usize,
    input: &[C32],
    plan: Option<u64>,
) -> (Vec<C32>, Vec<C32>) {
    let mut st = state(taps, n);
    let b = st.block_size();
    let mut out = vec![C32::default(); input.len()];
    match plan {
        None => {
            let p = kernel.xcorr_fft(&mut st, input, &mut out).unwrap();
            assert_eq!((p.consumed, p.produced), (input.len(), input.len()));
        }
        Some(k) => {
            let mut sizes = ChunkPlan::new(PLAN_SEED * 100 + k, MAX_CHUNK).sizes();
            let (mut pos, mut received) = (0, 0);
            while pos < input.len() {
                received = (received + sizes.next().unwrap()).min(input.len());
                let p = kernel
                    .xcorr_fft(&mut st, &input[pos..received], &mut out[pos..])
                    .unwrap();
                assert_eq!(p.consumed, p.produced);
                assert_eq!(p.consumed, (received - pos) / b * b);
                pos += p.consumed;
            }
        }
    }
    (out, st.history().to_vec())
}

/// The PRBS code of the case as chips `1 - 2 b` (shared conventions §B.1).
fn code_chips(poly: Poly, seed: u32, len: usize) -> Vec<f32> {
    let mut g = Prbs::new(poly, seed).unwrap();
    (0..len)
        .map(|_| 1.0 - 2.0 * f32::from(g.next_bit()))
        .collect()
}

fn poly_of(name: &str) -> Poly {
    match name {
        "prbs7" => Poly::Prbs7,
        "prbs10" => Poly::Prbs10,
        other => panic!("unexpected code {other}"),
    }
}

#[test]
fn the_cases_of_m50_are_present() {
    let mut want = CASES.map(String::from).to_vec();
    want.sort();
    assert_eq!(golden::list("xcorr_fft").unwrap(), want);
}

#[test]
fn xcorr_fft_matches_the_golden_cases() {
    for name in CASES {
        let case = golden::load("xcorr_fft", name).unwrap();
        let l = usize_param(&case, "taps_len");
        let n = usize_param(&case, "fft_size");
        let input = cf32(&case, "input");
        let taps = cf32(&case, "taps");
        assert_eq!(taps.len(), l, "{name}");
        assert_eq!(usize_param(&case, "block_size"), n - l + 1, "{name}");
        assert_eq!(input.len(), 4 * (n - l + 1), "{name}");
        let expected = case.array("expected").unwrap();
        if let Ok(code) = case.param_str("code") {
            // The taps are the conj-reversed code of the Rust PRBS generator.
            let seed = u32::try_from(case.param_u64("seed").unwrap()).unwrap();
            let chips = code_chips(poly_of(code), seed, l);
            let want: Vec<C32> = chips.iter().rev().map(|&c| C32::new(c, -0.0)).collect();
            for (t, w) in taps.iter().zip(&want) {
                assert_eq!(t.re, w.re, "{name}");
                assert_eq!(t.im, 0.0, "{name}");
            }
        }
        let (facade, facade_hist) = run(&Sdrk, &taps, n, &input, None);
        let (fallback, fallback_hist) = run(&Fallback, &taps, n, &input, None);
        let (reference, _) = run(&Reference, &taps, n, &input, None);
        for (who, out) in [
            ("facade", &facade),
            ("fallback", &fallback),
            ("reference", &reference),
        ] {
            let c = golden::assert_matches(&case, expected, out)
                .unwrap_or_else(|e| panic!("{name} {who}: {e}"));
            eprintln!("{name} {who}: {:.2} dB", c.snr_db);
        }
        if let Ok(peak) = case.param_u64("peak_index") {
            let best = fallback
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.norm_sqr().total_cmp(&b.1.norm_sqr()))
                .unwrap()
                .0;
            assert_eq!(best as u64, peak, "{name}");
            let offset = case.param_u64("offset").unwrap();
            assert_eq!(peak, offset + l as u64 - 1, "{name}");
        }
        for k in 0..PLANS {
            let (f, fh) = run(&Fallback, &taps, n, &input, Some(k));
            assert_eq!(bits(&f), bits(&fallback), "{name} fallback plan {k}");
            assert_eq!(bits(&fh), bits(&fallback_hist), "{name} fallback plan {k}");
            let (s, sh) = run(&Sdrk, &taps, n, &input, Some(k));
            assert_eq!(bits(&s), bits(&facade), "{name} facade plan {k}");
            assert_eq!(bits(&sh), bits(&facade_hist), "{name} facade plan {k}");
        }
    }
}

#[test]
fn a_code_in_noise_peaks_at_its_start_plus_l_minus_1() {
    let l = 127;
    let n = 512;
    let chips = code_chips(Poly::Prbs7, 1, l);
    let taps: Vec<C32> = chips.iter().rev().map(|&c| C32::new(c, 0.0)).collect();
    let b = n - l + 1;
    let len = 4 * b;
    let expect = (l as f64).sqrt();
    for trial in 0..10u64 {
        let mut awgn =
            Awgn::new(core::f64::consts::FRAC_1_SQRT_2, CORR_SEED * 100 + trial).unwrap();
        let m = 100 + 97 * trial as usize;
        let mut clean = vec![Complex64::new(0.0, 0.0); len];
        for (i, &c) in chips.iter().enumerate() {
            clean[m + i] = Complex64::new(f64::from(c), 0.0);
        }
        let mut noisy = Vec::new();
        awgn.process(&clean, &mut noisy);
        let input: Vec<C32> = noisy
            .iter()
            .map(|v| C32::new(v.re as f32, v.im as f32))
            .collect();
        for who in ["facade", "fallback"] {
            let (out, _) = if who == "facade" {
                run(&Sdrk, &taps, n, &input, None)
            } else {
                run(&Fallback, &taps, n, &input, None)
            };
            let mags: Vec<f64> = out.iter().map(|v| f64::from(v.norm_sqr()).sqrt()).collect();
            let (best, peak) = mags
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            assert_eq!(best, m + l - 1, "{who} trial {trial}");
            let rms = (mags.iter().map(|v| v * v).sum::<f64>() / mags.len() as f64).sqrt();
            let ratio = peak / rms;
            eprintln!(
                "{who} trial {trial}: peak at {best}, peak / rms {ratio:.2} (sqrt L {expect:.2})"
            );
            assert!(
                (0.8 * expect..=1.25 * expect).contains(&ratio),
                "{who} trial {trial}: peak / rms {ratio}"
            );
        }
    }
}
