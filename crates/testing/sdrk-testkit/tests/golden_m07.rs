//! Golden tests of the FFT kernels of M07 (`fft`, `rfft`, `irfft`) through the facade `Sdrk`.
//!
//! The `Q15` block-floating-point cases are compared after the output has been multiplied by
//! `2^exponent`, the fixed cases directly (their stored expectation is already divided by `N`).
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use sdrk::Sdrk;
use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, RealFft, RealFftPlan, Scaling};
use sdrk_core::{C32, CQ15};
use sdrk_testkit::golden::{self, Array, GoldenCase};

fn n_of(case: &GoldenCase) -> usize {
    usize::try_from(case.param_u64("n").unwrap()).unwrap()
}

fn direction(case: &GoldenCase) -> Direction {
    match case.param_str("direction").unwrap() {
        "forward" => Direction::Forward,
        "inverse" => Direction::Inverse,
        other => panic!("unknown direction {other}"),
    }
}

fn log2(n: usize) -> i32 {
    i32::try_from(n.trailing_zeros()).unwrap()
}

#[test]
fn all_cases_are_present() {
    let expect = [
        "c32_fwd_16",
        "c32_fwd_64",
        "c32_fwd_1024",
        "c32_fwd_4096",
        "c32_inv_1024",
        "rfft_f32_32",
        "rfft_f32_1024",
        "rfft_f32_4096",
        "irfft_f32_32",
        "irfft_f32_1024",
        "irfft_f32_4096",
        "cq15_fixed_tone_1024",
        "cq15_fixed_noise_1024",
        "cq15_fixed_noise_4096",
        "cq15_fixed_inv_1024",
        "cq15_bfp_tone_1024",
        "cq15_bfp_noise_1024",
        "cq15_bfp_weak_tone_1024",
    ];
    let names = golden::list("fft").unwrap();
    for c in expect {
        assert!(names.iter().any(|n| n == c), "missing fft/{c}");
    }
    assert_eq!(names.len(), expect.len(), "unexpected cases: {names:?}");
}

#[test]
fn c32_golden() {
    for name in golden::list("fft").unwrap() {
        if !name.starts_with("c32_") {
            continue;
        }
        let case = golden::load("fft", &name).unwrap();
        let n = n_of(&case);
        let Array::CF32(x) = case.array("input").unwrap() else {
            panic!("{name}: input is not cf32");
        };
        let want = case.array("expected").unwrap();
        let plan = FftPlan::<C32, _>::new(n, vec![C32::default(); n / 2]).unwrap();
        let dir = direction(&case);
        let mut data = x.clone();
        let scale = Fft::<C32>::fft(&Sdrk, &plan, &mut data, dir, Scaling::Default).unwrap();
        let want_exp = if dir == Direction::Forward {
            0
        } else {
            log2(n)
        };
        assert_eq!(scale.exponent, want_exp, "{name}");
        let c = golden::assert_matches(&case, want, &data).unwrap();
        assert!(c.snr_db >= 110.0, "{name}: {} dB", c.snr_db);
    }
}

#[test]
fn rfft_golden() {
    for name in golden::list("fft").unwrap() {
        if !name.starts_with("rfft_") {
            continue;
        }
        let case = golden::load("fft", &name).unwrap();
        let n = n_of(&case);
        let Array::F32(x) = case.array("input").unwrap() else {
            panic!("{name}: input is not f32");
        };
        let plan =
            RealFftPlan::new(n, vec![C32::default(); n / 4], vec![C32::default(); n / 2]).unwrap();
        let mut bins = vec![C32::default(); n / 2 + 1];
        let mut scratch = vec![C32::default(); n / 2];
        let scale = Sdrk.rfft(&plan, x, &mut bins, &mut scratch).unwrap();
        assert_eq!(scale.exponent, 0);
        golden::assert_matches(&case, case.array("expected").unwrap(), &bins).unwrap();
    }
}

#[test]
fn irfft_golden() {
    for name in golden::list("fft").unwrap() {
        if !name.starts_with("irfft_") {
            continue;
        }
        let case = golden::load("fft", &name).unwrap();
        let n = n_of(&case);
        let Array::CF32(bins) = case.array("input").unwrap() else {
            panic!("{name}: input is not cf32");
        };
        // The case deliberately carries imaginary parts in bin 0 and bin N/2.
        assert_ne!(bins[0].im, 0.0);
        assert_ne!(bins[n / 2].im, 0.0);
        let plan =
            RealFftPlan::new(n, vec![C32::default(); n / 4], vec![C32::default(); n / 2]).unwrap();
        let mut out = vec![0.0f32; n];
        let mut scratch = vec![C32::default(); n / 2];
        let scale = Sdrk.irfft(&plan, bins, &mut out, &mut scratch).unwrap();
        assert_eq!(scale.exponent, log2(n));
        golden::assert_matches(&case, case.array("expected").unwrap(), &out).unwrap();
    }
}

#[test]
fn cq15_fixed_golden() {
    for name in golden::list("fft").unwrap() {
        if !name.starts_with("cq15_fixed_") {
            continue;
        }
        let case = golden::load("fft", &name).unwrap();
        let n = n_of(&case);
        let Array::CI16(x) = case.array("input").unwrap() else {
            panic!("{name}: input is not ci16");
        };
        let plan = FftPlan::<CQ15, _>::new(n, vec![CQ15::default(); n / 2]).unwrap();
        let mut data = x.clone();
        let scale =
            Fft::<CQ15>::fft(&Sdrk, &plan, &mut data, direction(&case), Scaling::Default).unwrap();
        assert_eq!(scale.exponent, log2(n), "{name}");
        golden::assert_matches(&case, case.array("expected").unwrap(), &data).unwrap();
    }
}

#[test]
fn cq15_block_float_golden() {
    for name in golden::list("fft").unwrap() {
        if !name.starts_with("cq15_bfp_") {
            continue;
        }
        let case = golden::load("fft", &name).unwrap();
        let n = n_of(&case);
        let Array::CI16(x) = case.array("input").unwrap() else {
            panic!("{name}: input is not ci16");
        };
        let plan = FftPlan::<CQ15, _>::new(n, vec![CQ15::default(); n / 2]).unwrap();
        let mut data = x.clone();
        let scale = Fft::<CQ15>::fft(
            &Sdrk,
            &plan,
            &mut data,
            direction(&case),
            Scaling::BlockFloat,
        )
        .unwrap();
        assert!((0..=2 * log2(n)).contains(&scale.exponent), "{name}");
        let out = golden::scaled_block_float(&data, scale.exponent);
        golden::assert_matches_block_float(
            &case,
            case.array("expected").unwrap(),
            &data,
            scale.exponent,
        )
        .unwrap();

        if name == "cq15_bfp_weak_tone_1024" {
            // The tone bin must stand at least `min_margin_db` above the median of all other bins.
            let bin = usize::try_from(case.param_u64("bin").unwrap()).unwrap();
            let margin = case.param_f64("min_margin_db").unwrap();
            let mag = |v: &C32| libm::sqrt(f64::from(v.re).powi(2) + f64::from(v.im).powi(2));
            let tone = mag(&out[bin]);
            let mut others: Vec<f64> = out
                .iter()
                .enumerate()
                .filter(|(k, _)| *k != bin)
                .map(|(_, v)| mag(v))
                .collect();
            others.sort_by(f64::total_cmp);
            let median = others[others.len() / 2];
            // Written as a product so that a median of exactly zero (infinite margin) passes.
            assert!(
                tone >= median * libm::pow(10.0, margin / 20.0),
                "tone {tone}, median {median}"
            );
            // Full scale is about 2^exponent * 1: a weak tone must not be wiped out by scaling.
            assert!(
                scale.exponent <= 4,
                "weak tone needs little scaling: {}",
                scale.exponent
            );
        }
    }
}
