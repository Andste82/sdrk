//! Golden tests of the M35 kernel `am_mod` through the facade `Sdrk`, `Fallback` and `Reference`.
//!
//! Every case is run in one piece and in chunks of random size (the plan seed is the case seed);
//! both runs must match the golden file bit for bit. The tests also check that the inputs hold
//! the values beyond `+-1` and the NaNs the generator lists, that a NaN gives the bare carrier
//! and that the `CQ15` gains of the kernel are the ones of the generator. The golden cases of the
//! block `ssb_mod` (`blocks/ssb_mod_*`) are tested with the block in `sdrk-blocks`.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use sdrk::Sdrk;
use sdrk_core::kernels::modulate::AmMod;
use sdrk_core::{C32, CQ15, Fallback, Q15, Reference, Sample};
use sdrk_testkit::ChunkPlan;
use sdrk_testkit::golden::{self, Array, GoldenCase};

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
    ChunkPlan::new(case.seed, 300)
}

fn positions(case: &GoldenCase, key: &str) -> Vec<usize> {
    case.params[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| usize::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}

fn gains(case: &GoldenCase) -> (f32, f32) {
    (
        case.param_f64("carrier").unwrap() as f32,
        case.param_f64("depth").unwrap() as f32,
    )
}

/// Runs `kernel` whole and in the chunks of the case's plan and compares with `expected`.
fn run_case<R: Copy, T: Sample>(case: &GoldenCase, input: &[R], kernel: impl Fn(&[R], &mut [T])) {
    let want = case.array("expected").unwrap();
    for plan in [None, Some(plan_of(case))] {
        let mut out = vec![T::ZERO; input.len()];
        for (s, e) in chunks(input.len(), plan) {
            kernel(&input[s..e], &mut out[s..e]);
        }
        golden::assert_matches(case, want, &out).unwrap();
    }
}

fn c32_case<K: AmMod<C32>>(k: &K, case: &GoldenCase) {
    let input = match case.array("input").unwrap() {
        Array::F32(v) => v.clone(),
        other => panic!("f32 expected, got {}", other.dtype()),
    };
    let (carrier, depth) = gains(case);
    run_case(case, &input, |i, o: &mut [C32]| {
        k.am_mod(carrier, depth, i, o).unwrap();
    });
}

fn cq15_case<K: AmMod<CQ15>>(k: &K, case: &GoldenCase) {
    let input = case.array("input").unwrap().as_q15().unwrap();
    let (carrier, depth) = gains(case);
    run_case(case, &input, |i, o: &mut [CQ15]| {
        k.am_mod(carrier, depth, i, o).unwrap();
    });
}

macro_rules! for_each_kernel {
    ($f:ident, $case:expr) => {
        $f(&Sdrk, $case);
        $f(&Fallback, $case);
        $f(&Reference, $case);
    };
}

#[test]
fn the_cases_of_m35_are_present() {
    assert_eq!(golden::list("am_mod").unwrap(), ["c32_4096", "cq15_4096"]);
}

#[test]
fn am_mod_c32_golden() {
    let case = golden::load("am_mod", "c32_4096").unwrap();
    for_each_kernel!(c32_case, &case);
}

#[test]
fn am_mod_cq15_golden() {
    let case = golden::load("am_mod", "cq15_4096").unwrap();
    for_each_kernel!(cq15_case, &case);
}

#[test]
fn the_c32_case_holds_the_specials_and_a_nan_gives_the_carrier() {
    let case = golden::load("am_mod", "c32_4096").unwrap();
    let Array::F32(input) = case.array("input").unwrap() else {
        panic!("f32 input expected");
    };
    let Array::CF32(want) = case.array("expected").unwrap() else {
        panic!("cf32 expected values expected");
    };
    let (beyond, nans) = (
        positions(&case, "beyond_positions"),
        positions(&case, "nan_positions"),
    );
    assert_eq!((beyond.len(), nans.len()), (32, 8));
    assert!(beyond.iter().all(|&p| input[p].abs() > 1.0));
    assert!(input.contains(&f32::INFINITY) && input.contains(&f32::NEG_INFINITY));
    let (carrier, depth) = gains(&case);
    for &p in &nans {
        assert!(input[p].is_nan());
        assert_eq!(want[p], C32::new(carrier, 0.0));
    }
    // The clamp: every value beyond +-1 gives one end of the envelope.
    let (low, high) = (carrier * (1.0 - depth), carrier * (1.0 + depth));
    for &p in &beyond {
        let end = if input[p] > 0.0 { high } else { low };
        assert_eq!(want[p].re, end, "sample {p}");
    }
    assert!(want.iter().all(|s| s.im.to_bits() == 0));
}

#[test]
fn the_cq15_gains_are_the_ones_of_the_generator() {
    let case = golden::load("am_mod", "cq15_4096").unwrap();
    let (carrier, depth) = gains(&case);
    let c_q = i16::try_from(case.param_u64("carrier_q").unwrap()).unwrap();
    let m_q = i64::try_from(case.param_u64("depth_q").unwrap()).unwrap();
    // x = 0 gives re = c_q whatever the depth; with carrier 1 (c_q = 32767) the outputs for
    // x = +-16384 show m_q through t = round_shift(m_q x, 15).
    let mut out = [CQ15::default(); 2];
    Sdrk.am_mod(carrier, 1.0, &[Q15(0), Q15(0)], &mut out)
        .unwrap();
    assert_eq!(out[0].re, Q15(c_q));
    Sdrk.am_mod(1.0, depth, &[Q15(16384), Q15(-16384)], &mut out)
        .unwrap();
    let t = (m_q * 16384 + (1 << 14)) >> 15;
    let re = |x: i64| ((32767 * (32768 + x) + (1 << 14)) >> 15).min(32767);
    assert_eq!(i64::from(out[0].re.0), re(t));
    assert_eq!(i64::from(out[1].re.0), re((m_q * -16384 + (1 << 14)) >> 15));
}
