//! Host run of the M24 suite cases alone (the full suite takes minutes in a debug build).
#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use sdrk_testkit::{Arena, StdReporter, TestCase, Tier, all_cases, run_suite};

const NAMES: [&str; 14] = [
    "stereo_pll_f32",
    "stereo_pll_q15",
    "stereo_matrix_c32",
    "stereo_matrix_cq15",
    "extremes_m24",
    "block_fm_stereo_demod_separation",
    "block_fm_stereo_demod_mono",
    "e2e_fm_stereo_f32",
    "e2e_fm_stereo_q15",
    "e2e_fm_stereo_mono",
    "e2e_fm_stereo_f32_2s",
    "e2e_fm_stereo_q15_2s",
    "e2e_fm_stereo_switch_f32",
    "backend_vs_fallback_stereo_matrix",
];

#[test]
fn the_m24_cases_pass_in_the_full_tier() {
    let cases: Vec<TestCase> = NAMES
        .iter()
        .map(|n| *all_cases().iter().find(|c| c.name == *n).unwrap())
        .collect();
    let mut mem = vec![0u8; 512 * 1024];
    let mut arena = Arena::new(&mut mem);
    let summary = run_suite(&cases, Tier::Full, &mut arena, &mut StdReporter::new());
    assert_eq!(summary.failed, 0);
    assert_eq!(summary.passed as usize, NAMES.len());
}

#[test]
fn the_quick_m24_cases_fit_the_80_kib_arena_of_the_quick_tier() {
    let cases: Vec<TestCase> = NAMES
        .iter()
        .map(|n| *all_cases().iter().find(|c| c.name == *n).unwrap())
        .collect();
    let mut mem = vec![0u8; 80 * 1024];
    let mut arena = Arena::new(&mut mem);
    let summary = run_suite(&cases, Tier::Quick, &mut arena, &mut StdReporter::new());
    assert_eq!(summary.failed, 0);
    let quick = cases.iter().filter(|c| c.tier == Tier::Quick).count();
    assert_eq!(summary.passed as usize, quick);
}
