//! Host run of the M23 suite cases alone (the full suite takes minutes in a debug build).
#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use sdrk_testkit::{Arena, StdReporter, TestCase, Tier, all_cases, run_suite};

const NAMES: [&str; 13] = [
    "halfband_c32",
    "halfband_cq15",
    "halfband_real_f32",
    "halfband_real_q15",
    "real_to_complex_c32",
    "real_to_complex_cq15",
    "extremes_m23",
    "block_halfband_decimate_alias",
    "block_real_to_iq_tone",
    "e2e_real_fm_rx_f32",
    "e2e_real_fm_rx_q15",
    "e2e_real_fm_rx_f32_1s",
    "e2e_real_fm_rx_q15_1s",
];

#[test]
fn the_m23_cases_pass_in_the_full_tier() {
    let cases: Vec<TestCase> = NAMES
        .iter()
        .map(|n| *all_cases().iter().find(|c| c.name == *n).unwrap())
        .collect();
    let mut mem = vec![0u8; 256 * 1024];
    let mut arena = Arena::new(&mut mem);
    let summary = run_suite(&cases, Tier::Full, &mut arena, &mut StdReporter::new());
    assert_eq!(summary.failed, 0);
    assert_eq!(summary.passed as usize, NAMES.len());
}

#[test]
fn the_quick_m23_cases_fit_the_80_kib_arena_of_the_quick_tier() {
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
