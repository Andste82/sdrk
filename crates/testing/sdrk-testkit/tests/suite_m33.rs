//! Host run of the M33 suite cases alone (the full suite takes minutes in a debug build).
#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use sdrk_testkit::{Arena, StdReporter, TestCase, Tier, all_cases, run_suite};

const NAMES: [&str; 4] = [
    "noise_blank_c32",
    "noise_blank_cq15",
    "auto_notch_f32",
    "extremes_m33",
];

fn cases() -> Vec<TestCase> {
    NAMES
        .iter()
        .map(|n| *all_cases().iter().find(|c| c.name == *n).unwrap())
        .collect()
}

#[test]
fn the_m33_cases_pass_in_the_full_tier() {
    let cases = cases();
    let mut mem = vec![0u8; 512 * 1024];
    let mut arena = Arena::new(&mut mem);
    let summary = run_suite(&cases, Tier::Full, &mut arena, &mut StdReporter::new());
    assert_eq!(summary.failed, 0);
    assert_eq!(summary.passed as usize, NAMES.len());
}

#[test]
fn the_quick_m33_cases_fit_the_80_kib_arena_of_the_quick_tier() {
    let cases = cases();
    let mut mem = vec![0u8; 80 * 1024];
    let mut arena = Arena::new(&mut mem);
    let summary = run_suite(&cases, Tier::Quick, &mut arena, &mut StdReporter::new());
    assert_eq!(summary.failed, 0);
    let quick = cases.iter().filter(|c| c.tier == Tier::Quick).count();
    assert_eq!(summary.passed as usize, quick);
}
