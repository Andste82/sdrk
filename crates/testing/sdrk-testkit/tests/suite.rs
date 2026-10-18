//! Host run of the complete on-target suite.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::unnecessary_wraps,
    clippy::ignored_unit_patterns,
    clippy::items_after_statements,
    clippy::result_large_err,
    clippy::needless_pass_by_value,
    clippy::match_wildcard_for_single_variants,
    clippy::single_match_else
)]

use sdrk_testkit::{Arena, StdReporter, Tier, all_cases, run_suite};

#[test]
fn full_suite_passes_with_a_256_kib_arena() {
    let mut mem = vec![0u8; 256 * 1024];
    let mut arena = Arena::new(&mut mem);
    let summary = run_suite(all_cases(), Tier::Full, &mut arena, &mut StdReporter::new());
    assert_eq!(summary.failed, 0);
    assert_eq!(summary.passed as usize, all_cases().len());
    assert_eq!(summary.skipped, 0);
}

#[test]
fn quick_suite_passes_with_a_64_kib_arena() {
    let mut mem = vec![0u8; 64 * 1024];
    let mut arena = Arena::new(&mut mem);
    let summary = run_suite(
        all_cases(),
        Tier::Quick,
        &mut arena,
        &mut StdReporter::new(),
    );
    assert_eq!(summary.failed, 0);
    let quick = all_cases().iter().filter(|c| c.tier == Tier::Quick).count();
    assert_eq!(summary.passed as usize, quick);
    assert_eq!(summary.skipped as usize, all_cases().len() - quick);
}
