//! Host run of the M27 suite case alone.
#![cfg(feature = "std")]
#![allow(clippy::unwrap_used)]

use sdrk_testkit::{
    Arena, CHAIN_FM_RECEIVER_AUDIO, StdReporter, Tier, all_cases, chain_fm_receiver_digest,
    run_suite,
};

#[test]
fn chain_fm_receiver_passes_in_the_80_kib_arena_of_the_quick_tier() {
    let case = *all_cases()
        .iter()
        .find(|c| c.name == "chain_fm_receiver")
        .unwrap();
    assert_eq!(case.tier, Tier::Quick);
    let mut mem = vec![0u8; 80 * 1024];
    let mut arena = Arena::new(&mut mem);
    let summary = run_suite(&[case], Tier::Quick, &mut arena, &mut StdReporter::new());
    assert_eq!(summary.failed, 0);
    assert_eq!(summary.passed, 1);
}

#[test]
fn the_chain_makes_the_expected_number_of_audio_samples() {
    let mut mem = vec![0u8; 80 * 1024];
    let mut arena = Arena::new(&mut mem);
    let (count, _) = chain_fm_receiver_digest(&mut arena).unwrap();
    assert_eq!(count, CHAIN_FM_RECEIVER_AUDIO);
}
