//! Golden test of the block `snr_estimate` of M44.4 §1: `blocks/snr_estimate_qam1024_c32` (seed
//! 901, 1024-QAM at Es/N0 34, 20 and 8 dB in windows of 512 symbols) against the f64 model of the
//! estimator of the Python generator, within the tolerance of the case, fed in one call, one
//! window per call and one symbol per call.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation
)]

use sdrk::{C32, Rate};
use sdrk_blocks::{Block, SnrEstimateBlock, SnrEstimateConfig, SymbolMapConstellation};
use sdrk_testkit::VecStorage;
use sdrk_testkit::golden::{self, Array, Tolerance};

/// The estimate after every completed window from the second on, with the input fed in pieces
/// of `piece` symbols.
fn run(input: &[C32], window: usize, piece: usize) -> Vec<f64> {
    let cfg = SnrEstimateConfig {
        constellation: SymbolMapConstellation::Qam1024,
        window,
    };
    let mut est = SnrEstimateBlock::new(&cfg, Rate::hz(10_000).unwrap(), &mut VecStorage).unwrap();
    let mut out = vec![C32::default(); piece];
    let mut got = Vec::new();
    for x in input.chunks(piece) {
        let before = est.estimates();
        let p = est.process(x, &mut out[..x.len()]).unwrap();
        assert_eq!((p.consumed, p.produced), (x.len(), x.len()));
        assert_eq!(&out[..x.len()], x);
        // A piece of at most one window completes at most one window.
        if est.estimates() > before {
            got.push(est.snr_db());
        }
    }
    got
}

#[test]
fn snr_estimate_matches_the_model_of_the_estimator() {
    let case = golden::load("blocks", "snr_estimate_qam1024_c32").unwrap();
    assert_eq!(case.param_str("constellation").unwrap(), "qam1024");
    let window = case.param_u64("window").unwrap() as usize;
    let windows = case.param_u64("windows").unwrap() as usize;
    let Tolerance::MaxAbsErr { max, skip: 0, .. } = case.tolerance else {
        panic!("the case needs max_abs_err");
    };
    let Array::CF32(input) = case.array("input").unwrap() else {
        panic!("input must be cf32");
    };
    let Array::F64(want) = case.array("expected").unwrap() else {
        panic!("expected must be f64");
    };
    assert_eq!(input.len(), window * windows);
    assert_eq!(want.len(), windows - 1);
    for piece in [window, 1, 37] {
        let got = run(input, window, piece);
        assert_eq!(got.len(), want.len(), "piece {piece}");
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= max,
                "piece {piece}, window {}: {g} against {w}",
                k + 1
            );
        }
    }
    // In one call only the last estimate is visible.
    let all = run(input, window, input.len());
    assert!((all[0] - want[want.len() - 1]).abs() <= max);
}
