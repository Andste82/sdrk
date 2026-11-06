//! Golden tests of M40.1: the `slice` constellation `8psk` (`blocks/slice_8psk_c32`, the max-log
//! soft bits at the 8 points and the 8 decision boundaries) through `SliceBlock`, bit for bit,
//! in one call and one symbol per call. The order-8 Costas case `costas/8psk_lock_c32_8192` is
//! run by `golden_m40.rs` with the cases of M40.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation
)]

use sdrk::{C32, Rate};
use sdrk_blocks::{Block, SliceBlock, SliceConfig, SliceConstellation};
use sdrk_testkit::golden::{self, Array, Tolerance};
use sdrk_testkit::{VecStorage, bit_exact};

#[test]
fn slice_8psk_matches_the_float32_model() {
    let case = golden::load("blocks", "slice_8psk_c32").unwrap();
    assert_eq!(case.param_str("constellation").unwrap(), "8psk");
    assert_eq!(case.tolerance, Tolerance::BitExact);
    let Array::CF32(input) = case.array("input").unwrap() else {
        panic!("input must be cf32");
    };
    let Array::F32(want) = case.array("expected").unwrap() else {
        panic!("expected must be f32");
    };
    assert_eq!(want.len(), 3 * input.len());
    assert_eq!(case.param_u64("symbols").unwrap() as usize, input.len());
    let cfg = SliceConfig {
        constellation: SliceConstellation::Psk8,
        scale: case.param_f64("scale").unwrap(),
    };
    let rate = Rate::hz(1000).unwrap();
    let mut slice = SliceBlock::<C32, _>::new(&cfg, rate, &mut VecStorage).unwrap();
    let mut out = vec![0.0f32; want.len()];
    let p = slice.process(input, &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (input.len(), want.len()));
    assert_eq!(bit_exact(&out, want), None, "whole");
    let mut one = vec![0.0f32; want.len()];
    for (x, o) in input.iter().zip(one.chunks_exact_mut(3)) {
        let p = slice.process(core::slice::from_ref(x), o).unwrap();
        assert_eq!((p.consumed, p.produced), (1, 3));
    }
    assert_eq!(bit_exact(&one, want), None, "one symbol per call");
}
