//! The tunable resampler: output counts for a ppm correction over long streams, and the
//! position bookkeeping that keeps `pos` from overflowing.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_lossless,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::assert_is_empty,
    clippy::range_plus_one,
    clippy::large_stack_arrays,
    clippy::doc_markdown,
    clippy::multiple_bound_locations,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_pass_by_value,
    clippy::type_complexity,
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation
)]

mod common;

use common::rate;
use sdrk::{C32, CQ15, Q15};
use sdrk_blocks::{Block, ParamValue, ResampleBlock, ResampleConfig, ResampleRealBlock};
use sdrk_testkit::VecStorage;

/// Pushes `n` samples of a constant through the block in pieces of 4000 and counts the outputs.
fn count_outputs<B: Block>(block: &mut B, n: usize, sample: B::In) -> usize
where
    B::Out: Default,
{
    let input = vec![sample; 4000];
    let mut room = vec![B::Out::default(); 8192];
    let (mut fed, mut produced) = (0usize, 0usize);
    while fed < n {
        let piece = &input[..(n - fed).min(4000)];
        let mut pending = piece;
        while !pending.is_empty() {
            let p = block.process(pending, &mut room).unwrap();
            produced += p.produced;
            pending = &pending[p.consumed..];
        }
        fed += piece.len();
    }
    produced
}

#[test]
fn plus_100_ppm_over_a_million_samples() {
    let cfg = ResampleConfig {
        tunable: true,
        ppm: 100.0,
        ..ResampleConfig::default()
    };
    let mut block = ResampleBlock::<C32, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    let n = count_outputs(&mut block, 1_000_000, C32::new(0.25, 0.0));
    let expected = (1_000_000.0f64 * (1.0 + 1e-4)).round() as i64;
    assert!((n as i64 - expected).abs() <= 1, "{n} vs {expected}");
}

#[test]
fn the_ppm_set_later_gives_the_same_count_as_the_ppm_of_the_config() {
    let a = ResampleConfig {
        tunable: true,
        ppm: -250.0,
        ..ResampleConfig::default()
    };
    let b = ResampleConfig {
        tunable: true,
        ..ResampleConfig::default()
    };
    let mut block_a = ResampleBlock::<CQ15, _>::new(&a, rate(48_000), &mut VecStorage).unwrap();
    let mut block_b = ResampleBlock::<CQ15, _>::new(&b, rate(48_000), &mut VecStorage).unwrap();
    block_b.set_param("ppm", ParamValue::F64(-250.0)).unwrap();
    let s = sdrk::CQ15 {
        re: Q15(1000),
        im: Q15(-1000),
    };
    let (na, nb) = (
        count_outputs(&mut block_a, 200_000, s),
        count_outputs(&mut block_b, 200_000, s),
    );
    assert_eq!(na, nb);
    let expected = (200_000.0f64 * (1.0 - 250e-6)).round() as i64;
    assert!((na as i64 - expected).abs() <= 1, "{na} vs {expected}");
}

#[test]
fn a_ppm_change_in_the_middle_changes_the_rate_from_there_on() {
    let cfg = ResampleConfig {
        tunable: true,
        ..ResampleConfig::default()
    };
    let mut block = ResampleBlock::<C32, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    let s = C32::new(0.1, 0.1);
    let first = count_outputs(&mut block, 100_000, s);
    block.set_param("ppm", ParamValue::F64(500.0)).unwrap();
    let second = count_outputs(&mut block, 100_000, s);
    assert!((first as i64 - 100_000).abs() <= 1);
    assert!((second as i64 - 100_050).abs() <= 2, "{second}");
}

#[test]
fn the_real_resampler_counts_like_the_complex_one() {
    let cfg = ResampleConfig {
        rate_out_hz: 44_100.0,
        tunable: true,
        ppm: 100.0,
        ..ResampleConfig::default()
    };
    let mut real = ResampleRealBlock::<f32, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    let mut complex = ResampleBlock::<C32, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    let a = count_outputs(&mut real, 300_000, 0.3);
    let b = count_outputs(&mut complex, 300_000, C32::new(0.3, 0.0));
    assert_eq!(a, b);
    let expected = (300_000.0f64 * 44_100.0 / 48_000.0 * (1.0 + 1e-4)).round() as i64;
    assert!((a as i64 - expected).abs() <= 1, "{a} vs {expected}");
}

#[test]
fn a_rational_resampler_has_an_exact_count_and_no_drift() {
    let cfg = ResampleConfig {
        rate_out_hz: 44_100.0,
        max_taps: 8192,
        ..ResampleConfig::default()
    };
    let mut block = ResampleBlock::<C32, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    let n = count_outputs(&mut block, 480_000, C32::new(0.3, 0.0));
    assert_eq!(n, 441_000, "480000 * 147 / 160");
}
