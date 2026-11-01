//! `Block::map_index`: the first output sample that includes input sample `n`.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_lossless,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::assert_is_empty,
    clippy::large_stack_arrays,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    clippy::type_complexity,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation
)]

mod common;

use common::rate;
use sdrk::C32;
use sdrk_blocks::{
    Block, DdcBlock, DdcConfig, DucBlock, DucConfig, ResampleBlock, ResampleConfig,
    ResampleRealBlock, ScaleBlock, ScaleConfig,
};
use sdrk_testkit::VecStorage;

/// Feeds `n` samples in pieces of 777 and counts the outputs.
fn produced<B: Block>(block: &mut B, n: usize, sample: B::In) -> usize
where
    B::Out: Default,
{
    let input = vec![sample; 777];
    let mut room = vec![B::Out::default(); 20_000];
    let (mut fed, mut made) = (0usize, 0usize);
    while fed < n {
        let mut pending = &input[..(n - fed).min(777)];
        fed += pending.len();
        while !pending.is_empty() {
            let p = block.process(pending, &mut room).unwrap();
            made += p.produced;
            pending = &pending[p.consumed..];
            if p.consumed == 0 && p.produced == 0 {
                break;
            }
        }
    }
    made
}

const COUNTS: [usize; 7] = [1, 2, 9, 10, 11, 1234, 20_001];

#[test]
fn fixed_ratio_blocks_map_to_the_samples_they_produce() {
    let one = C32::new(0.5, 0.25);
    for n in COUNTS {
        let mut scale =
            ScaleBlock::<C32, _>::new(&ScaleConfig::default(), rate(1000), &mut VecStorage)
                .unwrap();
        assert_eq!(scale.map_index(n as u64), n as u64);
        assert_eq!(
            produced(&mut scale, n, one) as u64,
            scale.map_index(n as u64)
        );

        let cfg = DdcConfig {
            decimation: 10,
            ..DdcConfig::default()
        };
        let mut ddc = DdcBlock::<C32, _>::new(&cfg, rate(2_400_000), &mut VecStorage).unwrap();
        assert_eq!(ddc.map_index(n as u64), (n as u64).div_ceil(10));
        assert_eq!(
            produced(&mut ddc, n, one) as u64,
            ddc.map_index(n as u64),
            "ddc {n}"
        );

        let cfg = DucConfig {
            interpolation: 4,
            ..DucConfig::default()
        };
        let mut duc = DucBlock::<C32, _>::new(&cfg, rate(240_000), &mut VecStorage).unwrap();
        assert_eq!(duc.map_index(n as u64), 4 * n as u64);
        assert_eq!(
            produced(&mut duc, n, one) as u64,
            duc.map_index(n as u64),
            "duc {n}"
        );

        // 3/2 = 72 000 / 48 000.
        let cfg = ResampleConfig {
            rate_out_hz: 72_000.0,
            ..ResampleConfig::default()
        };
        let mut res = ResampleBlock::<C32, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
        assert_eq!(res.map_index(n as u64), (3 * n as u64).div_ceil(2));
        assert_eq!(
            produced(&mut res, n, one) as u64,
            res.map_index(n as u64),
            "resampler {n}"
        );
    }
    // A million samples do not overflow.
    let cfg = DdcConfig {
        decimation: 3,
        ..DdcConfig::default()
    };
    let ddc = DdcBlock::<C32, _>::new(&cfg, rate(2_400_000), &mut VecStorage).unwrap();
    assert_eq!(ddc.map_index(u64::MAX), u64::MAX.div_ceil(3));
}

#[test]
fn the_tuned_resampler_maps_with_its_position_step() {
    let nominal = |tunable_ppm: f64| ResampleConfig {
        rate_out_hz: 48_000.0,
        tunable: true,
        ppm: tunable_ppm,
        ..ResampleConfig::default()
    };
    let rate_in = rate(48_000);
    let zero = ResampleRealBlock::<f32, _>::new(&nominal(0.0), rate_in, &mut VecStorage).unwrap();
    assert_eq!(zero.map_index(0), 0);
    assert_eq!(zero.map_index(1_000_000), 1_000_000);
    let mut fast =
        ResampleRealBlock::<f32, _>::new(&nominal(1000.0), rate_in, &mut VecStorage).unwrap();
    // +1000 ppm: the output rate is 0.1 % higher, 1000 more samples per million.
    let m = fast.map_index(1_000_000);
    assert!((1_000_990..=1_001_010).contains(&m), "{m}");
    // It follows the setting at run time.
    fast.set_param("ppm", sdrk_blocks::ParamValue::F64(-1000.0))
        .unwrap();
    let m = fast.map_index(1_000_000);
    assert!((998_990..=999_010).contains(&m), "{m}");
    // Monotone, and the complex block agrees.
    let complex = ResampleBlock::<C32, _>::new(&nominal(250.0), rate_in, &mut VecStorage).unwrap();
    let mut last = 0;
    for n in (0..5000u64).step_by(37) {
        let m = complex.map_index(n);
        assert!(m >= last);
        last = m;
    }
    // The count of outputs that a long run really produces is close (the window holds back a
    // few samples).
    let mut tuned =
        ResampleRealBlock::<f32, _>::new(&nominal(500.0), rate_in, &mut VecStorage).unwrap();
    let n = 200_000usize;
    let made = produced(&mut tuned, n, 0.25f32) as i64;
    let mapped = tuned.map_index(n as u64) as i64;
    assert!((made - mapped).abs() <= 64, "{made} vs {mapped}");
    // No overflow at the largest index.
    let _ = tuned.map_index(u64::MAX);
}
