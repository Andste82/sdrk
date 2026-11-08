//! Spec tests 4 and 5 of M45: the blocks `ChannelizeBlock` and `ChannelSelectBlock`.
//!
//! - Test 4: the block equals the kernel `channelize` value for value (the same prototype and
//!   the same plan), chunk invariance with 20 plans of input chunks and output room (seed 664),
//!   the required `taps_per_branch` in the error of a prototype that is too short (3 cases; the
//!   pipeline message is checked in `sdrk-pipeline/tests/channelize_kinds.rs`).
//! - Test 5: every lane of 4, 16 and 256 lanes equals `step(i)[lane]` of the input, the signed
//!   mapping, the range error and chunk invariance (seed 665).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use sdrk::{C32, Delay, Ratio, RatioKind, Sdrk};
use sdrk_blocks::design::{channelizer_prototype, channelizer_taps_per_branch};
use sdrk_blocks::{
    Block, ChannelSelectBlock, ChannelSelectConfig, ChannelizeBlock, ChannelizeConfig, Error,
    LanesMut, LanesRef,
};
use sdrk_core::kernels::channelize::{Channelize, ChannelizeState, history_len};
use sdrk_testkit::{ChunkPlan, Pcg32, VecStorage};

const SEED_CHANNELIZE: u64 = 664;
const SEED_SELECT: u64 = 665;

fn noise(n: usize, seed: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(seed, 3);
    (0..n)
        .map(|_| C32::new(rng.normal() as f32 * 0.5, rng.normal() as f32 * 0.5))
        .collect()
}

/// The kernel run directly over the whole input, with the prototype of `cfg`.
fn kernel_output(cfg: &ChannelizeConfig, x: &[C32]) -> Vec<C32> {
    let (m, l) = (cfg.channels, cfg.taps_per_branch);
    let mut h = vec![0.0f64; m * l];
    channelizer_prototype(m, l, cfg.passband, cfg.stopband, cfg.attenuation_db, &mut h).unwrap();
    let taps: Vec<f32> = h.iter().map(|&v| v as f32).collect();
    let mut s = ChannelizeState::<C32, _, _, _>::new(
        taps,
        m,
        vec![C32::default(); history_len(m, l)],
        vec![C32::default(); m / 2],
    )
    .unwrap();
    let mut out = vec![C32::default(); x.len() / (m / 2) * m];
    let p = Sdrk.channelize(&mut s, x, &mut out).unwrap();
    out.truncate(p.produced);
    out
}

/// Runs the block over `x` in the chunks of `chunks`, each call with the output room of
/// `room` steps (at least 1); returns all lane values.
fn block_output(
    block: &mut ChannelizeBlock<VecStorage>,
    x: &[C32],
    chunks: &mut dyn Iterator<Item = usize>,
    room: &mut dyn Iterator<Item = usize>,
) -> Vec<C32> {
    let m = block.channels();
    let mut got = Vec::new();
    let mut buf = vec![C32::default(); m * 64];
    let mut at = 0;
    while at < x.len() {
        let end = (at + chunks.next().unwrap()).min(x.len());
        while at < end {
            let steps = room.next().unwrap().clamp(1, 64);
            let view = LanesMut::new(&mut buf[..steps * m], m).unwrap();
            let p = block.process(&x[at..end], view).unwrap();
            assert!(p.produced <= steps);
            got.extend_from_slice(&buf[..p.produced * m]);
            at += p.consumed;
        }
    }
    got
}

fn configs() -> [ChannelizeConfig; 4] {
    let grid = ChannelizeConfig::default();
    [
        grid,
        ChannelizeConfig {
            channels: 4,
            taps_per_branch: 16,
            ..grid
        },
        ChannelizeConfig {
            channels: 64,
            ..grid
        },
        ChannelizeConfig {
            passband: 0.4,
            stopband: 0.6,
            taps_per_branch: 24,
            ..grid
        },
    ]
}

#[test]
fn the_block_equals_the_kernel_value_for_value() {
    for cfg in configs() {
        let x = noise(6000, SEED_CHANNELIZE);
        let want = kernel_output(&cfg, &x);
        assert_eq!(want.len(), 6000 / (cfg.channels / 2) * cfg.channels);
        let mut block = ChannelizeBlock::new(&cfg, &mut VecStorage).unwrap();
        let got = block_output(
            &mut block,
            &x,
            &mut std::iter::repeat(6000),
            &mut std::iter::repeat(64),
        );
        assert!(got == want, "{cfg:?}");
        // The block keeps the prototype in the order of the kernel state (time-reversed).
        assert_eq!(block.taps().len(), cfg.channels * cfg.taps_per_branch);
    }
}

#[test]
fn the_block_is_chunk_invariant_with_20_plans() {
    for cfg in configs() {
        let x = noise(5003, SEED_CHANNELIZE + 1);
        let mut block = ChannelizeBlock::new(&cfg, &mut VecStorage).unwrap();
        let want = block_output(
            &mut block,
            &x,
            &mut std::iter::repeat(5003),
            &mut std::iter::repeat(64),
        );
        for k in 0..20u64 {
            let chunks = ChunkPlan::new(SEED_CHANNELIZE + k, 1 + 37 * k as usize);
            let room = ChunkPlan::new(SEED_CHANNELIZE + 100 + k, 1 + 3 * k as usize);
            block.reset();
            let got = block_output(&mut block, &x, &mut chunks.sizes(), &mut room.sizes());
            assert!(got == want, "{cfg:?}, plan {k}");
        }
    }
}

#[test]
fn the_timing_is_that_of_the_kernel() {
    let cfg = ChannelizeConfig::default();
    let block = ChannelizeBlock::new(&cfg, &mut VecStorage).unwrap();
    let t = block.timing();
    assert_eq!(t.ratio, Ratio::new(2, 16).unwrap());
    assert_eq!(t.kind, RatioKind::Constant);
    assert_eq!(t.delay, Delay::new(127, 2).unwrap());
    assert_eq!(block.min_input(), 8);
    let rate = sdrk::Rate::hz(200_000).unwrap();
    assert_eq!(
        ChannelizeBlock::<VecStorage>::rate_out(&cfg, rate).unwrap(),
        sdrk::Rate::hz(25_000).unwrap()
    );
}

#[test]
fn a_wrong_output_width_is_rejected() {
    let mut block = ChannelizeBlock::new(&ChannelizeConfig::default(), &mut VecStorage).unwrap();
    let mut buf = vec![C32::default(); 64];
    let e = block
        .process(&[C32::default(); 16], LanesMut::new(&mut buf, 8).unwrap())
        .unwrap_err();
    assert!(matches!(
        e,
        Error::Core(sdrk_core::Error::LengthMismatch { .. })
    ));
}

#[test]
fn a_short_prototype_names_the_required_taps_per_branch() {
    // Three cases: the PMR446 plan with the default 8 taps per branch, a sharp plan at 80 dB
    // with 64 channels, and one that needs more than 32 taps per branch.
    let grid = ChannelizeConfig::default();
    let cases = [
        (
            ChannelizeConfig {
                passband: 0.4,
                stopband: 0.6,
                ..grid
            },
            19,
        ),
        (
            ChannelizeConfig {
                channels: 64,
                passband: 0.45,
                stopband: 0.8,
                attenuation_db: 80.0,
                taps_per_branch: 12,
                ..grid
            },
            channelizer_taps_per_branch(64, 0.45, 0.8, 80.0).unwrap(),
        ),
        (
            ChannelizeConfig {
                passband: 0.48,
                stopband: 0.52,
                ..grid
            },
            channelizer_taps_per_branch(16, 0.48, 0.52, 60.0).unwrap(),
        ),
    ];
    for (cfg, need) in cases {
        assert_eq!(cfg.required_taps_per_branch().unwrap(), need);
        assert!(need > cfg.taps_per_branch);
        let want = Error::OutOfRange {
            name: "taps_per_branch",
            min: need as f64,
            max: 32.0,
            got: cfg.taps_per_branch as f64,
        };
        assert_eq!(cfg.validate(), Err(want.clone()), "{cfg:?}");
        assert_eq!(
            ChannelizeBlock::new(&cfg, &mut VecStorage).unwrap_err(),
            want
        );
        assert_eq!(
            ChannelizeBlock::<VecStorage>::storage_bytes(&cfg).unwrap_err(),
            want
        );
        // With the required taps per branch (where it exists) the configuration is valid.
        if need <= 32 {
            let ok = ChannelizeConfig {
                taps_per_branch: need,
                ..cfg
            };
            assert_eq!(ok.validate(), Ok(()));
        }
    }
    assert!(cases[2].1 > 32);
}

#[test]
fn invalid_channel_counts_and_edges_are_rejected() {
    let grid = ChannelizeConfig::default();
    for channels in [3, 12, 100, 512] {
        let cfg = ChannelizeConfig { channels, ..grid };
        assert!(
            matches!(
                cfg.validate(),
                Err(Error::InvalidParameter {
                    name: "channels",
                    ..
                })
            ),
            "{channels}"
        );
    }
    for (passband, stopband) in [(0.6, 0.6), (0.7, 0.5), (0.5, 1.6), (0.0, 1.0)] {
        let cfg = ChannelizeConfig {
            passband,
            stopband,
            ..grid
        };
        assert!(
            matches!(cfg.validate(), Err(Error::InvalidParameter { .. })),
            "{passband} / {stopband}"
        );
    }
}

/// Random lane values: `steps` steps of `width` lanes.
fn lanes(width: usize, steps: usize, seed: u64) -> Vec<C32> {
    noise(width * steps, seed)
}

/// Runs a selector over the lanes in the step chunks of `chunks`, each call with the output
/// room of `room` samples (at least 1).
fn select_output(
    block: &mut ChannelSelectBlock,
    data: &[C32],
    chunks: &mut dyn Iterator<Item = usize>,
    room: &mut dyn Iterator<Item = usize>,
) -> Vec<C32> {
    let width = block.width();
    let steps = data.len() / width;
    let mut got = Vec::new();
    let mut buf = vec![C32::default(); 512];
    let mut at = 0;
    while at < steps {
        let end = (at + chunks.next().unwrap()).min(steps);
        while at < end {
            let n = room.next().unwrap().clamp(1, 512);
            let view = LanesRef::new(&data[at * width..end * width], width).unwrap();
            let p = block.process(view, &mut buf[..n]).unwrap();
            assert_eq!(p.consumed, p.produced);
            assert_eq!(p.produced, n.min(end - at));
            got.extend_from_slice(&buf[..p.produced]);
            at += p.consumed;
        }
    }
    got
}

#[test]
fn every_lane_equals_the_step_values_of_the_input() {
    for width in [4usize, 16, 256] {
        let data = lanes(width, 300, SEED_SELECT);
        let view = LanesRef::new(&data, width).unwrap();
        let half = (width / 2) as i64;
        for channel in -half..half {
            let mut block =
                ChannelSelectBlock::new(&ChannelSelectConfig { channel }, width).unwrap();
            let lane = block.lane();
            let got = select_output(
                &mut block,
                &data,
                &mut std::iter::repeat(300),
                &mut std::iter::repeat(512),
            );
            assert_eq!(got.len(), 300);
            for (i, v) in got.iter().enumerate() {
                assert_eq!(
                    *v,
                    view.step(i).unwrap()[lane],
                    "{width} lanes, channel {channel}"
                );
            }
        }
    }
}

#[test]
fn the_channel_is_signed() {
    for width in [4usize, 16, 256] {
        let m = width as i64;
        for (channel, lane) in [
            (-m / 2, width / 2),
            (-1, width - 1),
            (0, 0),
            (1, 1),
            (m / 2 - 1, width / 2 - 1),
        ] {
            let block = ChannelSelectBlock::new(&ChannelSelectConfig { channel }, width).unwrap();
            assert_eq!(block.lane(), lane, "{width} lanes, channel {channel}");
            assert_eq!(ChannelSelectConfig { channel }.lane(width), Ok(lane));
        }
    }
}

#[test]
fn a_channel_outside_the_width_is_out_of_range() {
    for width in [4usize, 16, 256] {
        let half = (width / 2) as i64;
        for channel in [half, -half - 1] {
            let e = ChannelSelectBlock::new(&ChannelSelectConfig { channel }, width).unwrap_err();
            assert_eq!(
                e,
                Error::OutOfRange {
                    name: "channel",
                    min: -half as f64,
                    max: (half - 1) as f64,
                    got: channel as f64,
                }
            );
            // The message names the range.
            let text = e.to_string();
            assert!(
                text.contains(&format!("{} to {}", -half, half - 1)),
                "{text}"
            );
        }
    }
    assert!(matches!(
        ChannelSelectConfig { channel: 0 }.lane(0),
        Err(Error::InvalidParameter { name: "width", .. })
    ));
    // An input of another width than the block was made for is rejected.
    let mut block = ChannelSelectBlock::new(&ChannelSelectConfig { channel: 1 }, 16).unwrap();
    let data = lanes(8, 4, 1);
    let mut out = [C32::default(); 4];
    assert!(matches!(
        block.process(LanesRef::new(&data, 8).unwrap(), &mut out),
        Err(Error::Core(sdrk_core::Error::LengthMismatch { .. }))
    ));
}

#[test]
fn the_selector_is_chunk_invariant_with_20_plans() {
    let width = 16;
    let data = lanes(width, 3001, SEED_SELECT + 1);
    for channel in [-8i64, -1, 0, 5] {
        let mut block = ChannelSelectBlock::new(&ChannelSelectConfig { channel }, width).unwrap();
        let want = select_output(
            &mut block,
            &data,
            &mut std::iter::repeat(3001),
            &mut std::iter::repeat(512),
        );
        for k in 0..20u64 {
            let chunks = ChunkPlan::new(SEED_SELECT + k, 1 + 50 * k as usize);
            let room = ChunkPlan::new(SEED_SELECT + 100 + k, 1 + 20 * k as usize);
            let got = select_output(&mut block, &data, &mut chunks.sizes(), &mut room.sizes());
            assert!(got == want, "channel {channel}, plan {k}");
        }
    }
}

#[test]
fn the_selector_has_unit_timing() {
    let block = ChannelSelectBlock::new(&ChannelSelectConfig { channel: -2 }, 16).unwrap();
    let t = block.timing();
    assert_eq!(
        (t.ratio, t.kind, t.delay),
        (Ratio::ONE, RatioKind::Constant, Delay::ZERO)
    );
    assert_eq!(block.min_input(), 1);
}

#[test]
fn the_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ChannelizeBlock<VecStorage>>();
    assert_send_sync::<ChannelizeConfig>();
    assert_send_sync::<ChannelSelectBlock>();
    assert_send_sync::<ChannelSelectConfig>();
}
