//! Blocks `noise_blank` and `auto_notch` (M33): exact arena, rate and timing, validation,
//! runtime parameters, the clean pass-through of the blanker, reset and `Send`.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::items_after_statements
)]

mod common;

use common::{check_exact_storage, rate, run_block};
use num_complex::Complex64;
use sdrk::{C32, CQ15, Delay, Ratio, RatioKind, Sdrk};
use sdrk_blocks::{
    ArenaStorage, AutoNotchConfig, AutoNotchRealBlock, Block, BlockSample, Error, NoiseBlankBlock,
    NoiseBlankConfig, ParamValue,
};
use sdrk_core::kernels::noise_blank::NoiseBlank;
use sdrk_testkit::{Pcg32, VecStorage};

type A<'a> = ArenaStorage<'a>;

/// Seed of the block tests of M32, reused for the signals of these tests.
const SEED: u64 = 470;

fn noise_blank_configs() -> [(NoiseBlankConfig, u64); 4] {
    let d = NoiseBlankConfig::default();
    [
        (d, 192_000),
        (d, 48_000),
        (
            NoiseBlankConfig {
                lookahead_s: 0.0,
                hold_s: 0.0,
                ..d
            },
            48_000,
        ),
        (
            NoiseBlankConfig {
                lookahead_s: 0.001,
                hold_s: 0.01,
                tau_s: 1.0,
                ..d
            },
            2_400_000,
        ),
    ]
}

fn exact_noise_blank<T: BlockSample>()
where
    Sdrk: NoiseBlank<T>,
{
    for (cfg, fs) in noise_blank_configs() {
        let r = rate(fs);
        let d = (cfg.lookahead_s * fs as f64 + 0.5).floor() as usize;
        let bytes = NoiseBlankBlock::<T, A<'_>>::storage_bytes(&cfg, r).unwrap();
        assert_eq!(bytes, d * size_of::<T>(), "{cfg:?} at {fs}");
        check_exact_storage(bytes, |a| {
            NoiseBlankBlock::<T, _>::new(&cfg, r, a).map(|_| ())
        });
        let block = NoiseBlankBlock::<T, _>::new(&cfg, r, &mut VecStorage).unwrap();
        assert_eq!(block.lookahead(), d);
        let t = block.timing();
        assert_eq!(t.ratio, Ratio::ONE);
        assert_eq!(t.kind, RatioKind::Constant);
        assert_eq!(t.delay, Delay::new(d as u64, 1).unwrap());
        assert_eq!(NoiseBlankBlock::<T, A<'_>>::rate_out(&cfg, r).unwrap(), r);
    }
}

#[test]
fn storage_is_exact_and_timing_is_the_lookahead() {
    exact_noise_blank::<C32>();
    exact_noise_blank::<CQ15>();
    for cfg in [
        AutoNotchConfig::default(),
        AutoNotchConfig {
            taps: 8,
            delay: 1,
            mu: 0.1,
        },
        AutoNotchConfig {
            taps: 256,
            delay: 64,
            mu: 0.0001,
        },
    ] {
        let r = rate(8_000);
        let bytes = AutoNotchRealBlock::<A<'_>>::storage_bytes(&cfg, r).unwrap();
        assert_eq!(bytes, 4 * (2 * cfg.taps + cfg.delay), "{cfg:?}");
        check_exact_storage(bytes, |a| AutoNotchRealBlock::new(&cfg, r, a).map(|_| ()));
        let block = AutoNotchRealBlock::new(&cfg, r, &mut VecStorage).unwrap();
        let t = block.timing();
        assert_eq!(t.ratio, Ratio::ONE);
        assert_eq!(t.delay, Delay::ZERO);
        assert_eq!(AutoNotchRealBlock::<A<'_>>::rate_out(&cfg, r).unwrap(), r);
    }
}

#[test]
fn construction_is_validated() {
    let r = rate(48_000);
    let d = NoiseBlankConfig::default();
    for bad in [
        NoiseBlankConfig {
            threshold_db: 5.9,
            ..d
        },
        NoiseBlankConfig {
            threshold_db: 40.1,
            ..d
        },
        NoiseBlankConfig {
            threshold_db: f64::NAN,
            ..d
        },
        NoiseBlankConfig { tau_s: 0.0009, ..d },
        NoiseBlankConfig { tau_s: 1.5, ..d },
        NoiseBlankConfig { hold_s: -1e-9, ..d },
        NoiseBlankConfig { hold_s: 0.011, ..d },
        NoiseBlankConfig {
            lookahead_s: 0.0011,
            ..d
        },
        NoiseBlankConfig {
            lookahead_s: -1e-9,
            ..d
        },
    ] {
        assert!(
            matches!(bad.validate(r), Err(Error::OutOfRange { .. })),
            "{bad:?}"
        );
        assert!(NoiseBlankBlock::<C32, _>::new(&bad, r, &mut VecStorage).is_err());
        assert!(NoiseBlankBlock::<CQ15, A<'_>>::storage_bytes(&bad, r).is_err());
    }
    // tau_s of 1 ms needs at least 1.45 samples per time constant (alpha <= 0.5): 1 kHz fails.
    let fast = NoiseBlankConfig { tau_s: 0.001, ..d };
    assert!(fast.validate(rate(1_000)).is_err());
    assert!(fast.validate(rate(2_000)).is_ok());

    let a = AutoNotchConfig::default();
    for bad in [
        AutoNotchConfig { taps: 7, ..a },
        AutoNotchConfig { taps: 257, ..a },
        AutoNotchConfig { delay: 0, ..a },
        AutoNotchConfig { delay: 65, ..a },
        AutoNotchConfig { mu: 0.00009, ..a },
        AutoNotchConfig { mu: 0.11, ..a },
        AutoNotchConfig { mu: f64::NAN, ..a },
    ] {
        assert!(
            matches!(bad.validate(r), Err(Error::OutOfRange { .. })),
            "{bad:?}"
        );
        assert!(AutoNotchRealBlock::new(&bad, r, &mut VecStorage).is_err());
        assert!(AutoNotchRealBlock::<A<'_>>::storage_bytes(&bad, r).is_err());
    }
}

#[test]
fn runtime_parameters_are_checked_and_build_parameters_are_read_only() {
    let r = rate(192_000);
    let mut nb =
        NoiseBlankBlock::<CQ15, _>::new(&NoiseBlankConfig::default(), r, &mut VecStorage).unwrap();
    nb.set_param("threshold_db", ParamValue::F64(30.0)).unwrap();
    assert_eq!(nb.param("threshold_db"), Ok(ParamValue::F64(30.0)));
    for (name, value) in [
        ("threshold_db", ParamValue::F64(41.0)),
        ("threshold_db", ParamValue::F64(f64::NAN)),
        ("threshold_db", ParamValue::I64(20)),
    ] {
        assert!(nb.set_param(name, value).is_err(), "{name} {value:?}");
        assert_eq!(nb.param("threshold_db"), Ok(ParamValue::F64(30.0)));
    }
    for name in ["tau_s", "hold_s", "lookahead_s"] {
        assert_eq!(
            nb.set_param(name, ParamValue::F64(0.0005)),
            Err(Error::ReadOnlyParameter),
            "{name}"
        );
    }

    let mut an =
        AutoNotchRealBlock::new(&AutoNotchConfig::default(), rate(8_000), &mut VecStorage).unwrap();
    an.set_param("mu", ParamValue::F64(0.002)).unwrap();
    assert_eq!(an.param("mu"), Ok(ParamValue::F64(0.002)));
    for value in [
        ParamValue::F64(0.2),
        ParamValue::F64(0.0),
        ParamValue::I64(1),
    ] {
        assert!(an.set_param("mu", value).is_err(), "{value:?}");
        assert_eq!(an.param("mu"), Ok(ParamValue::F64(0.002)));
    }
    for name in ["taps", "delay"] {
        assert_eq!(
            an.set_param(name, ParamValue::I64(16)),
            Err(Error::ReadOnlyParameter),
            "{name}"
        );
    }
    assert_eq!(an.param("taps"), Ok(ParamValue::I64(32)));
    assert_eq!(an.param("delay"), Ok(ParamValue::I64(8)));
}

/// A signal without impulses passes unchanged, delayed by `D`, in one piece and in pieces;
/// after a reset the block starts again with a zero delay line.
fn clean_signal_passes<T: BlockSample + PartialEq + core::fmt::Debug>()
where
    Sdrk: NoiseBlank<T>,
{
    let fs = 192_000;
    let cfg = NoiseBlankConfig::default();
    let mut rng = Pcg32::new(SEED, 31);
    let input: Vec<T> = (0..6000)
        .map(|i| {
            let phase = std::f64::consts::TAU * 10_000.0 / fs as f64 * i as f64;
            let noise = Complex64::new(rng.normal(), rng.normal()) * 0.05;
            T::from_c64_round(Complex64::new(0.3 * phase.cos(), 0.3 * phase.sin()) + noise)
        })
        .collect();
    let mut block = NoiseBlankBlock::<T, _>::new(&cfg, rate(fs), &mut VecStorage).unwrap();
    let d = block.lookahead();
    assert_eq!(d, 2);
    let out = run_block(&mut block, &input, None);
    assert_eq!(block.blanked(), 0);
    assert!(out[..d].iter().all(|y| *y == T::ZERO));
    assert_eq!(&out[d..], &input[..input.len() - d]);
    block.reset();
    let again = run_block(
        &mut block,
        &input,
        Some(sdrk_testkit::ChunkPlan::new(SEED, 37)),
    );
    assert_eq!(again, out);
}

#[test]
fn a_clean_signal_passes_delayed_and_reset_restarts() {
    clean_signal_passes::<C32>();
    clean_signal_passes::<CQ15>();
}

#[test]
fn the_notch_reset_clears_the_weights() {
    let cfg = AutoNotchConfig::default();
    let mut block = AutoNotchRealBlock::new(&cfg, rate(8_000), &mut VecStorage).unwrap();
    let input: Vec<f32> = (0..2000)
        .map(|n| 0.5 * (std::f32::consts::TAU * 0.125 * n as f32).sin())
        .collect();
    let first = run_block(&mut block, &input, None);
    assert!(block.weights().iter().any(|w| *w != 0.0));
    block.reset();
    assert!(block.weights().iter().all(|w| *w == 0.0));
    let second = run_block(&mut block, &input, None);
    assert_eq!(
        first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        second.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn public_types_are_send() {
    assert_send::<NoiseBlankBlock<C32, A<'static>>>();
    assert_send::<NoiseBlankBlock<CQ15, A<'static>>>();
    assert_send::<NoiseBlankBlock<CQ15, VecStorage>>();
    assert_sync::<NoiseBlankBlock<C32, A<'static>>>();
    assert_send::<AutoNotchRealBlock<A<'static>>>();
    assert_send::<AutoNotchRealBlock<VecStorage>>();
    assert_sync::<AutoNotchRealBlock<A<'static>>>();
    assert_send::<NoiseBlankConfig>();
    assert_send::<AutoNotchConfig>();
    fn is_block<B: Block>() {
        assert_send::<B>();
    }
    is_block::<NoiseBlankBlock<CQ15, VecStorage>>();
    is_block::<AutoNotchRealBlock<A<'static>>>();
}
