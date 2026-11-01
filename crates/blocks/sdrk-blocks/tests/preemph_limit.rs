//! Block `preemph_limit`: description, construction in an exact arena, the output equals the
//! kernel, no sample above the limit for input up to 20 times full scale, a limit change from
//! the next call, the 40 kHz rule, validation, chunk invariance and reset.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::panic,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::needless_pass_by_value
)]

mod common;

use common::{plans, rate, run_block};
use sdrk::{Q15, Ratio, RatioKind, RealSample, Sdrk};
use sdrk_blocks::design::preemphasis_coefs;
use sdrk_blocks::{
    ArenaStorage, Block, BlockReal, Error, Mutability, ParamValue, PreemphLimitBlock,
    PreemphLimitConfig, all_descriptions,
};
use sdrk_core::kernels::preemph_limit::{PreemphLimit, PreemphLimitState};
use sdrk_testkit::{Pcg32, VecStorage};

const FS: u64 = 240_000;

fn new_block<R: BlockReal>(cfg: &PreemphLimitConfig, hz: u64) -> PreemphLimitBlock<R, VecStorage>
where
    Sdrk: PreemphLimit<R>,
{
    PreemphLimitBlock::<R, _>::new(cfg, rate(hz), &mut VecStorage).unwrap()
}

/// Noise up to `scale` times full scale, a 1 kHz tone, silence and spikes.
fn loud<R: RealSample>(n: usize, scale: f64, seed: u64) -> Vec<R> {
    let mut rng = Pcg32::new(seed, 5);
    (0..n)
        .map(|i| {
            let tone = (std::f64::consts::TAU * 1000.0 * i as f64 / FS as f64).sin();
            let v = if (i / 4000) % 3 == 2 {
                0.0
            } else {
                scale * (0.7 * rng.normal() * 0.3 + 0.3 * tone)
            };
            R::from_f64_round(v)
        })
        .collect()
}

fn as_f64<R: RealSample>(x: R) -> f64 {
    x.to_f64()
}

#[test]
fn the_description_follows_the_rules_of_the_blocks() {
    let d = all_descriptions()
        .iter()
        .find(|d| d.kind == "preemph_limit")
        .unwrap();
    assert!(d.summary.chars().count() <= 120 && d.summary.ends_with('.'));
    // The details name the position and the measurement behind it.
    assert!(d.details.contains("fm_mod"));
    assert!(d.details.contains("1.41"));
    let names: Vec<_> = d.params.iter().map(|p| (p.name, p.mutability)).collect();
    assert_eq!(
        names,
        [
            ("preemphasis_us", Mutability::Build),
            ("limit", Mutability::Runtime),
            ("hold_s", Mutability::Build),
            ("release_s", Mutability::Build),
        ]
    );
    // Build parameters are read-only, `limit` changes.
    let mut b = new_block::<f32>(&PreemphLimitConfig::default(), FS);
    assert_eq!(
        b.set_param("hold_s", ParamValue::F64(0.1)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(b.set_param("limit", ParamValue::F64(0.5)), Ok(()));
    assert_eq!(b.param("limit"), Ok(ParamValue::F64(0.5)));
    assert!(b.set_param("limit", ParamValue::F64(0.05)).is_err());
    assert!(b.set_param("limit", ParamValue::F64(1.5)).is_err());
    assert_eq!(b.param("nope"), Err(Error::UnknownParameter));
}

#[test]
fn the_figures_are_unit_timing_without_storage() {
    let cfg = PreemphLimitConfig::default();
    let b = new_block::<Q15>(&cfg, FS);
    let t = b.timing();
    assert_eq!((t.ratio, t.kind), (Ratio::ONE, RatioKind::Constant));
    assert_eq!(t.delay.as_f64(), 0.0);
    assert_eq!(b.min_input(), 1);
    assert_eq!(
        PreemphLimitBlock::<f32, VecStorage>::rate_out(&cfg, rate(FS)).unwrap(),
        rate(FS)
    );
    assert_eq!(
        PreemphLimitBlock::<f32, VecStorage>::storage_bytes(&cfg, rate(FS)).unwrap(),
        0
    );
    // An empty arena is enough.
    let mut arena = ArenaStorage::new(&mut []);
    assert!(PreemphLimitBlock::<f32, _>::new(&cfg, rate(FS), &mut arena).is_ok());
}

/// The block equals the kernel with the coefficients of the design function.
fn equals_the_kernel<R: BlockReal>()
where
    Sdrk: PreemphLimit<R>,
{
    let cfg = PreemphLimitConfig {
        preemphasis_us: 75.0,
        limit: 0.6,
        ..PreemphLimitConfig::default()
    };
    let input = loud::<R>(20_000, 3.0, 1);
    let mut want = vec![R::default(); input.len()];
    let mut s = PreemphLimitState::<R>::new(
        preemphasis_coefs(75e-6, FS as f64).unwrap(),
        0.6,
        0.02,
        0.2,
        rate(FS),
    )
    .unwrap();
    <Sdrk as PreemphLimit<R>>::preemph_limit(&Sdrk, &mut s, &input, &mut want).unwrap();
    let got = run_block(&mut new_block::<R>(&cfg, FS), &input, None);
    assert_eq!(got, want);
    // The chunk plans give the same bits.
    for plan in plans(77) {
        let chunked = run_block(&mut new_block::<R>(&cfg, FS), &input, Some(plan));
        assert!(chunked == want, "chunked run differs");
    }
}

#[test]
fn the_block_equals_the_kernel_and_is_chunk_invariant() {
    equals_the_kernel::<f32>();
    equals_the_kernel::<Q15>();
}

fn no_sample_above_the_limit<R: BlockReal>()
where
    Sdrk: PreemphLimit<R>,
{
    for (us, limit) in [(50.0, 1.0), (530.0, 0.85), (0.0, 0.3)] {
        let cfg = PreemphLimitConfig {
            preemphasis_us: us,
            limit,
            ..PreemphLimitConfig::default()
        };
        let mut block = new_block::<R>(&cfg, FS);
        let input = loud::<R>(30_000, 20.0, 2);
        let out = run_block(&mut block, &input, None);
        // The Q15 output limit is rounded to a raw value.
        let bound = (limit * 32768.0 + 0.5).floor().min(32767.0) / 32768.0;
        // The f32 limit is the f32 nearest to `limit`.
        let top = if std::mem::size_of::<R>() == 2 {
            bound
        } else {
            f64::from(limit as f32)
        };
        assert!(out.iter().all(|v| as_f64(*v).abs() <= top), "{us} {limit}");
        assert!(out.iter().any(|v| as_f64(*v).abs() > 0.9 * top));
        // A lower limit applies from the next call and keeps the state.
        block.set_param("limit", ParamValue::F64(0.25)).unwrap();
        let out = run_block(&mut block, &input, None);
        assert!(out.iter().all(|v| as_f64(*v).abs() <= 0.25), "{us} {limit}");
        assert!(out.iter().any(|v| as_f64(*v).abs() > 0.2), "{us} {limit}");
    }
}

#[test]
fn no_sample_exceeds_the_limit_before_and_after_a_change() {
    no_sample_above_the_limit::<f32>();
    no_sample_above_the_limit::<Q15>();
}

#[test]
fn a_limit_change_acts_from_the_next_sample() {
    let cfg = PreemphLimitConfig {
        preemphasis_us: 0.0,
        ..PreemphLimitConfig::default()
    };
    let mut block = new_block::<f32>(&cfg, FS);
    let mut out = [0.0f32; 4];
    block.process(&[0.9, 0.9, 0.9, 0.9], &mut out).unwrap();
    assert_eq!(out, [0.9; 4]);
    block.set_param("limit", ParamValue::F64(0.5)).unwrap();
    block.process(&[0.9, 0.2, 0.9, 0.2], &mut out).unwrap();
    assert_eq!(out[0], 0.5);
    assert!(out.iter().all(|v| *v <= 0.5));
}

#[test]
fn zero_pre_emphasis_passes_a_signal_below_the_limit() {
    let cfg = PreemphLimitConfig {
        preemphasis_us: 0.0,
        ..PreemphLimitConfig::default()
    };
    let input = loud::<f32>(5000, 1.0, 3);
    let input: Vec<f32> = input.iter().map(|v| v.clamp(-0.9, 0.9)).collect();
    let out = run_block(&mut new_block::<f32>(&cfg, 8_000), &input, None);
    assert_eq!(out, input);
}

#[test]
fn a_pre_emphasis_below_40_khz_is_rejected() {
    let cfg = PreemphLimitConfig::default();
    for hz in [8_000, 32_000, 39_999] {
        assert!(matches!(
            PreemphLimitBlock::<f32, VecStorage>::rate_out(&cfg, rate(hz)),
            Err(Error::OutOfRange { .. })
        ));
        assert!(PreemphLimitBlock::<Q15, _>::new(&cfg, rate(hz), &mut VecStorage).is_err());
    }
    assert!(cfg.validate(rate(40_000)).is_ok());
    // Without pre-emphasis any rate is fine.
    let off = PreemphLimitConfig {
        preemphasis_us: 0.0,
        ..cfg
    };
    assert!(off.validate(rate(8_000)).is_ok());
}

#[test]
fn invalid_parameters_are_rejected() {
    let d = PreemphLimitConfig::default();
    for cfg in [
        PreemphLimitConfig {
            preemphasis_us: -1.0,
            ..d
        },
        PreemphLimitConfig {
            preemphasis_us: 1000.5,
            ..d
        },
        PreemphLimitConfig {
            preemphasis_us: f64::NAN,
            ..d
        },
        PreemphLimitConfig { limit: 0.0, ..d },
        PreemphLimitConfig { limit: 0.09, ..d },
        PreemphLimitConfig { limit: 1.01, ..d },
        PreemphLimitConfig {
            limit: f64::NAN,
            ..d
        },
        PreemphLimitConfig { hold_s: -0.1, ..d },
        PreemphLimitConfig { hold_s: 1.5, ..d },
        PreemphLimitConfig {
            release_s: 0.0,
            ..d
        },
        PreemphLimitConfig {
            release_s: 0.0005,
            ..d
        },
        PreemphLimitConfig {
            release_s: 11.0,
            ..d
        },
    ] {
        assert!(cfg.validate(rate(FS)).is_err(), "{cfg:?}");
    }
    let mut cfg = d;
    assert_eq!(
        cfg.set("limit", ParamValue::F64(2.0)),
        Err(Error::OutOfRange {
            name: "limit",
            min: 0.1,
            max: 1.0,
            got: 2.0
        })
    );
    assert_eq!(
        cfg.set("nope", ParamValue::F64(1.0)),
        Err(Error::UnknownParameter)
    );
    assert!(cfg.set("hold_s", ParamValue::Bool(true)).is_err());
}

#[test]
fn reset_restores_the_start_state() {
    let cfg = PreemphLimitConfig::default();
    let input = loud::<f32>(6000, 5.0, 4);
    let mut block = new_block::<f32>(&cfg, FS);
    let first = run_block(&mut block, &input, None);
    assert!(block.gain() < 1.0);
    block.reset();
    assert_eq!(block.gain(), 1.0);
    assert_eq!(run_block(&mut block, &input, None), first);
}
