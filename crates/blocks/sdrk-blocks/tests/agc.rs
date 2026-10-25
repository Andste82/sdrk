//! Block `agc`: runtime parameters, validation, equality with the kernel, invariance and reset.
#![allow(
    clippy::unwrap_used,
    clippy::manual_assert_eq,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::panic,
    clippy::cast_lossless,
    clippy::items_after_statements,
    clippy::large_stack_frames,
    clippy::useless_vec,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::float_cmp,
    clippy::cast_possible_truncation
)]

mod common;

use common::{rate, run_block, signal};
use num_complex::Complex64;
use sdrk::{C32, CQ15, ComplexSample, Ratio, RatioKind, Sdrk};
use sdrk_blocks::{AgcBlock, AgcConfig, ArenaStorage, Block, BlockSample, Error, ParamValue};
use sdrk_core::kernels::agc::{Agc, AgcState};
use sdrk_testkit::{ChunkPlan, VecStorage};

const FS: u64 = 48_000;

fn config() -> AgcConfig {
    AgcConfig {
        reference_db: -6.0,
        attack_s: 0.002,
        release_s: 0.01,
        min_gain_db: -20.0,
        max_gain_db: 60.0,
        initial_gain_db: 0.0,
    }
}

fn new_block<T: BlockSample>(cfg: &AgcConfig) -> AgcBlock<T, VecStorage> {
    AgcBlock::<T, _>::new(cfg, rate(FS), &mut VecStorage).unwrap()
}

/// A complex tone of constant amplitude, 1 kHz.
fn tone<T: ComplexSample>(amplitude: f64, n: usize) -> Vec<T> {
    (0..n)
        .map(|i| {
            let p = std::f64::consts::TAU * 1000.0 * i as f64 / FS as f64;
            T::from_c64_round(Complex64::new(amplitude * p.cos(), amplitude * p.sin()))
        })
        .collect()
}

fn matches_the_kernel<T: BlockSample + PartialEq + core::fmt::Debug>()
where
    Sdrk: Agc<T>,
{
    let cfg = config();
    let mut input = tone::<T>(0.01, 3000);
    input.extend(tone::<T>(0.3, 3000));
    input.extend(signal::<T>(2000, 3, 0.02));
    let mut state = AgcState::<T>::new(
        10f64.powf(cfg.reference_db / 20.0),
        cfg.attack_s,
        cfg.release_s,
        cfg.min_gain_db,
        cfg.max_gain_db,
        cfg.initial_gain_db,
        rate(FS),
    )
    .unwrap();
    let mut want = vec![input[0]; input.len()];
    <Sdrk as Agc<T>>::agc(&Sdrk, &mut state, &input, &mut want).unwrap();
    let mut block = new_block::<T>(&cfg);
    let got = run_block(&mut block, &input, None);
    assert_eq!(got, want);
    assert_eq!(block.gain_db(), state.gain_db());
}

#[test]
fn the_block_equals_the_kernel() {
    matches_the_kernel::<C32>();
    matches_the_kernel::<CQ15>();
}

fn settles<T: BlockSample>()
where
    Sdrk: Agc<T>,
{
    // A weak tone is raised to the reference amplitude of 0.5.
    let mut block = new_block::<T>(&config());
    let out = run_block(&mut block, &tone::<T>(0.02, 4800), None);
    let last = out[out.len() - 2000..]
        .iter()
        .map(|s| s.to_c64().norm_sqr().sqrt())
        .fold(0.0, f64::max);
    assert!((last - 0.5).abs() < 0.02, "{last}");
    assert!((block.gain_db() - 20.0 * (0.5f64 / 0.02).log10()).abs() < 0.3);
}

#[test]
fn a_weak_tone_settles_at_the_reference() {
    settles::<C32>();
    settles::<CQ15>();
}

/// Runs `first` and `second` through a block, once with the parameter change between them and
/// once without, and returns both second halves.
fn with_and_without<T: BlockSample + PartialEq + core::fmt::Debug>(
    cfg: &AgcConfig,
    first: &[T],
    second: &[T],
    change: &[(&str, f64)],
) -> (Vec<T>, Vec<T>, AgcBlock<T, VecStorage>)
where
    Sdrk: Agc<T>,
{
    let mut a = new_block::<T>(cfg);
    let _ = run_block(&mut a, first, None);
    let base = run_block(&mut a, second, None);
    let mut b = new_block::<T>(cfg);
    let before = run_block(&mut b, first, None);
    let mut c = new_block::<T>(cfg);
    assert_eq!(run_block(&mut c, first, None), before);
    for (name, v) in change {
        b.set_param(name, ParamValue::F64(*v)).unwrap();
        assert_eq!(b.param(name), Ok(ParamValue::F64(*v)));
    }
    let changed = run_block(&mut b, second, None);
    (base, changed, b)
}

fn runtime_parameters<T: BlockSample + PartialEq + core::fmt::Debug>()
where
    Sdrk: Agc<T>,
{
    let loud = tone::<T>(0.4, 4000);
    let quiet = tone::<T>(0.01, 4000);

    // reference_db: the target level moves.
    let cfg = config();
    let (base, changed, _) = with_and_without(&cfg, &loud, &loud, &[("reference_db", -20.0)]);
    assert_ne!(base, changed);
    // The sample after the call starts is still scaled by the old gain; the loop acts on it.
    assert_ne!(base[1..], changed[1..]);

    // attack_s only acts while the output is too loud (reference -20 dBFS, input 0.4).
    let cfg = AgcConfig {
        reference_db: -20.0,
        ..config()
    };
    let (base, changed, _) = with_and_without(&cfg, &loud[..1000], &loud, &[("attack_s", 0.05)]);
    assert_ne!(base, changed);
    // ... and does not act while it is too quiet.
    let cfg = config();
    let (base, changed, _) =
        with_and_without(&cfg, &quiet[..100], &quiet[..100], &[("attack_s", 0.05)]);
    assert_eq!(base, changed);

    // release_s only acts while the output is too quiet.
    let (base, changed, _) = with_and_without(&cfg, &quiet, &quiet, &[("release_s", 0.1)]);
    assert_ne!(base, changed);
    let cfg2 = AgcConfig {
        reference_db: -20.0,
        initial_gain_db: 10.0,
        ..config()
    };
    let (base, changed, _) =
        with_and_without(&cfg2, &loud[..5], &loud[..10], &[("release_s", 0.1)]);
    assert_eq!(base, changed);

    // min_gain_db clamps the current gain at once.
    let cfg = AgcConfig {
        reference_db: -20.0,
        ..config()
    };
    let (base, changed, b) = with_and_without(&cfg, &loud, &loud[..16], &[("min_gain_db", -1.0)]);
    assert_ne!(base, changed);
    assert!(
        (b.gain_db() + 1.0).abs() < 0.2,
        "gain {} after the clamp",
        b.gain_db()
    );

    // max_gain_db clamps the current gain at once.
    let cfg = config();
    let (base, changed, b) = with_and_without(&cfg, &quiet, &quiet[..16], &[("max_gain_db", 1.0)]);
    assert_ne!(base, changed);
    assert!(b.gain_db() <= 1.001, "gain {}", b.gain_db());
    assert!(b.gain_db() > 0.5, "gain {}", b.gain_db());
}

#[test]
fn every_runtime_parameter_acts_from_the_next_call() {
    runtime_parameters::<C32>();
    runtime_parameters::<CQ15>();
}

#[test]
fn set_params_keeps_the_gain() {
    let mut block = new_block::<C32>(&config());
    let _ = run_block(&mut block, &tone::<C32>(0.02, 3000), None);
    let g = block.gain_db();
    assert!(g > 10.0);
    block.set_param("attack_s", ParamValue::F64(0.02)).unwrap();
    block
        .set_param("reference_db", ParamValue::F64(-10.0))
        .unwrap();
    assert_eq!(block.gain_db(), g);
    // The initial gain is a build parameter; reset goes back to it.
    block.reset();
    assert_eq!(block.gain_db(), 0.0);
}

#[test]
fn parameter_errors_leave_the_block_unchanged() {
    let mut block = new_block::<C32>(&config());
    for (name, v) in [
        ("reference_db", 0.5),
        ("reference_db", -30.5),
        ("reference_db", f64::NAN),
        ("attack_s", 0.0),
        ("attack_s", 101.0),
        ("release_s", f64::NAN),
        ("min_gain_db", -41.0),
        ("max_gain_db", 91.0),
    ] {
        assert!(
            matches!(
                block.set_param(name, ParamValue::F64(v)),
                Err(Error::OutOfRange { .. })
            ),
            "{name} = {v}"
        );
    }
    // Inside the range of the description, but rejected by the kernel: the minimum above
    // the maximum, and a time constant of less than one sample (rate above 0.5).
    for (name, v) in [
        ("min_gain_db", 70.0),
        ("max_gain_db", -30.0),
        ("attack_s", 1e-6),
    ] {
        assert!(
            block.set_param(name, ParamValue::F64(v)).is_err(),
            "{name} = {v}"
        );
    }
    assert_eq!(block.param("min_gain_db"), Ok(ParamValue::F64(-20.0)));
    assert_eq!(block.param("max_gain_db"), Ok(ParamValue::F64(60.0)));
    assert_eq!(block.param("attack_s"), Ok(ParamValue::F64(0.002)));
    assert_eq!(
        block.set_param("initial_gain_db", ParamValue::F64(3.0)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(
        block.set_param("attack_s", ParamValue::Bool(true)),
        Err(Error::WrongParamType)
    );
    assert_eq!(
        block.set_param("nope", ParamValue::F64(1.0)),
        Err(Error::UnknownParameter)
    );
    assert_eq!(block.param("nope"), Err(Error::UnknownParameter));
}

#[test]
fn construction_is_validated() {
    let r = rate(FS);
    let bad = [
        AgcConfig {
            reference_db: 1.0,
            ..config()
        },
        AgcConfig {
            reference_db: -31.0,
            ..config()
        },
        AgcConfig {
            attack_s: 0.0,
            ..config()
        },
        AgcConfig {
            release_s: f64::NAN,
            ..config()
        },
        // fs * tau of 0.048 samples: the rate is above 0.5.
        AgcConfig {
            attack_s: 1e-6,
            ..config()
        },
        AgcConfig {
            min_gain_db: -41.0,
            ..config()
        },
        AgcConfig {
            max_gain_db: 91.0,
            ..config()
        },
        AgcConfig {
            min_gain_db: 10.0,
            max_gain_db: 5.0,
            initial_gain_db: 7.0,
            ..config()
        },
        AgcConfig {
            initial_gain_db: 61.0,
            ..config()
        },
        AgcConfig {
            initial_gain_db: -21.0,
            ..config()
        },
    ];
    for cfg in bad {
        assert!(cfg.validate(r).is_err(), "{cfg:?}");
        assert!(AgcBlock::<C32, _>::new(&cfg, r, &mut VecStorage).is_err());
        assert!(AgcBlock::<CQ15, VecStorage>::storage_bytes(&cfg, r).is_err());
        assert!(AgcBlock::<CQ15, VecStorage>::rate_out(&cfg, r).is_err());
    }
    assert!(config().validate(r).is_ok());
    assert!(AgcConfig::default().validate(r).is_ok());
}

#[test]
fn rate_storage_and_timing() {
    let cfg = config();
    let r = rate(FS);
    assert_eq!(AgcBlock::<C32, VecStorage>::rate_out(&cfg, r), Ok(r));
    assert_eq!(AgcBlock::<CQ15, VecStorage>::storage_bytes(&cfg, r), Ok(0));
    // An arena of zero bytes is enough.
    let mut arena = ArenaStorage::new(&mut []);
    let block = AgcBlock::<C32, _>::new(&cfg, r, &mut arena).unwrap();
    let t = block.timing();
    assert_eq!(t.ratio, Ratio::ONE);
    assert_eq!(t.kind, RatioKind::Constant);
    assert_eq!(t.delay, sdrk::Delay::ZERO);
    assert_eq!(block.min_input(), 1);
    assert_eq!(block.description().kind, "agc");
}

fn twenty_plans() -> Vec<ChunkPlan> {
    (0..20u64)
        .map(|k| ChunkPlan::new(500 + k, 1 + ((k * 61) % 700) as usize))
        .collect()
}

fn invariant_and_reset<T: BlockSample + PartialEq + core::fmt::Debug>()
where
    Sdrk: Agc<T>,
{
    let cfg = config();
    let mut input = tone::<T>(0.01, 2500);
    input.extend(tone::<T>(0.3, 2500));
    input.extend(signal::<T>(2500, 5, 0.03));
    let whole = run_block(&mut new_block::<T>(&cfg), &input, None);
    for plan in twenty_plans() {
        let chunked = run_block(&mut new_block::<T>(&cfg), &input, Some(plan));
        assert_eq!(chunked, whole, "{plan:?}");
    }
    // After reset the block behaves like a new one.
    let mut block = new_block::<T>(&cfg);
    let _ = run_block(&mut block, &input[..4000], None);
    block.reset();
    assert_eq!(block.gain_db(), 0.0);
    assert_eq!(run_block(&mut block, &input, None), whole);
}

#[test]
fn block_size_invariance_and_reset() {
    invariant_and_reset::<C32>();
    invariant_and_reset::<CQ15>();
}

#[test]
fn a_nan_sample_leaves_the_gain_unchanged() {
    let cfg = config();
    let mut block = new_block::<C32>(&cfg);
    let _ = run_block(&mut block, &tone::<C32>(0.1, 500), None);
    let g = block.gain_db();
    let out = run_block(&mut block, &[C32::new(f32::NAN, 0.0)], None);
    assert!(out[0].re.is_nan());
    assert_eq!(block.gain_db(), g);
    let out = run_block(&mut block, &[C32::new(f32::INFINITY, 0.0)], None);
    assert!(out[0].re.is_infinite());
    assert!((block.gain_db() - cfg.min_gain_db).abs() < 1e-3);
}
