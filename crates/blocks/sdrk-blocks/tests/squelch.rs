//! Block `squelch`: gate timing, runtime parameters, validation, invariance and reset.
//! The expected switching samples follow from the closed form of the power average.
#![allow(
    clippy::unwrap_used,
    clippy::manual_assert_eq,
    clippy::manual_contains,
    clippy::indexing_slicing,
    clippy::too_many_lines,
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

use common::{plans, rate, run_block, signal};
use num_complex::Complex64;
use sdrk::{C32, CQ15, ComplexSample, Ratio, RatioKind, Sdrk};
use sdrk_blocks::{
    ArenaStorage, Block, BlockSample, Error, ParamValue, SquelchBlock, SquelchConfig,
};
use sdrk_core::SampleType;
use sdrk_core::kernels::squelch::{Squelch, SquelchState};
use sdrk_testkit::VecStorage;

const FS: u64 = 48_000;

fn config() -> SquelchConfig {
    SquelchConfig {
        threshold_db: -40.0,
        hysteresis_db: 3.0,
        tau_s: 0.002,
    }
}

fn alpha(cfg: &SquelchConfig) -> f64 {
    1.0 - (-1.0 / (FS as f64 * cfg.tau_s)).exp()
}

fn constant<T: ComplexSample>(amplitude: f64, n: usize) -> Vec<T> {
    vec![T::from_c64_round(Complex64::new(amplitude, 0.0)); n]
}

fn zeros<T: ComplexSample>(n: usize) -> Vec<T> {
    constant(0.0, n)
}

fn new_block<T: BlockSample>(cfg: &SquelchConfig) -> SquelchBlock<T, VecStorage> {
    SquelchBlock::<T, _>::new(cfg, rate(FS), &mut VecStorage).unwrap()
}

/// Index of the first output sample that equals the (non-zero) input sample.
fn first_pass<T: ComplexSample + PartialEq>(input: &[T], output: &[T]) -> Option<usize> {
    let zero = T::from_c64_round(Complex64::new(0.0, 0.0));
    input
        .iter()
        .zip(output)
        .position(|(i, o)| *i != zero && i == o)
}

/// The gate after every sample, from a model of the kernel written for this test: the `f32`
/// recurrence for `C32` and the `i64` recurrence of the Q30 power for `CQ15`.
fn gate_trace<T: ComplexSample>(cfg: &SquelchConfig, input: &[T]) -> Vec<bool> {
    let a = alpha(cfg);
    let (open_db, close_db) = (cfg.threshold_db, cfg.threshold_db - cfg.hysteresis_db);
    let mut gate = false;
    let mut trace = Vec::with_capacity(input.len());
    if T::TYPE == SampleType::CQ15 {
        let alpha_raw = (a * 32768.0 + 0.5).floor() as i64;
        let level = |db: f64| (10f64.powf(db / 10.0) * 2f64.powi(30) + 0.5).floor() as i64;
        let (open, close) = (level(open_db), level(close_db));
        let mut p = 0i64;
        for s in input {
            let c = s.to_c64();
            let (re, im) = (
                (c.re * 32768.0).round() as i64,
                (c.im * 32768.0).round() as i64,
            );
            p += (alpha_raw * (re * re + im * im - p) + 16384) >> 15;
            if p >= open {
                gate = true;
            } else if p < close {
                gate = false;
            }
            trace.push(gate);
        }
    } else {
        let alpha = a as f32;
        let (open, close) = (
            10f64.powf(open_db / 10.0) as f32,
            10f64.powf(close_db / 10.0) as f32,
        );
        let mut p = 0f32;
        for s in input {
            let c = s.to_c64();
            let (re, im) = (c.re as f32, c.im as f32);
            p += alpha * (re * re + im * im - p);
            if p >= open {
                gate = true;
            } else if p < close {
                gate = false;
            }
            trace.push(gate);
        }
    }
    trace
}

fn gate_opens_and_closes<T: BlockSample + PartialEq>()
where
    Sdrk: Squelch<T>,
{
    let cfg = config();
    let burst = 1500;
    let mut input = constant::<T>(0.05, burst);
    input.extend(zeros::<T>(1500));
    let trace = gate_trace(&cfg, &input);
    let mut block = new_block::<T>(&cfg);
    let out = run_block(&mut block, &input, None);

    // The first sample that passes is the first one with the gate open, exactly.
    let open_at = trace.iter().position(|g| *g).unwrap();
    assert!(open_at > 2 && open_at < burst, "{open_at}");
    let first = first_pass(&input, &out).unwrap();
    assert_eq!(first, open_at);
    let zero = T::from_c64_round(Complex64::new(0.0, 0.0));
    assert!(out[..first].iter().all(|s| *s == zero));
    assert!(out[first..burst].iter().all(|s| s == &input[0]));

    // The gate closes at the first sample after the burst where the model gate is closed:
    // one sample before it the block is open, with that sample it is closed.
    let close_at = burst + trace[burst..].iter().position(|g| !*g).unwrap();
    assert!(close_at > burst + 100 && close_at < 3000, "{close_at}");
    let mut replay = new_block::<T>(&cfg);
    let _ = run_block(&mut replay, &input[..close_at], None);
    assert!(replay.is_open(), "open before sample {close_at}");
    let _ = run_block(&mut replay, &input[close_at..=close_at], None);
    assert!(!replay.is_open(), "closed with sample {close_at}");
    assert!(!block.is_open());
}

#[test]
fn the_gate_opens_at_the_documented_sample() {
    gate_opens_and_closes::<C32>();
    gate_opens_and_closes::<CQ15>();
}

fn closing_sample<T: BlockSample + PartialEq>()
where
    Sdrk: Squelch<T>,
{
    // A small signal between the close and the open level keeps an open gate open and a
    // closed gate closed (hysteresis); a signal below the close level closes it.
    let cfg = config();
    let mut block = new_block::<T>(&cfg);
    let loud = constant::<T>(0.1, 1000);
    let _ = run_block(&mut block, &loud, None);
    assert!(block.is_open());
    // -41.5 dBFS: amplitude 10^(-41.5 / 20)
    let band = 10f64.powf(-41.5 / 20.0);
    let mid = constant::<T>(band, 4000);
    let out = run_block(&mut block, &mid, None);
    assert!(block.is_open(), "stays open inside the hysteresis band");
    assert!(out.last().unwrap() == mid.last().unwrap());
    let quiet = constant::<T>(0.001, 4000);
    let out = run_block(&mut block, &quiet, None);
    assert!(!block.is_open());
    let zero = T::from_c64_round(Complex64::new(0.0, 0.0));
    assert!(out.last().unwrap() == &zero);
    // A closed gate does not open inside the band either.
    let mut block = new_block::<T>(&cfg);
    let out = run_block(&mut block, &mid, None);
    assert!(!block.is_open());
    assert!(out.iter().all(|s| *s == zero));
}

#[test]
fn the_hysteresis_band_keeps_the_state() {
    closing_sample::<C32>();
    closing_sample::<CQ15>();
}

fn matches_the_kernel<T: BlockSample + PartialEq + core::fmt::Debug>()
where
    Sdrk: Squelch<T>,
{
    let cfg = config();
    let mut input = signal::<T>(2000, 3, 0.05);
    input.extend(zeros::<T>(1000));
    input.extend(signal::<T>(1500, 4, 0.1));
    let mut state =
        SquelchState::<T>::new(cfg.threshold_db, cfg.hysteresis_db, cfg.tau_s, rate(FS)).unwrap();
    let mut want = vec![input[0]; input.len()];
    <Sdrk as Squelch<T>>::squelch(&Sdrk, &mut state, &input, &mut want).unwrap();
    let got = run_block(&mut new_block::<T>(&cfg), &input, None);
    assert_eq!(got, want);
}

#[test]
fn the_block_equals_the_kernel() {
    matches_the_kernel::<C32>();
    matches_the_kernel::<CQ15>();
}

fn runtime_parameters<T: BlockSample + PartialEq>()
where
    Sdrk: Squelch<T>,
{
    let cfg = config();
    let mut block = new_block::<T>(&cfg);
    let zero = T::from_c64_round(Complex64::new(0.0, 0.0));
    let tone = constant::<T>(0.1, 1500); // -20 dBFS
    let out = run_block(&mut block, &tone, None);
    assert!(block.is_open());
    assert!(out.last().unwrap() == tone.last().unwrap());

    // Raising the level above the signal: the first sample of the next call is closed.
    block
        .set_param("threshold_db", ParamValue::F64(-10.0))
        .unwrap();
    assert_eq!(block.param("threshold_db"), Ok(ParamValue::F64(-10.0)));
    let out = run_block(&mut block, &tone[..10], None);
    assert!(out.iter().all(|s| *s == zero));
    assert!(!block.is_open());

    // Lowering it again opens at the next sample (the average was kept).
    block
        .set_param("threshold_db", ParamValue::F64(-50.0))
        .unwrap();
    let out = run_block(&mut block, &tone[..10], None);
    assert!(out.iter().all(|s| *s == tone[0]));
    assert!(block.is_open());

    // The hysteresis: closing level -20 dBFS - 0 = the signal power itself is just above it
    // while the settled average stays below the signal, so a wide hysteresis keeps it open.
    block
        .set_param("hysteresis_db", ParamValue::F64(40.0))
        .unwrap();
    assert_eq!(block.param("hysteresis_db"), Ok(ParamValue::F64(40.0)));
    block
        .set_param("threshold_db", ParamValue::F64(-15.0))
        .unwrap();
    let out = run_block(&mut block, &tone[..10], None);
    assert!(block.is_open(), "closes only below -55 dBFS");
    assert!(out.iter().all(|s| *s == tone[0]));
    block
        .set_param("hysteresis_db", ParamValue::F64(0.0))
        .unwrap();
    let out = run_block(&mut block, &tone[..10], None);
    assert!(!block.is_open());
    assert!(out.iter().all(|s| *s == zero));
}

#[test]
fn threshold_and_hysteresis_apply_from_the_next_call() {
    runtime_parameters::<C32>();
    runtime_parameters::<CQ15>();
}

#[test]
fn parameter_errors() {
    let mut block = new_block::<C32>(&config());
    for (name, v) in [
        ("threshold_db", 0.5),
        ("threshold_db", -150.5),
        ("threshold_db", f64::NAN),
        ("hysteresis_db", -0.5),
        ("hysteresis_db", 40.5),
    ] {
        assert!(
            matches!(
                block.set_param(name, ParamValue::F64(v)),
                Err(Error::OutOfRange { .. })
            ),
            "{name} = {v}"
        );
    }
    // Nothing changed after the rejections.
    assert_eq!(block.param("threshold_db"), Ok(ParamValue::F64(-40.0)));
    assert_eq!(block.param("hysteresis_db"), Ok(ParamValue::F64(3.0)));
    assert_eq!(
        block.set_param("tau_s", ParamValue::F64(0.1)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(
        block.set_param("threshold_db", ParamValue::Bool(true)),
        Err(Error::WrongParamType)
    );
    assert_eq!(
        block.set_param("nope", ParamValue::F64(1.0)),
        Err(Error::UnknownParameter)
    );
    assert_eq!(block.param("nope"), Err(Error::UnknownParameter));
    assert_eq!(block.param("tau_s"), Ok(ParamValue::F64(0.002)));
}

#[test]
fn construction_is_validated() {
    let r = rate(FS);
    let bad = [
        SquelchConfig {
            threshold_db: 1.0,
            ..config()
        },
        SquelchConfig {
            threshold_db: -151.0,
            ..config()
        },
        SquelchConfig {
            hysteresis_db: -1.0,
            ..config()
        },
        SquelchConfig {
            hysteresis_db: 41.0,
            ..config()
        },
        SquelchConfig {
            tau_s: 0.0,
            ..config()
        },
        SquelchConfig {
            tau_s: f64::NAN,
            ..config()
        },
        // fs * tau of 0.048 samples: the smoothing coefficient is above 0.5.
        SquelchConfig {
            tau_s: 1e-6,
            ..config()
        },
    ];
    for cfg in bad {
        assert!(cfg.validate(r).is_err(), "{cfg:?}");
        assert!(SquelchBlock::<C32, _>::new(&cfg, r, &mut VecStorage).is_err());
        assert!(SquelchBlock::<CQ15, VecStorage>::storage_bytes(&cfg, r).is_err());
        assert!(SquelchBlock::<CQ15, VecStorage>::rate_out(&cfg, r).is_err());
    }
    let cfg = SquelchConfig {
        tau_s: 100.0,
        ..config()
    };
    assert!(cfg.validate(r).is_ok());
}

#[test]
fn rate_storage_and_timing() {
    let cfg = config();
    let r = rate(FS);
    assert_eq!(SquelchBlock::<C32, VecStorage>::rate_out(&cfg, r), Ok(r));
    assert_eq!(
        SquelchBlock::<CQ15, VecStorage>::storage_bytes(&cfg, r),
        Ok(0)
    );
    // An arena of zero bytes is enough.
    let mut arena = ArenaStorage::new(&mut []);
    let block = SquelchBlock::<C32, _>::new(&cfg, r, &mut arena).unwrap();
    let t = block.timing();
    assert_eq!(t.ratio, Ratio::ONE);
    assert_eq!(t.kind, RatioKind::Constant);
    assert_eq!(t.delay, sdrk::Delay::ZERO);
    assert_eq!(block.min_input(), 1);
    assert_eq!(block.description().kind, "squelch");
}

fn invariant_and_reset<T: BlockSample + PartialEq + core::fmt::Debug>()
where
    Sdrk: Squelch<T>,
{
    let cfg = config();
    // Noise, a burst, noise: the gate switches inside the run.
    let mut input: Vec<T> = signal::<T>(1500, 5, 0.03)
        .iter()
        .map(|s| {
            let c = s.to_c64();
            T::from_c64_round(c * 0.01)
        })
        .collect();
    input.extend(signal::<T>(2500, 6, 0.04));
    input.extend(zeros::<T>(2500));
    let whole = run_block(&mut new_block::<T>(&cfg), &input, None);
    let zero = T::from_c64_round(Complex64::new(0.0, 0.0));
    assert!(whole.iter().any(|s| *s == zero) && whole.iter().any(|s| *s != zero));
    for plan in plans(21) {
        let chunked = run_block(&mut new_block::<T>(&cfg), &input, Some(plan));
        assert_eq!(chunked, whole, "{plan:?}");
    }
    // After reset the block starts closed with p = 0, like a new one.
    let mut block = new_block::<T>(&cfg);
    let _ = run_block(&mut block, &input[..4000], None);
    block.reset();
    assert!(!block.is_open());
    assert_eq!(run_block(&mut block, &input, None), whole);
}

#[test]
fn block_size_invariance_and_reset() {
    invariant_and_reset::<C32>();
    invariant_and_reset::<CQ15>();
}

#[test]
fn nan_input_is_passed_while_open_and_keeps_the_state() {
    let cfg = config();
    let bad = [
        C32::new(f32::NAN, 0.0),
        C32::new(f32::INFINITY, 1.0),
        C32::new(0.0, f32::NEG_INFINITY),
    ];
    // Loud, the three bad samples, then silence: the closing sample shows that p was kept.
    let mut input = constant::<C32>(0.1, 1000);
    input.extend(bad);
    input.extend(zeros::<C32>(1500));
    let bits = |v: &[C32]| -> Vec<(u32, u32)> {
        v.iter().map(|s| (s.re.to_bits(), s.im.to_bits())).collect()
    };
    let mut state =
        SquelchState::<C32>::new(cfg.threshold_db, cfg.hysteresis_db, cfg.tau_s, rate(FS)).unwrap();
    let mut want = vec![C32::ZERO; input.len()];
    Sdrk.squelch(&mut state, &input, &mut want).unwrap();
    let got = run_block(&mut new_block::<C32>(&cfg), &input, None);
    assert_eq!(bits(&got), bits(&want));
    assert_eq!(bits(&got[1000..1003]), bits(&bad));
    // The bad samples did not move p: the model without them closes at the same sample.
    let mut clean = constant::<C32>(0.1, 1000);
    clean.extend(zeros::<C32>(1500));
    let mut state =
        SquelchState::<C32>::new(cfg.threshold_db, cfg.hysteresis_db, cfg.tau_s, rate(FS)).unwrap();
    let mut want_clean = vec![C32::ZERO; clean.len()];
    Sdrk.squelch(&mut state, &clean, &mut want_clean).unwrap();
    let (a, b) = (&got[1003..], &want_clean[1000..]);
    assert_eq!(bits(a), bits(b));
    assert!(got.last().unwrap().re == 0.0);
}
