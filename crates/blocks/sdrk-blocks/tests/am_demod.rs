//! Block `am_demod`: the chain of kernels, the envelope level, validation, invariance, reset
//! and timing.
#![allow(
    clippy::unwrap_used,
    clippy::manual_assert_eq,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::panic,
    clippy::cast_lossless,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::float_cmp,
    clippy::type_complexity,
    clippy::cast_possible_truncation
)]

mod common;

use common::{check_exact_storage, rate, run_block, signal};
use num_complex::Complex64;
use sdrk::{C32, CQ15, ComplexSample, Ratio, RatioKind, RealSample, Sample, Sdrk};
use sdrk_blocks::{
    AmDemodBlock, AmDemodConfig, ArenaStorage, Block, BlockSample, DesignSpec, Error, ParamValue,
    TunedElem,
};
use sdrk_core::kernels::decim::{ResampleReal, ResampleRealState};
use sdrk_core::kernels::demod::{DcBlockReal, DcBlockRealState};
use sdrk_core::kernels::spectrum::Mag;
use sdrk_testkit::{ChunkPlan, VecStorage};

const FS: u64 = 48_000;

fn config(dc_cutoff_hz: f64, audio_decimation: usize) -> AmDemodConfig {
    AmDemodConfig {
        dc_cutoff_hz,
        audio_decimation,
        ..AmDemodConfig::default()
    }
}

fn new_block<T: BlockSample>(cfg: &AmDemodConfig) -> AmDemodBlock<T, VecStorage> {
    AmDemodBlock::<T, _>::new(cfg, rate(FS), &mut VecStorage).unwrap()
}

/// An AM signal: carrier 0.4, 1 kHz, modulation index 0.5, plus a little noise.
fn am_signal<T: ComplexSample>(n: usize) -> Vec<T> {
    let noise = signal::<C32>(n, 11, 0.0);
    (0..n)
        .map(|i| {
            let p = std::f64::consts::TAU * 1000.0 * i as f64 / FS as f64;
            let env = 0.4 * (1.0 + 0.5 * p.cos());
            let nz = noise[i].to_c64() * 0.01;
            T::from_c64_round(Complex64::new(env, 0.0) + nz)
        })
        .collect()
}

/// The chain written out with the kernels: `mag`, `dc_block_real`, `resample_real`.
fn chain<T: BlockSample>(cfg: &AmDemodConfig, input: &[T]) -> Vec<T::Real>
where
    Sdrk: Mag<T> + DcBlockReal<T::Real> + ResampleReal<T::Real>,
{
    let fs = FS as f64;
    let mut env = vec![T::Real::ZERO; input.len()];
    <Sdrk as Mag<T>>::mag(&Sdrk, input, &mut env).unwrap();
    if cfg.dc_cutoff_hz > 0.0 {
        let r = (-std::f64::consts::TAU * cfg.dc_cutoff_hz / fs).exp();
        let mut state = DcBlockRealState::new(T::Real::from_f64_round(r)).unwrap();
        let mut out = vec![T::Real::ZERO; env.len()];
        <Sdrk as DcBlockReal<T::Real>>::dc_block_real(&Sdrk, &mut state, &env, &mut out).unwrap();
        env = out;
    }
    if cfg.audio_decimation == 1 {
        return env;
    }
    let spec = DesignSpec {
        cutoff_hz: cfg.audio_cutoff_hz,
        transition_hz: cfg.audio_cutoff_hz / 4.0,
        attenuation_db: 60.0,
    };
    let mut design = vec![0.0f64; spec.num_taps(fs, 1023).unwrap()];
    spec.design(fs, 1023, &mut design).unwrap();
    let taps: Vec<T::Real> = design
        .iter()
        .map(|v| <T as TunedElem>::tap(*v).unwrap())
        .collect();
    let hist = vec![T::Real::ZERO; taps.len()];
    let mut state = ResampleRealState::new(taps, hist, 1, cfg.audio_decimation).unwrap();
    let mut out = vec![T::Real::ZERO; env.len() / cfg.audio_decimation + 2];
    let p =
        <Sdrk as ResampleReal<T::Real>>::resample_real(&Sdrk, &mut state, &env, &mut out).unwrap();
    assert_eq!(p.consumed, env.len());
    out.truncate(p.produced);
    out
}

fn matches_the_chain<T: BlockSample>()
where
    Sdrk: Mag<T> + DcBlockReal<T::Real> + ResampleReal<T::Real> + 'static,
    T::Real: PartialEq + core::fmt::Debug + Default,
{
    for (dc, dec) in [(30.0, 1), (0.0, 1), (30.0, 4), (0.0, 6), (200.0, 2)] {
        let cfg = AmDemodConfig {
            audio_cutoff_hz: 2_000.0,
            ..config(dc, dec)
        };
        let input = am_signal::<T>(5000);
        let want = chain::<T>(&cfg, &input);
        let got = run_block(&mut new_block::<T>(&cfg), &input, None);
        assert_eq!(got.len(), want.len(), "dc {dc}, decimation {dec}");
        assert_eq!(got, want, "dc {dc}, decimation {dec}");
    }
}

#[test]
fn the_block_equals_the_kernel_chain() {
    matches_the_chain::<C32>();
    matches_the_chain::<CQ15>();
}

#[test]
fn the_audio_level_is_the_modulation_depth_times_the_carrier() {
    // Carrier 0.4, index 0.5: after the DC blocker the 1 kHz tone has amplitude 0.2.
    for cfg in [config(30.0, 1), config(30.0, 4)] {
        let input = am_signal::<C32>(24_000);
        let out = run_block(&mut new_block::<C32>(&cfg), &input, None);
        let tail = &out[out.len() * 3 / 4..];
        let peak = tail.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!((peak - 0.2).abs() < 0.02, "{cfg:?}: {peak}");
        let mean = tail.iter().sum::<f32>() / tail.len() as f32;
        assert!(mean.abs() < 0.02, "{cfg:?}: DC {mean}");
    }
    // Q15 gives the same level.
    let input = am_signal::<CQ15>(24_000);
    let out = run_block(&mut new_block::<CQ15>(&config(30.0, 1)), &input, None);
    let tail = &out[out.len() * 3 / 4..];
    let peak = tail.iter().map(|v| v.0.unsigned_abs()).max().unwrap();
    assert!((f64::from(peak) / 32768.0 - 0.2).abs() < 0.02, "{peak}");
}

#[test]
fn without_the_dc_blocker_the_audio_is_the_magnitude() {
    let input = vec![C32::new(3.0, 4.0), C32::new(-0.5, 0.0), C32::new(0.0, 0.25)];
    let out = run_block(&mut new_block::<C32>(&config(0.0, 1)), &input, None);
    assert_eq!(out, vec![5.0, 0.5, 0.25]);
}

fn twenty_plans() -> Vec<ChunkPlan> {
    (0..20u64)
        .map(|k| ChunkPlan::new(700 + k, 1 + ((k * 53) % 600) as usize))
        .collect()
}

fn invariant_and_reset<T: BlockSample>()
where
    Sdrk: Mag<T> + DcBlockReal<T::Real> + ResampleReal<T::Real>,
    T::Real: PartialEq + core::fmt::Debug + Default,
{
    for (dc, dec) in [(30.0, 1), (30.0, 6), (0.0, 3)] {
        let cfg = AmDemodConfig {
            audio_cutoff_hz: 2_000.0,
            ..config(dc, dec)
        };
        let input = am_signal::<T>(6000);
        let whole = run_block(&mut new_block::<T>(&cfg), &input, None);
        for plan in twenty_plans() {
            let chunked = run_block(&mut new_block::<T>(&cfg), &input, Some(plan));
            assert_eq!(chunked, whole, "dc {dc}, decimation {dec}, {plan:?}");
        }
        // After reset the block behaves like a new one.
        let mut block = new_block::<T>(&cfg);
        let _ = run_block(&mut block, &input[..2501], None);
        block.reset();
        assert_eq!(run_block(&mut block, &input, None), whole);
    }
}

#[test]
fn block_size_invariance_and_reset() {
    invariant_and_reset::<C32>();
    invariant_and_reset::<CQ15>();
}

#[test]
fn a_small_output_buffer_never_overruns() {
    // The block takes only as much input as the output can hold.
    let cfg = config(30.0, 4);
    let mut block = new_block::<C32>(&cfg);
    let input = am_signal::<C32>(1000);
    let mut out = [0.0f32; 10];
    let p = block.process(&input, &mut out).unwrap();
    assert!(p.produced <= 10);
    assert!(p.consumed <= 40 + 4);
    assert!(p.consumed > 0);
}

#[test]
fn parameter_errors() {
    let mut block = new_block::<C32>(&config(30.0, 1));
    for name in ["dc_cutoff_hz", "audio_cutoff_hz"] {
        assert_eq!(
            block.set_param(name, ParamValue::F64(1.0)),
            Err(Error::ReadOnlyParameter)
        );
    }
    assert_eq!(
        block.set_param("audio_decimation", ParamValue::I64(2)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(
        block.set_param("nope", ParamValue::F64(1.0)),
        Err(Error::UnknownParameter)
    );
    assert_eq!(block.param("nope"), Err(Error::UnknownParameter));
    assert_eq!(block.param("dc_cutoff_hz"), Ok(ParamValue::F64(30.0)));
    assert_eq!(block.param("audio_cutoff_hz"), Ok(ParamValue::F64(5000.0)));
    assert_eq!(block.param("audio_decimation"), Ok(ParamValue::I64(1)));
}

#[test]
fn construction_is_validated() {
    let r = rate(FS);
    let bad = [
        config(-1.0, 1),
        config(f64::NAN, 1),
        // At or above half the sample rate.
        config(24_000.0, 1),
        config(30.0, 0),
        config(30.0, 65),
        AmDemodConfig {
            audio_cutoff_hz: 0.0,
            ..config(30.0, 1)
        },
        // The output Nyquist frequency (3 kHz) is below the cutoff plus the transition.
        AmDemodConfig {
            audio_cutoff_hz: 5_000.0,
            ..config(30.0, 8)
        },
        // The pole rounds to 1.0 in f32.
        config(1e-6, 1),
    ];
    for cfg in bad {
        assert!(cfg.validate(r).is_err(), "{cfg:?}");
        assert!(AmDemodBlock::<C32, _>::new(&cfg, r, &mut VecStorage).is_err());
        assert!(AmDemodBlock::<CQ15, VecStorage>::storage_bytes(&cfg, r).is_err());
        assert!(AmDemodBlock::<CQ15, VecStorage>::rate_out(&cfg, r).is_err());
    }
    assert!(config(0.0, 1).validate(r).is_ok());
    assert!(AmDemodConfig::default().validate(r).is_ok());
}

#[test]
fn rate_storage_and_timing() {
    let r = rate(FS);
    let cfg = AmDemodConfig {
        audio_cutoff_hz: 2_000.0,
        ..config(30.0, 4)
    };
    assert_eq!(
        AmDemodBlock::<C32, VecStorage>::rate_out(&cfg, r),
        Ok(rate(12_000))
    );
    assert_eq!(
        AmDemodBlock::<C32, VecStorage>::rate_out(&config(30.0, 1), r),
        Ok(r)
    );
    let taps = cfg.audio_taps(r).unwrap();
    assert!(taps > 3);
    assert_eq!(config(30.0, 1).audio_taps(r), Ok(0));

    let bytes = AmDemodBlock::<C32, VecStorage>::storage_bytes(&cfg, r).unwrap();
    check_exact_storage(bytes, |a| {
        AmDemodBlock::<C32, _>::new(&cfg, r, a).map(|_| ())
    });
    // Without the audio filter only the two work buffers remain.
    let plain = AmDemodBlock::<C32, VecStorage>::storage_bytes(&config(30.0, 1), r).unwrap();
    assert_eq!(plain, 2 * 256 * 4);
    let mut mem = common::Aligned::new(plain + 32);
    let mut arena = ArenaStorage::new(mem.bytes(plain));
    let block = AmDemodBlock::<C32, _>::new(&config(30.0, 1), r, &mut arena).unwrap();
    let t = block.timing();
    assert_eq!(t.ratio, Ratio::ONE);
    assert_eq!(t.kind, RatioKind::Constant);
    assert_eq!(t.delay, sdrk::Delay::ZERO);
    assert_eq!(block.min_input(), 1);
    assert_eq!(block.description().kind, "am_demod");

    let block = new_block::<CQ15>(&cfg);
    let t = block.timing();
    assert_eq!(t.ratio, Ratio::new(1, 4).unwrap());
    assert_eq!(
        t.delay,
        sdrk::Delay::new(taps as u64 - 1, 2).unwrap(),
        "(N - 1) / 2 input samples"
    );
    assert_eq!(block.min_input(), 4);
}

#[test]
fn the_audio_decays_after_the_input_stops() {
    // The carrier is gone (a closed squelch): the DC blocker lets the audio decay, then f32
    // is exactly zero again and Q15 stays at a constant offset of at most 127 LSB (the dead
    // band of the rounded pole).
    let mut input = am_signal::<C32>(4000);
    input.extend(vec![C32::new(0.0, 0.0); 40_000]);
    let out = run_block(&mut new_block::<C32>(&config(30.0, 1)), &input, None);
    assert!(
        out[4000..4100].iter().any(|v| v.abs() > 0.05),
        "starts large"
    );
    assert!(out[4000 + 5000..4000 + 5100].iter().all(|v| v.abs() < 1e-3));
    assert!(
        out[input.len() - 100..].iter().all(|v| *v == 0.0),
        "flushed"
    );

    let mut input = am_signal::<CQ15>(4000);
    input.extend(vec![CQ15::default(); 60_000]);
    let out = run_block(&mut new_block::<CQ15>(&config(30.0, 1)), &input, None);
    let tail = &out[input.len() - 100..];
    assert!(tail.iter().all(|v| *v == tail[0]), "constant");
    assert!(tail[0].0 != 0 && tail[0].0.abs() <= 127, "{:?}", tail[0]);
}
