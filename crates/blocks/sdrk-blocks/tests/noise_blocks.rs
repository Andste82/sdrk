//! The blocks of M48: `NoiseBlock` (`C32`, `CQ15`) and `NoiseRealBlock` (`f32`, `Q15`) of the
//! kind `noise`: equality with the kernel `gauss_noise` on whole buffers and in 20 chunk plans
//! (seed 688), a runtime change of `sigma`, `reset`, ignored input values, the description,
//! validation at the parameter, exact storage (none), timing, size, `Send` and `Sync`.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

mod common;

use common::{check_block, check_exact_storage, rate, same_bits};
use sdrk::{C32, CQ15, Delay, Q15, Ratio, RatioKind, Sdrk};
use sdrk_blocks::{
    ArenaStorage, Block, Error, Mutability, NoiseBlock, NoiseConfig, NoiseRealBlock, ParamKind,
    ParamValue, PortType, Unit, all_descriptions,
};
use sdrk_core::kernels::noise::{GaussNoise, GaussNoiseState};
use sdrk_testkit::VecStorage;

/// Seed of the chunk plans and of the generator of these tests (M48 range 680 to 689).
const SEED: u64 = 688;
/// Samples per test.
const N: usize = 3000;

fn cfg(sigma: f64) -> NoiseConfig {
    NoiseConfig {
        sigma,
        seed: SEED,
        stream: 5,
    }
}

/// The kernel output of `n` samples for the configuration, through the facade.
fn kernel<T>(c: &NoiseConfig, n: usize) -> Vec<T>
where
    Sdrk: GaussNoise<T>,
    T: sdrk_core::Sample + Copy + Default,
{
    let mut st = GaussNoiseState::new(c.seed, c.stream, c.sigma as f32).unwrap();
    let mut out = vec![T::default(); n];
    Sdrk.gauss_noise(&mut st, &mut out).unwrap();
    assert_eq!(st.index(), n as u64);
    out
}

/// The raw bytes of a sample buffer (bit comparison, so `-0.0` and `+0.0` differ).
fn bytes<T: bytemuck::Pod>(v: &[T]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

/// A clock whose values must not matter: NaN, infinities and large values for `f32`.
fn f32_clock(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| match i % 4 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => -1e30,
            _ => i as f32,
        })
        .collect()
}

#[test]
fn the_complex_blocks_equal_the_kernel_whole_and_in_twenty_plans() {
    let c = cfg(0.7);
    let r = rate(48_000);
    let clock: Vec<C32> = f32_clock(N)
        .chunks(2)
        .map(|p| C32::new(p[0], p[1]))
        .cycle()
        .take(N)
        .collect();
    let want = kernel::<C32>(&c, N);
    check_block(
        &want,
        &clock,
        SEED,
        || NoiseBlock::<C32, VecStorage>::new(&c, r, &mut VecStorage).unwrap(),
        "noise C32",
    );
    let got = common::run_block(
        &mut NoiseBlock::<C32, VecStorage>::new(&c, r, &mut VecStorage).unwrap(),
        &clock,
        None,
    );
    assert_eq!(bytes(&want), bytes(&got));

    let want = kernel::<CQ15>(&c, N);
    let clock = vec![
        CQ15 {
            re: Q15::MAX,
            im: Q15::MIN
        };
        N
    ];
    check_block(
        &want,
        &clock,
        SEED,
        || NoiseBlock::<CQ15, VecStorage>::new(&c, r, &mut VecStorage).unwrap(),
        "noise CQ15",
    );
}

#[test]
fn the_real_blocks_equal_the_kernel_whole_and_in_twenty_plans() {
    let c = cfg(1.3);
    let r = rate(8_000);
    let want = kernel::<f32>(&c, N);
    let clock = f32_clock(N);
    check_block(
        &want,
        &clock,
        SEED,
        || NoiseRealBlock::<f32, VecStorage>::new(&c, r, &mut VecStorage).unwrap(),
        "noise f32",
    );
    let got = common::run_block(
        &mut NoiseRealBlock::<f32, VecStorage>::new(&c, r, &mut VecStorage).unwrap(),
        &clock,
        None,
    );
    assert_eq!(bytes(&want), bytes(&got));

    // sigma 1.3 clips a large share of the Q15 values.
    let want = kernel::<Q15>(&c, N);
    assert!(want.contains(&Q15::MAX) && want.contains(&Q15::MIN));
    let clock = vec![Q15(12345); N];
    check_block(
        &want,
        &clock,
        SEED,
        || NoiseRealBlock::<Q15, VecStorage>::new(&c, r, &mut VecStorage).unwrap(),
        "noise Q15",
    );
}

#[test]
fn the_input_values_do_not_matter() {
    let c = cfg(0.25);
    let r = rate(48_000);
    let mut a = NoiseRealBlock::<f32, VecStorage>::new(&c, r, &mut VecStorage).unwrap();
    let mut b = NoiseRealBlock::<f32, VecStorage>::new(&c, r, &mut VecStorage).unwrap();
    let mut out_a = vec![0.0f32; 256];
    let mut out_b = vec![0.0f32; 256];
    a.process(&f32_clock(256), &mut out_a).unwrap();
    b.process(&[0.0f32; 256], &mut out_b).unwrap();
    assert_eq!(bytes(&out_a), bytes(&out_b));
    assert!(out_a.iter().all(|x| x.is_finite()));
}

/// Runs `first` samples, sets `sigma` at run time, runs `second` more; the same as the kernel
/// with `set_sigma` at the same sample index.
fn runtime_sigma<B, T>(mut block: B, first: usize, second: usize)
where
    B: Block<In = T, Out = T>,
    T: sdrk_core::Sample + bytemuck::Pod + Default + sdrk_blocks::StreamElem,
    Sdrk: GaussNoise<T>,
{
    let c = cfg(0.2);
    let mut st = GaussNoiseState::new(c.seed, c.stream, 0.2).unwrap();
    let mut want = vec![T::default(); first + second];
    Sdrk.gauss_noise(&mut st, &mut want[..first]).unwrap();
    st.set_sigma(0.9).unwrap();
    Sdrk.gauss_noise(&mut st, &mut want[first..]).unwrap();

    let clock = vec![T::default(); first + second];
    let mut got = vec![T::default(); first + second];
    let p = block.process(&clock[..first], &mut got[..first]).unwrap();
    assert_eq!((p.consumed, p.produced), (first, first));
    block.set_param("sigma", ParamValue::F64(0.9)).unwrap();
    assert_eq!(block.param("sigma"), Ok(ParamValue::F64(0.9)));
    let p = block.process(&clock[first..], &mut got[first..]).unwrap();
    assert_eq!(p.produced, second);
    assert_eq!(bytes(&want), bytes(&got));
}

#[test]
fn a_runtime_sigma_change_equals_set_sigma_at_the_same_index() {
    let c = cfg(0.2);
    let r = rate(48_000);
    let s = &mut VecStorage;
    runtime_sigma(NoiseBlock::<C32, _>::new(&c, r, s).unwrap(), 777, 500);
    runtime_sigma(NoiseBlock::<CQ15, _>::new(&c, r, s).unwrap(), 1, 999);
    runtime_sigma(NoiseRealBlock::<f32, _>::new(&c, r, s).unwrap(), 1024, 1024);
    runtime_sigma(NoiseRealBlock::<Q15, _>::new(&c, r, s).unwrap(), 333, 4);
}

#[test]
fn sigma_zero_gives_positive_zeros_and_keeps_the_position() {
    let r = rate(48_000);
    let mut block = NoiseBlock::<C32, VecStorage>::new(&cfg(0.0), r, &mut VecStorage).unwrap();
    let mut out = vec![C32::new(1.0, 1.0); 100];
    block.process(&[C32::default(); 100], &mut out).unwrap();
    assert!(
        out.iter()
            .all(|x| x.re.to_bits() == 0 && x.im.to_bits() == 0)
    );
    block.set_param("sigma", ParamValue::F64(1.0)).unwrap();
    block.process(&[C32::default(); 100], &mut out).unwrap();
    // Samples 100 to 199 of the sequence with sigma 1.
    let want = kernel::<C32>(&cfg(1.0), 200);
    assert_eq!(bytes(&out), bytes(&want[100..]));
}

#[test]
fn reset_starts_the_sequence_again() {
    let c = cfg(0.5);
    let r = rate(48_000);
    let mut block = NoiseRealBlock::<Q15, VecStorage>::new(&c, r, &mut VecStorage).unwrap();
    let mut first = vec![Q15(0); 500];
    let mut again = vec![Q15(0); 500];
    block.process(&[Q15(0); 500], &mut first).unwrap();
    block.process(&[Q15(0); 77], &mut again).unwrap();
    block.reset();
    block.process(&[Q15(0); 500], &mut again).unwrap();
    assert_eq!(first, again);
    // reset keeps a runtime sigma (the kernel's reset keeps it as well).
    let mut block = NoiseBlock::<C32, VecStorage>::new(&c, r, &mut VecStorage).unwrap();
    block.set_param("sigma", ParamValue::F64(0.125)).unwrap();
    block.reset();
    let mut out = vec![C32::default(); 64];
    block.process(&[C32::default(); 64], &mut out).unwrap();
    assert_eq!(bytes(&out), bytes(&kernel::<C32>(&cfg(0.125), 64)));
}

#[test]
fn the_output_room_limits_a_call() {
    let c = cfg(0.5);
    let mut block =
        NoiseRealBlock::<f32, VecStorage>::new(&c, rate(48_000), &mut VecStorage).unwrap();
    let mut out = vec![0.0f32; 10];
    let p = block.process(&[0.0; 25], &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (10, 10));
    let p = block.process(&[], &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
    let p = block.process(&[0.0; 25], &mut out[..0]).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
    let mut rest = vec![0.0f32; 10];
    block.process(&[0.0; 10], &mut rest).unwrap();
    out.extend_from_slice(&rest);
    assert_eq!(bytes(&out), bytes(&kernel::<f32>(&c, 20)));
}

#[test]
fn the_kernel_receives_sigma_as_f32() {
    // 0.3 is not an f32; the kernel takes 0.3 as f32, the block reports the f64 it was given.
    let c = cfg(0.3);
    let mut block = NoiseBlock::<C32, VecStorage>::new(&c, rate(1_000), &mut VecStorage).unwrap();
    assert_eq!(block.param("sigma"), Ok(ParamValue::F64(0.3)));
    let mut out = vec![C32::default(); 50];
    block.process(&[C32::default(); 50], &mut out).unwrap();
    let mut st = GaussNoiseState::new(SEED, 5, 0.3f32).unwrap();
    let mut want = vec![C32::default(); 50];
    Sdrk.gauss_noise(&mut st, &mut want).unwrap();
    assert_eq!(bytes(&out), bytes(&want));
}

fn out_of_range(e: Error, name: &str) {
    match e {
        Error::OutOfRange { name: n, .. } => assert_eq!(n, name),
        other => panic!("expected OutOfRange of {name}, got {other:?}"),
    }
}

#[test]
fn invalid_parameters_are_errors_at_their_parameter() {
    let r = rate(48_000);
    let s = &mut VecStorage;
    for sigma in [
        -0.1,
        -1e-300,
        16.000_001,
        16.5,
        f64::NAN,
        f64::INFINITY,
        -f64::INFINITY,
    ] {
        let c = cfg(sigma);
        out_of_range(c.validate(r).unwrap_err(), "sigma");
        out_of_range(NoiseBlock::<C32, _>::new(&c, r, s).unwrap_err(), "sigma");
        out_of_range(
            NoiseRealBlock::<Q15, _>::new(&c, r, s).unwrap_err(),
            "sigma",
        );
        out_of_range(
            NoiseBlock::<CQ15, VecStorage>::storage_bytes(&c, r).unwrap_err(),
            "sigma",
        );
        out_of_range(
            NoiseRealBlock::<f32, VecStorage>::rate_out(&c, r).unwrap_err(),
            "sigma",
        );
        // At run time: rejected, and nothing changes.
        let mut block = NoiseBlock::<C32, _>::new(&cfg(0.5), r, s).unwrap();
        out_of_range(
            block
                .set_param("sigma", ParamValue::F64(sigma))
                .unwrap_err(),
            "sigma",
        );
        assert_eq!(block.param("sigma"), Ok(ParamValue::F64(0.5)));
        let mut c = NoiseConfig::default();
        out_of_range(c.set("sigma", ParamValue::F64(sigma)).unwrap_err(), "sigma");
        assert_eq!(c, NoiseConfig::default());
    }
    for bad in [1u64 << 53, u64::MAX, i64::MAX as u64 + 1] {
        let c = NoiseConfig {
            seed: bad,
            ..NoiseConfig::default()
        };
        out_of_range(NoiseBlock::<C32, _>::new(&c, r, s).unwrap_err(), "seed");
        let c = NoiseConfig {
            stream: bad,
            ..NoiseConfig::default()
        };
        out_of_range(
            NoiseRealBlock::<f32, _>::new(&c, r, s).unwrap_err(),
            "stream",
        );
    }
    for name in ["seed", "stream"] {
        let mut c = NoiseConfig::default();
        out_of_range(c.set(name, ParamValue::I64(-1)).unwrap_err(), name);
        out_of_range(c.set(name, ParamValue::I64(1 << 53)).unwrap_err(), name);
        assert_eq!(
            c.set(name, ParamValue::F64(1.0)),
            Err(Error::WrongParamType)
        );
        c.set(name, ParamValue::I64((1 << 53) - 1)).unwrap();
        assert_eq!(c.get(name), Ok(ParamValue::I64((1 << 53) - 1)));
        // seed and stream are build parameters.
        let mut block = NoiseBlock::<C32, _>::new(&NoiseConfig::default(), r, s).unwrap();
        assert_eq!(
            block.set_param(name, ParamValue::I64(3)),
            Err(Error::ReadOnlyParameter)
        );
    }
    // The limits themselves are valid.
    for sigma in [0.0, -0.0, 16.0, 1e-30] {
        cfg(sigma).validate(r).unwrap();
        NoiseRealBlock::<Q15, _>::new(&cfg(sigma), r, s).unwrap();
    }
    let edge = NoiseConfig {
        sigma: 16.0,
        seed: (1 << 53) - 1,
        stream: (1 << 53) - 1,
    };
    let mut block = NoiseRealBlock::<Q15, _>::new(&edge, r, s).unwrap();
    let mut out = vec![Q15(0); 1000];
    block.process(&[Q15(0); 1000], &mut out).unwrap();
    // sigma 16 saturates nearly every value.
    let clipped = out
        .iter()
        .filter(|q| **q == Q15::MAX || **q == Q15::MIN)
        .count();
    assert!(clipped > 900, "{clipped}");
}

#[test]
fn the_description_follows_the_rules() {
    let d = all_descriptions()
        .iter()
        .find(|d| d.kind == "noise")
        .unwrap();
    assert!(d.summary.chars().count() <= 120 && d.summary.ends_with('.'));
    assert!(d.details.len() > d.summary.len() && !d.details.contains('\n'));
    for text in [
        "N0 = 2 sigma^2",
        "A / sqrt(2)",
        "nstd / sqrt(2)",
        "clock",
        "add",
    ] {
        assert!(d.details.contains(text), "details lack {text}");
    }
    let all = [PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15];
    assert_eq!(d.inputs.len(), 1);
    assert_eq!(d.inputs[0].name, "clock");
    assert_eq!(d.inputs[0].types, all);
    assert_eq!(d.outputs.len(), 1);
    assert_eq!(d.outputs[0].types, all);
    assert_eq!(d.required.len(), 0);
    let names: Vec<_> = d.params.iter().map(|p| p.name).collect();
    assert_eq!(names, ["sigma", "seed", "stream"]);
    let sigma = d.param("sigma").unwrap();
    assert_eq!(sigma.unit, Unit::None);
    assert_eq!(sigma.mutability, Mutability::Runtime);
    assert_eq!(
        sigma.kind,
        ParamKind::F64 {
            min: 0.0,
            max: 16.0,
            default: Some(0.1)
        }
    );
    for (name, default) in [("seed", 0), ("stream", 1)] {
        let p = d.param(name).unwrap();
        assert_eq!(p.unit, Unit::None);
        assert_eq!(p.mutability, Mutability::Build);
        assert_eq!(
            p.kind,
            ParamKind::I64 {
                min: 0,
                max: (1 << 53) - 1,
                default: Some(default)
            }
        );
    }
    let c = NoiseConfig::default();
    assert_eq!((c.sigma, c.seed, c.stream), (0.1, 0, 1));
}

#[test]
fn the_blocks_need_no_storage_and_keep_the_rate() {
    let c = cfg(0.5);
    let r = rate(250_000);
    assert_eq!(NoiseBlock::<C32, VecStorage>::storage_bytes(&c, r), Ok(0));
    assert_eq!(NoiseBlock::<CQ15, VecStorage>::storage_bytes(&c, r), Ok(0));
    assert_eq!(
        NoiseRealBlock::<f32, VecStorage>::storage_bytes(&c, r),
        Ok(0)
    );
    assert_eq!(
        NoiseRealBlock::<Q15, VecStorage>::storage_bytes(&c, r),
        Ok(0)
    );
    assert_eq!(NoiseBlock::<C32, VecStorage>::rate_out(&c, r), Ok(r));
    assert_eq!(NoiseRealBlock::<Q15, VecStorage>::rate_out(&c, r), Ok(r));
    check_exact_storage(0, |a| NoiseBlock::<C32, _>::new(&c, r, a).map(|_| ()));
    check_exact_storage(0, |a| NoiseBlock::<CQ15, _>::new(&c, r, a).map(|_| ()));
    check_exact_storage(0, |a| NoiseRealBlock::<f32, _>::new(&c, r, a).map(|_| ()));
    check_exact_storage(0, |a| NoiseRealBlock::<Q15, _>::new(&c, r, a).map(|_| ()));
}

fn check_timing<B: Block>(b: &B) {
    let t = b.timing();
    assert_eq!(t.ratio, Ratio::ONE);
    assert_eq!(t.kind, RatioKind::Constant);
    assert_eq!(t.delay, Delay::ZERO);
    assert_eq!(b.min_input(), 1);
    assert_eq!(b.description().kind, "noise");
}

#[test]
fn timing_is_ratio_one_and_delay_zero() {
    let c = cfg(0.5);
    let r = rate(48_000);
    let mut arena = ArenaStorage::new(&mut []);
    check_timing(&NoiseBlock::<C32, _>::new(&c, r, &mut arena).unwrap());
    check_timing(&NoiseBlock::<CQ15, _>::new(&c, r, &mut arena).unwrap());
    check_timing(&NoiseRealBlock::<f32, _>::new(&c, r, &mut arena).unwrap());
    let mut b = NoiseRealBlock::<Q15, _>::new(&c, r, &mut arena).unwrap();
    b.process(&[Q15(0); 9], &mut [Q15(0); 9]).unwrap();
    check_timing(&b);
}

#[test]
fn the_blocks_are_small_send_and_sync() {
    fn check<T: Send + Sync>(name: &str) {
        let size = std::mem::size_of::<T>();
        assert!(size <= 512, "{name} is {size} bytes");
    }
    check::<NoiseBlock<C32, ArenaStorage<'static>>>("NoiseBlock<C32>");
    check::<NoiseBlock<CQ15, VecStorage>>("NoiseBlock<CQ15, Vec>");
    check::<NoiseRealBlock<f32, ArenaStorage<'static>>>("NoiseRealBlock<f32>");
    check::<NoiseRealBlock<Q15, VecStorage>>("NoiseRealBlock<Q15, Vec>");
    check::<NoiseConfig>("NoiseConfig");
}

#[test]
fn two_streams_of_one_seed_differ() {
    let fs = rate(48_000);
    let one = NoiseConfig {
        sigma: 0.5,
        seed: SEED,
        stream: 1,
    };
    let two = NoiseConfig { stream: 2, ..one };
    let run = |c: &NoiseConfig| {
        let mut block = NoiseRealBlock::<f32, VecStorage>::new(c, fs, &mut VecStorage).unwrap();
        let mut out = vec![0.0f32; 64];
        block.process(&[0.0; 64], &mut out).unwrap();
        out
    };
    let (first, second) = (run(&one), run(&two));
    assert!(first.iter().zip(&second).all(|(p, q)| p != q));
    same_bits(&first, &kernel::<f32>(&one, 64), "stream 1");
}
