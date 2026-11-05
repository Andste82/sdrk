//! Blocks `am_mod` and `ssb_mod` (M35): exact arena, rate and timing, validation, runtime
//! parameters, the golden cases `blocks/ssb_mod_*` whole and in 20 chunk plans (seed 483) and
//! the phase-continuous sideband switch (seed 484).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::panic,
    clippy::cast_lossless,
    clippy::items_after_statements,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::multiple_bound_locations,
    clippy::large_stack_frames
)]

mod common;

use common::{check_exact_storage, norm, rate, run_block, same_bits, twenty_plans};
use num_complex::Complex64;
use sdrk::{C32, CQ15, Q15, Ratio, RatioKind, RealSample, Sample, Sdrk};
use sdrk_blocks::{
    AmModBlock, AmModConfig, ArenaStorage, Block, BlockSample, DesignSpec, Error, ParamValue,
    SSB_MOD_CHUNK, SsbModBlock, SsbModConfig, SsbSideband, StreamElem, TunedElem,
};
use sdrk_core::kernels::filter::Fir;
use sdrk_core::kernels::math::Interleave;
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_testkit::golden::{self, Array, GoldenCase};
use sdrk_testkit::{Pcg32, VecStorage};

type A<'a> = ArenaStorage<'a>;

const FS: u64 = 48_000;

/// Seed of the chunk plans of the golden tests (seed table of docs/milestones/README.md).
const CHUNK_SEED: u64 = 483;

/// Seed of the sideband-switch test.
const SWITCH_SEED: u64 = 484;

fn ssb(sideband: SsbSideband) -> SsbModConfig {
    SsbModConfig {
        sideband,
        ..SsbModConfig::default()
    }
}

fn new_ssb<T: BlockSample>(cfg: &SsbModConfig) -> SsbModBlock<T, VecStorage> {
    SsbModBlock::<T, _>::new(cfg, rate(FS), &mut VecStorage).unwrap()
}

fn to_c64<T: Sample>(v: &[T]) -> Vec<Complex64> {
    let bytes: &[u8] = bytemuck::cast_slice(v);
    match T::TYPE {
        sdrk::SampleType::C32 => bytemuck::cast_slice::<u8, f32>(bytes)
            .chunks_exact(2)
            .map(|c| Complex64::new(f64::from(c[0]), f64::from(c[1])))
            .collect(),
        sdrk::SampleType::CQ15 => bytemuck::cast_slice::<u8, i16>(bytes)
            .chunks_exact(2)
            .map(|c| Complex64::new(f64::from(c[0]) / 32768.0, f64::from(c[1]) / 32768.0))
            .collect(),
        other => panic!("complex type expected, got {other:?}"),
    }
}

/// Amplitude of the complex tone at `f_hz` (negative for the lower side) in `x`, averaged over
/// the whole slice (choose a whole number of periods).
fn tone_amplitude(x: &[Complex64], start: usize, f_hz: f64) -> f64 {
    let sum: Complex64 = x
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let n = (start + i) as f64;
            {
                let p = -std::f64::consts::TAU * f_hz * n / FS as f64;
                v * Complex64::new(p.cos(), p.sin())
            }
        })
        .sum();
    norm(sum / x.len() as f64)
}

/// A real cosine of `amp` at `f_hz`, plus Gaussian noise of standard deviation `sigma`.
fn audio<R: RealSample>(n: usize, f_hz: f64, amp: f64, sigma: f64, rng: &mut Pcg32) -> Vec<R> {
    (0..n)
        .map(|i| {
            let p = std::f64::consts::TAU * f_hz * i as f64 / FS as f64;
            R::from_f64_round(amp * p.cos() + sigma * rng.normal())
        })
        .collect()
}

// ---- am_mod -----------------------------------------------------------------------------

#[test]
fn am_mod_rate_storage_timing_and_validation() {
    let r = rate(FS);
    let cfg = AmModConfig::default();
    for bytes in [
        AmModBlock::<C32, A<'_>>::storage_bytes(&cfg, r).unwrap(),
        AmModBlock::<CQ15, A<'_>>::storage_bytes(&cfg, r).unwrap(),
    ] {
        assert_eq!(bytes, 0);
    }
    check_exact_storage(0, |a| AmModBlock::<C32, _>::new(&cfg, r, a).map(|_| ()));
    check_exact_storage(0, |a| AmModBlock::<CQ15, _>::new(&cfg, r, a).map(|_| ()));
    assert_eq!(AmModBlock::<C32, A<'_>>::rate_out(&cfg, r), Ok(r));
    let block = AmModBlock::<CQ15, _>::new(&cfg, r, &mut VecStorage).unwrap();
    let t = block.timing();
    assert_eq!((t.ratio, t.kind), (Ratio::ONE, RatioKind::Constant));
    assert_eq!(t.delay, sdrk::Delay::ZERO);
    assert_eq!(block.min_input(), 1);
    assert_eq!(block.description().kind, "am_mod");

    for bad in [
        AmModConfig {
            carrier: -0.1,
            ..cfg
        },
        AmModConfig {
            carrier: 1.5,
            ..cfg
        },
        AmModConfig {
            carrier: f64::NAN,
            ..cfg
        },
        AmModConfig { depth: -0.1, ..cfg },
        AmModConfig { depth: 1.01, ..cfg },
        AmModConfig {
            depth: f64::NAN,
            ..cfg
        },
    ] {
        assert!(
            matches!(bad.validate(), Err(Error::OutOfRange { .. })),
            "{bad:?}"
        );
        assert!(AmModBlock::<C32, _>::new(&bad, r, &mut VecStorage).is_err());
        assert!(AmModBlock::<CQ15, A<'_>>::storage_bytes(&bad, r).is_err());
        assert!(AmModBlock::<C32, A<'_>>::rate_out(&bad, r).is_err());
    }
}

#[test]
fn am_mod_runtime_depth_and_read_only_carrier() {
    let r = rate(FS);
    let mut block = AmModBlock::<C32, _>::new(&AmModConfig::default(), r, &mut VecStorage).unwrap();
    assert_eq!(block.param("carrier"), Ok(ParamValue::F64(0.5)));
    assert_eq!(block.param("depth"), Ok(ParamValue::F64(0.8)));
    assert_eq!(
        block.set_param("carrier", ParamValue::F64(0.4)),
        Err(Error::ReadOnlyParameter)
    );
    assert!(block.set_param("depth", ParamValue::F64(1.5)).is_err());
    assert!(block.set_param("depth", ParamValue::I64(1)).is_err());
    assert_eq!(
        block.set_param("nope", ParamValue::F64(0.0)),
        Err(Error::UnknownParameter)
    );
    // A failed change leaves the depth alone.
    assert_eq!(block.param("depth"), Ok(ParamValue::F64(0.8)));
    let mut out = [C32::default(); 2];
    block.process(&[1.0, -1.0], &mut out).unwrap();
    assert_eq!(
        out,
        [C32::new(0.5 * 1.8, 0.0), C32::new(0.5 * (1.0 - 0.8), 0.0)]
    );
    // Depth 0: the constant carrier; depth 1: the envelope touches 0.
    block.set_param("depth", ParamValue::F64(0.0)).unwrap();
    block.process(&[1.0, -1.0], &mut out).unwrap();
    assert_eq!(out, [C32::new(0.5, 0.0); 2]);
    block.set_param("depth", ParamValue::F64(1.0)).unwrap();
    assert_eq!(block.param("depth"), Ok(ParamValue::F64(1.0)));
    block.process(&[1.0, -1.0], &mut out).unwrap();
    assert_eq!(out, [C32::new(1.0, 0.0), C32::new(0.0, 0.0)]);
    // Reset changes nothing for a stateless block.
    block.reset();
    assert_eq!(block.param("depth"), Ok(ParamValue::F64(1.0)));
    // The output buffer limits the progress.
    let p = block.process(&[0.0; 5], &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (2, 2));
}

// ---- ssb_mod: construction --------------------------------------------------------------

#[test]
fn ssb_mod_rate_storage_and_timing() {
    let r = rate(FS);
    let cfg = SsbModConfig::default();
    let taps = cfg.num_taps(r).unwrap();
    assert_eq!(taps, 873);
    assert_eq!(SsbModBlock::<C32, A<'_>>::rate_out(&cfg, r), Ok(r));
    // The arena holds exactly the taps, the history and one work buffer.
    let bytes = SsbModBlock::<C32, A<'_>>::storage_bytes(&cfg, r).unwrap();
    assert_eq!(bytes, 4 * taps + 8 * (taps - 1) + 8 * SSB_MOD_CHUNK);
    check_exact_storage(bytes, |a| {
        SsbModBlock::<C32, _>::new(&cfg, r, a).map(|_| ())
    });
    let bytes = SsbModBlock::<CQ15, A<'_>>::storage_bytes(&cfg, r).unwrap();
    check_exact_storage(bytes, |a| {
        SsbModBlock::<CQ15, _>::new(&cfg, r, a).map(|_| ())
    });
    for cfg in [
        SsbModConfig {
            low_hz: 0.0,
            high_hz: 10_000.0,
            transition_hz: 1_000.0,
            ..cfg
        },
        SsbModConfig {
            sideband: SsbSideband::Lsb,
            attenuation_db: 80.0,
            max_taps: 2000,
            ..cfg
        },
    ] {
        let bytes = SsbModBlock::<C32, A<'_>>::storage_bytes(&cfg, r).unwrap();
        check_exact_storage(bytes, |a| {
            SsbModBlock::<C32, _>::new(&cfg, r, a).map(|_| ())
        });
        let bytes = SsbModBlock::<CQ15, A<'_>>::storage_bytes(&cfg, r).unwrap();
        check_exact_storage(bytes, |a| {
            SsbModBlock::<CQ15, _>::new(&cfg, r, a).map(|_| ())
        });
    }

    let block = new_ssb::<CQ15>(&cfg);
    let t = block.timing();
    assert_eq!((t.ratio, t.kind), (Ratio::ONE, RatioKind::Constant));
    assert_eq!(
        t.delay,
        sdrk::Delay::new(taps as u64 - 1, 2).unwrap(),
        "(taps - 1) / 2 samples"
    );
    assert_eq!(block.num_taps(), taps);
    assert_eq!(block.min_input(), 1);
    assert_eq!(block.description().kind, "ssb_mod");
}

#[test]
fn ssb_mod_construction_is_validated() {
    let r = rate(FS);
    let d = SsbModConfig::default();
    for bad in [
        SsbModConfig { low_hz: -1.0, ..d },
        SsbModConfig {
            low_hz: 3_000.0,
            ..d
        },
        SsbModConfig {
            low_hz: f64::NAN,
            ..d
        },
        SsbModConfig {
            high_hz: f64::NAN,
            ..d
        },
        // high_hz + transition_hz / 2 must not exceed fs / 2.
        SsbModConfig {
            high_hz: 23_901.0,
            ..d
        },
    ] {
        assert!(
            matches!(bad.validate(r), Err(Error::OutOfRange { .. })),
            "{bad:?}: {:?}",
            bad.validate(r)
        );
        assert!(SsbModBlock::<C32, _>::new(&bad, r, &mut VecStorage).is_err());
        assert!(SsbModBlock::<CQ15, A<'_>>::storage_bytes(&bad, r).is_err());
        assert!(SsbModBlock::<C32, A<'_>>::rate_out(&bad, r).is_err());
    }
    // The edge itself is valid.
    let edge = SsbModConfig {
        high_hz: 23_900.0,
        ..d
    };
    assert!(edge.validate(r).is_ok());
    // A design that needs more taps than max_taps fails.
    let small = SsbModConfig { max_taps: 100, ..d };
    assert!(small.validate(r).is_err());
    assert!(SsbModBlock::<CQ15, _>::new(&small, r, &mut VecStorage).is_err());
}

#[test]
fn ssb_mod_parameters() {
    let mut block = new_ssb::<C32>(&SsbModConfig::default());
    assert_eq!(block.param("sideband"), Ok(ParamValue::Choice(0)));
    for (name, v) in [
        ("low_hz", ParamValue::F64(200.0)),
        ("high_hz", ParamValue::F64(2_000.0)),
        ("transition_hz", ParamValue::F64(100.0)),
        ("attenuation_db", ParamValue::F64(70.0)),
        ("max_taps", ParamValue::I64(2000)),
    ] {
        assert_eq!(
            block.set_param(name, v),
            Err(Error::ReadOnlyParameter),
            "{name}"
        );
    }
    assert!(block.set_param("sideband", ParamValue::Choice(2)).is_err());
    assert!(block.set_param("sideband", ParamValue::I64(1)).is_err());
    assert_eq!(
        block.set_param("nope", ParamValue::Choice(0)),
        Err(Error::UnknownParameter)
    );
    assert_eq!(block.param("sideband"), Ok(ParamValue::Choice(0)));
    block.set_param("sideband", ParamValue::Choice(1)).unwrap();
    assert_eq!(block.param("sideband"), Ok(ParamValue::Choice(1)));
    assert_eq!(block.param("low_hz"), Ok(ParamValue::F64(300.0)));
    assert_eq!(block.param("max_taps"), Ok(ParamValue::I64(1023)));
}

// ---- ssb_mod: the kernel chain ----------------------------------------------------------

/// The chain of section 3 called sample block by sample block through `Sdrk` over the whole
/// signal at once: interleave, rotate by `-s c`, fir with the doubled taps, rotate by `+s c`.
fn chain<T: BlockSample>(cfg: &SsbModConfig, x: &[T::Real]) -> Vec<T>
where
    Sdrk: Interleave<T> + Rotate<T> + Fir<T>,
{
    let r = rate(FS);
    let spec = DesignSpec {
        cutoff_hz: (cfg.high_hz - cfg.low_hz) / 2.0,
        transition_hz: cfg.transition_hz,
        attenuation_db: cfg.attenuation_db,
    };
    let fs = FS as f64;
    let mut design = vec![0.0f64; spec.num_taps(fs, cfg.max_taps).unwrap()];
    spec.design(fs, cfg.max_taps, &mut design).unwrap();
    let taps: Vec<T::Tap> = design
        .iter()
        .map(|v| <T as TunedElem>::tap(2.0 * v).unwrap())
        .collect();
    let n = taps.len();
    let mut fir = sdrk_core::kernels::filter::FirState::new(taps, vec![T::ZERO; n - 1]).unwrap();
    let s = match cfg.sideband {
        SsbSideband::Usb => 1.0,
        _ => -1.0,
    };
    let c = f64::midpoint(cfg.low_hz, cfg.high_hz);
    let mut nco_1 = Nco::new(-s * c, r).unwrap();
    let mut nco_2 = Nco::new(s * c, r).unwrap();
    let mut a = vec![T::ZERO; x.len()];
    let mut b = vec![T::ZERO; x.len()];
    <Sdrk as Interleave<T>>::interleave(&Sdrk, x, None, &mut a).unwrap();
    <Sdrk as Rotate<T>>::rotate(&Sdrk, &mut nco_1, &a, &mut b).unwrap();
    <Sdrk as Fir<T>>::fir(&Sdrk, &mut fir, &b, &mut a).unwrap();
    <Sdrk as Rotate<T>>::rotate(&Sdrk, &mut nco_2, &a, &mut b).unwrap();
    b
}

fn equals_the_chain<T: BlockSample>()
where
    Sdrk: Interleave<T> + Rotate<T> + Fir<T>,
    T: PartialEq + core::fmt::Debug + StreamElem,
    T::Real: StreamElem,
{
    let mut rng = Pcg32::new(CHUNK_SEED, 3);
    let x: Vec<T::Real> = audio(3000, 1_234.5, 0.4, 0.1, &mut rng);
    for side in [SsbSideband::Usb, SsbSideband::Lsb] {
        let cfg = ssb(side);
        let want = chain::<T>(&cfg, &x);
        let got = run_block(&mut new_ssb::<T>(&cfg), &x, None);
        same_bits(&want, &got, &format!("{:?} {side:?}", T::TYPE));
    }
}

#[test]
fn ssb_mod_equals_the_kernel_chain() {
    equals_the_chain::<C32>();
    equals_the_chain::<CQ15>();
}

#[test]
fn ssb_mod_puts_a_tone_on_the_chosen_side_with_its_amplitude() {
    for side in [SsbSideband::Usb, SsbSideband::Lsb] {
        let s = if side == SsbSideband::Usb { 1.0 } else { -1.0 };
        let mut rng = Pcg32::new(CHUNK_SEED, 5);
        let x: Vec<f32> = audio(9600, 1000.0, 0.5, 0.0, &mut rng);
        let y = to_c64(&run_block(&mut new_ssb::<C32>(&ssb(side)), &x, None));
        // 4800 samples after the filter has settled: 100 periods of 1 kHz.
        let seg = &y[2400..7200];
        let wanted = tone_amplitude(seg, 2400, s * 1000.0);
        let mirror = tone_amplitude(seg, 2400, -s * 1000.0);
        assert!((wanted - 0.5).abs() < 0.5 * 0.01, "{side:?}: {wanted}");
        assert!(
            mirror < 0.5 * 10f64.powf(-50.0 / 20.0),
            "{side:?}: {mirror}"
        );
        let xq: Vec<Q15> = audio(9600, 1000.0, 0.5, 0.0, &mut rng);
        let yq = to_c64(&run_block(&mut new_ssb::<CQ15>(&ssb(side)), &xq, None));
        let seg = &yq[2400..7200];
        let wanted = tone_amplitude(seg, 2400, s * 1000.0);
        let mirror = tone_amplitude(seg, 2400, -s * 1000.0);
        assert!((wanted - 0.5).abs() < 0.5 * 0.01, "CQ15 {side:?}: {wanted}");
        assert!(
            mirror < 0.5 * 10f64.powf(-50.0 / 20.0),
            "CQ15 {side:?}: {mirror}"
        );
    }
}

#[test]
fn ssb_mod_silence_and_reset() {
    let cfg = SsbModConfig::default();
    let out = run_block(&mut new_ssb::<C32>(&cfg), &[0.0f32; 1000], None);
    assert!(out.iter().all(|s| *s == C32::new(0.0, 0.0)));
    let out = run_block(&mut new_ssb::<CQ15>(&cfg), &[Q15(0); 1000], None);
    assert!(out.iter().all(|s| *s == CQ15::default()));
    // Reset returns to the state after construction.
    let mut rng = Pcg32::new(CHUNK_SEED, 7);
    let x: Vec<Q15> = audio(1500, 700.0, 0.6, 0.05, &mut rng);
    let mut block = new_ssb::<CQ15>(&ssb(SsbSideband::Lsb));
    let first = run_block(&mut block, &x, None);
    block.reset();
    let second = run_block(&mut block, &x, None);
    assert_eq!(first, second);
}

// ---- ssb_mod: golden cases --------------------------------------------------------------

fn golden_config(case: &GoldenCase) -> SsbModConfig {
    let sideband = match case.param_str("sideband").unwrap() {
        "usb" => SsbSideband::Usb,
        "lsb" => SsbSideband::Lsb,
        other => panic!("unknown sideband {other}"),
    };
    let cfg = SsbModConfig {
        sideband,
        low_hz: case.param_f64("low_hz").unwrap(),
        high_hz: case.param_f64("high_hz").unwrap(),
        transition_hz: case.param_f64("transition_hz").unwrap(),
        attenuation_db: case.param_f64("attenuation_db").unwrap(),
        max_taps: usize::try_from(case.param_u64("max_taps").unwrap()).unwrap(),
    };
    assert_eq!(case.param_f64("tap_gain").unwrap(), 2.0);
    assert_eq!(case.param_u64("rate_hz").unwrap(), FS);
    let taps = cfg.num_taps(rate(FS)).unwrap();
    assert_eq!(taps as u64, case.param_u64("taps").unwrap());
    assert_eq!((taps as u64 - 1) / 2, case.param_u64("delay").unwrap());
    let s = if sideband == SsbSideband::Usb {
        1.0
    } else {
        -1.0
    };
    let c = f64::midpoint(cfg.low_hz, cfg.high_hz);
    assert_eq!(
        u64::from(Nco::new(-s * c, rate(FS)).unwrap().increment()),
        case.param_u64("inc_1").unwrap()
    );
    assert_eq!(
        u64::from(Nco::new(s * c, rate(FS)).unwrap().increment()),
        case.param_u64("inc_2").unwrap()
    );
    cfg
}

/// The block over the whole input and over 20 chunk plans of seed 483: every run matches the
/// golden file, and every chunked run equals the whole run bit for bit.
fn check_golden<T: BlockSample>(case: &GoldenCase, input: &[T::Real])
where
    SsbModBlock<T, VecStorage>: Block<In = T::Real, Out = T>,
    T: StreamElem + Default + Sample + PartialEq + core::fmt::Debug,
    T::Real: StreamElem,
{
    let cfg = golden_config(case);
    let expected = case.array("expected").unwrap();
    let whole = run_block(&mut new_ssb::<T>(&cfg), input, None);
    golden::assert_matches(case, expected, &whole).unwrap();
    for plan in twenty_plans(CHUNK_SEED) {
        let chunked = run_block(&mut new_ssb::<T>(&cfg), input, Some(plan));
        same_bits(&whole, &chunked, &format!("{} {plan:?}", case.case));
    }
}

#[test]
fn ssb_mod_golden_usb_c32() {
    let case = golden::load("blocks", "ssb_mod_usb_c32").unwrap();
    let Array::F32(input) = case.array("input").unwrap() else {
        panic!("f32 input expected")
    };
    check_golden::<C32>(&case, input);
}

#[test]
fn ssb_mod_golden_lsb_c32() {
    let case = golden::load("blocks", "ssb_mod_lsb_c32").unwrap();
    let Array::F32(input) = case.array("input").unwrap() else {
        panic!("f32 input expected")
    };
    check_golden::<C32>(&case, input);
}

#[test]
fn ssb_mod_golden_usb_cq15() {
    let case = golden::load("blocks", "ssb_mod_usb_cq15").unwrap();
    let input = case.array("input").unwrap().as_q15().unwrap();
    check_golden::<CQ15>(&case, &input);
}

// ---- ssb_mod: the sideband switch (Test 5) -----------------------------------------------

/// Largest step `|y[n] - y[n - 1]|` in `y[from..to]`.
fn max_step(y: &[Complex64], from: usize, to: usize) -> f64 {
    (from.max(1)..to)
        .map(|n| norm(y[n] - y[n - 1]))
        .fold(0.0, f64::max)
}

/// Switches USB to LSB (and back) at positions drawn with seed 484, between two calls: no step
/// of the output is larger than the signal amplitude, the tone moves from `+f` to `-f` (and
/// back), and switching twice at once leaves the output bit-identical to never switching.
fn sideband_switch<T: BlockSample>()
where
    SsbModBlock<T, VecStorage>: Block<In = T::Real, Out = T>,
    T: StreamElem + Default + Sample + PartialEq + core::fmt::Debug,
    T::Real: StreamElem,
{
    const AMP: f64 = 0.5;
    const F: f64 = 1000.0;
    let n = 24_000;
    let mut rng = Pcg32::new(SWITCH_SEED, 1);
    let x: Vec<T::Real> = audio(n, F, AMP, 0.01, &mut rng);
    let usb = run_block(&mut new_ssb::<T>(&ssb(SsbSideband::Usb)), &x, None);
    let usb_c = to_c64(&usb);
    let normal_step = max_step(&usb_c, 1000, n);
    for _ in 0..8 {
        // A switch to LSB and one back, at random positions of the settled signal.
        let at_1 = 2000 + rng.below(6000) as usize;
        let at_2 = 14_000 + rng.below(4000) as usize;
        let mut block = new_ssb::<T>(&ssb(SsbSideband::Usb));
        let mut y = run_block(&mut block, &x[..at_1], None);
        block.set_param("sideband", ParamValue::Choice(1)).unwrap();
        y.extend(run_block(&mut block, &x[at_1..at_2], None));
        block.set_param("sideband", ParamValue::Choice(0)).unwrap();
        y.extend(run_block(&mut block, &x[at_2..], None));
        let what = format!("{:?} switch at {at_1} and {at_2}", T::TYPE);
        assert_eq!(y[..at_1], usb[..at_1], "{what}: before the switch");
        let yc = to_c64(&y);
        let step = max_step(&yc, 1000, n);
        assert!(step <= AMP, "{what}: step {step} (normal {normal_step})");
        // With continuous phases the largest step is that of the tone whose rotation changes
        // from +c to -c at the switch, about AMP 2 sin(pi (f + c) / fs); a phase jump would give
        // up to 2 AMP.
        let c = 1650.0;
        let bound = 1.2 * AMP * 2.0 * (std::f64::consts::PI * (F + c) / FS as f64).sin();
        assert!(
            step <= bound,
            "{what}: step {step} (bound {bound}, normal {normal_step})"
        );
        // 2400 samples (50 periods) after the filter has settled on either side.
        let lsb = &yc[at_1 + 1500..at_1 + 3900];
        assert!(
            (tone_amplitude(lsb, at_1 + 1500, -F) - AMP).abs() < 0.02,
            "{what}: lsb"
        );
        assert!(
            tone_amplitude(lsb, at_1 + 1500, F) < 0.002,
            "{what}: usb leak"
        );
        let back = &yc[at_2 + 1500..at_2 + 3900];
        assert!(
            (tone_amplitude(back, at_2 + 1500, F) - AMP).abs() < 0.02,
            "{what}: usb"
        );
        assert!(
            tone_amplitude(back, at_2 + 1500, -F) < 0.002,
            "{what}: lsb leak"
        );
    }
    // Switching and switching back at once leaves the output bit-identical.
    let mut block = new_ssb::<T>(&ssb(SsbSideband::Usb));
    let mut y = run_block(&mut block, &x[..5000], None);
    block.set_param("sideband", ParamValue::Choice(1)).unwrap();
    block.set_param("sideband", ParamValue::Choice(0)).unwrap();
    y.extend(run_block(&mut block, &x[5000..], None));
    same_bits(&usb, &y, &format!("{:?} switch and back", T::TYPE));
}

#[test]
fn ssb_mod_sideband_switch_is_phase_continuous() {
    sideband_switch::<C32>();
    sideband_switch::<CQ15>();
}

// ---- Send -------------------------------------------------------------------------------

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn public_types_are_send() {
    assert_send::<AmModBlock<C32, A<'static>>>();
    assert_send::<AmModBlock<CQ15, A<'static>>>();
    assert_send::<AmModBlock<CQ15, VecStorage>>();
    assert_sync::<AmModBlock<C32, A<'static>>>();
    assert_send::<SsbModBlock<C32, A<'static>>>();
    assert_send::<SsbModBlock<CQ15, A<'static>>>();
    assert_send::<SsbModBlock<CQ15, VecStorage>>();
    assert_sync::<SsbModBlock<C32, A<'static>>>();
    assert_send::<AmModConfig>();
    assert_send::<SsbModConfig>();
    fn is_block<B: Block>() {
        assert_send::<B>();
    }
    is_block::<AmModBlock<CQ15, VecStorage>>();
    is_block::<SsbModBlock<C32, VecStorage>>();
}
