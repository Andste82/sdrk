//! `set_param`, `param` and the name-based config access: runtime parameters change the
//! behaviour from the next call, everything else is rejected with the right error.
#![allow(
    clippy::unwrap_used,
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

use common::{arg, norm, rate, run_block, run_frames};
use num_complex::Complex64;
use sdrk::{C32, CQ15, ComplexSample};
use sdrk_blocks::{
    Block, DdcBlock, DdcConfig, DucBlock, DucConfig, Error, FmModBlock, FmModConfig, FrameBlock,
    ParamValue, ResamplerBlock, ResamplerConfig, ScaleBlock, ScaleConfig, SpectrumBlock,
    SpectrumConfig,
};
use sdrk_testkit::VecStorage;

/// Mean phase advance per sample (radians) of a complex signal.
fn phase_step<T: ComplexSample>(y: &[T]) -> f64 {
    let acc: Complex64 = y
        .windows(2)
        .map(|w| w[1].to_c64() * w[0].to_c64().conj())
        .sum();
    arg(acc)
}

fn mean_abs<T: ComplexSample>(y: &[T]) -> f64 {
    y.iter().map(|s| norm(s.to_c64())).sum::<f64>() / y.len() as f64
}

#[test]
fn scale_gain_changes_from_the_next_call() {
    let r = rate(48_000);
    let mut block = ScaleBlock::<C32, _>::new(&ScaleConfig::default(), r, &mut VecStorage).unwrap();
    let input = vec![C32::new(0.5, -0.25); 16];
    let mut out = vec![C32::default(); 16];
    block.process(&input, &mut out).unwrap();
    assert_eq!(out[0], C32::new(0.5, -0.25));
    block.set_param("gain_db", ParamValue::F64(-6.0)).unwrap();
    assert_eq!(block.param("gain_db"), Ok(ParamValue::F64(-6.0)));
    block.process(&input, &mut out).unwrap();
    let g = 10f32.powf(-6.0 / 20.0);
    assert!((out[0].re - 0.5 * g).abs() < 1e-6 && (out[0].im + 0.25 * g).abs() < 1e-6);
    block.set_param("gain_db", ParamValue::F64(40.0)).unwrap();
    block.process(&input, &mut out).unwrap();
    assert!((out[0].re - 50.0).abs() < 1e-3);
}

#[test]
fn scale_limits_depend_on_the_sample_type() {
    let r = rate(48_000);
    let mut c32 = ScaleBlock::<C32, _>::new(&ScaleConfig::default(), r, &mut VecStorage).unwrap();
    assert!(c32.set_param("gain_db", ParamValue::F64(40.0)).is_ok());
    assert!(matches!(
        c32.set_param("gain_db", ParamValue::F64(40.5)),
        Err(Error::OutOfRange {
            name: "gain_db",
            ..
        })
    ));
    assert!(matches!(
        c32.set_param("gain_db", ParamValue::F64(-120.5)),
        Err(Error::OutOfRange { .. })
    ));
    let mut q = ScaleBlock::<CQ15, _>::new(&ScaleConfig::default(), r, &mut VecStorage).unwrap();
    assert!(q.set_param("gain_db", ParamValue::F64(0.0)).is_ok());
    match q.set_param("gain_db", ParamValue::F64(0.5)) {
        Err(Error::OutOfRange {
            name: "gain_db",
            min,
            max,
            got,
        }) => {
            assert_eq!((min, max, got), (-120.0, 0.0, 0.5));
        }
        other => panic!("{other:?}"),
    }
    // A rejected value leaves the block as it was.
    assert_eq!(q.param("gain_db"), Ok(ParamValue::F64(0.0)));
    let too_loud = ScaleConfig { gain_db: 3.0 };
    assert!(ScaleBlock::<CQ15, VecStorage>::new(&too_loud, r, &mut VecStorage).is_err());
    assert!(ScaleBlock::<C32, VecStorage>::new(&too_loud, r, &mut VecStorage).is_ok());
    // Q15 halves exactly at -6.02 dB: 0.5 * 0.5 = 0.25.
    let mut half = ScaleBlock::<CQ15, _>::new(
        &ScaleConfig {
            gain_db: -6.020_599_913_279_624,
        },
        r,
        &mut VecStorage,
    )
    .unwrap();
    let mut out = [CQ15::default(); 1];
    half.process(&[CQ15::from_c32(C32::new(0.5, -0.5))], &mut out)
        .unwrap();
    assert_eq!(out[0], CQ15::from_c32(C32::new(0.25, -0.25)));
}

fn ddc_tone_and_retune<T: ComplexSample>()
where
    sdrk::Sdrk: sdrk_core::kernels::fused::Ddc<T>,
    T: sdrk_blocks::BlockSample,
{
    // A tone at 6 kHz at 48 kHz, decimation 4 (output rate 12 kHz).
    let n = 4000;
    let input: Vec<T> = (0..n)
        .map(|i| {
            let p = std::f64::consts::TAU * 6_000.0 / 48_000.0 * i as f64;
            T::from_c64_round(Complex64::new(0.5 * p.cos(), 0.5 * p.sin()))
        })
        .collect();
    let cfg = DdcConfig {
        frequency_hz: 6_000.0,
        decimation: 4,
        ..DdcConfig::default()
    };
    let mut block = DdcBlock::<T, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    let first = run_block(&mut block, &input[..2000], None);
    let settled = &first[first.len() / 2..];
    assert!(phase_step(settled).abs() < 0.01, "tuned to the tone: DC");
    assert!((mean_abs(settled) - 0.5).abs() < 0.02);
    // Retune by 3 kHz: the output tone moves to +3 kHz = 2 pi * 3000 / 12000 per sample.
    block
        .set_param("frequency_hz", ParamValue::F64(3_000.0))
        .unwrap();
    assert_eq!(block.param("frequency_hz"), Ok(ParamValue::F64(3_000.0)));
    let second = run_block(&mut block, &input[2000..], None);
    let tail = &second[second.len() / 2..];
    let want = std::f64::consts::TAU * 3_000.0 / 12_000.0;
    assert!(
        (phase_step(tail) - want).abs() < 0.01,
        "{} vs {want}",
        phase_step(tail)
    );
    // Out of range: more than half the input rate.
    assert!(matches!(
        block.set_param("frequency_hz", ParamValue::F64(24_001.0)),
        Err(Error::OutOfRange {
            name: "frequency_hz",
            ..
        })
    ));
    assert!(
        block
            .set_param("frequency_hz", ParamValue::F64(-24_000.0))
            .is_ok()
    );
    assert!(matches!(
        block.set_param("frequency_hz", ParamValue::F64(f64::NAN)),
        Err(Error::OutOfRange { .. })
    ));
}

#[test]
fn ddc_frequency_moves_the_output_tone() {
    ddc_tone_and_retune::<C32>();
    ddc_tone_and_retune::<CQ15>();
}

#[test]
fn ddc_retuning_keeps_the_phase_continuous() {
    // Setting the same frequency again must not change the output at all.
    let cfg = DdcConfig {
        frequency_hz: 5_000.0,
        decimation: 3,
        ..DdcConfig::default()
    };
    let input = common::signal::<C32>(3000, 5, 5_000.0 / 48_000.0);
    let mut a = DdcBlock::<C32, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    let whole = run_block(&mut a, &input, None);
    let mut b = DdcBlock::<C32, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    let mut parts = run_block(&mut b, &input[..1500], None);
    b.set_param("frequency_hz", ParamValue::F64(5_000.0))
        .unwrap();
    parts.extend(run_block(&mut b, &input[1500..], None));
    assert_eq!(whole, parts);
}

#[test]
fn duc_frequency_moves_the_output_tone() {
    // Input: DC; output: a carrier at the frequency, at 4 times the input rate (48 kHz).
    let cfg = DucConfig {
        frequency_hz: 6_000.0,
        interpolation: 4,
        ..DucConfig::default()
    };
    let mut block = DucBlock::<C32, _>::new(&cfg, rate(12_000), &mut VecStorage).unwrap();
    let input = vec![C32::new(0.5, 0.0); 800];
    let first = run_block(&mut block, &input, None);
    assert_eq!(first.len(), 3200);
    let want = std::f64::consts::TAU * 6_000.0 / 48_000.0;
    assert!((phase_step(&first[1600..]) - want).abs() < 0.01);
    block
        .set_param("frequency_hz", ParamValue::F64(-9_000.0))
        .unwrap();
    assert_eq!(block.param("frequency_hz"), Ok(ParamValue::F64(-9_000.0)));
    let second = run_block(&mut block, &input, None);
    let want = -std::f64::consts::TAU * 9_000.0 / 48_000.0;
    assert!((phase_step(&second[1600..]) - want).abs() < 0.01);
    // Half the OUTPUT rate is the limit.
    assert!(
        block
            .set_param("frequency_hz", ParamValue::F64(24_000.0))
            .is_ok()
    );
    assert!(matches!(
        block.set_param("frequency_hz", ParamValue::F64(24_001.0)),
        Err(Error::OutOfRange {
            name: "frequency_hz",
            ..
        })
    ));
}

#[test]
fn fm_mod_deviation_changes_the_phase_step() {
    let r = rate(8_000);
    let cfg = FmModConfig {
        deviation_hz: 1_000.0,
    };
    let mut block = FmModBlock::<C32, _>::new(&cfg, r, &mut VecStorage).unwrap();
    let ones = vec![1.0f32; 64];
    let out = run_block(&mut block, &ones, None);
    assert!((phase_step(&out) - std::f64::consts::TAU / 8.0).abs() < 1e-4);
    block
        .set_param("deviation_hz", ParamValue::F64(2_000.0))
        .unwrap();
    let out2 = run_block(&mut block, &ones, None);
    assert!((phase_step(&out2) - std::f64::consts::TAU / 4.0).abs() < 1e-4);
    // The phase continues: the first sample after the change is one step of the new
    // deviation after the last sample before it.
    let jump = out2[0].to_c64() * out[63].to_c64().conj();
    assert!((arg(jump) - std::f64::consts::TAU / 4.0).abs() < 1e-4);
    assert!(matches!(
        block.set_param("deviation_hz", ParamValue::F64(4_001.0)),
        Err(Error::OutOfRange {
            name: "deviation_hz",
            ..
        })
    ));
    assert!(
        block
            .set_param("deviation_hz", ParamValue::F64(-1.0))
            .is_err()
    );
    assert_eq!(block.param("deviation_hz"), Ok(ParamValue::F64(2_000.0)));
}

#[test]
fn ppm_changes_the_output_rate_of_a_tunable_resampler() {
    let cfg = ResamplerConfig {
        tunable: true,
        ..ResamplerConfig::default()
    };
    let input = vec![C32::new(0.5, 0.0); 20_000];
    let mut block = ResamplerBlock::<C32, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    let base = run_block(&mut block, &input, None).len();
    assert!((19_999..=20_001).contains(&base));
    block.set_param("ppm", ParamValue::F64(1000.0)).unwrap();
    assert_eq!(block.param("ppm"), Ok(ParamValue::F64(1000.0)));
    let faster = run_block(&mut block, &input, None).len();
    assert!((20_019..=20_021).contains(&faster), "{faster}");
    block.set_param("ppm", ParamValue::F64(-1000.0)).unwrap();
    let slower = run_block(&mut block, &input, None).len();
    assert!((19_979..=19_981).contains(&slower), "{slower}");
    for bad in [1000.5, -1001.0, f64::NAN] {
        assert!(matches!(
            block.set_param("ppm", ParamValue::F64(bad)),
            Err(Error::OutOfRange { name: "ppm", .. })
        ));
    }
    assert_eq!(block.param("ppm"), Ok(ParamValue::F64(-1000.0)));
}

#[test]
fn spectrum_averaging_and_interval_apply_to_the_next_frame() {
    let cfg = SpectrumConfig {
        fft_size: 64,
        ..SpectrumConfig::default()
    };
    let mut block = SpectrumBlock::<C32, _>::new(&cfg, rate(64_000), &mut VecStorage).unwrap();
    let input = common::signal::<C32>(64 * 40, 3, 0.125);
    let frames = run_frames(&mut block, &input[..64 * 2], None);
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].1.averaged, 1);
    assert_eq!(frames[1].1.first_sample, 64);
    block.set_param("averaging", ParamValue::I64(4)).unwrap();
    assert_eq!(
        FrameBlock::param(&block, "averaging"),
        Ok(ParamValue::I64(4))
    );
    let frames = run_frames(&mut block, &input[128..128 + 64 * 8], None);
    assert_eq!(frames.len(), 2);
    assert!(frames.iter().all(|f| f.1.averaged == 4));
    assert_eq!(frames[0].1.first_sample, 128);
    assert_eq!(frames[1].1.first_sample, 128 + 256);
    // Overlapping windows: a new window every 16 samples.
    block.set_param("averaging", ParamValue::I64(1)).unwrap();
    block
        .set_param("interval_samples", ParamValue::I64(16))
        .unwrap();
    assert_eq!(
        FrameBlock::param(&block, "interval_samples"),
        Ok(ParamValue::I64(16))
    );
    let frames = run_frames(&mut block, &input[640..640 + 64 + 32], None);
    assert_eq!(frames.len(), 3, "windows complete at 64, 80 and 96 samples");
    assert_eq!(
        frames.iter().map(|f| f.1.first_sample).collect::<Vec<_>>(),
        [640, 656, 672]
    );
    assert!(matches!(
        block.set_param("averaging", ParamValue::I64(0)),
        Err(Error::OutOfRange {
            name: "averaging",
            ..
        })
    ));
    assert!(matches!(
        block.set_param("averaging", ParamValue::I64(1025)),
        Err(Error::OutOfRange { .. })
    ));
    assert!(matches!(
        block.set_param("interval_samples", ParamValue::I64(0)),
        Err(Error::OutOfRange { .. })
    ));
}

#[test]
fn errors_for_unknown_build_and_mistyped_parameters() {
    let r = rate(48_000);
    let mut block = ScaleBlock::<C32, _>::new(&ScaleConfig::default(), r, &mut VecStorage).unwrap();
    assert_eq!(
        block.set_param("volume", ParamValue::F64(1.0)),
        Err(Error::UnknownParameter)
    );
    assert_eq!(block.param("volume"), Err(Error::UnknownParameter));
    assert_eq!(
        block.set_param("gain_db", ParamValue::Bool(true)),
        Err(Error::WrongParamType)
    );
    assert_eq!(
        block.set_param("gain_db", ParamValue::I64(3)),
        Err(Error::WrongParamType)
    );
    let cfg = DdcConfig {
        decimation: 4,
        ..DdcConfig::default()
    };
    let mut ddc = DdcBlock::<C32, _>::new(&cfg, r, &mut VecStorage).unwrap();
    for name in [
        "decimation",
        "cutoff_hz",
        "transition_hz",
        "attenuation_db",
        "max_taps",
        "max_chunk",
    ] {
        let v = ddc.param(name).unwrap();
        assert_eq!(
            ddc.set_param(name, v),
            Err(Error::ReadOnlyParameter),
            "{name}"
        );
    }
    // An unknown name is reported before the type of the value is looked at.
    assert_eq!(
        ddc.set_param("nope", ParamValue::Bool(true)),
        Err(Error::UnknownParameter)
    );
    // The build parameters report the values in use, including the rate-dependent defaults.
    assert_eq!(ddc.param("decimation"), Ok(ParamValue::I64(4)));
    assert_eq!(ddc.param("cutoff_hz"), Ok(ParamValue::F64(0.4 * 12_000.0)));
    assert_eq!(
        ddc.param("transition_hz"),
        Ok(ParamValue::F64(0.1 * 12_000.0))
    );
}

#[test]
fn config_set_checks_names_kinds_and_ranges() {
    let mut cfg = DdcConfig::default();
    cfg.set("decimation", ParamValue::I64(8)).unwrap();
    cfg.set("cutoff_hz", ParamValue::F64(1234.0)).unwrap();
    assert_eq!(cfg.decimation, 8);
    assert_eq!(cfg.get("cutoff_hz"), Ok(ParamValue::F64(1234.0)));
    assert!(matches!(
        cfg.set("decimation", ParamValue::I64(1025)),
        Err(Error::OutOfRange { .. })
    ));
    assert!(matches!(
        cfg.set("decimation", ParamValue::F64(8.0)),
        Err(Error::WrongParamType)
    ));
    assert_eq!(
        cfg.set("nope", ParamValue::I64(1)),
        Err(Error::UnknownParameter)
    );
    assert_eq!(cfg.decimation, 8, "failed calls change nothing");
    // A parameter with a rate-dependent default has no value to read until it is set.
    assert!(DdcConfig::default().get("cutoff_hz").is_err());
}

#[test]
fn rate_dependent_limits_are_checked_when_the_block_is_built() {
    let r = rate(48_000);
    let too_high = DdcConfig {
        frequency_hz: 30_000.0,
        ..DdcConfig::default()
    };
    assert!(matches!(
        too_high.validate(r),
        Err(Error::OutOfRange {
            name: "frequency_hz",
            ..
        })
    ));
    let wide = DdcConfig {
        cutoff_hz: Some(23_800.0),
        decimation: 4,
        ..DdcConfig::default()
    };
    assert!(matches!(
        wide.validate(r),
        Err(Error::OutOfRange {
            name: "cutoff_hz",
            ..
        })
    ));
    let huge_chunk = DucConfig {
        interpolation: 8,
        max_chunk: 4,
        ..DucConfig::default()
    };
    assert!(matches!(
        huge_chunk.validate(r),
        Err(Error::OutOfRange {
            name: "max_chunk",
            ..
        })
    ));
    let few_taps = DdcConfig {
        max_taps: 10,
        ..DdcConfig::default()
    };
    assert!(matches!(
        few_taps.validate(r),
        Err(Error::OutOfRange {
            name: "max_taps",
            ..
        })
    ));
}
