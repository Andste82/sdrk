//! `timing()`: ratio, kind and group delay against what the blocks do to an impulse and to
//! sample counts.
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

use common::{norm, rate, run_block, signal};
use num_complex::Complex64;
use sdrk::{C32, CQ15, ComplexSample, Delay, Ratio, RatioKind, Sdrk};
use sdrk_blocks::{
    Block, BlockSample, ConvertBlock, ConvertConfig, ConvertOutBlock, ConvertOutConfig,
    DcBlockBlock, DcBlockConfig, DdcBlock, DdcConfig, DucBlock, DucConfig, FirBlock, FirConfig,
    FmDemodBlock, FmDemodConfig, FmModBlock, FmModConfig, ResampleBlock, ResampleConfig,
    ResampleRealBlock, ScaleBlock, ScaleConfig,
};
use sdrk_core::kernels::{
    decim::Resample,
    filter::Fir,
    fused::{Ddc, Duc},
};
use sdrk_device::formats;
use sdrk_testkit::VecStorage;

/// The group delay of the block in input samples as a float.
fn declared<B: Block>(b: &B) -> f64 {
    b.timing().delay.as_f64()
}

/// Sends a unit impulse at input index `at` and returns the index of the output with the
/// largest magnitude.
fn peak_index<T: ComplexSample, B: Block<In = T, Out = T>>(
    block: &mut B,
    at: usize,
    len: usize,
) -> usize
where
    T: Default,
{
    let mut x = vec![T::default(); len];
    x[at] = T::from_c64_round(Complex64::new(0.5, 0.0));
    let y = run_block(block, &x, None);
    y.iter()
        .enumerate()
        .max_by(|a, b| norm(a.1.to_c64()).total_cmp(&norm(b.1.to_c64())))
        .unwrap()
        .0
}

/// Measured delay in input samples: the best alignment of the impulse with the output grid
/// `output_index * step` (`step` input samples per output sample).
fn measured_delay<T: ComplexSample + Default, B: Block<In = T, Out = T>>(
    make: impl Fn() -> B,
    step: f64,
    alignments: usize,
    len: usize,
) -> f64 {
    let mut best: Option<(f64, f64)> = None;
    for at in 0..alignments {
        let mut block = make();
        let mut x = vec![T::default(); len];
        x[at] = T::from_c64_round(Complex64::new(0.5, 0.0));
        let y = run_block(&mut block, &x, None);
        let (i, peak) = y
            .iter()
            .enumerate()
            .map(|(i, s)| (i, norm(s.to_c64())))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        let delay = i as f64 * step - at as f64;
        if best.is_none_or(|(p, _)| peak > p) {
            best = Some((peak, delay));
        }
    }
    best.unwrap().1
}

fn check_fir<T: BlockSample + Default>()
where
    Sdrk: Fir<T>,
{
    let cfg = FirConfig {
        cutoff_hz: 4_000.0,
        transition_hz: 2_000.0,
        ..FirConfig::default()
    };
    let make = || FirBlock::<T, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    let block = make();
    let t = block.timing();
    assert_eq!((t.ratio, t.kind), (Ratio::ONE, RatioKind::Constant));
    let n = block.num_taps();
    assert_eq!(
        t.delay.reduced(),
        Delay::new((n as u64 - 1) / 2, 1).unwrap()
    );
    let measured = peak_index::<T, _>(&mut make(), 10, 400) as f64 - 10.0;
    assert!(
        (measured - declared(&block)).abs() <= 0.5,
        "{measured} vs {}",
        declared(&block)
    );
}

fn check_ddc<T: BlockSample + Default>()
where
    Sdrk: Ddc<T>,
{
    for dec in [1, 2, 4, 5] {
        let cfg = DdcConfig {
            decimation: dec,
            max_chunk: 64,
            ..DdcConfig::default()
        };
        let make = || DdcBlock::<T, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
        let t = make().timing();
        assert_eq!(
            (t.ratio, t.kind),
            (Ratio::new(1, dec as u64).unwrap(), RatioKind::Constant)
        );
        let measured = measured_delay::<T, _>(make, dec as f64, dec, 600);
        assert!(
            (measured - t.delay.as_f64()).abs() <= 0.5,
            "ddc {dec}: {measured} vs {}",
            t.delay.as_f64()
        );
    }
}

fn check_duc<T: BlockSample + Default>()
where
    Sdrk: Duc<T>,
{
    for l in [1, 2, 3, 4] {
        let cfg = DucConfig {
            interpolation: l,
            max_chunk: 64,
            ..DucConfig::default()
        };
        let make = || DucBlock::<T, _>::new(&cfg, rate(12_000), &mut VecStorage).unwrap();
        let t = make().timing();
        assert_eq!(
            (t.ratio, t.kind),
            (Ratio::new(l as u64, 1).unwrap(), RatioKind::Constant)
        );
        let measured = measured_delay::<T, _>(make, 1.0 / l as f64, 1, 200);
        assert!(
            (measured - t.delay.as_f64()).abs() <= 0.5,
            "duc {l}: {measured} vs {}",
            t.delay.as_f64()
        );
    }
}

fn check_resampler<T: BlockSample + Default>()
where
    Sdrk: Resample<T>,
{
    for (out_hz, l, m) in [
        (96_000u64, 2u64, 1u64),
        (24_000, 1, 2),
        (72_000, 3, 2),
        (32_000, 2, 3),
    ] {
        let cfg = ResampleConfig {
            rate_out_hz: out_hz as f64,
            ..ResampleConfig::default()
        };
        let make = || ResampleBlock::<T, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
        let t = make().timing();
        assert_eq!(
            (t.ratio, t.kind),
            (Ratio::new(l, m).unwrap(), RatioKind::Constant)
        );
        let measured = measured_delay::<T, _>(make, m as f64 / l as f64, m as usize, 600);
        assert!(
            (measured - t.delay.as_f64()).abs() <= 0.5,
            "resampler {l}/{m}: {measured} vs {}",
            t.delay.as_f64()
        );
    }
    // The tunable resampler declares a variable ratio and its own delay.
    let cfg = ResampleConfig {
        tunable: true,
        ..ResampleConfig::default()
    };
    let make = || ResampleBlock::<T, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    let t = make().timing();
    assert_eq!((t.ratio, t.kind), (Ratio::ONE, RatioKind::Variable));
    let measured = measured_delay::<T, _>(make, 1.0, 1, 200);
    assert!(
        (measured - t.delay.as_f64()).abs() <= 0.5,
        "tuned: {measured} vs {}",
        t.delay.as_f64()
    );
    let cfg = ResampleConfig {
        rate_out_hz: 44_100.0,
        tunable: true,
        ..cfg
    };
    let t = ResampleBlock::<T, _>::new(&cfg, rate(48_000), &mut VecStorage)
        .unwrap()
        .timing();
    assert_eq!(
        (t.ratio, t.kind),
        (Ratio::new(147, 160).unwrap(), RatioKind::Variable)
    );
}

#[test]
fn filters_and_resamplers_declare_the_delay_they_have() {
    check_fir::<C32>();
    check_fir::<CQ15>();
    check_ddc::<C32>();
    check_ddc::<CQ15>();
    check_duc::<C32>();
    check_duc::<CQ15>();
    check_resampler::<C32>();
    check_resampler::<CQ15>();
}

#[test]
fn the_real_resampler_declares_the_same_timing() {
    let cfg = ResampleConfig {
        rate_out_hz: 32_000.0,
        ..ResampleConfig::default()
    };
    let c = ResampleBlock::<C32, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    let r = ResampleRealBlock::<f32, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    assert_eq!(c.timing(), r.timing());
    let mut x = vec![0.0f32; 600];
    x[300] = 0.5;
    let mut block = ResampleRealBlock::<f32, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    let y = run_block(&mut block, &x, None);
    let i = y
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .unwrap()
        .0;
    let t = r.timing();
    let delay = i as f64 * 1.5 - 300.0;
    assert!((delay - t.delay.as_f64()).abs() <= 0.75);
}

#[test]
fn same_rate_blocks_have_ratio_one_and_no_delay() {
    let r = rate(48_000);
    let t = ScaleBlock::<C32, _>::new(&ScaleConfig::default(), r, &mut VecStorage)
        .unwrap()
        .timing();
    assert_eq!(
        (t.ratio, t.kind, t.delay),
        (Ratio::ONE, RatioKind::Constant, Delay::ZERO)
    );
    let t = DcBlockBlock::<CQ15, _>::new(&DcBlockConfig::default(), r, &mut VecStorage)
        .unwrap()
        .timing();
    assert_eq!(
        (t.ratio, t.kind, t.delay),
        (Ratio::ONE, RatioKind::Constant, Delay::ZERO)
    );
    let t = FmModBlock::<C32, _>::new(&FmModConfig::default(), rate(240_000), &mut VecStorage)
        .unwrap()
        .timing();
    assert_eq!(
        (t.ratio, t.kind, t.delay),
        (Ratio::ONE, RatioKind::Constant, Delay::ZERO)
    );
}

#[test]
fn convert_blocks_declare_samples_per_byte() {
    for (format, samples, bytes) in [
        (formats::CU8, 1u64, 2u64),
        (formats::CS16, 1, 4),
        (formats::CF32, 1, 8),
        (formats::PACKED12, 1, 3),
        (formats::PACKED10, 2, 5),
    ] {
        let t =
            ConvertBlock::<C32, _>::new(&ConvertConfig { format }, rate(1_000), &mut VecStorage)
                .unwrap()
                .timing();
        assert_eq!(t.ratio, Ratio::new(samples, bytes).unwrap(), "{format:?}");
        assert_eq!((t.kind, t.delay), (RatioKind::Constant, Delay::ZERO));
        let t = ConvertOutBlock::<CQ15, _>::new(
            &ConvertOutConfig { format },
            rate(1_000),
            &mut VecStorage,
        )
        .unwrap()
        .timing();
        assert_eq!(t.ratio, Ratio::new(bytes, samples).unwrap(), "{format:?}");
    }
}

#[test]
fn fm_demod_declares_the_discriminator_and_the_audio_filter() {
    let r = rate(240_000);
    let plain = FmDemodConfig {
        deemphasis_us: 0.0,
        audio_decimation: 1,
        ..FmDemodConfig::default()
    };
    let t = FmDemodBlock::<C32, _>::new(&plain, r, &mut VecStorage)
        .unwrap()
        .timing();
    assert_eq!(
        (t.ratio, t.kind, t.delay.reduced()),
        (Ratio::ONE, RatioKind::Constant, Delay::new(1, 2).unwrap())
    );

    let cfg = FmDemodConfig {
        deemphasis_us: 0.0,
        audio_decimation: 5,
        ..FmDemodConfig::default()
    };
    let make = || FmDemodBlock::<C32, _>::new(&cfg, r, &mut VecStorage).unwrap();
    let t = make().timing();
    assert_eq!(t.ratio, Ratio::new(1, 5).unwrap());
    // A frequency step at input sample 1000 (phase advance of 0.2 rad per sample afterwards).
    let n = 2000;
    let mut phase = 0.0f64;
    let x: Vec<C32> = (0..n)
        .map(|i| {
            if i > 1000 {
                phase += 0.2;
            }
            C32::new(phase.cos() as f32, phase.sin() as f32)
        })
        .collect();
    let y = run_block(&mut make(), &x, None);
    assert_eq!(y.len(), n / 5);
    // The step response of the symmetric filter crosses its half value at the declared delay
    // after the edge of the discriminator output (between samples 1000 and 1001).
    let high = y[y.len() - 10];
    let cross = y.iter().position(|v| *v > high / 2.0).unwrap();
    let (a, b) = (y[cross - 1], y[cross]);
    let frac = (high / 2.0 - a) / (b - a);
    let at = ((cross - 1) as f32 + frac) * 5.0;
    let expected = 1000.0 + t.delay.as_f64() as f32;
    assert!((at - expected).abs() < 1.5, "{at} vs {expected}");
}

#[test]
fn measured_output_counts_follow_the_declared_ratio() {
    fn count<B: Block>(mut block: B, input: &[B::In]) -> (usize, Ratio)
    where
        B::Out: Default,
    {
        let ratio = block.timing().ratio;
        (run_block(&mut block, input, None).len(), ratio)
    }
    let r = rate(48_000);
    let x = signal::<C32>(9_600, 1, 0.01);
    let cases = [
        count(
            DdcBlock::<C32, _>::new(
                &DdcConfig {
                    decimation: 8,
                    ..DdcConfig::default()
                },
                r,
                &mut VecStorage,
            )
            .unwrap(),
            &x,
        ),
        count(
            DucBlock::<C32, _>::new(
                &DucConfig {
                    interpolation: 5,
                    ..DucConfig::default()
                },
                r,
                &mut VecStorage,
            )
            .unwrap(),
            &x,
        ),
        count(
            ResampleBlock::<C32, _>::new(
                &ResampleConfig {
                    rate_out_hz: 44_100.0,
                    max_taps: 8192,
                    ..ResampleConfig::default()
                },
                r,
                &mut VecStorage,
            )
            .unwrap(),
            &x,
        ),
        count(
            ResampleBlock::<C32, _>::new(
                &ResampleConfig {
                    rate_out_hz: 44_100.0,
                    tunable: true,
                    ..ResampleConfig::default()
                },
                r,
                &mut VecStorage,
            )
            .unwrap(),
            &x,
        ),
        count(
            FirBlock::<C32, _>::new(&FirConfig::default(), r, &mut VecStorage).unwrap(),
            &x,
        ),
    ];
    for (n, ratio) in cases {
        let want = 9_600.0 * ratio.num() as f64 / ratio.den() as f64;
        assert!((n as f64 - want).abs() <= 1.0, "{n} vs {want} ({ratio:?})");
    }
}
