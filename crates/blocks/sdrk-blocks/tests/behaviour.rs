//! Block-size invariance, `reset`, extreme inputs and the contract of `process` for every block.
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

use common::{plans, rate, real_signal, run_block, run_frames, signal};
use sdrk::{C32, CQ15, Q15, Sdrk};
use sdrk_blocks::{
    Block, BlockSample, ConvertBlock, ConvertConfig, ConvertOutBlock, ConvertOutConfig,
    DcBlockBlock, DcBlockConfig, DdcBlock, DdcConfig, DucBlock, DucConfig, FirBlock, FirConfig,
    FmDemodBlock, FmDemodConfig, FmModBlock, FmModConfig, FrameBlock, ResampleBlock,
    ResampleConfig, ResampleRealBlock, ScaleBlock, ScaleConfig, SpectrumBlock, SpectrumConfig,
    SpectrumSample, SpectrumWindow,
};
use sdrk_core::kernels::{
    basic::Scale,
    convert::{Convert, ConvertOut},
    decim::Resample,
    demod::DcBlock,
    filter::Fir,
    fused::{Ddc, Duc, FmDemod},
    modulate::FmMod,
};
use sdrk_device::formats;
use sdrk_testkit::VecStorage;

fn same_bits<T: bytemuck::Pod>(a: &[T], b: &[T]) -> bool {
    bytemuck::cast_slice::<T, u8>(a) == bytemuck::cast_slice::<T, u8>(b)
}

/// Identical bits, or, for `f32` and `C32` outputs only, an SNR of at least 120 dB of `b`
/// against `a`: the blocks with the `f32`/`C32` `dc_block`, `dc_block_real` or `deemphasis`
/// inside, whose SIMD kernels are tolerance kernels whose bits may depend on the split
/// (maintainer decision of 2026-10-29). Every other sample type (`Q15`, `CQ15`) must keep
/// identical bits.
fn same_or_close<T: bytemuck::Pod>(a: &[T], b: &[T]) -> bool {
    use core::any::TypeId;
    if same_bits(a, b) {
        return true;
    }
    let float =
        TypeId::of::<T>() == TypeId::of::<f32>() || TypeId::of::<T>() == TypeId::of::<C32>();
    if !float || a.len() != b.len() {
        return false;
    }
    let (a, b) = (
        bytemuck::cast_slice::<T, f32>(a),
        bytemuck::cast_slice::<T, f32>(b),
    );
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        sig += f64::from(x) * f64::from(x);
        err += (f64::from(x) - f64::from(y)).powi(2);
    }
    10.0 * (sig / err).log10() >= 120.0
}

/// Runs a fresh block whole and in several chunkings and requires identical output.
fn invariant<B: Block>(name: &str, input: &[B::In], make: impl Fn() -> B)
where
    B::Out: Default + bytemuck::Pod,
{
    invariant_with(name, input, make, same_bits);
}

/// [`invariant`] with [`same_or_close`].
fn invariant_close<B: Block>(name: &str, input: &[B::In], make: impl Fn() -> B)
where
    B::Out: Default + bytemuck::Pod,
{
    invariant_with(name, input, make, same_or_close);
}

fn invariant_with<B: Block>(
    name: &str,
    input: &[B::In],
    make: impl Fn() -> B,
    same: fn(&[B::Out], &[B::Out]) -> bool,
) where
    B::Out: Default + bytemuck::Pod,
{
    let whole = run_block(&mut make(), input, None);
    assert!(!whole.is_empty(), "{name}: no output");
    for plan in plans(11) {
        let chunked = run_block(&mut make(), input, Some(plan));
        assert_eq!(
            chunked.len(),
            whole.len(),
            "{name}: output count with {plan:?}"
        );
        assert!(same(&whole, &chunked), "{name}: output with {plan:?}");
    }
}

/// After `reset` the block behaves like a new one.
fn resets<B: Block>(name: &str, input: &[B::In], make: impl Fn() -> B)
where
    B::Out: Default + bytemuck::Pod,
{
    let fresh = run_block(&mut make(), input, None);
    let mut block = make();
    let _ = run_block(&mut block, &input[..input.len() / 3 + 1], None);
    block.reset();
    let again = run_block(&mut block, input, None);
    assert_eq!(again.len(), fresh.len(), "{name}: count after reset");
    assert!(same_bits(&fresh, &again), "{name}: output after reset");
}

fn complex_blocks<T: BlockSample>(seed: u64)
where
    Sdrk: Scale<T>
        + DcBlock<T>
        + Fir<T>
        + Ddc<T>
        + Duc<T>
        + Resample<T>
        + FmDemod<T>
        + sdrk_core::kernels::demod::QuadDemod<T>
        + sdrk_core::kernels::decim::ResampleReal<T::Real>
        + FmMod<T>,
    T::Real: Default,
{
    let r = rate(48_000);
    let x = signal::<T>(3000, seed, 0.05);

    let cfg = ScaleConfig { gain_db: -3.0 };
    let make = || ScaleBlock::<T, _>::new(&cfg, r, &mut VecStorage).unwrap();
    invariant("scale", &x, make);
    resets("scale", &x, make);

    let cfg = DcBlockConfig { cutoff_hz: 50.0 };
    let make = || DcBlockBlock::<T, _>::new(&cfg, r, &mut VecStorage).unwrap();
    invariant_close("dc_block", &x, make);
    resets("dc_block", &x, make);

    let cfg = FirConfig {
        cutoff_hz: 5_000.0,
        transition_hz: 2_000.0,
        ..FirConfig::default()
    };
    let make = || FirBlock::<T, _>::new(&cfg, r, &mut VecStorage).unwrap();
    invariant("fir", &x, make);
    resets("fir", &x, make);

    let cfg = DdcConfig {
        frequency_hz: 2_000.0,
        decimation: 4,
        max_chunk: 50,
        ..DdcConfig::default()
    };
    let make = || DdcBlock::<T, _>::new(&cfg, r, &mut VecStorage).unwrap();
    invariant("ddc", &x, make);
    resets("ddc", &x, make);

    let cfg = DucConfig {
        frequency_hz: 7_000.0,
        interpolation: 3,
        max_chunk: 60,
        ..DucConfig::default()
    };
    let make = || DucBlock::<T, _>::new(&cfg, r, &mut VecStorage).unwrap();
    invariant("duc", &x, make);
    resets("duc", &x, make);

    for tunable in [false, true] {
        let cfg = ResampleConfig {
            rate_out_hz: 32_000.0,
            tunable,
            ppm: if tunable { 300.0 } else { 0.0 },
            ..ResampleConfig::default()
        };
        let make = || ResampleBlock::<T, _>::new(&cfg, r, &mut VecStorage).unwrap();
        invariant("resample", &x, make);
        resets("resample", &x, make);
    }
    // Decimation by 100 skips input between two windows.
    // (A sloppy 8 dB filter keeps the taps per phase small, so the step of 100 input samples
    // exceeds the work buffer and the block has to skip input.)
    let cfg = ResampleConfig {
        rate_out_hz: 480.0,
        cutoff_hz: Some(100.0),
        transition_hz: Some(200.0),
        attenuation_db: 8.0,
        tunable: true,
        ..ResampleConfig::default()
    };
    let big = signal::<T>(20_000, seed, 0.0001);
    let make = || ResampleBlock::<T, _>::new(&cfg, r, &mut VecStorage).unwrap();
    invariant("resample by 100", &big, make);
    resets("resample by 100", &big, make);

    let r240 = rate(240_000);
    for (dec, deemph) in [(1, 50.0), (5, 50.0), (4, 0.0)] {
        let cfg = FmDemodConfig {
            audio_decimation: dec,
            deemphasis_us: deemph,
            ..FmDemodConfig::default()
        };
        let make = || FmDemodBlock::<T, _>::new(&cfg, r240, &mut VecStorage).unwrap();
        let x = signal::<T>(3000, seed, 0.01);
        invariant_close("fm_demod", &x, make);
        resets("fm_demod", &x, make);
    }

    let cfg = FmModConfig {
        deviation_hz: 3_000.0,
    };
    let make = || FmModBlock::<T, _>::new(&cfg, r, &mut VecStorage).unwrap();
    let audio = real_signal::<T::Real>(2000, seed, 0.01);
    invariant("fm_mod", &audio, make);
    resets("fm_mod", &audio, make);
}

#[test]
fn complex_blocks_are_invariant_and_reset_c32() {
    complex_blocks::<C32>(1);
}

#[test]
fn complex_blocks_are_invariant_and_reset_cq15() {
    complex_blocks::<CQ15>(2);
}

fn real_resample_blocks<R: sdrk_blocks::BlockReal + sdrk::Sample>(seed: u64)
where
    Sdrk: sdrk_core::kernels::decim::ResampleReal<R>,
{
    let x = real_signal::<R>(3000, seed, 0.03);
    for (out_hz, tunable) in [(32_000.0, false), (44_100.0, true), (12_000.0, true)] {
        let cfg = ResampleConfig {
            rate_out_hz: out_hz,
            tunable,
            max_taps: 8192,
            ..ResampleConfig::default()
        };
        let make = || ResampleRealBlock::<R, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
        invariant("real resample", &x, make);
        resets("real resample", &x, make);
    }
}

#[test]
fn real_resample_blocks_are_invariant_and_reset() {
    real_resample_blocks::<f32>(3);
    real_resample_blocks::<Q15>(4);
}

fn converts<T: BlockSample>()
where
    Sdrk: Convert<T> + ConvertOut<T>,
{
    for format in [
        formats::CU8,
        formats::CS8,
        formats::CS16,
        formats::PACKED12,
        formats::PACKED10,
        formats::CF32,
        formats::CS12_LSB16,
    ] {
        let samples = 600;
        let x = signal::<T>(samples, 9, 0.05);
        // Encode with the block, decode with the block; the chunking must not matter.
        let cfg_out = ConvertOutConfig { format };
        let make_out =
            || ConvertOutBlock::<T, _>::new(&cfg_out, rate(1_000_000), &mut VecStorage).unwrap();
        let bytes = run_block(&mut make_out(), &x, None);
        assert_eq!(
            bytes.len(),
            format.bytes_for_samples(samples).unwrap(),
            "{format:?}"
        );
        invariant("convert_out", &x, make_out);
        resets("convert_out", &x, make_out);
        let cfg = ConvertConfig { format };
        let make = || ConvertBlock::<T, _>::new(&cfg, rate(1_000_000), &mut VecStorage).unwrap();
        invariant("convert", &bytes, make);
        resets("convert", &bytes, make);
        let back = run_block(&mut make(), &bytes, None);
        assert_eq!(back.len(), samples);
    }
}

#[test]
fn conversions_are_invariant_and_reset() {
    converts::<C32>();
    converts::<CQ15>();
}

#[test]
fn convert_only_takes_whole_groups() {
    // PACKED10 holds two complex samples (40 bits) in 5 bytes.
    let cfg = ConvertConfig {
        format: formats::PACKED10,
    };
    let mut block = ConvertBlock::<CQ15, _>::new(&cfg, rate(1_000_000), &mut VecStorage).unwrap();
    let group_bytes = block.min_input();
    let group_samples = formats::PACKED10.samples_in_bytes(group_bytes);
    assert_eq!((group_bytes, group_samples), (5, 2));
    let mut out = [CQ15::default(); 8];
    let bytes = [0x12u8; 23];
    let p = block.process(&bytes, &mut out).unwrap();
    assert_eq!(
        (p.consumed, p.produced),
        (20, 8),
        "four groups fit, 3 bytes are left"
    );
    assert_eq!(block.process(&bytes[..4], &mut out).unwrap().consumed, 0);
    // The output room limits the groups too.
    let p = block.process(&bytes, &mut out[..3]).unwrap();
    assert_eq!((p.consumed, p.produced), (5, 2));
    assert_eq!(
        block.timing().ratio,
        sdrk::Ratio::new(2, 5).unwrap(),
        "samples per byte"
    );
    // A planar-free format with one sample per group.
    let cs8 = ConvertBlock::<CQ15, _>::new(
        &ConvertConfig {
            format: formats::CS8,
        },
        rate(1),
        &mut VecStorage,
    )
    .unwrap();
    assert_eq!(cs8.min_input(), 2);

    let mut out_block = ConvertOutBlock::<CQ15, _>::new(
        &ConvertOutConfig {
            format: formats::PACKED10,
        },
        rate(1_000_000),
        &mut VecStorage,
    )
    .unwrap();
    assert_eq!(out_block.min_input(), 2);
    assert_eq!(out_block.timing().ratio, sdrk::Ratio::new(5, 2).unwrap());
    let mut raw = [0u8; 20];
    let samples = [CQ15::default(); 7];
    let p = out_block.process(&samples, &mut raw[..4]).unwrap();
    assert_eq!(
        (p.consumed, p.produced),
        (0, 0),
        "4 bytes hold less than one group of 5"
    );
    let p = out_block.process(&samples, &mut raw).unwrap();
    assert_eq!(
        (p.consumed, p.produced),
        (6, 15),
        "three groups of two samples"
    );
    let p = out_block.process(&samples[..1], &mut raw).unwrap();
    assert_eq!(
        (p.consumed, p.produced),
        (0, 0),
        "one sample is half a group"
    );
}

#[test]
fn planar_formats_are_rejected_and_real_ones_need_the_real_block() {
    // `convert` streams interleaved complex and real formats; planar ones cannot be streamed.
    assert!(
        ConvertConfig {
            format: formats::CS16_PLANAR
        }
        .validate(rate(1_000_000))
        .is_err()
    );
    for format in [formats::RS16, formats::RF32] {
        assert!(
            ConvertConfig { format }.validate(rate(1_000_000)).is_ok(),
            "{format:?}"
        );
        // The complex block refuses a real format, the real block a complex one.
        let mut arena = sdrk_blocks::ArenaStorage::new(&mut []);
        assert!(
            ConvertBlock::<C32, _>::new(&ConvertConfig { format }, rate(1_000_000), &mut arena)
                .is_err()
        );
    }
    let mut arena = sdrk_blocks::ArenaStorage::new(&mut []);
    assert!(
        sdrk_blocks::ConvertRealBlock::<f32, _>::new(
            &ConvertConfig::default(),
            rate(1_000_000),
            &mut arena
        )
        .is_err()
    );
    // `convert_out` has no real variant.
    for format in [formats::CS16_PLANAR, formats::RS16, formats::RF32] {
        assert!(
            ConvertOutConfig { format }
                .validate(rate(1_000_000))
                .is_err()
        );
    }
}

fn spectrum_cases<T: SpectrumSample>()
where
    Sdrk: sdrk_core::kernels::spectrum::Window<T>
        + sdrk_core::kernels::fft::Fft<T>
        + sdrk_core::kernels::spectrum::PowerDb<f32>,
{
    let x = signal::<T>(64 * 23 + 5, 21, 0.125);
    for (size, window, averaging, interval) in [
        (64usize, SpectrumWindow::Hann, 1usize, None),
        (64, SpectrumWindow::BlackmanHarris, 3, None),
        (64, SpectrumWindow::Kaiser, 2, Some(16)),
        (64, SpectrumWindow::Hann, 2, Some(100)),
        (128, SpectrumWindow::Hann, 1, Some(32)),
    ] {
        let cfg = SpectrumConfig {
            fft_size: size,
            window,
            averaging,
            interval_samples: interval,
            ..SpectrumConfig::default()
        };
        let make = || SpectrumBlock::<T, _>::new(&cfg, rate(64_000), &mut VecStorage).unwrap();
        let whole = run_frames(&mut make(), &x, None);
        assert!(!whole.is_empty(), "{cfg:?}");
        for plan in plans(31) {
            let chunked = run_frames(&mut make(), &x, Some(plan));
            assert_eq!(chunked, whole, "{cfg:?} {plan:?}");
        }
        // reset
        let mut block = make();
        let _ = run_frames(&mut block, &x[..200], None);
        block.reset();
        assert_eq!(run_frames(&mut block, &x, None), whole, "reset {cfg:?}");
    }
}

#[test]
fn spectrum_is_invariant_and_resets() {
    spectrum_cases::<C32>();
    spectrum_cases::<CQ15>();
}

// ---------------------------------------------------------------------------------------------
// extreme inputs

fn finite_or_nan_is_fine<B: Block>(name: &str, input: &[B::In], make: impl Fn() -> B) -> Vec<B::Out>
where
    B::Out: Default,
{
    let out = run_block(&mut make(), input, None);
    assert!(!out.is_empty(), "{name}: no output");
    out
}

#[test]
fn silence_and_full_scale_do_not_break_the_q15_blocks() {
    let r = rate(48_000);
    let silence = vec![CQ15::default(); 3000];
    let full: Vec<CQ15> = (0..3000)
        .map(|i| {
            let v = if i % 2 == 0 { 32767 } else { -32768 };
            CQ15 {
                re: Q15(v),
                im: Q15(-v.saturating_add(1)),
            }
        })
        .collect();
    for x in [&silence, &full] {
        let cfg = FirConfig {
            cutoff_hz: 5_000.0,
            transition_hz: 2_000.0,
            ..FirConfig::default()
        };
        finite_or_nan_is_fine("fir", x, || {
            FirBlock::<CQ15, _>::new(&cfg, r, &mut VecStorage).unwrap()
        });
        let cfg = DdcConfig {
            frequency_hz: 1_000.0,
            decimation: 4,
            ..DdcConfig::default()
        };
        finite_or_nan_is_fine("ddc", x, || {
            DdcBlock::<CQ15, _>::new(&cfg, r, &mut VecStorage).unwrap()
        });
        let cfg = DucConfig {
            interpolation: 4,
            ..DucConfig::default()
        };
        finite_or_nan_is_fine("duc", x, || {
            DucBlock::<CQ15, _>::new(&cfg, r, &mut VecStorage).unwrap()
        });
        let cfg = ResampleConfig {
            rate_out_hz: 44_100.0,
            tunable: true,
            ..ResampleConfig::default()
        };
        finite_or_nan_is_fine("resample", x, || {
            ResampleBlock::<CQ15, _>::new(&cfg, r, &mut VecStorage).unwrap()
        });
        let cfg = DcBlockConfig::default();
        finite_or_nan_is_fine("dc_block", x, || {
            DcBlockBlock::<CQ15, _>::new(&cfg, r, &mut VecStorage).unwrap()
        });
        let cfg = FmDemodConfig {
            audio_decimation: 3,
            ..FmDemodConfig::default()
        };
        finite_or_nan_is_fine("fm_demod", x, || {
            FmDemodBlock::<CQ15, _>::new(&cfg, rate(240_000), &mut VecStorage).unwrap()
        });
        let cfg = SpectrumConfig {
            fft_size: 64,
            ..SpectrumConfig::default()
        };
        let frames = run_frames(
            &mut SpectrumBlock::<CQ15, _>::new(&cfg, r, &mut VecStorage).unwrap(),
            x,
            None,
        );
        assert!(frames.iter().all(|(f, _)| f.iter().all(|v| v.is_finite())));
    }
    // Silence stays silent (apart from the filters' exact zero).
    let cfg = FirConfig {
        cutoff_hz: 5_000.0,
        transition_hz: 2_000.0,
        ..FirConfig::default()
    };
    let out = run_block(
        &mut FirBlock::<CQ15, _>::new(&cfg, r, &mut VecStorage).unwrap(),
        &silence,
        None,
    );
    assert!(out.iter().all(|s| *s == CQ15::default()));
}

#[test]
fn f32_blocks_survive_nan_and_infinity() {
    let r = rate(48_000);
    let specials = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        1e38,
        -1e38,
        1e-45,
    ];
    let mut x = signal::<C32>(3000, 6, 0.05);
    for (i, s) in specials.iter().enumerate() {
        x[100 + 7 * i] = C32::new(*s, *s);
        x[1500 + 3 * i] = C32::new(*s, 0.5);
    }
    let cfg = FirConfig {
        cutoff_hz: 5_000.0,
        transition_hz: 2_000.0,
        ..FirConfig::default()
    };
    let n = run_block(
        &mut FirBlock::<C32, _>::new(&cfg, r, &mut VecStorage).unwrap(),
        &x,
        None,
    )
    .len();
    assert_eq!(n, x.len());
    let cfg = DdcConfig {
        decimation: 4,
        ..DdcConfig::default()
    };
    let n = run_block(
        &mut DdcBlock::<C32, _>::new(&cfg, r, &mut VecStorage).unwrap(),
        &x,
        None,
    )
    .len();
    assert_eq!(n, 750);
    let cfg = ResampleConfig {
        rate_out_hz: 44_100.0,
        tunable: true,
        ..ResampleConfig::default()
    };
    let n = run_block(
        &mut ResampleBlock::<C32, _>::new(&cfg, r, &mut VecStorage).unwrap(),
        &x,
        None,
    )
    .len();
    assert!((2755..=2758).contains(&n), "{n}");
    let cfg = FmDemodConfig {
        audio_decimation: 5,
        ..FmDemodConfig::default()
    };
    let n = run_block(
        &mut FmDemodBlock::<C32, _>::new(&cfg, rate(240_000), &mut VecStorage).unwrap(),
        &x,
        None,
    )
    .len();
    assert_eq!(n, 600);
    let cfg = SpectrumConfig {
        fft_size: 64,
        ..SpectrumConfig::default()
    };
    let frames = run_frames(
        &mut SpectrumBlock::<C32, _>::new(&cfg, r, &mut VecStorage).unwrap(),
        &x,
        None,
    );
    assert!(!frames.is_empty()); // NaN bins are allowed, a panic is not
    let cfg = DcBlockConfig::default();
    let n = run_block(
        &mut DcBlockBlock::<C32, _>::new(&cfg, r, &mut VecStorage).unwrap(),
        &x,
        None,
    )
    .len();
    assert_eq!(n, x.len());
    let audio = [f32::NAN, f32::INFINITY, 0.5, -1e30, 0.0];
    let cfg = FmModConfig::default();
    let n = run_block(
        &mut FmModBlock::<C32, _>::new(&cfg, rate(240_000), &mut VecStorage).unwrap(),
        &audio,
        None,
    )
    .len();
    assert_eq!(n, 5);
    // After a NaN the feedback state of the DC blocker recovers only through a reset.
    let mut block = DcBlockBlock::<C32, _>::new(&cfg_dc(), r, &mut VecStorage).unwrap();
    let _ = run_block(&mut block, &[C32::new(f32::NAN, 0.0)], None);
    block.reset();
    let out = run_block(&mut block, &[C32::new(1.0, 0.0)], None);
    assert!(out[0].re.is_finite());
}

fn cfg_dc() -> DcBlockConfig {
    DcBlockConfig::default()
}

#[test]
fn small_output_buffers_never_lose_samples() {
    // Feed the whole input with output buffers of 1 to 3 samples; the output must not change.
    fn small<B: Block>(input: &[B::In], make: impl Fn() -> B)
    where
        B::Out: Default + bytemuck::Pod,
    {
        small_with(input, make, same_bits);
    }
    fn small_with<B: Block>(
        input: &[B::In],
        make: impl Fn() -> B,
        same: fn(&[B::Out], &[B::Out]) -> bool,
    ) where
        B::Out: Default + bytemuck::Pod,
    {
        let want = run_block(&mut make(), input, None);
        let mut block = make();
        let mut got = Vec::new();
        let mut pending: Vec<B::In> = input.to_vec();
        let mut room = [B::Out::default(); 3];
        // Calls with an empty input flush the outputs that were left pending.
        loop {
            let p = block.process(&pending, &mut room).unwrap();
            got.extend_from_slice(&room[..p.produced]);
            pending.drain(..p.consumed);
            if p.consumed == 0 && p.produced == 0 {
                break;
            }
        }
        assert!(pending.is_empty(), "the block stalled");
        assert_eq!(got.len(), want.len());
        assert!(same(&want, &got));
    }
    let r = rate(48_000);
    let x = signal::<C32>(1500, 4, 0.04);
    let cfg = DdcConfig {
        decimation: 4,
        ..DdcConfig::default()
    };
    small(&x, || {
        DdcBlock::<C32, _>::new(&cfg, r, &mut VecStorage).unwrap()
    });
    let cfg = DucConfig {
        interpolation: 3,
        ..DucConfig::default()
    };
    small(&x, || {
        DucBlock::<C32, _>::new(&cfg, r, &mut VecStorage).unwrap()
    });
    for tunable in [false, true] {
        let cfg = ResampleConfig {
            rate_out_hz: 32_000.0,
            tunable,
            ..ResampleConfig::default()
        };
        small(&x, || {
            ResampleBlock::<C32, _>::new(&cfg, r, &mut VecStorage).unwrap()
        });
    }
    let cfg = FmDemodConfig {
        audio_decimation: 5,
        ..FmDemodConfig::default()
    };
    small_with(
        &x,
        || FmDemodBlock::<C32, _>::new(&cfg, rate(240_000), &mut VecStorage).unwrap(),
        same_or_close,
    );
    let cfg = FirConfig {
        cutoff_hz: 5_000.0,
        transition_hz: 2_000.0,
        ..FirConfig::default()
    };
    small(&x, || {
        FirBlock::<C32, _>::new(&cfg, r, &mut VecStorage).unwrap()
    });
}

#[test]
fn empty_input_and_empty_output_make_no_progress_and_no_panic() {
    let r = rate(48_000);
    let mut block = FirBlock::<C32, _>::new(
        &FirConfig {
            cutoff_hz: 5_000.0,
            transition_hz: 2_000.0,
            ..FirConfig::default()
        },
        r,
        &mut VecStorage,
    )
    .unwrap();
    let mut out = [C32::default(); 4];
    assert_eq!(block.process(&[], &mut out).unwrap().consumed, 0);
    assert_eq!(
        block.process(&[C32::default()], &mut []).unwrap().produced,
        0
    );
    let cfg = ResampleConfig {
        tunable: true,
        ..ResampleConfig::default()
    };
    let mut rs = ResampleBlock::<C32, _>::new(&cfg, r, &mut VecStorage).unwrap();
    assert_eq!(rs.process(&[], &mut out).unwrap().produced, 0);
    let p = rs.process(&[C32::default(); 3], &mut []).unwrap();
    assert_eq!(p.produced, 0);
}

#[test]
fn the_tolerance_check_is_bit_exact_for_fixed_point() {
    // A 1-LSB error must fail for Q15/CQ15 even where their bytes would cast to `f32`.
    let a = vec![
        CQ15 {
            re: Q15(1000),
            im: Q15(-2000)
        };
        64
    ];
    let mut b = a.clone();
    b[3] = CQ15 {
        re: Q15(1001),
        im: Q15(-2000),
    };
    assert!(same_or_close(&a, &a));
    assert!(!same_or_close(&a, &b));
    let qa = vec![Q15(1000); 64];
    let mut qb = qa.clone();
    qb[5] = Q15(1001);
    assert!(!same_or_close(&qa, &qb));
    // `f32` within 120 dB still passes.
    let fa = vec![C32::new(0.5, -0.25); 64];
    let mut fb = fa.clone();
    fb[7] = C32::new(0.5 + 1e-7, -0.25);
    assert!(same_or_close(&fa, &fb));
}
