//! What the blocks do to signals: pass bands, stop bands, rates, levels and loops.
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

use std::f64::consts::TAU;

use common::{arg, norm, rate, run_block, run_frames};
use num_complex::Complex64;
use sdrk::{C32, CQ15, ComplexSample, Q15, RealSample, Sdrk};
use sdrk_blocks::{
    Block, BlockReal, BlockSample, DcBlockBlock, DcBlockConfig, DdcBlock, DdcConfig, DucBlock,
    DucConfig, FirBlock, FirConfig, FmDemodBlock, FmDemodConfig, FmModBlock, FmModConfig,
    FrameBlock, ResamplerBlock, ResamplerConfig, ResamplerRealBlock, SpectrumBlock, SpectrumConfig,
    SpectrumSample, SpectrumWindow,
};
use sdrk_core::kernels::{
    decim::{Resample, ResampleReal},
    demod::{DcBlock, QuadDemod},
    filter::Fir,
    fused::{Ddc, Duc, FmDemod},
    modulate::FmMod,
};
use sdrk_testkit::VecStorage;
use sdrk_testkit::meters::{ToneMeter, sinad_db};

fn tone<T: ComplexSample>(n: usize, freq_hz: f64, rate_hz: f64, amp: f64) -> Vec<T> {
    (0..n)
        .map(|i| {
            let p = TAU * freq_hz / rate_hz * i as f64;
            T::from_c64_round(Complex64::new(amp * p.cos(), amp * p.sin()))
        })
        .collect()
}

/// Amplitude of a complex tone at `freq_hz` fitted to the real part.
fn amplitude<T: ComplexSample>(y: &[T], freq_hz: f64, rate_hz: f64) -> f64 {
    let re: Vec<f64> = y.iter().map(|s| s.to_c64().re).collect();
    ToneMeter::measure(&re, rate_hz, freq_hz).amplitude
}

fn fir_pass_and_stop<T: BlockSample>(tolerance: f64)
where
    Sdrk: Fir<T>,
{
    let cfg = FirConfig {
        cutoff_hz: 4_000.0,
        transition_hz: 2_000.0,
        attenuation_db: 60.0,
        max_taps: 1023,
    };
    let r = 48_000.0;
    let mut block = FirBlock::<T, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    let pass = run_block(&mut block, &tone::<T>(4000, 1_000.0, r, 0.5), None);
    let a = amplitude(&pass[1000..], 1_000.0, r);
    assert!((a - 0.5).abs() < tolerance, "pass band amplitude {a}");
    block.reset();
    let stop = run_block(&mut block, &tone::<T>(4000, 9_000.0, r, 0.5), None);
    let a = amplitude(&stop[1000..], 9_000.0, r);
    // 60 dB below 0.5 is 5e-4; the Q15 output has a floor of a few LSB.
    assert!(a < 5.5e-4 + tolerance / 10.0, "stop band amplitude {a}");
}

#[test]
fn the_fir_block_passes_and_stops() {
    fir_pass_and_stop::<C32>(0.005);
    fir_pass_and_stop::<CQ15>(0.005);
}

fn dc_removal<T: BlockSample>()
where
    Sdrk: DcBlock<T>,
{
    let cfg = DcBlockConfig { cutoff_hz: 50.0 };
    let mut block = DcBlockBlock::<T, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    let mut x = tone::<T>(8000, 5_000.0, 48_000.0, 0.25);
    for s in &mut x {
        *s = T::from_c64_round(s.to_c64() + Complex64::new(0.25, -0.125));
    }
    let y = run_block(&mut block, &x, None);
    let mean: Complex64 = y[6000..].iter().map(|s| s.to_c64()).sum::<Complex64>() / 2000.0;
    assert!(norm(mean) < 0.01, "residual DC {mean}");
    assert!((amplitude(&y[6000..], 5_000.0, 48_000.0) - 0.25).abs() < 0.01);
}

#[test]
fn the_dc_blocker_removes_the_offset_and_keeps_the_tone() {
    dc_removal::<C32>();
    dc_removal::<CQ15>();
}

fn updown<T: BlockSample + Default>()
where
    Sdrk: Ddc<T> + Duc<T>,
{
    // 1 kHz at the base rate of 12 kHz goes up to a 9 kHz carrier at 48 kHz and back.
    let base = rate(12_000);
    let up = DucConfig {
        frequency_hz: 9_000.0,
        interpolation: 4,
        ..DucConfig::default()
    };
    let down = DdcConfig {
        frequency_hz: 9_000.0,
        decimation: 4,
        ..DdcConfig::default()
    };
    let mut duc = DucBlock::<T, _>::new(&up, base, &mut VecStorage).unwrap();
    let mut ddc = DdcBlock::<T, _>::new(&down, rate(48_000), &mut VecStorage).unwrap();
    let x = tone::<T>(3000, 1_000.0, 12_000.0, 0.4);
    let carrier = run_block(&mut duc, &x, None);
    assert_eq!(carrier.len(), 12_000);
    // The carrier is a tone at 9 kHz + 1 kHz = 10 kHz.
    assert!((amplitude(&carrier[4000..], 10_000.0, 48_000.0) - 0.4).abs() < 0.01);
    assert!(
        amplitude(&carrier[4000..], 8_000.0, 48_000.0) < 0.002,
        "image rejected"
    );
    let y = run_block(&mut ddc, &carrier, None);
    assert_eq!(y.len(), 3000);
    assert!((amplitude(&y[1000..], 1_000.0, 12_000.0) - 0.4).abs() < 0.01);
}

#[test]
fn a_tone_survives_duc_and_ddc() {
    updown::<C32>();
    updown::<CQ15>();
}

fn resampler_dc_and_tone<T: BlockSample + Default>()
where
    Sdrk: Resample<T>,
{
    for (out_hz, tunable) in [
        (44_100.0, false),
        (44_100.0, true),
        (48_000.0, true),
        (16_000.0, false),
        (96_000.0, true),
    ] {
        let cfg = ResamplerConfig {
            rate_out_hz: out_hz,
            tunable,
            max_taps: 8192,
            ..ResamplerConfig::default()
        };
        let mut block = ResamplerBlock::<T, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
        // DC passes with gain 1.
        let dc = vec![T::from_c64_round(Complex64::new(0.5, -0.25)); 3000];
        let y = run_block(&mut block, &dc, None);
        let last = y[y.len() - 1].to_c64();
        assert!(
            (last.re - 0.5).abs() < 2e-3 && (last.im + 0.25).abs() < 2e-3,
            "{out_hz} {tunable}: {last}"
        );
        // A 1 kHz tone keeps its amplitude and its frequency.
        block.reset();
        let x = tone::<T>(6000, 1_000.0, 48_000.0, 0.5);
        let y = run_block(&mut block, &x, None);
        let settled = &y[y.len() / 3..];
        let a = amplitude(settled, 1_000.0, out_hz);
        assert!((a - 0.5).abs() < 0.01, "{out_hz} {tunable}: amplitude {a}");
        let step: Complex64 = settled
            .windows(2)
            .map(|w| w[1].to_c64() * w[0].to_c64().conj())
            .sum();
        assert!(
            (arg(step) - TAU * 1_000.0 / out_hz).abs() < 1e-3,
            "{out_hz} {tunable}"
        );
    }
}

#[test]
fn complex_resamplers_keep_dc_gain_and_tone() {
    resampler_dc_and_tone::<C32>();
    resampler_dc_and_tone::<CQ15>();
}

fn real_resampler_tone<R: BlockReal + Default>()
where
    Sdrk: ResampleReal<R>,
{
    for (out_hz, tunable) in [
        (44_100.0, false),
        (44_100.0, true),
        (12_000.0, false),
        (8_000.0, true),
    ] {
        let cfg = ResamplerConfig {
            rate_out_hz: out_hz,
            tunable,
            max_taps: 8192,
            ..ResamplerConfig::default()
        };
        let mut block =
            ResamplerRealBlock::<R, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
        let x: Vec<R> = (0..6000)
            .map(|i| R::from_f64_round(0.5 * (TAU * 1_000.0 / 48_000.0 * i as f64).sin()))
            .collect();
        let y = run_block(&mut block, &x, None);
        let v: Vec<f64> = y[y.len() / 3..].iter().map(|s| s.to_f64()).collect();
        let t = ToneMeter::measure(&v, out_hz, 1_000.0);
        assert!(
            (t.amplitude - 0.5).abs() < 0.01,
            "{out_hz} {tunable}: {}",
            t.amplitude
        );
        assert!(
            t.residual_power < 1e-6,
            "{out_hz} {tunable}: residual {}",
            t.residual_power
        );
        // DC
        block.reset();
        let dc = vec![R::from_f64_round(0.25); 3000];
        let y = run_block(&mut block, &dc, None);
        assert!((y[y.len() - 1].to_f64() - 0.25).abs() < 2e-3);
    }
}

#[test]
fn real_resamplers_keep_dc_gain_and_tone() {
    real_resampler_tone::<f32>();
    real_resampler_tone::<Q15>();
}

fn fm_loop<T: BlockSample>(min_sinad: f64, deemphasis_us: f64)
where
    Sdrk: FmMod<T> + FmDemod<T> + QuadDemod<T> + ResampleReal<T::Real>,
    T::Real: Default,
{
    // Audio tone at 48 kHz, FM at 48 kHz (deviation 5 kHz), demodulated at the same rate.
    let audio: Vec<T::Real> = (0..9600)
        .map(|i| T::Real::from_f64_round(0.8 * (TAU * 1_000.0 / 48_000.0 * i as f64).sin()))
        .collect();
    let mut modulator = FmModBlock::<T, _>::new(
        &FmModConfig {
            deviation_hz: 5_000.0,
        },
        rate(48_000),
        &mut VecStorage,
    )
    .unwrap();
    let iq = run_block(&mut modulator, &audio, None);
    let cfg = FmDemodConfig {
        deviation_hz: 5_000.0,
        deemphasis_us,
        audio_decimation: 1,
        ..FmDemodConfig::default()
    };
    let mut demod = FmDemodBlock::<T, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    let out = run_block(&mut demod, &iq, None);
    assert_eq!(out.len(), audio.len());
    let v: Vec<f64> = out[2000..].iter().map(|s| s.to_f64()).collect();
    let sinad = sinad_db(&v, 48_000.0, 1_000.0);
    assert!(sinad > min_sinad, "SINAD {sinad} dB");
    if deemphasis_us == 0.0 {
        // The C32 gain maps the full deviation to 1.0; Q15 reads phase / pi.
        let t = ToneMeter::measure(&v, 48_000.0, 1_000.0);
        let expected = if T::TYPE_IS_FLOAT {
            0.8
        } else {
            0.8 * 5_000.0 / 24_000.0
        };
        assert!(
            (t.amplitude - expected).abs() < 0.01,
            "{} vs {expected}",
            t.amplitude
        );
    }
}

trait IsFloat {
    const TYPE_IS_FLOAT: bool;
}
impl<T: sdrk::Sample> IsFloat for T {
    const TYPE_IS_FLOAT: bool = matches!(T::TYPE, sdrk::SampleType::C32 | sdrk::SampleType::F32);
}

#[test]
fn fm_modulator_and_demodulator_form_a_loop() {
    fm_loop::<C32>(70.0, 0.0);
    fm_loop::<CQ15>(40.0, 0.0);
    // With de-emphasis the 1 kHz tone is attenuated slightly but stays clean.
    fm_loop::<C32>(50.0, 50.0);
}

fn spectrum_tone<T: SpectrumSample>(tolerance_db: f64)
where
    Sdrk: sdrk_core::kernels::spectrum::Window<T>
        + sdrk_core::kernels::fft::Fft<T>
        + sdrk_core::kernels::spectrum::PowerDb<f32>,
{
    let size = 1024usize;
    for window in [
        SpectrumWindow::Hann,
        SpectrumWindow::BlackmanHarris,
        SpectrumWindow::Kaiser,
    ] {
        for bin in [0i64, 100, -37, 511, -512] {
            let amp = 1.0 - 2.0f64.powi(-15);
            let x: Vec<T> = (0..size * 2)
                .map(|i| {
                    let p = TAU * (bin as f64) * i as f64 / size as f64;
                    T::from_c64_round(Complex64::new(amp * p.cos(), amp * p.sin()))
                })
                .collect();
            let cfg = SpectrumConfig {
                fft_size: size,
                window,
                averaging: 2,
                ..SpectrumConfig::default()
            };
            let mut block =
                SpectrumBlock::<T, _>::new(&cfg, rate(1_024_000), &mut VecStorage).unwrap();
            let frames = run_frames(&mut block, &x, None);
            assert_eq!(frames.len(), 1);
            let (frame, meta) = &frames[0];
            assert_eq!((meta.averaged, meta.first_sample, meta.exponent), (2, 0, 0));
            // DC is at index N/2; a tone at bin b sits at N/2 + b (bin -512 wraps to index 0).
            let index = (size as i64 / 2 + bin).rem_euclid(size as i64) as usize;
            let peak = frame
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            assert_eq!(peak.0, index, "{window:?} bin {bin}");
            assert!(
                f64::from(peak.1).abs() < tolerance_db,
                "{window:?} bin {bin}: {} dB",
                peak.1
            );
        }
    }
}

#[test]
fn a_full_scale_tone_reads_zero_dbfs_at_its_bin() {
    spectrum_tone::<C32>(0.05);
    spectrum_tone::<CQ15>(0.3);
}

#[test]
fn spectrum_skips_samples_when_the_interval_is_larger_than_the_window() {
    let size = 64;
    // Window k starts at 100 k; the tone changes frequency between the windows.
    let mut x = vec![C32::default(); 400];
    for (i, s) in x.iter_mut().enumerate() {
        let window = i / 100;
        let bin = [4.0, 10.0, 20.0, 30.0][window.min(3)];
        let p = TAU * bin * i as f64 / size as f64;
        *s = C32::new(p.cos() as f32, p.sin() as f32);
    }
    let cfg = SpectrumConfig {
        fft_size: size,
        interval_samples: Some(100),
        ..SpectrumConfig::default()
    };
    let mut block = SpectrumBlock::<C32, _>::new(&cfg, rate(64_000), &mut VecStorage).unwrap();
    let frames = run_frames(&mut block, &x, None);
    assert_eq!(
        frames.iter().map(|f| f.1.first_sample).collect::<Vec<_>>(),
        [0, 100, 200, 300]
    );
    // The window at 300 ends at 364 (beyond the 400 samples? no: 364 <= 400).
    for (k, bin) in [4usize, 10, 20, 30].into_iter().enumerate() {
        let peak = frames[k]
            .0
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap()
            .0;
        assert_eq!(peak, size / 2 + bin, "window {k}");
    }
}

#[test]
fn spectrum_push_stops_after_a_frame_and_reports_consumption() {
    let cfg = SpectrumConfig {
        fft_size: 16,
        ..SpectrumConfig::default()
    };
    let mut block = SpectrumBlock::<C32, _>::new(&cfg, rate(16_000), &mut VecStorage).unwrap();
    let x = vec![C32::new(0.5, 0.0); 40];
    let mut frame = [0.0f32; 16];
    assert_eq!(block.push(&x[..10], &mut frame).unwrap(), (10, None));
    let (used, meta) = block.push(&x[10..], &mut frame).unwrap();
    assert_eq!(used, 6, "the call stops right after the frame");
    assert_eq!(meta.unwrap().first_sample, 0);
    let (used, meta) = block.push(&x[16..], &mut frame).unwrap();
    assert_eq!(used, 16);
    assert_eq!(meta.unwrap().first_sample, 16);
    // A frame buffer of the wrong size is an error.
    assert!(block.push(&x, &mut [0.0; 8]).is_err());
    assert_eq!(block.frame_len(), 16);
    // DC reads -6 dB (amplitude 0.5) at index N/2 for the rectangular-ish Hann sum.
    assert!(
        (frame[8] - 20.0 * 0.5f32.log10()).abs() < 0.1,
        "{}",
        frame[8]
    );
}

#[test]
fn averaging_lowers_the_noise_variance() {
    let x = common::signal::<C32>(1024 * 16, 77, 0.0);
    let var = |averaging: usize| {
        let cfg = SpectrumConfig {
            fft_size: 1024,
            averaging,
            ..SpectrumConfig::default()
        };
        let mut block =
            SpectrumBlock::<C32, _>::new(&cfg, rate(1_000_000), &mut VecStorage).unwrap();
        let frames = run_frames(&mut block, &x, None);
        let f = &frames[0].0;
        // Bins far from DC contain noise only.
        let noise = &f[100..400];
        let mean = noise.iter().sum::<f32>() / noise.len() as f32;
        noise.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / noise.len() as f32
    };
    assert!(var(16) < var(1) / 4.0, "{} vs {}", var(16), var(1));
}
