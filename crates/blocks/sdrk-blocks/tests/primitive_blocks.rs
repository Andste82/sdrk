//! The primitive blocks `quad_demod`, `deemphasis`, `rotate`, `decimate`, `interpolate`, `mag`
//! and `delay`: equal to the kernel they wrap, equal to the blocks they overlap with, exact
//! storage, timing, validation and the run-time parameter of `rotate`.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::panic,
    clippy::cast_lossless,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::large_stack_arrays,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::multiple_bound_locations,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::float_cmp,
    clippy::cast_possible_truncation
)]

mod common;

use common::{
    check_block, check_block_with, check_exact_storage, rate, real_signal, run_block, same_bits,
    signal, taps_for,
};
use sdrk::{C32, CQ15, Delay, Q15, Rate, Ratio, RatioKind, Sample, Sdrk};
use sdrk_blocks::{
    AmDemodBlock, AmDemodConfig, ArenaStorage, BiquadElem, Block, BlockReal, BlockSample, DdcBlock,
    DdcConfig, DecimateBlock, DecimateConfig, DecimateRealBlock, DeemphasisBlock, DeemphasisConfig,
    DeemphasisRealBlock, DelayBlock, DelayConfig, DelayRealBlock, DucBlock, DucConfig, Error,
    FmStereoDemodBlock, FmStereoDemodConfig, InterpolateBlock, InterpolateConfig,
    InterpolateRealBlock, MagBlock, MagConfig, ParamValue, QuadDemodBlock, QuadDemodConfig,
    RotateBlock, RotateConfig,
};
use sdrk_core::kernels::decim::{FirDecimState, FirDecimate, ResampleReal, ResampleRealState};
use sdrk_core::kernels::demod::{Deemphasis, DeemphasisState, QuadDemod, QuadDemodState};
use sdrk_core::kernels::interp::{
    FirInterpRealState, FirInterpState, FirInterpolate, FirInterpolateReal,
};
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::kernels::spectrum::Mag;
use sdrk_testkit::VecStorage;

type A<'a> = ArenaStorage<'a>;

// ---- quad_demod -------------------------------------------------------------------------

fn quad_demod_equals_kernel<T: BlockSample>(seed: u64)
where
    Sdrk: QuadDemod<T>,
    T::Real: PartialEq + core::fmt::Debug,
{
    let cfg = QuadDemodConfig {
        deviation_hz: 5_000.0,
    };
    let fs = 48_000.0;
    let input = signal::<T>(3000, seed, 0.05);
    let gain = T::demod_gain(cfg.deviation_hz, fs);
    let mut want = vec![T::Real::ZERO; input.len()];
    let mut state = QuadDemodState::<T>::new();
    <Sdrk as QuadDemod<T>>::quad_demod(&Sdrk, &mut state, &input, &mut want, gain).unwrap();
    check_block(
        &want,
        &input,
        seed,
        || QuadDemodBlock::<T, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap(),
        "quad_demod",
    );
}

#[test]
fn quad_demod_equals_the_kernel() {
    quad_demod_equals_kernel::<C32>(455);
    quad_demod_equals_kernel::<CQ15>(455);
}

// ---- deemphasis -------------------------------------------------------------------------

fn deemphasis_equals_kernel<R: BlockReal>(seed: u64)
where
    Sdrk: Deemphasis<R>,
    R: PartialEq + core::fmt::Debug,
{
    let cfg = DeemphasisConfig { tau_us: 75.0 };
    let input = real_signal::<R>(3000, seed, 0.03);
    let mut want = vec![R::ZERO; input.len()];
    let mut state = DeemphasisState::<R>::new(75e-6, rate(48_000)).unwrap();
    <Sdrk as Deemphasis<R>>::deemphasis(&Sdrk, &mut state, &input, &mut want).unwrap();
    check_block_with(
        &want,
        &input,
        seed,
        || DeemphasisRealBlock::<R, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap(),
        "deemphasis",
        R::TYPE != sdrk::SampleType::F32,
    );
}

#[test]
fn deemphasis_equals_the_kernel() {
    deemphasis_equals_kernel::<f32>(456);
    deemphasis_equals_kernel::<Q15>(456);
}

/// The complex `deemphasis` (stereo audio) equals the de-emphasis inside `fm_stereo_demod` bit
/// for bit: the block after the decoder with its de-emphasis off gives the samples of the
/// decoder with it on.
fn complex_deemphasis_equals_the_stereo_decoder<T: BlockSample + BiquadElem>(seed: u64)
where
    T::Real: BiquadElem,
    Sdrk: sdrk_core::kernels::biquad::Biquad<T>
        + sdrk_core::kernels::biquad::BiquadReal<T::Real>
        + sdrk_core::kernels::demod::QuadDemod<T>
        + sdrk_core::kernels::stereo::StereoPll<T::Real>
        + sdrk_core::kernels::stereo::StereoMatrix<T>
        + sdrk_core::kernels::decim::FirDecimate<T>,
{
    let fs = rate(480_000);
    let with = FmStereoDemodConfig::default();
    let without = FmStereoDemodConfig {
        deemphasis_us: 0.0,
        ..with
    };
    let input = signal::<T>(24_000, seed, 0.05);
    let decode = |cfg: &FmStereoDemodConfig| {
        let mut block = FmStereoDemodBlock::<T, _>::new(cfg, fs, &mut VecStorage).unwrap();
        run_block(&mut block, &input, None)
    };
    let want = decode(&with);
    let plain = decode(&without);
    assert!(
        want.len() > 2000 && want != plain,
        "the filter does something"
    );
    let cfg = DeemphasisConfig { tau_us: 50.0 };
    let audio = rate(48_000);
    check_block_with(
        &want,
        &plain,
        seed,
        || DeemphasisBlock::<T, _>::new(&cfg, audio, &mut VecStorage).unwrap(),
        "complex deemphasis",
        true,
    );
}

#[test]
fn the_complex_deemphasis_equals_the_stereo_decoder() {
    complex_deemphasis_equals_the_stereo_decoder::<C32>(457);
    complex_deemphasis_equals_the_stereo_decoder::<CQ15>(457);
}

#[test]
fn the_complex_deemphasis_filters_both_channels_alike() {
    // Left and right with the same samples give the same filtered samples; a step settles.
    let cfg = DeemphasisConfig { tau_us: 50.0 };
    let mut block = DeemphasisBlock::<C32, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    let input: Vec<C32> = (0..500)
        .map(|i| C32::new(0.25 * (i % 7) as f32, 0.25 * (i % 7) as f32))
        .collect();
    let out = run_block(&mut block, &input, None);
    assert!(out.iter().all(|s| s.re == s.im));
    let step = run_block(&mut block, &[C32::new(1.0, -1.0); 500], None);
    assert!((step[499].re - 1.0).abs() < 1e-3 && (step[499].im + 1.0).abs() < 1e-3);
    // Reset clears the state; a bad time constant is refused.
    block.reset();
    assert!(
        DeemphasisBlock::<C32, _>::new(
            &DeemphasisConfig { tau_us: 0.0 },
            rate(48_000),
            &mut VecStorage
        )
        .is_err()
    );
    assert!(matches!(
        block.set_param("tau_us", ParamValue::F64(75.0)),
        Err(Error::ReadOnlyParameter)
    ));
}

#[test]
fn the_complex_deemphasis_storage_is_exact() {
    let cfg = DeemphasisConfig::default();
    let bytes =
        DeemphasisBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate(48_000)).unwrap();
    let mut mem = vec![0u8; bytes + 16];
    let mut arena = ArenaStorage::new(&mut mem);
    DeemphasisBlock::<C32, _>::new(&cfg, rate(48_000), &mut arena).unwrap();
    assert!(arena.remaining() <= 16, "{} bytes left", arena.remaining());
}

// ---- rotate -----------------------------------------------------------------------------

fn rotate_equals_kernel<T: BlockSample>(seed: u64)
where
    Sdrk: Rotate<T>,
    T: PartialEq + core::fmt::Debug,
{
    let cfg = RotateConfig {
        frequency_hz: -7_300.0,
    };
    let input = signal::<T>(3000, seed, 0.1);
    let mut want = vec![T::ZERO; input.len()];
    let mut nco = Nco::new(-7_300.0, rate(48_000)).unwrap();
    <Sdrk as Rotate<T>>::rotate(&Sdrk, &mut nco, &input, &mut want).unwrap();
    check_block(
        &want,
        &input,
        seed,
        || RotateBlock::<T, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap(),
        "rotate",
    );
}

#[test]
fn rotate_equals_the_kernel() {
    rotate_equals_kernel::<C32>(457);
    rotate_equals_kernel::<CQ15>(457);
}

#[test]
fn rotate_changes_its_frequency_with_continuous_phase() {
    let fs = 48_000;
    let input = signal::<C32>(400, 1, 0.1);
    let mut block = RotateBlock::<C32, _>::new(
        &RotateConfig {
            frequency_hz: 1_000.0,
        },
        rate(fs),
        &mut VecStorage,
    )
    .unwrap();
    let mut got = vec![C32::default(); 400];
    block.process(&input[..200], &mut got[..200]).unwrap();
    block
        .set_param("frequency_hz", ParamValue::F64(-3_000.0))
        .unwrap();
    assert_eq!(block.param("frequency_hz"), Ok(ParamValue::F64(-3_000.0)));
    block.process(&input[200..], &mut got[200..]).unwrap();
    // The kernel with the same change at the same sample.
    let mut want = vec![C32::default(); 400];
    let mut nco = Nco::new(1_000.0, rate(fs)).unwrap();
    <Sdrk as Rotate<C32>>::rotate(&Sdrk, &mut nco, &input[..200], &mut want[..200]).unwrap();
    nco.set_frequency(-3_000.0, rate(fs)).unwrap();
    <Sdrk as Rotate<C32>>::rotate(&Sdrk, &mut nco, &input[200..], &mut want[200..]).unwrap();
    same_bits(&want, &got, "rotate with a frequency change");
    // Errors of the run-time parameter.
    assert!(matches!(
        block.set_param("frequency_hz", ParamValue::F64(24_000.0)),
        Err(Error::OutOfRange { .. })
    ));
    assert_eq!(
        block.set_param("frequency_hz", ParamValue::I64(1)),
        Err(Error::WrongParamType)
    );
    assert_eq!(
        block.set_param("nope", ParamValue::F64(1.0)),
        Err(Error::UnknownParameter)
    );
}

// ---- decimate ---------------------------------------------------------------------------

fn decimate_equals_kernel<T: BlockSample>(seed: u64)
where
    Sdrk: FirDecimate<T>,
    T: PartialEq + core::fmt::Debug,
{
    let cfg = DecimateConfig {
        factor: 5,
        ..DecimateConfig::default()
    };
    let r = rate(240_000);
    let input = signal::<T>(4000, seed, 0.01);
    let taps = taps_for::<T>(&cfg.design_spec(r), 240_000.0, 1.0);
    let hist = vec![T::ZERO; taps.len() - 1];
    let mut state = FirDecimState::<T, _, _>::new(taps, hist, 5).unwrap();
    let mut want = vec![T::ZERO; input.len() / 5 + 2];
    let p = <Sdrk as FirDecimate<T>>::fir_decimate(&Sdrk, &mut state, &input, &mut want).unwrap();
    want.truncate(p.produced);
    check_block(
        &want,
        &input,
        seed,
        || DecimateBlock::<T, _>::new(&cfg, r, &mut VecStorage).unwrap(),
        "decimate",
    );
}

fn decimate_real_equals_kernel<R: BlockReal>(seed: u64)
where
    Sdrk: ResampleReal<R>,
    R: PartialEq + core::fmt::Debug,
{
    let cfg = DecimateConfig {
        factor: 5,
        ..DecimateConfig::default()
    };
    let r = rate(240_000);
    let input = real_signal::<R>(4000, seed, 0.01);
    let taps = taps_for::<R>(&cfg.design_spec(r), 240_000.0, 1.0);
    let hist = vec![R::ZERO; taps.len()];
    let mut state = ResampleRealState::<R, _, _>::new(taps, hist, 1, 5).unwrap();
    let mut want = vec![R::ZERO; input.len() / 5 + 2];
    let p = <Sdrk as ResampleReal<R>>::resample_real(&Sdrk, &mut state, &input, &mut want).unwrap();
    want.truncate(p.produced);
    check_block(
        &want,
        &input,
        seed,
        || DecimateRealBlock::<R, _>::new(&cfg, r, &mut VecStorage).unwrap(),
        "decimate real",
    );
}

#[test]
fn decimate_equals_the_kernel() {
    decimate_equals_kernel::<C32>(458);
    decimate_equals_kernel::<CQ15>(458);
    decimate_real_equals_kernel::<f32>(458);
    decimate_real_equals_kernel::<Q15>(458);
}

// ---- interpolate ------------------------------------------------------------------------

fn interpolate_equals_kernel<T: BlockSample>(seed: u64)
where
    Sdrk: FirInterpolate<T>,
    T: PartialEq + core::fmt::Debug,
{
    let cfg = InterpolateConfig {
        factor: 4,
        ..InterpolateConfig::default()
    };
    let r = rate(24_000);
    let input = signal::<T>(1000, seed, 0.02);
    let taps = taps_for::<T>(&cfg.design_spec(r), 96_000.0, 4.0);
    let hist = vec![T::ZERO; taps.len().div_ceil(4)];
    let mut state = FirInterpState::<T, _, _>::new(taps, hist, 4).unwrap();
    let mut want = vec![T::ZERO; input.len() * 4];
    let p =
        <Sdrk as FirInterpolate<T>>::fir_interpolate(&Sdrk, &mut state, &input, &mut want).unwrap();
    assert_eq!(p.consumed, input.len());
    want.truncate(p.produced);
    check_block(
        &want,
        &input,
        seed,
        || InterpolateBlock::<T, _>::new(&cfg, r, &mut VecStorage).unwrap(),
        "interpolate",
    );
}

fn interpolate_real_equals_kernel<R: BlockReal>(seed: u64)
where
    Sdrk: FirInterpolateReal<R>,
    R: PartialEq + core::fmt::Debug,
{
    let cfg = InterpolateConfig {
        factor: 4,
        ..InterpolateConfig::default()
    };
    let r = rate(24_000);
    let input = real_signal::<R>(1000, seed, 0.02);
    let taps = taps_for::<R>(&cfg.design_spec(r), 96_000.0, 4.0);
    let hist = vec![R::ZERO; taps.len().div_ceil(4)];
    let mut state = FirInterpRealState::<R, _, _>::new(taps, hist, 4).unwrap();
    let mut want = vec![R::ZERO; input.len() * 4];
    let p =
        <Sdrk as FirInterpolateReal<R>>::fir_interpolate_real(&Sdrk, &mut state, &input, &mut want)
            .unwrap();
    assert_eq!(p.consumed, input.len());
    want.truncate(p.produced);
    check_block(
        &want,
        &input,
        seed,
        || InterpolateRealBlock::<R, _>::new(&cfg, r, &mut VecStorage).unwrap(),
        "interpolate real",
    );
}

#[test]
fn interpolate_equals_the_kernel() {
    interpolate_equals_kernel::<C32>(459);
    interpolate_equals_kernel::<CQ15>(459);
    interpolate_real_equals_kernel::<f32>(459);
    interpolate_real_equals_kernel::<Q15>(459);
}

// ---- mag --------------------------------------------------------------------------------

fn mag_equals_kernel<T: BlockSample>(seed: u64)
where
    Sdrk: Mag<T>,
    T::Real: PartialEq + core::fmt::Debug,
{
    let input = signal::<T>(3000, seed, 0.05);
    let mut want = vec![T::Real::ZERO; input.len()];
    <Sdrk as Mag<T>>::mag(&Sdrk, &input, &mut want).unwrap();
    check_block(
        &want,
        &input,
        seed,
        || MagBlock::<T, _>::new(&MagConfig, rate(48_000), &mut VecStorage).unwrap(),
        "mag",
    );
}

#[test]
fn mag_equals_the_kernel() {
    mag_equals_kernel::<C32>(460);
    mag_equals_kernel::<CQ15>(460);
}

// ---- delay ------------------------------------------------------------------------------

fn delay_shifts<T: BlockSample + PartialEq + core::fmt::Debug>(seed: u64) {
    for samples in [0usize, 1, 7, 64, 1000] {
        let cfg = DelayConfig { samples };
        let input = signal::<T>(2500, seed, 0.03);
        let mut want = vec![T::ZERO; samples];
        want.extend_from_slice(&input[..input.len() - samples]);
        check_block(
            &want,
            &input,
            seed,
            || DelayBlock::<T, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap(),
            "delay",
        );
    }
}

fn delay_real_shifts<R: BlockReal + PartialEq + core::fmt::Debug>(seed: u64) {
    for samples in [0usize, 1, 7, 64, 1000] {
        let cfg = DelayConfig { samples };
        let input = real_signal::<R>(2500, seed, 0.03);
        let mut want = vec![R::ZERO; samples];
        want.extend_from_slice(&input[..input.len() - samples]);
        check_block(
            &want,
            &input,
            seed,
            || DelayRealBlock::<R, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap(),
            "delay real",
        );
    }
}

#[test]
fn delay_outputs_the_input_shifted_by_samples() {
    delay_shifts::<C32>(461);
    delay_shifts::<CQ15>(461);
    delay_real_shifts::<f32>(461);
    delay_real_shifts::<Q15>(461);
}

#[test]
fn delay_reset_clears_the_ring_and_timing_reports_the_delay() {
    let cfg = DelayConfig { samples: 3 };
    let mut block = DelayRealBlock::<f32, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    let mut out = [0.0f32; 5];
    block.process(&[1.0, 2.0, 3.0, 4.0, 5.0], &mut out).unwrap();
    assert_eq!(out, [0.0, 0.0, 0.0, 1.0, 2.0]);
    block.reset();
    block.process(&[9.0, 9.0, 9.0, 9.0, 9.0], &mut out).unwrap();
    assert_eq!(out, [0.0, 0.0, 0.0, 9.0, 9.0]);
    let t = block.timing();
    assert_eq!((t.ratio, t.kind), (Ratio::ONE, RatioKind::Constant));
    assert_eq!(t.delay.reduced(), Delay::new(3, 1).unwrap());
    assert_eq!(block.param("samples"), Ok(ParamValue::I64(3)));
    assert_eq!(
        block.set_param("samples", ParamValue::I64(4)),
        Err(Error::ReadOnlyParameter)
    );
}

// ---- overlap with existing blocks ---------------------------------------------------------

fn rotate_decimate_equals_ddc<T: BlockSample + PartialEq + core::fmt::Debug>()
where
    Sdrk: Rotate<T> + FirDecimate<T> + sdrk_core::kernels::fused::Ddc<T>,
{
    let r = rate(480_000);
    let input = signal::<T>(6000, 3, 0.1);
    let ddc = DdcConfig {
        frequency_hz: 60_000.0,
        decimation: 10,
        ..DdcConfig::default()
    };
    let want = run_block(
        &mut DdcBlock::<T, _>::new(&ddc, r, &mut VecStorage).unwrap(),
        &input,
        None,
    );
    let rotate = RotateConfig {
        frequency_hz: -60_000.0,
    };
    let decimate = DecimateConfig {
        factor: 10,
        ..DecimateConfig::default()
    };
    let mut rot = RotateBlock::<T, _>::new(&rotate, r, &mut VecStorage).unwrap();
    let mut dec = DecimateBlock::<T, _>::new(&decimate, r, &mut VecStorage).unwrap();
    let shifted = run_block(&mut rot, &input, None);
    let got = run_block(&mut dec, &shifted, None);
    same_bits(&want, &got, "rotate -> decimate vs ddc");
}

#[test]
fn rotate_then_decimate_equals_ddc() {
    rotate_decimate_equals_ddc::<C32>();
    rotate_decimate_equals_ddc::<CQ15>();
}

fn interpolate_rotate_equals_duc<T: BlockSample + PartialEq + core::fmt::Debug>()
where
    Sdrk: Rotate<T> + FirInterpolate<T> + sdrk_core::kernels::fused::Duc<T>,
{
    let r = rate(24_000);
    let r_out = rate(240_000);
    let input = signal::<T>(800, 4, 0.03);
    let duc = DucConfig {
        frequency_hz: 30_000.0,
        interpolation: 10,
        ..DucConfig::default()
    };
    let want = run_block(
        &mut DucBlock::<T, _>::new(&duc, r, &mut VecStorage).unwrap(),
        &input,
        None,
    );
    let interpolate = InterpolateConfig {
        factor: 10,
        ..InterpolateConfig::default()
    };
    let rotate = RotateConfig {
        frequency_hz: 30_000.0,
    };
    let mut int = InterpolateBlock::<T, _>::new(&interpolate, r, &mut VecStorage).unwrap();
    let mut rot = RotateBlock::<T, _>::new(&rotate, r_out, &mut VecStorage).unwrap();
    let up = run_block(&mut int, &input, None);
    let got = run_block(&mut rot, &up, None);
    same_bits(&want, &got, "interpolate -> rotate vs duc");
}

#[test]
fn interpolate_then_rotate_equals_duc() {
    interpolate_rotate_equals_duc::<C32>();
    interpolate_rotate_equals_duc::<CQ15>();
}

fn mag_equals_am_demod_stage<T: BlockSample>()
where
    Sdrk: Mag<T> + sdrk_core::kernels::demod::DcBlockReal<T::Real> + ResampleReal<T::Real>,
    T::Real: PartialEq + core::fmt::Debug,
{
    let cfg = AmDemodConfig {
        dc_cutoff_hz: 0.0,
        audio_decimation: 1,
        ..AmDemodConfig::default()
    };
    let input = signal::<T>(3000, 5, 0.02);
    let want = run_block(
        &mut AmDemodBlock::<T, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap(),
        &input,
        None,
    );
    let got = run_block(
        &mut MagBlock::<T, _>::new(&MagConfig, rate(48_000), &mut VecStorage).unwrap(),
        &input,
        None,
    );
    same_bits(&want, &got, "mag vs am_demod");
}

#[test]
fn mag_equals_the_first_stage_of_am_demod() {
    mag_equals_am_demod_stage::<C32>();
    mag_equals_am_demod_stage::<CQ15>();
}

// ---- storage, timing, validation ----------------------------------------------------------

fn exact_storage<T: BlockSample, R: BlockReal>() {
    let r: Rate = rate(240_000);
    let q = QuadDemodConfig::default();
    let b = QuadDemodBlock::<T, A<'_>>::storage_bytes(&q, r).unwrap();
    assert_eq!(b, 0);
    check_exact_storage(b, |a| QuadDemodBlock::<T, _>::new(&q, r, a).map(|_| ()));
    let d = DeemphasisConfig::default();
    let b = DeemphasisRealBlock::<R, A<'_>>::storage_bytes(&d, r).unwrap();
    check_exact_storage(b, |a| {
        DeemphasisRealBlock::<R, _>::new(&d, r, a).map(|_| ())
    });
    let c = RotateConfig {
        frequency_hz: 1_000.0,
    };
    let b = RotateBlock::<T, A<'_>>::storage_bytes(&c, r).unwrap();
    check_exact_storage(b, |a| RotateBlock::<T, _>::new(&c, r, a).map(|_| ()));
    let c = DecimateConfig {
        factor: 5,
        ..DecimateConfig::default()
    };
    let b = DecimateBlock::<T, A<'_>>::storage_bytes(&c, r).unwrap();
    assert!(b > 0);
    check_exact_storage(b, |a| DecimateBlock::<T, _>::new(&c, r, a).map(|_| ()));
    let b = DecimateRealBlock::<R, A<'_>>::storage_bytes(&c, r).unwrap();
    check_exact_storage(b, |a| DecimateRealBlock::<R, _>::new(&c, r, a).map(|_| ()));
    let c = InterpolateConfig {
        factor: 5,
        ..InterpolateConfig::default()
    };
    let b = InterpolateBlock::<T, A<'_>>::storage_bytes(&c, r).unwrap();
    assert!(b > 0);
    check_exact_storage(b, |a| InterpolateBlock::<T, _>::new(&c, r, a).map(|_| ()));
    let b = InterpolateRealBlock::<R, A<'_>>::storage_bytes(&c, r).unwrap();
    check_exact_storage(b, |a| {
        InterpolateRealBlock::<R, _>::new(&c, r, a).map(|_| ())
    });
    let b = MagBlock::<T, A<'_>>::storage_bytes(&MagConfig, r).unwrap();
    assert_eq!(b, 0);
    check_exact_storage(b, |a| MagBlock::<T, _>::new(&MagConfig, r, a).map(|_| ()));
    let c = DelayConfig { samples: 123 };
    let b = DelayBlock::<T, A<'_>>::storage_bytes(&c, r).unwrap();
    check_exact_storage(b, |a| DelayBlock::<T, _>::new(&c, r, a).map(|_| ()));
    let b = DelayRealBlock::<R, A<'_>>::storage_bytes(&c, r).unwrap();
    check_exact_storage(b, |a| DelayRealBlock::<R, _>::new(&c, r, a).map(|_| ()));
}

#[test]
fn storage_is_exact() {
    exact_storage::<C32, f32>();
    exact_storage::<CQ15, Q15>();
}

#[test]
fn blocks_stay_small_and_are_send_and_sync() {
    fn check<T: Send + Sync>() -> usize {
        core::mem::size_of::<T>()
    }
    type S = VecStorage;
    let sizes = [
        check::<QuadDemodBlock<C32, S>>(),
        check::<DeemphasisRealBlock<f32, S>>(),
        check::<RotateBlock<C32, S>>(),
        check::<DecimateBlock<C32, S>>(),
        check::<DecimateRealBlock<f32, S>>(),
        check::<InterpolateBlock<C32, S>>(),
        check::<InterpolateRealBlock<f32, S>>(),
        check::<MagBlock<C32, S>>(),
        check::<DelayBlock<C32, S>>(),
        check::<DelayRealBlock<f32, S>>(),
        check::<QuadDemodConfig>(),
        check::<DeemphasisConfig>(),
        check::<RotateConfig>(),
        check::<DecimateConfig>(),
        check::<InterpolateConfig>(),
        check::<MagConfig>(),
        check::<DelayConfig>(),
    ];
    for s in sizes {
        assert!(s <= 512, "a block is {s} bytes");
    }
}

#[test]
fn timing_and_rates() {
    let r = rate(240_000);
    let dec = DecimateConfig {
        factor: 5,
        ..DecimateConfig::default()
    };
    let b = DecimateBlock::<C32, _>::new(&dec, r, &mut VecStorage).unwrap();
    let t = b.timing();
    assert_eq!(
        (t.ratio, t.kind),
        (Ratio::new(1, 5).unwrap(), RatioKind::Constant)
    );
    assert_eq!(
        t.delay.reduced(),
        Delay::new(b.num_taps() as u64 - 1, 2).unwrap().reduced()
    );
    assert_eq!(
        DecimateBlock::<C32, A<'_>>::rate_out(&dec, r).unwrap(),
        rate(48_000)
    );
    let rb = DecimateRealBlock::<f32, _>::new(&dec, r, &mut VecStorage).unwrap();
    assert_eq!(rb.timing(), b.timing());
    let int = InterpolateConfig {
        factor: 5,
        ..InterpolateConfig::default()
    };
    let b = InterpolateBlock::<C32, _>::new(&int, r, &mut VecStorage).unwrap();
    assert_eq!(b.timing().ratio, Ratio::new(5, 1).unwrap());
    assert_eq!(
        InterpolateBlock::<C32, A<'_>>::rate_out(&int, r).unwrap(),
        rate(1_200_000)
    );
    // The discriminator looks at one previous sample; the IIR stage and the mixer report 0.
    let q = QuadDemodBlock::<C32, _>::new(&QuadDemodConfig::default(), r, &mut VecStorage).unwrap();
    assert_eq!(q.timing().delay.reduced(), Delay::new(1, 2).unwrap());
    let d = DeemphasisRealBlock::<f32, _>::new(&DeemphasisConfig::default(), r, &mut VecStorage)
        .unwrap();
    assert_eq!(d.timing().delay, Delay::ZERO);
    let m = MagBlock::<C32, _>::new(&MagConfig, r, &mut VecStorage).unwrap();
    assert_eq!(m.timing().delay, Delay::ZERO);
}

#[test]
fn validation_errors() {
    let r = rate(240_000);
    // decimate: the stop band must stay below the output Nyquist frequency.
    let ok = DecimateConfig {
        factor: 10,
        ..DecimateConfig::default()
    };
    assert!(ok.validate(r).is_ok());
    for bad in [
        DecimateConfig {
            cutoff_hz: Some(11_500.0),
            transition_hz: Some(2_000.0),
            ..ok
        },
        DecimateConfig { factor: 0, ..ok },
        DecimateConfig { factor: 1025, ..ok },
        DecimateConfig { max_taps: 3, ..ok },
    ] {
        assert!(
            matches!(bad.validate(r), Err(Error::OutOfRange { .. })),
            "{bad:?}"
        );
        assert!(DecimateBlock::<C32, _>::new(&bad, r, &mut VecStorage).is_err());
    }
    // interpolate: the stop band must stay below half the input rate.
    let ok = InterpolateConfig {
        factor: 4,
        ..InterpolateConfig::default()
    };
    assert!(ok.validate(r).is_ok());
    for bad in [
        InterpolateConfig {
            cutoff_hz: Some(118_000.0),
            transition_hz: Some(10_000.0),
            ..ok
        },
        InterpolateConfig { factor: 0, ..ok },
        InterpolateConfig { factor: 1025, ..ok },
    ] {
        assert!(
            matches!(bad.validate(r), Err(Error::OutOfRange { .. })),
            "{bad:?}"
        );
        assert!(InterpolateBlock::<C32, _>::new(&bad, r, &mut VecStorage).is_err());
    }
    // rotate: |f| < fs / 2.
    for f in [120_000.0, -120_000.0, 130_000.0, f64::NAN] {
        let cfg = RotateConfig { frequency_hz: f };
        assert!(
            matches!(cfg.validate(r), Err(Error::OutOfRange { .. })),
            "{f}"
        );
    }
    assert!(
        RotateConfig {
            frequency_hz: 119_999.0
        }
        .validate(r)
        .is_ok()
    );
    // deemphasis, quad_demod, delay.
    for tau in [0.0, 0.5, 10_001.0, f64::NAN] {
        assert!(
            DeemphasisConfig { tau_us: tau }.validate(r).is_err(),
            "{tau}"
        );
    }
    for dev in [0.0, -1.0, f64::NAN] {
        assert!(QuadDemodConfig { deviation_hz: dev }.validate(r).is_err());
    }
    assert!(DelayConfig { samples: 65_536 }.validate(r).is_ok());
    assert!(DelayConfig { samples: 65_537 }.validate(r).is_err());
    // Parameters by name.
    let mut d = DelayConfig::default();
    d.set("samples", ParamValue::I64(9)).unwrap();
    assert_eq!(d.get("samples"), Ok(ParamValue::I64(9)));
    assert!(matches!(
        d.set("samples", ParamValue::I64(-1)),
        Err(Error::OutOfRange { .. })
    ));
    assert_eq!(d.set("x", ParamValue::I64(1)), Err(Error::UnknownParameter));
    let mut m = MagConfig;
    assert_eq!(m.set("x", ParamValue::I64(1)), Err(Error::UnknownParameter));
    // A decimate block reports the resolved defaults.
    let b = DecimateBlock::<C32, _>::new(&DecimateConfig::default(), r, &mut VecStorage).unwrap();
    assert_eq!(b.param("cutoff_hz"), Ok(ParamValue::F64(9_600.0)));
    assert_eq!(b.param("transition_hz"), Ok(ParamValue::F64(2_400.0)));
    let mut b = b;
    assert_eq!(
        b.set_param("factor", ParamValue::I64(2)),
        Err(Error::ReadOnlyParameter)
    );
}
