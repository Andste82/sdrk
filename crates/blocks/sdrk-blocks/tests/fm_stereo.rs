//! Block `fm_stereo`: stereo decoding of a generated multiplex signal, the runtime parameter
//! `stereo`, `pilot_locked()`, validation, the equality with the hand-wired kernel chain,
//! invariance against the block size and extreme inputs.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::panic,
    clippy::cast_lossless,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::large_stack_arrays,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::multiple_bound_locations
)]

mod common;

use common::{plans, rate, run_block};
use num_complex::Complex64;
use sdrk::{C32, CQ15, ComplexSample, Ratio, RatioKind, Sdrk};
use sdrk_blocks::design::{BiquadResponse, biquad_coefs};
use sdrk_blocks::{
    BiquadElem, Block, BlockSample, Error, FmStereoBlock, FmStereoConfig, ParamValue, TunedElem,
};
use sdrk_core::RealSample;
use sdrk_core::kernels::biquad::{Biquad, BiquadReal, BiquadRealState, BiquadState};
use sdrk_core::kernels::decim::{FirDecimState, FirDecimate};
use sdrk_core::kernels::demod::{QuadDemod, QuadDemodState};
use sdrk_core::kernels::stereo::{StereoMatrix, StereoPll, StereoPllState};
use sdrk_testkit::fm::{fm_multitone_baseband, stereo_mpx_tones};
use sdrk_testkit::meters::MultitoneAccumulator;
use sdrk_testkit::{ChunkPlan, VecStorage};

const FS: u64 = 480_000;

/// A stereo FM baseband at 480 kHz (L 1 kHz, R 3 kHz, pilot `pilot`), amplitude 0.9.
fn mpx_signal<T: ComplexSample>(n: usize, pilot: f64, right: (f64, f64)) -> Vec<T> {
    let tones = stereo_mpx_tones((1.0, 1000.0), right, pilot);
    let mut buf = vec![Complex64::new(0.0, 0.0); n];
    fm_multitone_baseband(&mut buf, FS as f64, 0.0, 75_000.0, &tones, 0).unwrap();
    buf.iter()
        .map(|c| T::from_c64_round(*c * 0.9))
        .collect::<Vec<T>>()
}

fn block<T>(cfg: &FmStereoConfig) -> FmStereoBlock<T, VecStorage>
where
    T: BlockSample + BiquadElem,
    T::Real: BiquadElem,
{
    FmStereoBlock::<T, _>::new(cfg, rate(FS), &mut VecStorage).unwrap()
}

/// Separation in dB of the 1 kHz tone in L over R and of the 3 kHz tone in R over L, measured
/// from `skip` audio samples on.
fn separation<T: ComplexSample>(audio: &[T], skip: usize) -> (f64, f64) {
    let mut l = MultitoneAccumulator::new(48_000.0, &[1000.0, 3000.0]);
    let mut r = l;
    for v in &audio[skip..] {
        l.push(v.re().to_f64());
        r.push(v.im().to_f64());
    }
    let (al, ar) = (l.amplitudes(), r.amplitudes());
    (
        20.0 * (al[0] / ar[0]).log10(),
        20.0 * (ar[1] / al[1]).log10(),
    )
}

#[test]
fn it_decodes_stereo_and_reports_the_pilot() {
    let n = 96_000;
    let input = mpx_signal::<C32>(n, 0.1, (1.0, 3000.0));
    let mut b = block::<C32>(&FmStereoConfig::default());
    assert!(!b.pilot_locked());
    let out = run_block(&mut b, &input, None);
    assert_eq!(out.len(), n / 10);
    assert!(b.pilot_locked());
    let (sl, sr) = separation(&out, 4800);
    assert!(sl > 40.0 && sr > 40.0, "{sl} {sr}");
}

#[test]
fn the_q15_block_decodes_stereo_too() {
    let n = 96_000;
    let input = mpx_signal::<CQ15>(n, 0.1, (1.0, 3000.0));
    let mut b = block::<CQ15>(&FmStereoConfig::default());
    let out = run_block(&mut b, &input, None);
    assert!(b.pilot_locked());
    let (sl, sr) = separation(&out, 4800);
    assert!(sl > 35.0 && sr > 35.0, "{sl} {sr}");
}

#[test]
fn without_a_pilot_the_output_is_mono_and_the_loop_is_not_locked() {
    let input = mpx_signal::<C32>(48_000, 0.0, (1.0, 1000.0));
    let mut b = block::<C32>(&FmStereoConfig::default());
    let out = run_block(&mut b, &input, None);
    assert!(!b.pilot_locked());
    assert!(out.iter().all(|v| v.re.to_bits() == v.im.to_bits()));
}

#[test]
fn the_stereo_parameter_switches_between_auto_and_mono() {
    let n = 96_000;
    let input = mpx_signal::<C32>(n, 0.1, (1.0, 3000.0));
    let mut b = block::<C32>(&FmStereoConfig::default());
    assert_eq!(b.param("stereo").unwrap(), ParamValue::Choice(0));
    let first = run_block(&mut b, &input[..48_000], None);
    assert!(first[2400..].iter().any(|v| v.re != v.im));
    b.set_param("stereo", ParamValue::Choice(1)).unwrap();
    assert_eq!(b.param("stereo").unwrap(), ParamValue::Choice(1));
    let mono = run_block(&mut b, &input[48_000..], None);
    // The filter and the de-emphasis still hold the stereo signal for a few samples; after
    // 5 ms left and right are bit-identical.
    assert!(mono[240..].iter().all(|v| v.re.to_bits() == v.im.to_bits()));
    // The loop keeps running: the pilot is still reported.
    assert!(b.pilot_locked());
    // And back to stereo from the next call, without a new lock time.
    b.set_param("stereo", ParamValue::Choice(0)).unwrap();
    let again = run_block(&mut b, &input[..4800], None);
    assert!(again[240..].iter().any(|v| v.re != v.im));
    // A build parameter cannot change; unknown names and wrong kinds are errors.
    assert!(matches!(
        b.set_param("audio_decimation", ParamValue::I64(5)),
        Err(Error::ReadOnlyParameter)
    ));
    assert!(matches!(
        b.set_param("nope", ParamValue::I64(5)),
        Err(Error::UnknownParameter)
    ));
    assert!(b.set_param("stereo", ParamValue::Choice(2)).is_err());
}

#[test]
fn a_block_built_in_mono_is_mono_from_the_first_sample() {
    let mut cfg = FmStereoConfig::default();
    cfg.set("stereo", ParamValue::Choice(1)).unwrap();
    let input = mpx_signal::<C32>(24_000, 0.1, (1.0, 3000.0));
    let mut b = block::<C32>(&cfg);
    let out = run_block(&mut b, &input, None);
    assert!(out.iter().all(|v| v.re.to_bits() == v.im.to_bits()));
}

#[test]
fn the_configuration_is_validated() {
    let ok = FmStereoConfig::default();
    assert!(FmStereoBlock::<C32, VecStorage>::rate_out(&ok, rate(FS)).is_ok());
    assert_eq!(
        FmStereoBlock::<C32, VecStorage>::rate_out(&ok, rate(FS))
            .unwrap()
            .as_f64(),
        48_000.0
    );
    // Input rate below 120 kHz.
    assert!(matches!(
        ok.validate(rate(96_000)),
        Err(Error::OutOfRange { .. })
    ));
    assert!(ok.validate(rate(120_000)).is_err()); // 12 kHz output is too low for 15 kHz
    let at_min = FmStereoConfig {
        audio_decimation: 2,
        ..ok
    };
    assert!(at_min.validate(rate(120_000)).is_ok());
    // Cutoff above 16 kHz.
    let mut cfg = ok;
    assert!(
        cfg.set("audio_cutoff_hz", ParamValue::F64(16_001.0))
            .is_err()
    );
    assert!(cfg.set("pll_natural_hz", ParamValue::F64(191.0)).is_err());
    assert!(cfg.set("pll_natural_hz", ParamValue::F64(0.5)).is_err());
    assert!(cfg.set("audio_decimation", ParamValue::I64(65)).is_err());
    // An output rate too low for the cutoff.
    let low = FmStereoConfig {
        audio_decimation: 20,
        ..ok
    };
    assert!(matches!(
        low.validate(rate(FS)),
        Err(Error::OutOfRange { .. })
    ));
    let nan = FmStereoConfig {
        deviation_hz: f64::NAN,
        ..ok
    };
    assert!(nan.validate(rate(FS)).is_err());
}

#[test]
fn timing_is_the_decimation_and_the_filter_delay() {
    let b = block::<C32>(&FmStereoConfig::default());
    let t = b.timing();
    assert_eq!(t.ratio, Ratio::new(1, 10).unwrap());
    assert_eq!(t.kind, RatioKind::Constant);
    let taps = b.num_taps() as u64;
    assert_eq!(
        t.delay,
        sdrk::Delay::new(taps - 1, 2)
            .unwrap()
            .add(sdrk::Delay::new(1, 2).unwrap())
            .unwrap()
    );
    assert_eq!(b.min_input(), 10);
}

/// The chain of the milestone wired by hand from the kernels.
fn hand_chain<T>(cfg: &FmStereoConfig, input: &[T]) -> Vec<T>
where
    T: BlockSample + BiquadElem,
    T::Real: BiquadElem,
    Sdrk: QuadDemod<T>
        + BiquadReal<T::Real>
        + StereoPll<T::Real>
        + FirDecimate<T>
        + StereoMatrix<T>
        + Biquad<T>,
{
    let fs = FS as f64;
    let mut quad = QuadDemodState::<T>::new();
    let mut mpx = vec![T::Real::default(); input.len()];
    Sdrk.quad_demod(
        &mut quad,
        input,
        &mut mpx,
        T::demod_gain(cfg.deviation_hz, fs),
    )
    .unwrap();
    let bp = biquad_coefs(BiquadResponse::Bandpass, 19_000.0, 5.0, fs).unwrap();
    let mut coefs = vec![Default::default(); 1];
    <T::Real as BiquadElem>::convert(&[bp], &mut coefs).unwrap();
    let mut bp_state =
        BiquadRealState::<T::Real, _, _>::new(coefs, vec![T::Real::default(); 4]).unwrap();
    let mut pilot = vec![T::Real::default(); input.len()];
    Sdrk.biquad_real(&mut bp_state, &mpx, &mut pilot).unwrap();
    let level = if <T as BiquadElem>::Q15 {
        0.1 * 2.0 * cfg.deviation_hz / fs
    } else {
        0.1
    };
    let mut pll =
        StereoPllState::<T::Real>::new(19_000.0, cfg.pll_natural_hz, level, rate(FS)).unwrap();
    let mut split = vec![T::new(T::Real::default(), T::Real::default()); input.len()];
    Sdrk.stereo_pll(&mut pll, &mpx, &pilot, &mut split).unwrap();
    let spec = sdrk_blocks::DesignSpec {
        cutoff_hz: cfg.audio_cutoff_hz,
        transition_hz: cfg.audio_cutoff_hz / 4.0,
        attenuation_db: 60.0,
    };
    let n = spec.num_taps(fs, 1023).unwrap();
    let mut design = vec![0.0f64; n];
    spec.design(fs, 1023, &mut design).unwrap();
    let taps: Vec<T::Real> = design
        .iter()
        .map(|v| <T as TunedElem>::tap(*v).unwrap())
        .collect();
    let hist = vec![T::new(T::Real::default(), T::Real::default()); taps.len() - 1];
    let mut fir = FirDecimState::new(taps, hist, cfg.audio_decimation).unwrap();
    let mut dec = vec![
        T::new(T::Real::default(), T::Real::default());
        input.len() / cfg.audio_decimation + 2
    ];
    let p = Sdrk.fir_decimate(&mut fir, &split, &mut dec).unwrap();
    assert_eq!(p.consumed, input.len());
    dec.truncate(p.produced);
    let mut lr = vec![T::new(T::Real::default(), T::Real::default()); dec.len()];
    Sdrk.stereo_matrix(&dec, &mut lr).unwrap();
    if cfg.deemphasis_us > 0.0 {
        let alpha =
            1.0 - (-1.0 / (fs / cfg.audio_decimation as f64 * cfg.deemphasis_us * 1e-6)).exp();
        let mut coefs = vec![Default::default(); 1];
        <T as BiquadElem>::convert(&[[alpha, 0.0, 0.0, -(1.0 - alpha), 0.0]], &mut coefs).unwrap();
        let mut st = BiquadState::<T, _, _>::new(
            coefs,
            vec![T::new(T::Real::default(), T::Real::default()); 4],
        )
        .unwrap();
        let mut out = vec![T::new(T::Real::default(), T::Real::default()); lr.len()];
        Sdrk.biquad(&mut st, &lr, &mut out).unwrap();
        out
    } else {
        lr
    }
}

fn equals_hand_chain<T>()
where
    T: BlockSample + BiquadElem,
    T::Real: BiquadElem,
    Sdrk: QuadDemod<T>
        + BiquadReal<T::Real>
        + StereoPll<T::Real>
        + FirDecimate<T>
        + StereoMatrix<T>
        + Biquad<T>,
{
    let input = mpx_signal::<T>(24_000, 0.1, (1.0, 3000.0));
    for cfg in [
        FmStereoConfig::default(),
        FmStereoConfig {
            deemphasis_us: 0.0,
            audio_decimation: 5,
            ..FmStereoConfig::default()
        },
    ] {
        let want = hand_chain::<T>(&cfg, &input);
        let mut b = block::<T>(&cfg);
        let got = run_block(&mut b, &input, None);
        assert_eq!(got.len(), want.len());
        assert!(
            got.iter().zip(&want).all(|(a, b)| a.re().to_f64().to_bits()
                == b.re().to_f64().to_bits()
                && a.im().to_f64().to_bits() == b.im().to_f64().to_bits()),
            "block differs from the hand-wired chain"
        );
    }
}

#[test]
fn the_block_equals_the_hand_wired_kernel_chain() {
    equals_hand_chain::<C32>();
    equals_hand_chain::<CQ15>();
}

fn invariance<T>()
where
    T: BlockSample + BiquadElem,
    T::Real: BiquadElem,
    Sdrk: QuadDemod<T>
        + BiquadReal<T::Real>
        + StereoPll<T::Real>
        + FirDecimate<T>
        + StereoMatrix<T>
        + Biquad<T>,
{
    let input = mpx_signal::<T>(16_000, 0.1, (1.0, 3000.0));
    let cfg = FmStereoConfig::default();
    let whole = run_block(&mut block::<T>(&cfg), &input, None);
    assert!(whole.len() > 100);
    let mut max = 1usize;
    for seed in 0..20u64 {
        // Piece sizes from 1 sample to a few hundred (the block itself works in 256).
        let plan = ChunkPlan::new(7000 + seed, max);
        max = max * 3 / 2 + 1;
        let got = run_block(&mut block::<T>(&cfg), &input, Some(plan));
        assert_eq!(got.len(), whole.len(), "plan {seed}");
        assert!(
            got.iter()
                .zip(&whole)
                .all(
                    |(a, b)| a.re().to_f64().to_bits() == b.re().to_f64().to_bits()
                        && a.im().to_f64().to_bits() == b.im().to_f64().to_bits()
                ),
            "output depends on the block size (plan {seed})"
        );
    }
    let _ = plans(1);
}

#[test]
fn the_output_does_not_depend_on_the_block_size() {
    invariance::<C32>();
    invariance::<CQ15>();
}

#[test]
fn reset_returns_to_the_start() {
    let input = mpx_signal::<C32>(24_000, 0.1, (1.0, 3000.0));
    let cfg = FmStereoConfig::default();
    let mut b = block::<C32>(&cfg);
    let first = run_block(&mut b, &input, None);
    assert!(b.pilot_locked());
    b.reset();
    assert!(!b.pilot_locked());
    let second = run_block(&mut b, &input, None);
    assert_eq!(first, second);
    // reset keeps the runtime parameter
    b.set_param("stereo", ParamValue::Choice(1)).unwrap();
    b.reset();
    assert_eq!(b.param("stereo").unwrap(), ParamValue::Choice(1));
}

#[test]
fn extreme_inputs_do_not_panic() {
    let cfg = FmStereoConfig::default();
    let mut b = block::<C32>(&cfg);
    let mut out = vec![C32::default(); 4096];
    let zeros = vec![C32::new(0.0, 0.0); 2000];
    let full: Vec<C32> = (0..2000)
        .map(|i| {
            if i % 2 == 0 {
                C32::new(1.0, 1.0)
            } else {
                C32::new(-1.0, -1.0)
            }
        })
        .collect();
    let bad: Vec<C32> = (0..2000)
        .map(|i| match i % 4 {
            0 => C32::new(f32::NAN, 0.0),
            1 => C32::new(f32::INFINITY, f32::NEG_INFINITY),
            2 => C32::new(0.0, f32::NAN),
            _ => C32::new(1.0, -1.0),
        })
        .collect();
    for input in [&zeros, &full, &bad, &zeros] {
        let p = b.process(input, &mut out).unwrap();
        assert_eq!(p.consumed, input.len());
    }
    // Q15 corners.
    let mut q = block::<CQ15>(&cfg);
    let mut outq = vec![CQ15::default(); 4096];
    let corners: Vec<CQ15> = (0..2000)
        .map(|i| match i % 3 {
            0 => CQ15::new(sdrk::Q15(-32768), sdrk::Q15(-32768)),
            1 => CQ15::new(sdrk::Q15(32767), sdrk::Q15(-32768)),
            _ => CQ15::new(sdrk::Q15(0), sdrk::Q15(32767)),
        })
        .collect();
    let p = q.process(&corners, &mut outq).unwrap();
    assert_eq!(p.consumed, corners.len());
}

#[test]
fn the_block_builds_over_the_range_of_input_rates_in_both_types() {
    // The audio filter (transition 3750 Hz, 60 dB) needs about 1850 taps at 1.92 MS/s, more than
    // the 1023 the block allows (the rule of `fm_demod`): a wide input needs a `ddc` in front.
    for (fs, decimation, builds) in [
        (120_000u64, 2usize, true),
        (192_000, 4, true),
        (240_000, 5, true),
        (480_000, 10, true),
        (960_000, 20, true),
        (1_920_000, 40, false),
        (2_400_000, 50, false),
    ] {
        let cfg = FmStereoConfig {
            audio_decimation: decimation,
            ..FmStereoConfig::default()
        };
        let r = rate(fs);
        let c32 = FmStereoBlock::<C32, VecStorage>::new(&cfg, r, &mut VecStorage);
        let q15 = FmStereoBlock::<CQ15, VecStorage>::new(&cfg, r, &mut VecStorage);
        assert_eq!(c32.is_ok(), builds, "C32 at {fs} Hz");
        assert_eq!(q15.is_ok(), builds, "CQ15 at {fs} Hz");
        if !builds {
            assert!(matches!(c32, Err(Error::OutOfRange { .. })));
        }
    }
}

#[test]
fn extreme_deviations_are_rejected_before_the_build_in_q15() {
    // The CQ15 pilot level is 0.1 * 2 * deviation / fs and must lie in (0, 1]; C32 does not
    // use the deviation for the loop at all.
    let r = rate(FS);
    for deviation_hz in [2.5e6, 1e9, 1e11] {
        let cfg = FmStereoConfig {
            deviation_hz,
            ..FmStereoConfig::default()
        };
        assert!(
            cfg.validate(r).is_ok(),
            "{deviation_hz}: the config is fine for C32"
        );
        assert!(FmStereoBlock::<C32, VecStorage>::rate_out(&cfg, r).is_ok());
        assert!(FmStereoBlock::<C32, VecStorage>::storage_bytes(&cfg, r).is_ok());
        assert!(FmStereoBlock::<C32, VecStorage>::new(&cfg, r, &mut VecStorage).is_ok());
        assert!(
            matches!(
                FmStereoBlock::<CQ15, VecStorage>::rate_out(&cfg, r),
                Err(Error::OutOfRange {
                    name: "deviation_hz",
                    ..
                })
            ),
            "{deviation_hz}"
        );
        assert!(FmStereoBlock::<CQ15, VecStorage>::storage_bytes(&cfg, r).is_err());
        assert!(FmStereoBlock::<CQ15, VecStorage>::new(&cfg, r, &mut VecStorage).is_err());
    }
    // Whatever `new` rejects, `rate_out` and `storage_bytes` reject, too (small deviations).
    for deviation_hz in [1.0, 2.0, 10.0, 100.0, 75_000.0, 2.4e6] {
        let cfg = FmStereoConfig {
            deviation_hz,
            ..FmStereoConfig::default()
        };
        let built = FmStereoBlock::<CQ15, VecStorage>::new(&cfg, r, &mut VecStorage).is_ok();
        assert_eq!(
            FmStereoBlock::<CQ15, VecStorage>::rate_out(&cfg, r).is_ok(),
            built,
            "{deviation_hz}"
        );
        assert_eq!(
            FmStereoBlock::<CQ15, VecStorage>::storage_bytes(&cfg, r).is_ok(),
            built,
            "{deviation_hz}"
        );
    }
}
