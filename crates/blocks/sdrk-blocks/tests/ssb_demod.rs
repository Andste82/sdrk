//! Block `ssb_demod`: the chain of kernels, tone level and frequency in both sidebands, the
//! runtime sideband switch, validation, invariance, reset and timing.
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

use common::{check_exact_storage, rate, run_block};
use num_complex::Complex64;
use sdrk::{C32, CQ15, ComplexSample, Ratio, RatioKind, RealSample, Sdrk};
use sdrk_blocks::{
    ArenaStorage, Block, BlockSample, Error, ParamValue, SsbDemodBlock, SsbDemodConfig,
    SsbSideband, TunedElem,
};
use sdrk_core::kernels::decim::{FirDecimState, FirDecimate};
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_testkit::r#gen::{Sideband, ssb_tone};
use sdrk_testkit::meters::{ToneMeter, fit_frequency};
use sdrk_testkit::{ChunkPlan, Pcg32, VecStorage};

const FS: u64 = 48_000;

fn config(sideband: SsbSideband, audio_decimation: usize) -> SsbDemodConfig {
    SsbDemodConfig {
        sideband,
        audio_decimation,
        ..SsbDemodConfig::default()
    }
}

fn new_block<T: BlockSample>(cfg: &SsbDemodConfig) -> SsbDemodBlock<T, VecStorage> {
    SsbDemodBlock::<T, _>::new(cfg, rate(FS), &mut VecStorage).unwrap()
}

/// Converts f32 samples to the sample type.
fn convert<T: ComplexSample>(src: &[C32]) -> Vec<T> {
    src.iter()
        .map(|s| T::from_c64_round(Complex64::new(f64::from(s.re), f64::from(s.im))))
        .collect()
}

/// A tone `audio_hz` away from the carrier on the given side, amplitude `amp`.
fn tone<T: ComplexSample>(n: usize, audio_hz: f64, side: Sideband, amp: f64, start: u64) -> Vec<T> {
    let mut x = vec![C32::default(); n];
    ssb_tone(&mut x, rate(FS), audio_hz, side, amp, start);
    convert(&x)
}

/// The wanted tone on one side plus a tone of 1.5 kHz on the other, and a little noise.
fn two_sided<T: ComplexSample>(n: usize, wanted: Sideband) -> Vec<T> {
    let other = match wanted {
        Sideband::Usb => Sideband::Lsb,
        Sideband::Lsb => Sideband::Usb,
    };
    let a = tone::<C32>(n, 1000.0, wanted, 0.3, 0);
    let b = tone::<C32>(n, 1500.0, other, 0.3, 0);
    let mut rng = Pcg32::new(21, 5);
    let sum: Vec<C32> = a
        .iter()
        .zip(&b)
        .map(|(p, q)| {
            C32::new(
                p.re + q.re + 1e-3 * rng.normal() as f32,
                p.im + q.im + 1e-3 * rng.normal() as f32,
            )
        })
        .collect();
    convert(&sum)
}

fn as_f64<R: RealSample>(v: &[R]) -> Vec<f64> {
    v.iter().map(|s| s.to_f64()).collect()
}

fn sideband_of(side: Sideband) -> SsbSideband {
    match side {
        Sideband::Usb => SsbSideband::Usb,
        Sideband::Lsb => SsbSideband::Lsb,
    }
}

/// The chain written out with the kernels: `rotate`, `fir_decimate`, `rotate`, real part.
fn chain<T: BlockSample>(cfg: &SsbDemodConfig, input: &[T]) -> Vec<T::Real>
where
    Sdrk: Rotate<T> + FirDecimate<T>,
{
    let fs = FS as f64;
    let dec = cfg.audio_decimation;
    let spec = cfg.design_spec();
    let mut design = vec![0.0f64; spec.num_taps(fs, cfg.max_taps).unwrap()];
    spec.design(fs, cfg.max_taps, &mut design).unwrap();
    let taps: Vec<T::Real> = design
        .iter()
        .map(|v| <T as TunedElem>::tap(*v).unwrap())
        .collect();
    let hist = vec![T::ZERO; taps.len() - 1];
    let mut fir = FirDecimState::<T, _, _>::new(taps, hist, dec).unwrap();
    let centre = f64::midpoint(cfg.low_hz, cfg.high_hz);
    let s = if cfg.sideband == SsbSideband::Usb {
        1.0
    } else {
        -1.0
    };
    let mut nco_in = Nco::new(-s * centre, rate(FS)).unwrap();
    let mut nco_out = Nco::new(s * centre, rate(FS / dec as u64)).unwrap();
    let mut shifted = vec![T::ZERO; input.len()];
    <Sdrk as Rotate<T>>::rotate(&Sdrk, &mut nco_in, input, &mut shifted).unwrap();
    let mut narrow = vec![T::ZERO; input.len() / dec + 2];
    let p = <Sdrk as FirDecimate<T>>::fir_decimate(&Sdrk, &mut fir, &shifted, &mut narrow).unwrap();
    assert_eq!(p.consumed, input.len());
    narrow.truncate(p.produced);
    let mut back = vec![T::ZERO; narrow.len()];
    <Sdrk as Rotate<T>>::rotate(&Sdrk, &mut nco_out, &narrow, &mut back).unwrap();
    back.iter().map(|v| v.re()).collect()
}

fn matches_the_chain<T: BlockSample>()
where
    Sdrk: Rotate<T> + FirDecimate<T>,
    T::Real: PartialEq + core::fmt::Debug + Default,
{
    for (side, dec) in [
        (SsbSideband::Usb, 1),
        (SsbSideband::Lsb, 1),
        (SsbSideband::Usb, 3),
        (SsbSideband::Lsb, 2),
        (SsbSideband::Lsb, 4),
    ] {
        let cfg = config(side, dec);
        let input = two_sided::<T>(5000, Sideband::Usb);
        let want = chain::<T>(&cfg, &input);
        let got = run_block(&mut new_block::<T>(&cfg), &input, None);
        assert_eq!(got.len(), want.len(), "{side:?}, decimation {dec}");
        assert_eq!(got, want, "{side:?}, decimation {dec}");
    }
}

#[test]
fn the_block_equals_the_kernel_chain() {
    matches_the_chain::<C32>();
    matches_the_chain::<CQ15>();
}

fn tone_level_and_frequency<T: BlockSample>(amp_tolerance_db: f64)
where
    Sdrk: Rotate<T> + FirDecimate<T>,
    T::Real: Default,
{
    let amp = 0.5;
    for side in [Sideband::Usb, Sideband::Lsb] {
        for f in [400.0, 1000.0, 2500.0] {
            let cfg = config(sideband_of(side), 1);
            let input = tone::<T>(24_000, f, side, amp, 0);
            let out = as_f64(&run_block(&mut new_block::<T>(&cfg), &input, None));
            let tail = &out[out.len() / 2..];
            let m = ToneMeter::measure(tail, FS as f64, f);
            let db = 20.0 * (m.amplitude / amp).log10();
            assert!(db.abs() <= amp_tolerance_db, "{side:?} {f} Hz: {db} dB");
            let fit = fit_frequency(tail, FS as f64, f, 5.0);
            assert!((fit - f).abs() <= 0.1, "{side:?} {f} Hz: found {fit} Hz");
            // The audio is the tone itself: what is left after the fit is far below it.
            assert!(m.residual_power < m.amplitude * m.amplitude / 2.0 * 1e-3);
        }
    }
}

#[test]
fn a_tone_at_the_carrier_plus_f_gives_audio_at_f_with_the_input_amplitude() {
    tone_level_and_frequency::<C32>(0.5);
    tone_level_and_frequency::<CQ15>(0.5);
}

#[test]
fn the_opposite_sideband_is_rejected() {
    let cfg = config(SsbSideband::Usb, 1);
    for (wanted_side, sb) in [
        (Sideband::Usb, SsbSideband::Usb),
        (Sideband::Lsb, SsbSideband::Lsb),
    ] {
        let cfg = SsbDemodConfig {
            sideband: sb,
            ..cfg
        };
        let input = two_sided::<C32>(24_000, wanted_side);
        let out = as_f64(&run_block(&mut new_block::<C32>(&cfg), &input, None));
        let tail = &out[out.len() / 2..];
        let wanted = ToneMeter::measure(tail, FS as f64, 1000.0).amplitude;
        let other = ToneMeter::measure(tail, FS as f64, 1500.0).amplitude;
        assert!((wanted - 0.3).abs() < 0.01, "{sb:?}: wanted {wanted}");
        assert!(
            20.0 * (wanted / other).log10() > 50.0,
            "{sb:?}: opposite sideband only {} dB down",
            20.0 * (wanted / other).log10()
        );
    }
}

#[test]
fn a_mistuning_shifts_the_audio_by_the_same_amount() {
    // The carrier is 100 Hz too high: the audio tone moves by 100 Hz.
    let cfg = config(SsbSideband::Usb, 1);
    let input = tone::<C32>(24_000, 1100.0, Sideband::Usb, 0.4, 0);
    let out = as_f64(&run_block(&mut new_block::<C32>(&cfg), &input, None));
    let fit = fit_frequency(&out[12_000..], FS as f64, 1100.0, 5.0);
    assert!((fit - 1100.0).abs() < 0.1, "{fit}");
}

#[test]
fn the_sideband_switch_passes_the_lower_sideband_from_the_next_call() {
    let n = 12_000;
    let usb_first = two_sided::<C32>(2 * n, Sideband::Usb);
    let mut block = new_block::<C32>(&config(SsbSideband::Usb, 1));
    let mut out = run_block(&mut block, &usb_first[..n], None);
    block
        .set_param("sideband", ParamValue::Choice(SsbSideband::Lsb.index()))
        .unwrap();
    assert_eq!(block.param("sideband"), Ok(ParamValue::Choice(1)));
    out.extend(run_block(&mut block, &usb_first[n..], None));
    assert_eq!(out.len(), 2 * n);
    assert!(out.iter().all(|v| v.is_finite()), "NaN or infinity");

    // Before the switch the 1 kHz tone (USB) is audible, afterwards the 1.5 kHz tone (LSB).
    // 5760 samples hold whole cycles of both tones, so neither leaks into the other's fit.
    let before = as_f64(&out[4000..4000 + 5760]);
    let after = as_f64(&out[n + 4000..n + 4000 + 5760]);
    let (b1, b2) = (
        ToneMeter::measure(&before, FS as f64, 1000.0).amplitude,
        ToneMeter::measure(&before, FS as f64, 1500.0).amplitude,
    );
    let (a1, a2) = (
        ToneMeter::measure(&after, FS as f64, 1000.0).amplitude,
        ToneMeter::measure(&after, FS as f64, 1500.0).amplitude,
    );
    assert!(b1 > 0.29 && b2 < 0.001, "before: {b1}, {b2}");
    assert!(a2 > 0.29 && a1 < 0.001, "after: {a1}, {a2}");

    // The output is continuous: no jump larger than the amplitude of the signal (0.6 at most).
    let worst = out
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0f32, f32::max);
    assert!(worst <= 0.6, "jump of {worst}");

    // The same in Q15.
    let input = two_sided::<CQ15>(2 * n, Sideband::Usb);
    let mut block = new_block::<CQ15>(&config(SsbSideband::Usb, 1));
    let mut out = run_block(&mut block, &input[..n], None);
    block.set_param("sideband", ParamValue::Choice(1)).unwrap();
    out.extend(run_block(&mut block, &input[n..], None));
    let after = as_f64(&out[n + 4000..n + 4000 + 5760]);
    let a2 = ToneMeter::measure(&after, FS as f64, 1500.0).amplitude;
    let a1 = ToneMeter::measure(&after, FS as f64, 1000.0).amplitude;
    assert!(a2 > 0.29 && a1 < 0.002, "Q15 after: {a1}, {a2}");
}

#[test]
fn the_switch_keeps_both_phases_continuous() {
    // Switching and switching back at once leaves the output bit-identical to never switching.
    let input = two_sided::<C32>(3000, Sideband::Usb);
    let cfg = config(SsbSideband::Usb, 2);
    let want = run_block(&mut new_block::<C32>(&cfg), &input, None);
    let mut block = new_block::<C32>(&cfg);
    let mut got = run_block(&mut block, &input[..1000], None);
    block.set_param("sideband", ParamValue::Choice(1)).unwrap();
    block.set_param("sideband", ParamValue::Choice(0)).unwrap();
    got.extend(run_block(&mut block, &input[1000..], None));
    assert_eq!(got, want);
    // A real switch changes the output from the next call on, not before.
    let mut block = new_block::<C32>(&cfg);
    let mut got = run_block(&mut block, &input[..1000], None);
    assert_eq!(got, want[..got.len()]);
    block.set_param("sideband", ParamValue::Choice(1)).unwrap();
    got.extend(run_block(&mut block, &input[1000..], None));
    assert_ne!(got, want);
}

fn twenty_plans() -> Vec<ChunkPlan> {
    (0..20u64)
        .map(|k| ChunkPlan::new(900 + k, 1 + ((k * 53) % 600) as usize))
        .collect()
}

fn invariant_and_reset<T: BlockSample>()
where
    Sdrk: Rotate<T> + FirDecimate<T>,
    T::Real: PartialEq + core::fmt::Debug + Default,
{
    for (side, dec) in [
        (SsbSideband::Usb, 1),
        (SsbSideband::Lsb, 1),
        (SsbSideband::Lsb, 6),
    ] {
        let cfg = config(side, dec);
        let input = two_sided::<T>(6000, Sideband::Usb);
        let whole = run_block(&mut new_block::<T>(&cfg), &input, None);
        for plan in twenty_plans() {
            let chunked = run_block(&mut new_block::<T>(&cfg), &input, Some(plan));
            assert_eq!(chunked, whole, "{side:?}, decimation {dec}, {plan:?}");
        }
        // After reset the block behaves like a new one (both oscillators at phase 0).
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
fn a_small_output_buffer_never_overruns_and_keeps_the_phase() {
    for dec in [1usize, 4] {
        let cfg = config(SsbSideband::Lsb, dec);
        let input = two_sided::<C32>(3000, Sideband::Lsb);
        let whole = run_block(&mut new_block::<C32>(&cfg), &input, None);
        // Output buffers of 3 samples: the block takes only the input they can hold.
        let mut block = new_block::<C32>(&cfg);
        let mut got = Vec::new();
        let mut pos = 0;
        let mut out = [0.0f32; 3];
        while pos < input.len() {
            let p = block.process(&input[pos..], &mut out).unwrap();
            assert!(p.produced <= 3);
            assert!(p.consumed <= 3 * dec + dec);
            got.extend_from_slice(&out[..p.produced]);
            pos += p.consumed;
            assert!(p.consumed > 0 || p.produced > 0, "no progress");
        }
        assert_eq!(got, whole, "decimation {dec}");
    }
}

#[test]
fn parameter_errors() {
    let mut block = new_block::<C32>(&config(SsbSideband::Usb, 1));
    for name in ["low_hz", "high_hz", "transition_hz", "attenuation_db"] {
        assert_eq!(
            block.set_param(name, ParamValue::F64(1.0)),
            Err(Error::ReadOnlyParameter)
        );
    }
    for name in ["max_taps", "audio_decimation"] {
        assert_eq!(
            block.set_param(name, ParamValue::I64(2)),
            Err(Error::ReadOnlyParameter)
        );
    }
    assert_eq!(
        block.set_param("nope", ParamValue::F64(1.0)),
        Err(Error::UnknownParameter)
    );
    assert!(matches!(
        block.set_param("sideband", ParamValue::Choice(2)),
        Err(Error::OutOfRange { .. })
    ));
    assert_eq!(
        block.set_param("sideband", ParamValue::F64(1.0)),
        Err(Error::WrongParamType)
    );
    assert_eq!(block.param("nope"), Err(Error::UnknownParameter));
    assert_eq!(block.param("sideband"), Ok(ParamValue::Choice(0)));
    assert_eq!(block.param("low_hz"), Ok(ParamValue::F64(300.0)));
    assert_eq!(block.param("high_hz"), Ok(ParamValue::F64(3000.0)));
    assert_eq!(block.param("transition_hz"), Ok(ParamValue::F64(200.0)));
    assert_eq!(block.param("attenuation_db"), Ok(ParamValue::F64(60.0)));
    assert_eq!(block.param("max_taps"), Ok(ParamValue::I64(1023)));
    assert_eq!(block.param("audio_decimation"), Ok(ParamValue::I64(1)));
}

#[test]
fn construction_is_validated() {
    let r = rate(FS);
    let base = SsbDemodConfig::default();
    let bad = [
        // low_hz at or above high_hz, or negative, or NaN.
        SsbDemodConfig {
            low_hz: 3000.0,
            ..base
        },
        SsbDemodConfig {
            low_hz: 3500.0,
            ..base
        },
        SsbDemodConfig {
            low_hz: -1.0,
            ..base
        },
        SsbDemodConfig {
            low_hz: f64::NAN,
            ..base
        },
        // high_hz + transition_hz / 2 above half the output rate (24 kHz; 12 kHz at /4).
        SsbDemodConfig {
            high_hz: 23_950.0,
            ..base
        },
        SsbDemodConfig {
            high_hz: 11_950.0,
            audio_decimation: 2,
            ..base
        },
        SsbDemodConfig {
            audio_decimation: 8,
            high_hz: 3_000.0,
            transition_hz: 6_000.0,
            ..base
        },
        SsbDemodConfig {
            audio_decimation: 0,
            ..base
        },
        SsbDemodConfig {
            audio_decimation: 65,
            ..base
        },
        // The filter would need more than max_taps.
        SsbDemodConfig {
            max_taps: 100,
            ..base
        },
    ];
    for cfg in bad {
        assert!(cfg.validate(r).is_err(), "{cfg:?}");
        assert!(SsbDemodBlock::<C32, _>::new(&cfg, r, &mut VecStorage).is_err());
        assert!(SsbDemodBlock::<CQ15, VecStorage>::storage_bytes(&cfg, r).is_err());
        assert!(SsbDemodBlock::<CQ15, VecStorage>::rate_out(&cfg, r).is_err());
    }
    // The error kind of the band checks.
    assert!(matches!(
        SsbDemodConfig {
            low_hz: 3000.0,
            ..base
        }
        .validate(r),
        Err(Error::OutOfRange { name: "low_hz", .. })
    ));
    assert!(matches!(
        SsbDemodConfig {
            high_hz: 23_950.0,
            ..base
        }
        .validate(r),
        Err(Error::OutOfRange {
            name: "high_hz",
            ..
        })
    ));
    // Exactly at the limit is valid: 24 kHz - 100 Hz.
    assert!(
        SsbDemodConfig {
            high_hz: 23_900.0,
            ..base
        }
        .validate(r)
        .is_ok()
    );
    assert!(base.validate(r).is_ok());
    assert!(
        SsbDemodConfig {
            low_hz: 0.0,
            ..base
        }
        .validate(r)
        .is_ok()
    );
}

#[test]
fn rate_storage_and_timing() {
    let r = rate(FS);
    let cfg = config(SsbSideband::Usb, 4);
    assert_eq!(
        SsbDemodBlock::<C32, VecStorage>::rate_out(&cfg, r),
        Ok(rate(12_000))
    );
    assert_eq!(
        SsbDemodBlock::<C32, VecStorage>::rate_out(&config(SsbSideband::Usb, 1), r),
        Ok(r)
    );
    let taps = cfg.num_taps(r).unwrap();
    assert!(taps > 3 && taps % 2 == 1);

    // The arena holds exactly the taps, the history and the two work buffers.
    for dec in [1, 4] {
        let cfg = config(SsbSideband::Lsb, dec);
        let n = cfg.num_taps(r).unwrap();
        let bytes = SsbDemodBlock::<C32, VecStorage>::storage_bytes(&cfg, r).unwrap();
        assert_eq!(bytes, 4 * n + 8 * (n - 1) + 2 * 256 * 8);
        check_exact_storage(bytes, |a: &mut ArenaStorage<'_>| {
            SsbDemodBlock::<C32, _>::new(&cfg, r, a).map(|_| ())
        });
        let bytes = SsbDemodBlock::<CQ15, VecStorage>::storage_bytes(&cfg, r).unwrap();
        check_exact_storage(bytes, |a: &mut ArenaStorage<'_>| {
            SsbDemodBlock::<CQ15, _>::new(&cfg, r, a).map(|_| ())
        });
    }

    let block = new_block::<CQ15>(&cfg);
    let t = block.timing();
    assert_eq!(t.ratio, Ratio::new(1, 4).unwrap());
    assert_eq!(t.kind, RatioKind::Constant);
    assert_eq!(
        t.delay,
        sdrk::Delay::new(taps as u64 - 1, 2).unwrap(),
        "(N - 1) / 2 input samples"
    );
    assert_eq!(block.min_input(), 4);
    assert_eq!(block.num_taps(), taps);
    assert_eq!(block.description().kind, "ssb_demod");
    assert_eq!(
        new_block::<C32>(&config(SsbSideband::Usb, 1))
            .timing()
            .ratio,
        Ratio::ONE
    );
}

#[test]
fn silence_and_extreme_input() {
    let cfg = config(SsbSideband::Usb, 1);
    // Silence gives exact silence.
    let out = run_block(
        &mut new_block::<C32>(&cfg),
        &vec![C32::new(0.0, 0.0); 2000],
        None,
    );
    assert!(out.iter().all(|v| *v == 0.0));
    let out = run_block(
        &mut new_block::<CQ15>(&cfg),
        &vec![CQ15::default(); 2000],
        None,
    );
    assert!(out.iter().all(|v| v.0 == 0));
    // Full scale in Q15: the constant input is outside the band and never panics.
    let full = vec![
        CQ15 {
            re: sdrk::Q15(32767),
            im: sdrk::Q15(-32768),
        };
        3000
    ];
    let out = run_block(&mut new_block::<CQ15>(&cfg), &full, None);
    assert_eq!(out.len(), 3000);
    // A tone at 0.99 of full scale keeps its level and shape: a wrapping instead of a saturating
    // rotation or filter output would flip the sign of peaks and spoil the fit (rule R7).
    for side in [Sideband::Usb, Sideband::Lsb] {
        let cfg = config(sideband_of(side), 1);
        let loud = tone::<CQ15>(12_000, 1000.0, side, 0.99, 0);
        let out = as_f64(&run_block(&mut new_block::<CQ15>(&cfg), &loud, None));
        let tail = &out[out.len() / 2..];
        let m = ToneMeter::measure(tail, FS as f64, 1000.0);
        assert!(
            (m.amplitude - 0.99).abs() < 0.02,
            "{side:?}: {}",
            m.amplitude
        );
        assert!(
            m.residual_power < m.amplitude * m.amplitude / 2.0 * 1e-4,
            "{side:?}: residual {}",
            m.residual_power
        );
    }
    // NaN and infinity in f32 input do not panic or hang; once they have left the filter
    // history (873 taps) the output is finite again and the tone is back at its level.
    let mut bad = tone::<C32>(6000, 1000.0, Sideband::Usb, 0.5, 0);
    bad[10] = C32::new(f32::NAN, 0.0);
    bad[500] = C32::new(f32::INFINITY, f32::NEG_INFINITY);
    let out = run_block(&mut new_block::<C32>(&cfg), &bad, None);
    assert_eq!(out.len(), 6000);
    let tail = &out[500 + 873 + 100..];
    assert!(
        tail.iter().all(|v| v.is_finite()),
        "output stays non-finite"
    );
    let m = ToneMeter::measure(&as_f64(tail), FS as f64, 1000.0);
    assert!((m.amplitude - 0.5).abs() < 0.01, "{}", m.amplitude);
}
