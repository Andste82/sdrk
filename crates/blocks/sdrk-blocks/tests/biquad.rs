//! Block `biquad`: tone levels, the notch, runtime parameters, validation, invariance, reset and
//! the description.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_pass_by_value,
    clippy::type_complexity,
    clippy::manual_assert_eq,
    clippy::unreadable_literal,
    clippy::doc_markdown,
    clippy::cast_lossless,
    clippy::approx_constant
)]

mod common;

use common::{plans, rate, run_block};
use num_complex::Complex64;
use sdrk::{C32, CQ15, Q15, RealSample, Sdrk};
use sdrk_blocks::design::{BiquadResponse, butterworth_sos, to_f32_sos, to_q14};
use sdrk_blocks::{
    BiquadBlock, BiquadConfig, BiquadElem, BiquadRealBlock, Block, BlockReal, BlockSample, Error,
    Mutability, ParamValue, all_descriptions,
};
use sdrk_core::kernels::biquad::{BiquadReal, BiquadRealState};
use sdrk_testkit::{ChunkPlan, Pcg32, VecStorage};

const FS: u64 = 48_000;

fn cfg(response: BiquadResponse, order: usize, cutoff_hz: f64) -> BiquadConfig {
    BiquadConfig {
        response,
        order,
        cutoff_hz,
        ..BiquadConfig::default()
    }
}

fn lowpass4_2k() -> BiquadConfig {
    cfg(BiquadResponse::Lowpass, 4, 2_000.0)
}

fn real_block<T: BlockReal + BiquadElem>(c: &BiquadConfig) -> BiquadRealBlock<T, VecStorage>
where
    Sdrk: BiquadReal<T>,
{
    BiquadRealBlock::<T, _>::new(c, rate(FS), &mut VecStorage).unwrap()
}

fn sine<T: RealSample>(n: usize, hz: f64, amp: f64) -> Vec<T> {
    (0..n)
        .map(|i| T::from_f64_round(amp * (std::f64::consts::TAU * hz * i as f64 / FS as f64).sin()))
        .collect()
}

/// Amplitude of the tone `hz` in the last `window` samples (a Hann-windowed single-bin DFT).
fn level<T: RealSample>(x: &[T], hz: f64, window: usize) -> f64 {
    let tail = &x[x.len() - window..];
    let (mut re, mut im, mut w_sum) = (0.0, 0.0, 0.0);
    for (k, v) in tail.iter().enumerate() {
        let w = 0.5 - 0.5 * (std::f64::consts::TAU * k as f64 / window as f64).cos();
        let ph = std::f64::consts::TAU * hz * k as f64 / FS as f64;
        re += w * v.to_f64() * ph.cos();
        im -= w * v.to_f64() * ph.sin();
        w_sum += w;
    }
    2.0 * re.hypot(im) / w_sum
}

fn gain_db<T: RealSample>(out: &[T], hz: f64, amp: f64) -> f64 {
    20.0 * (level(out, hz, 1200).max(1e-12) / amp).log10()
}

/// Gain in dB of a fresh block for a tone.
fn tone_gain<T: BlockReal + BiquadElem>(c: &BiquadConfig, hz: f64) -> f64
where
    Sdrk: BiquadReal<T>,
{
    let mut block = real_block::<T>(c);
    let x = sine::<T>(4800, hz, 0.5);
    let y = run_block(&mut block, &x, None);
    gain_db(&y, hz, 0.5)
}

#[test]
fn lowpass_passes_1_khz_and_stops_8_khz() {
    let c = lowpass4_2k();
    // Less than 0.1 dB at 1 kHz, at least 48 dB at 8 kHz, in every sample type.
    for (pass, stop) in [
        (tone_gain::<f32>(&c, 1_000.0), tone_gain::<f32>(&c, 8_000.0)),
        (tone_gain::<Q15>(&c, 1_000.0), tone_gain::<Q15>(&c, 8_000.0)),
    ] {
        assert!(pass.abs() < 0.1, "1 kHz: {pass} dB");
        assert!(stop <= -48.0, "8 kHz: {stop} dB");
    }
}

#[test]
fn complex_blocks_filter_both_parts_alike() {
    fn check<T: BlockSample + BiquadElem>()
    where
        Sdrk: sdrk_core::kernels::biquad::Biquad<T>,
    {
        let c = lowpass4_2k();
        let mut block = BiquadBlock::<T, _>::new(&c, rate(FS), &mut VecStorage).unwrap();
        // re: 1 kHz, im: 8 kHz, so the parts must come out at the levels of the real block.
        let x: Vec<T> = (0..4800)
            .map(|i| {
                let t = i as f64 / FS as f64;
                T::from_c64_round(Complex64::new(
                    0.5 * (std::f64::consts::TAU * 1_000.0 * t).sin(),
                    0.5 * (std::f64::consts::TAU * 8_000.0 * t).sin(),
                ))
            })
            .collect();
        let y = run_block(&mut block, &x, None);
        let re: Vec<_> = y.iter().map(|s| s.re()).collect();
        let im: Vec<_> = y.iter().map(|s| s.im()).collect();
        assert!(gain_db(&re, 1_000.0, 0.5).abs() < 0.1);
        assert!(gain_db(&im, 8_000.0, 0.5) <= -48.0);
    }
    check::<C32>();
    check::<CQ15>();
}

#[test]
fn notch_removes_1_khz_with_q_10() {
    let c = BiquadConfig {
        response: BiquadResponse::Notch,
        cutoff_hz: 1_000.0,
        q: 10.0,
        ..BiquadConfig::default()
    };
    let f = tone_gain::<f32>(&c, 1_000.0);
    assert!(f <= -40.0, "f32 notch: {f} dB");
    // The notch is narrow: a tone 1 kHz away passes.
    assert!(tone_gain::<f32>(&c, 2_000.0).abs() < 1.0);
}

/// The Q15 notch depth: at least 34 dB, the separate Q15 threshold of test 3 (decision D23).
/// The zeros of the numerator `[1, -2 cos w0, 1] / a0` are rounded to Q2.14, which limits the
/// depth to 34.9 dB; `f32` keeps the 40 dB above.
#[test]
fn notch_removes_1_khz_with_q_10_in_q15() {
    let c = BiquadConfig {
        response: BiquadResponse::Notch,
        cutoff_hz: 1_000.0,
        q: 10.0,
        ..BiquadConfig::default()
    };
    let q = tone_gain::<Q15>(&c, 1_000.0);
    assert!(q <= -34.0, "Q15 notch: {q} dB");
}

#[test]
fn bandpass_has_unity_gain_at_the_centre() {
    let c = BiquadConfig {
        response: BiquadResponse::Bandpass,
        cutoff_hz: 3_000.0,
        q: 2.0,
        ..BiquadConfig::default()
    };
    assert!(tone_gain::<f32>(&c, 3_000.0).abs() < 0.05);
    assert!(tone_gain::<f32>(&c, 300.0) < -15.0);
}

/// The reference for runtime changes: the kernel on the first part with the old coefficients,
/// `set_coefs` and the second part with the new ones.
fn kernel_with_retune(old: &BiquadConfig, new: &BiquadConfig, x: &[f32], at: usize) -> Vec<f32> {
    let design = |c: &BiquadConfig| {
        let mut sos = vec![[0.0; 5]; c.sections()];
        butterworth_sos(c.response, c.order, c.cutoff_hz, FS as f64, &mut sos).unwrap();
        let mut f = vec![[0.0f32; 5]; c.sections()];
        to_f32_sos(&sos, &mut f).unwrap();
        f
    };
    let mut state =
        BiquadRealState::<f32, _, _>::new(design(old), vec![0.0f32; 4 * old.sections()]).unwrap();
    let mut out = vec![0.0f32; x.len()];
    let (a, b) = x.split_at(at);
    let (oa, ob) = out.split_at_mut(at);
    BiquadReal::<f32>::biquad_real(&Sdrk, &mut state, a, oa).unwrap();
    state.set_coefs(design(new)).unwrap();
    BiquadReal::<f32>::biquad_real(&Sdrk, &mut state, b, ob).unwrap();
    out
}

#[test]
fn a_cutoff_change_acts_from_the_next_call_and_keeps_the_history() {
    let old = lowpass4_2k();
    let new = cfg(BiquadResponse::Lowpass, 4, 500.0);
    let x = sine::<f32>(2000, 1_000.0, 0.5);
    let want = kernel_with_retune(&old, &new, &x, 1000);
    let mut block = real_block::<f32>(&old);
    let mut got = vec![0.0f32; 2000];
    block.process(&x[..1000], &mut got[..1000]).unwrap();
    block
        .set_param("cutoff_hz", ParamValue::F64(500.0))
        .unwrap();
    block.process(&x[1000..], &mut got[1000..]).unwrap();
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert!(got.iter().all(|v| v.is_finite()));
    assert_eq!(block.param("cutoff_hz"), Ok(ParamValue::F64(500.0)));
    // The 1 kHz tone is attenuated by the new 500 Hz low-pass.
    assert!(gain_db(&got, 1_000.0, 0.5) < -20.0);
}

#[test]
fn q_changes_the_notch_width_and_stays_finite() {
    let c = BiquadConfig {
        response: BiquadResponse::Notch,
        cutoff_hz: 1_000.0,
        q: 10.0,
        ..BiquadConfig::default()
    };
    let x = sine::<f32>(4800, 1_100.0, 0.5);
    let mut narrow = real_block::<f32>(&c);
    let y_narrow = run_block(&mut narrow, &x, None);
    let mut block = real_block::<f32>(&c);
    block.set_param("q", ParamValue::F64(1.0)).unwrap();
    let y_wide = run_block(&mut block, &x, None);
    assert!(y_wide.iter().all(|v| v.is_finite()));
    assert!(gain_db(&y_wide, 1_100.0, 0.5) < gain_db(&y_narrow, 1_100.0, 0.5) - 3.0);
    assert_eq!(block.param("q"), Ok(ParamValue::F64(1.0)));
}

#[test]
fn runtime_changes_in_q15_and_for_complex_blocks_work_too() {
    let mut q = real_block::<Q15>(&lowpass4_2k());
    let x = sine::<Q15>(2400, 1_000.0, 0.5);
    let mut y = vec![Q15(0); 2400];
    q.process(&x, &mut y).unwrap();
    q.set_param("cutoff_hz", ParamValue::F64(4_000.0)).unwrap();
    q.process(&x, &mut y).unwrap();
    assert!(gain_db(&y, 1_000.0, 0.5).abs() < 0.1);
    let mut c = BiquadBlock::<CQ15, _>::new(&lowpass4_2k(), rate(FS), &mut VecStorage).unwrap();
    c.set_param("cutoff_hz", ParamValue::F64(3_000.0)).unwrap();
    assert_eq!(c.param("cutoff_hz"), Ok(ParamValue::F64(3_000.0)));
    // A Q15 retune below fs / 500 is rejected and keeps the old design.
    assert!(matches!(
        c.set_param("cutoff_hz", ParamValue::F64(50.0)),
        Err(Error::OutOfRange { .. })
    ));
    assert_eq!(c.param("cutoff_hz"), Ok(ParamValue::F64(3_000.0)));
    // Build parameters are read-only.
    assert_eq!(
        c.set_param("order", ParamValue::I64(2)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(
        c.set_param("response", ParamValue::Choice(1)),
        Err(Error::ReadOnlyParameter)
    );
}

#[test]
fn q15_lowpass_and_highpass_need_a_cutoff_of_at_least_fs_over_500() {
    let r = rate(FS);
    let limit = FS as f64 / 500.0; // 96 Hz
    for response in [BiquadResponse::Lowpass, BiquadResponse::Highpass] {
        let ok = cfg(response, 2, limit);
        assert!(BiquadRealBlock::<Q15, _>::new(&ok, r, &mut VecStorage).is_ok());
        assert!(BiquadBlock::<CQ15, _>::new(&ok, r, &mut VecStorage).is_ok());
        let low = cfg(response, 2, limit * 0.99);
        assert!(matches!(
            BiquadRealBlock::<Q15, _>::new(&low, r, &mut VecStorage),
            Err(Error::OutOfRange {
                name: "cutoff_hz",
                ..
            })
        ));
        assert!(matches!(
            BiquadBlock::<CQ15, _>::new(&low, r, &mut VecStorage),
            Err(Error::OutOfRange {
                name: "cutoff_hz",
                ..
            })
        ));
        assert!(matches!(
            BiquadRealBlock::<Q15, VecStorage>::storage_bytes(&low, r),
            Err(Error::OutOfRange { .. })
        ));
        // The floating-point types have no such limit.
        assert!(BiquadRealBlock::<f32, _>::new(&low, r, &mut VecStorage).is_ok());
        assert!(BiquadBlock::<C32, _>::new(&low, r, &mut VecStorage).is_ok());
    }
    // Band-pass and notch are not low-frequency recursions around DC: no limit.
    for response in [BiquadResponse::Bandpass, BiquadResponse::Notch] {
        let c = cfg(response, 2, 10.0);
        assert!(BiquadRealBlock::<Q15, _>::new(&c, r, &mut VecStorage).is_ok());
    }
}

#[test]
fn invalid_configurations_are_errors() {
    let r = rate(FS);
    let new = |c: &BiquadConfig| BiquadRealBlock::<f32, _>::new(c, r, &mut VecStorage);
    // Odd, too small and too large orders.
    for order in [1, 3, 5, 7, 9, 10, 0] {
        assert!(
            matches!(
                new(&cfg(BiquadResponse::Lowpass, order, 1_000.0)),
                Err(Error::InvalidParameter { name: "order", .. })
            ),
            "order {order}"
        );
    }
    // Only order 2 for a band-pass and a notch.
    for response in [BiquadResponse::Bandpass, BiquadResponse::Notch] {
        assert!(matches!(
            new(&cfg(response, 4, 1_000.0)),
            Err(Error::InvalidParameter { name: "order", .. })
        ));
    }
    // The cutoff lies inside (0, fs / 2).
    for hz in [0.0, -5.0, 24_000.0, 30_000.0, f64::NAN] {
        assert!(
            matches!(
                new(&cfg(BiquadResponse::Lowpass, 2, hz)),
                Err(Error::OutOfRange {
                    name: "cutoff_hz",
                    ..
                })
            ),
            "cutoff {hz}"
        );
    }
    for q in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let c = BiquadConfig {
            response: BiquadResponse::Notch,
            cutoff_hz: 1_000.0,
            q,
            ..BiquadConfig::default()
        };
        assert!(
            matches!(new(&c), Err(Error::OutOfRange { name: "q", .. })),
            "q {q}"
        );
    }
    // Coefficients that do not fit Q2.14: a low-pass just below half the sample rate has
    // b1 = 2 - epsilon, which rounds to +2.0.
    let high = cfg(BiquadResponse::Lowpass, 2, 23_999.999);
    assert!(BiquadRealBlock::<f32, _>::new(&high, r, &mut VecStorage).is_ok());
    assert!(BiquadRealBlock::<Q15, _>::new(&high, r, &mut VecStorage).is_err());
    assert!(BiquadBlock::<CQ15, _>::new(&high, r, &mut VecStorage).is_err());
}

#[test]
fn invalid_runtime_values_are_rejected_and_change_nothing() {
    let mut block = real_block::<f32>(&lowpass4_2k());
    for (name, v) in [
        ("cutoff_hz", ParamValue::F64(30_000.0)),
        ("cutoff_hz", ParamValue::F64(0.0)),
        ("cutoff_hz", ParamValue::F64(f64::NAN)),
        ("cutoff_hz", ParamValue::Bool(true)),
        ("q", ParamValue::F64(0.0)),
        ("q", ParamValue::F64(-2.0)),
    ] {
        assert!(block.set_param(name, v).is_err(), "{name} {v:?}");
    }
    assert_eq!(block.param("cutoff_hz"), Ok(ParamValue::F64(2_000.0)));
    assert_eq!(block.param("q"), Ok(ParamValue::F64(0.7071)));
    assert_eq!(
        block.set_param("nope", ParamValue::F64(1.0)),
        Err(Error::UnknownParameter)
    );
}

#[test]
fn chunking_does_not_change_the_output() {
    let c = cfg(BiquadResponse::Highpass, 4, 300.0);
    let x = common::real_signal::<f32>(3000, 5, 0.01);
    let whole = run_block(&mut real_block::<f32>(&c), &x, None);
    let xq = common::real_signal::<Q15>(3000, 6, 0.01);
    let whole_q = run_block(&mut real_block::<Q15>(&c), &xq, None);
    let xc = common::signal::<C32>(3000, 7, 0.02);
    let whole_c = run_block(
        &mut BiquadBlock::<C32, _>::new(&c, rate(FS), &mut VecStorage).unwrap(),
        &xc,
        None,
    );
    let xcq = common::signal::<CQ15>(3000, 8, 0.02);
    let whole_cq = run_block(
        &mut BiquadBlock::<CQ15, _>::new(&c, rate(FS), &mut VecStorage).unwrap(),
        &xcq,
        None,
    );
    for seed in 0..20u64 {
        let plan = ChunkPlan::new(7000 + seed, 1 + (seed as usize * 37) % 500);
        let y = run_block(&mut real_block::<f32>(&c), &x, Some(plan));
        assert_eq!(bits(&y), bits(&whole), "f32 plan {seed}");
        let plan = ChunkPlan::new(7100 + seed, 1 + (seed as usize * 37) % 500);
        let y = run_block(&mut real_block::<Q15>(&c), &xq, Some(plan));
        assert_eq!(y, whole_q, "Q15 plan {seed}");
        let plan = ChunkPlan::new(7200 + seed, 1 + (seed as usize * 37) % 500);
        let y = run_block(
            &mut BiquadBlock::<C32, _>::new(&c, rate(FS), &mut VecStorage).unwrap(),
            &xc,
            Some(plan),
        );
        assert_eq!(cbits(&y), cbits(&whole_c), "C32 plan {seed}");
        let plan = ChunkPlan::new(7300 + seed, 1 + (seed as usize * 37) % 500);
        let y = run_block(
            &mut BiquadBlock::<CQ15, _>::new(&c, rate(FS), &mut VecStorage).unwrap(),
            &xcq,
            Some(plan),
        );
        assert_eq!(y, whole_cq, "CQ15 plan {seed}");
    }
    let _ = plans(1);
    let _ = Pcg32::new(1, 1);
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

fn cbits(x: &[C32]) -> Vec<(u32, u32)> {
    x.iter().map(|v| (v.re.to_bits(), v.im.to_bits())).collect()
}

#[test]
fn reset_clears_the_history_and_keeps_the_design() {
    let c = lowpass4_2k();
    let x = sine::<f32>(500, 1_000.0, 0.5);
    let mut block = real_block::<f32>(&c);
    let first = run_block(&mut block, &x, None);
    block.reset();
    let again = run_block(&mut block, &x, None);
    assert_eq!(bits(&first), bits(&again));
    block
        .set_param("cutoff_hz", ParamValue::F64(1_000.0))
        .unwrap();
    block.reset();
    assert_eq!(block.param("cutoff_hz"), Ok(ParamValue::F64(1_000.0)));
}

#[test]
fn nan_and_infinity_do_not_panic_and_leave_the_filter_usable_after_reset() {
    let mut block = real_block::<f32>(&lowpass4_2k());
    let mut x = vec![0.5f32; 16];
    x[3] = f32::NAN;
    x[9] = f32::INFINITY;
    let mut y = vec![0.0f32; 16];
    block.process(&x, &mut y).unwrap();
    block.reset();
    let x = sine::<f32>(2400, 1_000.0, 0.5);
    let y = run_block(&mut block, &x, None);
    assert!(y.iter().all(|v| v.is_finite()));
}

#[test]
fn the_block_is_registered_and_described() {
    let d = all_descriptions()
        .iter()
        .copied()
        .find(|d| d.kind == "biquad")
        .unwrap();
    assert_eq!(d.required, &["cutoff_hz"]);
    let names: Vec<_> = d.params.iter().map(|p| p.name).collect();
    assert_eq!(names, ["response", "cutoff_hz", "order", "q"]);
    for (name, mutability) in [
        ("response", Mutability::Build),
        ("cutoff_hz", Mutability::Runtime),
        ("order", Mutability::Build),
        ("q", Mutability::Runtime),
    ] {
        assert_eq!(d.param(name).unwrap().mutability, mutability, "{name}");
    }
    // Defaults of the specification: lowpass, order 2, q 0.7071; the cutoff has none.
    assert_eq!(
        d.param("order").unwrap().default_value(),
        Some(ParamValue::I64(2))
    );
    assert_eq!(
        d.param("q").unwrap().default_value(),
        Some(ParamValue::F64(0.7071))
    );
    assert_eq!(
        d.param("response").unwrap().default_value(),
        Some(ParamValue::Choice(0))
    );
    assert_eq!(d.param("cutoff_hz").unwrap().default_value(), None);
    assert_eq!(
        BiquadConfig::default().get("response"),
        Ok(ParamValue::Choice(0))
    );
    // The details name the pitfall of the Q15 cutoff limit.
    assert!(d.details.contains("sample rate / 500"));
}

#[test]
fn q15_coefficients_equal_the_designed_ones() {
    // The block rounds the f64 design to Q2.14 once: the filter of the block equals the kernel
    // on `to_q14(butterworth_sos(..))`.
    let c = cfg(BiquadResponse::Highpass, 2, 480.0);
    let mut sos = [[0.0; 5]; 1];
    butterworth_sos(c.response, 2, 480.0, FS as f64, &mut sos).unwrap();
    let mut q = [[0i16; 5]; 1];
    to_q14(&sos, &mut q).unwrap();
    let x = sine::<Q15>(1000, 2_000.0, 0.5);
    let mut state = BiquadRealState::<Q15, _, _>::new(q, [Q15(0); 4]).unwrap();
    let mut want = vec![Q15(0); 1000];
    BiquadReal::<Q15>::biquad_real(&Sdrk, &mut state, &x, &mut want).unwrap();
    let got = run_block(&mut real_block::<Q15>(&c), &x, None);
    assert_eq!(got, want);
}
