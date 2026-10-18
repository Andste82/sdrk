//! Tests of the receive kernels of `sdrk-core` (M06): `fir_decimate`, `resample`,
//! `quad_demod`, `deemphasis`, `dc_block`, `window`, `mag_sq`, `power_db` and the
//! window generators.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::identity_op,
    clippy::useless_vec,
    clippy::many_single_char_names,
    clippy::unreadable_literal,
    clippy::items_after_statements,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::excessive_precision,
    clippy::cast_lossless
)]

use core::f32::consts::PI as PI32;

use sdrk_core::kernels::decim::{
    FirDecimState, FirDecimate, Resample, ResampleReal, ResampleRealState, ResampleState,
};
use sdrk_core::kernels::demod::{
    DcBlock, DcBlockState, Deemphasis, DeemphasisState, QuadDemod, QuadDemodState,
};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::spectrum::{MagSq, PowerDb, Window};
use sdrk_core::timing::{Delay, RatioKind, Timing};
use sdrk_core::window::{Symmetry, bessel_i0, blackman_harris, hann, kaiser, window_gain};
use sdrk_core::{C32, CQ15, Error, Fallback, Progress, Q15, Rate, Ratio, Sample};
use sdrk_device::SdrkError;

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn unit(state: &mut u64) -> f32 {
    (lcg(state) >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
}

fn random_c32(n: usize, seed: u64) -> Vec<C32> {
    let mut s = seed;
    (0..n)
        .map(|_| C32::new(unit(&mut s), unit(&mut s)))
        .collect()
}

fn random_cq15(n: usize, seed: u64) -> Vec<CQ15> {
    let mut s = seed;
    (0..n)
        .map(|_| CQ15 {
            re: Q15(lcg(&mut s) as i16),
            im: Q15(lcg(&mut s) as i16),
        })
        .collect()
}

fn random_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| unit(&mut s)).collect()
}

fn random_q15(n: usize, seed: u64) -> Vec<Q15> {
    let mut s = seed;
    (0..n).map(|_| Q15(lcg(&mut s) as i16)).collect()
}

fn taps_f32(n: usize, seed: u64) -> Vec<f32> {
    random_f32(n, seed).iter().map(|v| v * 0.25).collect()
}

fn taps_q15(n: usize, seed: u64) -> Vec<Q15> {
    random_q15(n, seed).iter().map(|q| Q15(q.0 / 4)).collect()
}

fn q(re: i16, im: i16) -> CQ15 {
    CQ15 {
        re: Q15(re),
        im: Q15(im),
    }
}

// -------------------------------------------------------------- fir_decimate

#[test]
fn fir_decimate_state_validation_and_timing() {
    let ok = FirDecimState::<C32, _, _>::new([0.5f32; 5], [C32::ZERO; 4], 3).unwrap();
    assert_eq!(ok.factor(), 3);
    assert_eq!(ok.countdown(), 0);
    assert_eq!(ok.taps().len(), 5);
    assert_eq!(ok.history().len(), 4);
    assert_eq!(ok.ratio(), Ratio::new(1, 3).unwrap());
    assert_eq!(ok.ratio_kind(), RatioKind::Constant);
    assert_eq!(ok.group_delay(), Delay::new(4, 2).unwrap());
    assert_eq!(
        FirDecimState::<C32, _, _>::new([0.5f32; 5], [C32::ZERO; 3], 3).unwrap_err(),
        Error::LengthMismatch {
            expected: 4,
            got: 3
        }
    );
    for bad in [
        FirDecimState::<C32, _, _>::new([0.5f32; 5], [C32::ZERO; 4], 0).map(|_| ()),
        FirDecimState::<C32, _, _>::new([0.5f32; 0], [C32::ZERO; 0], 1).map(|_| ()),
        FirDecimState::<C32, _, _>::new([0.5f32; 1], [C32::ZERO; 0], usize::MAX).map(|_| ()),
    ] {
        let e = bad.unwrap_err();
        assert!(matches!(e, Error::InvalidParameter(_)));
        assert_eq!(e.kind(), sdrk_device::ErrorKind::InvalidArgument);
    }
}

#[test]
fn fir_decimate_takes_every_mth_output_starting_with_the_first() {
    let mut st = FirDecimState::<C32, _, _>::new([1.0f32, 1.0, 1.0], [C32::ZERO; 2], 4).unwrap();
    let x: Vec<C32> = (1..=12).map(|v| C32::new(v as f32, -(v as f32))).collect();
    let mut y = [C32::ZERO; 3];
    let p = Fallback.fir_decimate(&mut st, &x, &mut y).unwrap();
    assert_eq!(
        p,
        Progress {
            consumed: 12,
            produced: 3
        }
    );
    // y_full = running sum of three samples; outputs at the input positions 0, 4, 8.
    assert_eq!(y[0], C32::new(1.0, -1.0));
    assert_eq!(y[1], C32::new(3.0 + 4.0 + 5.0, -12.0));
    assert_eq!(y[2], C32::new(7.0 + 8.0 + 9.0, -24.0));
    // The phase carries on: the next output is due at input 12, i.e. no sample is left to skip.
    assert_eq!(st.countdown(), 0);
    assert_eq!(st.history(), &x[10..]);
}

#[test]
fn fir_decimate_reset_restarts_the_phase_and_the_history() {
    let mut st = FirDecimState::<C32, _, _>::new([1.0f32, 1.0], [C32::ZERO; 1], 3).unwrap();
    let x = random_c32(10, 1);
    let mut y = [C32::ZERO; 10];
    let first = Fallback.fir_decimate(&mut st, &x, &mut y).unwrap();
    let a = y[..first.produced].to_vec();
    st.reset();
    assert_eq!(st.countdown(), 0);
    assert!(st.history().iter().all(|s| *s == C32::ZERO));
    let again = Fallback.fir_decimate(&mut st, &x, &mut y).unwrap();
    assert_eq!(first, again);
    assert_eq!(a, y[..again.produced]);
}

#[test]
fn fir_decimate_with_factor_one_equals_fir() {
    let x = random_c32(300, 2);
    let taps = taps_f32(21, 3);
    let mut fir = FirState::<C32, _, _>::new(taps.clone(), vec![C32::ZERO; 20]).unwrap();
    let mut dec = FirDecimState::<C32, _, _>::new(taps, vec![C32::ZERO; 20], 1).unwrap();
    let (mut a, mut b) = (vec![C32::ZERO; 300], vec![C32::ZERO; 300]);
    Fallback.fir(&mut fir, &x, &mut a).unwrap();
    let p = Fallback.fir_decimate(&mut dec, &x, &mut b).unwrap();
    assert_eq!((p.consumed, p.produced), (300, 300));
    assert_eq!(
        bytemuck::cast_slice::<C32, u32>(&a),
        bytemuck::cast_slice::<C32, u32>(&b)
    );
    assert_eq!(fir.history(), dec.history());

    let x = random_cq15(300, 4);
    let taps = taps_q15(21, 5);
    let mut fir = FirState::<CQ15, _, _>::new(taps.clone(), vec![CQ15::ZERO; 20]).unwrap();
    let mut dec = FirDecimState::<CQ15, _, _>::new(taps, vec![CQ15::ZERO; 20], 1).unwrap();
    let (mut a, mut b) = (vec![CQ15::ZERO; 300], vec![CQ15::ZERO; 300]);
    Fallback.fir(&mut fir, &x, &mut a).unwrap();
    Fallback.fir_decimate(&mut dec, &x, &mut b).unwrap();
    assert_eq!(a, b);
}

#[test]
fn fir_decimate_with_a_single_tap_and_the_history_of_short_blocks() {
    // N = 1 has no history; M = 2 keeps samples 0, 2, 4.
    let mut st = FirDecimState::<CQ15, _, _>::new([Q15(16384)], [], 2).unwrap();
    let x: Vec<CQ15> = (0..5).map(|v| q(1000 * (v + 1), 0)).collect();
    let mut y = [CQ15::ZERO; 5];
    let p = Fallback.fir_decimate(&mut st, &x, &mut y).unwrap();
    assert_eq!((p.consumed, p.produced), (5, 3));
    assert_eq!(&y[..3], &[q(500, 0), q(1500, 0), q(2500, 0)]);
    // Blocks shorter than the filter still build up the history.
    let mut st =
        FirDecimState::<C32, _, _>::new([1.0f32, 2.0, 3.0, 4.0], [C32::ZERO; 3], 1).unwrap();
    let mut y = [C32::ZERO; 1];
    for v in 1..=4 {
        Fallback
            .fir_decimate(&mut st, &[C32::new(v as f32, 0.0)], &mut y)
            .unwrap();
    }
    assert_eq!(
        y[0],
        C32::new(1.0 * 4.0 + 2.0 * 3.0 + 3.0 * 2.0 + 4.0 * 1.0, 0.0)
    );
}

#[test]
fn fir_decimate_progress_stops_at_the_right_sample_and_resubmission_matches() {
    let taps = taps_f32(9, 6);
    let x = random_c32(103, 7);
    let new = || FirDecimState::<C32, _, _>::new(taps.clone(), vec![C32::ZERO; 8], 4).unwrap();
    let mut whole = vec![C32::ZERO; 40];
    let total = Fallback.fir_decimate(&mut new(), &x, &mut whole).unwrap();
    assert_eq!(total.consumed, 103);
    assert_eq!(total.produced, 26); // inputs 0, 4, ..., 100
    // Room for 5 outputs: the 6th producing sample (input 20) is not consumed.
    let mut st = new();
    let mut small = vec![C32::ZERO; 5];
    let p = Fallback.fir_decimate(&mut st, &x, &mut small).unwrap();
    assert_eq!((p.consumed, p.produced), (20, 5));
    // Resubmit the unconsumed rest in small steps.
    let mut got = small.clone();
    let mut pos = p.consumed;
    while pos < x.len() {
        let mut room = vec![C32::ZERO; 3];
        let p = Fallback
            .fir_decimate(&mut st, &x[pos..], &mut room)
            .unwrap();
        assert!(p.consumed + p.produced > 0);
        got.extend_from_slice(&room[..p.produced]);
        pos += p.consumed;
    }
    assert_eq!(got.len(), 26);
    assert_eq!(
        bytemuck::cast_slice::<C32, u32>(&got),
        bytemuck::cast_slice::<C32, u32>(&whole[..26])
    );
    // No room at all: samples that need no output are consumed up to the first output.
    let mut st = new();
    Fallback
        .fir_decimate(&mut st, &x[..1], &mut whole[..1])
        .unwrap();
    let p = Fallback.fir_decimate(&mut st, &x[1..], &mut []).unwrap();
    assert_eq!((p.consumed, p.produced), (3, 0));
    // Length zero input is a no-op.
    let p = Fallback.fir_decimate(&mut st, &[], &mut whole).unwrap();
    assert_eq!(p, Progress::default());
}

// ------------------------------------------------------------------ resample

#[test]
fn resample_state_validation_and_timing() {
    // History length is ceil(N / L).
    let st = ResampleState::<C32, _, _>::new([0.1f32; 10], [C32::ZERO; 4], 3, 2).unwrap();
    assert_eq!((st.up(), st.down()), (3, 2));
    assert_eq!(st.history().len(), 4);
    assert_eq!(st.taps().len(), 10);
    assert_eq!(st.phase(), 0);
    assert_eq!(st.ratio(), Ratio::new(3, 2).unwrap());
    assert_eq!(st.group_delay(), Delay::new(9, 6).unwrap());
    assert_eq!(
        ResampleState::<C32, _, _>::new([0.1f32; 10], [C32::ZERO; 3], 3, 2).unwrap_err(),
        Error::LengthMismatch {
            expected: 4,
            got: 3
        }
    );
    for bad in [
        ResampleState::<C32, _, _>::new([0.1f32; 10], [C32::ZERO; 4], 0, 2).map(|_| ()),
        ResampleState::<C32, _, _>::new([0.1f32; 10], [C32::ZERO; 4], 3, 0).map(|_| ()),
        ResampleState::<C32, _, _>::new([0.1f32; 0], [C32::ZERO; 0], 1, 1).map(|_| ()),
        ResampleState::<C32, _, _>::new([0.1f32; 1], [C32::ZERO; 1], usize::MAX, 1).map(|_| ()),
    ] {
        assert!(matches!(bad.unwrap_err(), Error::InvalidParameter(_)));
    }
    let real = ResampleRealState::<Q15, _, _>::new([Q15(1); 7], [Q15(0); 4], 2, 5).unwrap();
    assert_eq!(real.ratio(), Ratio::new(2, 5).unwrap());
    assert_eq!(real.group_delay(), Delay::new(6, 4).unwrap());
    assert_eq!(real.ratio_kind(), RatioKind::Constant);
}

#[test]
fn resample_output_count_follows_the_formula() {
    for (up, down) in [
        (1usize, 1usize),
        (1, 5),
        (3, 2),
        (2, 3),
        (147, 160),
        (5, 1),
        (4, 4),
    ] {
        for n in [1usize, 2, 7, 50, 161] {
            let taps = vec![0.01f32; 12];
            let mut st = ResampleState::<C32, _, _>::new(
                taps,
                vec![C32::ZERO; 12usize.div_ceil(up)],
                up,
                down,
            )
            .unwrap();
            let x = random_c32(n, 8);
            let mut y = vec![C32::ZERO; n * up + 1];
            let p = Fallback.resample(&mut st, &x, &mut y).unwrap();
            let k = (n * up - 1) / down + 1;
            assert_eq!((p.consumed, p.produced), (n, k), "L={up} M={down} n={n}");
        }
    }
    // The worked example of the specification: L = 3, M = 2, one input sample gives two outputs.
    let mut st = ResampleState::<C32, _, _>::new([1.0f32; 3], [C32::ZERO; 1], 3, 2).unwrap();
    let mut y = [C32::ZERO; 4];
    let p = Fallback
        .resample(&mut st, &[C32::new(1.0, 0.0)], &mut y)
        .unwrap();
    assert_eq!((p.consumed, p.produced), (1, 2));
}

#[test]
fn resample_zero_order_hold_and_decimation_by_value() {
    // Upsampling by 3 with three taps of 1 repeats every sample three times.
    let mut st = ResampleState::<CQ15, _, _>::new([Q15(32767); 3], [CQ15::ZERO; 1], 3, 1).unwrap();
    let x = [q(1000, -1000), q(2000, -2000)];
    let mut y = [CQ15::ZERO; 8];
    let p = Fallback.resample(&mut st, &x, &mut y).unwrap();
    assert_eq!((p.consumed, p.produced), (2, 6));
    let want = |v: i16| {
        // 32767 / 32768 * v rounded half up
        let r = ((i32::from(v) * 32767 + 16384) >> 15) as i16;
        q(r, -r)
    };
    assert_eq!(
        &y[..6],
        &[
            want(1000),
            want(1000),
            want(1000),
            want(2000),
            want(2000),
            want(2000)
        ]
    );
    // A single unit tap with M = 3 keeps every third sample.
    let mut st = ResampleRealState::<f32, _, _>::new([1.0f32], [0.0f32; 1], 1, 3).unwrap();
    let x: Vec<f32> = (0..10).map(|v| v as f32).collect();
    let mut y = [0.0f32; 10];
    let p = Fallback.resample_real(&mut st, &x, &mut y).unwrap();
    assert_eq!((p.consumed, p.produced), (10, 4));
    assert_eq!(&y[..4], &[0.0, 3.0, 6.0, 9.0]);
}

#[test]
fn resample_with_l_and_m_one_equals_fir() {
    let x = random_c32(200, 9);
    let taps = taps_f32(17, 10);
    let mut fir = FirState::<C32, _, _>::new(taps.clone(), vec![C32::ZERO; 16]).unwrap();
    let mut rs = ResampleState::<C32, _, _>::new(taps, vec![C32::ZERO; 17], 1, 1).unwrap();
    let (mut a, mut b) = (vec![C32::ZERO; 200], vec![C32::ZERO; 200]);
    Fallback.fir(&mut fir, &x, &mut a).unwrap();
    let p = Fallback.resample(&mut rs, &x, &mut b).unwrap();
    assert_eq!((p.consumed, p.produced), (200, 200));
    assert_eq!(
        bytemuck::cast_slice::<C32, u32>(&a),
        bytemuck::cast_slice::<C32, u32>(&b)
    );

    let x = random_cq15(200, 11);
    let taps = taps_q15(17, 12);
    let mut fir = FirState::<CQ15, _, _>::new(taps.clone(), vec![CQ15::ZERO; 16]).unwrap();
    let mut rs = ResampleState::<CQ15, _, _>::new(taps, vec![CQ15::ZERO; 17], 1, 1).unwrap();
    let (mut a, mut b) = (vec![CQ15::ZERO; 200], vec![CQ15::ZERO; 200]);
    Fallback.fir(&mut fir, &x, &mut a).unwrap();
    Fallback.resample(&mut rs, &x, &mut b).unwrap();
    assert_eq!(a, b);

    // The real variant is the same filter on the real axis.
    let x = random_f32(100, 13);
    let taps = taps_f32(9, 14);
    let mut rs = ResampleRealState::<f32, _, _>::new(taps.clone(), vec![0.0f32; 9], 1, 1).unwrap();
    let mut y = vec![0.0f32; 100];
    Fallback.resample_real(&mut rs, &x, &mut y).unwrap();
    let cx: Vec<C32> = x.iter().map(|&v| C32::new(v, 0.0)).collect();
    let mut fir = FirState::<C32, _, _>::new(taps, vec![C32::ZERO; 8]).unwrap();
    let mut cy = vec![C32::ZERO; 100];
    Fallback.fir(&mut fir, &cx, &mut cy).unwrap();
    assert!(
        y.iter()
            .zip(&cy)
            .all(|(a, b)| a.to_bits() == b.re.to_bits())
    );
}

#[test]
fn resample_progress_stops_at_the_right_sample_and_resubmission_matches() {
    for (up, down) in [(3usize, 2usize), (1, 5), (4, 1), (147, 160)] {
        let taps = taps_f32(24, 15);
        let hist = taps.len().div_ceil(up);
        let new = || {
            ResampleState::<C32, _, _>::new(taps.clone(), vec![C32::ZERO; hist], up, down).unwrap()
        };
        let x = random_c32(97, 16);
        let mut whole = vec![C32::ZERO; 97 * up + 4];
        let total = Fallback.resample(&mut new(), &x, &mut whole).unwrap();
        assert_eq!(total.consumed, 97);
        whole.truncate(total.produced);
        // Tiny output buffers, resubmitting the unconsumed rest each time.
        let mut st = new();
        let mut got = Vec::new();
        let mut pos = 0;
        let mut room = vec![C32::ZERO; 2];
        let mut calls = 0;
        loop {
            let p = Fallback.resample(&mut st, &x[pos..], &mut room).unwrap();
            got.extend_from_slice(&room[..p.produced]);
            pos += p.consumed;
            calls += 1;
            assert!(calls < 10_000, "no progress");
            if pos == x.len() && p.produced < room.len() {
                break;
            }
        }
        assert_eq!(
            bytemuck::cast_slice::<C32, u32>(&got),
            bytemuck::cast_slice::<C32, u32>(&whole),
            "L={up} M={down}"
        );
    }
    // One output of room for the 3/2 resampler: input 0 gives outputs 0 and 1, so the
    // second output stays pending until the next call.
    let mut st = ResampleState::<C32, _, _>::new([1.0f32; 3], [C32::ZERO; 1], 3, 2).unwrap();
    let x = [C32::new(1.0, 0.0), C32::new(2.0, 0.0)];
    let mut one = [C32::ZERO; 1];
    let p = Fallback.resample(&mut st, &x, &mut one).unwrap();
    assert_eq!((p.consumed, p.produced), (1, 1));
    assert!(st.phase() < 0, "one output is still pending");
    let p = Fallback.resample(&mut st, &x[1..], &mut one).unwrap();
    // The pending output is produced first; the next sample is consumed because nothing
    // is pending any more, and the output that follows from it waits for room.
    assert_eq!((p.consumed, p.produced), (1, 1));
    assert!(st.phase() < 0);
}

#[test]
fn resample_reset_restarts_everything() {
    let taps = taps_q15(10, 17);
    let mut st = ResampleState::<CQ15, _, _>::new(taps, vec![CQ15::ZERO; 4], 3, 2).unwrap();
    let x = random_cq15(40, 18);
    let mut y = vec![CQ15::ZERO; 130];
    let a = Fallback.resample(&mut st, &x, &mut y).unwrap();
    let first = y[..a.produced].to_vec();
    st.reset();
    assert_eq!(st.phase(), 0);
    assert!(st.history().iter().all(|s| *s == CQ15::ZERO));
    let b = Fallback.resample(&mut st, &x, &mut y).unwrap();
    assert_eq!(a, b);
    assert_eq!(first, y[..b.produced]);
}

// ----------------------------------------------------------------- quad_demod

#[test]
fn quad_demod_of_a_pure_tone_is_constant() {
    for f in [0.01f32, 0.1, 0.25, -0.2, 0.4] {
        let x: Vec<C32> = (0..512)
            .map(|n| {
                let ph = f64::from(f) * n as f64 * 2.0 * core::f64::consts::PI;
                C32::new(libm::cos(ph) as f32, libm::sin(ph) as f32)
            })
            .collect();
        let mut st = QuadDemodState::<C32>::new();
        let mut y = vec![0f32; 512];
        let gain = 2.5f32;
        Fallback.quad_demod(&mut st, &x, &mut y, gain).unwrap();
        let want = 2.0 * PI32 * f * gain;
        assert_eq!(y[0], 0.0, "the first output has an all-zero predecessor");
        for (n, v) in y.iter().enumerate().skip(1) {
            assert!((v - want).abs() < 1e-5, "f={f} n={n}: {v} vs {want}");
        }
        // Without gain the error is below 1e-6 as well.
        let mut st = QuadDemodState::<C32>::new();
        Fallback.quad_demod(&mut st, &x, &mut y, 1.0).unwrap();
        let want = 2.0 * PI32 * f;
        assert!(y.iter().skip(1).all(|v| (v - want).abs() < 1e-6), "f={f}");
    }
}

#[test]
fn quad_demod_q15_tone_and_special_values() {
    let q15_unit = |f: f64| -> Vec<CQ15> {
        (0..256)
            .map(|n| {
                let ph = f * n as f64 * 2.0 * core::f64::consts::PI;
                q(
                    (libm::cos(ph) * 20000.0).round() as i16,
                    (libm::sin(ph) * 20000.0).round() as i16,
                )
            })
            .collect()
    };
    // f = 0.125 cycles per sample is a phase step of pi / 4 = 0.25 in units of pi.
    let x = q15_unit(0.125);
    let mut st = QuadDemodState::<CQ15>::new();
    let mut y = vec![Q15(0); 256];
    Fallback
        .quad_demod(&mut st, &x, &mut y, Q15(32767))
        .unwrap();
    assert_eq!(y[0], Q15(0));
    for v in &y[1..] {
        assert!((i32::from(v.0) - 8192).abs() <= 3, "{v:?}");
    }
    // Half gain halves the output (Q15 multiply of the phase value).
    let mut st = QuadDemodState::<CQ15>::new();
    Fallback
        .quad_demod(&mut st, &x, &mut y, Q15(16384))
        .unwrap();
    assert!(y[1..].iter().all(|v| (i32::from(v.0) - 4096).abs() <= 3));
    // Zero samples: arg(0) = 0 in both types, also with an infinite gain.
    let mut st = QuadDemodState::<C32>::new();
    let mut yf = [1.0f32; 3];
    Fallback
        .quad_demod(&mut st, &[C32::ZERO; 3], &mut yf, f32::INFINITY)
        .unwrap_or_default();
    assert!(yf.iter().all(|v| v.is_nan() || *v == 0.0));
    let mut st = QuadDemodState::<C32>::new();
    Fallback
        .quad_demod(
            &mut st,
            &[C32::new(-0.0, 0.0), C32::new(-1.0, -0.0)],
            &mut yf[..2],
            1.0,
        )
        .unwrap();
    assert_eq!(yf[..2], [0.0, 0.0]); // 0 * conj(0) is zero whatever the sign of the zeros
    let mut st = QuadDemodState::<CQ15>::new();
    let mut yq = [Q15(7); 3];
    Fallback
        .quad_demod(&mut st, &[CQ15::ZERO; 3], &mut yq, Q15(32767))
        .unwrap();
    assert_eq!(yq, [Q15(0); 3]);
    // A half turn (phase pi) is +1.0 in units of pi, i.e. 32767 after rounding, times the gain.
    let mut st = QuadDemodState::<CQ15>::new();
    Fallback
        .quad_demod(
            &mut st,
            &[q(10000, 0), q(-10000, 0)],
            &mut yq[..2],
            Q15(32767),
        )
        .unwrap();
    assert_eq!(yq[1], Q15(32766)); // q15_mul(32767, 32767)
}

#[test]
fn quad_demod_state_carries_over_and_resets() {
    let x = random_c32(64, 19);
    let mut st = QuadDemodState::<C32>::new();
    let (mut a, mut b) = (vec![0f32; 64], vec![0f32; 64]);
    Fallback.quad_demod(&mut st, &x, &mut a, 1.7).unwrap();
    assert_eq!(st.prev(), x[63]);
    let mut st = QuadDemodState::<C32>::default();
    Fallback
        .quad_demod(&mut st, &x[..20], &mut b[..20], 1.7)
        .unwrap();
    Fallback
        .quad_demod(&mut st, &x[20..], &mut b[20..], 1.7)
        .unwrap();
    assert_eq!(a, b);
    st.reset();
    assert_eq!(st.prev(), C32::ZERO);
    assert_eq!(st.ratio(), Ratio::ONE);
    assert_eq!(st.group_delay(), Delay::new(1, 2).unwrap());
    // Wrong output length.
    assert_eq!(
        Fallback.quad_demod(&mut st, &x, &mut a[..3], 1.0),
        Err(Error::LengthMismatch {
            expected: 64,
            got: 3
        })
    );
}

// ---------------------------------------------------------------- deemphasis

#[test]
fn deemphasis_alpha_and_validation() {
    let rate = Rate::hz(48_000).unwrap();
    let a = DeemphasisState::<f32>::new(50e-6, rate).unwrap().alpha();
    let exact = 1.0 - (-1.0 / (48_000.0 * 50e-6f64)).exp();
    assert_eq!(a, exact as f32);
    let aq = DeemphasisState::<Q15>::new(50e-6, rate).unwrap().alpha();
    assert_eq!(aq, Q15((exact * 32768.0 + 0.5).floor() as i16));
    for tau in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let e = DeemphasisState::<f32>::new(tau, rate).unwrap_err();
        assert!(matches!(e, Error::InvalidParameter(_)), "{tau}");
    }
    let st = DeemphasisState::<f32>::new(75e-6, rate).unwrap();
    assert_eq!(st.ratio(), Ratio::ONE);
    assert_eq!(st.group_delay(), Delay::ZERO);
    assert_eq!(st.last(), 0.0);
}

#[test]
fn deemphasis_step_response_reaches_63_percent_after_tau_fs_samples() {
    // fs = 1 kHz, tau = 100 ms: tau * fs = 100 samples.
    let rate = Rate::hz(1000).unwrap();
    let n = 100usize;
    let mut st = DeemphasisState::<f32>::new(0.1, rate).unwrap();
    let x = vec![1.0f32; 400];
    let mut y = vec![0.0f32; 400];
    Fallback.deemphasis(&mut st, &x, &mut y).unwrap();
    let first = y.iter().position(|&v| v >= 0.632_120_6).unwrap();
    // `first` is the 0-based index; after `first + 1` samples the level is reached.
    assert!((first + 1).abs_diff(n) <= 1, "{first}");
    assert!((y[n - 1] - 0.632).abs() < 0.01);
    assert!(y.windows(2).all(|w| w[1] >= w[0]));
    assert!((y[399] - 1.0).abs() < 0.03);

    let mut st = DeemphasisState::<Q15>::new(0.1, rate).unwrap();
    let x = vec![Q15(20000); 400];
    let mut y = vec![Q15(0); 400];
    Fallback.deemphasis(&mut st, &x, &mut y).unwrap();
    let level = 20000.0 * 0.632_120_6;
    let first = y.iter().position(|v| f64::from(v.0) >= level).unwrap();
    assert!((first + 1).abs_diff(n) <= 1, "{first}");
    assert!(y.windows(2).all(|w| w[1].0 >= w[0].0));
}

#[test]
fn deemphasis_state_carries_over_flushes_and_resets() {
    let rate = Rate::hz(48_000).unwrap();
    let x = random_f32(300, 20);
    let (mut a, mut b) = (vec![0f32; 300], vec![0f32; 300]);
    let mut st = DeemphasisState::<f32>::new(50e-6, rate).unwrap();
    Fallback.deemphasis(&mut st, &x, &mut a).unwrap();
    let mut st2 = DeemphasisState::<f32>::new(50e-6, rate).unwrap();
    Fallback
        .deemphasis(&mut st2, &x[..77], &mut b[..77])
        .unwrap();
    Fallback
        .deemphasis(&mut st2, &x[77..], &mut b[77..])
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(st.last(), a[299]);
    // A decaying state is flushed to exactly zero instead of walking through the denormals.
    st.reset();
    assert_eq!(st.last(), 0.0);
    let mut y = [0f32; 1];
    Fallback.deemphasis(&mut st, &[1.0], &mut y).unwrap();
    let silence = vec![0f32; 20_000];
    let mut out = vec![1f32; 20_000];
    Fallback.deemphasis(&mut st, &silence, &mut out).unwrap();
    assert_eq!(*out.last().unwrap(), 0.0);
    assert!(out.iter().all(|v| *v == 0.0 || v.abs() >= 1e-30));
    // Q15: full scale in, saturation holds and the state is the last output.
    let mut st = DeemphasisState::<Q15>::new(1e-6, rate).unwrap(); // alpha close to 1
    let mut y = [Q15(0); 4];
    Fallback
        .deemphasis(&mut st, &[Q15::MAX, Q15::MIN, Q15::MAX, Q15::MIN], &mut y)
        .unwrap();
    // alpha rounds to 32767 / 32768, so the first output is 32767 * 32767 / 2^30 -> 32766.
    assert_eq!(st.alpha(), Q15::MAX);
    assert_eq!(y[0], Q15(32766));
    assert!(y[1].0 < 0 && y[2].0 > 0 && y[3].0 < 0);
    assert_eq!(st.last(), y[3]);
    assert_eq!(
        Fallback.deemphasis(&mut st, &[Q15(1)], &mut y),
        Err(Error::LengthMismatch {
            expected: 1,
            got: 4
        })
    );
}

// ------------------------------------------------------------------ dc_block

#[test]
fn dc_block_state_validation() {
    for r in [0.0f32, 1.0, -0.5, 1.5, f32::NAN] {
        assert!(matches!(
            DcBlockState::<C32>::new(r).unwrap_err(),
            Error::InvalidParameter(_)
        ));
    }
    for r in [Q15(0), Q15(-5)] {
        assert!(DcBlockState::<CQ15>::new(r).is_err());
    }
    let st = DcBlockState::<CQ15>::new(Q15(32767)).unwrap();
    assert_eq!(st.r(), Q15(32767));
    assert_eq!(st.ratio(), Ratio::ONE);
    assert_eq!(st.group_delay(), Delay::ZERO);
    assert_eq!(st.x_prev(), CQ15::ZERO);
    assert_eq!(st.y_prev(), CQ15::ZERO);
}

#[test]
fn dc_block_removes_a_constant() {
    let mut st = DcBlockState::<C32>::new(0.999).unwrap();
    let x = vec![C32::new(0.5, 0.25); 10_000];
    let mut y = vec![C32::ZERO; 10_000];
    Fallback.dc_block(&mut st, &x, &mut y).unwrap();
    assert_eq!(y[0], C32::new(0.5, 0.25));
    let last = y[9_999];
    assert!(last.re.hypot(last.im) < 1e-3, "{last}");
    // The decay is geometric: y[n] = r^n * x0.
    assert!((y[100].re - 0.5 * 0.999f32.powi(100)).abs() < 1e-5);

    // Q15 has a dead band of 496 LSB per component (documented limit).
    let r = Q15(32735);
    let mut st = DcBlockState::<CQ15>::new(r).unwrap();
    let x = vec![q(16384, 8192); 10_000];
    let mut y = vec![CQ15::ZERO; 10_000];
    Fallback.dc_block(&mut st, &x, &mut y).unwrap();
    let last = y[9_999];
    assert!(i32::from(last.re.0).abs() <= 496, "{last:?}");
    assert!(i32::from(last.im.0).abs() <= 496, "{last:?}");
    let mag = last.to_c32().re.hypot(last.to_c32().im);
    assert!(mag < 0.022, "{mag}");
    // The bound is tight: the state really does get stuck on the dead band.
    assert!(y[9_000..].iter().any(|s| s.re.0 != 0));
}

#[test]
fn dc_block_state_carries_over_and_resets() {
    let x = random_c32(200, 21);
    let (mut a, mut b) = (vec![C32::ZERO; 200], vec![C32::ZERO; 200]);
    let mut st = DcBlockState::<C32>::new(0.95).unwrap();
    Fallback.dc_block(&mut st, &x, &mut a).unwrap();
    assert_eq!(st.x_prev(), x[199]);
    let mut st = DcBlockState::<C32>::new(0.95).unwrap();
    Fallback.dc_block(&mut st, &x[..33], &mut b[..33]).unwrap();
    Fallback.dc_block(&mut st, &x[33..], &mut b[33..]).unwrap();
    assert_eq!(a, b);
    st.reset();
    assert_eq!(st.y_prev(), C32::ZERO);
    // Silence after a burst decays into an exact zero (denormal flush of the state).
    let mut st = DcBlockState::<C32>::new(0.5).unwrap();
    Fallback
        .dc_block(&mut st, &[C32::new(1.0, 1.0)], &mut a[..1])
        .unwrap();
    let silence = vec![C32::ZERO; 400];
    let mut out = vec![C32::new(1.0, 1.0); 400];
    Fallback.dc_block(&mut st, &silence, &mut out).unwrap();
    assert_eq!(st.y_prev(), C32::ZERO);
    assert!(Fallback.dc_block(&mut st, &silence, &mut out[..2]).is_err());
}

// ------------------------------------------------------------------- windows

#[test]
fn symmetric_windows_are_symmetric_bit_for_bit() {
    for n in [2usize, 3, 8, 255, 256, 1023] {
        let mut h = vec![0f32; n];
        let mut b = vec![0f32; n];
        let mut k = vec![0f32; n];
        hann(&mut h, Symmetry::Symmetric).unwrap();
        blackman_harris(&mut b, Symmetry::Symmetric).unwrap();
        kaiser(&mut k, 8.0, Symmetry::Symmetric).unwrap();
        for w in [&h, &b, &k] {
            for i in 0..n / 2 {
                assert_eq!(w[i].to_bits(), w[n - 1 - i].to_bits(), "n={n} i={i}");
            }
        }
        assert_eq!(h[0], 0.0);
        assert!((b[0] - 6e-5).abs() < 1e-6);
    }
    // Q15 windows too.
    let mut q = vec![Q15(0); 129];
    hann(&mut q, Symmetry::Symmetric).unwrap();
    assert_eq!(q[64], Q15::MAX); // the peak 1.0 saturates
    assert!((0..64).all(|i| q[i] == q[128 - i]));
}

#[test]
fn periodic_window_is_the_symmetric_one_of_length_n_plus_one_without_the_last_value() {
    for n in [1usize, 4, 16, 100, 1024] {
        let (mut p, mut s) = (vec![0f32; n], vec![0f32; n + 1]);
        hann(&mut p, Symmetry::Periodic).unwrap();
        hann(&mut s, Symmetry::Symmetric).unwrap();
        assert_eq!(p[..], s[..n]);
        blackman_harris(&mut p, Symmetry::Periodic).unwrap();
        blackman_harris(&mut s, Symmetry::Symmetric).unwrap();
        assert_eq!(p[..], s[..n]);
        kaiser(&mut p, 6.5, Symmetry::Periodic).unwrap();
        kaiser(&mut s, 6.5, Symmetry::Symmetric).unwrap();
        assert_eq!(p[..], s[..n]);
    }
}

#[test]
fn window_generators_edge_cases() {
    let mut none: [f32; 0] = [];
    hann(&mut none, Symmetry::Periodic).unwrap();
    hann(&mut none, Symmetry::Symmetric).unwrap();
    let mut one = [7.0f32];
    hann(&mut one, Symmetry::Symmetric).unwrap();
    assert_eq!(one, [1.0]);
    blackman_harris(&mut one, Symmetry::Periodic).unwrap();
    assert!((one[0] - 6e-5).abs() < 1e-6); // periodic N = 1: the first value of a length-2 window
    let mut one = [0.0f32];
    kaiser(&mut one, 9.0, Symmetry::Symmetric).unwrap();
    assert_eq!(one, [1.0]);
    // Invalid beta.
    let mut w = [0f32; 8];
    for beta in [-0.1, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            kaiser(&mut w, beta, Symmetry::Symmetric).unwrap_err(),
            Error::InvalidParameter(_)
        ));
    }
    // beta = 0 is rectangular, the Kaiser peak is exactly 1.
    kaiser(&mut w, 0.0, Symmetry::Symmetric).unwrap();
    assert_eq!(w, [1.0; 8]);
    let mut k = vec![0f32; 129];
    kaiser(&mut k, 8.6, Symmetry::Symmetric).unwrap();
    assert_eq!(k[64], 1.0);
    assert!(k[0] > 0.0 && k[0] < 0.01);
}

#[test]
fn hann_values_and_window_gain() {
    let mut w = [0f32; 8];
    hann(&mut w, Symmetry::Periodic).unwrap();
    let want = [
        0.0,
        0.146_446_6,
        0.5,
        0.853_553_4,
        1.0,
        0.853_553_4,
        0.5,
        0.146_446_6,
    ];
    for (a, b) in w.iter().zip(want) {
        assert!((a - b).abs() < 1e-7, "{a} {b}");
    }
    let mut w = vec![0f32; 4096];
    hann(&mut w, Symmetry::Periodic).unwrap();
    let (cg, nebw) = window_gain(&w);
    assert!((cg - 0.5).abs() < 1e-9, "{cg}");
    assert!((nebw - 1.5).abs() < 1e-9, "{nebw}");
    // Rectangular window: gain 1, bandwidth 1.
    let (cg, nebw) = window_gain(&[1.0f32; 64]);
    assert_eq!((cg, nebw), (1.0, 1.0));
    assert_eq!(window_gain(&[]), (0.0, 0.0));
    // Blackman-Harris: coherent gain a0, bandwidth about 2.0044.
    let mut w = vec![0f32; 4096];
    blackman_harris(&mut w, Symmetry::Periodic).unwrap();
    let (cg, nebw) = window_gain(&w);
    assert!((cg - 0.35875).abs() < 1e-6);
    assert!((nebw - 2.0044).abs() < 1e-3, "{nebw}");
}

#[test]
fn bessel_i0_matches_known_values() {
    let cases = [
        (0.0, 1.0),
        (0.5, 1.063_483_370_741_323),
        (1.0, 1.266_065_877_752_008),
        (2.0, 2.279_585_302_336_067),
        (5.0, 27.239_871_823_604_44),
        (8.0, 427.564_115_721_804_8),
        (20.0, 43_558_282.559_553_5),
    ];
    for (x, want) in cases {
        let got = bessel_i0(x);
        assert!(
            ((got - want) / want).abs() < 1e-13,
            "I0({x}) = {got}, want {want}"
        );
        assert_eq!(bessel_i0(-x), got, "I0 is even");
    }
    assert!(bessel_i0(f64::NAN).is_nan());
}

#[test]
fn window_kernel_multiplies() {
    let x = random_c32(64, 22);
    let mut w = vec![0f32; 64];
    hann(&mut w, Symmetry::Symmetric).unwrap();
    let mut y = vec![C32::ZERO; 64];
    Fallback.window(&x, &mut y, &w).unwrap();
    for ((a, b), w) in y.iter().zip(&x).zip(&w) {
        assert_eq!(a.re.to_bits(), (b.re * w).to_bits());
        assert_eq!(a.im.to_bits(), (b.im * w).to_bits());
    }
    let x = random_cq15(64, 23);
    let mut wq = vec![Q15(0); 64];
    hann(&mut wq, Symmetry::Symmetric).unwrap();
    let mut y = vec![CQ15::ZERO; 64];
    Fallback.window(&x, &mut y, &wq).unwrap();
    for ((a, b), w) in y.iter().zip(&x).zip(&wq) {
        assert_eq!(a.re, sdrk_core::num::q15_mul(b.re, *w));
        assert_eq!(a.im, sdrk_core::num::q15_mul(b.im, *w));
    }
    // Full scale times the saturated peak and the -1 * -1 corner.
    let mut y = [CQ15::ZERO; 2];
    Fallback
        .window(
            &[q(-32768, 32767), q(-32768, -32768)],
            &mut y,
            &[Q15::MIN, Q15::MIN],
        )
        .unwrap();
    assert_eq!(y[0], q(32767, -32767));
    assert_eq!(y[1], q(32767, 32767));
    // Length checks cover the window too.
    let mut y = [C32::ZERO; 2];
    assert_eq!(
        Fallback.window(&[C32::ZERO; 2], &mut y, &[1.0]),
        Err(Error::LengthMismatch {
            expected: 2,
            got: 1
        })
    );
    assert!(
        Fallback
            .window(&[C32::ZERO; 2], &mut y[..1], &[1.0; 2])
            .is_err()
    );
}

// -------------------------------------------------------- mag_sq and power_db

#[test]
fn mag_sq_values() {
    let mut y = [0f32; 3];
    Fallback
        .mag_sq(
            &[C32::new(3.0, 4.0), C32::new(-0.5, 0.5), C32::ZERO],
            &mut y,
        )
        .unwrap();
    assert_eq!(y, [25.0, 0.5, 0.0]);
    let mut yq = [Q15(0); 5];
    Fallback
        .mag_sq(
            &[
                q(16384, 0),
                q(16384, 16384),
                q(-32768, -32768),
                q(32767, 32767),
                q(100, 0),
            ],
            &mut yq,
        )
        .unwrap();
    // 0.5^2 = 0.25 -> 8192; 2 * 0.25 = 0.5 -> 16384; full scale saturates; 100^2 / 32768 = 0.305 -> 0.
    assert_eq!(yq, [Q15(8192), Q15(16384), Q15::MAX, Q15::MAX, Q15(0)]);
    let mut yq = [Q15(9)];
    Fallback.mag_sq(&[q(181, 181)], &mut yq).unwrap();
    assert_eq!(yq[0], Q15(2)); // 65522 / 32768 = 1.9995 rounds to 2
    assert!(Fallback.mag_sq(&[C32::ZERO], &mut y).is_err());
}

#[test]
fn power_db_values_and_floor() {
    let mut y = [0f32; 8];
    let p = [1.0f32, 0.1, 0.01, 1e-20, 1e-30, 0.0, -1.0, f32::NAN];
    Fallback.power_db(&p, &mut y, 0.0).unwrap();
    let want = [0.0, -10.0, -20.0, -200.0, -200.0, -200.0, -200.0, -200.0];
    for (a, b) in y.iter().zip(want) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
    // The offset is added after the logarithm, also for the floor.
    Fallback.power_db(&p, &mut y, 3.0).unwrap();
    assert!((y[0] - 3.0).abs() < 1e-6);
    assert!((y[7] + 197.0).abs() < 1e-4);
    // Infinity stays infinite, negative infinity is the floor.
    let mut y2 = [0f32; 2];
    Fallback
        .power_db(&[f32::INFINITY, f32::NEG_INFINITY], &mut y2, 0.0)
        .unwrap();
    assert_eq!(y2[0], f32::INFINITY);
    assert!((y2[1] + 200.0).abs() < 1e-4);
    // Q15: raw / 32768 is the linear power.
    let mut yq = [0f32; 4];
    Fallback
        .power_db(&[Q15(16384), Q15(32767), Q15(0), Q15(-5)], &mut yq, 0.0)
        .unwrap();
    assert!((yq[0] + 3.0103).abs() < 1e-3);
    assert!(yq[1].abs() < 1e-3);
    assert!((yq[2] + 200.0).abs() < 1e-4 && (yq[3] + 200.0).abs() < 1e-4);
    assert!(Fallback.power_db(&[1.0f32], &mut yq, 0.0).is_err());
}

#[test]
fn power_db_never_panics_on_any_f32_bit_pattern() {
    let mut s = 99u64;
    let p: Vec<f32> = (0..20_000).map(|_| f32::from_bits(lcg(&mut s))).collect();
    let mut y = vec![0f32; p.len()];
    Fallback.power_db(&p, &mut y, 0.0).unwrap();
    for (&p, &d) in p.iter().zip(&y) {
        if p > 1e-20 && p.is_finite() {
            assert!(d.is_finite());
        } else if !p.is_finite() && p > 0.0 {
            assert_eq!(d, f32::INFINITY);
        } else {
            assert!((d + 200.0).abs() < 1e-4, "{p} -> {d}");
        }
    }
}

// ---------------------------------------------------------------------- misc

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<FirDecimState<C32, Vec<f32>, Vec<C32>>>();
    assert_sync::<FirDecimState<CQ15, [Q15; 4], [CQ15; 3]>>();
    assert_send::<FirDecimState<CQ15, &'static [Q15], &'static mut [CQ15]>>();
    assert_send::<ResampleState<C32, Vec<f32>, Vec<C32>>>();
    assert_sync::<ResampleState<CQ15, [Q15; 4], [CQ15; 2]>>();
    assert_send::<ResampleRealState<f32, Vec<f32>, Vec<f32>>>();
    assert_sync::<ResampleRealState<Q15, [Q15; 4], [Q15; 2]>>();
    assert_send::<QuadDemodState<C32>>();
    assert_sync::<QuadDemodState<CQ15>>();
    assert_send::<DeemphasisState<f32>>();
    assert_sync::<DeemphasisState<Q15>>();
    assert_send::<DcBlockState<C32>>();
    assert_sync::<DcBlockState<CQ15>>();
    assert_send::<Symmetry>();
    assert_sync::<Symmetry>();
}

#[cfg(feature = "reference")]
mod against_the_reference {
    use super::*;
    use sdrk_core::Reference;

    fn snr_db(want: &[f64], got: &[f64]) -> f64 {
        let sig: f64 = want.iter().map(|v| v * v).sum();
        let err: f64 = want.iter().zip(got).map(|(a, b)| (a - b) * (a - b)).sum();
        if err == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (sig / err).log10()
        }
    }

    fn flat(x: &[C32]) -> Vec<f64> {
        x.iter()
            .flat_map(|s| [f64::from(s.re), f64::from(s.im)])
            .collect()
    }

    #[test]
    fn q15_decimator_and_resampler_are_bit_exact() {
        for (up, down) in [(1usize, 1usize), (1, 5), (3, 2), (7, 3), (2, 9)] {
            let taps = taps_q15(30, 30);
            let x = random_cq15(150, 31);
            let hist = 30usize.div_ceil(up);
            let mut a =
                ResampleState::<CQ15, _, _>::new(taps.clone(), vec![CQ15::ZERO; hist], up, down)
                    .unwrap();
            let mut b = a.clone();
            let (mut ya, mut yb) = (
                vec![CQ15::ZERO; 150 * up + 2],
                vec![CQ15::ZERO; 150 * up + 2],
            );
            let pa = Fallback.resample(&mut a, &x, &mut ya).unwrap();
            let pb = Reference.resample(&mut b, &x, &mut yb).unwrap();
            assert_eq!(pa, pb, "L={up} M={down}");
            assert_eq!(ya, yb, "L={up} M={down}");
            assert_eq!(a.history(), b.history());
            assert_eq!(a.phase(), b.phase());
        }
        for m in [1usize, 2, 7, 10] {
            let taps = taps_q15(25, 32);
            let x = random_cq15(200, 33);
            let mut a = FirDecimState::<CQ15, _, _>::new(taps, vec![CQ15::ZERO; 24], m).unwrap();
            let mut b = a.clone();
            let (mut ya, mut yb) = (vec![CQ15::ZERO; 200], vec![CQ15::ZERO; 200]);
            let pa = Fallback.fir_decimate(&mut a, &x, &mut ya).unwrap();
            let pb = Reference.fir_decimate(&mut b, &x, &mut yb).unwrap();
            assert_eq!((pa, &ya), (pb, &yb));
            assert_eq!((a.history(), a.countdown()), (b.history(), b.countdown()));
        }
        // Real Q15 resampling.
        let taps = taps_q15(20, 34);
        let x = random_q15(120, 35);
        let mut a = ResampleRealState::<Q15, _, _>::new(taps, vec![Q15(0); 7], 3, 4).unwrap();
        let mut b = a.clone();
        let (mut ya, mut yb) = (vec![Q15(0); 130], vec![Q15(0); 130]);
        let pa = Fallback.resample_real(&mut a, &x, &mut ya).unwrap();
        let pb = Reference.resample_real(&mut b, &x, &mut yb).unwrap();
        assert_eq!((pa, ya), (pb, yb));
    }

    #[test]
    fn f32_decimator_and_resampler_reach_110_db() {
        let taps = taps_f32(63, 36);
        let x = random_c32(2000, 37);
        let mut a = FirDecimState::<C32, _, _>::new(taps.clone(), vec![C32::ZERO; 62], 10).unwrap();
        let mut b = a.clone();
        let (mut ya, mut yb) = (vec![C32::ZERO; 200], vec![C32::ZERO; 200]);
        Fallback.fir_decimate(&mut a, &x, &mut ya).unwrap();
        Reference.fir_decimate(&mut b, &x, &mut yb).unwrap();
        assert!(snr_db(&flat(&yb), &flat(&ya)) > 110.0);
        let mut a = ResampleState::<C32, _, _>::new(taps, vec![C32::ZERO; 21], 3, 2).unwrap();
        let mut b = a.clone();
        let (mut ya, mut yb) = (vec![C32::ZERO; 3001], vec![C32::ZERO; 3001]);
        let pa = Fallback.resample(&mut a, &x, &mut ya).unwrap();
        let pb = Reference.resample(&mut b, &x, &mut yb).unwrap();
        assert_eq!(pa, pb);
        assert!(snr_db(&flat(&yb), &flat(&ya)) > 110.0);
    }

    #[test]
    fn demod_chain_kernels_agree_with_the_reference() {
        let x = random_cq15(500, 38);
        let (mut a, mut b) = (QuadDemodState::<CQ15>::new(), QuadDemodState::<CQ15>::new());
        let (mut ya, mut yb) = (vec![Q15(0); 500], vec![Q15(0); 500]);
        Fallback
            .quad_demod(&mut a, &x, &mut ya, Q15(20000))
            .unwrap();
        Reference
            .quad_demod(&mut b, &x, &mut yb, Q15(20000))
            .unwrap();
        // Double rounding in the fallback: at most one LSB apart.
        assert!(
            ya.iter()
                .zip(&yb)
                .all(|(p, r)| (i32::from(p.0) - i32::from(r.0)).abs() <= 1)
        );
        assert_eq!(a, b);

        let rate = Rate::hz(48_000).unwrap();
        let x = random_q15(1000, 39);
        let mut a = DeemphasisState::<Q15>::new(50e-6, rate).unwrap();
        let mut b = a;
        let (mut ya, mut yb) = (vec![Q15(0); 1000], vec![Q15(0); 1000]);
        Fallback.deemphasis(&mut a, &x, &mut ya).unwrap();
        Reference.deemphasis(&mut b, &x, &mut yb).unwrap();
        assert_eq!((ya, a), (yb, b));

        let x = random_cq15(1000, 40);
        let mut a = DcBlockState::<CQ15>::new(Q15(32735)).unwrap();
        let mut b = a;
        let (mut ya, mut yb) = (vec![CQ15::ZERO; 1000], vec![CQ15::ZERO; 1000]);
        Fallback.dc_block(&mut a, &x, &mut ya).unwrap();
        Reference.dc_block(&mut b, &x, &mut yb).unwrap();
        assert_eq!((ya, a), (yb, b));

        let x = random_c32(1000, 41);
        let mut a = DcBlockState::<C32>::new(0.999).unwrap();
        let mut b = a;
        let (mut ya, mut yb) = (vec![C32::ZERO; 1000], vec![C32::ZERO; 1000]);
        Fallback.dc_block(&mut a, &x, &mut ya).unwrap();
        Reference.dc_block(&mut b, &x, &mut yb).unwrap();
        assert!(snr_db(&flat(&yb), &flat(&ya)) > 100.0);
    }

    #[test]
    fn window_mag_sq_and_power_db_agree_with_the_reference() {
        let x = random_c32(500, 42);
        let mut w = vec![0f32; 500];
        kaiser(&mut w, 5.0, Symmetry::Periodic).unwrap();
        let (mut a, mut b) = (vec![C32::ZERO; 500], vec![C32::ZERO; 500]);
        Fallback.window(&x, &mut a, &w).unwrap();
        Reference.window(&x, &mut b, &w).unwrap();
        assert_eq!(
            bytemuck::cast_slice::<C32, u32>(&a),
            bytemuck::cast_slice::<C32, u32>(&b)
        );
        let (mut ma, mut mb) = (vec![0f32; 500], vec![0f32; 500]);
        Fallback.mag_sq(&x, &mut ma).unwrap();
        Reference.mag_sq(&x, &mut mb).unwrap();
        let (fa, fb): (Vec<f64>, Vec<f64>) = (
            ma.iter().map(|&v| f64::from(v)).collect(),
            mb.iter().map(|&v| f64::from(v)).collect(),
        );
        assert!(snr_db(&fb, &fa) > 140.0);
        let xq = random_cq15(500, 43);
        let (mut qa, mut qb) = (vec![Q15(0); 500], vec![Q15(0); 500]);
        Fallback.mag_sq(&xq, &mut qa).unwrap();
        Reference.mag_sq(&xq, &mut qb).unwrap();
        assert_eq!(qa, qb);
        let p: Vec<f32> = ma.iter().map(|v| v * 1e-3).collect();
        let (mut da, mut db) = (vec![0f32; 500], vec![0f32; 500]);
        Fallback.power_db(&p, &mut da, 1.5).unwrap();
        Reference.power_db(&p, &mut db, 1.5).unwrap();
        assert!(da.iter().zip(&db).all(|(a, b)| (a - b).abs() < 1e-4));
        let (mut da, mut db) = (vec![0f32; 500], vec![0f32; 500]);
        Fallback.power_db(&qa, &mut da, 0.0).unwrap();
        Reference.power_db(&qa, &mut db, 0.0).unwrap();
        assert!(da.iter().zip(&db).all(|(a, b)| (a - b).abs() < 1e-4));
    }
}
