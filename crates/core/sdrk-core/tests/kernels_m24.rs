//! Tests of the M24 items of `sdrk-core`: the construction rules of the stereo states, timing,
//! the behaviour of the pilot loop (lock, offset, no pilot, enable and disable), the matrix
//! corners and the thread-safety of the public types.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::items_after_statements,
    clippy::indexing_slicing,
    clippy::too_many_lines
)]

use sdrk_core::kernels::stereo::{StereoMatrix, StereoPll, StereoPllState};
use sdrk_core::{
    C32, CQ15, ComplexSample, Delay, Error, Fallback, Q15, Rate, Ratio, RealSample, Sample, Timing,
};
use sdrk_device::{ErrorKind, SdrkError};

const FS: u64 = 480_000;

fn rate() -> Rate {
    Rate::hz(FS).unwrap()
}

fn level_of<T: RealSample>() -> f64 {
    if T::TYPE == sdrk_core::SampleType::Q15 {
        0.0625
    } else {
        0.1
    }
}

fn new_state<T: RealSample>() -> StereoPllState<T> {
    StereoPllState::new(19_000.0, 20.0, level_of::<T>(), rate()).unwrap()
}

#[test]
fn construction_validates_every_parameter() {
    let r = rate();
    assert!(StereoPllState::<f32>::new(19_000.0, 20.0, 0.1, r).is_ok());
    assert!(StereoPllState::<Q15>::new(19_000.0, 20.0, 0.0625, r).is_ok());
    // pilot at or above fs / 4, zero, negative, not finite
    let quarter = FS as f64 / 4.0;
    for pilot in [
        quarter,
        quarter + 1.0,
        0.0,
        -19_000.0,
        f64::NAN,
        f64::INFINITY,
    ] {
        let e = StereoPllState::<f32>::new(pilot, 1.0, 0.1, r).unwrap_err();
        assert!(matches!(e, Error::InvalidParameter(_)), "{pilot}");
        assert_eq!(e.kind(), ErrorKind::InvalidArgument);
        assert!(StereoPllState::<Q15>::new(pilot, 1.0, 0.1, r).is_err());
    }
    assert!(StereoPllState::<f32>::new(quarter - 1.0, 1.0, 0.1, r).is_ok());
    // natural_hz: 0, above pilot / 100, negative, NaN; the limit itself is allowed
    for natural in [0.0, 190.1, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            StereoPllState::<f32>::new(19_000.0, natural, 0.1, r).is_err(),
            "{natural}"
        );
    }
    assert!(StereoPllState::<f32>::new(19_000.0, 190.0, 0.1, r).is_ok());
    // pilot level: 0, negative, above 1, NaN
    for level in [0.0, -0.1, 1.0001, f64::NAN, f64::INFINITY] {
        assert!(
            StereoPllState::<f32>::new(19_000.0, 20.0, level, r).is_err(),
            "{level}"
        );
    }
    assert!(StereoPllState::<f32>::new(19_000.0, 20.0, 1.0, r).is_ok());
}

#[test]
fn a_q15_gain_that_does_not_fit_is_rejected() {
    // A tiny pilot level makes the detector gain huge: kp_q 2^30 + imax_q overflows i64, and f32
    // still holds the coefficients.
    let r = rate();
    let e = StereoPllState::<Q15>::new(19_000.0, 20.0, 1e-9, r).unwrap_err();
    assert!(matches!(e, Error::InvalidParameter(_)));
    assert!(StereoPllState::<f32>::new(19_000.0, 20.0, 1e-9, r).is_ok());
    // A very low natural frequency: ki rounds to zero in Q15 (ki 2^34 < 0.5), f32 is fine.
    let e = StereoPllState::<Q15>::new(19_000.0, 1e-6, 0.0625, r).unwrap_err();
    assert!(matches!(e, Error::InvalidParameter(_)));
    assert!(StereoPllState::<f32>::new(19_000.0, 1e-6, 0.0625, r).is_ok());
    // A rate so high that the lock coefficient rounds to zero: beta 2^31 < 0.5.
    let fast = Rate::hz(4_000_000_000).unwrap();
    assert!(StereoPllState::<Q15>::new(19_000.0, 100.0, 0.0625, fast).is_err());
}

#[test]
fn a_new_state_is_not_stereo_and_enabled() {
    let st = new_state::<f32>();
    assert!(!st.is_stereo() && st.is_enabled() && st.phase() == 0);
    assert_eq!(st.integrator(), 0.0);
    assert_eq!(st.lock_level(), 0.0);
    let c = st.coefficients();
    assert_eq!(
        c.step0,
        (19_000.0 / 480_000.0 * 4_294_967_296.0 + 0.5) as u32
    );
    assert_eq!(c.imax, 50.0 / 480_000.0);
    assert!((c.on / c.off - 4.0 / 3.0).abs() < 1e-12);
}

#[test]
fn timing_is_one_to_one_without_delay() {
    let c = new_state::<f32>();
    let q = new_state::<Q15>();
    assert_eq!(c.ratio(), Ratio::ONE);
    assert_eq!(q.ratio(), Ratio::ONE);
    assert_eq!(c.group_delay(), Delay::ZERO);
    assert_eq!(q.group_delay(), Delay::ZERO);
}

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send<T: Send + Sync>() {}
    assert_send::<StereoPllState<f32>>();
    assert_send::<StereoPllState<Q15>>();
    assert_send::<sdrk_core::kernels::stereo::StereoPllCoefficients>();
}

// ---------------------------------------------------------------------------------------------
// Behaviour of the loop
// ---------------------------------------------------------------------------------------------

/// Deterministic normal noise (sum of twelve uniform values), a different stream per seed.
struct Noise(u64);
impl Noise {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / 9_007_199_254_740_992.0
    }
    fn normal(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

/// The pilot starts a seventh of a turn ahead of the loop.
const START_TURNS: f64 = 1.0 / 7.0;

fn pilot_at(hz: f64, amp: f64, n: usize) -> f64 {
    let turns = (hz * n as f64 / FS as f64 + START_TURNS).fract();
    amp * (2.0 * core::f64::consts::PI * turns).sin()
}

fn conv<T: RealSample>(v: f64) -> T {
    T::from_f64_round(v)
}

/// Runs the loop sample by sample on a clean pilot of `hz` and returns the phase error in
/// degrees, wrapped to +-180, for every sample.
fn phase_errors<T: RealSample>(hz: f64, count: usize) -> Vec<f64>
where
    Fallback: StereoPll<T>,
{
    let mut st = new_state::<T>();
    let amp = level_of::<T>();
    let mut out = vec![<T::Complex as Sample>::ZERO; 1];
    let mut errors = Vec::new();
    for n in 0..count {
        let p = conv::<T>(pilot_at(hz, amp, n));
        // The detector uses the phase before the update: compare the phase of this sample.
        let loop_turns = f64::from(st.phase()) / 4_294_967_296.0;
        let want = (hz * n as f64 / FS as f64 + START_TURNS).fract();
        let mut d = (want - loop_turns).rem_euclid(1.0);
        if d > 0.5 {
            d -= 1.0;
        }
        errors.push(d * 360.0);
        Fallback.stereo_pll(&mut st, &[p], &[p], &mut out).unwrap();
    }
    errors
}

fn check_settles<T: RealSample>(hz: f64)
where
    Fallback: StereoPll<T>,
{
    let tenth = FS as usize / 10;
    let errors = phase_errors::<T>(hz, 3 * tenth);
    // Within 0.1 s the error is below 0.5 degrees and stays there.
    let worst = errors[tenth..].iter().fold(0.0f64, |m, e| m.max(e.abs()));
    assert!(
        worst < 0.5,
        "{hz} Hz: phase error {worst} degrees after 0.1 s"
    );
    // ... and the test is not vacuous: the loop starts 51 degrees away.
    assert!(errors[0].abs() > 50.0);
}

#[test]
fn a_clean_pilot_settles_below_half_a_degree_within_a_tenth_of_a_second() {
    check_settles::<f32>(19_000.0);
    check_settles::<Q15>(19_000.0);
}

#[test]
fn an_offset_pilot_settles_too() {
    check_settles::<f32>(19_002.0);
    check_settles::<Q15>(19_002.0);
    // The integrator carries the offset: 2 Hz is 2 / fs turns per sample.
    let mut st = new_state::<f32>();
    let mut out = vec![C32::ZERO; 1];
    for n in 0..100_000 {
        let p = pilot_at(19_002.0, 0.1, n) as f32;
        Fallback.stereo_pll(&mut st, &[p], &[p], &mut out).unwrap();
    }
    let want = 2.0 / FS as f64;
    assert!(
        (st.integrator() - want).abs() < 0.05 * want,
        "{}",
        st.integrator()
    );
    assert!(st.integrator().abs() <= st.integrator_limit());
    assert!(st.is_stereo());
}

fn no_pilot_stays_mono<T: RealSample>()
where
    Fallback: StereoPll<T>,
{
    let mut st = new_state::<T>();
    let mut noise = Noise(7);
    let n = FS as usize / 2;
    let mpx: Vec<T> = (0..n).map(|_| conv(0.3 * noise.normal())).collect();
    let pilot: Vec<T> = (0..n).map(|_| conv(0.3 * noise.normal())).collect();
    let mut out = vec![<T::Complex as Sample>::ZERO; n];
    for (m, (p, o)) in mpx
        .chunks(997)
        .zip(pilot.chunks(997).zip(out.chunks_mut(997)))
    {
        Fallback.stereo_pll(&mut st, m, p, o).unwrap();
        assert!(!st.is_stereo());
        assert!(st.integrator().abs() <= st.integrator_limit());
    }
    assert!(out.iter().all(|s| s.im().to_f64() == 0.0));
}

#[test]
fn without_a_pilot_the_integrator_is_limited_and_the_output_is_mono() {
    no_pilot_stays_mono::<f32>();
    no_pilot_stays_mono::<Q15>();
}

fn the_lock_detector_switches_off_after_the_pilot_vanishes<T: RealSample>()
where
    Fallback: StereoPll<T>,
{
    let amp = level_of::<T>();
    let mut st = new_state::<T>();
    let mut out = [<T::Complex as Sample>::ZERO];
    let mut on_at = None;
    let mut off_at = None;
    for n in 0..FS as usize {
        // Pilot for the first 0.3 s, then silence.
        let p = conv::<T>(if n < 144_000 {
            pilot_at(19_000.0, amp, n)
        } else {
            0.0
        });
        Fallback.stereo_pll(&mut st, &[p], &[p], &mut out).unwrap();
        if st.is_stereo() && on_at.is_none() {
            on_at = Some(n);
        }
        if on_at.is_some() && !st.is_stereo() && off_at.is_none() {
            off_at = Some(n);
        }
    }
    // 50 ms time constant: on after about 0.5 tau (25 ms), off about 1.2 tau (60 ms) after the
    // pilot has gone.
    let on = on_at.unwrap() as f64 / FS as f64;
    let off = (off_at.unwrap() as f64 - 144_000.0) / FS as f64;
    assert!((0.015..0.05).contains(&on), "on after {on} s");
    assert!((0.04..0.09).contains(&off), "off after {off} s");
}

#[test]
fn the_lock_decision_has_hysteresis_in_both_directions() {
    the_lock_detector_switches_off_after_the_pilot_vanishes::<f32>();
    the_lock_detector_switches_off_after_the_pilot_vanishes::<Q15>();
}

fn enable_disable<T: RealSample>()
where
    Fallback: StereoPll<T>,
{
    let amp = level_of::<T>();
    let mut a = new_state::<T>();
    let mut b = new_state::<T>();
    let (lock, off, total) = (FS as usize / 5, FS as usize / 10, FS as usize / 5);
    let mut pos = 0;
    let step = |a: &mut StereoPllState<T>, b: &mut StereoPllState<T>, n: usize| {
        let mut oa = [<T::Complex as Sample>::ZERO];
        let mut ob = [<T::Complex as Sample>::ZERO];
        // x is the multiplex signal: a 38 kHz carrier-like tone so that the difference output
        // is not accidentally zero.
        let x = conv::<T>(0.5 * pilot_at(38_000.0, 1.0, n) + 0.3);
        let p = conv::<T>(pilot_at(19_000.0, amp, n));
        Fallback.stereo_pll(a, &[x], &[p], &mut oa).unwrap();
        Fallback.stereo_pll(b, &[x], &[p], &mut ob).unwrap();
        (oa[0], ob[0])
    };
    while pos < lock {
        step(&mut a, &mut b, pos);
        pos += 1;
    }
    assert!(a.is_stereo());
    // b runs on enabled as the control; a is disabled from the next sample.
    a.set_enabled(false);
    for _ in 0..off {
        let (oa, ob) = step(&mut a, &mut b, pos);
        pos += 1;
        assert_eq!(oa.im().to_f64(), 0.0, "disabled output at {pos}");
        assert_eq!(oa.re(), ob.re());
        // the loop keeps running: the same phase and decision as the control
        assert_eq!(a.phase(), b.phase());
        assert_eq!(a.is_stereo(), b.is_stereo());
    }
    assert!(a.is_stereo(), "is_stereo is independent of enabled");
    a.set_enabled(true);
    let mut seen = false;
    for _ in 0..total {
        let (oa, ob) = step(&mut a, &mut b, pos);
        pos += 1;
        assert_eq!(oa, ob, "restored output at {pos} without a new lock time");
        seen |= oa.im().to_f64() != 0.0;
    }
    assert!(seen);
    // reset keeps the enabled flag and clears the loop
    a.set_enabled(false);
    a.reset();
    assert!(!a.is_enabled() && a.phase() == 0 && !a.is_stereo());
    assert_eq!((a.integrator(), a.lock_level()), (0.0, 0.0));
}

#[test]
fn disabling_zeroes_the_difference_and_enabling_restores_it_immediately() {
    enable_disable::<f32>();
    enable_disable::<Q15>();
}

#[test]
fn unequal_lengths_are_rejected_and_change_nothing() {
    let mut st = new_state::<f32>();
    let before = st;
    let mut out = [C32::new(7.0, 7.0); 4];
    for (m, p, o) in [(4, 3, 4), (3, 4, 4), (4, 4, 3)] {
        let e = Fallback
            .stereo_pll(&mut st, &vec![0.1; m], &vec![0.1; p], &mut out[..o])
            .unwrap_err();
        assert!(matches!(e, Error::LengthMismatch { .. }));
        assert_eq!(st, before);
        assert_eq!(out[0], C32::new(7.0, 7.0));
        #[cfg(feature = "reference")]
        {
            let e = sdrk_core::Reference
                .stereo_pll(&mut st, &vec![0.1; m], &vec![0.1; p], &mut out[..o])
                .unwrap_err();
            assert!(matches!(e, Error::LengthMismatch { .. }));
        }
    }
    let mut o = [C32::new(7.0, 7.0); 2];
    let e = Fallback.stereo_matrix(&[C32::ZERO; 3], &mut o).unwrap_err();
    assert_eq!(
        e,
        Error::LengthMismatch {
            expected: 3,
            got: 2
        }
    );
    assert_eq!(o[0], C32::new(7.0, 7.0));
    let mut o = [CQ15::ZERO; 2];
    assert!(Fallback.stereo_matrix(&[CQ15::ZERO; 3], &mut o).is_err());
}

// ---------------------------------------------------------------------------------------------
// stereo_matrix
// ---------------------------------------------------------------------------------------------

#[test]
fn the_c32_matrix_is_the_exact_sum_and_difference() {
    let input = [
        C32::new(0.5, 0.125),
        C32::new(-1.0, 0.25),
        C32::new(f32::NAN, 1.0),
        C32::new(f32::INFINITY, f32::INFINITY),
        C32::new(0.0, -0.0),
    ];
    let mut out = [C32::ZERO; 5];
    Fallback.stereo_matrix(&input, &mut out).unwrap();
    assert_eq!(out[0], C32::new(0.75, 0.25));
    assert_eq!(out[1], C32::new(-0.5, -1.5));
    assert!(out[2].re.is_nan() && out[2].im.is_nan());
    assert!(out[3].re.is_infinite() || out[3].re.is_nan());
    assert!(out[3].im.is_nan(), "inf - inf");
    // 0 + (-0 + -0) = 0 and 0 - (-0) = 0
    assert_eq!(out[4].re.to_bits(), 0.0f32.to_bits());
    assert_eq!(out[4].im.to_bits(), 0.0f32.to_bits());
}

#[test]
fn the_cq15_matrix_saturates_once_at_the_corners() {
    let q = |s: i16, d: i16| CQ15 {
        re: Q15(s),
        im: Q15(d),
    };
    let input = [
        q(0, 0),
        q(1000, 100),
        q(32767, 1),
        q(-32768, -1),
        q(32767, 32767),
        q(-32768, -32768),
        q(-32768, 32767),
        q(32767, -32768),
        q(30000, -16384),
    ];
    let mut out = [CQ15::ZERO; 9];
    Fallback.stereo_matrix(&input, &mut out).unwrap();
    let want = [
        (0, 0),
        (1200, 800),
        (32767, 32765),
        (-32768, -32766),
        (32767, -32767),
        (-32768, 32767),
        (32766, -32768),
        (-32768, 32767),
        (-2768, 32767),
    ];
    for (i, (o, (l, r))) in out.iter().zip(want).enumerate() {
        assert_eq!((o.re.0, o.im.0), (l, r), "{:?}", input[i]);
    }
    // s + 2 d is exact before the single saturation: -32768 + 2 * 32767 = 32766 is not
    // saturated, unlike a chain that saturates d + d first (CMSIS: 32767 - 32768 = -1).
    let mut o = [CQ15::ZERO];
    Fallback.stereo_matrix(&[q(-32768, 32767)], &mut o).unwrap();
    assert_eq!(o[0].re.0, 32766);
}

/// A pilot 80 Hz away from the loop (beyond the 50 Hz of the integrator limit) saturates
/// the integrator; `integrator_limit()` is the value that the active type clamps to, so the
/// saturated integrator equals it at every rate, also where `50 / fs` rounded to `f32` is above
/// the `f64` value.
fn saturates_at_the_limit<T: RealSample>(fs: u64)
where
    Fallback: StereoPll<T>,
{
    let rate = Rate::hz(fs).unwrap();
    let mut st = StereoPllState::<T>::new(19_000.0, 20.0, level_of::<T>(), rate).unwrap();
    let n = 100_000;
    let mpx: Vec<T> = (0..n)
        .map(|k| conv(0.5 * (2.0 * core::f64::consts::PI * 19_080.0 * k as f64 / fs as f64).sin()))
        .collect();
    let mut out = vec![<T::Complex as Sample>::ZERO; n];
    Fallback.stereo_pll(&mut st, &mpx, &mpx, &mut out).unwrap();
    assert!(
        st.integrator().abs() <= st.integrator_limit(),
        "fs {fs}: {} against {}",
        st.integrator(),
        st.integrator_limit()
    );
    assert!(
        st.integrator().abs() > 0.99 * st.integrator_limit(),
        "fs {fs}: the integrator did not saturate"
    );
}

#[test]
fn the_integrator_limit_is_the_clamp_of_the_active_type() {
    for fs in [96_000, 120_000, 192_000, 240_000, 480_000, 960_000] {
        saturates_at_the_limit::<f32>(fs);
        saturates_at_the_limit::<Q15>(fs);
    }
    // The limits of the two types are the clamps themselves.
    for fs in [96_000_u64, 120_000, 480_000] {
        let rate = Rate::hz(fs).unwrap();
        let f = StereoPllState::<f32>::new(19_000.0, 20.0, 0.1, rate).unwrap();
        assert_eq!(f.integrator_limit(), f64::from((50.0 / fs as f64) as f32));
        let q = StereoPllState::<Q15>::new(19_000.0, 20.0, 0.0625, rate).unwrap();
        assert!((q.integrator_limit() - 50.0 / fs as f64).abs() < 1e-15);
    }
}
