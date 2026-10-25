//! Tests of the `squelch` kernel of M19 that need the testkit: block-size invariance over
//! 20 chunk plans per type, the gate semantics, `set_thresholds` timing, extreme inputs and
//! the equality of `Fallback`, `Reference` and the facade.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

use sdrk::Sdrk;
use sdrk_core::kernels::squelch::{Squelch, SquelchState};
use sdrk_core::{C32, CQ15, Fallback, Progress, Q15, Rate, Reference, Sample};
use sdrk_testkit::{ChunkPlan, Pcg32, bit_exact, check_block_invariance};

const FS: u64 = 48_000;
const TAU: f64 = 0.002;
const N: usize = 6000;

fn rate() -> Rate {
    Rate::hz(FS).unwrap()
}

/// Noise with loud and quiet stretches so the gate opens and closes several times.
fn amplitude(i: usize) -> f64 {
    match (i / 750) % 4 {
        0 => 0.001,
        1 => 0.15,
        2 => 0.0006,
        _ => 0.04,
    }
}

fn c32_input(rng: &mut Pcg32) -> Vec<C32> {
    (0..N)
        .map(|i| {
            let a = amplitude(i);
            C32::new((rng.normal() * a) as f32, (rng.normal() * a) as f32)
        })
        .collect()
}

fn cq_input(rng: &mut Pcg32) -> Vec<CQ15> {
    let q = |v: f64| Q15((v * 32768.0).round().clamp(-32768.0, 32767.0) as i16);
    (0..N)
        .map(|i| {
            let a = amplitude(i);
            CQ15 {
                re: q(rng.normal() * a),
                im: q(rng.normal() * a),
            }
        })
        .collect()
}

fn same_rate<I: Copy, O>(
    input: &[I],
    output: &mut [O],
    kernel: impl FnOnce(&[I], &mut [O]) -> sdrk_core::Result<()>,
) -> sdrk_core::Result<Progress> {
    let n = input.len().min(output.len());
    kernel(&input[..n], &mut output[..n])?;
    Ok(Progress {
        consumed: n,
        produced: n,
    })
}

#[test]
fn squelch_is_block_size_invariant() {
    let mut rng = Pcg32::new(91, 1);
    let (c32, cq) = (c32_input(&mut rng), cq_input(&mut rng));
    for seed in 0..20u64 {
        let plan = ChunkPlan::new(seed, 1 + (seed as usize * 37) % 500);
        let (mut a, mut b) = (vec![C32::ZERO; N], vec![C32::ZERO; N]);
        let r = check_block_invariance(
            &c32,
            &mut a,
            &mut b,
            plan,
            || SquelchState::<C32>::new(-30.0, 3.0, TAU, rate()).unwrap(),
            |st, i, o| same_rate(i, o, |i, o| Sdrk.squelch(st, i, o)),
        )
        .unwrap();
        assert_eq!(r.outputs, N);
        let closed = a.iter().filter(|s| **s == C32::ZERO).count();
        assert!(closed > 500 && closed < N - 500, "the gate must toggle");
        let (mut a, mut b) = (vec![CQ15::ZERO; N], vec![CQ15::ZERO; N]);
        check_block_invariance(
            &cq,
            &mut a,
            &mut b,
            plan,
            || SquelchState::<CQ15>::new(-30.0, 3.0, TAU, rate()).unwrap(),
            |st, i, o| same_rate(i, o, |i, o| Sdrk.squelch(st, i, o)),
        )
        .unwrap();
        let closed = a.iter().filter(|s| **s == CQ15::ZERO).count();
        assert!(closed > 500 && closed < N - 500, "the gate must toggle");
    }
}

#[test]
fn fallback_reference_and_facade_agree_bit_for_bit() {
    let mut rng = Pcg32::new(92, 1);
    let (c32, cq) = (c32_input(&mut rng), cq_input(&mut rng));
    let new32 = || SquelchState::<C32>::new(-35.0, 4.0, TAU, rate()).unwrap();
    let (mut sa, mut sb, mut sc) = (new32(), new32(), new32());
    let (mut a, mut b, mut c) = (vec![C32::ZERO; N], vec![C32::ZERO; N], vec![C32::ZERO; N]);
    Fallback.squelch(&mut sa, &c32, &mut a).unwrap();
    Reference.squelch(&mut sb, &c32, &mut b).unwrap();
    Sdrk.squelch(&mut sc, &c32, &mut c).unwrap();
    assert_eq!(bit_exact(&a, &b), None);
    assert_eq!(bit_exact(&a, &c), None);
    assert_eq!(sa, sb);
    assert_eq!(sa, sc);
    let newq = || SquelchState::<CQ15>::new(-35.0, 4.0, TAU, rate()).unwrap();
    let (mut sa, mut sb) = (newq(), newq());
    let (mut a, mut b) = (vec![CQ15::ZERO; N], vec![CQ15::ZERO; N]);
    Fallback.squelch(&mut sa, &cq, &mut a).unwrap();
    Reference.squelch(&mut sb, &cq, &mut b).unwrap();
    assert_eq!(bit_exact(&a, &b), None);
    assert_eq!(sa, sb);
}

/// `P_open` and `P_close` of the state as the kernel computes them (f32 for C32).
fn levels32(threshold_db: f64, hysteresis_db: f64) -> (f64, f64) {
    let lin = |db: f64| f64::from((10f64.powf(db / 10.0)) as f32);
    (lin(threshold_db), lin(threshold_db - hysteresis_db))
}

#[test]
fn c32_gate_opens_holds_in_the_band_and_closes_with_exact_zeros() {
    let (open_p, close_p) = levels32(-40.0, 3.0);
    let mut st = SquelchState::<C32>::new(-40.0, 3.0, TAU, rate()).unwrap();
    assert!(!st.is_open());
    assert_eq!(st.average_power(), 0.0);

    // Loud tone: the gate opens on the first sample where p >= P_open, and not before.
    let loud = C32::new(0.05, 0.0); // power 2.5e-3, -26 dBFS
    let mut first_open = None;
    for i in 0..400 {
        let mut out = [C32::new(7.0, 7.0)];
        Sdrk.squelch(&mut st, &[loud], &mut out).unwrap();
        let p = st.average_power();
        assert_eq!(st.is_open(), p >= open_p, "sample {i}: p = {p}");
        if st.is_open() {
            first_open.get_or_insert(i);
            assert_eq!(out[0], loud, "an open gate passes the sample unchanged");
        } else {
            assert_eq!(out[0].re.to_bits(), 0.0f32.to_bits());
            assert_eq!(out[0].im.to_bits(), 0.0f32.to_bits());
        }
    }
    assert!(first_open.is_some() && first_open.unwrap() > 0);

    // A level between the close and the open threshold (-41.5 dBFS) keeps the gate open.
    let amp = 10f32.powf(-41.5 / 20.0);
    let band = C32::new(amp, 0.0);
    for i in 0..4000 {
        let mut out = [C32::ZERO];
        Sdrk.squelch(&mut st, &[band], &mut out).unwrap();
        assert!(
            st.is_open(),
            "sample {i}: left the band at p = {}",
            st.average_power()
        );
        assert_eq!(out[0], band);
    }
    let p = st.average_power();
    assert!(p >= close_p && p < open_p, "p = {p} is in the band");

    // Silence: the gate closes on the first sample where p < P_close; that sample and all
    // later ones are exactly (+0.0, +0.0).
    let mut closed_at = None;
    for i in 0..2000 {
        let mut out = [C32::new(1.0, 1.0)];
        Sdrk.squelch(&mut st, &[C32::new(0.0, 0.0)], &mut out)
            .unwrap();
        let p = st.average_power();
        assert_eq!(st.is_open(), p >= close_p, "sample {i}: p = {p}");
        if !st.is_open() {
            closed_at.get_or_insert(i);
            assert_eq!(out[0].re.to_bits(), 0u32);
            assert_eq!(out[0].im.to_bits(), 0u32);
        }
    }
    assert!(closed_at.is_some());
    // A closed gate stays closed in the band: the same -41.5 dBFS level does not reopen it.
    for _ in 0..4000 {
        let mut out = [C32::new(1.0, 1.0)];
        Sdrk.squelch(&mut st, &[band], &mut out).unwrap();
        assert!(!st.is_open());
        assert_eq!(out[0].re.to_bits(), 0u32);
    }
}

#[test]
fn cq15_gate_opens_holds_in_the_band_and_closes_with_exact_zeros() {
    let to_q30 = |db: f64| (10f64.powf(db / 10.0) * 2f64.powi(30) + 0.5).floor();
    let (open_p, close_p) = (to_q30(-40.0) / 2f64.powi(30), to_q30(-43.0) / 2f64.powi(30));
    let mut st = SquelchState::<CQ15>::new(-40.0, 3.0, TAU, rate()).unwrap();
    let cq = |re: i16, im: i16| CQ15 {
        re: Q15(re),
        im: Q15(im),
    };
    let loud = cq(1600, 0); // 0.0488, -26 dBFS
    let mut first_open = None;
    for i in 0..400 {
        let mut out = [cq(5, 5)];
        Sdrk.squelch(&mut st, &[loud], &mut out).unwrap();
        let p = st.average_power();
        assert_eq!(st.is_open(), p >= open_p, "sample {i}: p = {p}");
        if st.is_open() {
            first_open.get_or_insert(i);
            assert_eq!(out[0], loud);
        } else {
            assert_eq!(out[0], CQ15::ZERO);
        }
    }
    assert!(first_open.is_some() && first_open.unwrap() > 0);
    // 275 / 32768 = -41.5 dBFS.
    let band = cq(275, 0);
    for i in 0..4000 {
        let mut out = [CQ15::ZERO];
        Sdrk.squelch(&mut st, &[band], &mut out).unwrap();
        assert!(st.is_open(), "sample {i}: p = {}", st.average_power());
        assert_eq!(out[0], band);
    }
    let p = st.average_power();
    assert!(p >= close_p && p < open_p, "p = {p} is in the band");
    let mut closed = false;
    for i in 0..2000 {
        let mut out = [cq(1, 1)];
        Sdrk.squelch(&mut st, &[CQ15::ZERO], &mut out).unwrap();
        let p = st.average_power();
        assert_eq!(st.is_open(), p >= close_p, "sample {i}: p = {p}");
        if !st.is_open() {
            closed = true;
            assert_eq!(out[0], CQ15::ZERO);
        }
    }
    assert!(closed);
    for _ in 0..4000 {
        let mut out = [cq(1, 1)];
        Sdrk.squelch(&mut st, &[band], &mut out).unwrap();
        assert!(!st.is_open());
        assert_eq!(out[0], CQ15::ZERO);
    }
}

#[test]
fn set_thresholds_applies_from_the_next_sample_and_keeps_the_power() {
    // C32: open gate with p in the band; raising the thresholds closes it at the next sample.
    let mut st = SquelchState::<C32>::new(-40.0, 3.0, TAU, rate()).unwrap();
    let loud = C32::new(0.05, 0.0);
    let mut out = vec![C32::ZERO; 600];
    Sdrk.squelch(&mut st, &vec![loud; 600], &mut out).unwrap();
    assert!(st.is_open());
    let before = st.average_power();
    st.set_thresholds(-10.0, 0.0).unwrap();
    assert!(
        st.is_open(),
        "set_thresholds alone does not evaluate the gate"
    );
    assert_eq!(st.average_power(), before);
    assert_eq!((st.threshold_db(), st.hysteresis_db()), (-10.0, 0.0));
    let mut one = [C32::ZERO];
    Sdrk.squelch(&mut st, &[loud], &mut one).unwrap();
    assert!(!st.is_open());
    assert_eq!(
        one[0],
        C32::ZERO,
        "the new level already applies to the next sample"
    );

    // Lowering them opens a closed gate at the next sample as well.
    let mut st = SquelchState::<C32>::new(-20.0, 3.0, TAU, rate()).unwrap();
    let mid = C32::new(0.02, 0.0); // power 4e-4 = -34 dBFS, below -20 dB
    let mut out = vec![C32::ZERO; 600];
    Sdrk.squelch(&mut st, &vec![mid; 600], &mut out).unwrap();
    assert!(!st.is_open());
    assert!(st.average_power() > 3.9e-4 && st.average_power() < 4.1e-4);
    st.set_thresholds(-40.0, 3.0).unwrap();
    assert!(!st.is_open());
    Sdrk.squelch(&mut st, &[mid], &mut one).unwrap();
    assert!(st.is_open());
    assert_eq!(one[0], mid);

    // CQ15: the same two transitions.
    let cq = |re: i16| CQ15 {
        re: Q15(re),
        im: Q15(0),
    };
    let mut st = SquelchState::<CQ15>::new(-40.0, 3.0, TAU, rate()).unwrap();
    let mut out = vec![CQ15::ZERO; 600];
    Sdrk.squelch(&mut st, &vec![cq(1600); 600], &mut out)
        .unwrap();
    assert!(st.is_open());
    let before = st.average_power();
    st.set_thresholds(-10.0, 0.0).unwrap();
    assert!(st.is_open());
    assert_eq!(st.average_power(), before);
    let mut one = [CQ15::ZERO];
    Sdrk.squelch(&mut st, &[cq(1600)], &mut one).unwrap();
    assert!(!st.is_open());
    assert_eq!(one[0], CQ15::ZERO);
    st.set_thresholds(-40.0, 3.0).unwrap();
    Sdrk.squelch(&mut st, &[cq(1600)], &mut one).unwrap();
    assert!(st.is_open());
    assert_eq!(one[0], cq(1600));

    // A rejected change keeps everything.
    let snapshot = st;
    assert!(st.set_thresholds(f64::NAN, 3.0).is_err());
    assert!(st.set_thresholds(-40.0, -0.5).is_err());
    assert!(st.set_thresholds(-151.0, 3.0).is_err());
    assert!(st.set_thresholds(0.5, 3.0).is_err());
    assert_eq!(st, snapshot);
}

#[test]
fn reset_closes_the_gate_and_clears_the_power() {
    let mut st = SquelchState::<C32>::new(-40.0, 3.0, TAU, rate()).unwrap();
    let mut out = vec![C32::ZERO; 600];
    Sdrk.squelch(&mut st, &vec![C32::new(0.1, 0.1); 600], &mut out)
        .unwrap();
    assert!(st.is_open() && st.average_power() > 0.0);
    st.reset();
    assert!(!st.is_open());
    assert_eq!(st.average_power(), 0.0);
    assert_eq!(
        st,
        SquelchState::<C32>::new(-40.0, 3.0, TAU, rate()).unwrap()
    );
}

#[test]
fn c32_extremes_never_panic_and_non_finite_samples_keep_the_state() {
    let nan = f32::NAN;
    let inf = f32::INFINITY;
    let odd = [
        C32::new(nan, 0.0),
        C32::new(0.0, nan),
        C32::new(inf, 0.0),
        C32::new(0.0, -inf),
        C32::new(inf, nan),
        C32::new(f32::MAX, f32::MAX), // q overflows to +inf
    ];
    // Closed gate: p and the state stay, the output is exactly zero.
    let mut st = SquelchState::<C32>::new(-40.0, 3.0, TAU, rate()).unwrap();
    let mut out = [C32::new(1.0, 1.0); 6];
    Sdrk.squelch(&mut st, &odd, &mut out).unwrap();
    assert!(!st.is_open());
    assert_eq!(st.average_power(), 0.0);
    assert!(
        out.iter()
            .all(|s| s.re.to_bits() == 0 && s.im.to_bits() == 0)
    );
    // Open gate: the sample is passed as it is, p and the state stay.
    let mut loud = vec![C32::new(0.1, 0.1); 300];
    let mut tmp = vec![C32::ZERO; 300];
    Sdrk.squelch(&mut st, &loud, &mut tmp).unwrap();
    assert!(st.is_open());
    let p = st.average_power();
    let mut out = [C32::ZERO; 6];
    Sdrk.squelch(&mut st, &odd, &mut out).unwrap();
    assert!(st.is_open());
    assert_eq!(st.average_power(), p);
    for (o, i) in out.iter().zip(&odd) {
        assert_eq!(
            (o.re.to_bits(), o.im.to_bits()),
            (i.re.to_bits(), i.im.to_bits())
        );
    }
    // The reference agrees on the same input.
    let mut sa = SquelchState::<C32>::new(-40.0, 3.0, TAU, rate()).unwrap();
    let mut sb = sa;
    loud.extend_from_slice(&odd);
    loud.extend(std::iter::repeat_n(C32::ZERO, 50));
    let (mut a, mut b) = (vec![C32::ZERO; loud.len()], vec![C32::ZERO; loud.len()]);
    Fallback.squelch(&mut sa, &loud, &mut a).unwrap();
    Reference.squelch(&mut sb, &loud, &mut b).unwrap();
    assert_eq!(bit_exact(&a, &b), None);
    assert_eq!(sa, sb);
}

#[test]
fn silence_and_full_scale_for_both_types() {
    // C32 silence: the gate never opens; full scale: it opens and stays open.
    let mut st = SquelchState::<C32>::new(-40.0, 3.0, TAU, rate()).unwrap();
    let mut out = vec![C32::new(1.0, 1.0); 1000];
    Sdrk.squelch(&mut st, &vec![C32::ZERO; 1000], &mut out)
        .unwrap();
    assert!(!st.is_open() && out.iter().all(|s| *s == C32::ZERO));
    let full = vec![C32::new(1.0, -1.0); 1000];
    Sdrk.squelch(&mut st, &full, &mut out).unwrap();
    assert!(st.is_open() && out == full);
    assert!(st.average_power() > 1.9);

    // CQ15 corners: -32768 on both components is q = 2^31 and must not overflow.
    let corner = |re: i16, im: i16| CQ15 {
        re: Q15(re),
        im: Q15(im),
    };
    let mut st = SquelchState::<CQ15>::new(0.0, 0.0, TAU, rate()).unwrap();
    let input: Vec<CQ15> = [
        corner(-32768, -32768),
        corner(32767, 32767),
        corner(-32768, 32767),
        corner(32767, -32768),
    ]
    .into_iter()
    .cycle()
    .take(2000)
    .collect();
    let mut out = vec![CQ15::ZERO; 2000];
    Sdrk.squelch(&mut st, &input, &mut out).unwrap();
    assert!(st.is_open() && out[1999] == input[1999]);
    assert!(st.average_power() <= 2.0 + 1e-9);
    // Then silence closes it again (threshold 0 dB, hysteresis 0).
    let mut out = vec![corner(1, 1); 2000];
    Sdrk.squelch(&mut st, &vec![CQ15::ZERO; 2000], &mut out)
        .unwrap();
    assert!(!st.is_open() && out[1999] == CQ15::ZERO);
    // The lowest threshold opens at once (P_open rounds to zero in Q30).
    let mut st = SquelchState::<CQ15>::new(-150.0, 0.0, TAU, rate()).unwrap();
    let mut out = [corner(1, 1)];
    Sdrk.squelch(&mut st, &[CQ15::ZERO], &mut out).unwrap();
    assert!(st.is_open());
}

#[test]
fn length_mismatch_is_an_error_and_changes_nothing() {
    let mut st = SquelchState::<C32>::new(-40.0, 3.0, TAU, rate()).unwrap();
    let snapshot = st;
    let mut out = [C32::ZERO; 2];
    let err = Sdrk
        .squelch(&mut st, &[C32::new(1.0, 1.0); 3], &mut out)
        .unwrap_err();
    assert_eq!(
        err,
        sdrk_core::Error::LengthMismatch {
            expected: 3,
            got: 2
        }
    );
    assert_eq!(st, snapshot);
}

/// Without the denormal flush the decaying average of `C32` would stall at a denormal number
/// (it needs about 5900 samples at alpha = 0.0104 to fall below 1e-30) and never reach zero.
#[test]
fn c32_average_decays_to_exactly_zero_and_the_gate_closes() {
    let new = || SquelchState::<C32>::new(-40.0, 3.0, TAU, rate()).unwrap();
    let mut input = vec![C32::new(0.5, 0.5); 600];
    input.extend(std::iter::repeat_n(C32::ZERO, 20_000));
    let (mut sa, mut sb) = (new(), new());
    let (mut a, mut b) = (
        vec![C32::new(1.0, 1.0); input.len()],
        vec![C32::new(1.0, 1.0); input.len()],
    );
    Fallback.squelch(&mut sa, &input, &mut a).unwrap();
    Reference.squelch(&mut sb, &input, &mut b).unwrap();
    assert_eq!(sa.average_power(), 0.0);
    assert!(!sa.is_open());
    assert_eq!(bit_exact(&a, &b), None);
    assert_eq!(sa, sb);
}

/// The `CQ15` recurrence in `i128`, written independently of the kernel.
fn cq15_model(alpha_raw: i128, input: &[(i16, i16)], mut p: i128) -> Vec<i128> {
    input
        .iter()
        .map(|&(re, im)| {
            let q = i128::from(re) * i128::from(re) + i128::from(im) * i128::from(im);
            p += (alpha_raw * (q - p) + 16384) >> 15;
            p
        })
        .collect()
}

fn alpha_raw() -> i128 {
    let alpha = 1.0 - (-1.0 / (FS as f64 * TAU)).exp();
    (alpha * 32768.0 + 0.5).floor() as i128
}

fn cq(re: i16, im: i16) -> CQ15 {
    CQ15 {
        re: Q15(re),
        im: Q15(im),
    }
}

/// Run `input` one sample at a time and return the gate after each sample.
fn gate_trace<T: sdrk_core::ComplexSample>(st: &mut SquelchState<T>, input: &[T]) -> Vec<bool>
where
    Sdrk: Squelch<T>,
{
    input
        .iter()
        .map(|s| {
            let mut o = [*s];
            Sdrk.squelch(st, std::slice::from_ref(s), &mut o).unwrap();
            st.is_open()
        })
        .collect()
}

#[test]
fn cq15_comparisons_switch_exactly_at_the_levels() {
    let ar = alpha_raw();
    let db = |p: i128| 10.0 * (p as f64 / 2f64.powi(30)).log10();

    // Opening: after one sample p1 = round(alpha * q) equals P_open exactly: `>=` opens.
    let p1 = cq15_model(ar, &[(2000, 0)], 0)[0];
    assert!(p1 > 100);
    for (level, opens) in [(p1, true), (p1 + 1, false), (p1 - 1, true)] {
        let mut st = SquelchState::<CQ15>::new(db(level), 20.0, TAU, rate()).unwrap();
        let trace = gate_trace(&mut st, &[cq(2000, 0)]);
        assert_eq!(trace, [opens], "P_open = {level}, p = {p1}");
        assert_eq!(st.average_power() * 2f64.powi(30), p1 as f64);
    }

    // Closing: open the gate with permissive levels, then one zero sample gives p_b. With
    // P_close == p_b the gate must stay open (`p < P_close` is false); with P_close = p_b + 1
    // it closes on that sample.
    let warm = vec![cq(1600, 0); 400];
    let warm_model: Vec<(i16, i16)> = vec![(1600, 0); 400];
    let p_a = *cq15_model(ar, &warm_model, 0).last().unwrap();
    let p_b = cq15_model(ar, &[(0, 0)], p_a)[0];
    assert!(p_b < p_a && p_b > 1000);
    for (close, stays_open) in [(p_b, true), (p_b + 1, false), (p_b - 1, true)] {
        let mut st = SquelchState::<CQ15>::new(-60.0, 0.0, TAU, rate()).unwrap();
        let mut out = vec![CQ15::ZERO; warm.len()];
        Sdrk.squelch(&mut st, &warm, &mut out).unwrap();
        assert!(st.is_open());
        assert_eq!(st.average_power() * 2f64.powi(30), p_a as f64);
        st.set_thresholds(db(close) + 3.0, 3.0).unwrap();
        let trace = gate_trace(&mut st, &[CQ15::ZERO]);
        assert_eq!(trace, [stays_open], "P_close = {close}, p = {p_b}");
        assert_eq!(st.average_power() * 2f64.powi(30), p_b as f64);
    }
}

#[test]
fn c32_comparisons_switch_exactly_at_the_levels() {
    let alpha = (1.0 - (-1.0 / (FS as f64 * TAU)).exp()) as f32;
    let db = |p: f32| 10.0 * f64::from(p).log10();
    let step = |p: f32, power: f32| p + alpha * (power - p);

    // Opening at p1 == P_open exactly.
    let p1 = step(0.0, 0.05f32 * 0.05);
    for (level, opens) in [
        (p1, true),
        (f32::from_bits(p1.to_bits() + 1), false),
        (f32::from_bits(p1.to_bits() - 1), true),
    ] {
        let mut st = SquelchState::<C32>::new(db(level), 20.0, TAU, rate()).unwrap();
        let trace = gate_trace(&mut st, &[C32::new(0.05, 0.0)]);
        assert_eq!(trace, [opens], "P_open = {level}, p = {p1}");
        assert_eq!(st.average_power(), f64::from(p1));
    }

    // Closing at p_b == P_close exactly.
    let mut p_a = 0.0f32;
    for _ in 0..400 {
        p_a = step(p_a, 0.05f32 * 0.05);
    }
    let p_b = step(p_a, 0.0);
    for (close, stays_open) in [
        (p_b, true),
        (f32::from_bits(p_b.to_bits() + 1), false),
        (f32::from_bits(p_b.to_bits() - 1), true),
    ] {
        let mut st = SquelchState::<C32>::new(-60.0, 0.0, TAU, rate()).unwrap();
        let mut out = vec![C32::ZERO; 400];
        Sdrk.squelch(&mut st, &vec![C32::new(0.05, 0.0); 400], &mut out)
            .unwrap();
        assert_eq!(st.average_power(), f64::from(p_a));
        st.set_thresholds(db(close) + 3.0, 3.0).unwrap();
        let trace = gate_trace(&mut st, &[C32::ZERO]);
        assert_eq!(trace, [stays_open], "P_close = {close}, p = {p_b}");
        assert_eq!(st.average_power(), f64::from(p_b));
    }
}

#[test]
fn cq15_average_follows_an_independent_integer_model_at_the_corners() {
    let ar = alpha_raw();
    let corners = [
        (-32768i16, -32768i16),
        (32767, 32767),
        (-32768, 32767),
        (32767, -32768),
        (0, 0),
        (-32768, 0),
        (1, -1),
    ];
    // Every corner on its own, then a long alternating full-scale run, then silence.
    let mut seq: Vec<(i16, i16)> = corners.to_vec();
    seq.extend(
        [(-32768, -32768), (32767, 32767)]
            .into_iter()
            .cycle()
            .take(2000),
    );
    seq.extend(std::iter::repeat_n((0, 0), 500));
    let model = cq15_model(ar, &seq, 0);
    let input: Vec<CQ15> = seq.iter().map(|&(r, i)| cq(r, i)).collect();
    let mut st = SquelchState::<CQ15>::new(-20.0, 3.0, TAU, rate()).unwrap();
    for (k, s) in input.iter().enumerate() {
        let mut o = [CQ15::ZERO];
        Sdrk.squelch(&mut st, std::slice::from_ref(s), &mut o)
            .unwrap();
        assert_eq!(
            st.average_power() * 2f64.powi(30),
            model[k] as f64,
            "sample {k}"
        );
    }
    // The same through one call of Fallback and Reference.
    let (mut sa, mut sb) = (
        SquelchState::<CQ15>::new(-20.0, 3.0, TAU, rate()).unwrap(),
        SquelchState::<CQ15>::new(-20.0, 3.0, TAU, rate()).unwrap(),
    );
    let (mut a, mut b) = (vec![CQ15::ZERO; input.len()], vec![CQ15::ZERO; input.len()]);
    Fallback.squelch(&mut sa, &input, &mut a).unwrap();
    Reference.squelch(&mut sb, &input, &mut b).unwrap();
    assert_eq!(
        sa.average_power() * 2f64.powi(30),
        *model.last().unwrap() as f64
    );
    assert_eq!(bit_exact(&a, &b), None);
    assert_eq!(sa, sb);
}

/// The Q30 limits of `CQ15` (documented in `docs/kernels.md`): the rounding of the update
/// stalls the decaying average near -66 dBFS, and an open level that rounds to zero opens the
/// gate even for silence.
#[test]
fn cq15_stalls_near_minus_66_db_and_a_very_low_threshold_always_opens() {
    let fs = Rate::hz(48_000).unwrap();
    let mut st = SquelchState::<CQ15>::new(-90.0, 3.0, 0.01, fs).unwrap();
    let mut out = vec![CQ15::ZERO; 2000];
    Sdrk.squelch(&mut st, &vec![cq(1600, 0); 2000], &mut out)
        .unwrap();
    assert!(st.is_open());
    let mut out = vec![CQ15::ZERO; 100_000];
    Sdrk.squelch(&mut st, &vec![CQ15::ZERO; 100_000], &mut out)
        .unwrap();
    let p_db = 10.0 * st.average_power().log10();
    assert!(
        (-68.0..-64.0).contains(&p_db),
        "the average stalls at {p_db} dBFS"
    );
    assert!(st.is_open(), "a close level below the stall never closes");

    for db in [-95.0, -100.0, -150.0] {
        let mut st = SquelchState::<CQ15>::new(db, 0.0, 0.01, fs).unwrap();
        let mut out = [cq(1, 1)];
        Sdrk.squelch(&mut st, &[CQ15::ZERO], &mut out).unwrap();
        assert!(st.is_open(), "{db} dB opens on silence");
    }
    let mut st = SquelchState::<CQ15>::new(-90.0, 0.0, 0.01, fs).unwrap();
    let mut out = [cq(1, 1)];
    Sdrk.squelch(&mut st, &[CQ15::ZERO], &mut out).unwrap();
    assert!(!st.is_open(), "-90 dB does not open on silence");
}
