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
    let (open_p, close_p) = (level_q61(-40.0) as f64 / Q61, level_q61(-43.0) as f64 / Q61);
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
    // The lowest CQ15 close level (-90 dB) keeps the gate closed on silence.
    let mut st = SquelchState::<CQ15>::new(-87.0, 3.0, TAU, rate()).unwrap();
    let mut out = vec![corner(1, 1); 2000];
    Sdrk.squelch(&mut st, &vec![CQ15::ZERO; 2000], &mut out)
        .unwrap();
    assert!(!st.is_open() && out.iter().all(|s| *s == CQ15::ZERO));
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

/// Scale of the Q61 average of `CQ15`.
const Q61: f64 = 2_305_843_009_213_693_952.0;

/// The `CQ15` recurrence in `i128`, written independently of the kernel: `q` in Q30, the
/// average in Q61, the coefficient in Q31.
fn cq15_model(alpha_q31: i128, input: &[(i16, i16)], mut p: i128) -> Vec<i128> {
    input
        .iter()
        .map(|&(re, im)| {
            let q = i128::from(re) * i128::from(re) + i128::from(im) * i128::from(im);
            p += alpha_q31 * (q - ((p + (1 << 30)) >> 31));
            p
        })
        .collect()
}

fn alpha_f64(tau: f64) -> f64 {
    1.0 - (-1.0 / (FS as f64 * tau)).exp()
}

fn alpha_q31() -> i128 {
    (alpha_f64(TAU) * 2f64.powi(31) + 0.5).floor() as i128
}

/// The Q61 level of `db`, computed exactly as the kernel does.
fn level_q61(db: f64) -> i128 {
    libm::floor(libm::pow(10.0, db / 10.0) * Q61 + 0.5) as i128
}

/// A level in dB whose Q61 value is exactly `target` and for which `(db + 4) - 4 == db`, so
/// that it can also be reached as `threshold - hysteresis` with a hysteresis of 4 dB. Below
/// about -36 dBFS one step of `db` moves the level by less than one, so every target is hit.
fn db_for_level(target: i128) -> f64 {
    let mut db = 10.0 * (target as f64 / Q61).log10();
    while level_q61(db) >= target {
        db = db.next_down();
    }
    for _ in 0..100_000 {
        db = db.next_up();
        let l = level_q61(db);
        assert!(l <= target, "level {target} is skipped at {db} dB");
        if l == target && (db + 4.0) - 4.0 == db {
            return db;
        }
    }
    panic!("no level in dB for {target}");
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
    let a = alpha_q31();

    // Opening: after one sample p1 = alpha * q equals P_open exactly: `>=` opens.
    let p1 = cq15_model(a, &[(2000, 0)], 0)[0];
    for (level, opens) in [(p1, true), (p1 + 1, false), (p1 - 1, true)] {
        let mut st = SquelchState::<CQ15>::new(db_for_level(level), 20.0, TAU, rate()).unwrap();
        let trace = gate_trace(&mut st, &[cq(2000, 0)]);
        assert_eq!(trace, [opens], "P_open = {level}, p = {p1}");
        assert_eq!(st.average_power(), p1 as f64 / Q61);
    }

    // Closing: open the gate with permissive levels, then one zero sample gives p_b. With
    // P_close == p_b the gate must stay open (`p < P_close` is false); with P_close = p_b + 1
    // it closes on that sample.
    let warm = vec![cq(180, 0); 400];
    let warm_model: Vec<(i16, i16)> = vec![(180, 0); 400];
    let p_a = *cq15_model(a, &warm_model, 0).last().unwrap();
    let p_b = cq15_model(a, &[(0, 0)], p_a)[0];
    assert!(p_b < p_a);
    for (close, stays_open) in [(p_b, true), (p_b + 1, false), (p_b - 1, true)] {
        let mut st = SquelchState::<CQ15>::new(-60.0, 0.0, TAU, rate()).unwrap();
        let mut out = vec![CQ15::ZERO; warm.len()];
        Sdrk.squelch(&mut st, &warm, &mut out).unwrap();
        assert!(st.is_open());
        assert_eq!(st.average_power(), p_a as f64 / Q61);
        st.set_thresholds(db_for_level(close) + 4.0, 4.0).unwrap();
        let trace = gate_trace(&mut st, &[CQ15::ZERO]);
        assert_eq!(trace, [stays_open], "P_close = {close}, p = {p_b}");
        assert_eq!(st.average_power(), p_b as f64 / Q61);
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
    let ar = alpha_q31();
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
        assert_eq!(st.average_power(), model[k] as f64 / Q61, "sample {k}");
    }
    // The same through one call of Fallback and Reference.
    let (mut sa, mut sb) = (
        SquelchState::<CQ15>::new(-20.0, 3.0, TAU, rate()).unwrap(),
        SquelchState::<CQ15>::new(-20.0, 3.0, TAU, rate()).unwrap(),
    );
    let (mut a, mut b) = (vec![CQ15::ZERO; input.len()], vec![CQ15::ZERO; input.len()]);
    Fallback.squelch(&mut sa, &input, &mut a).unwrap();
    Reference.squelch(&mut sb, &input, &mut b).unwrap();
    assert_eq!(sa.average_power(), *model.last().unwrap() as f64 / Q61);
    assert_eq!(bit_exact(&a, &b), None);
    assert_eq!(sa, sb);
}

/// Q15 noise of `level_db` dBFS (complex Gaussian, rounded and saturated).
fn q15_noise(rng: &mut Pcg32, n: usize, level_db: f64) -> Vec<CQ15> {
    let sigma = (10f64.powf(level_db / 10.0) / 2.0).sqrt() * 32768.0;
    let q = |v: f64| Q15(v.round().clamp(-32768.0, 32767.0) as i16);
    (0..n)
        .map(|_| CQ15 {
            re: q(rng.normal() * sigma),
            im: q(rng.normal() * sigma),
        })
        .collect()
}

/// The `CQ15` average follows small powers: noise at -70 and -80 dBFS with the default time
/// constant (10 ms at 48 kHz) is measured within 0.2 dB of its actual power, and a gate 3 dB
/// below it opens. A rounded Q15 step (`round_shift(alpha_q15 * (q - p), 15)`) stalled near
/// -66 dBFS here and never left zero at -80 dBFS; GNU Radio's `pwr_squelch_cc` (double)
/// measures the same values as this kernel on these signals.
#[test]
fn cq15_follows_low_noise_powers() {
    for (level, seed) in [(-70.0, 70u64), (-80.0, 80)] {
        let input = q15_noise(&mut Pcg32::new(seed, 1), 96_000, level);
        let mut st = SquelchState::<CQ15>::new(level - 3.0, 0.0, 0.01, rate()).unwrap();
        let (mut sum_p, mut sum_q, mut open) = (0.0, 0.0, 0usize);
        for (k, s) in input.iter().enumerate() {
            let mut o = [*s];
            Sdrk.squelch(&mut st, std::slice::from_ref(s), &mut o)
                .unwrap();
            if k >= 48_000 {
                sum_p += st.average_power();
                let (re, im) = (f64::from(s.re.0), f64::from(s.im.0));
                sum_q += (re * re + im * im) / 2f64.powi(30);
                open += usize::from(st.is_open());
            }
        }
        let diff_db = 10.0 * (sum_p / sum_q).log10();
        assert!(
            diff_db.abs() < 0.2,
            "{level} dBFS: average off by {diff_db} dB"
        );
        assert!(
            open >= 47_500,
            "{level} dBFS: gate open for {open} of 48000 samples"
        );
    }
}

/// After a signal, digital silence lets the `CQ15` average decay below `2^-31` (-93.3 dBFS),
/// and a gate with the close level -70 dB closes within one sample of the ideal time
/// `ln(P / P_close) / -ln(1 - alpha)` (115 ms here, as GNU Radio's `pwr_squelch_cc`).
#[test]
fn cq15_closes_in_digital_silence_at_the_ideal_time() {
    let mut st = SquelchState::<CQ15>::new(-70.0, 0.0, 0.01, rate()).unwrap();
    let mut out = vec![CQ15::ZERO; 24_000];
    Sdrk.squelch(&mut st, &vec![cq(3277, 0); 24_000], &mut out)
        .unwrap();
    assert!(st.is_open());
    let p = st.average_power();
    let silence = vec![CQ15::ZERO; 200_000];
    let trace = gate_trace(&mut st, &silence);
    let closed_at = trace.iter().position(|o| !o).unwrap();
    let ideal = (p / 1e-7).ln() / -(1.0 - alpha_f64(0.01)).ln();
    assert!(
        (closed_at as f64 - ideal).abs() <= 1.0,
        "closed after {closed_at} samples, ideal {ideal}"
    );
    assert!(st.average_power() < 2f64.powi(-31));
}

/// Long time constants keep their value in `CQ15`: after `N = fs * tau` samples of a
/// constant input the average is within 0.1 % of `q * (1 - (1 - alpha)^N)` (the rounding of
/// the Q31 coefficient is the only error, 2.7e-6 at `N = 20000`). The Q15
/// coefficient of the first version gave an 18 % shorter time constant at `N = 20000` and
/// rounded to zero (no movement at all) above about 65 500.
#[test]
fn cq15_keeps_long_time_constants() {
    for n in [20_000usize, 70_000, 1_000_000] {
        let tau = n as f64 / FS as f64;
        let mut st = SquelchState::<CQ15>::new(-30.0, 0.0, tau, rate()).unwrap();
        let mut out = vec![CQ15::ZERO; n];
        Sdrk.squelch(&mut st, &vec![cq(3277, 0); n], &mut out)
            .unwrap();
        let q = 3277f64 * 3277.0 / 2f64.powi(30);
        let ideal = q * (1.0 - (1.0 - alpha_f64(tau)).powi(n as i32));
        let rel = (st.average_power() - ideal).abs() / ideal;
        assert!(rel < 1e-3, "N = {n}: relative error {rel}");
    }
}

/// The `CQ15` close level must be at least -90 dB (the power of one input LSB); `C32` keeps
/// the range down to -150 dB. A refused `set_thresholds` changes nothing.
#[test]
fn cq15_close_level_is_limited_to_minus_90_db() {
    for (t, h) in [(-95.0, 0.0), (-150.0, 0.0), (-88.0, 3.0), (-60.0, 31.0)] {
        assert!(
            SquelchState::<CQ15>::new(t, h, 0.01, rate()).is_err(),
            "{t} {h}"
        );
        assert!(
            SquelchState::<C32>::new(t, h, 0.01, rate()).is_ok(),
            "{t} {h}"
        );
    }
    assert!(SquelchState::<CQ15>::new(-90.0, 0.0, 0.01, rate()).is_ok());
    assert!(SquelchState::<CQ15>::new(-60.0, 30.0, 0.01, rate()).is_ok());
    let mut st = SquelchState::<CQ15>::new(-40.0, 3.0, 0.01, rate()).unwrap();
    let before = st;
    assert!(st.set_thresholds(-89.0, 3.0).is_err());
    assert_eq!(st, before);
}

/// Runs `Fallback` and `Reference` over `input` in calls of random lengths (0 to 40 samples,
/// so that every exit of the fallback's gate loops lands at every position of a call) and
/// checks outputs and state bit for bit after every call.
fn agree_in_random_calls<T: sdrk_core::ComplexSample>(
    rng: &mut Pcg32,
    new: impl Fn() -> SquelchState<T>,
    input: &[T],
) -> usize
where
    Fallback: Squelch<T>,
    Reference: Squelch<T>,
{
    let (mut sa, mut sb) = (new(), new());
    let mut a = vec![T::ZERO; input.len()];
    let mut b = vec![T::ZERO; input.len()];
    let (mut at, mut toggles, mut was_open) = (0, 0, sa.is_open());
    while at < input.len() {
        let n = (rng.below(41) as usize).min(input.len() - at);
        let range = at..at + n;
        Fallback
            .squelch(&mut sa, &input[range.clone()], &mut a[range.clone()])
            .unwrap();
        Reference
            .squelch(&mut sb, &input[range.clone()], &mut b[range.clone()])
            .unwrap();
        assert_eq!(
            bit_exact(&a[range.clone()], &b[range]),
            None,
            "call at {at}"
        );
        assert_eq!(sa, sb, "state after the call at {at}");
        assert_eq!(
            sa.average_power().to_bits(),
            sb.average_power().to_bits(),
            "average after the call at {at}"
        );
        toggles += usize::from(sa.is_open() != was_open);
        was_open = sa.is_open();
        at += n;
    }
    toggles
}

/// `alpha = 1 - exp(-1 / (fs tau))` for a wanted `alpha` (inverse of the rule of `new`).
fn tau_for_alpha(alpha: f64) -> f64 {
    -1.0 / (FS as f64 * (1.0 - alpha).ln())
}

/// Every path of the `C32` fallback against the reference: the open and the closed loop,
/// their exits at the levels, the flush to zero reached from both gate states (silence after
/// a burst decays below `1e-30`), averages next to `1e-30`, non-finite powers (NaN, an
/// infinite component, a power that overflows) in both gate states, and coefficients from
/// the largest (0.5) to a small one.
#[test]
fn c32_fallback_matches_the_reference_on_every_path() {
    let mut rng = Pcg32::new(95, 1);
    let mut input = Vec::new();
    for burst in 0..24 {
        let loud = 10f64.powf(-f64::from(burst % 6));
        for _ in 0..rng.below(200) + 20 {
            input.push(C32::new(
                (rng.normal() * loud) as f32,
                (rng.normal() * loud) as f32,
            ));
        }
        match burst % 4 {
            // Silence long enough for the average to be flushed (at alpha >= 0.05).
            0 => input.extend(std::iter::repeat_n(C32::ZERO, 2000)),
            // Powers next to the flush level and to the smallest normal numbers.
            1 => {
                for _ in 0..400 {
                    let a = 10f64.powf(-14.0 - 2.0 * rng.next_f64());
                    input.push(C32::new(
                        (rng.normal() * a) as f32,
                        (rng.normal() * a) as f32,
                    ));
                }
            }
            // Non-finite powers while the gate is open (right after the burst) and closed.
            2 => {
                for (k, x) in [
                    C32::new(f32::NAN, 0.0),
                    C32::new(f32::INFINITY, 1.0),
                    C32::new(f32::MAX, f32::MAX),
                    C32::new(0.0, f32::NEG_INFINITY),
                ]
                .into_iter()
                .enumerate()
                {
                    input.push(x);
                    input.extend(std::iter::repeat_n(C32::new(1e-9, 0.0), 30 * k));
                }
            }
            _ => input.extend(std::iter::repeat_n(C32::new(3e-3, -2e-3), 300)),
        }
    }
    for alpha in [0.5, 0.31, 0.05] {
        // The last four put the close level below `1e-30` (a subnormal one, the smallest
        // subnormal and zero): the open gate then needs the flush above the close level.
        for (t, h) in [
            (-40.0, 3.0),
            (-150.0, 0.0),
            (0.0, 20.0),
            (-60.0, 0.0),
            (-150.0, 155.0),
            (-10.0, 400.0),
            (-150.0, 300.0),
            (0.0, 1000.0),
        ] {
            let new = || SquelchState::<C32>::new(t, h, tau_for_alpha(alpha), rate()).unwrap();
            let toggles = agree_in_random_calls(&mut rng, new, &input);
            if t == -40.0 {
                assert!(toggles >= 10, "alpha {alpha}: only {toggles} gate changes");
            }
        }
    }
    // The flush is reached: the average ends at exactly zero after the last silence.
    let mut st = SquelchState::<C32>::new(-40.0, 3.0, tau_for_alpha(0.5), rate()).unwrap();
    let mut out = vec![C32::ZERO; 400];
    Fallback
        .squelch(&mut st, &[C32::new(0.5, 0.5); 400], &mut out)
        .unwrap();
    Fallback
        .squelch(&mut st, &[C32::ZERO; 400], &mut out)
        .unwrap();
    assert_eq!(st.average_power().to_bits(), 0f64.to_bits());
}

/// A large hysteresis below a low threshold puts the `C32` close level below `1e-30` (or at
/// zero): an open gate whose average decays into `(close, 1e-30)` is flushed to zero and,
/// for a close level above zero, closed, as in the reference.
#[test]
fn c32_fallback_flushes_above_a_close_level_below_1e_30() {
    for (t, h, alpha) in [
        (-150.0, 155.0, None),
        (-150.0, 170.0, None),
        (-150.0, 200.0, None),
        (-150.0, 300.0, None),
        (-10.0, 400.0, Some(0.5)),
        (0.0, 1000.0, Some(0.5)),
    ] {
        let tau = alpha.map_or(1e-4, tau_for_alpha);
        let new = || SquelchState::<C32>::new(t, h, tau, rate()).unwrap();
        let (loud, quiet) = if alpha.is_some() {
            (C32::new(1.0, 0.0), C32::new(1e-17, 0.0))
        } else {
            (C32::new(0.5, 0.5), C32::new(1e-30, 0.0))
        };
        let mut input = vec![loud; 100];
        input.extend(std::iter::repeat_n(quiet, 3000));
        let (mut sa, mut sb) = (new(), new());
        let mut a = vec![C32::ZERO; input.len()];
        let mut b = vec![C32::ZERO; input.len()];
        Fallback.squelch(&mut sa, &input, &mut a).unwrap();
        Reference.squelch(&mut sb, &input, &mut b).unwrap();
        assert_eq!(bit_exact(&a, &b), None, "threshold {t}, hysteresis {h}");
        assert_eq!(sa, sb, "threshold {t}, hysteresis {h}");
        assert_eq!(
            sa.average_power().to_bits(),
            sb.average_power().to_bits(),
            "threshold {t}, hysteresis {h}"
        );
        assert_eq!(sa.average_power().to_bits(), 0f64.to_bits());
    }
}

/// Every path of the `CQ15` fallback (the carried `p + 2^30` and its split into the high and
/// the low 31 bits) against the reference: full-scale runs that drive the average to its
/// largest value (`(-32768, -32768)` has the power `2^31` in Q30), digital silence, single
/// LSBs, random noise, and the largest and smallest coefficients.
#[test]
fn cq15_fallback_matches_the_reference_on_every_path() {
    let mut rng = Pcg32::new(96, 1);
    let cq = |re: i32, im: i32| CQ15 {
        re: Q15(re as i16),
        im: Q15(im as i16),
    };
    let mut input = Vec::new();
    for round in 0..16 {
        input.extend(std::iter::repeat_n(cq(-32768, -32768), 300 + 100 * round));
        input.extend(std::iter::repeat_n(cq(32767, -32768), 50));
        input.extend(std::iter::repeat_n(cq(0, 0), 500));
        input.extend(std::iter::repeat_n(cq(1, 0), 200));
        input.extend(std::iter::repeat_n(cq(-1, 1), 200));
        for _ in 0..300 {
            let a = 10f64.powf(-4.0 * rng.next_f64());
            let v = |r: &mut Pcg32| (r.normal() * a * 32768.0).round().clamp(-32768.0, 32767.0);
            input.push(cq(v(&mut rng) as i32, v(&mut rng) as i32));
        }
    }
    // alpha 0.5, a typical one, and the smallest Q31 coefficient (1 / 2^31).
    let smallest = tau_for_alpha(1.0 / 2_147_483_648.0);
    for tau in [tau_for_alpha(0.5), TAU, smallest] {
        for (t, h) in [(-40.0, 3.0), (-90.0, 0.0), (0.0, 0.0), (-3.0, 60.0)] {
            if t - h < -90.0 {
                continue;
            }
            let new = || SquelchState::<CQ15>::new(t, h, tau, rate()).unwrap();
            agree_in_random_calls(&mut rng, new, &input);
        }
    }
    // The average after a long full-scale run is the largest the recursion reaches: just
    // below 2.0 (2^62 in Q61) and the same in both implementations.
    let full = vec![cq(-32768, -32768); 20_000];
    let mut out = vec![CQ15::ZERO; full.len()];
    let new = || SquelchState::<CQ15>::new(-40.0, 3.0, tau_for_alpha(0.5), rate()).unwrap();
    let (mut sa, mut sb) = (new(), new());
    Fallback.squelch(&mut sa, &full, &mut out).unwrap();
    Reference.squelch(&mut sb, &full, &mut out).unwrap();
    assert_eq!(sa, sb);
    assert!(sa.average_power() > 1.99 && sa.average_power() <= 2.0);
}
