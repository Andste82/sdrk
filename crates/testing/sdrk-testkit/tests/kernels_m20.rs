//! Tests of the M20 kernels `mag`, `dc_block_real` and `agc` that need the testkit:
//! block-size invariance over 20 chunk plans, `Fallback`, `Reference` and facade agreement,
//! the behaviour of the AGC loop, the `mag` sweeps and extreme inputs.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]

use sdrk::Sdrk;
use sdrk_core::kernels::agc::{Agc, AgcState};
use sdrk_core::kernels::demod::{DcBlockReal, DcBlockRealState};
use sdrk_core::kernels::spectrum::Mag;
use sdrk_core::{C32, CQ15, Fallback, Progress, Q15, Rate, Reference, Sample};
use sdrk_testkit::{ChunkPlan, Pcg32, bit_exact, check_block_invariance};

const FS: u64 = 48_000;
const N: usize = 6000;

fn rate() -> Rate {
    Rate::hz(FS).unwrap()
}

fn q(v: f64) -> Q15 {
    Q15((v * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
}

/// Level of the test signal: quiet and loud stretches so the gain moves up and down.
fn amplitude(i: usize) -> f64 {
    match (i / 1000) % 3 {
        0 => 0.01,
        1 => 0.3,
        _ => 0.003,
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

fn agc32() -> AgcState<C32> {
    AgcState::new(0.5, 0.005, 0.05, -20.0, 60.0, 0.0, rate()).unwrap()
}

fn agcq() -> AgcState<CQ15> {
    AgcState::new(0.5, 0.005, 0.05, -20.0, 60.0, 0.0, rate()).unwrap()
}

// ---------------------------------------------------------------------------------------------
// Invariance and agreement

#[test]
fn dc_block_real_is_block_size_invariant() {
    let mut rng = Pcg32::new(201, 1);
    let f: Vec<f32> = (0..N).map(|_| (rng.normal() * 0.2 + 0.3) as f32).collect();
    let qs: Vec<Q15> = (0..N).map(|_| q(rng.normal() * 0.2 + 0.3)).collect();
    for seed in 0..20u64 {
        let plan = ChunkPlan::new(seed, 1 + (seed as usize * 37) % 500);
        let (mut a, mut b) = (vec![0.0f32; N], vec![0.0f32; N]);
        let r = check_block_invariance(
            &f,
            &mut a,
            &mut b,
            plan,
            || DcBlockRealState::<f32>::new(0.995).unwrap(),
            |st, i, o| same_rate(i, o, |i, o| Sdrk.dc_block_real(st, i, o)),
        )
        .unwrap();
        assert_eq!(r.outputs, N);
        let (mut a, mut b) = (vec![Q15(0); N], vec![Q15(0); N]);
        check_block_invariance(
            &qs,
            &mut a,
            &mut b,
            plan,
            || DcBlockRealState::<Q15>::new(Q15(32600)).unwrap(),
            |st, i, o| same_rate(i, o, |i, o| Sdrk.dc_block_real(st, i, o)),
        )
        .unwrap();
    }
}

#[test]
fn agc_is_block_size_invariant() {
    let mut rng = Pcg32::new(202, 1);
    let (c32, cq) = (c32_input(&mut rng), cq_input(&mut rng));
    for seed in 0..20u64 {
        let plan = ChunkPlan::new(seed, 1 + (seed as usize * 37) % 500);
        let (mut a, mut b) = (vec![C32::ZERO; N], vec![C32::ZERO; N]);
        let r = check_block_invariance(&c32, &mut a, &mut b, plan, agc32, |st, i, o| {
            same_rate(i, o, |i, o| Sdrk.agc(st, i, o))
        })
        .unwrap();
        assert_eq!(r.outputs, N);
        let (mut a, mut b) = (vec![CQ15::ZERO; N], vec![CQ15::ZERO; N]);
        check_block_invariance(&cq, &mut a, &mut b, plan, agcq, |st, i, o| {
            same_rate(i, o, |i, o| Sdrk.agc(st, i, o))
        })
        .unwrap();
    }
}

#[test]
fn fallback_reference_and_facade_agree() {
    let mut rng = Pcg32::new(203, 1);
    let (c32, cq) = (c32_input(&mut rng), cq_input(&mut rng));

    // agc CQ15: bit-exact, including the state.
    let (mut sa, mut sb, mut sc) = (agcq(), agcq(), agcq());
    let (mut a, mut b, mut c) = (
        vec![CQ15::ZERO; N],
        vec![CQ15::ZERO; N],
        vec![CQ15::ZERO; N],
    );
    Fallback.agc(&mut sa, &cq, &mut a).unwrap();
    Reference.agc(&mut sb, &cq, &mut b).unwrap();
    Sdrk.agc(&mut sc, &cq, &mut c).unwrap();
    assert_eq!(bit_exact(&a, &b), None);
    assert_eq!(bit_exact(&a, &c), None);
    assert_eq!(sa, sb);
    assert_eq!(sa, sc);

    // agc C32: the f64 reference stays close to the f32 fallback.
    let (mut sa, mut sb) = (agc32(), agc32());
    let (mut a, mut b) = (vec![C32::ZERO; N], vec![C32::ZERO; N]);
    Fallback.agc(&mut sa, &c32, &mut a).unwrap();
    Reference.agc(&mut sb, &c32, &mut b).unwrap();
    let snr = snr_c32(&b, &a);
    assert!(snr > 80.0, "agc C32 fallback against reference: {snr} dB");
    assert!((sa.gain_db() - sb.gain_db()).abs() < 1e-3);

    // dc_block_real: Q15 bit-exact, f32 by SNR.
    let rq: Vec<Q15> = cq.iter().map(|s| s.re).collect();
    let newq = || DcBlockRealState::<Q15>::new(Q15(32700)).unwrap();
    let (mut sa, mut sb) = (newq(), newq());
    let (mut a, mut b) = (vec![Q15(0); N], vec![Q15(0); N]);
    Fallback.dc_block_real(&mut sa, &rq, &mut a).unwrap();
    Reference.dc_block_real(&mut sb, &rq, &mut b).unwrap();
    assert_eq!(bit_exact(&a, &b), None);
    assert_eq!(sa, sb);
    let rf: Vec<f32> = c32.iter().map(|s| s.re + 0.3).collect();
    let new32 = || DcBlockRealState::<f32>::new(0.999).unwrap();
    let (mut sa, mut sb) = (new32(), new32());
    let (mut a, mut b) = (vec![0.0f32; N], vec![0.0f32; N]);
    Fallback.dc_block_real(&mut sa, &rf, &mut a).unwrap();
    Reference.dc_block_real(&mut sb, &rf, &mut b).unwrap();
    let (num, den) = a.iter().zip(&b).fold((0.0f64, 0.0f64), |(n, d), (x, y)| {
        (n + f64::from(*y).powi(2), d + f64::from(x - y).powi(2))
    });
    assert!(10.0 * (num / den.max(1e-300)).log10() > 100.0);

    // mag: CQ15 bit-exact, C32 by SNR.
    let (mut a, mut b) = (vec![Q15(0); N], vec![Q15(0); N]);
    Fallback.mag(&cq, &mut a).unwrap();
    Reference.mag(&cq, &mut b).unwrap();
    assert_eq!(bit_exact(&a, &b), None);
    let (mut a, mut b) = (vec![0.0f32; N], vec![0.0f32; N]);
    Fallback.mag(&c32, &mut a).unwrap();
    Reference.mag(&c32, &mut b).unwrap();
    let (num, den) = a.iter().zip(&b).fold((0.0f64, 0.0f64), |(n, d), (x, y)| {
        (n + f64::from(*y).powi(2), d + f64::from(x - y).powi(2))
    });
    assert!(10.0 * (num / den.max(1e-300)).log10() > 130.0);
}

fn snr_c32(reference: &[C32], actual: &[C32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (r, a) in reference.iter().zip(actual) {
        num += f64::from(r.re).powi(2) + f64::from(r.im).powi(2);
        den += f64::from(r.re - a.re).powi(2) + f64::from(r.im - a.im).powi(2);
    }
    10.0 * (num / den.max(1e-300)).log10()
}

// ---------------------------------------------------------------------------------------------
// AGC behaviour

/// Constant-modulus complex tone of 1 kHz with amplitude `a`, `n` samples from `start`.
fn tone(a: f64, start: usize, n: usize) -> Vec<C32> {
    (start..start + n)
        .map(|i| {
            let ph = 2.0 * core::f64::consts::PI * ((i as f64 * 1000.0 / FS as f64) % 1.0);
            C32::new((a * ph.cos()) as f32, (a * ph.sin()) as f32)
        })
        .collect()
}

fn tone_q(a: f64, start: usize, n: usize) -> Vec<CQ15> {
    tone(a, start, n)
        .iter()
        .map(|s| CQ15::from_c32(*s))
        .collect()
}

fn level_db_c32(out: &[C32]) -> f64 {
    let m = out.iter().map(|s| f64::from(s.re).hypot(f64::from(s.im)));
    20.0 * (m.sum::<f64>() / out.len() as f64).log10()
}

fn level_db_q(out: &[CQ15]) -> f64 {
    let m = out
        .iter()
        .map(|s| f64::from(s.re.0).hypot(f64::from(s.im.0)) / 32768.0);
    20.0 * (m.sum::<f64>() / out.len() as f64).log10()
}

const REFERENCE_DB: f64 = -6.020_599_913_279_624; // 20 log10(0.5)

#[test]
fn c32_step_up_and_down_settle_within_one_db() {
    let (attack, release) = (0.005, 0.05);
    let mut st = AgcState::<C32>::new(0.5, attack, release, -20.0, 60.0, 0.0, rate()).unwrap();
    let settle = |st: &mut AgcState<C32>, amp: f64, secs: f64| -> Vec<C32> {
        let n = (secs * FS as f64) as usize;
        let x = tone(amp, 0, n);
        let mut y = vec![C32::ZERO; n];
        Sdrk.agc(st, &x, &mut y).unwrap();
        y
    };
    // -40 dBFS tone first: the loop settles at the reference (release time constants).
    let y = settle(&mut st, 0.01, 10.0 * release);
    assert!((level_db_c32(&y[y.len() - 100..]) - REFERENCE_DB).abs() < 0.1);
    // Step up to -10 dBFS: the output is within 1 dB after 5 attack time constants.
    let y = settle(&mut st, 0.316_227_766, 5.0 * attack + 0.005);
    let n5 = (5.0 * attack * FS as f64) as usize;
    let tail = &y[n5..n5 + 100];
    assert!(
        (level_db_c32(tail) - REFERENCE_DB).abs() < 1.0,
        "{} dB",
        level_db_c32(tail)
    );
    // The first samples of the step are far too loud: the attack acts quickly.
    assert!(level_db_c32(&y[..4]) > REFERENCE_DB + 10.0);
    // Back down by 15 dB to -25 dBFS: within 1 dB after 5 release time constants.
    let y = settle(&mut st, 0.056_234_132, 5.0 * release + 0.005);
    let n5 = (5.0 * release * FS as f64) as usize;
    let tail = &y[n5..n5 + 100];
    assert!(
        (level_db_c32(tail) - REFERENCE_DB).abs() < 1.0,
        "{} dB",
        level_db_c32(tail)
    );
    // The full 30 dB step down (-10 to -40 dBFS) needs about 5.5 release time constants for
    // 1 dB (the loop is logistic: M20-1 in docs/open-items.md): check it at 6.
    let mut st = AgcState::<C32>::new(0.5, attack, release, -20.0, 60.0, 0.0, rate()).unwrap();
    settle(&mut st, 0.316_227_766, 10.0 * release);
    let y = settle(&mut st, 0.01, 6.0 * release + 0.005);
    let n6 = (6.0 * release * FS as f64) as usize;
    assert!((level_db_c32(&y[n6..n6 + 100]) - REFERENCE_DB).abs() < 1.0);
}

#[test]
fn cq15_step_up_and_down_settle_within_one_db() {
    let (attack, release) = (0.005, 0.05);
    let mut st = AgcState::<CQ15>::new(0.5, attack, release, -20.0, 60.0, 0.0, rate()).unwrap();
    let settle = |st: &mut AgcState<CQ15>, amp: f64, secs: f64| -> Vec<CQ15> {
        let n = (secs * FS as f64) as usize;
        let x = tone_q(amp, 0, n);
        let mut y = vec![CQ15::ZERO; n];
        Sdrk.agc(st, &x, &mut y).unwrap();
        y
    };
    let y = settle(&mut st, 0.01, 10.0 * release);
    assert!((level_db_q(&y[y.len() - 100..]) - REFERENCE_DB).abs() < 0.2);
    let y = settle(&mut st, 0.316_227_766, 5.0 * attack + 0.005);
    let n5 = (5.0 * attack * FS as f64) as usize;
    assert!((level_db_q(&y[n5..n5 + 100]) - REFERENCE_DB).abs() < 1.0);
    let y = settle(&mut st, 0.056_234_132, 5.0 * release + 0.005);
    let n5 = (5.0 * release * FS as f64) as usize;
    assert!((level_db_q(&y[n5..n5 + 100]) - REFERENCE_DB).abs() < 1.0);
    // The full 30 dB step down needs about 5.5 release time constants (M20-1).
    let mut st = AgcState::<CQ15>::new(0.5, attack, release, -20.0, 60.0, 0.0, rate()).unwrap();
    settle(&mut st, 0.316_227_766, 10.0 * release);
    let y = settle(&mut st, 0.01, 6.0 * release + 0.005);
    let n6 = (6.0 * release * FS as f64) as usize;
    assert!((level_db_q(&y[n6..n6 + 100]) - REFERENCE_DB).abs() < 1.0);
}

#[test]
fn the_gain_never_leaves_its_limits() {
    let mut rng = Pcg32::new(204, 1);
    let (c32, cq) = (c32_input(&mut rng), cq_input(&mut rng));
    let silence = vec![C32::ZERO; 60_000];
    let silence_q = vec![CQ15::ZERO; 60_000];
    let loud = vec![C32::new(1.0, 1.0); 30_000];
    let loud_q = vec![
        CQ15 {
            re: Q15::MAX,
            im: Q15::MAX
        };
        30_000
    ];
    for (min, max, init) in [(-20.0, 60.0, 0.0), (10.0, 20.0, 15.0), (-40.0, 90.0, 30.0)] {
        let mk32 = || AgcState::<C32>::new(0.5, 0.005, 0.05, min, max, init, rate()).unwrap();
        let mkq = || AgcState::<CQ15>::new(0.5, 0.005, 0.05, min, max, init, rate()).unwrap();
        let (mut s32, mut sq) = (mk32(), mkq());
        let (tol32, tolq) = (1e-4, 1e-6);
        for x in [&c32, &silence, &loud, &c32] {
            let mut y = vec![C32::ZERO; x.len()];
            Sdrk.agc(&mut s32, x, &mut y).unwrap();
            let g = s32.gain_db();
            assert!(
                g >= min - tol32 && g <= max + tol32,
                "{g} not in {min}..{max}"
            );
        }
        for x in [&cq, &silence_q, &loud_q, &cq] {
            let mut y = vec![CQ15::ZERO; x.len()];
            Sdrk.agc(&mut sq, x, &mut y).unwrap();
            let g = sq.gain_db();
            assert!(
                g >= min - tolq && g <= max + tolq,
                "{g} not in {min}..{max}"
            );
        }
    }
    // Silence drives the gain to the maximum; a full-scale signal settles at the reference
    // divided by its amplitude (|x| = sqrt 2), after a first sample that drives the gain to
    // the minimum (the step is far larger than the gain).
    let mut st = agc32();
    let mut y = vec![C32::ZERO; silence.len()];
    Sdrk.agc(&mut st, &silence, &mut y).unwrap();
    assert!((st.gain_db() - 60.0).abs() < 1e-3);
    let mut y = vec![C32::ZERO; loud.len()];
    Sdrk.agc(&mut st, &loud, &mut y).unwrap();
    assert!((st.gain_db() - 20.0 * (0.5f64 / 2f64.sqrt()).log10()).abs() < 0.05);
    assert!(y[0].re >= 1000.0 && y[1].re < 0.2, "{:?} {:?}", y[0], y[1]);
    let mut st = agcq();
    let mut y = vec![CQ15::ZERO; silence_q.len()];
    Sdrk.agc(&mut st, &silence_q, &mut y).unwrap();
    assert!((st.gain_db() - 60.0).abs() < 1e-6);
    let mut y = vec![CQ15::ZERO; loud_q.len()];
    Sdrk.agc(&mut st, &loud_q, &mut y).unwrap();
    assert!((st.gain_db() - 20.0 * (0.5f64 / 2f64.sqrt()).log10()).abs() < 0.05);
}

#[test]
fn a_nan_sample_leaves_the_gain_unchanged() {
    let mut st = agc32();
    let mut y = vec![C32::ZERO; 3000];
    Sdrk.agc(&mut st, &tone(0.01, 0, 3000), &mut y).unwrap();
    let before = st;
    let mut out = [C32::ZERO; 3];
    let input = [
        C32::new(f32::NAN, 0.1),
        C32::new(0.1, f32::NAN),
        C32::new(f32::NAN, f32::NAN),
    ];
    Sdrk.agc(&mut st, &input, &mut out).unwrap();
    assert_eq!(st, before);
    assert!(out.iter().all(|s| s.re.is_nan() || s.im.is_nan()));
    // The next finite sample moves the gain again.
    Sdrk.agc(&mut st, &[C32::new(0.01, 0.0)], &mut out[..1])
        .unwrap();
    assert_ne!(st.gain_db(), before.gain_db());
}

#[test]
fn an_infinite_sample_drives_the_gain_to_the_minimum() {
    let mut st = agc32();
    let mut out = [C32::ZERO; 1];
    Sdrk.agc(&mut st, &[C32::new(f32::INFINITY, 0.0)], &mut out)
        .unwrap();
    assert_eq!(out[0].re, f32::INFINITY);
    assert!((st.gain_db() + 20.0).abs() < 1e-4, "{}", st.gain_db());
    // It recovers afterwards.
    let mut y = vec![C32::ZERO; 60_000];
    Sdrk.agc(&mut st, &vec![C32::ZERO; 60_000], &mut y).unwrap();
    assert!(st.gain_db() > 59.0);
}

#[test]
fn set_params_keeps_the_gain_and_clamps_it() {
    for c32_type in [true, false] {
        if c32_type {
            let mut st = AgcState::<C32>::new(0.5, 0.005, 0.05, -20.0, 60.0, 20.0, rate()).unwrap();
            let g = st.gain_db();
            st.set_params(0.25, 0.01, 0.1, -30.0, 80.0, rate()).unwrap();
            assert_eq!(st.gain_db(), g);
            st.set_params(0.25, 0.01, 0.1, -30.0, 10.0, rate()).unwrap();
            assert!((st.gain_db() - 10.0).abs() < 1e-5);
            st.set_params(0.25, 0.01, 0.1, 30.0, 40.0, rate()).unwrap();
            assert!((st.gain_db() - 30.0).abs() < 1e-5);
            // A rejected change leaves everything as it was.
            let copy = st;
            assert!(st.set_params(2.0, 0.01, 0.1, 30.0, 40.0, rate()).is_err());
            assert!(st.set_params(0.25, 0.01, 0.1, 50.0, 40.0, rate()).is_err());
            assert_eq!(st, copy);
        } else {
            let mut st =
                AgcState::<CQ15>::new(0.5, 0.005, 0.05, -20.0, 60.0, 20.0, rate()).unwrap();
            let g = st.gain_db();
            st.set_params(0.25, 0.01, 0.1, -30.0, 80.0, rate()).unwrap();
            assert_eq!(st.gain_db(), g);
            st.set_params(0.25, 0.01, 0.1, -30.0, 10.0, rate()).unwrap();
            assert!((st.gain_db() - 10.0).abs() < 1e-6);
            st.reset();
            assert!(
                (st.gain_db() - 10.0).abs() < 1e-6,
                "reset clamps to the range"
            );
        }
    }
    // A changed set point acts from the next sample: the loop settles at the new reference.
    let mut st = agc32();
    let mut y = vec![C32::ZERO; 24_000];
    Sdrk.agc(&mut st, &tone(0.1, 0, 24_000), &mut y).unwrap();
    st.set_params(0.25, 0.005, 0.05, -20.0, 60.0, rate())
        .unwrap();
    Sdrk.agc(&mut st, &tone(0.1, 24_000, 24_000), &mut y)
        .unwrap();
    assert!((level_db_c32(&y[23_000..]) - 20.0 * 0.25f64.log10()).abs() < 0.1);
}

#[test]
fn reset_returns_to_the_initial_gain() {
    let mut st = AgcState::<C32>::new(0.5, 0.005, 0.05, -20.0, 60.0, 12.0, rate()).unwrap();
    let initial = st.gain_db();
    let mut y = vec![C32::ZERO; 5000];
    Sdrk.agc(&mut st, &tone(0.01, 0, 5000), &mut y).unwrap();
    assert!(st.gain_db() > initial + 1.0);
    st.reset();
    assert_eq!(st.gain_db(), initial);
}

// ---------------------------------------------------------------------------------------------
// mag

/// The rounded square root through f64: exact for sums below 2^31.
fn want_mag(re: i16, im: i16) -> i16 {
    let s = f64::from(re).powi(2) + f64::from(im).powi(2);
    (s.sqrt() + 0.5).floor().min(32767.0) as i16
}

#[test]
fn mag_cq15_equals_the_rounded_square_root_over_sweeps() {
    let mut pairs: Vec<CQ15> = Vec::new();
    let mk = |re: i32, im: i32| CQ15 {
        re: Q15(re as i16),
        im: Q15(im as i16),
    };
    // Every value on both axes and on the diagonal.
    for v in -32768..=32767 {
        pairs.push(mk(v, 0));
        pairs.push(mk(0, v));
        pairs.push(mk(v, v));
        pairs.push(mk(v, -1 - v));
    }
    // A coarse grid, which includes the four full-scale corners.
    for re in (-32768..=32767).step_by(257).chain([32767]) {
        for im in (-32768..=32767).step_by(263).chain([32767]) {
            pairs.push(mk(re, im));
        }
    }
    // The boundaries r^2 + r and r^2 + r + 1 around every integer root up to 181 (tie edge).
    for r in 0..=181i32 {
        pairs.push(mk(r, 0));
        pairs.push(mk(0, r));
    }
    let mut out_f = vec![Q15(0); pairs.len()];
    let mut out_r = vec![Q15(0); pairs.len()];
    Fallback.mag(&pairs, &mut out_f).unwrap();
    Reference.mag(&pairs, &mut out_r).unwrap();
    assert_eq!(out_f, out_r);
    for ((p, f), r) in pairs.iter().zip(&out_f).zip(&out_r) {
        assert_eq!(f.0, want_mag(p.re.0, p.im.0), "({}, {})", p.re.0, p.im.0);
        assert_eq!(f, r);
    }
    // Spot values: the corners saturate, 3-4-5 is exact.
    let mut y = [Q15(0); 6];
    Fallback
        .mag(
            &[
                mk(-32768, -32768),
                mk(32767, 32767),
                mk(3, 4),
                mk(0, 0),
                mk(1, 1),
                mk(-32768, 0),
            ],
            &mut y,
        )
        .unwrap();
    assert_eq!(
        y,
        [Q15(32767), Q15(32767), Q15(5), Q15(0), Q15(1), Q15(32767)]
    );
}

#[test]
fn mag_cq15_squares_that_need_rounding_up() {
    // s = r^2 + r is rounded down, s = r^2 + r + 1 up, as is the first integer past it.
    for r in 1..=180i32 {
        let want_lo = r as i16;
        for (s, want) in [(r * r + r, want_lo), (r * r + r + 1, want_lo + 1)] {
            // Find a pair (re, im) with re^2 + im^2 = s when one exists.
            let found = (0..=r + 1).find_map(|a| {
                let b2 = s - a * a;
                let b = f64::from(b2).sqrt().round() as i32;
                (b2 >= 0 && b * b == b2).then_some((a, b))
            });
            if let Some((a, b)) = found {
                let mut y = [Q15(0)];
                Fallback
                    .mag(
                        &[CQ15 {
                            re: Q15(a as i16),
                            im: Q15(b as i16),
                        }],
                        &mut y,
                    )
                    .unwrap();
                assert_eq!(y[0].0, want, "s = {s}");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Extremes

#[test]
fn mag_c32_handles_overflow_nan_and_infinity() {
    let input = [
        C32::new(0.0, 0.0),
        C32::new(1.0e20, 0.0),
        C32::new(f32::NAN, 1.0),
        C32::new(1.0, f32::NAN),
        C32::new(f32::INFINITY, 0.0),
        C32::new(f32::NEG_INFINITY, f32::INFINITY),
        C32::new(f32::MAX, f32::MAX),
        C32::new(-1.0, -1.0),
    ];
    let mut y = [0.0f32; 8];
    Fallback.mag(&input, &mut y).unwrap();
    assert_eq!(y[0], 0.0);
    assert_eq!(
        y[1],
        f32::INFINITY,
        "above about 1.8e19 the square overflows"
    );
    assert!(y[2].is_nan() && y[3].is_nan());
    assert_eq!(
        (y[4], y[5], y[6]),
        (f32::INFINITY, f32::INFINITY, f32::INFINITY)
    );
    assert_eq!(y[7], core::f32::consts::SQRT_2);
    // The facade and the reference do not panic on the same input.
    let mut z = [0.0f32; 8];
    Sdrk.mag(&input, &mut z).unwrap();
    Reference.mag(&input, &mut z).unwrap();
}

#[test]
fn dc_block_real_extremes_do_not_panic_and_q15_saturates() {
    let mut st = DcBlockRealState::<f32>::new(0.999).unwrap();
    let input = [
        0.0,
        1.0,
        -1.0,
        f32::MAX,
        -f32::MAX,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.5,
    ];
    let mut y = [0.0f32; 9];
    Sdrk.dc_block_real(&mut st, &input, &mut y).unwrap();
    assert_eq!(y[1], 1.0);
    // Q15: a jump from the lowest to the highest value saturates.
    let mut st = DcBlockRealState::<Q15>::new(Q15(32767)).unwrap();
    let mut y = [Q15(0); 5];
    Sdrk.dc_block_real(
        &mut st,
        &[Q15::MIN, Q15::MAX, Q15::MIN, Q15::MAX, Q15(0)],
        &mut y,
    )
    .unwrap();
    assert_eq!(y[0], Q15::MIN);
    assert_eq!(y[1], Q15::MAX);
    assert_eq!(y[2], Q15::MIN);
    // Silence stays silence.
    let mut st = DcBlockRealState::<Q15>::new(Q15(32000)).unwrap();
    let mut y = [Q15(7); 100];
    Sdrk.dc_block_real(&mut st, &[Q15(0); 100], &mut y).unwrap();
    assert!(y.iter().all(|s| s.0 == 0));
    // Reference and fallback agree on the Q15 corners.
    let corners = [
        Q15::MIN,
        Q15::MAX,
        Q15(0),
        Q15::MAX,
        Q15::MIN,
        Q15(1),
        Q15(-1),
    ];
    let new = || DcBlockRealState::<Q15>::new(Q15(32767)).unwrap();
    let (mut sa, mut sb) = (new(), new());
    let (mut a, mut b) = ([Q15(0); 7], [Q15(0); 7]);
    Fallback.dc_block_real(&mut sa, &corners, &mut a).unwrap();
    Reference.dc_block_real(&mut sb, &corners, &mut b).unwrap();
    assert_eq!(a, b);
}

#[test]
fn agc_extremes_do_not_panic() {
    let weird = [
        C32::ZERO,
        C32::new(1.0, 1.0),
        C32::new(-1.0, -1.0),
        C32::new(f32::MAX, f32::MAX),
        C32::new(f32::NAN, 0.0),
        C32::new(0.0, f32::NAN),
        C32::new(f32::INFINITY, 0.0),
        C32::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
        C32::new(1e-38, 1e-38),
        C32::new(0.25, 0.0),
    ];
    for engine in 0..3 {
        let mut st = agc32();
        let mut y = [C32::ZERO; 10];
        match engine {
            0 => Sdrk.agc(&mut st, &weird, &mut y).unwrap(),
            1 => Fallback.agc(&mut st, &weird, &mut y).unwrap(),
            _ => Reference.agc(&mut st, &weird, &mut y).unwrap(),
        }
        let g = st.gain_db();
        assert!(
            g.is_finite() && (-20.001..=60.001).contains(&g),
            "engine {engine}: {g}"
        );
    }
    // Q15 corners.
    let corners = [
        CQ15::ZERO,
        CQ15 {
            re: Q15::MIN,
            im: Q15::MIN,
        },
        CQ15 {
            re: Q15::MAX,
            im: Q15::MAX,
        },
        CQ15 {
            re: Q15::MIN,
            im: Q15::MAX,
        },
        CQ15 {
            re: Q15(1),
            im: Q15(-1),
        },
        CQ15::ZERO,
    ];
    let (mut sa, mut sb) = (agcq(), agcq());
    let (mut a, mut b) = ([CQ15::ZERO; 6], [CQ15::ZERO; 6]);
    Fallback.agc(&mut sa, &corners, &mut a).unwrap();
    Reference.agc(&mut sb, &corners, &mut b).unwrap();
    assert_eq!(a, b);
    assert_eq!(sa, sb);
    // The extreme gains with full-scale input stay in range of the integer arithmetic.
    let mut st = AgcState::<CQ15>::new(
        1.0 / 32.0,
        1.0 / 48_000.0 * 1.45,
        0.05,
        -40.0,
        90.0,
        90.0,
        rate(),
    )
    .unwrap();
    let loud = vec![
        CQ15 {
            re: Q15::MIN,
            im: Q15::MIN
        };
        2000
    ];
    let mut y = vec![CQ15::ZERO; 2000];
    Sdrk.agc(&mut st, &loud, &mut y).unwrap();
    assert!(st.gain_db() < -30.0, "{}", st.gain_db());
    let mut y = vec![CQ15::ZERO; 200_000];
    Sdrk.agc(&mut st, &vec![CQ15::ZERO; 200_000], &mut y)
        .unwrap();
    assert!(st.gain_db() > 80.0);
}

#[test]
fn length_mismatch_is_an_error_and_changes_nothing() {
    let mut st = agc32();
    let before = st;
    let mut y = [C32::ZERO; 3];
    assert!(Sdrk.agc(&mut st, &[C32::ZERO; 4], &mut y).is_err());
    assert_eq!(st, before);
    let mut st = DcBlockRealState::<f32>::new(0.9).unwrap();
    let mut y = [0.0f32; 3];
    assert!(Sdrk.dc_block_real(&mut st, &[0.0; 4], &mut y).is_err());
    assert!(Sdrk.mag(&[C32::ZERO; 4], &mut y).is_err());
    let mut y = [Q15(0); 3];
    assert!(Sdrk.mag(&[CQ15::ZERO; 4], &mut y).is_err());
}

/// A time constant so long that the `CQ15` loop coefficient would round to zero is rejected
/// instead of freezing the gain silently.
#[test]
fn a_time_constant_that_freezes_the_loop_is_rejected() {
    assert!(AgcState::<CQ15>::new(0.03125, 1e9, 0.5, -20.0, 60.0, 0.0, rate()).is_err());
    assert!(AgcState::<CQ15>::new(0.03125, 0.1, 1e9, -20.0, 60.0, 0.0, rate()).is_err());
    let mut st = AgcState::<CQ15>::new(0.03125, 0.1, 0.5, -20.0, 60.0, 0.0, rate()).unwrap();
    let before = st;
    assert!(
        st.set_params(0.03125, 1e9, 0.5, -20.0, 60.0, rate())
            .is_err()
    );
    assert_eq!(st, before);
}

/// A NaN input poisons the stored state of the `f32` DC blocker until `reset()`, as in
/// `dc_block`.
#[test]
fn dc_block_real_f32_stays_nan_after_a_nan_until_reset() {
    let mut st = DcBlockRealState::<f32>::new(0.9).unwrap();
    let mut y = [0.0f32; 4];
    Sdrk.dc_block_real(&mut st, &[0.5, f32::NAN, 0.5, 0.5], &mut y)
        .unwrap();
    assert!(y[1].is_nan() && y[2].is_nan() && y[3].is_nan());
    st.reset();
    Sdrk.dc_block_real(&mut st, &[0.5; 4], &mut y).unwrap();
    assert!(y.iter().all(|v| v.is_finite()));
}
