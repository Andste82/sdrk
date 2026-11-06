//! Tests of the M43 kernels of `sdrk-core`: `dot_complex` (the order of its contract, the
//! accuracy against an f64 sum, the extremes) and `equalize` (construction rules, timing, the
//! length check, a clean QPSK signal through an identity channel, CMA convergence on a two-tap
//! channel, decision-directed convergence, the phase tracker, the decision feedback, the mode
//! switch, `reset`, non-finite input, huge inputs, block-size invariance and the thread-safety of
//! the public types). The
//! comparison of `Fallback` with `Reference` runs only with the feature `reference`.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::approx_constant
)]

use sdrk_core::kernels::dot::{DOT_COMPLEX_PARTIALS, DotComplex};
use sdrk_core::kernels::equalize::{EqConstellation, EqMode, EqParams, Equalize, EqualizeState};
use sdrk_core::{C32, Delay, Error, Fallback, Ratio, Timing};
use sdrk_device::{ErrorKind, SdrkError};

/// The seed of the `dot_complex` tests (the M43 seeds of the new kernel, 596 and 597).
const SEED_DOT: u64 = 596;
/// The seed of the `equalize` unit tests.
const SEED_EQ: u64 = 597;
/// The seed of the chunk plans (test 6 of M43).
const SEED_PLANS: u64 = 595;

/// Deterministic generator (xorshift64*), uniform in [-1, 1).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (v as f64 / (1u64 << 52) as f64 - 1.0) as f32
    }

    fn below(&mut self, n: usize) -> usize {
        ((f32::midpoint(self.next(), 1.0) * n as f32) as usize).min(n - 1)
    }
}

const H: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// QPSK points `(+-1 +-j) / sqrt 2`.
fn qpsk(rng: &mut Rng, n: usize) -> Vec<C32> {
    (0..n)
        .map(|_| {
            C32::new(
                if rng.next() < 0.0 { -H } else { H },
                if rng.next() < 0.0 { -H } else { H },
            )
        })
        .collect()
}

/// `x` through the FIR channel `h` (`h[0]` multiplies the current symbol).
fn channel(x: &[C32], h: &[C32]) -> Vec<C32> {
    (0..x.len())
        .map(|n| {
            h.iter()
                .enumerate()
                .filter(|&(k, _)| k <= n)
                .fold(C32::new(0.0, 0.0), |acc, (k, &c)| acc + c * x[n - k])
        })
        .collect()
}

type State = EqualizeState<Vec<C32>, Vec<C32>>;

/// `taps` forward and `feedback` feedback taps, both step sizes `mu`, the tracker at `phase_bw`.
fn state_with(
    taps: usize,
    feedback: usize,
    c: EqConstellation,
    mu: f32,
    cma: u32,
    phase_bw: f64,
) -> State {
    EqualizeState::new(
        vec![C32::default(); taps + feedback],
        vec![C32::default(); taps + feedback],
        EqParams {
            constellation: c,
            feedback_taps: feedback,
            mu_cma: mu,
            mu_dd: mu,
            cma_symbols: cma,
            phase_bw,
        },
    )
    .unwrap()
}

/// A linear equalizer without the phase tracker.
fn state(taps: usize, c: EqConstellation, mu: f32, cma: u32) -> State {
    state_with(taps, 0, c, mu, cma, 0.0)
}

fn polar(m: f32, a: f32) -> C32 {
    C32::new(m * a.cos(), m * a.sin())
}

/// `v` turned by `r` quarter turns.
fn quarter(v: C32, r: u32) -> C32 {
    (0..r).fold(v, |v, _| C32::new(-v.im, v.re))
}

fn bits(v: &[C32]) -> Vec<(u32, u32)> {
    let b = |x: f32| if x.is_nan() { u32::MAX } else { x.to_bits() };
    v.iter().map(|c| (b(c.re), b(c.im))).collect()
}

// ---------------------------------------------------------------------------------------------
// dot_complex

/// The order of the contract written out: eight partial sums, then the tree.
fn contract_dot(x: &[C32], w: &[C32]) -> C32 {
    let mut re = [0.0f32; 8];
    let mut im = [0.0f32; 8];
    for (i, (a, b)) in x.iter().zip(w).enumerate() {
        re[i % 8] += a.re * b.re - a.im * b.im;
        im[i % 8] += a.re * b.im + a.im * b.re;
    }
    let t = |a: [f32; 8]| ((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7]));
    C32::new(t(re), t(im))
}

#[test]
fn dot_complex_follows_the_contract_order() {
    assert_eq!(DOT_COMPLEX_PARTIALS, 8);
    let mut rng = Rng::new(SEED_DOT);
    let scales = [1.0f32, -1e-3, 1e4, -7.5, 3e-20];
    for n in 0..=130 {
        let mut draw = || {
            let s = scales[rng.below(scales.len())];
            C32::new(rng.next() * s, rng.next() * s)
        };
        let x: Vec<C32> = (0..n).map(|_| draw()).collect();
        let w: Vec<C32> = (0..n).map(|_| draw()).collect();
        let got = Fallback.dot_complex(&x, &w).unwrap();
        assert_eq!(bits(&[got]), bits(&[contract_dot(&x, &w)]), "n {n}");
    }
}

#[test]
fn dot_complex_is_accurate_against_the_f64_sum() {
    // The f64 accuracy check (D34): unit-scale inputs, lengths up to 4096.
    let mut rng = Rng::new(SEED_DOT + 1);
    for n in [1usize, 11, 63, 1000, 4096] {
        let x: Vec<C32> = (0..n).map(|_| C32::new(rng.next(), rng.next())).collect();
        let w: Vec<C32> = (0..n).map(|_| C32::new(rng.next(), rng.next())).collect();
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (a, b) in x.iter().zip(&w) {
            let (ar, ai, br, bi) = (
                f64::from(a.re),
                f64::from(a.im),
                f64::from(b.re),
                f64::from(b.im),
            );
            re += ar * br - ai * bi;
            im += ar * bi + ai * br;
        }
        let got = Fallback.dot_complex(&x, &w).unwrap();
        let err = (f64::from(got.re) - re).hypot(f64::from(got.im) - im);
        // The bound of a float sum of n products of magnitude <= 2 in chains of n / 8.
        let bound = 2.0 * (n as f64 / 8.0 + 3.0) * f64::from(f32::EPSILON) * 2.0 * n as f64;
        assert!(err <= bound.max(1e-6), "n {n}: error {err}, bound {bound}");
    }
}

#[test]
fn dot_complex_extremes() {
    // Length 0 and 1, and the length check.
    assert_eq!(
        bits(&[Fallback.dot_complex(&[], &[]).unwrap()]),
        bits(&[C32::new(0.0, 0.0)])
    );
    let a = C32::new(0.5, -2.0);
    let b = C32::new(3.0, 0.25);
    assert_eq!(
        Fallback.dot_complex(&[a], &[b]).unwrap(),
        C32::new(2.0, -5.875)
    );
    let e = Fallback.dot_complex(&[a, a], &[b]).unwrap_err();
    assert_eq!(
        e,
        Error::LengthMismatch {
            expected: 2,
            got: 1
        }
    );
    assert_eq!(e.kind(), ErrorKind::LengthMismatch);
    // Silence: +0, also for -0 inputs (the partial sums start at +0).
    let z = vec![C32::new(-0.0, -0.0); 17];
    let got = Fallback.dot_complex(&z, &z).unwrap();
    assert_eq!((got.re.to_bits(), got.im.to_bits()), (0, 0));
    // Full scale: 63 products of (1, 1)(1, -1) = 2 + 0j.
    let x = vec![C32::new(1.0, 1.0); 63];
    let w = vec![C32::new(1.0, -1.0); 63];
    assert_eq!(Fallback.dot_complex(&x, &w).unwrap(), C32::new(126.0, 0.0));
    // NaN and infinities propagate.
    let mut x = vec![C32::new(0.5, 0.5); 20];
    x[13] = C32::new(f32::NAN, 0.0);
    let got = Fallback.dot_complex(&x, &w[..20]).unwrap();
    assert!(got.re.is_nan() && got.im.is_nan());
    x[13] = C32::new(f32::INFINITY, 0.0);
    let got = Fallback.dot_complex(&x, &w[..20]).unwrap();
    assert_eq!((got.re, got.im), (f32::INFINITY, f32::NEG_INFINITY));
    // An overflow of the products gives infinity, not a wrapped value.
    let big = vec![C32::new(f32::MAX, 0.0); 2];
    assert_eq!(Fallback.dot_complex(&big, &big).unwrap().re, f32::INFINITY);
}

// ---------------------------------------------------------------------------------------------
// equalize

#[test]
fn construction_rules() {
    let q = EqConstellation::Qpsk;
    let params = |feedback_taps: usize, mu_cma: f32, mu_dd: f32, phase_bw: f64| EqParams {
        constellation: q,
        feedback_taps,
        mu_cma,
        mu_dd,
        cma_symbols: 10,
        phase_bw,
    };
    let new = |n: usize, m: usize, p: EqParams| {
        EqualizeState::new(vec![C32::default(); n], vec![C32::default(); m], p)
    };
    for (n, f) in [(3usize, 0usize), (5, 2), (11, 0), (63, 0), (3, 63), (15, 5)] {
        let st = new(n + f, n + f, params(f, 0.05, 0.02, 0.02)).unwrap();
        assert_eq!((st.taps(), st.feedback_taps()), (n, f));
        for (i, w) in st.weights().iter().enumerate() {
            assert_eq!(
                *w,
                if i == (n - 1) / 2 {
                    C32::new(1.0, 0.0)
                } else {
                    C32::default()
                }
            );
        }
        assert!(st.history().iter().all(|u| *u == C32::default()));
        assert_eq!((st.phase(), st.frequency()), (0, 0.0));
    }
    for n in [0usize, 1, 2, 4, 10, 62, 64, 65] {
        let e = new(n, n, params(0, 0.01, 0.01, 0.02)).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidArgument, "{n}");
    }
    // The forward part must be odd from 3 to 63; at most 63 feedback taps.
    for (len, f) in [(5usize, 3usize), (5, 6), (14, 4), (127, 64), (68, 4)] {
        let e = new(len, len, params(f, 0.01, 0.01, 0.02)).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidArgument, "{len} {f}");
    }
    assert_eq!(
        new(11, 12, params(0, 0.01, 0.01, 0.02)).unwrap_err(),
        Error::LengthMismatch {
            expected: 11,
            got: 12
        }
    );
    for mu in [1e-5f32, 0.5, 1.0] {
        assert!(new(11, 11, params(0, mu, mu, 0.02)).is_ok());
    }
    for mu in [0.0f32, 9e-6, 1.01, -0.01, f32::NAN, f32::INFINITY] {
        assert!(new(11, 11, params(0, mu, 0.01, 0.02)).is_err(), "{mu}");
        assert!(new(11, 11, params(0, 0.01, mu, 0.02)).is_err(), "{mu}");
    }
    for bw in [0.0f64, 0.0001, 0.2] {
        assert!(new(11, 11, params(0, 0.01, 0.01, bw)).is_ok(), "{bw}");
    }
    for bw in [5e-5f64, 0.21, -0.01, f64::NAN, f64::INFINITY] {
        assert!(new(11, 11, params(0, 0.01, 0.01, bw)).is_err(), "{bw}");
    }
    // The tracker gains are those of costas; phase_bw 0 gives 0.
    let st = new(11, 11, params(0, 0.01, 0.01, 0.02)).unwrap();
    let g = sdrk_core::kernels::costas::CostasState::new(4, 0.02, 0.7071, 0.05)
        .unwrap()
        .gains();
    assert_eq!(st.phase_gains(), (g.kp, g.ki));
    let st = new(11, 11, params(0, 0.01, 0.01, 0.0)).unwrap();
    assert_eq!(st.phase_gains(), (0.0, 0.0));
    // Garbage in the caller's storage is overwritten.
    let st = EqualizeState::new(
        [C32::new(5.0, 5.0); 3],
        [C32::new(7.0, 7.0); 3],
        EqParams {
            cma_symbols: 0,
            mu_cma: 0.01,
            mu_dd: 0.01,
            ..EqParams::default()
        },
    )
    .unwrap();
    assert_eq!(
        st.weights(),
        &[C32::default(), C32::new(1.0, 0.0), C32::default()]
    );
    assert_eq!(st.history(), &[C32::default(); 3]);
    assert_eq!(
        (st.mode(), st.mse(), st.cma_symbols()),
        (EqMode::Decision, 1.0, 0)
    );
    assert_eq!(state(5, q, 0.01, 1).mode(), EqMode::Cma);
    assert_eq!(st.constellation(), q);
    assert_eq!((st.mu_cma(), st.mu_dd()), (0.01, 0.01));
    assert_eq!(st.params().phase_bw, 0.02);
}

#[test]
fn timing_and_length_check() {
    for n in [3usize, 11, 63] {
        let st = state(n, EqConstellation::Bpsk, 0.01, 5);
        assert_eq!(st.ratio(), Ratio::ONE);
        assert_eq!(st.group_delay(), Delay::new((n as u64 - 1) / 2, 1).unwrap());
    }
    let mut st = state(5, EqConstellation::Bpsk, 0.01, 5);
    let before = st.clone();
    let mut out = [C32::default(); 3];
    let e = Fallback
        .equalize(&mut st, &[C32::new(1.0, 0.0); 4], &mut out)
        .unwrap_err();
    assert_eq!(
        e,
        Error::LengthMismatch {
            expected: 4,
            got: 3
        }
    );
    assert_eq!(st.weights(), before.weights());
    assert_eq!(st.history(), before.history());
}

#[test]
fn decisions_follow_the_rule_of_slice_and_costas() {
    let b = EqConstellation::Bpsk;
    assert_eq!(b.decide(C32::new(0.3, -9.0)), C32::new(1.0, 0.0));
    assert_eq!(b.decide(C32::new(-1e-30, 9.0)), C32::new(-1.0, 0.0));
    assert_eq!(b.decide(C32::new(-0.0, 0.0)), C32::new(1.0, 0.0));
    let q = EqConstellation::Qpsk;
    for (y, d) in [
        (C32::new(0.1, 0.1), C32::new(H, H)),
        (C32::new(-0.1, 0.1), C32::new(-H, H)),
        (C32::new(-0.1, -0.1), C32::new(-H, -H)),
        (C32::new(0.1, -0.1), C32::new(H, -H)),
        (C32::new(0.0, -0.0), C32::new(H, H)),
    ] {
        assert_eq!(q.decide(y), d, "{y}");
    }
    // 8PSK: every point decides on itself, and the octant boundaries k pi / 4 belong to k.
    for k in 0..8u32 {
        let a = f64::from(2 * k + 1) * core::f64::consts::PI / 8.0;
        let p = C32::new(a.cos() as f32, a.sin() as f32);
        assert_eq!(EqConstellation::Psk8.decide(p), p, "point {k}");
        let a = f64::from(k) * core::f64::consts::PI / 4.0 + 1e-6;
        let y = C32::new(a.cos() as f32, a.sin() as f32);
        assert_eq!(EqConstellation::Psk8.decide(y), p, "boundary {k}");
    }
    for c in [b, q, EqConstellation::Psk8] {
        assert_eq!(c.cma_r2(), 1.0);
    }
}

#[test]
fn clean_qpsk_through_an_identity_channel_keeps_the_centre_tap() {
    let mut rng = Rng::new(SEED_EQ);
    let x = qpsk(&mut rng, 4000);
    // Decision mode: the first symbols, while y is still near 0, give a large decision error
    // (the start-up transient); afterwards the weights return to the start within rounding and
    // the output is the input delayed by (N - 1) / 2 = 5 symbols.
    let mut st = state(11, EqConstellation::Qpsk, 0.1, 0);
    let mut y = vec![C32::default(); x.len()];
    Fallback.equalize(&mut st, &x, &mut y).unwrap();
    for (i, w) in st.weights().iter().enumerate() {
        let want = if i == 5 {
            C32::new(1.0, 0.0)
        } else {
            C32::default()
        };
        assert!((w - want).norm_sqr() < 1e-10, "tap {i}: {w}");
    }
    // After the start-up transient of the first symbols.
    for (k, (a, b)) in y[5..].iter().zip(&x).enumerate().skip(3500) {
        assert!((a - b).norm_sqr() < 1e-10, "symbol {k}: {a} {b}");
    }
    assert!(st.mse() < 1e-10, "mse {}", st.mse());
    // CMA: |y|^2 is 1 within rounding, so the weights stay within rounding of the start.
    let mut st = state(11, EqConstellation::Qpsk, 0.1, u32::MAX);
    Fallback.equalize(&mut st, &x, &mut y).unwrap();
    assert_eq!(st.mode(), EqMode::Cma);
    for (i, w) in st.weights().iter().enumerate() {
        let want = if i == 5 {
            C32::new(1.0, 0.0)
        } else {
            C32::default()
        };
        assert!((w - want).norm_sqr() < 1e-8, "tap {i}: {w}");
    }
}

/// The combined response `h * w` of the channel and the equalizer: its largest tap and the
/// power of all the others.
fn combined(h: &[C32], w: &[C32]) -> (f32, f32) {
    let mut c = vec![C32::default(); h.len() + w.len() - 1];
    for (i, &a) in h.iter().enumerate() {
        for (k, &b) in w.iter().enumerate() {
            c[i + k] += a * b;
        }
    }
    let p: Vec<f32> = c.iter().map(C32::norm_sqr).collect();
    let max = p.iter().copied().fold(0.0f32, f32::max);
    (max.sqrt(), p.iter().sum::<f32>() - max)
}

#[test]
fn cma_converges_on_a_two_tap_channel() {
    let mut rng = Rng::new(SEED_EQ + 1);
    let h = [
        C32::new(1.0, 0.0),
        C32::new(0.4 * 0.5f32.cos(), 0.4 * 0.5f32.sin()),
    ];
    let x = channel(&qpsk(&mut rng, 20000), &h);
    let (_, isi0) = combined(&h, &[C32::new(1.0, 0.0)]);
    let mut st = state_with(11, 0, EqConstellation::Qpsk, 0.05, u32::MAX, 0.02);
    let mut y = vec![C32::default(); x.len()];
    Fallback.equalize(&mut st, &x, &mut y).unwrap();
    assert_eq!(st.mode(), EqMode::Cma);
    let (main, isi) = combined(&h, st.weights());
    // The ISI of the channel (0.16 of the main tap's power) falls below -25 dB; the gain is 1
    // and the phase is free (CMA; the tracker turns the output, not the weights).
    assert!((main - 1.0).abs() < 0.05, "main tap {main}");
    assert!(
        isi < 3e-3 && isi < isi0 / 50.0,
        "isi {isi} (channel {isi0})"
    );
    let dispersion: f32 = y[19000..]
        .iter()
        .map(|v| (v.norm_sqr() - 1.0).abs())
        .sum::<f32>()
        / 1000.0;
    assert!(dispersion < 0.05, "dispersion {dispersion}");
}

#[test]
fn cma_then_decisions_remove_the_echo() {
    // The phase is right; the decision mode drives mse down.
    let mut rng = Rng::new(SEED_EQ + 2);
    let h = [C32::new(1.0, 0.0), C32::new(0.3, -0.2), C32::new(0.0, 0.1)];
    let x = channel(&qpsk(&mut rng, 8000), &h);
    let mut st = state_with(11, 0, EqConstellation::Qpsk, 0.02, 2000, 0.02);
    let mut y = vec![C32::default(); x.len()];
    Fallback.equalize(&mut st, &x, &mut y).unwrap();
    assert_eq!(st.mode(), EqMode::Decision);
    let db = 10.0 * st.mse().log10();
    assert!(db < -30.0, "mse {db} dB");
    let (main, isi) = combined(&h, st.weights());
    assert!(
        (main - 1.0).abs() < 0.02 && isi < 1e-3,
        "main {main}, isi {isi}"
    );
}

#[test]
fn the_tracker_removes_a_carrier_phase_and_a_frequency_offset() {
    // QPSK through 1 + 0.3 e^(-j0.6) z^-1, turned by 0.6 rad plus 0.002 cycles per symbol:
    // after CMA and the decision mode, the output z sits on the transmitted points (delayed by
    // 5), and the tracker's frequency integrator holds the offset. Without the tracker the
    // decision mode cannot follow the rotation.
    let mut rng = Rng::new(SEED_EQ + 8);
    let s = qpsk(&mut rng, 12000);
    let h = [C32::new(1.0, 0.0), polar(0.3, -0.6)];
    let x: Vec<C32> = channel(&s, &h)
        .iter()
        .enumerate()
        .map(|(k, v)| v * polar(1.0, 0.6 + core::f32::consts::TAU * 0.002 * k as f32))
        .collect();
    let mut st = state_with(11, 0, EqConstellation::Qpsk, 0.05, 1000, 0.02);
    let mut y = vec![C32::default(); x.len()];
    Fallback.equalize(&mut st, &x, &mut y).unwrap();
    assert!(st.mse() < 0.01, "mse {}", st.mse());
    assert!((st.frequency() - 0.002).abs() < 2e-4, "{}", st.frequency());
    // The tail decides like the transmitted symbols up to a fixed quarter turn.
    let turn = (0..4)
        .min_by_key(|&r| {
            y[11_000..]
                .iter()
                .zip(&s[10_995..])
                .filter(|(a, b)| {
                    let d = EqConstellation::Qpsk.decide(quarter(**a, r));
                    (d - **b).norm_sqr() > 1e-6
                })
                .count()
        })
        .unwrap();
    let errors = y[11_000..]
        .iter()
        .zip(&s[10_995..])
        .filter(|(a, b)| {
            let d = EqConstellation::Qpsk.decide(quarter(**a, turn));
            (d - **b).norm_sqr() > 1e-6
        })
        .count();
    assert_eq!(errors, 0);
    let mut off = state(11, EqConstellation::Qpsk, 0.05, 1000);
    Fallback.equalize(&mut off, &x, &mut y).unwrap();
    assert!(off.mse() > 0.1, "mse without the tracker {}", off.mse());
}

#[test]
fn decision_feedback_cancels_the_postcursors() {
    // BPSK through 1 + 0.9 z^-1 + 0.5 z^-2 (a deep null for a linear equalizer): with 2 feedback
    // taps the feedback weights learn the postcursors (as d e^(j theta) with theta 0 here) and
    // mse ends far below that of the linear equalizer of the same forward length.
    let mut rng = Rng::new(SEED_EQ + 9);
    let s: Vec<C32> = (0..20000)
        .map(|_| C32::new(if rng.next() < 0.0 { -1.0 } else { 1.0 }, 0.0))
        .collect();
    let h = [C32::new(1.0, 0.0), C32::new(0.9, 0.0), C32::new(0.5, 0.0)];
    let x: Vec<C32> = channel(&s, &h)
        .iter()
        .map(|v| v + C32::new(rng.next() * 0.02, rng.next() * 0.02))
        .collect();
    let mut dfe = state_with(5, 2, EqConstellation::Bpsk, 0.05, 0, 0.0);
    let mut lin = state_with(5, 0, EqConstellation::Bpsk, 0.05, 0, 0.0);
    let mut y = vec![C32::default(); x.len()];
    Fallback.equalize(&mut dfe, &x, &mut y).unwrap();
    Fallback.equalize(&mut lin, &x, &mut y).unwrap();
    assert!(dfe.mse() < 1e-3, "dfe mse {}", dfe.mse());
    assert!(
        lin.mse() > 10.0 * dfe.mse(),
        "linear {} dfe {}",
        lin.mse(),
        dfe.mse()
    );
    // The feedback history holds the last two decisions, newest first.
    let fb = &dfe.history()[5..];
    assert!(
        fb.iter().all(|d| d.im == 0.0 && d.re.abs() == 1.0),
        "{fb:?}"
    );
    // In CMA mode nothing is fed back.
    let mut cma = state_with(5, 2, EqConstellation::Bpsk, 0.05, u32::MAX, 0.0);
    Fallback
        .equalize(&mut cma, &x[..500], &mut y[..500])
        .unwrap();
    assert!(cma.history()[5..].iter().all(|d| *d == C32::default()));
    assert!(cma.weights()[5..].iter().all(|w| *w == C32::default()));
}

#[test]
fn the_mode_switches_once_after_cma_symbols() {
    let mut rng = Rng::new(SEED_EQ + 3);
    let x = qpsk(&mut rng, 40);
    let mut st = state(5, EqConstellation::Qpsk, 0.01, 10);
    let mut y = [C32::default(); 40];
    for k in 0..40 {
        assert_eq!(
            st.mode(),
            if k < 10 {
                EqMode::Cma
            } else {
                EqMode::Decision
            },
            "{k}"
        );
        Fallback
            .equalize(&mut st, &x[k..=k], &mut y[k..=k])
            .unwrap();
    }
    // A non-finite symbol counts as a symbol.
    let mut st = state(5, EqConstellation::Qpsk, 0.01, 3);
    let nan = [C32::new(f32::NAN, 0.0); 3];
    Fallback.equalize(&mut st, &nan, &mut y[..3]).unwrap();
    assert_eq!(st.mode(), EqMode::Decision);
    // u32::MAX never switches; 0 starts in decision mode.
    let mut st = state(5, EqConstellation::Qpsk, 0.01, u32::MAX);
    Fallback.equalize(&mut st, &x, &mut y).unwrap();
    assert_eq!(st.mode(), EqMode::Cma);
    assert_eq!(
        state(5, EqConstellation::Qpsk, 0.01, 0).mode(),
        EqMode::Decision
    );
}

#[test]
fn reset_returns_to_the_start() {
    let mut rng = Rng::new(SEED_EQ + 4);
    let x = channel(
        &qpsk(&mut rng, 500),
        &[C32::new(1.0, 0.0), C32::new(0.5, 0.5)],
    );
    let mut st = state_with(7, 2, EqConstellation::Psk8, 0.1, 100, 0.05);
    let fresh = st.clone();
    let mut y = vec![C32::default(); x.len()];
    Fallback.equalize(&mut st, &x, &mut y).unwrap();
    assert_eq!(st.mode(), EqMode::Decision);
    st.set_mu_cma(0.02).unwrap();
    st.reset();
    assert_eq!(st.weights(), fresh.weights());
    assert_eq!(st.history(), fresh.history());
    assert_eq!((st.mode(), st.mse()), (EqMode::Cma, 1.0));
    assert_eq!((st.phase(), st.frequency()), (0, 0.0));
    // The parameters are kept, the run repeats bit for bit.
    assert_eq!(st.mu_cma(), 0.02);
    st.set_mu_cma(0.1).unwrap();
    let mut z = vec![C32::default(); x.len()];
    Fallback.equalize(&mut st, &x, &mut z).unwrap();
    assert_eq!(bits(&y), bits(&z));
}

#[test]
fn non_finite_input_gives_zero_and_keeps_the_weights() {
    let mut rng = Rng::new(SEED_EQ + 5);
    let x = channel(
        &qpsk(&mut rng, 300),
        &[C32::new(1.0, 0.0), C32::new(0.2, 0.1)],
    );
    let mut st = state_with(5, 1, EqConstellation::Qpsk, 0.1, 50, 0.02);
    let mut y = vec![C32::default(); 300];
    Fallback.equalize(&mut st, &x, &mut y).unwrap();
    for bad in [
        C32::new(f32::NAN, 0.5),
        C32::new(0.5, f32::INFINITY),
        C32::new(f32::NEG_INFINITY, f32::NAN),
    ] {
        let (w, mse) = (st.weights().to_vec(), st.mse());
        let (phase, freq) = (st.phase(), st.frequency());
        let mut o = [C32::new(9.0, 9.0)];
        Fallback.equalize(&mut st, &[bad], &mut o).unwrap();
        assert_eq!(o[0], C32::default());
        assert_eq!((st.weights(), st.mse()), (&w[..], mse));
        assert_eq!((st.phase(), st.frequency()), (phase, freq));
        // Zero enters the forward history and the fed-back decisions.
        assert_eq!(st.history()[0], C32::default());
        assert_eq!(st.history()[5], C32::default());
        assert!(
            st.history()
                .iter()
                .all(|u| u.re.is_finite() && u.im.is_finite())
        );
    }
    // Afterwards the output is finite again.
    let mut o = vec![C32::default(); 50];
    Fallback.equalize(&mut st, &x[..50], &mut o).unwrap();
    assert!(o.iter().all(|v| v.re.is_finite() && v.im.is_finite()));
}

#[test]
fn silence_full_scale_and_overflow() {
    // Silence: zero output, the CMA error is zero, nothing moves.
    let mut st = state(11, EqConstellation::Qpsk, 0.1, 100);
    let start = st.weights().to_vec();
    let mut y = vec![C32::new(1.0, 1.0); 200];
    Fallback
        .equalize(&mut st, &vec![C32::default(); 200], &mut y)
        .unwrap();
    assert!(y.iter().all(|v| *v == C32::default()));
    assert_eq!(st.weights(), &start[..]);
    // Full scale (|x| = 1 on both components) stays finite.
    let mut rng = Rng::new(SEED_EQ + 6);
    let x: Vec<C32> = (0..2000)
        .map(|_| {
            C32::new(
                if rng.next() < 0.0 { -1.0 } else { 1.0 },
                if rng.next() < 0.0 { -1.0 } else { 1.0 },
            )
        })
        .collect();
    let mut st = state(11, EqConstellation::Qpsk, 0.1, 500);
    let mut y = vec![C32::default(); x.len()];
    Fallback.equalize(&mut st, &x, &mut y).unwrap();
    assert!(y.iter().all(|v| v.re.is_finite() && v.im.is_finite()));
    // Huge finite inputs: the normalization by the window power keeps the step small, a
    // non-finite step leaves the weights; the state stays finite and the output too.
    for v in [1e10f32, 1e12, 1e15, 1e19, f32::MAX] {
        let mut st = state_with(5, 2, EqConstellation::Qpsk, 1.0, 0, 0.2);
        let mut y = vec![C32::default(); 20];
        Fallback
            .equalize(&mut st, &[C32::new(v, -v); 20], &mut y)
            .unwrap();
        assert!(
            st.weights()
                .iter()
                .all(|w| w.re.is_finite() && w.im.is_finite()),
            "{v}"
        );
        assert!(st.mse().is_finite() && st.frequency().is_finite());
        assert!(
            y.iter().all(|o| o.re.is_finite() && o.im.is_finite()),
            "{v}"
        );
    }
}

#[test]
fn block_size_invariance() {
    let mut rng = Rng::new(SEED_PLANS);
    let x = channel(
        &qpsk(&mut rng, 3000),
        &[C32::new(1.0, 0.0), C32::new(0.3, 0.4)],
    );
    for c in [
        EqConstellation::Bpsk,
        EqConstellation::Qpsk,
        EqConstellation::Psk8,
    ] {
        let mut whole = state_with(9, 2, c, 0.05, 700, 0.02);
        let mut y = vec![C32::default(); x.len()];
        Fallback.equalize(&mut whole, &x, &mut y).unwrap();
        for _plan in 0..20 {
            let mut st = state_with(9, 2, c, 0.05, 700, 0.02);
            let mut z = vec![C32::default(); x.len()];
            let mut at = 0;
            while at < x.len() {
                let len = (1 + rng.below(400)).min(x.len() - at);
                Fallback
                    .equalize(&mut st, &x[at..at + len], &mut z[at..at + len])
                    .unwrap();
                at += len;
            }
            assert_eq!(bits(&y), bits(&z));
            assert_eq!(st.weights(), whole.weights());
            assert_eq!(st.history(), whole.history());
            assert_eq!(
                (
                    st.mode(),
                    st.mse().to_bits(),
                    st.phase(),
                    st.frequency().to_bits()
                ),
                (
                    whole.mode(),
                    whole.mse().to_bits(),
                    whole.phase(),
                    whole.frequency().to_bits()
                )
            );
        }
    }
}

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<EqMode>();
    assert_send::<EqConstellation>();
    assert_send::<EqParams>();
    assert_sync::<EqParams>();
    assert_send::<EqualizeState<[C32; 11], [C32; 11]>>();
    assert_send::<EqualizeState<&mut [C32], &mut [C32]>>();
    assert_sync::<EqualizeState<[C32; 11], [C32; 11]>>();
}

#[cfg(feature = "reference")]
mod against_the_reference {
    use super::*;
    use sdrk_core::Reference;

    const SPECIALS: [f32; 9] = [
        0.0,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE / 2.0,
        f32::MAX,
        1e15,
        1.0,
    ];

    #[test]
    fn dot_complex_fallback_equals_reference() {
        let mut rng = Rng::new(SEED_DOT + 2);
        for n in 0..=100 {
            for _ in 0..4 {
                let mut draw = || {
                    if rng.below(5) == 0 {
                        SPECIALS[rng.below(SPECIALS.len())]
                    } else {
                        rng.next() * 100.0
                    }
                };
                let x: Vec<C32> = (0..n).map(|_| C32::new(draw(), draw())).collect();
                let w: Vec<C32> = (0..n).map(|_| C32::new(draw(), draw())).collect();
                assert_eq!(
                    bits(&[Fallback.dot_complex(&x, &w).unwrap()]),
                    bits(&[Reference.dot_complex(&x, &w).unwrap()]),
                    "n {n}"
                );
            }
        }
    }

    #[test]
    fn equalize_fallback_equals_reference() {
        let mut rng = Rng::new(SEED_EQ + 7);
        for (k, c) in [
            EqConstellation::Bpsk,
            EqConstellation::Qpsk,
            EqConstellation::Psk8,
        ]
        .into_iter()
        .enumerate()
        {
            for (taps, feedback, phase_bw) in [(3usize, 0usize, 0.0), (11, 3, 0.02), (51, 12, 0.2)]
            {
                let mut x = channel(
                    &qpsk(&mut rng, 3000),
                    &[C32::new(1.0, 0.0), C32::new(0.3, -0.1)],
                );
                for v in &mut x {
                    *v += C32::new(rng.next() * 0.1, rng.next() * 0.1);
                }
                for (j, s) in SPECIALS.iter().enumerate() {
                    x[500 + 7 * j] = C32::new(*s, 0.3);
                    x[900 + 7 * j] = C32::new(-0.3, *s);
                }
                let mu = [1.0f32, 0.03, 1e-5][k];
                let mut a = state_with(taps, feedback, c, mu, 1000, phase_bw);
                let mut b = a.clone();
                let mut ya = vec![C32::default(); x.len()];
                let mut yb = vec![C32::default(); x.len()];
                Fallback.equalize(&mut a, &x, &mut ya).unwrap();
                Reference.equalize(&mut b, &x, &mut yb).unwrap();
                assert_eq!(bits(&ya), bits(&yb), "{c:?} taps {taps}");
                assert_eq!(bits(a.weights()), bits(b.weights()));
                assert_eq!(bits(a.history()), bits(b.history()));
                assert_eq!((a.mode(), a.mse().to_bits()), (b.mode(), b.mse().to_bits()));
                assert_eq!(
                    (a.phase(), a.frequency().to_bits()),
                    (b.phase(), b.frequency().to_bits())
                );
            }
        }
    }
}
