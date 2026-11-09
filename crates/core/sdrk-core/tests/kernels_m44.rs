//! Tests of the M44 kernel paths of `sdrk-core`: the `costas` constellations `qam16` and `qam64`
//! (the decision-directed detector `arg(y conj d)` against the grid decision, the lock metric
//! `cos(8 err)` with the QAM decision; M44 §3) and the `equalize` constellations `qam16` and
//! `qam64` (the grid decision and the CMA radius). Lock and frequency tracking on noisy QAM, the
//! detector against `libm::atan2f`, the lock levels on QAM and on noise (D1208), block-size
//! invariance, `reset`, silence, full scale and beyond, non-finite input, and the convergence of
//! the equalizer on 16-QAM and 64-QAM through a channel. The comparison of `Fallback` with
//! `Reference` runs only with the feature `reference`. Seeds 600 to 607 of M44 (this file: 605
//! for the loops, 604 for the channels).

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::many_single_char_names
)]

use sdrk_core::kernels::constellation::Qam;
use sdrk_core::kernels::costas::{Costas, CostasConstellation, CostasState};
use sdrk_core::kernels::equalize::{EqConstellation, EqMode, EqParams, Equalize, EqualizeState};
use sdrk_core::{C32, Fallback};

#[allow(
    clippy::approx_constant,
    reason = "the damping of the block defaults, written as in the spec"
)]
const DAMP: f64 = 0.7071;
/// The seed of the carrier-loop signals (M44 test 6).
const SEED: u64 = 605;
/// The seed of the equalizer signals (M44 test 5).
const SEED_EQ: u64 = 604;
/// The level above which the lock detector turns on.
const LOCK_ON_LEVEL: f32 = 0.5;

const QAMS: [(Qam, CostasConstellation); 2] = [
    (Qam::Qam16, CostasConstellation::Qam16),
    (Qam::Qam64, CostasConstellation::Qam64),
];

/// Deterministic generator (xorshift64*), uniform in [-1, 1).
struct Rng(u64);

impl Rng {
    fn new(stream: u64) -> Self {
        Self(SEED * 1000 + stream)
    }

    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (v as f64 / (1u64 << 52) as f64 - 1.0) as f32
    }

    fn symbol(&mut self, q: Qam) -> u32 {
        let v = f32::midpoint(self.next(), 1.0) * q.points() as f32;
        (v as u32).min(q.points() as u32 - 1)
    }
}

/// Random points of `q` turned by `phase0 + 2 pi offset k` plus uniform noise of `noise` per
/// component; also returns the symbols.
fn qam_symbols(
    stream: u64,
    n: usize,
    q: Qam,
    phase0: f64,
    offset: f64,
    noise: f32,
) -> (Vec<C32>, Vec<u32>) {
    let mut rng = Rng::new(stream);
    let mut syms = Vec::with_capacity(n);
    let x = (0..n)
        .map(|k| {
            let s = rng.symbol(q);
            syms.push(s);
            let p = q.point(s);
            let turns = (phase0 / core::f64::consts::TAU + offset * k as f64).rem_euclid(1.0);
            let (sn, c) = (core::f64::consts::TAU * turns).sin_cos();
            let (sn, c) = (sn as f32, c as f32);
            C32::new(
                p.re * c - p.im * sn + noise * rng.next(),
                p.re * sn + p.im * c + noise * rng.next(),
            )
        })
        .collect();
    (x, syms)
}

fn run(st: &mut CostasState, x: &[C32]) -> Vec<C32> {
    let mut y = vec![C32::default(); x.len()];
    Fallback.costas(st, x, &mut y).unwrap();
    y
}

fn bits(x: &[C32]) -> Vec<(u32, u32)> {
    x.iter().map(|v| (v.re.to_bits(), v.im.to_bits())).collect()
}

/// Symbol errors of the decisions of `y` against `syms`, under the quarter turn with the fewest.
fn errors_best_rotation(q: Qam, y: &[C32], syms: &[u32]) -> usize {
    (0..4)
        .map(|r| {
            y.iter()
                .zip(syms)
                .filter(|&(v, &s)| {
                    // `j^r p` of the transmitted point against the decision.
                    let mut p = q.point(s);
                    for _ in 0..r {
                        p = C32::new(-p.im, p.re);
                    }
                    let d = q.decide(*v);
                    (d.re - p.re).abs() > 1e-6 || (d.im - p.im).abs() > 1e-6
                })
                .count()
        })
        .min()
        .unwrap()
}

#[test]
fn the_constellations_have_their_detectors() {
    for (q, c) in QAMS {
        assert_eq!(c.qam(), Some(q));
        assert_eq!(c.psk_order(), None);
        let st = CostasState::new(c, 0.005, DAMP, 0.05).unwrap();
        assert_eq!(st.constellation(), c);
    }
    for (c, order) in [
        (CostasConstellation::Bpsk, 2),
        (CostasConstellation::Qpsk, 4),
        (CostasConstellation::Psk8, 8),
    ] {
        assert_eq!((c.psk_order(), c.qam()), (Some(order), None));
    }
}

/// The detector is the exact phase `atan2f(im(y conj d), re(y conj d))` against the grid
/// decision: from a fresh loop the first symbol moves the integrator to `ki e` exactly.
#[test]
fn the_qam_detector_is_the_phase_against_the_grid_decision() {
    for (q, c) in QAMS {
        let mut rng = Rng::new(1);
        for _ in 0..500 {
            let p = q.point(rng.symbol(q));
            let a = 0.2 * rng.next();
            let y = C32::new(
                p.re * a.cos() - p.im * a.sin() + 0.002 * rng.next(),
                p.re * a.sin() + p.im * a.cos() + 0.002 * rng.next(),
            );
            let mut st = CostasState::new(c, 0.02, DAMP, 0.25).unwrap();
            let out = run(&mut st, &[y]);
            // phase 0: the derotation is the table's (cos 0, sin 0) = (1, 0) up to its rounding
            let d = q.decide(out[0]);
            let r = out[0].re * d.re + out[0].im * d.im;
            let i = out[0].im * d.re - out[0].re * d.im;
            let e = libm::atan2f(i, r);
            let want = st.gains().ki * e;
            assert_eq!(st.frequency() as f32, want, "{q:?} {y}");
            // the sign convention of M40: a sample ahead of its point gives e > 0
            if a > 0.05 && d == p {
                assert!(e > 0.0, "{q:?} {a} {e}");
            }
        }
    }
}

#[test]
fn qam_locks_onto_the_grid_and_follows_an_offset() {
    for (q, c) in QAMS {
        for (stream, phase0, offset) in [(10, 0.3, 0.0), (11, -0.5, 0.001), (12, 0.6, -0.002)] {
            let noise = if q == Qam::Qam16 { 0.05 } else { 0.02 };
            let (x, syms) = qam_symbols(stream, 8000, q, phase0, offset, noise);
            let mut st = CostasState::new(c, 0.02, DAMP, 0.05).unwrap();
            let y = run(&mut st, &x);
            assert!(st.locked(), "{q:?} {phase0} {offset}: {}", st.lock_level());
            assert!(
                (st.frequency() - offset).abs() < 2e-4,
                "{q:?} {phase0} {offset}: {}",
                st.frequency()
            );
            assert_eq!(
                errors_best_rotation(q, &y[4000..], &syms[4000..]),
                0,
                "{q:?} {phase0} {offset}"
            );
        }
    }
}

/// D1208: the lock metric `cos(8 err)` against the grid decision. On noisy 16-QAM it is well above
/// the on level and on noise near 0, so the 16-QAM loop locks on the signal and never on noise;
/// on 64-QAM the dense grid keeps every sample close to a point, so the metric stays near the on
/// level on noise as well (open item I44-2). The measured levels are guarded.
#[test]
#[allow(
    clippy::print_stdout,
    reason = "the measured lock levels of the QAM metric (D1208)"
)]
fn qam_lock_levels_on_the_signal_and_on_noise() {
    let mut noise_rng = Rng::new(41);
    let noise: Vec<C32> = (0..20_000)
        .map(|_| C32::new(noise_rng.next(), noise_rng.next()))
        .collect();
    for (q, c) in QAMS {
        let (x, _) = qam_symbols(40, 4000, q, 0.4, 0.0, 0.02);
        let mut st = CostasState::new(c, 0.02, DAMP, 0.05).unwrap();
        run(&mut st, &x);
        let signal = st.lock_level();
        assert!(st.locked() && signal > 0.8, "{q:?}: {signal}");
        let mut st = CostasState::new(c, 0.02, DAMP, 0.05).unwrap();
        let mut levels = Vec::new();
        let mut on = 0usize;
        let mut y = [C32::default(); 1];
        for v in &noise {
            Fallback
                .costas(&mut st, core::slice::from_ref(v), &mut y)
                .unwrap();
            on += usize::from(st.locked());
            levels.push(st.lock_level());
        }
        let tail = &levels[2000..];
        let mean = tail.iter().map(|&l| f64::from(l)).sum::<f64>() / tail.len() as f64;
        let max = tail.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "{q:?} lock level: signal {signal:.4}; uniform noise mean {mean:.4}, max {max:.4}, \
             locked in {on} of {} symbols",
            noise.len()
        );
        if q == Qam::Qam16 {
            assert_eq!(on, 0, "16-QAM locked on noise");
            assert!(mean.abs() < 0.3 && max < LOCK_ON_LEVEL, "{mean} {max}");
        } else {
            // the measured value of I44-2, guarded against regressions in both directions
            assert!((0.3..0.8).contains(&mean), "{mean}");
        }
    }
}

#[test]
fn output_does_not_depend_on_the_block_size() {
    for (stream, (q, c)) in QAMS.into_iter().enumerate() {
        let (x, _) = qam_symbols(50 + stream as u64, 1500, q, 2.0, 0.004, 0.1);
        let mut whole = CostasState::new(c, 0.03, DAMP, 0.05).unwrap();
        let want = run(&mut whole, &x);
        for size in [1usize, 2, 3, 7, 64, 499, 1500] {
            let mut st = CostasState::new(c, 0.03, DAMP, 0.05).unwrap();
            let mut y = vec![C32::default(); x.len()];
            for (xi, yi) in x.chunks(size).zip(y.chunks_mut(size)) {
                Fallback.costas(&mut st, xi, yi).unwrap();
            }
            assert_eq!(bits(&y), bits(&want), "{q:?}, size {size}");
            assert_eq!(st, whole);
        }
    }
}

#[test]
fn reset_returns_to_the_new_state() {
    for (q, c) in QAMS {
        let mut st = CostasState::new(c, 0.02, DAMP, 0.05).unwrap();
        let (x, _) = qam_symbols(60, 1000, q, 1.0, 0.002, 0.05);
        let first = run(&mut st, &x);
        st.reset();
        assert_eq!(st, CostasState::new(c, 0.02, DAMP, 0.05).unwrap());
        assert_eq!(bits(&run(&mut st, &x)), bits(&first));
    }
}

#[test]
fn silence_gives_zeros_and_a_resting_loop() {
    for (_, c) in QAMS {
        let mut st = CostasState::new(c, 0.02, DAMP, 0.05).unwrap();
        st.set_phase(0x1234_5678);
        let y = run(&mut st, &[C32::new(0.0, 0.0); 500]);
        assert!(y.iter().all(|v| v.re == 0.0 && v.im == 0.0));
        // atan2f(0, 0) = 0 and the metric 0 / 0 is skipped: nothing moves.
        assert_eq!(st.phase(), 0x1234_5678);
        assert_eq!(
            (st.frequency(), st.lock_level(), st.locked()),
            (0.0, 0.0, false)
        );
    }
}

#[test]
fn full_scale_and_beyond_keep_the_loop_bounded() {
    for (q, c) in QAMS {
        for amp in [1.0f32, 1e18, 1e30, f32::MAX] {
            let (x, _) = qam_symbols(70, 600, q, 0.8, 0.003, 0.0);
            let x: Vec<C32> = x.iter().map(|v| C32::new(v.re * amp, v.im * amp)).collect();
            let mut st = CostasState::new(c, 0.02, DAMP, 0.05).unwrap();
            run(&mut st, &x);
            assert!(st.frequency().abs() <= 0.05 + 1e-9, "{q:?} {amp}");
            assert!(st.lock_level().is_finite() && st.lock_level().abs() <= 1.0 + 1e-6);
        }
    }
}

#[test]
fn non_finite_input_stays_in_its_own_symbols() {
    for (q, c) in QAMS {
        let (mut x, _) = qam_symbols(80, 3000, q, 0.2, 0.0, 0.01);
        for (i, v) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
            .into_iter()
            .enumerate()
        {
            x[2000 + i] = C32::new(v, 0.5);
            x[2010 + i] = C32::new(0.5, v);
        }
        let mut st = CostasState::new(c, 0.02, DAMP, 0.05).unwrap();
        let y = run(&mut st, &x);
        assert!(
            y[..2000]
                .iter()
                .all(|v| v.re.is_finite() && v.im.is_finite())
        );
        assert!(!y[2000].re.is_finite() && !y[2010].im.is_finite());
        assert!(
            y[2013..]
                .iter()
                .all(|v| v.re.is_finite() && v.im.is_finite())
        );
        assert!(st.locked() && st.frequency().is_finite(), "{q:?}");
    }
}

// --------------------------------------------------------------------------------------------
// equalize
// --------------------------------------------------------------------------------------------

#[test]
fn the_equalizer_decides_on_the_grid_with_the_qam_radius() {
    assert_eq!(EqConstellation::Qam16.cma_r2(), (33.0f64 / 25.0) as f32);
    assert_eq!(EqConstellation::Qam64.cma_r2(), (29.0f64 / 21.0) as f32);
    assert!((f64::from(EqConstellation::Qam16.cma_r2()) - 1.32).abs() < 5e-4);
    assert!((f64::from(EqConstellation::Qam64.cma_r2()) - 1.381).abs() < 5e-4);
    for (q, e) in [
        (Qam::Qam16, EqConstellation::Qam16),
        (Qam::Qam64, EqConstellation::Qam64),
    ] {
        let mut rng = Rng::new(90);
        for _ in 0..2000 {
            let y = C32::new(1.5 * rng.next(), 1.5 * rng.next());
            assert_eq!(e.decide(y), q.decide(y));
        }
        for &p in q.point_table() {
            assert_eq!(e.decide(p), p);
        }
        let d = e.decide(C32::new(f32::NAN, f32::INFINITY));
        let l = q.levels();
        assert_eq!(d, C32::new(l[0], l[l.len() - 1]));
    }
}

/// QAM through a symbol-spaced channel with a carrier phase: CMA with the QAM radius, then the
/// decisions on the grid with the phase tracker; after convergence every decision is the
/// transmitted symbol (delayed by the centre tap, turned by the quarter turn of the blind
/// start), and the blocks cut the input in any way without changing a bit.
#[test]
fn the_equalizer_converges_on_qam_through_a_channel() {
    for (q, e, cma_symbols, noise) in [
        (Qam::Qam16, EqConstellation::Qam16, 4000u32, 0.02f32),
        (Qam::Qam64, EqConstellation::Qam64, 6000, 0.005),
    ] {
        let n = 20_000;
        let mut rng = Rng(SEED_EQ * 1000 + 1);
        let syms: Vec<u32> = (0..n).map(|_| rng.symbol(q)).collect();
        let s: Vec<C32> = syms.iter().map(|&k| q.point(k)).collect();
        let (h1, h2) = (C32::new(0.2, 0.12), C32::new(-0.05, 0.06));
        let turn = C32::new(0.2f32.cos(), 0.2f32.sin());
        let x: Vec<C32> = (0..n)
            .map(|k| {
                let mut v = s[k];
                if k >= 1 {
                    v += s[k - 1] * h1;
                }
                if k >= 2 {
                    v += s[k - 2] * h2;
                }
                v * turn + C32::new(noise * rng.next(), noise * rng.next())
            })
            .collect();
        let p = EqParams {
            constellation: e,
            mu_cma: 0.02,
            mu_dd: 0.01,
            cma_symbols,
            ..EqParams::default()
        };
        let mut st =
            EqualizeState::new(vec![C32::default(); 11], vec![C32::default(); 11], p).unwrap();
        let mut y = vec![C32::default(); n];
        Fallback.equalize(&mut st, &x, &mut y).unwrap();
        assert_eq!(st.mode(), EqMode::Decision);
        assert!(st.mse() < 0.01, "{q:?}: mse {}", st.mse());
        let delay = 5;
        assert_eq!(
            errors_best_rotation(q, &y[n / 2..], &syms[n / 2 - delay..n - delay]),
            0,
            "{q:?}"
        );
        // block-size invariance
        for size in [1usize, 7, 64, 333] {
            let mut st2 =
                EqualizeState::new(vec![C32::default(); 11], vec![C32::default(); 11], p).unwrap();
            let mut y2 = vec![C32::default(); n];
            for (a, b) in x.chunks(size).zip(y2.chunks_mut(size)) {
                Fallback.equalize(&mut st2, a, b).unwrap();
            }
            assert_eq!(bits(&y2), bits(&y), "{q:?} size {size}");
        }
    }
}

#[cfg(feature = "reference")]
mod against_the_reference {
    use super::*;
    use sdrk_core::Reference;

    const SPECIALS: [f32; 8] = [
        0.0,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE / 2.0,
        f32::MAX,
        1.0,
    ];

    fn same(a: &[C32], b: &[C32]) -> bool {
        a.iter().zip(b).all(|(p, q)| {
            (p.re.to_bits() == q.re.to_bits() || (p.re.is_nan() && q.re.is_nan()))
                && (p.im.to_bits() == q.im.to_bits() || (p.im.is_nan() && q.im.is_nan()))
        })
    }

    #[test]
    fn costas_fallback_equals_reference() {
        for (q, c) in QAMS {
            for (k, offset) in [0.0, 0.002, -0.01].into_iter().enumerate() {
                let (mut x, _) = qam_symbols(95 + k as u64, 3000, q, 1.0, offset, 0.1);
                for (j, s) in SPECIALS.iter().enumerate() {
                    x[500 + 7 * j] = C32::new(*s, 0.3);
                    x[900 + 7 * j] = C32::new(-0.3, *s);
                }
                let mut a = CostasState::new(c, 0.05, DAMP, 0.05).unwrap();
                let mut b = a;
                let mut ya = vec![C32::default(); x.len()];
                let mut yb = vec![C32::default(); x.len()];
                Fallback.costas(&mut a, &x, &mut ya).unwrap();
                Reference.costas(&mut b, &x, &mut yb).unwrap();
                assert!(same(&ya, &yb), "{q:?}, offset {offset}");
                assert_eq!(a, b);
            }
            // Every pair of special values, and values on the decision boundaries.
            let mut x: Vec<C32> = SPECIALS
                .iter()
                .flat_map(|&a| SPECIALS.iter().map(move |&b| C32::new(a, b)))
                .collect();
            let l = q.levels();
            for w in l.windows(2) {
                let mid = f32::midpoint(w[0], w[1]);
                x.push(C32::new(mid, w[0]));
                x.push(C32::new(w[1], mid));
            }
            let mut a = CostasState::new(c, 0.2, 4.0, 0.25).unwrap();
            let mut b = a;
            let mut ya = vec![C32::default(); x.len()];
            let mut yb = vec![C32::default(); x.len()];
            Fallback.costas(&mut a, &x, &mut ya).unwrap();
            Reference.costas(&mut b, &x, &mut yb).unwrap();
            assert!(same(&ya, &yb), "{q:?} specials");
            assert_eq!(a, b, "{q:?}");
        }
    }

    #[test]
    fn equalize_fallback_equals_reference() {
        for (q, e) in [
            (Qam::Qam16, EqConstellation::Qam16),
            (Qam::Qam64, EqConstellation::Qam64),
        ] {
            for feedback in [0usize, 3] {
                let (mut x, _) = qam_symbols(97, 3000, q, 0.3, 0.0, 0.05);
                for (j, s) in SPECIALS.iter().enumerate() {
                    x[700 + 5 * j] = C32::new(*s, 0.2);
                }
                let p = EqParams {
                    constellation: e,
                    feedback_taps: feedback,
                    cma_symbols: 1000,
                    ..EqParams::default()
                };
                let len = 11 + feedback;
                let mk = || {
                    EqualizeState::new(vec![C32::default(); len], vec![C32::default(); len], p)
                        .unwrap()
                };
                let (mut a, mut b) = (mk(), mk());
                let mut ya = vec![C32::default(); x.len()];
                let mut yb = vec![C32::default(); x.len()];
                Fallback.equalize(&mut a, &x, &mut ya).unwrap();
                Reference.equalize(&mut b, &x, &mut yb).unwrap();
                assert!(same(&ya, &yb), "{q:?} feedback {feedback}");
                assert!(same(a.weights(), b.weights()));
            }
        }
    }
}

#[test]
fn new_public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CostasConstellation>();
    assert_send_sync::<Qam>();
}
