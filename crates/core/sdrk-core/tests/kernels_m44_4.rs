//! Tests of the M44.4 §3 blind start of `equalize` for the dense grids (seed 907): 256-QAM and
//! 1024-QAM start with the multi-modulus cost (`EqBlind::Mma`, the radius `Qam::mma_r2` of one
//! axis), every other constellation keeps the constant-modulus cost; the multi-modulus start turns
//! the output onto the axes of the grid without the phase tracker, which the constant-modulus
//! start cannot; the split of a call at the switch to the decision mode (the blind symbols in
//! their own loop, the rest in the loop of the decision mode) keeps the bits for every cut, with
//! and without feedback taps, across `reset`; and with the feature `reference` `Fallback` against
//! `Reference` over the blind start (never switching, and with special values).

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing
)]

use sdrk_core::kernels::constellation::Qam;
use sdrk_core::kernels::equalize::{
    EqBlind, EqConstellation, EqMode, EqParams, Equalize, EqualizeState,
};
use sdrk_core::{C32, Fallback};

/// The seed of the signals of this file.
const SEED: u64 = 907;

/// The dense orders with the multi-modulus start.
const DENSE: [(Qam, EqConstellation); 2] = [
    (Qam::Qam256, EqConstellation::Qam256),
    (Qam::Qam1024, EqConstellation::Qam1024),
];

type State = EqualizeState<Vec<C32>, Vec<C32>>;

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

    fn unit_f64(&mut self) -> f64 {
        f64::from(f32::midpoint(self.next(), 1.0))
    }

    fn symbol(&mut self, q: Qam) -> u32 {
        let v = f32::midpoint(self.next(), 1.0) * q.points() as f32;
        (v as u32).min(q.points() as u32 - 1)
    }

    /// A standard normal value (Box-Muller).
    fn normal(&mut self) -> f64 {
        let u1 = self.unit_f64().max(1e-12);
        let u2 = self.unit_f64();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }
}

/// `n` symbols of `q` through `1 + (0.15 + 0.08 j) z^-1 + (-0.04 + 0.03 j) z^-2`, turned by
/// `turn` rad, plus Gaussian noise for `ebn0_db` at the power of the channel output; also returns
/// the symbols.
fn channel_signal(
    rng: &mut Rng,
    n: usize,
    q: Qam,
    turn: f32,
    ebn0_db: f64,
) -> (Vec<C32>, Vec<u32>) {
    let syms: Vec<u32> = (0..n).map(|_| rng.symbol(q)).collect();
    let s: Vec<C32> = syms.iter().map(|&k| q.point(k)).collect();
    let (h1, h2) = (C32::new(0.15, 0.08), C32::new(-0.04, 0.03));
    let rot = C32::new(turn.cos(), turn.sin());
    let clean: Vec<C32> = (0..n)
        .map(|k| {
            let mut v = s[k];
            if k >= 1 {
                v += s[k - 1] * h1;
            }
            if k >= 2 {
                v += s[k - 2] * h2;
            }
            v * rot
        })
        .collect();
    let power = clean.iter().map(|v| f64::from(v.norm_sqr())).sum::<f64>() / n as f64;
    let n0 = power / (q.bits_per_symbol() as f64 * 10f64.powf(ebn0_db / 10.0));
    let sigma = (n0 / 2.0).sqrt();
    let x = clean
        .iter()
        .map(|v| {
            C32::new(
                v.re + (sigma * rng.normal()) as f32,
                v.im + (sigma * rng.normal()) as f32,
            )
        })
        .collect();
    (x, syms)
}

/// The decision errors of `y` against `syms` at the best quarter turn.
fn errors_best_rotation(q: Qam, y: &[C32], syms: &[u32]) -> usize {
    (0..4)
        .map(|r| {
            y.iter()
                .zip(syms)
                .filter(|&(v, &s)| {
                    let mut p = q.point(s);
                    for _ in 0..r {
                        p = C32::new(-p.im, p.re);
                    }
                    q.decide(*v) != p
                })
                .count()
        })
        .min()
        .unwrap()
}

fn state(
    c: EqConstellation,
    feedback: usize,
    mu_cma: f32,
    cma_symbols: u32,
    phase_bw: f64,
) -> State {
    let p = EqParams {
        constellation: c,
        feedback_taps: feedback,
        mu_cma,
        mu_dd: 0.005,
        cma_symbols,
        phase_bw,
    };
    let len = 11 + feedback;
    EqualizeState::new(vec![C32::default(); len], vec![C32::default(); len], p).unwrap()
}

fn bits(x: &[C32]) -> Vec<(u32, u32)> {
    x.iter().map(|v| (v.re.to_bits(), v.im.to_bits())).collect()
}

#[test]
fn the_dense_grids_start_with_the_multi_modulus_cost() {
    for c in [
        EqConstellation::Bpsk,
        EqConstellation::Qpsk,
        EqConstellation::Psk8,
        EqConstellation::Qam16,
        EqConstellation::Qam64,
    ] {
        assert_eq!(c.blind(), EqBlind::Cma, "{c:?}");
    }
    for (_, c) in DENSE {
        assert_eq!(c.blind(), EqBlind::Mma, "{c:?}");
    }
    // R = E[a^4] / E[a^2] of one axis, from the levels of the grid.
    for q in [Qam::Qpsk, Qam::Qam16, Qam::Qam64, Qam::Qam256, Qam::Qam1024] {
        let levels: Vec<f64> = q.levels().iter().map(|&v| f64::from(v)).collect();
        let a2 = levels.iter().map(|v| v * v).sum::<f64>();
        let a4 = levels.iter().map(|v| v * v * v * v).sum::<f64>();
        assert!((q.mma_r2() - a4 / a2).abs() < 1e-6, "{q:?}");
    }
    assert_eq!(Qam::Qam256.mma_r2(), 761.0 / 850.0);
    assert_eq!(Qam::Qam1024.mma_r2(), 613.0 / 682.0);
}

#[test]
fn the_multi_modulus_start_turns_the_output_onto_the_axes() {
    // Without the phase tracker: the multi-modulus start of 1024-QAM takes out the turn of
    // 0.3 rad (modulo a quarter turn) and the decision mode settles on the grid; the
    // constant-modulus start is blind to the phase, so 64-QAM through the same channel keeps
    // the turn and its decisions fail.
    let n = 30_000;
    let mut rng = Rng::new(1);
    let (x, syms) = channel_signal(&mut rng, n, Qam::Qam1024, 0.3, 32.0);
    let mut st = state(EqConstellation::Qam1024, 0, 0.01, 8000, 0.0);
    let mut y = vec![C32::default(); n];
    Fallback.equalize(&mut st, &x, &mut y).unwrap();
    assert_eq!(st.mode(), EqMode::Decision);
    let tail = 10_000;
    let errors = errors_best_rotation(Qam::Qam1024, &y[n - tail..], &syms[n - tail - 5..n - 5]);
    assert!(errors <= 5, "1024-QAM: {errors} errors, mse {}", st.mse());
    // The squared half level spacing of 1024-QAM is 1 / 682.
    assert!(st.mse() < 0.3 / 682.0, "mse {}", st.mse());

    let mut rng = Rng::new(2);
    let (x, syms) = channel_signal(&mut rng, n, Qam::Qam64, 0.3, 30.0);
    let mut st = state(EqConstellation::Qam64, 0, 0.01, 8000, 0.0);
    Fallback.equalize(&mut st, &x, &mut y).unwrap();
    let errors = errors_best_rotation(Qam::Qam64, &y[n - tail..], &syms[n - tail - 5..n - 5]);
    assert!(errors > tail / 10, "64-QAM with CMA: {errors} errors");
}

#[test]
fn the_split_at_the_switch_keeps_the_bits_for_every_cut() {
    for (stream, (q, c)) in DENSE.into_iter().enumerate() {
        for feedback in [0usize, 3] {
            let mut rng = Rng::new(10 + stream as u64);
            let (x, _) = channel_signal(&mut rng, 6000, q, 0.4, 30.0);
            let mut whole = state(c, feedback, 0.01, 2500, 0.005);
            let mut want = vec![C32::default(); x.len()];
            Fallback.equalize(&mut whole, &x, &mut want).unwrap();
            assert_eq!(whole.mode(), EqMode::Decision);
            for plan in 0..20u64 {
                let mut cut = Rng::new(100 + 20 * stream as u64 + plan);
                let mut st = state(c, feedback, 0.01, 2500, 0.005);
                let mut got = vec![C32::default(); x.len()];
                let mut at = 0;
                // Cuts of 1 to 700 symbols, one of them exactly at the switch.
                while at < x.len() {
                    let mut len = 1 + (f32::midpoint(cut.next(), 1.0) * 700.0) as usize;
                    if at < 2500 && at + len > 2500 && plan % 2 == 0 {
                        len = 2500 - at;
                    }
                    let end = (at + len).min(x.len());
                    Fallback
                        .equalize(&mut st, &x[at..end], &mut got[at..end])
                        .unwrap();
                    at = end;
                }
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "{q:?} feedback {feedback} plan {plan}"
                );
                assert_eq!(bits(st.weights()), bits(whole.weights()));
                assert_eq!(bits(st.history()), bits(whole.history()));
                assert_eq!(st.mse().to_bits(), whole.mse().to_bits());
                assert_eq!(
                    (st.phase(), st.frequency().to_bits()),
                    (whole.phase(), whole.frequency().to_bits())
                );
            }
            // `reset` starts the blind symbols again: the same output as a new state.
            whole.reset();
            let mut again = vec![C32::default(); x.len()];
            Fallback.equalize(&mut whole, &x, &mut again).unwrap();
            assert_eq!(
                bits(&again),
                bits(&want),
                "{q:?} feedback {feedback} after reset"
            );
        }
    }
}

#[test]
fn the_multi_modulus_start_stays_finite_on_extreme_input() {
    for (_, c) in DENSE {
        let mut st = state(c, 2, 1.0, u32::MAX, 0.2);
        let mut y = vec![C32::default(); 64];
        for v in [
            C32::new(0.0, 0.0),
            C32::new(1.0, -1.0),
            C32::new(1e6, -1e6),
            C32::new(f32::MAX, f32::MAX),
            C32::new(f32::NAN, 0.5),
            C32::new(f32::INFINITY, f32::NEG_INFINITY),
        ] {
            Fallback.equalize(&mut st, &[v; 64], &mut y).unwrap();
            assert_eq!(st.mode(), EqMode::Cma, "{c:?}");
            assert!(
                y.iter().all(|z| z.re.is_finite() && z.im.is_finite()),
                "{c:?} {v:?}"
            );
            assert!(
                st.weights()
                    .iter()
                    .all(|w| w.re.is_finite() && w.im.is_finite())
            );
            assert!(st.mse().is_finite());
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
    fn the_multi_modulus_start_equals_the_reference() {
        for (stream, (q, c)) in DENSE.into_iter().enumerate() {
            for (feedback, cma_symbols, mu, bw) in [
                (0usize, u32::MAX, 0.01f32, 0.005f64),
                (0, 3000, 0.05, 0.0),
                (3, u32::MAX, 0.01, 0.02),
                (5, 2000, 0.02, 0.005),
            ] {
                let mut rng = Rng::new(40 + stream as u64);
                let (mut x, _) = channel_signal(&mut rng, 5000, q, 0.7, 28.0);
                for (j, s) in SPECIALS.iter().enumerate() {
                    x[900 + 5 * j] = C32::new(*s, 0.2);
                    x[1300 + 5 * j] = C32::new(-0.4, *s);
                }
                let (mut a, mut b) = (
                    state(c, feedback, mu, cma_symbols, bw),
                    state(c, feedback, mu, cma_symbols, bw),
                );
                let mut ya = vec![C32::default(); x.len()];
                let mut yb = vec![C32::default(); x.len()];
                Fallback.equalize(&mut a, &x, &mut ya).unwrap();
                Reference.equalize(&mut b, &x, &mut yb).unwrap();
                let tag = format!("{q:?} feedback {feedback} cma_symbols {cma_symbols}");
                assert!(same(&ya, &yb), "{tag}");
                assert!(same(a.weights(), b.weights()), "{tag}");
                assert_eq!(a.mse().to_bits(), b.mse().to_bits(), "{tag}");
                assert_eq!((a.mode(), a.phase()), (b.mode(), b.phase()), "{tag}");
            }
        }
    }
}
