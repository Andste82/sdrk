//! Tests of the M44.2 carrier loop and equalizer of `sdrk-core` for the new square orders (§3):
//! `costas` and `equalize` with the constellations `qam256` and `qam1024` (the grid decision of
//! §1, the decision-directed loop `arg(y conj d)`, the lock metric of D1400 with `g = 1.5 E = M -
//! 1`, the CMA radius of §1), and `qpsk` as the 4-QAM grid (D1510). Test 6 of M44.2: the lock
//! detector stays off on uniform (seed 832) and Gaussian (seed 833) noise for `qpsk`, `qam256`
//! and `qam1024` and turns on at the Eb/N0 of the BER tests (4-QAM 8 and 10 dB, 256-QAM 22 and
//! 24 dB, 1024-QAM 27 and 29 dB; signals of seed 832, streams 100 and up). Also: lock and
//! frequency tracking, block-size invariance, `reset`, silence, full scale and non-finite input
//! of the new loops, the convergence of the equalizer on 256-QAM and 1024-QAM through a channel
//! with the step sizes and CMA lengths that the measurement of D1511 found to converge, and with
//! the feature `reference` `Fallback` against `Reference` (seed 839 for these signals).

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::many_single_char_names
)]

use sdrk_core::kernels::constellation::{QPSK_POINTS, Qam};
use sdrk_core::kernels::costas::{Costas, CostasConstellation, CostasState};
use sdrk_core::kernels::equalize::{EqConstellation, EqMode, EqParams, Equalize, EqualizeState};
use sdrk_core::{C32, Fallback};

#[allow(
    clippy::approx_constant,
    reason = "the damping of the block defaults, written as in the spec"
)]
const DAMP: f64 = 0.7071;
/// The seed of the uniform noise of test 6 and of the signals at the BER SNRs.
const SEED_UNIFORM: u64 = 832;
/// The seed of the Gaussian noise of test 6.
const SEED_GAUSS: u64 = 833;
/// The seed of the other signals of this file.
const SEED: u64 = 839;
/// The level above which the lock detector turns on, and the one below which it turns off.
const LOCK_ON_LEVEL: f32 = 0.5;
const LOCK_OFF_LEVEL: f32 = 0.3;

/// The new QAM orders of the loops.
const QAMS: [(Qam, CostasConstellation, EqConstellation); 2] = [
    (
        Qam::Qam256,
        CostasConstellation::Qam256,
        EqConstellation::Qam256,
    ),
    (
        Qam::Qam1024,
        CostasConstellation::Qam1024,
        EqConstellation::Qam1024,
    ),
];

/// Deterministic generator (xorshift64*), uniform in [-1, 1).
struct Rng(u64);

impl Rng {
    fn new(seed: u64, stream: u64) -> Self {
        Self(seed * 1000 + stream)
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

/// Random points of `q` turned by `phase0 + 2 pi offset k` plus Gaussian noise of `sigma` per
/// component; also returns the symbols.
fn qam_signal(
    rng: &mut Rng,
    n: usize,
    q: Qam,
    phase0: f64,
    offset: f64,
    sigma: f64,
) -> (Vec<C32>, Vec<u32>) {
    let mut syms = Vec::with_capacity(n);
    let x = (0..n)
        .map(|k| {
            let s = rng.symbol(q);
            syms.push(s);
            let p = q.point(s);
            let turns = (phase0 / core::f64::consts::TAU + offset * k as f64).rem_euclid(1.0);
            let (sn, c) = (core::f64::consts::TAU * turns).sin_cos();
            let (re, im) = (f64::from(p.re), f64::from(p.im));
            C32::new(
                (re * c - im * sn + sigma * rng.normal()) as f32,
                (re * sn + im * c + sigma * rng.normal()) as f32,
            )
        })
        .collect();
    (x, syms)
}

/// The noise `sigma` per component of a unit-power signal of `q` at `ebn0_db`.
fn sigma_of(q: Qam, ebn0_db: f64) -> f64 {
    let n0 = 1.0 / (q.bits_per_symbol() as f64 * 10f64.powf(ebn0_db / 10.0));
    (n0 / 2.0).sqrt()
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

/// Feeds `x` symbol by symbol from a fresh loop of `loop_bw`; returns the levels after every
/// symbol and the count of symbols with the flag on.
fn levels_on(c: CostasConstellation, loop_bw: f64, x: &[C32]) -> (Vec<f32>, usize) {
    let mut st = CostasState::new(c, loop_bw, DAMP, 0.05).unwrap();
    let mut levels = Vec::with_capacity(x.len());
    let mut on = 0usize;
    let mut y = [C32::default(); 1];
    for v in x {
        Fallback
            .costas(&mut st, core::slice::from_ref(v), &mut y)
            .unwrap();
        on += usize::from(st.locked());
        levels.push(st.lock_level());
    }
    (levels, on)
}

/// Mean and maximum of the levels after the first 2000 symbols.
fn tail_stats(levels: &[f32]) -> (f64, f32) {
    let tail = &levels[2000..];
    let mean = tail.iter().map(|&l| f64::from(l)).sum::<f64>() / tail.len() as f64;
    let max = tail.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (mean, max)
}

// --------------------------------------------------------------------------------------------
// The constellations
// --------------------------------------------------------------------------------------------

#[test]
fn the_new_orders_have_their_grid_and_radius() {
    for (q, c, e) in QAMS {
        assert_eq!((c.qam(), c.psk_order()), (Some(q), None));
        let st = CostasState::new(c, 0.005, DAMP, 0.05).unwrap();
        assert_eq!(st.constellation(), c);
        let mut rng = Rng::new(SEED, 1);
        for _ in 0..5000 {
            let y = C32::new(1.6 * rng.next(), 1.6 * rng.next());
            assert_eq!(e.decide(y), q.decide(y), "{q:?} {y}");
        }
        for &p in q.point_table() {
            assert_eq!(e.decide(p), p);
        }
        let l = q.levels();
        let d = e.decide(C32::new(f32::NAN, f32::INFINITY));
        assert_eq!(d, C32::new(l[0], l[l.len() - 1]));
    }
    // R2 of §1, the exact fractions rounded once to f32, and to four digits.
    assert_eq!(EqConstellation::Qam256.cma_r2(), (593.0f64 / 425.0) as f32);
    assert_eq!(EqConstellation::Qam1024.cma_r2(), (477.0f64 / 341.0) as f32);
    assert!((f64::from(EqConstellation::Qam256.cma_r2()) - 1.3953).abs() < 5e-5);
    assert!((f64::from(EqConstellation::Qam1024.cma_r2()) - 1.3988).abs() < 5e-5);
}

/// D1510: `qpsk` is the 4-QAM grid of `costas` and `equalize`. The points of `equalize` `qpsk`
/// are the 4-QAM points bit for bit, its decision is the 4-QAM grid decision of §1 on every
/// value except the negative ones above about `-4.2e-8` (`v (sqrt 2 / 2) + 1` rounds to 1 in
/// `f32`, so the rounding formula takes the positive level and the sign rule the negative one)
/// and NaN (the sign rule counts NaN as not negative, the
/// grid takes level 0), and its CMA radius is the `R2` = 1 of §1; `costas` `qpsk` keeps the PSK
/// detector and metric.
#[test]
fn qpsk_is_the_4qam_grid() {
    let q = Qam::Qpsk;
    for &p in &QPSK_POINTS {
        assert_eq!(EqConstellation::Qpsk.decide(p), p);
        assert_eq!(q.decide(p), p);
    }
    assert_eq!(EqConstellation::Qpsk.cma_r2(), q.cma_r2() as f32);
    let mut rng = Rng::new(SEED, 2);
    let mut values: Vec<f32> = (0..20_000).map(|_| 3.0 * rng.next()).collect();
    values.extend([
        0.0,
        -0.0,
        1e-7,
        -1e-7,
        f32::MIN_POSITIVE,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        -f32::MAX,
    ]);
    for (i, &a) in values.iter().enumerate() {
        let b = values[(i * 7 + 3) % values.len()];
        let y = C32::new(a, b);
        assert_eq!(EqConstellation::Qpsk.decide(y), q.decide(y), "{y}");
    }
    // The two documented exceptions.
    let h = core::f32::consts::FRAC_1_SQRT_2;
    let tiny = C32::new(-1e-9, f32::NAN);
    assert_eq!(EqConstellation::Qpsk.decide(tiny), C32::new(-h, h));
    assert_eq!(q.decide(tiny), C32::new(h, -h));
    assert_eq!(
        (
            CostasConstellation::Qpsk.qam(),
            CostasConstellation::Qpsk.psk_order()
        ),
        (None, Some(4))
    );
}

// --------------------------------------------------------------------------------------------
// Test 6: the lock detector
// --------------------------------------------------------------------------------------------

/// Test 6 of M44.2 (D1512): from a fresh loop over 20000 symbols, at the block default
/// `loop_bw` 0.02 and at the `carrier_bw` 0.0025 of `qam_demod`, the lock flag never turns on for
/// `qpsk`, `qam256` and `qam1024` on uniform noise in `[-1, 1)` (seed 832) or on Gaussian noise of
/// unit power (seed 833); the thresholds 0.5 and 0.3 of M40 hold for every order. The measured
/// levels are guarded and printed.
#[test]
#[allow(
    clippy::print_stdout,
    reason = "the measured lock levels of the new orders (D1512)"
)]
fn the_lock_stays_off_on_noise_for_every_new_order() {
    let mut rng = Rng::new(SEED_UNIFORM, 0);
    let uniform: Vec<C32> = (0..20_000)
        .map(|_| C32::new(rng.next(), rng.next()))
        .collect();
    let mut rng = Rng::new(SEED_GAUSS, 0);
    let s = core::f64::consts::FRAC_1_SQRT_2;
    let gauss: Vec<C32> = (0..20_000)
        .map(|_| C32::new((s * rng.normal()) as f32, (s * rng.normal()) as f32))
        .collect();
    let all = [
        CostasConstellation::Qpsk,
        CostasConstellation::Qam256,
        CostasConstellation::Qam1024,
    ];
    for c in all {
        for loop_bw in [0.02, 0.0025] {
            let (levels, on) = levels_on(c, loop_bw, &uniform);
            let (mean, max) = tail_stats(&levels);
            let (glevels, gon) = levels_on(c, loop_bw, &gauss);
            let (gmean, gmax) = tail_stats(&glevels);
            println!(
                "{c:?} loop_bw {loop_bw}: uniform noise mean {mean:.4}, max {max:.4}, locked in \
                 {on}; Gaussian noise mean {gmean:.4}, max {gmax:.4}, locked in {gon} of 20000"
            );
            assert_eq!(on, 0, "{c:?} {loop_bw} locked on uniform noise");
            assert_eq!(gon, 0, "{c:?} {loop_bw} locked on Gaussian noise");
            if c == CostasConstellation::Qpsk {
                // The PSK metric of M40: the uniform square is not round.
                assert!(mean.abs() < 0.2 && max < 0.45, "{c:?}: {mean} {max}");
                assert!(gmean.abs() < 0.05 && gmax < 0.3, "{c:?}: {gmean} {gmax}");
            } else {
                // The decision error: 0 within a cell, below 0 outside the grid.
                assert!(mean.abs() < 0.05 && max < 0.3, "{c:?}: {mean} {max}");
                assert!(gmean < -0.05 && gmax < 0.3, "{c:?}: {gmean} {gmax}");
            }
        }
    }
}

/// Test 6 of M44.2 (D1512): at the Eb/N0 points of the BER tests (Gaussian noise; 4-QAM 8 and
/// 10 dB, 256-QAM 22 and 24 dB, 1024-QAM 27 and 29 dB) the loop at `loop_bw` 0.02 locks within
/// 1000 symbols from a start phase of 0.1 rad, and from 0.05 rad with an offset of 0.0002
/// cycles per symbol, and stays locked (the level never falls below 0.3 after it first passed
/// 0.5); for the QAM the smoothed level is near the expected `1 - 1.5 E N0` of the metric.
#[test]
#[allow(
    clippy::print_stdout,
    reason = "the measured lock levels of the new orders (D1512)"
)]
fn the_lock_turns_on_at_the_snrs_of_the_ber_tests() {
    let cases = [
        (Qam::Qpsk, CostasConstellation::Qpsk, [8.0f64, 10.0]),
        (Qam::Qam256, CostasConstellation::Qam256, [22.0, 24.0]),
        (Qam::Qam1024, CostasConstellation::Qam1024, [27.0, 29.0]),
    ];
    let mut stream = 100;
    for (q, c, snrs) in cases {
        for ebn0_db in snrs {
            for (phase0, offset) in [(0.1, 0.0), (0.05, 0.0002)] {
                stream += 1;
                let mut rng = Rng::new(SEED_UNIFORM, stream);
                let (x, _) = qam_signal(&mut rng, 8000, q, phase0, offset, sigma_of(q, ebn0_db));
                let (levels, _) = levels_on(c, 0.02, &x);
                let first = levels.iter().position(|&l| l > LOCK_ON_LEVEL).unwrap();
                let (mean, _) = tail_stats(&levels);
                let n0 = 1.0 / (q.bits_per_symbol() as f64 * 10f64.powf(ebn0_db / 10.0));
                let want = 1.0 - 1.5 * q.energy() * n0;
                println!(
                    "{q:?} at {ebn0_db} dB, phase {phase0}, offset {offset}: first locked at \
                     {first}, mean level {mean:.4}, decision-error metric {want:.4}"
                );
                assert!(first < 1000, "{q:?} {ebn0_db}: first locked at {first}");
                assert!(
                    levels[first..].iter().all(|&l| l > LOCK_OFF_LEVEL),
                    "{q:?} {ebn0_db} lost lock"
                );
                if q == Qam::Qpsk {
                    assert!(mean > 0.65, "{q:?} {ebn0_db}: {mean}");
                } else {
                    assert!((mean - want).abs() < 0.05, "{q:?}: {mean} against {want}");
                }
            }
        }
    }
}

// --------------------------------------------------------------------------------------------
// costas
// --------------------------------------------------------------------------------------------

#[test]
fn the_new_loops_lock_and_follow_an_offset() {
    for (stream, (q, c, _)) in QAMS.into_iter().enumerate() {
        // The decision-directed loop pulls in from small offsets only, the less the denser the
        // grid (D1512); the acquisition of `qam_demod` brings larger ones within its range.
        let offsets = if q == Qam::Qam256 {
            [(0.1, 0.0), (-0.08, 0.0005), (0.05, -0.001)]
        } else {
            [(0.1, 0.0), (-0.08, 0.0002), (0.05, -0.0003)]
        };
        for (k, (phase0, offset)) in offsets.into_iter().enumerate() {
            let mut rng = Rng::new(SEED, 10 + 3 * stream as u64 + k as u64);
            // Eb/N0 30 dB (256-QAM) and 34 dB (1024-QAM): no decision error in 4000 symbols.
            let ebn0 = if q == Qam::Qam256 { 30.0 } else { 34.0 };
            let (x, syms) = qam_signal(&mut rng, 8000, q, phase0, offset, sigma_of(q, ebn0));
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

#[test]
fn the_new_loops_do_not_depend_on_the_block_size_and_reset() {
    for (stream, (q, c, _)) in QAMS.into_iter().enumerate() {
        let mut rng = Rng::new(SEED, 20 + stream as u64);
        let (x, _) = qam_signal(&mut rng, 1500, q, 1.0, 0.003, 0.02);
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
        whole.reset();
        assert_eq!(whole, CostasState::new(c, 0.03, DAMP, 0.05).unwrap());
        assert_eq!(bits(&run(&mut whole, &x)), bits(&want));
    }
}

#[test]
fn the_new_loops_handle_silence_full_scale_and_non_finite_input() {
    for (stream, (q, c, _)) in QAMS.into_iter().enumerate() {
        // Silence: atan2f(0, 0) = 0 and y = 0 has no lock metric (D1400), nothing moves.
        let mut st = CostasState::new(c, 0.02, DAMP, 0.05).unwrap();
        st.set_phase(0x1234_5678);
        let y = run(&mut st, &[C32::new(0.0, 0.0); 500]);
        assert!(y.iter().all(|v| v.re == 0.0 && v.im == 0.0));
        assert_eq!(st.phase(), 0x1234_5678);
        assert_eq!(
            (st.frequency(), st.lock_level(), st.locked()),
            (0.0, 0.0, false)
        );
        // Full scale and beyond: the integrator stays within its limit, the level within [-1, 1].
        let mut rng = Rng::new(SEED, 30 + stream as u64);
        let (x, _) = qam_signal(&mut rng, 600, q, 0.8, 0.003, 0.0);
        for amp in [1.0f32, 1e18, 1e30, f32::MAX] {
            let x: Vec<C32> = x.iter().map(|v| C32::new(v.re * amp, v.im * amp)).collect();
            let mut st = CostasState::new(c, 0.02, DAMP, 0.05).unwrap();
            run(&mut st, &x);
            assert!(st.frequency().abs() <= 0.05 + 1e-9, "{q:?} {amp}");
            assert!(st.lock_level().is_finite() && st.lock_level().abs() <= 1.0 + 1e-6);
        }
        // Non-finite input stays in its own symbols.
        let (mut x, _) = qam_signal(&mut rng, 3000, q, 0.05, 0.0, 0.002);
        for (i, v) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
            .into_iter()
            .enumerate()
        {
            x[2000 + i] = C32::new(v, 0.5);
            x[2010 + i] = C32::new(0.5, v);
        }
        let mut st = CostasState::new(c, 0.02, DAMP, 0.05).unwrap();
        let y = run(&mut st, &x);
        let finite = |v: &C32| v.re.is_finite() && v.im.is_finite();
        assert!(y[..2000].iter().all(finite));
        assert!(!y[2000].re.is_finite() && !y[2010].im.is_finite());
        assert!(y[2013..].iter().all(finite));
        assert!(st.locked() && st.frequency().is_finite(), "{q:?}");
    }
}

// --------------------------------------------------------------------------------------------
// equalize
// --------------------------------------------------------------------------------------------

/// D1511: 256-QAM and 1024-QAM through the symbol-spaced channel `1 + (0.2 + 0.12 j) z^-1 +
/// (-0.05 + 0.06 j) z^-2` turned by 0.3 rad at the upper Eb/N0 of the BER tests (24 and 29 dB),
/// 11 taps, the tracker at 0.005 and `mu_dd` 0.005 (those of `qam_demod`): CMA with the radius
/// of the order converges with `mu_cma` 0.02 over 4000 symbols (256-QAM) and `mu_cma` 0.005 over
/// 16000 symbols (1024-QAM), the step sizes and lengths that converged in every run of the
/// measurement; after convergence the decisions are the transmitted symbols up to the noise
/// (at most 5 errors in the last 20000 symbols), and the blocks cut the input in any way
/// without changing a bit. Since M44.4 these two orders start with the multi-modulus algorithm
/// instead of CMA (D2507), so the test runs MMA with the step sizes and lengths of D1511; it
/// still converges in both (the blind start of `qam_demod` takes the values of D2508 and D2509).
#[test]
fn the_equalizer_converges_on_256_and_1024_qam_through_a_channel() {
    for (q, e, mu_cma, cma_symbols, ebn0) in [
        (Qam::Qam256, EqConstellation::Qam256, 0.02f32, 4000u32, 24.0),
        (Qam::Qam1024, EqConstellation::Qam1024, 0.005, 16_000, 29.0),
    ] {
        let n = 60_000;
        let mut rng = Rng::new(SEED, 40 + q.bits_per_axis() as u64);
        let syms: Vec<u32> = (0..n).map(|_| rng.symbol(q)).collect();
        let s: Vec<C32> = syms.iter().map(|&k| q.point(k)).collect();
        let (h1, h2) = (C32::new(0.2, 0.12), C32::new(-0.05, 0.06));
        let turn = C32::new(0.3f32.cos(), 0.3f32.sin());
        let clean: Vec<C32> = (0..n)
            .map(|k| {
                let mut v = s[k];
                if k >= 1 {
                    v += s[k - 1] * h1;
                }
                if k >= 2 {
                    v += s[k - 2] * h2;
                }
                v * turn
            })
            .collect();
        let power = clean.iter().map(|v| f64::from(v.norm_sqr())).sum::<f64>() / n as f64;
        let sigma = sigma_of(q, ebn0) * power.sqrt();
        let x: Vec<C32> = clean
            .iter()
            .map(|v| {
                C32::new(
                    v.re + (sigma * rng.normal()) as f32,
                    v.im + (sigma * rng.normal()) as f32,
                )
            })
            .collect();
        let p = EqParams {
            constellation: e,
            mu_cma,
            mu_dd: 0.005,
            cma_symbols,
            phase_bw: 0.005,
            ..EqParams::default()
        };
        let mut st =
            EqualizeState::new(vec![C32::default(); 11], vec![C32::default(); 11], p).unwrap();
        let mut y = vec![C32::default(); n];
        Fallback.equalize(&mut st, &x, &mut y).unwrap();
        assert_eq!(st.mode(), EqMode::Decision);
        let delay = 5;
        let errors =
            errors_best_rotation(q, &y[n - 20_000..], &syms[n - 20_000 - delay..n - delay]);
        assert!(errors <= 5, "{q:?}: {errors} errors, mse {}", st.mse());
        for size in [1usize, 7, 333, 4096] {
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
        for (stream, (q, c, _)) in QAMS.into_iter().enumerate() {
            for (k, offset) in [0.0, 0.002, -0.01].into_iter().enumerate() {
                let mut rng = Rng::new(SEED, 50 + 3 * stream as u64 + k as u64);
                let (mut x, _) = qam_signal(&mut rng, 3000, q, 1.0, offset, 0.02);
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
        for (stream, (q, _, e)) in QAMS.into_iter().enumerate() {
            for feedback in [0usize, 3] {
                let mut rng = Rng::new(SEED, 60 + stream as u64);
                let (mut x, _) = qam_signal(&mut rng, 3000, q, 0.3, 0.0, 0.01);
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
