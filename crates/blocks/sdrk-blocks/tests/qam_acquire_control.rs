//! The building blocks of the receiver of M44.1: `qam_acquire` (test 3: the estimates bit for bit
//! those of the acquisition of `qam_demod` before M44.1 on the same windows, `correct` true, seed
//! 842, `search` and `verify`; test 11: 20 cut plans of seed 851), the control interface of
//! `equalize`, `costas` and `symbol_sync` (test 4: each method on the state it forwards to, no
//! change of the output without calls, seed 853) and the states of `LockWatch` (the transitions of
//! §5, the part of test 6 that step 1 has).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

mod common;

use common::{check_block, rate, run_block, same_bits};
use sdrk::{C32, Sdrk};
use sdrk_blocks::{
    AcquireWindow, Block, CostasBlock, CostasConfig, CostasConstellation, EqConstellation,
    EqualizeBlock, EqualizeConfig, LockAction, LockCheck, LockPolicy, LockState, LockTransition,
    LockWatch, QAM_ACQUISITION_SKIP, QAM_ACQUISITION_SYMBOLS, QamAcquireBlock, QamAcquireConfig,
    SymbolSyncBlock, SymbolSyncConfig,
};
use sdrk_core::kernels::constellation::Qam;
use sdrk_core::kernels::costas::{Costas, CostasState};
use sdrk_core::kernels::equalize::{EqParams, Equalize, EqualizeState};
use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, Scaling};
use sdrk_core::kernels::symbol_sync::{SymbolSync, SymbolSyncState, Ted};
use sdrk_core::num::flush_denormal;
use sdrk_testkit::{Pcg32, VecStorage};

/// The seed of test 3, of test 4 and of the cut plans of test 11.
const SEED: u64 = 842;
const SEED_CONTROL: u64 = 853;
const SEED_CHUNKS: u64 = 851;

// ---- test 3: qam_acquire ------------------------------------------------------------------

/// `n` random 16-QAM (or 64-QAM) symbols turned by `f` cycles per symbol from the phase `p0`
/// (turns), with uniform noise of `+-noise` per component; also the symbols before the turn.
fn qam_signal(
    qam: Qam,
    n: usize,
    f: f64,
    p0: f64,
    noise: f32,
    stream: u64,
) -> (Vec<C32>, Vec<C32>) {
    let mut rng = Pcg32::new(SEED, stream);
    let s: Vec<C32> = (0..n)
        .map(|_| qam.point(rng.below(qam.points() as u32)))
        .collect();
    let x = s
        .iter()
        .enumerate()
        .map(|(k, v)| {
            let t = core::f64::consts::TAU * (p0 + f * k as f64);
            let (sn, cs) = (t.sin(), t.cos());
            let (re, im) = (f64::from(v.re), f64::from(v.im));
            C32::new(
                (re * cs - im * sn) as f32 + noise * (2.0 * rng.next_f64() as f32 - 1.0),
                (re * sn + im * cs) as f32 + noise * (2.0 * rng.next_f64() as f32 - 1.0),
            )
        })
        .collect();
    (x, s)
}

/// Gaussian noise of unit power.
fn noise(n: usize, stream: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(SEED, stream);
    let mut g = || (rng.normal() * core::f64::consts::FRAC_1_SQRT_2) as f32;
    (0..n).map(|_| C32::new(g(), g())).collect()
}

fn acquire(correct: bool) -> QamAcquireBlock<VecStorage> {
    QamAcquireBlock::new(&QamAcquireConfig { correct }, rate(1000), &mut VecStorage).unwrap()
}

/// The acquisition of `qam_demod` before M44.1 (`blocks/qam_demod.rs` at ab8b729, `Acquisition`:
/// `collect` and `estimate_now`, written out here): the model of test 3.
struct OldWindow {
    twiddles: Vec<C32>,
    buf: Vec<C32>,
    power: f32,
    seen: usize,
}

impl OldWindow {
    fn new() -> Self {
        let mut twiddles = vec![C32::default(); 512];
        FftPlan::new(1024, &mut twiddles[..]).unwrap();
        Self {
            twiddles,
            buf: vec![C32::default(); 1024],
            power: 0.0,
            seen: 0,
        }
    }

    /// The symbols of one window from `seen` (0, or 256 after a window without a clear peak) to
    /// 768, then the estimate.
    fn window(&mut self, x: &[C32]) -> Option<f64> {
        for &v in x {
            let p = v.re * v.re + v.im * v.im;
            if p.is_finite() {
                self.power = flush_denormal(self.power + (1.0 / 64.0) * (p - self.power));
            }
            if let Some(i) = self.seen.checked_sub(256).filter(|&i| i < 512) {
                let a = C32::new(v.re * v.re - v.im * v.im, 2.0 * v.re * v.im);
                let pp = self.power * self.power;
                let q = C32::new((a.re * a.re - a.im * a.im) / pp, (2.0 * a.re * a.im) / pp);
                self.buf[i] = if self.power > 0.0 && q.re.is_finite() && q.im.is_finite() {
                    q
                } else {
                    C32::new(0.0, 0.0)
                };
                self.buf[i + 512] = C32::new(0.0, 0.0);
            }
            self.seen += 1;
        }
        assert_eq!(self.seen, 768);
        let energy: f32 = self.buf[..512]
            .iter()
            .map(|v| v.re * v.re + v.im * v.im)
            .sum();
        let plan = FftPlan::from_twiddles(1024, &self.twiddles[..]).unwrap();
        <Sdrk as Fft<C32>>::fft(
            &Sdrk,
            &plan,
            &mut self.buf,
            Direction::Forward,
            Scaling::Default,
        )
        .unwrap();
        let power = |v: &C32| v.re * v.re + v.im * v.im;
        let mut bin = 0usize;
        let mut peak = -1.0f32;
        for (i, v) in self.buf.iter().enumerate() {
            let p = power(v);
            if p > peak {
                peak = p;
                bin = i;
            }
        }
        if !(energy.is_finite() && peak.is_finite() && energy > 0.0) || peak < 12.0 * energy {
            self.seen = 256;
            return None;
        }
        let mag = |i: usize| f64::from(power(&self.buf[i % 1024])).sqrt();
        let (below, top, above) = (mag(bin + 1023), mag(bin), mag(bin + 1));
        let den = below - 2.0 * top + above;
        let delta = if den < 0.0 {
            0.5 * (below - above) / den
        } else {
            0.0
        };
        let mut cycles = (bin as f64 + delta) / 1024.0;
        if cycles >= 0.5 {
            cycles -= 1.0;
        }
        Some(cycles / 4.0)
    }
}

/// Feeds `x` in pieces of `piece` symbols; returns every window that completed.
fn feed(b: &mut QamAcquireBlock<VecStorage>, x: &[C32], piece: usize) -> Vec<AcquireWindow> {
    let mut y = x.to_vec();
    let mut windows = Vec::new();
    for chunk in y.chunks_mut(piece) {
        if let Some(w) = b.process_in_place(chunk).unwrap() {
            windows.push(w);
        }
    }
    let bits = |v: &[C32]| -> Vec<(u32, u32)> {
        v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
    };
    assert_eq!(bits(x), bits(&y), "correct false keeps the input");
    windows
}

#[test]
fn qam_acquire_estimates_bit_for_bit_as_the_old_acquisition() {
    for (stream, (qam, f)) in [
        (Qam::Qam16, 0.001),
        (Qam::Qam64, -0.0005),
        (Qam::Qam16, 0.04),
        (Qam::Qam64, -0.1),
        (Qam::Qam16, 0.124),
    ]
    .into_iter()
    .enumerate()
    {
        let stream = stream as u64;
        // Two windows of noise (no clear peak), then the signal: the third window is clear.
        let mut x = noise(768 + 512, 10 + stream);
        x.extend(qam_signal(qam, 3000, f, 0.1, 0.02, 20 + stream).0);
        // (a call returns the last window that completed in it: at most one per piece here)
        for piece in [1usize, 7, 100, 512, 768] {
            let what = format!("{qam:?} {f} piece {piece}");
            let mut b = acquire(false);
            let mut old = OldWindow::new();
            let want = [
                old.window(&x[..768]),
                old.window(&x[768..1280]),
                old.window(&x[1280..1792]),
            ];
            assert_eq!(want[..2], [None, None], "{what}: the noise is not clear");
            let got = feed(&mut b, &x[..1792], piece);
            assert_eq!(got.len(), 3, "{what}");
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.estimate.map(f64::to_bits), w.map(f64::to_bits), "{what}");
                assert!(!g.verify, "{what}");
            }
            let e = want[2].unwrap();
            assert!((e - f).abs() < 6e-5, "{what}: estimate {e}");
            assert_eq!(b.estimate().map(f64::to_bits), Some(e.to_bits()), "{what}");
            assert_eq!(
                (b.window_clear(), b.active()),
                (Some(true), false),
                "{what}"
            );
            // Held: no window; then a verify window from the next symbol on (the power runs on
            // from the search, as in `qam_demod`, where a check follows after the estimate).
            assert!(feed(&mut b, &x[1792..2000], piece).is_empty(), "{what}");
            b.verify();
            assert_eq!(b.remaining(), 768, "{what}");
            old.seen = 0;
            let v = old.window(&x[2000..2768]);
            let got = feed(&mut b, &x[2000..2768], piece);
            assert_eq!(got.len(), 1, "{what}");
            assert!(got[0].verify, "{what}");
            assert_eq!(
                got[0].estimate.map(f64::to_bits),
                v.map(f64::to_bits),
                "{what}"
            );
            assert_eq!(
                b.estimate().map(f64::to_bits),
                Some(e.to_bits()),
                "{what}: kept"
            );
            // A new search drops the estimate and takes its window after the skip.
            b.search();
            assert_eq!((b.estimate(), b.remaining()), (None, 768), "{what}");
            old.seen = 0;
            let s = old.window(&x[2768..3536]);
            let got = feed(&mut b, &x[2768..3536], piece);
            assert_eq!(got.len(), 1, "{what}");
            assert_eq!(
                got[0].estimate.map(f64::to_bits),
                s.map(f64::to_bits),
                "{what}"
            );
            assert_eq!(
                b.estimate().map(f64::to_bits),
                s.map(f64::to_bits),
                "{what}"
            );
        }
    }
}

#[test]
fn qam_acquire_reports_windows_without_a_clear_peak() {
    let mut b = acquire(false);
    assert_eq!((b.window_clear(), b.remaining()), (None, 768));
    let x = noise(768 + 3 * 512, 30);
    let got = feed(&mut b, &x, 333);
    assert_eq!(got.len(), 4);
    assert!(got.iter().all(|w| w.estimate.is_none() && !w.verify));
    // the next window follows at once, without the skip
    assert_eq!(
        (b.window_clear(), b.estimate(), b.remaining()),
        (Some(false), None, 512)
    );
    b.reset();
    assert_eq!(
        (b.window_clear(), b.remaining(), b.active()),
        (None, 768, true)
    );
}

#[test]
fn qam_acquire_corrects_a_tone_of_a_thousandth_cycle_per_symbol() {
    for qam in [Qam::Qam16, Qam::Qam64] {
        let f = 0.001;
        let (x, s) = qam_signal(qam, 6000, f, 0.03, 0.0, 40);
        let mut b = acquire(true);
        let mut y = vec![C32::default(); x.len()];
        let p = b.process(&x, &mut y).unwrap();
        assert_eq!((p.consumed, p.produced), (6000, 6000));
        let e = b.estimate().unwrap();
        assert!((e - f).abs() < 5.6e-5, "{qam:?}: estimate {e}");
        // Until the window completes the output is the input; then each symbol is turned back
        // by the estimate: y / s turns at f - e from symbol 768 on.
        same_bits(&x[..768], &y[..768], "before the estimate");
        let arg = |k: usize| {
            let (a, v) = (y[k], s[k]);
            let re = f64::from(a.re) * f64::from(v.re) + f64::from(a.im) * f64::from(v.im);
            let im = f64::from(a.im) * f64::from(v.re) - f64::from(a.re) * f64::from(v.im);
            im.atan2(re) / core::f64::consts::TAU
        };
        let mut turn = 0.0;
        let mut prev = arg(768);
        for k in 769..6000 {
            let a = arg(k);
            turn += (a - prev) - (a - prev).round();
            prev = a;
        }
        let residual = turn / (6000 - 769) as f64;
        assert!(
            (residual - (f - e)).abs() < 1e-6,
            "{qam:?}: the output turns at {residual}, the error of the estimate is {}",
            f - e
        );
        assert!(residual.abs() < 5.6e-5, "{qam:?}");
    }
}

#[test]
fn qam_acquire_is_chunk_invariant() {
    for correct in [false, true] {
        let mut x = noise(900, 50);
        x.extend(qam_signal(Qam::Qam16, 4000, 0.002, 0.2, 0.02, 51).0);
        let want = run_block(&mut acquire(correct), &x, None);
        for seed in [SEED, SEED_CHUNKS] {
            check_block(
                &want,
                &x,
                seed,
                || acquire(correct),
                &format!("qam_acquire correct {correct}"),
            );
        }
    }
}

#[test]
fn qam_acquire_timing_and_parameters() {
    let b = acquire(true);
    assert_eq!(b.timing().ratio, sdrk::Ratio::ONE);
    assert_eq!(b.timing().delay, sdrk::Delay::ZERO);
    assert_eq!(b.min_input(), 1);
    assert_eq!(b.config(), QamAcquireConfig { correct: true });
    assert_eq!(
        b.param("correct").unwrap(),
        sdrk_blocks::ParamValue::Bool(true)
    );
    let mut b = b;
    assert!(
        b.set_param("correct", sdrk_blocks::ParamValue::Bool(false))
            .is_err()
    );
    assert_eq!(QAM_ACQUISITION_SKIP + QAM_ACQUISITION_SYMBOLS, 768);
    // the extreme inputs: NaN and infinities give no estimate and pass unchanged
    let mut x = vec![C32::new(f32::NAN, 0.0); 800];
    x[300] = C32::new(f32::INFINITY, f32::NEG_INFINITY);
    let mut b = acquire(false);
    let got = feed(&mut b, &x, 800);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].estimate, None);
}

// ---- test 4: the control interface --------------------------------------------------------

fn qam16_symbols(n: usize, stream: u64) -> Vec<C32> {
    qam_signal(Qam::Qam16, n, 0.0007, 0.05, 0.03, stream).0
}

fn eq_config() -> EqualizeConfig {
    EqualizeConfig {
        constellation: EqConstellation::Qam16,
        cma_symbols: Some(300),
        phase_bw: 0.005,
        ..EqualizeConfig::default()
    }
}

fn eq_state(cfg: &EqualizeConfig) -> EqualizeState<Vec<C32>, Vec<C32>> {
    EqualizeState::new(
        vec![C32::default(); cfg.taps],
        vec![C32::default(); cfg.taps],
        EqParams {
            constellation: cfg.constellation,
            feedback_taps: cfg.feedback_taps,
            mu_cma: cfg.mu_cma() as f32,
            mu_dd: cfg.mu_dd as f32,
            cma_symbols: cfg.cma_symbols(),
            phase_bw: cfg.phase_bw,
        },
    )
    .unwrap()
}

#[test]
fn equalize_control_calls_forward_to_the_state() {
    let cfg = eq_config();
    let x = qam16_symbols(3000, SEED_CONTROL);
    let mut block = EqualizeBlock::new(&cfg, rate(1000), &mut VecStorage).unwrap();
    let mut st = eq_state(&cfg);
    let mut yb = vec![C32::default(); 500];
    let mut ys = vec![C32::default(); 500];
    let mut step =
        |block: &mut EqualizeBlock<VecStorage>, st: &mut EqualizeState<_, _>, k: usize| {
            let x = &x[500 * k..500 * (k + 1)];
            block.process(x, &mut yb).unwrap();
            <Sdrk as Equalize>::equalize(&Sdrk, st, x, &mut ys).unwrap();
            same_bits(&ys, &yb, &format!("equalize piece {k}"));
            assert_eq!(block.state().weights(), st.weights());
            let energy: f32 = st.weights().iter().map(|w| w.re * w.re + w.im * w.im).sum();
            assert_eq!(block.weight_energy(), energy);
        };
    // no calls: the block is the kernel
    step(&mut block, &mut st, 0);
    block.set_frequency(0.0007);
    st.set_frequency(0.0007f32);
    step(&mut block, &mut st, 1);
    block.set_phase(0x1234_5678);
    st.set_phase(0x1234_5678);
    step(&mut block, &mut st, 2);
    block.set_phase_bw(0.0015).unwrap();
    st.set_phase_bw(0.0015).unwrap();
    assert_eq!(block.config().phase_bw, 0.0015);
    assert!(block.set_phase_bw(0.5).is_err());
    step(&mut block, &mut st, 3);
    block.restart();
    st.reset();
    assert_eq!((block.state().phase(), block.state().frequency()), (0, 0.0));
    assert_eq!(block.weight_energy(), 1.0);
    step(&mut block, &mut st, 4);
    // the parameters stay those of the configuration
    assert_eq!(
        block.config(),
        EqualizeConfig {
            phase_bw: 0.0015,
            ..cfg
        }
    );
    assert_eq!(
        block.param("mu_dd").unwrap(),
        sdrk_blocks::ParamValue::F64(cfg.mu_dd)
    );
    step(&mut block, &mut st, 5);
}

#[test]
fn costas_control_calls_forward_to_the_state() {
    let cfg = CostasConfig {
        constellation: CostasConstellation::Qam16,
        loop_bw: 0.0025,
        ..CostasConfig::default()
    };
    let x = qam16_symbols(3000, SEED_CONTROL + 1);
    let mut block = CostasBlock::new(&cfg, rate(1000), &mut VecStorage).unwrap();
    let mut st =
        CostasState::new(cfg.constellation, cfg.loop_bw, cfg.damping, cfg.max_offset).unwrap();
    let mut yb = vec![C32::default(); 750];
    let mut ys = vec![C32::default(); 750];
    let mut step = |block: &mut CostasBlock<VecStorage>, st: &mut CostasState, k: usize| {
        let x = &x[750 * k..750 * (k + 1)];
        block.process(x, &mut yb).unwrap();
        <Sdrk as Costas>::costas(&Sdrk, st, x, &mut ys).unwrap();
        same_bits(&ys, &yb, &format!("costas piece {k}"));
        assert_eq!(block.state(), &*st);
        assert_eq!(block.lock_level(), st.lock_level());
    };
    step(&mut block, &mut st, 0);
    block.set_frequency(0.0007);
    st.set_frequency(0.0007);
    assert_eq!(block.frequency(), f64::from(0.0007f32));
    step(&mut block, &mut st, 1);
    block.restart();
    st.reset();
    assert_eq!(block.state(), &st);
    step(&mut block, &mut st, 2);
    assert_eq!(block.config(), cfg);
    step(&mut block, &mut st, 3);
}

#[test]
fn symbol_sync_restart_forwards_to_the_state() {
    let cfg = SymbolSyncConfig {
        symbol_rate_hz: 1000.0,
        loop_bw: 0.005,
        ..SymbolSyncConfig::default()
    };
    let mut rng = Pcg32::new(SEED_CONTROL, 3);
    let x: Vec<C32> = (0..8000)
        .map(|n| {
            let s = (n / 4) as u32;
            let b = (s.wrapping_mul(2_654_435_761) >> 30) as f32;
            C32::new(b - 1.5 + 0.01 * rng.next_f64() as f32, 1.5 - b)
        })
        .collect();
    let mut block = SymbolSyncBlock::<C32, _>::new(&cfg, rate(4000), &mut VecStorage).unwrap();
    let mut st = SymbolSyncState::<C32>::new(
        4.0,
        Ted::Gardner,
        0.005,
        cfg.damping,
        cfg.max_deviation,
        1.0,
    )
    .unwrap();
    let mut yb = vec![C32::default(); 2000];
    let mut ys = vec![C32::default(); 2000];
    for k in 0..4 {
        if k == 2 {
            block.restart();
            st.reset();
            assert_eq!(block.state(), &st);
        }
        let x = &x[2000 * k..2000 * (k + 1)];
        let pb = block.process(x, &mut yb).unwrap();
        let ps = <Sdrk as SymbolSync<C32>>::symbol_sync(&Sdrk, &mut st, x, &mut ys).unwrap();
        assert_eq!((pb.consumed, pb.produced), (ps.consumed, ps.produced));
        same_bits(
            &ys[..ps.produced],
            &yb[..pb.produced],
            &format!("symbol_sync piece {k}"),
        );
        assert_eq!(block.state(), &st);
    }
    assert_eq!(block.config(), cfg);
    assert_eq!(block.min_input(), 9);
}

// ---- LockWatch: the transitions of §5 (part of test 6) ---------------------------------------

struct Policy;

impl LockPolicy for Policy {
    const CHECK: u16 = 4096;
    const TOLERANCE: f64 = 2e-4;
    const WEIGHT_CHECKS: u8 = 2;
    const GAP_WINDOWS: u8 = 6;
}

/// The inputs of a check: the lock, and a weight energy above the floor.
fn chk(locked: bool) -> impl FnOnce() -> LockCheck {
    move || LockCheck {
        locked,
        weight_energy: Some(1.0),
        weight_floor: 0.75,
    }
}

#[test]
fn lock_watch_takes_every_transition_of_step_one() {
    let mut w = LockWatch::<Policy>::new();
    assert_eq!(
        (w.state(), w.remaining(), w.acquired()),
        (LockState::Search, usize::MAX, None)
    );
    // Search: no check counts
    assert_eq!(w.advance(5000, chk(false)), None);
    assert_eq!(
        w.window(Some(0.001), 0.0),
        LockTransition::Acquire { frequency: 0.001 }
    );
    assert_eq!((w.state(), w.remaining()), (LockState::PullIn, 4096));
    assert_eq!(w.advance(4000, chk(false)), None);
    assert_eq!(w.remaining(), 96);
    assert_eq!(w.advance(96, chk(true)), Some(LockTransition::Settle));
    assert_eq!(w.state(), LockState::Track);
    assert_eq!(w.advance(4096, chk(true)), Some(LockTransition::Hold));
    assert_eq!(w.advance(4096, chk(false)), Some(LockTransition::Doubt));
    assert_eq!(w.state(), LockState::Verify);
    // a verify window without a clear peak, or within the tolerance (modulo a quarter cycle)
    assert_eq!(w.advance(768, chk(false)), None);
    assert_eq!(w.window(None, 0.001), LockTransition::Confirm);
    assert_eq!((w.state(), w.remaining()), (LockState::Track, 4096 - 768));
    assert_eq!(
        w.advance(4096 - 768, chk(false)),
        Some(LockTransition::Doubt)
    );
    assert_eq!(
        w.window(Some(0.0011), 0.001 - 0.25),
        LockTransition::Confirm
    );
    assert_eq!(w.advance(4096, chk(false)), Some(LockTransition::Doubt));
    assert_eq!(w.window(Some(0.0013), 0.001), LockTransition::Restart);
    assert_eq!(
        (w.state(), w.restarts(), w.acquired()),
        (LockState::Search, 1, Some(0.001))
    );
    // a miss before the estimate: the equalizer starts again
    assert_eq!(w.window(None, 0.0), LockTransition::Miss);
    assert_eq!(w.window(None, 0.0), LockTransition::Miss);
    assert_eq!(
        w.window(Some(-0.002), 0.0),
        LockTransition::AcquireAfterMiss { frequency: -0.002 }
    );
    assert_eq!(
        w.advance(4096, chk(false)),
        Some(LockTransition::SettleUnlocked)
    );
    assert_eq!(w.state(), LockState::Verify);
    // the next search after a restart is fresh again
    assert_eq!(w.window(Some(0.01), -0.002), LockTransition::Restart);
    assert_eq!(
        w.window(Some(0.01), 0.0),
        LockTransition::Acquire { frequency: 0.01 }
    );
    assert_eq!(w.restarts(), 2);
    assert!(w.recovery());
    w.reset();
    assert_eq!(
        format!("{w:?}"),
        format!("{:?}", LockWatch::<Policy>::new())
    );
}

/// The inputs of a check with the weight energy `energy` against the floor 0.75.
fn chk_e(locked: bool, energy: f32) -> impl FnOnce() -> LockCheck {
    move || LockCheck {
        locked,
        weight_energy: Some(energy),
        weight_floor: 0.75,
    }
}

/// A watch in `Track` after an estimate of 0.001 cycles per symbol.
fn tracking<P: LockPolicy>() -> LockWatch<P> {
    let mut w = LockWatch::<P>::new();
    assert_eq!(
        w.window(Some(0.001), 0.0),
        LockTransition::Acquire { frequency: 0.001 }
    );
    assert_eq!(w.advance(4096, chk(true)), Some(LockTransition::Settle));
    w
}

#[test]
fn lock_watch_takes_the_weight_watch_of_step_two() {
    // the weight watch: low checks in a row in `Track`, with or without lock, also across a
    // verify window; a check above the floor starts the count again
    let mut w = tracking::<Policy>();
    assert_eq!(
        w.advance(4096, chk_e(true, 0.74)),
        Some(LockTransition::Hold)
    );
    assert_eq!(w.low_checks(), 1);
    assert_eq!(
        w.advance(4096, chk_e(true, 0.75)),
        Some(LockTransition::Hold)
    );
    assert_eq!(w.low_checks(), 0);
    assert_eq!(
        w.advance(4096, chk_e(false, 0.5)),
        Some(LockTransition::Doubt)
    );
    assert_eq!(w.advance(768, chk_e(false, 0.5)), None);
    assert_eq!(w.window(None, 0.001), LockTransition::Confirm);
    assert_eq!(
        (w.state(), w.low_checks(), w.misses()),
        (LockState::Track, 1, 1)
    );
    assert_eq!(
        w.advance(4096 - 768, chk_e(false, f32::NAN)),
        Some(LockTransition::Shrink)
    );
    assert_eq!(LockTransition::Shrink.action(), LockAction::RestartAll);
    assert_eq!(
        (w.state(), w.restarts(), w.low_checks(), w.misses()),
        (LockState::Search, 1, 0, 0)
    );
    assert_eq!(w.acquired(), Some(0.001));
    // the first check after an estimate does not count; without an equalizer nothing counts
    let mut w = LockWatch::<Policy>::new();
    w.window(Some(0.001), 0.0);
    assert_eq!(
        w.advance(4096, chk_e(true, 0.1)),
        Some(LockTransition::Settle)
    );
    assert_eq!(w.low_checks(), 0);
    let none = || LockCheck {
        locked: true,
        weight_energy: None,
        weight_floor: 0.75,
    };
    for _ in 0..10 {
        assert_eq!(w.advance(4096, none), Some(LockTransition::Hold));
    }
    assert_eq!(w.restarts(), 0);
}

#[test]
fn lock_watch_takes_the_signal_back_after_a_gap_and_names_the_actions() {
    // the signal back after a gap: a clear verify window within the tolerance after at least
    // `GAP_WINDOWS` verify windows in a row without a clear peak
    let mut w = tracking::<Policy>();
    for k in 1..=6u8 {
        assert_eq!(w.advance(4096, chk(false)), Some(LockTransition::Doubt));
        assert_eq!(w.window(None, 0.001), LockTransition::Confirm);
        assert_eq!(w.misses(), k);
    }
    assert_eq!(w.advance(4096, chk(false)), Some(LockTransition::Doubt));
    let back = w.window(Some(0.00105), 0.001);
    assert_eq!(back, LockTransition::Return { frequency: 0.00105 });
    assert_eq!(back.action(), LockAction::RestartEqualizer);
    assert_eq!(back.frequency(), Some(0.00105));
    assert_eq!(
        (
            w.state(),
            w.remaining(),
            w.acquired(),
            w.restarts(),
            w.misses()
        ),
        (LockState::PullIn, 4096, Some(0.00105), 1, 0)
    );
    assert_eq!(w.advance(4096, chk(true)), Some(LockTransition::Settle));
    // five windows without a clear peak are not a gap; a clear one in between starts the count
    // again (the hysteresis for weak signals, D1619)
    let mut w = tracking::<Policy>();
    for clear in [
        false, false, false, false, false, true, false, false, false, false, false,
    ] {
        assert_eq!(w.advance(4096, chk(false)), Some(LockTransition::Doubt));
        let est = clear.then_some(0.00101);
        assert_eq!(w.window(est, 0.001), LockTransition::Confirm);
    }
    assert_eq!(w.advance(4096, chk(false)), Some(LockTransition::Doubt));
    assert_eq!(w.window(Some(0.00101), 0.001), LockTransition::Confirm);
    assert_eq!((w.restarts(), w.misses()), (0, 0));
    // a clear verify window away from the loops after a gap is a full restart
    let mut w = tracking::<Policy>();
    for _ in 0..8 {
        w.advance(4096, chk(false));
        w.window(None, 0.001);
    }
    w.advance(4096, chk(false));
    assert_eq!(w.window(Some(0.003), 0.001), LockTransition::Restart);
    assert_eq!(w.state(), LockState::Search);

    // the actions of the transitions of step one
    for (t, a) in [
        (LockTransition::Miss, LockAction::Keep),
        (LockTransition::Acquire { frequency: 0.0 }, LockAction::Keep),
        (
            LockTransition::AcquireAfterMiss { frequency: 0.0 },
            LockAction::RestartEqualizer,
        ),
        (LockTransition::Settle, LockAction::Keep),
        (LockTransition::SettleUnlocked, LockAction::Verify),
        (LockTransition::Hold, LockAction::Keep),
        (LockTransition::Doubt, LockAction::Verify),
        (LockTransition::Confirm, LockAction::Keep),
        (LockTransition::Restart, LockAction::RestartAll),
    ] {
        assert_eq!(t.action(), a, "{t:?}");
    }
}

/// A policy with both conditions of step two off.
struct Off;

impl LockPolicy for Off {
    const CHECK: u16 = 4096;
    const TOLERANCE: f64 = 2e-4;
    const WEIGHT_CHECKS: u8 = 0;
    const GAP_WINDOWS: u8 = 0;
}

#[test]
fn lock_watch_without_the_conditions_of_step_two_keeps_step_one() {
    let mut w = tracking::<Off>();
    for _ in 0..100 {
        assert_eq!(
            w.advance(4096, chk_e(false, 0.0)),
            Some(LockTransition::Doubt)
        );
        assert_eq!(w.window(None, 0.001), LockTransition::Confirm);
    }
    assert_eq!((w.misses(), w.low_checks()), (100, 31));
    assert_eq!(w.advance(4096, chk(false)), Some(LockTransition::Doubt));
    assert_eq!(w.window(Some(0.001), 0.001), LockTransition::Confirm);
    assert_eq!((w.restarts(), w.misses(), w.low_checks()), (0, 0, 0));
}
