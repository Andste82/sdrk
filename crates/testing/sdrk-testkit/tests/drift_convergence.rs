//! Closed-loop tests of `DriftController` against `TwoClockSim` (M28 tests 1 to 3).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_precision_loss,
    clippy::float_cmp
)]

use sdrk_blocks::{DriftConfig, DriftController};
use sdrk_testkit::drift::TwoClockSim;

const FS: f64 = 48_000.0;
const CAPACITY: usize = 9_600;
const TARGET: f64 = 4_800.0;
const BLOCK: usize = 480;
const SEED: u64 = 462;
const TAU: f64 = 30.0;
const SLEW: f64 = 20.0;
/// Peak fill error allowed after `5 tau`: the jitter of the consumer call times alone moves
/// the fill by up to 96 samples (20 % of a 480-sample block), so the bound is 0.0025 fs.
const PEAK_FILL: f64 = 0.0025 * FS;
/// Peak ripple of the output allowed after `5 tau`. The jitter of the smoothed error drives
/// the proportional term; measured 23 ppm.
const PEAK_PPM: f64 = 30.0;
/// How long the drift of 2000 ppm lasts in the limits test. The slew limit needs 25 s to
/// reach -500 ppm, and while the output sits at the limit the buffer gains 72 samples/s.
const HOLD_S: f64 = 50.0;

fn controller() -> DriftController {
    DriftController::new(DriftConfig::new(FS, TARGET)).unwrap()
}

/// One closed-loop call: simulate, update the controller, apply its output.
fn tick(sim: &mut TwoClockSim, ctl: &mut DriftController) -> (f64, f64, f64) {
    let (now, fill) = sim.step();
    let ppm = ctl.update(now, fill);
    sim.set_ppm(ppm);
    (now, fill, ppm)
}

/// Results of one run, so that the guard tests and the spec tests share the loops.
struct Settled {
    overruns: u64,
    underruns: u64,
    worst_fill: f64,
    worst_ppm: f64,
    /// Mean of `fill - target` and of `ppm + drift` over the settled part of the run.
    mean_fill: f64,
    mean_ppm: f64,
}

/// Collects the statistics of the settled part of a run.
#[derive(Default)]
struct Window {
    n: f64,
    sum_fill: f64,
    sum_ppm: f64,
    worst_fill: f64,
    worst_ppm: f64,
}

impl Window {
    fn add(&mut self, fill_err: f64, ppm_err: f64) {
        self.n += 1.0;
        self.sum_fill += fill_err;
        self.sum_ppm += ppm_err;
        self.worst_fill = self.worst_fill.max(fill_err.abs());
        self.worst_ppm = self.worst_ppm.max(ppm_err.abs());
    }

    fn finish(&self, sim: &TwoClockSim) -> Settled {
        Settled {
            overruns: sim.overruns(),
            underruns: sim.underruns(),
            worst_fill: self.worst_fill,
            worst_ppm: self.worst_ppm,
            mean_fill: self.sum_fill / self.n,
            mean_ppm: self.sum_ppm / self.n,
        }
    }
}

fn run_constant(drift: f64) -> Settled {
    let mut sim = TwoClockSim::new(FS, drift, CAPACITY, BLOCK, SEED);
    let mut ctl = controller();
    let mut prev: Option<(f64, f64)> = None;
    let mut win = Window::default();
    loop {
        let (now, fill, ppm) = tick(&mut sim, &mut ctl);
        if let Some((t0, p0)) = prev {
            assert!(
                (ppm - p0).abs() <= SLEW * (now - t0) + 1e-9,
                "slew violated at {now} s (drift {drift})"
            );
        }
        prev = Some((now, ppm));
        if now > 5.0 * TAU {
            win.add(fill - TARGET, ppm + drift);
        }
        if now >= 7_200.0 {
            break;
        }
    }
    win.finish(&sim)
}

fn run_step() -> Settled {
    let mut sim = TwoClockSim::new(FS, 0.0, CAPACITY, BLOCK, SEED);
    let mut ctl = controller();
    let mut stepped = false;
    let mut t_step = 0.0;
    let mut win = Window::default();
    loop {
        let (now, fill, ppm) = tick(&mut sim, &mut ctl);
        if !stepped && now >= 600.0 {
            sim.set_drift_ppm(100.0);
            stepped = true;
            t_step = now;
        }
        if stepped && now > t_step + 5.0 * TAU {
            win.add(fill - TARGET, ppm + 100.0);
        }
        if now >= 1_800.0 {
            break;
        }
    }
    win.finish(&sim)
}

/// Drift of 2000 ppm for `HOLD_S`, then 0. Returns `(ppm at the end of the hold, seconds
/// until the output leaves the limit, excess fill at the end of the hold, overruns)`.
fn run_limits() -> (f64, f64, f64, u64) {
    let mut sim = TwoClockSim::new(FS, 2_000.0, CAPACITY, BLOCK, SEED);
    let mut ctl = controller();
    let mut at_hold: Option<(f64, f64, f64)> = None;
    let mut left = None;
    loop {
        let (now, fill, ppm) = tick(&mut sim, &mut ctl);
        match at_hold {
            None if now >= HOLD_S => {
                at_hold = Some((now, ppm, fill - TARGET));
                sim.set_drift_ppm(0.0);
            }
            Some((t0, ..)) if left.is_none() && ppm > -500.0 + 1e-9 => left = Some(now - t0),
            _ => {}
        }
        if now >= HOLD_S + 10.0 * TAU {
            break;
        }
    }
    let (_, ppm_at_hold, excess) = at_hold.unwrap();
    (
        ppm_at_hold,
        left.unwrap_or(f64::INFINITY),
        excess,
        sim.overruns(),
    )
}

/// Spec test 1. The jitter of the consumer call times moves the fill by up to 96 samples on
/// its own and the output ripples around its mean (D310), so after `5 tau` the means must
/// be exact (fill within 0.002 fs, output within 1 ppm of `-drift`) and the peaks bounded.
#[test]
fn converges_for_every_drift() {
    for drift in [-100.0, -20.0, 0.0, 20.0, 100.0] {
        let r = run_constant(drift);
        assert_eq!(r.overruns, 0, "overruns at {drift} ppm");
        assert_eq!(r.underruns, 0, "underruns at {drift} ppm");
        assert!(
            r.mean_fill.abs() <= 0.002 * FS,
            "mean fill off by {} at {drift} ppm",
            r.mean_fill
        );
        assert!(
            r.mean_ppm.abs() <= 1.0,
            "mean ppm off by {} at {drift} ppm",
            r.mean_ppm
        );
        assert!(
            r.worst_fill <= PEAK_FILL,
            "fill off by {} at {drift} ppm",
            r.worst_fill
        );
        assert!(
            r.worst_ppm <= PEAK_PPM,
            "ppm off by {} at {drift} ppm",
            r.worst_ppm
        );
    }
}

/// Spec test 2: the same bounds, counted from `5 tau` after the step.
#[test]
fn settles_again_after_a_drift_step() {
    let r = run_step();
    assert_eq!(r.overruns, 0);
    assert_eq!(r.underruns, 0);
    assert!(
        r.mean_fill.abs() <= 0.002 * FS,
        "mean fill off by {}",
        r.mean_fill
    );
    assert!(r.mean_ppm.abs() <= 1.0, "mean ppm off by {}", r.mean_ppm);
    assert!(r.worst_fill <= PEAK_FILL, "fill off by {}", r.worst_fill);
    assert!(r.worst_ppm <= PEAK_PPM, "ppm off by {}", r.worst_ppm);
}

/// Spec test 3. The output saturates at `-max_ppm`. After the drift returns to 0 the buffer
/// still holds the excess gathered while the output was slewing to and sitting at the limit,
/// and that excess can only drain at `max_ppm * fs` samples per second. Without windup the
/// output leaves the limit once the excess has drained plus `2 tau`; a wound-up integrator
/// would add to that.
#[test]
fn saturates_at_the_limit_and_leaves_it_after_the_drift_returns() {
    let (ppm_at_hold, left_after, excess, overruns) = run_limits();
    assert_eq!(ppm_at_hold, -500.0);
    assert_eq!(overruns, 0);
    let drain_s = excess / (500.0 * 1e-6 * FS);
    assert!(
        left_after <= drain_s + 2.0 * TAU,
        "left the limit after {left_after} s, excess drains in {drain_s} s"
    );
}
