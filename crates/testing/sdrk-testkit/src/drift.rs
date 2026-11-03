//! Two-clock buffer simulation for the clock drift tests (M28).
//!
//! A producer (the `resample` node, running at `rate_hz` corrected by the
//! drift and the applied ppm) fills a buffer, and a consumer (a sound card)
//! takes `consumer_block` samples at its own, jittery callback times. The
//! simulation works on fill counts only, no samples, so two simulated hours
//! take well under a second in a release build.
//!
//! ```
//! use sdrk_testkit::drift::TwoClockSim;
//! let mut sim = TwoClockSim::new(48_000.0, 0.0, 9600, 480, 462);
//! let (now_s, fill) = sim.step();
//! assert!(now_s > 0.0);
//! assert!((fill - 4800.0).abs() < 100.0);
//! assert_eq!(sim.overruns() + sim.underruns(), 0);
//! ```

use crate::rng::Pcg32;

/// Relative half width of the uniform jitter of the consumer period.
const JITTER: f64 = 0.2;

/// Producer and consumer on two clocks with a shared buffer.
#[derive(Debug, Clone)]
pub struct TwoClockSim {
    rate_hz: f64,
    drift_ppm: f64,
    ppm_set: f64,
    capacity: f64,
    consumer_block: f64,
    rng: Pcg32,
    calls: u64,
    now_s: f64,
    fill: f64,
    overruns: u64,
    underruns: u64,
}

impl TwoClockSim {
    /// Creates a simulation. The buffer starts half full. The producer rate
    /// is `rate_hz * (1 + (drift_ppm + ppm_set) * 1e-6)`; the consumer takes
    /// `consumer_block` samples every `consumer_block / rate_hz` seconds, each
    /// call time is shifted from its nominal moment by a uniform offset of up to
    /// `±0.2` periods drawn from a PCG32 with `jitter_seed` (the offsets do not
    /// accumulate).
    pub fn new(
        rate_hz: f64,
        drift_ppm: f64,
        buffer_capacity: usize,
        consumer_block: usize,
        jitter_seed: u64,
    ) -> Self {
        let capacity = buffer_capacity as f64;
        Self {
            rate_hz,
            drift_ppm,
            ppm_set: 0.0,
            capacity,
            consumer_block: consumer_block as f64,
            rng: Pcg32::new(jitter_seed, 54),
            calls: 0,
            now_s: 0.0,
            fill: 0.5 * capacity,
            overruns: 0,
            underruns: 0,
        }
    }

    /// Advances to the next consumer call and returns `(now_s, fill)`, the
    /// time and the fill level after the consumer took its block. A fill above
    /// the capacity before the take counts as an overrun, a fill below zero
    /// after the take as an underrun; the fill is clamped to `0 ..= capacity`.
    pub fn step(&mut self) -> (f64, f64) {
        let period = self.consumer_block / self.rate_hz;
        // The nominal schedule is exact, only each call time is jittered; the jitter does
        // not accumulate, so the consumer clock keeps its long-term rate.
        self.calls += 1;
        let t = self.calls as f64 * period + period * JITTER * (2.0 * self.rng.next_f64() - 1.0);
        let dt = t - self.now_s;
        self.now_s = t;
        let producer_rate = self.rate_hz * (1.0 + (self.drift_ppm + self.ppm_set) * 1e-6);
        self.fill += producer_rate * dt;
        if self.fill > self.capacity {
            self.overruns += 1;
            self.fill = self.capacity;
        }
        self.fill -= self.consumer_block;
        if self.fill < 0.0 {
            self.underruns += 1;
            self.fill = 0.0;
        }
        (self.now_s, self.fill)
    }

    /// Applies the controller output (ppm of the producer).
    pub fn set_ppm(&mut self, ppm: f64) {
        self.ppm_set = ppm;
    }

    /// Changes the clock drift between the two clocks (for step tests).
    pub fn set_drift_ppm(&mut self, drift_ppm: f64) {
        self.drift_ppm = drift_ppm;
    }

    /// The ppm applied last.
    pub fn ppm(&self) -> f64 {
        self.ppm_set
    }

    /// Simulated time of the last consumer call in seconds.
    pub fn now_s(&self) -> f64 {
        self.now_s
    }

    /// Current fill level in samples.
    pub fn fill(&self) -> f64 {
        self.fill
    }

    /// Number of steps in which the buffer overflowed.
    pub fn overruns(&self) -> u64 {
        self.overruns
    }

    /// Number of steps in which the buffer ran empty.
    pub fn underruns(&self) -> u64 {
        self.underruns
    }
}
