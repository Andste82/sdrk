//! Clock-drift controller: keeps a buffer between two clock domains at a set fill level by
//! steering the `ppm` parameter of the tunable `resample` block.
//!
//! The buffer sits between a producer (the `resample` node, at its output rate corrected by
//! `ppm`) and a consumer (a sound card or a transmitter, at its own clock). The application
//! calls [`DriftController::update`] at roughly regular moments with the current fill level.
//! See `docs/blocks.md`, section "Clock drift", for the formulas.

use crate::error::{Error, Result};

/// Gain of the plant: one ppm changes the error by `1e-6` seconds per second.
const PPM_TO_RATE: f64 = 1e-6;

/// Largest `ppm` the tunable `resample` block accepts.
const RESAMPLE_MAX_PPM: f64 = 1000.0;

/// Parameters of a [`DriftController`].
///
/// ```
/// use sdrk_blocks::DriftConfig;
///
/// let cfg = DriftConfig::new(48_000.0, 4_800.0);
/// assert_eq!(cfg.time_constant_s, 30.0);
/// assert_eq!(cfg.max_ppm, 500.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftConfig {
    /// Output rate of the `resample` node in Hz (`fs`), greater than 0.
    pub rate_hz: f64,
    /// Target fill level in samples, greater than 0.
    pub target: f64,
    /// Loop time constant in seconds, 1 to 600 (default 30).
    pub time_constant_s: f64,
    /// Limit of the output in ppm, greater than 0 and at most 1000, the range of `resample`
    /// (default 500).
    pub max_ppm: f64,
    /// Largest change of the output per second in ppm, greater than 0 (default 20).
    pub slew_ppm_per_s: f64,
    /// Time constant of the fill measurement filter in seconds, greater than 0 (default 1).
    pub smoothing_s: f64,
}

impl DriftConfig {
    /// A configuration with the given rate and target and the defaults for everything else.
    ///
    /// ```
    /// use sdrk_blocks::DriftConfig;
    ///
    /// let cfg = DriftConfig::new(44_100.0, 2_000.0);
    /// assert_eq!((cfg.rate_hz, cfg.target), (44_100.0, 2_000.0));
    /// assert_eq!(cfg.slew_ppm_per_s, 20.0);
    /// assert_eq!(cfg.smoothing_s, 1.0);
    /// ```
    #[must_use]
    pub const fn new(rate_hz: f64, target: f64) -> Self {
        Self {
            rate_hz,
            target,
            time_constant_s: 30.0,
            max_ppm: 500.0,
            slew_ppm_per_s: 20.0,
            smoothing_s: 1.0,
        }
    }

    fn validate(&self) -> Result<()> {
        let bad = |name: &'static str, reason: &'static str| {
            Err(Error::InvalidParameter { name, reason })
        };
        if !(self.rate_hz.is_finite() && self.rate_hz > 0.0) {
            return bad("rate_hz", "must be finite and greater than 0");
        }
        if !(self.target.is_finite() && self.target > 0.0) {
            return bad("target", "must be finite and greater than 0");
        }
        if !(1.0..=600.0).contains(&self.time_constant_s) {
            return bad("time_constant_s", "must be in 1 to 600");
        }
        if !(self.max_ppm > 0.0 && self.max_ppm <= RESAMPLE_MAX_PPM) {
            return bad("max_ppm", "must be greater than 0 and at most 1000");
        }
        if !(self.slew_ppm_per_s.is_finite() && self.slew_ppm_per_s > 0.0) {
            return bad("slew_ppm_per_s", "must be finite and greater than 0");
        }
        if !(self.smoothing_s.is_finite() && self.smoothing_s > 0.0) {
            return bad("smoothing_s", "must be finite and greater than 0");
        }
        Ok(())
    }
}

/// PI controller that turns a buffer fill level into a `ppm` value for the tunable
/// `resample` block. Heap-free, f64 arithmetic only.
///
/// ```
/// use sdrk_blocks::{DriftConfig, DriftController};
///
/// let mut c = DriftController::new(DriftConfig::new(48_000.0, 4_800.0))?;
/// // The first call only initialises the state and returns the initial ppm.
/// assert_eq!(c.update(0.0, 4_800.0), 0.0);
/// // A buffer that fills needs a lower ppm.
/// let ppm = c.update(1.0, 5_300.0);
/// assert!(ppm < 0.0);
/// assert_eq!(c.ppm(), ppm);
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct DriftController {
    config: DriftConfig,
    initial_ppm: f64,
    smoothed_error: f64,
    integrator: f64,
    output_ppm: f64,
    last_s: f64,
    started: bool,
}

impl DriftController {
    /// Creates a controller. Fails with `InvalidParameter` when a value is outside its range.
    ///
    /// ```
    /// use sdrk_blocks::{DriftConfig, DriftController};
    ///
    /// assert!(DriftController::new(DriftConfig::new(48_000.0, 4_800.0)).is_ok());
    /// let mut bad = DriftConfig::new(48_000.0, 4_800.0);
    /// bad.max_ppm = 2000.0;
    /// assert!(DriftController::new(bad).is_err());
    /// ```
    pub fn new(config: DriftConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            initial_ppm: 0.0,
            smoothed_error: 0.0,
            integrator: 0.0,
            output_ppm: 0.0,
            last_s: 0.0,
            started: false,
        })
    }

    /// Observes the buffer and returns the `ppm` value to set.
    ///
    /// `now_s` is the monotonic time of the observation in seconds, `fill` the number of
    /// samples in the buffer. The first call only initialises the state and returns the initial
    /// ppm. A call with `dt <= 0` (or a non-finite time or fill) changes nothing and returns
    /// the current output.
    ///
    /// ```
    /// use sdrk_blocks::{DriftConfig, DriftController};
    ///
    /// let mut c = DriftController::new(DriftConfig::new(48_000.0, 4_800.0))?;
    /// c.update(10.0, 4_800.0);
    /// let before = c.ppm();
    /// assert_eq!(c.update(10.0, 9_000.0), before); // dt = 0: nothing changes
    /// # Ok::<(), sdrk_blocks::Error>(())
    /// ```
    pub fn update(&mut self, now_s: f64, fill: f64) -> f64 {
        let cfg = &self.config;
        if !(now_s.is_finite() && fill.is_finite()) {
            return self.output_ppm;
        }
        // 1. error in seconds
        let e = (fill - cfg.target) / cfg.rate_hz;
        if !self.started {
            self.started = true;
            self.last_s = now_s;
            self.smoothed_error = e;
            self.integrator = 0.0;
            self.output_ppm = self.initial_ppm;
            return self.output_ppm;
        }
        let dt = now_s - self.last_s;
        if dt <= 0.0 {
            return self.output_ppm;
        }
        self.last_s = now_s;
        // 2. smoothing of the measurement
        let a = 1.0 - libm::exp(-dt / cfg.smoothing_s);
        self.smoothed_error += a * (e - self.smoothed_error);
        // 3. gains of a critically damped second-order loop
        let wn = 1.0 / cfg.time_constant_s;
        let kp = 2.0 * wn / PPM_TO_RATE;
        let ki = wn * wn / PPM_TO_RATE;
        // 4. integrator with anti-windup
        self.integrator = clamp(self.integrator + ki * self.smoothed_error * dt, cfg.max_ppm);
        // 5. target value
        let u = clamp(
            self.initial_ppm - (kp * self.smoothed_error + self.integrator),
            cfg.max_ppm,
        );
        // 6. slew limit
        let max_step = cfg.slew_ppm_per_s * dt;
        let step = (u - self.output_ppm).clamp(-max_step, max_step);
        // 7. store
        self.output_ppm += step;
        self.output_ppm
    }

    /// The current output in ppm (the initial ppm before the first update).
    ///
    /// ```
    /// use sdrk_blocks::{DriftConfig, DriftController};
    ///
    /// let c = DriftController::new(DriftConfig::new(48_000.0, 4_800.0))?;
    /// assert_eq!(c.ppm(), 0.0);
    /// # Ok::<(), sdrk_blocks::Error>(())
    /// ```
    #[must_use]
    pub const fn ppm(&self) -> f64 {
        self.output_ppm
    }

    /// Sets a known offset (for example a measured device error), clamped to `max_ppm`. It is
    /// the output until the first update and the centre of the loop afterwards. A NaN is
    /// treated as 0.
    ///
    /// ```
    /// use sdrk_blocks::{DriftConfig, DriftController};
    ///
    /// let mut c = DriftController::new(DriftConfig::new(48_000.0, 4_800.0))?;
    /// c.set_initial_ppm(37.5);
    /// assert_eq!(c.ppm(), 37.5);
    /// c.set_initial_ppm(9_999.0);
    /// assert_eq!(c.ppm(), 500.0);
    /// # Ok::<(), sdrk_blocks::Error>(())
    /// ```
    pub fn set_initial_ppm(&mut self, ppm: f64) {
        let ppm = if ppm.is_nan() { 0.0 } else { ppm };
        self.initial_ppm = clamp(ppm, self.config.max_ppm);
        if !self.started {
            self.output_ppm = self.initial_ppm;
        }
    }

    /// Forgets the history; the config and the initial ppm are kept and the next update
    /// starts again from the initial ppm.
    ///
    /// ```
    /// use sdrk_blocks::{DriftConfig, DriftController};
    ///
    /// let mut c = DriftController::new(DriftConfig::new(48_000.0, 4_800.0))?;
    /// c.set_initial_ppm(10.0);
    /// c.update(0.0, 4_800.0);
    /// c.update(5.0, 6_000.0);
    /// c.reset();
    /// assert_eq!(c.ppm(), 10.0);
    /// # Ok::<(), sdrk_blocks::Error>(())
    /// ```
    pub fn reset(&mut self) {
        self.smoothed_error = 0.0;
        self.integrator = 0.0;
        self.output_ppm = self.initial_ppm;
        self.last_s = 0.0;
        self.started = false;
    }
}

fn clamp(x: f64, limit: f64) -> f64 {
    x.clamp(-limit, limit)
}
