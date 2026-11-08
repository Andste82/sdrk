//! Short-range device transmissions of the ISM bands (M45.1): bursts of on-off keying and
//! two-level FSK, and scenes of many of them with noise, a common clock error and interferers.
//!
//! # Transmissions
//!
//! A [`Transmission`] is a burst of `bits` in one of four [`Coding`]s, sent `repeats` times:
//!
//! - [`Coding::Pwm`]: every bit is a pulse and a gap of `gap_us`; a 1 is a pulse of
//!   `short_us`, a 0 one of `long_us`.
//! - [`Coding::Ppm`]: every bit is a pulse of `pulse_us` and a gap; a 0 is a gap of
//!   `short_gap_us`, a 1 one of `long_gap_us`. A final pulse of `pulse_us` ends the last gap.
//! - [`Coding::Manchester`]: every bit is two halves of `half_bit_us`; a 1 is carrier on then
//!   off (a falling edge in the middle), a 0 off then on.
//! - [`Coding::Fsk`]: every bit is `bit_us` at `freq_offset_hz + deviation_hz` for a 1 and at
//!   `freq_offset_hz - deviation_hz` for a 0, with a continuous phase (NRZ).
//!
//! A copy ends with its last time of carrier on (a trailing gap or off half is not sent); the
//! next copy starts `repeat_gap_us` later, carrier off in between (also for FSK). Every duration
//! is multiplied by `1 + timing_error` (the clock error of the device). [`Transmission::runs`]
//! lists the result as [`Run`]s of one [`Level`] each (adjacent equal levels joined), the exact
//! description the detector of M45.1 should recover. Time 0 of a transmission is the start of
//! its first bit.
//!
//! The rendered signal at time `t` (seconds from the start) is
//!
//! ```text
//! s(t) = A a(t) exp(j 2 pi (f t + dev D(t) + phi0)),   A = 10^(level_db / 20)
//! ```
//!
//! with `f` the frequency offset, `D(t)` the integral of +1 (upper frequency), -1 (lower) and 0
//! (OOK, carrier off) over the runs up to `t` (so the phase is continuous at every FSK
//! transition), and the envelope `a(t)` 1 while a run is on, 0 while it is off, with a raised
//! cosine of [`RISE_US`] centred on every edge between on and off (the envelope crosses 1/2
//! exactly at the nominal edge, so widths measured at half amplitude are the nominal widths).
//! [`Transmission::render`] adds a transmission to a buffer at a sample rate.
//!
//! # Scenes
//!
//! A [`Scene`] places transmissions at start times (in a [`Placed`] each) and adds complex
//! Gaussian noise over the whole band, [`Interferer`]s (a CW carrier, a wideband burst of
//! Gaussian noise) and a common clock error `clock_error_ppm` of the receiver: with
//! `e = clock_error_ppm * 1e-6` the receiver samples at `fs (1 + e)` while it believes `fs`
//! and its local oscillator is `center_hz * e` too high, so sample `n` is the true time
//! `n / (fs (1 + e))` and every carrier is shifted by `-center_hz * e`.
//! [`Scene::apparent_offset_hz`] and [`Scene::apparent_us`] give what the receiver measures.
//! [`Scene::renderer`] checks the scene and returns a [`SceneRenderer`] that writes the band in
//! pieces of any length; every value is a function of the sample index and the seeded random
//! streams, so the same scene gives the same samples in any chunking.
//!
//! The random streams: the band noise uses `Pcg32::new(seed, 0)`, interferer `i`
//! `Pcg32::new(seed, 1 + i)` (the start phase of a carrier, the noise of a burst), and the
//! start phase `phi0` of transmission `i` is the first `next_f64` of
//! `Pcg32::new(seed, 0x1_0000 + i)`.
//!
//! ```
//! use num_complex::Complex32;
//! use sdrk_testkit::ism::{Coding, Level, Run, Transmission};
//!
//! let t = Transmission {
//!     coding: Coding::Pwm { short_us: 100.0, long_us: 300.0, gap_us: 200.0 },
//!     bits: &[true, false],
//!     repeats: 1,
//!     repeat_gap_us: 0.0,
//!     freq_offset_hz: 50_000.0,
//!     level_db: -20.0,
//!     timing_error: 0.0,
//! };
//! let runs: Vec<Run> = t.runs().collect();
//! assert_eq!(runs, [
//!     Run { level: Level::On, us: 100.0 },
//!     Run { level: Level::Off, us: 200.0 },
//!     Run { level: Level::On, us: 300.0 },
//! ]);
//! let mut x = vec![Complex32::new(0.0, 0.0); 2000];
//! t.render(1e6, &mut x)?;
//! assert!((x[50].norm_sqr().sqrt() - 0.1).abs() < 1e-6); // carrier on at -20 dBFS
//! assert_eq!(x[200].norm_sqr(), 0.0); // the gap
//! # Ok::<(), sdrk_testkit::ism::IsmError>(())
//! ```

// The renderer keeps up to `MAX_SCENE_TRANSMISSIONS` cursors inline (no heap); a test helper,
// not a block.
#![allow(clippy::large_stack_frames)]

use core::f64::consts::PI;

use num_complex::{Complex32, Complex64};

use crate::rng::Pcg32;

/// Rise and fall time of the OOK envelope (and of the FSK carrier at the start and end of a
/// copy) in µs: a raised cosine of this length centred on every edge.
pub const RISE_US: f64 = 2.0;

/// The most transmissions a [`Scene`] holds.
pub const MAX_SCENE_TRANSMISSIONS: usize = 64;

/// The most interferers a [`Scene`] holds.
pub const MAX_SCENE_INTERFERERS: usize = 16;

/// The line code of a [`Transmission`] (see the [module documentation](self)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Coding {
    /// Pulse-width modulation: a 1 is a short pulse, a 0 a long one, each followed by a gap.
    Pwm {
        /// Pulse width of a 1 in µs.
        short_us: f64,
        /// Pulse width of a 0 in µs.
        long_us: f64,
        /// Gap after every pulse in µs.
        gap_us: f64,
    },
    /// Pulse-position modulation: equal pulses, a 0 is a short gap after the pulse, a 1 a long
    /// one; a final pulse ends the last gap.
    Ppm {
        /// Pulse width in µs.
        pulse_us: f64,
        /// Gap of a 0 in µs.
        short_gap_us: f64,
        /// Gap of a 1 in µs.
        long_gap_us: f64,
    },
    /// Manchester code: a 1 is on then off, a 0 off then on.
    Manchester {
        /// Half a bit in µs.
        half_bit_us: f64,
    },
    /// Two-level FSK (NRZ): a 1 at the upper, a 0 at the lower frequency.
    Fsk {
        /// Bit duration in µs.
        bit_us: f64,
        /// Frequency deviation in Hz (the two frequencies are the offset ± this).
        deviation_hz: f64,
    },
}

/// One burst of a short-range device (see the [module documentation](self)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transmission<'a> {
    /// The line code.
    pub coding: Coding,
    /// The bits of one copy, first bit first (at least one).
    pub bits: &'a [bool],
    /// Number of copies sent (at least 1).
    pub repeats: u32,
    /// Carrier off between the end of a copy and the start of the next in µs.
    pub repeat_gap_us: f64,
    /// Carrier frequency relative to the centre of the band in Hz.
    pub freq_offset_hz: f64,
    /// Level of the carrier in dBFS (amplitude `10^(level_db / 20)`).
    pub level_db: f64,
    /// Relative error of every duration (0.01: everything 1 % longer); above -1.
    pub timing_error: f64,
}

/// The state of a [`Run`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Level {
    /// Carrier off.
    Off,
    /// OOK carrier on.
    On,
    /// FSK at the lower frequency.
    Low,
    /// FSK at the upper frequency.
    High,
}

impl Level {
    fn amplitude(self) -> f64 {
        match self {
            Self::Off => 0.0,
            Self::On | Self::Low | Self::High => 1.0,
        }
    }

    fn sign(self) -> f64 {
        match self {
            Self::Off | Self::On => 0.0,
            Self::Low => -1.0,
            Self::High => 1.0,
        }
    }
}

/// A time of one [`Level`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Run {
    /// The level.
    pub level: Level,
    /// Duration in µs (timing error included).
    pub us: f64,
}

/// A transmission or scene that cannot be rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IsmError {
    /// More than [`MAX_SCENE_TRANSMISSIONS`] transmissions or [`MAX_SCENE_INTERFERERS`]
    /// interferers.
    TooMany,
    /// A value is out of its range; the name says which.
    InvalidParameter(&'static str),
}

impl core::fmt::Display for IsmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooMany => write!(
                f,
                "a scene holds at most {MAX_SCENE_TRANSMISSIONS} transmissions and {MAX_SCENE_INTERFERERS} interferers"
            ),
            Self::InvalidParameter(name) => write!(f, "invalid ism parameter `{name}`"),
        }
    }
}

impl core::error::Error for IsmError {}

fn positive(v: f64, name: &'static str) -> Result<(), IsmError> {
    if v.is_finite() && v > 0.0 {
        Ok(())
    } else {
        Err(IsmError::InvalidParameter(name))
    }
}

impl<'a> Transmission<'a> {
    /// Checks the description.
    ///
    /// # Errors
    /// [`IsmError::InvalidParameter`] for no bits, no copies, a duration or deviation that is
    /// not positive and finite, a repeat gap that is negative or not finite, a frequency or level
    /// that is not finite or a timing error that is not finite or at most -1.
    pub fn validate(&self) -> Result<(), IsmError> {
        if self.bits.is_empty() {
            return Err(IsmError::InvalidParameter("bits"));
        }
        if self.repeats == 0 {
            return Err(IsmError::InvalidParameter("repeats"));
        }
        match self.coding {
            Coding::Pwm {
                short_us,
                long_us,
                gap_us,
            } => {
                positive(short_us, "short_us")?;
                positive(long_us, "long_us")?;
                positive(gap_us, "gap_us")?;
            }
            Coding::Ppm {
                pulse_us,
                short_gap_us,
                long_gap_us,
            } => {
                positive(pulse_us, "pulse_us")?;
                positive(short_gap_us, "short_gap_us")?;
                positive(long_gap_us, "long_gap_us")?;
            }
            Coding::Manchester { half_bit_us } => positive(half_bit_us, "half_bit_us")?,
            Coding::Fsk {
                bit_us,
                deviation_hz,
            } => {
                positive(bit_us, "bit_us")?;
                positive(deviation_hz, "deviation_hz")?;
            }
        }
        if !(self.repeat_gap_us.is_finite() && self.repeat_gap_us >= 0.0) {
            return Err(IsmError::InvalidParameter("repeat_gap_us"));
        }
        if !self.freq_offset_hz.is_finite() {
            return Err(IsmError::InvalidParameter("freq_offset_hz"));
        }
        if !self.level_db.is_finite() {
            return Err(IsmError::InvalidParameter("level_db"));
        }
        if !(self.timing_error.is_finite() && self.timing_error > -1.0) {
            return Err(IsmError::InvalidParameter("timing_error"));
        }
        Ok(())
    }

    /// The runs of the whole transmission, all copies, in order (see the
    /// [module documentation](self)). It may start with [`Level::Off`] (a Manchester 0) and
    /// ends with a run that is not off.
    pub fn runs(&self) -> Runs<'a> {
        Runs {
            t: *self,
            copy: 0,
            slot: 0,
            pending: None,
            scale: 1.0 + self.timing_error,
        }
    }

    /// The duration from the start of the first bit to the end of the last run in µs.
    pub fn duration_us(&self) -> f64 {
        self.runs().map(|r| r.us).sum()
    }

    /// The largest absolute instantaneous frequency in Hz (offset plus deviation).
    fn max_abs_hz(&self) -> f64 {
        self.freq_offset_hz.abs() + self.deviation_hz()
    }

    /// The FSK deviation in Hz, 0 for OOK.
    fn deviation_hz(&self) -> f64 {
        match self.coding {
            Coding::Fsk { deviation_hz, .. } => deviation_hz,
            _ => 0.0,
        }
    }

    /// Adds the transmission to `out` at the sample rate `fs`, its time 0 at `out[0]`, with
    /// the start phase 0 and without clock error.
    ///
    /// # Errors
    /// The errors of [`Transmission::validate`]; [`IsmError::InvalidParameter`] for a sample
    /// rate that is not positive and finite (`"fs"`) or a frequency (offset plus deviation)
    /// at or beyond `fs / 2` (`"freq_offset_hz"`).
    pub fn render(&self, fs: f64, out: &mut [Complex32]) -> Result<(), IsmError> {
        self.validate()?;
        positive(fs, "fs")?;
        if self.max_abs_hz() >= fs / 2.0 {
            return Err(IsmError::InvalidParameter("freq_offset_hz"));
        }
        let mut c = Cursor::new(*self, self.freq_offset_hz, 0.0, 0.0);
        for (n, o) in out.iter_mut().enumerate() {
            let t = n as f64 / fs;
            if t > c.end_s {
                break;
            }
            let v = c.value(t);
            *o += Complex32::new(v.re as f32, v.im as f32);
        }
        Ok(())
    }
}

/// The runs of a [`Transmission`], from [`Transmission::runs`].
#[derive(Debug, Clone)]
pub struct Runs<'a> {
    t: Transmission<'a>,
    copy: u32,
    slot: usize,
    pending: Option<Run>,
    scale: f64,
}

impl Runs<'_> {
    /// Slots of one copy: the elements of every bit and the final PPM pulse.
    fn slots(&self) -> usize {
        let n = self.t.bits.len();
        match self.t.coding {
            Coding::Pwm { .. } | Coding::Manchester { .. } => 2 * n,
            Coding::Ppm { .. } => 2 * n + 1,
            Coding::Fsk { .. } => n,
        }
    }

    /// Element `slot` of a copy (level and nominal µs).
    fn element(&self, slot: usize) -> (Level, f64) {
        let bit = |k: usize| self.t.bits.get(k).copied().unwrap_or(false);
        match self.t.coding {
            Coding::Pwm {
                short_us,
                long_us,
                gap_us,
            } => {
                if !slot.is_multiple_of(2) {
                    (Level::Off, gap_us)
                } else if bit(slot / 2) {
                    (Level::On, short_us)
                } else {
                    (Level::On, long_us)
                }
            }
            Coding::Ppm {
                pulse_us,
                short_gap_us,
                long_gap_us,
            } => {
                if slot.is_multiple_of(2) {
                    (Level::On, pulse_us)
                } else if bit(slot / 2) {
                    (Level::Off, long_gap_us)
                } else {
                    (Level::Off, short_gap_us)
                }
            }
            Coding::Manchester { half_bit_us } => {
                let first_on = bit(slot / 2);
                let on = first_on == slot.is_multiple_of(2);
                (if on { Level::On } else { Level::Off }, half_bit_us)
            }
            Coding::Fsk { bit_us, .. } => {
                if bit(slot) {
                    (Level::High, bit_us)
                } else {
                    (Level::Low, bit_us)
                }
            }
        }
    }

    /// The next element before joining, or `None` at the end.
    fn raw(&mut self) -> Option<Run> {
        loop {
            if self.copy >= self.t.repeats {
                return None;
            }
            let slots = self.slots();
            let slot = self.slot;
            self.slot += 1;
            if slot < slots {
                let (level, us) = self.element(slot);
                // A copy ends with its last time of carrier on.
                if slot + 1 == slots && level == Level::Off {
                    continue;
                }
                return Some(Run {
                    level,
                    us: us * self.scale,
                });
            }
            self.slot = 0;
            self.copy += 1;
            if self.copy < self.t.repeats {
                return Some(Run {
                    level: Level::Off,
                    us: self.t.repeat_gap_us * self.scale,
                });
            }
        }
    }
}

impl Iterator for Runs<'_> {
    type Item = Run;

    fn next(&mut self) -> Option<Run> {
        let mut cur = match self.pending.take() {
            Some(r) => r,
            None => self.raw()?,
        };
        loop {
            match self.raw() {
                Some(r) if r.level == cur.level || r.us <= 0.0 => cur.us += r.us,
                Some(r) if cur.us <= 0.0 => cur = r,
                Some(r) => {
                    self.pending = Some(r);
                    return Some(cur);
                }
                None => return Some(cur),
            }
        }
    }
}

/// Weight `0 ... 1` of the level after an edge at the distance `u` seconds from it (raised
/// cosine over `RISE_US`, 1/2 at the edge).
fn ramp(u: f64) -> f64 {
    let r = RISE_US * 1e-6;
    if u <= -r / 2.0 {
        0.0
    } else if u >= r / 2.0 {
        1.0
    } else {
        0.5 - 0.5 * libm::cos(PI * (u / r + 0.5))
    }
}

/// Fractional part `x - floor(x)`.
fn frac(x: f64) -> f64 {
    x - libm::floor(x)
}

/// Evaluates one transmission at increasing times.
#[derive(Debug, Clone)]
struct Cursor<'a> {
    runs: Runs<'a>,
    amplitude: f64,
    /// Carrier offset in Hz (clock error included).
    freq_hz: f64,
    deviation_hz: f64,
    phase0: f64,
    /// Start of the transmission in seconds of the evaluation time.
    start_s: f64,
    /// Time after which the transmission contributes nothing (end plus half a ramp).
    end_s: f64,
    prev: Level,
    cur: Level,
    /// Start and end of the current run relative to the start, in seconds.
    cur_start: f64,
    cur_end: f64,
    next: Option<Run>,
    /// Integral of the FSK sign up to the start of the current run, in seconds.
    d_start: f64,
}

impl<'a> Cursor<'a> {
    fn new(t: Transmission<'a>, freq_hz: f64, phase0: f64, start_s: f64) -> Self {
        let mut runs = t.runs();
        let next = runs.next();
        Self {
            runs,
            amplitude: libm::pow(10.0, t.level_db / 20.0),
            freq_hz,
            deviation_hz: t.deviation_hz(),
            phase0,
            start_s,
            end_s: start_s + (t.duration_us() + RISE_US) * 1e-6,
            prev: Level::Off,
            cur: Level::Off,
            cur_start: f64::NEG_INFINITY,
            cur_end: 0.0,
            next,
            d_start: 0.0,
        }
    }

    /// The signal at the evaluation time `t` (not before the previous call's).
    fn value(&mut self, t: f64) -> Complex64 {
        let r = RISE_US * 1e-6;
        let local = t - self.start_s;
        if local < -r || t > self.end_s {
            return Complex64::new(0.0, 0.0);
        }
        while local >= self.cur_end {
            if self.cur_start.is_finite() {
                self.d_start += self.cur.sign() * (self.cur_end - self.cur_start);
            }
            self.prev = self.cur;
            self.cur_start = self.cur_end;
            if let Some(run) = self.next.take() {
                self.cur = run.level;
                self.cur_end = self.cur_start + run.us * 1e-6;
                self.next = self.runs.next();
            } else {
                self.cur = Level::Off;
                self.cur_end = f64::INFINITY;
            }
        }
        let a_cur = self.cur.amplitude();
        let mut a = a_cur;
        if local - self.cur_start < r / 2.0 {
            a += (self.prev.amplitude() - a_cur) * (1.0 - ramp(local - self.cur_start));
        }
        if self.cur_end - local < r / 2.0 {
            let a_next = self.next.map_or(0.0, |n| n.level.amplitude());
            a += (a_next - a_cur) * ramp(local - self.cur_end);
        }
        if a <= 0.0 {
            return Complex64::new(0.0, 0.0);
        }
        let d = if self.cur_start.is_finite() {
            self.d_start + self.cur.sign() * (local - self.cur_start)
        } else {
            0.0
        };
        let cycles = frac(self.freq_hz * local) + frac(self.deviation_hz * d) + self.phase0;
        let arg = 2.0 * PI * frac(cycles);
        Complex64::new(libm::cos(arg), libm::sin(arg)) * (self.amplitude * a)
    }
}

/// A [`Transmission`] at a start time of a [`Scene`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placed<'a> {
    /// Start (time 0 of the transmission) in seconds of true time from sample 0.
    pub start_s: f64,
    /// The transmission.
    pub transmission: Transmission<'a>,
}

/// An interfering signal of a [`Scene`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Interferer {
    /// An unmodulated carrier for the whole scene.
    Carrier {
        /// Frequency relative to the centre of the band in Hz.
        freq_offset_hz: f64,
        /// Level in dBFS.
        level_db: f64,
    },
    /// Complex Gaussian noise over the whole band for a short time.
    WidebandBurst {
        /// Start in seconds of true time.
        start_s: f64,
        /// Duration in seconds.
        duration_s: f64,
        /// Total power in dBFS.
        level_db: f64,
    },
}

/// A band with transmissions, noise, a clock error and interferers (see the
/// [module documentation](self)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scene<'a> {
    /// Nominal sample rate in Hz.
    pub fs: f64,
    /// RF centre of the band in Hz (only for the oscillator error of `clock_error_ppm`).
    pub center_hz: f64,
    /// Clock error of the receiver in ppm (sample clock and local oscillator).
    pub clock_error_ppm: f64,
    /// Total power of the complex Gaussian band noise in dBFS; `f64::NEG_INFINITY` for none.
    pub noise_db: f64,
    /// The transmissions, at most [`MAX_SCENE_TRANSMISSIONS`].
    pub transmissions: &'a [Placed<'a>],
    /// The interferers, at most [`MAX_SCENE_INTERFERERS`].
    pub interferers: &'a [Interferer],
    /// Seed of the random streams.
    pub seed: u64,
}

impl<'a> Scene<'a> {
    fn e(&self) -> f64 {
        self.clock_error_ppm * 1e-6
    }

    /// The frequency the receiver measures for a carrier at `freq_offset_hz` (oscillator error
    /// and sample-clock error applied).
    pub fn apparent_offset_hz(&self, freq_offset_hz: f64) -> f64 {
        (freq_offset_hz - self.center_hz * self.e()) / (1.0 + self.e())
    }

    /// The duration the receiver measures for `us` µs of true time.
    pub fn apparent_us(&self, us: f64) -> f64 {
        us * (1.0 + self.e())
    }

    /// The noise power in dBFS within `bandwidth_hz` of the band.
    pub fn noise_db_in(&self, bandwidth_hz: f64) -> f64 {
        self.noise_db + 10.0 * libm::log10(bandwidth_hz / self.fs)
    }

    /// Checks the scene and returns a renderer at sample 0.
    ///
    /// # Errors
    /// [`IsmError::TooMany`] for too many transmissions or interferers; the errors of
    /// [`Transmission::validate`]; [`IsmError::InvalidParameter`] for a sample rate that is not
    /// positive and finite, a centre, clock error, start or duration that is not finite (or a
    /// clock error at or below -1e6 ppm, a negative burst duration), a level of an interferer
    /// that is not finite, a `noise_db` of NaN or `+inf` and an apparent frequency (offset plus
    /// deviation) at or beyond `fs / 2`.
    pub fn renderer(&self) -> Result<SceneRenderer<'a>, IsmError> {
        if self.transmissions.len() > MAX_SCENE_TRANSMISSIONS
            || self.interferers.len() > MAX_SCENE_INTERFERERS
        {
            return Err(IsmError::TooMany);
        }
        positive(self.fs, "fs")?;
        if !self.center_hz.is_finite() {
            return Err(IsmError::InvalidParameter("center_hz"));
        }
        if !(self.clock_error_ppm.is_finite() && self.clock_error_ppm > -1e6) {
            return Err(IsmError::InvalidParameter("clock_error_ppm"));
        }
        if self.noise_db.is_nan() || self.noise_db == f64::INFINITY {
            return Err(IsmError::InvalidParameter("noise_db"));
        }
        let lo_hz = -self.center_hz * self.e();
        let in_band = |hz: f64| (hz + lo_hz).abs() / (1.0 + self.e()) < self.fs / 2.0;
        let mut cursors = heapless::Vec::new();
        for (i, p) in self.transmissions.iter().enumerate() {
            let t = p.transmission;
            t.validate()?;
            if !p.start_s.is_finite() {
                return Err(IsmError::InvalidParameter("start_s"));
            }
            let dev = t.deviation_hz();
            if !(in_band(t.freq_offset_hz + dev) && in_band(t.freq_offset_hz - dev)) {
                return Err(IsmError::InvalidParameter("freq_offset_hz"));
            }
            let phase0 = Pcg32::new(self.seed, 0x1_0000 + i as u64).next_f64();
            let c = Cursor::new(t, t.freq_offset_hz + lo_hz, phase0, p.start_s);
            if cursors.push(c).is_err() {
                return Err(IsmError::TooMany);
            }
        }
        let mut interferers = heapless::Vec::new();
        for (i, f) in self.interferers.iter().enumerate() {
            let mut rng = Pcg32::new(self.seed, 1 + i as u64);
            let state = match *f {
                Interferer::Carrier {
                    freq_offset_hz,
                    level_db,
                } => {
                    if !freq_offset_hz.is_finite() || !in_band(freq_offset_hz) {
                        return Err(IsmError::InvalidParameter("freq_offset_hz"));
                    }
                    if !level_db.is_finite() {
                        return Err(IsmError::InvalidParameter("level_db"));
                    }
                    InterfererState::Carrier {
                        freq_hz: freq_offset_hz + lo_hz,
                        amplitude: libm::pow(10.0, level_db / 20.0),
                        phase0: rng.next_f64(),
                    }
                }
                Interferer::WidebandBurst {
                    start_s,
                    duration_s,
                    level_db,
                } => {
                    if !start_s.is_finite() {
                        return Err(IsmError::InvalidParameter("start_s"));
                    }
                    if !(duration_s.is_finite() && duration_s >= 0.0) {
                        return Err(IsmError::InvalidParameter("duration_s"));
                    }
                    if !level_db.is_finite() {
                        return Err(IsmError::InvalidParameter("level_db"));
                    }
                    InterfererState::Burst {
                        start_s,
                        end_s: start_s + duration_s,
                        sigma: libm::sqrt(libm::pow(10.0, level_db / 10.0) / 2.0),
                        rng,
                    }
                }
            };
            if interferers.push(state).is_err() {
                return Err(IsmError::TooMany);
            }
        }
        Ok(SceneRenderer {
            cursors,
            interferers,
            noise_sigma: libm::sqrt(libm::pow(10.0, self.noise_db / 10.0) / 2.0),
            noise: Pcg32::new(self.seed, 0),
            true_fs: self.fs * (1.0 + self.e()),
            n: 0,
        })
    }
}

#[derive(Debug, Clone)]
enum InterfererState {
    Carrier {
        freq_hz: f64,
        amplitude: f64,
        phase0: f64,
    },
    Burst {
        start_s: f64,
        end_s: f64,
        sigma: f64,
        rng: Pcg32,
    },
}

/// Writes the band of a [`Scene`] piece by piece (see the [module documentation](self)).
#[derive(Debug, Clone)]
pub struct SceneRenderer<'a> {
    cursors: heapless::Vec<Cursor<'a>, MAX_SCENE_TRANSMISSIONS>,
    interferers: heapless::Vec<InterfererState, MAX_SCENE_INTERFERERS>,
    noise_sigma: f64,
    noise: Pcg32,
    /// The true sample rate `fs (1 + e)`.
    true_fs: f64,
    n: u64,
}

impl SceneRenderer<'_> {
    /// Writes the next `out.len()` samples.
    pub fn render(&mut self, out: &mut [Complex32]) {
        for o in out.iter_mut() {
            let t = self.n as f64 / self.true_fs;
            let mut sum = Complex64::new(0.0, 0.0);
            for c in &mut self.cursors {
                sum += c.value(t);
            }
            for f in &mut self.interferers {
                match f {
                    InterfererState::Carrier {
                        freq_hz,
                        amplitude,
                        phase0,
                    } => {
                        let arg = 2.0 * PI * frac(frac(*freq_hz * t) + *phase0);
                        sum += Complex64::new(libm::cos(arg), libm::sin(arg)) * *amplitude;
                    }
                    InterfererState::Burst {
                        start_s,
                        end_s,
                        sigma,
                        rng,
                    } => {
                        if t >= *start_s && t < *end_s {
                            sum += Complex64::new(rng.normal(), rng.normal()) * *sigma;
                        }
                    }
                }
            }
            if self.noise_sigma > 0.0 {
                sum += Complex64::new(self.noise.normal(), self.noise.normal()) * self.noise_sigma;
            }
            *o = Complex32::new(sum.re as f32, sum.im as f32);
            self.n = self.n.wrapping_add(1);
        }
    }

    /// Index of the next sample to render.
    #[must_use]
    pub fn position(&self) -> u64 {
        self.n
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::many_single_char_names
)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec;
    use std::vec::Vec;

    fn abs(c: Complex32) -> f32 {
        libm::sqrtf(c.norm_sqr())
    }

    fn zeros(n: usize) -> Vec<Complex32> {
        vec![Complex32::new(0.0, 0.0); n]
    }

    const BITS: [bool; 8] = [true, false, false, true, true, true, false, true];

    fn tx(coding: Coding) -> Transmission<'static> {
        Transmission {
            coding,
            bits: &BITS,
            repeats: 1,
            repeat_gap_us: 0.0,
            freq_offset_hz: 0.0,
            level_db: 0.0,
            timing_error: 0.0,
        }
    }

    fn levels(t: &Transmission<'_>) -> Vec<(Level, f64)> {
        t.runs().map(|r| (r.level, r.us)).collect()
    }

    #[test]
    fn the_codings_give_their_runs() {
        use Level::{High, Low, Off, On};
        let pwm = tx(Coding::Pwm {
            short_us: 100.0,
            long_us: 300.0,
            gap_us: 200.0,
        });
        let mut want = Vec::new();
        for &b in &BITS {
            want.push((On, if b { 100.0 } else { 300.0 }));
            want.push((Off, 200.0));
        }
        want.pop();
        assert_eq!(levels(&pwm), want);

        let ppm = tx(Coding::Ppm {
            pulse_us: 50.0,
            short_gap_us: 100.0,
            long_gap_us: 250.0,
        });
        let mut want = Vec::new();
        for &b in &BITS {
            want.push((On, 50.0));
            want.push((Off, if b { 250.0 } else { 100.0 }));
        }
        want.push((On, 50.0));
        assert_eq!(levels(&ppm), want);

        // 1 0 0 1 1 1 0 1 -> on off | off on | off on | on off | on off | on off | off on | on off
        let man = tx(Coding::Manchester { half_bit_us: 10.0 });
        assert_eq!(
            levels(&man),
            [
                (On, 10.0),
                (Off, 20.0),
                (On, 10.0),
                (Off, 10.0),
                (On, 20.0),
                (Off, 10.0),
                (On, 10.0),
                (Off, 10.0),
                (On, 10.0),
                (Off, 20.0),
                (On, 20.0),
            ]
        );
        let fsk = tx(Coding::Fsk {
            bit_us: 20.0,
            deviation_hz: 50e3,
        });
        assert_eq!(
            levels(&fsk),
            [
                (High, 20.0),
                (Low, 40.0),
                (High, 60.0),
                (Low, 20.0),
                (High, 20.0),
            ]
        );
        assert_eq!(fsk.duration_us(), 160.0);
    }

    #[test]
    fn repeats_and_timing_error_scale_and_separate_the_copies() {
        let mut t = tx(Coding::Manchester { half_bit_us: 10.0 });
        let zero_first = [false, true];
        t.bits = &zero_first;
        t.repeats = 3;
        t.repeat_gap_us = 1000.0;
        t.timing_error = 0.1;
        // 0 1 -> off on on off: the copy is off 10, on 20; the trailing off is dropped.
        let r = levels(&t);
        let s = 1.1;
        let want = [
            (Level::Off, 10.0 * s),
            (Level::On, 20.0 * s),
            (Level::Off, (1000.0 + 10.0) * s),
            (Level::On, 20.0 * s),
            (Level::Off, (1000.0 + 10.0) * s),
            (Level::On, 20.0 * s),
        ];
        assert_eq!(r.len(), want.len());
        for (a, b) in r.iter().zip(&want) {
            assert_eq!(a.0, b.0);
            assert!((a.1 - b.1).abs() < 1e-9, "{a:?} {b:?}");
        }
        // FSK copies are separated by carrier off.
        let mut f = tx(Coding::Fsk {
            bit_us: 10.0,
            deviation_hz: 1e3,
        });
        f.repeats = 2;
        f.repeat_gap_us = 500.0;
        let r = levels(&f);
        assert_eq!(r.iter().filter(|x| x.0 == Level::Off).count(), 1);
        assert_eq!(f.duration_us(), 2.0 * 80.0 + 500.0);
    }

    /// The times where `|x|` crosses half the amplitude, by linear interpolation, in µs.
    fn half_crossings(x: &[Complex32], fs: f64, amp: f64) -> Vec<f64> {
        let mut out = Vec::new();
        for n in 1..x.len() {
            let (a, b) = (
                f64::from(abs(x[n - 1])) - amp / 2.0,
                f64::from(abs(x[n])) - amp / 2.0,
            );
            if (a < 0.0) != (b < 0.0) {
                let frac = a / (a - b);
                out.push(((n - 1) as f64 + frac) / fs * 1e6);
            }
        }
        out
    }

    #[test]
    fn rendered_ook_edges_are_at_the_nominal_times() {
        let fs = 2e6;
        for coding in [
            Coding::Pwm {
                short_us: 123.4,
                long_us: 371.9,
                gap_us: 245.3,
            },
            Coding::Ppm {
                pulse_us: 61.0,
                short_gap_us: 97.7,
                long_gap_us: 203.1,
            },
            Coding::Manchester { half_bit_us: 31.25 },
        ] {
            let mut t = tx(coding);
            t.freq_offset_hz = 312_500.0;
            t.level_db = -6.0;
            t.repeats = 2;
            t.repeat_gap_us = 777.0;
            t.timing_error = -0.03;
            // Delay by 10 µs so that the first rise lies inside the buffer.
            let pad = 20;
            let mut x = zeros(pad + (t.duration_us() * 1e-6 * fs) as usize + 40);
            t.render(fs, &mut x[pad..]).unwrap();
            let amp = libm::pow(10.0, -6.0 / 20.0);
            let edges: Vec<f64> = half_crossings(&x, fs, amp)
                .into_iter()
                .map(|e| e - pad as f64 / fs * 1e6)
                .collect();
            let mut want = Vec::new();
            let mut time = 0.0;
            let mut prev = Level::Off;
            for r in t.runs() {
                if r.level != prev {
                    want.push(time);
                }
                prev = r.level;
                time += r.us;
            }
            want.push(time);
            assert_eq!(edges.len(), want.len(), "{coding:?}");
            for (e, w) in edges.iter().zip(&want) {
                // The raised cosine is not linear between two samples: 0.05 µs.
                assert!((e - w).abs() < 0.05, "{coding:?}: edge {e} want {w}");
            }
            // Carrier on is exactly the level; the carrier is at its frequency.
            let mid = pad + 20 + (want[0] * 1e-6 * fs) as usize;
            assert!((f64::from(abs(x[mid])) - amp).abs() < 1e-6);
            let d = x[mid + 1] * x[mid].conj();
            let f = f64::from(d.im.atan2(d.re)) / (2.0 * PI) * fs;
            assert!((f - 312_500.0).abs() < 50.0, "{f}");
        }
    }

    #[test]
    fn fsk_has_a_continuous_phase_and_the_two_frequencies() {
        let fs = 1e6;
        let dev = 40e3;
        let mut t = tx(Coding::Fsk {
            bit_us: 52.3,
            deviation_hz: dev,
        });
        t.freq_offset_hz = -100e3;
        t.level_db = -10.0;
        let mut x = zeros((t.duration_us() * 1e-6 * fs) as usize + 10);
        t.render(fs, &mut x).unwrap();
        let amp = libm::pow(10.0, -10.0 / 20.0);
        let runs: Vec<Run> = t.runs().collect();
        // Every sample at least a ramp away from the ends has the full amplitude, and the phase
        // step between two samples is the frequency of the bit, or between the two frequencies
        // across a transition (never a jump).
        let mut bounds = Vec::new();
        let mut time = 0.0;
        for r in &runs {
            bounds.push((time, time + r.us, r.level));
            time += r.us;
        }
        let end = time;
        for n in 3..(end * 1e-6 * fs) as usize - 3 {
            assert!((f64::from(abs(x[n])) - amp).abs() < 1e-5, "n = {n}");
            let d = x[n + 1] * x[n].conj();
            let f = f64::from(d.im.atan2(d.re)) / (2.0 * PI) * fs;
            let (t0, t1) = (n as f64 / fs * 1e6, (n + 1) as f64 / fs * 1e6);
            let lo = -100e3 - dev - 1.0;
            let hi = -100e3 + dev + 1.0;
            assert!(f > lo && f < hi, "n = {n}: {f}");
            if let Some(&(_, _, level)) = bounds.iter().find(|b| b.0 <= t0 && t1 <= b.1) {
                let want = -100e3 + if level == Level::High { dev } else { -dev };
                assert!((f - want).abs() < 2.0, "n = {n}: {f} want {want}");
            }
        }
        // Across a transition the phase step is the time-weighted mean of the two frequencies.
        let edge = runs[0].us; // 52.3 µs: High to Low between samples 52 and 53
        let n = (edge * 1e-6 * fs) as usize;
        let d = x[n + 1] * x[n].conj();
        let f = f64::from(d.im.atan2(d.re)) / (2.0 * PI) * fs;
        let w_high = edge - n as f64; // 0.3 of the sample at the upper frequency
        let want = -100e3 + dev * (w_high - (1.0 - w_high));
        assert!((f - want).abs() < 2.0, "{f} want {want}");
    }

    fn scene_parts() -> ([Placed<'static>; 2], [Interferer; 2]) {
        let mut a = tx(Coding::Pwm {
            short_us: 100.0,
            long_us: 300.0,
            gap_us: 200.0,
        });
        a.freq_offset_hz = -400e3;
        a.level_db = -20.0;
        let mut b = tx(Coding::Fsk {
            bit_us: 50.0,
            deviation_hz: 30e3,
        });
        b.freq_offset_hz = 300e3;
        b.level_db = -25.0;
        (
            [
                Placed {
                    start_s: 100e-6,
                    transmission: a,
                },
                Placed {
                    start_s: 250e-6,
                    transmission: b,
                },
            ],
            [
                Interferer::Carrier {
                    freq_offset_hz: 123e3,
                    level_db: -30.0,
                },
                Interferer::WidebandBurst {
                    start_s: 1e-3,
                    duration_s: 200e-6,
                    level_db: -15.0,
                },
            ],
        )
    }

    fn scene<'a>(p: &'a [Placed<'a>], i: &'a [Interferer]) -> Scene<'a> {
        Scene {
            fs: 2e6,
            center_hz: 433.92e6,
            clock_error_ppm: 20.0,
            noise_db: -40.0,
            transmissions: p,
            interferers: i,
            seed: 11,
        }
    }

    #[test]
    fn scenes_are_deterministic_in_any_chunking_and_follow_the_seed() {
        let (p, i) = scene_parts();
        let s = scene(&p, &i);
        let mut whole = zeros(4000);
        s.renderer().unwrap().render(&mut whole);
        let mut again = zeros(4000);
        s.renderer().unwrap().render(&mut again);
        assert_eq!(whole, again);
        let mut chunked = zeros(4000);
        let mut r = s.renderer().unwrap();
        let mut rng = Pcg32::new(5, 5);
        let mut at = 0;
        while at < chunked.len() {
            let n = (1 + rng.below(300) as usize).min(chunked.len() - at);
            r.render(&mut chunked[at..at + n]);
            at += n;
        }
        assert_eq!(whole, chunked);
        assert_eq!(r.position(), 4000);
        let mut other = s;
        other.seed = 12;
        let mut x = zeros(4000);
        other.renderer().unwrap().render(&mut x);
        assert_ne!(whole, x);
    }

    #[test]
    fn the_scene_applies_the_clock_error_and_adds_its_parts() {
        let (p, _) = scene_parts();
        let mut s = scene(&p[..1], &[]);
        s.noise_db = f64::NEG_INFINITY;
        let mut x = zeros(4000);
        s.renderer().unwrap().render(&mut x);
        // The apparent frequency of the carrier: -400 kHz shifted by -433.92 MHz * 20 ppm.
        let want = s.apparent_offset_hz(-400e3);
        assert!((want - (-400e3 - 8678.4) / (1.0 + 20e-6)).abs() < 1e-6);
        let n = 260; // inside the first pulse (100 ... 200 µs)
        let d = x[n + 1] * x[n].conj();
        let f = f64::from(d.im.atan2(d.re)) / (2.0 * PI) * s.fs;
        assert!((f - want).abs() < 20.0, "{f} want {want}");
        // The edges move by the clock error: start 100 µs, first pulse 100 µs.
        let edges = half_crossings(&x, s.fs, 0.1);
        assert!((edges[0] - s.apparent_us(100.0)).abs() < 0.05, "{edges:?}");
        assert!((edges[1] - s.apparent_us(200.0)).abs() < 0.05, "{edges:?}");
        assert!((s.apparent_us(1e6) - 1e6 - 20.0).abs() < 1e-6);

        // Noise and interferers: the noise power, the carrier and the burst.
        let (_, i) = scene_parts();
        let mut s = scene(&[], &i[1..]);
        s.noise_db = -40.0;
        let mut x = zeros(4000);
        s.renderer().unwrap().render(&mut x);
        let power = |v: &[Complex32]| {
            10.0 * libm::log10(
                v.iter().map(|c| f64::from(c.norm_sqr())).sum::<f64>() / v.len() as f64,
            )
        };
        assert!(
            (power(&x[..1900]) + 40.0).abs() < 0.3,
            "{}",
            power(&x[..1900])
        );
        // The burst: 1.0 ... 1.2 ms of true time, samples 2000 ... 2400.
        assert!((power(&x[2010..2390]) + 15.0).abs() < 0.6);
        assert!((power(&x[2410..]) + 40.0).abs() < 0.3);
        assert!((s.noise_db_in(200e3) + 50.0).abs() < 1e-9);
        let mut s = scene(&[], &i[..1]);
        s.noise_db = f64::NEG_INFINITY;
        let mut x = zeros(100);
        s.renderer().unwrap().render(&mut x);
        assert!(
            x.iter()
                .all(|&c| (f64::from(abs(c)) - 10f64.powf(-1.5)).abs() < 1e-6)
        );
    }

    #[test]
    fn invalid_descriptions_are_rejected() {
        let good = tx(Coding::Pwm {
            short_us: 1.0,
            long_us: 2.0,
            gap_us: 1.0,
        });
        assert_eq!(good.validate(), Ok(()));
        let mut t = good;
        t.bits = &[];
        assert_eq!(t.validate(), Err(IsmError::InvalidParameter("bits")));
        let mut t = good;
        t.repeats = 0;
        assert_eq!(t.validate(), Err(IsmError::InvalidParameter("repeats")));
        let mut t = good;
        t.coding = Coding::Fsk {
            bit_us: 10.0,
            deviation_hz: 0.0,
        };
        assert_eq!(
            t.validate(),
            Err(IsmError::InvalidParameter("deviation_hz"))
        );
        let mut t = good;
        t.coding = Coding::Manchester {
            half_bit_us: f64::NAN,
        };
        assert_eq!(t.validate(), Err(IsmError::InvalidParameter("half_bit_us")));
        let mut t = good;
        t.timing_error = -1.0;
        assert_eq!(
            t.validate(),
            Err(IsmError::InvalidParameter("timing_error"))
        );
        let mut t = good;
        t.repeat_gap_us = -1.0;
        assert_eq!(
            t.validate(),
            Err(IsmError::InvalidParameter("repeat_gap_us"))
        );
        let mut t = good;
        t.freq_offset_hz = 1.1e6;
        let mut x = zeros(4);
        assert_eq!(
            t.render(2e6, &mut x),
            Err(IsmError::InvalidParameter("freq_offset_hz"))
        );
        assert_eq!(
            good.render(0.0, &mut x),
            Err(IsmError::InvalidParameter("fs"))
        );
        let p = [Placed {
            start_s: 0.0,
            transmission: t,
        }];
        assert_eq!(
            scene(&p, &[]).renderer().err(),
            Some(IsmError::InvalidParameter("freq_offset_hz"))
        );
        let many = vec![
            Placed {
                start_s: 0.0,
                transmission: good,
            };
            MAX_SCENE_TRANSMISSIONS + 1
        ];
        assert_eq!(scene(&many, &[]).renderer().err(), Some(IsmError::TooMany));
        assert!(
            scene(&many[..MAX_SCENE_TRANSMISSIONS], &[])
                .renderer()
                .is_ok()
        );
        let mut s = scene(&[], &[]);
        s.noise_db = f64::NAN;
        assert_eq!(
            s.renderer().err(),
            Some(IsmError::InvalidParameter("noise_db"))
        );
        let bad = [Interferer::WidebandBurst {
            start_s: 0.0,
            duration_s: -1.0,
            level_db: 0.0,
        }];
        assert_eq!(
            scene(&[], &bad).renderer().err(),
            Some(IsmError::InvalidParameter("duration_s"))
        );
    }
}
