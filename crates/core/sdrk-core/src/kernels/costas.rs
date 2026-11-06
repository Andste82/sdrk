//! Carrier recovery for phase-shift keying: a Costas loop of order 2 (BPSK), 4 (QPSK) or 8
//! (8PSK) that runs on one sample per symbol (after `symbol_sync`), removes the carrier phase
//! and frequency offset and reports whether it is locked.
//!
//! Per symbol the input is rotated by the phase of a numerically controlled oscillator
//! (`y = x exp(-j phi)`, sine and cosine from the tables of `rotate`), a decision-directed phase
//! detector measures the remaining phase error, and a proportional-integral loop filter corrects
//! the frequency integrator and the phase. A lock detector smooths `cos(order arg y)` (for
//! QPSK its negative, so that the points at 45 degrees give +1, D601; for 8PSK the metric of
//! M40.1 §1, `cos(2 err)` against the nearest point). The arithmetic is fixed (float32 model,
//! see [`Costas`] and `docs/kernels.md`), so `Fallback` and `Reference` are bit-identical.

use core::f64::consts::TAU;

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::num::flush_denormal;
use crate::timing::{Delay, Timing};
use crate::types::C32;

/// The smoothing factor of the lock detector, `l = l + LOCK_ALPHA (c - l)`.
pub(crate) const LOCK_ALPHA: f32 = 0.01;
/// The lock detector turns on above this level.
pub(crate) const LOCK_ON: f32 = 0.5;
/// The lock detector turns off below this level.
pub(crate) const LOCK_OFF: f32 = 0.3;
/// `cos(pi / 8)` rounded once to `f32`: the larger part of the 8PSK points.
#[allow(
    clippy::excessive_precision,
    reason = "the literal is the shortest decimal that reads back as the f32 of cos(pi / 8)"
)]
pub(crate) const C8: f32 = 0.923_879_5;
/// `sin(pi / 8)` rounded once to `f32`: the smaller part of the 8PSK points.
#[allow(
    clippy::excessive_precision,
    reason = "the literal is the shortest decimal that reads back as the f32 of sin(pi / 8)"
)]
pub(crate) const S8: f32 = 0.382_683_43;

/// The loop constants of [`CostasState`], computed in f64 by [`CostasState::new`] and kept as
/// the `f32` values the loop uses.
///
/// ```
/// use sdrk_core::kernels::costas::CostasState;
/// let st = CostasState::new(4, 0.02, 0.7071, 0.05)?;
/// let g = st.gains();
/// assert!(g.kp > 0.0 && g.ki > 0.0 && g.ki < g.kp);
/// assert_eq!(g.max_offset, 0.05f32);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostasGains {
    /// Proportional gain in cycles per symbol per unit of the detector output,
    /// `((4 damping theta) / d) / (2 pi)`, rounded to `f32`.
    pub kp: f32,
    /// Integral gain `((4 theta^2) / d) / (2 pi)`, rounded to `f32`.
    pub ki: f32,
    /// The limit of the frequency integrator in cycles per symbol, `max_offset` as `f32`,
    /// 0 when that is below 1e-30 (flushed, R6.3).
    pub max_offset: f32,
}

/// State of the Costas loop: the NCO phase (`u32`, a full turn is `2^32`), the frequency
/// integrator (`f32`, cycles per symbol), the loop constants, the order and the lock detector.
///
/// ```
/// use sdrk_core::{kernels::costas::CostasState, Ratio, Timing};
/// let st = CostasState::new(2, 0.02, 0.7071, 0.05)?;
/// assert_eq!((st.order(), st.phase(), st.frequency(), st.locked()), (2, 0, 0.0, false));
/// assert_eq!(st.ratio(), Ratio::ONE);
/// // Order 8 is 8PSK; order 3 and a loop bandwidth above 0.2 are rejected.
/// assert_eq!(CostasState::new(8, 0.02, 0.7071, 0.05)?.order(), 8);
/// assert!(CostasState::new(3, 0.02, 0.7071, 0.05).is_err());
/// assert!(CostasState::new(4, 0.3, 0.7071, 0.05).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostasState {
    pub(crate) order: u32,
    pub(crate) loop_bw: f64,
    pub(crate) damping: f64,
    pub(crate) max_offset: f64,
    pub(crate) kp: f32,
    pub(crate) ki: f32,
    pub(crate) max: f32,
    /// NCO phase before the next symbol, in units of `2^-32` turn.
    pub(crate) phase: u32,
    /// Frequency integrator in cycles per symbol.
    pub(crate) freq: f32,
    /// Smoothed lock metric.
    pub(crate) level: f32,
    pub(crate) locked: bool,
}

fn invalid(msg: &'static str) -> Error {
    Error::InvalidParameter(msg)
}

impl CostasState {
    /// Creates a loop with phase 0, a zero frequency integrator and the lock detector at 0
    /// (unlocked).
    ///
    /// `order` is 2 (BPSK), 4 (QPSK) or 8 (8PSK, M40.1); any other order gives
    /// `InvalidParameter`. `loop_bw` (0.0001 to 0.2) is the loop bandwidth normalized to the
    /// symbol rate, `damping` 0.1 to 4 (1/√2 usual), `max_offset` (0 to 0.25) the limit of
    /// the frequency integrator in cycles per symbol (as `f32`; a value below 1e-30 is flushed
    /// to 0, so the integrator never holds a subnormal). The loop constants, in f64 (the
    /// formulas of `symbol_sync` with `ted_gain` 1, without the factor `sps` and divided by
    /// `2 pi`): `theta = loop_bw / (damping + 1 / (4 damping))`,
    /// `d = (1 + (2 damping) theta) + theta theta`, `kp = (((4 damping) theta) / d) / (2 pi)`,
    /// `ki = ((4 (theta theta)) / d) / (2 pi)`, then rounded to `f32`. Anything else, including
    /// NaN, gives `InvalidParameter`.
    ///
    /// ```
    /// use sdrk_core::kernels::costas::CostasState;
    /// assert!(CostasState::new(2, 0.0001, 0.1, 0.0).is_ok());
    /// assert!(CostasState::new(4, 0.2, 4.0, 0.25).is_ok());
    /// assert!(CostasState::new(8, 0.02, 0.7071, 0.05).is_ok());
    /// assert!(CostasState::new(3, 0.02, 0.7071, 0.05).is_err());
    /// assert!(CostasState::new(4, 0.02, 0.7071, 0.3).is_err());
    /// assert!(CostasState::new(4, f64::NAN, 0.7071, 0.05).is_err());
    /// ```
    pub fn new(order: u32, loop_bw: f64, damping: f64, max_offset: f64) -> Result<Self> {
        if !matches!(order, 2 | 4 | 8) {
            return Err(invalid("costas order must be 2, 4 or 8"));
        }
        if !(loop_bw.is_finite() && (0.0001..=0.2).contains(&loop_bw)) {
            return Err(invalid(
                "costas loop bandwidth must be between 0.0001 and 0.2",
            ));
        }
        if !(damping.is_finite() && (0.1..=4.0).contains(&damping)) {
            return Err(invalid("costas damping must be between 0.1 and 4"));
        }
        if !(max_offset.is_finite() && (0.0..=0.25).contains(&max_offset)) {
            return Err(invalid(
                "costas maximum offset must be between 0 and 0.25 cycles per symbol",
            ));
        }
        let theta = loop_bw / (damping + 1.0 / (4.0 * damping));
        let d = (1.0 + (2.0 * damping) * theta) + theta * theta;
        let kp = (((4.0 * damping) * theta) / d) / TAU;
        let ki = ((4.0 * (theta * theta)) / d) / TAU;
        let (kp, ki) = (kp as f32, ki as f32);
        if !(kp.is_finite() && ki.is_finite()) {
            return Err(invalid("costas loop gains must be finite"));
        }
        Ok(Self {
            order,
            loop_bw,
            damping,
            max_offset,
            kp,
            ki,
            // Flushed (R6.3): a limit below 1e-30 would keep the integrator subnormal.
            max: flush_denormal(max_offset as f32),
            phase: 0,
            freq: 0.0,
            level: 0.0,
            locked: false,
        })
    }

    /// Returns to the state after construction: phase 0, zero frequency integrator, lock
    /// detector at 0 and unlocked (the parameters are kept).
    ///
    /// ```
    /// use sdrk_core::{kernels::costas::{Costas, CostasState}, Fallback, C32};
    /// let mut st = CostasState::new(2, 0.05, 0.7071, 0.05)?;
    /// let input = [C32::new(0.6, 0.8); 64];
    /// let mut out = [C32::default(); 64];
    /// Fallback.costas(&mut st, &input, &mut out)?;
    /// assert_ne!(st.phase(), 0);
    /// st.reset();
    /// assert_eq!(st, CostasState::new(2, 0.05, 0.7071, 0.05)?);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn reset(&mut self) {
        self.phase = 0;
        self.freq = 0.0;
        self.level = 0.0;
        self.locked = false;
    }

    /// The frequency integrator in cycles per symbol (the estimated carrier offset).
    pub fn frequency(&self) -> f64 {
        f64::from(self.freq)
    }

    /// Whether the lock detector reports lock (on above 0.5, off below 0.3). With order 8 the
    /// level stays near 0.9 on noise, so lock is reported on any input (open item I40.1-1).
    pub fn locked(&self) -> bool {
        self.locked
    }

    /// The smoothed lock metric `l` (about 1 when locked, about 0 on noise for orders 2 and 4;
    /// for order 8 about 0.9 on noise, the mean `sin(pi / 4) / (pi / 4)` of `cos(2 err)` with
    /// `err` uniform within an octant, open item I40.1-1).
    pub fn lock_level(&self) -> f32 {
        self.level
    }

    /// The NCO phase before the next symbol, in units of `2^-32` turn.
    pub fn phase(&self) -> u32 {
        self.phase
    }

    /// Sets the NCO phase (for tests that start the loop at a given phase).
    ///
    /// ```
    /// use sdrk_core::kernels::costas::CostasState;
    /// let mut st = CostasState::new(4, 0.02, 0.7071, 0.05)?;
    /// st.set_phase(0x4000_0000);
    /// assert_eq!(st.phase(), 0x4000_0000);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn set_phase(&mut self, phase: u32) {
        self.phase = phase;
    }

    /// The order: 2 (BPSK), 4 (QPSK) or 8 (8PSK).
    pub fn order(&self) -> u32 {
        self.order
    }

    /// The loop bandwidth, normalized to the symbol rate.
    pub fn loop_bw(&self) -> f64 {
        self.loop_bw
    }

    /// The damping of the loop.
    pub fn damping(&self) -> f64 {
        self.damping
    }

    /// The limit of the frequency integrator in cycles per symbol.
    pub fn max_offset(&self) -> f64 {
        self.max_offset
    }

    /// The loop constants.
    pub fn gains(&self) -> CostasGains {
        CostasGains {
            kp: self.kp,
            ki: self.ki,
            max_offset: self.max,
        }
    }
}

impl Timing for CostasState {
    /// One output per input symbol.
    fn ratio(&self) -> Ratio {
        Ratio::ONE
    }

    /// Zero; the output belongs to the input symbol at the same index.
    fn group_delay(&self) -> Delay {
        Delay::ZERO
    }
}

/// `sign(x)`: 1 above zero, -1 below, 0 for zero and NaN.
#[inline]
pub(crate) fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// The derotated symbol `y = x exp(-j phi)` from `c = cos phi`, `s = sin phi`, in the order
/// of the contract: `(x.re c + x.im s, x.im c - x.re s)`.
#[inline]
pub(crate) fn derotate(x: C32, c: f32, s: f32) -> C32 {
    C32::new(x.re * c + x.im * s, x.im * c - x.re * s)
}

/// The 8PSK point `(dr, di)` of the octant of `y`: point `k` (at `(2k + 1) pi / 8`) for the
/// half-open octant `[k pi / 4, (k + 1) pi / 4)`. The quadrant comes from the signs (`re > 0,
/// im >= 0`; `re <= 0, im > 0`; `re < 0, im <= 0`; `re >= 0, im < 0`), the half of it from
/// `|re| > |im|`; zero and NaN give point 0 (D641).
#[inline]
pub(crate) fn octant_point(y: C32) -> (f32, f32) {
    let wide = libm::fabsf(y.re) > libm::fabsf(y.im);
    let tall = libm::fabsf(y.im) > libm::fabsf(y.re);
    if y.re > 0.0 && y.im >= 0.0 {
        if wide { (C8, S8) } else { (S8, C8) }
    } else if y.re <= 0.0 && y.im > 0.0 {
        if tall { (-S8, C8) } else { (-C8, S8) }
    } else if y.re < 0.0 && y.im <= 0.0 {
        if wide { (-C8, -S8) } else { (-S8, -C8) }
    } else if y.re >= 0.0 && y.im < 0.0 {
        if tall { (S8, -C8) } else { (C8, -S8) }
    } else {
        (C8, S8)
    }
}

/// The phase detector of the contract; a non-finite value counts as 0 (D603).
#[inline]
pub(crate) fn detector(order: u32, y: C32) -> f32 {
    let e = match order {
        2 => y.re * y.im,
        4 => sign(y.re) * y.im - sign(y.im) * y.re,
        _ => {
            let (dr, di) = octant_point(y);
            y.im * dr - y.re * di
        }
    };
    if e.is_finite() { e } else { 0.0 }
}

/// The lock metric `c` of the contract (order 2: `cos(2 arg y)`, order 4: `-cos(4 arg y)`,
/// D601; order 8: `((r r) / p) 2 - 1 = cos(2 err)` with `r = re(y conj d)` against the octant
/// point `d`, D640); `None` when it is not finite (`y = 0`, NaN, overflow), and the detector
/// keeps its level (D603).
#[allow(clippy::many_single_char_names)]
#[inline]
pub(crate) fn lock_metric(order: u32, y: C32) -> Option<f32> {
    let a = y.re * y.re;
    let b = y.im * y.im;
    let p = a + b;
    let c = match order {
        2 => (a - b) / p,
        4 => {
            let d = a - b;
            ((4.0 * a) * b - d * d) / (p * p)
        }
        _ => {
            let (dr, di) = octant_point(y);
            let r = y.re * dr + y.im * di;
            ((r * r) / p) * 2.0 - 1.0
        }
    };
    c.is_finite().then_some(c)
}

/// Carrier recovery with a Costas loop, one sample per symbol.
///
/// ```
/// use sdrk_core::{kernels::costas::{Costas, CostasState}, Fallback, C32};
/// // QPSK symbols rotated by 30 degrees: the loop turns them back onto the diagonals.
/// let (s, c) = (0.5f32, 0.75f32.sqrt());
/// let input: Vec<C32> = (0..2000)
///     .map(|k| {
///         let p = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)][(k * 7 % 11) % 4];
///         let (re, im) = (p.0 * 0.7071f32, p.1 * 0.7071f32);
///         C32::new(re * c - im * s, re * s + im * c)
///     })
///     .collect();
/// let mut st = CostasState::new(4, 0.02, 0.7071, 0.05)?;
/// let mut out = vec![C32::default(); input.len()];
/// Fallback.costas(&mut st, &input, &mut out)?;
/// assert!(st.locked());
/// let y = out[1999];
/// assert!((y.re.abs() - 0.7071).abs() < 1e-3 && (y.im.abs() - 0.7071).abs() < 1e-3);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Costas {
    /// Per symbol, in f32 without FMA, with `phi` the NCO phase before the symbol:
    ///
    /// 1. `c = cos phi`, `s = sin phi` from the tables of `rotate` with its interpolation (the
    ///    cosine at `phi` and at `phi - 2^30`).
    /// 2. `y = (x.re c + x.im s, x.im c - x.re s)` (`x exp(-j phi)`); output `y`.
    /// 3. The detector `e`: order 2 `y.re y.im`; order 4 `sign(y.re) y.im - sign(y.im) y.re`
    ///    (`sign` is 0 for zero and NaN); order 8 `y.im dr - y.re di` with `(dr, di)` the 8PSK
    ///    point of the octant of `y` (`(+-cos(pi/8), +-sin(pi/8))` or swapped, as `f32`; zero
    ///    and NaN take point 0); a non-finite `e` counts as 0.
    /// 4. `freq = clamp(flush_denormal(freq + ki e), -max, max)`.
    /// 5. `phi += (floor((freq + kp e) 2^32 + 0.5) as i64) as u32` (the product and the
    ///    rounding in f64, wrapping).
    /// 6. Lock detector with `a = y.re y.re`, `b = y.im y.im`, `p = a + b`, `d = a - b`: the
    ///    metric `c = d / p` (order 2), `c = ((4 a) b - d d) / (p p)` (order 4) or
    ///    `c = ((r r) / p) 2 - 1` with `r = y.re dr + y.im di` (order 8); if `c` is
    ///    finite, `l = flush_denormal(l + 0.01 (c - l))`, locked when `l > 0.5`, unlocked when
    ///    `l < 0.3`; otherwise the detector is unchanged.
    ///
    /// The lengths must be equal, otherwise `LengthMismatch` and nothing changes.
    fn costas(&self, s: &mut CostasState, input: &[C32], output: &mut [C32]) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::{C8, C32, S8, octant_point};

    #[test]
    fn the_8psk_constants_are_the_f64_values_rounded_once() {
        let a = core::f64::consts::PI / 8.0;
        assert_eq!(C8.to_bits(), (libm::cos(a) as f32).to_bits());
        assert_eq!(S8.to_bits(), (libm::sin(a) as f32).to_bits());
    }

    #[test]
    fn octant_points_follow_the_half_open_rule() {
        let h = core::f32::consts::FRAC_1_SQRT_2;
        let cases = [
            // The boundaries k pi / 4 belong to octant k.
            (C32::new(1.0, 0.0), (C8, S8)),
            (C32::new(h, h), (S8, C8)),
            (C32::new(0.0, 1.0), (-S8, C8)),
            (C32::new(-h, h), (-C8, S8)),
            (C32::new(-1.0, 0.0), (-C8, -S8)),
            (C32::new(-h, -h), (-S8, -C8)),
            (C32::new(0.0, -1.0), (S8, -C8)),
            (C32::new(h, -h), (C8, -S8)),
            // Signed zeros on the axes, zero and NaN.
            (C32::new(-0.0, 1.0), (-S8, C8)),
            (C32::new(-1.0, -0.0), (-C8, -S8)),
            (C32::new(1.0, -0.0), (C8, S8)),
            (C32::new(0.0, 0.0), (C8, S8)),
            (C32::new(-0.0, -0.0), (C8, S8)),
            (C32::new(f32::NAN, 1.0), (C8, S8)),
            (C32::new(-1.0, f32::NAN), (C8, S8)),
        ];
        for (y, want) in cases {
            let got = octant_point(y);
            assert_eq!(
                (got.0.to_bits(), got.1.to_bits()),
                (want.0.to_bits(), want.1.to_bits()),
                "{y}"
            );
        }
    }
}
