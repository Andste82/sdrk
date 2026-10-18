//! Frequency mixing with a numerically controlled oscillator.

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::timing::{Delay, Rate, Timing};
use crate::types::ComplexSample;

/// Numerically controlled oscillator state: phase accumulator in units of 2π/2^32.
///
/// The phase advances by `inc` per sample (wrapping), so a frequency `f` at
/// sample rate `fs` corresponds to `inc = round(frac(f / fs) · 2^32)`; negative
/// frequencies map to `inc > 2^31`. A retune keeps the phase, so the signal stays
/// phase-continuous.
///
/// ```
/// use sdrk_core::{kernels::mix::Nco, Rate};
/// let nco = Nco::new(250_000.0, Rate::hz(1_000_000)?)?;
/// assert_eq!(nco.increment(), 1 << 30);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Nco {
    phase: u32,
    inc: u32,
}

const TWO_POW_32: f64 = 4_294_967_296.0;

/// `round(frac(freq_hz / rate) * 2^32)` modulo `2^32`.
fn increment_for(freq_hz: f64, rate: Rate) -> Result<u32> {
    let cycles = freq_hz / rate.as_f64();
    if !cycles.is_finite() {
        return Err(Error::InvalidParameter("frequency must be finite"));
    }
    let frac = cycles - libm::floor(cycles);
    // Round half to even, like Python's `round`; the value is in `0..=2^32`.
    let scaled = libm::rint(frac * TWO_POW_32);
    Ok((scaled as u64) as u32)
}

impl Nco {
    /// Creates an oscillator at `freq_hz` for samples at `rate`, starting at phase 0.
    /// NaN and infinite frequencies are rejected with `InvalidParameter`.
    ///
    /// ```
    /// use sdrk_core::{kernels::mix::Nco, Rate};
    /// let nco = Nco::new(-250_000.0, Rate::hz(1_000_000)?)?;
    /// assert_eq!(nco.increment(), 3 << 30);
    /// assert!(Nco::new(f64::NAN, Rate::hz(1)?).is_err());
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn new(freq_hz: f64, rate: Rate) -> Result<Nco> {
        Ok(Nco {
            phase: 0,
            inc: increment_for(freq_hz, rate)?,
        })
    }

    /// Creates an oscillator from a raw phase increment, starting at phase 0.
    pub const fn from_increment(inc: u32) -> Nco {
        Nco { phase: 0, inc }
    }

    /// Changes the frequency and keeps the phase (phase-continuous retune).
    ///
    /// ```
    /// use sdrk_core::{kernels::mix::Nco, Rate};
    /// let rate = Rate::hz(1000)?;
    /// let mut nco = Nco::new(100.0, rate)?;
    /// nco.set_phase(12345);
    /// nco.set_frequency(250.0, rate)?;
    /// assert_eq!((nco.phase(), nco.increment()), (12345, 1 << 30));
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn set_frequency(&mut self, freq_hz: f64, rate: Rate) -> Result<()> {
        self.inc = increment_for(freq_hz, rate)?;
        Ok(())
    }

    /// The current phase in units of 2π/2^32.
    pub const fn phase(&self) -> u32 {
        self.phase
    }

    /// Sets the phase.
    pub const fn set_phase(&mut self, p: u32) {
        self.phase = p;
    }

    /// The phase increment per sample.
    pub const fn increment(&self) -> u32 {
        self.inc
    }

    /// Sets phase to 0 and keeps the increment.
    pub const fn reset(&mut self) {
        self.phase = 0;
    }

    /// Advances the phase by `n` samples (wrapping). Used by kernels after a call.
    pub(crate) const fn advance(&mut self, n: usize) {
        self.phase = self.phase.wrapping_add(self.inc.wrapping_mul(n as u32));
    }
}

impl Timing for Nco {
    fn ratio(&self) -> Ratio {
        Ratio::ONE
    }

    fn group_delay(&self) -> Delay {
        Delay::ZERO
    }
}

/// Multiplies a complex signal by `e^{j·phase}` of an oscillator (frequency shift).
///
/// ```
/// use sdrk_core::{kernels::mix::{Nco, Rotate}, Fallback, C32};
/// // A quarter turn per sample: 1 -> 1, j, -1, -j (up to table rounding).
/// let mut nco = Nco::from_increment(1 << 30);
/// let input = [C32::new(1.0, 0.0); 4];
/// let mut output = [C32::default(); 4];
/// Fallback.rotate(&mut nco, &input, &mut output)?;
/// assert!((output[1].im - 1.0).abs() < 1e-6);
/// assert!((output[2].re + 1.0).abs() < 1e-6);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Rotate<T: ComplexSample> {
    /// `output[n] = input[n] · e^{j·2π·phase_n / 2^32}`; `phase_{n+1} = phase_n + inc` (wrapping).
    /// Positive frequency moves the spectrum up; to bring a signal at +f to 0 Hz use −f.
    /// The lengths must be equal; the oscillator phase advances by `input.len()`.
    fn rotate(&self, nco: &mut Nco, input: &[T], output: &mut [T]) -> Result<()>;
}
