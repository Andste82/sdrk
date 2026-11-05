//! Automatic notch: a normalized LMS line enhancer that removes steady tones from real audio.
//!
//! An adaptive filter predicts the input from the samples `delay` and more samples back. Steady
//! tones (carriers, heterodynes) are predictable over that distance, noise and speech are
//! not, so the prediction error, which is the output, keeps the noise and the speech and loses
//! the tones. The filter follows a tone whose frequency drifts.

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::timing::{Delay, Timing};

/// Smallest and largest number of taps.
const TAPS: core::ops::RangeInclusive<usize> = 8..=256;
/// Smallest and largest delay in samples.
const DELAY: core::ops::RangeInclusive<usize> = 1..=64;

/// Mutable view of the state, used by the kernel implementations of this crate.
pub(crate) struct AutoNotchParts<'a> {
    pub(crate) weights: &'a mut [f32],
    pub(crate) hist: &'a mut [f32],
    pub(crate) pos: &'a mut usize,
    pub(crate) mu: f32,
}

/// State of the automatic notch: `L` weights, a ring of the last `L + delay` input samples
/// and the step size `mu`.
///
/// The caller provides the storage (`[f32; N]`, `&mut [f32]` or, in upper layers,
/// `Vec<f32>`). The state starts with zero weights and a zero history.
///
/// ```
/// use sdrk_core::kernels::auto_notch::AutoNotchState;
/// let state = AutoNotchState::new([0.0f32; 32], [0.0f32; 40], 8, 0.01)?;
/// assert_eq!((state.taps(), state.delay(), state.mu()), (32, 8, 0.01));
/// // The history must hold exactly taps + delay samples.
/// assert!(AutoNotchState::new([0.0f32; 32], [0.0f32; 39], 8, 0.01).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct AutoNotchState<W: AsRef<[f32]> + AsMut<[f32]>, X: AsRef<[f32]> + AsMut<[f32]>> {
    weights: W,
    hist: X,
    delay: usize,
    mu: f32,
    /// Index of the oldest sample of the ring (the next one to be overwritten).
    pos: usize,
}

fn check_mu(mu: f32) -> Result<()> {
    if mu.is_finite() && (0.0001..=0.1).contains(&mu) {
        Ok(())
    } else {
        Err(Error::InvalidParameter(
            "auto notch step size must be between 0.0001 and 0.1",
        ))
    }
}

impl<W: AsRef<[f32]> + AsMut<[f32]>, X: AsRef<[f32]> + AsMut<[f32]>> AutoNotchState<W, X> {
    /// Creates a state: `L = weights.len()` taps (8 to 256), `delay` (`Δ`, 1 to 64 samples)
    /// between the newest sample and the newest sample the prediction uses, and the step size
    /// `mu` (0.0001 to 0.1) of the normalized LMS update. `history.len()` must be `L + delay`
    /// (`LengthMismatch` otherwise); the other violations, including NaN, give
    /// `InvalidParameter`. Weights and history are zeroed.
    pub fn new(mut weights: W, mut history: X, delay: usize, mu: f32) -> Result<Self> {
        let taps = weights.as_ref().len();
        if !TAPS.contains(&taps) {
            return Err(Error::InvalidParameter("auto notch needs 8 to 256 taps"));
        }
        if !DELAY.contains(&delay) {
            return Err(Error::InvalidParameter(
                "auto notch delay must be between 1 and 64 samples",
            ));
        }
        check_mu(mu)?;
        let expected = taps + delay;
        let got = history.as_ref().len();
        if got != expected {
            return Err(Error::LengthMismatch { expected, got });
        }
        weights.as_mut().fill(0.0);
        history.as_mut().fill(0.0);
        Ok(Self {
            weights,
            hist: history,
            delay,
            mu,
            pos: 0,
        })
    }

    /// Changes the step size (0.0001 to 0.1, `InvalidParameter` otherwise and nothing
    /// changes); weights and history are kept and the new value applies from the next sample.
    ///
    /// ```
    /// use sdrk_core::kernels::auto_notch::AutoNotchState;
    /// let mut state = AutoNotchState::new([0.0f32; 8], [0.0f32; 9], 1, 0.01)?;
    /// state.set_mu(0.05)?;
    /// assert_eq!(state.mu(), 0.05);
    /// assert!(state.set_mu(0.2).is_err());
    /// assert_eq!(state.mu(), 0.05);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn set_mu(&mut self, mu: f32) -> Result<()> {
        check_mu(mu)?;
        self.mu = mu;
        Ok(())
    }

    /// Clears the weights and the history (taps, delay and step size are kept).
    pub fn reset(&mut self) {
        self.weights.as_mut().fill(0.0);
        self.hist.as_mut().fill(0.0);
        self.pos = 0;
    }

    /// The number of taps `L`.
    pub fn taps(&self) -> usize {
        self.weights.as_ref().len()
    }

    /// The delay `Δ` in samples.
    pub fn delay(&self) -> usize {
        self.delay
    }

    /// The step size.
    pub fn mu(&self) -> f32 {
        self.mu
    }

    /// The weights `w[0..L]`; `w[i]` multiplies the input `delay + i` samples back.
    pub fn weights(&self) -> &[f32] {
        self.weights.as_ref()
    }

    pub(crate) fn parts_mut(&mut self) -> AutoNotchParts<'_> {
        AutoNotchParts {
            weights: self.weights.as_mut(),
            hist: self.hist.as_mut(),
            pos: &mut self.pos,
            mu: self.mu,
        }
    }
}

impl<W: AsRef<[f32]> + AsMut<[f32]>, X: AsRef<[f32]> + AsMut<[f32]>> Timing
    for AutoNotchState<W, X>
{
    fn ratio(&self) -> Ratio {
        Ratio::ONE
    }

    /// Zero; the output is the input minus a prediction from older samples.
    fn group_delay(&self) -> Delay {
        Delay::ZERO
    }
}

/// Automatic notch (`f32` audio).
///
/// ```
/// use sdrk_core::{kernels::auto_notch::{AutoNotch, AutoNotchState}, Fallback};
/// let mut state = AutoNotchState::new([0.0f32; 32], [0.0f32; 40], 8, 0.05)?;
/// let input: Vec<f32> =
///     (0..8000).map(|n| 0.5 * libm::sinf(2.0 * core::f32::consts::PI * 0.125 * n as f32)).collect();
/// let mut output = vec![0.0f32; 8000];
/// Fallback.auto_notch(&mut state, &input, &mut output)?;
/// // After the adaptation the tone is gone.
/// assert!(output[7000..].iter().all(|y| y.abs() < 0.01));
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait AutoNotch {
    /// Per sample `x[n]`: the reference vector `u = (x[n - Δ], ..., x[n - Δ - L + 1])` (zero
    /// before the first sample), the prediction `y = Σ w[i] u[i]` and the power
    /// `P = Σ u[i] u[i]` (each a sum from `+0.0` in ascending `i`), the error
    /// `e = x[n] - y`, which is the output, and the update
    /// `w[i] = flush_denormal(w[i] + f * u[i])` with `f = (mu * e) / (P + 1e-6)` computed
    /// once; all in f32 without FMA. A non-finite input is stored as `0.0` in the history,
    /// the output is `+0.0` and the weights stay. A finite input so large that `f` is not
    /// finite leaves the weights unchanged, and a non-finite `e` gives the output `+0.0`, so
    /// the weights and the output stay finite. The lengths must be equal. Bit-exact
    /// between `Fallback` and `Reference` (see `docs/kernels.md`).
    fn auto_notch<W, X>(
        &self,
        s: &mut AutoNotchState<W, X>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()>
    where
        W: AsRef<[f32]> + AsMut<[f32]>,
        X: AsRef<[f32]> + AsMut<[f32]>;
}
