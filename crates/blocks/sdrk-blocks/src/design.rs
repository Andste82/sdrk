//! FIR filter design, computed in `f64` and written to caller slices.
//!
//! The functions mirror `SciPy` (`firwin`, `kaiserord`, `kaiser_beta`) so that they can be
//! checked against golden vectors. Blocks use the same arithmetic one tap at a time, so they
//! never need a temporary `f64` buffer.

use core::f64::consts::PI;

use sdrk_core::window::bessel_i0;
use sdrk_core::{Q15, RealSample};

use crate::error::{Error, Result};

/// Largest number of taps any design accepts (a guard against absurd parameters).
const TAP_LIMIT: usize = 1 << 24;

/// Window of the windowed-sinc design (symmetric, as in `scipy.signal.get_window` with
/// `fftbins=False`).
///
/// ```
/// use sdrk_blocks::design::DesignWindow;
/// assert_ne!(DesignWindow::Hann, DesignWindow::Kaiser { beta: 5.0 });
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum DesignWindow {
    /// `0.5 - 0.5 cos(2 pi n / (N - 1))`.
    Hann,
    /// `0.54 - 0.46 cos(2 pi n / (N - 1))`.
    Hamming,
    /// `0.42 - 0.5 cos(2 pi n / (N - 1)) + 0.08 cos(4 pi n / (N - 1))`.
    Blackman,
    /// `I0(beta sqrt(1 - (2 n / (N - 1) - 1)^2)) / I0(beta)`.
    Kaiser {
        /// Shape parameter, finite and not negative.
        beta: f64,
    },
}

/// A low-pass design that yields one tap at a time (so no `f64` buffer is needed).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Lowpass {
    n: usize,
    fc: f64,
    window: DesignWindow,
    sum: f64,
}

impl Lowpass {
    /// Prepares the design (computes the normalisation sum).
    pub(crate) fn new(
        cutoff_hz: f64,
        fs_hz: f64,
        window: DesignWindow,
        num_taps: usize,
    ) -> Result<Self> {
        if !(fs_hz.is_finite() && fs_hz > 0.0) {
            return Err(Error::InvalidParameter {
                name: "fs_hz",
                reason: "the sample rate must be finite and positive",
            });
        }
        let fc = cutoff_hz / fs_hz;
        if !(fc > 0.0 && fc < 0.5) {
            return Err(Error::OutOfRange {
                name: "cutoff_hz",
                min: 0.0,
                max: fs_hz / 2.0,
                got: cutoff_hz,
            });
        }
        if num_taps == 0 || num_taps > TAP_LIMIT {
            return Err(Error::InvalidParameter {
                name: "num_taps",
                reason: "the number of taps must be between 1 and 2^24",
            });
        }
        if let DesignWindow::Kaiser { beta } = window
            && !(beta.is_finite() && beta >= 0.0)
        {
            return Err(Error::InvalidParameter {
                name: "beta",
                reason: "the kaiser beta must be finite and not negative",
            });
        }
        let mut lp = Self {
            n: num_taps,
            fc,
            window,
            sum: 1.0,
        };
        let mut sum = 0.0f64;
        for i in 0..num_taps {
            sum += lp.raw(i);
        }
        if !(sum.is_finite() && sum != 0.0) {
            return Err(Error::InvalidParameter {
                name: "window",
                reason: "the window leaves no DC gain to normalise (too few taps)",
            });
        }
        lp.sum = sum;
        Ok(lp)
    }

    /// Number of taps.
    pub(crate) fn len(&self) -> usize {
        self.n
    }

    fn window_at(&self, i: usize) -> f64 {
        if self.n == 1 {
            return 1.0;
        }
        let d = (self.n - 1) as f64;
        let x = 2.0 * PI * i as f64 / d;
        match self.window {
            DesignWindow::Hann => 0.5 - 0.5 * libm::cos(x),
            DesignWindow::Hamming => 0.54 - 0.46 * libm::cos(x),
            DesignWindow::Blackman => 0.42 - 0.5 * libm::cos(x) + 0.08 * libm::cos(2.0 * x),
            DesignWindow::Kaiser { beta } => {
                let t = 2.0 * i as f64 / d - 1.0;
                let arg = (1.0 - t * t).max(0.0);
                bessel_i0(beta * libm::sqrt(arg)) / bessel_i0(beta)
            }
        }
    }

    fn raw(&self, i: usize) -> f64 {
        let m = (self.n - 1) as f64 / 2.0;
        let x = 2.0 * self.fc * (i as f64 - m);
        let sinc = if x == 0.0 {
            1.0
        } else {
            libm::sin(PI * x) / (PI * x)
        };
        2.0 * self.fc * sinc * self.window_at(i)
    }

    /// Tap `i`, normalised to a DC gain of 1.
    pub(crate) fn tap(&self, i: usize) -> f64 {
        self.raw(i) / self.sum
    }
}

/// Windowed-sinc low-pass, equal to
/// `scipy.signal.firwin(num_taps, cutoff_hz, window=…, pass_zero=True, scale=True, fs=fs_hz)`
/// with `num_taps = taps.len()`.
///
/// With `fc = cutoff_hz / fs_hz` (`0 < fc < 0.5`, otherwise `OutOfRange`), `N = taps.len()` and
/// `m = (N - 1) / 2`:
///
/// ```text
/// h[n] = 2 fc sinc(2 fc (n - m)) w[n],      sinc(x) = sin(pi x) / (pi x),  sinc(0) = 1
/// ```
///
/// where `w` is the symmetric window (Hann `0.5 - 0.5 cos(2 pi n / (N - 1))`, Hamming
/// `0.54 - 0.46 cos(…)`, Blackman `0.42 - 0.5 cos(2 pi n / (N - 1)) + 0.08 cos(4 pi n / (N - 1))`,
/// Kaiser `I0(beta sqrt(1 - (2 n / (N - 1) - 1)^2)) / I0(beta)` with
/// [`sdrk_core::window::bessel_i0`]). Finally `h` is divided by `sum h`, so the DC gain is 1.
/// One tap is `[1.0]`.
///
/// ```
/// use sdrk_blocks::design::{firwin_lowpass, DesignWindow};
/// let mut h = [0.0f64; 31];
/// firwin_lowpass(1_000.0, 10_000.0, DesignWindow::Hamming, &mut h)?;
/// assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// assert_eq!(h[0], h[30]); // symmetric
/// assert!(firwin_lowpass(6_000.0, 10_000.0, DesignWindow::Hann, &mut h).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
pub fn firwin_lowpass(
    cutoff_hz: f64,
    fs_hz: f64,
    window: DesignWindow,
    taps: &mut [f64],
) -> Result<()> {
    let lp = Lowpass::new(cutoff_hz, fs_hz, window, taps.len())?;
    for (i, t) in taps.iter_mut().enumerate() {
        *t = lp.tap(i);
    }
    Ok(())
}

/// Kaiser window shape for a stop-band attenuation, equal to `scipy.signal.kaiser_beta(a)`:
/// `a > 50`: `0.1102 (a - 8.7)`; `21 <= a <= 50`: `0.5842 (a - 21)^0.4 + 0.07886 (a - 21)`;
/// otherwise `0`.
///
/// ```
/// use sdrk_blocks::design::kaiser_beta;
/// assert!((kaiser_beta(60.0) - 5.65326).abs() < 1e-9);
/// assert_eq!(kaiser_beta(10.0), 0.0);
/// ```
pub fn kaiser_beta(attenuation_db: f64) -> f64 {
    let a = attenuation_db;
    if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a >= 21.0 {
        0.5842 * libm::pow(a - 21.0, 0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    }
}

/// Number of taps and Kaiser `beta` for a ripple and a transition width, equal to
/// `scipy.signal.kaiserord(ripple_db, width = 2 * transition_hz / fs_hz)` with the tap count
/// forced odd (one more if it is even).
///
/// With `A = |ripple_db|` (at least 8) and `width = 2 transition_hz / fs_hz`:
/// `N = ceil((A - 7.95) / 2.285 / (pi width) + 1)`, `beta = kaiser_beta(A)`.
///
/// ```
/// use sdrk_blocks::design::kaiserord;
/// let (n, beta) = kaiserord(60.0, 1_000.0, 48_000.0)?;
/// assert_eq!(n % 2, 1);
/// assert!(n > 100 && (beta - 5.65326).abs() < 1e-4);
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
pub fn kaiserord(ripple_db: f64, transition_hz: f64, fs_hz: f64) -> Result<(usize, f64)> {
    let a = libm::fabs(ripple_db);
    if !a.is_finite() || a < 8.0 {
        return Err(Error::InvalidParameter {
            name: "attenuation_db",
            reason: "the attenuation must be at least 8 dB",
        });
    }
    let width = 2.0 * transition_hz / fs_hz;
    if !(width.is_finite() && width > 0.0) {
        return Err(Error::InvalidParameter {
            name: "transition_hz",
            reason: "the transition width and the sample rate must be finite and positive",
        });
    }
    let num = (a - 7.95) / 2.285 / (PI * width) + 1.0;
    if num > TAP_LIMIT as f64 {
        return Err(Error::OutOfRange {
            name: "transition_hz",
            min: 0.0,
            max: fs_hz,
            got: transition_hz,
        });
    }
    let mut n = libm::ceil(num) as usize;
    if n % 2 == 0 {
        n += 1;
    }
    Ok((n, kaiser_beta(a)))
}

/// Converts taps to `f32`.
///
/// ```
/// use sdrk_blocks::design::to_f32;
/// let mut dst = [0.0f32; 2];
/// to_f32(&[0.5, 0.25], &mut dst)?;
/// assert_eq!(dst, [0.5, 0.25]);
/// assert!(to_f32(&[0.5], &mut dst).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
pub fn to_f32(src: &[f64], dst: &mut [f32]) -> Result<()> {
    check_len(src.len(), dst.len())?;
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = s as f32;
    }
    Ok(())
}

/// Converts taps to Q15 with the rounding rule of the numeric conventions.
///
/// Fails with `OutOfRange` if any `|tap|` exceeds `1 - 2^-15` by more than 1 %, because such a
/// tap would saturate badly.
///
/// ```
/// use sdrk_blocks::design::to_q15;
/// use sdrk_core::Q15;
/// let mut dst = [Q15(0); 2];
/// to_q15(&[0.5, -0.25], &mut dst)?;
/// assert_eq!(dst, [Q15(16384), Q15(-8192)]);
/// assert!(to_q15(&[1.5, 0.0], &mut dst).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
pub fn to_q15(src: &[f64], dst: &mut [Q15]) -> Result<()> {
    check_len(src.len(), dst.len())?;
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = q15_tap(s)?;
    }
    Ok(())
}

/// One tap as Q15 with the range check of [`to_q15`].
pub(crate) fn q15_tap(v: f64) -> Result<Q15> {
    const LIMIT: f64 = (1.0 - 1.0 / 32768.0) * 1.01;
    if libm::fabs(v)
        .partial_cmp(&LIMIT)
        .is_none_or(core::cmp::Ordering::is_gt)
    {
        return Err(Error::OutOfRange {
            name: "tap",
            min: -LIMIT,
            max: LIMIT,
            got: v,
        });
    }
    Ok(Q15::from_f64_round(v))
}

fn check_len(expected: usize, got: usize) -> Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::Core(sdrk_core::Error::LengthMismatch {
            expected,
            got,
        }))
    }
}

/// The common parameter set of the filtering blocks: a low-pass with a cutoff, a transition
/// width and a stop-band attenuation, designed with [`kaiserord`] and a Kaiser window.
///
/// ```
/// use sdrk_blocks::design::DesignSpec;
/// let spec = DesignSpec { cutoff_hz: 10_000.0, transition_hz: 2_000.0, attenuation_db: 60.0 };
/// let n = spec.num_taps(48_000.0, 1023)?;
/// let mut h = vec![0.0f64; n];
/// spec.design(48_000.0, 1023, &mut h)?;
/// assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// assert!(spec.num_taps(48_000.0, 10).is_err()); // more than the cap of 10 taps
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DesignSpec {
    /// Cutoff (the -6 dB point of the window design) in Hz.
    pub cutoff_hz: f64,
    /// Width of the transition band in Hz, centred on the cutoff.
    pub transition_hz: f64,
    /// Stop-band attenuation in dB (at least 8).
    pub attenuation_db: f64,
}

impl DesignSpec {
    fn validate(&self, fs_hz: f64) -> Result<()> {
        if !(fs_hz.is_finite() && fs_hz > 0.0) {
            return Err(Error::InvalidParameter {
                name: "fs_hz",
                reason: "the sample rate must be finite and positive",
            });
        }
        if !(self.transition_hz.is_finite() && self.transition_hz > 0.0) {
            return Err(Error::OutOfRange {
                name: "transition_hz",
                min: 0.0,
                max: fs_hz / 2.0,
                got: self.transition_hz,
            });
        }
        let max = fs_hz / 2.0 - self.transition_hz / 2.0;
        if !(self.cutoff_hz > 0.0 && self.cutoff_hz <= max) {
            return Err(Error::OutOfRange {
                name: "cutoff_hz",
                min: 0.0,
                max,
                got: self.cutoff_hz,
            });
        }
        Ok(())
    }

    /// Number of taps (odd) at the sample rate `fs_hz`.
    ///
    /// # Errors
    /// `OutOfRange` if the cutoff plus half the transition exceeds `fs_hz / 2` or the
    /// design needs more than `max_taps` taps.
    pub fn num_taps(&self, fs_hz: f64, max_taps: usize) -> Result<usize> {
        self.validate(fs_hz)?;
        let (n, _) = kaiserord(self.attenuation_db, self.transition_hz, fs_hz)?;
        if n > max_taps {
            return Err(Error::OutOfRange {
                name: "max_taps",
                min: n as f64,
                max: TAP_LIMIT as f64,
                got: max_taps as f64,
            });
        }
        Ok(n)
    }

    /// Designs the taps into `taps`, whose length must be [`DesignSpec::num_taps`].
    pub fn design(&self, fs_hz: f64, max_taps: usize, taps: &mut [f64]) -> Result<()> {
        let n = self.num_taps(fs_hz, max_taps)?;
        check_len(n, taps.len())?;
        let (_, beta) = kaiserord(self.attenuation_db, self.transition_hz, fs_hz)?;
        firwin_lowpass(self.cutoff_hz, fs_hz, DesignWindow::Kaiser { beta }, taps)
    }

    /// The streaming design used by the blocks: one tap at a time, no `f64` buffer.
    pub(crate) fn lowpass(&self, fs_hz: f64, max_taps: usize) -> Result<Lowpass> {
        let n = self.num_taps(fs_hz, max_taps)?;
        let (_, beta) = kaiserord(self.attenuation_db, self.transition_hz, fs_hz)?;
        Lowpass::new(self.cutoff_hz, fs_hz, DesignWindow::Kaiser { beta }, n)
    }
}
