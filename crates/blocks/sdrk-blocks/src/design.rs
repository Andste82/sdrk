//! FIR filter design, computed in `f64` and written to caller slices.
//!
//! The functions mirror `SciPy` (`firwin`, `kaiserord`, `kaiser_beta`) so that they can be
//! checked against golden vectors. Blocks use the same arithmetic one tap at a time, so they
//! never need a temporary `f64` buffer.

use core::f64::consts::PI;

use sdrk_core::num::q14_from_f64;
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
    if n.is_multiple_of(2) {
        n += 1;
    }
    Ok((n, kaiser_beta(a)))
}

/// Tap count of a halfband filter: `kaiserord(attenuation_db, transition_hz, fs_hz)` rounded up
/// to the next `N` with `N mod 4 == 3` (so `N >= 3`).
///
/// ```
/// use sdrk_blocks::design::halfband_len;
/// let n = halfband_len(0.1, 60.0, 1.0)?;
/// assert_eq!(n % 4, 3);
/// assert!(halfband_len(0.1, 60.0, 1.0)? <= halfband_len(0.05, 60.0, 1.0)?);
/// assert!(halfband_len(0.1, 3.0, 1.0).is_err()); // below the 8 dB limit of kaiserord
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
pub fn halfband_len(transition_hz: f64, attenuation_db: f64, fs_hz: f64) -> Result<usize> {
    let (n, _) = kaiserord(attenuation_db, transition_hz, fs_hz)?;
    // `kaiserord` returns an odd count: 4 k + 1 becomes 4 k + 3.
    Ok(if n % 4 == 1 { n + 2 } else { n })
}

/// Halfband taps of the length `taps.len()` (`N mod 4 == 3`, `N >= 3`) with a Kaiser window of
/// the given `beta`, exactly symmetric, with the sum 1.
///
/// With `m = (N - 1) / 2` (odd): `h = firwin_lowpass(fs / 4, fs, Kaiser { beta })` for
/// `fs = 1`, then `h[m ± 2 i] = 0.0` exactly for `i >= 1` (they are about 1e-17 from
/// `sin(k pi)`), the right half is set to the mirror of the left one (the window and the sinc
/// are symmetric only up to rounding), and `h` is scaled so that `sum h = 1`. About half of the taps are zero, the
/// outermost taps (even index) are not, and the centre tap is close to 0.5. The usable band
/// after a decimation by 2 is `±(fs / 4 - transition_hz / 2)` of the input rate.
///
/// ```
/// use sdrk_blocks::design::{halfband_taps, kaiser_beta};
/// let mut h = [0.0f64; 23];
/// halfband_taps(kaiser_beta(60.0), &mut h)?;
/// assert_eq!(h[11 - 2], 0.0); // zero at the even offsets from the centre
/// assert_eq!(h[1], 0.0);
/// assert!(h[0] != 0.0 && h[22] == h[0]);
/// assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-14);
/// assert!(halfband_taps(5.0, &mut [0.0; 21]).is_err()); // 21 mod 4 == 1
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
pub fn halfband_taps(beta: f64, taps: &mut [f64]) -> Result<()> {
    let n = taps.len();
    if n < 3 || n % 4 != 3 {
        return Err(Error::InvalidParameter {
            name: "taps",
            reason: "a halfband filter needs 4 k + 3 taps",
        });
    }
    firwin_lowpass(0.25, 1.0, DesignWindow::Kaiser { beta }, taps)?;
    let m = (n - 1) / 2;
    for (k, t) in taps.iter_mut().enumerate() {
        if k != m && k.abs_diff(m).is_multiple_of(2) {
            *t = 0.0;
        }
    }
    // The window and the sinc are symmetric only up to rounding (a few units in the last place):
    // the right half becomes the exact mirror of the left one, so that the taps are exactly
    // symmetric (a pre-add of the two samples of a tap pair in an accelerated path is then exact).
    let (left, right) = taps.split_at_mut(m + 1);
    for (r, l) in right.iter_mut().zip(left.iter().rev().skip(1)) {
        *r = *l;
    }
    let sum: f64 = taps.iter().sum();
    if !(sum.is_finite() && sum != 0.0) {
        return Err(Error::InvalidParameter {
            name: "beta",
            reason: "the halfband taps do not have a usable sum",
        });
    }
    for t in taps.iter_mut() {
        *t /= sum;
    }
    Ok(())
}

/// The halfband design of [`halfband_taps`], one tap at a time (so the blocks need no `f64`
/// buffer). `tap(k)` equals `halfband_taps(beta, ..)[k]` bit for bit.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HalfbandDesign {
    lp: Lowpass,
    n: usize,
    sum: f64,
}

impl HalfbandDesign {
    /// Prepares the design of `n` taps (`n mod 4 == 3`).
    pub(crate) fn new(beta: f64, n: usize) -> Result<Self> {
        if n < 3 || n % 4 != 3 {
            return Err(Error::InvalidParameter {
                name: "taps",
                reason: "a halfband filter needs 4 k + 3 taps",
            });
        }
        let lp = Lowpass::new(0.25, 1.0, DesignWindow::Kaiser { beta }, n)?;
        let mut d = Self { lp, n, sum: 1.0 };
        let mut sum = 0.0f64;
        for k in 0..n {
            sum += d.unscaled(k);
        }
        if !(sum.is_finite() && sum != 0.0) {
            return Err(Error::InvalidParameter {
                name: "beta",
                reason: "the halfband taps do not have a usable sum",
            });
        }
        d.sum = sum;
        Ok(d)
    }

    /// Mirrored tap with the even offsets from the centre set to zero, before the scaling.
    fn unscaled(&self, k: usize) -> f64 {
        let m = (self.n - 1) / 2;
        if k != m && k.abs_diff(m).is_multiple_of(2) {
            0.0
        } else {
            self.lp.tap(k.min(self.n - 1 - k))
        }
    }

    /// Tap `k` (scaled so that the taps sum to 1).
    pub(crate) fn tap(&self, k: usize) -> f64 {
        self.unscaled(k) / self.sum
    }
}

/// Smallest and largest `bt` of [`gaussian`].
const GAUSSIAN_BT: (f64, f64) = (0.2, 1.0);

/// Smallest and largest `span` (in symbols) of [`gaussian`].
const GAUSSIAN_SPAN: (usize, usize) = (2, 8);

/// The Gaussian pulse of [`gaussian`], one tap at a time (so a block needs no `f64` buffer).
/// `tap(n)` equals `gaussian(bt, sps, span, ..)[n]` bit for bit.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Gaussian {
    /// `2 pi^2 bt^2 / ln 2`.
    a: f64,
    /// Samples per symbol.
    sps: f64,
    /// `(len - 1) / 2`.
    mid: f64,
    /// Number of taps, `span sps + 1`.
    n: usize,
    /// Sum of the unscaled taps.
    sum: f64,
}

impl Gaussian {
    /// Validates the parameters and computes the normalisation sum.
    pub(crate) fn new(bt: f64, sps: usize, span: usize) -> Result<Self> {
        let (bt_min, bt_max) = GAUSSIAN_BT;
        if !(bt.is_finite() && (bt_min..=bt_max).contains(&bt)) {
            return Err(Error::OutOfRange {
                name: "bt",
                min: bt_min,
                max: bt_max,
                got: bt,
            });
        }
        let (span_min, span_max) = GAUSSIAN_SPAN;
        if !(span_min..=span_max).contains(&span) {
            return Err(Error::OutOfRange {
                name: "span",
                min: span_min as f64,
                max: span_max as f64,
                got: span as f64,
            });
        }
        let sps_max = (TAP_LIMIT - 1) / span;
        if !(2..=sps_max).contains(&sps) {
            return Err(Error::OutOfRange {
                name: "sps",
                min: 2.0,
                max: sps_max as f64,
                got: sps as f64,
            });
        }
        // `span sps <= TAP_LIMIT - 1`, so neither operation overflows.
        let n = span * sps + 1;
        let mut g = Self {
            a: 2.0 * PI * PI * bt * bt / core::f64::consts::LN_2,
            sps: sps as f64,
            mid: (n - 1) as f64 / 2.0,
            n,
            sum: 1.0,
        };
        let mut sum = 0.0f64;
        for i in 0..n {
            sum += g.raw(i);
        }
        // The centre region is close to 1 for every valid `bt`, so the sum is at least 1.
        g.sum = sum;
        Ok(g)
    }

    /// Number of taps, `span sps + 1`.
    pub(crate) fn len(&self) -> usize {
        self.n
    }

    /// Unscaled tap `exp(-a (t/T)^2)` with `t/T = (i - mid) / sps`.
    fn raw(&self, i: usize) -> f64 {
        let x = (i as f64 - self.mid) / self.sps;
        libm::exp(-(self.a * (x * x)))
    }

    /// Tap `i`, normalised so that the taps sum to 1.
    pub(crate) fn tap(&self, i: usize) -> f64 {
        self.raw(i) / self.sum
    }
}

/// Gaussian pulse-shaping filter of GFSK and GMSK, normalised to sum 1.
///
/// With `N = span sps + 1` taps (`taps.len()` must be `N`), the symbol period `T`,
/// `t/T = (n - (N - 1) / 2) / sps` and the bandwidth-time product `bt`:
///
/// ```text
/// h[n] = exp(-(2 pi^2 bt^2 / ln 2) (t/T)^2),   then h = h / sum h
/// ```
///
/// `bt` is 0.2 to 1 (0.5 with a modulation index of 0.5 is GMSK), `sps` (samples per symbol)
/// at least 2, `span` (the length in symbols) 2 to 8. The filter is symmetric up to rounding;
/// its peak is at the centre. Computed in `f64`, equal to the Python formula of the golden case
/// `design/gaussian_bt0p5_sps8_span4` within 1e-12.
///
/// ```
/// use sdrk_blocks::design::gaussian;
/// let mut h = [0.0f64; 4 * 8 + 1];
/// gaussian(0.5, 8, 4, &mut h)?;
/// assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// assert!(h[16] > h[15] && (h[15] - h[17]).abs() < 1e-15); // peak at the centre
/// assert!(gaussian(0.1, 8, 4, &mut h).is_err()); // bt below 0.2
/// assert!(gaussian(0.5, 8, 4, &mut [0.0; 32]).is_err()); // wrong length
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
///
/// # Errors
/// `OutOfRange` for `bt` outside 0.2 to 1 (NaN included), `sps` below 2 (or so large that
/// the filter exceeds 2^24 taps) or `span` outside 2 to 8; `Core(LengthMismatch)` if
/// `taps.len()` is not `span sps + 1`.
pub fn gaussian(bt: f64, sps: usize, span: usize, taps: &mut [f64]) -> Result<()> {
    let g = Gaussian::new(bt, sps, span)?;
    check_len(g.len(), taps.len())?;
    for (i, t) in taps.iter_mut().enumerate() {
        *t = g.tap(i);
    }
    Ok(())
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

/// Response of a second-order section of the RBJ Audio EQ Cookbook, and of the Butterworth
/// cascades (low-pass and high-pass only).
///
/// ```
/// use sdrk_blocks::design::BiquadResponse;
/// assert_ne!(BiquadResponse::Lowpass, BiquadResponse::Notch);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BiquadResponse {
    /// Second-order low-pass, DC gain 1.
    Lowpass,
    /// Second-order high-pass, gain 1 at Nyquist.
    Highpass,
    /// Band-pass with 0 dB peak gain at the centre frequency.
    Bandpass,
    /// Notch: zeros on the unit circle at the centre frequency.
    Notch,
}

/// Coefficients `[b0, b1, b2, a1, a2]` (with `a0 = 1`) of one section of the RBJ Audio EQ
/// Cookbook (R. Bristow-Johnson), computed in `f64` and normalized by `a0`.
///
/// `0 < f0_hz < fs_hz / 2` and `q > 0` (else `OutOfRange`). With `w0 = 2 pi f0 / fs`,
/// `c = cos w0` and `alpha = sin(w0) / (2 q)` the numerator before the normalization is
///
/// - `Lowpass`: `b = [(1 - c) / 2, 1 - c, (1 - c) / 2]`
/// - `Highpass`: `b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]`
/// - `Bandpass` (0 dB peak gain): `b = [alpha, 0, -alpha]`
/// - `Notch`: `b = [1, -2 c, 1]`
///
/// and the denominator is `a = [1 + alpha, -2 c, 1 - alpha]` for all four. The result is
/// `[b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0]` with `a0 = 1 + alpha`.
///
/// ```
/// use sdrk_blocks::design::{biquad_coefs, BiquadResponse};
/// // A low-pass has the DC gain (b0 + b1 + b2) / (1 + a1 + a2) = 1.
/// let c = biquad_coefs(BiquadResponse::Lowpass, 1_000.0, 0.7071, 48_000.0)?;
/// assert!(((c[0] + c[1] + c[2]) / (1.0 + c[3] + c[4]) - 1.0).abs() < 1e-12);
/// // The centre frequency must be below Nyquist.
/// assert!(biquad_coefs(BiquadResponse::Notch, 30_000.0, 10.0, 48_000.0).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[allow(clippy::similar_names)]
pub fn biquad_coefs(response: BiquadResponse, f0_hz: f64, q: f64, fs_hz: f64) -> Result<[f64; 5]> {
    if !(f0_hz > 0.0 && f0_hz < fs_hz / 2.0 && fs_hz.is_finite()) {
        return Err(Error::OutOfRange {
            name: "f0_hz",
            min: 0.0,
            max: fs_hz / 2.0,
            got: f0_hz,
        });
    }
    if !(q > 0.0 && q.is_finite()) {
        return Err(Error::OutOfRange {
            name: "q",
            min: 0.0,
            max: f64::INFINITY,
            got: q,
        });
    }
    let w0 = 2.0 * PI * f0_hz / fs_hz;
    let c = libm::cos(w0);
    let alpha = libm::sin(w0) / (2.0 * q);
    let hp = 1.0 + c;
    let b = match response {
        BiquadResponse::Lowpass => [(1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0],
        BiquadResponse::Highpass => [hp / 2.0, -hp, hp / 2.0],
        BiquadResponse::Bandpass => [alpha, 0.0, -alpha],
        BiquadResponse::Notch => [1.0, -2.0 * c, 1.0],
    };
    let a = [1.0 + alpha, -2.0 * c, 1.0 - alpha];
    Ok([
        b[0] / a[0],
        b[1] / a[0],
        b[2] / a[0],
        a[1] / a[0],
        a[2] / a[0],
    ])
}

/// Lowest sample rate that `preemphasis_coefs` accepts, in hertz. Below it the bilinear mapping
/// without pre-warping differs from the ideal `1 + j 2 pi f tau` by more than 0.2 dB inside the
/// audio band (1 dB at 32 kHz).
pub const PREEMPHASIS_MIN_RATE_HZ: f64 = 40_000.0;

/// Largest time constant that `preemphasis_coefs` accepts, in seconds (1000 us).
pub const PREEMPHASIS_MAX_TAU_S: f64 = 1e-3;

/// First-order pre-emphasis for the de-emphasis time constant `tau_s`: `[b0, b1, a1]` of
/// `H(z) = (b0 + b1 z^-1) / (1 + a1 z^-1)` with unity gain at DC.
///
/// The ideal pre-emphasis `1 + s tau` has no upper limit, so the design adds a pole at
/// `f_p = fs / 2.4` (20 kHz at 48 kHz) and maps `H(s) = (1 + s tau) / (1 + s tau_p)` with
/// `tau_p = 1 / (2 pi f_p)` by the bilinear transform without pre-warping: with `K = 2 fs`,
/// `b0 = (1 + K tau) / (1 + K tau_p)`, `b1 = (1 - K tau) / (1 + K tau_p)` and
/// `a1 = (1 - K tau_p) / (1 + K tau_p)`. The magnitude differs from `|1 + j 2 pi f tau|` by at
/// most 0.2 dB from 30 Hz to `min(15 kHz, 0.45 fs)`. The gain at Nyquist is `tau / tau_p`
/// (16 dB for 50 us at 48 kHz, 36 dB for 530 us at 48 kHz, 56 dB for 1000 us at 240 kHz), which
/// is why the `preemph_limit` kernel keeps the pre-emphasized value in a wide format.
///
/// `tau_s` must be in `(0, 1e-3]` and `fs_hz` at least 40 000 and finite, else `OutOfRange`
/// (NaN included).
///
/// ```
/// use sdrk_blocks::design::preemphasis_coefs;
/// let [b0, b1, a1] = preemphasis_coefs(50e-6, 48_000.0)?;
/// // Unity gain at DC.
/// assert!(((b0 + b1) / (1.0 + a1) - 1.0).abs() < 1e-12);
/// // 75 us is fine, 2 ms and a sample rate below 40 kHz are rejected.
/// assert!(preemphasis_coefs(75e-6, 48_000.0).is_ok());
/// assert!(preemphasis_coefs(2e-3, 48_000.0).is_err());
/// assert!(preemphasis_coefs(50e-6, 32_000.0).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
pub fn preemphasis_coefs(tau_s: f64, fs_hz: f64) -> Result<[f64; 3]> {
    if !(tau_s > 0.0 && tau_s <= PREEMPHASIS_MAX_TAU_S) {
        return Err(Error::OutOfRange {
            name: "tau_s",
            min: 0.0,
            max: PREEMPHASIS_MAX_TAU_S,
            got: tau_s,
        });
    }
    if !(fs_hz >= PREEMPHASIS_MIN_RATE_HZ && fs_hz.is_finite()) {
        return Err(Error::OutOfRange {
            name: "fs_hz",
            min: PREEMPHASIS_MIN_RATE_HZ,
            max: f64::INFINITY,
            got: fs_hz,
        });
    }
    let tau_p = 1.0 / (2.0 * PI * (fs_hz / 2.4));
    let k = 2.0 * fs_hz;
    let den = 1.0 + k * tau_p;
    Ok([
        (1.0 + k * tau_s) / den,
        (1.0 - k * tau_s) / den,
        (1.0 - k * tau_p) / den,
    ])
}

/// Butterworth filter of even order `2 ..= 8` as `order / 2` sections, low-pass or high-pass
/// (`Bandpass` and `Notch` give `InvalidParameter`), the lowest `Q` first.
///
/// `sos.len()` must be `order / 2` (else `LengthMismatch`). Section `k = 1 ..= order / 2` is
/// `biquad_coefs(response, cutoff_hz, Q_k, fs_hz)` with
/// `Q_k = 1 / (2 cos((2 k - 1) pi / (2 order)))`. The cookbook and
/// `scipy.signal.butter(order, cutoff_hz, btype, fs=fs_hz, output='sos')` both use the bilinear
/// transform prewarped at the cutoff, so the cascade has the response of that `SciPy` call (the
/// golden generator checks it); the gain is -3.0103 dB at the cutoff for every order.
///
/// ```
/// use sdrk_blocks::design::{butterworth_sos, BiquadResponse};
/// let mut sos = [[0.0f64; 5]; 2];
/// butterworth_sos(BiquadResponse::Lowpass, 4, 1_000.0, 48_000.0, &mut sos)?;
/// // The first section has the lower Q, hence the smaller a2.
/// assert!(sos[0][4] < sos[1][4]);
/// // Odd orders and a wrong slice length are rejected.
/// assert!(butterworth_sos(BiquadResponse::Lowpass, 3, 1_000.0, 48_000.0, &mut sos[..1]).is_err());
/// assert!(butterworth_sos(BiquadResponse::Lowpass, 4, 1_000.0, 48_000.0, &mut sos[..1]).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
pub fn butterworth_sos(
    response: BiquadResponse,
    order: usize,
    cutoff_hz: f64,
    fs_hz: f64,
    sos: &mut [[f64; 5]],
) -> Result<()> {
    if !matches!(response, BiquadResponse::Lowpass | BiquadResponse::Highpass) {
        return Err(Error::InvalidParameter {
            name: "response",
            reason: "a Butterworth cascade is a low-pass or a high-pass",
        });
    }
    if !(2..=8).contains(&order) || !order.is_multiple_of(2) {
        return Err(Error::InvalidParameter {
            name: "order",
            reason: "the order must be even, from 2 to 8",
        });
    }
    check_len(order / 2, sos.len())?;
    for (i, section) in sos.iter_mut().enumerate() {
        let k = (i + 1) as f64;
        let q = 1.0 / (2.0 * libm::cos((2.0 * k - 1.0) * PI / (2.0 * order as f64)));
        *section = biquad_coefs(response, cutoff_hz, q, fs_hz)?;
    }
    Ok(())
}

/// Converts coefficients to Q2.14 (`sdrk_core::num::q14_from_f64`: `floor(x 2^14 + 0.5)`).
///
/// Fails with `OutOfRange` if a coefficient does not fit (`-2 <= x < 2`) and with
/// `LengthMismatch` if the slices differ in length.
///
/// ```
/// use sdrk_blocks::design::to_q14;
/// let mut dst = [[0i16; 5]; 1];
/// to_q14(&[[0.5, -2.0, 1.0, 0.25, 0.0]], &mut dst)?;
/// assert_eq!(dst, [[8192, -32768, 16384, 4096, 0]]);
/// assert!(to_q14(&[[2.0, 0.0, 0.0, 0.0, 0.0]], &mut dst).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
pub fn to_q14(src: &[[f64; 5]], dst: &mut [[i16; 5]]) -> Result<()> {
    check_len(src.len(), dst.len())?;
    for (d, s) in dst.iter_mut().zip(src) {
        for (dv, &sv) in d.iter_mut().zip(s) {
            *dv = q14_from_f64(sv).ok_or(Error::OutOfRange {
                name: "coefficient",
                min: -2.0,
                max: 2.0 - 1.0 / 16384.0,
                got: sv,
            })?;
        }
    }
    Ok(())
}

/// Converts coefficients to `f32`.
///
/// ```
/// use sdrk_blocks::design::to_f32_sos;
/// let mut dst = [[0.0f32; 5]; 1];
/// to_f32_sos(&[[0.5, 0.25, 0.0, -1.5, 0.75]], &mut dst)?;
/// assert_eq!(dst, [[0.5, 0.25, 0.0, -1.5, 0.75]]);
/// assert!(to_f32_sos(&[], &mut dst).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
pub fn to_f32_sos(src: &[[f64; 5]], dst: &mut [[f32; 5]]) -> Result<()> {
    check_len(src.len(), dst.len())?;
    for (d, s) in dst.iter_mut().zip(src) {
        for (dv, &sv) in d.iter_mut().zip(s) {
            *dv = sv as f32;
        }
    }
    Ok(())
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

/// The smallest roll-off of [`rrc`].
pub const RRC_MIN_ROLLOFF: f64 = 0.05;
/// The smallest span of [`rrc`] in symbols.
pub const RRC_MIN_SPAN: usize = 4;
/// The largest span of [`rrc`] in symbols.
pub const RRC_MAX_SPAN: usize = 32;
/// The largest samples per symbol of [`rrc`] (D600).
pub const RRC_MAX_SPS: usize = 1024;

/// The singular points `|t| = 1 / (4 rolloff)` of the root-raised-cosine formula are detected
/// with this tolerance on `|1 - (4 rolloff t)^2|` (the golden generator uses the same rule).
const RRC_SINGULAR: f64 = 1e-9;

/// Tap count of [`rrc`]: `span sps + 1`, after the checks of the parameters.
///
/// ```
/// use sdrk_blocks::design::rrc_len;
/// assert_eq!(rrc_len(8, 10)?, 81);
/// assert!(rrc_len(1, 10).is_err());
/// assert!(rrc_len(8, 3).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
pub fn rrc_len(sps: usize, span: usize) -> Result<usize> {
    if !(2..=RRC_MAX_SPS).contains(&sps) {
        return Err(Error::OutOfRange {
            name: "samples_per_symbol",
            min: 2.0,
            max: RRC_MAX_SPS as f64,
            got: sps as f64,
        });
    }
    if !(RRC_MIN_SPAN..=RRC_MAX_SPAN).contains(&span) {
        return Err(Error::OutOfRange {
            name: "span",
            min: RRC_MIN_SPAN as f64,
            max: RRC_MAX_SPAN as f64,
            got: span as f64,
        });
    }
    Ok(span * sps + 1)
}

/// The root-raised-cosine design of [`rrc`], one tap at a time (so the blocks need no `f64`
/// buffer). `tap(k)` equals `rrc(..)[k]` bit for bit.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Rrc {
    rolloff: f64,
    sps: usize,
    n: usize,
    /// `1 / sqrt(sum h^2)` of the unnormalized taps.
    norm: f64,
}

impl Rrc {
    /// Checks the parameters and computes the energy of the unnormalized taps.
    pub(crate) fn new(rolloff: f64, sps: usize, span: usize) -> Result<Self> {
        if !(rolloff.is_finite() && (RRC_MIN_ROLLOFF..=1.0).contains(&rolloff)) {
            return Err(Error::OutOfRange {
                name: "rolloff",
                min: RRC_MIN_ROLLOFF,
                max: 1.0,
                got: rolloff,
            });
        }
        let n = rrc_len(sps, span)?;
        let mut d = Self {
            rolloff,
            sps,
            n,
            norm: 1.0,
        };
        let mut energy = 0.0f64;
        for k in 0..n {
            let h = d.unscaled(k);
            energy += h * h;
        }
        d.norm = 1.0 / libm::sqrt(energy);
        Ok(d)
    }

    /// The number of taps.
    pub(crate) fn len(&self) -> usize {
        self.n
    }

    /// The impulse response at tap `k` before the normalization, for a symbol period of 1.
    #[allow(clippy::many_single_char_names)]
    fn unscaled(&self, k: usize) -> f64 {
        let b = self.rolloff;
        let t = (k as f64 - (self.n - 1) as f64 / 2.0) / self.sps as f64;
        if t == 0.0 {
            return (1.0 - b) + 4.0 * b / PI;
        }
        let x = 4.0 * b * t;
        let den = 1.0 - x * x;
        if libm::fabs(den) < RRC_SINGULAR {
            let a = PI / (4.0 * b);
            return (b / core::f64::consts::SQRT_2)
                * ((1.0 + 2.0 / PI) * libm::sin(a) + (1.0 - 2.0 / PI) * libm::cos(a));
        }
        let num = libm::sin(PI * t * (1.0 - b)) + x * libm::cos(PI * t * (1.0 + b));
        num / (PI * t * den)
    }

    /// Tap `k`, normalized to unit energy.
    pub(crate) fn tap(&self, k: usize) -> f64 {
        self.unscaled(k) * self.norm
    }
}

/// Root-raised-cosine taps (M40): the standard impulse response for a symbol period of `sps`
/// samples, truncated to `span` symbols (`span sps + 1` taps, centred), normalized to unit
/// energy (`sum h^2 = 1`).
///
/// With `t` the time from the centre in symbols and `b = rolloff`:
/// `h(t) = (sin(pi t (1 - b)) + 4 b t cos(pi t (1 + b))) / (pi t (1 - (4 b t)^2))`, with the
/// closed forms `h(0) = 1 - b + 4 b / pi` and, where `|1 - (4 b t)^2| < 1e-9` (the points
/// `|t| = 1 / (4 b)`), `h = b / sqrt(2) ((1 + 2 / pi) sin(pi / (4 b)) + (1 - 2 / pi) cos(pi / (4 b)))`.
/// `rolloff` 0.05 to 1, `sps` 2 to 1024, `span` 4 to 32 ([`OutOfRange`](Error::OutOfRange)
/// otherwise); `taps.len()` must be [`rrc_len`]`(sps, span)`.
///
/// ```
/// use sdrk_blocks::design::{rrc, rrc_len};
/// let mut h = vec![0.0f64; rrc_len(8, 8)?];
/// rrc(0.5, 8, 8, &mut h)?;
/// assert!((h.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12);
/// assert_eq!(h[0], h[64]); // symmetric
/// assert!(h[32] > h[31]); // the peak is in the centre
/// assert!(rrc(0.01, 8, 8, &mut h).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
pub fn rrc(rolloff: f64, sps: usize, span: usize, taps: &mut [f64]) -> Result<()> {
    let d = Rrc::new(rolloff, sps, span)?;
    check_len(d.len(), taps.len())?;
    for (k, t) in taps.iter_mut().enumerate() {
        *t = d.tap(k);
    }
    Ok(())
}

/// The largest stop-band edge of [`channelizer_prototype`] in channel spacings: the largest edge
/// that does not alias into the passband at the oversampled output rate `2 fs / M`.
pub const CHANNELIZER_MAX_STOPBAND: f64 = 1.5;
/// The smallest stop-band attenuation of [`channelizer_prototype`] in dB.
pub const CHANNELIZER_MIN_ATTENUATION_DB: f64 = 20.0;
/// The largest stop-band attenuation of [`channelizer_prototype`] in dB.
pub const CHANNELIZER_MAX_ATTENUATION_DB: f64 = 120.0;

/// Checks the edges and the attenuation of a channelizer prototype.
fn check_channelizer_edges(passband: f64, stopband: f64, attenuation_db: f64) -> Result<()> {
    if !(passband.is_finite() && passband > 0.0) {
        return Err(Error::InvalidParameter {
            name: "passband",
            reason: "the passband edge must be a finite fraction of the channel spacing above 0",
        });
    }
    if !(stopband.is_finite() && stopband > passband && stopband <= CHANNELIZER_MAX_STOPBAND) {
        return Err(Error::InvalidParameter {
            name: "stopband",
            reason: "the stopband edge must lie above the passband edge and at most at 1.5 channel spacings",
        });
    }
    if !(CHANNELIZER_MIN_ATTENUATION_DB..=CHANNELIZER_MAX_ATTENUATION_DB).contains(&attenuation_db)
    {
        return Err(Error::OutOfRange {
            name: "attenuation_db",
            min: CHANNELIZER_MIN_ATTENUATION_DB,
            max: CHANNELIZER_MAX_ATTENUATION_DB,
            got: attenuation_db,
        });
    }
    Ok(())
}

/// Checks the channel count of a channelizer: a power of two from 4 to 256.
fn check_channelizer_channels(channels: usize) -> Result<()> {
    use sdrk_core::kernels::channelize::{MAX_CHANNELS, MIN_CHANNELS};
    if channels.is_power_of_two() && (MIN_CHANNELS..=MAX_CHANNELS).contains(&channels) {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name: "channels",
            reason: "the channel count must be a power of two from 4 to 256",
        })
    }
}

/// The Kaiser tap estimate of a channelizer prototype: the number of taps that
/// `scipy.signal.kaiserord(attenuation_db, width)` returns for the transition from `passband` to
/// `stopband` channel spacings, `width = 2 (stopband - passband) / M` (the tap count as `SciPy`
/// returns it, not forced odd): `ceil((A - 7.95) / 2.285 / (pi width) + 1)`.
///
/// ```
/// use sdrk_blocks::design::channelizer_taps;
/// // 16 channels, 60 dB: the search grid of 0.5 / 1.0 and the sharp PMR446 plan of 0.4 / 0.6.
/// assert_eq!(channelizer_taps(16, 0.5, 1.0, 60.0)?, 118);
/// assert_eq!(channelizer_taps(16, 0.4, 0.6, 60.0)?, 292);
/// assert!(channelizer_taps(16, 0.6, 0.4, 60.0).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
///
/// # Errors
/// `InvalidParameter` for a channel count that is not a power of two from 4 to 256 or for edges
/// outside `0 < passband < stopband <= 1.5`; `OutOfRange` for an attenuation outside 20 to
/// 120 dB.
pub fn channelizer_taps(
    channels: usize,
    passband: f64,
    stopband: f64,
    attenuation_db: f64,
) -> Result<usize> {
    check_channelizer_channels(channels)?;
    check_channelizer_edges(passband, stopband, attenuation_db)?;
    let width = 2.0 * (stopband - passband) / channels as f64;
    let num = (attenuation_db - 7.95) / 2.285 / (PI * width) + 1.0;
    // A transition band of almost zero width needs an absurd number of taps.
    if !(num.is_finite() && num <= TAP_LIMIT as f64) {
        return Err(Error::InvalidParameter {
            name: "stopband",
            reason: "the transition band is too narrow for any channelizer prototype",
        });
    }
    Ok(libm::ceil(num) as usize)
}

/// The prototype taps per branch `L` that the Kaiser estimate of [`channelizer_taps`] needs:
/// `ceil(channelizer_taps(..) / M)`.
///
/// ```
/// use sdrk_blocks::design::channelizer_taps_per_branch;
/// assert_eq!(channelizer_taps_per_branch(16, 0.5, 1.0, 60.0)?, 8);
/// assert_eq!(channelizer_taps_per_branch(16, 0.4, 0.6, 60.0)?, 19);
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
///
/// # Errors
/// As [`channelizer_taps`].
pub fn channelizer_taps_per_branch(
    channels: usize,
    passband: f64,
    stopband: f64,
    attenuation_db: f64,
) -> Result<usize> {
    Ok(channelizer_taps(channels, passband, stopband, attenuation_db)?.div_ceil(channels))
}

/// The streaming design of [`channelizer_prototype`] (one tap at a time, no `f64` buffer).
pub(crate) fn channelizer_lowpass(
    channels: usize,
    taps_per_branch: usize,
    passband: f64,
    stopband: f64,
    attenuation_db: f64,
) -> Result<Lowpass> {
    use sdrk_core::kernels::channelize::{MAX_TAPS_PER_BRANCH, MIN_TAPS_PER_BRANCH};
    check_channelizer_channels(channels)?;
    if !(MIN_TAPS_PER_BRANCH..=MAX_TAPS_PER_BRANCH).contains(&taps_per_branch) {
        return Err(Error::InvalidParameter {
            name: "taps_per_branch",
            reason: "a channelizer needs 4 to 32 taps per branch",
        });
    }
    check_channelizer_edges(passband, stopband, attenuation_db)?;
    let cutoff = (passband + stopband) / (2.0 * channels as f64);
    Lowpass::new(
        cutoff,
        1.0,
        DesignWindow::Kaiser {
            beta: kaiser_beta(attenuation_db),
        },
        channels * taps_per_branch,
    )
}

/// The prototype low-pass of the channelizer (`sdrk_core::kernels::channelize`): `M L` taps
/// (`M = channels`, a power of two from 4 to 256, `L = taps_per_branch`, 4 to 32;
/// `taps.len()` must be `M L`), a Kaiser-windowed low-pass with the passband edge at
/// `passband fs / M` and the stopband from `stopband fs / M` (both fractions of the channel
/// spacing `fs / M`, `0 < passband < stopband <= 1.5`; 1.5 is the largest edge that does not
/// alias into the passband at the output rate `2 fs / M`), the stop-band attenuation
/// `attenuation_db` (20 to 120, 60 is the usual value), normalized to a DC gain of 1.
///
/// Equal to `scipy.signal.firwin(M L, (passband + stopband) / (2 M), window=('kaiser',
/// kaiser_beta(attenuation_db)), fs=1)` within 1e-12: the cutoff (the -6 dB point) lies in the
/// middle of the transition band. The tap count is a parameter; [`channelizer_taps`] says how
/// many the edges need. The search grid of a band (edges 0.5 / 1.0) passes half a spacing on each
/// side of a channel centre flat, so neighbouring channels overlap by half a spacing; a channel
/// plan (PMR446: 0.4 / 0.6) rejects its neighbours sharply.
///
/// ```
/// use sdrk_blocks::design::channelizer_prototype;
/// let mut h = vec![0.0f64; 16 * 8];
/// channelizer_prototype(16, 8, 0.5, 1.0, 60.0, &mut h)?;
/// assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// assert!(channelizer_prototype(16, 8, 1.0, 0.5, 60.0, &mut h).is_err()); // edges swapped
/// assert!(channelizer_prototype(16, 8, 0.5, 1.6, 60.0, &mut h).is_err()); // above 1.5
/// assert!(channelizer_prototype(16, 8, 0.5, 1.0, 60.0, &mut [0.0; 100]).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
///
/// # Errors
/// `InvalidParameter` for a channel count, a tap count per branch or edges outside the ranges;
/// `OutOfRange` for the attenuation; `Core(LengthMismatch)` if `taps.len()` is not `M L`.
pub fn channelizer_prototype(
    channels: usize,
    taps_per_branch: usize,
    passband: f64,
    stopband: f64,
    attenuation_db: f64,
    taps: &mut [f64],
) -> Result<()> {
    let lp = channelizer_lowpass(
        channels,
        taps_per_branch,
        passband,
        stopband,
        attenuation_db,
    )?;
    check_len(lp.len(), taps.len())?;
    for (i, t) in taps.iter_mut().enumerate() {
        *t = lp.tap(i);
    }
    Ok(())
}
