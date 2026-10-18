//! Window generators.
//!
//! All windows are computed in f64 with `libm` and rounded into caller-provided
//! slices of `f32` or `Q15`. [`Symmetry::Symmetric`] uses the denominator `N - 1`;
//! [`Symmetry::Periodic`] computes the symmetric window of length `N + 1` and drops
//! the last value, like `scipy.signal.get_window(..., fftbins=True)`. A window of
//! length 1 is `[1.0]` (and an empty slice is left alone).

use core::f64::consts::PI;

use crate::error::{Error, Result};
use crate::types::RealSample;

/// Whether the window is symmetric (filter design) or periodic (spectral analysis).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Symmetry {
    /// Periodic: the symmetric window of length `N + 1` without its last value.
    Periodic,
    /// Symmetric: denominator `N - 1`.
    Symmetric,
}

/// Fills `out` with `f(n, D)` where `D` is the denominator of the chosen symmetry.
fn fill<R: RealSample>(out: &mut [R], sym: Symmetry, f: impl Fn(f64, f64) -> f64) {
    let n = out.len();
    let d = match sym {
        Symmetry::Periodic => n,
        Symmetry::Symmetric => n.saturating_sub(1),
    };
    for (i, o) in out.iter_mut().enumerate() {
        let v = if d == 0 { 1.0 } else { f(i as f64, d as f64) };
        *o = R::from_f64_round(v);
    }
}

/// Hann window `0.5 - 0.5 cos(2 pi n / D)`.
///
/// ```
/// use sdrk_core::window::{hann, Symmetry};
/// let mut w = [0.0f32; 4];
/// hann(&mut w, Symmetry::Periodic)?;
/// assert_eq!(w, [0.0, 0.5, 1.0, 0.5]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub fn hann<R: RealSample>(out: &mut [R], sym: Symmetry) -> Result<()> {
    fill(out, sym, |n, d| 0.5 - 0.5 * libm::cos(2.0 * PI * n / d));
    Ok(())
}

/// Four-term Blackman-Harris window with `a0 = 0.35875`, `a1 = 0.48829`, `a2 = 0.14128`,
/// `a3 = 0.01168`: `a0 - a1 cos(2 pi n / D) + a2 cos(4 pi n / D) - a3 cos(6 pi n / D)`.
///
/// ```
/// use sdrk_core::window::{blackman_harris, Symmetry};
/// let mut w = [0.0f32; 8];
/// blackman_harris(&mut w, Symmetry::Periodic)?;
/// assert!((w[0] - 6e-5).abs() < 1e-6 && (w[4] - 1.0).abs() < 1e-6);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub fn blackman_harris<R: RealSample>(out: &mut [R], sym: Symmetry) -> Result<()> {
    const A0: f64 = 0.35875;
    const A1: f64 = 0.48829;
    const A2: f64 = 0.14128;
    const A3: f64 = 0.01168;
    fill(out, sym, |n, d| {
        let x = 2.0 * PI * n / d;
        A0 - A1 * libm::cos(x) + A2 * libm::cos(2.0 * x) - A3 * libm::cos(3.0 * x)
    });
    Ok(())
}

/// Kaiser window `I0(beta sqrt(1 - (2 n / D - 1)^2)) / I0(beta)`. `beta` must be finite
/// and not negative, otherwise `InvalidParameter`.
///
/// ```
/// use sdrk_core::window::{kaiser, Symmetry};
/// let mut w = [0.0f32; 5];
/// kaiser(&mut w, 0.0, Symmetry::Symmetric)?; // beta 0 is rectangular
/// assert_eq!(w, [1.0; 5]);
/// assert!(kaiser(&mut w, -1.0, Symmetry::Symmetric).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub fn kaiser<R: RealSample>(out: &mut [R], beta: f64, sym: Symmetry) -> Result<()> {
    if !(beta.is_finite() && beta >= 0.0) {
        return Err(Error::InvalidParameter(
            "kaiser beta must be finite and not negative",
        ));
    }
    let norm = bessel_i0(beta);
    fill(out, sym, |n, d| {
        let t = 2.0 * n / d - 1.0;
        let arg = (1.0 - t * t).max(0.0);
        bessel_i0(beta * libm::sqrt(arg)) / norm
    });
    Ok(())
}

/// Modified Bessel function of the first kind and order 0, by the power series
/// `sum ((x / 2)^k / k!)^2`; the sum stops when a term is below `1e-17` times the
/// sum, after at most 500 terms.
///
/// ```
/// use sdrk_core::window::bessel_i0;
/// assert_eq!(bessel_i0(0.0), 1.0);
/// assert!((bessel_i0(1.0) - 1.266_065_877_752_008_4).abs() < 1e-15);
/// ```
pub fn bessel_i0(x: f64) -> f64 {
    let half = x / 2.0;
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    for k in 1..=500u32 {
        let f = half / f64::from(k);
        term *= f * f;
        sum += term;
        if term < 1e-17 * sum {
            break;
        }
    }
    sum
}

/// `(coherent gain, noise-equivalent bandwidth)` of a window:
/// `sum(w) / N` and `N sum(w^2) / sum(w)^2` (in bins), summed in f64. An empty window
/// gives `(0.0, 0.0)`; a window that sums to zero gives an infinite or NaN bandwidth.
///
/// ```
/// use sdrk_core::window::{hann, window_gain, Symmetry};
/// let mut w = [0.0f32; 4096];
/// hann(&mut w, Symmetry::Periodic)?;
/// let (cg, nebw) = window_gain(&w);
/// assert!((cg - 0.5).abs() < 1e-9 && (nebw - 1.5).abs() < 1e-9);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub fn window_gain(w: &[f32]) -> (f64, f64) {
    if w.is_empty() {
        return (0.0, 0.0);
    }
    let n = w.len() as f64;
    let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
    for &v in w {
        let v = f64::from(v);
        sum += v;
        sum_sq += v * v;
    }
    (sum / n, n * sum_sq / (sum * sum))
}
