//! Comparison metrics: SNR, error and bit-exact comparison.

use num_complex::Complex64;
use sdrk_core::Sample;

/// Result of comparing an actual signal with the expected one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Comparison {
    /// `10 log10(sum |ref|^2 / sum |ref - out|^2)` over the compared samples;
    /// `+inf` when the error is zero, `-inf` when only the signal energy is
    /// zero.
    pub snr_db: f64,
    /// Largest absolute error.
    pub max_abs_err: f64,
    /// Root mean square of the error.
    pub rms_err: f64,
    /// Absolute index of the first sample with a non-zero (or NaN) error.
    pub first_mismatch: Option<usize>,
    /// Number of compared samples (`skip..`, at most the shorter length).
    pub compared: usize,
}

#[derive(Default)]
struct Acc {
    sig: f64,
    err: f64,
    max: f64,
    first: Option<usize>,
    n: usize,
}

impl Acc {
    fn add(&mut self, index: usize, sig: f64, err_sq: f64, abs_err: f64) {
        self.sig += sig;
        self.err += err_sq;
        self.n += 1;
        // NaN counts as a mismatch.
        if abs_err.is_nan() || abs_err > 0.0 {
            if self.first.is_none() {
                self.first = Some(index);
            }
            if abs_err.is_nan() || abs_err > self.max {
                self.max = abs_err;
            }
        }
    }

    fn finish(self) -> Comparison {
        let snr_db = if self.err == 0.0 {
            f64::INFINITY
        } else if self.sig == 0.0 {
            f64::NEG_INFINITY
        } else {
            10.0 * libm::log10(self.sig / self.err)
        };
        let rms_err = if self.n == 0 {
            0.0
        } else {
            libm::sqrt(self.err / self.n as f64)
        };
        Comparison {
            snr_db,
            max_abs_err: self.max,
            rms_err,
            first_mismatch: self.first,
            compared: self.n,
        }
    }
}

/// Compares complex samples from index `skip` on. Stops at the shorter of
/// `expected` and `actual`.
///
/// ```
/// use num_complex::Complex64;
/// use sdrk_testkit::metrics::compare_complex;
/// let e = [Complex64::new(1.0, 0.0); 4];
/// let c = compare_complex(&e, e.iter().copied(), 0);
/// assert_eq!(c.snr_db, f64::INFINITY);
/// assert_eq!(c.first_mismatch, None);
/// ```
pub fn compare_complex(
    expected: &[Complex64],
    actual: impl Iterator<Item = Complex64>,
    skip: usize,
) -> Comparison {
    let mut acc = Acc::default();
    for (index, (e, a)) in expected.iter().zip(actual).enumerate().skip(skip) {
        let d = e - a;
        let err_sq = d.re * d.re + d.im * d.im;
        acc.add(index, e.re * e.re + e.im * e.im, err_sq, libm::sqrt(err_sq));
    }
    acc.finish()
}

/// Compares real samples from index `skip` on. Stops at the shorter of
/// `expected` and `actual`.
pub fn compare_real(
    expected: &[f64],
    actual: impl Iterator<Item = f64>,
    skip: usize,
) -> Comparison {
    let mut acc = Acc::default();
    for (index, (e, a)) in expected.iter().zip(actual).enumerate().skip(skip) {
        let d = e - a;
        acc.add(index, e * e, d * d, libm::fabs(d));
    }
    acc.finish()
}

/// Index of the first element whose raw bytes differ, or `None` if the slices
/// are bit-identical. Slices of different length differ at the length of the
/// shorter one.
///
/// ```
/// use sdrk_core::Q15;
/// use sdrk_testkit::metrics::bit_exact;
/// assert_eq!(bit_exact(&[Q15(1), Q15(2)], &[Q15(1), Q15(3)]), Some(1));
/// assert_eq!(bit_exact(&[0.0f32], &[-0.0f32]), Some(0));
/// ```
pub fn bit_exact<T: Sample>(a: &[T], b: &[T]) -> Option<usize> {
    let differs = a
        .iter()
        .zip(b)
        .position(|(x, y)| bytemuck::bytes_of(x) != bytemuck::bytes_of(y));
    match differs {
        Some(i) => Some(i),
        None if a.len() != b.len() => Some(a.len().min(b.len())),
        None => None,
    }
}
