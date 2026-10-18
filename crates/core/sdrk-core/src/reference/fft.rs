//! Reference implementation of the FFT kernels: the naive O(N^2) DFT in f64.
//!
//! The twiddle for the product `k * n` is `e^(-+j 2 pi ((k n) mod N) / N)` with the product
//! reduced modulo `N` in integers first, evaluated with `libm::cos`/`libm::sin`.
//!
//! A DFT needs the whole input for every output, so it cannot run in place without extra
//! memory. [`Reference::dft`] is the out-of-place form without a size limit; the in-place trait
//! method `Fft::fft` of `Reference` copies the input to a small stack buffer and therefore
//! accepts sizes up to [`REFERENCE_IN_PLACE_MAX`] only.

use num_complex::Complex64;

use crate::Reference;
use crate::error::{Error, Result};
use crate::kernels::check_len;
use crate::kernels::fft::{
    Direction, Fft, FftPlan, FftScale, RealFft, RealFftPlan, Scaling, check_real_lengths, log2_of,
};
use crate::types::{C32, ComplexSample, SampleType};

/// Largest size of the in-place `Fft::fft` of [`Reference`]; use [`Reference::dft`] above.
pub const REFERENCE_IN_PLACE_MAX: usize = 256;

/// `(cos, sin)` of `2 pi ((k n) mod N) / N`.
fn cos_sin(k: usize, n: usize, size: usize) -> (f64, f64) {
    let idx = (k as u64 * n as u64) % size as u64;
    let theta = 2.0 * core::f64::consts::PI * idx as f64 / size as f64;
    (libm::cos(theta), libm::sin(theta))
}

/// `sum x[n] e^(-+j 2 pi k n / N)` for one output index.
fn dft_bin<T: ComplexSample>(input: &[T], k: usize, inverse: bool) -> Complex64 {
    let size = input.len();
    let mut acc = Complex64::new(0.0, 0.0);
    for (n, x) in input.iter().enumerate() {
        let (c, s) = cos_sin(k, n, size);
        let w = Complex64::new(c, if inverse { s } else { -s });
        acc += x.to_c64() * w;
    }
    acc
}

impl Reference {
    /// Out-of-place DFT of `input` into `output` with the conventions of the FFT kernel
    /// (`docs/kernels.md`): `C32` forward is unnormalised (exponent 0), `C32` inverse is divided by
    /// `N`, `CQ15` is divided by `N` in both directions (the exponent is `log2 N`, also for
    /// `BlockFloat`, which the one exponent rule allows). The lengths must be equal and a power of
    /// two from 16 to 65536.
    ///
    /// ```
    /// use sdrk_core::kernels::fft::{Direction, Scaling};
    /// use sdrk_core::{Reference, C32};
    /// let mut x = [C32::default(); 16];
    /// x[0] = C32::new(1.0, 0.0);
    /// let mut y = [C32::default(); 16];
    /// Reference.dft(&x, &mut y, Direction::Forward, Scaling::Default)?;
    /// assert_eq!(y[5], C32::new(1.0, 0.0));
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn dft<T: ComplexSample>(
        &self,
        input: &[T],
        output: &mut [T],
        dir: Direction,
        scaling: Scaling,
    ) -> Result<FftScale> {
        let is_c32 = T::TYPE == SampleType::C32;
        if scaling == Scaling::BlockFloat && is_c32 {
            return Err(Error::Unsupported(
                "block floating point is not available for c32",
            ));
        }
        let log2n = log2_of(input.len())?;
        check_len(input.len(), output.len())?;
        let inverse = dir == Direction::Inverse;
        let divide = !is_c32 || inverse;
        let g = if divide {
            1.0 / input.len() as f64
        } else {
            1.0
        };
        for (k, out) in output.iter_mut().enumerate() {
            *out = T::from_c64_round(dft_bin(input, k, inverse) * g);
        }
        Ok(FftScale {
            exponent: if divide { log2n as i32 } else { 0 },
        })
    }
}

impl<T: ComplexSample> Fft<T> for Reference {
    /// In place for sizes up to [`REFERENCE_IN_PLACE_MAX`]; larger sizes are `Unsupported`
    /// (use [`Reference::dft`]).
    fn fft<Tw: AsRef<[T]>>(
        &self,
        plan: &FftPlan<T, Tw>,
        data: &mut [T],
        dir: Direction,
        scaling: Scaling,
    ) -> Result<FftScale> {
        check_len(plan.len(), data.len())?;
        if data.len() > REFERENCE_IN_PLACE_MAX {
            return Err(Error::Unsupported(
                "the in-place reference fft is limited to 256 points; use Reference::dft",
            ));
        }
        let mut copy = [T::ZERO; REFERENCE_IN_PLACE_MAX];
        let (head, _) = copy
            .split_at_mut_checked(data.len())
            .ok_or(Error::Internal("reference copy"))?;
        head.copy_from_slice(data);
        self.dft(head, data, dir, scaling)
    }
}

impl RealFft for Reference {
    fn rfft<Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>>(
        &self,
        plan: &RealFftPlan<Tw, Tw2>,
        input: &[f32],
        output: &mut [C32],
        scratch: &mut [C32],
    ) -> Result<FftScale> {
        let n = plan.len();
        check_real_lengths(n, input.len(), output.len(), scratch.len())?;
        for (k, out) in output.iter_mut().enumerate() {
            let mut acc = Complex64::new(0.0, 0.0);
            for (i, &x) in input.iter().enumerate() {
                let (c, s) = cos_sin(k, i, n);
                acc += Complex64::new(f64::from(x) * c, -f64::from(x) * s);
            }
            *out = C32::from_c64_round(acc);
        }
        Ok(FftScale { exponent: 0 })
    }

    fn irfft<Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>>(
        &self,
        plan: &RealFftPlan<Tw, Tw2>,
        input: &[C32],
        output: &mut [f32],
        scratch: &mut [C32],
    ) -> Result<FftScale> {
        let n = plan.len();
        check_real_lengths(n, output.len(), input.len(), scratch.len())?;
        let h = n / 2;
        for (i, out) in output.iter_mut().enumerate() {
            // Hermitian extension: bins 0 and N/2 count with their real part only, every
            // other bin k < N/2 appears twice (k and N - k), which doubles its real part.
            let mut acc = 0.0f64;
            for (k, x) in input.iter().enumerate() {
                let (c, s) = cos_sin(k, i, n);
                let term = f64::from(x.re) * c - f64::from(x.im) * s;
                acc += if k == 0 || k == h {
                    f64::from(x.re) * c
                } else {
                    2.0 * term
                };
            }
            *out = (acc / n as f64) as f32;
        }
        Ok(FftScale {
            exponent: n.trailing_zeros() as i32,
        })
    }
}
