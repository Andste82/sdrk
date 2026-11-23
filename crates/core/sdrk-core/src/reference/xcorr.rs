//! Reference implementation of `xcorr_fft`: the direct convolution `y[n] = sum_k h[k] x[n - k]`
//! in f64, with the block boundaries and the history of the overlap-save driver
//! ([`xcorr_fft_with`]), so a call consumes and produces what the fallback does.
//!
//! The state holds only the spectrum `H`, not the taps (a state from `from_spectrum` has none).
//! The reference recovers the taps once per call as `h[k] = (1/N) sum_j H[j] e^(+j 2 pi j k / N)`
//! for `k < L` in f64 (the twiddle of `j k mod N` from `libm::cos`/`libm::sin`) and keeps them in
//! the scratch as two `C32` parts, `hi = round(h)` in `scratch[..L]` and `lo = round(h - hi)` in
//! `scratch[L..2 L]` (`N >= 2 L`), so the taps keep about 48 bits. For a state of `new` they are
//! the taps up to the `f32` error of the fallback FFT of `H`. Each output is then the f64 sum of
//! `(hi + lo) w` over the `L` window samples, rounded once to `C32`. Cost: `L N` for the taps per
//! call and `L` per output (clarity over speed; the reference serves a few thousand outputs per
//! case).

use num_complex::Complex64;

use crate::Reference;
use crate::error::{Error, Result};
use crate::kernels::xcorr::{XcorrFft, XcorrFftState, xcorr_fft_with};
use crate::timing::Progress;
use crate::types::{C32, ComplexSample};

/// `e^(+j 2 pi ((k n) mod N) / N)` in f64.
fn inverse_twiddle(k: usize, n: usize, size: usize) -> Complex64 {
    let idx = (k as u64 * n as u64) % size as u64;
    let theta = 2.0 * core::f64::consts::PI * idx as f64 / size as f64;
    Complex64::new(libm::cos(theta), libm::sin(theta))
}

fn internal() -> Error {
    Error::Internal("reference xcorr_fft index")
}

/// Writes the taps recovered from `spectrum` into `scratch` (`hi` in `..taps`, `lo` in
/// `taps..2 taps`).
fn recover_taps(spectrum: &[C32], taps: usize, scratch: &mut [C32]) -> Result<()> {
    let n = spectrum.len();
    let (hi, rest) = scratch.split_at_mut_checked(taps).ok_or_else(internal)?;
    let lo = rest.get_mut(..taps).ok_or_else(internal)?;
    let g = 1.0 / n as f64;
    for (k, (h, l)) in hi.iter_mut().zip(lo.iter_mut()).enumerate() {
        let mut acc = Complex64::new(0.0, 0.0);
        for (j, x) in spectrum.iter().enumerate() {
            acc += x.to_c64() * inverse_twiddle(j, k, n);
        }
        let v = acc * g;
        *h = C32::from_c64_round(v);
        *l = C32::from_c64_round(v - h.to_c64());
    }
    Ok(())
}

impl XcorrFft<C32> for Reference {
    fn xcorr_fft<S, Sh, Sc, Tw>(
        &self,
        state: &mut XcorrFftState<C32, S, Sh, Sc, Tw>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<Progress>
    where
        S: AsRef<[C32]>,
        Sh: AsRef<[C32]> + AsMut<[C32]>,
        Sc: AsRef<[C32]> + AsMut<[C32]>,
        Tw: AsRef<[C32]>,
    {
        let taps = state.taps_len();
        let mut ready = false;
        xcorr_fft_with(
            state,
            input,
            output,
            |(history, x, spectrum, scratch, _, y)| {
                if !ready {
                    recover_taps(spectrum, taps, scratch)?;
                    ready = true;
                }
                let hi = scratch.get(..taps).ok_or_else(internal)?;
                let lo = scratch.get(taps..2 * taps).ok_or_else(internal)?;
                // Output `j` of the block is window position `L - 1 + j`:
                // `sum_k h[k] w[L - 1 + j - k] = sum_{i = j}^{j + L - 1} h[j + L - 1 - i] w[i]`.
                for (j, out) in y.iter_mut().enumerate() {
                    let window = history.iter().chain(x.iter()).skip(j).take(taps);
                    let mut acc = Complex64::new(0.0, 0.0);
                    for (w, (a, b)) in window.zip(hi.iter().zip(lo).rev()) {
                        acc += (a.to_c64() + b.to_c64()) * w.to_c64();
                    }
                    *out = C32::from_c64_round(acc);
                }
                Ok(())
            },
        )
    }
}
