//! Chaining functions of the fused kernels.
//!
//! Each function builds a fused kernel from the single kernels of any implementation `k`
//! (`Fallback`, `Reference`, the facade or a backend). The traits live in
//! [`kernels::fused`](crate::kernels::fused).

use crate::error::{Error, Result};
use crate::kernels::decim::{FirDecimState, FirDecimate};
use crate::kernels::demod::{Deemphasis, DeemphasisState, QuadDemod, QuadDemodState};
use crate::kernels::interp::{FirInterpState, FirInterpolate};
use crate::kernels::mix::{Nco, Rotate};
use crate::timing::Progress;
use crate::types::ComplexSample;

fn split<T>(slice: &[T], n: usize) -> Result<&[T]> {
    slice.get(..n).ok_or(Error::Internal("fused split"))
}

fn split_mut<T>(slice: &mut [T], n: usize) -> Result<&mut [T]> {
    slice.get_mut(..n).ok_or(Error::Internal("fused split"))
}

/// Digital down-converter built from `rotate` and `fir_decimate`.
///
/// It processes `n_in = min(input.len(), scratch.len(), fir.max_input_for_output(output.len()))`
/// samples: `input[..n_in]` is rotated into `scratch[..n_in]` (the oscillator advances by exactly
/// `n_in`), then decimated into `output`. The decimator must consume all `n_in` samples
/// (`Internal` otherwise). This keeps the oscillator phase exact when the output space is
/// limited. Returns `Progress { consumed: n_in, produced }`.
///
/// ```
/// use sdrk_core::{fused::ddc_chained, kernels::{decim::FirDecimState, mix::Nco}, Fallback, C32};
/// let mut nco = Nco::from_increment(0);
/// let mut fir = FirDecimState::<C32, _, _>::new([1.0f32], [], 3)?;
/// let input = [C32::new(1.0, 0.0); 9];
/// let (mut output, mut scratch) = ([C32::default(); 2], [C32::default(); 9]);
/// // Two outputs fit: the call stops after 6 of the 9 input samples.
/// let p = ddc_chained(&Fallback, &mut nco, &mut fir, &input, &mut output, &mut scratch)?;
/// assert_eq!((p.consumed, p.produced), (6, 2));
/// assert_eq!(nco.phase(), 0);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub fn ddc_chained<T, K, Taps, Hist>(
    k: &K,
    nco: &mut Nco,
    fir: &mut FirDecimState<T, Taps, Hist>,
    input: &[T],
    output: &mut [T],
    scratch: &mut [T],
) -> Result<Progress>
where
    T: ComplexSample,
    K: Rotate<T> + FirDecimate<T>,
    Taps: AsRef<[T::Real]>,
    Hist: AsRef<[T]> + AsMut<[T]>,
{
    let n_in = fir
        .max_input_for_output(output.len())
        .min(input.len())
        .min(scratch.len());
    let rotated = split_mut(scratch, n_in)?;
    k.rotate(nco, split(input, n_in)?, rotated)?;
    let p = k.fir_decimate(fir, rotated, output)?;
    if p.consumed != n_in {
        return Err(Error::Internal(
            "decimator did not consume the rotated block",
        ));
    }
    Ok(p)
}

/// Digital up-converter built from `fir_interpolate` and `rotate`.
///
/// It interpolates `min(input.len(), output.len() / L, scratch.len() / L)` whole input
/// samples into `scratch` and rotates the interpolated samples into `output`. Returns
/// `Progress { consumed, produced }` of the interpolator.
///
/// ```
/// use sdrk_core::{fused::duc_chained, kernels::{interp::FirInterpState, mix::Nco}, Fallback, C32};
/// let mut nco = Nco::from_increment(0);
/// let mut fir = FirInterpState::<C32, _, _>::new([1.0f32; 4], [C32::default(); 1], 4)?;
/// let input = [C32::new(1.0, 0.0); 5];
/// let (mut output, mut scratch) = ([C32::default(); 10], [C32::default(); 20]);
/// // The output holds 10 samples, which is 2 whole input samples.
/// let p = duc_chained(&Fallback, &mut fir, &mut nco, &input, &mut output, &mut scratch)?;
/// assert_eq!((p.consumed, p.produced), (2, 8));
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub fn duc_chained<T, K, Taps, Hist>(
    k: &K,
    fir: &mut FirInterpState<T, Taps, Hist>,
    nco: &mut Nco,
    input: &[T],
    output: &mut [T],
    scratch: &mut [T],
) -> Result<Progress>
where
    T: ComplexSample,
    K: Rotate<T> + FirInterpolate<T>,
    Taps: AsRef<[T::Real]>,
    Hist: AsRef<[T]> + AsMut<[T]>,
{
    let l = fir.factor();
    let n_in = input.len().min(output.len() / l).min(scratch.len() / l);
    let room = split_mut(scratch, n_in.saturating_mul(l))?;
    let p = k.fir_interpolate(fir, split(input, n_in)?, room)?;
    let upsampled = split(room, p.produced)?;
    k.rotate(nco, upsampled, split_mut(output, p.produced)?)?;
    Ok(p)
}

/// FM demodulator built from `quad_demod` and `deemphasis`.
///
/// `output_audio.len()` must equal `input.len()`. The input is processed in pieces of
/// `scratch_real.len()` samples; both states carry over between pieces, so the result does not
/// depend on the scratch size. A scratch of length 0 with a non-empty input is `BufferTooSmall`.
///
/// ```
/// use sdrk_core::{fused::fm_demod_chained, kernels::demod::{DeemphasisState, QuadDemodState}, Fallback, Rate, C32};
/// let mut quad = QuadDemodState::<C32>::new();
/// let mut deemph = DeemphasisState::<f32>::new(50e-6, Rate::hz(240_000)?)?;
/// let input = [C32::new(1.0, 0.0); 4];
/// let (mut audio, mut scratch) = ([1.0f32; 4], [0.0f32; 2]);
/// fm_demod_chained(&Fallback, &mut quad, &mut deemph, &input, &mut audio, &mut scratch, 1.0)?;
/// assert_eq!(audio, [0.0; 4]); // a constant carrier has no deviation
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub fn fm_demod_chained<T, K>(
    k: &K,
    quad: &mut QuadDemodState<T>,
    deemph: &mut DeemphasisState<T::Real>,
    input: &[T],
    output_audio: &mut [T::Real],
    scratch_real: &mut [T::Real],
    gain: T::Real,
) -> Result<()>
where
    T: ComplexSample,
    K: QuadDemod<T> + Deemphasis<T::Real>,
{
    if output_audio.len() != input.len() {
        return Err(Error::LengthMismatch {
            expected: input.len(),
            got: output_audio.len(),
        });
    }
    if input.is_empty() {
        return Ok(());
    }
    if scratch_real.is_empty() {
        return Err(Error::BufferTooSmall { needed: 1, got: 0 });
    }
    for (i, a) in input
        .chunks(scratch_real.len())
        .zip(output_audio.chunks_mut(scratch_real.len()))
    {
        let s = split_mut(scratch_real, i.len())?;
        k.quad_demod(quad, i, s, gain)?;
        k.deemphasis(deemph, s, a)?;
    }
    Ok(())
}
