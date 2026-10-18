//! Fused kernels: receive and transmit chains that a backend may implement in one pass.
//!
//! Each trait has a generic chaining function in [`crate::fused`]; `Fallback`, `Reference`
//! and the facade implement the trait by calling it with the best single kernels. A
//! backend may provide a truly fused version, which must be bit-identical to the chained
//! result of the same backend's single kernels.

use crate::error::Result;
use crate::kernels::decim::FirDecimState;
use crate::kernels::demod::{DeemphasisState, QuadDemodState};
use crate::kernels::interp::FirInterpState;
use crate::kernels::mix::Nco;
use crate::timing::Progress;
use crate::types::ComplexSample;

/// Digital down-converter: `rotate` followed by `fir_decimate`.
///
/// ```
/// use sdrk_core::{kernels::{decim::FirDecimState, fused::Ddc, mix::Nco}, Fallback, C32};
/// // Mix a tone at a quarter of the sample rate to DC and decimate by 2 (one tap).
/// let mut nco = Nco::from_increment(3 << 30); // -fs/4
/// let mut fir = FirDecimState::<C32, _, _>::new([1.0f32], [], 2)?;
/// let input: Vec<C32> = (0..8)
///     .map(|n| C32::new([1.0, 0.0, -1.0, 0.0][n % 4], [0.0, 1.0, 0.0, -1.0][n % 4]))
///     .collect();
/// let (mut output, mut scratch) = ([C32::default(); 4], [C32::default(); 8]);
/// let p = Fallback.ddc(&mut nco, &mut fir, &input, &mut output, &mut scratch)?;
/// assert_eq!((p.consumed, p.produced), (8, 4));
/// assert!(output.iter().all(|s| (s.re - 1.0).abs() < 1e-5 && s.im.abs() < 1e-5));
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Ddc<T: ComplexSample> {
    /// Rotates `input` by the oscillator and decimates it. `scratch` holds the rotated
    /// samples; the call processes `min(input.len(), scratch.len(), fir.max_input_for_output(output.len()))`
    /// input samples, so the oscillator phase stays exact when the output space is limited.
    /// Returns the consumed and produced counts; the caller resubmits unconsumed input.
    fn ddc<Taps, Hist>(
        &self,
        nco: &mut Nco,
        fir: &mut FirDecimState<T, Taps, Hist>,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<Progress>
    where
        Taps: AsRef<[T::Real]>,
        Hist: AsRef<[T]> + AsMut<[T]>;
}

/// Digital up-converter: `fir_interpolate` followed by `rotate`.
///
/// ```
/// use sdrk_core::{kernels::{fused::Duc, interp::FirInterpState, mix::Nco}, Fallback, C32};
/// // Zero-order hold by 2, then mix up by fs/4.
/// let mut nco = Nco::from_increment(1 << 30);
/// let mut fir = FirInterpState::<C32, _, _>::new([1.0f32, 1.0], [C32::default(); 1], 2)?;
/// let input = [C32::new(1.0, 0.0); 4];
/// let (mut output, mut scratch) = ([C32::default(); 8], [C32::default(); 8]);
/// let p = Fallback.duc(&mut fir, &mut nco, &input, &mut output, &mut scratch)?;
/// assert_eq!((p.consumed, p.produced), (4, 8));
/// assert!((output[1].im - 1.0).abs() < 1e-5);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Duc<T: ComplexSample> {
    /// Interpolates as many whole input samples as `output` (and `scratch`) can take, then
    /// rotates the interpolated samples into `output`. Returns the consumed and produced
    /// counts; the caller resubmits unconsumed input.
    fn duc<Taps, Hist>(
        &self,
        fir: &mut FirInterpState<T, Taps, Hist>,
        nco: &mut Nco,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<Progress>
    where
        Taps: AsRef<[T::Real]>,
        Hist: AsRef<[T]> + AsMut<[T]>;
}

/// FM demodulator: `quad_demod` followed by `deemphasis`.
///
/// ```
/// use sdrk_core::{kernels::{demod::{DeemphasisState, QuadDemodState}, fused::FmDemod}, Fallback, Rate, C32};
/// let mut quad = QuadDemodState::<C32>::new();
/// let mut deemph = DeemphasisState::<f32>::new(1e-9, Rate::hz(1000)?)?; // almost no filtering
/// let input = [C32::new(1.0, 0.0), C32::new(0.0, 1.0), C32::new(-1.0, 0.0)];
/// let (mut audio, mut scratch) = ([0.0f32; 3], [0.0f32; 3]);
/// Fallback.fm_demod(&mut quad, &mut deemph, &input, &mut audio, &mut scratch, 1.0)?;
/// assert!((audio[1] - core::f32::consts::FRAC_PI_2).abs() < 1e-5);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait FmDemod<T: ComplexSample> {
    /// Demodulates `input` into `output_audio` (`output_audio.len() == input.len()`).
    /// `scratch_real` holds the discriminator output; a shorter scratch makes the kernel work
    /// in several pieces, a scratch of length 0 is `BufferTooSmall` for a non-empty input.
    fn fm_demod(
        &self,
        quad: &mut QuadDemodState<T>,
        deemph: &mut DeemphasisState<T::Real>,
        input: &[T],
        output_audio: &mut [T::Real],
        scratch_real: &mut [T::Real],
        gain: T::Real,
    ) -> Result<()>;
}
