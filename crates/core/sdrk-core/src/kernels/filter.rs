//! FIR filters.

use core::marker::PhantomData;

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::timing::{Delay, Timing};
use crate::types::ComplexSample;

/// FIR state with real taps and a history of the last `taps.len() - 1` input samples.
///
/// The caller provides the storage (`[T; N]`, `&mut [T]` or, in upper layers,
/// `Vec<T>`).
///
/// ```
/// use sdrk_core::{kernels::filter::FirState, C32};
/// let state = FirState::<C32, _, _>::new([0.5f32, 0.5], [C32::default(); 1])?;
/// assert_eq!(state.taps(), &[0.5, 0.5]);
/// assert!(FirState::<C32, _, _>::new([0.5f32, 0.5], [C32::default(); 2]).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct FirState<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>> {
    taps: Taps,
    hist: Hist,
    _t: PhantomData<T>,
}

impl<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>>
    FirState<T, Taps, Hist>
{
    /// Creates a state: `taps.len() >= 1` and `history.len() == taps.len() - 1`.
    /// The history is zeroed.
    pub fn new(taps: Taps, mut history: Hist) -> Result<Self> {
        let n = taps.as_ref().len();
        if n == 0 {
            return Err(Error::InvalidParameter("a filter needs at least one tap"));
        }
        let expected = n - 1;
        let got = history.as_ref().len();
        if got != expected {
            return Err(Error::LengthMismatch { expected, got });
        }
        history.as_mut().fill(T::ZERO);
        Ok(Self {
            taps,
            hist: history,
            _t: PhantomData,
        })
    }

    /// Clears the history (the taps are kept).
    pub fn reset(&mut self) {
        self.hist.as_mut().fill(T::ZERO);
    }

    /// The filter taps.
    pub fn taps(&self) -> &[T::Real] {
        self.taps.as_ref()
    }

    /// The history, oldest sample first (`taps().len() - 1` samples).
    pub fn history(&self) -> &[T] {
        self.hist.as_ref()
    }

    /// Taps and mutable history at once (for kernel implementations).
    pub(crate) fn parts_mut(&mut self) -> (&[T::Real], &mut [T]) {
        (self.taps.as_ref(), self.hist.as_mut())
    }
}

impl<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>> Timing
    for FirState<T, Taps, Hist>
{
    fn ratio(&self) -> Ratio {
        Ratio::ONE
    }

    /// `(N - 1) / 2` input samples for `N` taps (exact for symmetric taps).
    fn group_delay(&self) -> Delay {
        let n = self.taps.as_ref().len() as u64;
        Delay::new(n.saturating_sub(1), 2).unwrap_or(Delay::ZERO)
    }
}

/// Complex FIR filter with real taps.
///
/// ```
/// use sdrk_core::{kernels::filter::{Fir, FirState}, Fallback, C32};
/// let mut state = FirState::<C32, _, _>::new([0.5f32, 0.5], [C32::default(); 1])?;
/// let input = [C32::new(2.0, 0.0), C32::new(4.0, 2.0)];
/// let mut output = [C32::default(); 2];
/// Fallback.fir(&mut state, &input, &mut output)?;
/// assert_eq!(output, [C32::new(1.0, 0.0), C32::new(3.0, 1.0)]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Fir<T: ComplexSample> {
    /// `y[n] = Σ_{k=0}^{N−1} h[k] · x[n−k]`, where `x[n−k]` for `n−k < 0` comes from the history.
    /// The lengths must be equal. After the call the history holds the last `N−1` samples of
    /// (old history ++ input).
    fn fir<Taps, Hist>(
        &self,
        state: &mut FirState<T, Taps, Hist>,
        input: &[T],
        output: &mut [T],
    ) -> Result<()>
    where
        Taps: AsRef<[T::Real]>,
        Hist: AsRef<[T]> + AsMut<[T]>;
}

/// Sets the history to the last `hist.len()` samples of (old history ++ `input`).
pub(crate) fn update_history<T: Copy>(hist: &mut [T], input: &[T]) -> Result<()> {
    let internal = || Error::Internal("history update");
    let keep = hist.len();
    if input.len() >= keep {
        let tail = input.get(input.len() - keep..).ok_or_else(internal)?;
        hist.copy_from_slice(tail);
    } else {
        // Shift the old samples left and append the new ones.
        hist.rotate_left(input.len());
        let start = keep - input.len();
        let dst = hist.get_mut(start..).ok_or_else(internal)?;
        dst.copy_from_slice(input);
    }
    Ok(())
}
