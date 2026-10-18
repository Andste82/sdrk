//! Interpolating FIR kernels (transmit side): zero stuffing by `L` followed by a low-pass.

use sdrk_device::Ratio;

use crate::error::Result;
use crate::kernels::decim::{ResampleRealState, ResampleState};
use crate::timing::{Delay, Progress, Timing};
use crate::types::{ComplexSample, RealSample};

macro_rules! interp_state_api {
    () => {
        /// Clears the history and restarts the output counter (taps and factor are kept).
        pub fn reset(&mut self) {
            self.inner.reset();
        }

        /// The interpolation factor `L`.
        pub fn factor(&self) -> usize {
            self.inner.up()
        }

        /// The history of the last `ceil(N / L)` input samples, oldest first.
        pub fn history(&self) -> &[T] {
            self.inner.history()
        }

        /// Output phase of the underlying polyphase structure; it is `0` between whole
        /// input samples and negative while outputs of the last input sample are pending.
        pub fn phase(&self) -> i64 {
            self.inner.phase()
        }
    };
}

/// State of the interpolating FIR for complex signals: real taps designed at the *output*
/// rate (the interpolation gain `L` is included by the caller), the factor `L >= 1` and a
/// history of `ceil(N / L)` input samples.
///
/// The output is `scipy.signal.upfirdn(h, x, up=L, down=1)` restricted to the outputs that
/// depend on received input: every input sample yields exactly `L` outputs. It is the
/// [`Resample`](crate::kernels::decim::Resample) kernel with `up = L` and `down = 1`.
///
/// ```
/// use sdrk_core::{kernels::interp::{FirInterpolate, FirInterpState}, Fallback, C32};
/// // Zero-order hold by 3: three taps of 1.
/// let mut state = FirInterpState::<C32, _, _>::new([1.0f32; 3], [C32::default(); 1], 3)?;
/// let input = [C32::new(1.0, 0.0), C32::new(2.0, 0.0)];
/// let mut output = [C32::default(); 6];
/// let p = Fallback.fir_interpolate(&mut state, &input, &mut output)?;
/// assert_eq!((p.consumed, p.produced), (2, 6));
/// assert_eq!(output.map(|s| s.re), [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
/// assert_eq!(state.factor(), 3);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct FirInterpState<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>> {
    inner: ResampleState<T, Taps, Hist>,
}

impl<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>>
    FirInterpState<T, Taps, Hist>
{
    /// Creates a state: `taps.len() >= 1`, `factor >= 1` and
    /// `history.len() == ceil(taps.len() / factor)`. The history is zeroed.
    pub fn new(taps: Taps, history: Hist, factor: usize) -> Result<Self> {
        Ok(Self {
            inner: ResampleState::new(taps, history, factor, 1)?,
        })
    }

    /// The filter taps.
    pub fn taps(&self) -> &[T::Real] {
        self.inner.taps()
    }

    interp_state_api!();

    pub(crate) fn inner_mut(&mut self) -> &mut ResampleState<T, Taps, Hist> {
        &mut self.inner
    }
}

impl<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>> Timing
    for FirInterpState<T, Taps, Hist>
{
    /// `L / 1`.
    fn ratio(&self) -> Ratio {
        self.inner.ratio()
    }

    /// `(N - 1) / (2 L)` input samples for `N` taps, built as `Delay::new(N - 1, 2 * L)`.
    fn group_delay(&self) -> Delay {
        self.inner.group_delay()
    }
}

/// Interpolating FIR for complex signals with real taps.
///
/// Like [`Resample`](crate::kernels::decim::Resample) it consumes an input sample only
/// while the output has room for its `L` outputs; the caller resubmits unconsumed input.
pub trait FirInterpolate<T: ComplexSample> {
    /// Interpolates by `L`, see [`FirInterpState`] for the definition. Returns the consumed
    /// and produced counts.
    fn fir_interpolate<Taps, Hist>(
        &self,
        state: &mut FirInterpState<T, Taps, Hist>,
        input: &[T],
        output: &mut [T],
    ) -> Result<Progress>
    where
        Taps: AsRef<[T::Real]>,
        Hist: AsRef<[T]> + AsMut<[T]>;
}

/// State of the interpolating FIR for real signals (`f32` or `Q15` samples and taps).
///
/// The definition is the one of [`FirInterpState`].
///
/// ```
/// use sdrk_core::{kernels::interp::{FirInterpolateReal, FirInterpRealState}, Fallback};
/// let mut state = FirInterpRealState::<f32, _, _>::new([1.0f32, 1.0], [0.0f32; 1], 2)?;
/// let mut output = [0.0f32; 4];
/// let p = Fallback.fir_interpolate_real(&mut state, &[3.0, 4.0], &mut output)?;
/// assert_eq!((p.consumed, p.produced), (2, 4));
/// assert_eq!(output, [3.0, 3.0, 4.0, 4.0]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct FirInterpRealState<T: RealSample, Taps: AsRef<[T]>, Hist: AsRef<[T]> + AsMut<[T]>> {
    inner: ResampleRealState<T, Taps, Hist>,
}

impl<T: RealSample, Taps: AsRef<[T]>, Hist: AsRef<[T]> + AsMut<[T]>>
    FirInterpRealState<T, Taps, Hist>
{
    /// Creates a state: `taps.len() >= 1`, `factor >= 1` and
    /// `history.len() == ceil(taps.len() / factor)`. The history is zeroed.
    pub fn new(taps: Taps, history: Hist, factor: usize) -> Result<Self> {
        Ok(Self {
            inner: ResampleRealState::new(taps, history, factor, 1)?,
        })
    }

    /// The filter taps.
    pub fn taps(&self) -> &[T] {
        self.inner.taps()
    }

    interp_state_api!();

    pub(crate) fn inner_mut(&mut self) -> &mut ResampleRealState<T, Taps, Hist> {
        &mut self.inner
    }
}

impl<T: RealSample, Taps: AsRef<[T]>, Hist: AsRef<[T]> + AsMut<[T]>> Timing
    for FirInterpRealState<T, Taps, Hist>
{
    /// `L / 1`.
    fn ratio(&self) -> Ratio {
        self.inner.ratio()
    }

    /// `(N - 1) / (2 L)` input samples, built as `Delay::new(N - 1, 2 * L)`.
    fn group_delay(&self) -> Delay {
        self.inner.group_delay()
    }
}

/// Interpolating FIR for real signals.
pub trait FirInterpolateReal<T: RealSample> {
    /// Interpolates by `L` with the definition of [`FirInterpState`].
    fn fir_interpolate_real<Taps, Hist>(
        &self,
        state: &mut FirInterpRealState<T, Taps, Hist>,
        input: &[T],
        output: &mut [T],
    ) -> Result<Progress>
    where
        Taps: AsRef<[T]>,
        Hist: AsRef<[T]> + AsMut<[T]>;
}
