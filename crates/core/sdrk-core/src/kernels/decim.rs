//! Rate-changing FIR kernels: decimating FIR and rational polyphase resampler.

use core::marker::PhantomData;

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::timing::{Delay, Progress, Timing};
use crate::types::{ComplexSample, RealSample};

/// Largest decimation or interpolation factor accepted by the states.
const MAX_FACTOR: usize = i32::MAX as usize;

/// State of the decimating FIR: real taps, a history of `taps.len() - 1` samples,
/// the decimation factor `M >= 1` and the number of input samples to skip before
/// the next output.
///
/// With `y_full[n] = sum h[k] x[n-k]` over the whole stream (zero history at the
/// start and after `reset`), the outputs are `y[m] = y_full[m * M]`: the first
/// input sample after a reset produces the first output.
///
/// ```
/// use sdrk_core::{kernels::decim::{FirDecimate, FirDecimState}, Fallback, C32};
/// let mut state = FirDecimState::<C32, _, _>::new([1.0f32, 1.0], [C32::default(); 1], 2)?;
/// let input: Vec<C32> = (1..=6).map(|v| C32::new(v as f32, 0.0)).collect();
/// let mut output = [C32::default(); 3];
/// let p = Fallback.fir_decimate(&mut state, &input, &mut output)?;
/// assert_eq!((p.consumed, p.produced), (6, 3));
/// assert_eq!(output.map(|s| s.re), [1.0, 5.0, 9.0]);
/// assert_eq!(state.factor(), 2);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct FirDecimState<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>> {
    taps: Taps,
    hist: Hist,
    factor: usize,
    countdown: usize,
    _t: PhantomData<T>,
}

/// Borrowed parts of a [`FirDecimState`] for kernel implementations in backend crates.
pub struct DecimParts<'a, T: ComplexSample> {
    /// The filter taps.
    pub taps: &'a [T::Real],
    /// The history, oldest sample first.
    pub hist: &'a mut [T],
    /// The decimation factor.
    pub factor: usize,
    /// Input samples still to skip before the next output.
    pub countdown: &'a mut usize,
}

impl<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>>
    FirDecimState<T, Taps, Hist>
{
    /// Creates a state: `taps.len() >= 1`, `history.len() == taps.len() - 1` and
    /// `factor >= 1`. The history is zeroed.
    pub fn new(taps: Taps, mut history: Hist, factor: usize) -> Result<Self> {
        let n = taps.as_ref().len();
        if n == 0 {
            return Err(Error::InvalidParameter("a filter needs at least one tap"));
        }
        if factor == 0 || factor > MAX_FACTOR {
            return Err(Error::InvalidParameter(
                "decimation factor must be between 1 and 2^31 - 1",
            ));
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
            factor,
            countdown: 0,
            _t: PhantomData,
        })
    }

    /// Clears the history and restarts the decimation phase (the taps and the factor are kept).
    pub fn reset(&mut self) {
        self.hist.as_mut().fill(T::ZERO);
        self.countdown = 0;
    }

    /// The decimation factor `M`.
    pub fn factor(&self) -> usize {
        self.factor
    }

    /// The filter taps.
    pub fn taps(&self) -> &[T::Real] {
        self.taps.as_ref()
    }

    /// The history, oldest sample first (`taps().len() - 1` samples).
    pub fn history(&self) -> &[T] {
        self.hist.as_ref()
    }

    /// Number of input samples that are skipped before the next output is computed.
    pub fn countdown(&self) -> usize {
        self.countdown
    }

    /// The largest number of input samples that [`FirDecimate::fir_decimate`] consumes while
    /// producing at most `output_len` outputs, counted from the current state:
    /// `countdown + output_len * factor` (saturating). Submitting exactly this many samples
    /// (or fewer) with an output of `output_len` samples consumes all of them, which is how
    /// fused kernels keep the oscillator phase exact when the output space is limited.
    ///
    /// ```
    /// use sdrk_core::{kernels::decim::FirDecimState, C32};
    /// let state = FirDecimState::<C32, _, _>::new([1.0f32], [], 4)?;
    /// // Outputs appear at input indices 0, 4, 8: two outputs fit in 8 samples; the sample at
    /// // index 8 would produce a third one.
    /// assert_eq!(state.max_input_for_output(2), 8);
    /// assert_eq!(state.max_input_for_output(0), 0);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn max_input_for_output(&self, output_len: usize) -> usize {
        self.countdown
            .saturating_add(output_len.saturating_mul(self.factor))
    }

    /// The borrowed parts of the state, for kernel implementations in backend crates.
    pub fn parts_mut(&mut self) -> DecimParts<'_, T> {
        DecimParts {
            taps: self.taps.as_ref(),
            hist: self.hist.as_mut(),
            factor: self.factor,
            countdown: &mut self.countdown,
        }
    }
}

impl<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>> Timing
    for FirDecimState<T, Taps, Hist>
{
    /// `1 / M`.
    fn ratio(&self) -> Ratio {
        Ratio::new(1, self.factor as u64).unwrap_or(Ratio::ONE)
    }

    /// `(N - 1) / 2` input samples for `N` taps, built as `Delay::new(N - 1, 2)`.
    fn group_delay(&self) -> Delay {
        let n = self.taps.as_ref().len() as u64;
        Delay::new(n.saturating_sub(1), 2).unwrap_or(Delay::ZERO)
    }
}

/// Complex FIR with real taps followed by decimation by `M`.
///
/// The kernel consumes input only as far as the output space allows: an input
/// sample that would produce an output is not consumed when `output` is full. The
/// caller resubmits the unconsumed input.
pub trait FirDecimate<T: ComplexSample> {
    /// Computes `y[m] = y_full[m * M]` (see [`FirDecimState`]) and returns how many
    /// input samples were consumed and how many outputs were produced.
    /// Accumulation as in [`Fir`](crate::kernels::filter::Fir): `f32` ascending in `k` without FMA,
    /// `CQ15` in `i64` with one rounding and saturation per output.
    fn fir_decimate<Taps, Hist>(
        &self,
        state: &mut FirDecimState<T, Taps, Hist>,
        input: &[T],
        output: &mut [T],
    ) -> Result<Progress>
    where
        Taps: AsRef<[T::Real]>,
        Hist: AsRef<[T]> + AsMut<[T]>;
}

/// Shared state of the rational resamplers.
#[derive(Debug, Clone)]
pub(crate) struct ResampleCore<T, P, Taps, Hist> {
    taps: Taps,
    hist: Hist,
    up: usize,
    down: usize,
    /// `j * M - L * c` where `j` outputs were produced and `c` input samples consumed;
    /// an output is ready while it is negative, and it stays in `-L ..`.
    q: i64,
    _t: PhantomData<(T, P)>,
}

/// Borrowed parts of a resampler state for kernel implementations in backend crates.
pub struct ResampleParts<'a, P, T> {
    /// The filter taps.
    pub taps: &'a [P],
    /// The history, oldest sample first.
    pub hist: &'a mut [T],
    /// The interpolation factor `L`.
    pub up: usize,
    /// The decimation factor `M`.
    pub down: usize,
    /// The phase counter `j * M - L * c`.
    pub q: &'a mut i64,
}

impl<T: Copy + Default, P, Taps: AsRef<[P]>, Hist: AsRef<[T]> + AsMut<[T]>>
    ResampleCore<T, P, Taps, Hist>
{
    fn new(taps: Taps, mut history: Hist, up: usize, down: usize) -> Result<Self> {
        let n = taps.as_ref().len();
        if n == 0 {
            return Err(Error::InvalidParameter("a filter needs at least one tap"));
        }
        if up == 0 || up > MAX_FACTOR || down == 0 || down > MAX_FACTOR {
            return Err(Error::InvalidParameter(
                "up and down factors must be between 1 and 2^31 - 1",
            ));
        }
        let expected = n.div_ceil(up);
        let got = history.as_ref().len();
        if got != expected {
            return Err(Error::LengthMismatch { expected, got });
        }
        history.as_mut().fill(T::default());
        Ok(Self {
            taps,
            hist: history,
            up,
            down,
            q: 0,
            _t: PhantomData,
        })
    }

    fn reset(&mut self) {
        self.hist.as_mut().fill(T::default());
        self.q = 0;
    }

    fn parts_mut(&mut self) -> ResampleParts<'_, P, T> {
        ResampleParts {
            taps: self.taps.as_ref(),
            hist: self.hist.as_mut(),
            up: self.up,
            down: self.down,
            q: &mut self.q,
        }
    }

    fn ratio(&self) -> Ratio {
        Ratio::new(self.up as u64, self.down as u64).unwrap_or(Ratio::ONE)
    }

    fn group_delay(&self) -> Delay {
        let n = self.taps.as_ref().len() as u64;
        Delay::new(n.saturating_sub(1), 2 * self.up as u64).unwrap_or(Delay::ZERO)
    }
}

macro_rules! resample_state_api {
    () => {
        /// Clears the history and restarts the output counter (taps and factors are kept).
        pub fn reset(&mut self) {
            self.core.reset();
        }

        /// The interpolation factor `L`.
        pub fn up(&self) -> usize {
            self.core.up
        }

        /// The decimation factor `M`.
        pub fn down(&self) -> usize {
            self.core.down
        }

        /// The history of the last `ceil(N / L)` input samples, oldest first.
        pub fn history(&self) -> &[T] {
            self.core.hist.as_ref()
        }

        /// Output phase `j * M - L * c` (`j` outputs produced, `c` input samples consumed); an output is
        /// ready while it is negative.
        pub fn phase(&self) -> i64 {
            self.core.q
        }
    };
}

/// State of the rational complex resampler `L / M`.
///
/// The taps `h` are designed at the upsampled rate, the interpolation gain `L` is
/// included by the caller. The output is
/// `scipy.signal.upfirdn(h, x, up=L, down=M)` restricted to the outputs that depend
/// only on received input: `y[j] = sum_k h[k] u[j M - k]` with `u[i] = x[i / L]` for
/// `i mod L == 0` and 0 otherwise. Output `j` is produced as soon as input sample
/// `floor(j M / L)` has been received. For `n` input samples after a reset the kernel
/// produces `K = floor((n L - 1) / M) + 1` outputs.
///
/// ```
/// use sdrk_core::{kernels::decim::{Resample, ResampleState}, Fallback, C32};
/// // Zero-order hold by 2: two taps of 1.
/// let mut state = ResampleState::<C32, _, _>::new([1.0f32, 1.0], [C32::default(); 1], 2, 1)?;
/// let input = [C32::new(1.0, 0.0), C32::new(2.0, 0.0)];
/// let mut output = [C32::default(); 4];
/// let p = Fallback.resample(&mut state, &input, &mut output)?;
/// assert_eq!((p.consumed, p.produced), (2, 4));
/// assert_eq!(output.map(|s| s.re), [1.0, 1.0, 2.0, 2.0]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct ResampleState<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>> {
    core: ResampleCore<T, T::Real, Taps, Hist>,
}

impl<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>>
    ResampleState<T, Taps, Hist>
{
    /// Creates a state: `taps.len() >= 1`, `up >= 1`, `down >= 1` and
    /// `history.len() == ceil(taps.len() / up)`. The history is zeroed.
    pub fn new(taps: Taps, history: Hist, up: usize, down: usize) -> Result<Self> {
        Ok(Self {
            core: ResampleCore::new(taps, history, up, down)?,
        })
    }

    /// The filter taps.
    pub fn taps(&self) -> &[T::Real] {
        self.core.taps.as_ref()
    }

    resample_state_api!();

    /// The borrowed parts of the state, for kernel implementations in backend crates.
    pub fn parts_mut(&mut self) -> ResampleParts<'_, T::Real, T> {
        self.core.parts_mut()
    }
}

impl<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>> Timing
    for ResampleState<T, Taps, Hist>
{
    /// `L / M`.
    fn ratio(&self) -> Ratio {
        self.core.ratio()
    }

    /// `(N - 1) / (2 L)` input samples, built as `Delay::new(N - 1, 2 * L)`.
    fn group_delay(&self) -> Delay {
        self.core.group_delay()
    }
}

/// Rational polyphase resampler for complex signals with real taps.
///
/// Like [`FirDecimate`] it consumes input only as far as the output space allows.
pub trait Resample<T: ComplexSample> {
    /// Resamples by `L / M`, see [`ResampleState`] for the definition. Returns the
    /// consumed and produced counts; the caller resubmits unconsumed input.
    fn resample<Taps, Hist>(
        &self,
        state: &mut ResampleState<T, Taps, Hist>,
        input: &[T],
        output: &mut [T],
    ) -> Result<Progress>
    where
        Taps: AsRef<[T::Real]>,
        Hist: AsRef<[T]> + AsMut<[T]>;
}

/// State of the rational resampler for real signals (`f32` or `Q15` samples and taps).
///
/// The definition is the one of [`ResampleState`].
///
/// ```
/// use sdrk_core::{kernels::decim::{ResampleReal, ResampleRealState}, Fallback};
/// let mut state = ResampleRealState::<f32, _, _>::new([1.0f32], [0.0f32; 1], 1, 2)?;
/// let mut output = [0.0f32; 2];
/// let p = Fallback.resample_real(&mut state, &[1.0, 2.0, 3.0, 4.0], &mut output)?;
/// assert_eq!((p.consumed, p.produced), (4, 2));
/// assert_eq!(output, [1.0, 3.0]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct ResampleRealState<T: RealSample, Taps: AsRef<[T]>, Hist: AsRef<[T]> + AsMut<[T]>> {
    core: ResampleCore<T, T, Taps, Hist>,
}

impl<T: RealSample, Taps: AsRef<[T]>, Hist: AsRef<[T]> + AsMut<[T]>>
    ResampleRealState<T, Taps, Hist>
{
    /// Creates a state: `taps.len() >= 1`, `up >= 1`, `down >= 1` and
    /// `history.len() == ceil(taps.len() / up)`. The history is zeroed.
    pub fn new(taps: Taps, history: Hist, up: usize, down: usize) -> Result<Self> {
        Ok(Self {
            core: ResampleCore::new(taps, history, up, down)?,
        })
    }

    /// The filter taps.
    pub fn taps(&self) -> &[T] {
        self.core.taps.as_ref()
    }

    resample_state_api!();

    /// The borrowed parts of the state, for kernel implementations in backend crates.
    pub fn parts_mut(&mut self) -> ResampleParts<'_, T, T> {
        self.core.parts_mut()
    }
}

impl<T: RealSample, Taps: AsRef<[T]>, Hist: AsRef<[T]> + AsMut<[T]>> Timing
    for ResampleRealState<T, Taps, Hist>
{
    /// `L / M`.
    fn ratio(&self) -> Ratio {
        self.core.ratio()
    }

    /// `(N - 1) / (2 L)` input samples, built as `Delay::new(N - 1, 2 * L)`.
    fn group_delay(&self) -> Delay {
        self.core.group_delay()
    }
}

/// Rational polyphase resampler for real signals.
pub trait ResampleReal<T: RealSample> {
    /// Resamples by `L / M` with the definition of [`ResampleState`].
    fn resample_real<Taps, Hist>(
        &self,
        state: &mut ResampleRealState<T, Taps, Hist>,
        input: &[T],
        output: &mut [T],
    ) -> Result<Progress>
    where
        Taps: AsRef<[T]>,
        Hist: AsRef<[T]> + AsMut<[T]>;
}
