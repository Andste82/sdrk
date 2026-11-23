//! The FFT fast convolution `xcorr_fft` (M50): a `C32` stream convolved with a fixed complex
//! filter of up to `N / 2` taps by the overlap-save method, the engine of a long matched filter
//! (a preamble or spreading-code correlator).
//!
//! **Definition.** With the taps `h` (length `L`) and the input `x` (zero before the first sample
//! after a reset), output `n` is the linear convolution
//!
//! ```text
//! y[n] = sum_{k = 0}^{L - 1} h[k] x[n - k]
//! ```
//!
//! **Computation (overlap-save).** An `N`-point FFT (`N` a power of two, 16 to 65536,
//! `N >= 2 L`) and the stored spectrum `H = FFT_N{h padded with zeros}` give `B = N - L + 1`
//! outputs per transform block: the window `w = [history (L - 1 samples) | next B inputs]` is
//! copied into the scratch, `W = FFT_N{w}`, `Y = W H` element by element with the fixed product
//! formula of [`spectral_product`] (four products, one subtraction, one addition, no FMA), and
//! `IFFT_N{Y}` holds the valid outputs at `L - 1 .. N`. The history becomes the last `L - 1`
//! samples of the window (a copy). The forward `C32` FFT is unnormalised and the inverse divides
//! by `N`, so no scaling is applied.
//!
//! **Correlation.** The kernel convolves. A correlation with a reference `r` takes the taps
//! `h[k] = conj(r[L - 1 - k])`; a reference starting at input index `m` then peaks at output
//! index `m + L - 1`.

use core::marker::PhantomData;

use sdrk_device::Ratio;

use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::fft::{Direction, Fft, FftPlan, Scaling, log2_of};
use crate::timing::{Delay, Progress, Timing};
use crate::types::{C32, ComplexSample};

/// Smallest FFT size of [`XcorrFftState`].
pub const MIN_FFT_SIZE: usize = 16;
/// Largest FFT size of [`XcorrFftState`].
pub const MAX_FFT_SIZE: usize = 65_536;

/// The error of a violated internal invariant of the kernel.
fn internal() -> Error {
    Error::Internal("xcorr_fft index")
}

/// Checks the tap count `L` against the FFT size `N`: `1 <= L <= N / 2`, `N` a supported size.
fn check_sizes(taps_len: usize, n: usize) -> Result<()> {
    log2_of(n)?;
    if taps_len == 0 {
        return Err(Error::InvalidParameter("xcorr_fft needs at least one tap"));
    }
    if taps_len > n / 2 {
        return Err(Error::InvalidParameter(
            "xcorr_fft needs an fft size of at least twice the tap count",
        ));
    }
    Ok(())
}

/// `Ok` when `got == expected`, else `LengthMismatch`.
fn check_len(expected: usize, got: usize) -> Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::LengthMismatch { expected, got })
    }
}

/// Writes `H = FFT_N{taps padded to N}` into `spectrum` with the `Fallback` forward FFT of the
/// plan, exactly as [`XcorrFftState::new`] does (for build scripts and offline tools that
/// generate a table for [`XcorrFftState::from_spectrum`]).
///
/// # Errors
/// `InvalidParameter` for no taps or more than `N / 2` taps; `LengthMismatch` if
/// `spectrum.len()` is not `N`.
///
/// ```
/// use sdrk_core::kernels::fft::FftPlan;
/// use sdrk_core::kernels::xcorr::xcorr_spectrum;
/// use sdrk_core::C32;
/// let plan = FftPlan::<C32, _>::new(16, [C32::default(); 8])?;
/// let mut h = [C32::default(); 16];
/// // One tap 1 at index 0: a flat spectrum.
/// xcorr_spectrum(&[C32::new(1.0, 0.0)], &mut h, &plan)?;
/// assert!(h.iter().all(|v| *v == C32::new(1.0, 0.0)));
/// assert!(xcorr_spectrum(&[C32::default(); 9], &mut h, &plan).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub fn xcorr_spectrum<Tw: AsRef<[C32]>>(
    taps: &[C32],
    spectrum: &mut [C32],
    fft: &FftPlan<C32, Tw>,
) -> Result<()> {
    let n = fft.len();
    check_sizes(taps.len(), n)?;
    check_len(n, spectrum.len())?;
    let (head, tail) = spectrum
        .split_at_mut_checked(taps.len())
        .ok_or_else(internal)?;
    head.copy_from_slice(taps);
    tail.fill(C32::ZERO);
    Fallback.fft(fft, spectrum, Direction::Forward, Scaling::Default)?;
    Ok(())
}

/// State of `xcorr_fft`: the tap spectrum `H` (`N` values, read-only after construction), the
/// history of the last `L - 1` input samples, the scratch of `N` values in which the transforms
/// run, the tap count `L` and the FFT plan of size `N`.
///
/// Storage: `H` and the scratch take `N` values each, the history `L - 1`, the plan `N / 2`
/// twiddles. With `from_spectrum` and a plan over `FftPlan::from_static_4096` (feature
/// `static-twiddles`, `N <= 4096`) `H` and the twiddles can be `static` tables, so only the
/// history and the scratch need RAM.
///
/// ```
/// use sdrk_core::kernels::fft::FftPlan;
/// use sdrk_core::kernels::xcorr::{XcorrFft, XcorrFftState};
/// use sdrk_core::{C32, Fallback};
/// // A two-tap filter h = [1, 0.5] with a 16-point FFT: B = 15 outputs per block.
/// let plan = FftPlan::<C32, _>::new(16, vec![C32::default(); 8])?;
/// let taps = [C32::new(1.0, 0.0), C32::new(0.5, 0.0)];
/// let mut state = XcorrFftState::new(&taps, vec![C32::default(); 16], vec![C32::default(); 1], vec![C32::default(); 16], plan)?;
/// assert_eq!(state.block_size(), 15);
/// let mut input = [C32::default(); 30];
/// input[3] = C32::new(1.0, 0.0); // an impulse at index 3
/// let mut output = [C32::default(); 30];
/// let p = Fallback.xcorr_fft(&mut state, &input, &mut output)?;
/// assert_eq!((p.consumed, p.produced), (30, 30));
/// // The impulse reproduces the taps at outputs 3 and 4 (within the FFT error).
/// assert!((output[3].re - 1.0).abs() < 1e-6 && (output[4].re - 0.5).abs() < 1e-6);
/// assert!(output[5].norm_sqr() < 1e-12);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct XcorrFftState<T, S, Sh, Sc, Tw>
where
    T: ComplexSample,
    S: AsRef<[T]>,
    Sh: AsRef<[T]> + AsMut<[T]>,
    Sc: AsRef<[T]> + AsMut<[T]>,
    Tw: AsRef<[T]>,
{
    spectrum: S,
    history: Sh,
    scratch: Sc,
    plan: FftPlan<T, Tw>,
    taps_len: usize,
    _t: PhantomData<fn() -> T>,
}

/// Checks the storage lengths of a state of `taps_len` taps over a plan of size `n`.
fn check_storage(
    taps_len: usize,
    n: usize,
    spectrum: usize,
    history: usize,
    scratch: usize,
) -> Result<()> {
    check_sizes(taps_len, n)?;
    check_len(n, spectrum)?;
    check_len(taps_len - 1, history)?;
    check_len(n, scratch)
}

impl<S, Sh, Sc, Tw> XcorrFftState<C32, S, Sh, Sc, Tw>
where
    S: AsRef<[C32]> + AsMut<[C32]>,
    Sh: AsRef<[C32]> + AsMut<[C32]>,
    Sc: AsRef<[C32]> + AsMut<[C32]>,
    Tw: AsRef<[C32]>,
{
    /// Creates a state from the `L` filter `taps` (already reversed and conjugated for a
    /// correlation): `fft` is a plan of size `N` with `N >= 2 L`; `h_store` (`N` values) is
    /// filled with `H = FFT_N{taps padded to N}` by the `Fallback` forward FFT (so `H` has the
    /// same bits on every backend), the `history` (`L - 1` values) is zeroed and the `scratch`
    /// takes `N` values.
    ///
    /// # Errors
    /// `InvalidParameter` for no taps or `N < 2 L`; `LengthMismatch` for storage of another
    /// length.
    ///
    /// ```
    /// use sdrk_core::kernels::fft::FftPlan;
    /// use sdrk_core::kernels::xcorr::XcorrFftState;
    /// use sdrk_core::C32;
    /// let plan = || FftPlan::<C32, _>::new(64, vec![C32::default(); 32]).unwrap();
    /// let z = |n: usize| vec![C32::default(); n];
    /// assert!(XcorrFftState::new(&z(32), z(64), z(31), z(64), plan()).is_ok());
    /// // 33 taps need at least 66 points; a history of 30 values is too short.
    /// assert!(XcorrFftState::new(&z(33), z(64), z(32), z(64), plan()).is_err());
    /// assert!(XcorrFftState::new(&z(32), z(64), z(30), z(64), plan()).is_err());
    /// ```
    pub fn new(
        taps: &[C32],
        mut h_store: S,
        mut history: Sh,
        scratch: Sc,
        fft: FftPlan<C32, Tw>,
    ) -> Result<Self> {
        let n = fft.len();
        check_storage(
            taps.len(),
            n,
            h_store.as_ref().len(),
            history.as_ref().len(),
            scratch.as_ref().len(),
        )?;
        xcorr_spectrum(taps, h_store.as_mut(), &fft)?;
        history.as_mut().fill(C32::ZERO);
        Ok(Self {
            spectrum: h_store,
            history,
            scratch,
            plan: fft,
            taps_len: taps.len(),
            _t: PhantomData,
        })
    }
}

impl<S, Sh, Sc, Tw> XcorrFftState<C32, S, Sh, Sc, Tw>
where
    S: AsRef<[C32]>,
    Sh: AsRef<[C32]> + AsMut<[C32]>,
    Sc: AsRef<[C32]> + AsMut<[C32]>,
    Tw: AsRef<[C32]>,
{
    /// Creates a state from a precomputed spectrum `H` of `N` values (read-only, for example a
    /// `static` table in flash), the spectrum that [`xcorr_spectrum`] (or [`new`](Self::new))
    /// gives for the same taps and `N`; `taps_len` is `L`. The spectrum is not checked against
    /// any taps (it cannot be). The `history` (`L - 1` values) is zeroed.
    ///
    /// # Errors
    /// The rules of [`new`](Self::new): `InvalidParameter` for `taps_len` 0 or `N < 2 L`;
    /// `LengthMismatch` for a spectrum other than `N` values or other storage of another length.
    ///
    /// ```
    /// use sdrk_core::kernels::fft::FftPlan;
    /// use sdrk_core::kernels::xcorr::{xcorr_spectrum, XcorrFftState};
    /// use sdrk_core::C32;
    /// let plan = FftPlan::<C32, _>::new(16, [C32::default(); 8])?;
    /// let mut table = [C32::default(); 16];
    /// xcorr_spectrum(&[C32::new(0.5, 0.0); 3], &mut table, &plan)?;
    /// let state = XcorrFftState::from_spectrum(&table[..], 3, [C32::default(); 2], [C32::default(); 16], plan)?;
    /// assert_eq!((state.taps_len(), state.block_size()), (3, 14));
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn from_spectrum(
        spectrum: S,
        taps_len: usize,
        mut history: Sh,
        scratch: Sc,
        fft: FftPlan<C32, Tw>,
    ) -> Result<Self> {
        let n = fft.len();
        check_storage(
            taps_len,
            n,
            spectrum.as_ref().len(),
            history.as_ref().len(),
            scratch.as_ref().len(),
        )?;
        history.as_mut().fill(C32::ZERO);
        Ok(Self {
            spectrum,
            history,
            scratch,
            plan: fft,
            taps_len,
            _t: PhantomData,
        })
    }
}

impl<T, S, Sh, Sc, Tw> XcorrFftState<T, S, Sh, Sc, Tw>
where
    T: ComplexSample,
    S: AsRef<[T]>,
    Sh: AsRef<[T]> + AsMut<[T]>,
    Sc: AsRef<[T]> + AsMut<[T]>,
    Tw: AsRef<[T]>,
{
    /// The tap count `L`.
    pub fn taps_len(&self) -> usize {
        self.taps_len
    }

    /// The FFT size `N`.
    pub fn fft_size(&self) -> usize {
        self.plan.len()
    }

    /// The block size `B = N - L + 1`: the outputs of one transform block, and the input and
    /// output samples a call needs at least to produce anything.
    pub fn block_size(&self) -> usize {
        self.plan.len() - self.taps_len + 1
    }

    /// The tap spectrum `H` (`N` values).
    pub fn spectrum(&self) -> &[T] {
        self.spectrum.as_ref()
    }

    /// The history: the last `L - 1` input samples of the last block, oldest first (zero after
    /// construction and [`reset`](Self::reset)).
    pub fn history(&self) -> &[T] {
        self.history.as_ref()
    }

    /// The FFT plan of size `N`.
    pub fn plan(&self) -> &FftPlan<T, Tw> {
        &self.plan
    }

    /// Clears the history (the spectrum and the plan are kept); the next block starts the
    /// stream anew.
    pub fn reset(&mut self) {
        self.history.as_mut().fill(T::ZERO);
    }
}

impl<T, S, Sh, Sc, Tw> Timing for XcorrFftState<T, S, Sh, Sc, Tw>
where
    T: ComplexSample,
    S: AsRef<[T]>,
    Sh: AsRef<[T]> + AsMut<[T]>,
    Sc: AsRef<[T]> + AsMut<[T]>,
    Tw: AsRef<[T]>,
{
    /// `1/1`.
    fn ratio(&self) -> Ratio {
        Ratio::ONE
    }

    /// `L - 1` input samples (the correlation alignment: a reference that starts at input index
    /// `m` peaks at output index `m + L - 1`).
    fn group_delay(&self) -> Delay {
        Delay::new(self.taps_len.saturating_sub(1) as u64, 1).unwrap_or(Delay::ZERO)
    }
}

/// The FFT fast convolution (see the [module documentation](self) for the definition).
///
/// A call runs `k = min(input.len() / B, output.len() / B)` whole transform blocks
/// (`B = state.block_size()`), consumes `k B` input samples and writes `k B` outputs;
/// `Progress` counts both. A remainder of fewer than `B` input samples is not consumed: the
/// caller resubmits it with the next input. The block boundaries are those of the stream (every
/// `B` samples from the start or the last reset), so the output does not depend on the chunking
/// (bit for bit on one backend and level).
///
/// `C32`: the class is *tolerance* (SNR >= 100 dB against the f64 direct convolution of
/// `Reference`): one forward FFT, the spectral product of [`spectral_product`] (no FMA) and one
/// inverse FFT per block, `H` from the `Fallback` FFT of the taps on every backend.
pub trait XcorrFft<T: ComplexSample> {
    /// Convolves `input` with the taps of the state (see the trait documentation).
    ///
    /// # Errors
    /// Only the errors of the FFT (none for a valid state).
    fn xcorr_fft<S, Sh, Sc, Tw>(
        &self,
        state: &mut XcorrFftState<T, S, Sh, Sc, Tw>,
        input: &[T],
        output: &mut [T],
    ) -> Result<Progress>
    where
        S: AsRef<[T]>,
        Sh: AsRef<[T]> + AsMut<[T]>,
        Sc: AsRef<[T]> + AsMut<[T]>,
        Tw: AsRef<[T]>;
}

/// One block of the overlap-save driver: the history (`L - 1` values), the `B` new input
/// samples, the spectrum `H`, the scratch (`N` values), the plan and the `B` outputs.
pub type BlockArgs<'a, T, Tw> = (
    &'a [T],
    &'a [T],
    &'a [T],
    &'a mut [T],
    &'a FftPlan<T, Tw>,
    &'a mut [T],
);

/// The driver of every `xcorr_fft` implementation: runs the whole blocks that fit the input and
/// the output, calls `block(history, x, spectrum, scratch, plan, y)` for each (`x` the `B` new
/// input samples, `y` the `B` outputs of the block) and then advances the history to the last
/// `L - 1` samples of `x` (a copy; `B > L - 1` because `N >= 2 L`). The driver does not touch the
/// scratch, so `block` may keep data in it from one block of a call to the next.
///
/// # Errors
/// The errors of `block`.
#[allow(
    clippy::inline_always,
    reason = "a backend calls it at its dispatched level, whose target features the block must inherit"
)]
#[inline(always)]
pub fn xcorr_fft_with<T, S, Sh, Sc, Tw, K>(
    s: &mut XcorrFftState<T, S, Sh, Sc, Tw>,
    input: &[T],
    output: &mut [T],
    mut block: K,
) -> Result<Progress>
where
    T: ComplexSample,
    S: AsRef<[T]>,
    Sh: AsRef<[T]> + AsMut<[T]>,
    Sc: AsRef<[T]> + AsMut<[T]>,
    Tw: AsRef<[T]>,
    K: FnMut(BlockArgs<'_, T, Tw>) -> Result<()>,
{
    let b = s.block_size();
    let keep = s.taps_len - 1;
    let blocks = (input.len() / b).min(output.len() / b);
    let XcorrFftState {
        spectrum,
        history,
        scratch,
        plan,
        ..
    } = s;
    let spectrum = spectrum.as_ref();
    let history = history.as_mut();
    let scratch = scratch.as_mut();
    for (x, y) in input
        .chunks_exact(b)
        .zip(output.chunks_exact_mut(b))
        .take(blocks)
    {
        block((&*history, x, spectrum, &mut *scratch, plan, y))?;
        copy_samples(history, x.get(b - keep..).ok_or_else(internal)?);
    }
    let done = blocks * b;
    Ok(Progress {
        consumed: done,
        produced: done,
    })
}

/// Copies `src` to `dst` sample by sample (equal lengths; exact). An element loop instead of
/// `copy_from_slice`, as the history copy of `channelize` (D920): `copy_from_slice` of a
/// variable length is a call of `memcpy`, which glibc does with `rep movsb` for the copies of
/// thousands of samples, counted once per byte by the counters of the cycle simulation.
#[allow(
    clippy::inline_always,
    reason = "a backend calls it at its dispatched level, whose target features it must inherit"
)]
#[inline(always)]
fn copy_samples<T: Copy>(dst: &mut [T], src: &[T]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = *s;
    }
}

/// The spectral product of one bin, `w h`, with the fixed formula of the contract: the four
/// products `wr hr`, `wi hi`, `wr hi`, `wi hr` first, then `re = wr hr - wi hi` and
/// `im = wr hi + wi hr` (one subtraction, one addition, no FMA; the `mul` fallback model).
///
/// ```
/// use sdrk_core::kernels::xcorr::spectral_product_one;
/// use sdrk_core::C32;
/// assert_eq!(spectral_product_one(C32::new(1.0, 2.0), C32::new(3.0, -1.0)), C32::new(5.0, 5.0));
/// ```
#[inline]
pub fn spectral_product_one(w: C32, h: C32) -> C32 {
    let rr = w.re * h.re;
    let ii = w.im * h.im;
    let ri = w.re * h.im;
    let ir = w.im * h.re;
    C32::new(rr - ii, ri + ir)
}

/// `w[k] = w[k] h[k]` for every bin with the formula of [`spectral_product_one`].
///
/// # Errors
/// `LengthMismatch` if the lengths differ.
///
/// ```
/// use sdrk_core::kernels::xcorr::spectral_product;
/// use sdrk_core::C32;
/// let mut w = [C32::new(0.0, 1.0); 4];
/// spectral_product(&mut w, &[C32::new(0.0, 1.0); 4])?;
/// assert!(w.iter().all(|v| *v == C32::new(-1.0, 0.0)));
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[allow(
    clippy::inline_always,
    reason = "a backend calls it at its dispatched level, whose target features it must inherit"
)]
#[inline(always)]
pub fn spectral_product(w: &mut [C32], h: &[C32]) -> Result<()> {
    check_len(w.len(), h.len())?;
    for (a, b) in w.iter_mut().zip(h) {
        *a = spectral_product_one(*a, *b);
    }
    Ok(())
}

/// One overlap-save block with the FFT `fft`: the window `[history | x]` into the scratch, the
/// forward FFT, the spectral product with `spectrum`, the inverse FFT, and the last `B` values
/// of the scratch into `y`. The block function of `Fallback` with `fft = Fallback`.
///
/// # Errors
/// `Internal` if the lengths do not fit (the state checks them), and the errors of `fft`.
#[allow(
    clippy::inline_always,
    reason = "a backend calls it at its dispatched level, whose target features it must inherit"
)]
#[inline(always)]
pub fn overlap_save_block<F: Fft<C32>, Tw: AsRef<[C32]>>(
    fft: &F,
    (history, x, spectrum, scratch, plan, y): BlockArgs<'_, C32, Tw>,
) -> Result<()> {
    let keep = history.len();
    if keep + x.len() != scratch.len() || y.len() != x.len() {
        return Err(internal());
    }
    let (head, tail) = scratch.split_at_mut(keep);
    copy_samples(head, history);
    copy_samples(tail, x);
    fft.fft(plan, scratch, Direction::Forward, Scaling::Default)?;
    spectral_product(scratch, spectrum)?;
    fft.fft(plan, scratch, Direction::Inverse, Scaling::Default)?;
    copy_samples(y, scratch.get(keep..).ok_or_else(internal)?);
    Ok(())
}
