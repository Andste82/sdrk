//! The polyphase channelizer `channelize` (M45): an analysis filter bank that splits a band into
//! `M` channels at once, oversampled by 2 (one output vector of `M` channel samples per `M / 2`
//! input samples, the "M/2 hop" form).
//!
//! **Definition.** With the prototype low-pass `h` of `N = M L` taps (`L` taps per branch), the
//! input `x` (zero before the first sample after a reset) and the hop `D = M / 2`, output step
//! `m` is taken at the input index `t_m = (m + 1) D - 1` (the newest sample of hop `m`), and
//! channel `k` of it is
//!
//! ```text
//! y_k[m] = sum_{r = 0}^{N - 1} h[r] x[t_m - r] e^(-j 2 pi k (t_m - r) / M)
//! ```
//!
//! the input mixed down by `k fs / M` (with the mixer phase 0 at the first sample after a reset),
//! filtered by `h` and taken every `D` samples. Channel `k` (FFT order: 0 is the band centre,
//! `k > M / 2` are the negative frequencies) is centred at `k fs / M`; its rate is `2 fs / M`, and
//! it keeps its phase from step to step.
//!
//! **Computation.** Per hop, the window `w` of the newest `N` samples is multiplied by the
//! time-reversed prototype `g[i] = h[N - 1 - i]`, and the `L` rows of `M` products are summed per
//! column: `z[c] = sum_p g[p M + c] w[p M + c]` (the `M` branch filters). Column `c` holds the
//! samples with `n mod M = (c + s) mod M`, where `s = D` for an even hop index and `s = 0` for an
//! odd one, so `u[(c + s) mod M] = z[c]` (the circular shift by the parity of the output index)
//! and `y[m] = FFT_M(u)` (forward, unnormalised).

// The names are those of the formulas in the module documentation.
#![allow(clippy::many_single_char_names)]

use core::marker::PhantomData;

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::kernels::fft::{Direction, Fft, FftPlan, Scaling, log2_of, twiddle_f64};
use crate::timing::{Delay, Progress, Timing};
use crate::types::{C32, ComplexSample};

/// Fewest channels `M` of [`ChannelizeState`].
pub const MIN_CHANNELS: usize = 4;
/// Most channels `M` of [`ChannelizeState`].
pub const MAX_CHANNELS: usize = 256;
/// Fewest prototype taps per branch `L` of [`ChannelizeState`].
pub const MIN_TAPS_PER_BRANCH: usize = 4;
/// Most prototype taps per branch `L` of [`ChannelizeState`].
pub const MAX_TAPS_PER_BRANCH: usize = 32;

/// Smallest size of the FFT kernel; smaller channel counts use the radix-2 transform of
/// [`channel_fft`].
const FFT_MIN: usize = 16;

/// The error of a violated internal invariant of the channelizer.
fn internal() -> Error {
    Error::Internal("channelize index")
}

/// Checks the channel count: a power of two from 4 to 256.
fn check_channels(channels: usize) -> Result<()> {
    if channels.is_power_of_two() && (MIN_CHANNELS..=MAX_CHANNELS).contains(&channels) {
        Ok(())
    } else {
        Err(Error::InvalidParameter(
            "channelize needs a power of two from 4 to 256 channels",
        ))
    }
}

/// State of the channelizer: the time-reversed prototype (`M L` taps), the history of the input,
/// the position within the current hop, the parity of the hop index and the FFT plan of size `M`.
///
/// The history holds the newest `N - D` samples of the last complete hop plus the samples of
/// the current hop; it is written in place and moved to its start only when it is full, so a
/// longer history moves less often (`2 N` moves `N - D` samples every `(N + D) / D` hops; the
/// values do not depend on its length).
///
/// ```
/// use sdrk_core::kernels::channelize::{Channelize, ChannelizeState};
/// use sdrk_core::{C32, Fallback};
/// // 4 channels with 4 taps per branch: a prototype of 16 taps (here a plain average).
/// let mut state =
///     ChannelizeState::<C32, _, _, _>::new([1.0f32 / 16.0; 16], 4, [C32::default(); 32], [C32::default(); 2])?;
/// // A tone exactly on channel 1 (fs / 4).
/// let input: Vec<C32> = (0..64)
///     .map(|n| {
///         let p = core::f32::consts::FRAC_PI_2 * (n % 4) as f32;
///         C32::new(p.cos(), p.sin())
///     })
///     .collect();
/// let mut output = vec![C32::default(); 4 * 32];
/// let p = Fallback.channelize(&mut state, &input, &mut output)?;
/// assert_eq!((p.consumed, p.produced), (64, 128)); // 32 steps of 4 channels
/// let last = &output[124..128];
/// assert!(last[0].norm_sqr() < 1e-12 && last[2].norm_sqr() < 1e-12 && last[3].norm_sqr() < 1e-12);
/// // The averaging prototype has gain 1 at DC; the mixer of channel 1 removes the tone.
/// assert!((last[1].norm_sqr() - 1.0).abs() < 1e-5);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct ChannelizeState<T, Taps, Hist, Tw>
where
    T: ComplexSample,
    Taps: AsRef<[T::Real]>,
    Hist: AsRef<[T]> + AsMut<[T]>,
    Tw: AsRef<[T]>,
{
    taps: Taps,
    hist: Hist,
    plan: FftPlan<T, Tw>,
    channels: usize,
    taps_per_branch: usize,
    /// Write position in the history (the window of the current hop ends here).
    pos: usize,
    /// Samples of the current hop received so far, `0 .. D`.
    phase: usize,
    /// Whether the index of the next output step is odd.
    odd: bool,
    _t: PhantomData<fn() -> T>,
}

impl<T, Taps, Hist, Tw> ChannelizeState<T, Taps, Hist, Tw>
where
    T: ComplexSample,
    Taps: AsRef<[T::Real]> + AsMut<[T::Real]>,
    Hist: AsRef<[T]> + AsMut<[T]>,
    Tw: AsRef<[T]> + AsMut<[T]>,
{
    /// Creates a state for `channels` channels `M` (a power of two, 4 to 256) from the prototype
    /// `taps` `h` (`M L` taps with `L` from 4 to 32, in time order; the state reverses them in
    /// place, see [`taps`](Self::taps)), a `history` of at least `M L` samples (2 `M L`
    /// recommended, [`history_len`]) and the FFT `twiddles` (`M / 2` values, filled with
    /// `e^(-j 2 pi k / M)` as by [`FftPlan::new`]). The history is zeroed.
    ///
    /// # Errors
    /// `InvalidParameter` for a channel count or a tap count outside the ranges;
    /// `LengthMismatch` for a history shorter than `M L` or a twiddle table other than `M / 2`.
    ///
    /// ```
    /// use sdrk_core::kernels::channelize::ChannelizeState;
    /// use sdrk_core::C32;
    /// let ok = ChannelizeState::<C32, _, _, _>::new(vec![0.0f32; 128], 16, vec![C32::default(); 256], vec![C32::default(); 8]);
    /// assert!(ok.is_ok());
    /// // 12 channels are not a power of two; 16 channels with 3 taps per branch are too few.
    /// assert!(ChannelizeState::<C32, _, _, _>::new(vec![0.0f32; 48], 12, vec![C32::default(); 96], vec![C32::default(); 6]).is_err());
    /// assert!(ChannelizeState::<C32, _, _, _>::new(vec![0.0f32; 48], 16, vec![C32::default(); 96], vec![C32::default(); 8]).is_err());
    /// ```
    pub fn new(
        mut taps: Taps,
        channels: usize,
        mut history: Hist,
        mut twiddles: Tw,
    ) -> Result<Self> {
        check_channels(channels)?;
        let n = taps.as_ref().len();
        let taps_per_branch = n / channels;
        if !n.is_multiple_of(channels)
            || !(MIN_TAPS_PER_BRANCH..=MAX_TAPS_PER_BRANCH).contains(&taps_per_branch)
        {
            return Err(Error::InvalidParameter(
                "channelize needs channels * taps_per_branch taps with 4 to 32 taps per branch",
            ));
        }
        let got = history.as_ref().len();
        if got < n {
            return Err(Error::LengthMismatch { expected: n, got });
        }
        let got = twiddles.as_ref().len();
        if got != channels / 2 {
            return Err(Error::LengthMismatch {
                expected: channels / 2,
                got,
            });
        }
        let plan = if channels >= FFT_MIN {
            FftPlan::new(channels, twiddles)?
        } else {
            // The FFT kernel starts at 16 points; 4 and 8 channels use the radix-2 transform of
            // `channel_fft` over the same twiddles.
            for (k, w) in twiddles.as_mut().iter_mut().enumerate() {
                *w = T::from_c64_round(twiddle_f64(k, channels));
            }
            FftPlan {
                n: channels,
                log2n: channels.trailing_zeros(),
                twiddles,
                tw_stride: 1,
                _sample: PhantomData,
            }
        };
        taps.as_mut().reverse();
        history.as_mut().fill(T::ZERO);
        Ok(Self {
            taps,
            hist: history,
            plan,
            channels,
            taps_per_branch,
            pos: n - channels / 2,
            phase: 0,
            odd: false,
            _t: PhantomData,
        })
    }
}

/// The recommended history length of a [`ChannelizeState`] for `channels` channels with
/// `taps_per_branch` taps per branch: `2 M L` (the history moves `N - D` samples once every
/// `(N + D) / D` hops, about one copy per input sample). Any length from `M L` on works.
///
/// ```
/// use sdrk_core::kernels::channelize::history_len;
/// assert_eq!(history_len(16, 8), 256);
/// ```
pub const fn history_len(channels: usize, taps_per_branch: usize) -> usize {
    channels.saturating_mul(taps_per_branch).saturating_mul(2)
}

impl<T, Taps, Hist, Tw> ChannelizeState<T, Taps, Hist, Tw>
where
    T: ComplexSample,
    Taps: AsRef<[T::Real]>,
    Hist: AsRef<[T]> + AsMut<[T]>,
    Tw: AsRef<[T]>,
{
    /// Clears the history and restarts at hop 0 (the taps and the plan are kept).
    pub fn reset(&mut self) {
        self.hist.as_mut().fill(T::ZERO);
        self.pos = self.channels * self.taps_per_branch - self.channels / 2;
        self.phase = 0;
        self.odd = false;
    }

    /// The number of channels `M`.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// The prototype taps per branch `L`.
    pub fn taps_per_branch(&self) -> usize {
        self.taps_per_branch
    }

    /// The hop `D = M / 2`: input samples per output step.
    pub fn hop(&self) -> usize {
        self.channels / 2
    }

    /// The prototype, **time-reversed**: `taps()[i] = h[M L - 1 - i]`.
    pub fn taps(&self) -> &[T::Real] {
        self.taps.as_ref()
    }

    /// The history storage (its layout is an implementation detail; for tests).
    pub fn history(&self) -> &[T] {
        self.hist.as_ref()
    }

    /// The FFT plan of size `M` for 16 channels and more; `None` for 4 and 8 channels, whose
    /// transform is the radix-2 path of [`channel_fft`] and lies outside the sizes the FFT kernel
    /// accepts (16 to 65536 points), so the plan never reaches [`Fft::fft`].
    pub fn plan(&self) -> Option<&FftPlan<T, Tw>> {
        (self.channels >= FFT_MIN).then_some(&self.plan)
    }

    /// The `M / 2` twiddles `e^(-j 2 pi k / M)` of the channel transform (for every `M`).
    pub fn twiddles(&self) -> &[T] {
        self.plan.twiddles()
    }

    /// Input samples of the current hop received so far (`0 .. M / 2`).
    pub fn phase(&self) -> usize {
        self.phase
    }

    /// Whether the index of the next output step is odd (hop 0 after a reset is even).
    pub fn odd_step(&self) -> bool {
        self.odd
    }

    /// The largest number of input samples that [`Channelize::channelize`] consumes with an
    /// output of `steps` steps (`steps M` values), counted from the current state:
    /// `steps D - phase + D - 1` (saturating): the samples that complete `steps` hops plus the
    /// part of the next hop that fits without an output.
    ///
    /// ```
    /// use sdrk_core::kernels::channelize::ChannelizeState;
    /// use sdrk_core::C32;
    /// let state = ChannelizeState::<C32, _, _, _>::new([0.0f32; 64], 16, [C32::default(); 128], [C32::default(); 8])?;
    /// assert_eq!(state.max_input_for_output(0), 7);
    /// assert_eq!(state.max_input_for_output(2), 23);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn max_input_for_output(&self, steps: usize) -> usize {
        let d = self.channels / 2;
        steps.saturating_mul(d).saturating_add(d - 1 - self.phase)
    }
}

impl<T, Taps, Hist, Tw> Timing for ChannelizeState<T, Taps, Hist, Tw>
where
    T: ComplexSample,
    Taps: AsRef<[T::Real]>,
    Hist: AsRef<[T]> + AsMut<[T]>,
    Tw: AsRef<[T]>,
{
    /// `2 / M`: output steps (of `M` values) per input sample.
    fn ratio(&self) -> Ratio {
        Ratio::new(2, self.channels as u64).unwrap_or(Ratio::ONE)
    }

    /// `(M L - 1) / 2` input samples.
    fn group_delay(&self) -> Delay {
        let n = (self.channels * self.taps_per_branch) as u64;
        Delay::new(n.saturating_sub(1), 2).unwrap_or(Delay::ZERO)
    }
}

/// The polyphase channelizer (see the [module documentation](self) for the definition).
///
/// `output` is filled in groups of `M` values (channel 0 to `M - 1` of one time step); its length
/// must be a multiple of `M` (`LengthMismatch`). The kernel consumes the input as far as the
/// output space allows: a sample that would complete a hop without room for its output step is
/// not consumed (the caller resubmits it); a partial hop at the end of the input is kept in the
/// history. `Progress` counts the input samples consumed and the output values produced (a
/// multiple of `M`). The hop boundaries do not depend on the chunking, so any split of the input
/// gives the same output.
///
/// `C32`: the class is *tolerance* (SNR >= 100 dB against the f64 model of `Reference`): the
/// branch sums in f32 without FMA (ascending rows), then the `fft` kernel (or a radix-2 transform
/// of the same order for 4 and 8 channels).
pub trait Channelize<T: ComplexSample> {
    /// Runs the filter bank over `input` (see the trait documentation).
    ///
    /// # Errors
    /// `LengthMismatch` if `output.len()` is not a multiple of the channel count.
    fn channelize<Taps, Hist, Tw>(
        &self,
        s: &mut ChannelizeState<T, Taps, Hist, Tw>,
        input: &[T],
        output: &mut [T],
    ) -> Result<Progress>
    where
        Taps: AsRef<[T::Real]>,
        Hist: AsRef<[T]> + AsMut<[T]>,
        Tw: AsRef<[T]>;
}

/// The driver of every channelizer implementation: consumes the input into the history hop by
/// hop and calls `hop(taps, window, swap, plan, group)` for every complete hop, with the
/// time-reversed taps, the window of the newest `M L` samples (oldest first), whether the
/// column sums are swapped by half (`swap`: an even output index, `s = D`), the plan and the
/// output group of `M` values.
///
/// # Errors
/// `LengthMismatch` for an output that is not a multiple of `M`, and the errors of `hop`.
#[allow(
    clippy::inline_always,
    reason = "a backend calls it at its dispatched level, whose target features the hop must inherit"
)]
#[inline(always)]
pub fn channelize_with<T, Taps, Hist, Tw, H>(
    s: &mut ChannelizeState<T, Taps, Hist, Tw>,
    input: &[T],
    output: &mut [T],
    mut hop: H,
) -> Result<Progress>
where
    T: ComplexSample,
    Taps: AsRef<[T::Real]>,
    Hist: AsRef<[T]> + AsMut<[T]>,
    Tw: AsRef<[T]>,
    H: FnMut(&[T::Real], &[T], bool, &FftPlan<T, Tw>, &mut [T]) -> Result<()>,
{
    let m = s.channels;
    let d = m / 2;
    let n = m * s.taps_per_branch;
    let whole = output.len() - output.len() % m;
    if whole != output.len() {
        return Err(Error::LengthMismatch {
            expected: whole,
            got: output.len(),
        });
    }
    let ChannelizeState {
        taps,
        hist,
        plan,
        pos,
        phase,
        odd,
        ..
    } = s;
    let taps = taps.as_ref();
    let hist = hist.as_mut();
    let cap = hist.len();
    let mut groups = output.chunks_exact_mut(m);
    let mut rest = input;
    let mut consumed = 0;
    let mut produced = 0;
    while !rest.is_empty() {
        if *phase == 0 && *pos + d > cap {
            // Move the newest `N - D` samples to the start of the history.
            let keep = n - d;
            let from = pos.checked_sub(keep).ok_or_else(internal)?;
            hist.get_mut(..*pos)
                .ok_or_else(internal)?
                .copy_within(from.., 0);
            *pos = keep;
        }
        let need = d - *phase;
        let blocked = rest.len() >= need && groups.len() == 0;
        let take = if blocked {
            need - 1
        } else {
            need.min(rest.len())
        };
        let (now, later) = rest.split_at(take);
        // An element loop, not `copy_from_slice`: a hop is short (`M / 2` samples), and the
        // call of `memcpy` that the slice copy becomes costs more than the copy itself.
        for (h, x) in hist
            .get_mut(*pos..*pos + take)
            .ok_or_else(internal)?
            .iter_mut()
            .zip(now)
        {
            *h = *x;
        }
        *pos += take;
        *phase += take;
        consumed += take;
        rest = later;
        if blocked {
            break;
        }
        if *phase == d {
            let group = groups.next().ok_or_else(internal)?;
            let start = pos.checked_sub(n).ok_or_else(internal)?;
            let window = hist.get(start..*pos).ok_or_else(internal)?;
            hop(taps, window, !*odd, plan, group)?;
            *odd = !*odd;
            *phase = 0;
            produced += m;
        }
    }
    Ok(Progress { consumed, produced })
}

/// Sums of `B` columns starting at `off` over all rows, in f32 without FMA (ascending rows), into
/// `dst` (`B` values).
#[allow(
    clippy::inline_always,
    reason = "the column block is the loop body of the branch sums; it must not become a call"
)]
#[inline(always)]
fn column_block<const B: usize>(
    taps: &[f32],
    window: &[C32],
    m: usize,
    off: usize,
    dst: &mut [C32],
) -> Result<()> {
    // Whole rows of `M` values: `M` is a power of two, and written as a shift the compiler
    // counts the rows with a shift instead of a division; the columns `off .. off + B` of a row
    // are checked against the row length, which does not change, so the check leaves the loop.
    let m = 1usize << m.trailing_zeros();
    let mut rows = taps
        .chunks_exact(m)
        .zip(window.chunks_exact(m))
        .map(|(t, w)| {
            t.get(off..off + B)
                .and_then(|t| t.first_chunk::<B>())
                .zip(w.get(off..off + B).and_then(|w| w.first_chunk::<B>()))
                .ok_or_else(internal)
        });
    let mut acc = [C32::new(0.0, 0.0); B];
    let (t0, w0) = rows.next().ok_or_else(internal)??;
    for ((a, &g), x) in acc.iter_mut().zip(t0).zip(w0) {
        *a = C32::new(g * x.re, g * x.im);
    }
    for r in rows {
        let (t, w) = r?;
        for ((a, &g), x) in acc.iter_mut().zip(t).zip(w) {
            *a = C32::new(a.re + g * x.re, a.im + g * x.im);
        }
    }
    dst.get_mut(..B).ok_or_else(internal)?.copy_from_slice(&acc);
    Ok(())
}

/// Sums the columns `off .. off + dst.len()` in blocks of `B`.
#[allow(
    clippy::inline_always,
    reason = "a backend calls it at its dispatched level, whose target features it must inherit"
)]
#[inline(always)]
fn column_range<const B: usize>(
    taps: &[f32],
    window: &[C32],
    m: usize,
    off: usize,
    dst: &mut [C32],
) -> Result<()> {
    for (b, chunk) in dst.chunks_exact_mut(B).enumerate() {
        column_block::<B>(taps, window, m, off + b * B, chunk)?;
    }
    Ok(())
}

/// The branch sums of one hop: `z[c] = sum_p g[p M + c] w[p M + c]` for the time-reversed taps
/// `taps` and the `window` (both `M L` long, `M = group.len()`), each column summed over the rows
/// in ascending order in f32 without FMA (the first product, then one addition per row), written
/// to `group[(c + s) mod M]` with `s = M / 2` when `swap` is set and `s = 0` otherwise.
///
/// The columns are summed in blocks of eight (four, two for `M` 8 and 4) whose partial sums stay
/// in registers over all rows; the compiler vectorizes the blocks.
///
/// # Errors
/// `Internal` if the lengths do not fit (they are checked by the state).
#[allow(
    clippy::inline_always,
    reason = "a backend calls it at its dispatched level, whose target features it must inherit"
)]
#[inline(always)]
pub fn branch_sums(taps: &[f32], window: &[C32], swap: bool, group: &mut [C32]) -> Result<()> {
    let m = group.len();
    let d = m / 2;
    if m < MIN_CHANNELS
        || !m.is_power_of_two()
        || taps.len() != window.len()
        || taps.len() & (m - 1) != 0
    {
        return Err(internal());
    }
    let (lo, hi) = group.split_at_mut(d);
    let (first, second) = if swap { (hi, lo) } else { (lo, hi) };
    match d {
        2 => {
            column_range::<2>(taps, window, m, 0, first)?;
            column_range::<2>(taps, window, m, d, second)
        }
        4 => {
            column_range::<4>(taps, window, m, 0, first)?;
            column_range::<4>(taps, window, m, d, second)
        }
        _ => {
            column_range::<8>(taps, window, m, 0, first)?;
            column_range::<8>(taps, window, m, d, second)
        }
    }
}

/// The forward unnormalised FFT of one output group: the `fft` kernel of `fft` for 16 points and
/// more, an in-place radix-2 decimation-in-time transform over the plan's twiddles for 4 and 8
/// points (the order of the fallback FFT: the bit-reversal permutation, then the stages with the
/// butterfly `a + w b`, `a - w b`, the product in f32 without FMA).
///
/// # Errors
/// `LengthMismatch` if `data.len()` differs from the plan, and the errors of `fft`.
#[allow(
    clippy::inline_always,
    reason = "a backend calls it at its dispatched level, whose target features it must inherit"
)]
#[inline(always)]
pub fn channel_fft<F: Fft<C32>, Tw: AsRef<[C32]>>(
    fft: &F,
    plan: &FftPlan<C32, Tw>,
    data: &mut [C32],
) -> Result<()> {
    let n = plan.len();
    if data.len() != n {
        return Err(Error::LengthMismatch {
            expected: n,
            got: data.len(),
        });
    }
    if log2_of(n).is_ok() {
        fft.fft(plan, data, Direction::Forward, Scaling::Default)?;
        return Ok(());
    }
    // 4 or 8 points: bit reversal, then the radix-2 stages.
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            data.swap(i, j);
        }
    }
    let tw = plan.twiddles();
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = n / len;
        for block in data.chunks_exact_mut(len) {
            let (a, b) = block.split_at_mut(half);
            for (j, (a, b)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
                let w = *tw.get(j * step).ok_or_else(internal)?;
                let t = C32::new(b.re * w.re - b.im * w.im, b.re * w.im + b.im * w.re);
                let u = *a;
                *a = C32::new(u.re + t.re, u.im + t.im);
                *b = C32::new(u.re - t.re, u.im - t.im);
            }
        }
        len *= 2;
    }
    Ok(())
}
