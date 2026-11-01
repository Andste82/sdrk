//! Fallback implementation of `fir_interpolate`: the polyphase resampler with `down = 1`.
//!
//! `C32` and `f32` call the engine of `resample` (`fallback/decim.rs`). `Q15` and `CQ15` have
//! their own loop over the outputs of a branch, after the scalar FIR filters of CMSIS-DSP
//! (`arm_fir_interpolate_q15`): a few outputs together with their samples in a ring of
//! registers, two loads and one product per output for each tap ([`sums_contiguous`]). The
//! engine's groups of 16 or 8 outputs load every sample of the group again for each tap; that
//! only pays where the compiler turns them into wide vector products (`CQ15` in an AVX2 build,
//! which keeps the engine).

use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::decim::{Resample, ResampleReal};
use crate::kernels::filter::update_history;
use crate::kernels::interp::{
    FirInterpRealState, FirInterpState, FirInterpolate, FirInterpolateReal,
};
use crate::num::sat16;
use crate::timing::Progress;
use crate::types::{C32, CQ15, Q15};

use super::decim::{FbMac, Out, branch_len, single};

macro_rules! interpolate_impl {
    ($tr:ident, $method:ident, $inner:ident, $resample:ident, $state:ident, $t:ty, $tap:ty) => {
        impl $tr<$t> for Fallback {
            fn $method<Taps, Hist>(
                &self,
                state: &mut $state<$t, Taps, Hist>,
                input: &[$t],
                output: &mut [$t],
            ) -> Result<Progress>
            where
                Taps: AsRef<[$tap]>,
                Hist: AsRef<[$t]> + AsMut<[$t]>,
            {
                $inner::<$t>::$resample(self, state.inner_mut(), input, output)
            }
        }
    };
}

interpolate_impl!(
    FirInterpolate,
    fir_interpolate,
    Resample,
    resample,
    FirInterpState,
    C32,
    f32
);
interpolate_impl!(
    FirInterpolateReal,
    fir_interpolate_real,
    ResampleReal,
    resample_real,
    FirInterpRealState,
    f32,
    f32
);
// In an AVX2 build the compiler turns the 8 outputs of the engine's `CQ15` group into vector
// products, which simulated faster on `znver3` than the ring (79.42 against 82.77 cycles per
// sample); the SIMD backend runs this fallback at `avx2` and `avx512`.
#[cfg(target_feature = "avx2")]
interpolate_impl!(
    FirInterpolate,
    fir_interpolate,
    Resample,
    resample,
    FirInterpState,
    CQ15,
    Q15
);

#[cfg(not(target_feature = "avx2"))]
impl FirInterpolate<CQ15> for Fallback {
    fn fir_interpolate<Taps, Hist>(
        &self,
        state: &mut FirInterpState<CQ15, Taps, Hist>,
        input: &[CQ15],
        output: &mut [CQ15],
    ) -> Result<Progress>
    where
        Taps: AsRef<[Q15]>,
        Hist: AsRef<[CQ15]> + AsMut<[CQ15]>,
    {
        let p = state.inner_mut().parts_mut();
        interpolate::<CQ15, CQ15_GROUP>(p.taps, p.hist, p.up, p.q, input, output)
    }
}

impl FirInterpolateReal<Q15> for Fallback {
    fn fir_interpolate_real<Taps, Hist>(
        &self,
        state: &mut FirInterpRealState<Q15, Taps, Hist>,
        input: &[Q15],
        output: &mut [Q15],
    ) -> Result<Progress>
    where
        Taps: AsRef<[Q15]>,
        Hist: AsRef<[Q15]> + AsMut<[Q15]>,
    {
        let p = state.inner_mut().parts_mut();
        interpolate::<Q15, Q15_GROUP>(p.taps, p.hist, p.up, p.q, input, output)
    }
}

/// Outputs of one branch that [`groups`] computes together for `Q15` (one `i64` sum each): 8 on
/// AArch64, x86-64 and wasm32 (8 sums and 8 samples in registers or locals), 4 elsewhere (32-bit
/// Arm has 13 registers for `i64` sums of two each).
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
))]
const Q15_GROUP: usize = 8;
#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
)))]
const Q15_GROUP: usize = 4;
/// The same for `CQ15` (two sums each): 4, 8 on wasm32, 2 on 32-bit Arm (not used in an AVX2
/// build).
#[cfg(not(any(target_arch = "arm", target_arch = "wasm32")))]
#[cfg_attr(target_feature = "avx2", allow(dead_code))]
const CQ15_GROUP: usize = 4;
#[cfg(target_arch = "wasm32")]
const CQ15_GROUP: usize = 8;
#[cfg(target_arch = "arm")]
const CQ15_GROUP: usize = 2;
/// The longest branch whose taps [`branch`] copies into a contiguous buffer on the stack
/// (256 bytes), and whose outputs at the start of a call take their windows from a stack copy
/// of history and input (`TAP_COPY + G - 1` samples); longer branches read their taps with the
/// stride and sum those outputs one at a time.
const TAP_COPY: usize = 128;
/// The largest group size (of the window buffer of [`branch`]).
const MAX_GROUP: usize = 8;

/// `resample` with `up = l` and `down = 1` (the plan of `fallback::decim::polyphase`), each
/// branch evaluated by [`branch`].
fn interpolate<T: Ring, const G: usize>(
    taps: &[T::Tap],
    hist: &mut [T],
    l: usize,
    q: &mut i64,
    input: &[T],
    output: &mut [T],
) -> Result<Progress> {
    let internal = || Error::Internal("interpolator window");
    let ll = i64::try_from(l).map_err(|_| internal())?;
    let q0 = *q;
    let (consumed, produced) = plan(l, q, input.len(), output.len())?;
    // Output `j` is ready after `c_j` samples (the fewest that make `q0 + j - L c_j` negative)
    // with the branch `q0 + j - L c_j + L`; outputs `j` and `j + L` share the branch, and their
    // windows lie one sample apart.
    let out = output.get_mut(..produced).ok_or_else(internal)?;
    for first in 0..l.min(produced) {
        let at = i128::from(q0) + first as i128;
        let ready = if at < 0 { 0 } else { at / i128::from(ll) + 1 };
        let phase = usize::try_from(at - i128::from(ll) * ready + i128::from(ll))
            .map_err(|_| internal())?;
        let seen = usize::try_from(ready).map_err(|_| internal())?;
        let count = (produced - first).div_ceil(l);
        let dst = out
            .get_mut(first..)
            .unwrap_or_default()
            .iter_mut()
            .step_by(l);
        branch::<T, G>(taps, l, hist, input, phase, seen, count, dst)?;
    }
    update_history(hist, input.get(..consumed).ok_or_else(internal)?)?;
    Ok(Progress { consumed, produced })
}

/// The state machine of `fallback::decim::polyphase` with `down = 1` (`q < 0`: `-q` outputs of
/// the last sample are pending; produce one and add 1, else consume a sample and subtract `L`)
/// run over `inputs` samples and `room` outputs: `(consumed, produced)`, `q` updated. With
/// `-L <= q <= 0`, the range the machine keeps, the counts follow in closed form; another `q`
/// runs the machine.
fn plan(l: usize, q: &mut i64, inputs: usize, room: usize) -> Result<(usize, usize)> {
    let internal = || Error::Internal("interpolator plan");
    let ll = i64::try_from(l).map_err(|_| internal())?;
    if (-ll..=0).contains(&*q) {
        let pending = usize::try_from(-*q).map_err(|_| internal())?;
        if room < pending {
            *q += i64::try_from(room).map_err(|_| internal())?;
            return Ok((0, room));
        }
        // After the pending outputs every sample gives `L` outputs; the machine consumes a
        // sample whenever none is pending and one is left, also when the output is full.
        let (full, part) = ((room - pending) / l, (room - pending) % l);
        if inputs <= full {
            *q = 0;
            return Ok((inputs, pending + l * inputs));
        }
        *q = i64::try_from(part).map_err(|_| internal())? - ll;
        return Ok((full + 1, room));
    }
    let (mut consumed, mut produced) = (0usize, 0usize);
    loop {
        if *q < 0 {
            if produced == room {
                break;
            }
            produced += 1;
            *q += 1;
        } else {
            if consumed == inputs {
                break;
            }
            consumed += 1;
            *q -= ll;
        }
    }
    Ok((consumed, produced))
}

/// The `count` outputs of the branch `phase` whose windows end after `seen, seen + 1, ...`
/// samples of `input`, written to `dst` in order, `G` at a time with [`groups`] (the outputs
/// whose windows reach into the history from a stack copy of history and input) and the last
/// ones of the call one by one with [`single_exact`]. A branch of up to [`TAP_COPY`] taps is
/// first copied into a contiguous buffer, so that the group reads its taps without the stride
/// and without a bounds check per tap.
#[allow(clippy::too_many_arguments)]
fn branch<'a, T: Ring + 'a, const G: usize>(
    taps: &[T::Tap],
    l: usize,
    hist: &[T],
    input: &[T],
    phase: usize,
    seen: usize,
    count: usize,
    mut dst: impl Iterator<Item = &'a mut T>,
) -> Result<()> {
    let internal = || Error::Internal("interpolator branch");
    let len = branch_len(taps.len(), l, phase);
    let branch = taps.get(phase..).unwrap_or(&[]);
    // The groups and `single_exact` sum `Q15` products in an `i64` without saturation: exact
    // while the sum of their magnitudes (at most 2^30 each) stays below 2^63, so for fewer than
    // 2^32 taps. An empty branch (`L > N`) gives zeros.
    if len == 0 || u32::try_from(len).is_err() {
        for (t, y) in (0..count).zip(dst) {
            *y = single(
                taps,
                l,
                hist,
                input,
                Out {
                    phase,
                    seen: seen + t,
                },
            )?;
        }
        return Ok(());
    }
    let mut copy = [Q15(0); TAP_COPY];
    let mut t = 0;
    let hs = if len <= TAP_COPY {
        for (d, h) in copy.iter_mut().zip(branch.iter().step_by(l)) {
            *d = *h;
        }
        let hs = BranchTaps::Contiguous(copy.get(..len).ok_or_else(internal)?);
        // The windows of the outputs `at < len` begin in the history: `window[j]` is sample
        // `j - len + at` of `input` (before 0: of the history).
        let mut window = [T::default(); TAP_COPY + MAX_GROUP];
        while seen + t < len && t + G <= count {
            let at = seen + t;
            let (from_hist, from_input) = window
                .get_mut(..len + G - 1)
                .ok_or_else(internal)?
                .split_at_mut(len - at);
            let tail = hist
                .get(hist.len().checked_sub(len - at).ok_or_else(internal)?..)
                .ok_or_else(internal)?;
            from_hist.copy_from_slice(tail);
            let head = input.get(..at + G - 1).ok_or_else(internal)?;
            from_input.copy_from_slice(head);
            let stitched = window.get(..len + G - 1).ok_or_else(internal)?;
            groups::<T, G, _>(hs, len, stitched, len, 1, &mut dst)?;
            t += G;
        }
        hs
    } else {
        BranchTaps::Strided(branch, l)
    };
    while t < count {
        let at = seen + t;
        if at >= len && t + G <= count {
            let n = (count - t) / G;
            groups::<T, G, _>(hs, len, input, at, n, &mut dst)?;
            t += n * G;
        } else {
            *dst.next().ok_or_else(internal)? = single_exact(hs, len, hist, input, at)?;
            t += 1;
        }
    }
    Ok(())
}

/// The engine's `single` with the exact sums of [`Ring`] (a branch of `len < 2^32` taps): the
/// output whose window ends after `seen` samples of `input`, the samples of `input` first, then
/// those of the history.
fn single_exact<T: Ring>(
    hs: BranchTaps<'_>,
    len: usize,
    hist: &[T],
    input: &[T],
    seen: usize,
) -> Result<T> {
    let internal = || Error::Internal("interpolator window");
    let (taps, step) = match hs {
        BranchTaps::Contiguous(taps) => (taps, 1),
        BranchTaps::Strided(taps, l) => (taps, l),
    };
    let mut hs = taps.iter().step_by(step).take(len);
    let mut sum = T::empty();
    // The samples first: the zip stops on them without taking a tap too many.
    let xs = input.get(seen - len.min(seen)..seen).ok_or_else(internal)?;
    for (x, h) in xs.iter().rev().zip(hs.by_ref()) {
        T::add(&mut sum, i32::from(h.0), x.widen());
    }
    if seen < len {
        let from = hist.len().checked_sub(len - seen).ok_or_else(internal)?;
        let xs = hist.get(from..).ok_or_else(internal)?;
        for (x, h) in xs.iter().rev().zip(hs) {
            T::add(&mut sum, i32::from(h.0), x.widen());
        }
    }
    Ok(T::round(sum))
}

/// A sample type of the integer interpolator: the widened sample of the products and the exact
/// sums of one output.
trait Ring: FbMac<Tap = Q15> + Default {
    /// The sample in 32-bit integers (kept in the ring, so it is extended once per tap).
    type Wide: Copy;
    /// The sums of one output: products of at most 2^30 in magnitude, fewer than 2^32 of them,
    /// in an `i64` without saturation (exact, see [`branch`]).
    type Sum: Copy;
    fn widen(self) -> Self::Wide;
    fn empty() -> Self::Sum;
    fn add(sum: &mut Self::Sum, h: i32, x: Self::Wide);
    /// `sat16(round_shift(sum, 15))`, the finish of `Acc64` (see [`round_q15`]).
    fn round(sum: Self::Sum) -> Self;
}

/// `sat16(round_shift(sum, 15))` for a sum of [`Ring`]: `|sum| <= 2^62`, so the rounding
/// constant is added without the saturation of `round_shift` (the same value).
#[inline]
fn round_q15(sum: i64) -> Q15 {
    Q15(sat16(sum.wrapping_add(1 << 14) >> 15))
}

impl Ring for Q15 {
    type Wide = i32;
    type Sum = i64;
    #[inline]
    fn widen(self) -> i32 {
        i32::from(self.0)
    }
    #[inline]
    fn empty() -> i64 {
        0
    }
    #[inline]
    fn add(sum: &mut i64, h: i32, x: i32) {
        *sum = sum.wrapping_add(i64::from(h) * i64::from(x));
    }
    #[inline]
    fn round(sum: i64) -> Q15 {
        round_q15(sum)
    }
}

impl Ring for CQ15 {
    type Wide = (i32, i32);
    type Sum = (i64, i64);
    #[inline]
    fn widen(self) -> (i32, i32) {
        (i32::from(self.re.0), i32::from(self.im.0))
    }
    #[inline]
    fn empty() -> (i64, i64) {
        (0, 0)
    }
    #[inline]
    fn add(sum: &mut (i64, i64), h: i32, x: (i32, i32)) {
        sum.0 = sum.0.wrapping_add(i64::from(h) * i64::from(x.0));
        sum.1 = sum.1.wrapping_add(i64::from(h) * i64::from(x.1));
    }
    #[inline]
    fn round(sum: (i64, i64)) -> CQ15 {
        CQ15 {
            re: round_q15(sum.0),
            im: round_q15(sum.1),
        }
    }
}

/// Where the taps of a branch come from: copied into a contiguous buffer (branches of up to
/// [`TAP_COPY`] taps), or read with the stride `l` from the taps of the filter.
#[derive(Clone, Copy)]
enum BranchTaps<'a> {
    Contiguous(&'a [Q15]),
    Strided(&'a [Q15], usize),
}

/// `n` groups of `G` outputs of a branch of `len >= 1` taps, written to `dst` in order: group
/// `g` holds the outputs whose windows end after `at + g G, at + g G + 1, ...` samples of `x`.
/// One function for all groups of a call, so that the outputs go to `dst` from registers.
#[inline(never)]
fn groups<'a, T: Ring + 'a, const G: usize, D: Iterator<Item = &'a mut T>>(
    hs: BranchTaps<'_>,
    len: usize,
    x: &[T],
    at: usize,
    n: usize,
    dst: &mut D,
) -> Result<()> {
    let internal = || Error::Internal("interpolator group");
    let window = |g: usize| {
        let end = at + g * G;
        x.get(end - len..end + G - 1).ok_or_else(internal)
    };
    match hs {
        BranchTaps::Contiguous(hs) => {
            for g in 0..n {
                for s in sums_contiguous::<T, G>(hs, len, window(g)?)? {
                    *dst.next().ok_or_else(internal)? = T::round(s);
                }
            }
        }
        BranchTaps::Strided(branch, l) => {
            for g in 0..n {
                for s in sums_strided::<T, G>(branch, l, len, window(g)?)? {
                    *dst.next().ok_or_else(internal)? = T::round(s);
                }
            }
        }
    }
    Ok(())
}

/// The ring before tap 0 of a group (the samples `w[len - 1 + j]`; position 0 is loaded again
/// by tap 0) and the new samples of the taps (oldest first: tap `i` takes `samples[len - 1 - i]`).
#[allow(
    clippy::inline_always,
    reason = "the group size must be a constant in the hot loops"
)]
#[inline(always)]
fn start<T: Ring, const G: usize>(len: usize, w: &[T]) -> Result<([T::Wide; G], &[T])> {
    let internal = || Error::Internal("interpolator group");
    let w = w.get(..len + G - 1).ok_or_else(internal)?;
    let samples = w.get(..len).ok_or_else(internal)?;
    let top: &[T; G] = w
        .get(len - 1..)
        .and_then(|top| top.try_into().ok())
        .ok_or_else(internal)?;
    Ok((top.map(T::widen), samples))
}

/// The sums of the `G` outputs of a branch with the taps `hs` (`len >= 1` of them), output `t`
/// with the window `w[t..t + len]`: tap `i` multiplies `w[len - 1 - i + t]` into output `t`.
///
/// The samples of tap `i` are those of tap `i - 1` with `w[len - 1 - i]` in place of
/// `w[len - 1 - i + G]`: a ring of `G` samples (`w[n]` at `(n - len + 1) mod G`, so the sample
/// of output `t` at tap `i` is at `(t - i) mod G`) takes one new sample per tap, and a tap costs
/// two loads and `G` products, as in the scalar FIR filters of CMSIS-DSP
/// (`arm_fir_interpolate_q15` keeps four samples in variables the same way). The taps go in
/// blocks of `G`, in which the ring positions are constants. Each output adds its products in
/// ascending tap order into its own exact `i64` sum, so the result is that of the
/// one-at-a-time evaluation.
#[allow(
    clippy::inline_always,
    reason = "the group size must be a constant in the hot loops"
)]
#[inline(always)]
fn sums_contiguous<T: Ring, const G: usize>(
    hs: &[Q15],
    len: usize,
    w: &[T],
) -> Result<[T::Sum; G]> {
    let internal = || Error::Internal("interpolator group");
    let hs = hs.get(..len).ok_or_else(internal)?;
    let (mut ring, samples) = start::<T, G>(len, w)?;
    let mut sums = [T::empty(); G];
    let (tap_blocks, sample_blocks) = (hs.chunks_exact(G), samples.rchunks_exact(G));
    let (h_rest, x_rest) = (tap_blocks.remainder(), sample_blocks.remainder());
    for (h, xs) in tap_blocks.zip(sample_blocks) {
        let h: &[Q15; G] = h.try_into().map_err(|_| internal())?;
        let xs: &[T; G] = xs.try_into().map_err(|_| internal())?;
        ring_taps::<T, G>(&mut sums, &mut ring, h, xs, G)?;
    }
    ring_taps::<T, G>(&mut sums, &mut ring, h_rest, x_rest, x_rest.len())?;
    Ok(sums)
}

/// [`sums_contiguous`] with the taps `branch[i l]` (`len` of them).
#[allow(
    clippy::inline_always,
    reason = "the group size must be a constant in the hot loops"
)]
#[inline(always)]
fn sums_strided<T: Ring, const G: usize>(
    branch: &[Q15],
    l: usize,
    len: usize,
    w: &[T],
) -> Result<[T::Sum; G]> {
    let internal = || Error::Internal("interpolator group");
    let mut taps = branch.iter().step_by(l);
    let (mut ring, samples) = start::<T, G>(len, w)?;
    let mut sums = [T::empty(); G];
    let mut h = [Q15(0); G];
    let sample_blocks = samples.rchunks_exact(G);
    let x_rest = sample_blocks.remainder();
    for xs in sample_blocks {
        let xs: &[T; G] = xs.try_into().map_err(|_| internal())?;
        for d in &mut h {
            *d = *taps.next().ok_or_else(internal)?;
        }
        ring_taps::<T, G>(&mut sums, &mut ring, &h, xs, G)?;
    }
    for d in h.iter_mut().take(x_rest.len()) {
        *d = *taps.next().ok_or_else(internal)?;
    }
    ring_taps::<T, G>(&mut sums, &mut ring, &h, x_rest, x_rest.len())?;
    Ok(sums)
}

/// `n <= G` taps `h[k]` of a block that begins with a tap `i = 0 mod G`; tap `k` takes the new
/// sample `xs[n - 1 - k]` into the ring position `-i mod G`.
#[allow(
    clippy::inline_always,
    reason = "the ring positions must be constants in the hot loop"
)]
#[inline(always)]
fn ring_taps<T: Ring, const G: usize>(
    sums: &mut [T::Sum; G],
    ring: &mut [T::Wide; G],
    h: &[Q15],
    xs: &[T],
    n: usize,
) -> Result<()> {
    let internal = || Error::Internal("interpolator group");
    for k in 0..G {
        if k == n {
            break;
        }
        let slot = (G - k) % G;
        *ring.get_mut(slot).ok_or_else(internal)? = xs.get(n - 1 - k).ok_or_else(internal)?.widen();
        let h = i32::from(h.get(k).ok_or_else(internal)?.0);
        for (t, s) in sums.iter_mut().enumerate() {
            T::add(s, h, *ring.get((slot + t) % G).ok_or_else(internal)?);
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
mod tests {
    //! The ring groups must give the bits of the definition evaluated one output at a time
    //! (taps ascending, one saturating `Acc64` per output) for every factor, branch length
    //! (every tail of a tap block), split into calls and output buffer size, also at the
    //! extremes of `Q15`.

    use std::vec;
    use std::vec::Vec;

    use super::{CQ15, Fallback, FirInterpolate, FirInterpolateReal, Q15};
    use crate::kernels::interp::{FirInterpRealState, FirInterpState};
    use crate::num::Acc64;

    fn next(s: &mut u32) -> u32 {
        *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *s >> 8
    }

    fn q(s: &mut u32, extreme: bool) -> Q15 {
        if extreme {
            Q15(if next(s) & 1 == 0 { i16::MIN } else { i16::MAX })
        } else {
            Q15(next(s) as i16)
        }
    }

    /// `upfirdn(h, x, l, 1)` for the outputs of the received input, one output at a time.
    fn model(taps: &[Q15], l: usize, x: &[(Q15, Q15)]) -> Vec<(Q15, Q15)> {
        (0..x.len() * l)
            .map(|j| {
                let (p, b) = (j % l, j / l);
                let (mut re, mut im) = (Acc64::default(), Acc64::default());
                for (i, h) in taps.iter().skip(p).step_by(l).enumerate() {
                    if i <= b {
                        re.mac(*h, x[b - i].0);
                        im.mac(*h, x[b - i].1);
                    }
                }
                (re.finish(), im.finish())
            })
            .collect()
    }

    /// Feeds `x` in calls of random lengths with output buffers of random sizes (resubmitting
    /// what was not consumed) to the `CQ15` and the `Q15` interpolators and compares the bits
    /// with [`model`] (`Q15`: the real parts).
    fn check(seed: u32, n_taps: usize, l: usize, extreme: bool) {
        let s = &mut (seed.wrapping_mul(2_654_435_761) | 1);
        let taps: Vec<Q15> = (0..n_taps).map(|_| q(s, extreme)).collect();
        let x: Vec<(Q15, Q15)> = (0..500).map(|_| (q(s, extreme), q(s, extreme))).collect();
        let want = model(&taps, l, &x);
        let hist = n_taps.div_ceil(l);
        let mut cq = FirInterpState::new(taps.clone(), vec![CQ15::default(); hist], l).unwrap();
        let mut rq = FirInterpRealState::new(taps, vec![Q15::default(); hist], l).unwrap();
        let xc: Vec<CQ15> = x.iter().map(|&(re, im)| CQ15 { re, im }).collect();
        let xr: Vec<Q15> = x.iter().map(|&(re, _)| re).collect();
        let (mut got_c, mut got_r) = (Vec::new(), Vec::new());
        let (mut at, mut stalls) = (0, 0);
        while at < x.len() {
            // Short calls (windows in the history) and long ones (groups in the input).
            let take =
                (next(s) as usize % if next(s) & 1 == 0 { 30 } else { 200 }).min(x.len() - at);
            let room = next(s) as usize % 500;
            let mut oc = vec![CQ15::default(); room];
            let mut or = vec![Q15::default(); room];
            let pc = Fallback
                .fir_interpolate(&mut cq, &xc[at..at + take], &mut oc)
                .unwrap();
            let pr = Fallback
                .fir_interpolate_real(&mut rq, &xr[at..at + take], &mut or)
                .unwrap();
            assert_eq!((pc.consumed, pc.produced), (pr.consumed, pr.produced));
            got_c.extend_from_slice(&oc[..pc.produced]);
            got_r.extend_from_slice(&or[..pr.produced]);
            at += pc.consumed;
            stalls = if pc.consumed == 0 { stalls + 1 } else { 0 };
            assert!(stalls < 1000, "no progress");
        }
        // The outputs still pending of the last sample.
        let mut oc = vec![CQ15::default(); 8];
        let mut or = vec![Q15::default(); 8];
        let pc = Fallback.fir_interpolate(&mut cq, &[], &mut oc).unwrap();
        let pr = Fallback
            .fir_interpolate_real(&mut rq, &[], &mut or)
            .unwrap();
        got_c.extend_from_slice(&oc[..pc.produced]);
        got_r.extend_from_slice(&or[..pr.produced]);
        assert_eq!(got_c.len(), want.len(), "outputs, {n_taps} taps, L = {l}");
        assert_eq!(got_r.len(), want.len(), "outputs, {n_taps} taps, L = {l}");
        for (j, ((c, r), w)) in got_c.iter().zip(&got_r).zip(&want).enumerate() {
            assert_eq!(
                (c.re.0, c.im.0),
                (w.0.0, w.1.0),
                "CQ15 output {j}, {n_taps} taps, L = {l}"
            );
            assert_eq!(r.0, w.0.0, "Q15 output {j}, {n_taps} taps, L = {l}");
        }
    }

    #[test]
    fn q15_and_cq15_groups_equal_the_definition_bit_for_bit() {
        for l in [1, 2, 3, 5] {
            // Branch lengths 1 to 10 cover every tail of a tap block and, with `L = 5`, the
            // empty branches of `L > N`.
            for n_taps in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 20, 31, 41, 64, 65, 128] {
                for extreme in [false, true] {
                    check(n_taps as u32 * 10 + l as u32, n_taps, l, extreme);
                }
            }
        }
    }

    #[test]
    fn branches_longer_than_the_tap_copy_equal_the_definition_bit_for_bit() {
        // Branches of `TAP_COPY` taps (copied) and of more (read with the stride), with every
        // tail of a tap block.
        let copy = super::TAP_COPY;
        for (l, n_taps) in [
            (1, copy),
            (1, copy + 1),
            (2, 2 * copy + 3),
            (3, 3 * copy + 7),
        ] {
            for extreme in [false, true] {
                check(n_taps as u32 + l as u32, n_taps, l, extreme);
            }
        }
    }

    #[test]
    fn the_closed_form_plan_equals_the_state_machine() {
        for l in 1..7usize {
            for q0 in -(l as i64)..=0 {
                for inputs in 0..9 {
                    for room in 0..30 {
                        // The state machine of `fallback::decim::polyphase` with `down = 1`.
                        let (mut q, mut consumed, mut produced) = (q0, 0, 0);
                        loop {
                            if q < 0 {
                                if produced == room {
                                    break;
                                }
                                produced += 1;
                                q += 1;
                            } else {
                                if consumed == inputs {
                                    break;
                                }
                                consumed += 1;
                                q -= l as i64;
                            }
                        }
                        let mut got = q0;
                        let counts = super::plan(l, &mut got, inputs, room).unwrap();
                        assert_eq!(counts, (consumed, produced), "L {l} q {q0} {inputs} {room}");
                        assert_eq!(got, q, "L {l} q {q0} {inputs} {room}");
                    }
                }
            }
        }
    }
}
