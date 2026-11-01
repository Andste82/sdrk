//! Fallback implementations of `fir_decimate` and `resample`.

use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::decim::{
    FirDecimState, FirDecimate, Resample, ResampleReal, ResampleRealState, ResampleState,
};
use crate::kernels::filter::update_history;
use crate::num::Acc64;
use crate::timing::Progress;
use crate::types::{C32, CQ15, Q15};

/// Multiply-accumulate rules of the fallback for one sample type.
pub(super) trait FbMac: Copy {
    /// Tap type.
    type Tap: Copy;
    /// Accumulator.
    type Acc: Copy;
    /// Whether [`WIDE_GROUP`] outputs are computed together (the real types, whose outputs
    /// take one accumulator each) instead of [`NARROW_GROUP`].
    const WIDE: bool;
    fn zero() -> Self::Acc;
    fn mac(acc: &mut Self::Acc, h: Self::Tap, x: Self);
    /// [`FbMac::mac`] where the caller guarantees that the sum cannot saturate (fewer than
    /// 2^32 products of at most 2^30 each in an `i64`): the same result without the check.
    #[inline]
    fn mac_exact(acc: &mut Self::Acc, h: Self::Tap, x: Self) {
        Self::mac(acc, h, x);
    }
    fn finish(acc: Self::Acc) -> Self;
    /// The `count` outputs of `fir_decimate` (windows `s` samples apart, the first one after
    /// `seen` samples of `input`): [`progression`], unless the type has a faster form with the
    /// same bits for the target.
    #[inline]
    fn decimate_outputs(
        taps: &[Self::Tap],
        hist: &[Self],
        input: &[Self],
        seen: usize,
        s: usize,
        out: &mut [Self],
    ) -> Result<()> {
        let count = out.len();
        progression(taps, 1, hist, input, 0, seen, s, count, out.iter_mut())
    }
}

impl FbMac for C32 {
    const WIDE: bool = false;
    type Tap = f32;
    type Acc = (f32, f32);
    #[inline]
    fn zero() -> Self::Acc {
        (0.0, 0.0)
    }
    #[inline]
    fn mac(acc: &mut Self::Acc, h: f32, x: C32) {
        acc.0 += h * x.re;
        acc.1 += h * x.im;
    }
    #[inline]
    fn finish(acc: Self::Acc) -> C32 {
        C32::new(acc.0, acc.1)
    }
}

impl FbMac for CQ15 {
    const WIDE: bool = false;
    type Tap = Q15;
    type Acc = (Acc64, Acc64);
    #[inline]
    fn zero() -> Self::Acc {
        (Acc64::default(), Acc64::default())
    }
    #[inline]
    fn mac(acc: &mut Self::Acc, h: Q15, x: CQ15) {
        acc.0.mac(h, x.re);
        acc.1.mac(h, x.im);
    }
    #[inline]
    fn mac_exact(acc: &mut Self::Acc, h: Q15, x: CQ15) {
        acc.0.0 = acc.0.0.wrapping_add(i64::from(h.0) * i64::from(x.re.0));
        acc.1.0 = acc.1.0.wrapping_add(i64::from(h.0) * i64::from(x.im.0));
    }
    #[inline]
    fn finish(acc: Self::Acc) -> CQ15 {
        CQ15 {
            re: acc.0.finish(),
            im: acc.1.finish(),
        }
    }
    #[cfg(target_arch = "aarch64")]
    #[inline]
    fn decimate_outputs(
        taps: &[Q15],
        hist: &[CQ15],
        input: &[CQ15],
        seen: usize,
        s: usize,
        out: &mut [CQ15],
    ) -> Result<()> {
        a64::decimate(taps, hist, input, seen, s, out)
    }
}

impl FbMac for f32 {
    const WIDE: bool = true;
    type Tap = f32;
    type Acc = f32;
    #[inline]
    fn zero() -> f32 {
        0.0
    }
    #[inline]
    fn mac(acc: &mut f32, h: f32, x: f32) {
        *acc += h * x;
    }
    #[inline]
    fn finish(acc: f32) -> f32 {
        acc
    }
}

impl FbMac for Q15 {
    const WIDE: bool = true;
    type Tap = Q15;
    type Acc = Acc64;
    #[inline]
    fn zero() -> Acc64 {
        Acc64::default()
    }
    #[inline]
    fn mac(acc: &mut Acc64, h: Q15, x: Q15) {
        acc.mac(h, x);
    }
    #[inline]
    fn mac_exact(acc: &mut Acc64, h: Q15, x: Q15) {
        acc.0 = acc.0.wrapping_add(i64::from(h.0) * i64::from(x.0));
    }
    #[inline]
    fn finish(acc: Acc64) -> Q15 {
        acc.finish()
    }
}

/// Outputs that [`group`] computes together: 16 for the real types ([`FbMac::WIDE`]) with
/// windows one sample apart, 8 otherwise.
///
/// The outputs of a group share their branch of taps, and their windows lie `s` samples apart
/// (`s = 1` for `fir` and `fir_interpolate`, whose windows of one tap are then consecutive
/// samples). Each output keeps its own accumulator and adds its products in ascending tap order,
/// exactly as it would alone, so the outputs do not depend on the grouping (nor on the split into
/// calls). The group gives the core independent add chains (one output alone is bound by the
/// latency of its chain), and for `s = 1` the products of a tap are an element-wise product of
/// the tap with consecutive samples, which the compiler turns into vector operations.
const WIDE_GROUP: usize = 16;
/// See [`WIDE_GROUP`].
const NARROW_GROUP: usize = 8;

/// One output of a polyphase branch: `y = sum_{i >= 0, phase + i l < N} taps[phase + i l] x(i)`,
/// where `x(i)` is the `i`-th sample before the newest one: `input[seen - 1 - i]` while
/// `i < seen`, then the history from its newest sample backwards. `fir` and `fir_decimate` are
/// the case `l = 1`, `phase = 0`.
#[derive(Clone, Copy, Debug)]
struct Out {
    /// Index of the first tap of the branch (`< l`).
    phase: usize,
    /// Number of samples of `input` up to the newest one of the output.
    seen: usize,
}

/// Number of taps of the branch of `phase`: `ceil((N - phase) / l)`.
#[inline]
fn branch_len(n_taps: usize, l: usize, phase: usize) -> usize {
    n_taps.saturating_sub(phase).div_ceil(l)
}

/// The sum of `out` over its whole branch, in ascending order: the samples of `input` first,
/// then those of the history.
#[inline]
fn single<T: FbMac>(taps: &[T::Tap], l: usize, hist: &[T], input: &[T], out: Out) -> Result<T> {
    let internal = || Error::Internal("filter window");
    let mut acc = T::zero();
    let len = branch_len(taps.len(), l, out.phase);
    let in_end = len.min(out.seen);
    // A branch beyond the taps (`L > N`) is empty.
    let ts = taps.get(out.phase..).unwrap_or_default();
    if in_end > 0 {
        let xs = input
            .get(out.seen - in_end..out.seen)
            .ok_or_else(internal)?;
        for (h, x) in ts.iter().step_by(l).zip(xs.iter().rev()) {
            T::mac(&mut acc, *h, *x);
        }
    }
    if out.seen < len {
        let h_len = hist.len();
        let xs = hist
            .get(h_len.checked_sub(len - out.seen).ok_or_else(internal)?..)
            .ok_or_else(internal)?;
        let ts = taps.get(out.phase + out.seen * l..).ok_or_else(internal)?;
        for (h, x) in ts.iter().step_by(l).zip(xs.iter().rev()) {
            T::mac(&mut acc, *h, *x);
        }
    }
    Ok(T::finish(acc))
}

/// `G` accumulators plus the products of the tap `h` with the consecutive samples `xs`.
#[allow(
    clippy::inline_always,
    reason = "the group size and the strides must be constants in the hot loops"
)]
#[inline(always)]
fn add_consecutive<T: FbMac, const G: usize>(
    acc: &mut [T::Acc; G],
    h: T::Tap,
    xs: &[T],
) -> Result<()> {
    let xs: &[T; G] = xs.try_into().map_err(|_| Error::Internal("filter group"))?;
    for (a, x) in acc.iter_mut().zip(xs) {
        T::mac_exact(a, h, *x);
    }
    Ok(())
}

// The three forms of a group are separate functions, so that each hot loop is compiled on its
// own (inlined into one function they changed each other's register allocation).

/// The `G` outputs of contiguous taps (`fir`): output `t` has the window `w[t..t + taps.len()]`.
#[inline(never)]
fn group_unit<T: FbMac, const G: usize>(taps: &[T::Tap], w: &[T]) -> Result<[T; G]> {
    let mut acc = [T::zero(); G];
    // Tap `i` multiplies the consecutive samples `len - 1 - i .. len - 1 - i + G`.
    let w = w
        .get(..taps.len() + G - 1)
        .ok_or(Error::Internal("filter group"))?;
    for (h, xs) in taps.iter().zip(w.windows(G).rev()) {
        add_consecutive::<T, G>(&mut acc, *h, xs)?;
    }
    Ok(acc.map(T::finish))
}

/// The `G` outputs of a branch with the taps `branch[i l]` (`len` of them, `l > 1`), output `t`
/// with the window `w[t..t + len]` (`fir_interpolate`).
#[inline(never)]
fn group_rows<T: FbMac, const G: usize>(
    branch: &[T::Tap],
    l: usize,
    len: usize,
    w: &[T],
) -> Result<[T; G]> {
    let internal = || Error::Internal("filter group");
    let mut acc = [T::zero(); G];
    // Tap `i` is the first of row `i` of `l` taps; the last tap may be a partial row.
    let rows = branch.chunks_exact(l);
    let last = rows.remainder().first();
    let w = w.get(..len + G - 1).ok_or_else(internal)?;
    let mut wins = w.windows(G).rev();
    for (row, xs) in rows.zip(wins.by_ref()) {
        add_consecutive::<T, G>(&mut acc, *row.first().ok_or_else(internal)?, xs)?;
    }
    if let (Some(h), Some(xs)) = (last, wins.next()) {
        add_consecutive::<T, G>(&mut acc, *h, xs)?;
    }
    Ok(acc.map(T::finish))
}

/// The `G` outputs of a branch with the taps `branch[i l]` (`len` of them), output `t` with the
/// window `w[t s..t s + len]` (`fir_decimate`, `resample`).
#[inline(never)]
fn group_strided<T: FbMac, const G: usize>(
    branch: &[T::Tap],
    l: usize,
    len: usize,
    w: &[T],
    s: usize,
) -> Result<[T; G]> {
    let internal = || Error::Internal("filter group");
    let mut acc = [T::zero(); G];
    let rows = branch.chunks_exact(l);
    let last = rows.remainder().first();
    let mut win: [&[T]; G] = [&[]; G];
    for (t, v) in win.iter_mut().enumerate() {
        *v = w.get(t * s..t * s + len).ok_or_else(internal)?;
    }
    let mut back = len;
    for row in rows.take(len) {
        back -= 1;
        add_strided::<T, G>(&mut acc, *row.first().ok_or_else(internal)?, &win, back)?;
    }
    if let (Some(h), Some(b)) = (last, back.checked_sub(1)) {
        add_strided::<T, G>(&mut acc, *h, &win, b)?;
    }
    Ok(acc.map(T::finish))
}

/// `G` accumulators plus the products of the tap `h` with the samples `back` of the windows
/// `win` (inlined: as a closure the compiler kept it out of line, one call per tap).
#[allow(
    clippy::inline_always,
    reason = "the group size and the strides must be constants in the hot loops"
)]
#[inline(always)]
fn add_strided<T: FbMac, const G: usize>(
    acc: &mut [T::Acc; G],
    h: T::Tap,
    win: &[&[T]; G],
    back: usize,
) -> Result<()> {
    for (a, v) in acc.iter_mut().zip(win) {
        T::mac_exact(a, h, *v.get(back).ok_or(Error::Internal("filter group"))?);
    }
    Ok(())
}

/// The `G` outputs of one branch (its tap `i` is `branch[i l]`, `len` taps) whose windows
/// begin at the samples `t s` of `w` (`t < G`) and lie inside it.
#[allow(
    clippy::inline_always,
    reason = "the group size and the strides must be constants in the hot loops"
)]
#[inline(always)]
fn group<T: FbMac, const G: usize>(
    branch: &[T::Tap],
    l: usize,
    len: usize,
    w: &[T],
    s: usize,
) -> Result<[T; G]> {
    match (l, s) {
        (1, 1) => group_unit::<T, G>(branch.get(..len).unwrap_or_default(), w),
        (_, 1) => group_rows::<T, G>(branch, l, len, w),
        _ => group_strided::<T, G>(branch, l, len, w, s),
    }
}

/// The `count` outputs of one phase whose windows lie `s` samples apart, the first one after
/// `seen` samples of `input`, written to `dst` in order. The outputs whose windows reach into the
/// history are computed one by one, the rest `G` at a time.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::inline_always,
    reason = "the group size and the strides must be constants in the hot loops"
)]
#[inline(always)]
fn progression_of<'a, T: FbMac + 'a, const G: usize>(
    taps: &[T::Tap],
    l: usize,
    hist: &[T],
    input: &[T],
    phase: usize,
    seen: usize,
    s: usize,
    count: usize,
    mut dst: impl Iterator<Item = &'a mut T>,
) -> Result<()> {
    let internal = || Error::Internal("filter progression");
    let len = branch_len(taps.len(), l, phase);
    let branch = taps.get(phase..).unwrap_or(&[]);
    // A group sums `Q15` products in an `i64` without saturation: exact while the sum of their
    // magnitudes (at most 2^30 each) stays below 2^63, so for fewer than 2^32 taps.
    let grouped = u32::try_from(len).is_ok();
    let mut t = 0;
    while t < count {
        let at = seen + t * s;
        if grouped && at >= len && t + G <= count {
            let w = input.get(at - len..at + (G - 1) * s).ok_or_else(internal)?;
            let ys = group::<T, G>(branch, l, len, w, s)?;
            for y in ys {
                *dst.next().ok_or_else(internal)? = y;
            }
            t += G;
        } else {
            let out = Out { phase, seen: at };
            *dst.next().ok_or_else(internal)? = single(taps, l, hist, input, out)?;
            t += 1;
        }
    }
    Ok(())
}

/// [`progression_of`] with the group size of `T`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn progression<'a, T: FbMac + 'a>(
    taps: &[T::Tap],
    l: usize,
    hist: &[T],
    input: &[T],
    phase: usize,
    seen: usize,
    s: usize,
    count: usize,
    dst: impl Iterator<Item = &'a mut T>,
) -> Result<()> {
    // Windows `s > 1` samples apart take one slice per output: 16 of them spilled.
    if T::WIDE && s == 1 {
        progression_of::<T, WIDE_GROUP>(taps, l, hist, input, phase, seen, s, count, dst)
    } else {
        progression_of::<T, NARROW_GROUP>(taps, l, hist, input, phase, seen, s, count, dst)
    }
}

/// `fir`: one output per input sample; the history holds the last `N - 1` samples.
pub(super) fn fir<T: FbMac>(
    taps: &[T::Tap],
    hist: &mut [T],
    input: &[T],
    output: &mut [T],
) -> Result<()> {
    if input.len() != output.len() {
        return Err(Error::LengthMismatch {
            expected: input.len(),
            got: output.len(),
        });
    }
    let len = input.len();
    progression(taps, 1, hist, input, 0, 1, 1, len, output.iter_mut())?;
    update_history(hist, input)
}

fn decimate<T: FbMac>(
    taps: &[T::Tap],
    hist: &mut [T],
    factor: usize,
    countdown: &mut usize,
    input: &[T],
    output: &mut [T],
) -> Result<Progress> {
    let internal = || Error::Internal("decimation window");
    let len = input.len();
    // The samples that produce an output are `first, first + M, ...`.
    let first = *countdown;
    let avail = if first < len {
        (len - 1 - first) / factor + 1
    } else {
        0
    };
    let produced = avail.min(output.len());
    let (consumed, next) = if produced < avail {
        // The output is full: stop right before the next producing sample.
        (first + produced * factor, 0)
    } else {
        (len, first + produced * factor - len)
    };
    let out = output.get_mut(..produced).ok_or_else(internal)?;
    T::decimate_outputs(taps, hist, input, first + 1, factor, out)?;
    *countdown = next;
    update_history(hist, input.get(..consumed).ok_or_else(internal)?)?;
    Ok(Progress { consumed, produced })
}

/// `fir_decimate` `CQ15` on AArch64 (scalar, D40-style form per target): the grouped engine with
/// [`a64::GROUP`] outputs per group, and the first outputs of a call, whose windows reach into
/// the history, grouped as well in a stack copy of the history and the start of the input.
///
/// The hot loop is the one of [`group_strided`]: per tap one `smaddl` for each component of
/// every output of the group and two `ldrsh` for its sample. With 8 outputs the 16
/// accumulators, the tap and the window pointers do not fit into the 31 integer registers of
/// AArch64 (two accumulators spilled, four more loads and stores per tap); 7 outputs fit. The
/// history outputs (`ceil(N / M)` per call, 13 of 410 at 127 taps and decimation by 10) took
/// the one-output loop of [`single`]. The sums are the same exact `i64` sums, so the outputs are
/// those of every other form bit for bit.
#[cfg(target_arch = "aarch64")]
mod a64 {
    use super::{progression, progression_of};
    use crate::error::{Error, Result};
    use crate::types::{CQ15, Q15};

    /// Outputs per group.
    pub(super) const GROUP: usize = 7;
    /// Longest filter whose first outputs are grouped in the stack copy (it holds `2 (N - 1)`
    /// samples).
    const HEAD_TAPS: usize = 256;

    /// [`super::FbMac::decimate_outputs`] of `CQ15`.
    pub(super) fn decimate(
        taps: &[Q15],
        hist: &[CQ15],
        input: &[CQ15],
        seen: usize,
        s: usize,
        out: &mut [CQ15],
    ) -> Result<()> {
        let len = taps.len();
        // Outputs `t` with `seen + t s < len` reach into the history.
        let n_head = if seen < len && s > 0 {
            (len - seen).div_ceil(s).min(out.len())
        } else {
            0
        };
        let (oh, ob) = out.split_at_mut(n_head);
        if n_head > 0 {
            if len <= HEAD_TAPS && hist.len() + 1 == len {
                head(taps, hist, input, seen, s, oh)?;
            } else {
                progression(taps, 1, hist, input, 0, seen, s, n_head, oh.iter_mut())?;
            }
        }
        grouped(taps, hist, input, seen + n_head * s, s, ob)
    }

    /// The outputs of `out` (the first one after `at` samples) in groups of [`GROUP`], the
    /// remaining fewer than [`GROUP`] in groups of 4 and 2 (a single output is bound by the
    /// latency of its two sums).
    fn grouped(
        taps: &[Q15],
        hist: &[CQ15],
        input: &[CQ15],
        at: usize,
        s: usize,
        out: &mut [CQ15],
    ) -> Result<()> {
        let main = out.len() - out.len() % GROUP;
        let (om, rest) = out.split_at_mut(main);
        progression_of::<CQ15, GROUP>(taps, 1, hist, input, 0, at, s, main, om.iter_mut())?;
        let at = at + main * s;
        let fours = rest.len() & !3;
        let (o4, o2) = rest.split_at_mut(fours);
        progression_of::<CQ15, 4>(taps, 1, hist, input, 0, at, s, fours, o4.iter_mut())?;
        let count = o2.len();
        progression_of::<CQ15, 2>(
            taps,
            1,
            hist,
            input,
            0,
            at + fours * s,
            s,
            count,
            o2.iter_mut(),
        )
    }

    /// The first outputs, whose windows end before sample `N` of the input: in the stack copy
    /// `history ++ input[..N - 1]`, where every window lies inside the copy.
    #[inline(never)]
    fn head(
        taps: &[Q15],
        hist: &[CQ15],
        input: &[CQ15],
        seen: usize,
        s: usize,
        out: &mut [CQ15],
    ) -> Result<()> {
        let internal = || Error::Internal("decimation head");
        let mut buf = [CQ15::default(); 2 * HEAD_TAPS];
        let n = hist.len();
        let take = input.len().min(n);
        let buf = buf.get_mut(..n + take).ok_or_else(internal)?;
        let (bh, bi) = buf.split_at_mut(n);
        bh.copy_from_slice(hist);
        bi.copy_from_slice(input.get(..take).ok_or_else(internal)?);
        grouped(taps, &[], buf, seen + n, s, out)
    }
}

/// Greatest common divisor.
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[allow(clippy::too_many_arguments)]
fn polyphase<T: FbMac>(
    taps: &[T::Tap],
    hist: &mut [T],
    up: usize,
    down: usize,
    q: &mut i64,
    input: &[T],
    output: &mut [T],
) -> Result<Progress> {
    let internal = || Error::Internal("resampler window");
    let (l, m) = (up as i64, down as i64);
    let q0 = *q;
    // The plan: the state machine without the sums.
    let (mut consumed, mut produced) = (0usize, 0usize);
    loop {
        if *q < 0 {
            if produced == output.len() {
                break;
            }
            produced += 1;
            *q += m;
        } else {
            if consumed == input.len() {
                break;
            }
            consumed += 1;
            *q -= l;
        }
    }
    // Output `j` is ready after `c_j` samples (the fewest that make `q0 + j M - L c_j` negative)
    // with the branch `q0 + j M - L c_j + L`. Outputs `j` and `j + L'` (`L' = L / gcd`) share the
    // branch, and their windows lie `M' = M / gcd` samples apart.
    let div = gcd(up, down);
    let (lp, mp) = (up / div, down / div);
    let out = output.get_mut(..produced).ok_or_else(internal)?;
    for first in 0..lp.min(produced) {
        let at = i128::from(q0) + first as i128 * i128::from(m);
        let ready = if at < 0 { 0 } else { at / i128::from(l) + 1 };
        let phase =
            usize::try_from(at - i128::from(l) * ready + i128::from(l)).map_err(|_| internal())?;
        let seen = usize::try_from(ready).map_err(|_| internal())?;
        let count = (produced - first).div_ceil(lp);
        let dst = out
            .get_mut(first..)
            .unwrap_or_default()
            .iter_mut()
            .step_by(lp);
        progression(taps, up, hist, input, phase, seen, mp, count, dst)?;
    }
    update_history(hist, input.get(..consumed).ok_or_else(internal)?)?;
    Ok(Progress { consumed, produced })
}

macro_rules! fir_decimate_impl {
    ($t:ty, $tap:ty) => {
        impl FirDecimate<$t> for Fallback {
            fn fir_decimate<Taps, Hist>(
                &self,
                state: &mut FirDecimState<$t, Taps, Hist>,
                input: &[$t],
                output: &mut [$t],
            ) -> Result<Progress>
            where
                Taps: AsRef<[$tap]>,
                Hist: AsRef<[$t]> + AsMut<[$t]>,
            {
                let p = state.parts_mut();
                decimate::<$t>(p.taps, p.hist, p.factor, p.countdown, input, output)
            }
        }
    };
}

fir_decimate_impl!(C32, f32);
fir_decimate_impl!(CQ15, Q15);

macro_rules! resample_impl {
    ($tr:ident, $method:ident, $state:ident, $t:ty, $tap:ty) => {
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
                let p = state.parts_mut();
                polyphase::<$t>(p.taps, p.hist, p.up, p.down, p.q, input, output)
            }
        }
    };
}

resample_impl!(Resample, resample, ResampleState, C32, f32);
resample_impl!(Resample, resample, ResampleState, CQ15, Q15);
resample_impl!(ResampleReal, resample_real, ResampleRealState, f32, f32);
resample_impl!(ResampleReal, resample_real, ResampleRealState, Q15, Q15);

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::panic,
    clippy::many_single_char_names
)]
mod tests {
    //! The batched engine must give the bits of the definition evaluated one output at a time
    //! (taps ascending, one accumulator per output, no FMA) for every split into calls, every
    //! output buffer size, every phase and every tail.

    use std::vec;
    use std::vec::Vec;

    use super::{C32, CQ15, FbMac, Q15, decimate, fir, polyphase};

    /// A small linear congruential generator.
    fn next(s: &mut u32) -> u32 {
        *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *s >> 8
    }

    /// Sample types of the tests, with a random value and an exact bit pattern to compare.
    trait Sample: FbMac + Default + core::fmt::Debug {
        fn random(s: &mut u32, extreme: bool) -> Self;
        fn tap(s: &mut u32, extreme: bool) -> Self::Tap;
        fn bits(self) -> (u32, u32);
    }

    fn f(s: &mut u32) -> f32 {
        (next(s) as f32 / 8_388_608.0) - 1.0
    }

    fn q(s: &mut u32, extreme: bool) -> Q15 {
        if extreme {
            if next(s) & 1 == 0 {
                Q15(i16::MIN)
            } else {
                Q15(i16::MAX)
            }
        } else {
            Q15(next(s) as i16)
        }
    }

    impl Sample for C32 {
        fn random(s: &mut u32, _: bool) -> Self {
            C32::new(f(s), f(s))
        }
        fn tap(s: &mut u32, _: bool) -> f32 {
            f(s) * 0.3
        }
        fn bits(self) -> (u32, u32) {
            (self.re.to_bits(), self.im.to_bits())
        }
    }

    impl Sample for f32 {
        fn random(s: &mut u32, _: bool) -> Self {
            f(s)
        }
        fn tap(s: &mut u32, _: bool) -> f32 {
            f(s) * 0.3
        }
        fn bits(self) -> (u32, u32) {
            (self.to_bits(), 0)
        }
    }

    impl Sample for CQ15 {
        fn random(s: &mut u32, extreme: bool) -> Self {
            CQ15 {
                re: q(s, extreme),
                im: q(s, extreme),
            }
        }
        fn tap(s: &mut u32, extreme: bool) -> Q15 {
            q(s, extreme)
        }
        fn bits(self) -> (u32, u32) {
            (self.re.0 as u32, self.im.0 as u32)
        }
    }

    impl Sample for Q15 {
        fn random(s: &mut u32, extreme: bool) -> Self {
            q(s, extreme)
        }
        fn tap(s: &mut u32, extreme: bool) -> Q15 {
            q(s, extreme)
        }
        fn bits(self) -> (u32, u32) {
            (self.0 as u32, 0)
        }
    }

    /// `upfirdn(h, x, up, down)` up to the outputs of the received input, one output at a time:
    /// `y[j] = sum_i h[p + i up] x[b - i]` (taps ascending, zeros before the start).
    fn model<T: Sample>(taps: &[T::Tap], up: usize, down: usize, x: &[T]) -> Vec<T> {
        if x.is_empty() {
            return Vec::new();
        }
        let outputs = (x.len() * up - 1) / down + 1;
        (0..outputs)
            .map(|j| {
                let (p, b) = ((j * down) % up, (j * down) / up);
                let mut acc = T::zero();
                for (i, h) in taps.iter().skip(p).step_by(up).enumerate() {
                    let x = if i <= b { x[b - i] } else { T::default() };
                    T::mac(&mut acc, *h, x);
                }
                T::finish(acc)
            })
            .collect()
    }

    /// Which entry point runs.
    #[derive(Clone, Copy)]
    enum Kind {
        Fir,
        Decimate(usize),
        Resample(usize, usize),
    }

    /// Runs `kind` over `x` in calls of random lengths with output buffers of random sizes
    /// (resubmitting what was not consumed) and compares the bits with [`model`].
    fn check<T: Sample>(seed: u32, n_taps: usize, kind: Kind, extreme: bool) {
        let s = &mut (seed.wrapping_mul(2_654_435_761) | 1);
        let taps: Vec<T::Tap> = (0..n_taps).map(|_| T::tap(s, extreme)).collect();
        let x: Vec<T> = (0..700).map(|_| T::random(s, extreme)).collect();
        let (up, down) = match kind {
            Kind::Fir => (1, 1),
            Kind::Decimate(m) => (1, m),
            Kind::Resample(l, m) => (l, m),
        };
        let want = model::<T>(&taps, up, down, &x);
        let hist_len = match kind {
            Kind::Resample(l, _) => n_taps.div_ceil(l),
            _ => n_taps - 1,
        };
        let mut hist = vec![T::default(); hist_len];
        let (mut countdown, mut q) = (0usize, 0i64);
        let mut got: Vec<T> = Vec::new();
        let mut at = 0;
        let mut stalls = 0;
        while at < x.len() {
            // Short calls (windows in the history) and long ones (groups in the input).
            let take =
                (next(s) as usize % if next(s) & 1 == 0 { 40 } else { 300 }).min(x.len() - at);
            let input = &x[at..at + take];
            let room = match kind {
                Kind::Fir => take,
                _ => next(s) as usize % 400,
            };
            let mut out = vec![T::default(); room];
            let (consumed, produced) = match kind {
                Kind::Fir => {
                    fir(&taps, &mut hist, input, &mut out).unwrap();
                    (take, take)
                }
                Kind::Decimate(m) => {
                    let p = decimate(&taps, &mut hist, m, &mut countdown, input, &mut out).unwrap();
                    (p.consumed, p.produced)
                }
                Kind::Resample(l, m) => {
                    let p = polyphase(&taps, &mut hist, l, m, &mut q, input, &mut out).unwrap();
                    (p.consumed, p.produced)
                }
            };
            got.extend_from_slice(&out[..produced]);
            at += consumed;
            stalls = if consumed == 0 { stalls + 1 } else { 0 };
            assert!(stalls < 1000, "no progress");
        }
        // The outputs pending after the last sample.
        if let Kind::Resample(l, m) = kind {
            let mut out = vec![T::default(); 4096];
            let p = polyphase(&taps, &mut hist, l, m, &mut q, &[], &mut out).unwrap();
            got.extend_from_slice(&out[..p.produced]);
        }
        assert_eq!(got.len(), want.len(), "output count, {n_taps} taps");
        for (j, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.bits(), w.bits(), "output {j} of {n_taps} taps");
        }
    }

    fn all<T: Sample>(extreme: bool) {
        for n_taps in [1, 2, 3, 4, 5, 7, 8, 13, 31, 64, 65] {
            let seed = n_taps as u32;
            check::<T>(seed, n_taps, Kind::Fir, extreme);
            for m in [1, 2, 3, 5, 8] {
                check::<T>(seed + 100 * m as u32, n_taps, Kind::Decimate(m), extreme);
            }
            for (l, m) in [
                (1, 1),
                (2, 1),
                (3, 1),
                (5, 1),
                (3, 2),
                (2, 3),
                (4, 3),
                (7, 5),
            ] {
                check::<T>(
                    seed + 7 * l as u32 + m as u32,
                    n_taps,
                    Kind::Resample(l, m),
                    extreme,
                );
            }
        }
    }

    #[test]
    fn c32_batches_equal_the_definition_bit_for_bit() {
        all::<C32>(false);
    }

    #[test]
    fn f32_batches_equal_the_definition_bit_for_bit() {
        all::<f32>(false);
    }

    #[test]
    fn cq15_batches_equal_the_definition_bit_for_bit() {
        all::<CQ15>(false);
        all::<CQ15>(true);
    }

    /// `CQ15` decimation of longer filters: the first outputs of a call grouped in the stack
    /// copy of the AArch64 form (up to 256 taps) and beyond it (257, 300), every factor where
    /// the groups of 7, 4 and 2 and the single outputs occur at the start and the end of a call.
    #[test]
    fn cq15_decimation_of_long_filters_equals_the_definition_bit_for_bit() {
        for n_taps in [127, 255, 256, 257, 300] {
            for m in [1, 2, 6, 10, 13, 40] {
                for extreme in [false, true] {
                    check::<CQ15>(
                        n_taps as u32 + 1000 * m as u32,
                        n_taps,
                        Kind::Decimate(m),
                        extreme,
                    );
                }
            }
        }
    }

    #[test]
    fn q15_batches_equal_the_definition_bit_for_bit() {
        all::<Q15>(false);
        all::<Q15>(true);
    }
}
