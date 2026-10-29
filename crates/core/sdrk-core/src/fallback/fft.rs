//! Fallback implementation of the FFT kernels: iterative in-place radix-2 with a bit-reversal
//! permutation, decimation in time. The order of operations is part of the numeric contract.
//!
//! Some stages run in pairs: one pass over the data applies the butterflies of two consecutive
//! radix-2 stages to every group of four samples while it is in registers (`C32`: all stages
//! after a single first one for odd `log2 N`; `CQ15` with fixed scaling: the first two). The
//! butterflies and their operands are those of the stage-by-stage order, so the result is the
//! same bit for bit; only the loads, stores and twiddle reads of every second stage are saved.
//! Block floating point needs the largest magnitude between the stages and runs them one by one.

// `(a + b) * 0.5` is the specified, exact halving; `f32::midpoint` is not part of the contract.
#![allow(clippy::manual_midpoint, clippy::many_single_char_names)]
#![allow(
    clippy::inline_always,
    reason = "the butterflies are the loop bodies of the stages; they must not become calls"
)]

use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::check_len;
use crate::kernels::fft::{
    Direction, Fft, FftPlan, FftScale, RealFft, RealFftPlan, Scaling, irfft_with, rfft_with,
};
use crate::types::{C32, CQ15, Q15};

fn tw_error() -> Error {
    Error::Internal("twiddle index")
}

/// Largest component magnitude above which a butterfly may need a shift of 1 (see `docs/kernels.md`).
const BFP_SHIFT1_ABOVE: i32 = 13_572;
/// Largest component magnitude above which a butterfly needs a shift of 2.
const BFP_SHIFT2_ABOVE: i32 = 27_145;

/// Swaps `data[i]` and `data[rev(i)]` for `i < rev(i)`; `data.len()` is `2^log2n`.
///
/// On x86 and wasm32, which have no bit-reverse instruction but count trailing zeros in one,
/// `rev(i)` is carried along: going from `i` to `i + 1` flips the `t + 1` lowest bits of `i`
/// (`t` = its trailing ones), which are the `t + 1` highest bits of `rev(i)`. Elsewhere (Arm's
/// `rbit`, and the embedded targets) every `rev(i)` is computed with `reverse_bits`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]
fn bit_reverse<T>(data: &mut [T], log2n: u32) {
    let n = data.len();
    if n != 1 << log2n {
        bit_reverse_by_bits(data, log2n);
        return;
    }
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            data.swap(i, j);
        }
        // `t + 1 <= log2n + 1` as `i < n`, so the shift is in range; after the last `i` the
        // value of `j` is not used.
        let t = (!i).trailing_zeros();
        j ^= n - (n >> (t + 1));
    }
}

/// [`bit_reverse`] with `reverse_bits` for every index.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32")))]
fn bit_reverse<T>(data: &mut [T], log2n: u32) {
    bit_reverse_by_bits(data, log2n);
}

/// Swaps `data[i]` and `data[rev(i)]` for `i < rev(i)`, with `reverse_bits` for every index.
fn bit_reverse_by_bits<T>(data: &mut [T], log2n: u32) {
    let shift = usize::BITS - log2n;
    for i in 0..data.len() {
        let j = i.reverse_bits() >> shift;
        if i < j && j < data.len() {
            data.swap(i, j);
        }
    }
}

/// Checks that the twiddles `tw[k * step]` for `k < count` exist: once per stage instead of
/// once per butterfly.
fn check_twiddles<T>(tw: &[T], count: usize, step: usize) -> Result<()> {
    if count == 0 || step == 0 || (count - 1).saturating_mul(step) >= tw.len() {
        Err(tw_error())
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// C32

/// The radix-2 butterfly of `C32`: `a + w c` into `a` and `a - w c` into `c` (`conj(w)` for the
/// inverse), in the order of the specification and without FMA.
#[inline(always)]
fn butterfly_c32<const INV: bool>(a: &mut C32, c: &mut C32, w: C32) {
    let wi = if INV { -w.im } else { w.im };
    let tr = w.re * c.re - wi * c.im;
    let ti = w.re * c.im + wi * c.re;
    let x = *a;
    *a = C32::new(x.re + tr, x.im + ti);
    *c = C32::new(x.re - tr, x.im - ti);
}

/// The butterflies of two stages on one group of four samples: `(x0, x1)` and `(x2, x3)` with
/// `w`, then `(x0, x2)` with `wa` and `(x1, x3)` with `wb`. The samples are read once into
/// locals and written once.
#[inline(always)]
fn quad_c32<const INV: bool>(x: [&mut C32; 4], w: C32, wa: C32, wb: C32) {
    let [x0, x1, x2, x3] = x;
    let (mut a0, mut a1, mut a2, mut a3) = (*x0, *x1, *x2, *x3);
    butterfly_c32::<INV>(&mut a0, &mut a1, w);
    butterfly_c32::<INV>(&mut a2, &mut a3, w);
    butterfly_c32::<INV>(&mut a0, &mut a2, wa);
    butterfly_c32::<INV>(&mut a1, &mut a3, wb);
    (*x0, *x1, *x2, *x3) = (a0, a1, a2, a3);
}

/// One radix-2 stage of `C32` data: `half` is the half block length, `step` the twiddle step.
fn stage_c32<const INV: bool>(
    data: &mut [C32],
    tw: &[C32],
    half: usize,
    step: usize,
) -> Result<()> {
    check_twiddles(tw, half, step)?;
    if half == 1 {
        let w = tw.first().copied().ok_or_else(tw_error)?;
        for pair in data.chunks_exact_mut(2) {
            if let [a, c] = pair {
                butterfly_c32::<INV>(a, c, w);
            }
        }
        return Ok(());
    }
    for block in data.chunks_exact_mut(2 * half) {
        let (lo, hi) = block.split_at_mut(half);
        for ((a, c), w) in lo
            .iter_mut()
            .zip(hi.iter_mut())
            .zip(tw.iter().step_by(step))
        {
            butterfly_c32::<INV>(a, c, *w);
        }
    }
    Ok(())
}

/// Two radix-2 stages of `C32` data in one pass: the stage of half block length `half` and
/// twiddle step `step`, then the stage of `2 * half` and `step / 2`. The four samples `k`,
/// `k + half`, `k + 2 half` and `k + 3 half` of a block of `4 * half` go through the
/// butterflies of both stages while they are in registers, with the operands of two separate
/// stages: the result is the same bit for bit.
fn stage_pair_c32<const INV: bool>(
    data: &mut [C32],
    tw: &[C32],
    half: usize,
    step: usize,
) -> Result<()> {
    let step2 = step / 2;
    check_twiddles(tw, half, step)?;
    check_twiddles(tw, 2 * half, step2)?;
    if half == 1 {
        // The first two stages: the twiddles `tw[0]` and `tw[step2]` serve every block.
        let w0 = tw.first().copied().ok_or_else(tw_error)?;
        let w1 = tw.get(step2).copied().ok_or_else(tw_error)?;
        for quad in data.chunks_exact_mut(4) {
            if let [x0, x1, x2, x3] = quad {
                quad_c32::<INV>([x0, x1, x2, x3], w0, w0, w1);
            }
        }
        return Ok(());
    }
    let tw_hi = tw.get(half * step2..).ok_or_else(tw_error)?;
    for block in data.chunks_exact_mut(4 * half) {
        let (lo, hi) = block.split_at_mut(2 * half);
        let (q0, q1) = lo.split_at_mut(half);
        let (q2, q3) = hi.split_at_mut(half);
        let groups = q0
            .iter_mut()
            .zip(q1.iter_mut())
            .zip(q2.iter_mut().zip(q3.iter_mut()));
        let twiddles = tw
            .iter()
            .step_by(step)
            .zip(tw.iter().step_by(step2).zip(tw_hi.iter().step_by(step2)));
        for (((x0, x1), (x2, x3)), (w, (wa, wb))) in groups.zip(twiddles) {
            quad_c32::<INV>([x0, x1, x2, x3], *w, *wa, *wb);
        }
    }
    Ok(())
}

/// The stages of the `C32` transform after the bit reversal: one radix-2 stage first if
/// `log2 N` is odd, then pairs of stages.
fn stages_c32<const INV: bool>(
    data: &mut [C32],
    tw: &[C32],
    n: usize,
    log2n: u32,
    stride: usize,
) -> Result<()> {
    let mut s = 1;
    if log2n % 2 == 1 {
        stage_c32::<INV>(data, tw, 1, (n >> 1) * stride)?;
        s = 2;
    }
    while s < log2n {
        stage_pair_c32::<INV>(data, tw, 1 << (s - 1), (n >> s) * stride)?;
        s += 2;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// CQ15

/// Saturates the rounded value `r` to `Q15` (`sat16`); counts a clamp if `COUNT`.
#[inline(always)]
fn saturate_q15<const COUNT: bool>(r: i32, clamped: &mut usize) -> Q15 {
    // `sat16` as a clamp, which compiles without branches.
    let s = r.clamp(-32768, 32767);
    if COUNT {
        *clamped += usize::from(s != r);
    }
    Q15(s as i16)
}

/// Rounds `v` by `SHIFT` bits and saturates it to `Q15`; counts a clamp if `COUNT`.
///
/// The same value as `sat16(round_shift(v, SHIFT))` of `crate::num`: `v` is far from the range
/// of `i32` (a butterfly sum below `2^17` in magnitude), so the saturating addition of
/// `round_shift` is a plain one and the rounding is `(v + 2^(SHIFT - 1)) >> SHIFT`.
#[cfg(any(test, not(target_pointer_width = "64")))]
#[inline(always)]
fn store_q15<const SHIFT: u32, const COUNT: bool>(v: i32, clamped: &mut usize) -> Q15 {
    let r = if SHIFT == 0 {
        v
    } else {
        (v + (1 << (SHIFT - 1))) >> SHIFT
    };
    saturate_q15::<COUNT>(r, clamped)
}

/// The twiddle products of a `CQ15` butterfly, `w c` (`conj(w)` for the inverse) before the
/// rounding: below `2^31 + 1` in magnitude (`|w| = |c| = 2^15`), so they need 64 bits.
#[inline(always)]
fn products_q15<const INV: bool>(c: CQ15, w: CQ15) -> (i64, i64) {
    let (wr, wi) = (i64::from(w.re.0), i64::from(w.im.0));
    let wi = if INV { -wi } else { wi };
    let (cr, ci) = (i64::from(c.re.0), i64::from(c.im.0));
    (wr * cr - wi * ci, wr * ci + wi * cr)
}

/// The radix-2 butterfly of `CQ15` with the output shift `SHIFT`: the twiddle product `w c`
/// (`conj(w)` for the inverse) rounded to Q15 without saturation, then `a +- w c` shifted with
/// rounding and saturated once.
///
/// On 64-bit targets this is [`butterfly_q15_wide`], on the others (the embedded targets)
/// [`butterfly_q15_narrow`]; both give the same values.
#[inline(always)]
fn butterfly_q15<const INV: bool, const SHIFT: u32, const COUNT: bool>(
    a: &mut CQ15,
    c: &mut CQ15,
    w: CQ15,
    clamped: &mut usize,
) {
    #[cfg(target_pointer_width = "64")]
    butterfly_q15_wide::<INV, SHIFT, COUNT>(a, c, w, clamped);
    #[cfg(not(target_pointer_width = "64"))]
    butterfly_q15_narrow::<INV, SHIFT, COUNT>(a, c, w, clamped);
}

/// [`butterfly_q15`] with 32-bit sums: the rounded product `t = (p + 2^14) >> 15` is at most
/// `2^16` in magnitude, so `a +- t` and its rounding shift are `i32`.
#[cfg(any(test, not(target_pointer_width = "64")))]
#[inline(always)]
fn butterfly_q15_narrow<const INV: bool, const SHIFT: u32, const COUNT: bool>(
    a: &mut CQ15,
    c: &mut CQ15,
    w: CQ15,
    clamped: &mut usize,
) {
    let (pr, pi) = products_q15::<INV>(*c, w);
    let tr = ((pr + (1 << 14)) >> 15) as i32;
    let ti = ((pi + (1 << 14)) >> 15) as i32;
    let (ar, ai) = (i32::from(a.re.0), i32::from(a.im.0));
    *a = CQ15 {
        re: store_q15::<SHIFT, COUNT>(ar + tr, clamped),
        im: store_q15::<SHIFT, COUNT>(ai + ti, clamped),
    };
    *c = CQ15 {
        re: store_q15::<SHIFT, COUNT>(ar - tr, clamped),
        im: store_q15::<SHIFT, COUNT>(ai - ti, clamped),
    };
}

/// [`butterfly_q15`] with one 64-bit rounding shift per output in place of two.
///
/// With `h = 2^(SHIFT - 1)` (0 for `SHIFT = 0`), `S = 15 + SHIFT` and `b = (a + h) 2^15 + 2^14`:
/// `a + t + h = floor((b + p) / 2^15)` because `(a + h) 2^15` is a multiple of `2^15`, and
/// `a - t + h = floor((b - 1 - p) / 2^15)` because `-floor(x / 2^15) = floor((2^15 - 1 - x) /
/// 2^15)`. A further `floor(/ 2^SHIFT)` of a floor is the floor of the quotient, so the outputs
/// before the saturation are `(b + p) >> S` and `(b - 1 - p) >> S`: the values of
/// [`butterfly_q15_narrow`].
#[cfg(any(test, target_pointer_width = "64"))]
#[inline(always)]
fn butterfly_q15_wide<const INV: bool, const SHIFT: u32, const COUNT: bool>(
    a: &mut CQ15,
    c: &mut CQ15,
    w: CQ15,
    clamped: &mut usize,
) {
    let (pr, pi) = products_q15::<INV>(*c, w);
    let h: i64 = if SHIFT == 0 { 0 } else { 1 << (SHIFT - 1) };
    let base = |x: Q15| ((i64::from(x.0) + h) << 15) + (1 << 14);
    let (br, bi) = (base(a.re), base(a.im));
    // Below `2^17` in magnitude after the shift, so the casts are exact.
    let out = |v: i64| (v >> (15 + SHIFT)) as i32;
    *a = CQ15 {
        re: saturate_q15::<COUNT>(out(br + pr), clamped),
        im: saturate_q15::<COUNT>(out(bi + pi), clamped),
    };
    *c = CQ15 {
        re: saturate_q15::<COUNT>(out(br - 1 - pr), clamped),
        im: saturate_q15::<COUNT>(out(bi - 1 - pi), clamped),
    };
}

/// The butterflies of two stages of fixed scaling on one group of four samples, as
/// [`quad_c32`].
#[inline(always)]
fn quad_q15<const INV: bool, const COUNT: bool>(
    x: [&mut CQ15; 4],
    w: CQ15,
    wa: CQ15,
    wb: CQ15,
    clamped: &mut usize,
) {
    let [x0, x1, x2, x3] = x;
    let (mut a0, mut a1, mut a2, mut a3) = (*x0, *x1, *x2, *x3);
    butterfly_q15::<INV, 1, COUNT>(&mut a0, &mut a1, w, clamped);
    butterfly_q15::<INV, 1, COUNT>(&mut a2, &mut a3, w, clamped);
    butterfly_q15::<INV, 1, COUNT>(&mut a0, &mut a2, wa, clamped);
    butterfly_q15::<INV, 1, COUNT>(&mut a1, &mut a3, wb, clamped);
    (*x0, *x1, *x2, *x3) = (a0, a1, a2, a3);
}

/// One radix-2 stage of `CQ15` data with the output shift `shift` (1 for fixed scaling, 0 to 2
/// for block floating point), with the butterfly of [`butterfly_q15`]. Returns the number of
/// clamped components (if `COUNT`).
fn stage_q15<const INV: bool, const COUNT: bool>(
    data: &mut [CQ15],
    tw: &[CQ15],
    half: usize,
    step: usize,
    shift: u32,
) -> Result<usize> {
    // One copy per shift, so that the rounding shifts are constants.
    match shift {
        0 => stage_q15_by::<INV, 0, COUNT>(data, tw, half, step),
        1 => stage_q15_by::<INV, 1, COUNT>(data, tw, half, step),
        2 => stage_q15_by::<INV, 2, COUNT>(data, tw, half, step),
        _ => Err(Error::Internal("block-floating-point shift")),
    }
}

/// [`stage_q15`] with the output shift `SHIFT`.
fn stage_q15_by<const INV: bool, const SHIFT: u32, const COUNT: bool>(
    data: &mut [CQ15],
    tw: &[CQ15],
    half: usize,
    step: usize,
) -> Result<usize> {
    check_twiddles(tw, half, step)?;
    let mut clamped = 0usize;
    if half == 1 {
        let w = tw.first().copied().ok_or_else(tw_error)?;
        for pair in data.chunks_exact_mut(2) {
            if let [a, c] = pair {
                butterfly_q15::<INV, SHIFT, COUNT>(a, c, w, &mut clamped);
            }
        }
        return Ok(clamped);
    }
    for block in data.chunks_exact_mut(2 * half) {
        let (lo, hi) = block.split_at_mut(half);
        for ((a, c), w) in lo
            .iter_mut()
            .zip(hi.iter_mut())
            .zip(tw.iter().step_by(step))
        {
            butterfly_q15::<INV, SHIFT, COUNT>(a, c, *w, &mut clamped);
        }
    }
    Ok(clamped)
}

/// The first two radix-2 stages of `CQ15` data with fixed scaling (shift 1 each) in one pass
/// over groups of four samples, as [`stage_pair_c32`] with `half = 1`: their twiddles `tw[0]`
/// and `tw[step2]` serve every group, and the butterflies have the operands of two calls of
/// [`stage_q15`], so the result is the same bit for bit. `step2` is the twiddle step of the
/// second stage. Returns the number of clamped components (if `COUNT`).
///
/// The later stages run one by one: a group of four `CQ15` butterflies with three twiddles
/// does not fit the general registers of x86-64 (the paired loop spilled and was slower than
/// two passes on `znver3` and at `sse2`).
fn first_stages_q15<const INV: bool, const COUNT: bool>(
    data: &mut [CQ15],
    tw: &[CQ15],
    step2: usize,
) -> Result<usize> {
    check_twiddles(tw, 2, step2)?;
    let w0 = tw.first().copied().ok_or_else(tw_error)?;
    let w1 = tw.get(step2).copied().ok_or_else(tw_error)?;
    let mut clamped = 0usize;
    for quad in data.chunks_exact_mut(4) {
        if let [x0, x1, x2, x3] = quad {
            quad_q15::<INV, COUNT>([x0, x1, x2, x3], w0, w0, w1, &mut clamped);
        }
    }
    Ok(clamped)
}

/// The block-floating-point shift of the next stage from the largest component magnitude.
fn bfp_shift_of(m: i32) -> u32 {
    if m > BFP_SHIFT2_ABOVE {
        2
    } else {
        u32::from(m > BFP_SHIFT1_ABOVE)
    }
}

/// The block-floating-point shift of the next stage from the largest component magnitude of
/// `data`. The magnitude is `max(largest component, -smallest component)`: two running
/// extremes of the plain `i16` values, which a compiler can vectorize.
fn bfp_shift(data: &[CQ15]) -> u32 {
    let (hi, lo) = data.iter().fold((0i16, 0i16), |(hi, lo), x| {
        (hi.max(x.re.0).max(x.im.0), lo.min(x.re.0).min(x.im.0))
    });
    bfp_shift_of(i32::from(hi).max(-i32::from(lo)))
}

/// The stages of the `CQ15` transform after the bit reversal; returns the exponent and the
/// number of clamped components (if `COUNT`).
fn stages_q15<const INV: bool, const COUNT: bool>(
    data: &mut [CQ15],
    tw: &[CQ15],
    n: usize,
    log2n: u32,
    stride: usize,
    scaling: Scaling,
) -> Result<(i32, usize)> {
    let mut clamped = 0usize;
    if scaling == Scaling::BlockFloat {
        // The shift of a stage comes from the largest magnitude of the data before it.
        let mut exponent = 0i32;
        for s in 1..=log2n {
            let shift = bfp_shift(data);
            let half = 1usize << (s - 1);
            let step = (n >> s) * stride;
            clamped += stage_q15::<INV, COUNT>(data, tw, half, step, shift)?;
            exponent += shift as i32;
        }
        return Ok((exponent, clamped));
    }
    // `log2 N >= 4`: the first two stages in one pass, then one stage per pass.
    clamped += first_stages_q15::<INV, COUNT>(data, tw, (n >> 2) * stride)?;
    for s in 3..=log2n {
        clamped += stage_q15_by::<INV, 1, COUNT>(data, tw, 1 << (s - 1), (n >> s) * stride)?;
    }
    Ok((log2n as i32, clamped))
}

/// The `CQ15` transform; with `COUNT` it also returns how many stored components were clamped
/// (always 0 for block floating point with inputs of complex magnitude up to 1).
pub(crate) fn fft_cq15_counted<const COUNT: bool, Tw: AsRef<[CQ15]>>(
    plan: &FftPlan<CQ15, Tw>,
    data: &mut [CQ15],
    dir: Direction,
    scaling: Scaling,
) -> Result<(FftScale, usize)> {
    check_len(plan.n, data.len())?;
    let tw = plan.twiddles.as_ref();
    bit_reverse(data, plan.log2n);
    let (n, log2n, stride) = (plan.n, plan.log2n, plan.tw_stride);
    let (exponent, clamped) = if dir == Direction::Inverse {
        stages_q15::<true, COUNT>(data, tw, n, log2n, stride, scaling)?
    } else {
        stages_q15::<false, COUNT>(data, tw, n, log2n, stride, scaling)?
    };
    Ok((FftScale { exponent }, clamped))
}

impl Fft<C32> for Fallback {
    fn fft<Tw: AsRef<[C32]>>(
        &self,
        plan: &FftPlan<C32, Tw>,
        data: &mut [C32],
        dir: Direction,
        scaling: Scaling,
    ) -> Result<FftScale> {
        if scaling == Scaling::BlockFloat {
            return Err(Error::Unsupported(
                "block floating point is not available for c32",
            ));
        }
        check_len(plan.n, data.len())?;
        let tw = plan.twiddles.as_ref();
        let inverse = dir == Direction::Inverse;
        bit_reverse(data, plan.log2n);
        let (n, log2n, stride) = (plan.n, plan.log2n, plan.tw_stride);
        if inverse {
            stages_c32::<true>(data, tw, n, log2n, stride)?;
            let g = 1.0 / plan.n as f32;
            for x in data.iter_mut() {
                *x = C32::new(x.re * g, x.im * g);
            }
            Ok(FftScale {
                exponent: plan.log2n as i32,
            })
        } else {
            stages_c32::<false>(data, tw, n, log2n, stride)?;
            Ok(FftScale { exponent: 0 })
        }
    }
}

impl Fft<CQ15> for Fallback {
    fn fft<Tw: AsRef<[CQ15]>>(
        &self,
        plan: &FftPlan<CQ15, Tw>,
        data: &mut [CQ15],
        dir: Direction,
        scaling: Scaling,
    ) -> Result<FftScale> {
        fft_cq15_counted::<false, Tw>(plan, data, dir, scaling).map(|(scale, _)| scale)
    }
}

impl RealFft for Fallback {
    fn rfft<Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>>(
        &self,
        plan: &RealFftPlan<Tw, Tw2>,
        input: &[f32],
        output: &mut [C32],
        scratch: &mut [C32],
    ) -> Result<FftScale> {
        rfft_with(self, plan, input, output, scratch)
    }

    fn irfft<Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>>(
        &self,
        plan: &RealFftPlan<Tw, Tw2>,
        input: &[C32],
        output: &mut [f32],
        scratch: &mut [C32],
    ) -> Result<FftScale> {
        irfft_with(self, plan, input, output, scratch)
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_lossless,
    clippy::many_single_char_names
)]
mod tests {
    use super::*;
    use crate::types::ComplexSample;
    use core::marker::PhantomData;
    use std::format;

    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state >> 33
    }

    fn plan(n: usize) -> FftPlan<CQ15, std::vec::Vec<CQ15>> {
        FftPlan::new(n, std::vec![CQ15::default(); n / 2]).unwrap()
    }

    /// A plan of `n` points over the table of `big` points (every `big / n`-th twiddle), as
    /// `from_static_4096` builds one.
    fn strided<T: ComplexSample>(n: usize, big: usize) -> FftPlan<T, std::vec::Vec<T>> {
        let full = FftPlan::<T, _>::new(big, std::vec![T::default(); big / 2]).unwrap();
        FftPlan {
            n,
            log2n: n.trailing_zeros(),
            twiddles: full.twiddles,
            tw_stride: big / n,
            _sample: PhantomData,
        }
    }

    /// Runs the block-floating-point transform and returns the number of clamped components.
    fn clamps(input: &[CQ15], dir: Direction) -> usize {
        let p = plan(input.len());
        let mut data = input.to_vec();
        fft_cq15_counted::<true, _>(&p, &mut data, dir, Scaling::BlockFloat)
            .unwrap()
            .1
    }

    fn sizes() -> impl Iterator<Item = usize> {
        (4..=12).map(|b| 1usize << b)
    }

    /// A Q15 value over the whole range with the extremes mixed in (`-32768` makes the largest
    /// products and sums, and with shift 0 most clamps).
    fn q15_value(s: &mut u64) -> Q15 {
        let r = lcg(s);
        Q15(match r % 8 {
            0 => -32768,
            1 => 32767,
            _ => (r >> 3) as u16 as i16,
        })
    }

    fn cq15_values(s: &mut u64, n: usize) -> std::vec::Vec<CQ15> {
        (0..n)
            .map(|_| CQ15 {
                re: q15_value(s),
                im: q15_value(s),
            })
            .collect()
    }

    /// Noise of up to `amplitude` per component.
    fn cq15_noise(s: &mut u64, n: usize, amplitude: i32) -> std::vec::Vec<CQ15> {
        let mut v = || Q15(((lcg(s) % (2 * amplitude as u64 + 1)) as i32 - amplitude) as i16);
        (0..n).map(|_| CQ15 { re: v(), im: v() }).collect()
    }

    fn c32_noise(s: &mut u64, n: usize) -> std::vec::Vec<C32> {
        let mut v = || (lcg(s) % 2_000_001) as f32 / 1_000_000.0 - 1.0;
        (0..n).map(|_| C32::new(v(), v())).collect()
    }

    /// `stage_q15` with a run-time direction, counting clamps.
    fn stage_q15_counted(
        data: &mut [CQ15],
        tw: &[CQ15],
        half: usize,
        step: usize,
        inverse: bool,
        shift: u32,
    ) -> Result<usize> {
        if inverse {
            stage_q15::<true, true>(data, tw, half, step, shift)
        } else {
            stage_q15::<false, true>(data, tw, half, step, shift)
        }
    }

    /// The transform as it was specified: bit reversal, then `log2 N` radix-2 stages one by
    /// one, block floating point with a separate pass for the largest magnitude before every
    /// stage. The paired passes must give these bits.
    fn fft_cq15_stage_by_stage<Tw: AsRef<[CQ15]>>(
        plan: &FftPlan<CQ15, Tw>,
        data: &mut [CQ15],
        dir: Direction,
        scaling: Scaling,
    ) -> (i32, usize) {
        let tw = plan.twiddles.as_ref();
        bit_reverse(data, plan.log2n);
        let (mut exponent, mut clamped) = (0, 0);
        for s in 1..=plan.log2n {
            let shift = if scaling == Scaling::BlockFloat {
                let m = data
                    .iter()
                    .map(|x| i32::from(x.re.0).abs().max(i32::from(x.im.0).abs()))
                    .max()
                    .unwrap_or(0);
                if m > BFP_SHIFT2_ABOVE {
                    2
                } else {
                    u32::from(m > BFP_SHIFT1_ABOVE)
                }
            } else {
                1
            };
            let half = 1usize << (s - 1);
            let step = (plan.n >> s) * plan.tw_stride;
            let inverse = dir == Direction::Inverse;
            clamped += stage_q15_formula(data, tw, half, step, inverse, shift);
            exponent += shift as i32;
        }
        (exponent, clamped)
    }

    /// The `C32` transform stage by stage, as it was specified (see [`fft_cq15_stage_by_stage`]).
    fn fft_c32_stage_by_stage<Tw: AsRef<[C32]>>(
        plan: &FftPlan<C32, Tw>,
        data: &mut [C32],
        dir: Direction,
    ) {
        let tw = plan.twiddles.as_ref();
        let inverse = dir == Direction::Inverse;
        bit_reverse(data, plan.log2n);
        for s in 1..=plan.log2n {
            let half = 1usize << (s - 1);
            let step = (plan.n >> s) * plan.tw_stride;
            for block in data.chunks_exact_mut(2 * half) {
                let (lo, hi) = block.split_at_mut(half);
                for (k, (a, c)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                    let w = tw[k * step];
                    let wi = if inverse { -w.im } else { w.im };
                    let tr = w.re * c.re - wi * c.im;
                    let ti = w.re * c.im + wi * c.re;
                    let x = *a;
                    *a = C32::new(x.re + tr, x.im + ti);
                    *c = C32::new(x.re - tr, x.im - ti);
                }
            }
        }
        if inverse {
            let g = 1.0 / plan.n as f32;
            for x in data.iter_mut() {
                *x = C32::new(x.re * g, x.im * g);
            }
        }
    }

    fn bits(x: &[C32]) -> std::vec::Vec<(u32, u32)> {
        x.iter().map(|v| (v.re.to_bits(), v.im.to_bits())).collect()
    }

    #[test]
    fn c32_paired_stages_equal_the_stage_by_stage_transform_bit_for_bit() {
        // Every size from 16 to 4096 (odd and even `log2 N`), both directions, plans of their
        // own and over a larger table (stride > 1).
        let mut s = 11u64;
        for n in sizes() {
            for big in [n, 4096.max(n)] {
                let p = strided::<C32>(n, big);
                for dir in [Direction::Forward, Direction::Inverse] {
                    let x = c32_noise(&mut s, n);
                    let mut want = x.clone();
                    fft_c32_stage_by_stage(&p, &mut want, dir);
                    let mut got = x.clone();
                    Fallback.fft(&p, &mut got, dir, Scaling::Default).unwrap();
                    assert_eq!(bits(&got), bits(&want), "n={n} big={big} {dir:?}");
                }
            }
        }
        // The largest size, and non-finite values, which propagate the same way.
        let n = 65_536;
        let p = strided::<C32>(n, n);
        let mut x = c32_noise(&mut s, n);
        x[3] = C32::new(f32::INFINITY, 0.0);
        x[700] = C32::new(f32::NAN, -0.0);
        for (i, v) in x.iter_mut().enumerate().skip(1000).take(64) {
            *v = C32::new(
                f32::MAX * if i % 2 == 0 { 1.0 } else { -1.0 },
                f32::MIN_POSITIVE,
            );
        }
        let mut want = x.clone();
        fft_c32_stage_by_stage(&p, &mut want, Direction::Forward);
        let mut got = x;
        Fallback
            .fft(&p, &mut got, Direction::Forward, Scaling::Default)
            .unwrap();
        assert_eq!(bits(&got), bits(&want));
    }

    #[test]
    fn cq15_paired_stages_equal_the_stage_by_stage_transform_bit_for_bit() {
        // Every size from 16 to 4096, both directions, both scalings, plans of their own and
        // over a larger table, with inputs over the whole range (with clamps), full-scale noise
        // and weak noise (block floating point shifts by 0 there).
        let mut s = 23u64;
        for n in sizes() {
            for big in [n, 4096.max(n)] {
                let p = strided::<CQ15>(n, big);
                for dir in [Direction::Forward, Direction::Inverse] {
                    for scaling in [Scaling::Default, Scaling::BlockFloat] {
                        for x in [
                            cq15_values(&mut s, n),
                            cq15_noise(&mut s, n, 23_000),
                            cq15_noise(&mut s, n, 40),
                        ] {
                            let mut want = x.clone();
                            let (e, c) = fft_cq15_stage_by_stage(&p, &mut want, dir, scaling);
                            let mut got = x.clone();
                            let (scale, clamped) =
                                fft_cq15_counted::<true, _>(&p, &mut got, dir, scaling).unwrap();
                            let what = format!("n={n} big={big} {dir:?} {scaling:?}");
                            assert_eq!(got, want, "{what}");
                            assert_eq!((scale.exponent, clamped), (e, c), "{what}");
                            let mut uncounted = x.clone();
                            let scale2 = Fallback.fft(&p, &mut uncounted, dir, scaling).unwrap();
                            assert_eq!((uncounted, scale2), (got, scale), "{what}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn bfp_never_saturates_for_alternating_full_scale() {
        for n in sizes() {
            for dir in [Direction::Forward, Direction::Inverse] {
                let alt: std::vec::Vec<CQ15> = (0..n)
                    .map(|i| {
                        let v = if i % 2 == 0 { 32767 } else { -32767 };
                        CQ15 {
                            re: Q15(v),
                            im: Q15(v),
                        }
                    })
                    .collect();
                assert_eq!(clamps(&alt, dir), 0, "n={n} {dir:?}");
            }
        }
    }

    #[test]
    fn bfp_never_saturates_for_random_inputs_up_to_unit_magnitude() {
        for seed in 0..20u64 {
            for n in sizes() {
                let mut s = seed * 1000 + n as u64;
                let x: std::vec::Vec<CQ15> = (0..n)
                    .map(|_| {
                        // Magnitude uniform in [0, 1 - 2^-15], random phase.
                        let mag = (lcg(&mut s) % 32768) as f64 / 32768.0 * (1.0 - 2f64.powi(-15));
                        let ph = (lcg(&mut s) % 1_000_000) as f64 / 1_000_000.0
                            * 2.0
                            * core::f64::consts::PI;
                        CQ15::from_c64_round(num_complex::Complex64::new(
                            mag * libm::cos(ph),
                            mag * libm::sin(ph),
                        ))
                    })
                    .collect();
                for dir in [Direction::Forward, Direction::Inverse] {
                    assert_eq!(clamps(&x, dir), 0, "seed={seed} n={n} {dir:?}");
                }
            }
        }
    }

    #[test]
    fn bfp_never_saturates_for_unit_circle_inputs() {
        // Worst-case magnitude 1 everywhere with a structured phase.
        for n in sizes() {
            let x: std::vec::Vec<CQ15> = (0..n)
                .map(|i| {
                    let ph = 2.0 * core::f64::consts::PI * (i * i) as f64 / n as f64;
                    CQ15::from_c64_round(num_complex::Complex64::new(libm::cos(ph), libm::sin(ph)))
                })
                .collect();
            assert_eq!(clamps(&x, Direction::Forward), 0, "chirp n={n}");
        }
    }

    #[test]
    fn counter_detects_saturation_of_fixed_corner_input() {
        // The counter itself works: a sum that exceeds the range with shift 0 is counted.
        let tw = [CQ15 {
            re: Q15(32767),
            im: Q15(0),
        }];
        let c = CQ15 {
            re: Q15(30000),
            im: Q15(0),
        };
        let mut data = [c, c];
        let clamped = stage_q15_counted(&mut data, &tw, 1, 1, false, 0).unwrap();
        assert_eq!(clamped, 1);
        assert_eq!(data[0].re, Q15(32767));
        // Without counting, the same values and no count.
        let mut data2 = [c, c];
        let uncounted = stage_q15::<false, false>(&mut data2, &tw, 1, 1, 0).unwrap();
        assert_eq!((data2, uncounted), (data, 0));
    }

    /// Both forms of the butterfly for one direction, shift and counting.
    fn both_butterflies<const INV: bool, const SHIFT: u32, const COUNT: bool>(
        a: CQ15,
        c: CQ15,
        w: CQ15,
    ) -> [(CQ15, CQ15, usize); 2] {
        let (mut a1, mut c1, mut n1) = (a, c, 0);
        butterfly_q15_narrow::<INV, SHIFT, COUNT>(&mut a1, &mut c1, w, &mut n1);
        let (mut a2, mut c2, mut n2) = (a, c, 0);
        butterfly_q15_wide::<INV, SHIFT, COUNT>(&mut a2, &mut c2, w, &mut n2);
        [(a1, c1, n1), (a2, c2, n2)]
    }

    #[test]
    fn the_wide_and_the_narrow_q15_butterfly_agree() {
        // The corners of every operand (with `-32768` the products reach `2^31`), values next
        // to the rounding boundaries, and random values over the whole range.
        let corners = [
            -32768i16, -32767, -16385, -16384, -1, 0, 1, 16383, 16384, 32767,
        ];
        let mut values: std::vec::Vec<CQ15> = corners
            .iter()
            .flat_map(|&re| {
                corners.iter().map(move |&im| CQ15 {
                    re: Q15(re),
                    im: Q15(im),
                })
            })
            .collect();
        let mut s = 41u64;
        values.extend(cq15_values(&mut s, 400));
        let mut checked = 0usize;
        for (i, &a) in values.iter().enumerate() {
            for (j, &c) in values.iter().enumerate().skip(i % 7).step_by(7) {
                let w = values[(i * 31 + j * 17) % values.len()];
                for [x, y] in [
                    both_butterflies::<false, 0, true>(a, c, w),
                    both_butterflies::<false, 1, true>(a, c, w),
                    both_butterflies::<false, 2, true>(a, c, w),
                    both_butterflies::<true, 0, true>(a, c, w),
                    both_butterflies::<true, 1, true>(a, c, w),
                    both_butterflies::<true, 2, true>(a, c, w),
                    both_butterflies::<true, 1, false>(a, c, w),
                ] {
                    assert_eq!(x, y, "a={a:?} c={c:?} w={w:?}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 100_000);
    }

    #[test]
    fn the_bfp_shift_is_that_of_the_largest_magnitude() {
        // The extremes of either sign and either component decide, also `-32768` and the values
        // next to the thresholds.
        let at = |re: i16, im: i16| {
            let mut x = std::vec![CQ15 { re: Q15(5), im: Q15(-5) }; 16];
            x[9] = CQ15 {
                re: Q15(re),
                im: Q15(im),
            };
            bfp_shift(&x)
        };
        for v in [
            0, 1, 13_571, 13_572, 13_573, 27_144, 27_145, 27_146, 32_767, 32_768,
        ] {
            let want = bfp_shift_of(v);
            let pos = i16::try_from(v).ok();
            let neg = i16::try_from(-v).unwrap();
            assert_eq!(at(neg, 0), want, "{v}");
            assert_eq!(at(0, neg), want, "{v}");
            if let Some(p) = pos {
                assert_eq!(at(p, 0), want, "{v}");
                assert_eq!(at(0, p), want, "{v}");
            }
        }
        assert_eq!(bfp_shift(&[]), 0);
    }

    #[test]
    fn the_carried_bit_reversal_is_the_bit_reversal() {
        for log2n in 1..=16 {
            let n = 1usize << log2n;
            let mut got: std::vec::Vec<usize> = (0..n).collect();
            let mut want = got.clone();
            bit_reverse(&mut got, log2n);
            bit_reverse_by_bits(&mut want, log2n);
            assert_eq!(got, want, "log2n={log2n}");
            assert!(
                got.iter()
                    .enumerate()
                    .all(|(i, &j)| i == j.reverse_bits() >> (usize::BITS - log2n))
            );
        }
    }

    /// `stage_q15` as the formula of `docs/kernels.md` with `round_shift` and `sat16`, written
    /// out per butterfly: the twiddle product rounded to Q15, the sum shifted with rounding and
    /// saturated, and the clamps counted.
    fn stage_q15_formula(
        data: &mut [CQ15],
        tw: &[CQ15],
        half: usize,
        step: usize,
        inverse: bool,
        shift: u32,
    ) -> usize {
        use crate::num::{round_shift, sat16};
        let mut clamped = 0;
        let mut store = |v: i64| {
            let r = round_shift(v, shift);
            let s = sat16(r);
            clamped += usize::from(i64::from(s) != r);
            Q15(s)
        };
        for block in data.chunks_exact_mut(2 * half) {
            let (lo, hi) = block.split_at_mut(half);
            for (k, (a, c)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                let w = tw[k * step];
                let (wr, wi) = (i64::from(w.re.0), i64::from(w.im.0));
                let wi = if inverse { -wi } else { wi };
                let (cr, ci) = (i64::from(c.re.0), i64::from(c.im.0));
                let tr = round_shift(wr * cr - wi * ci, 15);
                let ti = round_shift(wr * ci + wi * cr, 15);
                let (ar, ai) = (i64::from(a.re.0), i64::from(a.im.0));
                *a = CQ15 {
                    re: store(ar + tr),
                    im: store(ai + ti),
                };
                *c = CQ15 {
                    re: store(ar - tr),
                    im: store(ai - ti),
                };
            }
        }
        clamped
    }

    #[test]
    fn the_q15_stage_is_the_formula_bit_for_bit() {
        // Random components and twiddles over the whole range, with the extremes mixed in.
        let mut s = 7u64;
        for log2 in 1..=6 {
            let n = 1usize << log2;
            let x = cq15_values(&mut s, n);
            let tw = cq15_values(&mut s, n / 2);
            for stage in 1..=log2 {
                let half = 1usize << (stage - 1);
                let step = n >> stage;
                for inverse in [false, true] {
                    for shift in 0..=2 {
                        let mut want = x.clone();
                        let mut got = x.clone();
                        let clamps = stage_q15_formula(&mut want, &tw, half, step, inverse, shift);
                        let counted =
                            stage_q15_counted(&mut got, &tw, half, step, inverse, shift).unwrap();
                        let what = format!("n={n} half={half} {inverse} shift {shift}");
                        assert_eq!(got, want, "{what}");
                        assert_eq!(counted, clamps, "{what}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_first_q15_stages_are_two_stages_of_the_formula_bit_for_bit() {
        // Arbitrary twiddles over the whole range (also `-32768`, whose conjugate is `+32768`
        // in the products of the inverse), both directions.
        let mut s = 31u64;
        for log2 in 2..=7 {
            let n = 1usize << log2;
            for _ in 0..8 {
                let x = cq15_values(&mut s, n);
                let tw = cq15_values(&mut s, n / 2);
                for inverse in [false, true] {
                    let mut want = x.clone();
                    let mut clamps = stage_q15_formula(&mut want, &tw, 1, n / 2, inverse, 1);
                    clamps += stage_q15_formula(&mut want, &tw, 2, n / 4, inverse, 1);
                    let mut got = x.clone();
                    let counted = if inverse {
                        first_stages_q15::<true, true>(&mut got, &tw, n / 4)
                    } else {
                        first_stages_q15::<false, true>(&mut got, &tw, n / 4)
                    }
                    .unwrap();
                    assert_eq!(got, want, "n={n} {inverse}");
                    assert_eq!(counted, clamps, "n={n} {inverse}");
                }
            }
        }
    }

    #[test]
    fn the_c32_stage_pair_is_two_stages_bit_for_bit() {
        // Arbitrary twiddles (not those of a plan), every pair of stages, both directions.
        let mut s = 37u64;
        for log2 in 2..=8 {
            let n = 1usize << log2;
            let x = c32_noise(&mut s, n);
            let tw = c32_noise(&mut s, n / 2);
            for stage in 1..log2 {
                let half = 1usize << (stage - 1);
                let step = n >> stage;
                for inverse in [false, true] {
                    let mut want = x.clone();
                    let mut got = x.clone();
                    if inverse {
                        stage_c32::<true>(&mut want, &tw, half, step).unwrap();
                        stage_c32::<true>(&mut want, &tw, 2 * half, step / 2).unwrap();
                        stage_pair_c32::<true>(&mut got, &tw, half, step).unwrap();
                    } else {
                        stage_c32::<false>(&mut want, &tw, half, step).unwrap();
                        stage_c32::<false>(&mut want, &tw, 2 * half, step / 2).unwrap();
                        stage_pair_c32::<false>(&mut got, &tw, half, step).unwrap();
                    }
                    assert_eq!(bits(&got), bits(&want), "n={n} half={half} {inverse}");
                }
            }
        }
    }

    #[test]
    fn the_stages_reject_a_short_twiddle_table() {
        let mut q = [CQ15::default(); 8];
        let tw = [CQ15::default(); 3];
        // The butterflies of half 4 need the twiddles 0 to 3 (step 1), those of half 2 and step
        // 3 the twiddles 0 and 3.
        assert!(stage_q15_counted(&mut q, &tw, 4, 1, false, 1).is_err());
        assert!(stage_q15_counted(&mut q, &tw, 2, 1, false, 3).is_err());
        assert!(stage_q15_counted(&mut q, &tw, 2, 2, false, 3).is_err());
        assert!(stage_q15_counted(&mut q, &tw, 2, 2, false, 1).is_ok());
        // The first two stages with the second step 2 need the twiddles 0 and 2.
        assert!(first_stages_q15::<false, false>(&mut q, &tw, 2).is_ok());
        assert!(first_stages_q15::<true, false>(&mut q, &tw, 3).is_err());
        let mut c = [C32::default(); 8];
        let tw = [C32::default(); 3];
        assert!(stage_c32::<false>(&mut c, &tw, 4, 1).is_err());
        assert!(stage_c32::<false>(&mut c, &tw, 2, 3).is_err());
        assert!(stage_c32::<false>(&mut c, &tw, 2, 2).is_ok());
        assert!(stage_c32::<false>(&mut c, &tw, 2, 1).is_ok());
        // A pair of half 1 and step 4 needs the twiddles 0 and 2; of half 2 and step 2 the
        // twiddles 0, 2 (first stage) and 0 to 3 (second stage).
        assert!(stage_pair_c32::<false>(&mut c, &tw, 1, 4).is_ok());
        assert!(stage_pair_c32::<false>(&mut c, &tw, 1, 8).is_err());
        assert!(stage_pair_c32::<true>(&mut c, &tw, 2, 2).is_err());
        let tw4 = [C32::default(); 4];
        assert!(stage_pair_c32::<true>(&mut c, &tw4, 2, 2).is_ok());
    }
}
