//! Fallback implementations of `quad_demod`, `deemphasis` and `dc_block`.

use core::f64::consts::PI;

use crate::Fallback;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::demod::{
    DcBlock, DcBlockReal, DcBlockRealState, DcBlockState, Deemphasis, DeemphasisState, QuadDemod,
    QuadDemodState,
};
use crate::num::{flush_denormal, q15_from_f64, q15_mul, round_shift, sat16};
use crate::types::{C32, CQ15, Q15};

impl QuadDemod<C32> for Fallback {
    fn quad_demod(
        &self,
        state: &mut QuadDemodState<C32>,
        input: &[C32],
        output: &mut [f32],
        gain: f32,
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let mut prev = state.prev();
        for (o, &x) in output.iter_mut().zip(input) {
            let re = x.re * prev.re + x.im * prev.im;
            let im = x.im * prev.re - x.re * prev.im;
            let arg = if re == 0.0 && im == 0.0 {
                0.0
            } else {
                libm::atan2f(im, re)
            };
            *o = arg * gain;
            prev = x;
        }
        state.set_prev(prev);
        Ok(())
    }
}

impl QuadDemod<CQ15> for Fallback {
    fn quad_demod(
        &self,
        state: &mut QuadDemodState<CQ15>,
        input: &[CQ15],
        output: &mut [Q15],
        gain: Q15,
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let mut prev = state.prev();
        for (o, &x) in output.iter_mut().zip(input) {
            let (ar, ai) = (i64::from(x.re.0), i64::from(x.im.0));
            let (br, bi) = (i64::from(prev.re.0), i64::from(prev.im.0));
            let re = ar * br + ai * bi;
            let im = ai * br - ar * bi;
            let phi = if re == 0 && im == 0 {
                0.0
            } else {
                libm::atan2(im as f64, re as f64) / PI
            };
            *o = q15_mul(q15_from_f64(phi), gain);
            prev = x;
        }
        state.set_prev(prev);
        Ok(())
    }
}

// The recursions of `deemphasis` and `dc_block` have two forms with the same bits: the loop of the
// contract as written (docs/kernels.md), and a form arranged so that the chain from one output to
// the next, which bounds their speed, is shorter:
//
// - `f32`: the flush is taken off the chain. The comparison of the unflushed value runs beside the
//   arithmetic of the next step and selects between the two possible results (`dc_block`: the
//   product `r y` or 0, which is `r * 0.0` for the positive `r`; `deemphasis`: the step from the
//   value or from 0).
// - Q15: the terms that do not depend on the previous output (the input difference or
//   `alpha x`, and the rounding constant) are added in `i32` first, which is exact, and the sign
//   extension keeps the compiler from adding the rounding constant after the product again: the
//   chain is the product, one addition and the shift. `y + ((v + 2^14) >> 15)` equals
//   `(y 2^15 + v + 2^14) >> 15` for every integer `y`, so the rounding is the same.
//
// The short forms cost instructions (the second result of the `f32` select, the `i32` terms), so
// each kernel uses them only on the targets where they were measured faster on both core models
// of every level (issue #20, simulated cycles per sample): on x86-64 by `llvm-mca` at `sse2`,
// `sse4.2`, `avx2` and `avx512`, on AArch64 at `cortex-a72` and `cortex-a76`, on wasm32 by
// instructions. Elsewhere, and on the in-order embedded cores, the contract loop runs.

/// `dc_block_real` `f32`: on x86-64 (`znver3` / `alderlake` at `avx2` 9.03 / 13.04 to
/// 7.03 / 9.04 cycles per sample) and wasm32 (28.53 to 27.54 instructions); not on AArch64, where
/// `cortex-a72` loses 7 %.
const SHORT_DC_REAL_F32: bool = cfg!(any(target_arch = "x86_64", target_family = "wasm"));
/// `dc_block` `C32`: on x86-64 (as `dc_block_real`) and AArch64 (`cortex-a72` / `cortex-a76`
/// 21.05 / 11.05 to 16.04 / 8.04); not on wasm32 (26.53 to 32.03 instructions).
const SHORT_DC_C32: bool = cfg!(any(target_arch = "x86_64", target_arch = "aarch64"));
/// `deemphasis` `f32`: on x86-64 (12.03 / 16.05 to 9.54 / 11.57 at `avx2`); not on AArch64
/// (`cortex-a72` 5 % slower) and wasm32 (27.53 to 42.03 instructions).
const SHORT_DEEMPHASIS_F32: bool = cfg!(target_arch = "x86_64");
/// `deemphasis` `Q15`: with AVX2 (7.04 / 7.05 to 5.53 / 5.53 at `avx2`, 7.07 / 7.08 to
/// 5.77 / 5.52 at `avx512`); not below it (`znver2` 3 % slower at `sse2` and `sse4.2`), on
/// AArch64 (8 % slower) and wasm32 (5 % more instructions).
const SHORT_DEEMPHASIS_Q15: bool = cfg!(all(target_arch = "x86_64", target_feature = "avx2"));
/// `dc_block_real` `Q15`: on x86-64 (7.02 / 6.64 to 5.53 / 5.56 at `avx2`, 7.00 / 7.76 to
/// 6.06 / 7.54 at `sse2`); not on AArch64 (10 % slower) and wasm32 (no fewer instructions).
const SHORT_DC_REAL_Q15: bool = cfg!(target_arch = "x86_64");
/// `dc_block` `CQ15`: with AVX2 but not AVX-512 (11.07 / 11.10 to 6.30 / 6.12 at `avx2`, where
/// the contract loop compiles to a longer chain; `sapphirerapids` 7.79 to 15.79 at `avx512`,
/// `skylake` 7 % slower at `sse2`) and on wasm32 (87.03 to 53.03 instructions); not on AArch64
/// (9 % slower).
const SHORT_DC_CQ15: bool = cfg!(any(
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ),
    target_family = "wasm"
));

/// `true` where [`flush_denormal`] replaces `x` by 0.
#[inline]
fn is_flushed(x: f32) -> bool {
    libm::fabsf(x) < 1e-30
}

/// `deemphasis` `f32` as written: `y += alpha (x - y)`, flushed.
#[inline]
fn deemphasis_f32_plain(alpha: f32, last: &mut f32, input: &[f32], output: &mut [f32]) {
    let mut y = *last;
    for (o, &x) in output.iter_mut().zip(input) {
        y += alpha * (x - y);
        y = flush_denormal(y);
        *o = y;
    }
    *last = y;
}

/// `deemphasis` `f32` with the flush off the chain.
#[inline]
fn deemphasis_f32_short(alpha: f32, last: &mut f32, input: &[f32], output: &mut [f32]) {
    // `u` is the value before the flush; the stored state is flushed already, and flushing it
    // again changes nothing. The output is `flush(u)`, and the next `u` is `y + alpha * (x - y)`
    // with `y = flush(u)`: with `y = u` or with `y = 0`.
    let mut u = *last;
    for (o, &x) in output.iter_mut().zip(input) {
        let kept = u + alpha * (x - u);
        let flushed = 0.0 + alpha * x;
        u = if is_flushed(u) { flushed } else { kept };
        *o = flush_denormal(u);
    }
    *last = flush_denormal(u);
}

impl Deemphasis<f32> for Fallback {
    fn deemphasis(
        &self,
        state: &mut DeemphasisState<f32>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (alpha, last) = state.parts_mut();
        if SHORT_DEEMPHASIS_F32 {
            deemphasis_f32_short(alpha, last, input, output);
        } else {
            deemphasis_f32_plain(alpha, last, input, output);
        }
        Ok(())
    }
}

/// `deemphasis` `Q15` as written: `y = sat16(y + round_shift(alpha (x - y), 15))`.
#[inline]
fn deemphasis_q15_plain(alpha: Q15, last: &mut Q15, input: &[Q15], output: &mut [Q15]) {
    let mut y = i64::from(last.0);
    for (o, &x) in output.iter_mut().zip(input) {
        let d = i64::from(x.0) - y;
        y = i64::from(sat16(y + round_shift(i64::from(alpha.0) * d, 15)));
        *o = Q15(y as i16);
    }
    *last = Q15(y as i16);
}

/// `deemphasis` `Q15` as `sat16(((2^15 - alpha) y + alpha x + 2^14) >> 15)`.
#[inline]
fn deemphasis_q15_short(alpha: Q15, last: &mut Q15, input: &[Q15], output: &mut [Q15]) {
    let keep = (1i64 << 15) - i64::from(alpha.0);
    let mut y = i64::from(last.0);
    for (o, &x) in output.iter_mut().zip(input) {
        // `|alpha x| <= 2^30`: exact in `i32`.
        let c = i32::from(alpha.0) * i32::from(x.0) + (1 << 14);
        y = i64::from(sat16((keep * y + i64::from(c)) >> 15));
        *o = Q15(y as i16);
    }
    *last = Q15(y as i16);
}

impl Deemphasis<Q15> for Fallback {
    fn deemphasis(
        &self,
        state: &mut DeemphasisState<Q15>,
        input: &[Q15],
        output: &mut [Q15],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (alpha, last) = state.parts_mut();
        if SHORT_DEEMPHASIS_Q15 {
            deemphasis_q15_short(alpha, last, input, output);
        } else {
            deemphasis_q15_plain(alpha, last, input, output);
        }
        Ok(())
    }
}

/// `(x - x_prev) + r * y_prev` of `dc_block` as written.
#[inline]
fn dc_f32(r: f32, x: f32, x_prev: f32, y_prev: f32) -> f32 {
    (x - x_prev) + r * y_prev
}

/// [`dc_f32`] with the difference `d = x - x_prev` and the unflushed `y`: the flush is applied
/// to the product, beside the multiplication (`r * 0.0 = 0.0` for the positive `r`).
#[inline]
fn dc_f32_short(r: f32, d: f32, y: f32) -> f32 {
    let p = r * y;
    d + if is_flushed(y) { 0.0 } else { p }
}

/// `sat16((x - x_prev) + round_shift(r * y_prev, 15))` of `dc_block` as written.
#[inline]
fn dc_q15(r: Q15, x: Q15, x_prev: Q15, y_prev: Q15) -> Q15 {
    let v = i64::from(x.0) - i64::from(x_prev.0)
        + round_shift(i64::from(r.0) * i64::from(y_prev.0), 15);
    Q15(sat16(v))
}

/// [`dc_q15`] as `sat16((r y + (x - x_prev) 2^15 + 2^14) >> 15)`.
#[inline]
fn dc_q15_short(r: i64, x: Q15, x_prev: Q15, y: i64) -> i64 {
    // `|x - x_prev| <= 65535`, so `(x - x_prev) 2^15 + 2^14` is exact in `i32`.
    let c = (i32::from(x.0) - i32::from(x_prev.0)) * (1 << 15) + (1 << 14);
    i64::from(sat16((r * y + i64::from(c)) >> 15))
}

/// `dc_block` `C32` as written.
#[inline]
fn dc_block_c32_plain(
    r: f32,
    x_prev: &mut C32,
    y_prev: &mut C32,
    input: &[C32],
    output: &mut [C32],
) {
    for (o, &x) in output.iter_mut().zip(input) {
        let y = C32::new(
            dc_f32(r, x.re, x_prev.re, y_prev.re),
            dc_f32(r, x.im, x_prev.im, y_prev.im),
        );
        *o = y;
        *x_prev = x;
        *y_prev = C32::new(flush_denormal(y.re), flush_denormal(y.im));
    }
}

/// `dc_block` `C32` with the flush off the chain.
#[inline]
fn dc_block_c32_short(
    r: f32,
    x_prev: &mut C32,
    y_prev: &mut C32,
    input: &[C32],
    output: &mut [C32],
) {
    // `y` is the last output, unflushed; the stored `y_prev` is flushed already.
    let (mut xp, mut y) = (*x_prev, *y_prev);
    for (o, &x) in output.iter_mut().zip(input) {
        y = C32::new(
            dc_f32_short(r, x.re - xp.re, y.re),
            dc_f32_short(r, x.im - xp.im, y.im),
        );
        *o = y;
        xp = x;
    }
    *x_prev = xp;
    *y_prev = C32::new(flush_denormal(y.re), flush_denormal(y.im));
}

impl DcBlock<C32> for Fallback {
    fn dc_block(
        &self,
        state: &mut DcBlockState<C32>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (r, x_prev, y_prev) = state.parts_mut();
        if SHORT_DC_C32 {
            dc_block_c32_short(r, x_prev, y_prev, input, output);
        } else {
            dc_block_c32_plain(r, x_prev, y_prev, input, output);
        }
        Ok(())
    }
}

/// `dc_block` `CQ15` as written.
#[inline]
fn dc_block_cq15_plain(
    r: Q15,
    x_prev: &mut CQ15,
    y_prev: &mut CQ15,
    input: &[CQ15],
    output: &mut [CQ15],
) {
    for (o, &x) in output.iter_mut().zip(input) {
        let y = CQ15 {
            re: dc_q15(r, x.re, x_prev.re, y_prev.re),
            im: dc_q15(r, x.im, x_prev.im, y_prev.im),
        };
        *o = y;
        *x_prev = x;
        *y_prev = y;
    }
}

/// `dc_block` `CQ15` with the input terms added before the product.
#[inline]
fn dc_block_cq15_short(
    r: Q15,
    x_prev: &mut CQ15,
    y_prev: &mut CQ15,
    input: &[CQ15],
    output: &mut [CQ15],
) {
    let r = i64::from(r.0);
    let mut xp = *x_prev;
    let (mut yr, mut yi) = (i64::from(y_prev.re.0), i64::from(y_prev.im.0));
    for (o, &x) in output.iter_mut().zip(input) {
        yr = dc_q15_short(r, x.re, xp.re, yr);
        yi = dc_q15_short(r, x.im, xp.im, yi);
        *o = CQ15 {
            re: Q15(yr as i16),
            im: Q15(yi as i16),
        };
        xp = x;
    }
    *x_prev = xp;
    *y_prev = CQ15 {
        re: Q15(yr as i16),
        im: Q15(yi as i16),
    };
}

impl DcBlock<CQ15> for Fallback {
    fn dc_block(
        &self,
        state: &mut DcBlockState<CQ15>,
        input: &[CQ15],
        output: &mut [CQ15],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (r, x_prev, y_prev) = state.parts_mut();
        if SHORT_DC_CQ15 {
            dc_block_cq15_short(r, x_prev, y_prev, input, output);
        } else {
            dc_block_cq15_plain(r, x_prev, y_prev, input, output);
        }
        Ok(())
    }
}

/// `dc_block_real` `f32` as written.
#[inline]
fn dc_block_real_f32_plain(
    r: f32,
    x_prev: &mut f32,
    y_prev: &mut f32,
    input: &[f32],
    output: &mut [f32],
) {
    for (o, &x) in output.iter_mut().zip(input) {
        let y = dc_f32(r, x, *x_prev, *y_prev);
        *o = y;
        *x_prev = x;
        *y_prev = flush_denormal(y);
    }
}

/// `dc_block_real` `f32` with the flush off the chain.
#[inline]
fn dc_block_real_f32_short(
    r: f32,
    x_prev: &mut f32,
    y_prev: &mut f32,
    input: &[f32],
    output: &mut [f32],
) {
    // `y` is the last output, unflushed; the stored `y_prev` is flushed already.
    let (mut xp, mut y) = (*x_prev, *y_prev);
    for (o, &x) in output.iter_mut().zip(input) {
        y = dc_f32_short(r, x - xp, y);
        *o = y;
        xp = x;
    }
    *x_prev = xp;
    *y_prev = flush_denormal(y);
}

impl DcBlockReal<f32> for Fallback {
    fn dc_block_real(
        &self,
        state: &mut DcBlockRealState<f32>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (r, x_prev, y_prev) = state.parts_mut();
        if SHORT_DC_REAL_F32 {
            dc_block_real_f32_short(r, x_prev, y_prev, input, output);
        } else {
            dc_block_real_f32_plain(r, x_prev, y_prev, input, output);
        }
        Ok(())
    }
}

/// `dc_block_real` `Q15` as written.
#[inline]
fn dc_block_real_q15_plain(
    r: Q15,
    x_prev: &mut Q15,
    y_prev: &mut Q15,
    input: &[Q15],
    output: &mut [Q15],
) {
    for (o, &x) in output.iter_mut().zip(input) {
        let y = dc_q15(r, x, *x_prev, *y_prev);
        *o = y;
        *x_prev = x;
        *y_prev = y;
    }
}

/// `dc_block_real` `Q15` with the input terms added before the product.
#[inline]
fn dc_block_real_q15_short(
    r: Q15,
    x_prev: &mut Q15,
    y_prev: &mut Q15,
    input: &[Q15],
    output: &mut [Q15],
) {
    let r = i64::from(r.0);
    let (mut xp, mut y) = (*x_prev, i64::from(y_prev.0));
    for (o, &x) in output.iter_mut().zip(input) {
        y = dc_q15_short(r, x, xp, y);
        *o = Q15(y as i16);
        xp = x;
    }
    *x_prev = xp;
    *y_prev = Q15(y as i16);
}

impl DcBlockReal<Q15> for Fallback {
    fn dc_block_real(
        &self,
        state: &mut DcBlockRealState<Q15>,
        input: &[Q15],
        output: &mut [Q15],
    ) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (r, x_prev, y_prev) = state.parts_mut();
        if SHORT_DC_REAL_Q15 {
            dc_block_real_q15_short(r, x_prev, y_prev, input, output);
        } else {
            dc_block_real_q15_plain(r, x_prev, y_prev, input, output);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! Both forms of every recursion (whichever the target compiles into the kernel) against the
    //! loops of the contract as written (docs/kernels.md), bit for bit, over inputs that reach the
    //! flush of `f32`, the saturation of Q15, signed zeros, NaN and infinities, with the input
    //! split into blocks of every length up to 9, and through the kernel of the target.

    use std::vec;
    use std::vec::Vec;

    use super::{
        dc_block_c32_plain, dc_block_c32_short, dc_block_cq15_plain, dc_block_cq15_short,
        dc_block_real_f32_plain, dc_block_real_f32_short, dc_block_real_q15_plain,
        dc_block_real_q15_short, deemphasis_f32_plain, deemphasis_f32_short, deemphasis_q15_plain,
        deemphasis_q15_short,
    };
    use crate::Fallback;
    use crate::kernels::demod::{
        DcBlock, DcBlockReal, DcBlockRealState, DcBlockState, Deemphasis, DeemphasisState,
    };
    use crate::num::{flush_denormal, round_shift, sat16};
    use crate::timing::Rate;
    use crate::types::{C32, CQ15, Q15};

    /// The two forms of each recursion.
    type DeemphasisF32 = fn(f32, &mut f32, &[f32], &mut [f32]);
    type DeemphasisQ15 = fn(Q15, &mut Q15, &[Q15], &mut [Q15]);
    type RealF32 = fn(f32, &mut f32, &mut f32, &[f32], &mut [f32]);
    type ComplexF32 = fn(f32, &mut C32, &mut C32, &[C32], &mut [C32]);
    type RealQ15 = fn(Q15, &mut Q15, &mut Q15, &[Q15], &mut [Q15]);
    type ComplexQ15 = fn(Q15, &mut CQ15, &mut CQ15, &[CQ15], &mut [CQ15]);

    /// A small linear congruential generator.
    fn next(s: &mut u32) -> u32 {
        *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *s >> 8
    }

    /// `f32` inputs: noise with an offset, special values, silence long enough for the output to
    /// decay below the flush threshold of 1e-30, then a tiny signal around it.
    fn f32_input(seed: u32) -> Vec<f32> {
        let mut s = seed;
        let mut v: Vec<f32> = (0..300)
            .map(|_| next(&mut s) as f32 / 8_388_608.0 - 0.7)
            .collect();
        v.extend([
            0.0,
            -0.0,
            1e30,
            -1e30,
            1e-31,
            -1e-31,
            f32::MIN_POSITIVE,
            -0.0,
            f32::INFINITY,
            1.0,
            f32::NAN,
            0.5,
        ]);
        v.extend(core::iter::repeat_n(0.0, 1500));
        v.extend((0..400).map(|i| match i % 4 {
            0 => 2e-30,
            1 => -1e-30,
            2 => 0.0,
            _ => -0.0,
        }));
        v
    }

    /// Q15 inputs: noise, full-scale steps and alternation, silence.
    fn q15_input(seed: u32) -> Vec<Q15> {
        let mut s = seed;
        let mut v: Vec<Q15> = (0..400).map(|_| Q15((next(&mut s) >> 8) as i16)).collect();
        v.extend([Q15::MAX; 50]);
        v.extend([Q15::MIN; 50]);
        v.extend((0..100).map(|i| if i % 2 == 0 { Q15::MAX } else { Q15::MIN }));
        v.extend([Q15(0); 100]);
        v
    }

    /// Splits `n` samples into blocks of 1 to 9 samples (and one empty block).
    fn blocks(n: usize, seed: u32) -> Vec<(usize, usize)> {
        let mut s = seed;
        let (mut out, mut at) = (vec![(0, 0)], 0);
        while at < n {
            let len = (1 + next(&mut s) as usize % 9).min(n - at);
            out.push((at, at + len));
            at += len;
        }
        out
    }

    /// The bits of `x`, with every NaN as one value (Rust leaves the sign and payload of a NaN
    /// produced by arithmetic, such as `inf - inf`, unspecified).
    fn bits(x: &[f32]) -> Vec<u32> {
        x.iter().map(|&v| canonical(v)).collect()
    }

    fn canonical(v: f32) -> u32 {
        if v.is_nan() {
            f32::NAN.to_bits()
        } else {
            v.to_bits()
        }
    }

    #[test]
    fn deemphasis_f32_is_the_contract_loop() {
        let rate = Rate::hz(48_000).unwrap();
        for (k, tau) in [50e-6, 75e-6, 1e-7, 1.0].into_iter().enumerate() {
            let x = f32_input(k as u32);
            let state = DeemphasisState::<f32>::new(tau, rate).unwrap();
            let alpha = state.alpha();
            let mut want = Vec::new();
            let mut y = 0.0f32;
            for &x in &x {
                y += alpha * (x - y);
                y = flush_denormal(y);
                want.push(y);
            }
            let forms: [DeemphasisF32; 2] = [deemphasis_f32_plain, deemphasis_f32_short];
            for (f, form) in forms.into_iter().enumerate() {
                let (mut last, mut got) = (0.0f32, vec![0.0f32; x.len()]);
                for (a, b) in blocks(x.len(), 7 + k as u32) {
                    form(alpha, &mut last, &x[a..b], &mut got[a..b]);
                }
                assert_eq!(bits(&want), bits(&got), "tau {tau} form {f}");
                assert_eq!(canonical(last), canonical(y), "tau {tau} form {f}");
            }
            let (mut st, mut got) = (state, vec![0.0f32; x.len()]);
            Deemphasis::<f32>::deemphasis(&Fallback, &mut st, &x, &mut got).unwrap();
            assert_eq!(bits(&want), bits(&got), "tau {tau}");
        }
    }

    #[test]
    fn deemphasis_q15_is_the_contract_loop() {
        let rate = Rate::hz(48_000).unwrap();
        // `alpha` about 0.24 and 0.34, and the extremes 32767 and 0.
        for (k, tau) in [75e-6, 50e-6, 1e-9, 1e3].into_iter().enumerate() {
            let x = q15_input(k as u32);
            let state = DeemphasisState::<Q15>::new(tau, rate).unwrap();
            let alpha = state.alpha();
            let mut want = Vec::new();
            let mut y = 0i64;
            for &x in &x {
                let d = i64::from(x.0) - y;
                y = i64::from(sat16(y + round_shift(i64::from(alpha.0) * d, 15)));
                want.push(Q15(y as i16));
            }
            let forms: [DeemphasisQ15; 2] = [deemphasis_q15_plain, deemphasis_q15_short];
            for (f, form) in forms.into_iter().enumerate() {
                let (mut last, mut got) = (Q15(0), vec![Q15(0); x.len()]);
                for (a, b) in blocks(x.len(), 11 + k as u32) {
                    form(alpha, &mut last, &x[a..b], &mut got[a..b]);
                }
                assert_eq!(want, got, "tau {tau} form {f}");
                assert_eq!(last, Q15(y as i16), "tau {tau} form {f}");
            }
            let (mut st, mut got) = (state, vec![Q15(0); x.len()]);
            Deemphasis::<Q15>::deemphasis(&Fallback, &mut st, &x, &mut got).unwrap();
            assert_eq!(want, got, "tau {tau}");
        }
    }

    /// `(x - x_prev) + r * y_prev` as written, with the stored `y_prev` flushed.
    fn dc_f32_contract(r: f32, x: &[f32]) -> (Vec<f32>, f32, f32) {
        let (mut xp, mut yp) = (0.0f32, 0.0f32);
        let mut out = Vec::new();
        for &x in x {
            let y = (x - xp) + r * yp;
            out.push(y);
            xp = x;
            yp = flush_denormal(y);
        }
        (out, xp, yp)
    }

    /// `sat16((x - x_prev) + round_shift(r * y_prev, 15))` as written.
    fn dc_q15_contract(r: Q15, x: &[Q15]) -> (Vec<Q15>, Q15, Q15) {
        let (mut xp, mut yp) = (Q15(0), Q15(0));
        let mut out = Vec::new();
        for &x in x {
            let v = i64::from(x.0) - i64::from(xp.0)
                + round_shift(i64::from(r.0) * i64::from(yp.0), 15);
            let y = Q15(sat16(v));
            out.push(y);
            xp = x;
            yp = y;
        }
        (out, xp, yp)
    }

    #[test]
    fn dc_block_f32_is_the_contract_loop() {
        for (k, r) in [0.999f32, 0.5, 1e-6, 0.999_999_9].into_iter().enumerate() {
            let re = f32_input(k as u32);
            let im = f32_input(100 + k as u32);
            let (want_re, xr, yr) = dc_f32_contract(r, &re);
            let (want_im, xi, yi) = dc_f32_contract(r, &im);
            let forms: [RealF32; 2] = [dc_block_real_f32_plain, dc_block_real_f32_short];
            for (f, form) in forms.into_iter().enumerate() {
                let (mut xp, mut yp, mut got) = (0.0f32, 0.0f32, vec![0.0f32; re.len()]);
                for (a, b) in blocks(re.len(), 13 + k as u32) {
                    form(r, &mut xp, &mut yp, &re[a..b], &mut got[a..b]);
                }
                assert_eq!(bits(&want_re), bits(&got), "r {r} form {f}");
                assert_eq!([xp, yp].map(canonical), [xr, yr].map(canonical));
            }
            let mut st = DcBlockRealState::<f32>::new(r).unwrap();
            let mut got = vec![0.0f32; re.len()];
            DcBlockReal::<f32>::dc_block_real(&Fallback, &mut st, &re, &mut got).unwrap();
            assert_eq!(bits(&want_re), bits(&got), "r {r}");

            let x: Vec<C32> = re.iter().zip(&im).map(|(&a, &b)| C32::new(a, b)).collect();
            let forms: [ComplexF32; 2] = [dc_block_c32_plain, dc_block_c32_short];
            for (f, form) in forms.into_iter().enumerate() {
                let (mut xp, mut yp) = (C32::default(), C32::default());
                let mut got = vec![C32::default(); x.len()];
                for (a, b) in blocks(x.len(), 17 + k as u32) {
                    form(r, &mut xp, &mut yp, &x[a..b], &mut got[a..b]);
                }
                let got_re: Vec<f32> = got.iter().map(|c| c.re).collect();
                let got_im: Vec<f32> = got.iter().map(|c| c.im).collect();
                assert_eq!(bits(&want_re), bits(&got_re), "r {r} form {f}");
                assert_eq!(bits(&want_im), bits(&got_im), "r {r} form {f}");
                assert_eq!(
                    [xp.re, xp.im, yp.re, yp.im].map(canonical),
                    [xr, xi, yr, yi].map(canonical)
                );
            }
            let mut st = DcBlockState::<C32>::new(r).unwrap();
            let mut got = vec![C32::default(); x.len()];
            DcBlock::<C32>::dc_block(&Fallback, &mut st, &x, &mut got).unwrap();
            let got_re: Vec<f32> = got.iter().map(|c| c.re).collect();
            assert_eq!(bits(&want_re), bits(&got_re), "r {r}");
        }
    }

    #[test]
    fn dc_block_q15_is_the_contract_loop() {
        for (k, r) in [Q15(32_735), Q15(32_767), Q15(1), Q15(16_384)]
            .into_iter()
            .enumerate()
        {
            let re = q15_input(k as u32);
            let im = q15_input(100 + k as u32);
            let (want_re, xr, yr) = dc_q15_contract(r, &re);
            let (want_im, xi, yi) = dc_q15_contract(r, &im);
            let forms: [RealQ15; 2] = [dc_block_real_q15_plain, dc_block_real_q15_short];
            for (f, form) in forms.into_iter().enumerate() {
                let (mut xp, mut yp, mut got) = (Q15(0), Q15(0), vec![Q15(0); re.len()]);
                for (a, b) in blocks(re.len(), 19 + k as u32) {
                    form(r, &mut xp, &mut yp, &re[a..b], &mut got[a..b]);
                }
                assert_eq!(want_re, got, "r {r:?} form {f}");
                assert_eq!((xp, yp), (xr, yr));
            }
            let mut st = DcBlockRealState::<Q15>::new(r).unwrap();
            let mut got = vec![Q15(0); re.len()];
            DcBlockReal::<Q15>::dc_block_real(&Fallback, &mut st, &re, &mut got).unwrap();
            assert_eq!(want_re, got, "r {r:?}");

            let x: Vec<CQ15> = re
                .iter()
                .zip(&im)
                .map(|(&re, &im)| CQ15 { re, im })
                .collect();
            let want: Vec<CQ15> = want_re
                .iter()
                .zip(&want_im)
                .map(|(&re, &im)| CQ15 { re, im })
                .collect();
            let forms: [ComplexQ15; 2] = [dc_block_cq15_plain, dc_block_cq15_short];
            for (f, form) in forms.into_iter().enumerate() {
                let (mut xp, mut yp) = (CQ15::default(), CQ15::default());
                let mut got = vec![CQ15::default(); x.len()];
                for (a, b) in blocks(x.len(), 23 + k as u32) {
                    form(r, &mut xp, &mut yp, &x[a..b], &mut got[a..b]);
                }
                assert_eq!(want, got, "r {r:?} form {f}");
                assert_eq!(xp, CQ15 { re: xr, im: xi });
                assert_eq!(yp, CQ15 { re: yr, im: yi });
            }
            let mut st = DcBlockState::<CQ15>::new(r).unwrap();
            let mut got = vec![CQ15::default(); x.len()];
            DcBlock::<CQ15>::dc_block(&Fallback, &mut st, &x, &mut got).unwrap();
            assert_eq!(want, got, "r {r:?}");
        }
    }
}
