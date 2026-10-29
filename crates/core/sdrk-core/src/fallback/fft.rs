//! Fallback implementation of the FFT kernels: iterative in-place radix-2 with a bit-reversal
//! permutation, decimation in time. The order of operations is part of the numeric contract.

// `(a + b) * 0.5` is the specified, exact halving; `f32::midpoint` is not part of the contract.
#![allow(clippy::manual_midpoint, clippy::many_single_char_names)]

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

/// Swaps `data[i]` and `data[rev(i)]` for `i < rev(i)`.
fn bit_reverse<T>(data: &mut [T], log2n: u32) {
    let shift = usize::BITS - log2n;
    for i in 0..data.len() {
        let j = i.reverse_bits() >> shift;
        if i < j && j < data.len() {
            data.swap(i, j);
        }
    }
}

/// One radix-2 stage of `C32` data: `half` is the half block length, `step` the twiddle step.
fn stage_c32(data: &mut [C32], tw: &[C32], half: usize, step: usize, inverse: bool) -> Result<()> {
    // The twiddles of a block are `tw[k * step]` for `k < half`: checked once here instead of
    // once per butterfly.
    if half == 0 || step == 0 || (half - 1).saturating_mul(step) >= tw.len() {
        return Err(tw_error());
    }
    for block in data.chunks_exact_mut(2 * half) {
        let (lo, hi) = block.split_at_mut(half);
        for ((a, c), w) in lo
            .iter_mut()
            .zip(hi.iter_mut())
            .zip(tw.iter().step_by(step))
        {
            let wi = if inverse { -w.im } else { w.im };
            let tr = w.re * c.re - wi * c.im;
            let ti = w.re * c.im + wi * c.re;
            let x = *a;
            *a = C32::new(x.re + tr, x.im + ti);
            *c = C32::new(x.re - tr, x.im - ti);
        }
    }
    Ok(())
}

/// One radix-2 stage of `CQ15` data with the output shift `shift` (1 for fixed scaling, 0 to 2
/// for block floating point). The twiddle product is rounded to Q15 without saturation; the
/// butterfly sum is shifted with rounding and saturated once. Returns the number of stored
/// components that had to be clamped.
fn stage_q15(
    data: &mut [CQ15],
    tw: &[CQ15],
    half: usize,
    step: usize,
    inverse: bool,
    shift: u32,
) -> Result<usize> {
    // One copy per shift, so that the rounding shifts are constants.
    match shift {
        0 => stage_q15_by::<0>(data, tw, half, step, inverse),
        1 => stage_q15_by::<1>(data, tw, half, step, inverse),
        2 => stage_q15_by::<2>(data, tw, half, step, inverse),
        _ => Err(Error::Internal("block-floating-point shift")),
    }
}

/// [`stage_q15`] with the output shift `SHIFT`.
///
/// The same values as `sat16(round_shift(x, s))` of `crate::num`: the operands are far from the
/// range of `i64` (a twiddle product below `2^31` in magnitude, a sum below `2^17`), so the
/// saturating addition of `round_shift` is a plain one and the rounding is `(x + 2^(s - 1)) >> s`
/// with the shifts as constants.
fn stage_q15_by<const SHIFT: u32>(
    data: &mut [CQ15],
    tw: &[CQ15],
    half: usize,
    step: usize,
    inverse: bool,
) -> Result<usize> {
    let mut clamped = 0usize;
    let mut store = |v: i64| -> Q15 {
        let r = if SHIFT == 0 {
            v
        } else {
            (v + (1 << (SHIFT - 1))) >> SHIFT
        };
        // `sat16` as a clamp, which compiles without branches.
        let s = r.clamp(-32768, 32767);
        clamped += usize::from(s != r);
        Q15(s as i16)
    };
    for block in data.chunks_exact_mut(2 * half) {
        let (lo, hi) = block.split_at_mut(half);
        for (k, (a, c)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
            let w = tw.get(k * step).ok_or_else(tw_error)?;
            let (wr, wi) = (i64::from(w.re.0), i64::from(w.im.0));
            let wi = if inverse { -wi } else { wi };
            let (cr, ci) = (i64::from(c.re.0), i64::from(c.im.0));
            let tr = (wr * cr - wi * ci + (1 << 14)) >> 15;
            let ti = (wr * ci + wi * cr + (1 << 14)) >> 15;
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
    Ok(clamped)
}

/// The block-floating-point shift for the next stage from the largest component magnitude.
fn bfp_shift(data: &[CQ15]) -> u32 {
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
}

/// The `CQ15` transform; also returns how many stored components were clamped (always 0 for
/// block floating point with inputs of complex magnitude up to 1).
pub(crate) fn fft_cq15_counted<Tw: AsRef<[CQ15]>>(
    plan: &FftPlan<CQ15, Tw>,
    data: &mut [CQ15],
    dir: Direction,
    scaling: Scaling,
) -> Result<(FftScale, usize)> {
    check_len(plan.n, data.len())?;
    let tw = plan.twiddles.as_ref();
    let inverse = dir == Direction::Inverse;
    bit_reverse(data, plan.log2n);
    let mut exponent = 0i32;
    let mut clamped = 0usize;
    for s in 1..=plan.log2n {
        let shift = if scaling == Scaling::BlockFloat {
            bfp_shift(data)
        } else {
            1
        };
        let half = 1usize << (s - 1);
        let step = (plan.n >> s) * plan.tw_stride;
        clamped += stage_q15(data, tw, half, step, inverse, shift)?;
        exponent += shift as i32;
    }
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
        for s in 1..=plan.log2n {
            let half = 1usize << (s - 1);
            let step = (plan.n >> s) * plan.tw_stride;
            stage_c32(data, tw, half, step, inverse)?;
        }
        if inverse {
            let g = 1.0 / plan.n as f32;
            for x in data.iter_mut() {
                *x = C32::new(x.re * g, x.im * g);
            }
            Ok(FftScale {
                exponent: plan.log2n as i32,
            })
        } else {
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
        fft_cq15_counted(plan, data, dir, scaling).map(|(scale, _)| scale)
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

    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state >> 33
    }

    fn plan(n: usize) -> FftPlan<CQ15, std::vec::Vec<CQ15>> {
        FftPlan::new(n, std::vec![CQ15::default(); n / 2]).unwrap()
    }

    /// Runs the block-floating-point transform and returns the number of clamped components.
    fn clamps(input: &[CQ15], dir: Direction) -> usize {
        let p = plan(input.len());
        let mut data = input.to_vec();
        fft_cq15_counted(&p, &mut data, dir, Scaling::BlockFloat)
            .unwrap()
            .1
    }

    fn sizes() -> impl Iterator<Item = usize> {
        (4..=12).map(|b| 1usize << b)
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
        let clamped = stage_q15(&mut data, &tw, 1, 1, false, 0).unwrap();
        assert_eq!(clamped, 1);
        assert_eq!(data[0].re, Q15(32767));
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
        // Random components and twiddles over the whole range, with the extremes mixed in
        // (`-32768` makes the largest products and sums, and with shift 0 most clamps).
        let mut s = 7u64;
        let value = |s: &mut u64| -> Q15 {
            let r = lcg(s);
            Q15(match r % 8 {
                0 => -32768,
                1 => 32767,
                _ => (r >> 3) as u16 as i16,
            })
        };
        for log2 in 1..=6 {
            let n = 1usize << log2;
            let x: std::vec::Vec<CQ15> = (0..n)
                .map(|_| CQ15 {
                    re: value(&mut s),
                    im: value(&mut s),
                })
                .collect();
            let tw: std::vec::Vec<CQ15> = (0..n / 2)
                .map(|_| CQ15 {
                    re: value(&mut s),
                    im: value(&mut s),
                })
                .collect();
            for stage in 1..=log2 {
                let half = 1usize << (stage - 1);
                let step = n >> stage;
                for inverse in [false, true] {
                    for shift in 0..=2 {
                        let mut want = x.clone();
                        let mut got = x.clone();
                        let clamps = stage_q15_formula(&mut want, &tw, half, step, inverse, shift);
                        let counted = stage_q15(&mut got, &tw, half, step, inverse, shift).unwrap();
                        assert_eq!(got, want, "n={n} half={half} {inverse} shift {shift}");
                        assert_eq!(counted, clamps, "n={n} half={half} {inverse} shift {shift}");
                    }
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
        assert!(stage_q15(&mut q, &tw, 4, 1, false, 1).is_err());
        assert!(stage_q15(&mut q, &tw, 2, 1, false, 3).is_err());
        let mut c = [C32::default(); 8];
        let tw = [C32::default(); 3];
        assert!(stage_c32(&mut c, &tw, 4, 1, false).is_err());
        assert!(stage_c32(&mut c, &tw, 2, 3, false).is_err());
        assert!(stage_c32(&mut c, &tw, 2, 2, false).is_ok());
        assert!(stage_c32(&mut c, &tw, 2, 1, false).is_ok());
    }
}
