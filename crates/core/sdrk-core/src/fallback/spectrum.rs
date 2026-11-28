//! Fallback implementations of `window`, `mag_sq`, `mag` and `power_db`.
//!
//! The loops are written so that the compiler can vectorize them (no calls, no branches, no
//! 64-bit products): the `Q15` arithmetic in 32-bit lanes, the rounded root of `mag` from a
//! single-precision root and one correction step, the logarithm of `power_db` as the steps of
//! `libm::log10f` without its special cases (the `f32` loop keeps the call on AArch64), and on
//! x86 the `C32` `window` with duplicated weights. Every form gives the bits of the rule of the
//! contract (`docs/kernels.md`); the tests below compare them with it.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::check_len;
use crate::kernels::spectrum::{Mag, MagSq, PowerDb, Window};
use crate::types::{C32, CQ15, Q15};

/// Smallest linear power that `power_db` evaluates; smaller, negative and NaN input use it.
const FLOOR: f32 = 1e-20;

/// `q15_mul(x, w) = sat16(round_shift(x w, 15))` in `i32`: the product is at most `2^30` (the
/// one product that saturates, `(-32768)^2`) and at least `-32768 * 32767`, whose result is
/// -32767, so only the upper limit is needed.
#[inline]
fn q15_mul_i32(x: i16, w: i16) -> i16 {
    ((i32::from(x) * i32::from(w) + (1 << 14)) >> 15).min(32767) as i16
}

/// Samples per block of the `C32` `window` loop with duplicated weights.
const WINDOW_BLOCK: usize = 8;

/// Whether the `C32` `window` loop duplicates the weights first (see the loop). Only on x86 up
/// to AVX2 (instructions per sample at `sse2` / `avx2`: 3.14 / 1.64 against 4.02 / 2.08 for the
/// plain loop). With AVX-512 the plain loop is the shorter one (0.70 against
/// 2.26); AArch64 compiles it to `ld2`/`st2`, which simulated faster than the duplicated weights
/// (`cortex-a72` / `cortex-a76`: 2.03 / 1.53 against 2.27 / 1.90 cycles per sample); the
/// embedded targets keep the plain loop.
const WINDOW_DUPLICATED: bool = cfg!(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    not(target_feature = "avx512f")
));

impl Window<C32> for Fallback {
    fn window(&self, input: &[C32], output: &mut [C32], w: &[f32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        check_len(input.len(), w.len())?;
        if !WINDOW_DUPLICATED {
            for ((o, x), &w) in output.iter_mut().zip(input).zip(w) {
                *o = C32::new(x.re * w, x.im * w);
            }
            return Ok(());
        }
        // On the interleaved components: each weight is written twice into a small array first,
        // which the compiler turns into one shuffle per vector of weights (`vpermps`,
        // `unpcklps`) and whole-vector products of the samples as they lie in memory. As a loop
        // over the samples it splits the parts and interleaves the products again (14 shuffles
        // per 16 samples at AVX2). The products are the same.
        let x: &[f32] = bytemuck::cast_slice(input);
        let o: &mut [f32] = bytemuck::cast_slice_mut(output);
        let mut oc = o.chunks_exact_mut(2 * WINDOW_BLOCK);
        let mut xc = x.chunks_exact(2 * WINDOW_BLOCK);
        let mut wc = w.chunks_exact(WINDOW_BLOCK);
        for ((o, x), w) in (&mut oc).zip(&mut xc).zip(&mut wc) {
            let mut d = [0f32; 2 * WINDOW_BLOCK];
            for (d, &w) in d.chunks_exact_mut(2).zip(w) {
                d.fill(w);
            }
            for ((o, x), d) in o.iter_mut().zip(x).zip(&d) {
                *o = x * d;
            }
        }
        let rest = oc.into_remainder().chunks_exact_mut(2);
        for ((o, x), &w) in rest.zip(xc.remainder().chunks_exact(2)).zip(wc.remainder()) {
            for (o, x) in o.iter_mut().zip(x) {
                *o = x * w;
            }
        }
        Ok(())
    }
}

impl Window<CQ15> for Fallback {
    fn window(&self, input: &[CQ15], output: &mut [CQ15], w: &[Q15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        check_len(input.len(), w.len())?;
        for ((o, x), &w) in output.iter_mut().zip(input).zip(w) {
            *o = CQ15 {
                re: Q15(q15_mul_i32(x.re.0, w.0)),
                im: Q15(q15_mul_i32(x.im.0, w.0)),
            };
        }
        Ok(())
    }
}

impl MagSq<C32> for Fallback {
    fn mag_sq(&self, input: &[C32], output: &mut [f32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (o, x) in output.iter_mut().zip(input) {
            *o = x.re * x.re + x.im * x.im;
        }
        Ok(())
    }
}

/// `sat16(round_shift(re^2 + im^2, 15))` in `u32`: each square is at most `2^30`, the sum at
/// most `2^31`, the sum with the rounding term below `2^32`.
#[inline]
fn mag_sq_q15(x: CQ15) -> i16 {
    let (re, im) = (i32::from(x.re.0), i32::from(x.im.0));
    let s = (re * re) as u32 + (im * im) as u32;
    ((s + (1 << 14)) >> 15).min(32767) as i16
}

impl MagSq<CQ15> for Fallback {
    fn mag_sq(&self, input: &[CQ15], output: &mut [Q15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (o, &x) in output.iter_mut().zip(input) {
            *o = Q15(mag_sq_q15(x));
        }
        Ok(())
    }
}

impl Mag<C32> for Fallback {
    fn mag(&self, input: &[C32], output: &mut [f32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (o, x) in output.iter_mut().zip(input) {
            *o = crate::num::sqrt_f32(x.re * x.re + x.im * x.im);
        }
        Ok(())
    }
}

/// The root of `s` rounded to the nearest integer (`round_sqrt` of the contract), for
/// `0 <= s <= 2 * 32767^2`, as the `i32` the smallest `y >= 0` with `y (y + 1) >= s`.
///
/// `fl(s)` is within a relative `2^-24` of `s`, its correctly rounded root within about
/// `1.5 * 2^-24` of `sqrt(s)`, at most `46341 * 2^-23.4 < 0.01` away; adding `2^23` rounds it
/// to the nearest integer (the root is below `2^22`, so the ulp of the sum is 1 and its low 23
/// bits are the integer). That integer is within one of the rounded root, which is the `y`
/// with `(y - 1) y < s <= y (y + 1)` (from `(y - 1/2)^2 <= s < (y + 1/2)^2` for integers), so
/// one step up where `y (y + 1) < s` or down where `(y - 1) y >= s` gives it (for `s = 0` the
/// step down gives -1, which the caller's limit to `[0, 32767]` turns back into 0). Every
/// product is below `2^31`.
#[inline]
fn round_sqrt_q15(s: i32) -> i32 {
    const MAGIC: f32 = 8_388_608.0; // 2^23
    let r = crate::num::sqrt_f32(s as f32);
    let y = (r + MAGIC).to_bits() as i32 - MAGIC.to_bits() as i32;
    let yy = y * y;
    y + i32::from(yy + y < s) - i32::from(yy - y >= s)
}

/// `sat16(round_sqrt(re^2 + im^2))`. A part of -32768 is limited to -32767 first: its magnitude
/// is at least 32768 either way, so the output saturates as before, and the sum stays below
/// `2^31`.
#[inline]
fn mag_q15(x: CQ15) -> i16 {
    let re = i32::from(x.re.0.max(-32767));
    let im = i32::from(x.im.0.max(-32767));
    round_sqrt_q15(re * re + im * im).clamp(0, 32767) as i16
}

impl Mag<CQ15> for Fallback {
    fn mag(&self, input: &[CQ15], output: &mut [Q15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (o, &x) in output.iter_mut().zip(input) {
            *o = Q15(mag_q15(x));
        }
        Ok(())
    }
}

/// `libm::log10f(x)` for a positive normal `x` or `+inf` (FreeBSD `e_log10f.c` as in `libm`,
/// step for step and without fused multiply-add, so with its bits): the subnormal, zero,
/// negative and NaN cases of `log10f` cannot occur after the floor of `power_db`, and the case
/// `x = 1` gives 0 in the general steps too. Without branches, so that the loops vectorize.
#[allow(
    clippy::many_single_char_names,
    reason = "the names of `e_log10f.c`, so that the steps can be compared with it"
)]
#[inline]
fn log10f_normal(x: f32) -> f32 {
    const IVLN10HI: f32 = f32::from_bits(0x3ede_6000);
    const IVLN10LO: f32 = f32::from_bits(0xb804_ead9);
    const LOG10_2HI: f32 = f32::from_bits(0x3e9a_2080);
    const LOG10_2LO: f32 = f32::from_bits(0x3554_27db);
    const LG1: f32 = f32::from_bits(0x3f2a_aaaa);
    const LG2: f32 = f32::from_bits(0x3ecc_ce13);
    const LG3: f32 = f32::from_bits(0x3e91_e9ee);
    const LG4: f32 = f32::from_bits(0x3e78_9e26);
    // Reduce `x` to `2^k m` with `m` in `[sqrt(2)/2, sqrt(2))`.
    let ix = x.to_bits() + (0x3f80_0000 - 0x3f35_04f3);
    let k = (ix >> 23) as i32 - 0x7f;
    let m = f32::from_bits((ix & 0x007f_ffff) + 0x3f35_04f3);
    let f = m - 1.0;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * LG4);
    let t2 = z * (LG1 + w * LG3);
    let r = t2 + t1;
    let hfsq = 0.5 * f * f;
    let hi = f32::from_bits((f - hfsq).to_bits() & 0xffff_f000);
    let lo = f - hi - hfsq + s * (hfsq + r);
    let dk = k as f32;
    let y = dk * LOG10_2LO + (lo + hi) * IVLN10LO + lo * IVLN10HI + hi * IVLN10HI + dk * LOG10_2HI;
    if x == f32::INFINITY { x } else { y }
}

/// `10 log10f(max(p, 1e-20)) + offset_db`; NaN takes the floor.
#[inline]
fn to_db(p: f32, offset_db: f32) -> f32 {
    let p = if p > FLOOR { p } else { FLOOR };
    10.0 * log10f_normal(p) + offset_db
}

/// [`to_db`] with `libm::log10f` itself (the same bits), for the `f32` loop on AArch64: there
/// the branch-free steps vectorize but simulated slower than the scalar calls (per block,
/// `cortex-a72` / `cortex-a76`, cycles per sample: 21.88 / 9.94 against 19.63 / 8.81), so the
/// `f32` fallback keeps the call there (target-dependent form as in D40). The `Q15` loop gains
/// from the steps on AArch64 too and uses them everywhere.
#[cfg(target_arch = "aarch64")]
#[inline]
fn to_db_libm(p: f32, offset_db: f32) -> f32 {
    let p = if p > FLOOR { p } else { FLOOR };
    10.0 * libm::log10f(p) + offset_db
}

impl PowerDb<f32> for Fallback {
    fn power_db(&self, input: &[f32], output: &mut [f32], offset_db: f32) -> Result<()> {
        check_len(input.len(), output.len())?;
        #[cfg(target_arch = "aarch64")]
        let to_db = to_db_libm;
        for (o, &p) in output.iter_mut().zip(input) {
            *o = to_db(p, offset_db);
        }
        Ok(())
    }
}

impl PowerDb<Q15> for Fallback {
    fn power_db(&self, input: &[Q15], output: &mut [f32], offset_db: f32) -> Result<()> {
        check_len(input.len(), output.len())?;
        for (o, &p) in output.iter_mut().zip(input) {
            *o = to_db(p.to_f32(), offset_db);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! The vectorizable forms against the rules of the contract (`q15_mul`, `round_shift`,
    //! `round_sqrt`, `libm::log10f`), bit for bit.
    #![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

    use super::*;
    use crate::num::{q15_mul, round_shift, round_sqrt, sat16};

    /// Values that reach the corners of the Q15 rules.
    const EDGES: [i16; 13] = [
        i16::MIN,
        -32767,
        -23171,
        -16384,
        -181,
        -1,
        0,
        1,
        181,
        16384,
        23170,
        32766,
        i16::MAX,
    ];

    fn cq15(re: i16, im: i16) -> CQ15 {
        CQ15 {
            re: Q15(re),
            im: Q15(im),
        }
    }

    #[test]
    fn c32_windows_equal_the_products_at_every_length_and_start() {
        // Every length up to five blocks, so every tail after the blocks, at three starts; the
        // values include zeros of both signs, infinities, NaN and subnormals.
        let special = [0.0, -0.0, 1.0, -1.5, f32::INFINITY, f32::NAN, 1e-40, -3e38];
        let x: [C32; 48] = core::array::from_fn(|i| {
            let v = (i as f32 * 0.37).sin();
            C32::new(special[i % 8] * v + v, special[(i * 3) % 8] - v)
        });
        let w: [f32; 48] = core::array::from_fn(|i| special[(i * 5) % 8] + (i as f32 * 0.11).cos());
        let mut y = [C32::new(0.0, 0.0); 48];
        for start in 0..3 {
            for n in 0..=(48 - start) {
                let (x, w) = (&x[start..start + n], &w[start..start + n]);
                Fallback.window(x, &mut y[..n], w).unwrap();
                for ((y, x), &w) in y.iter().zip(x).zip(w) {
                    for (got, want) in [(y.re, x.re * w), (y.im, x.im * w)] {
                        assert!(
                            got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                            "n = {n}, start = {start}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn q15_products_equal_q15_mul_for_every_value() {
        for x in i16::MIN..=i16::MAX {
            for &w in &EDGES {
                assert_eq!(q15_mul_i32(x, w), q15_mul(Q15(x), Q15(w)).0, "{x} * {w}");
                assert_eq!(q15_mul_i32(w, x), q15_mul(Q15(w), Q15(x)).0, "{w} * {x}");
            }
        }
    }

    #[test]
    fn squared_magnitudes_equal_the_rule_for_every_part() {
        let rule = |re: i16, im: i16| {
            let (re, im) = (i64::from(re), i64::from(im));
            sat16(round_shift(re * re + im * im, 15))
        };
        for re in i16::MIN..=i16::MAX {
            for &im in &EDGES {
                assert_eq!(mag_sq_q15(cq15(re, im)), rule(re, im), "{re}, {im}");
                assert_eq!(mag_sq_q15(cq15(im, re)), rule(im, re), "{im}, {re}");
            }
        }
    }

    #[test]
    fn rounded_roots_equal_round_sqrt_on_both_sides_of_every_step() {
        // (`s = 0` gives -1, which `mag_q15` limits to 0.) The rounded root changes between
        // `n^2 + n` and `n^2 + n + 1`; the estimate is within
        // one of it, so both sides of every step and the squares cover every case.
        let max = 2 * 32767i64 * 32767;
        for n in 0..=46341i64 {
            for s in [n * n - n, n * n - n + 1, n * n, n * n + n, n * n + n + 1] {
                if (0..=max).contains(&s) {
                    assert_eq!(
                        i64::from(round_sqrt_q15(s as i32).max(0)),
                        round_sqrt(s),
                        "s = {s}"
                    );
                }
            }
        }
        for s in (0..=max).step_by(9973).chain([max - 1, max]) {
            assert_eq!(
                i64::from(round_sqrt_q15(s as i32).max(0)),
                round_sqrt(s),
                "s = {s}"
            );
        }
    }

    #[test]
    fn magnitudes_equal_the_rule_for_every_part() {
        let rule = |re: i16, im: i16| {
            let (re, im) = (i64::from(re), i64::from(im));
            sat16(round_sqrt(re * re + im * im))
        };
        for re in i16::MIN..=i16::MAX {
            for &im in &EDGES {
                assert_eq!(mag_q15(cq15(re, im)), rule(re, im), "{re}, {im}");
                assert_eq!(mag_q15(cq15(im, re)), rule(im, re), "{im}, {re}");
            }
        }
    }

    /// `libm::log10f` and the vectorizable steps on `x`, bit for bit.
    fn check_log10(x: f32) {
        assert_eq!(
            log10f_normal(x).to_bits(),
            libm::log10f(x).to_bits(),
            "log10f({x:e}) ({:#x})",
            x.to_bits()
        );
    }

    #[test]
    fn the_logarithm_equals_libm_in_every_binade_from_the_floor() {
        // Every binade of positive normal numbers from that of the floor, its ends and a
        // stride through its mantissas.
        let first = FLOOR.to_bits() & 0x7f80_0000;
        for e in (first..0x7f80_0000).step_by(0x0080_0000) {
            for m in (0..0x0080_0000)
                .step_by(1021)
                .chain([1, 0x7f_fffe, 0x7f_ffff])
            {
                check_log10(f32::from_bits(e | m));
            }
        }
        check_log10(1.0);
        check_log10(f32::MAX);
        check_log10(f32::INFINITY);
    }

    /// Every positive normal `f32` from the floor up (about 2.1e9 values; a few minutes in a
    /// release build): `cargo test -p sdrk-core --release -- --ignored every_normal`.
    #[test]
    #[ignore = "exhaustive; run by hand in a release build"]
    fn the_logarithm_equals_libm_for_every_normal_from_the_floor() {
        for b in FLOOR.to_bits()..0x7f80_0000 {
            check_log10(f32::from_bits(b));
        }
    }

    #[test]
    fn power_db_equals_the_libm_rule_with_the_floor() {
        let rule = |p: f32, offset: f32| {
            let p = if p > FLOOR { p } else { FLOOR };
            10.0 * libm::log10f(p) + offset
        };
        let special = [
            f32::NAN,
            -f32::NAN,
            f32::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            1e-21,
            FLOOR,
            f32::from_bits(FLOOR.to_bits() + 1),
            0.5,
            1.0,
            2.0,
            10.0,
            1e30,
            f32::MAX,
            f32::INFINITY,
        ];
        let mut out = [0f32; 18];
        for offset in [0.0, -3.0, 12.5] {
            Fallback.power_db(&special, &mut out, offset).unwrap();
            for (&p, &y) in special.iter().zip(&out) {
                assert_eq!(y.to_bits(), rule(p, offset).to_bits(), "power_db({p:e})");
            }
        }
        // Every Q15 power.
        let (mut chunk, mut y) = ([Q15(0); 256], [0f32; 256]);
        for hi in 0..256i32 {
            for (lo, p) in (0i32..).zip(chunk.iter_mut()) {
                *p = Q15((hi * 256 + lo - 32768) as i16);
            }
            Fallback.power_db(&chunk, &mut y, -3.0).unwrap();
            for (&p, &y) in chunk.iter().zip(&y) {
                assert_eq!(y.to_bits(), rule(p.to_f32(), -3.0).to_bits(), "{}", p.0);
            }
        }
    }
}
