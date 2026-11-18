//! Suite cases of M49: the kernel `scramble` and its extremes (test 5 of the milestone).
//!
//! `scramble_c32` and `scramble_cq15` run every `bits` 1 to 8 in both directions on 1024 random
//! samples with random phase bytes: `Fallback` against `Reference` (bit for bit where the residue
//! is 0, within the contract SNR elsewhere: 120 dB `C32`, 80 dB `CQ15`), the facade against
//! `Fallback` bit for bit, the round trip (exact at `bits` 1 and 2 except the saturating `CQ15`
//! negation of −32768; within 120 dB or 2 LSB at `bits` 3 to 8), and 20 chunk plans (seed 695,
//! chunks of 1 to 300 samples) of `Fallback` and the facade (alternating) that must give the
//! bits of the run in one call. `extremes_m49` (test 4) checks an empty input, every index 0 to
//! 255 at `bits` 8 in both directions, the `CQ15` corners (the saturating negation of the
//! quadrant map and the saturation of the residual product), NaN and the infinities in `C32`
//! (`±∞` stays `±∞` where the residue is 0) and the parameter and length errors, which write
//! nothing. No files are read.

#![allow(
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::too_many_arguments
)]

use sdrk::Sdrk;
use sdrk_core::kernels::scramble::{Scramble, phase_index, split_index};
use sdrk_core::num::{c32_quadrant, cq15_quadrant};
use sdrk_core::{C32, CQ15, ComplexSample, Error, Fallback, Q15, Reference};

use super::ensure;
use crate::arena::Arena;
use crate::invariance::ChunkPlan;
use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;

/// The seed of the inputs and of the chunk plans (M49 test 3).
const SEED: u64 = 695;
/// Chunk plans per type.
const PLANS: u64 = 20;
/// Largest chunk of the plans.
const MAX_CHUNK: usize = 300;
/// Samples per run.
const N: usize = 1024;

/// A sample type of the kernel with the per-type parts of the checks.
trait Scrambled: ComplexSample {
    /// A random sample (`CQ15`: of complex magnitude at most 1).
    fn random(rng: &mut Pcg32) -> Self;
    /// The bits of a sample, for the exact comparisons (`C32`: a NaN component counts by class).
    fn same(a: Self, b: Self) -> bool;
    /// Signal and error energy of `got` against `want`.
    fn energy(want: Self, got: Self) -> (f64, f64);
    /// The round trip of `x` came back as `back`: exact (`bits ≤ 2`, `CQ15` −32768 in a negated
    /// component as −32767) or within the bound of `bits ≥ 3`.
    fn round_trip_ok(x: Self, back: Self, bits: u32, q: u32) -> bool;
    /// The minimum SNR of the contract in dB.
    const SNR_DB: f64;
}

impl Scrambled for C32 {
    fn random(rng: &mut Pcg32) -> Self {
        C32::new(
            (rng.next_f64() * 2.0 - 1.0) as f32,
            (rng.next_f64() * 2.0 - 1.0) as f32,
        )
    }

    fn same(a: Self, b: Self) -> bool {
        let one = |x: f32, y: f32| (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits();
        one(a.re, b.re) && one(a.im, b.im)
    }

    fn energy(want: Self, got: Self) -> (f64, f64) {
        let (wr, wi) = (f64::from(want.re), f64::from(want.im));
        let (dr, di) = (wr - f64::from(got.re), wi - f64::from(got.im));
        (wr * wr + wi * wi, dr * dr + di * di)
    }

    fn round_trip_ok(x: Self, back: Self, bits: u32, _q: u32) -> bool {
        if bits <= 2 {
            x.re.to_bits() == back.re.to_bits() && x.im.to_bits() == back.im.to_bits()
        } else {
            // Within a few f32 ulps of the magnitude (the table phasor has |c + js| = 1 ± 2^-24).
            let (_, e) = Self::energy(x, back);
            e <= 1e-12
        }
    }

    const SNR_DB: f64 = 120.0;
}

impl Scrambled for CQ15 {
    fn random(rng: &mut Pcg32) -> Self {
        loop {
            let re = (rng.next_u32() >> 16) as i32 - 32768;
            let im = (rng.next_u32() >> 16) as i32 - 32768;
            if re * re + im * im <= 1 << 30 {
                return CQ15 {
                    re: Q15(re as i16),
                    im: Q15(im as i16),
                };
            }
        }
    }

    fn same(a: Self, b: Self) -> bool {
        a == b
    }

    fn energy(want: Self, got: Self) -> (f64, f64) {
        let (wr, wi) = (f64::from(want.re.0), f64::from(want.im.0));
        let (dr, di) = (wr - f64::from(got.re.0), wi - f64::from(got.im.0));
        (wr * wr + wi * wi, dr * dr + di * di)
    }

    fn round_trip_ok(x: Self, back: Self, bits: u32, q: u32) -> bool {
        if bits <= 2 {
            let re_neg = matches!(q & 3, 2 | 3);
            let im_neg = matches!(q & 3, 1 | 2);
            let want = |v: Q15, neg: bool| if neg && v.0 == i16::MIN { -32767 } else { v.0 };
            back.re.0 == want(x.re, re_neg) && back.im.0 == want(x.im, im_neg)
        } else {
            (i32::from(x.re.0) - i32::from(back.re.0)).abs() <= 2
                && (i32::from(x.im.0) - i32::from(back.im.0)).abs() <= 2
        }
    }

    const SNR_DB: f64 = 80.0;
}

fn get_mut<X>(s: &mut [X], from: usize, to: usize) -> Result<&mut [X], Failure> {
    s.get_mut(from..to)
        .ok_or_else(|| Failure::new("m49", "slice range"))
}

fn get<X>(s: &[X], from: usize, to: usize) -> Result<&[X], Failure> {
    s.get(from..to)
        .ok_or_else(|| Failure::new("m49", "slice range"))
}

/// Runs `k` over `x` in the chunks of `plan`.
fn chunked<T: ComplexSample, K: Scramble<T>>(
    case: &'static str,
    k: &K,
    x: &[T],
    p: &[u8],
    bits: u32,
    inverse: bool,
    out: &mut [T],
    plan: ChunkPlan,
) -> Result<(), Failure> {
    let mut sizes = plan.sizes();
    let mut pos = 0;
    while pos < x.len() {
        let n = sizes
            .next()
            .ok_or_else(|| Failure::new(case, "plan"))?
            .min(x.len() - pos);
        let e = pos + n;
        k.scramble(
            get(x, pos, e)?,
            get(p, pos, e)?,
            bits,
            inverse,
            get_mut(out, pos, e)?,
        )?;
        pos = e;
    }
    Ok(())
}

/// One type: every `bits` in both directions.
fn one_type<T: Scrambled>(case: &'static str, arena: &mut Arena<'_>, j: u64) -> Result<(), Failure>
where
    Fallback: Scramble<T>,
    Reference: Scramble<T>,
    Sdrk: Scramble<T>,
{
    arena.reset();
    let mut c = arena.carver();
    let x = c.take::<T>(N)?;
    let p = c.take::<u8>(N)?;
    let f = c.take::<T>(N)?;
    let r = c.take::<T>(N)?;
    let s = c.take::<T>(N)?;
    let mut rng = Pcg32::new(SEED, j);
    for v in x.iter_mut() {
        *v = T::random(&mut rng);
    }
    for v in p.iter_mut() {
        *v = rng.next_u32() as u8;
    }
    for bits in 1..=8u32 {
        for inverse in [false, true] {
            Fallback.scramble(x, p, bits, inverse, f)?;
            Reference.scramble(x, p, bits, inverse, r)?;
            Sdrk.scramble(x, p, bits, inverse, s)?;
            if let Some(i) = bit_exact(f, s) {
                return Err(Failure::new(case, "the facade differs from the fallback")
                    .detail(format_args!("bits {bits}, inverse {inverse}, sample {i}")));
            }
            let (mut sig, mut err) = (0.0f64, 0.0f64);
            for (i, ((&a, &b), &v)) in f.iter().zip(r.iter()).zip(p.iter()).enumerate() {
                let (_, res) = split_index(phase_index(v, bits, inverse), bits);
                if res == 0 {
                    ensure(T::same(a, b), || {
                        Failure::new(case, "fallback differs from the reference at r = 0")
                            .detail(format_args!("bits {bits}, inverse {inverse}, sample {i}"))
                    })?;
                } else {
                    let (es, ee) = T::energy(b, a);
                    sig += es;
                    err += ee;
                }
            }
            if err > 0.0 {
                let snr = 10.0 * libm::log10(sig / err);
                ensure(snr >= T::SNR_DB, || {
                    Failure::new(case, "fallback misses the contract SNR")
                        .detail(format_args!("bits {bits}, inverse {inverse}, {snr:.1} dB"))
                })?;
            }
            // The round trip: descramble the fallback output into `r`.
            if !inverse {
                Fallback.scramble(f, p, bits, true, r)?;
                for (i, ((&a, &b), &v)) in x.iter().zip(r.iter()).zip(p.iter()).enumerate() {
                    let (q, _) = split_index(phase_index(v, bits, false), bits);
                    ensure(T::round_trip_ok(a, b, bits, q), || {
                        Failure::new(case, "the round trip leaves its bound")
                            .detail(format_args!("bits {bits}, sample {i}"))
                    })?;
                }
            }
        }
    }
    // 20 chunk plans at bits 2 and 8 against the run in one call.
    for (bits, inverse) in [(2, false), (8, true)] {
        Fallback.scramble(x, p, bits, inverse, f)?;
        for plan in 0..PLANS {
            let seed = SEED * 1000 + 100 * j + plan;
            s.fill(T::ZERO);
            let pl = ChunkPlan::new(seed, MAX_CHUNK);
            if plan % 2 == 0 {
                chunked(case, &Fallback, x, p, bits, inverse, s, pl)?;
            } else {
                chunked(case, &Sdrk, x, p, bits, inverse, s, pl)?;
            }
            if let Some(i) = bit_exact(f, s) {
                return Err(Failure::new(case, "a chunk plan changes the output")
                    .detail(format_args!("bits {bits}, plan seed {seed}, sample {i}")));
            }
        }
    }
    Ok(())
}

/// `C32`.
pub(super) fn scramble_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    one_type::<C32>("scramble_c32", arena, 0)
}

/// `CQ15`.
pub(super) fn scramble_cq15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    one_type::<CQ15>("scramble_cq15", arena, 1)
}

/// The extremes of test 4.
pub(super) fn extremes_m49(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_m49";
    // An empty input.
    for bits in 1..=8 {
        Fallback.scramble(&[] as &[C32], &[], bits, false, &mut [])?;
        Sdrk.scramble(&[] as &[CQ15], &[], bits, true, &mut [])?;
    }

    // Every index 0 to 255 at bits 8, both directions: the facade and the fallback agree, the
    // reference within the contract, and the magnitude is kept.
    arena.reset();
    let mut c = arena.carver();
    let p = c.take::<u8>(256)?;
    for (i, v) in p.iter_mut().enumerate() {
        *v = i as u8;
    }
    let x = c.take::<C32>(256)?;
    let f = c.take::<C32>(256)?;
    let s = c.take::<C32>(256)?;
    let xq = c.take::<CQ15>(256)?;
    let fq = c.take::<CQ15>(256)?;
    let sq = c.take::<CQ15>(256)?;
    let mut rng = Pcg32::new(SEED, 7);
    for v in x.iter_mut() {
        *v = C32::random(&mut rng);
    }
    for v in xq.iter_mut() {
        *v = CQ15::random(&mut rng);
    }
    for inverse in [false, true] {
        Fallback.scramble(x, p, 8, inverse, f)?;
        Sdrk.scramble(x, p, 8, inverse, s)?;
        Fallback.scramble(xq, p, 8, inverse, fq)?;
        Sdrk.scramble(xq, p, 8, inverse, sq)?;
        ensure(
            bit_exact(f, s).is_none() && bit_exact(fq, sq).is_none(),
            || Failure::new(CASE, "bits 8: the facade differs from the fallback"),
        )?;
        for (k, (a, b)) in x.iter().zip(f.iter()).enumerate() {
            let d = libm::fabs(f64::from(a.norm_sqr()) - f64::from(b.norm_sqr()));
            ensure(d <= 1e-6, || {
                Failure::new(CASE, "bits 8: the magnitude changes")
                    .detail(format_args!("index {k}, inverse {inverse}"))
            })?;
        }
        Reference.scramble(xq, p, 8, inverse, sq)?;
        let (mut sig, mut err) = (0.0, 0.0);
        for (&a, &b) in sq.iter().zip(fq.iter()) {
            let (es, ee) = CQ15::energy(a, b);
            sig += es;
            err += ee;
        }
        ensure(err == 0.0 || 10.0 * libm::log10(sig / err) >= 80.0, || {
            Failure::new(CASE, "bits 8: CQ15 misses the contract SNR")
        })?;
    }

    // CQ15 corners: the saturating negation of the quadrant map, the saturation of the residual
    // product (the fallback against the reference: bit for bit at r = 0).
    let corners = [
        (i16::MIN, i16::MIN),
        (i16::MIN, i16::MAX),
        (i16::MAX, i16::MIN),
        (i16::MAX, i16::MAX),
    ];
    for bits in 1..=8u32 {
        for k in 0..(1u32 << bits) {
            let (q, r) = split_index(k, bits);
            for &(re, im) in &corners {
                let x = [CQ15 {
                    re: Q15(re),
                    im: Q15(im),
                }];
                let mut y = [CQ15::default()];
                let mut z = [CQ15::default()];
                Sdrk.scramble(&x, &[k as u8], bits, false, &mut y)?;
                Reference.scramble(&x, &[k as u8], bits, false, &mut z)?;
                if r == 0 {
                    ensure(y[0] == cq15_quadrant(x[0], q) && y == z, || {
                        Failure::new(CASE, "a CQ15 corner is not the saturating quadrant map")
                            .detail(format_args!("bits {bits}, k {k}"))
                    })?;
                } else {
                    let d = |a: Q15, b: Q15| (i32::from(a.0) - i32::from(b.0)).abs();
                    ensure(d(y[0].re, z[0].re) <= 1 && d(y[0].im, z[0].im) <= 1, || {
                        Failure::new(CASE, "a CQ15 corner differs from the reference")
                            .detail(format_args!("bits {bits}, k {k}"))
                    })?;
                }
            }
        }
    }

    // NaN and infinities in C32: ±∞ stays ±∞ where r = 0 (the exact quadrant map).
    let inf = f32::INFINITY;
    let specials = [
        C32::new(inf, 1.0),
        C32::new(-inf, -0.0),
        C32::new(0.0, inf),
        C32::new(inf, -inf),
        C32::new(f32::NAN, 2.0),
        C32::new(-0.0, 0.0),
    ];
    for bits in 1..=8u32 {
        for k in 0..(1u32 << bits) {
            let (q, r) = split_index(k, bits);
            for &x in &specials {
                let mut y = [C32::default()];
                Sdrk.scramble(&[x], &[k as u8], bits, false, &mut y)?;
                let ok = if r == 0 {
                    let w = c32_quadrant(x, q);
                    y[0].re.to_bits() == w.re.to_bits() && y[0].im.to_bits() == w.im.to_bits()
                } else if x.re.is_infinite() && x.im.is_finite() {
                    y[0].re.is_infinite() && y[0].im.is_infinite()
                } else if x.re.is_nan() {
                    y[0].re.is_nan() && y[0].im.is_nan()
                } else {
                    true
                };
                ensure(ok, || {
                    Failure::new(CASE, "a special C32 value is not mapped by the contract")
                        .detail(format_args!("bits {bits}, k {k}"))
                })?;
            }
        }
    }

    // Parameter and length errors write nothing.
    let mut y = [C32::new(5.0, 5.0); 2];
    for bits in [0, 9] {
        ensure(
            matches!(
                Sdrk.scramble(&[C32::default(); 2], &[0; 2], bits, false, &mut y),
                Err(Error::InvalidParameter(_))
            ),
            || Failure::new(CASE, "an invalid bits accepted").detail(format_args!("{bits}")),
        )?;
    }
    ensure(
        matches!(
            Sdrk.scramble(&[C32::default(); 2], &[0; 1], 2, false, &mut y),
            Err(Error::LengthMismatch { .. })
        ),
        || Failure::new(CASE, "a short phase accepted"),
    )?;
    ensure(y == [C32::new(5.0, 5.0); 2], || {
        Failure::new(CASE, "a refused call wrote the output")
    })?;
    Ok(())
}
