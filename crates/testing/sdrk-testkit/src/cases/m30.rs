//! Suite cases of M30: the kernels `add`, `add_real`, `mul`, `mul_real`, `oscillate`,
//! `interleave` and `deinterleave` and their extreme inputs.
//!
//! Every kernel case compares `Fallback` with `Reference`, the facade with `Fallback` and runs
//! the facade in 20 chunk plans; no files are read, so the cases run on microcontrollers too.
//! All kernels are bit-exact in both types except the `C32` oscillator, which is judged by its
//! SNR against the float64 `exp` (at least 120 dB, like `rotate`), and the facade's `C32`
//! product, which the SIMD backend fuses on AVX2, AVX-512 and NEON (D326: at least 140 dB
//! against the fallback, and the same bits whatever the block size). The block cases
//! `block_nco_phase_lock` and `block_make_complex_roundtrip` run the blocks of the milestone.

#![allow(
    clippy::cast_possible_truncation,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]

use num_complex::Complex64;
use sdrk::{Rate, Sdrk};
use sdrk_blocks::BlockSample;
use sdrk_blocks::{
    ArenaStorage, Block, ImagPartBlock, MakeComplexBlock, NcoBlock, NcoConfig, RealPartBlock,
};
use sdrk_core::kernels::math::{Add, AddReal, Interleave, Mul, MulReal};
use sdrk_core::kernels::mix::{Nco, Oscillate, Rotate};
use sdrk_core::{
    C32, CQ15, ComplexSample, Error, Fallback, Q15, Reference, Result as CoreResult, Sample,
};

use super::ensure;
use super::m05::{random_c32, random_cq15};
use crate::arena::Arena;
use crate::invariance::ChunkPlan;
use crate::metrics::{bit_exact, compare_complex};
use crate::rng::Pcg32;
use crate::suite::Failure;

/// Length of the element-wise signals.
const N: usize = 500;
/// Chunk plans per case.
const PLANS: u64 = 20;

type Binary<T> = fn(&[T], &[T], &mut [T]) -> CoreResult<()>;
/// Whether the facade's output (second) is close enough to the fallback's (first).
type Close<T> = fn(&[T], &[T]) -> bool;

/// The three implementations of an element-wise kernel.
struct Impls<T> {
    fallback: Binary<T>,
    reference: Binary<T>,
    facade: Binary<T>,
    /// How the facade's output (second) is compared with the fallback's (first); `None`: bit
    /// for bit.
    close: Option<Close<T>>,
}

fn random_q15(rng: &mut Pcg32) -> Q15 {
    Q15((rng.next_u32() >> 16) as i16)
}

fn random_f32(rng: &mut Pcg32) -> f32 {
    ((rng.next_f64() * 2.0 - 1.0) * 1.5) as f32
}

fn range<'a, X>(
    s: &'a [X],
    from: usize,
    to: usize,
    case: &'static str,
) -> Result<&'a [X], Failure> {
    s.get(from..to)
        .ok_or_else(|| Failure::new(case, "slice range"))
}

fn range_mut<'a, X>(
    s: &'a mut [X],
    from: usize,
    to: usize,
    case: &'static str,
) -> Result<&'a mut [X], Failure> {
    s.get_mut(from..to)
        .ok_or_else(|| Failure::new(case, "slice range"))
}

/// Fallback and Reference agree bit for bit, the facade equals the fallback (or is close to it,
/// `Impls::close`), unequal lengths are `LengthMismatch` and change nothing, and the facade does
/// not depend on the block size (20 plans).
fn binary_case<T: Sample>(
    arena: &mut Arena<'_>,
    case: &'static str,
    seed: u64,
    fill: fn(&mut Pcg32) -> T,
    impls: &Impls<T>,
) -> Result<(), Failure> {
    let mut c = arena.carver();
    let a = c.take::<T>(N)?;
    let b = c.take::<T>(N)?;
    let (fb, rf) = (c.take::<T>(N)?, c.take::<T>(N)?);
    let mut rng = Pcg32::new(seed, 1);
    for s in a.iter_mut().chain(b.iter_mut()) {
        *s = fill(&mut rng);
    }
    (impls.fallback)(a, b, fb)?;
    (impls.reference)(a, b, rf)?;
    ensure(bit_exact(fb, rf).is_none(), || {
        Failure::new(case, "reference differs from the fallback")
    })?;
    rf.fill(T::ZERO);
    (impls.facade)(a, b, rf)?;
    let close = impls
        .close
        .map_or_else(|| bit_exact(fb, rf).is_none(), |close| close(fb, rf));
    ensure(close, || {
        Failure::new(case, "facade differs from the fallback")
    })?;
    // From here on the facade is compared with its own output in one call.
    fb.copy_from_slice(rf);
    // Unequal lengths: an error, and the output stays as it was.
    for f in [impls.fallback, impls.reference, impls.facade] {
        let short_b = range(b, 0, N - 1, case)?;
        ensure(
            matches!(f(a, short_b, rf), Err(Error::LengthMismatch { .. })),
            || Failure::new(case, "a shorter second input was accepted"),
        )?;
        let short_out = range_mut(rf, 0, N - 1, case)?;
        ensure(
            matches!(f(a, b, short_out), Err(Error::LengthMismatch { .. })),
            || Failure::new(case, "a shorter output was accepted"),
        )?;
        ensure(bit_exact(fb, rf).is_none(), || {
            Failure::new(case, "a rejected call wrote to the output")
        })?;
    }
    for plan_seed in 0..PLANS {
        let plan = ChunkPlan::new(seed * 100 + plan_seed, 1 + (plan_seed as usize * 37) % 300);
        rf.fill(T::ZERO);
        let mut pos = 0;
        for size in plan.sizes() {
            if pos >= N {
                break;
            }
            let end = (pos + size).min(N);
            (impls.facade)(
                range(a, pos, end, case)?,
                range(b, pos, end, case)?,
                range_mut(rf, pos, end, case)?,
            )?;
            pos = end;
        }
        ensure(bit_exact(fb, rf).is_none(), || {
            Failure::new(case, "output depends on the block size")
        })?;
    }
    Ok(())
}

/// The SNR of `got` against `want` in dB.
fn snr_db_c32(want: &[C32], got: &[C32]) -> f64 {
    let (mut signal, mut noise) = (0.0f64, 0.0f64);
    for (w, g) in want.iter().zip(got) {
        let (wr, wi) = (f64::from(w.re), f64::from(w.im));
        let (er, ei) = (wr - f64::from(g.re), wi - f64::from(g.im));
        signal += wr * wr + wi * wi;
        noise += er * er + ei * ei;
    }
    if noise == 0.0 {
        f64::INFINITY
    } else {
        10.0 * libm::log10(signal / noise)
    }
}

pub(super) fn add_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    binary_case::<C32>(
        arena,
        "add_c32",
        4630,
        |r| random_c32(r, 1.5),
        &Impls {
            fallback: |a, b, o| Add::<C32>::add(&Fallback, a, b, o),
            reference: |a, b, o| Add::<C32>::add(&Reference, a, b, o),
            facade: |a, b, o| Add::<C32>::add(&Sdrk, a, b, o),
            close: None,
        },
    )
}

pub(super) fn add_cq15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    binary_case::<CQ15>(
        arena,
        "add_cq15",
        4631,
        |r| random_cq15(r, 1.0),
        &Impls {
            fallback: |a, b, o| Add::<CQ15>::add(&Fallback, a, b, o),
            reference: |a, b, o| Add::<CQ15>::add(&Reference, a, b, o),
            facade: |a, b, o| Add::<CQ15>::add(&Sdrk, a, b, o),
            close: None,
        },
    )
}

pub(super) fn mul_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    binary_case::<C32>(
        arena,
        "mul_c32",
        4650,
        |r| random_c32(r, 1.5),
        &Impls {
            fallback: |a, b, o| Mul::<C32>::mul(&Fallback, a, b, o),
            reference: |a, b, o| Mul::<C32>::mul(&Reference, a, b, o),
            facade: |a, b, o| Mul::<C32>::mul(&Sdrk, a, b, o),
            // The SIMD backend fuses the product on AVX2, AVX-512 and NEON (D326).
            close: Some(|fb, fac| snr_db_c32(fb, fac) >= 140.0),
        },
    )
}

pub(super) fn mul_cq15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    binary_case::<CQ15>(
        arena,
        "mul_cq15",
        4651,
        |r| random_cq15(r, 1.0),
        &Impls {
            fallback: |a, b, o| Mul::<CQ15>::mul(&Fallback, a, b, o),
            reference: |a, b, o| Mul::<CQ15>::mul(&Reference, a, b, o),
            facade: |a, b, o| Mul::<CQ15>::mul(&Sdrk, a, b, o),
            close: None,
        },
    )
}

pub(super) fn mul_real_q15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    binary_case::<Q15>(
        arena,
        "mul_real_q15",
        4661,
        random_q15,
        &Impls {
            fallback: |a, b, o| MulReal::<Q15>::mul_real(&Fallback, a, b, o),
            reference: |a, b, o| MulReal::<Q15>::mul_real(&Reference, a, b, o),
            facade: |a, b, o| MulReal::<Q15>::mul_real(&Sdrk, a, b, o),
            close: None,
        },
    )
}

// ---------------------------------------------------------------------------------------
// oscillate
// ---------------------------------------------------------------------------------------

/// Oscillator settings: increment, start phase, harmonic.
const OSC_SETTINGS: [(u32, u32, u32); 4] = [
    (0x1F2E_3D4C, 0, 1),
    (0xC000_0001, 0xFFFF_FF00, 2),
    (1 << 30, 77, 3),
    (0x0FED_CBA9, 0x1234_5678, 8),
];

fn nco_at(inc: u32, phase: u32) -> Nco {
    let mut n = Nco::from_increment(inc);
    n.set_phase(phase);
    n
}

/// Runs `f` over `out` in the chunks of `plan` with one oscillator.
fn oscillate_chunked<T: ComplexSample>(
    nco: &mut Nco,
    out: &mut [T],
    plan: ChunkPlan,
    case: &'static str,
    f: impl Fn(&mut Nco, &mut [T]) -> CoreResult<()>,
) -> Result<(), Failure> {
    let n = out.len();
    let mut pos = 0;
    for size in plan.sizes() {
        if pos >= n {
            break;
        }
        let end = (pos + size).min(n);
        f(nco, range_mut(out, pos, end, case)?)?;
        pos = end;
    }
    Ok(())
}

pub(super) fn oscillate_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "oscillate_c32";
    const M: usize = 1024;
    let mut c = arena.carver();
    let (fb, rf, fac, tmp) = (
        c.take::<C32>(M)?,
        c.take::<C32>(M)?,
        c.take::<C32>(M)?,
        c.take::<C32>(M)?,
    );
    let ones = c.take::<C32>(M)?;
    let want = c.take::<Complex64>(M)?;
    ones.fill(C32::new(1.0, 0.0));
    for (inc, phase, k) in OSC_SETTINGS {
        let (mut a, mut b, mut d) = (nco_at(inc, phase), nco_at(inc, phase), nco_at(inc, phase));
        Oscillate::<C32>::oscillate(&Fallback, &mut a, k, fb)?;
        Oscillate::<C32>::oscillate(&Reference, &mut b, k, rf)?;
        Oscillate::<C32>::oscillate(&Sdrk, &mut d, k, fac)?;
        ensure(a == b && a == d, || {
            Failure::new(CASE, "phase advanced differently")
        })?;
        for (w, r) in want.iter_mut().zip(rf.iter()) {
            *w = r.to_c64();
        }
        for (what, got) in [("fallback", &*fb), ("facade", &*fac)] {
            let cmp = compare_complex(want, got.iter().map(|s| s.to_c64()), 0);
            ensure(cmp.snr_db >= 120.0, || {
                Failure::new(CASE, "below 120 dB").detail(format_args!(
                    "{what}, inc {inc:#x} harmonic {k}: {} dB",
                    cmp.snr_db
                ))
            })?;
        }
        // The block size does not change the facade output.
        for plan_seed in 0..PLANS {
            let plan = ChunkPlan::new(4670 + plan_seed, 1 + (plan_seed as usize * 37) % 300);
            let mut n = nco_at(inc, phase);
            oscillate_chunked(&mut n, tmp, plan, CASE, |nco, o| {
                Oscillate::<C32>::oscillate(&Sdrk, nco, k, o)
            })?;
            ensure(bit_exact(tmp, fac).is_none() && n == d, || {
                Failure::new(CASE, "output depends on the block size")
            })?;
        }
    }
    // Harmonic 1 equals the mixer applied to the constant (1, 0), bit for bit.
    let (mut a, mut b) = (nco_at(0x0123_4567, 99), nco_at(0x0123_4567, 99));
    Oscillate::<C32>::oscillate(&Fallback, &mut a, 1, fb)?;
    Rotate::<C32>::rotate(&Fallback, &mut b, ones, rf)?;
    ensure(bit_exact(fb, rf).is_none() && a == b, || {
        Failure::new(CASE, "harmonic 1 differs from rotate of (1, 0)")
    })?;
    // A harmonic outside 1 .. 8 is rejected and leaves the oscillator alone.
    for k in [0u32, 9, u32::MAX] {
        let mut n = nco_at(5, 6);
        let before = n;
        ensure(
            matches!(
                Oscillate::<C32>::oscillate(&Sdrk, &mut n, k, tmp),
                Err(Error::InvalidParameter(_))
            ) && n == before,
            || Failure::new(CASE, "an invalid harmonic was accepted"),
        )?;
    }
    Ok(())
}

pub(super) fn oscillate_cq15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "oscillate_cq15";
    const M: usize = 1024;
    let mut c = arena.carver();
    let (fb, rf, fac, tmp) = (
        c.take::<CQ15>(M)?,
        c.take::<CQ15>(M)?,
        c.take::<CQ15>(M)?,
        c.take::<CQ15>(M)?,
    );
    for (inc, phase, k) in OSC_SETTINGS {
        let (mut a, mut b, mut d) = (nco_at(inc, phase), nco_at(inc, phase), nco_at(inc, phase));
        Oscillate::<CQ15>::oscillate(&Fallback, &mut a, k, fb)?;
        Oscillate::<CQ15>::oscillate(&Reference, &mut b, k, rf)?;
        Oscillate::<CQ15>::oscillate(&Sdrk, &mut d, k, fac)?;
        ensure(a == b && a == d, || {
            Failure::new(CASE, "phase advanced differently")
        })?;
        ensure(bit_exact(fb, rf).is_none(), || {
            Failure::new(CASE, "reference differs from the fallback")
        })?;
        ensure(bit_exact(fb, fac).is_none(), || {
            Failure::new(CASE, "facade differs from the fallback")
        })?;
        for plan_seed in 0..PLANS {
            let plan = ChunkPlan::new(4680 + plan_seed, 1 + (plan_seed as usize * 37) % 300);
            let mut n = nco_at(inc, phase);
            oscillate_chunked(&mut n, tmp, plan, CASE, |nco, o| {
                Oscillate::<CQ15>::oscillate(&Sdrk, nco, k, o)
            })?;
            ensure(bit_exact(tmp, fb).is_none() && n == d, || {
                Failure::new(CASE, "output depends on the block size")
            })?;
        }
    }
    for k in [0u32, 9] {
        let mut n = nco_at(5, 6);
        let before = n;
        ensure(
            matches!(
                Oscillate::<CQ15>::oscillate(&Sdrk, &mut n, k, tmp),
                Err(Error::InvalidParameter(_))
            ) && n == before,
            || Failure::new(CASE, "an invalid harmonic was accepted"),
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// interleave
// ---------------------------------------------------------------------------------------

/// The three implementations agree bit for bit on `interleave` (with and without `im`) and
/// `deinterleave` (both outputs, either one), the round trip returns the inputs, and wrong
/// lengths are `LengthMismatch`.
fn interleave_case<T, K1, K2, K3>(
    arena: &mut Arena<'_>,
    case: &'static str,
    kernels: (&K1, &K2, &K3),
    fill: fn(&mut Pcg32) -> T::Real,
) -> Result<(), Failure>
where
    T: ComplexSample,
    K1: Interleave<T>,
    K2: Interleave<T>,
    K3: Interleave<T>,
{
    const M: usize = 300;
    let (kf, kr, ks) = kernels;
    let mut c = arena.carver();
    let re = c.take::<T::Real>(M)?;
    let im = c.take::<T::Real>(M)?;
    let (o1, o2, o3) = (c.take::<T>(M)?, c.take::<T>(M)?, c.take::<T>(M)?);
    let (r1, i1) = (c.take::<T::Real>(M)?, c.take::<T::Real>(M)?);
    let mut rng = Pcg32::new(4690, 1);
    for s in re.iter_mut().chain(im.iter_mut()) {
        *s = fill(&mut rng);
    }
    let fail = |msg: &'static str| Failure::new(case, msg);
    for with_im in [true, false] {
        let imp = with_im.then_some(&*im);
        kf.interleave(re, imp, o1)?;
        kr.interleave(re, imp, o2)?;
        ks.interleave(re, imp, o3)?;
        ensure(
            bit_exact(o1, o2).is_none() && bit_exact(o1, o3).is_none(),
            || fail("interleave differs between implementations"),
        )?;
        let parts_ok = o1.iter().zip(re.iter().zip(im.iter())).all(|(&o, (r, i))| {
            bytemuck::bytes_of(&o.re()) == bytemuck::bytes_of(r)
                && (!with_im || bytemuck::bytes_of(&o.im()) == bytemuck::bytes_of(i))
                && (with_im || bytemuck::bytes_of(&o.im()) == bytemuck::bytes_of(&T::Real::ZERO))
        });
        ensure(parts_ok, || fail("interleave placed a component wrongly"))?;
    }
    kf.interleave(re, Some(im), o1)?;
    for k in 0..3 {
        r1.fill(T::Real::ZERO);
        i1.fill(T::Real::ZERO);
        match k {
            0 => kf.deinterleave(o1, Some(&mut *r1), Some(&mut *i1))?,
            1 => kr.deinterleave(o1, Some(&mut *r1), Some(&mut *i1))?,
            _ => ks.deinterleave(o1, Some(&mut *r1), Some(&mut *i1))?,
        }
        ensure(
            bit_exact(r1, re).is_none() && bit_exact(i1, im).is_none(),
            || fail("deinterleave did not return the inputs"),
        )?;
    }
    // One output only.
    r1.fill(T::Real::ZERO);
    ks.deinterleave(o1, Some(&mut *r1), None)?;
    ensure(bit_exact(r1, re).is_none(), || {
        fail("deinterleave of the real part alone differs")
    })?;
    i1.fill(T::Real::ZERO);
    ks.deinterleave(o1, None, Some(&mut *i1))?;
    ensure(bit_exact(i1, im).is_none(), || {
        fail("deinterleave of the imaginary part alone differs")
    })?;
    ks.deinterleave(o1, None, None)?;
    // Length errors change nothing.
    kf.interleave(re, Some(im), o2)?;
    let short = range(im, 0, M - 1, case)?;
    ensure(
        matches!(
            ks.interleave(re, Some(short), o2),
            Err(Error::LengthMismatch { .. })
        ),
        || fail("a shorter im was accepted"),
    )?;
    let out_short = range_mut(o2, 0, M - 1, case)?;
    ensure(
        matches!(
            ks.interleave(re, Some(im), out_short),
            Err(Error::LengthMismatch { .. })
        ),
        || fail("a shorter output was accepted"),
    )?;
    let r_short = range_mut(r1, 0, M - 1, case)?;
    ensure(
        matches!(
            ks.deinterleave(o1, Some(r_short), None),
            Err(Error::LengthMismatch { .. })
        ),
        || fail("a shorter re output was accepted"),
    )?;
    ensure(bit_exact(o2, o1).is_none(), || {
        fail("a rejected call wrote to the output")
    })
}

pub(super) fn interleave_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    interleave_case::<C32, _, _, _>(
        arena,
        "interleave_c32",
        (&Fallback, &Reference, &Sdrk),
        random_f32,
    )
}

pub(super) fn interleave_cq15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    interleave_case::<CQ15, _, _, _>(
        arena,
        "interleave_cq15",
        (&Fallback, &Reference, &Sdrk),
        random_q15,
    )
}

// ---------------------------------------------------------------------------------------
// extremes
// ---------------------------------------------------------------------------------------

/// The f32 specials (`0`, `-0`, NaN, `+-inf`, a denormal, `f32::MAX`, one).
const SPECIALS: [f32; 8] = [
    0.0,
    -0.0,
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::MIN_POSITIVE / 2.0,
    f32::MAX,
    1.0,
];

/// The Q15 corner values.
const CORNERS: [i16; 7] = [-32768, -32767, -1, 0, 1, 32766, 32767];

fn special(i: usize) -> f32 {
    SPECIALS.get(i % SPECIALS.len()).copied().unwrap_or(0.0)
}

fn corner(i: usize) -> Q15 {
    Q15(CORNERS.get(i % CORNERS.len()).copied().unwrap_or(0))
}

/// Bit equality, or both NaN (which payload an operation on two NaNs returns is a property of
/// the instruction, see D157).
fn same_or_nan(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn same_c32_or_nan(a: &[C32], b: &[C32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| same_or_nan(x.re, y.re) && same_or_nan(x.im, y.im))
}

fn same_f32_or_nan(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same_or_nan(*x, *y))
}

pub(super) fn extremes_m30(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_m30";
    const P: usize = 64;
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let mut c = arena.carver();

    // f32 and C32: every pair of specials. Fallback and Reference agree bit for bit (NaN with
    // NaN), the facade up to the payload of a NaN.
    let (xa, xb) = (c.take::<f32>(P)?, c.take::<f32>(P)?);
    let (f1, f2, f3) = (c.take::<f32>(P)?, c.take::<f32>(P)?, c.take::<f32>(P)?);
    for (k, (a, b)) in xa.iter_mut().zip(xb.iter_mut()).enumerate() {
        *a = special(k / 8);
        *b = special(k);
    }
    AddReal::<f32>::add_real(&Fallback, xa, xb, f1)?;
    AddReal::<f32>::add_real(&Reference, xa, xb, f2)?;
    AddReal::<f32>::add_real(&Sdrk, xa, xb, f3)?;
    ensure(
        bit_exact(f1, f2).is_none() && same_f32_or_nan(f1, f3),
        || fail("add_real specials differ between implementations"),
    )?;
    MulReal::<f32>::mul_real(&Fallback, xa, xb, f1)?;
    MulReal::<f32>::mul_real(&Reference, xa, xb, f2)?;
    MulReal::<f32>::mul_real(&Sdrk, xa, xb, f3)?;
    ensure(
        bit_exact(f1, f2).is_none() && same_f32_or_nan(f1, f3),
        || fail("mul_real specials differ between implementations"),
    )?;
    let (ca, cb) = (c.take::<C32>(P)?, c.take::<C32>(P)?);
    let (g1, g2, g3) = (c.take::<C32>(P)?, c.take::<C32>(P)?, c.take::<C32>(P)?);
    for (k, (a, b)) in ca.iter_mut().zip(cb.iter_mut()).enumerate() {
        *a = C32::new(special(k), special(k / 8 + 3));
        *b = C32::new(special(k / 8), special(k + 5));
    }
    Add::<C32>::add(&Fallback, ca, cb, g1)?;
    Add::<C32>::add(&Reference, ca, cb, g2)?;
    Add::<C32>::add(&Sdrk, ca, cb, g3)?;
    ensure(
        bit_exact(g1, g2).is_none() && same_c32_or_nan(g1, g3),
        || fail("add C32 specials differ between implementations"),
    )?;
    Mul::<C32>::mul(&Fallback, ca, cb, g1)?;
    Mul::<C32>::mul(&Reference, ca, cb, g2)?;
    Mul::<C32>::mul(&Sdrk, ca, cb, g3)?;
    ensure(
        bit_exact(g1, g2).is_none() && same_c32_or_nan(g1, g3),
        || fail("mul C32 specials differ between implementations"),
    )?;

    // Silence in, silence out (positive zero).
    ca.fill(C32::ZERO);
    cb.fill(C32::ZERO);
    for sum in [true, false] {
        if sum {
            Add::<C32>::add(&Sdrk, ca, cb, g3)?;
        } else {
            Mul::<C32>::mul(&Sdrk, ca, cb, g3)?;
        }
        ensure(
            g3.iter()
                .all(|s| s.re.to_bits() == 0 && s.im.to_bits() == 0),
            || fail("silence did not give a zero output"),
        )?;
    }

    // Q15: every pair of corner values, real and complex.
    const Q: usize = 49;
    let (qa, qb) = (c.take::<Q15>(Q)?, c.take::<Q15>(Q)?);
    let (q1, q2, q3) = (c.take::<Q15>(Q)?, c.take::<Q15>(Q)?, c.take::<Q15>(Q)?);
    for (k, (a, b)) in qa.iter_mut().zip(qb.iter_mut()).enumerate() {
        *a = corner(k / 7);
        *b = corner(k);
    }
    AddReal::<Q15>::add_real(&Fallback, qa, qb, q1)?;
    AddReal::<Q15>::add_real(&Reference, qa, qb, q2)?;
    AddReal::<Q15>::add_real(&Sdrk, qa, qb, q3)?;
    ensure(
        bit_exact(q1, q2).is_none() && bit_exact(q1, q3).is_none(),
        || fail("add_real corners differ between implementations"),
    )?;
    let all_sat = qa.iter().zip(qb.iter()).zip(q1.iter()).all(|((a, b), s)| {
        i32::from(s.0) == (i32::from(a.0) + i32::from(b.0)).clamp(-32768, 32767)
    });
    ensure(all_sat, || fail("add_real is not the saturated sum"))?;
    MulReal::<Q15>::mul_real(&Fallback, qa, qb, q1)?;
    MulReal::<Q15>::mul_real(&Reference, qa, qb, q2)?;
    MulReal::<Q15>::mul_real(&Sdrk, qa, qb, q3)?;
    ensure(
        bit_exact(q1, q2).is_none() && bit_exact(q1, q3).is_none(),
        || fail("mul_real corners differ between implementations"),
    )?;
    let hit = qa
        .iter()
        .zip(qb.iter())
        .zip(q1.iter())
        .find(|((a, b), _)| a.0 == -32768 && b.0 == -32768);
    ensure(hit.is_some_and(|(_, s)| s.0 == 32767), || {
        fail("-1 * -1 is not 32767")
    })?;
    let (ia, ib) = (c.take::<CQ15>(Q)?, c.take::<CQ15>(Q)?);
    let (p1, p2, p3) = (c.take::<CQ15>(Q)?, c.take::<CQ15>(Q)?, c.take::<CQ15>(Q)?);
    for (k, (a, b)) in ia.iter_mut().zip(ib.iter_mut()).enumerate() {
        *a = CQ15 {
            re: corner(k / 7),
            im: corner(k),
        };
        *b = CQ15 {
            re: corner(k),
            im: corner(k / 7 + 3),
        };
    }
    for (name, sum) in [("add", true), ("mul", false)] {
        if sum {
            Add::<CQ15>::add(&Fallback, ia, ib, p1)?;
            Add::<CQ15>::add(&Reference, ia, ib, p2)?;
            Add::<CQ15>::add(&Sdrk, ia, ib, p3)?;
        } else {
            Mul::<CQ15>::mul(&Fallback, ia, ib, p1)?;
            Mul::<CQ15>::mul(&Reference, ia, ib, p2)?;
            Mul::<CQ15>::mul(&Sdrk, ia, ib, p3)?;
        }
        ensure(
            bit_exact(p1, p2).is_none() && bit_exact(p1, p3).is_none(),
            || {
                Failure::new(CASE, "Q15 corners differ between implementations")
                    .detail(format_args!("{name}"))
            },
        )?;
    }
    // (-1 - j) * (-1 - j) = (1 - 1) + j(1 + 1): re 0, im saturated to 32767.
    let one = CQ15 {
        re: Q15(-32768),
        im: Q15(-32768),
    };
    let mut out = [CQ15::ZERO];
    Mul::<CQ15>::mul(&Sdrk, &[one], &[one], &mut out)?;
    ensure(
        out.first().is_some_and(|o| o.re.0 == 0 && o.im.0 == 32767),
        || fail("complex corner product wrong"),
    )?;

    // Oscillator at the extremes of the increment and the phase, harmonic 8.
    let (o1, o2) = (c.take::<CQ15>(P)?, c.take::<CQ15>(P)?);
    let (e1, e2) = (c.take::<C32>(P)?, c.take::<C32>(P)?);
    for (inc, phase) in [
        (0u32, 0u32),
        (u32::MAX, u32::MAX),
        (1 << 31, 1 << 31),
        (0x8000_0001, 0xFFFF_FFFF),
    ] {
        let (mut a, mut b) = (nco_at(inc, phase), nco_at(inc, phase));
        Oscillate::<CQ15>::oscillate(&Fallback, &mut a, 8, o1)?;
        Oscillate::<CQ15>::oscillate(&Reference, &mut b, 8, o2)?;
        ensure(bit_exact(o1, o2).is_none() && a == b, || {
            fail("CQ15 oscillator corners differ between implementations")
        })?;
        let (mut a, mut b) = (nco_at(inc, phase), nco_at(inc, phase));
        Oscillate::<C32>::oscillate(&Fallback, &mut a, 8, e1)?;
        Oscillate::<C32>::oscillate(&Reference, &mut b, 8, e2)?;
        let bounded = e1.iter().all(|s| {
            s.re.is_finite() && s.im.is_finite() && s.re.abs() <= 1.0001 && s.im.abs() <= 1.0001
        });
        ensure(bounded && a == b, || {
            fail("C32 oscillator left the unit square")
        })?;
    }

    // interleave copies NaN and infinity bit for bit; the zero-length call is fine.
    let (re, im) = (c.take::<f32>(P)?, c.take::<f32>(P)?);
    for (k, (r, i)) in re.iter_mut().zip(im.iter_mut()).enumerate() {
        *r = special(k);
        *i = special(k / 3 + 2);
    }
    Interleave::<C32>::interleave(&Sdrk, re, Some(im), g1)?;
    let copied = g1
        .iter()
        .zip(re.iter().zip(im.iter()))
        .all(|(o, (r, i))| o.re.to_bits() == r.to_bits() && o.im.to_bits() == i.to_bits());
    ensure(copied, || fail("interleave changed a special value"))?;
    Interleave::<C32>::interleave(&Sdrk, &[], None, &mut [])?;
    Interleave::<C32>::deinterleave(&Sdrk, &[], None, None)?;
    Add::<C32>::add(&Sdrk, &[], &[], &mut [])?;
    Mul::<CQ15>::mul(&Sdrk, &[], &[], &mut [])?;
    Ok(())
}

/// Samples of `block_nco_phase_lock`.
const LOCK_SAMPLES: usize = 1_000_000;
/// Samples per call of the block cases.
const BLOCK_CHUNK: usize = 256;

/// Two `nco` blocks on the same clock at 19 kHz (192 kHz rate), harmonic 1 and harmonic 2: the
/// second equals the square of the first within 1e-6 in every sample, over 10^6 samples.
pub(super) fn block_nco_phase_lock(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "block_nco_phase_lock";
    let fs = Rate::hz(192_000)?;
    let one = NcoConfig {
        frequency_hz: 19_000.0,
        ..NcoConfig::default()
    };
    let two = NcoConfig { harmonic: 2, ..one };
    let mut storage = ArenaStorage::new(&mut []);
    let mut first = NcoBlock::<C32, _>::new(&one, fs, &mut storage)?;
    let mut second = NcoBlock::<C32, _>::new(&two, fs, &mut storage)?;
    let mut c = arena.carver();
    // The clock: its values are ignored.
    let clock = c.take::<C32>(BLOCK_CHUNK)?;
    let (o1, o2) = (c.take::<C32>(BLOCK_CHUNK)?, c.take::<C32>(BLOCK_CHUNK)?);
    let mut done = 0;
    let mut worst = 0.0f64;
    while done < LOCK_SAMPLES {
        let n = BLOCK_CHUNK.min(LOCK_SAMPLES - done);
        let clk = range(clock, 0, n, CASE)?;
        let a = first.process(clk, range_mut(o1, 0, n, CASE)?)?;
        let b = second.process(clk, range_mut(o2, 0, n, CASE)?)?;
        ensure((a.produced, b.produced) == (n, n), || {
            Failure::new(CASE, "a block did not produce a sample per clock sample")
        })?;
        for (x, y) in o1.iter().zip(o2.iter()).take(n) {
            let (xr, xi) = (f64::from(x.re), f64::from(x.im));
            let re = xr * xr - xi * xi;
            let im = 2.0 * xr * xi;
            let err = (re - f64::from(y.re))
                .abs()
                .max((im - f64::from(y.im)).abs());
            worst = worst.max(err);
        }
        done += n;
    }
    ensure(worst <= 1e-6, || {
        Failure::new(CASE, "harmonic 2 differs from the square of harmonic 1")
            .detail(format_args!("largest error {worst:e}"))
    })
}

fn make_complex_roundtrip<T: BlockSample>(
    arena: &mut Arena<'_>,
    fill: fn(&mut Pcg32, usize) -> T::Real,
) -> Result<(), Failure>
where
    Sdrk: Interleave<T>,
{
    const CASE: &str = "block_make_complex_roundtrip";
    const N: usize = 600;
    let fs = Rate::hz(48_000)?;
    let mut storage = ArenaStorage::new(&mut []);
    let mut make = MakeComplexBlock::<T, _>::new(fs, &mut storage)?;
    let mut real = RealPartBlock::<T, _>::new(fs, &mut storage)?;
    let mut imag = ImagPartBlock::<T, _>::new(fs, &mut storage)?;
    let mut c = arena.carver();
    let (re, im) = (c.take::<T::Real>(N)?, c.take::<T::Real>(N)?);
    let (re_out, im_out) = (c.take::<T::Real>(N)?, c.take::<T::Real>(N)?);
    let joined = c.take::<T>(N)?;
    let mut rng = Pcg32::new(4680, 3);
    for (k, (a, b)) in re.iter_mut().zip(im.iter_mut()).enumerate() {
        *a = fill(&mut rng, k);
        *b = fill(&mut rng, k + 1);
    }
    let mut pos = 0;
    let plan = ChunkPlan::new(4681, 97);
    for size in plan.sizes() {
        if pos >= N {
            break;
        }
        let end = (pos + size).min(N);
        let n = end - pos;
        let p = make.process(
            (range(re, pos, end, CASE)?, range(im, pos, end, CASE)?),
            range_mut(joined, pos, end, CASE)?,
        )?;
        ensure(p.consumed == n && p.produced == n, || {
            Failure::new(CASE, "make_complex did not take the whole piece")
        })?;
        real.process(
            range(joined, pos, end, CASE)?,
            range_mut(re_out, pos, end, CASE)?,
        )?;
        imag.process(
            range(joined, pos, end, CASE)?,
            range_mut(im_out, pos, end, CASE)?,
        )?;
        pos = end;
    }
    ensure(
        bit_exact(re, re_out).is_none() && bit_exact(im, im_out).is_none(),
        || Failure::new(CASE, "the parts differ from the inputs"),
    )
}

/// `make_complex` followed by `real_part` and `imag_part` returns the inputs bit for bit, for
/// `f32` (with NaN, infinities and negative zero) and `Q15`, in pieces of random size.
pub(super) fn block_make_complex_roundtrip(arena: &mut Arena<'_>) -> Result<(), Failure> {
    make_complex_roundtrip::<C32>(arena, |rng, k| match k % 50 {
        0 => f32::NAN,
        1 => f32::INFINITY,
        2 => f32::NEG_INFINITY,
        3 => -0.0,
        _ => random_f32(rng),
    })?;
    make_complex_roundtrip::<CQ15>(arena, |rng, k| match k % 50 {
        0 => Q15(i16::MIN),
        1 => Q15(i16::MAX),
        _ => random_q15(rng),
    })
}
