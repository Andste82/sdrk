//! Suite cases of M40: the kernel `costas` (with order 8 of M40.1) and its extreme inputs.
//!
//! The kernel is bit-exact (the float32 model) and not accelerated (the facade is the
//! fallback). The kernel case compares `Fallback` with `Reference` (output and final state) and
//! the facade with `Fallback`, and runs the facade in 20 chunk plans (seed 559, chunks of 1 to
//! 300 symbols; seed 564 for the order-8 configurations of M40.1); every run must give the
//! outputs and the final state of the run in one call. The signals are BPSK, QPSK and 8PSK
//! symbols from `Pcg32`, one sample per symbol, turned by a start phase and a carrier offset,
//! plus noise; no files are read, so the cases run on microcontrollers too.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

use sdrk::Sdrk;
use sdrk_core::kernels::costas::{Costas, CostasState};
use sdrk_core::{C32, Fallback, Reference};

use super::ensure;
use crate::arena::Arena;
use crate::invariance::ChunkPlan;
use crate::rng::Pcg32;
use crate::suite::Failure;

/// Symbols per signal of the kernel case.
const N: usize = 2048;
/// Chunk plans per configuration.
const PLANS: u64 = 20;
/// The seed of the chunk plans (test 6 of M40); plan `k` of configuration `j` uses
/// `PLAN_SEED * 1000 + 100 j + k`.
const PLAN_SEED: u64 = 559;
/// The seed of the chunk plans and the signals of order 8 (test 5 of M40.1, D646).
const PLAN_SEED_8PSK: u64 = 564;
/// Largest chunk of the plans.
const MAX_CHUNK: usize = 300;
/// The loop parameters of the kernel case: the block defaults with a wider loop.
const LOOP_BW: f64 = 0.03;
#[allow(
    clippy::approx_constant,
    reason = "the damping of the block defaults, written as in the spec"
)]
const DAMPING: f64 = 0.7071;
const MAX_OFFSET: f64 = 0.05;

/// Bit equality; NaN equals NaN.
fn same(a: &[C32], b: &[C32]) -> bool {
    let eq = |p: f32, q: f32| p.to_bits() == q.to_bits() || (p.is_nan() && q.is_nan());
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| eq(x.re, y.re) && eq(x.im, y.im))
}

fn get<X>(s: &[X], from: usize, to: usize) -> Result<&[X], Failure> {
    s.get(from..to)
        .ok_or_else(|| Failure::new("m40", "slice range"))
}

fn get_mut<X>(s: &mut [X], from: usize, to: usize) -> Result<&mut [X], Failure> {
    s.get_mut(from..to)
        .ok_or_else(|| Failure::new("m40", "slice range"))
}

/// Symbols of `order` (BPSK `+-1`, QPSK `(+-1 +-j) / sqrt 2`, 8PSK `exp(j (2m + 1) pi / 8)`)
/// turned by `phase0 + 2 pi offset k` (radians, cycles per symbol), plus normal noise of `sigma`
/// per component, from the PCG32 stream `stream` of the seed 559 (564 for order 8).
fn psk_symbols(x: &mut [C32], order: u32, phase0: f64, offset: f64, sigma: f64, stream: u64) {
    let seed = if order == 8 {
        PLAN_SEED_8PSK
    } else {
        PLAN_SEED
    };
    let mut rng = Pcg32::new(seed, stream);
    let r = core::f64::consts::FRAC_1_SQRT_2;
    for (k, v) in x.iter_mut().enumerate() {
        let b = rng.next_u32();
        let (re, im) = match order {
            2 => (if b & 1 == 1 { 1.0 } else { -1.0 }, 0.0),
            4 => (
                if b & 1 == 1 { r } else { -r },
                if b & 2 == 2 { r } else { -r },
            ),
            _ => {
                let a = f64::from(2 * (b & 7) + 1) * core::f64::consts::PI / 8.0;
                (libm::cos(a), libm::sin(a))
            }
        };
        let turns = phase0 / core::f64::consts::TAU + offset * k as f64;
        let a = core::f64::consts::TAU * (turns - libm::floor(turns));
        let (s, c) = (libm::sin(a), libm::cos(a));
        *v = C32::new(
            (re * c - im * s + sigma * rng.normal()) as f32,
            (re * s + im * c + sigma * rng.normal()) as f32,
        );
    }
}

/// One configuration: Fallback against Reference (outputs and final state), the facade against
/// Fallback, the facade in 20 chunk plans. Returns the final state.
fn compare_all(
    case: &'static str,
    x: &[C32],
    out: &mut [C32],
    other: &mut [C32],
    order: u32,
    plan_base: u64,
) -> Result<CostasState, Failure> {
    let fail = |msg: &'static str| Failure::new(case, msg);
    let new = || CostasState::new(order, LOOP_BW, DAMPING, MAX_OFFSET);
    let mut fb = new()?;
    Fallback.costas(&mut fb, x, out)?;
    let mut rf = new()?;
    Reference.costas(&mut rf, x, other)?;
    ensure(same(out, other) && fb == rf, || {
        fail("reference differs from the fallback").detail(format_args!("order {order}"))
    })?;
    let mut fc = new()?;
    Sdrk.costas(&mut fc, x, other)?;
    ensure(same(out, other) && fc == fb, || {
        fail("facade differs from the fallback")
    })?;
    for k in 0..PLANS {
        let mut st = new()?;
        let mut sizes = ChunkPlan::new(plan_base + k, MAX_CHUNK).sizes();
        let mut pos = 0usize;
        while pos < x.len() {
            let a = sizes.next().unwrap_or(1).min(x.len() - pos);
            Sdrk.costas(
                &mut st,
                get(x, pos, pos + a)?,
                get_mut(other, pos, pos + a)?,
            )?;
            pos += a;
        }
        ensure(same(out, other) && st == fb, || {
            fail("output depends on the block size").detail(format_args!("order {order}, plan {k}"))
        })?;
    }
    Ok(fb)
}

/// `C32`: BPSK, QPSK and 8PSK with start phases, offsets and noise; the loop must lock.
pub(super) fn costas_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "costas_c32";
    let mut c = arena.carver();
    let x = c.take::<C32>(N)?;
    let out = c.take::<C32>(N)?;
    let other = c.take::<C32>(N)?;
    let configs = [
        (2u32, 0.7, 0.0, 0.1),
        (2, -2.0, 0.005, 0.2),
        (4, 1.3, 0.0, 0.1),
        (4, 2.5, -0.01, 0.1),
        (8, -0.4, 0.0, 0.05),
        (8, 2.2, 0.006, 0.05),
    ];
    for (j, (order, phase0, offset, sigma)) in configs.into_iter().enumerate() {
        psk_symbols(x, order, phase0, offset, sigma, 4000 + j as u64);
        let seed = if order == 8 {
            PLAN_SEED_8PSK
        } else {
            PLAN_SEED
        };
        let st = compare_all(CASE, x, out, other, order, seed * 1000 + 100 * j as u64)?;
        ensure(
            st.locked() && libm::fabs(st.frequency() - offset) < 0.002,
            || {
                Failure::new(CASE, "the loop did not lock").detail(format_args!(
                    "order {order}, offset {offset}, frequency {}",
                    st.frequency()
                ))
            },
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// Extreme inputs
// ---------------------------------------------------------------------------------------

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

fn special(i: usize) -> f32 {
    SPECIALS.get(i % SPECIALS.len()).copied().unwrap_or(0.0)
}

/// Runs Fallback and Reference on `x` and requires equal outputs (NaN equals NaN), equal final
/// states, a frequency within its limit and a finite lock level in `[-1, 1]`.
fn agree(
    case: &'static str,
    msg: &'static str,
    x: &[C32],
    o1: &mut [C32],
    o2: &mut [C32],
    st: CostasState,
) -> Result<CostasState, Failure> {
    let (mut a, mut b) = (st, st);
    Fallback.costas(&mut a, x, o1)?;
    Reference.costas(&mut b, x, o2)?;
    ensure(same(o1, o2) && a == b, || {
        Failure::new(case, msg).detail(format_args!(
            "order {}: the implementations differ",
            st.order()
        ))
    })?;
    let l = a.lock_level();
    ensure(
        libm::fabs(a.frequency()) <= a.max_offset() + 1e-9
            && l.is_finite()
            && libm::fabsf(l) <= 1.0,
        || Failure::new(case, msg).detail(format_args!("frequency {}, level {l}", a.frequency())),
    )?;
    Ok(a)
}

/// Silence, full scale, values beyond full scale, NaN and infinities, the parameter bounds and
/// `max_offset` 0, for every order.
pub(super) fn extremes_m40(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_m40";
    const M: usize = 600;
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let mut c = arena.carver();
    let x = c.take::<C32>(M)?;
    let o1 = c.take::<C32>(M)?;
    let o2 = c.take::<C32>(M)?;

    for order in [2u32, 4, 8] {
        let st = CostasState::new(order, LOOP_BW, DAMPING, MAX_OFFSET)?;

        // Silence: zeros out, the loop does not move and stays unlocked.
        x.fill(C32::new(0.0, 0.0));
        let end = agree(CASE, "silence", x, o1, o2, st)?;
        ensure(
            o1.iter().all(|v| v.re == 0.0 && v.im == 0.0)
                && end.phase() == 0
                && end.frequency() == 0.0
                && !end.locked(),
            || fail("silence did not give zeros with a resting loop"),
        )?;

        // Full scale: the symbols of the order at amplitude 1 (with the largest component 1).
        psk_symbols(x, order, 0.9, 0.002, 0.0, 4010);
        let end = agree(CASE, "full scale", x, o1, o2, st)?;
        ensure(end.locked(), || fail("full scale did not lock"))?;

        // Far beyond full scale: the products overflow to infinities and NaN; a non-finite
        // error counts as 0 and a non-finite lock metric is skipped (D603).
        for v in x.iter_mut() {
            *v = C32::new(v.re * f32::MAX, v.im * f32::MAX);
        }
        agree(CASE, "f32::MAX", x, o1, o2, st)?;

        // NaN, infinities, a denormal and f32::MAX in every phase of either component.
        for offset in 0..SPECIALS.len() {
            for (i, v) in x.iter_mut().enumerate() {
                *v = C32::new(special(i / 3 + offset), special(i / 5 + offset + 3));
            }
            agree(CASE, "non-finite", x, o1, o2, st)?;
        }

        // A burst of NaN in a clean signal: the loop keeps its state, the outputs are finite
        // again after it and the lock holds.
        psk_symbols(x, order, 0.4, 0.0, 0.05, 4020);
        for v in x.get_mut(300..310).unwrap_or(&mut []) {
            *v = C32::new(f32::NAN, f32::NAN);
        }
        let end = agree(CASE, "NaN burst", x, o1, o2, st)?;
        ensure(
            get(o1, 310, M)?
                .iter()
                .all(|v| v.re.is_finite() && v.im.is_finite())
                && end.locked(),
            || fail("the outputs after a NaN burst are not finite or the lock is lost"),
        )?;

        // The parameter bounds and max_offset 0 (the integrator never moves).
        psk_symbols(x, order, 1.1, 0.03, 0.1, 4030);
        for (bw, d, m) in [(0.0001, 0.1, 0.0), (0.2, 4.0, 0.25), (0.2, 0.1, 0.25)] {
            let s = CostasState::new(order, bw, d, m)?;
            let end = agree(CASE, "parameter bounds", x, o1, o2, s)?;
            ensure(m != 0.0 || end.frequency() == 0.0, || {
                fail("max_offset 0: the frequency moved")
            })?;
        }
    }
    ensure(
        CostasState::new(3, LOOP_BW, DAMPING, MAX_OFFSET).is_err()
            && CostasState::new(6, LOOP_BW, DAMPING, MAX_OFFSET).is_err()
            && CostasState::new(4, 0.21, DAMPING, MAX_OFFSET).is_err(),
        || fail("an invalid order or bandwidth was accepted"),
    )?;
    Ok(())
}
