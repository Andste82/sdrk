//! Suite cases of M48: the kernel `gauss_noise` and its extremes (test 7 of the milestone).
//!
//! `gauss_noise_c32` (`C32` and `CQ15`) and `gauss_noise_q15` (`Q15` and `f32`) compare
//! `Fallback` with `Reference` and the facade with `Fallback` bit for bit, from the start and
//! after a `seek`, check the index after every call and run `Fallback` and the facade in 20 chunk
//! plans (seed 687, chunks of 1 to 300 samples); every chunked run must give the bits of the run
//! in one call. `extremes_m48` (test 6) checks `sigma` 0 and 9e-31 (flushed: `+0.0` and zeros,
//! the generator still advances), `sigma` 1e-30 (not flushed), `sigma` 16 (nearly every `Q15`
//! value saturates, `f32` stays finite and within `830729 · 2^-17 · 16`), the largest index
//! `u64::MAX` for real and complex types (wrapping; for complex types it is the index
//! `u64::MAX − 2^63`), an empty output, the invalid `sigma` values and the `InvalidState` of a
//! type change. No files are read and the kernel needs no `f64`, so the cases run on
//! microcontrollers too.

#![allow(
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use sdrk::Sdrk;
use sdrk_core::kernels::noise::{GaussNoise, GaussNoiseState};
use sdrk_core::{C32, CQ15, Error, Fallback, Q15, Reference, Sample};

use super::ensure;
use crate::arena::Arena;
use crate::invariance::ChunkPlan;
use crate::metrics::bit_exact;
use crate::suite::Failure;

/// The seed of the chunk plans (M48 test 5); plan `k` of type `j` uses `SEED * 1000 + 100 j + k`.
const SEED: u64 = 687;
/// Chunk plans per type.
const PLANS: u64 = 20;
/// Largest chunk of the plans.
const MAX_CHUNK: usize = 300;
/// Samples per run.
const N: usize = 1024;
/// The index of the second run (after a `seek`).
const SEEK: u64 = 1_000_003;

fn get_mut<X>(s: &mut [X], from: usize, to: usize) -> Result<&mut [X], Failure> {
    s.get_mut(from..to)
        .ok_or_else(|| Failure::new("m48", "slice range"))
}

/// A state of the plan seed and stream `stream`, at `start`.
fn state(stream: u64, sigma: f32, start: u64) -> Result<GaussNoiseState, Failure> {
    let mut st = GaussNoiseState::new(SEED, stream, sigma)?;
    st.seek(start);
    Ok(st)
}

/// Fills `out` in one call and checks the index.
fn whole<T: Sample, K: GaussNoise<T>>(
    case: &'static str,
    k: &K,
    st: &mut GaussNoiseState,
    out: &mut [T],
) -> Result<(), Failure> {
    let start = st.index();
    k.gauss_noise(st, out)?;
    ensure(st.index() == start.wrapping_add(out.len() as u64), || {
        Failure::new(case, "the index did not advance by the output length")
    })
}

/// Fills `out` in the chunks of `plan` and checks the index after every call.
fn chunked<T: Sample, K: GaussNoise<T>>(
    case: &'static str,
    k: &K,
    st: &mut GaussNoiseState,
    out: &mut [T],
    plan: ChunkPlan,
) -> Result<(), Failure> {
    let mut sizes = plan.sizes();
    let mut pos = 0;
    while pos < out.len() {
        let n = sizes
            .next()
            .ok_or_else(|| Failure::new(case, "plan"))?
            .min(out.len() - pos);
        whole(case, k, st, get_mut(out, pos, pos + n)?)?;
        pos += n;
    }
    Ok(())
}

/// One type: `Fallback` against `Reference`, the facade against `Fallback`, from index 0 and
/// from `SEEK`, and 20 chunk plans of `Fallback` and the facade (alternating).
fn one_type<T: Sample>(
    case: &'static str,
    arena: &mut Arena<'_>,
    j: u64,
    sigma: f32,
) -> Result<(), Failure>
where
    Fallback: GaussNoise<T>,
    Reference: GaussNoise<T>,
    Sdrk: GaussNoise<T>,
{
    arena.reset();
    let mut c = arena.carver();
    let r = c.take::<T>(N)?;
    let f = c.take::<T>(N)?;
    let s = c.take::<T>(N)?;
    let k = c.take::<T>(N)?;
    for start in [0, SEEK] {
        whole(case, &Reference, &mut state(j, sigma, start)?, r)?;
        whole(case, &Fallback, &mut state(j, sigma, start)?, f)?;
        whole(case, &Sdrk, &mut state(j, sigma, start)?, s)?;
        if let Some(i) = bit_exact(r, f) {
            return Err(Failure::new(case, "fallback differs from the reference")
                .detail(format_args!("start {start}, sample {i}")));
        }
        if let Some(i) = bit_exact(f, s) {
            return Err(Failure::new(case, "the facade differs from the fallback")
                .detail(format_args!("start {start}, sample {i}")));
        }
    }
    // `r` holds the run from `SEEK`; the plans start there too.
    for p in 0..PLANS {
        let seed = SEED * 1000 + 100 * j + p;
        let plan = ChunkPlan::new(seed, MAX_CHUNK);
        let mut st = state(j, sigma, SEEK)?;
        k.fill(T::ZERO);
        if p % 2 == 0 {
            chunked(case, &Fallback, &mut st, k, plan)?;
        } else {
            chunked(case, &Sdrk, &mut st, k, plan)?;
        }
        if let Some(i) = bit_exact(r, k) {
            return Err(Failure::new(case, "a chunk plan changes the output")
                .detail(format_args!("plan seed {seed}, sample {i}")));
        }
    }
    Ok(())
}

/// `C32` and `CQ15`.
pub(super) fn gauss_noise_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "gauss_noise_c32";
    one_type::<C32>(CASE, arena, 0, 1.0)?;
    one_type::<CQ15>(CASE, arena, 1, 0.25)
}

/// `Q15` and `f32`.
pub(super) fn gauss_noise_q15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "gauss_noise_q15";
    one_type::<Q15>(CASE, arena, 2, 0.6)?;
    one_type::<f32>(CASE, arena, 3, 1.0)
}

/// `sigma` 0 (or a `sigma` below 1e-30, which is flushed) gives `+0.0` (or 0) in every sample
/// and still advances the generator: the samples after switching to `sigma` 0.5 equal those of
/// a fresh state at the same index.
fn zero_sigma<T: Sample + PartialEq>(
    arena: &mut Arena<'_>,
    j: u64,
    sigma: f32,
    is_zero: impl Fn(&T) -> bool,
) -> Result<(), Failure>
where
    Fallback: GaussNoise<T>,
    Reference: GaussNoise<T>,
    Sdrk: GaussNoise<T>,
{
    const CASE: &str = "extremes_m48";
    arena.reset();
    let mut c = arena.carver();
    let a = c.take::<T>(64)?;
    let b = c.take::<T>(64)?;
    let want = c.take::<T>(64)?;
    for who in ["fallback", "reference", "facade"] {
        let mut st = state(j, sigma, 0)?;
        let call = |st: &mut GaussNoiseState, out: &mut [T]| match who {
            "fallback" => whole(CASE, &Fallback, st, out),
            "reference" => whole(CASE, &Reference, st, out),
            _ => whole(CASE, &Sdrk, st, out),
        };
        call(&mut st, a)?;
        ensure(a.iter().all(&is_zero), || {
            Failure::new(CASE, "a zero sigma gives a value other than +0")
                .detail(format_args!("{who}, sigma {sigma}"))
        })?;
        st.set_sigma(0.5)?;
        call(&mut st, b)?;
        whole(CASE, &Reference, &mut state(j, 0.5, 64)?, want)?;
        if let Some(i) = bit_exact(want, b) {
            return Err(
                Failure::new(CASE, "a zero sigma does not advance the generator")
                    .detail(format_args!("{who}, sigma {sigma}, sample {i}")),
            );
        }
    }
    Ok(())
}

/// The largest index `u64::MAX` of one type: `Fallback`, `Reference` and the facade agree there,
/// and the run equals the run of a fresh state from `wrapped`, the index the generator position
/// `d · u64::MAX` (wrapping) belongs to: `u64::MAX` for real types (`d = 1`), `u64::MAX − 2^63`
/// for complex types (`2 · u64::MAX = 2^64 − 2`). A real run continues with sample 0 after the
/// wrap of the index, which the comparison with a fresh run from `u64::MAX` cannot show, so
/// the samples after the first are also compared with a fresh run from index 0.
fn largest_index<T: Sample>(arena: &mut Arena<'_>, j: u64, complex: bool) -> Result<(), Failure>
where
    Fallback: GaussNoise<T>,
    Reference: GaussNoise<T>,
    Sdrk: GaussNoise<T>,
{
    const CASE: &str = "extremes_m48";
    arena.reset();
    let mut c = arena.carver();
    let a = c.take::<T>(64)?;
    let b = c.take::<T>(64)?;
    let s = c.take::<T>(64)?;
    let w = c.take::<T>(64)?;
    whole(CASE, &Reference, &mut state(j, 1.0, u64::MAX)?, a)?;
    whole(CASE, &Fallback, &mut state(j, 1.0, u64::MAX)?, b)?;
    whole(CASE, &Sdrk, &mut state(j, 1.0, u64::MAX)?, s)?;
    ensure(
        bit_exact(a, b).is_none() && bit_exact(b, s).is_none(),
        || {
            Failure::new(
                CASE,
                "index u64::MAX: fallback, reference and facade differ",
            )
        },
    )?;
    if complex {
        whole(
            CASE,
            &Fallback,
            &mut state(j, 1.0, u64::MAX - (1 << 63))?,
            w,
        )?;
        if let Some(i) = bit_exact(a, w) {
            return Err(
                Failure::new(CASE, "complex index u64::MAX differs from u64::MAX - 2^63")
                    .detail(format_args!("sample {i}")),
            );
        }
    } else {
        whole(CASE, &Fallback, &mut state(j, 1.0, 0)?, w)?;
        let (_, after) = a.split_at(1);
        let (head, _) = w.split_at(after.len());
        if let Some(i) = bit_exact(after, head) {
            return Err(
                Failure::new(CASE, "real index u64::MAX does not wrap to sample 0")
                    .detail(format_args!("sample {i}")),
            );
        }
    }
    Ok(())
}

/// Extreme parameters and states.
pub(super) fn extremes_m48(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_m48";
    // sigma 0 and 9e-31 (below 1e-30: flushed to 0).
    for (j, sigma) in [(10, 0.0_f32), (30, 9e-31)] {
        zero_sigma::<C32>(arena, j, sigma, |v| {
            v.re.to_bits() == 0 && v.im.to_bits() == 0
        })?;
        zero_sigma::<f32>(arena, j + 1, sigma, |v| v.to_bits() == 0)?;
        zero_sigma::<CQ15>(arena, j + 2, sigma, |v| v.re.0 == 0 && v.im.0 == 0)?;
        zero_sigma::<Q15>(arena, j + 3, sigma, |v| v.0 == 0)?;
    }

    // sigma 1e-30 is not flushed: f32 values of at most 6.338e-30 (normal numbers), Q15 zeros.
    arena.reset();
    let mut c = arena.carver();
    let x = c.take::<f32>(64)?;
    let xf = c.take::<f32>(64)?;
    whole(CASE, &Reference, &mut state(18, 1e-30, 0)?, x)?;
    whole(CASE, &Fallback, &mut state(18, 1e-30, 0)?, xf)?;
    ensure(bit_exact(x, xf).is_none(), || {
        Failure::new(
            CASE,
            "sigma 1e-30: fallback differs from the reference (f32)",
        )
    })?;
    ensure(
        x.iter()
            .all(|v| v.to_bits() == 0 || (v.is_normal() && libm::fabsf(*v) <= 6.34e-30)),
        || {
            Failure::new(
                CASE,
                "sigma 1e-30 gives a value that is not a small normal number",
            )
        },
    )?;
    ensure(x.iter().filter(|v| **v != 0.0).count() >= 60, || {
        Failure::new(CASE, "sigma 1e-30 is flushed")
    })?;
    let q = c.take::<Q15>(64)?;
    whole(CASE, &Fallback, &mut state(19, 1e-30, 0)?, q)?;
    ensure(q.iter().all(|v| v.0 == 0), || {
        Failure::new(CASE, "sigma 1e-30 gives a Q15 value other than 0")
    })?;

    // sigma 16: nearly every Q15 value saturates (only |z| < 1/16, about 5 % of the values,
    // stays inside), f32 stays finite and within the tail limit 830729 · 2^-17 · 16 (exact in
    // f32).
    arena.reset();
    let mut c = arena.carver();
    let q = c.take::<Q15>(N)?;
    let qf = c.take::<Q15>(N)?;
    whole(CASE, &Reference, &mut state(20, 16.0, 0)?, q)?;
    whole(CASE, &Fallback, &mut state(20, 16.0, 0)?, qf)?;
    ensure(bit_exact(q, qf).is_none(), || {
        Failure::new(CASE, "sigma 16: fallback differs from the reference (Q15)")
    })?;
    let clipped = q
        .iter()
        .filter(|v| v.0 == i16::MAX || v.0 == i16::MIN)
        .count();
    ensure(clipped * 100 >= 85 * N, || {
        Failure::new(CASE, "sigma 16 saturates too few Q15 values")
            .detail(format_args!("{clipped} of {N}"))
    })?;
    let x = c.take::<f32>(N)?;
    let xf = c.take::<f32>(N)?;
    whole(CASE, &Reference, &mut state(21, 16.0, 0)?, x)?;
    whole(CASE, &Fallback, &mut state(21, 16.0, 0)?, xf)?;
    ensure(bit_exact(x, xf).is_none(), || {
        Failure::new(CASE, "sigma 16: fallback differs from the reference (f32)")
    })?;
    let zmax = 830_729.0_f32 / 8192.0;
    ensure(
        x.iter().all(|v| v.is_finite() && libm::fabsf(*v) <= zmax),
        || {
            Failure::new(
                CASE,
                "sigma 16 gives a value that is not finite or beyond the tail limit",
            )
        },
    )?;

    // The largest index u64::MAX, real and complex.
    largest_index::<f32>(arena, 22, false)?;
    largest_index::<Q15>(arena, 23, false)?;
    largest_index::<C32>(arena, 24, true)?;
    largest_index::<CQ15>(arena, 25, true)?;

    // An empty output changes nothing and does not fix the type.
    let mut st = state(17, 1.0, 5)?;
    let before = st;
    Fallback.gauss_noise(&mut st, &mut [] as &mut [C32])?;
    ensure(st == before, || {
        Failure::new(CASE, "an empty output changes the state")
    })?;
    Fallback.gauss_noise(&mut st, &mut [0.0f32; 4])?;

    // A type change is InvalidState; reset clears the type.
    let r = Fallback.gauss_noise(&mut st, &mut [C32::default(); 4]);
    ensure(matches!(r, Err(Error::InvalidState(_))), || {
        Failure::new(CASE, "a type change is not InvalidState")
    })?;
    ensure(st.index() == 9, || {
        Failure::new(CASE, "a refused call moves the index")
    })?;
    st.reset();
    Fallback.gauss_noise(&mut st, &mut [C32::default(); 4])?;

    // Invalid sigma values.
    for sigma in [
        -1.0_f32,
        -1e-30,
        16.000_002,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        ensure(
            matches!(
                GaussNoiseState::new(1, 1, sigma),
                Err(Error::InvalidParameter(_))
            ),
            || Failure::new(CASE, "an invalid sigma accepted").detail(format_args!("{sigma}")),
        )?;
        ensure(
            matches!(st.set_sigma(sigma), Err(Error::InvalidParameter(_))),
            || {
                Failure::new(CASE, "set_sigma accepts an invalid sigma")
                    .detail(format_args!("{sigma}"))
            },
        )?;
    }
    ensure(st.sigma() == 1.0, || {
        Failure::new(CASE, "a refused set_sigma changed sigma")
    })?;
    Ok(())
}
