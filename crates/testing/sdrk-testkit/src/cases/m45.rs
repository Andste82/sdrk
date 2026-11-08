//! Suite cases of M45: the kernel `channelize` and its extreme inputs.
//!
//! `channelize_c32` compares `Fallback` and the facade with `Reference` (the f64 model, SNR at
//! least 100 dB, the contract) for 4, 8, 16 and 64 channels with random taps and noise (seed 662),
//! and runs both in 20 chunk plans (seed 662, chunks of 1 to 300 samples, an output space of 0 to
//! 4 steps per call); every chunked run must give the bits of the run in one call and the same
//! hop position. `extremes_m45` feeds silence (exact zeros), full scale (finite and within the
//! sum of the tap magnitudes), NaN, infinities and `f32::MAX` (only the steps whose window holds
//! the value are affected, the steps after it equal those of the clean input bit for bit) and
//! checks the invalid parameters. No files are read, so the cases run on microcontrollers too.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use sdrk::Sdrk;
use sdrk_core::kernels::channelize::{Channelize, ChannelizeState};
use sdrk_core::{C32, Error, Fallback, Reference};

use super::ensure;
use crate::arena::{Arena, Carver};
use crate::invariance::ChunkPlan;
use crate::rng::Pcg32;
use crate::suite::Failure;

/// The seed of the signals, the taps and the chunk plans (M45 test 1); plan `k` of
/// configuration `j` uses `SEED * 1000 + 100 j + k`.
const SEED: u64 = 662;
/// Chunk plans per configuration.
const PLANS: u64 = 20;
/// Largest chunk of the plans.
const MAX_CHUNK: usize = 300;
/// Input samples per configuration.
const N: usize = 512;
/// The configurations: channels, taps per branch.
const CONFIGS: [(usize, usize); 4] = [(4, 4), (8, 5), (16, 8), (64, 4)];

type St<'a> = ChannelizeState<C32, &'a mut [f32], &'a mut [C32], &'a mut [C32]>;

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
        .ok_or_else(|| Failure::new("m45", "slice range"))
}

fn get_mut<X>(s: &mut [X], from: usize, to: usize) -> Result<&mut [X], Failure> {
    s.get_mut(from..to)
        .ok_or_else(|| Failure::new("m45", "slice range"))
}

/// Random taps with a sum of about 1.
fn random_taps(rng: &mut Pcg32, taps: &mut [f32]) {
    let n = taps.len() as f64;
    for t in taps.iter_mut() {
        *t = ((1.0 + (2.0 * rng.next_f64() - 1.0)) / n) as f32;
    }
}

/// Complex normal values of 0.3 per component.
fn normals(rng: &mut Pcg32, x: &mut [C32]) {
    for v in x.iter_mut() {
        *v = C32::new((0.3 * rng.normal()) as f32, (0.3 * rng.normal()) as f32);
    }
}

/// A state over storage carved from `c`: a tap storage of `2 M L` with the prototype `taps` in
/// its first half, a history of `2 M L`.
fn state<'a>(c: &mut Carver<'a>, taps: &[f32], m: usize) -> Result<St<'a>, Failure> {
    let t = c.take::<f32>(2 * taps.len())?;
    t.get_mut(..taps.len())
        .ok_or_else(|| Failure::new("m45", "tap storage"))?
        .copy_from_slice(taps);
    let h = c.take::<C32>(2 * taps.len())?;
    let w = c.take::<C32>(m / 2)?;
    Ok(ChannelizeState::new(t, m, h, w)?)
}

/// The whole input in one call; returns the values produced.
fn whole<K: Channelize<C32>>(
    k: &K,
    st: &mut St<'_>,
    input: &[C32],
    out: &mut [C32],
) -> Result<usize, Failure> {
    let p = k.channelize(st, input, out)?;
    ensure(p.consumed == input.len(), || {
        Failure::new("m45", "not all input consumed")
    })?;
    Ok(p.produced)
}

/// The SNR of `got` against `want` in dB.
fn snr_db(want: &[C32], got: &[C32]) -> f64 {
    let (mut p, mut e) = (0.0f64, 0.0f64);
    for (w, g) in want.iter().zip(got) {
        let (wr, wi) = (f64::from(w.re), f64::from(w.im));
        p += wr * wr + wi * wi;
        let (dr, di) = (wr - f64::from(g.re), wi - f64::from(g.im));
        e += dr * dr + di * di;
    }
    if e == 0.0 {
        f64::INFINITY
    } else {
        10.0 * libm::log10(p / e)
    }
}

/// Runs `input` through `k` in the chunks of `plan` with an output space of 0 to 4 steps per
/// call (drawn from `room`); returns the values produced.
fn chunked<K: Channelize<C32>>(
    k: &K,
    st: &mut St<'_>,
    input: &[C32],
    out: &mut [C32],
    plan: ChunkPlan,
    room: &mut Pcg32,
) -> Result<usize, Failure> {
    const CASE: &str = "channelize_c32";
    let m = st.channels();
    let mut sizes = plan.sizes();
    let (mut pos, mut written, mut stalls) = (0, 0, 0);
    while pos < input.len() {
        let size = sizes
            .next()
            .ok_or_else(|| Failure::new(CASE, "plan"))?
            .min(input.len() - pos);
        let steps = room.below(5) as usize;
        let end = (written + steps * m).min(out.len());
        let max = st.max_input_for_output((end - written) / m);
        let p = k.channelize(
            st,
            get(input, pos, pos + size)?,
            get_mut(out, written, end)?,
        )?;
        ensure(p.consumed == size.min(max), || {
            Failure::new(CASE, "consumption differs from max_input_for_output").detail(
                format_args!("size {size}, max {max}, consumed {}", p.consumed),
            )
        })?;
        pos += p.consumed;
        written += p.produced;
        stalls = if p.consumed == 0 { stalls + 1 } else { 0 };
        ensure(stalls < 100, || Failure::new(CASE, "no progress"))?;
    }
    Ok(written)
}

/// `Fallback` and the facade against `Reference` within the contract (100 dB), and both in 20
/// chunk plans bit for bit against their run in one call.
pub(super) fn channelize_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "channelize_c32";
    let mut rng = Pcg32::new(SEED, 1);
    for (j, &(m, l)) in CONFIGS.iter().enumerate() {
        arena.reset();
        let mut c = arena.carver();
        let taps = c.take::<f32>(m * l)?;
        random_taps(&mut rng, taps);
        let input = c.take::<C32>(N)?;
        normals(&mut rng, input);
        let values = N / (m / 2) * m;
        let r = c.take::<C32>(values)?;
        let f = c.take::<C32>(values)?;
        let s = c.take::<C32>(values)?;
        let k = c.take::<C32>(values)?;
        let mut st = state(&mut c, taps, m)?;
        let got = whole(&Reference, &mut st, input, r)?;
        ensure(got == values, || {
            Failure::new(CASE, "reference: output count")
        })?;
        let mut st = state(&mut c, taps, m)?;
        whole(&Fallback, &mut st, input, f)?;
        let mut st = state(&mut c, taps, m)?;
        whole(&Sdrk, &mut st, input, s)?;
        for (who, out) in [("fallback", &*f), ("facade", &*s)] {
            let snr = snr_db(r, out);
            ensure(snr >= 100.0, || {
                Failure::new(CASE, "below 100 dB against the reference")
                    .detail(format_args!("{who}, M {m}, L {l}: {snr:.1} dB"))
            })?;
        }
        let mut st = state(&mut c, taps, m)?;
        for plan in 0..PLANS {
            let seed = SEED * 1000 + 100 * j as u64 + plan;
            for (who, want) in [("fallback", &*f), ("facade", &*s)] {
                st.reset();
                k.fill(C32::default());
                let mut room = Pcg32::new(seed, 2);
                let plan = ChunkPlan::new(seed, MAX_CHUNK);
                let got = if who == "fallback" {
                    chunked(&Fallback, &mut st, input, k, plan, &mut room)?
                } else {
                    chunked(&Sdrk, &mut st, input, k, plan, &mut room)?
                };
                ensure(got == values && same(want, k), || {
                    Failure::new(CASE, "a chunk plan changes the output")
                        .detail(format_args!("{who}, M {m}, plan seed {seed}"))
                })?;
                let hops = N / (m / 2);
                ensure(
                    st.phase() == N % (m / 2) && st.odd_step() == (hops % 2 == 1),
                    || Failure::new(CASE, "a chunk plan changes the hop position"),
                )?;
            }
        }
    }
    Ok(())
}

/// Extreme inputs and invalid parameters.
pub(super) fn extremes_m45(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_m45";
    let mut rng = Pcg32::new(SEED, 3);
    for &(m, l) in &[(4usize, 4usize), (16, 8), (64, 4)] {
        arena.reset();
        let mut c = arena.carver();
        let n = 6 * m * l;
        let values = n / (m / 2) * m;
        let taps = c.take::<f32>(m * l)?;
        random_taps(&mut rng, taps);
        let gain: f32 = taps.iter().map(|t| libm::fabsf(*t)).sum();
        let input = c.take::<C32>(n)?;
        let out = c.take::<C32>(values)?;
        let clean = c.take::<C32>(values)?;
        // Silence gives exact zeros.
        input.fill(C32::default());
        let mut st = state(&mut c, taps, m)?;
        whole(&Fallback, &mut st, input, out)?;
        ensure(out.iter().all(|v| v.re == 0.0 && v.im == 0.0), || {
            Failure::new(CASE, "silence gives a non-zero output").detail(format_args!("M {m}"))
        })?;
        // Full scale, alternating: finite and within the sum of the tap magnitudes.
        let a = 1.0 - 1.0 / 32768.0;
        for (i, v) in input.iter_mut().enumerate() {
            let s = if i % 2 == 0 { a } else { -a };
            *v = C32::new(s, -s);
        }
        st.reset();
        whole(&Fallback, &mut st, input, out)?;
        let bound = gain * core::f32::consts::SQRT_2 * 1.0001;
        ensure(
            out.iter().all(|v| libm::sqrtf(v.norm_sqr()) <= bound),
            || Failure::new(CASE, "full scale exceeds the tap gain").detail(format_args!("M {m}")),
        )?;
        // NaN, infinities and f32::MAX at one sample.
        normals(&mut rng, input);
        st.reset();
        whole(&Fallback, &mut st, input, clean)?;
        let d = m / 2;
        let at = 2 * m * l + 3;
        let steps = values / m;
        // Steps `first .. last` read the sample (window `t - N + 1 ..= t`, `t = (s + 1) D - 1`).
        let first = (at + 1).div_ceil(d) - 1;
        let last = ((at + m * l) / d).min(steps);
        for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
            let keep = *get(input, at, at + 1)?
                .first()
                .ok_or_else(|| Failure::new(CASE, "at"))?;
            if let Some(x) = input.get_mut(at) {
                *x = C32::new(v, 0.5);
            }
            st.reset();
            whole(&Fallback, &mut st, input, out)?;
            for (s, group) in out.chunks_exact(m).enumerate() {
                let finite = group.iter().all(|x| x.re.is_finite() && x.im.is_finite());
                ensure(finite || (first..last).contains(&s), || {
                    Failure::new(CASE, "a special value spoils a step outside its window")
                        .detail(format_args!("M {m}, value {v}, step {s}"))
                })?;
            }
            ensure(
                same(get(out, last * m, values)?, get(clean, last * m, values)?),
                || {
                    Failure::new(CASE, "the steps after a special value differ")
                        .detail(format_args!("M {m}, value {v}"))
                },
            )?;
            if let Some(x) = input.get_mut(at) {
                *x = keep;
            }
        }
        // The facade on the same specials keeps going too.
        st.reset();
        whole(&Sdrk, &mut st, input, out)?;
    }
    // Invalid parameters.
    arena.reset();
    let mut c = arena.carver();
    let bad = |r: Result<St<'_>, Error>, msg: &'static str| -> Result<(), Failure> {
        ensure(r.is_err(), || Failure::new(CASE, msg))
    };
    for (taps, m, hist, tw, msg) in [
        // The tap storage holds every tap twice: `2 M L` values.
        (96usize, 12usize, 96usize, 6usize, "12 channels accepted"),
        (32, 2, 32, 1, "2 channels accepted"),
        (4096, 512, 4096, 256, "512 channels accepted"),
        (96, 16, 96, 8, "3 taps per branch accepted"),
        (2 * 16 * 33, 16, 16 * 33, 8, "33 taps per branch accepted"),
        (
            258,
            16,
            258,
            8,
            "a tap count that is not a multiple of M accepted",
        ),
        (255, 16, 256, 8, "a tap storage of odd length accepted"),
        (256, 16, 127, 8, "a short history accepted"),
        (256, 16, 256, 7, "a short twiddle table accepted"),
    ] {
        let t = c.take::<f32>(taps)?;
        let h = c.take::<C32>(hist)?;
        let w = c.take::<C32>(tw)?;
        bad(ChannelizeState::new(t, m, h, w), msg)?;
    }
    // An output that is not a multiple of M.
    let t = c.take::<f32>(64)?;
    let mut st = state(&mut c, t, 16)?;
    let x = c.take::<C32>(64)?;
    let o = c.take::<C32>(24)?;
    ensure(Fallback.channelize(&mut st, x, o).is_err(), || {
        Failure::new(CASE, "an output of 24 values accepted for 16 channels")
    })?;
    ensure(st.phase() == 0, || {
        Failure::new(CASE, "a rejected call consumed input")
    })?;
    Ok(())
}
