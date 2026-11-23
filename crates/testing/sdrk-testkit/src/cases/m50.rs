//! Suite cases of M50: the kernel `xcorr_fft` and its extreme inputs (test 6 of the milestone).
//!
//! `xcorr_fft_c32` compares `Fallback` and the facade with `Reference` (the f64 direct
//! convolution, SNR at least 100 dB, the contract) for `L`/`N` of 1/16, 8/16, 65/256 and
//! 127/512 with complex Gaussian taps and input (seed 705), and runs both in 20 chunk plans
//! (seed 705, chunks of 1 to 300 samples received per call, the samples a call does not consume
//! resubmitted, an output room of 0 to 3 blocks per call); every chunked run must give the bits of
//! the run in one call, and the history after it the last `L - 1` input samples.
//! `extremes_m50` feeds silence (exact zeros), a full-scale impulse (the taps within 100 dB),
//! NaN and the infinities in one sample (every output of the at most two blocks whose windows
//! hold it is non-finite, every other block equals the clean run bit for bit) and checks the
//! invalid parameters. No files are read, so the cases run on microcontrollers too.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use sdrk::Sdrk;
use sdrk_core::kernels::fft::FftPlan;
use sdrk_core::kernels::xcorr::{XcorrFft, XcorrFftState};
use sdrk_core::{C32, Error, Fallback, Reference};

use super::ensure;
use crate::arena::{Arena, Carver};
use crate::invariance::ChunkPlan;
use crate::rng::Pcg32;
use crate::suite::Failure;

/// The seed of the signals, the taps and the chunk plans (M50 tests 4 and 6); plan `k` of
/// configuration `j` uses `SEED * 1000 + 100 j + k`.
const SEED: u64 = 705;
/// Chunk plans per configuration.
const PLANS: u64 = 20;
/// Largest chunk of the plans.
const MAX_CHUNK: usize = 300;
/// The configurations: taps `L`, FFT size `N`.
const CONFIGS: [(usize, usize); 4] = [(1, 16), (8, 16), (65, 256), (127, 512)];
/// The contract floor in dB.
const FLOOR_DB: f64 = 100.0;

type St<'a> = XcorrFftState<C32, &'a mut [C32], &'a mut [C32], &'a mut [C32], &'a mut [C32]>;

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
        .ok_or_else(|| Failure::new("m50", "slice range"))
}

fn get_mut<X>(s: &mut [X], from: usize, to: usize) -> Result<&mut [X], Failure> {
    s.get_mut(from..to)
        .ok_or_else(|| Failure::new("m50", "slice range"))
}

/// Complex normal values of `sigma` per component.
fn normals(rng: &mut Pcg32, x: &mut [C32], sigma: f64) {
    for v in x.iter_mut() {
        *v = C32::new((sigma * rng.normal()) as f32, (sigma * rng.normal()) as f32);
    }
}

/// A state over storage carved from `c` for the `taps` and the FFT size `n`.
fn state<'a>(c: &mut Carver<'a>, taps: &[C32], n: usize) -> Result<St<'a>, Failure> {
    let h = c.take::<C32>(n)?;
    let hist = c.take::<C32>(taps.len().saturating_sub(1))?;
    let scratch = c.take::<C32>(n)?;
    let tw = c.take::<C32>(n / 2)?;
    let plan = FftPlan::new(n, tw)?;
    Ok(XcorrFftState::new(taps, h, hist, scratch, plan)?)
}

/// The whole input (a multiple of `B`) in one call.
fn whole<K: XcorrFft<C32>>(
    k: &K,
    st: &mut St<'_>,
    input: &[C32],
    out: &mut [C32],
) -> Result<(), Failure> {
    let p = k.xcorr_fft(st, input, out)?;
    ensure(
        p.consumed == input.len() && p.produced == input.len(),
        || Failure::new("m50", "not all input consumed"),
    )
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

/// Runs `input` (a multiple of `B`) through `k` in the chunks of `plan`, with an output room of
/// 0 to 3 blocks per call (drawn from `room`); returns the samples produced.
fn chunked<K: XcorrFft<C32>>(
    k: &K,
    st: &mut St<'_>,
    input: &[C32],
    out: &mut [C32],
    plan: ChunkPlan,
    room: &mut Pcg32,
) -> Result<usize, Failure> {
    const CASE: &str = "xcorr_fft_c32";
    let b = st.block_size();
    let mut sizes = plan.sizes();
    let (mut pos, mut received, mut stalls) = (0, 0, 0);
    while pos < input.len() {
        let size = sizes.next().ok_or_else(|| Failure::new(CASE, "plan"))?;
        received = (received + size).min(input.len());
        let end = (pos + room.below(4) as usize * b).min(out.len());
        let p = k.xcorr_fft(st, get(input, pos, received)?, get_mut(out, pos, end)?)?;
        let want = (received - pos).min(end - pos) / b * b;
        ensure(p.consumed == want && p.produced == want, || {
            Failure::new(CASE, "a call does not run the whole blocks that fit").detail(
                format_args!(
                    "input {}, output {}, consumed {}",
                    received - pos,
                    end - pos,
                    p.consumed
                ),
            )
        })?;
        pos += p.consumed;
        stalls = if p.consumed == 0 { stalls + 1 } else { 0 };
        ensure(stalls < 1000, || Failure::new(CASE, "no progress"))?;
    }
    Ok(pos)
}

/// `Fallback` and the facade against `Reference` within the contract (100 dB), and both in 20
/// chunk plans bit for bit against their run in one call.
pub(super) fn xcorr_fft_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "xcorr_fft_c32";
    let mut rng = Pcg32::new(SEED, 1);
    for (j, &(l, n)) in CONFIGS.iter().enumerate() {
        arena.reset();
        let mut c = arena.carver();
        let b = n - l + 1;
        let len = 3 * b;
        let taps = c.take::<C32>(l)?;
        normals(&mut rng, taps, libm::sqrt(0.5 / l as f64));
        let input = c.take::<C32>(len)?;
        normals(&mut rng, input, core::f64::consts::FRAC_1_SQRT_2);
        let r = c.take::<C32>(len)?;
        let f = c.take::<C32>(len)?;
        let s = c.take::<C32>(len)?;
        let k = c.take::<C32>(len)?;
        let mut st = state(&mut c, taps, n)?;
        whole(&Reference, &mut st, input, r)?;
        st.reset();
        whole(&Fallback, &mut st, input, f)?;
        st.reset();
        whole(&Sdrk, &mut st, input, s)?;
        for (who, out) in [("fallback", &*f), ("facade", &*s)] {
            let snr = snr_db(r, out);
            ensure(snr >= FLOOR_DB, || {
                Failure::new(CASE, "below 100 dB against the reference")
                    .detail(format_args!("{who}, L {l}, N {n}: {snr:.1} dB"))
            })?;
        }
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
                ensure(got == len && same(want, k), || {
                    Failure::new(CASE, "a chunk plan changes the output")
                        .detail(format_args!("{who}, L {l}, N {n}, plan seed {seed}"))
                })?;
                ensure(same(st.history(), get(input, len - (l - 1), len)?), || {
                    Failure::new(CASE, "a chunk plan changes the history")
                        .detail(format_args!("{who}, L {l}, N {n}, plan seed {seed}"))
                })?;
            }
        }
    }
    Ok(())
}

/// Extreme inputs and invalid parameters.
pub(super) fn extremes_m50(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_m50";
    let mut rng = Pcg32::new(SEED, 3);
    for &(l, n) in &[(1usize, 16usize), (8, 16), (65, 256), (127, 512)] {
        arena.reset();
        let mut c = arena.carver();
        let b = n - l + 1;
        let len = 4 * b;
        let taps = c.take::<C32>(l)?;
        normals(&mut rng, taps, libm::sqrt(0.5 / l as f64));
        let input = c.take::<C32>(len)?;
        let out = c.take::<C32>(len)?;
        let clean = c.take::<C32>(len)?;
        let mut st = state(&mut c, taps, n)?;
        // Silence gives exact zeros.
        input.fill(C32::default());
        whole(&Fallback, &mut st, input, out)?;
        ensure(out.iter().all(|v| v.re == 0.0 && v.im == 0.0), || {
            Failure::new(CASE, "silence gives a non-zero output").detail(format_args!("L {l}"))
        })?;
        // A full-scale impulse at the start of block 1 reproduces the taps scaled by it.
        let a = 1.0 - 1.0 / 32768.0;
        let imp = C32::new(a, -a);
        if let Some(x) = input.get_mut(b) {
            *x = imp;
        }
        st.reset();
        whole(&Fallback, &mut st, input, out)?;
        clean.fill(C32::default());
        for (w, t) in get_mut(clean, b, b + l)?.iter_mut().zip(taps.iter()) {
            *w = C32::new(t.re * imp.re - t.im * imp.im, t.re * imp.im + t.im * imp.re);
        }
        let snr = snr_db(clean, out);
        ensure(snr >= FLOOR_DB, || {
            Failure::new(CASE, "a full-scale impulse does not give the taps")
                .detail(format_args!("L {l}, N {n}: {snr:.1} dB"))
        })?;
        // NaN and the infinities at one sample: in the middle of block 1 and among its last
        // L - 1 samples (then the history carries it into block 2).
        normals(&mut rng, input, core::f64::consts::FRAC_1_SQRT_2);
        st.reset();
        whole(&Fallback, &mut st, input, clean)?;
        for at in [b + b / 2, 2 * b - 1] {
            for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let keep = *get(input, at, at + 1)?
                    .first()
                    .ok_or_else(|| Failure::new(CASE, "at"))?;
                if let Some(x) = input.get_mut(at) {
                    *x = C32::new(v, 0.5);
                }
                st.reset();
                whole(&Fallback, &mut st, input, out)?;
                let first = at / b;
                let last = if at % b + l > b { first + 1 } else { first };
                for (blk, (o, k)) in out.chunks_exact(b).zip(clean.chunks_exact(b)).enumerate() {
                    let ok = if (first..=last).contains(&blk) {
                        o.iter().all(|s| !s.re.is_finite() || !s.im.is_finite())
                    } else {
                        same(o, k)
                    };
                    ensure(ok, || {
                        Failure::new(CASE, "a special value spoils the wrong blocks")
                            .detail(format_args!("L {l}, value {v}, at {at}, block {blk}"))
                    })?;
                }
                // The facade on the same input keeps going too.
                st.reset();
                whole(&Sdrk, &mut st, input, out)?;
                if let Some(x) = input.get_mut(at) {
                    *x = keep;
                }
            }
        }
    }
    // Invalid parameters.
    arena.reset();
    let mut c = arena.carver();
    let bad = |r: Result<St<'_>, Error>, msg: &'static str| -> Result<(), Failure> {
        ensure(r.is_err(), || Failure::new(CASE, msg))
    };
    for (l, h, hist, scr, msg) in [
        (0usize, 64usize, 0usize, 64usize, "no taps accepted"),
        (33, 64, 32, 64, "33 taps with 64 points accepted"),
        (10, 63, 9, 64, "a short spectrum accepted"),
        (10, 64, 8, 64, "a short history accepted"),
        (10, 64, 10, 64, "a long history accepted"),
        (10, 64, 9, 128, "a long scratch accepted"),
    ] {
        let t = c.take::<C32>(l)?;
        let hs = c.take::<C32>(h)?;
        let hi = c.take::<C32>(hist)?;
        let sc = c.take::<C32>(scr)?;
        let tw = c.take::<C32>(32)?;
        let plan = FftPlan::new(64, tw)?;
        bad(XcorrFftState::new(t, hs, hi, sc, plan), msg)?;
    }
    for (n, msg) in [
        (24usize, "a plan of 24 points accepted"),
        (8, "a plan of 8 points accepted"),
    ] {
        let tw = c.take::<C32>(n / 2)?;
        ensure(FftPlan::<C32, _>::new(n, tw).is_err(), || {
            Failure::new(CASE, msg)
        })?;
    }
    Ok(())
}
