//! Suite cases of M43: the kernels `dot_complex` and `equalize` and their extreme inputs.
//!
//! Both kernels are bit-exact (the float32 order of their contracts) and accelerated by the SIMD
//! backend (the facade delegates to it). `dot_complex_c32` compares `Fallback` and the facade
//! with `Reference` for every length from 0 to 80 and some long ones (seed 596) and checks the
//! accuracy against the float64 sum. `equalize_c32` compares `Fallback` with `Reference` (output
//! and final state) and the facade with `Fallback`, and runs the facade in 20 chunk plans (seed
//! 595, chunks of 1 to 300 symbols, test 6 of M43); every run must give the outputs and the final
//! state of the run in one call. The signals are BPSK, QPSK and 8PSK symbols from `Pcg32` (seed
//! 595), one sample per symbol, through a short symbol-spaced channel, turned by a constant
//! phase, plus noise; no files are read, so the cases run on microcontrollers too.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use sdrk::Sdrk;
use sdrk_core::kernels::dot::DotComplex;
use sdrk_core::kernels::equalize::{EqConstellation, EqMode, EqParams, Equalize, EqualizeState};
use sdrk_core::{C32, Fallback, Reference};

use super::ensure;
use crate::arena::Arena;
use crate::invariance::ChunkPlan;
use crate::rng::Pcg32;
use crate::suite::Failure;

/// The seed of the `dot_complex` signals.
const DOT_SEED: u64 = 596;
/// The seed of the chunk plans and the signals of `equalize` (test 6 of M43); plan `k` of
/// configuration `j` uses `PLAN_SEED * 1000 + 100 j + k`.
const PLAN_SEED: u64 = 595;
/// Chunk plans per configuration.
const PLANS: u64 = 20;
/// Largest chunk of the plans.
const MAX_CHUNK: usize = 300;
/// Symbols per signal of the `equalize` case.
const N: usize = 2048;
/// The most weights of the equalizer (forward and feedback together).
const MAX_TAPS: usize = 63;

/// Bit equality; NaN equals NaN.
fn same(a: &[C32], b: &[C32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same1(*x, *y))
}

fn same1(x: C32, y: C32) -> bool {
    let eq = |p: f32, q: f32| p.to_bits() == q.to_bits() || (p.is_nan() && q.is_nan());
    eq(x.re, y.re) && eq(x.im, y.im)
}

fn get<X>(s: &[X], from: usize, to: usize) -> Result<&[X], Failure> {
    s.get(from..to)
        .ok_or_else(|| Failure::new("m43", "slice range"))
}

fn get_mut<X>(s: &mut [X], from: usize, to: usize) -> Result<&mut [X], Failure> {
    s.get_mut(from..to)
        .ok_or_else(|| Failure::new("m43", "slice range"))
}

fn finite(v: C32) -> bool {
    v.re.is_finite() && v.im.is_finite()
}

// ---------------------------------------------------------------------------------------
// dot_complex
// ---------------------------------------------------------------------------------------

/// Complex normal values of 0.3 per component.
fn normals(rng: &mut Pcg32, x: &mut [C32]) {
    for v in x.iter_mut() {
        *v = C32::new((0.3 * rng.normal()) as f32, (0.3 * rng.normal()) as f32);
    }
}

/// `Fallback` and the facade must give the bits of `Reference`.
fn dot_agree(case: &'static str, msg: &'static str, x: &[C32], w: &[C32]) -> Result<C32, Failure> {
    let r = Reference.dot_complex(x, w)?;
    let f = Fallback.dot_complex(x, w)?;
    let s = Sdrk.dot_complex(x, w)?;
    ensure(same1(r, f) && same1(r, s), || {
        Failure::new(case, msg).detail(format_args!(
            "length {}: reference {r}, fallback {f}, facade {s}",
            x.len()
        ))
    })?;
    Ok(r)
}

/// Every length from 0 to 80 and 255, 256, 1021, 1024: the implementations agree bit for bit,
/// and the result is within 1e-5 of the float64 sum relative to the sum of the magnitudes of the
/// products.
pub(super) fn dot_complex_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "dot_complex_c32";
    const LONG: usize = 1024;
    let mut c = arena.carver();
    let x = c.take::<C32>(LONG)?;
    let w = c.take::<C32>(LONG)?;
    let mut rng = Pcg32::new(DOT_SEED, 1);
    for n in (0..=80).chain([255, 256, 1021, LONG]) {
        let xs = get_mut(x, 0, n)?;
        normals(&mut rng, xs);
        let ws = get_mut(w, 0, n)?;
        normals(&mut rng, ws);
        let (xs, ws) = (get(x, 0, n)?, get(w, 0, n)?);
        let y = dot_agree(CASE, "the implementations differ", xs, ws)?;
        let (mut re, mut im, mut mag) = (0.0f64, 0.0f64, 0.0f64);
        for (a, b) in xs.iter().zip(ws) {
            let (ar, ai) = (f64::from(a.re), f64::from(a.im));
            let (br, bi) = (f64::from(b.re), f64::from(b.im));
            re += ar * br - ai * bi;
            im += ar * bi + ai * br;
            mag += libm::sqrt((ar * ar + ai * ai) * (br * br + bi * bi));
        }
        let err = libm::fabs(f64::from(y.re) - re) + libm::fabs(f64::from(y.im) - im);
        ensure(err <= 1e-5 * mag, || {
            Failure::new(CASE, "inaccurate against the float64 sum")
                .detail(format_args!("length {n}: error {err}, magnitude sum {mag}"))
        })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// equalize
// ---------------------------------------------------------------------------------------

/// The final state of a run, as bits.
#[derive(Clone, Copy)]
struct Snap {
    w: [C32; MAX_TAPS],
    h: [C32; MAX_TAPS],
    mode: EqMode,
    mse: u32,
    phase: u32,
    freq: u32,
}

impl Snap {
    fn of(st: &EqualizeState<&mut [C32], &mut [C32]>) -> Self {
        let mut s = Self {
            w: [C32::default(); MAX_TAPS],
            h: [C32::default(); MAX_TAPS],
            mode: st.mode(),
            mse: st.mse().to_bits(),
            phase: st.phase(),
            freq: st.frequency().to_bits(),
        };
        for (d, v) in s.w.iter_mut().zip(st.weights()) {
            *d = *v;
        }
        for (d, v) in s.h.iter_mut().zip(st.history()) {
            *d = *v;
        }
        s
    }

    fn same(&self, o: &Self) -> bool {
        same(&self.w, &o.w)
            && same(&self.h, &o.h)
            && self.mode == o.mode
            && self.mse == o.mse
            && self.phase == o.phase
            && self.freq == o.freq
    }
}

/// One equalizer configuration.
#[derive(Clone, Copy)]
struct Config {
    constellation: EqConstellation,
    /// Forward taps.
    taps: usize,
    feedback_taps: usize,
    mu_cma: f32,
    mu_dd: f32,
    cma_symbols: u32,
    phase_bw: f64,
}

impl Config {
    /// A fresh state on the storage `w`, `h`.
    fn state<'a>(
        &self,
        w: &'a mut [C32; MAX_TAPS],
        h: &'a mut [C32; MAX_TAPS],
    ) -> Result<EqualizeState<&'a mut [C32], &'a mut [C32]>, Failure> {
        let n = self.taps + self.feedback_taps;
        Ok(EqualizeState::new(
            get_mut(w, 0, n)?,
            get_mut(h, 0, n)?,
            EqParams {
                constellation: self.constellation,
                feedback_taps: self.feedback_taps,
                mu_cma: self.mu_cma,
                mu_dd: self.mu_dd,
                cma_symbols: self.cma_symbols,
                phase_bw: self.phase_bw,
            },
        )?)
    }
}

/// Symbols of `c` (BPSK `+-1`, QPSK `(+-1 +-j) / sqrt 2`, 8PSK `exp(j (2m + 1) pi / 8)`), times
/// `amp`, through `1 + 0.3 e^(j0.6) z^-1 + 0.1 e^(-j2) z^-2`, turned by `phase` (radians), plus
/// normal noise of `sigma` per component, from the PCG32 stream `stream` of the seed 595.
fn psk_signal(x: &mut [C32], c: EqConstellation, amp: f64, phase: f64, sigma: f64, stream: u64) {
    let mut rng = Pcg32::new(PLAN_SEED, stream);
    let r = core::f64::consts::FRAC_1_SQRT_2;
    let echo = [(0.3, 0.6), (0.1, -2.0)];
    let (mut s1, mut s2) = ((0.0, 0.0), (0.0, 0.0));
    let (pc, ps) = (libm::cos(phase), libm::sin(phase));
    for v in x.iter_mut() {
        let b = rng.next_u32();
        let s = match c {
            EqConstellation::Bpsk => (if b & 1 == 1 { 1.0 } else { -1.0 }, 0.0),
            EqConstellation::Qpsk => (
                if b & 1 == 1 { r } else { -r },
                if b & 2 == 2 { r } else { -r },
            ),
            _ => {
                let a = f64::from(2 * (b & 7) + 1) * core::f64::consts::PI / 8.0;
                (libm::cos(a), libm::sin(a))
            }
        };
        let mut re = s.0;
        let mut im = s.1;
        for ((m, ph), p) in echo.iter().zip([s1, s2]) {
            let (gr, gi) = (m * libm::cos(*ph), m * libm::sin(*ph));
            re += gr * p.0 - gi * p.1;
            im += gr * p.1 + gi * p.0;
        }
        s2 = s1;
        s1 = s;
        let (re, im) = (amp * (re * pc - im * ps), amp * (re * ps + im * pc));
        *v = C32::new(
            (re + sigma * rng.normal()) as f32,
            (im + sigma * rng.normal()) as f32,
        );
    }
}

/// `Fallback` against `Reference` (outputs and final state) and the facade against `Fallback`
/// on `x`; returns the final state of `Fallback`.
fn eq_agree(
    case: &'static str,
    msg: &'static str,
    cfg: &Config,
    x: &[C32],
    o1: &mut [C32],
    o2: &mut [C32],
) -> Result<Snap, Failure> {
    let (mut w, mut h) = ([C32::default(); MAX_TAPS], [C32::default(); MAX_TAPS]);
    let mut st = cfg.state(&mut w, &mut h)?;
    Fallback.equalize(&mut st, x, o1)?;
    let fb = Snap::of(&st);
    let (mut w, mut h) = ([C32::default(); MAX_TAPS], [C32::default(); MAX_TAPS]);
    let mut st = cfg.state(&mut w, &mut h)?;
    Reference.equalize(&mut st, x, o2)?;
    ensure(same(o1, o2) && fb.same(&Snap::of(&st)), || {
        Failure::new(case, msg).detail(format_args!("{} taps: reference differs", cfg.taps))
    })?;
    let (mut w, mut h) = ([C32::default(); MAX_TAPS], [C32::default(); MAX_TAPS]);
    let mut st = cfg.state(&mut w, &mut h)?;
    Sdrk.equalize(&mut st, x, o2)?;
    ensure(same(o1, o2) && fb.same(&Snap::of(&st)), || {
        Failure::new(case, msg).detail(format_args!("{} taps: facade differs", cfg.taps))
    })?;
    // The weights stay finite whatever the input.
    ensure(fb.w.iter().all(|v| finite(*v)), || {
        Failure::new(case, msg).detail(format_args!("{} taps: a weight is not finite", cfg.taps))
    })?;
    Ok(fb)
}

/// `C32`: CMA then decisions (QPSK with the phase tracker, BPSK with 3 feedback taps), decisions
/// only (8PSK with 2 feedback taps and the tracker) and CMA only (QPSK, no tracker), each against
/// the reference, the facade and 20 chunk plans; the equalizers must converge.
pub(super) fn equalize_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "equalize_c32";
    let mut c = arena.carver();
    let x = c.take::<C32>(N)?;
    let out = c.take::<C32>(N)?;
    let other = c.take::<C32>(N)?;
    let cfg = |constellation, taps, feedback_taps, mu: (f32, f32), cma_symbols, phase_bw| Config {
        constellation,
        taps,
        feedback_taps,
        mu_cma: mu.0,
        mu_dd: mu.1,
        cma_symbols,
        phase_bw,
    };
    let configs = [
        (
            cfg(EqConstellation::Qpsk, 11, 0, (0.05, 0.02), 1000, 0.02),
            1.3,
            0.05,
        ),
        (
            cfg(EqConstellation::Bpsk, 9, 3, (0.04, 0.04), 600, 0.0),
            -0.4,
            0.05,
        ),
        (
            cfg(EqConstellation::Psk8, 11, 2, (0.05, 0.03), 0, 0.01),
            0.2,
            0.03,
        ),
        (
            cfg(EqConstellation::Qpsk, 21, 0, (0.05, 0.02), u32::MAX, 0.0),
            2.2,
            0.05,
        ),
    ];
    for (j, (cfg, phase, sigma)) in configs.into_iter().enumerate() {
        psk_signal(x, cfg.constellation, 1.0, phase, sigma, 5000 + j as u64);
        let fb = eq_agree(CASE, "implementations differ", &cfg, x, out, other)?;
        for k in 0..PLANS {
            let (mut w, mut h) = ([C32::default(); MAX_TAPS], [C32::default(); MAX_TAPS]);
            let mut st = cfg.state(&mut w, &mut h)?;
            let mut sizes =
                ChunkPlan::new(PLAN_SEED * 1000 + 100 * j as u64 + k, MAX_CHUNK).sizes();
            let mut pos = 0usize;
            while pos < x.len() {
                let a = sizes.next().unwrap_or(1).min(x.len() - pos);
                Sdrk.equalize(
                    &mut st,
                    get(x, pos, pos + a)?,
                    get_mut(other, pos, pos + a)?,
                )?;
                pos += a;
            }
            ensure(same(out, other) && fb.same(&Snap::of(&st)), || {
                Failure::new(CASE, "output depends on the block size")
                    .detail(format_args!("configuration {j}, plan {k}"))
            })?;
        }
        // Convergence: in decision mode the smoothed error power ends below -20 dB; in CMA only
        // the modulus of the last 512 outputs is restored.
        let ok = if cfg.cma_symbols == u32::MAX {
            let tail = get(out, N - 512, N)?;
            let d: f32 = tail
                .iter()
                .map(|v| {
                    let m = v.re * v.re + v.im * v.im - 1.0;
                    m * m
                })
                .sum::<f32>()
                / 512.0;
            fb.mode == EqMode::Cma && d < 0.05
        } else {
            fb.mode == EqMode::Decision && f32::from_bits(fb.mse) < 0.01
        };
        ensure(ok, || {
            Failure::new(CASE, "the equalizer did not converge").detail(format_args!(
                "configuration {j}: mse {}",
                f32::from_bits(fb.mse)
            ))
        })?;
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

/// `dot_complex`: length 0 and 1, silence, full scale, NaN and infinities, overflow and a length
/// mismatch.
fn dot_extremes(case: &'static str, x: &mut [C32], w: &mut [C32]) -> Result<(), Failure> {
    let fail = |msg: &'static str| Failure::new(case, msg);
    let m = x.len().min(w.len());
    // dot_complex: length 0 gives +0, length 1 the exact product.
    let y = dot_agree(case, "length 0", &[], &[])?;
    ensure(y.re.to_bits() == 0 && y.im.to_bits() == 0, || {
        fail("length 0 is not +0")
    })?;
    let (a, b) = (C32::new(0.75, -1.5), C32::new(-2.0, 0.25));
    let y = dot_agree(case, "length 1", &[a], &[b])?;
    ensure(y == C32::new(-1.125, 3.1875), || {
        fail("length 1 is not the product")
    })?;
    // Silence and full scale (components +-1, the products and sums exact).
    for n in [7usize, 64, 129] {
        x.fill(C32::new(0.0, 0.0));
        w.fill(C32::new(0.5, -0.5));
        let y = dot_agree(case, "silence", get(x, 0, n)?, get(w, 0, n)?)?;
        ensure(y.re.to_bits() == 0 && y.im.to_bits() == 0, || {
            fail("silence is not +0")
        })?;
        for (i, (v, u)) in x.iter_mut().zip(w.iter_mut()).enumerate() {
            let s = |k: usize| if k.is_multiple_of(3) { -1.0 } else { 1.0 };
            *v = C32::new(s(i), s(i + 1));
            *u = C32::new(s(i + 2), -s(i));
        }
        dot_agree(case, "full scale", get(x, 0, n)?, get(w, 0, n)?)?;
    }
    // NaN, infinities, a denormal and f32::MAX in every position of either operand.
    for offset in 0..SPECIALS.len() {
        for (i, (v, u)) in x.iter_mut().zip(w.iter_mut()).enumerate() {
            *v = C32::new(special(i / 3 + offset), special(i / 5 + offset + 3));
            *u = C32::new(special(i / 7 + offset + 1), 0.5);
        }
        for n in [1usize, 9, 40, m] {
            dot_agree(case, "non-finite", get(x, 0, n)?, get(w, 0, n)?)?;
        }
    }
    ensure(
        Fallback.dot_complex(get(x, 0, 3)?, get(w, 0, 2)?).is_err(),
        || fail("a length mismatch was accepted"),
    )?;
    Ok(())
}

/// `equalize` with one constellation: silence, full scale, NaN and infinities, inputs far beyond
/// full scale, 3 and 63 taps (the impulse response of the start weights), the step-size bounds
/// and `cma_symbols` 0 and 1.
fn equalize_extremes(
    case: &'static str,
    constellation: EqConstellation,
    x: &mut [C32],
    o1: &mut [C32],
    o2: &mut [C32],
) -> Result<(), Failure> {
    let fail = |msg: &'static str| Failure::new(case, msg);
    let base = Config {
        constellation,
        taps: 11,
        feedback_taps: 0,
        mu_cma: 0.02,
        mu_dd: 0.02,
        cma_symbols: 200,
        phase_bw: 0.02,
    };

    // Silence: zeros out, the weights stay the start weights.
    x.fill(C32::new(0.0, 0.0));
    let end = eq_agree(case, "silence", &base, x, o1, o2)?;
    ensure(
        o1.iter().all(|v| v.re == 0.0 && v.im == 0.0)
            && end.w.iter().enumerate().all(|(i, v)| {
                *v == if i == 5 {
                    C32::new(1.0, 0.0)
                } else {
                    C32::new(0.0, 0.0)
                }
            }),
        || fail("silence did not give zeros with the start weights"),
    )?;

    // Full scale: unit-amplitude symbols with the largest component 1.
    let amp = if constellation == EqConstellation::Qpsk {
        core::f64::consts::SQRT_2
    } else {
        1.0
    };
    psk_signal(x, constellation, amp, 0.0, 0.0, 5010);
    eq_agree(case, "full scale", &base, x, o1, o2)?;

    // Far beyond full scale: the products overflow, the outputs are 0 and the weights stay
    // finite.
    for v in x.iter_mut() {
        *v = C32::new(v.re * 1e30, v.im * 1e30);
    }
    eq_agree(case, "1e30", &base, x, o1, o2)?;
    for v in x.iter_mut() {
        *v = C32::new(f32::MAX, -f32::MAX);
    }
    eq_agree(case, "f32::MAX", &base, x, o1, o2)?;

    // NaN, infinities, a denormal and f32::MAX in every phase of either component: a
    // non-finite input gives a zero output.
    for offset in 0..SPECIALS.len() {
        for (i, v) in x.iter_mut().enumerate() {
            *v = C32::new(special(i / 3 + offset), special(i / 5 + offset + 3));
        }
        eq_agree(case, "non-finite", &base, x, o1, o2)?;
        ensure(
            x.iter()
                .zip(o1.iter())
                .all(|(a, b)| finite(*a) || (b.re == 0.0 && b.im == 0.0)),
            || fail("a non-finite input did not give 0"),
        )?;
    }

    // A burst of NaN in a clean signal: the outputs are finite and the equalizer converges.
    psk_signal(x, constellation, 1.0, 0.0, 0.03, 5020);
    for v in x.get_mut(300..310).unwrap_or(&mut []) {
        *v = C32::new(f32::NAN, f32::NAN);
    }
    eq_agree(case, "NaN burst", &base, x, o1, o2)?;
    ensure(o1.iter().all(|v| finite(*v)), || {
        fail("an output around a NaN burst is not finite")
    })?;

    // 3 and 63 taps: in CMA mode an impulse comes out unchanged after (N - 1) / 2 symbols
    // (y = 0 and |y| = 1 give no update), then the parameter bounds.
    for taps in [3usize, 63] {
        let cfg = Config { taps, ..base };
        x.fill(C32::new(0.0, 0.0));
        if let Some(v) = x.first_mut() {
            *v = C32::new(1.0, 0.0);
        }
        eq_agree(case, "impulse", &cfg, x, o1, o2)?;
        let centre = (taps - 1) / 2;
        ensure(
            o1.iter().enumerate().all(|(i, v)| {
                *v == if i == centre {
                    C32::new(1.0, 0.0)
                } else {
                    C32::new(0.0, 0.0)
                }
            }),
            || {
                fail("the impulse response is not the centre tap")
                    .detail(format_args!("{taps} taps"))
            },
        )?;
        psk_signal(x, constellation, 1.0, 0.5, 0.05, 5030);
        for (mu_cma, mu_dd) in [(0.000_01, 0.000_01), (1.0, 1.0)] {
            eq_agree(
                case,
                "parameter bounds",
                &Config {
                    mu_cma,
                    mu_dd,
                    ..cfg
                },
                x,
                o1,
                o2,
            )?;
        }
    }

    // 51 forward and 12 feedback taps (63 weights), feedback in decision mode and the tracker
    // bounds.
    psk_signal(x, constellation, 1.0, 0.3, 0.05, 5035);
    for (taps, feedback_taps, phase_bw) in [(3usize, 1usize, 0.2), (51, 12, 0.0001), (5, 0, 0.0)] {
        let cfg = Config {
            taps,
            feedback_taps,
            phase_bw,
            cma_symbols: 100,
            ..base
        };
        eq_agree(case, "feedback and tracker bounds", &cfg, x, o1, o2)?;
    }

    // cma_symbols 0 starts and stays in decision mode; 1 switches after the first symbol.
    psk_signal(x, constellation, 1.0, 0.0, 0.05, 5040);
    let end = eq_agree(
        case,
        "cma_symbols 0",
        &Config {
            cma_symbols: 0,
            ..base
        },
        x,
        o1,
        o2,
    )?;
    let (mut wb, mut hb) = ([C32::default(); MAX_TAPS], [C32::default(); MAX_TAPS]);
    let mut st = Config {
        cma_symbols: 1,
        ..base
    }
    .state(&mut wb, &mut hb)?;
    let start = st.mode();
    Fallback.equalize(&mut st, get(x, 0, 1)?, get_mut(o1, 0, 1)?)?;
    ensure(
        end.mode == EqMode::Decision && start == EqMode::Cma && st.mode() == EqMode::Decision,
        || fail("the mode switch is wrong"),
    )?;
    let (mut wb, mut hb) = ([C32::default(); MAX_TAPS], [C32::default(); MAX_TAPS]);
    let st = Config {
        cma_symbols: 0,
        ..base
    }
    .state(&mut wb, &mut hb)?;
    ensure(st.mode() == EqMode::Decision, || {
        fail("cma_symbols 0 did not start in decision mode")
    })?;
    Ok(())
}

/// Invalid numbers of taps, history lengths, step sizes and tracker bandwidths are rejected.
fn equalize_invalid(case: &'static str) -> Result<(), Failure> {
    let fail = |msg: &'static str| Failure::new(case, msg);
    let (mut wb, mut hb) = (
        [C32::default(); MAX_TAPS + 1],
        [C32::default(); MAX_TAPS + 1],
    );
    let q = EqConstellation::Qpsk;
    // (weights, history, feedback taps, mu_cma, mu_dd, phase_bw)
    let bad = [
        (1usize, 1usize, 0usize, 0.01f32, 0.01f32, 0.02f64),
        (4, 4, 0, 0.01, 0.01, 0.02),
        (MAX_TAPS + 1, MAX_TAPS + 1, 0, 0.01, 0.01, 0.02),
        (5, 3, 0, 0.01, 0.01, 0.02),
        (5, 5, 0, 0.0, 0.01, 0.02),
        (5, 5, 0, 0.01, 1.01, 0.02),
        (5, 5, 0, f32::NAN, 0.01, 0.02),
        (5, 5, 0, 0.01, f32::INFINITY, 0.02),
        // 2 forward taps, 4 feedback taps of 5 weights, 64 feedback taps.
        (5, 5, 3, 0.01, 0.01, 0.02),
        (5, 5, 6, 0.01, 0.01, 0.02),
        (MAX_TAPS + 1, MAX_TAPS + 1, 64, 0.01, 0.01, 0.02),
        (5, 5, 0, 0.01, 0.01, 0.000_05),
        (5, 5, 0, 0.01, 0.01, 0.21),
        (5, 5, 0, 0.01, 0.01, f64::NAN),
        (5, 5, 0, 0.01, 0.01, -0.01),
    ];
    for (nw, nh, feedback_taps, mu_cma, mu_dd, phase_bw) in bad {
        let r = EqualizeState::new(
            get_mut(&mut wb, 0, nw)?,
            get_mut(&mut hb, 0, nh)?,
            EqParams {
                constellation: q,
                feedback_taps,
                mu_cma,
                mu_dd,
                cma_symbols: 100,
                phase_bw,
            },
        );
        ensure(r.is_err(), || {
            fail("invalid parameters were accepted").detail(format_args!(
                "{nw} {nh} {feedback_taps} {mu_cma} {mu_dd} {phase_bw}"
            ))
        })?;
    }
    Ok(())
}

/// `dot_complex`: length 0 and 1, silence, full scale, NaN and infinities, overflow and a length
/// mismatch. `equalize`: silence, full scale, NaN and infinities, inputs far beyond full scale,
/// 3 and 63 taps (the impulse response of the start weights), `cma_symbols` 0 and 1 and the
/// parameter bounds.
pub(super) fn extremes_m43(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_m43";
    const M: usize = 600;
    let mut c = arena.carver();
    let x = c.take::<C32>(M)?;
    let o1 = c.take::<C32>(M)?;
    let o2 = c.take::<C32>(M)?;
    let w = c.take::<C32>(M)?;
    dot_extremes(CASE, x, w)?;
    for constellation in [
        EqConstellation::Bpsk,
        EqConstellation::Qpsk,
        EqConstellation::Psk8,
    ] {
        equalize_extremes(CASE, constellation, x, o1, o2)?;
    }
    equalize_invalid(CASE)
}
