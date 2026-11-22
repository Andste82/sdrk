//! The baselines of M44.2 (spec §2 and §3, test 1): the outputs of the 16-QAM and 64-QAM
//! kernels as they are before the soft bits and the grid decision become one generic
//! implementation over the bits per axis, recorded from the unchanged code in
//! `tests/baseline/m44_2/` (`sdrk_testkit::baseline`). The same checks then hold the generic
//! instances for two and three bits per axis to the same bits.
//!
//! - **`constants`:** the levels, `sqrt E`, `E`, the point tables and `R2` of `Qam` (also
//!   rounded to `f32`, the radius `EqConstellation::cma_r2` gave until D2536), and the rotation
//!   map of every symbol by 0 to 3 quarter turns.
//! - **`qam_soft_bits`:** the soft bits through the facade `Sdrk` and through `Fallback` at the
//!   scales 1, 0.5, -2, 7.25, 1e-30, 3e30 and 0, on the axis values of [`axis_values`] (a fine
//!   sweep, every decision boundary and every kink of the soft bits with the 4 values on either
//!   side, the special values, random values of seed 838), whole and as the first 1 to 33
//!   symbols (the tails of every vector width); at the scale 1 also the full output of the 600
//!   symbols after the sweep (the boundaries, the special values and the first random values).
//! - **`qam_decide`:** `Qam::decide_level` of both components, `Qam::decide_symbol`,
//!   `Qam::decide` and `EqConstellation::decide` on the same symbols.
//! - **`costas`:** the loop with the constellations `qam16` and `qam64` on a locked signal
//!   with a phase and frequency offset, on Gaussian and on uniform noise and on the axis values,
//!   through the facade in pieces of 1000 symbols and through `Fallback` whole: the output and,
//!   after every piece, the phase, the frequency, the lock level and the lock flag.
//! - **`equalize`:** the equalizer with the constellations `qam16` and `qam64` through a
//!   three-tap channel with noise (CMA then decision mode), with 5 feedback taps, from the
//!   start in decision mode with the phase tracker, and on the axis values: the output, the
//!   `mse` and the mode after every piece, and the final weights.
//!
//! The kernels are bit-exact on every SIMD level (`docs/kernels.md`: every implementation of
//! `qam_soft_bits` gives the bits of the fallback, `costas` and `equalize` give the bits of
//! the `Reference`), so every table has one line per case at the level `all`, and every level
//! is checked against it. The sign and payload of a NaN are not part of the contracts, so every
//! NaN is digested as `0x7fc0_0000`.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::too_many_lines,
    clippy::large_stack_frames,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::approx_constant
)]

use sdrk::Sdrk;
use sdrk_core::kernels::constellation::{Qam, QamSoftBits, rotate_symbol};
use sdrk_core::kernels::costas::{Costas, CostasConstellation, CostasState};
use sdrk_core::kernels::equalize::{EqConstellation, EqMode, EqParams, Equalize, EqualizeState};
use sdrk_core::{C32, Fallback};
use sdrk_testkit::Pcg32;
use sdrk_testkit::baseline::{self, Baseline, Digest};

/// The baseline set of M44.2.
const SET: &str = "m44_2";
/// The level of every line: the kernels are bit-exact on every level.
const LEVEL: &str = "all";
/// The seed of the inputs.
const SEED: u64 = 838;
/// The orders whose outputs are recorded.
const ORDERS: [Qam; 2] = [Qam::Qam16, Qam::Qam64];
/// Symbols per piece of the facade runs of the loops.
const PIECE: usize = 1000;
/// Values of the sweep at the start of [`axis_values`].
const SWEEP: usize = 4001;
/// Symbols after the sweep whose soft bits at the scale 1 are stored in full.
const EDGE_SYMBOLS: usize = 600;

fn table(name: &str) -> Baseline {
    Baseline::load(SET, name, LEVEL).unwrap()
}

fn name(q: Qam) -> &'static str {
    match q {
        Qam::Qam16 => "qam16",
        Qam::Qam64 => "qam64",
        _ => panic!("no baseline for {q:?}"),
    }
}

/// `x` with every NaN as `0x7fc0_0000`.
fn fold(x: f32) -> f32 {
    if x.is_nan() {
        f32::from_bits(0x7fc0_0000)
    } else {
        x
    }
}

fn fold_c(v: C32) -> C32 {
    C32::new(fold(v.re), fold(v.im))
}

fn digest_f32(d: &mut Digest, v: &[f32]) {
    for &x in v {
        d.f32s(&[fold(x)]);
    }
}

fn digest_c32(d: &mut Digest, v: &[C32]) {
    for &x in v {
        d.c32s(&[fold_c(x)]);
    }
}

/// The `n` values on either side of `x` and `x` itself, in order (none past the infinities).
fn around(x: f32, n: u32, out: &mut Vec<f32>) {
    let step = |v: f32, up: bool| -> f32 {
        if v == 0.0 {
            let tiny = f32::from_bits(1);
            return if up { tiny } else { -tiny };
        }
        let b = v.to_bits();
        // Away from zero is +1 on the magnitude, toward zero -1.
        let away = (v > 0.0) == up;
        f32::from_bits(if away { b + 1 } else { b - 1 })
    };
    let mut lo = Vec::new();
    let mut v = x;
    for _ in 0..n {
        v = step(v, false);
        lo.push(v);
    }
    out.extend(lo.iter().rev());
    out.push(x);
    let mut v = x;
    for _ in 0..n {
        v = step(v, true);
        out.push(v);
    }
}

/// The axis values of the soft-bit and decision cases: a sweep over `-1.6 ..= 1.6` in 4001
/// steps, every decision boundary `(k - m / 2) / (sqrt E / 2)` and every integer `v` up to
/// `m` on the scale of both orders (the kinks of the soft bits) with the 4 values on either
/// side, the special values, and 4096 random values of seed 838 (uniform in `-1.5 .. 1.5`,
/// then Gaussian with the standard deviation 0.6).
fn axis_values() -> Vec<f32> {
    let mut v: Vec<f32> = (0..SWEEP)
        .map(|i| (-1.6 + 3.2 * i as f64 / (SWEEP - 1) as f64) as f32)
        .collect();
    for q in ORDERS {
        let m = q.levels_per_axis() as i32;
        let c = q.sqrt_energy();
        let h = c / 2.0;
        for k in 1..m {
            around((k as f32 - m as f32 / 2.0) / h, 4, &mut v);
        }
        for k in -m..=m {
            around(k as f32 / c, 4, &mut v);
        }
    }
    v.extend([
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1e-40,
        -1e-40,
        f32::MAX,
        f32::MIN,
        1e20,
        -1e20,
        1e-20,
        -1e-20,
    ]);
    let mut rng = Pcg32::new(SEED, 1);
    v.extend((0..2048).map(|_| (rng.next_f64() * 3.0 - 1.5) as f32));
    v.extend((0..2048).map(|_| (0.6 * rng.normal()) as f32));
    v
}

/// The symbols of the axis values: value `i` as the in-phase component, value `n - 1 - 3 i`
/// (modulo `n`) as the quadrature component, so that every value meets many others.
fn axis_symbols() -> Vec<C32> {
    let v = axis_values();
    let n = v.len();
    (0..n)
        .map(|i| C32::new(v[i], v[(n - 1 + 2 * n - (3 * i) % n) % n]))
        .collect()
}

#[test]
fn the_constants_keep_their_baseline() {
    let t = table("constants");
    for q in ORDERS {
        let mut d = Digest::new();
        d.f32s(q.levels());
        d.f32s(&[q.sqrt_energy()]);
        d.bytes(&q.energy().to_bits().to_le_bytes());
        d.bytes(&q.cma_r2().to_bits().to_le_bytes());
        d.c32s(q.point_table());
        // The `f32` CMA radius that `EqConstellation::cma_r2` gave before the square grids took
        // the multi-modulus start (D2536): `Qam::cma_r2` rounded once, the same bits.
        d.f32s(&[q.cma_r2() as f32]);
        let b = q.bits_per_axis() as u32;
        for s in 0..q.points() as u32 {
            d.f32s(&[q.point(s).re, q.point(s).im]);
            for r in 0..4 {
                d.bytes(&rotate_symbol(s, b, r).to_le_bytes());
            }
        }
        t.check(name(q), &d).unwrap();
    }
}

fn soft_bits<K: QamSoftBits<C32>>(k: &K, q: Qam, x: &[C32], scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len() * q.bits_per_symbol()];
    k.qam_soft_bits(q, x, &mut out, scale).unwrap();
    out
}

#[test]
fn qam_soft_bits_keeps_its_baseline() {
    let t = table("qam_soft_bits");
    let x = axis_symbols();
    for q in ORDERS {
        for (si, scale) in [1.0f32, 0.5, -2.0, 7.25, 1e-30, 3e30, 0.0]
            .into_iter()
            .enumerate()
        {
            let case = format!("{}_scale{si}", name(q));
            let whole = soft_bits(&Sdrk, q, &x, scale);
            let fb = soft_bits(&Fallback, q, &x, scale);
            let mut d = Digest::new();
            digest_f32(&mut d, &whole);
            let mut e = Digest::new();
            digest_f32(&mut e, &fb);
            assert_eq!(d, e, "{case}: the facade differs from the fallback");
            for n in 1..=33 {
                digest_f32(&mut d, &soft_bits(&Sdrk, q, &x[..n], scale));
            }
            t.check(&case, &d).unwrap();
            if si == 0 {
                // The full output of the boundaries, the special values and the first random
                // values (the symbols after the sweep).
                let k = q.bits_per_symbol();
                let edges = &whole[SWEEP * k..(SWEEP + EDGE_SYMBOLS) * k];
                let folded: Vec<f32> = edges.iter().map(|&v| fold(v)).collect();
                t.check_vector(&format!("{case}_edges"), &baseline::f32_bytes(&folded))
                    .unwrap();
            }
        }
    }
}

#[test]
fn the_grid_decisions_keep_their_baseline() {
    let t = table("qam_decide");
    let x = axis_symbols();
    for q in ORDERS {
        let eq = match q {
            Qam::Qam16 => EqConstellation::Qam16,
            _ => EqConstellation::Qam64,
        };
        let mut levels = Digest::new();
        let mut symbols = Digest::new();
        let mut points = Digest::new();
        let mut eq_points = Digest::new();
        for &y in &x {
            levels.bytes(&q.decide_level(y.re).to_le_bytes());
            levels.bytes(&q.decide_level(y.im).to_le_bytes());
            symbols.bytes(&q.decide_symbol(y).to_le_bytes());
            digest_c32(&mut points, &[q.decide(y)]);
            digest_c32(&mut eq_points, &[eq.decide(y)]);
        }
        t.check(&format!("{}_level", name(q)), &levels).unwrap();
        t.check(&format!("{}_symbol", name(q)), &symbols).unwrap();
        t.check(&format!("{}_point", name(q)), &points).unwrap();
        t.check(&format!("{}_eq_point", name(q)), &eq_points)
            .unwrap();
    }
}

/// `n` random points of `q` (stream `stream`), turned by `phase0 + 2 pi offset k` at symbol
/// `k`, with Gaussian noise of `noise` per component (stream `stream + 1`).
fn signal(q: Qam, n: usize, phase0: f64, offset: f64, noise: f64, stream: u64) -> Vec<C32> {
    let mut sym = Pcg32::new(SEED, stream);
    let mut rng = Pcg32::new(SEED, stream + 1);
    (0..n)
        .map(|k| {
            let p = q.point(sym.below(q.points() as u32));
            let ph = phase0 + core::f64::consts::TAU * offset * k as f64;
            let (s, c) = (libm::sin(ph), libm::cos(ph));
            let (re, im) = (f64::from(p.re), f64::from(p.im));
            C32::new(
                (re * c - im * s + noise * rng.normal()) as f32,
                (re * s + im * c + noise * rng.normal()) as f32,
            )
        })
        .collect()
}

fn gaussian(n: usize, sigma: f64, stream: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(SEED, stream);
    (0..n)
        .map(|_| C32::new((sigma * rng.normal()) as f32, (sigma * rng.normal()) as f32))
        .collect()
}

fn uniform(n: usize, stream: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(SEED, stream);
    (0..n)
        .map(|_| {
            C32::new(
                (rng.next_f64() * 2.0 - 1.0) as f32,
                (rng.next_f64() * 2.0 - 1.0) as f32,
            )
        })
        .collect()
}

/// The `n`-tap channel `taps` applied to `x` (the first output is `taps[0] x[0]`).
fn channel(x: &[C32], taps: &[C32]) -> Vec<C32> {
    (0..x.len())
        .map(|k| {
            taps.iter()
                .enumerate()
                .filter(|(j, _)| *j <= k)
                .fold(C32::new(0.0, 0.0), |acc, (j, &h)| acc + h * x[k - j])
        })
        .collect()
}

/// The inputs of the loop cases of `q`: their names and symbols.
fn loop_inputs(q: Qam) -> Vec<(&'static str, Vec<C32>)> {
    let (noise, stream) = match q {
        Qam::Qam16 => (0.03, 10),
        _ => (0.015, 20),
    };
    vec![
        ("lock", signal(q, 6000, 0.3, 0.0005, noise, stream)),
        ("gaussian", gaussian(3000, 0.7, stream + 2)),
        ("uniform", uniform(3000, stream + 3)),
        ("axis", axis_symbols()),
    ]
}

fn costas_constellation(q: Qam) -> CostasConstellation {
    match q {
        Qam::Qam16 => CostasConstellation::Qam16,
        _ => CostasConstellation::Qam64,
    }
}

/// The output of `costas` and the state after every piece of `piece` symbols (all at once
/// when `piece` is the length).
fn run_costas<K: Costas>(k: &K, q: Qam, x: &[C32], piece: usize) -> (Vec<C32>, Digest) {
    let mut st = CostasState::new(costas_constellation(q), 0.01, 0.7071, 0.05).unwrap();
    let mut out = vec![C32::default(); x.len()];
    let mut state = Digest::new();
    for (i, o) in x.chunks(piece).zip(out.chunks_mut(piece)) {
        k.costas(&mut st, i, o).unwrap();
        state.bytes(&st.phase().to_le_bytes());
        state.bytes(&st.frequency().to_bits().to_le_bytes());
        state.f32s(&[fold(st.lock_level())]);
        state.bytes(&[u8::from(st.locked())]);
    }
    (out, state)
}

#[test]
fn costas_keeps_its_baseline() {
    let t = table("costas");
    for q in ORDERS {
        for (input, x) in loop_inputs(q) {
            let case = format!("{}_{input}", name(q));
            let (out, state) = run_costas(&Sdrk, q, &x, PIECE);
            let (whole, _) = run_costas(&Fallback, q, &x, x.len());
            let (fb, fb_state) = run_costas(&Fallback, q, &x, PIECE);
            let mut d = Digest::new();
            digest_c32(&mut d, &out);
            let mut e = Digest::new();
            digest_c32(&mut e, &whole);
            assert_eq!(d, e, "{case}: the pieces differ from the whole run");
            let mut f = Digest::new();
            digest_c32(&mut f, &fb);
            assert_eq!((d, state), (f, fb_state), "{case}: the facade differs");
            t.check(&format!("{case}_out"), &d).unwrap();
            t.check(&format!("{case}_state"), &state).unwrap();
        }
    }
}

/// The equalizer cases: name, forward taps, parameters.
fn equalize_cases(q: Qam) -> Vec<(&'static str, usize, EqParams)> {
    let c = match q {
        Qam::Qam16 => EqConstellation::Qam16,
        _ => EqConstellation::Qam64,
    };
    let base = EqParams {
        constellation: c,
        ..EqParams::default()
    };
    vec![
        ("linear", 11, base),
        (
            "dfe",
            11,
            EqParams {
                feedback_taps: 5,
                phase_bw: 0.0,
                ..base
            },
        ),
        (
            "decision",
            7,
            EqParams {
                cma_symbols: 0,
                mu_dd: 0.01,
                phase_bw: 0.005,
                ..base
            },
        ),
    ]
}

fn run_equalize<K: Equalize>(
    k: &K,
    taps: usize,
    p: EqParams,
    x: &[C32],
    piece: usize,
) -> (Vec<C32>, Digest) {
    let n = taps + p.feedback_taps;
    let mut st = EqualizeState::new(vec![C32::default(); n], vec![C32::default(); n], p).unwrap();
    let mut out = vec![C32::default(); x.len()];
    let mut state = Digest::new();
    for (i, o) in x.chunks(piece).zip(out.chunks_mut(piece)) {
        k.equalize(&mut st, i, o).unwrap();
        state.f32s(&[fold(st.mse())]);
        state.bytes(&[u8::from(st.mode() == EqMode::Decision)]);
        state.bytes(&st.phase().to_le_bytes());
        state.f32s(&[fold(st.frequency())]);
    }
    digest_c32(&mut state, st.weights());
    (out, state)
}

#[test]
fn equalize_keeps_its_baseline() {
    let t = table("equalize");
    let taps = [C32::new(1.0, 0.0), C32::new(0.2, -0.1), C32::new(0.0, 0.05)];
    for q in ORDERS {
        let (noise, stream) = match q {
            Qam::Qam16 => (0.02, 30),
            _ => (0.01, 40),
        };
        let inputs = [
            (
                "multipath",
                channel(&signal(q, 8000, 0.0, 0.0, noise, stream), &taps),
            ),
            ("turned", signal(q, 6000, 0.05, 0.0002, noise, stream + 2)),
            ("axis", axis_symbols()),
        ];
        for (cname, ntaps, p) in equalize_cases(q) {
            for (input, x) in &inputs {
                let case = format!("{}_{cname}_{input}", name(q));
                let (out, state) = run_equalize(&Sdrk, ntaps, p, x, PIECE);
                let (whole, _) = run_equalize(&Fallback, ntaps, p, x, x.len());
                let (fb, fb_state) = run_equalize(&Fallback, ntaps, p, x, PIECE);
                let mut d = Digest::new();
                digest_c32(&mut d, &out);
                let mut e = Digest::new();
                digest_c32(&mut e, &whole);
                assert_eq!(d, e, "{case}: the pieces differ from the whole run");
                let mut f = Digest::new();
                digest_c32(&mut f, &fb);
                assert_eq!((d, state), (f, fb_state), "{case}: the facade differs");
                t.check(&format!("{case}_out"), &d).unwrap();
                t.check(&format!("{case}_state"), &state).unwrap();
            }
        }
    }
}
